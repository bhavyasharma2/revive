"""FastAPI app: pick or upload a notebook, watch it being repaired, download the result.

Run locally:   uvicorn revive.web.app:app --port 8000
Settings (environment variables):
  REVIVE_ALLOW_UPLOADS=1     accept user notebooks (they execute code; on a public host also
                             run as root with REVIVE_SANDBOX_USER set, see revive/sandbox.py)
  REVIVE_SANDBOX_USER=name   run notebook kernels as this unprivileged user
  REVIVE_LIVE=0              disable live repairs entirely (replays only)
  REVIVE_LIVE_PER_CLIENT=3   live repairs per client per day
  REVIVE_LIVE_PER_DAY=20     live repairs per day for the whole server
  REVIVE_MAX_CONCURRENT=2    repairs running at once
  REVIVE_RECORDINGS=path     folder of recorded runs to replay (default benchmark/recordings)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from .. import sandbox
from .. import preflight
from ..notebook import load_notebook, loads_notebook
from ..timemachine import validate_date
from .jobs import CachedRun, JobManager, LimitExceeded, RateLimiter, RepairFn

# The repository root (holds benchmark/). Override with REVIVE_HOME if the
# package is installed somewhere other than its source checkout.
ROOT = Path(os.environ.get("REVIVE_HOME") or Path(__file__).resolve().parents[2])
STATIC = Path(__file__).resolve().parent / "static"
SAMPLES = ROOT / "benchmark" / "notebooks"
MANIFEST = ROOT / "benchmark" / "manifest.json"
# Sample sets shown in the app: (folder with manifest.json and notebooks/). The real
# public notebooks (benchmark/real/) come after the benchmark's own.
SAMPLE_SETS = (ROOT / "benchmark", ROOT / "benchmark" / "real")
MAX_UPLOAD_BYTES = 2_000_000
# Data files sent with an uploaded notebook (CSVs, images, helper modules).
MAX_DATA_FILES = 30
MAX_DATA_BYTES = 15_000_000
SSE_POLL_S = 0.2
SSE_KEEPALIVE_S = 15.0


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def default_repair() -> RepairFn:
    """Real repairs: Nemotron on Token Factory, Tavily search, an isolated env per job."""
    from ..agent import ReviveAgent
    from ..env import NotebookEnv
    from ..llm import TokenFactoryClient
    from ..search import TavilyClient, TavilyError

    def repair(nb, emit, options):
        llm = TokenFactoryClient()
        try:
            search = TavilyClient()
        except TavilyError:
            search = None
        with tempfile.TemporaryDirectory(prefix="revive-web-") as tmp:
            # The notebook may write files in work/; the environment in venv/ stays out of its reach.
            work = sandbox.make_workdir(Path(tmp) / "work")
            write_data_files(work, options.get("files") or {})
            env = NotebookEnv.create(Path(tmp) / "venv")
            agent = ReviveAgent(
                llm,
                search=search,
                env=env,
                max_attempts=4,
                max_total_attempts=10,
                cell_timeout=180,
                cwd=str(work),
                on_event=emit,
                time_machine=options.get("time_machine", "auto"),
                as_of=options.get("as_of"),
            )
            result = agent.repair(nb)
        return result, llm.usage

    return repair


def decode_data_files(files: dict[str, str]) -> dict[str, bytes]:
    """Validate and decode {relative path: base64 content}; raises HTTPException with a plain reason."""
    import base64
    import binascii

    if len(files) > MAX_DATA_FILES:
        raise HTTPException(413, f"at most {MAX_DATA_FILES} data files can be added")
    decoded: dict[str, bytes] = {}
    total = 0
    for raw_path, content in files.items():
        path = preflight.normalize_path(str(raw_path))
        if path is None or not all(re.fullmatch(r"[\w.\- ()]+", part) for part in path.split("/")):
            raise HTTPException(400, f"not a usable file name: {raw_path!r} (use a name like data/train.csv)")
        if path in decoded:
            raise HTTPException(400, f"the file {path} was added twice")
        try:
            data = base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError, TypeError) as exc:
            raise HTTPException(400, f"could not read the file {path}") from exc
        total += len(data)
        if total > MAX_DATA_BYTES:
            raise HTTPException(413, f"data files may add up to {MAX_DATA_BYTES // 1_000_000} MB")
        decoded[path] = data
    return decoded


def write_data_files(work: Path, files: dict[str, bytes]) -> None:
    """Place uploaded data files next to the notebook, in the folders its code expects."""
    root = work.resolve()
    for path, data in files.items():
        target = (root / path).resolve()
        if root not in target.parents:  # decode_data_files already refuses this; never write outside work/
            raise ValueError(f"refusing to write outside the working folder: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if sandbox.active():
            for folder in [target.parent, *target.parent.parents]:
                if folder == root:
                    break
                os.chmod(folder, 0o777)  # the notebook may write next to its data, like it would at home
        target.write_bytes(data)
        os.chmod(target, 0o644)


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_samples() -> dict[str, tuple[dict, Path]]:
    """Every bundled notebook that exists on disk: name -> (manifest entry, path)."""
    found: dict[str, tuple[dict, Path]] = {}
    for folder in SAMPLE_SETS:
        manifest = folder / "manifest.json"
        if not manifest.exists():
            continue
        for entry in json.loads(manifest.read_text(encoding="utf-8")):
            path = folder / "notebooks" / f"{entry['name']}.ipynb"
            if path.exists() and entry["name"] not in found:
                found[entry["name"]] = (entry, path)
    return found


def load_recordings(manager: JobManager, folder: Path) -> int:
    """Seed the replay cache with runs recorded by scripts/run_benchmark.py."""
    count = 0
    if not folder.is_dir():
        return 0
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # name.json -> normal run; name.tm.json -> run with the Time Machine forced on
            key = f"sample:{path.stem[:-3]}:tm" if path.stem.endswith(".tm") else f"sample:{path.stem}"
            manager.seed_cache(
                key,
                CachedRun(
                    events=data["events"],
                    notebook_json=data.get("notebook_json"),
                    report=data.get("report"),
                    outcome=data["outcome"],
                    recorded_at=float(data["recorded_at"]),
                ),
            )
            count += 1
        except (KeyError, ValueError, TypeError):
            continue
    return count


class JobRequest(BaseModel):
    sample: Optional[str] = None
    notebook: Optional[str] = None
    live: bool = False
    time_machine: bool = False  # force the Time Machine even when outputs are saved
    as_of: Optional[str] = None  # uploads: when the notebook was last changed (YYYY-MM-DD), for the Time Machine
    files: Optional[dict[str, str]] = None  # uploads: data files, {relative path: base64 content}


class PreflightRequest(BaseModel):
    notebook: str
    files: list[str] = []  # relative paths of the data files that will be added


def create_app(repair: Optional[RepairFn] = None, *, recordings: Optional[Path] = None) -> FastAPI:
    allow_uploads = _flag("REVIVE_ALLOW_UPLOADS", False)
    if allow_uploads and sandbox.requested() and not sandbox.active():
        # The operator asked for sandboxed kernels but this process cannot provide them
        # (not root, or the user does not exist): uploaded code would see the API keys.
        print("revive: REVIVE_SANDBOX_USER is set but kernels cannot be sandboxed here; uploads are disabled")
        allow_uploads = False
    live_enabled = _flag("REVIVE_LIVE", True)
    if repair is None and live_enabled and not os.environ.get("NEBIUS_API_KEY", "").strip():
        # Without a key every live repair would fail; offer replays only instead of an error.
        print("revive: NEBIUS_API_KEY is not set, so live repairs and uploads are off (replays still work)")
        live_enabled = False
    limiter = RateLimiter(per_client=_int("REVIVE_LIVE_PER_CLIENT", 3), per_day=_int("REVIVE_LIVE_PER_DAY", 20))
    manager = JobManager(repair or default_repair(), max_concurrent=_int("REVIVE_MAX_CONCURRENT", 2), limiter=limiter)
    folder = recordings or Path(os.environ.get("REVIVE_RECORDINGS", ROOT / "benchmark" / "recordings"))
    load_recordings(manager, folder)

    sample_files = load_samples()
    samples = {name: entry for name, (entry, _path) in sample_files.items()}

    app = FastAPI(title="Revive", docs_url=None, redoc_url=None)
    app.state.manager = manager

    def client_id(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "")
        return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")

    @app.api_route("/", methods=["GET", "HEAD"])
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.get("/api/config")
    def config() -> dict:
        return {
            "allow_uploads": allow_uploads and live_enabled,
            "live_enabled": live_enabled,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_data_files": MAX_DATA_FILES,
            "max_data_bytes": MAX_DATA_BYTES,
            "sandboxed": sandbox.active(),
        }

    @app.get("/api/samples")
    def list_samples() -> list[dict]:
        out = []
        for name, entry in samples.items():
            cached = manager.cache.get(f"sample:{name}")
            cached_tm = manager.cache.get(f"sample:{name}:tm")
            out.append({
                **entry,
                "recorded_at": cached.recorded_at if cached else None,
                "recorded_tm_at": cached_tm.recorded_at if cached_tm else None,
            })
        return out

    def parse_upload(text: str):
        if len(text.encode("utf-8")) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "notebook is larger than 2 MB")
        try:
            return loads_notebook(text)
        except Exception as exc:  # noqa: BLE001 - nbformat raises several types
            raise HTTPException(400, f"this file is not a valid .ipynb notebook: {exc}") from exc

    @app.post("/api/preflight")
    def check_notebook(body: PreflightRequest) -> dict:
        """What would stop a repair of this notebook, before anything runs (reads it, never executes it)."""
        return preflight.check(parse_upload(body.notebook), body.files).to_dict()

    @app.post("/api/jobs")
    def create_job(body: JobRequest, request: Request) -> JSONResponse:
        if bool(body.sample) == bool(body.notebook):
            raise HTTPException(400, "send either a sample name or a notebook, not both")
        if body.sample:
            if body.sample not in samples:
                raise HTTPException(404, f"unknown sample: {body.sample}")
            nb = load_notebook(sample_files[body.sample][1])
            label, key = samples[body.sample]["title"], f"sample:{body.sample}" + (":tm" if body.time_machine else "")
        else:
            if not allow_uploads:
                raise HTTPException(403, "uploads are turned off on this server; pick one of the sample notebooks")
            nb = parse_upload(body.notebook)
            data_files = decode_data_files(body.files or {})
            check = preflight.check(nb, data_files.keys())
            if not check.ok:
                missing = f" Missing: {', '.join(check.missing_files)}." if check.missing_files else ""
                raise HTTPException(400, "this notebook cannot be repaired here yet: " + " ".join(check.blockers()) + missing)
            label, key = "Uploaded notebook", None
            try:
                upload_as_of = validate_date(body.as_of)
            except ValueError as exc:
                raise HTTPException(400, f"last changed date: {exc}") from exc

        live = body.live or key is None or key not in manager.cache
        if live and not live_enabled:
            raise HTTPException(403, "live repairs are turned off on this server; only recorded replays are available")
        try:
            options = {"time_machine": "always" if body.time_machine else "auto"}
            if body.sample:
                # Bundled samples carry the date their version control would give (see benchmark/manifest.json).
                options["as_of"] = samples[body.sample].get("as_of")
            else:
                options["as_of"] = upload_as_of
                options["files"] = data_files
            job = manager.start(nb, label=label, client=client_id(request), cache_key=key, live=live, options=options)
        except LimitExceeded as exc:
            raise HTTPException(429, str(exc)) from exc
        return JSONResponse(job.summary(), status_code=201)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job.summary()

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")

        async def stream():
            sent = 0
            last_beat = time.monotonic()
            while True:
                pending = job.events[sent:]
                for event in pending:
                    yield f"data: {json.dumps(event)}\n\n"
                sent += len(pending)
                if job.done and sent >= len(job.events):
                    yield f"event: end\ndata: {json.dumps(job.summary())}\n\n"
                    return
                if time.monotonic() - last_beat > SSE_KEEPALIVE_S:
                    yield ": keepalive\n\n"
                    last_beat = time.monotonic()
                await asyncio.sleep(SSE_POLL_S)

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    @app.get("/api/jobs/{job_id}/notebook")
    def job_notebook(job_id: str) -> Response:
        job = manager.get(job_id)
        if job is None or job.notebook_json is None:
            raise HTTPException(404, "no repaired notebook for this job")
        return Response(
            job.notebook_json,
            media_type="application/x-ipynb+json",
            headers={"Content-Disposition": 'attachment; filename="revived.ipynb"'},
        )

    @app.get("/api/jobs/{job_id}/report")
    def job_report(job_id: str) -> Any:
        job = manager.get(job_id)
        if job is None or job.report is None:
            raise HTTPException(404, "no report for this job yet")
        return job.report

    @app.api_route("/healthz", methods=["GET", "HEAD"])
    def health() -> dict:
        return {"ok": True}

    return app


# `uvicorn revive.web.app:app` imports this. Building it needs no API keys;
# they are only read when a live repair starts.
app = create_app()
