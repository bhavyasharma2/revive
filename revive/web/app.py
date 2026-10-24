"""FastAPI app: pick or upload a notebook, watch it being repaired, download the result.

Run locally:   uvicorn revive.web.app:app --port 8000
Settings (environment variables):
  REVIVE_ALLOW_UPLOADS=1     accept user notebooks (they execute code; keep off on public hosts)
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
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from ..notebook import load_notebook, loads_notebook
from .jobs import CachedRun, JobManager, LimitExceeded, RateLimiter, RepairFn

# The repository root (holds benchmark/). Override with REVIVE_HOME if the
# package is installed somewhere other than its source checkout.
ROOT = Path(os.environ.get("REVIVE_HOME") or Path(__file__).resolve().parents[2])
STATIC = Path(__file__).resolve().parent / "static"
SAMPLES = ROOT / "benchmark" / "notebooks"
MANIFEST = ROOT / "benchmark" / "manifest.json"
MAX_UPLOAD_BYTES = 2_000_000
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
            env = NotebookEnv.create(Path(tmp) / "venv")
            agent = ReviveAgent(
                llm,
                search=search,
                env=env,
                max_attempts=4,
                cell_timeout=180,
                cwd=tmp,
                on_event=emit,
                time_machine=options.get("time_machine", "auto"),
                as_of=options.get("as_of"),
            )
            result = agent.repair(nb)
        return result, llm.usage

    return repair


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


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


def create_app(repair: Optional[RepairFn] = None, *, recordings: Optional[Path] = None) -> FastAPI:
    allow_uploads = _flag("REVIVE_ALLOW_UPLOADS", False)
    live_enabled = _flag("REVIVE_LIVE", True)
    limiter = RateLimiter(per_client=_int("REVIVE_LIVE_PER_CLIENT", 3), per_day=_int("REVIVE_LIVE_PER_DAY", 20))
    manager = JobManager(repair or default_repair(), max_concurrent=_int("REVIVE_MAX_CONCURRENT", 2), limiter=limiter)
    folder = recordings or Path(os.environ.get("REVIVE_RECORDINGS", ROOT / "benchmark" / "recordings"))
    load_recordings(manager, folder)

    samples = {entry["name"]: entry for entry in load_manifest() if (SAMPLES / f"{entry['name']}.ipynb").exists()}

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
        return {"allow_uploads": allow_uploads, "live_enabled": live_enabled}

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

    @app.post("/api/jobs")
    def create_job(body: JobRequest, request: Request) -> JSONResponse:
        if bool(body.sample) == bool(body.notebook):
            raise HTTPException(400, "send either a sample name or a notebook, not both")
        if body.sample:
            if body.sample not in samples:
                raise HTTPException(404, f"unknown sample: {body.sample}")
            nb = load_notebook(SAMPLES / f"{body.sample}.ipynb")
            label, key = samples[body.sample]["title"], f"sample:{body.sample}" + (":tm" if body.time_machine else "")
        else:
            if not allow_uploads:
                raise HTTPException(403, "uploads are turned off on this server; pick one of the sample notebooks")
            if len(body.notebook.encode("utf-8")) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "notebook is larger than 2 MB")
            try:
                nb = loads_notebook(body.notebook)
            except Exception as exc:  # noqa: BLE001 - nbformat raises several types
                raise HTTPException(400, f"this file is not a valid .ipynb notebook: {exc}") from exc
            label, key = "Uploaded notebook", None

        live = body.live or key is None or key not in manager.cache
        if live and not live_enabled:
            raise HTTPException(403, "live repairs are turned off on this server; only recorded replays are available")
        try:
            options = {"time_machine": "always" if body.time_machine else "auto"}
            if body.sample:
                # Bundled samples carry the date their version control would give (see benchmark/manifest.json).
                options["as_of"] = samples[body.sample].get("as_of")
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
