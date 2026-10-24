"""Time Machine: rebuild the environment a notebook was written for, and run it there.

Saved outputs are the best evidence of what a notebook is supposed to compute,
but many notebooks are shared with outputs cleared. The Time Machine recovers
that evidence: Nemotron infers the Python and library versions the notebook was
written against (from its code, its metadata and any warnings it printed), uv
builds that environment in seconds, and the untouched original runs there. Its
outputs become the reference that a modernised version must reproduce.

When a notebook does have saved outputs, the same run checks that the
reconstruction is faithful: the original should reproduce its own saved outputs
in the rebuilt environment.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from nbformat import NotebookNode

from .compare import ComparisonReport, comparable_text, compare_notebooks
from .imports import notebook_distributions
from .notebook import has_stored_outputs
from .patches import PatchParseError, extract_json_object, validate_pip_spec
from .runner import ExecutionResult, RunnerError, execute_notebook

SUPPORTED_PYTHONS = ("3.8", "3.9", "3.10", "3.11", "3.12")
KERNEL_NAME = "revive-legacy"
INSTALL_TIMEOUT_S = 900
MAX_CELL_CHARS = 2500

Runner = Callable[..., subprocess.CompletedProcess]
Emit = Callable[[dict], None]

INFER_SYSTEM = """You reconstruct the Python environment a Jupyter notebook was originally written for, so that it can run again exactly as its author ran it.

Choose a Python version and exact versions of the notebook's libraries such that:
- every API the notebook uses existed and behaved as the code expects (functions, methods, parameters and defaults that were later removed or changed must still be present);
- the versions are mutually compatible and have pre-built wheels for the Python version you choose;
- among versions that satisfy this, prefer the most recent ones.
Always pin numpy (and scipy when scikit-learn or scipy is used), even if the notebook does not import them directly: compiled libraries only work with the NumPy generation they were built for.
Use the evidence given: Python version in the metadata, warnings the notebook printed, and the APIs in the code.

Python must be one of: 3.8, 3.9, 3.10, 3.11, 3.12.

Keep your reasoning short: decide from the strongest evidence and answer.
Answer with a single JSON object and nothing else:
{"python": "3.10", "packages": {"numpy": "1.22.4", "pandas": "1.5.3"}, "reasoning": "<one or two sentences>"}
"""


class EnvError(RuntimeError):
    """A legacy environment could not be built."""


@dataclass
class EnvSpec:
    python: str
    packages: dict[str, str] = field(default_factory=dict)
    reasoning: str = ""
    unpinned: list[str] = field(default_factory=list)  # imported libraries the model did not pin
    as_of: Optional[str] = None  # YYYY-MM-DD: only packages uploaded before this date may be installed
    corrections: list[str] = field(default_factory=list)  # pins Revive changed because they post-date as_of

    def requirements(self) -> list[str]:
        reqs = [f"{name}=={version}" for name, version in self.packages.items()] + list(self.unpinned)
        if self.packages and "numpy" not in self.packages:
            # Compiled libraries from before mid-2024 break against NumPy 2 ("numpy.dtype size changed").
            reqs.append("numpy<2")
        return reqs

    def describe(self) -> str:
        pins = ", ".join(f"{name} {version}" for name, version in self.packages.items())
        newest = f"latest as of {self.as_of}" if self.as_of else "latest"
        extra = ", ".join(f"{name} ({newest})" for name in self.unpinned)
        parts = [p for p in (pins, extra) if p]
        return f"Python {self.python}" + (f", {', '.join(parts)}" if parts else "")

    def to_dict(self) -> dict[str, Any]:
        return {"python": self.python, "packages": dict(self.packages), "unpinned": list(self.unpinned),
                "as_of": self.as_of, "corrections": list(self.corrections), "reasoning": self.reasoning}


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_date(value: Optional[str]) -> Optional[str]:
    """Return a YYYY-MM-DD date string, or raise ValueError."""
    if value is None or value == "":
        return None
    import datetime as _dt

    text = str(value).strip()[:10]
    if not _DATE_RE.match(text):
        raise ValueError(f"dates must look like 2023-02-01, got {value!r}")
    _dt.date.fromisoformat(text)  # raises ValueError for impossible dates
    return text


def git_last_changed(path: Path, runner: Runner = subprocess.run) -> Optional[str]:
    """Date (YYYY-MM-DD) of the last commit that changed ``path``, or None outside git."""
    path = Path(path).resolve()
    try:
        proc = runner(
            ["git", "log", "-1", "--format=%cs", "--", path.name],
            cwd=str(path.parent), capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return validate_date(proc.stdout.strip())
    except ValueError:
        return None


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# Libraries that are not always imported directly but whose version decides
# whether everything else works (binary compatibility, transitive behaviour).
SUPPORTING_LIBRARIES = ("numpy", "scipy", "pandas", "joblib", "threadpoolctl", "matplotlib")


def _aliases(allowed: Sequence[str]) -> dict[str, str]:
    """Normalised name -> distribution name, accepting import names too (sklearn -> scikit-learn)."""
    from .imports import _DIST_NAMES

    table = {_norm(a): a for a in allowed}
    for module, dist in _DIST_NAMES.items():
        if _norm(dist) in table:
            table[_norm(module)] = table[_norm(dist)]
    return table


def parse_env_spec(text: str, allowed: Sequence[str]) -> EnvSpec:
    """Validate the model's environment proposal. Only the notebook's own libraries may be pinned."""
    obj = extract_json_object(text)
    python = str(obj.get("python", "")).strip()
    match = re.match(r"^3\.(\d+)", python)
    if not match or f"3.{match.group(1)}" not in SUPPORTED_PYTHONS:
        raise PatchParseError(f"python must be one of {SUPPORTED_PYTHONS}, got {python!r}")
    python = f"3.{match.group(1)}"

    raw = obj.get("packages") or {}
    if not isinstance(raw, dict):
        raise PatchParseError("'packages' must be an object mapping names to versions")
    names = _aliases(list(allowed) + [lib for lib in SUPPORTING_LIBRARIES if lib not in allowed])
    packages: dict[str, str] = {}
    for name, version in raw.items():
        if not isinstance(name, str) or not isinstance(version, (str, int, float)):
            raise PatchParseError(f"invalid package entry: {name!r}: {version!r}")
        canonical = names.get(_norm(name))
        if canonical is None:
            continue  # ignore anything the notebook does not import
        spec = validate_pip_spec(f"{canonical}=={str(version).strip()}")
        packages[canonical] = spec.split("==", 1)[1]
    unpinned = [a for a in allowed if a not in packages]
    reasoning = obj.get("reasoning", "")
    return EnvSpec(python=python, packages=packages, reasoning=str(reasoning).strip(), unpinned=unpinned)


def collect_evidence(nb: NotebookNode) -> dict[str, Any]:
    """Facts in the notebook that hint at when, and with what, it was written."""
    language = nb.metadata.get("language_info", {}) or {}
    warnings: list[str] = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream" and output.get("name") == "stderr":
                text = output.get("text", "")
                text = "".join(text) if isinstance(text, list) else str(text)
                if "Warning" in text:
                    warnings.append(text.strip()[:400])
    return {
        "python_in_metadata": language.get("version"),
        "kernel": (nb.metadata.get("kernelspec", {}) or {}).get("display_name"),
        "warnings_printed": warnings[:8],
        "libraries": notebook_distributions(nb),
        "has_saved_outputs": has_stored_outputs(nb),
    }


def infer_prompt(nb: NotebookNode, evidence: dict[str, Any], history: Sequence[str]) -> str:
    lines = ["Evidence:"]
    if evidence.get("as_of"):
        lines.append(
            f"- The notebook was last changed on {evidence['as_of']} (from version control). Its author ran it with "
            f"versions released before that date. The build only allows packages released before {evidence['as_of']}, "
            "so pin versions from before that date; libraries you leave unpinned get the newest release from before it."
        )
    else:
        lines.append(
            "- The date the notebook was written is unknown. Note that behaviour can change between versions without "
            "any API change (for example changed defaults); prefer the versions most widely used when the APIs in the code were current."
        )
    lines.append(f"- Python version recorded in the notebook metadata: {evidence['python_in_metadata'] or 'not recorded'}")
    lines.append(f"- Libraries imported: {', '.join(evidence['libraries']) or 'none'}")
    if evidence["warnings_printed"]:
        lines.append("- Warnings the notebook printed when its author ran it:")
        lines.extend(f"    {w}" for w in evidence["warnings_printed"])
    lines.append("")
    lines.append("Notebook code cells:")
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            lines.append(f"### cell {index}\n{cell.source[:MAX_CELL_CHARS]}")
    if history:
        lines.append("")
        lines.append("Previous reconstructions that did not work (choose differently):")
        lines.extend(f"- {item}" for item in history)
    return "\n".join(lines)


def _uv() -> list[str]:
    """Command prefix for uv: the installed module (cross-platform) or a uv binary on PATH."""
    try:
        import uv  # noqa: F401

        return [sys.executable, "-m", "uv"]
    except ImportError:
        found = shutil.which("uv")
        if found:
            return [found]
    raise EnvError("uv is required for the Time Machine: pip install uv")


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


class LegacyEnv:
    """A standalone environment (its own Python version) built with uv."""

    def __init__(self, root: Path, spec: EnvSpec, *, runner: Runner = subprocess.run) -> None:
        self.root = Path(root).resolve()
        self.spec = spec
        self.python = _venv_python(self.root)
        self.kernel_dir = self.root / "kernels"
        self._run = runner

    @classmethod
    def create(cls, root: Path, spec: EnvSpec, *, runner: Runner = subprocess.run) -> "LegacyEnv":
        env = cls(root, spec, runner=runner)
        env._build()
        return env

    def _call(self, args: list[str], what: str) -> str:
        try:
            proc = self._run(args, capture_output=True, text=True, timeout=INSTALL_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            raise EnvError(f"{what} timed out after {INSTALL_TIMEOUT_S}s") from exc
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if proc.returncode != 0:
            raise EnvError(f"{what} failed: {output[-1500:]}")
        return output

    def _build(self) -> None:
        uv = _uv()
        self._call([*uv, "venv", "--quiet", "--python", self.spec.python, str(self.root)], f"creating Python {self.spec.python}")
        install = [*uv, "pip", "install", "--quiet", "--python", str(self.python)]
        if self.spec.as_of:
            # Resolve everything (including unpinned and indirect dependencies) as of that date.
            install += ["--exclude-newer", self.spec.as_of]
        install += ["ipykernel", *self.spec.requirements()]
        self._call(install, f"installing {self.spec.describe()}")
        spec_dir = self.kernel_dir / KERNEL_NAME
        spec_dir.mkdir(parents=True, exist_ok=True)
        kernel = {
            "argv": [str(self.python), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": f"Revive Time Machine ({self.spec.describe()})",
            "language": "python",
        }
        (spec_dir / "kernel.json").write_text(json.dumps(kernel, indent=1), encoding="utf-8")

    def kernel_manager(self):
        from jupyter_client.kernelspec import KernelSpecManager
        from jupyter_client.manager import AsyncKernelManager

        ksm = KernelSpecManager(kernel_dirs=[str(self.kernel_dir)])
        return AsyncKernelManager(kernel_name=KERNEL_NAME, kernel_spec_manager=ksm)

    def installed_versions(self, names: Sequence[str]) -> dict[str, str]:
        code = (
            "import json, importlib.metadata as m, sys\n"
            "out = {'python': sys.version.split()[0]}\n"
            f"for d in {list(names)!r}:\n"
            "    try: out[d] = m.version(d)\n"
            "    except m.PackageNotFoundError: out[d] = 'not installed'\n"
            "print(json.dumps(out))"
        )
        try:
            return json.loads(self._call([str(self.python), "-c", code], "reading versions").splitlines()[-1])
        except (EnvError, ValueError, IndexError):
            return {}

    def destroy(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@dataclass
class Reconstruction:
    ok: bool
    spec: Optional[EnvSpec] = None
    reference: Optional[NotebookNode] = None
    installed: dict[str, str] = field(default_factory=dict)
    vs_saved: Optional[ComparisonReport] = None
    attempts: list[dict] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "environment": self.spec.to_dict() if self.spec else None,
            "installed": self.installed,
            "reproduces_saved_outputs": None if self.vs_saved is None or self.vs_saved.unverifiable else self.vs_saved.verified,
            "vs_saved": self.vs_saved.summary() if self.vs_saved is not None and not self.vs_saved.unverifiable else None,
            "attempts": self.attempts,
            "error": self.error,
        }


def reference_outputs(nb: NotebookNode) -> dict[int, str]:
    """Comparable text output of every code cell, for display."""
    return {
        index: comparable_text(cell.get("outputs", []))
        for index, cell in enumerate(nb.cells)
        if cell.cell_type == "code"
    }


MAX_DATE_CORRECTIONS = 4
_FILTERED_RE = re.compile(r"`([A-Za-z0-9][A-Za-z0-9._-]*)` was filtered by `exclude-newer`")


def _released_after_date(message: str, spec: EnvSpec) -> Optional[str]:
    """The pinned package uv rejected because it was released after the cutoff date, if any."""
    if not spec.as_of:
        return None
    pinned = {_norm(name): name for name in spec.packages}
    for match in _FILTERED_RE.finditer(message):
        name = pinned.get(_norm(match.group(1)))
        if name is not None:
            return name
    return None


def reconstruct(
    nb: NotebookNode,
    llm: Any,
    *,
    workdir: Path,
    as_of: Optional[str] = None,
    max_tries: int = 4,
    cell_timeout: int = 300,
    cwd: Optional[str] = None,
    emit: Optional[Emit] = None,
    runner: Runner = subprocess.run,
    env_factory: Optional[Callable[[Path, EnvSpec], Any]] = None,
    rtol: float = 1e-3,
) -> Reconstruction:
    """Infer, build and run the notebook's original environment, retrying with feedback."""
    send = emit or (lambda event: None)
    as_of = validate_date(as_of)
    evidence = collect_evidence(nb)
    evidence["as_of"] = as_of
    libraries = evidence["libraries"]
    build = env_factory or (lambda root, spec: LegacyEnv.create(root, spec, runner=runner))
    history: list[str] = []
    result = Reconstruction(ok=False)

    for attempt in range(1, max_tries + 1):
        send({"type": "reconstruct_start", "attempt": attempt})
        try:
            reply = llm.chat(
                [{"role": "system", "content": INFER_SYSTEM}, {"role": "user", "content": infer_prompt(nb, evidence, history)}],
                role="deep",
                temperature=0.2,
                max_tokens=32768,
            )
            spec = parse_env_spec(reply, libraries)
            spec.as_of = as_of
        except PatchParseError as exc:
            history.append(f"attempt {attempt}: the answer was not a valid environment ({exc})")
            result.attempts.append({"attempt": attempt, "result": f"invalid proposal: {exc}"})
            send({"type": "reconstruct_result", "attempt": attempt, "ok": False, "error": f"invalid proposal: {exc}"})
            continue
        except Exception as exc:  # noqa: BLE001 - API failure ends this attempt only
            result.attempts.append({"attempt": attempt, "result": f"model call failed: {exc}"})
            send({"type": "reconstruct_result", "attempt": attempt, "ok": False, "error": f"model call failed: {exc}"})
            continue

        result.spec = spec
        send({"type": "reconstruct_env", "attempt": attempt, "python": spec.python, "packages": spec.packages,
              "reasoning": spec.reasoning, "description": spec.describe(), "as_of": as_of})

        env = None
        failure = ""
        for build_try in range(1, MAX_DATE_CORRECTIONS + 2):
            root = Path(workdir) / f"legacy-{attempt}-{build_try}"
            try:
                env = build(root, spec)
                break
            except EnvError as exc:
                failure = str(exc)
                too_new = _released_after_date(failure, spec)
                if not too_new or build_try > MAX_DATE_CORRECTIONS:
                    break
                # The date is hard evidence: a pinned version released after it cannot be what
                # the author used. Drop that pin and let the date choose (newest before it).
                version = spec.packages.pop(too_new)
                if too_new in libraries and too_new not in spec.unpinned:
                    spec.unpinned.append(too_new)
                note = f"{too_new} {version} was released after {as_of}; using the newest {too_new} from before that date"
                spec.corrections.append(note)
                send({"type": "reconstruct_adjust", "attempt": attempt, "package": too_new, "version": version,
                      "as_of": as_of, "message": note})
        if env is None:
            history.append(f"{spec.describe()}: could not be installed: {failure[-600:]}")
            result.attempts.append({"attempt": attempt, "environment": spec.to_dict(), "result": "install failed"})
            send({"type": "reconstruct_result", "attempt": attempt, "ok": False, "error": failure[-600:]})
            continue

        try:
            run: ExecutionResult = execute_notebook(nb, cwd=cwd, timeout=cell_timeout, kernel_manager_factory=env.kernel_manager)
            installed = env.installed_versions(libraries) if hasattr(env, "installed_versions") else {}
        except RunnerError as exc:
            history.append(f"{spec.describe()}: the notebook kernel could not start: {exc}")
            result.attempts.append({"attempt": attempt, "environment": spec.to_dict(), "result": f"kernel failed: {exc}"})
            send({"type": "reconstruct_result", "attempt": attempt, "ok": False, "error": f"kernel failed: {exc}"})
            continue
        finally:
            if hasattr(env, "destroy"):
                env.destroy()

        if not run.ok:
            history.append(f"{spec.describe()}: the original notebook failed there: {run.error_summary()[:300]}")
            result.attempts.append({"attempt": attempt, "environment": spec.to_dict(), "result": run.error_summary()})
            send({"type": "reconstruct_result", "attempt": attempt, "ok": False, "error": run.error_summary()})
            continue

        result.ok = True
        result.reference = run.notebook
        result.installed = installed
        if evidence["has_saved_outputs"]:
            result.vs_saved = compare_notebooks(nb, run.notebook, rtol=rtol)
        result.attempts.append({"attempt": attempt, "environment": spec.to_dict(), "result": "original ran"})
        send({
            "type": "reconstruct_result",
            "attempt": attempt,
            "ok": True,
            "installed": installed,
            "duration_s": run.duration_s,
            "outputs": {str(k): v for k, v in reference_outputs(run.notebook).items()},
            "vs_saved": result.vs_saved.summary() if result.vs_saved and not result.vs_saved.unverifiable else None,
            "reproduces_saved": (result.vs_saved.verified if result.vs_saved and not result.vs_saved.unverifiable else None),
        })
        return result

    result.error = history[-1] if history else "no environment could be reconstructed"
    return result
