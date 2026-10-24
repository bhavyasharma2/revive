"""The repair loop: run, research, patch, re-run, verify.

1. Run the notebook in its sandbox environment.
2. If a cell fails, a fast Nemotron model writes a search query and Tavily
   finds the changelog or migration note behind the error.
3. A large Nemotron model reads the notebook, the error, the library versions
   and that evidence, and answers with a JSON patch.
4. The patch is applied (cell edits plus validated pip pins) and the notebook
   runs again. Repeat until it runs or attempts run out.
5. Once it runs, every cell's output is compared with the outputs the author
   saved. Only then is a notebook called verified.

Time Machine: when the notebook has no saved outputs (or when asked), the
original environment is rebuilt first and the untouched notebook is run there;
its outputs become the reference the repaired notebook must reproduce.
"""

from __future__ import annotations

import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

from nbformat import NotebookNode

from .compare import ComparisonReport, Status, compare_notebooks, matching_cells
from .imports import notebook_distributions
from .notebook import has_stored_outputs
from .timemachine import Reconstruction, reconstruct

TIME_MACHINE_MODES = ("off", "auto", "always")
from .patches import (
    Patch,
    PatchApplyError,
    PatchParseError,
    apply_patch,
    changed_cells,
    notebook_diff,
    parse_patch,
    syntax_errors,
)
from .prompts import QUERY_SYSTEM, REPAIR_SYSTEM, query_prompt, repair_prompt
from .runner import ExecutionResult, execute_notebook

Event = dict[str, Any]
EventSink = Callable[[Event], None]


class ChatModel(Protocol):
    def chat(self, messages: Sequence[dict[str, Any]], *, role: str = ..., **kwargs: Any) -> str: ...

    def model_for(self, role: str) -> str: ...


class Searcher(Protocol):
    def search(self, query: str, *, max_results: int = ...) -> list: ...


class Outcome(str, Enum):
    VERIFIED = "verified"                  # runs, and every reference output matches
    ALREADY_WORKING = "already_working"    # ran and matched without any change
    RUNS_UNVERIFIED = "runs_unverified"    # runs, but there was no reference output to check
    DRIFT = "drift"                        # runs, but outputs differ from the reference
    FAILED = "failed"                      # could not be made to run


# Where the outputs a repair is checked against came from.
REFERENCE_SAVED = "saved"                  # saved in the notebook by its author
REFERENCE_RECONSTRUCTED = "reconstructed"  # produced by the Time Machine in the original environment
REFERENCE_NONE = "none"


@dataclass
class Attempt:
    number: int
    error: str
    search_query: str = ""
    sources: list[dict] = field(default_factory=list)
    explanation: str = ""
    diff: str = ""
    pip_install: list[str] = field(default_factory=list)
    result: str = ""
    model: str = ""


@dataclass
class RepairResult:
    outcome: Outcome
    notebook: NotebookNode
    attempts: list[Attempt]
    comparison: Optional[ComparisonReport]
    last_error: str = ""
    installed: list[str] = field(default_factory=list)
    reference: str = REFERENCE_SAVED
    reconstruction: Optional[Reconstruction] = None

    @property
    def runs(self) -> bool:
        return self.outcome != Outcome.FAILED

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "runs": self.runs,
            "reference": self.reference,
            "last_error": self.last_error,
            "installed": self.installed,
            "attempts": [asdict(a) for a in self.attempts],
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "time_machine": self.reconstruction.to_dict() if self.reconstruction else None,
        }


def _drift_summary(report: ComparisonReport, limit: int = 5, max_chars: int = 1500, label: str = "saved output") -> str:
    """Per differing cell: the reference output, what the notebook prints now, and the first difference."""
    blocks = []
    for cell in report.failures[:limit]:
        blocks.append(
            f"cell {cell.cell_index} ({cell.status.value}): {cell.detail}\n"
            f"  {label}:\n{_indent(cell.stored[:max_chars])}\n"
            f"  output now:\n{_indent(cell.fresh[:max_chars]) if cell.fresh else '    (no output)'}"
        )
    return "\n".join(blocks)


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines()) or "    (empty)"


_WARNING_RE = re.compile(r"(?:Future|Deprecation|PendingDeprecation)Warning: .+")


def printed_warning_lines(nb: NotebookNode, limit: int = 12) -> list[str]:
    """Distinct deprecation-style warnings in a notebook's stderr outputs, with their cell."""
    found: list[str] = []
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream" or output.get("name") != "stderr":
                continue
            text = output.get("text", "")
            text = "".join(text) if isinstance(text, list) else str(text)
            for match in _WARNING_RE.finditer(text):
                line = f"cell {index}: {match.group(0).strip()[:300]}"
                if line not in found:
                    found.append(line)
    return found[:limit]


_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}
_TRANSIENT_NAMES = {"APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError"}


def is_transient(exc: BaseException) -> bool:
    """Server-side hiccups worth retrying (an HTTP 503 from Token Factory, a dropped connection)."""
    status = getattr(exc, "status_code", None)
    return (isinstance(status, int) and status in _TRANSIENT_STATUS) or type(exc).__name__ in _TRANSIENT_NAMES


def fallback_query(error_name: Optional[str], error_value: Optional[str], dists: Sequence[str]) -> str:
    value = re.sub(r"\s+", " ", (error_value or "")).strip()[:120]
    library = dists[0] if dists else "python"
    return f"{library} {error_name or 'error'} {value} deprecated removed migration".strip()


class ReviveAgent:
    def __init__(
        self,
        llm: ChatModel,
        *,
        search: Optional[Searcher] = None,
        env: Any = None,
        max_attempts: int = 4,
        cell_timeout: int = 300,
        cwd: Optional[str] = None,
        on_event: Optional[EventSink] = None,
        fix_drift: bool = True,
        rtol: float = 1e-3,
        time_machine: str = "auto",
        reconstructor: Optional[Callable[..., Reconstruction]] = None,
        retry_delays: Sequence[float] = (5.0, 15.0),
        as_of: Optional[str] = None,
    ) -> None:
        self.retry_delays = tuple(retry_delays)
        self.as_of = as_of  # when the notebook was last changed, if known (e.g. its last git commit)
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if time_machine not in TIME_MACHINE_MODES:
            raise ValueError(f"time_machine must be one of {TIME_MACHINE_MODES}")
        self.time_machine = time_machine
        self._reconstruct = reconstructor or reconstruct
        self.llm = llm
        self.search = search
        self.env = env
        self.max_attempts = max_attempts
        self.cell_timeout = cell_timeout
        self.cwd = cwd
        self.fix_drift = fix_drift
        self.rtol = rtol
        self._sink = on_event

    # -- helpers ---------------------------------------------------------------

    def _emit(self, kind: str, **data: Any) -> None:
        if self._sink is not None:
            self._sink({"type": kind, **data})

    def _run(self, nb: NotebookNode) -> ExecutionResult:
        factory = self.env.kernel_manager if self.env is not None else None
        return execute_notebook(nb, cwd=self.cwd, timeout=self.cell_timeout, kernel_manager_factory=factory)

    def _versions(self, dists: Sequence[str]) -> dict[str, str]:
        if self.env is not None:
            return self.env.package_versions(list(dists))
        import importlib.metadata as metadata
        import sys

        out = {"python": sys.version.split()[0]}
        for dist in dists:
            try:
                out[dist] = metadata.version(dist)
            except metadata.PackageNotFoundError:
                out[dist] = "not installed"
        return out

    def _research(self, error_name, error_value, error_text, versions, dists) -> tuple[str, list[dict]]:
        if self.search is None:
            return "", []
        query = ""
        try:
            query = self.llm.chat(
                [
                    {"role": "system", "content": QUERY_SYSTEM},
                    {"role": "user", "content": query_prompt(error_text, versions)},
                ],
                role="fast",
                temperature=0.0,
                max_tokens=2048,
            ).strip().strip('"').splitlines()[0][:200]
        except Exception as exc:  # noqa: BLE001 - a failed helper call must not stop the repair
            self._emit("warning", message=f"query model failed, using a fallback query: {exc}")
        if not query:
            query = fallback_query(error_name, error_value, dists)
        try:
            hits = self.search.search(query, max_results=5)
        except Exception as exc:  # noqa: BLE001
            self._emit("warning", message=f"web search failed: {exc}")
            return query, []
        sources = [{"title": h.title, "url": h.url, "content": h.content} for h in hits]
        self._emit("search", query=query, sources=[{"title": s["title"], "url": s["url"]} for s in sources])
        return query, sources

    def _ask_for_patch(self, prompt: str) -> tuple[str, str]:
        model = ""
        try:
            model = self.llm.model_for("deep")
        except Exception:  # noqa: BLE001 - only used for reporting
            pass
        messages = [{"role": "system", "content": REPAIR_SYSTEM}, {"role": "user", "content": prompt}]
        for delay in (*self.retry_delays, None):
            try:
                # Generous budget: reasoning models think before answering, and a cut-off answer has no patch.
                return self.llm.chat(messages, role="deep", temperature=0.2, max_tokens=32768), model
            except Exception as exc:  # noqa: BLE001 - only temporary server errors are retried
                if delay is None or not is_transient(exc):
                    raise
                self._emit("warning", message=f"Token Factory temporarily unavailable ({type(exc).__name__}); retrying in {delay}s")
                time.sleep(delay)
        raise AssertionError("unreachable")

    # -- main loop -------------------------------------------------------------

    def _time_travel(self, original: NotebookNode) -> Optional[Reconstruction]:
        has_saved = has_stored_outputs(original)
        if self.time_machine == "off" or (self.time_machine == "auto" and has_saved):
            return None
        self._emit("time_machine", reason="no saved outputs" if not has_saved else "requested")
        with tempfile.TemporaryDirectory(prefix="revive-tm-") as tmp:
            rec = self._reconstruct(
                original,
                self.llm,
                workdir=Path(tmp),
                as_of=self.as_of,
                cell_timeout=self.cell_timeout,
                cwd=self.cwd,
                emit=self._sink,
                rtol=self.rtol,
            )
        return rec

    def repair(self, original: NotebookNode) -> RepairResult:
        dists = notebook_distributions(original)
        versions = self._versions(dists)
        current = original
        attempts: list[Attempt] = []
        history: list[str] = []
        changed = False
        # Cells that ran and printed exactly what the reference shows. Patches
        # may not touch them: models sometimes "modernise" working code (for
        # example swapping a random number generator), which silently changes results.
        locked: set[int] = set()

        # What the repaired notebook must reproduce: the author's saved outputs,
        # or, without them, the outputs of the untouched original in its rebuilt environment.
        reconstruction = self._time_travel(original)
        oracle, self._reference = original, (REFERENCE_SAVED if has_stored_outputs(original) else REFERENCE_NONE)
        self._reconstruction = reconstruction
        original_env = ""
        if reconstruction is not None and reconstruction.ok:
            original_env = reconstruction.spec.describe()
            if self._reference == REFERENCE_NONE:
                oracle, self._reference = reconstruction.reference, REFERENCE_RECONSTRUCTED
        label = "reference output (from the original environment)" if self._reference == REFERENCE_RECONSTRUCTED else "saved output"
        # Deprecation warnings printed in the notebook's own era list what is about to break.
        printed_warnings = printed_warning_lines(original)
        if reconstruction is not None and reconstruction.ok:
            for line in printed_warning_lines(reconstruction.reference):
                if line not in printed_warnings:
                    printed_warnings.append(line)

        run: Optional[ExecutionResult] = None
        ran: Optional[NotebookNode] = None  # the notebook version `run` belongs to
        for round_number in range(self.max_attempts + 1):
            # A rejected patch leaves the notebook unchanged: reuse that run instead of repeating it.
            fresh = run is None or ran is not current
            if fresh:
                self._emit("run_start", round=round_number)
                run, ran = self._run(current), current
                self._emit(
                    "run_result",
                    round=round_number,
                    ok=run.ok,
                    failed_cell=run.failed_cell,
                    error=run.error_summary(),
                    duration_s=run.duration_s,
                )

            drift_text: Optional[str] = None
            if run.ok:
                report = compare_notebooks(oracle, run.notebook, rtol=self.rtol)
                if fresh:
                    self._emit("compare", reference=self._reference, **report.to_dict())
                locked |= {c.cell_index for c in report.cells if c.status == Status.MATCH}
                if report.unverifiable:
                    return self._finish(Outcome.RUNS_UNVERIFIED, run.notebook, attempts, report)
                if report.verified:
                    outcome = Outcome.VERIFIED if changed else Outcome.ALREADY_WORKING
                    return self._finish(outcome, run.notebook, attempts, report)
                if not self.fix_drift or round_number == self.max_attempts:
                    return self._finish(Outcome.DRIFT, run.notebook, attempts, report)
                drift_text = _drift_summary(report, label=label)
                error_text = "outputs differ from the saved outputs" if self._reference == REFERENCE_SAVED else (
                    "outputs differ from the original environment's outputs"
                )
                first = report.failures[0]
                focus_cell: Optional[int] = first.cell_index
                first_line = current.cells[first.cell_index].source.strip().splitlines()[:1]
                search_name, search_value = "results changed after upgrade", (first_line[0] if first_line else "")
                search_text = (
                    "This code runs without errors after a library upgrade but prints different numbers "
                    f"({first.detail}):\n{current.cells[first.cell_index].source[:400]}"
                )
            else:
                locked |= matching_cells(oracle, run.notebook, upto=run.failed_cell, rtol=self.rtol)
                error_text = run.error_summary()
                if round_number == self.max_attempts:
                    return self._finish(Outcome.FAILED, current, attempts, None, last_error=error_text)
                focus_cell = run.failed_cell
                search_name, search_value, search_text = run.error_name, run.error_value, error_text

            attempt = Attempt(number=len(attempts) + 1, error=error_text)
            attempts.append(attempt)

            attempt.search_query, attempt.sources = self._research(
                search_name, search_value, search_text, versions, dists
            )

            prompt = repair_prompt(
                current,
                failed_cell=focus_cell,
                error=error_text,
                traceback=run.traceback,
                versions=versions,
                evidence=attempt.sources,
                history=history,
                drift=drift_text,
                locked=sorted(locked),
                original_env=original_env,
                printed_warnings=printed_warnings,
            )

            self._emit("thinking", attempt=attempt.number)
            try:
                reply, attempt.model = self._ask_for_patch(prompt)
                patch: Patch = parse_patch(reply)
                patched = apply_patch(current, patch)
                broken = syntax_errors(current, patched)
                if broken:
                    raise PatchApplyError("the edited code is not valid Python: " + "; ".join(broken))
                touched_locked = sorted(set(changed_cells(current, patched)) & locked)
                if touched_locked:
                    cells = ", ".join(str(i) for i in touched_locked)
                    raise PatchApplyError(
                        f"it changed cell(s) {cells}, which already run and print exactly what the author saved; "
                        "leave those cells unchanged and fix only the cells that fail or print different results"
                    )
            except (PatchParseError, PatchApplyError) as exc:
                attempt.result = f"invalid patch: {exc}"
                history.append(f"attempt {attempt.number}: produced an invalid patch ({exc})")
                self._emit("patch_rejected", attempt=attempt.number, reason=str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 - API errors end the attempt, not the run
                attempt.result = f"model call failed: {exc}"
                history.append(f"attempt {attempt.number}: the model call failed")
                self._emit("warning", message=f"model call failed: {exc}")
                continue

            attempt.explanation = patch.explanation
            attempt.diff = notebook_diff(current, patched, patch.pip_install)
            attempt.pip_install = list(patch.pip_install)
            self._emit(
                "patch",
                attempt=attempt.number,
                explanation=patch.explanation,
                diff=attempt.diff,
                pip_install=attempt.pip_install,
                model=attempt.model,
                edits=[
                    {"cell_index": i, "new_source": patched.cells[i].source}
                    for i in changed_cells(current, patched)
                ],
            )

            if patch.pip_install:
                if self.env is None:
                    attempt.result = "needs packages, but no sandbox environment is configured"
                    history.append(
                        f"attempt {attempt.number}: requested pip install {patch.pip_install}, "
                        "which is not possible here; fix the code instead"
                    )
                    self._emit("install", ok=False, specs=patch.pip_install, output=attempt.result)
                    if not patch.code_changes():
                        continue
                    # The code changes were already applied to `patched`; only the pins are dropped.
                else:
                    installed = self.env.install(patch.pip_install)
                    self._emit("install", ok=installed.ok, specs=installed.specs, output=installed.output[-800:])
                    if not installed.ok:
                        attempt.result = "pip install failed"
                        history.append(
                            f"attempt {attempt.number}: pip install {patch.pip_install} failed: "
                            f"{installed.output[-300:]}"
                        )
                        continue
                    versions = self._versions(dists)

            current = patched
            changed = True
            attempt.result = "applied"
            history.append(f"attempt {attempt.number}: {patch.explanation or 'patch applied'}; it was not enough")

        # The loop always returns; this guards against future edits breaking that.
        return self._finish(Outcome.FAILED, current, attempts, None, last_error="attempts exhausted")

    def _finish(self, outcome, notebook, attempts, report, last_error: str = "") -> RepairResult:
        installed = list(getattr(self.env, "installed", []) or [])
        result = RepairResult(
            outcome=outcome,
            notebook=notebook,
            attempts=attempts,
            comparison=report,
            last_error=last_error,
            installed=installed,
            reference=getattr(self, "_reference", REFERENCE_SAVED),
            reconstruction=getattr(self, "_reconstruction", None),
        )
        self._emit("done", outcome=outcome.value, attempts=len(attempts), last_error=last_error, reference=result.reference)
        return result
