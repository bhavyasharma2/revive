"""Execute a notebook in a fresh kernel and report exactly where it breaks."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

from jupyter_client.kernelspec import NoSuchKernel
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError
from nbformat import NotebookNode

from .notebook import expected_errors, strip_outputs

# nbclient's per-cell switch for "this cell is expected to raise".
RAISES_TAG = "raises-exception"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Set REVIVE_KERNEL_LOGS=1 to see the kernel process's own log lines (debugging only).
_SHOW_KERNEL_LOGS = os.environ.get("REVIVE_KERNEL_LOGS") == "1"


def _use_selector_loop_on_windows() -> None:
    """pyzmq needs a selector event loop; Windows defaults to Proactor and warns."""
    if sys.platform != "win32":
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # policies are deprecated in 3.14+
        policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if policy is not None and not isinstance(asyncio.get_event_loop_policy(), policy):
            asyncio.set_event_loop_policy(policy())


class RunnerError(RuntimeError):
    """The kernel could not be started or reached (infrastructure, not a notebook bug)."""


@dataclass
class ExecutionResult:
    """Outcome of one full execution attempt."""

    ok: bool
    notebook: NotebookNode
    failed_cell: Optional[int] = None
    error_name: Optional[str] = None
    error_value: Optional[str] = None
    traceback: Optional[str] = None
    duration_s: float = 0.0
    timed_out: bool = False
    kernel_died: bool = False

    def error_summary(self) -> str:
        """One-line description of the failure, or ``"ok"``."""
        if self.ok:
            return "ok"
        where = f"cell {self.failed_cell}" if self.failed_cell is not None else "unknown cell"
        what = self.error_name or "error"
        detail = f": {self.error_value}" if self.error_value else ""
        return f"{what} in {where}{detail}"


def _clean(text: Optional[str]) -> str:
    return _ANSI_RE.sub("", text or "")


def _find_error(nb: NotebookNode, expected: Optional[dict[int, str]] = None):
    """Locate the first unexpected error output; returns (cell_index, ename, evalue, traceback) or None.

    A cell listed in ``expected`` that raised the same exception as in the saved
    notebook is behaving as its author intended and is skipped.
    """
    expected = expected or {}
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                if index in expected and output.get("ename") == expected[index]:
                    continue
                trace = _clean("\n".join(output.get("traceback", [])))
                return index, output.get("ename"), output.get("evalue"), trace
    return None


def _startup_timeout() -> int:
    """Seconds to wait for a kernel to start. Slow hosts (a fraction of a CPU) need more:
    set REVIVE_KERNEL_STARTUP_TIMEOUT."""
    try:
        return max(10, int(os.environ.get("REVIVE_KERNEL_STARTUP_TIMEOUT", "60")))
    except ValueError:
        return 60


def execute_notebook(
    nb: NotebookNode,
    *,
    cwd: Optional[Union[str, Path]] = None,
    timeout: int = 120,
    startup_timeout: Optional[int] = None,
    kernel_name: str = "python3",
    kernel_manager_factory: Optional[Callable[[], Any]] = None,
) -> ExecutionResult:
    """Run every cell of ``nb`` top to bottom in a new kernel.

    The input notebook is never modified: execution happens on a copy whose
    stored outputs have been cleared, so stale outputs can never be mistaken
    for fresh ones. Execution stops at the first failing cell.

    ``timeout`` is the per-cell limit in seconds. ``cwd`` is the kernel's
    working directory (so relative paths to datasets resolve).
    ``kernel_manager_factory`` returns a fresh kernel manager; use it to run
    the notebook inside an isolated environment (see ``revive.env``).
    """
    _use_selector_loop_on_windows()
    # Cells whose saved output is an error were meant to raise: let execution continue past them.
    expected = expected_errors(nb)
    work = strip_outputs(nb)
    added_tags: list[int] = []
    for index in expected:
        tags = work.cells[index].metadata.setdefault("tags", [])
        if RAISES_TAG not in tags:
            tags.append(RAISES_TAG)
            added_tags.append(index)

    def untag() -> None:
        for index in added_tags:
            tags = work.cells[index].metadata.get("tags", [])
            if RAISES_TAG in tags:
                tags.remove(RAISES_TAG)
            if not tags:
                work.cells[index].metadata.pop("tags", None)
    state: dict[str, Optional[int]] = {"current": None}

    def _on_cell_execute(*, cell, cell_index, **_kwargs):
        state["current"] = cell_index

    kwargs = dict(
        timeout=timeout,
        startup_timeout=startup_timeout if startup_timeout is not None else _startup_timeout(),
        kernel_name=kernel_name,
        allow_errors=False,
        record_timing=False,
        on_cell_execute=_on_cell_execute,
    )
    if cwd is not None:
        kwargs["resources"] = {"metadata": {"path": str(Path(cwd).resolve())}}

    km = kernel_manager_factory() if kernel_manager_factory is not None else None
    client = NotebookClient(work, km=km, **kwargs)
    started = time.monotonic()

    def elapsed() -> float:
        return round(time.monotonic() - started, 3)

    try:
        # cleanup_kc=True: always shut the kernel down afterwards. nbclient only
        # does this by default when it created the kernel manager itself.
        # Cell output (including stderr) travels over ZeroMQ, not the kernel
        # process's stderr, so silencing the latter only hides startup chatter.
        launch = {} if _SHOW_KERNEL_LOGS else {"stderr": subprocess.DEVNULL}
        client.execute(cleanup_kc=True, **launch)
    except CellTimeoutError as exc:
        untag()
        return ExecutionResult(
            ok=False,
            notebook=work,
            failed_cell=state["current"],
            error_name="CellTimeoutError",
            error_value=f"cell exceeded the {timeout}s time limit",
            traceback=_clean(str(exc)),
            duration_s=elapsed(),
            timed_out=True,
        )
    except DeadKernelError as exc:
        untag()
        return ExecutionResult(
            ok=False,
            notebook=work,
            failed_cell=state["current"],
            error_name="DeadKernelError",
            error_value="the kernel died (out of memory or a native crash)",
            traceback=_clean(str(exc)),
            duration_s=elapsed(),
            kernel_died=True,
        )
    except CellExecutionError as exc:
        untag()
        found = _find_error(work, expected)
        if found is not None:
            index, ename, evalue, trace = found
        else:  # defensive: fall back to what the exception itself carries
            index = state["current"]
            ename = getattr(exc, "ename", None) or "CellExecutionError"
            evalue = getattr(exc, "evalue", None)
            trace = _clean(getattr(exc, "traceback", None) or str(exc))
        return ExecutionResult(
            ok=False,
            notebook=work,
            failed_cell=index,
            error_name=ename,
            error_value=evalue,
            traceback=trace,
            duration_s=elapsed(),
        )
    except NoSuchKernel as exc:
        raise RunnerError(f"kernel '{kernel_name}' is not installed: {exc}") from exc
    except RuntimeError as exc:
        raise RunnerError(f"could not run the notebook kernel: {exc}") from exc

    untag()
    # An intentional error cell that now raises something else (or a removed API) is a real failure.
    found = _find_error(work, expected)
    if found is not None:
        index, ename, evalue, trace = found
        return ExecutionResult(
            ok=False, notebook=work, failed_cell=index, error_name=ename,
            error_value=evalue, traceback=trace, duration_s=elapsed(),
        )
    return ExecutionResult(ok=True, notebook=work, duration_s=elapsed())
