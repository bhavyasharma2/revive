"""Compare a re-executed notebook against the outputs saved in the original.

"It runs again" is not the same as "it is still correct". A notebook that was
repaired by swapping a deprecated call can execute cleanly and quietly produce
different numbers. Every ``.ipynb`` already stores the results its author saw,
so we use those as an oracle: re-run, then check each cell's text output
against what was stored, with a numeric tolerance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

from nbformat import NotebookNode

DEFAULT_RTOL = 1e-3
DEFAULT_ATOL = 1e-8

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")

# NumPy 2 changed scalar reprs: ``0.5`` became ``np.float64(0.5)``. Notebooks
# saved before that change would otherwise look different for purely cosmetic
# reasons, so these wrappers are removed before comparing.
_NUMPY_SCALAR_RE = re.compile(r"\b(?:np|numpy)\.(?:float|int|uint|bool|complex|str)\w*\(([^()]*)\)")

# Lines whose content depends on the machine or the clock, not on the result.
_VOLATILE_LINE_RE = re.compile(
    r"(wall time|cpu times|\bit/s\b|\bs/it\b|\d+%\||ms/step|\bETA\b|elapsed|runtime:)",
    re.IGNORECASE,
)

# A number that is not glued to a letter, digit, underscore or dot on its left.
_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?!\w)"
)


class Status(str, Enum):
    """Result of comparing one cell."""

    MATCH = "match"        # output agrees with the stored one within tolerance
    DRIFT = "drift"        # same shape of output, numbers differ beyond tolerance
    CHANGED = "changed"    # the text itself differs (not just the numbers)
    MISSING = "missing"    # the original had output, the re-run produced none
    EXTRA = "extra"        # the re-run produced output, the original had none
    SKIPPED = "skipped"    # nothing comparable on either side


FAILING = (Status.DRIFT, Status.CHANGED, Status.MISSING)
COMPARED = (Status.MATCH, Status.DRIFT, Status.CHANGED, Status.MISSING)


@dataclass
class CellComparison:
    cell_index: int
    status: Status
    max_rel_diff: float = 0.0
    detail: str = ""
    stored: str = ""
    fresh: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_index": self.cell_index,
            "status": self.status.value,
            "max_rel_diff": self.max_rel_diff,
            "detail": self.detail,
            "stored": self.stored,
            "fresh": self.fresh,
        }


@dataclass
class ComparisonReport:
    cells: list[CellComparison] = field(default_factory=list)
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL

    def count(self, status: Status) -> int:
        return sum(1 for c in self.cells if c.status == status)

    @property
    def failures(self) -> list[CellComparison]:
        return [c for c in self.cells if c.status in FAILING]

    @property
    def compared(self) -> int:
        """Number of cells that had stored output to check against."""
        return sum(1 for c in self.cells if c.status in COMPARED)

    @property
    def unverifiable(self) -> bool:
        """True when the original notebook had no comparable saved output at all."""
        return self.compared == 0

    @property
    def verified(self) -> bool:
        """True when there was something to check and every check passed."""
        return self.compared > 0 and not self.failures

    @property
    def match_rate(self) -> float:
        return self.count(Status.MATCH) / self.compared if self.compared else 0.0

    def summary(self) -> str:
        if self.unverifiable:
            return "no saved outputs in the original notebook to verify against"
        verdict = "verified" if self.verified else "NOT verified"
        return (
            f"{verdict}: {self.count(Status.MATCH)}/{self.compared} cells match "
            f"({self.count(Status.DRIFT)} drift, {self.count(Status.CHANGED)} changed, "
            f"{self.count(Status.MISSING)} missing)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "unverifiable": self.unverifiable,
            "compared": self.compared,
            "match_rate": self.match_rate,
            "rtol": self.rtol,
            "atol": self.atol,
            "summary": self.summary(),
            "cells": [c.to_dict() for c in self.cells],
        }


def _join(text: Any) -> str:
    """nbformat may hold multiline text as a string or as a list of strings."""
    if isinstance(text, (list, tuple)):
        return "".join(str(part) for part in text)
    return "" if text is None else str(text)


def comparable_text(outputs: Iterable[Any]) -> str:
    """Concatenate the parts of a cell's outputs that are worth comparing.

    Kept: stdout streams and the ``text/plain`` form of results and displays.
    Ignored: stderr (warnings differ between library versions), errors and
    images (plots are not byte-stable).
    """
    pieces: list[str] = []
    for output in outputs:
        kind = output.get("output_type")
        if kind == "stream":
            if output.get("name") == "stdout":
                pieces.append(_join(output.get("text")))
        elif kind in ("execute_result", "display_data"):
            data = output.get("data", {})
            if "text/plain" in data:
                pieces.append(_join(data["text/plain"]))
    text = ""
    for piece in pieces:
        if text and not text.endswith("\n"):
            text += "\n"
        text += piece
    return text


def normalize_text(text: str) -> str:
    """Strip everything that varies run to run but does not change the result."""
    text = _ANSI_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _NUMPY_SCALAR_RE.sub(r"\1", text)
    lines: list[str] = []
    for line in text.split("\n"):
        if _VOLATILE_LINE_RE.search(line):
            continue
        line = _ADDR_RE.sub("0xADDR", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def split_numbers(text: str) -> tuple[str, list[float]]:
    """Return ``text`` with every number replaced by ``#`` plus the numbers found."""
    skeleton, tokens = _split_tokens(text)
    return skeleton, [float(t) for t in tokens]


def _split_tokens(text: str) -> tuple[str, list[str]]:
    tokens: list[str] = []

    def _take(match: re.Match) -> str:
        tokens.append(match.group(0))
        return "#"

    return _NUMBER_RE.sub(_take, text), tokens


# Printed with this many decimals or more, a difference of one unit in the last
# printed digit is treated as rounding noise (e.g. 0.182 vs 0.181 from two
# solvers that agree to 1e-4). Coarser prints get no such allowance.
LAST_DIGIT_MIN_DECIMALS = 3


def _last_digit_tolerance(token: str) -> float:
    mantissa = token.lower().split("e")[0]
    if "." not in mantissa:
        return 0.0
    decimals = len(mantissa.split(".")[1])
    if decimals < LAST_DIGIT_MIN_DECIMALS:
        return 0.0
    exponent = int(token.lower().split("e")[1]) if "e" in token.lower() else 0
    return 10.0 ** (exponent - decimals) * 1.000001


def _rel_diff(a: float, b: float) -> float:
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b))
    if scale == 0 or math.isinf(scale):
        return math.inf
    return abs(a - b) / scale


def _first_difference(a: str, b: str) -> str:
    for left, right in zip(a.split("\n"), b.split("\n")):
        if left != right:
            return f"stored: {left[:160]!r} / fresh: {right[:160]!r}"
    return "outputs have a different number of lines"


def compare_text(
    stored: str,
    fresh: str,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> tuple[Status, float, str]:
    """Compare two already-extracted outputs. Returns (status, max_rel_diff, detail)."""
    a = normalize_text(stored)
    b = normalize_text(fresh)
    if a == b:
        return Status.MATCH, 0.0, ""

    skeleton_a, tokens_a = _split_tokens(a)
    skeleton_b, tokens_b = _split_tokens(b)
    if skeleton_a != skeleton_b or len(tokens_a) != len(tokens_b):
        return Status.CHANGED, 0.0, _first_difference(a, b)

    worst = 0.0
    drifted: Optional[tuple[float, float, float]] = None
    for token_a, token_b in zip(tokens_a, tokens_b):
        x, y = float(token_a), float(token_b)
        rel = _rel_diff(x, y)
        worst = max(worst, rel)
        allowed = max(atol, _last_digit_tolerance(token_a))
        if drifted is None and not math.isclose(x, y, rel_tol=rtol, abs_tol=allowed):
            drifted = (x, y, rel)

    if drifted is None:
        return Status.MATCH, worst, ""
    x, y, rel = drifted
    return Status.DRIFT, worst, f"{x:g} -> {y:g} (relative difference {rel:.2e})"


def matching_cells(
    original: NotebookNode,
    executed: NotebookNode,
    *,
    upto: Optional[int] = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> set[int]:
    """Code cells that ran and printed what the author saved (only cells with saved output count).

    ``upto`` limits the check to cells before that index, for a run that
    stopped at a failing cell (later cells never executed).
    """
    found: set[int] = set()
    limit = len(original.cells) if upto is None else min(upto, len(original.cells))
    for index in range(min(limit, len(executed.cells))):
        before, after = original.cells[index], executed.cells[index]
        if before.cell_type != "code" or after.cell_type != "code":
            continue
        stored = comparable_text(before.get("outputs", []))
        if not normalize_text(stored):
            continue
        fresh = comparable_text(after.get("outputs", []))
        if not normalize_text(fresh):
            continue
        if compare_text(stored, fresh, rtol=rtol, atol=atol)[0] == Status.MATCH:
            found.add(index)
    return found


def compare_notebooks(
    original: NotebookNode,
    executed: NotebookNode,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> ComparisonReport:
    """Compare every code cell of a fully executed notebook with the original.

    ``original`` carries the saved outputs; ``executed`` is the result of a
    successful run. Both must have the same cells in the same order, which the
    patch format guarantees by only ever editing a cell's source in place.
    """
    if len(original.cells) != len(executed.cells):
        raise ValueError(
            f"notebooks have different cell counts ({len(original.cells)} vs {len(executed.cells)})"
        )

    report = ComparisonReport(rtol=rtol, atol=atol)
    for index, (before, after) in enumerate(zip(original.cells, executed.cells)):
        if before.cell_type != after.cell_type:
            raise ValueError(f"cell {index} changed type ({before.cell_type} vs {after.cell_type})")
        if before.cell_type != "code":
            continue

        stored = comparable_text(before.get("outputs", []))
        fresh = comparable_text(after.get("outputs", []))
        has_stored = bool(normalize_text(stored))
        has_fresh = bool(normalize_text(fresh))

        if not has_stored and not has_fresh:
            status, rel, detail = Status.SKIPPED, 0.0, ""
        elif has_stored and not has_fresh:
            status, rel, detail = Status.MISSING, 0.0, "the re-run produced no output for this cell"
        elif has_fresh and not has_stored:
            status, rel, detail = Status.EXTRA, 0.0, "the original had no output for this cell"
        else:
            status, rel, detail = compare_text(stored, fresh, rtol=rtol, atol=atol)

        report.cells.append(
            CellComparison(
                cell_index=index,
                status=status,
                max_rel_diff=rel,
                detail=detail,
                stored=normalize_text(stored)[:2000],
                fresh=normalize_text(fresh)[:2000],
            )
        )
    return report
