"""Compare a re-executed notebook against the outputs saved in the original.

"It runs again" is not the same as "it is still correct". A notebook that was
repaired by swapping a deprecated call can execute cleanly and quietly produce
different numbers. Every ``.ipynb`` already stores the results its author saw,
so we use those as an oracle: re-run, then check each cell's text output
against what was stored, with a numeric tolerance.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

from nbformat import NotebookNode

DEFAULT_RTOL = 1e-3
DEFAULT_ATOL = 1e-8

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ADDR_RE = re.compile(r"\b0x[0-9a-fA-F]+")  # a memory address, not the "0x400" in "800x400"

# NumPy 2 changed scalar reprs: ``0.5`` became ``np.float64(0.5)``. Notebooks
# saved before that change would otherwise look different for purely cosmetic
# reasons, so these wrappers are removed before comparing.
_NUMPY_SCALAR_RE = re.compile(r"\b(?:np|numpy)\.(?:float|int|uint|bool|complex|str)\w*\(([^()]*)\)")

# Lines whose content depends on the machine or the clock, not on the result
# (including %timeit's "2.87 ms ± 17.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)").
_VOLATILE_LINE_RE = re.compile(
    r"(wall time|cpu times|\bit/s\b|\bs/it\b|\d+%\||ms/step|\bETA\b|elapsed|runtime:"
    r"|per loop|loops each|best of \d+)",
    re.IGNORECASE,
)

# Plots are not compared, and a figure's text stand-in ("<Figure size 576x288 with 1 Axes>")
# changes with matplotlib's default resolution, not with the result.
_FIGURE_LINE_RE = re.compile(r"^<Figure size [\d.]+x[\d.]+ with \d+ Axes>$")

# Formatting that libraries changed without changing any value. Each pair maps an
# old or new spelling to one canonical form; numbers are never touched.
_COSMETIC_SUBS = (
    # pandas 2.0 removed Int64Index/UInt64Index/Float64Index: the same index now prints as Index.
    (re.compile(r"\b(?:Int64|UInt64|Float64)Index\("), "Index("),
    # pandas 3.0 stores text as its own string dtype instead of object.
    (re.compile(r"dtype: str\b"), "dtype: object"),
    (re.compile(r"dtype='str'"), "dtype='object'"),
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
    for pattern, replacement in _COSMETIC_SUBS:
        text = pattern.sub(replacement, text)
    lines: list[str] = []
    for line in text.split("\n"):
        if _VOLATILE_LINE_RE.search(line):
            continue
        line = _ADDR_RE.sub("0xADDR", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line and not _FIGURE_LINE_RE.match(line):
            lines.append(line)
    return "\n".join(lines)


# ---------- scikit-learn estimator printouts ----------
# Since scikit-learn 0.23 an estimator prints only the parameters that differ from
# their defaults; older versions printed every parameter. The two describe the
# same model when every parameter the new printout shows appears, with the same
# value, in the old one.
_ESTIMATOR_RE = re.compile(r"^([A-Z][A-Za-z0-9_]*)\((.*)\)$", re.S)


def _split_args(text: str) -> Optional[list[str]]:
    parts, depth, quote, start = [], 0, "", 0
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return None
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    if depth or quote:
        return None
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _estimator_params(text: str) -> Optional[tuple[str, dict[str, str]]]:
    m = _ESTIMATOR_RE.match(text.strip())
    if not m:
        return None
    args = _split_args(m.group(2))
    if args is None:
        return None
    params: dict[str, str] = {}
    for arg in args:
        key, sep, value = arg.partition("=")
        if not sep or not key.strip().isidentifier():
            return None
        params[key.strip()] = value.strip()
    return m.group(1), params


def _same_value(old: str, new: str) -> bool:
    """Equal parameter values, allowing for nested estimators and for long lists the
    new printout shortens with "..." (every element it does show must be in the old one, in order)."""
    a, b = re.sub(r"\s+", "", old), re.sub(r"\s+", "", new)
    if a == b or _same_estimator(old, new):
        return True
    if "..." not in b:
        return False
    pieces = b.split("...")
    if not a.startswith(pieces[0]) or not a.endswith(pieces[-1]):
        return False
    position = len(pieces[0])
    for piece in pieces[1:-1]:
        found = a.find(piece, position)
        if found < 0:
            return False
        position = found + len(piece)
    return position <= len(a) - len(pieces[-1])


def _same_estimator(old: str, new: str) -> bool:
    """True when ``new`` (changed-only printout) describes the model ``old`` printed in full."""
    a, b = _estimator_params(old.replace("\n", " ")), _estimator_params(new.replace("\n", " "))
    if a is None or b is None or a[0] != b[0] or not b[1].keys() <= a[1].keys():
        return False
    return all(_same_value(a[1][key], value) for key, value in b[1].items())


def _same_estimator_output(old: str, new: str) -> bool:
    """An estimator printout, possibly after or before identical lines (progress messages)."""
    a, b = old.split("\n"), new.split("\n")
    while a and b and a[0] == b[0]:
        a, b = a[1:], b[1:]
    while a and b and a[-1] == b[-1]:
        a, b = a[:-1], b[:-1]
    return bool(a) and bool(b) and _same_estimator("\n".join(a), "\n".join(b))


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
    if _same_estimator_output(a, b):
        return Status.MATCH, 0.0, "same model; newer scikit-learn prints only the parameters that were set"

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


# ---------- notebooks that print their own code ----------
# Some notebooks print code they evaluate, e.g. display('df1.append(df2)') shows the
# expression above its result. Fixing that expression necessarily changes the label.
_CODE_STRING_RE = re.compile(r"'((?:[^'\\\n]|\\.)*)'" + r'|"((?:[^"\\\n]|\\.)*)"')


def _code_strings(source: str) -> list[str]:
    found = []
    for match in _CODE_STRING_RE.finditer(source):
        value = match.group(1) if match.group(1) is not None else match.group(2)
        if value and "(" in value:  # an expression, not data such as a column name
            found.append(value)
    return found


def _renamed_code_strings(old_source: str, new_source: str) -> list[tuple[str, str]]:
    old, new = _code_strings(old_source), _code_strings(new_source)
    pairs = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag == "replace" and i2 - i1 == j2 - j1:
            pairs.extend(zip(old[i1:i2], new[j1:j2]))
    return pairs


def _explain_own_code(stored: str, fresh: str, old_source: str, new_source: str, rtol: float, atol: float) -> Optional[str]:
    """If the only differences are code strings the fix rewrote and the notebook prints, say which."""
    pairs = [(o, n) for o, n in _renamed_code_strings(old_source, new_source) if o in stored]
    if not pairs:
        return None
    for old, new in pairs:
        stored = stored.replace(old, new)
    if compare_text(stored, fresh, rtol=rtol, atol=atol)[0] != Status.MATCH:
        return None
    return "; ".join(f"the notebook prints its own code, which the fix changed: {o!r} -> {n!r}" for o, n in pairs)


# ---------- cells that only report the environment ----------
# A cell such as "%watermark -v -p numpy" prints library versions and the date. It
# describes the machine, not a result, and can never match on a newer setup.
_ENV_REPORT_LINE_RE = re.compile(
    r"^\s*(?:%load_ext\s+watermark\b|%watermark\b|%version_information\b|%load_ext\s+version_information\b"
    r"|!\s*(?:pip|conda)\s+(?:freeze|list)\b|!\s*python3?\s+(?:-V|--version)\b|!\s*nvidia-smi\b)"
)


def environment_report(source: str) -> bool:
    """True when every line of a cell only reports versions, the date or the hardware."""
    lines = [line for line in source.splitlines() if line.strip() and not line.strip().startswith("#")]
    return bool(lines) and all(_ENV_REPORT_LINE_RE.match(line) for line in lines)


def _error_name(cell: Any) -> str:
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            return str(output.get("ename") or "")
    return ""


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

        stored_error, fresh_error = _error_name(before), _error_name(after)
        if environment_report(before.source):
            status, rel, detail = Status.SKIPPED, 0.0, "reports versions and the date, not a result; not compared"
        elif stored_error and fresh_error != stored_error:
            status, rel, detail = Status.CHANGED, 0.0, (
                f"the original raised {stored_error} here; the re-run "
                + (f"raised {fresh_error}" if fresh_error else "raised nothing")
            )
        elif stored_error and not has_stored and not has_fresh:
            # Shown on purpose by the author, and it still happens.
            status, rel, detail = Status.MATCH, 0.0, f"raised {stored_error}, as in the original"
        elif not has_stored and not has_fresh:
            status, rel, detail = Status.SKIPPED, 0.0, ""
        elif has_stored and not has_fresh:
            status, rel, detail = Status.MISSING, 0.0, "the re-run produced no output for this cell"
        elif has_fresh and not has_stored:
            status, rel, detail = Status.EXTRA, 0.0, "the original had no output for this cell"
        else:
            status, rel, detail = compare_text(stored, fresh, rtol=rtol, atol=atol)
            if status in (Status.DRIFT, Status.CHANGED) and before.source != after.source:
                explained = _explain_own_code(stored, fresh, before.source, after.source, rtol, atol)
                if explained:
                    status, rel, detail = Status.MATCH, 0.0, explained

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
