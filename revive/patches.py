"""The fix format the model must produce, and safe ways to parse and apply it.

The model answers with JSON::

    {
      "explanation": "np.float was removed in NumPy 1.24; use np.float64",
      "pip_install": ["scikit-learn==1.3.2"],
      "replacements": [{"cell_index": 3, "find": "np.float)", "replace": "np.float64)"}],
      "edits": [{"cell_index": 5, "new_source": "<the complete new source of cell 5>"}]
    }

``replacements`` (preferred) change one exact snippet of a cell and leave the
rest of the cell byte-for-byte untouched. That matters: when a model retypes
a whole cell inside JSON it can silently corrupt unrelated code, such as the
escapes in a string literal, and the notebook then runs but computes something
else. ``edits`` replace a whole cell and are kept for larger rewrites.

Patches never add, remove or reorder cells, so cell indices stay valid across
attempts and outputs can be compared cell by cell.
"""

from __future__ import annotations

import copy
import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import nbformat
from nbformat import NotebookNode

from .textutil import strip_reasoning

MAX_PIP_SPECS = 10
MAX_SOURCE_CHARS = 200_000
MAX_CHANGES = 50

# One requirement: name, optional extras, optional single version specifier.
# No URLs, no flags, no environment markers, no whitespace: the model's output
# ends up in a pip command line, so only a tight allowlist is accepted.
_PIP_SPEC_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r"(?:\[[A-Za-z0-9,._-]+\])?"
    r"(?:(?:==|>=|<=|~=|!=|<|>)[A-Za-z0-9.*+!_-]+)?$"
)


class PatchParseError(ValueError):
    """The model's answer could not be turned into a valid patch."""


class PatchApplyError(ValueError):
    """A syntactically valid patch does not fit the notebook."""


@dataclass
class CellEdit:
    cell_index: int
    new_source: str


@dataclass
class CellReplace:
    cell_index: int
    find: str
    replace: str


@dataclass
class Patch:
    edits: list[CellEdit] = field(default_factory=list)
    pip_install: list[str] = field(default_factory=list)
    explanation: str = ""
    replacements: list[CellReplace] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.edits and not self.pip_install and not self.replacements

    def code_changes(self) -> bool:
        return bool(self.edits or self.replacements)


def extract_json_object(text: str) -> dict:
    """Find the first JSON object in a model response.

    Tolerates reasoning blocks, prose around the JSON and ```json fences.
    """
    cleaned = strip_reasoning(text)
    candidates = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)]
    candidates.append(cleaned)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                obj, _end = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise PatchParseError("no JSON object found in the model response")


def validate_pip_spec(spec: object) -> str:
    """Return a cleaned requirement string or raise ``PatchParseError``."""
    if not isinstance(spec, str):
        raise PatchParseError(f"pip requirement must be a string, got {type(spec).__name__}")
    cleaned = re.sub(r"\s+", "", spec)
    if not _PIP_SPEC_RE.match(cleaned):
        raise PatchParseError(f"rejected pip requirement: {spec!r}")
    return cleaned


def _cell_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PatchParseError(f"invalid cell_index: {value!r}")
    return value


def _text(value: object, name: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise PatchParseError(f"{name} must be a string")
    if len(value) > MAX_SOURCE_CHARS:
        raise PatchParseError(f"{name} is too large")
    if not allow_empty and not value:
        raise PatchParseError(f"{name} must not be empty")
    return value


def _list(obj: dict, key: str) -> list:
    value = obj.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise PatchParseError(f"'{key}' must be a list")
    return value


def parse_patch(text: str) -> Patch:
    """Turn a raw model response into a validated ``Patch``."""
    obj = extract_json_object(text)

    edits: list[CellEdit] = []
    for item in _list(obj, "edits"):
        if not isinstance(item, dict):
            raise PatchParseError("each edit must be an object")
        edits.append(CellEdit(_cell_index(item.get("cell_index")), _text(item.get("new_source"), "new_source")))

    replacements: list[CellReplace] = []
    for item in _list(obj, "replacements"):
        if not isinstance(item, dict):
            raise PatchParseError("each replacement must be an object")
        replacements.append(
            CellReplace(
                _cell_index(item.get("cell_index")),
                _text(item.get("find"), "find", allow_empty=False),
                _text(item.get("replace"), "replace"),
            )
        )

    if len(edits) + len(replacements) > MAX_CHANGES:
        raise PatchParseError(f"too many changes in one patch (max {MAX_CHANGES})")

    raw_pip = _list(obj, "pip_install")
    if len(raw_pip) > MAX_PIP_SPECS:
        raise PatchParseError(f"too many pip requirements (max {MAX_PIP_SPECS})")
    pip_install = [validate_pip_spec(spec) for spec in raw_pip]

    explanation = obj.get("explanation", "")
    if not isinstance(explanation, str):
        explanation = str(explanation)

    patch = Patch(edits=edits, pip_install=pip_install, explanation=explanation.strip(), replacements=replacements)
    if patch.is_empty():
        raise PatchParseError("the patch contains no changes and no pip requirements")
    return patch


def _code_cell(nb: NotebookNode, index: int) -> NotebookNode:
    if index >= len(nb.cells):
        raise PatchApplyError(f"cell_index {index} is out of range (notebook has {len(nb.cells)} cells)")
    cell = nb.cells[index]
    if cell.cell_type != "code":
        raise PatchApplyError(f"cell {index} is a {cell.cell_type} cell, not code")
    return cell


def _variants(rep: CellReplace) -> list[tuple[str, str]]:
    """Ways to read the snippet, most literal first.

    1. As given. 2. Without stray surrounding whitespace. 3. With doubled
    backslashes collapsed: models often over-escape inside JSON and send
    ``\\\\s`` where the cell (a raw string) contains ``\\s``.
    """
    variants = [(rep.find, rep.replace)]
    stripped = rep.find.strip()
    if stripped and stripped != rep.find:
        variants.append((stripped, rep.replace.strip()))
    for find, replace in list(variants):
        if "\\\\" in find:
            variants.append((find.replace("\\\\", "\\"), replace.replace("\\\\", "\\")))
    return variants


def _python_part(source: str) -> str:
    """Cell source without IPython-only lines (magics, shell escapes), for syntax checking."""
    return "\n".join("" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines())


def syntax_errors(before: NotebookNode, after: NotebookNode) -> list[str]:
    """Edited cells that were valid Python before and are not any more."""
    problems: list[str] = []
    for index in changed_cells(before, after):
        try:
            compile(_python_part(before.cells[index].source), f"cell {index}", "exec")
        except SyntaxError:
            continue  # it did not parse before either (IPython syntax we do not model); do not judge
        try:
            compile(_python_part(after.cells[index].source), f"cell {index}", "exec")
        except SyntaxError as exc:
            problems.append(f"cell {index}: {exc.msg} (line {exc.lineno}): {(exc.text or '').strip()[:120]}")
    return problems


def _locate(nb: NotebookNode, rep: CellReplace) -> tuple[int, str, str]:
    """Find where a replacement applies: (cell index, find, replace).

    The snippet must occur exactly once in the named code cell. Models often
    miscount cell positions by one, so if the snippet is not in the named cell
    but occurs exactly once in the whole notebook, that unique cell is used.
    Anything ambiguous is rejected with a reason the model can act on.
    """
    named = nb.cells[rep.cell_index] if rep.cell_index < len(nb.cells) else None
    if named is not None and named.cell_type == "code":
        for find, replace in _variants(rep):
            count = named.source.count(find)
            if count == 1:
                return rep.cell_index, find, replace
            if count > 1:
                raise PatchApplyError(
                    f"replacement for cell {rep.cell_index}: the text {rep.find[:80]!r} appears {count} times in that cell; "
                    "include more surrounding code so it is unique"
                )

    for find, replace in _variants(rep):
        hits = [
            (index, cell.source.count(find))
            for index, cell in enumerate(nb.cells)
            if cell.cell_type == "code" and find in cell.source
        ]
        if len(hits) == 1 and hits[0][1] == 1:
            return hits[0][0], find, replace
        if hits:
            raise PatchApplyError(
                f"replacement for cell {rep.cell_index}: the text {rep.find[:80]!r} is not in that cell and appears "
                f"in several other cells ({', '.join(str(i) for i, _ in hits)}); use the right cell_index and a unique snippet"
            )

    if named is None:
        raise PatchApplyError(f"cell_index {rep.cell_index} is out of range (notebook has {len(nb.cells)} cells)")
    raise PatchApplyError(
        f"replacement for cell {rep.cell_index}: the text {rep.find[:80]!r} does not appear anywhere in the notebook; "
        "copy 'find' exactly from the cell source"
    )


def apply_patch(nb: NotebookNode, patch: Patch) -> NotebookNode:
    """Return a copy of ``nb`` with the patch's code changes applied.

    Whole-cell edits are applied first, then replacements. A replacement for
    a cell that also has a whole-cell edit is ignored (the edit is the cell's
    complete new source). A replacement's ``find`` must occur exactly once in
    its cell, ignoring stray whitespace around it; otherwise the patch is
    rejected and the reason is sent back to the model. ``pip_install`` is handled by the sandbox environment, not here.
    The input notebook is never modified.
    """
    patched = copy.deepcopy(nb)
    touched: set[int] = set()

    for edit in patch.edits:
        cell = _code_cell(patched, edit.cell_index)
        cell.source = edit.new_source
        touched.add(edit.cell_index)

    edited = {edit.cell_index for edit in patch.edits}
    for rep in patch.replacements:
        if rep.cell_index in edited:
            # The whole-cell edit already states the cell's complete new source.
            continue
        index, find, replace = _locate(patched, rep)
        cell = patched.cells[index]
        cell.source = cell.source.replace(find, replace, 1)
        touched.add(index)

    for index in touched:
        patched.cells[index].outputs = []
        patched.cells[index].execution_count = None
    nbformat.validate(patched)
    return patched


def changed_cells(before: NotebookNode, after: NotebookNode) -> list[int]:
    """Indices of code cells whose source differs between two versions of a notebook."""
    return [
        index
        for index, (a, b) in enumerate(zip(before.cells, after.cells))
        if a.cell_type == "code" and a.source != b.source
    ]


def notebook_diff(before: NotebookNode, after: NotebookNode, pip_install: Optional[list[str]] = None) -> str:
    """Unified diff of every code cell that changed, plus any pip installs."""
    chunks: list[str] = []
    for index in changed_cells(before, after):
        diff = difflib.unified_diff(
            before.cells[index].source.splitlines(),
            after.cells[index].source.splitlines(),
            fromfile=f"cell {index} (before)",
            tofile=f"cell {index} (after)",
            lineterm="",
        )
        chunks.append("\n".join(diff))
    if pip_install:
        chunks.append("pip install " + " ".join(pip_install))
    return "\n\n".join(chunk for chunk in chunks if chunk)


def patch_diff(nb: NotebookNode, patch: Patch) -> str:
    """Human-readable unified diff of what the patch changes in ``nb``."""
    try:
        after = apply_patch(nb, patch)
    except PatchApplyError:
        # Fall back to describing the requested whole-cell edits only.
        after = copy.deepcopy(nb)
        for edit in patch.edits:
            if 0 <= edit.cell_index < len(after.cells) and after.cells[edit.cell_index].cell_type == "code":
                after.cells[edit.cell_index].source = edit.new_source
    return notebook_diff(nb, after, patch.pip_install)


# ---------- patches that fake the check instead of fixing the code ----------

_PRINT_HOOKS = ("__repr__", "__str__", "_repr_html_", "_repr_pretty_", "__format__")
_MONKEY_PATCH_RE = re.compile(r"^\s*[A-Za-z_][\w.]*\.[A-Za-z_]\w*\s*=\s*(?:lambda\b|types\.MethodType\b|functools\.partial\b)", re.M)
# A new subclass of a pandas/NumPy/scikit-learn type, typically to bring back a removed method.
_LIBRARY_SUBCLASS_RE = re.compile(r"^\s*class\s+\w+\s*\(\s*(?:pd|pandas|np|numpy|sklearn)\.[\w.]+\s*[,)]", re.M)
_STRING_RE = re.compile(r"""(?:[rRbBuUfF]{0,2})(?:'((?:[^'\\\n]|\\.)*)'|"((?:[^"\\\n]|\\.)*)")""")
MIN_PASTED_CHARS = 4


def gaming_reasons(original: NotebookNode, before: NotebookNode, after: NotebookNode, reference: NotebookNode) -> list[str]:
    """Why a patch looks like it fakes the output check rather than fixing the code.

    Three signs, each checked only in lines the patch added:
    - a string literal copied from the expected output that the original notebook never contained
      (for example printing "Int64Index(" because old pandas did);
    - a new __repr__/__str__-style hook, which changes how results print, not what they are;
    - re-attaching a method to an object (``df.append = lambda ...``), which hides a removed API.
    """
    from .compare import comparable_text  # local import: compare does not depend on patches

    original_source = "\n".join(c.source for c in original.cells if c.cell_type == "code")
    expected = "\n".join(comparable_text(c.get("outputs", [])) for c in reference.cells if c.cell_type == "code")
    reasons: list[str] = []
    for index in changed_cells(before, after):
        old = before.cells[index].source
        old_lines = {line.strip() for line in old.splitlines()}
        added = "\n".join(line for line in after.cells[index].source.splitlines() if line.strip() not in old_lines)
        if not added:
            continue
        for hook in _PRINT_HOOKS:
            if hook in added and hook not in old:
                reasons.append(f"cell {index} adds {hook}, which changes how results are printed instead of fixing them")
                break
        if _LIBRARY_SUBCLASS_RE.search(added) and not _LIBRARY_SUBCLASS_RE.search(old):
            reasons.append(f"cell {index} subclasses a library type to bring back old behaviour; change the call itself")
        if _MONKEY_PATCH_RE.search(added):
            reasons.append(f"cell {index} re-attaches a method to an object (monkey-patching); change the call itself")
        for match in _STRING_RE.finditer(added):
            value = (match.group(1) if match.group(1) is not None else match.group(2) or "").strip()
            if len(value) >= MIN_PASTED_CHARS and value in expected and value not in original_source:
                reasons.append(
                    f"cell {index} pastes text from the expected output ({value[:60]!r}) into the code, "
                    "so it would print the saved output instead of computing it"
                )
                break
    return reasons
