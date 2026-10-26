"""Prompts sent to the Nemotron models."""

from __future__ import annotations

from typing import Optional, Sequence

from nbformat import NotebookNode

MAX_CELL_CHARS = 4000
MAX_TRACEBACK_CHARS = 4000
MAX_SNIPPET_CHARS = 600

REPAIR_SYSTEM = """You repair Jupyter notebooks that stopped working because the libraries they use changed (removed or renamed APIs, changed defaults, version conflicts, missing packages).

Rules:
- Fix the root cause with the smallest change that keeps the notebook's original intent and results.
- Never hard-code, print or fake an expected output. The fixed notebook is verified by re-running it and comparing each cell's output with the outputs the author saved, so faking results will be caught.
- Prefer updating code to the current API. Pin an older package version only when there is no reasonable modern equivalent, and only with exact pins such as "pandas==1.5.3".
- Change only the code that is broken: code that raises an error, or whose output differs from the saved output. Leave every other character of the notebook exactly as it is.
- Code in a later cell that will certainly raise an error with the installed versions (it uses an API that was removed or renamed, as the changelog or the printed warnings show) also counts as broken: fix it in the same patch instead of waiting for it to fail.
- Do not "modernise" code that still works, and do not fix typos you only suspect. If code runs and prints the saved output, it is correct for this notebook. For example np.random.RandomState still works in NumPy 2; replacing it changes the random numbers and every result after it.
- Never change how something is printed just to make it look like the saved output: do not rewrite printed text or error messages, do not override __repr__/__str__, and do not re-attach a removed method to an object (monkey-patching). Patches that do this are rejected. Revive already ignores known formatting changes (timings, figure sizes, Int64Index vs Index, object vs str dtypes, scikit-learn printing only non-default parameters). If what remains is only a difference in how a library formats its output, it is not something to fix.
- If the notebook prints or evaluates its own code as a string (for example display('df1.append(df2)')), update that string like any other code; Revive accounts for the changed label. Do not subclass or monkey-patch library types to keep an old call working.
- Timings (%timeit, %time) are never compared. Never pin or change library versions to change speed or formatting.
- Treat an API as removed only when the error, a printed warning or the search results say so. Many old APIs still exist (for example np.nansum, np.nanmin, Series.isnull and Series.notnull).
- You cannot add, delete or reorder cells.
- If several cells use the same broken API, fix all of them in one patch.

How to write changes:
- "cell_index" is the number shown after "### cell" in the listing (it counts every cell, including markdown cells, starting at 0). The cell that failed is marked FAILED HERE.
- Prefer "replacements": each one finds an exact snippet in one cell and replaces it. Copy "find" character for character from the cell source, and make it long enough to appear exactly once in that cell. Keep snippets short: just the broken call or line.
- Use "edits" (the complete new source of a cell) only when a cell needs a larger rewrite.
- Both are JSON strings, so a backslash in Python code is written as \\\\ in JSON. For example the Python code '\\n' inside a string literal is written as '\\\\n'.

Answer with a single JSON object and nothing else:
{"explanation": "<one or two sentences: what broke and why>",
 "pip_install": ["<exact requirement>", ...],
 "replacements": [{"cell_index": <int>, "find": "<exact snippet>", "replace": "<new snippet>"}],
 "edits": [{"cell_index": <int>, "new_source": "<the complete new source of that cell>"}]}
"""

QUERY_SYSTEM = """You write one web search query that will find the official changelog, migration guide or deprecation note explaining a Python error. Answer with the query only, at most 12 words, no quotes."""


def render_notebook(nb: NotebookNode, highlight: Optional[int] = None) -> str:
    parts: list[str] = []
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source
        if len(source) > MAX_CELL_CHARS:
            source = source[:MAX_CELL_CHARS] + "\n# ... (cell truncated)"
        marker = "   <-- FAILED HERE" if index == highlight else ""
        parts.append(f"### cell {index}{marker}\n{source}")
    return "\n\n".join(parts)


def _tail(text: Optional[str], limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else "...\n" + text[-limit:]


def repair_prompt(
    nb: NotebookNode,
    *,
    failed_cell: Optional[int],
    error: str,
    traceback: Optional[str],
    versions: dict[str, str],
    evidence: Sequence[dict],
    history: Sequence[str],
    drift: Optional[str] = None,
    locked: Sequence[int] = (),
    original_env: str = "",
    printed_warnings: Sequence[str] = (),
) -> str:
    lines: list[str] = []
    if versions:
        env = ", ".join(f"{name} {version}" for name, version in versions.items())
        lines.append(f"Environment: {env}")
    if original_env:
        lines.append(
            f"The untouched notebook runs correctly in its original environment ({original_env}), "
            "so the breakage comes from changes between those versions and the current ones."
        )
    lines.append("")
    lines.append("Notebook code cells (indices count every cell, including markdown):")
    lines.append(render_notebook(nb, failed_cell))
    lines.append("")
    if drift:
        lines.append("The notebook now runs, but its outputs no longer match the outputs the author saved:")
        lines.append(drift)
        lines.append("Find the library behaviour change that causes this difference and fix it at the source.")
    else:
        lines.append(f"Error: {error}")
        if traceback:
            lines.append("Traceback:")
            lines.append(_tail(traceback, MAX_TRACEBACK_CHARS))
    if printed_warnings:
        lines.append("")
        lines.append(
            "When the notebook ran with its original library versions it printed these warnings. "
            "Each names an API that was about to be removed or changed, so fix every one that applies in this patch:"
        )
        lines.extend(f"- {w}" for w in printed_warnings)
    if locked:
        lines.append("")
        lines.append(
            "Cells " + ", ".join(str(i) for i in locked) + " already run and print exactly what the author saved. "
            "Do not change them; a patch that does is rejected."
        )
    if evidence:
        lines.append("")
        lines.append("Web search results that may explain the change:")
        for hit in evidence:
            snippet = (hit.get("content") or "")[:MAX_SNIPPET_CHARS]
            lines.append(f"- {hit.get('title', '')} ({hit.get('url', '')}): {snippet}")
    if history:
        lines.append("")
        lines.append("Previous attempts that did not work (do not repeat them):")
        lines.extend(f"- {item}" for item in history)
    return "\n".join(lines)


def query_prompt(error: str, versions: dict[str, str]) -> str:
    env = ", ".join(f"{name} {version}" for name, version in versions.items())
    return f"Error: {error}\nEnvironment: {env or 'unknown'}"
