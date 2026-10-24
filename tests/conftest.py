"""Shared helpers for building small notebooks in tests."""

from __future__ import annotations

from typing import Optional

import nbformat
import pytest
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output


def stdout_output(text: str) -> NotebookNode:
    return new_output("stream", name="stdout", text=text)


def result_output(text: str) -> NotebookNode:
    return new_output("execute_result", data={"text/plain": text}, execution_count=1)


def make_notebook(*cells) -> NotebookNode:
    """Build a notebook. Each cell is a source string, or (source, [outputs]),
    or ('md', source) for a markdown cell."""
    built = []
    for cell in cells:
        if isinstance(cell, tuple) and cell[0] == "md":
            built.append(new_markdown_cell(cell[1]))
            continue
        if isinstance(cell, tuple):
            source, outputs = cell
        else:
            source, outputs = cell, []
        code = new_code_cell(source)
        code.outputs = list(outputs)
        built.append(code)
    nb = new_notebook(cells=built)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.validate(nb)
    return nb


@pytest.fixture
def broken_numpy_notebook() -> NotebookNode:
    """A notebook written for old NumPy: ``np.float`` no longer exists.

    The stored outputs are what the original author saw.
    """
    return make_notebook(
        ("md", "# Stats demo"),
        "import numpy as np",
        (
            "x = np.array([1, 2, 3, 4])\nprint(x.astype(np.float).mean())",
            [stdout_output("2.5\n")],
        ),
        ("x.std()", [result_output("1.118033988749895")]),
    )
