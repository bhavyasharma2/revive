"""Notebook loading, saving and stored-output helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Union

import nbformat
from nbformat import NotebookNode

NBFORMAT_VERSION = 4

PathLike = Union[str, Path]


def load_notebook(path: PathLike) -> NotebookNode:
    """Read a notebook from disk, upgrading older formats to nbformat 4."""
    return nbformat.read(str(path), as_version=NBFORMAT_VERSION)


def loads_notebook(text: str) -> NotebookNode:
    """Parse a notebook from a JSON string."""
    return nbformat.reads(text, as_version=NBFORMAT_VERSION)


def save_notebook(nb: NotebookNode, path: PathLike) -> None:
    """Write a notebook to disk."""
    nbformat.write(nb, str(path))


def dumps_notebook(nb: NotebookNode) -> str:
    """Serialise a notebook to a JSON string."""
    return nbformat.writes(nb)


def code_cell_indices(nb: NotebookNode) -> list[int]:
    """Indices (into ``nb.cells``) of all code cells."""
    return [i for i, cell in enumerate(nb.cells) if cell.cell_type == "code"]


def strip_outputs(nb: NotebookNode) -> NotebookNode:
    """Deep copy of ``nb`` with every code cell's outputs and counter cleared."""
    clean = copy.deepcopy(nb)
    for cell in clean.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    return clean


def has_stored_outputs(nb: NotebookNode) -> bool:
    """True if at least one code cell carries saved outputs."""
    return any(
        cell.cell_type == "code" and len(cell.get("outputs", [])) > 0
        for cell in nb.cells
    )


def expected_errors(nb: NotebookNode) -> dict[int, str]:
    """Code cells whose saved output is an error, mapped to the exception's name.

    Tutorials often show an error on purpose ("this raises a TypeError").
    Such a cell is not broken: it must raise the same kind of error again.
    """
    found: dict[int, str] = {}
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                found[index] = str(output.get("ename") or "")
                break
    return found
