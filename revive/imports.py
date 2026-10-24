"""Work out which libraries a notebook uses, to report their versions to the model."""

from __future__ import annotations

import re
from typing import Iterable

from nbformat import NotebookNode

_IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w]*)|import\s+([A-Za-z_][\w]*(?:\s*,\s*[A-Za-z_][\w]*)*))", re.M)

# import name -> distribution name, where they differ
_DIST_NAMES = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "tf": "tensorflow",
    "keras": "keras",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}

_STDLIB_SKIP = {
    "os", "sys", "re", "json", "math", "time", "random", "itertools", "functools",
    "collections", "pathlib", "typing", "warnings", "datetime", "string", "pickle",
    "csv", "glob", "shutil", "subprocess", "copy", "io", "statistics", "logging",
    "abc", "dataclasses", "enum", "operator", "textwrap", "uuid", "gc", "zipfile",
}


def imported_modules(sources: Iterable[str]) -> list[str]:
    """Top-level module names imported anywhere in the given sources, in first-seen order."""
    found: list[str] = []
    for source in sources:
        for match in _IMPORT_RE.finditer(source):
            names = match.group(1) or match.group(2)
            for name in re.split(r"\s*,\s*", names):
                top = name.split(".")[0].strip()
                if top and top not in found:
                    found.append(top)
    return found


def notebook_distributions(nb: NotebookNode) -> list[str]:
    """Distribution names for third-party libraries the notebook imports."""
    sources = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    dists: list[str] = []
    for module in imported_modules(sources):
        if module in _STDLIB_SKIP:
            continue
        dist = _DIST_NAMES.get(module, module)
        if dist not in dists:
            dists.append(dist)
    return dists
