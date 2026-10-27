"""Check a notebook before running it: what would stop Revive from repairing and verifying it.

Many real notebooks depend on things that are not in the .ipynb file: a CSV
next to it, an image it displays, a helper module, a deep-learning framework
or a GPU. Running such a notebook only to fail minutes later wastes time and
API credits, so the web app runs this check as soon as a notebook is chosen
and explains, in plain words, what is missing and what to add.

The check only reads the notebook; it never runs any of its code. It is
deliberately conservative: it reports what it can see (literal file names in
calls that read files, imports, magics) and does not guess about paths built
at run time.
"""

from __future__ import annotations

import importlib.util
import posixpath
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from nbformat import NotebookNode

from .compare import environment_report
from .imports import imported_modules
from .notebook import expected_errors, has_stored_outputs

BLOCK, WARN, INFO, GOOD = "block", "warn", "info", "good"

# Not supported on the hosted demo yet (large downloads, usually a GPU).
UNSUPPORTED = {
    "tensorflow": "TensorFlow", "keras": "Keras", "torch": "PyTorch", "torchvision": "PyTorch",
    "jax": "JAX", "jaxlib": "JAX", "transformers": "Hugging Face Transformers",
    "lightning": "PyTorch Lightning", "pytorch_lightning": "PyTorch Lightning",
    "tensorflow_datasets": "TensorFlow Datasets", "tensorflow_hub": "TensorFlow Hub",
    "tf_agents": "TF-Agents", "gym": "OpenAI Gym", "gymnasium": "Gymnasium",
    "pyspark": "Spark", "theano": "Theano", "cupy": "CuPy",
}

LONG_NOTEBOOK_CELLS = 120

# Calls that read a file whose name is given as a string literal, e.g.
# pd.read_csv("data/train.csv"), np.load("x.npy"), open("notes.txt"), Image(filename="images/a.png").
_READ_CALL_RE = re.compile(
    r"\b(read_csv|read_table|read_excel|read_json|read_parquet|read_pickle|read_feather|read_hdf|read_fwf"
    r"|read_stata|read_sas|read_spss|read_orc|loadtxt|genfromtxt|fromfile|loadmat|load|imread|load_img"
    r"|open|Image|ImageFile|run)\s*\(\s*(?:(?:filename|fname|filepath_or_buffer|path|file|filepath)\s*=\s*)?"
    r"[rRbBuU]?(['\"])([^'\"\n]+)\2(.{0,60})",
)
# %run helper.py runs another file.
_RUN_MAGIC_RE = re.compile(r"^\s*%run\s+(?:-\w+\s+)*([^\s;#]+)", re.M)
# Calls that write files: anything the notebook writes itself is not missing.
_WRITE_CALL_RE = re.compile(
    r"\b(to_csv|to_excel|to_json|to_parquet|to_pickle|to_feather|to_hdf|savefig|save|savez|savez_compressed"
    r"|savetxt|dump|imsave|imwrite|urlretrieve|write|writefile|export_graphviz|makedirs|mkdir)\s*\(([^)]*)\)"
)
_WRITEFILE_MAGIC_RE = re.compile(r"^\s*%%writefile\s+(?:-a\s+)?(\S+)", re.M)
_OPEN_FOR_WRITING_RE = re.compile(r"^\s*,\s*(?:mode\s*=\s*)?[rRbBuU]?['\"][^'\"]*[wax]")
_STRING_RE = re.compile(r"[rRbBuU]?(['\"])([^'\"\n]+)\1")
_NETWORK_RE = re.compile(r"\b(?:urlretrieve|urlopen|requests\.get|fetch_openml|fetch_[a-z0-9_]+|load_dataset|wget)\s*\(|https?://\S+\.(?:csv|zip|tgz|gz|json|txt|data|xlsx?|parquet|npz?)\b")
_FILE_EXTENSIONS = (
    ".csv", ".tsv", ".txt", ".json", ".xlsx", ".xls", ".parquet", ".feather", ".pkl", ".pickle", ".npy", ".npz",
    ".h5", ".hdf5", ".hdf", ".mat", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".svg",
    ".data", ".dat", ".zip", ".gz", ".tgz", ".db", ".sqlite", ".py", ".joblib", ".xml", ".yaml", ".yml", ".arff",
)


@dataclass
class Finding:
    level: str  # "block", "warn", "info" or "good"
    message: str
    paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"level": self.level, "message": self.message, "paths": self.paths}


@dataclass
class Preflight:
    findings: list[Finding] = field(default_factory=list)
    needed_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.level == BLOCK for f in self.findings)

    def blockers(self) -> list[str]:
        return [f.message for f in self.findings if f.level == BLOCK]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "findings": [f.to_dict() for f in self.findings],
            "needed_files": self.needed_files,
            "missing_files": self.missing_files,
        }


def normalize_path(path: str) -> Optional[str]:
    """A notebook-relative file path in canonical form, or None if it is not a safe relative path."""
    p = path.strip().replace("\\", "/")
    if not p or p.startswith(("/", "~")) or re.match(r"^[A-Za-z]:", p) or "\x00" in p:
        return None
    p = posixpath.normpath(p)
    if p in (".", "") or p.startswith("../") or p == ".." or "/../" in p:
        return None
    return p


def _looks_like_file(path: str) -> bool:
    return path.lower().endswith(_FILE_EXTENSIONS)


def _code(nb: NotebookNode) -> list[str]:
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _without_comments(source: str) -> str:
    return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))


def _written_paths(sources: Iterable[str]) -> set[str]:
    written: set[str] = set()
    for source in sources:
        for match in _WRITE_CALL_RE.finditer(source):
            for literal in _STRING_RE.finditer(match.group(2)):
                norm = normalize_path(literal.group(2))
                if norm:
                    written.add(norm)
        for match in _WRITEFILE_MAGIC_RE.finditer(source):
            norm = normalize_path(match.group(1))
            if norm:
                written.add(norm)
        for match in _READ_CALL_RE.finditer(source):
            if match.group(1) == "open" and _OPEN_FOR_WRITING_RE.match(match.group(4)):
                norm = normalize_path(match.group(3))
                if norm:
                    written.add(norm)
    return written


def _is_written(path: str, written: set[str]) -> bool:
    # Written directly, or created inside a folder the notebook makes or writes into.
    return path in written or any(path.startswith(w.rstrip("/") + "/") for w in written)


def _read_paths(sources: Iterable[str]) -> tuple[list[str], list[str]]:
    """Relative files the notebook reads, and absolute paths it reads (which nobody can supply)."""
    relative: list[str] = []
    absolute: list[str] = []
    for source in sources:
        for match in _READ_CALL_RE.finditer(source):
            func, raw, rest = match.group(1), match.group(3).strip(), match.group(4)
            if raw.lower().startswith(("http://", "https://", "ftp://", "s3://", "gs://", "data:")):
                continue
            if func == "open" and _OPEN_FOR_WRITING_RE.match(rest):
                continue
            if func in ("load", "run", "Image", "ImageFile") and not _looks_like_file(raw):
                continue  # e.g. json.load(f) is not caught here; Image("some text") is not a file
            norm = normalize_path(raw)
            if norm is None:
                if raw not in absolute:
                    absolute.append(raw)
            elif norm not in relative:
                relative.append(norm)
        for match in _RUN_MAGIC_RE.finditer(source):
            norm = normalize_path(match.group(1))
            if norm and norm not in relative:
                relative.append(norm)
    return relative, absolute


def check(nb: NotebookNode, provided: Iterable[str] = ()) -> Preflight:
    """Everything that would stop a repair, given the files that will be placed next to the notebook."""
    report = Preflight()
    sources = [_without_comments(s) for s in _code(nb)]
    provided_set = {p for p in (normalize_path(x) for x in provided) if p}

    if not sources:
        report.findings.append(Finding(BLOCK, "This notebook has no code cells."))
        return report

    # Frameworks the demo does not support yet.
    modules = imported_modules(sources)
    heavy = sorted({UNSUPPORTED[m] for m in modules if m in UNSUPPORTED})
    if heavy:
        report.findings.append(Finding(
            BLOCK, f"Uses {', '.join(heavy)}, which Revive does not support yet (deep-learning frameworks and GPUs are next)."
        ))

    # Files it reads.
    written = _written_paths(sources)
    reads, absolute = _read_paths(sources)
    needed = [p for p in reads if not _is_written(p, written)]
    report.needed_files = needed
    report.missing_files = [p for p in needed if p not in provided_set]
    if report.missing_files:
        report.findings.append(Finding(
            BLOCK,
            "Reads files that are not included. Add them below (they are placed next to the notebook).",
            list(report.missing_files),
        ))
    elif needed:
        report.findings.append(Finding(GOOD, "Every file it reads is included.", list(needed)))
    if absolute:
        report.findings.append(Finding(
            BLOCK,
            "Reads files by an absolute path on the author's computer, which cannot be supplied here. "
            "Change them to paths next to the notebook, or run Revive locally.",
            absolute,
        ))

    # Libraries that are neither installed here nor supplied as .py files.
    missing_modules = []
    for module in modules:
        if module in UNSUPPORTED or f"{module}.py" in provided_set:
            continue
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing_modules.append(module)
    if missing_modules:
        report.findings.append(Finding(
            WARN,
            "Imports modules that are not installed here: Revive will try to install them. "
            "If one is a .py file that sits next to the notebook, add it below.",
            missing_modules,
        ))

    # Things that can never match, or depend on the internet.
    env_cells = [s for s in _code(nb) if environment_report(s)]
    if env_cells:
        report.findings.append(Finding(
            INFO, "Has cells that only print library versions or the date; Revive runs them but does not compare them."
        ))
    if any(_NETWORK_RE.search(s) for s in sources):
        report.findings.append(Finding(
            WARN, "Downloads data from the internet while it runs; that works only if the address still exists."
        ))

    # What kind of check is possible.
    if not has_stored_outputs(nb):
        report.findings.append(Finding(
            INFO, "Saved without outputs: the Time Machine will rebuild the original environment first, which takes longer."
        ))
    if expected_errors(nb):
        report.findings.append(Finding(
            INFO, "Shows some errors on purpose; those cells must still raise the same error."
        ))
    if len(sources) > LONG_NOTEBOOK_CELLS:
        report.findings.append(Finding(
            WARN, f"Long notebook ({len(sources)} code cells): a live repair can take several minutes."
        ))
    return report
