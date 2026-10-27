"""The pre-flight check and data files sent with an uploaded notebook."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import nbformat
import pytest
from fastapi.testclient import TestClient

from revive.compare import Status, compare_notebooks
from revive.imports import imported_modules
from revive.preflight import check, normalize_path
from revive.web.app import create_app, write_data_files


def nb_of(*sources, outputs=True):
    nb = nbformat.v4.new_notebook()
    for src in sources:
        cell = nbformat.v4.new_code_cell(src)
        if outputs:
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="1\n")]
        nb.cells.append(cell)
    return nb


def levels(report):
    return [f.level for f in report.findings]


# ---------- what the check sees ----------

def test_files_the_notebook_reads_are_needed_until_provided():
    nb = nb_of("import pandas as pd\ndf = pd.read_csv('data/train.csv')",
               "from IPython.display import Image\nImage(filename='./images/fig 1.png', width=400)")
    report = check(nb)
    assert report.needed_files == ["data/train.csv", "images/fig 1.png"]
    assert report.missing_files == report.needed_files and not report.ok
    assert check(nb, ["data/train.csv", "images/fig 1.png"]).ok


def test_files_the_notebook_writes_itself_are_not_needed():
    nb = nb_of("df.to_csv('out.csv')\nback = pd.read_csv('out.csv')",
               "with open('notes.txt', 'w') as f:\n    f.write('x')\nprint(open('notes.txt').read())",
               "import os\nos.makedirs('figures', exist_ok=True)\nnp.save('figures/a.npy', x)\nnp.load('figures/a.npy')")
    report = check(nb)
    assert report.needed_files == [] and report.ok


def test_urls_and_comments_are_not_files_but_downloads_are_flagged():
    nb = nb_of("# pd.read_csv('old/commented.csv')\ndf = pd.read_csv('https://example.org/data.csv')")
    report = check(nb)
    assert report.needed_files == []
    assert "warn" in levels(report)


def test_absolute_paths_and_unsupported_frameworks_block():
    assert not check(nb_of("pd.read_csv('/home/alice/data.csv')")).ok
    assert not check(nb_of("pd.read_csv('C:\\\\Users\\\\bob\\\\data.csv')")).ok
    report = check(nb_of("import tensorflow as tf"))
    assert not report.ok and "TensorFlow" in report.findings[0].message


def test_run_magic_and_local_modules():
    report = check(nb_of("%run helpers.py"))
    assert report.missing_files == ["helpers.py"]
    report = check(nb_of("import helpers_05_08"))
    assert report.ok and any("helpers_05_08" in f.paths for f in report.findings)
    assert not any("helpers_05_08" in f.paths for f in check(nb_of("import helpers_05_08"), ["helpers_05_08.py"]).findings)


def test_environment_reports_and_missing_outputs_are_explained():
    report = check(nb_of("%load_ext watermark\n%watermark -v -p numpy", "print(1)", outputs=False))
    messages = " ".join(f.message for f in report.findings)
    assert report.ok and "versions" in messages and "Time Machine" in messages


def test_every_bundled_notebook_passes():
    root = Path(__file__).resolve().parents[1] / "benchmark"
    for path in [*root.glob("notebooks/*.ipynb"), *root.glob("real/notebooks/*.ipynb")]:
        assert check(nbformat.read(path, as_version=4)).ok, path.name


@pytest.mark.parametrize("raw, expected", [
    ("data/train.csv", "data/train.csv"), ("./images/a.png", "images/a.png"), ("a\\b.csv", "a/b.csv"),
    ("../secret.txt", None), ("/etc/passwd", None), ("C:/x.csv", None), ("~/.ssh/id", None), ("a/../../b", None), ("", None),
])
def test_normalize_path(raw, expected):
    assert normalize_path(raw) == expected


def test_load_ext_counts_as_a_package_and_version_cells_are_not_compared():
    assert imported_modules(["%load_ext watermark\n%load_ext autoreload\nimport numpy"]) == ["watermark", "numpy"]
    before = nb_of("%load_ext watermark\n%watermark -v -d")
    after = nb_of("%load_ext watermark\n%watermark -v -d")
    after.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="Python 3.13 2026-10-27\n")]
    report = compare_notebooks(before, after)
    assert report.cells[0].status == Status.SKIPPED and "versions" in report.cells[0].detail


# ---------- the web API ----------

def _client(monkeypatch, seen):
    monkeypatch.setenv("REVIVE_ALLOW_UPLOADS", "1")
    monkeypatch.delenv("REVIVE_SANDBOX_USER", raising=False)

    def repair(nb, emit, options):
        seen.append(options)
        raise RuntimeError("stub")

    return TestClient(create_app(repair=repair))


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def test_preflight_endpoint_and_jobs_refuse_missing_files(monkeypatch):
    seen = []
    client = _client(monkeypatch, seen)
    text = nbformat.writes(nb_of("import pandas as pd\nprint(pd.read_csv('data/train.csv').shape)"))
    pf = client.post("/api/preflight", json={"notebook": text}).json()
    assert pf["ok"] is False and pf["missing_files"] == ["data/train.csv"]
    assert client.post("/api/preflight", json={"notebook": text, "files": ["data/train.csv"]}).json()["ok"] is True

    res = client.post("/api/jobs", json={"notebook": text, "live": True})
    assert res.status_code == 400 and "data/train.csv" in res.json()["detail"]
    assert seen == []  # nothing ran

    res = client.post("/api/jobs", json={"notebook": text, "live": True, "files": {"data/train.csv": _b64(b"a,b\n1,2\n")}})
    assert res.status_code == 201
    for _ in range(50):
        if seen:
            break
        import time; time.sleep(0.05)
    assert seen and seen[0]["files"] == {"data/train.csv": b"a,b\n1,2\n"}


@pytest.mark.parametrize("files, status", [
    ({"../escape.csv": "YQ=="}, 400),
    ({"/etc/x": "YQ=="}, 400),
    ({"ok.csv": "not base64!"}, 400),
    ({f"f{i}.csv": "YQ==" for i in range(31)}, 413),
])
def test_bad_data_files_are_refused(monkeypatch, files, status):
    client = _client(monkeypatch, [])
    text = nbformat.writes(nb_of("print(1)"))
    assert client.post("/api/jobs", json={"notebook": text, "live": True, "files": files}).status_code == status


def test_data_files_too_large_in_total(monkeypatch):
    import revive.web.app as app_module

    monkeypatch.setattr(app_module, "MAX_DATA_BYTES", 10)
    client = _client(monkeypatch, [])
    text = nbformat.writes(nb_of("print(1)"))
    res = client.post("/api/jobs", json={"notebook": text, "live": True, "files": {"a.csv": _b64(b"x" * 11)}})
    assert res.status_code == 413


def test_write_data_files_places_files_and_never_escapes(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    write_data_files(work, {"images/a.png": b"png", "train.csv": b"1"})
    assert (work / "images" / "a.png").read_bytes() == b"png" and (work / "train.csv").read_bytes() == b"1"
    with pytest.raises(ValueError):
        write_data_files(work, {"../outside.txt": b"x"})
    assert not (tmp_path / "outside.txt").exists()
