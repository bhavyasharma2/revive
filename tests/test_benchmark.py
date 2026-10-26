"""Benchmark notebooks are real failures, and the runner works end to end with a scripted model."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from revive.compare import compare_notebooks
from revive.notebook import load_notebook
from revive.runner import execute_notebook

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmark"
MANIFEST = json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_matches_files():
    files = sorted(p.stem for p in (BENCH / "notebooks").glob("*.ipynb"))
    assert sorted(m["name"] for m in MANIFEST) == files
    assert len(files) >= 15


@pytest.mark.parametrize("entry", MANIFEST, ids=[m["name"] for m in MANIFEST])
def test_every_benchmark_notebook_is_broken_today(entry):
    nb = load_notebook(BENCH / "notebooks" / f"{entry['name']}.ipynb")
    if entry["kind"] == "no_outputs":
        # Same code as its base notebook, saved without outputs; the metadata (Python version) is kept.
        from revive.notebook import has_stored_outputs

        base = load_notebook(BENCH / "notebooks" / f"{entry['base']}.ipynb")
        assert not has_stored_outputs(nb)
        assert [c.source for c in nb.cells] == [c.source for c in base.cells]
        assert nb.metadata["language_info"]["version"] == base.metadata["language_info"]["version"]
        return
    result = execute_notebook(nb, timeout=120)
    if entry["kind"] == "error":
        assert not result.ok, f"{entry['name']} unexpectedly runs"
    else:
        assert result.ok
        assert not compare_notebooks(nb, result.notebook).verified, f"{entry['name']} unexpectedly matches"


def test_saved_outputs_are_authentic_old_library_outputs():
    """Saved outputs come from running the old code on old libraries (Python 3.8/3.10), not from a rewrite."""
    for entry in MANIFEST:
        if entry["kind"] == "no_outputs":
            continue
        nb = load_notebook(BENCH / "notebooks" / f"{entry['name']}.ipynb")
        assert nb.metadata["language_info"]["version"].startswith(("3.8", "3.10")), entry["name"]
        assert entry["written_for"].startswith("Python 3.")


class OneShotLLM:
    """Patches the silent KMeans notebook the way a correct model would."""

    def __init__(self):
        self.usage = {}

    def reset_usage(self):
        self.usage = {}

    def model_for(self, role):
        return "fake/model"

    def chat(self, messages, *, role="deep", **kwargs):
        self.usage.setdefault("fake/model", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})["calls"] += 1
        return json.dumps({
            "explanation": "KMeans n_init default changed to 'auto' in 1.4; restore n_init=10",
            "edits": [{"cell_index": 3, "new_source": "km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(X)\nprint(round(km.inertia_, 2))"}],
        })


def test_runner_catches_silent_drift_then_verifies_the_fix(tmp_path, monkeypatch):
    bench = load_script("run_benchmark")
    monkeypatch.setattr(bench, "RESULTS", tmp_path)
    monkeypatch.setattr(bench, "RECORDINGS", tmp_path / "recordings")
    args = SimpleNamespace(max_attempts=2, timeout=60, no_env=True)
    row = bench.run_one(BENCH / "notebooks" / "sklearn_kmeans_silent.ipynb", OneShotLLM(), None, args)
    assert row["outcome"] == "verified"
    assert row["drift_caught"] is True        # the untouched notebook ran but its numbers were wrong
    assert row["cells_matched"] == "1/1"     # one cell in this notebook prints output
    assert (tmp_path / "revived" / "sklearn_kmeans_silent.ipynb").exists()
    recording = json.loads((tmp_path / "recordings" / "sklearn_kmeans_silent.json").read_text())
    assert recording["outcome"] == "verified"
    assert recording["events"][0]["type"] == "notebook" and recording["events"][-1]["type"] == "done"

    markdown = bench.write_markdown([row], {m["name"]: m for m in MANIFEST})
    assert "Verified repairs: **1/1**" in markdown and "sklearn_kmeans_silent" in markdown


def test_reconstruction_faithfulness_is_measured_against_the_hidden_real_outputs():
    from revive.timemachine import Reconstruction, EnvSpec
    from revive.notebook import strip_outputs

    bench = load_script("run_benchmark")
    entry = {m["name"]: m for m in MANIFEST}["sklearn_kmeans_silent_no_outputs"]
    real = load_notebook(BENCH / "notebooks" / "sklearn_kmeans_silent.ipynb")
    good = SimpleNamespace(reconstruction=Reconstruction(ok=True, spec=EnvSpec("3.10"), reference=real))
    assert bench.reconstruction_vs_real(good, entry).startswith("verified")
    wrong = SimpleNamespace(reconstruction=Reconstruction(ok=True, spec=EnvSpec("3.10"), reference=execute_notebook(strip_outputs(real), timeout=60).notebook))
    assert bench.reconstruction_vs_real(wrong, entry).startswith("NOT verified")   # today's sklearn is not the original
    assert bench.reconstruction_vs_real(good, {"kind": "error"}) is None


REAL = BENCH / "real"


def test_real_notebooks_are_credited_and_present():
    entries = json.loads((REAL / "manifest.json").read_text(encoding="utf-8"))
    files = sorted(p.stem for p in (REAL / "notebooks").glob("*.ipynb"))
    assert sorted(e["name"] for e in entries) == files
    for e in entries:
        assert e["kind"] == "real"
        assert e["source"].startswith("https://github.com/")
        assert e["author"] and e["license"] and e["breaks"] and e["title"]
    assert (REAL / "licenses" / "handson-ml-LICENSE.txt").exists()
    assert (REAL / "licenses" / "PythonDataScienceHandbook-LICENSE-CODE.txt").exists()
    readme = (REAL / "README.md").read_text(encoding="utf-8")
    assert all(e["name"] in readme for e in entries)


def test_handbook_notebooks_hold_only_code_cells_and_a_credit():
    for name in ("pdsh_03_04_missing_values", "pdsh_03_06_concat_and_append"):
        nb = json.loads((REAL / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
        first, rest = nb["cells"][0], nb["cells"][1:]
        assert first["cell_type"] == "markdown" and "VanderPlas" in "".join(first["source"])
        assert rest and all(c["cell_type"] == "code" for c in rest)


def test_app_lists_real_notebooks():
    from fastapi.testclient import TestClient
    from revive.web.app import create_app

    client = TestClient(create_app(repair=lambda nb, emit, options: None))
    real = [s for s in client.get("/api/samples").json() if s.get("kind") == "real"]
    assert len(real) == 4 and all(s["source"] for s in real)
