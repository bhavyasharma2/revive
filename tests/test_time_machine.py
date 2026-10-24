"""Time Machine: infer, rebuild and run the original environment.

Most tests use a fake environment builder. Tests marked `uv` build real old
Python environments with uv (a few seconds each, needs network the first time).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from revive.agent import REFERENCE_RECONSTRUCTED, REFERENCE_SAVED, Outcome, ReviveAgent
from revive.notebook import load_notebook, strip_outputs
from revive.patches import PatchParseError
from revive.timemachine import EnvError, EnvSpec, Reconstruction, collect_evidence, parse_env_spec, reconstruct

ROOT = Path(__file__).resolve().parent.parent
CHURN = ROOT / "examples" / "legacy_churn_model.ipynb"
KMEANS = ROOT / "benchmark" / "notebooks" / "sklearn_kmeans_silent.ipynb"
OLD_SPEC = {"python": "3.10", "packages": {"numpy": "1.22.4", "pandas": "1.5.3", "scikit-learn": "1.1.3"}, "reasoning": "np.float removed in 1.24"}

needs_uv = pytest.mark.skipif(
    shutil.which("uv") is None and __import__("importlib").util.find_spec("uv") is None,
    reason="uv is not installed",
)


# ---- parsing the model's proposal ---------------------------------------------

def test_parse_env_spec_maps_import_names_and_drops_unknown_packages():
    spec = parse_env_spec(json.dumps({"python": "3.10.4", "packages": {"sklearn": "1.1.3", "numpy": "1.22.4", "evil-pkg": "1.0"}}),
                          ["numpy", "pandas", "scikit-learn"])
    assert spec.python == "3.10"
    assert spec.packages == {"scikit-learn": "1.1.3", "numpy": "1.22.4"}
    assert spec.unpinned == ["pandas"]
    assert spec.requirements() == ["scikit-learn==1.1.3", "numpy==1.22.4", "pandas"]
    assert "pandas (latest)" in spec.describe()


def test_supporting_libraries_can_be_pinned_and_numpy2_is_avoided():
    # The notebook imports only scikit-learn, but numpy/scipy versions decide whether it works.
    spec = parse_env_spec(json.dumps({"python": "3.10", "packages": {"scikit-learn": "1.1.3", "scipy": "1.9.3"}}), ["scikit-learn"])
    assert spec.packages == {"scikit-learn": "1.1.3", "scipy": "1.9.3"}
    assert spec.unpinned == []
    assert spec.requirements()[-1] == "numpy<2"      # never mix NumPy 2 with 2022-era compiled libraries
    pinned = parse_env_spec(json.dumps({"python": "3.10", "packages": {"scikit-learn": "1.1.3", "numpy": "1.22.4"}}), ["scikit-learn"])
    assert "numpy<2" not in pinned.requirements()


@pytest.mark.parametrize("bad", [
    {"python": "2.7", "packages": {}},
    {"python": "3.13", "packages": {}},
    {"python": "three", "packages": {}},
    {"python": "3.10", "packages": ["numpy"]},
    {"python": "3.10", "packages": {"numpy": "1.0; rm -rf /"}},
    {"python": "3.10", "packages": {"numpy": "git+https://x"}},
])
def test_parse_env_spec_rejects_bad_proposals(bad):
    with pytest.raises(PatchParseError):
        parse_env_spec(json.dumps(bad), ["numpy"])


def test_evidence_includes_metadata_python_and_printed_warnings():
    nb = load_notebook(CHURN)
    nb.metadata["language_info"] = {"name": "python", "version": "3.9.7"}
    nb.cells[2].outputs.append({"output_type": "stream", "name": "stderr", "text": "FutureWarning: The frame.append method is deprecated\n"})
    evidence = collect_evidence(nb)
    assert evidence["python_in_metadata"] == "3.9.7"
    assert "frame.append" in evidence["warnings_printed"][0]
    assert evidence["libraries"] == ["numpy", "pandas", "scikit-learn"]


# ---- the reconstruction loop with a fake environment ---------------------------

class FakeEnv:
    """Runs in the current interpreter; refuses specs that would not work."""

    def __init__(self, fail_install=False):
        self.fail_install = fail_install

    def kernel_manager(self):
        return None  # unused: execute_notebook gets a None factory below

    def destroy(self):
        pass


class Replies:
    def __init__(self, *replies):
        self.replies = [json.dumps(r) if isinstance(r, dict) else r for r in replies]
        self.prompts = []

    def model_for(self, role):
        return "fake/" + role

    def chat(self, messages, *, role="deep", **kwargs):
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0)


def test_reconstruct_retries_with_feedback_until_install_succeeds(tmp_path):
    calls = []

    def build(root, spec):
        calls.append(spec.describe())
        if spec.packages.get("numpy") == "0.0.1":
            raise EnvError("installing numpy 0.0.1 failed: no matching distribution")
        env = FakeEnv()
        env.kernel_manager = None
        return env

    llm = Replies({"python": "3.10", "packages": {"numpy": "0.0.1"}}, {"python": "3.10", "packages": {"numpy": "2.0.0"}})
    events = []
    nb = load_notebook(ROOT / "benchmark" / "notebooks" / "numpy_asscalar.ipynb")
    # numpy 2 in the current interpreter: the untouched notebook still fails -> attempts used up
    rec = reconstruct(nb, llm, workdir=tmp_path, env_factory=build, emit=events.append, max_tries=2)
    assert not rec.ok
    assert "no matching distribution" in llm.prompts[1]
    assert [e["type"] for e in events].count("reconstruct_env") == 2
    assert "asscalar" in rec.error


def test_invalid_proposal_is_fed_back(tmp_path):
    llm = Replies("I think python 3.10 is fine", {"python": "2.7"})
    rec = reconstruct(load_notebook(CHURN), llm, workdir=tmp_path, env_factory=lambda r, s: FakeEnv(), max_tries=2)
    assert not rec.ok and "not a valid environment" in llm.prompts[1]


# ---- agent integration with a fake reconstructor -------------------------------

def fake_reconstructor(reference_nb, ok=True):
    def run(nb, llm, **kwargs):
        emit = kwargs.get("emit") or (lambda e: None)
        emit({"type": "reconstruct_env", "attempt": 1, "python": "3.10", "packages": {}, "reasoning": "", "description": "Python 3.10"})
        if not ok:
            return Reconstruction(ok=False, error="could not rebuild")
        return Reconstruction(ok=True, spec=EnvSpec("3.10", {"numpy": "1.22.4"}), reference=reference_nb)
    return run


def kmeans_patch():
    return {"explanation": "n_init default changed in 1.4", "replacements": [
        {"cell_index": 3, "find": "KMeans(n_clusters=6, random_state=0)", "replace": "KMeans(n_clusters=6, n_init=10, random_state=0)"}]}


def test_silent_change_without_saved_outputs_is_caught_only_by_the_time_machine():
    saved = load_notebook(KMEANS)          # outputs from the original environment
    stripped = strip_outputs(saved)        # what people often share

    # Without the Time Machine there is nothing to check: it "runs" and that is all we can say.
    plain = ReviveAgent(Replies(), time_machine="off", cell_timeout=60).repair(stripped)
    assert plain.outcome == Outcome.RUNS_UNVERIFIED

    # With it, the reconstructed reference exposes the silent change, and the fix is verified.
    llm = Replies(kmeans_patch())
    events = []
    agent = ReviveAgent(llm, cell_timeout=60, on_event=events.append, reconstructor=fake_reconstructor(saved))
    result = agent.repair(stripped)
    assert result.outcome == Outcome.VERIFIED
    assert result.reference == REFERENCE_RECONSTRUCTED
    assert "original environment (Python 3.10, numpy 1.22.4)" in llm.prompts[0]
    assert "reference output (from the original environment)" in llm.prompts[0]
    compares = [e for e in events if e["type"] == "compare"]
    assert compares[0]["reference"] == REFERENCE_RECONSTRUCTED and not compares[0]["verified"]
    assert result.to_dict()["time_machine"]["ok"] is True


def test_auto_mode_skips_the_time_machine_when_outputs_are_saved():
    called = []

    def recon(*args, **kwargs):
        called.append(1)
        raise AssertionError("should not run")

    llm = Replies(kmeans_patch())
    result = ReviveAgent(llm, cell_timeout=60, reconstructor=recon).repair(load_notebook(KMEANS))
    assert called == [] and result.reference == REFERENCE_SAVED and result.outcome == Outcome.VERIFIED


def test_always_mode_keeps_saved_outputs_as_the_reference_and_reports_the_rebuild():
    saved = load_notebook(KMEANS)
    llm = Replies(kmeans_patch())
    result = ReviveAgent(llm, cell_timeout=60, time_machine="always", reconstructor=fake_reconstructor(saved)).repair(saved)
    assert result.reference == REFERENCE_SAVED and result.outcome == Outcome.VERIFIED
    assert result.to_dict()["time_machine"]["environment"]["python"] == "3.10"


def test_failed_reconstruction_falls_back_to_unverified():
    stripped = strip_outputs(load_notebook(KMEANS))
    result = ReviveAgent(Replies(), cell_timeout=60, reconstructor=fake_reconstructor(None, ok=False)).repair(stripped)
    assert result.outcome == Outcome.RUNS_UNVERIFIED
    assert result.to_dict()["time_machine"]["error"] == "could not rebuild"


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        ReviveAgent(Replies(), time_machine="sometimes")


# ---- real environments ---------------------------------------------------------

@needs_uv
def test_real_rebuild_reproduces_the_authors_outputs_from_a_stripped_notebook(tmp_path):
    original = load_notebook(CHURN)
    rec = reconstruct(strip_outputs(original), Replies(OLD_SPEC), workdir=tmp_path, cell_timeout=120)
    assert rec.ok, rec.error
    assert rec.installed["numpy"] == "1.22.4" and rec.installed["python"].startswith("3.10")
    from revive.compare import compare_notebooks

    assert compare_notebooks(original, rec.reference).verified


@needs_uv
def test_real_end_to_end_stripped_silent_kmeans(tmp_path):
    stripped = strip_outputs(load_notebook(KMEANS))
    spec = {"python": "3.10", "packages": {"scikit-learn": "1.1.3", "numpy": "1.22.4"}}
    llm = Replies(spec, kmeans_patch())
    result = ReviveAgent(llm, cell_timeout=120).repair(stripped)
    assert result.reference == REFERENCE_RECONSTRUCTED
    assert result.outcome == Outcome.VERIFIED
    assert "n_init=10" in result.notebook.cells[3].source


# ---- date evidence -----------------------------------------------------------

def test_validate_date():
    from revive.timemachine import validate_date

    assert validate_date("2023-02-01") == "2023-02-01"
    assert validate_date("2023-02-01T10:00:00+05:30") == "2023-02-01"
    assert validate_date(None) is None and validate_date("") is None
    for bad in ("2023-02-30", "01/02/2023", "yesterday"):
        with pytest.raises(ValueError):
            validate_date(bad)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_last_changed(tmp_path):
    import subprocess

    from revive.timemachine import git_last_changed

    nb = tmp_path / "old.ipynb"
    nb.write_text("{}")
    assert git_last_changed(nb) is None                      # not a repository
    env = {"GIT_AUTHOR_DATE": "2021-06-15T12:00:00", "GIT_COMMITTER_DATE": "2021-06-15T12:00:00",
           "HOME": str(tmp_path), "PATH": __import__("os").environ["PATH"]}
    for cmd in (["git", "init", "-q"], ["git", "-c", "user.email=a@b", "-c", "user.name=a", "add", "old.ipynb"],
                ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q", "-m", "add"]):
        subprocess.run(cmd, cwd=tmp_path, env=env, check=True)
    assert git_last_changed(nb) == "2021-06-15"


def test_date_reaches_the_model_and_the_spec(tmp_path):
    seen = []

    def build(root, spec):
        seen.append(spec)
        raise EnvError("stop here")

    llm = Replies({"python": "3.10", "packages": {"numpy": "1.22.4"}})
    reconstruct(load_notebook(CHURN), llm, workdir=tmp_path, as_of="2023-02-01", env_factory=build, max_tries=1)
    assert "last changed on 2023-02-01" in llm.prompts[0]
    assert seen[0].as_of == "2023-02-01"
    assert "latest as of 2023-02-01" in seen[0].describe()


def test_undated_prompt_warns_about_silent_changes(tmp_path):
    llm = Replies("not json")
    reconstruct(load_notebook(CHURN), llm, workdir=tmp_path, env_factory=lambda r, s: FakeEnv(), max_tries=1)
    assert "date the notebook was written is unknown" in llm.prompts[0]


def test_agent_passes_the_date_to_the_time_machine():
    got = {}

    def recon(nb, llm, **kwargs):
        got.update(kwargs)
        return Reconstruction(ok=False, error="x")

    ReviveAgent(Replies(), cell_timeout=60, reconstructor=recon, as_of="2023-02-01").repair(strip_outputs(load_notebook(KMEANS)))
    assert got["as_of"] == "2023-02-01"


@needs_uv
def test_real_dated_rebuild_recovers_from_a_too_new_numpy(tmp_path):
    """As of 2023-02-01 the newest NumPy (1.24) had already removed np.float.
    Leaving NumPy unpinned therefore fails, and the error steers the next guess."""
    original = load_notebook(CHURN)
    first = {"python": "3.10", "packages": {"pandas": "1.5.3", "scikit-learn": "1.1.3", "numpy": "1.24.1"}}
    second = {"python": "3.10", "packages": {"pandas": "1.5.3", "scikit-learn": "1.1.3", "numpy": "1.23.5"}}
    llm = Replies(first, second)
    rec = reconstruct(strip_outputs(original), llm, workdir=tmp_path, as_of="2023-02-01", cell_timeout=120)
    assert rec.ok, rec.error
    assert "np.float" in llm.prompts[1] or "float" in llm.prompts[1]
    assert rec.installed["numpy"] == "1.23.5"
    from revive.compare import compare_notebooks

    assert compare_notebooks(original, rec.reference).verified


@needs_uv
def test_real_dated_rebuild_corrects_versions_newer_than_the_date(tmp_path):
    # scikit-learn 1.5.2 (2024) cannot be what a notebook last changed in Feb 2023 used:
    # the pin is dropped and the date picks the newest release before it, 1.2.1.
    llm = Replies({"python": "3.10", "packages": {"scikit-learn": "1.5.2"}})
    rec = reconstruct(strip_outputs(load_notebook(KMEANS)), llm, workdir=tmp_path, as_of="2023-02-01", cell_timeout=120)
    assert rec.ok, rec.error
    assert len(llm.prompts) == 1
    assert rec.installed["scikit-learn"] == "1.2.1"
    assert compare_notebooks_real(rec)


def compare_notebooks_real(rec):
    from revive.compare import compare_notebooks

    return compare_notebooks(load_notebook(KMEANS), rec.reference).verified


def test_released_after_date_is_read_from_uv_message():
    from revive.timemachine import _released_after_date

    spec = EnvSpec("3.10", {"numpy": "1.23.5", "scipy": "1.10.1"}, as_of="2023-02-01")
    message = ("Because there is no version of scipy==1.10.1 ... hint: `scipy` was filtered by `exclude-newer` "
               "to only include packages uploaded before 2023-02-01T18:30:00Z.")
    assert _released_after_date(message, spec) == "scipy"
    assert _released_after_date("some other failure", spec) is None
    assert _released_after_date(message, EnvSpec("3.10", {"scipy": "1.10.1"})) is None   # no date: not our call


def test_too_new_pins_are_corrected_without_another_model_call(tmp_path):
    builds = []

    def build(root, spec):
        builds.append(dict(spec.packages))
        if "scipy" in spec.packages:
            raise EnvError("installing failed: hint: `scipy` was filtered by `exclude-newer` to only include packages uploaded before 2023-02-01")
        raise EnvError("stop: reached the corrected build")

    llm = Replies({"python": "3.10", "packages": {"numpy": "1.23.5", "scipy": "1.10.1", "scikit-learn": "1.2.1"}})
    events = []
    rec = reconstruct(load_notebook(KMEANS), llm, workdir=tmp_path, as_of="2023-02-01", env_factory=build,
                      emit=events.append, max_tries=1)
    assert builds == [{"numpy": "1.23.5", "scipy": "1.10.1", "scikit-learn": "1.2.1"}, {"numpy": "1.23.5", "scikit-learn": "1.2.1"}]
    assert len(llm.prompts) == 1                                     # no extra model call
    adjust = [e for e in events if e["type"] == "reconstruct_adjust"]
    assert adjust and "scipy 1.10.1 was released after 2023-02-01" in adjust[0]["message"]
    assert rec.spec.corrections == [adjust[0]["message"]]


@needs_uv
def test_real_rebuild_with_the_exact_failing_guess_from_a_benchmark_run(tmp_path):
    """Oct 24 run: Nemotron pinned scipy 1.10.1 (released 2023-02-19) for a notebook dated 2023-02-01, three times."""
    original = load_notebook(CHURN)
    guess = {"python": "3.10", "packages": {"numpy": "1.23.5", "pandas": "1.5.3", "scikit-learn": "1.2.1", "scipy": "1.10.1"}}
    rec = reconstruct(strip_outputs(original), Replies(guess), workdir=tmp_path, as_of="2023-02-01", cell_timeout=120, max_tries=1)
    assert rec.ok, rec.error
    assert rec.spec.corrections and "scipy" in rec.spec.corrections[0]
    from revive.compare import compare_notebooks

    assert compare_notebooks(original, rec.reference).verified
