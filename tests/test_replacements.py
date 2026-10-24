"""Snippet replacements: the safer patch format, and the drift case it prevents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from revive.agent import Outcome, ReviveAgent
from revive.compare import Status, compare_notebooks
from revive.notebook import load_notebook
from revive.patches import CellReplace, Patch, PatchApplyError, PatchParseError, apply_patch, parse_patch, patch_diff
from revive.runner import execute_notebook

from conftest import make_notebook

IO_FILL = Path(__file__).resolve().parent.parent / "benchmark" / "notebooks" / "pandas_io_fill.ipynb"


def test_parse_replacements():
    patch = parse_patch(json.dumps({"replacements": [{"cell_index": 2, "find": "np.float)", "replace": "float)"}]}))
    assert patch.replacements == [CellReplace(2, "np.float)", "float)")]
    assert patch.code_changes() and not patch.edits


@pytest.mark.parametrize(
    "item",
    [{"cell_index": 1, "find": "", "replace": "x"}, {"cell_index": 1, "find": 3, "replace": "x"},
     {"cell_index": "1", "find": "a", "replace": "b"}, {"cell_index": 1, "find": "a"}],
)
def test_bad_replacements_rejected(item):
    with pytest.raises(PatchParseError):
        parse_patch(json.dumps({"replacements": [item]}))


def test_null_lists_are_treated_as_empty():
    patch = parse_patch(json.dumps({"edits": None, "pip_install": None, "replacements": [{"cell_index": 0, "find": "a", "replace": "b"}]}))
    assert patch.edits == [] and patch.pip_install == []


def test_too_many_changes_rejected():
    reps = [{"cell_index": 0, "find": "a", "replace": "b"}] * 51
    with pytest.raises(PatchParseError):
        parse_patch(json.dumps({"replacements": reps}))


def test_replacement_changes_only_the_snippet_and_keeps_backslashes():
    source = "raw = 'a b\\n1 2'\ndf = pd.read_csv(io.StringIO(raw), delim_whitespace=True)"
    nb = make_notebook(source)
    patched = apply_patch(nb, Patch(replacements=[CellReplace(0, "delim_whitespace=True", "sep=r'\\s+'")]))
    assert patched.cells[0].source == "raw = 'a b\\n1 2'\ndf = pd.read_csv(io.StringIO(raw), sep=r'\\s+')"
    assert nb.cells[0].source == source


def test_replacement_must_match_exactly_once():
    nb = make_notebook("x = np.float(1)\ny = np.float(2)")
    with pytest.raises(PatchApplyError, match="appears 2 times"):
        apply_patch(nb, Patch(replacements=[CellReplace(0, "np.float(", "float(")]))
    with pytest.raises(PatchApplyError, match="does not appear anywhere"):
        apply_patch(nb, Patch(replacements=[CellReplace(0, "np.int(", "int(")]))
    fixed = apply_patch(nb, Patch(replacements=[CellReplace(0, "x = np.float(", "x = float("), CellReplace(0, "y = np.float(", "y = float(")]))
    assert fixed.cells[0].source == "x = float(1)\ny = float(2)"


def test_off_by_one_cell_index_is_corrected_when_the_snippet_is_unique():
    # Seen in real runs: the fix for cell 2 labelled "cell 1".
    nb = make_notebook(("md", "# title"), "import numpy as np", "print(np.asscalar(best) * 100)")
    patched = apply_patch(nb, Patch(replacements=[CellReplace(1, "np.asscalar(best)", "best.item()")]))
    assert patched.cells[2].source == "print(best.item() * 100)"
    assert patched.cells[1].source == "import numpy as np"
    # also when the named cell is the markdown cell
    patched = apply_patch(nb, Patch(replacements=[CellReplace(0, "np.asscalar(best)", "best.item()")]))
    assert patched.cells[2].source == "print(best.item() * 100)"


def test_snippet_in_several_other_cells_is_rejected():
    nb = make_notebook("a = 1", "x = np.float(1)", "y = np.float(2)")
    with pytest.raises(PatchApplyError, match="several other cells"):
        apply_patch(nb, Patch(replacements=[CellReplace(0, "np.float(", "float(")]))


def test_out_of_range_index_with_unique_snippet_elsewhere_is_corrected():
    nb = make_notebook("x = np.float(1)")
    patched = apply_patch(nb, Patch(replacements=[CellReplace(9, "np.float(1)", "float(1)")]))
    assert patched.cells[0].source == "x = float(1)"
    with pytest.raises(PatchApplyError, match="out of range"):
        apply_patch(nb, Patch(replacements=[CellReplace(9, "nothing", "x")]))


def test_agent_corrects_an_off_by_one_cell_index(broken_numpy_notebook):
    import json
    from revive.agent import Outcome, ReviveAgent

    nb = broken_numpy_notebook  # np.float is in cell 2; the model says cell 1
    replies = [json.dumps({"explanation": "fix", "replacements": [
        {"cell_index": 1, "find": "x.astype(np.float)", "replace": "x.astype(np.float64)"}]})]

    class LLM:
        def model_for(self, role):
            return "fake"

        def chat(self, messages, *, role="deep", **kwargs):
            return replies.pop(0)

    result = ReviveAgent(LLM(), cell_timeout=60).repair(nb)
    assert result.outcome == Outcome.VERIFIED          # the off-by-one index was corrected to cell 2
    assert "np.float64" in result.notebook.cells[2].source


def test_whole_cell_edit_wins_over_replacement_in_the_same_cell():
    from revive.patches import CellEdit

    nb = make_notebook("a = 1", "b = 1")
    patched = apply_patch(nb, Patch(edits=[CellEdit(0, "a = 5")],
                                    replacements=[CellReplace(0, "a = 1", "a = 9"), CellReplace(1, "1", "2")]))
    assert patched.cells[0].source == "a = 5"      # the replacement aimed at the old text is ignored
    assert patched.cells[1].source == "b = 2"


def test_replacement_tolerates_surrounding_whitespace():
    nb = make_notebook("x = np.asscalar(a)\nprint(x)")
    patched = apply_patch(nb, Patch(replacements=[CellReplace(0, "\nx = np.asscalar(a)  \n", "\nx = a.item()\n")]))
    assert patched.cells[0].source == "x = a.item()\nprint(x)"


def test_patch_diff_covers_replacements():
    nb = make_notebook("print(np.asscalar(a))")
    diff = patch_diff(nb, Patch(replacements=[CellReplace(0, "np.asscalar(a)", "a.item()")]))
    assert "-print(np.asscalar(a))" in diff and "+print(a.item())" in diff


def test_corrupted_whole_cell_rewrite_runs_but_is_caught_as_drift():
    """The failure mode replacements prevent: retyping a cell turns '\\n' into a literal backslash."""
    nb = load_notebook(IO_FILL)
    cell = nb.cells[2].source
    corrupted = cell.replace("\\n", "\\\\n").replace("delim_whitespace=True", "sep=r'\\s+'")
    from revive.patches import CellEdit

    run = execute_notebook(apply_patch(nb, Patch(edits=[CellEdit(2, corrupted)])), timeout=60)
    if run.ok:
        report = compare_notebooks(nb, run.notebook)
        assert not report.verified
        assert report.cells[0].status in (Status.DRIFT, Status.CHANGED)
    # (if pandas rejects the corrupted text outright, that is also not a silent success)


def test_replacement_fix_for_io_fill_verifies():
    nb = load_notebook(IO_FILL)
    patch = Patch(replacements=[
        CellReplace(2, "delim_whitespace=True", "sep=r'\\s+'"),
        CellReplace(3, ".fillna(method='ffill')", ".ffill()"),
    ])
    run = execute_notebook(apply_patch(nb, patch), timeout=60)
    assert run.ok, run.error_summary()
    assert compare_notebooks(nb, run.notebook).verified


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def model_for(self, role):
        return "fake/" + role

    def chat(self, messages, *, role="deep", **kwargs):
        if role == "fast":
            return "pandas read_csv delim_whitespace removed"
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0)


class Search:
    def __init__(self):
        self.queries = []

    def search(self, query, *, max_results=5):
        self.queries.append(query)
        return []


def test_agent_with_replacements_and_drift_feedback_details():
    nb = load_notebook(IO_FILL)
    # A plausible wrong fix: it runs, but backward-fill changes the values.
    bad = json.dumps({"explanation": "replace removed arguments", "replacements": [
        {"cell_index": 2, "find": "delim_whitespace=True", "replace": "sep=r'\\s+'"},
        {"cell_index": 3, "find": ".fillna(method='ffill')", "replace": ".bfill()"}]})
    good = json.dumps({"explanation": "fillna(method='ffill') is ffill(), not bfill()", "replacements": [
        {"cell_index": 3, "find": ".bfill()", "replace": ".ffill()"}]})
    llm = ScriptedLLM([bad, good])
    search = Search()
    events = []
    result = ReviveAgent(llm, search=search, cell_timeout=60, on_event=events.append).repair(nb)
    assert result.outcome == Outcome.VERIFIED, [e for e in events if e["type"] in ("run_result", "compare")]
    assert result.attempts[1].error == "outputs differ from the saved outputs"
    drift_prompt = llm.prompts[1]
    assert "saved output:" in drift_prompt and "output now:" in drift_prompt
    assert "[21.5, 21.5, 23.0, 23.0, 24.5]" in drift_prompt      # what the author saved
    assert "[21.5, 23.0, 23.0, 24.5, 24.5]" in drift_prompt      # what the bad fix printed
    # drift also triggers a web search for the behaviour change
    assert len(search.queries) == 2
    patch_events = [e for e in events if e["type"] == "patch"]
    assert {e["cell_index"] for e in patch_events[0]["edits"]} == {2, 3}
    assert "delim_whitespace" not in result.notebook.cells[2].source
    assert "\\n" in result.notebook.cells[2].source                 # data string untouched
