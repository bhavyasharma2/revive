"""Cells that already match their saved output are locked against edits.

Regression test for a real benchmark run: Nemotron replaced np.random.RandomState
(which still works) with a different generator, producing different data and
four drifting cells in the churn notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

from revive.agent import Outcome, ReviveAgent
from revive.compare import matching_cells
from revive.notebook import load_notebook
from revive.runner import execute_notebook

CHURN = Path(__file__).resolve().parent.parent / "examples" / "legacy_churn_model.ipynb"

GOOD_FIX = {
    "explanation": "append, np.float/np.int, sparse and iteritems were removed",
    "replacements": [
        {"cell_index": 3, "find": "df = df.append(late_signups, ignore_index=True)", "replace": "df = pd.concat([df, late_signups], ignore_index=True)"},
        {"cell_index": 4, "find": "astype(np.float)", "replace": "astype(float)"},
        {"cell_index": 4, "find": "astype(np.int)", "replace": "astype(int)"},
        {"cell_index": 5, "find": "OneHotEncoder(sparse=False)", "replace": "OneHotEncoder(sparse_output=False)"},
        {"cell_index": 7, "find": ".iteritems()", "replace": ".items()"},
    ],
}

# What the model actually did: "modernise" the RNG in cell 2 as well.
OVERREACHING_FIX = {
    "explanation": "numpy RandomState was removed; use Generator",
    "replacements": GOOD_FIX["replacements"] + [
        {"cell_index": 2, "find": "rng = np.random.RandomState(0)", "replace": "rng = np.random.Generator(np.random.MT19937(0))"},
    ],
}


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = [json.dumps(r) for r in replies]
        self.prompts = []

    def model_for(self, role):
        return "fake/" + role

    def chat(self, messages, *, role="deep", **kwargs):
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0)


def test_matching_cells_before_the_failure():
    nb = load_notebook(CHURN)
    run = execute_notebook(nb, timeout=60)
    assert run.failed_cell == 3
    # cell 1 has no saved output (imports), cell 2 prints the dataset shape and churn rate
    assert matching_cells(nb, run.notebook, upto=run.failed_cell) == {2}


def test_patch_touching_a_locked_cell_is_rejected_and_explained():
    nb = load_notebook(CHURN)
    llm = ScriptedLLM([OVERREACHING_FIX, GOOD_FIX])
    events = []
    result = ReviveAgent(llm, cell_timeout=60, on_event=events.append).repair(nb)

    assert result.outcome == Outcome.VERIFIED
    assert result.attempts[0].result.startswith("invalid patch")
    assert "cell(s) 2" in result.attempts[0].result
    rejected = [e for e in events if e["type"] == "patch_rejected"]
    assert rejected and "already run and print exactly what the author saved" in rejected[0]["reason"]
    # the model is told up front which cells are locked, and why its patch failed
    assert "Cells 2 already run" in llm.prompts[0]
    assert "cell(s) 2" in llm.prompts[1]
    # the original random generator survives
    assert "np.random.RandomState(0)" in result.notebook.cells[2].source


def test_off_by_one_relocation_cannot_bypass_the_lock():
    nb = load_notebook(CHURN)
    sneaky = json.loads(json.dumps(OVERREACHING_FIX))
    sneaky["replacements"][-1]["cell_index"] = 1      # names the imports cell; the snippet lives in locked cell 2
    llm = ScriptedLLM([sneaky, GOOD_FIX])
    result = ReviveAgent(llm, cell_timeout=60).repair(nb)
    assert result.outcome == Outcome.VERIFIED
    assert "cell(s) 2" in result.attempts[0].result
    assert "np.random.RandomState(0)" in result.notebook.cells[2].source


def test_cells_matching_after_a_fix_become_locked_for_drift_retries():
    nb = load_notebook(CHURN)
    # The first fix repairs every error but also changes the train/test split, so cell 6 drifts.
    drift_fix = json.loads(json.dumps(GOOD_FIX))
    drift_fix["replacements"].append({"cell_index": 6, "find": "test_size=0.25", "replace": "test_size=0.3"})
    undo = {"explanation": "restore the original split", "replacements": [{"cell_index": 6, "find": "test_size=0.3", "replace": "test_size=0.25"}]}
    llm = ScriptedLLM([drift_fix, undo])
    result = ReviveAgent(llm, cell_timeout=60).repair(nb)
    assert result.outcome == Outcome.VERIFIED
    drift_prompt = llm.prompts[1]
    assert "no longer match" in drift_prompt
    # every cell except the drifting one (6) is locked during the retry
    assert "Cells 2, 3, 4, 5, 7 already run" in drift_prompt
