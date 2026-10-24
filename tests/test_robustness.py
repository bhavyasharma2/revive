"""Fixes for failure modes seen in real benchmark runs (Oct 24)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from revive.agent import Outcome, ReviveAgent, is_transient, printed_warning_lines
from revive.notebook import load_notebook
from revive.patches import CellReplace, Patch, apply_patch, syntax_errors

from conftest import make_notebook

BENCH = Path(__file__).resolve().parent.parent / "benchmark" / "notebooks"


class Replies:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def model_for(self, role):
        return "fake/" + role

    def chat(self, messages, *, role="deep", **kwargs):
        self.prompts.append(messages[-1]["content"])
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


# --- over-escaped backslashes ----------------------------------------------------

def test_over_escaped_backslashes_still_match_a_raw_string():
    # Real run: the cell has  sep=r'\s+}'  and the model's find was  sep=r'\\s+}'
    nb = make_notebook("df = pd.read_csv(io.StringIO(raw), sep=r'\\s+}')")
    fixed = apply_patch(nb, Patch(replacements=[CellReplace(0, "sep=r'\\\\s+}'", "sep=r'\\\\s+'")]))
    assert fixed.cells[0].source == "df = pd.read_csv(io.StringIO(raw), sep=r'\\s+')"


# --- syntax check before running -------------------------------------------------

def test_syntax_errors_are_found_before_running():
    before = make_notebook("x = f(a, b)")
    after = apply_patch(before, Patch(replacements=[CellReplace(0, "f(a, b)", "f(a, b}")]))
    problems = syntax_errors(before, after)
    assert len(problems) == 1 and problems[0].startswith("cell 0:")


def test_cells_with_ipython_magics_are_not_misjudged():
    before = make_notebook("%matplotlib inline\nx = 1")
    after = apply_patch(before, Patch(replacements=[CellReplace(0, "x = 1", "x = 2")]))
    assert syntax_errors(before, after) == []
    shell = make_notebook("!pip install foo\ny = [1, 2")         # already invalid before: not judged
    edited = apply_patch(shell, Patch(replacements=[CellReplace(0, "y = [1, 2", "y = [1, 3")]))
    assert syntax_errors(shell, edited) == []


def test_agent_rejects_a_syntax_breaking_patch_without_running_it():
    nb = load_notebook(BENCH / "pandas_io_fill.ipynb")
    typo = {"explanation": "removed kwargs", "replacements": [
        {"cell_index": 2, "find": "delim_whitespace=True", "replace": "sep=r'\\s+'}"}]}   # stray } outside the string
    good = {"explanation": "removed kwargs", "replacements": [
        {"cell_index": 2, "find": "delim_whitespace=True", "replace": "sep=r'\\s+'"},
        {"cell_index": 3, "find": ".fillna(method='ffill')", "replace": ".ffill()"}]}
    llm = Replies(typo, good)
    events = []
    result = ReviveAgent(llm, cell_timeout=60, on_event=events.append, time_machine="off").repair(nb)
    assert result.outcome == Outcome.VERIFIED
    assert "not valid Python" in result.attempts[0].result
    assert sum(1 for e in events if e["type"] == "run_start") == 2   # the broken patch never ran


# --- temporary server errors -----------------------------------------------------

class HTTP503(Exception):
    status_code = 503


def test_transient_errors_are_recognised():
    assert is_transient(HTTP503())
    assert is_transient(SimpleNamespace.__class__("APIConnectionError", (Exception,), {})())
    assert not is_transient(RuntimeError("bad request"))
    err = Exception()
    err.status_code = 400
    assert not is_transient(err)


def test_a_503_is_retried_without_using_up_an_attempt():
    nb = load_notebook(BENCH / "numpy_asscalar.ipynb")
    fix = {"explanation": "asscalar removed", "replacements": [{"cell_index": 2, "find": "np.asscalar(best)", "replace": "best.item()"}]}
    llm = Replies(HTTP503(), fix)
    events = []
    result = ReviveAgent(llm, cell_timeout=60, on_event=events.append, time_machine="off", retry_delays=(0,)).repair(nb)
    assert result.outcome == Outcome.VERIFIED and len(result.attempts) == 1
    assert any("temporarily unavailable" in e.get("message", "") for e in events if e["type"] == "warning")


def test_retries_are_limited():
    nb = load_notebook(BENCH / "numpy_asscalar.ipynb")
    llm = Replies(HTTP503(), HTTP503(), HTTP503(), HTTP503())
    result = ReviveAgent(llm, cell_timeout=60, time_machine="off", retry_delays=(0,), max_attempts=1).repair(nb)
    assert result.outcome == Outcome.FAILED
    assert result.attempts[0].result.startswith("model call failed")


# --- warnings from the notebook's own era ----------------------------------------

def test_era_warnings_reach_the_model_up_front():
    nb = load_notebook(BENCH / "legacy_churn_model.ipynb")
    warnings = printed_warning_lines(nb)
    assert any("frame.append" in w for w in warnings) and any("iteritems" in w for w in warnings)
    llm = Replies("not json", "not json")
    ReviveAgent(llm, cell_timeout=60, time_machine="off", max_attempts=1).repair(nb)
    assert "printed these warnings" in llm.prompts[0] and "iteritems is deprecated" in llm.prompts[0]
