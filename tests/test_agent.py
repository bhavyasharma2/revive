"""The agent loop, with a scripted model and real notebook execution."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from revive.agent import Outcome, ReviveAgent, fallback_query
from revive.env import InstallResult
from revive.search import SearchHit

from conftest import make_notebook, stdout_output

FIX_CELL_2 = "x = np.array([1, 2, 3, 4])\nprint(x.astype(np.float64).mean())"
WRONG_CELL_2 = "x = np.array([1, 2, 3, 4])\nprint(x.astype(np.float64).sum())"


def patch_reply(cell_index, source, explanation="np.float was removed in NumPy 1.24", pip=None):
    body = {"explanation": explanation, "edits": [{"cell_index": cell_index, "new_source": source}]}
    if pip is not None:
        body["pip_install"] = pip
    return "<think>reasoning</think>" + json.dumps(body)


class ScriptedLLM:
    """Returns queued replies for 'deep' calls and a fixed query for 'fast' calls."""

    def __init__(self, deep_replies, query="numpy np.float removed 1.24", fail_fast=False):
        self.deep_replies = list(deep_replies)
        self.query = query
        self.fail_fast = fail_fast
        self.calls = []

    def model_for(self, role):
        return {"deep": "nvidia/Nemotron-3-Ultra-550b-a55b", "fast": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"}[role]

    def chat(self, messages, *, role="deep", **kwargs):
        self.calls.append({"role": role, "messages": messages, **kwargs})
        if role == "fast":
            if self.fail_fast:
                raise RuntimeError("fast model unavailable")
            return self.query
        if not self.deep_replies:
            raise AssertionError("the agent asked for more patches than scripted")
        reply = self.deep_replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def deep_prompts(self):
        return [c["messages"][-1]["content"] for c in self.calls if c["role"] == "deep"]


class FakeSearch:
    def __init__(self, fail=False):
        self.queries = []
        self.fail = fail

    def search(self, query, *, max_results=5):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("tavily down")
        return [SearchHit(title="NumPy 1.24 release notes", url="https://numpy.org/doc/stable/release/1.24.0-notes.html",
                          content="The deprecation for the aliases np.float ... has expired.", score=0.9)]


@dataclass
class FakeEnv:
    """Records installs; runs notebooks in the default kernel."""

    install_ok: bool = True

    def __post_init__(self):
        self.installed = []
        self.install_calls = []

    kernel_manager = None  # the agent passes None -> default kernel

    def install(self, specs):
        self.install_calls.append(list(specs))
        if self.install_ok:
            self.installed.extend(specs)
        return InstallResult(ok=self.install_ok, specs=list(specs), output="" if self.install_ok else "ERROR: no matching distribution")

    def package_versions(self, dists):
        return {"python": "3.x", **{d: "test" for d in dists}}


def make_agent(llm, **kwargs):
    events = []
    kwargs.setdefault("cell_timeout", 60)
    kwargs.setdefault("time_machine", "off")  # Time Machine behaviour is tested in test_time_machine.py
    agent = ReviveAgent(llm, on_event=events.append, **kwargs)
    return agent, events


def kinds(events):
    return [e["type"] for e in events]


def test_fixes_and_verifies_in_one_attempt(broken_numpy_notebook):
    llm = ScriptedLLM([patch_reply(2, FIX_CELL_2)])
    search = FakeSearch()
    agent, events = make_agent(llm, search=search)

    result = agent.repair(broken_numpy_notebook)

    assert result.outcome == Outcome.VERIFIED and result.runs
    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert "AttributeError" in attempt.error
    assert attempt.search_query == "numpy np.float removed 1.24"
    assert attempt.sources[0]["url"].startswith("https://numpy.org")
    assert attempt.model == "nvidia/Nemotron-3-Ultra-550b-a55b"
    assert "+print(x.astype(np.float64).mean())" in attempt.diff
    assert result.comparison.verified and result.comparison.compared == 2
    assert search.queries == ["numpy np.float removed 1.24"]
    # the fast model wrote the query, the deep model wrote the patch
    assert [c["role"] for c in llm.calls] == ["fast", "deep"]
    # the evidence and the failing cell reach the deep model
    prompt = llm.deep_prompts()[0]
    assert "numpy.org" in prompt and "FAILED HERE" in prompt and "### cell 2" in prompt
    # fresh outputs are in the returned notebook
    assert result.notebook.cells[2].outputs[0].text == "2.5\n"
    assert kinds(events)[0] == "run_start" and kinds(events)[-1] == "done"
    assert "search" in kinds(events) and "patch" in kinds(events) and "compare" in kinds(events)


def test_already_working_notebook_needs_no_model():
    nb = make_notebook(("print(1 + 1)", [stdout_output("2\n")]))
    llm = ScriptedLLM([])
    agent, _ = make_agent(llm, search=FakeSearch())
    result = agent.repair(nb)
    assert result.outcome == Outcome.ALREADY_WORKING
    assert result.attempts == [] and llm.calls == []


def test_notebook_without_saved_outputs_is_runs_unverified():
    nb = make_notebook("import numpy as np", "print(np.float(1))")
    llm = ScriptedLLM([patch_reply(1, "print(np.float64(1))")])
    agent, _ = make_agent(llm)
    result = agent.repair(nb)
    assert result.outcome == Outcome.RUNS_UNVERIFIED
    assert result.comparison.unverifiable


def test_invalid_patch_is_retried_and_reported_to_the_model(broken_numpy_notebook):
    llm = ScriptedLLM(["I think you should use float64.", patch_reply(2, FIX_CELL_2)])
    agent, events = make_agent(llm)
    result = agent.repair(broken_numpy_notebook)
    assert result.outcome == Outcome.VERIFIED
    assert len(result.attempts) == 2
    assert result.attempts[0].result.startswith("invalid patch")
    assert "invalid patch" in llm.deep_prompts()[1]
    assert "patch_rejected" in kinds(events)


def test_patch_for_markdown_cell_is_rejected_not_crashing(broken_numpy_notebook):
    llm = ScriptedLLM([patch_reply(0, "oops"), patch_reply(2, FIX_CELL_2)])
    agent, _ = make_agent(llm)
    result = agent.repair(broken_numpy_notebook)
    assert result.outcome == Outcome.VERIFIED
    assert "not code" in result.attempts[0].result


def test_model_api_error_uses_up_an_attempt_but_does_not_crash(broken_numpy_notebook):
    llm = ScriptedLLM([RuntimeError("503 from API"), patch_reply(2, FIX_CELL_2)])
    agent, events = make_agent(llm)
    result = agent.repair(broken_numpy_notebook)
    assert result.outcome == Outcome.VERIFIED
    assert result.attempts[0].result.startswith("model call failed")


def test_gives_up_after_max_attempts(broken_numpy_notebook):
    still_broken = patch_reply(2, "x = np.array([1, 2, 3, 4])\nprint(x.astype(np.float).mean())")
    llm = ScriptedLLM([still_broken, still_broken])
    agent, events = make_agent(llm, max_attempts=2)
    result = agent.repair(broken_numpy_notebook)
    assert result.outcome == Outcome.FAILED and not result.runs
    assert len(result.attempts) == 2
    assert "AttributeError" in result.last_error
    assert events[-1] == {"type": "done", "outcome": "failed", "attempts": 2, "last_error": result.last_error, "reference": "saved"}


def test_fix_that_changes_results_gets_a_drift_retry(broken_numpy_notebook):
    llm = ScriptedLLM([patch_reply(2, WRONG_CELL_2), patch_reply(2, FIX_CELL_2, explanation="restore the mean")])
    agent, _ = make_agent(llm)
    result = agent.repair(broken_numpy_notebook)
    assert result.outcome == Outcome.VERIFIED
    assert result.attempts[1].error == "outputs differ from the saved outputs"
    drift_prompt = llm.deep_prompts()[1]
    assert "no longer match" in drift_prompt and "2.5 -> 10" in drift_prompt


def test_drift_is_reported_when_attempts_run_out(broken_numpy_notebook):
    llm = ScriptedLLM([patch_reply(2, WRONG_CELL_2)])
    agent, _ = make_agent(llm, max_attempts=1)
    result = agent.repair(broken_numpy_notebook)
    assert result.outcome == Outcome.DRIFT and result.runs
    assert not result.comparison.verified


def test_drift_retry_can_be_disabled(broken_numpy_notebook):
    llm = ScriptedLLM([patch_reply(2, WRONG_CELL_2)])
    agent, _ = make_agent(llm, fix_drift=False)
    assert agent.repair(broken_numpy_notebook).outcome == Outcome.DRIFT


def test_search_and_query_failures_do_not_stop_the_repair(broken_numpy_notebook):
    llm = ScriptedLLM([patch_reply(2, FIX_CELL_2)], fail_fast=True)
    search = FakeSearch(fail=True)
    agent, events = make_agent(llm, search=search)
    result = agent.repair(broken_numpy_notebook)
    assert result.outcome == Outcome.VERIFIED
    assert "AttributeError" in search.queries[0]  # fallback query was used
    warnings = [e["message"] for e in events if e["type"] == "warning"]
    assert any("query model failed" in w for w in warnings)
    assert any("web search failed" in w for w in warnings)


def test_without_search_no_query_model_call(broken_numpy_notebook):
    llm = ScriptedLLM([patch_reply(2, FIX_CELL_2)])
    agent, _ = make_agent(llm)
    agent.repair(broken_numpy_notebook)
    assert [c["role"] for c in llm.calls] == ["deep"]


def test_pip_pins_are_installed_in_the_env(broken_numpy_notebook):
    env = FakeEnv()
    llm = ScriptedLLM([patch_reply(2, FIX_CELL_2, pip=["numpy==2.0.0"])])
    agent, events = make_agent(llm, env=env)
    result = agent.repair(broken_numpy_notebook)
    assert result.outcome == Outcome.VERIFIED
    assert env.install_calls == [["numpy==2.0.0"]]
    assert result.installed == ["numpy==2.0.0"]
    assert result.attempts[0].pip_install == ["numpy==2.0.0"]


def test_failed_install_is_fed_back(broken_numpy_notebook):
    env = FakeEnv(install_ok=False)
    llm = ScriptedLLM([patch_reply(2, FIX_CELL_2, pip=["numpy==0.0.1"]), patch_reply(2, FIX_CELL_2)])
    agent, _ = make_agent(llm, env=env)
    result = agent.repair(broken_numpy_notebook)
    assert result.outcome == Outcome.VERIFIED
    assert result.attempts[0].result == "pip install failed"
    assert "no matching distribution" in llm.deep_prompts()[1]


def test_without_env_pins_are_dropped_but_code_edits_still_apply(broken_numpy_notebook):
    llm = ScriptedLLM([patch_reply(2, FIX_CELL_2, pip=["numpy==1.26.4"])])
    agent, events = make_agent(llm)
    result = agent.repair(broken_numpy_notebook)
    assert result.outcome == Outcome.VERIFIED
    install = [e for e in events if e["type"] == "install"][0]
    assert install["ok"] is False


def test_result_to_dict_is_json_serialisable(broken_numpy_notebook):
    llm = ScriptedLLM([patch_reply(2, FIX_CELL_2)])
    agent, _ = make_agent(llm, search=FakeSearch())
    data = json.loads(json.dumps(agent.repair(broken_numpy_notebook).to_dict()))
    assert data["outcome"] == "verified" and data["attempts"][0]["sources"]


def test_original_notebook_is_never_modified(broken_numpy_notebook):
    before = json.dumps(broken_numpy_notebook)
    llm = ScriptedLLM([patch_reply(2, FIX_CELL_2)])
    make_agent(llm)[0].repair(broken_numpy_notebook)
    assert json.dumps(broken_numpy_notebook) == before


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError):
        ReviveAgent(ScriptedLLM([]), max_attempts=0)


def test_fallback_query_mentions_library_and_error():
    q = fallback_query("AttributeError", "module 'numpy' has no attribute 'float'", ["numpy", "pandas"])
    assert q.startswith("numpy AttributeError") and "float" in q


def _three_breaks():
    return make_notebook(
        "import numpy as np",
        ("print(int(np.int(1)))", [stdout_output("1\n")]),
        ("print(float(np.float(2)))", [stdout_output("2.0\n")]),
        ("print(np.asscalar(np.array([5])))", [stdout_output("5\n")]),
    )


THREE_FIXES = [
    patch_reply(1, "print(int(int(1)))", "np.int was removed"),
    patch_reply(2, "print(float(float(2)))", "np.float was removed"),
    patch_reply(3, "print(np.array([5]).item())", "np.asscalar was removed"),
]


def test_attempts_that_make_progress_do_not_use_up_the_budget():
    agent = ReviveAgent(ScriptedLLM(list(THREE_FIXES)), max_attempts=1, max_total_attempts=5, cell_timeout=60)
    result = agent.repair(_three_breaks())
    assert result.outcome == Outcome.VERIFIED and len(result.attempts) == 3


def test_without_an_extension_the_budget_is_a_hard_limit():
    agent = ReviveAgent(ScriptedLLM(list(THREE_FIXES)), max_attempts=2, cell_timeout=60)
    result = agent.repair(_three_breaks())
    assert result.outcome == Outcome.FAILED and len(result.attempts) == 2


def test_attempts_without_progress_still_stop_the_repair():
    useless = [patch_reply(1, "print(int(np.int(1)))  # unchanged call", "try again")] * 3
    agent = ReviveAgent(ScriptedLLM(useless), max_attempts=2, max_total_attempts=10, cell_timeout=60)
    result = agent.repair(_three_breaks())
    assert result.outcome == Outcome.FAILED and len(result.attempts) == 2


def test_a_patch_may_not_bring_back_code_that_already_failed():
    nb = make_notebook(
        "import numpy as np",
        ("print(int(np.int(7)))", [stdout_output("7\n")]),
    )
    replies = [
        patch_reply(1, "print(int(np.int(7)) + 1)", "first try"),  # still raises: np.int is gone
        patch_reply(1, "print(int(np.int(7)))", "back to the original"),  # identical to the failed original
        patch_reply(1, "print(int(7))", "np.int was removed"),
    ]
    events = []
    agent = ReviveAgent(ScriptedLLM(replies), max_attempts=4, cell_timeout=60, on_event=events.append)
    result = agent.repair(nb)
    assert result.outcome == Outcome.VERIFIED
    rejected = [e for e in events if e["type"] == "patch_rejected"]
    assert len(rejected) == 1 and "already failed" in rejected[0]["reason"] and "AttributeError" in rejected[0]["reason"]
