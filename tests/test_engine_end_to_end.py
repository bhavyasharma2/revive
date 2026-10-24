"""The core promise of Revive, without any model in the loop.

A broken notebook is run, fails at the right cell, gets a patch of the kind the
model produces, runs again, and is verified against the outputs the original
author saved. A second patch that runs cleanly but changes the numbers must be
caught as drift: that is the difference between "it runs" and "it is correct".
"""

from __future__ import annotations

import json

from revive.compare import Status, compare_notebooks
from revive.patches import apply_patch, parse_patch
from revive.runner import execute_notebook


def _model_reply(cell_index: int, source: str) -> str:
    return "<think>np.float was removed in NumPy 1.24</think>```json\n" + json.dumps(
        {"explanation": "use a concrete dtype", "edits": [{"cell_index": cell_index, "new_source": source}]}
    ) + "\n```"


def test_broken_notebook_is_fixed_and_verified(broken_numpy_notebook):
    first = execute_notebook(broken_numpy_notebook, timeout=60)
    assert not first.ok and first.failed_cell == 2

    patch = parse_patch(
        _model_reply(2, "x = np.array([1, 2, 3, 4])\nprint(x.astype(np.float64).mean())")
    )
    fixed = apply_patch(broken_numpy_notebook, patch)

    second = execute_notebook(fixed, timeout=60)
    assert second.ok, second.error_summary()

    report = compare_notebooks(broken_numpy_notebook, second.notebook)
    assert report.verified, report.summary()
    assert report.compared == 2 and report.match_rate == 1.0


def test_fix_that_runs_but_changes_results_is_flagged_as_drift(broken_numpy_notebook):
    # This "fix" removes the error but also silently changes what the notebook computes.
    patch = parse_patch(
        _model_reply(2, "x = np.array([1, 2, 3, 4])\nprint(x.astype(np.float64).sum())")
    )
    wrong = apply_patch(broken_numpy_notebook, patch)

    run = execute_notebook(wrong, timeout=60)
    assert run.ok  # it runs...

    report = compare_notebooks(broken_numpy_notebook, run.notebook)
    assert not report.verified  # ...but it is not the same notebook
    drift = [c for c in report.cells if c.status == Status.DRIFT]
    assert len(drift) == 1 and drift[0].cell_index == 2
    assert "2.5 -> 10" in drift[0].detail
