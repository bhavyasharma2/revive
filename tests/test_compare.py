from __future__ import annotations

import pytest
from nbformat.v4 import new_output

from revive.compare import (
    Status,
    compare_notebooks,
    compare_text,
    comparable_text,
    normalize_text,
    split_numbers,
)

from conftest import make_notebook, result_output, stdout_output


def test_identical_text_matches():
    status, rel, detail = compare_text("accuracy: 0.91\n", "accuracy: 0.91\n")
    assert status == Status.MATCH and rel == 0.0 and detail == ""


def test_numbers_within_tolerance_match():
    status, rel, _ = compare_text("loss 0.123456", "loss 0.123457", rtol=1e-3)
    assert status == Status.MATCH
    assert 0 < rel < 1e-3


def test_numbers_beyond_tolerance_are_drift():
    status, rel, detail = compare_text("accuracy 0.9123", "accuracy 0.9487", rtol=1e-3)
    assert status == Status.DRIFT
    assert rel == pytest.approx(0.0384, abs=1e-3)
    assert "0.9123" in detail and "0.9487" in detail


def test_last_printed_digit_flip_is_rounding_noise_at_three_or_more_decimals():
    # Real case: old and new scikit-learn solvers print 0.182 vs 0.181.
    assert compare_text("[0.016, 0.182, 0.701]", "[0.016, 0.181, 0.701]")[0] == Status.MATCH
    assert compare_text("rmse 12.3457", "rmse 12.3456")[0] == Status.MATCH
    assert compare_text("1.234e-05", "1.235e-05")[0] == Status.MATCH


def test_bigger_or_coarse_differences_still_drift():
    assert compare_text("[0.016, 0.182]", "[0.016, 0.184]")[0] == Status.DRIFT   # two units
    assert compare_text("accuracy 0.94", "accuracy 0.95")[0] == Status.DRIFT     # only 2 decimals
    assert compare_text("402", "403")[0] == Status.DRIFT                          # integers exact
    assert compare_text("1975.13", "2166.14")[0] == Status.DRIFT


def test_different_words_are_changed_not_drift():
    status, _, detail = compare_text("accuracy 0.91", "precision 0.91")
    assert status == Status.CHANGED
    assert "accuracy" in detail and "precision" in detail


def test_different_number_of_numbers_is_changed():
    status, _, _ = compare_text("values 1 2 3", "values 1 2")
    assert status == Status.CHANGED


def test_timing_lines_are_ignored():
    stored = "result 42\nWall time: 1.2 s\n"
    fresh = "result 42\nWall time: 9.9 s\n"
    assert compare_text(stored, fresh)[0] == Status.MATCH


def test_progress_bars_are_ignored():
    stored = "done\n100%|██████████| 10/10 [00:01<00:00, 9.1it/s]\n"
    fresh = "done\n100%|██████████| 10/10 [00:02<00:00, 4.3it/s]\n"
    assert compare_text(stored, fresh)[0] == Status.MATCH


def test_memory_addresses_are_ignored():
    assert compare_text("<Model at 0x7f3a1c>", "<Model at 0x55d0aa>")[0] == Status.MATCH


def test_whitespace_differences_are_ignored():
    assert compare_text("a   b\n\n\nc", "a b\nc")[0] == Status.MATCH


def test_numpy2_scalar_repr_is_not_a_difference():
    # NumPy >= 2 prints np.float64(2.0) where older versions printed 2.0.
    assert compare_text("2.0", "np.float64(2.0)")[0] == Status.MATCH
    assert compare_text("[1, 2]", "[np.int64(1), np.int64(2)]")[0] == Status.MATCH


def test_numbers_inside_identifiers_stay_literal():
    skeleton, values = split_numbers("layer1 weight 0.5 v2.1")
    assert values == [0.5]
    assert "layer1" in skeleton and "v2.1" in skeleton


def test_scientific_notation_and_negatives():
    skeleton, values = split_numbers("lr=1e-05 delta=-0.25")
    assert values == [1e-05, -0.25]
    assert skeleton == "lr=# delta=#"


def test_comparable_text_keeps_stdout_and_results_only():
    outputs = [
        stdout_output("hello\n"),
        new_output("stream", name="stderr", text="UserWarning: old api\n"),
        result_output("42"),
        new_output("display_data", data={"image/png": "AAAA", "text/plain": "<Figure size 640x480>"}),
        new_output("error", ename="ValueError", evalue="x", traceback=["tb"]),
    ]
    text = comparable_text(outputs)
    assert "hello" in text and "42" in text and "<Figure size 640x480>" in text
    assert "UserWarning" not in text and "ValueError" not in text and "AAAA" not in text


def test_comparable_text_handles_list_text():
    outputs = [{"output_type": "stream", "name": "stdout", "text": ["a\n", "b\n"]}]
    assert comparable_text(outputs) == "a\nb\n"


def test_normalize_text_drops_empty_lines():
    assert normalize_text("\n  \nfoo\n\n") == "foo"


# ---- whole-notebook comparison -------------------------------------------------


def _pair(stored_outputs, fresh_outputs):
    original = make_notebook(("print('x')", stored_outputs))
    executed = make_notebook(("print('x')", fresh_outputs))
    return compare_notebooks(original, executed)


def test_notebook_all_match_is_verified():
    report = _pair([stdout_output("1.0\n")], [stdout_output("1.0\n")])
    assert report.verified and report.compared == 1 and report.match_rate == 1.0
    assert "verified" in report.summary()


def test_notebook_drift_is_not_verified():
    report = _pair([stdout_output("1.0\n")], [stdout_output("2.0\n")])
    assert not report.verified
    assert report.count(Status.DRIFT) == 1
    assert report.failures[0].cell_index == 0
    assert "NOT verified" in report.summary()


def test_notebook_missing_output_fails():
    report = _pair([stdout_output("1.0\n")], [])
    assert report.cells[0].status == Status.MISSING and not report.verified


def test_notebook_extra_output_does_not_fail():
    report = _pair([stdout_output("")], [stdout_output("hi\n")])
    assert report.cells[0].status == Status.EXTRA
    assert report.unverifiable and not report.verified  # nothing to check against


def test_notebook_without_outputs_is_unverifiable():
    report = _pair([], [])
    assert report.cells[0].status == Status.SKIPPED
    assert report.unverifiable and not report.verified
    assert "no saved outputs" in report.summary()


def test_markdown_cells_are_ignored_and_indices_follow_cells():
    original = make_notebook(("md", "# title"), ("print(1)", [stdout_output("1\n")]))
    executed = make_notebook(("md", "# title"), ("print(1)", [stdout_output("1\n")]))
    report = compare_notebooks(original, executed)
    assert [c.cell_index for c in report.cells] == [1]


def test_mismatched_notebooks_raise():
    one = make_notebook("a = 1")
    two = make_notebook("a = 1", "b = 2")
    with pytest.raises(ValueError):
        compare_notebooks(one, two)
    with pytest.raises(ValueError):
        compare_notebooks(make_notebook(("md", "x")), make_notebook("x"))


def test_report_to_dict_is_json_serialisable():
    import json

    report = _pair([stdout_output("1.0\n")], [stdout_output("2.0\n")])
    data = json.loads(json.dumps(report.to_dict()))
    assert data["verified"] is False and data["cells"][0]["status"] == "drift"
