"""These tests start real Jupyter kernels."""

from __future__ import annotations

import pytest

from revive.runner import execute_notebook

from conftest import make_notebook, stdout_output


def test_successful_run_collects_outputs_and_ignores_stale_ones():
    nb = make_notebook(
        ("md", "# title"),
        ("print('fresh')", [stdout_output("STALE\n")]),
        "1 + 1",
    )
    result = execute_notebook(nb, timeout=60)
    assert result.ok and result.failed_cell is None
    assert result.error_summary() == "ok"
    texts = [o.get("text") or o.get("data", {}).get("text/plain") for o in result.notebook.cells[1].outputs]
    assert texts == ["fresh\n"]
    assert result.notebook.cells[2].outputs[0].data["text/plain"] == "2"
    assert nb.cells[1].outputs[0].text == "STALE\n"  # the input is never modified


def test_failing_cell_is_located_with_error_details():
    nb = make_notebook("a = 1", "print('before')", "raise ValueError('boom')", "print('never runs')")
    result = execute_notebook(nb, timeout=60)
    assert not result.ok
    assert result.failed_cell == 2
    assert result.error_name == "ValueError" and result.error_value == "boom"
    assert "ValueError" in result.traceback and "\x1b[" not in result.traceback
    assert "cell 2" in result.error_summary()
    assert result.notebook.cells[3].outputs == []  # execution stopped at the failure


def test_missing_attribute_error_for_removed_api(broken_numpy_notebook):
    result = execute_notebook(broken_numpy_notebook, timeout=60)
    assert not result.ok
    assert result.failed_cell == 2          # index counts the markdown cell
    assert result.error_name == "AttributeError"
    assert "float" in result.error_value


def test_import_error_is_reported():
    nb = make_notebook("import definitely_not_a_real_module_xyz")
    result = execute_notebook(nb, timeout=60)
    assert result.error_name == "ModuleNotFoundError" and result.failed_cell == 0


def test_timeout_is_reported_with_the_right_cell():
    nb = make_notebook("print('quick')", "import time\ntime.sleep(30)", "print('never')")
    result = execute_notebook(nb, timeout=2)
    assert not result.ok and result.timed_out
    assert result.failed_cell == 1
    assert result.error_name == "CellTimeoutError"
    assert result.duration_s < 25


def test_dead_kernel_is_reported():
    nb = make_notebook("print('ok')", "import os\nos._exit(1)")
    result = execute_notebook(nb, timeout=30)
    assert not result.ok and result.kernel_died
    assert result.failed_cell == 1
    assert result.error_name == "DeadKernelError"


def test_cwd_controls_relative_paths(tmp_path):
    (tmp_path / "data.txt").write_text("hello from data")
    nb = make_notebook("print(open('data.txt').read())")
    result = execute_notebook(nb, cwd=tmp_path, timeout=60)
    assert result.ok
    assert result.notebook.cells[0].outputs[0].text == "hello from data\n"


def test_unknown_kernel_is_an_infrastructure_error():
    from revive.runner import RunnerError

    nb = make_notebook("1")
    with pytest.raises(RunnerError):
        execute_notebook(nb, kernel_name="no-such-kernel-xyz", timeout=10)
