"""Isolated notebook environments. Creating a real venv takes a few seconds."""

from __future__ import annotations

import json
import subprocess

import pytest

from revive.env import KERNEL_NAME, NotebookEnv
from revive.imports import imported_modules, notebook_distributions
from revive.runner import execute_notebook

from conftest import make_notebook


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("env") / "venv"
    environment = NotebookEnv.create(root)
    yield environment
    environment.destroy()


def test_kernel_runs_on_the_env_python_and_sees_parent_packages(env):
    nb = make_notebook("import sys, numpy\nprint(sys.executable)")
    result = execute_notebook(nb, kernel_manager_factory=env.kernel_manager, timeout=60)
    assert result.ok, result.error_summary()
    assert str(env.root) in result.notebook.cells[0].outputs[0].text


@pytest.mark.skipif(__import__("os").name == "nt", reason="os.kill(pid, 0) terminates processes on Windows")
def test_env_kernel_is_shut_down_after_each_run(env):
    import os

    nb = make_notebook("import os\nprint(os.getpid())")
    result = execute_notebook(nb, kernel_manager_factory=env.kernel_manager, timeout=60)
    pid = int(result.notebook.cells[0].outputs[0].text)
    try:
        os.kill(pid, 0)  # raises if the process no longer exists
    except ProcessLookupError:
        return
    # Allow a moment for the process to be reaped.
    import time

    for _ in range(50):
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
    raise AssertionError(f"kernel process {pid} is still running")


def test_kernelspec_points_at_env_python(env):
    spec = json.loads((env.kernel_dir / KERNEL_NAME / "kernel.json").read_text())
    assert spec["argv"][0] == str(env.python)


def test_package_versions(env):
    versions = env.package_versions(["numpy", "definitely-not-installed-xyz"])
    assert versions["numpy"][0].isdigit()
    assert versions["definitely-not-installed-xyz"] == "not installed"
    assert "python" in versions


def test_unsafe_specs_never_reach_pip(tmp_path):
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    env = NotebookEnv(tmp_path, runner=runner)
    result = env.install(["numpy==1.26.4", "x; rm -rf /"])
    assert not result.ok and result.rejected == ["x; rm -rf /"]
    assert calls == []


def test_install_builds_pip_command_and_records_success(tmp_path):
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "Successfully installed", "")

    env = NotebookEnv(tmp_path, runner=runner)
    result = env.install(["pandas==1.5.3"])
    assert result.ok and env.installed == ["pandas==1.5.3"]
    assert calls[0][1:4] == ["-m", "pip", "install"] and calls[0][-1] == "pandas==1.5.3"


def test_failed_install_is_not_recorded(tmp_path):
    def runner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "ERROR: No matching distribution")

    env = NotebookEnv(tmp_path, runner=runner)
    result = env.install(["numpy==0.0.1"])
    assert not result.ok and "No matching distribution" in result.output
    assert env.installed == []


def test_pip_timeout_is_reported(tmp_path):
    def runner(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    result = NotebookEnv(tmp_path, runner=runner).install(["numpy"])
    assert not result.ok and "timed out" in result.output


def test_empty_install_is_a_no_op(tmp_path):
    def runner(cmd, **kwargs):
        raise AssertionError("pip should not run")

    assert NotebookEnv(tmp_path, runner=runner).install([]).ok


def test_imported_modules():
    sources = ["import numpy as np\nimport os, sys\nfrom sklearn.model_selection import train_test_split",
               "  import pandas.api as pa\nx = 'import fake'"]
    assert imported_modules(sources) == ["numpy", "os", "sys", "sklearn", "pandas"]


def test_notebook_distributions_maps_names_and_skips_stdlib():
    nb = make_notebook("import numpy as np, os", "from sklearn import datasets", "from PIL import Image", ("md", "import x"))
    assert notebook_distributions(nb) == ["numpy", "scikit-learn", "Pillow"]
