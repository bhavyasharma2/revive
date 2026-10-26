"""Kernels for uploaded notebooks must not be able to read the server's secrets."""

from __future__ import annotations

import os
import pwd
import sys

import nbformat
import pytest

from revive import sandbox
from revive.env import NotebookEnv
from revive.runner import execute_notebook

SANDBOX_TEST_USER = "revivesbx"


def test_plain_kernel_without_sandbox(monkeypatch):
    monkeypatch.delenv(sandbox.SANDBOX_ENV, raising=False)
    assert not sandbox.requested()
    assert not sandbox.active()
    assert sandbox.kernel_argv("/x/python") == ["/x/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"]


def test_unknown_user_means_no_sandbox(monkeypatch):
    monkeypatch.setenv(sandbox.SANDBOX_ENV, "no-such-user-revive")
    assert sandbox.requested()
    assert not sandbox.active()
    assert sandbox.kernel_argv("/x/python")[0] == "/x/python"


def test_uploads_refused_when_sandbox_requested_but_unavailable(monkeypatch):
    from fastapi.testclient import TestClient
    from revive.web.app import create_app

    monkeypatch.setenv("REVIVE_ALLOW_UPLOADS", "1")
    monkeypatch.setenv(sandbox.SANDBOX_ENV, "no-such-user-revive")
    client = TestClient(create_app(repair=lambda nb, emit, options: None))
    assert client.get("/api/config").json()["allow_uploads"] is False
    res = client.post("/api/jobs", json={"notebook": "{}", "live": True})
    assert res.status_code == 403


def _have_sandbox_user() -> bool:
    if os.name != "posix" or os.geteuid() != 0:
        return False
    try:
        pwd.getpwnam(SANDBOX_TEST_USER)
    except KeyError:
        return False
    return True


THIEF = [
    "import os\nprint('uid', os.getuid() != 0)",
    "print('secret visible', 'REVIVE_TEST_SECRET' in os.environ)",
    (
        "try:\n"
        "    open(f'/proc/{os.getppid()}/environ', 'rb').read()\n"
        "    print('parent environ readable')\n"
        "except PermissionError:\n"
        "    print('parent environ blocked')"
    ),
    (
        "import sysconfig\n"
        "try:\n"
        "    open(os.path.join(sysconfig.get_paths()['purelib'], 'evil.pth'), 'w').write('x')\n"
        "    print('venv writable')\n"
        "except PermissionError:\n"
        "    print('venv read-only')"
    ),
    "open('result.txt', 'w').write('ok')\nprint('workdir writable')",
]


@pytest.mark.skipif(not _have_sandbox_user(), reason=f"needs root and a '{SANDBOX_TEST_USER}' user")
def test_sandboxed_kernel_cannot_read_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv(sandbox.SANDBOX_ENV, SANDBOX_TEST_USER)
    monkeypatch.setenv("REVIVE_TEST_SECRET", "sk-should-never-leak")
    assert sandbox.active()

    env = NotebookEnv.create(tmp_path / "venv")
    work = sandbox.make_workdir(tmp_path / "work")
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(src) for src in THIEF]

    result = execute_notebook(nb, cwd=str(work), timeout=120, kernel_manager_factory=env.kernel_manager)
    assert result.ok, result.error_summary()
    printed = "".join(
        o.get("text", "") for c in result.notebook.cells for o in c.get("outputs", []) if o.get("output_type") == "stream"
    )
    assert "uid True" in printed
    assert "secret visible False" in printed
    assert "parent environ blocked" in printed
    assert "venv read-only" in printed
    assert "workdir writable" in printed
    assert "sk-should-never-leak" not in printed
    assert (work / "result.txt").read_text() == "ok"
