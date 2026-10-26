"""Run notebook kernels as an unprivileged user.

An uploaded notebook is arbitrary code. On a public server it must not be able
to read the server's secrets (the Token Factory and Tavily API keys live in its
environment) or tamper with the environments Revive installs packages into.

When the server runs as root and ``REVIVE_SANDBOX_USER`` names an existing
user, every kernel is started through this module:

    python -m revive.sandbox <user> <python> -m ipykernel_launcher -f <connection file>

It runs as root just long enough to hand the kernel a copy of its connection
file, then drops to the sandbox user with a minimal environment and replaces
itself with the kernel (same process id, so Jupyter can still stop it). The
kernel can no longer see the server's environment variables, read
``/proc/<server pid>/environ`` (owned by root), or write to the root-owned
virtual environments.

Without that setup (local use, Windows, non-root containers) kernels start
normally.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

SANDBOX_ENV = "REVIVE_SANDBOX_USER"

# Per-user process limit (threads count too) and the largest file a notebook may write.
MAX_PROCESSES = 512
MAX_FILE_BYTES = 512 * 1024 * 1024


def requested() -> bool:
    """True when the operator asked for sandboxed kernels."""
    return bool(os.environ.get(SANDBOX_ENV, "").strip())


def sandbox_user() -> Optional[str]:
    """The user kernels run as, or None when sandboxing is not possible here."""
    name = os.environ.get(SANDBOX_ENV, "").strip()
    if not name or os.name != "posix" or os.geteuid() != 0:
        return None
    import pwd

    try:
        pwd.getpwnam(name)
    except KeyError:
        return None
    return name


def active() -> bool:
    return sandbox_user() is not None


def kernel_argv(python: Path | str) -> list[str]:
    """The kernelspec argv for a kernel running on ``python``."""
    base = [str(python), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    user = sandbox_user()
    if user is None:
        return base
    return [sys.executable, "-m", "revive.sandbox", user, *base]


def make_traversable(path: Path | str) -> None:
    """Let the sandbox user reach (read and execute, never write) ``path``.

    Temporary directories are created with mode 0700; a kernel whose Python
    lives inside one could not start. Only directories under the system temp
    folder are touched.
    """
    if not active():
        return
    tmp_root = Path(tempfile.gettempdir()).resolve()
    p = Path(path).resolve()
    for d in [p, *p.parents]:
        if d == tmp_root or tmp_root not in d.parents:
            break
        if d.is_dir():
            os.chmod(d, os.stat(d).st_mode | 0o055)


def make_workdir(path: Path | str) -> Path:
    """Create a directory the notebook may write to (its working directory)."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    if active():
        make_traversable(p.parent)
        os.chmod(p, 0o777)
    return p


def _clean_env(home: str) -> dict[str, str]:
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": home,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
    }
    parent = os.environ.get("JPY_PARENT_PID")
    if parent:
        env["JPY_PARENT_PID"] = parent
    return env


def main(argv: Sequence[str]) -> int:
    import pwd
    import resource

    if len(argv) < 2:
        print("usage: python -m revive.sandbox <user> <kernel command...>", file=sys.stderr)
        return 2
    if os.geteuid() != 0:
        print("revive.sandbox must start as root", file=sys.stderr)
        return 2
    user, cmd = argv[0], list(argv[1:])
    pw = pwd.getpwnam(user)

    try:
        i = cmd.index("-f")
        conn = cmd[i + 1]
    except (ValueError, IndexError):
        print("revive.sandbox: the kernel command has no -f <connection file>", file=sys.stderr)
        return 2

    # The connection file is private to root (mode 0600). Give the kernel its own copy.
    fd, copy = tempfile.mkstemp(prefix="revive-kernel-", suffix=".json")
    os.close(fd)
    shutil.copyfile(conn, copy)
    os.chown(copy, pw.pw_uid, pw.pw_gid)
    os.chmod(copy, 0o600)
    cmd[i + 1] = copy

    resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    os.setgroups([])
    os.setgid(pw.pw_gid)
    os.setuid(pw.pw_uid)
    if os.getuid() == 0 or os.geteuid() == 0:  # never run a notebook as root by accident
        print("revive.sandbox: could not drop privileges", file=sys.stderr)
        return 1

    os.execve(cmd[0], cmd, _clean_env(pw.pw_dir))
    return 1  # unreachable


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
