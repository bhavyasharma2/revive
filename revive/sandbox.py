"""Run notebook kernels as an unprivileged user.

An uploaded notebook is arbitrary code. On a public server it must not be able
to read the server's secrets (the Token Factory and Tavily API keys live in its
environment) or tamper with the environments Revive installs packages into.

When the server runs as root and ``REVIVE_SANDBOX_USER`` names an existing
user, every kernel is started through this module:

    python -m revive.sandbox <user> <python> -m ipykernel_launcher -f <connection file>

The process Jupyter starts is a small supervisor. It hands the kernel a copy
of its connection file, then forks: the child drops to the sandbox user with a
minimal environment and becomes the kernel; the parent keeps only the right to
be signalled by the server, passes every signal on to the kernel (interrupt,
terminate), and exits when the kernel does. If the supervisor is killed, the
kernel dies with it.

Why not simply switch user and become the kernel? Some hosts (Render, for one)
take away root's right to signal other users' processes (CAP_KILL), and then
Jupyter could not interrupt or stop the kernel ("Operation not permitted").
The supervisor keeps root as its saved user id, which is what lets the server
signal it, while its real and effective user ids are the sandbox user's,
which is what lets it signal the kernel.

The kernel cannot see the server's environment variables or read them from
``/proc`` (the server and the supervisor are not readable by it), and cannot
write to the root-owned virtual environments. A self-check at startup verifies
all of this works on the host; if it does not, sandboxing is reported as
unavailable (and the web app turns uploads off).

Without that setup (local use, Windows, non-root containers) kernels start
normally.
"""

from __future__ import annotations

import functools
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

SANDBOX_ENV = "REVIVE_SANDBOX_USER"

# Per-user process limit (threads count too) and the largest file a notebook may write.
MAX_PROCESSES = 512
MAX_FILE_BYTES = 512 * 1024 * 1024

FORWARDED_SIGNALS = ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2")
PROBE_READY = "revive-sandbox-ready"


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
    return name if _works(name) else None


def active() -> bool:
    return sandbox_user() is not None


@functools.lru_cache(maxsize=None)
def _works(user: str) -> bool:
    """Start a sandboxed process the way kernels are started, then stop it the way Jupyter does."""
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "revive.sandbox", "--probe", user],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
    except OSError:
        return False
    try:
        ready = proc.stdout.readline().strip() if proc.stdout else ""
        if ready != PROBE_READY:
            return False
        os.killpg(proc.pid, signal.SIGINT)  # what an interrupt does
        time.sleep(0.2)
        proc.send_signal(signal.SIGTERM)  # what a shutdown does
        return proc.wait(timeout=15) == 0
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return False
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except OSError:
                pass


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


def _die_with_parent() -> None:
    """Ask the kernel to receive SIGKILL when the supervisor exits (Linux prctl PR_SET_PDEATHSIG)."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")


def _run_child(pw, cmd: list[str], supervisor: int) -> None:
    """In the forked child: become the sandbox user, then the kernel. Never returns."""
    import resource

    try:
        os.setpgid(0, 0)  # its own process group: signals reach it only through the supervisor
        resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setgroups([])
        os.setresgid(pw.pw_gid, pw.pw_gid, pw.pw_gid)
        os.setresuid(pw.pw_uid, pw.pw_uid, pw.pw_uid)
        if 0 in os.getresuid() or 0 in os.getresgid():  # never run a notebook as root by accident
            os._exit(1)
        _die_with_parent()  # after the user switch, which would clear it
        if os.getppid() != supervisor:  # the supervisor is already gone
            os._exit(1)
        os.execve(cmd[0], cmd, _clean_env(pw.pw_dir))
    except BaseException:  # noqa: BLE001 - a failed child must never fall back into the parent's code
        os._exit(127)


def main(argv: Sequence[str]) -> int:
    import pwd

    args = list(argv)
    probe = bool(args) and args[0] == "--probe"
    if probe:
        args = args[1:]
    if not args or (not probe and len(args) < 2):
        print("usage: python -m revive.sandbox [--probe] <user> <kernel command...>", file=sys.stderr)
        return 2
    if os.geteuid() != 0:
        print("revive.sandbox must start as root", file=sys.stderr)
        return 2
    user, cmd = args[0], args[1:]
    pw = pwd.getpwnam(user)

    if probe:
        cmd = [sys.executable, "-c", f"import time; print({PROBE_READY!r}, flush=True); time.sleep(60)"]
    else:
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

    supervisor = os.getpid()
    child = os.fork()
    if child == 0:
        _run_child(pw, cmd, supervisor)

    # Supervisor: keep root only as the saved user id, so the server may still signal us,
    # and use the sandbox user's ids, so we may signal the kernel.
    os.setgroups([])
    os.setresgid(pw.pw_gid, pw.pw_gid, 0)
    os.setresuid(pw.pw_uid, pw.pw_uid, 0)

    def forward(signum, _frame):
        try:
            os.kill(child, signum)
        except ProcessLookupError:
            pass

    for name in FORWARDED_SIGNALS:
        signal.signal(getattr(signal, name), forward)

    while True:
        try:
            _, status = os.waitpid(child, 0)
            break
        except ChildProcessError:
            return 1
    if probe:
        # The probe is stopped on purpose: success means the kernel was reached and ended.
        return 0 if os.WIFSIGNALED(status) or os.WIFEXITED(status) else 1
    code = os.waitstatus_to_exitcode(status)
    return 128 - code if code < 0 else code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
