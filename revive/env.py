"""Isolated Python environments, one per notebook being repaired.

Each environment is a virtualenv layered on top of the environment Revive
itself runs in: a ``.pth`` file makes the parent's packages importable, and
anything the agent installs into the child (for example an older pinned
library) is found first and shadows the parent's copy. Creating one takes
seconds because nothing is copied, and the parent environment is never
modified.

Executing a notebook runs arbitrary code. These environments isolate
*packages*, not the machine: run untrusted notebooks inside a container.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import venv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from .patches import PatchParseError, validate_pip_spec
from .sandbox import kernel_argv, make_traversable

KERNEL_NAME = "revive-env"
PIP_TIMEOUT_S = 900
_PTH_NAME = "_revive_parent.pth"

Runner = Callable[..., subprocess.CompletedProcess]


class EnvError(RuntimeError):
    """The environment could not be created or used."""


@dataclass
class InstallResult:
    ok: bool
    specs: list[str] = field(default_factory=list)
    output: str = ""
    rejected: list[str] = field(default_factory=list)


def parent_site_paths() -> list[str]:
    """Site-packages directories of the interpreter running Revive."""
    seen: list[str] = []
    for entry in sys.path:
        if entry and "site-packages" in entry.replace("\\", "/") and os.path.isdir(entry):
            resolved = str(Path(entry).resolve())
            if resolved not in seen:
                seen.append(resolved)
    return seen


def _venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


class NotebookEnv:
    """A throwaway environment with its own pip installs and Jupyter kernel."""

    def __init__(self, root: Path, *, runner: Runner = subprocess.run) -> None:
        self.root = Path(root).resolve()
        self.python = _venv_python(self.root)
        self.kernel_dir = self.root / "kernels"
        self.installed: list[str] = []
        self._run = runner

    @classmethod
    def create(cls, root: Path, *, runner: Runner = subprocess.run) -> "NotebookEnv":
        env = cls(root, runner=runner)
        env._build()
        return env

    def _build(self) -> None:
        try:
            venv.EnvBuilder(with_pip=True, clear=True, symlinks=(os.name != "nt")).create(str(self.root))
        except (OSError, subprocess.CalledProcessError) as exc:
            raise EnvError(f"could not create a virtualenv at {self.root}: {exc}") from exc
        if not self.python.exists():
            raise EnvError(f"virtualenv python not found at {self.python}")

        site_dir = Path(self._query("import sysconfig; print(sysconfig.get_paths()['purelib'])"))
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / _PTH_NAME).write_text("\n".join(parent_site_paths()) + "\n", encoding="utf-8")

        spec_dir = self.kernel_dir / KERNEL_NAME
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "argv": kernel_argv(self.python),
            "display_name": "Revive sandbox",
            "language": "python",
        }
        (spec_dir / "kernel.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")

        check = self._query("import ipykernel; print(ipykernel.__version__)")
        if not check:
            raise EnvError("ipykernel is not importable inside the environment")

    def _query(self, code: str) -> str:
        proc = self._run(
            [str(self.python), "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise EnvError(f"environment python failed: {proc.stderr.strip()[-500:]}")
        return proc.stdout.strip()

    def kernel_manager(self):
        """A fresh kernel manager whose kernel runs on this environment's python."""
        from jupyter_client.kernelspec import KernelSpecManager
        from jupyter_client.manager import AsyncKernelManager

        make_traversable(self.root)  # a sandboxed kernel must be able to reach this python
        ksm = KernelSpecManager(kernel_dirs=[str(self.kernel_dir)])
        return AsyncKernelManager(kernel_name=KERNEL_NAME, kernel_spec_manager=ksm)

    def install(self, specs: Sequence[str]) -> InstallResult:
        """pip-install validated requirement strings into this environment only."""
        clean: list[str] = []
        rejected: list[str] = []
        for spec in specs:
            try:
                clean.append(validate_pip_spec(spec))
            except PatchParseError:
                rejected.append(str(spec))
        if rejected:
            return InstallResult(ok=False, rejected=rejected, output="rejected unsafe requirement(s)")
        if not clean:
            return InstallResult(ok=True)

        cmd = [
            str(self.python), "-m", "pip", "install",
            "--disable-pip-version-check", "--no-input", "--progress-bar", "off",
            *clean,
        ]
        try:
            proc = self._run(cmd, capture_output=True, text=True, timeout=PIP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return InstallResult(ok=False, specs=clean, output=f"pip timed out after {PIP_TIMEOUT_S}s")
        output = ((proc.stdout or "") + (proc.stderr or ""))[-3000:]
        if proc.returncode == 0:
            self.installed.extend(clean)
        return InstallResult(ok=proc.returncode == 0, specs=clean, output=output)

    def package_versions(self, distributions: Sequence[str]) -> dict[str, str]:
        """Installed versions (as seen from this environment) of the given distributions."""
        if not distributions:
            return {}
        code = (
            "import json, importlib.metadata as m, sys\n"
            "out = {'python': sys.version.split()[0]}\n"
            f"for d in {list(distributions)!r}:\n"
            "    try: out[d] = m.version(d)\n"
            "    except m.PackageNotFoundError: out[d] = 'not installed'\n"
            "print(json.dumps(out))"
        )
        try:
            return json.loads(self._query(code))
        except (EnvError, ValueError):
            return {}

    def destroy(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
