"""Deploy the Revive web app to a Hugging Face Space (free CPU hardware).

One-time setup:
  1. Create a free account at https://huggingface.co
  2. Create an access token with "write" permission: https://huggingface.co/settings/tokens
  3. pip install huggingface_hub

Deploy (and redeploy after changes) from the repository folder:
  PowerShell:
    $env:HF_TOKEN="hf_..."; $env:NEBIUS_API_KEY="..."; $env:TAVILY_API_KEY="..."
    python scripts/deploy_huggingface.py --github https://github.com/<you>/revive
  bash:
    HF_TOKEN=hf_... NEBIUS_API_KEY=... TAVILY_API_KEY=... python scripts/deploy_huggingface.py

The API keys are stored as Space secrets (encrypted, never in the files).
Use --dry-run to see what would be uploaded without uploading anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INCLUDE = [
    "Dockerfile",
    ".dockerignore",
    "pyproject.toml",
    "LICENSE",
    "requirements-demo.txt",
    "revive/**/*.py",
    "revive/web/static/*",
    "benchmark/manifest.json",
    "benchmark/notebooks/*.ipynb",
    "benchmark/recordings/*.json",
    "benchmark/results/RESULTS.md",
    "examples/*.ipynb",
]
EXCLUDE_PARTS = {"__pycache__", ".venv", ".pytest_cache"}

SPACE_README = """---
title: Revive
emoji: 🩺
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
license: mit
short_description: Repairs broken ML notebooks and verifies the fix
---

# Revive

Revive repairs ML notebooks that broke when their libraries changed, using NVIDIA Nemotron
models on Nebius Token Factory and Tavily web search, then proves the fix: every cell is
re-run and checked against the output its author saved.

{source_line}
"""


def collect_files(root: Path = ROOT) -> list[Path]:
    """Files to upload, relative to ``root``, in a stable order."""
    found: set[Path] = set()
    for pattern in INCLUDE:
        for path in root.glob(pattern):
            if path.is_file() and not (set(path.relative_to(root).parts) & EXCLUDE_PARTS):
                if path.name.endswith((".revived.ipynb", ".pyc")):
                    continue
                found.add(path.relative_to(root))
    return sorted(found, key=lambda p: p.as_posix())


def space_readme(github: str | None) -> str:
    line = f"Source code, setup instructions and benchmark: {github}" if github else ""
    return SPACE_README.format(source_line=line).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deploy Revive to a Hugging Face Space.")
    parser.add_argument("--space", help="owner/name of the Space (default: <your username>/revive)")
    parser.add_argument("--github", help="public GitHub URL of the project, linked from the Space page")
    parser.add_argument("--dry-run", action="store_true", help="list what would be uploaded and stop")
    args = parser.parse_args(argv)

    files = collect_files()
    recordings = [f for f in files if f.parts[:2] == ("benchmark", "recordings")]
    print(f"{len(files)} files to upload ({len(recordings)} recorded runs for replay)")
    if not recordings:
        print("warning: no recorded runs found; run scripts/run_benchmark.py first so the demo can replay them")
    if args.dry_run:
        for f in files:
            print("  ", f.as_posix())
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("error: set HF_TOKEN to a Hugging Face access token with write permission", file=sys.stderr)
        return 2
    missing = [k for k in ("NEBIUS_API_KEY", "TAVILY_API_KEY") if not os.environ.get(k)]
    if "NEBIUS_API_KEY" in missing:
        print("error: set NEBIUS_API_KEY so the Space can run live repairs", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    except ImportError:
        print("error: run  pip install huggingface_hub  first", file=sys.stderr)
        return 2

    api = HfApi(token=token)
    space = args.space or f"{api.whoami()['name']}/revive"
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_secret(space, "NEBIUS_API_KEY", os.environ["NEBIUS_API_KEY"])
    if os.environ.get("TAVILY_API_KEY"):
        api.add_space_secret(space, "TAVILY_API_KEY", os.environ["TAVILY_API_KEY"])
    else:
        print("note: TAVILY_API_KEY not set; live repairs will run without web search")

    wanted = {f.as_posix() for f in files} | {"README.md"}
    existing = set(api.list_repo_files(space, repo_type="space"))
    operations = [CommitOperationAdd(path_in_repo=f.as_posix(), path_or_fileobj=str(ROOT / f)) for f in files]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=space_readme(args.github).encode("utf-8")))
    operations += [CommitOperationDelete(path_in_repo=p) for p in sorted(existing - wanted) if p != ".gitattributes"]

    api.create_commit(space, operations=operations, commit_message="Deploy Revive", repo_type="space")
    owner, name = space.split("/", 1)
    print(f"\nDeployed. The Space builds for a few minutes, then the app is live at:")
    print(f"  https://{owner}-{name}.hf.space".lower().replace("_", "-"))
    print(f"Build logs and settings: https://huggingface.co/spaces/{space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
