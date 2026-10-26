from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load():
    spec = importlib.util.spec_from_file_location("deploy_huggingface", ROOT / "scripts" / "deploy_huggingface.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collects_app_files_and_skips_junk():
    files = {p.as_posix() for p in load().collect_files()}
    for required in ("Dockerfile", "pyproject.toml", "LICENSE", "revive/web/app.py", "revive/web/static/index.html",
                     "benchmark/manifest.json", "benchmark/notebooks/pandas_append.ipynb"):
        assert required in files
    assert not any("__pycache__" in f or f.startswith(".venv") or f.startswith("tests/") for f in files)
    assert "README.md" not in files  # the Space gets its own README with the required header


def test_space_readme_has_docker_header_and_link():
    text = load().space_readme("https://github.com/someone/revive")
    assert text.startswith("---\n") and "sdk: docker" in text and "app_port: 7860" in text
    assert "https://github.com/someone/revive" in text


def test_dry_run_needs_no_token(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert load().main(["--dry-run"]) == 0
    assert "files to upload" in capsys.readouterr().out


def test_missing_token_is_an_error(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert load().main([]) == 2
    assert "HF_TOKEN" in capsys.readouterr().err


def test_dockerfile_matches_space_requirements():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "--uid 1000" in dockerfile and "PORT=7860" in dockerfile
    # Uploaded notebooks must run as the unprivileged sandbox user on the hosted demo.
    assert "REVIVE_SANDBOX_USER=sandbox" in dockerfile and "useradd" in dockerfile and "USER " not in dockerfile
    assert "requirements-demo.txt" in dockerfile
