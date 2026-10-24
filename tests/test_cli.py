from __future__ import annotations

from pathlib import Path

from revive.__main__ import main
from revive.notebook import has_stored_outputs, load_notebook
from revive.runner import execute_notebook

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "legacy_churn_model.ipynb"


def test_missing_notebook_returns_2(tmp_path, capsys):
    assert main([str(tmp_path / "nope.ipynb")]) == 2
    assert "not found" in capsys.readouterr().err


def test_missing_api_key_returns_2(monkeypatch, capsys):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    assert main([str(EXAMPLE), "--no-env", "--no-search"]) == 2
    assert "NEBIUS_API_KEY" in capsys.readouterr().err


def test_example_notebook_is_broken_today_and_has_saved_outputs():
    nb = load_notebook(EXAMPLE)
    assert has_stored_outputs(nb)
    result = execute_notebook(nb, timeout=120)
    assert not result.ok and result.failed_cell == 3
    assert "append" in result.error_value


def test_invalid_written_before_date_returns_2(monkeypatch, capsys):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    assert main([str(EXAMPLE), "--no-env", "--no-search", "--written-before", "last year"]) == 2
    assert "--written-before" in capsys.readouterr().err
