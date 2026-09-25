"""Generic CLI behaviour: path resolution and signal translation."""

from __future__ import annotations

import importlib.util
import os
import signal
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("ejepa_cli_under_test", REPO_ROOT / "src" / "ejepa_cli" / "cli.py")
cli = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cli
_spec.loader.exec_module(cli)


def test_absolute_cli_path_resolves_relative_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "sessions" / "wm"
    checkpoint.mkdir(parents=True)

    assert cli._absolute_cli_path("sessions/wm") == str(checkpoint.resolve())
    assert cli._absolute_cli_path("  ") is None
    assert cli._absolute_cli_path(None) is None


def test_cli_translates_sigterm_to_keyboard_interrupt() -> None:
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"), cli._translate_termination_signals():
        os.kill(os.getpid(), signal.SIGTERM)
