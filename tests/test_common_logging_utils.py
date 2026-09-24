from __future__ import annotations

import logging

from common.logging_utils import configure_logging, get_logger


def test_configure_logging_uses_env_level_and_stdout(monkeypatch, capsys) -> None:
    monkeypatch.setenv("BENCHMARK_LOG_LEVEL", "debug")

    configure_logging()
    logger = logging.getLogger("benchmarks.logging-test")
    logger.debug("debug-visible")

    captured = capsys.readouterr()
    assert logging.getLogger().level == logging.DEBUG
    assert "DEBUG benchmarks.logging-test: debug-visible" in captured.out


def test_configure_logging_falls_back_to_info_for_unknown_level(monkeypatch, capsys) -> None:
    monkeypatch.setenv("BENCHMARK_LOG_LEVEL", "not-a-level")

    configure_logging()
    logger = logging.getLogger("benchmarks.logging-test")
    logger.debug("debug-hidden")
    logger.info("info-visible")

    captured = capsys.readouterr()
    assert logging.getLogger().level == logging.INFO
    assert "debug-hidden" not in captured.out
    assert "INFO benchmarks.logging-test: info-visible" in captured.out


def test_get_logger_configures_root_when_no_handlers(monkeypatch) -> None:
    monkeypatch.delenv("BENCHMARK_LOG_LEVEL", raising=False)
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    logger = get_logger("benchmarks.lazy")

    assert logger.name == "benchmarks.lazy"
    assert root.handlers
    assert root.level == logging.INFO
