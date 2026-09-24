import logging
import os
import sys


def configure_logging() -> None:
    """標準出力向けの簡易ロギング設定を初期化する。"""
    level_name = os.getenv("BENCHMARK_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """共通設定済みのロガーを返す。"""
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
