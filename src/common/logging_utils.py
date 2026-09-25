import logging
import os
import sys


def configure_logging() -> None:
    """Initialize simple stdout logging."""
    level_name = os.getenv("BENCHMARK_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the shared configuration applied."""
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
