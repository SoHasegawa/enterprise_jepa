"""Verbosity helpers shared by launchers and config loaders."""

from __future__ import annotations

VERBOSITY_LEVELS = ("quiet", "progress", "verbose")


def validate_verbosity(value: str, *, label: str) -> None:
    """Reject unsupported verbosity values with a clear config-specific message."""

    if value not in VERBOSITY_LEVELS:
        allowed = ", ".join(VERBOSITY_LEVELS)
        raise ValueError(f"{label} verbosity must be one of: {allowed}.")


def progress_enabled(value: str) -> bool:
    """Return whether normal progress messages should be printed."""

    return value in {"progress", "verbose"}


def verbose_enabled(value: str) -> bool:
    """Return whether diagnostic messages should be printed."""

    return value == "verbose"
