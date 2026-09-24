"""Subprocess error formatting helpers."""

from __future__ import annotations

import subprocess


def process_error(label: str, completed: subprocess.CompletedProcess) -> str:
    """Return a readable error for a failed subprocess."""

    stderr = decode_process_output(completed.stderr)
    stdout = decode_process_output(completed.stdout)
    detail = stderr or stdout
    return f"{label} failed with exit code {completed.returncode}: {detail.strip()}"


def decode_process_output(output: bytes | str | None) -> str:
    """Decode optional subprocess output without raising on invalid UTF-8."""

    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def exception_summary(error: BaseException) -> str:
    """Return a compact exception summary for cleanup error messages."""

    text = str(error)
    if text:
        return f"{type(error).__name__}: {text}"
    return type(error).__name__
