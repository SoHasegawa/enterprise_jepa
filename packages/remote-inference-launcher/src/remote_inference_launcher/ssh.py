"""SSH command helpers shared by remote launchers."""

from __future__ import annotations

import subprocess

SSH_TRANSPORT_FAILURE_ATTEMPTS = 8

_TRANSIENT_SSH_ERROR_MARKERS = (
    "connection reset",
    "connection closed",
    "connection timed out",
    "connection refused",
    "kex_exchange_identification",
    "broken pipe",
)


class NonRetryableTransientSshError(RuntimeError):
    """Raised when an unsafe remote command hits an ambiguous SSH transport failure."""


def ssh_options() -> list[str]:
    """Return SSH options used by long-running remote launcher operations."""

    return [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "NumberOfPasswordPrompts=0",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]


def is_transient_ssh_failure(completed: subprocess.CompletedProcess) -> bool:
    """Return whether an SSH process failure is worth retrying."""

    if completed.returncode != 255:
        return False
    detail = f"{completed.stderr}\n{completed.stdout}".lower()
    return any(marker in detail for marker in _TRANSIENT_SSH_ERROR_MARKERS)
