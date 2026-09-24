"""Shared Slurm command rendering and parsing helpers."""

from __future__ import annotations

import shlex
import uuid

SQUEUE_JOB_FORMAT = "%T|%N|%R"
SQUEUE_JOB_ID_COMMENT_FORMAT = "%i|%k"


def squeue_job_info_command(job_id: str) -> str:
    """Return a remote command that queries one Slurm job without hiding failures."""

    return " ".join(
        [
            "squeue",
            "-j",
            shlex.quote(job_id),
            "-h",
            "-o",
            shlex.quote(SQUEUE_JOB_FORMAT),
        ]
    )


def new_slurm_submission_comment() -> str:
    """Return a unique Slurm comment used to recover ambiguous submissions."""

    return f"ril-{uuid.uuid4().hex[:16]}"


def squeue_job_ids_by_comment_command(comment: str) -> str:
    """Return a remote command that finds active job ids by exact Slurm comment."""

    return (
        f'squeue -u "$USER" -h -o {shlex.quote(SQUEUE_JOB_ID_COMMENT_FORMAT)} | '
        "while IFS='|' read -r job_id job_comment; do "
        f'if [ "$job_comment" = {shlex.quote(comment)} ]; then '
        "printf '%s\\n' \"$job_id\"; "
        "fi; "
        "done"
    )


def parse_squeue_job_info(output: str) -> tuple[str, str, str] | None:
    """Parse the first non-empty `squeue` job info line, or return None."""

    raw = _first_non_empty_line(output)
    if raw is None:
        return None
    parts = raw.split("|", maxsplit=2)
    state = parts[0] if parts else "UNKNOWN"
    node = parts[1] if len(parts) > 1 else ""
    reason = parts[2] if len(parts) > 2 else ""
    return state, node, reason


def scontrol_hostnames_command(node_expression: str) -> str:
    """Return a remote command that expands a Slurm node expression."""

    return " ".join(["scontrol", "show", "hostnames", shlex.quote(node_expression)])


def scontrol_show_job_command(job_id: str) -> str:
    """Return a remote command that prints Slurm job diagnostics."""

    return " ".join(["scontrol", "show", "job", shlex.quote(job_id)])


def squeue_start_command(job_id: str) -> str:
    """Return a remote command that prints advisory Slurm start-time estimates."""

    return " ".join(["squeue", "--start", "-j", shlex.quote(job_id), "-h"])


def scancel_job_command(job_id: str) -> str:
    """Return a remote command that cancels one Slurm job without hiding failures."""

    return " ".join(["scancel", shlex.quote(job_id)])


def first_scontrol_hostname(output: str, *, node_expression: str) -> str:
    """Return the first hostname from `scontrol show hostnames` output."""

    hostname = _first_non_empty_line(output)
    if hostname is None:
        raise RuntimeError(
            f"Remote scontrol returned no hostnames for Slurm node expression {node_expression!r}."
        )
    return hostname


def _first_non_empty_line(output: str) -> str | None:
    for line in output.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None
