from __future__ import annotations

import pytest

from remote_inference_launcher.slurm import (
    first_scontrol_hostname,
    parse_squeue_job_info,
    scancel_job_command,
    squeue_job_ids_by_comment_command,
    squeue_job_info_command,
)


def test_squeue_job_info_command_does_not_hide_remote_errors() -> None:
    command = squeue_job_info_command("12345")

    assert command == "squeue -j 12345 -h -o '%T|%N|%R'"
    assert "2>/dev/null" not in command
    assert "head -n 1" not in command


def test_squeue_job_ids_by_comment_command_filters_exact_comment() -> None:
    command = squeue_job_ids_by_comment_command("ril-submit-abc123")

    assert "squeue -u \"$USER\" -h -o '%i|%k'" in command
    assert '[ "$job_comment" = ril-submit-abc123 ]' in command
    assert "printf '%s\\n' \"$job_id\"" in command


def test_scancel_job_command_does_not_hide_remote_errors() -> None:
    command = scancel_job_command("12345")

    assert command == "scancel 12345"
    assert "2>/dev/null" not in command
    assert "|| true" not in command


def test_parse_squeue_job_info_returns_first_non_empty_line() -> None:
    assert parse_squeue_job_info("\nRUNNING|node001|None\nPENDING|(null)|Resources\n") == (
        "RUNNING",
        "node001",
        "None",
    )


def test_parse_squeue_job_info_returns_none_for_missing_job() -> None:
    assert parse_squeue_job_info("\n") is None


def test_first_scontrol_hostname_rejects_empty_resolution() -> None:
    with pytest.raises(RuntimeError, match="node\\[001-002\\]"):
        first_scontrol_hostname("", node_expression="node[001-002]")
