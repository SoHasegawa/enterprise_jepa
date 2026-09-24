from __future__ import annotations

import subprocess

from remote_inference_launcher.remote_execution import (
    RemoteCommandPolicy,
    run_ssh_command,
    transient_transport_code,
)


def test_retry_safe_ssh_command_retries_transient_transport_failure() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            255,
            stdout="",
            stderr="Connection reset by peer",
        )

    result = run_ssh_command(
        ssh_target="cluster.example",
        name="safe-status",
        command="squeue --me",
        command_runner=runner,
        policy=RemoteCommandPolicy(
            attempts=3,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            jitter_seconds=0,
        ),
        retry_safe=True,
    )

    assert result.returncode == 255
    assert result.attempts == 3
    assert len(calls) == 3
    assert transient_transport_code(result.to_completed_process()) == "ssh_connection_reset"
    assert "cluster.example" not in result.command_summary
    assert "<ssh-target>" in result.command_summary


def test_non_retryable_ssh_command_does_not_replay_after_ambiguous_failure() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            255,
            stdout="",
            stderr="Connection closed by remote host",
        )

    result = run_ssh_command(
        ssh_target="cluster.example",
        name="sbatch-submit",
        command="sbatch launch.sbatch",
        command_runner=runner,
        policy=RemoteCommandPolicy(
            attempts=3,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            jitter_seconds=0,
        ),
        retry_safe=False,
    )

    assert result.returncode == 255
    assert result.attempts == 1
    assert len(calls) == 1
    assert transient_transport_code(result.to_completed_process()) == "ssh_connection_closed"
