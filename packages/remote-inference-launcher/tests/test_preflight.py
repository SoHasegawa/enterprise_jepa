from __future__ import annotations

import subprocess

from remote_inference_launcher.fleet import FleetConfig
from remote_inference_launcher.preflight import RemoteCommandPolicy, run_preflight_checks
from remote_inference_launcher.slurm_vllm import SlurmVllmConfig
from remote_inference_launcher.ssh_vllm import SshVllmConfig


def _remote_batch_stdout(
    check_count: int,
    failures: dict[int, tuple[int, str, str]] | None = None,
) -> str:
    failures = failures or {}
    chunks: list[str] = []
    for index in range(check_count):
        returncode, stderr, stdout = failures.get(index, (0, "", ""))
        chunks.extend(
            [
                "__RIL_PREFLIGHT_CHECK_BEGIN__",
                f"index={index}",
                f"returncode={returncode}",
                "__RIL_PREFLIGHT_STDERR__",
                stderr,
                "__RIL_PREFLIGHT_STDOUT__",
                stdout,
                "__RIL_PREFLIGHT_CHECK_END__",
            ]
        )
    return "\n".join(chunks)


def test_slurm_preflight_runs_scheduler_advisory_commands() -> None:
    commands: list[str] = []

    def runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        commands.append(" ".join(command))
        return subprocess.CompletedProcess(command, 0, stdout=_remote_batch_stdout(12), stderr="")

    checks = run_preflight_checks(
        SlurmVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            partition="gpu",
            walltime="00:30:00",
            num_gpus=1,
            memory="32GB",
            cpus_per_task=4,
        ),
        source="slurm.yaml",
        command_runner=runner,
    )

    names = {check.name for check in checks}
    command_text = "\n".join(commands)
    assert "slurm-sbatch-test-only" in names
    assert "slurm-squeue-start" in names
    assert "slurm-partition-state" in names
    assert "slurm-gres-gpu" in names
    assert "slurm-partition-walltime" in names
    assert "sbatch --test-only" in command_text
    assert "squeue --start" in command_text
    assert len(commands) == 1


def test_slurm_preflight_skips_login_node_vllm_import() -> None:
    commands: list[str] = []

    def runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        command_text = " ".join(command)
        commands.append(command_text)
        assert "import vllm" not in command_text
        return subprocess.CompletedProcess(command, 0, stdout=_remote_batch_stdout(12), stderr="")

    checks = run_preflight_checks(
        SlurmVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            partition="gpu",
            walltime="00:30:00",
            num_gpus=1,
            memory="32GB",
            cpus_per_task=4,
        ),
        source="slurm.yaml",
        command_runner=runner,
    )

    vllm_check = next(check for check in checks if check.name == "vllm-import")
    assert vllm_check.outcome == "skipped"
    assert vllm_check.code == "requires_compute_allocation"
    assert commands


def test_slurm_partition_walltime_mismatch_is_durable() -> None:
    def runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_remote_batch_stdout(
                12,
                {
                    10: (
                        1,
                        "requested walltime 24:00:00 exceeds partition max 06:00:00",
                        "",
                    )
                },
            ),
            stderr="",
        )

    checks = run_preflight_checks(
        SlurmVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            partition="gpu-short",
            walltime="24:00:00",
            num_gpus=1,
            memory="32GB",
            cpus_per_task=4,
        ),
        source="slurm.yaml",
        command_runner=runner,
        remote_command_policy=RemoteCommandPolicy(
            attempts=3,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            jitter_seconds=0,
        ),
    )

    walltime_check = next(check for check in checks if check.name == "slurm-partition-walltime")
    assert walltime_check.outcome == "durable_failure"
    assert walltime_check.code == "slurm_partition_walltime_exceeded"
    assert walltime_check.attempts == 1


def test_slurm_partition_without_schedulable_states_is_durable() -> None:
    def runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_remote_batch_stdout(
                12,
                {
                    8: (
                        1,
                        "partition gpu has no schedulable node states: down drain",
                        "",
                    )
                },
            ),
            stderr="",
        )

    checks = run_preflight_checks(
        SlurmVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            partition="gpu",
            walltime="00:30:00",
            num_gpus=1,
            memory="32GB",
            cpus_per_task=4,
        ),
        source="slurm.yaml",
        command_runner=runner,
        remote_command_policy=RemoteCommandPolicy(
            attempts=3,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            jitter_seconds=0,
        ),
    )

    state_check = next(check for check in checks if check.name == "slurm-partition-state")
    assert state_check.outcome == "durable_failure"
    assert state_check.code == "slurm_partition_unavailable"
    assert "no schedulable node states" in state_check.detail
    assert state_check.attempts == 1


def test_slurm_gres_gpu_mismatch_is_durable() -> None:
    def runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_remote_batch_stdout(
                12,
                {
                    9: (
                        1,
                        "partition gpu exposes at most 1 GPUs via GRES but 2 were requested",
                        "",
                    )
                },
            ),
            stderr="",
        )

    checks = run_preflight_checks(
        SlurmVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            partition="gpu",
            walltime="00:30:00",
            num_gpus=2,
            memory="32GB",
            cpus_per_task=4,
        ),
        source="slurm.yaml",
        command_runner=runner,
        remote_command_policy=RemoteCommandPolicy(
            attempts=3,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            jitter_seconds=0,
        ),
    )

    gres_check = next(check for check in checks if check.name == "slurm-gres-gpu")
    assert gres_check.outcome == "durable_failure"
    assert gres_check.code == "slurm_gres_insufficient"
    assert "2 were requested" in gres_check.detail
    assert gres_check.attempts == 1


def test_slurm_preflight_includes_base_resource_candidate_when_requested() -> None:
    commands: list[str] = []

    def runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        commands.append(" ".join(command))
        return subprocess.CompletedProcess(command, 0, stdout=_remote_batch_stdout(12), stderr="")

    checks = run_preflight_checks(
        SlurmVllmConfig(
            name="logical",
            ssh_target="base-cluster",
            model="vendor/model",
            partition="gpu",
            walltime="00:30:00",
            num_gpus=1,
            memory="32GB",
            cpus_per_task=4,
            include_base_resource_candidate=True,
            resource_preferences=(
                {
                    "name": "preferred",
                    "ssh_target": "preferred-cluster",
                    "partition": "gpu-short",
                },
            ),
        ),
        source="slurm.yaml",
        command_runner=runner,
    )

    sources = {check.source for check in checks}
    command_text = "\n".join(commands)
    assert "slurm.yaml resource 'preferred'" in sources
    assert "slurm.yaml resource 'logical-base'" in sources
    assert "preferred-cluster" in command_text
    assert "base-cluster" in command_text
    assert len(commands) == 2


def test_fleet_preflight_batches_shared_ssh_target() -> None:
    commands: list[str] = []

    def runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        commands.append(" ".join(command))
        return subprocess.CompletedProcess(command, 0, stdout=_remote_batch_stdout(6), stderr="")

    checks = run_preflight_checks(
        FleetConfig(
            endpoints={
                "one": SshVllmConfig(ssh_target="cluster", model="vendor/model-a"),
                "two": SshVllmConfig(ssh_target="cluster", model="vendor/model-b"),
            },
        ),
        source="fleet.yaml",
        command_runner=runner,
    )

    assert len(commands) == 1
    assert {check.source for check in checks} == {
        "fleet.yaml endpoint 'one'",
        "fleet.yaml endpoint 'two'",
    }
    assert [check.name for check in checks].count("ssh-reachable") == 2
    assert [check.name for check in checks].count("vllm-import") == 2


def test_ssh_preflight_retries_transient_transport_failure() -> None:
    calls: list[str] = []

    def runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        calls.append(" ".join(command))
        return subprocess.CompletedProcess(
            command,
            255,
            stdout="",
            stderr="Connection reset by peer",
        )

    checks = run_preflight_checks(
        SshVllmConfig(
            ssh_target="cluster.example",
            model="vendor/model",
        ),
        source="ssh.yaml",
        command_runner=runner,
        remote_command_policy=RemoteCommandPolicy(
            attempts=3,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            jitter_seconds=0,
            per_attempt_timeout_seconds=5,
        ),
    )

    reachable = next(check for check in checks if check.name == "ssh-reachable")
    assert reachable.outcome == "transient_failure"
    assert reachable.code == "ssh_connection_reset"
    assert reachable.attempts == 3
    assert "cluster.example" not in reachable.command_summary
    assert "<ssh-target>" in reachable.command_summary
    assert len(calls) == 3


def test_remote_python_missing_is_durable_without_retry() -> None:
    calls: list[str] = []

    def runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        calls.append(" ".join(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_remote_batch_stdout(
                3,
                {1: (127, "bash: /missing/python: No such file or directory", "")},
            ),
            stderr="",
        )

    checks = run_preflight_checks(
        SshVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            python_bin="/missing/python",
        ),
        source="ssh.yaml",
        command_runner=runner,
        remote_command_policy=RemoteCommandPolicy(
            attempts=3,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            jitter_seconds=0,
        ),
    )

    python_check = next(check for check in checks if check.name == "python-version")
    assert python_check.outcome == "durable_failure"
    assert python_check.code == "remote_python_missing"
    assert python_check.attempts == 1
    assert len(calls) == 1


def test_missing_slurm_binary_is_durable() -> None:
    def runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_remote_batch_stdout(12, {1: (1, "sbatch not found", "")}),
            stderr="",
        )

    checks = run_preflight_checks(
        SlurmVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            partition="gpu",
            walltime="00:30:00",
            num_gpus=1,
            memory="32GB",
            cpus_per_task=4,
        ),
        source="slurm.yaml",
        command_runner=runner,
        remote_command_policy=RemoteCommandPolicy(
            attempts=3,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            jitter_seconds=0,
        ),
    )

    sbatch_check = next(check for check in checks if check.name == "slurm-sbatch")
    assert sbatch_check.outcome == "durable_failure"
    assert sbatch_check.code == "slurm_command_missing"
    assert sbatch_check.attempts == 1
