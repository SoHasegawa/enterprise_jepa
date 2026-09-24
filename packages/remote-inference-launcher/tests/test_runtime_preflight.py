from __future__ import annotations

import json
import subprocess

from remote_inference_launcher.remote_execution import RemoteCommandPolicy
from remote_inference_launcher.runtime_preflight import (
    RuntimePreflightPolicy,
    render_slurm_runtime_preflight_sbatch,
    run_runtime_preflight_checks,
)
from remote_inference_launcher.slurm_vllm import SlurmVllmConfig, with_slurm_vllm_defaults


def _runtime_policy() -> RuntimePreflightPolicy:
    return RuntimePreflightPolicy(
        timeout_seconds=1,
        poll_interval_seconds=0,
        ssh_policy=RemoteCommandPolicy(
            attempts=1,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            jitter_seconds=0,
            per_attempt_timeout_seconds=1,
        ),
    )


def _manifest(*, vllm_returncode: int = 0) -> str:
    return json.dumps(
        {
            "schema_version": "ril-slurm-runtime-preflight/v1",
            "status": "ok" if vllm_returncode == 0 else "failed",
            "job_id": "12345",
            "node": "node-a",
            "checks": [
                {
                    "name": "runtime-python-version",
                    "returncode": 0,
                    "code": "ok",
                    "stdout_excerpt": "Python 3.12.0",
                    "stderr_excerpt": "",
                },
                {
                    "name": "runtime-vllm-import",
                    "returncode": vllm_returncode,
                    "code": "ok" if vllm_returncode == 0 else "runtime_vllm_import_failed",
                    "stdout_excerpt": "",
                    "stderr_excerpt": "" if vllm_returncode == 0 else "No module named vllm",
                },
                {
                    "name": "runtime-torch-device",
                    "returncode": 0,
                    "code": "ok",
                    "stdout_excerpt": "torch_cuda_available=True",
                    "stderr_excerpt": "",
                },
            ],
        }
    )


def test_render_slurm_runtime_preflight_sbatch_checks_runtime_only() -> None:
    script = render_slurm_runtime_preflight_sbatch(
        with_slurm_vllm_defaults(
            SlurmVllmConfig(
                ssh_target="cluster",
                model="/models/qwen",
                partition="gpu",
                walltime="00:10:00",
                num_gpus=1,
                memory="32GB",
                cpus_per_task=4,
                setup_cmd="module load vllm",
                target_device="rocm",
            ),
            run_id="run-abc",
        ),
        "/tmp/runtime_preflight_manifest.json",
    )

    assert "#SBATCH --partition=gpu" in script
    assert "runtime-python-version" in script
    assert "runtime-vllm-import" in script
    assert "runtime-torch-device" in script
    assert "runtime-model-path" in script
    assert "torch.cuda.is_available" in script
    assert "vllm serve" not in script


def test_slurm_runtime_preflight_reads_success_manifest() -> None:
    commands: list[str] = []

    def runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        command_text = " ".join(command)
        commands.append(command_text)
        if "cd && pwd" in command_text:
            return subprocess.CompletedProcess(command, 0, stdout="/home/user\n", stderr="")
        if "cat >" in command_text and "runtime_preflight.sbatch" in command_text:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if "sbatch --parsable" in command_text:
            return subprocess.CompletedProcess(command, 0, stdout="12345\n", stderr="")
        if "runtime_preflight_manifest.json" in command_text and "if [ -s" in command_text:
            return subprocess.CompletedProcess(command, 0, stdout=_manifest(), stderr="")
        raise AssertionError(f"unexpected command: {command_text}")

    checks = run_runtime_preflight_checks(
        SlurmVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            partition="gpu",
            walltime="00:10:00",
            num_gpus=1,
            memory="32GB",
            cpus_per_task=4,
            target_device="rocm",
        ),
        source="slurm.yaml",
        command_runner=runner,
        policy=_runtime_policy(),
    )

    by_name = {check.name: check for check in checks}
    assert by_name["runtime-preflight-job"].outcome == "ok"
    assert by_name["runtime-vllm-import"].outcome == "ok"
    assert by_name["runtime-torch-device"].outcome == "ok"
    assert all(check.layer == "compute-runtime" for check in checks)
    assert any("sbatch --parsable" in command for command in commands)


def test_slurm_runtime_preflight_reports_vllm_import_failure() -> None:
    def runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        command_text = " ".join(command)
        if "cd && pwd" in command_text:
            return subprocess.CompletedProcess(command, 0, stdout="/home/user\n", stderr="")
        if "cat >" in command_text and "runtime_preflight.sbatch" in command_text:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if "sbatch --parsable" in command_text:
            return subprocess.CompletedProcess(command, 0, stdout="12345\n", stderr="")
        if "runtime_preflight_manifest.json" in command_text and "if [ -s" in command_text:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_manifest(vllm_returncode=1),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command_text}")

    checks = run_runtime_preflight_checks(
        SlurmVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            partition="gpu",
            walltime="00:10:00",
            num_gpus=1,
            memory="32GB",
            cpus_per_task=4,
        ),
        source="slurm.yaml",
        command_runner=runner,
        policy=_runtime_policy(),
    )

    vllm = next(check for check in checks if check.name == "runtime-vllm-import")
    assert vllm.outcome == "durable_failure"
    assert vllm.code == "runtime_vllm_import_failed"
    assert "No module named vllm" in vllm.detail
