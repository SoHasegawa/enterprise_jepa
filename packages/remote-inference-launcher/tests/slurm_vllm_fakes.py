from __future__ import annotations

import json
from pathlib import Path

from remote_inference_launcher.slurm_vllm import SlurmVllmConfig, SlurmVllmLauncher

TEST_MODEL = "vendor/test-model"


def base_config(**overrides: object) -> SlurmVllmConfig:
    values = {
        "ssh_target": "cluster",
        "model": TEST_MODEL,
        "served_model_name": TEST_MODEL,
        "partition": "batch-2gpu",
        "walltime": "6:00:00",
        "num_gpus": 2,
        "memory": "384GB",
        "cpus_per_task": 20,
    }
    values.update(overrides)
    return SlurmVllmConfig(**values)


class _FailingLauncher(SlurmVllmLauncher):
    def __init__(self, config: SlurmVllmConfig) -> None:
        super().__init__(config)
        self.cancelled_jobs: list[str] = []

    def _ssh_text(self, command: str) -> str:
        if command == "cd && pwd":
            return "/remote/home\n"
        if command.startswith("mkdir -p"):
            return ""
        if command == "scancel 12345":
            self.cancelled_jobs.append(command.split()[1])
            return ""
        raise AssertionError(f"unexpected SSH command: {command}")

    def _copy_to_remote(self, local_path, remote_path: str) -> None:
        assert local_path.is_file()
        assert remote_path.endswith("/launch_vllm.sbatch")

    def _submit_remote_job(self, remote_script: str) -> str:
        assert remote_script.endswith("/launch_vllm.sbatch")
        return "12345"

    def _wait_for_running_node(self) -> str:
        raise TimeoutError("not ready")


class _RecordingSummary:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.payloads: list[dict[str, object]] = []
        self._reserved = False

    def reserve(self) -> Path:
        if not self._reserved:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text('{"lifecycle_state": "RESERVED"}\n', encoding="utf-8")
            self._reserved = True
        return self.path

    def write(self, payload: dict[str, object]) -> None:
        self.reserve()
        self.payloads.append(dict(payload))
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class _ReadyLauncher(SlurmVllmLauncher):
    def _ssh_text(self, command: str) -> str:
        if command == "cd && pwd":
            return "/remote/home\n"
        if command.startswith("mkdir -p"):
            return ""
        if command == "scancel 12345":
            return ""
        raise AssertionError(f"unexpected SSH command: {command}")

    def _copy_to_remote(self, local_path, remote_path: str) -> None:
        assert local_path.is_file()
        assert remote_path.endswith("/launch_vllm.sbatch")

    def _submit_remote_job(self, remote_script: str) -> str:
        assert remote_script.endswith("/launch_vllm.sbatch")
        return "12345"

    def _wait_for_running_node(self) -> str:
        return "node001"

    def _wait_for_remote_state_port(self) -> int:
        return 18817

    def _start_tunnel(self, local_port: int, remote_port: int, **_kwargs: object) -> None:
        assert local_port == 8123
        assert remote_port == 18817

    def _wait_for_ready(self, api_base: str) -> None:
        self._log_progress(f"Remote vLLM is ready at {api_base}.")


class _OutDirCaptureLauncher(_ReadyLauncher):
    def __init__(self, config: SlurmVllmConfig) -> None:
        super().__init__(config)
        self.remote_script = ""

    def _copy_to_remote(self, local_path, remote_path: str) -> None:
        assert local_path.is_file()
        self.remote_script = remote_path


class _RetryTunnelLauncher(_ReadyLauncher):
    def __init__(self, config: SlurmVllmConfig) -> None:
        super().__init__(config)
        self.tunnel_ports: list[int] = []
        self.ready_api_base = ""

    def _start_tunnel(self, local_port: int, remote_port: int, **_kwargs: object) -> None:
        assert remote_port == 18817
        self.tunnel_ports.append(local_port)
        if len(self.tunnel_ports) == 1:
            raise RuntimeError("port was claimed")

    def _wait_for_ready(self, api_base: str) -> None:
        self.ready_api_base = api_base


class _FinalTunnelFailureLauncher(_ReadyLauncher):
    def __init__(self, config: SlurmVllmConfig) -> None:
        super().__init__(config)
        self.squeue_queries = 0

    def _start_tunnel(self, local_port: int, remote_port: int, **_kwargs: object) -> None:
        del local_port, remote_port
        raise RuntimeError("tunnel bind failed")

    def _ssh_text(self, command: str) -> str:
        if command == "squeue -j 12345 -h -o '%T|%N|%R'":
            self.squeue_queries += 1
            return "RUNNING|node001|None\n"
        return super()._ssh_text(command)


class _CapacityLogLauncher(SlurmVllmLauncher):
    def _safe_remote_log_tail(self, *, lines: int = 80) -> str:
        del lines
        return "\n".join(
            [
                "GPU KV cache size: 100,000 tokens",
                "Maximum concurrency for 8192 tokens per request: 3.50x",
            ]
        )


class _SqueueFailureLauncher(SlurmVllmLauncher):
    def _ssh_text(self, command: str) -> str:
        assert command == "squeue -j 12345 -h -o '%T|%N|%R'"
        raise RuntimeError("squeue unavailable")


class _ScontrolFailureLauncher(SlurmVllmLauncher):
    def _ssh_text(self, command: str) -> str:
        if command == "squeue -j 12345 -h -o '%T|%N|%R'":
            return "RUNNING|node[001-002]|None\n"
        assert command == "scontrol show hostnames 'node[001-002]'"
        raise RuntimeError("scontrol unavailable")


class _ScancelFailureLauncher(SlurmVllmLauncher):
    def _ssh_text(self, command: str) -> str:
        if command == "scancel 12345":
            raise RuntimeError("scancel failed")
        if command == "squeue -j 12345 -h -o '%T|%N|%R'":
            return "RUNNING|node001|None\n"
        raise AssertionError(f"unexpected SSH command: {command}")


class _ScancelAlreadyGoneLauncher(SlurmVllmLauncher):
    def _ssh_text(self, command: str) -> str:
        if command == "scancel 12345":
            raise RuntimeError("invalid job id")
        if command == "squeue -j 12345 -h -o '%T|%N|%R'":
            return ""
        raise AssertionError(f"unexpected SSH command: {command}")


class _ScancelTransientThenSuccessLauncher(SlurmVllmLauncher):
    def __init__(self, config: SlurmVllmConfig) -> None:
        super().__init__(config)
        self.scancel_attempts = 0

    def _ssh_text(self, command: str) -> str:
        if command == "scancel 12345":
            self.scancel_attempts += 1
            if self.scancel_attempts == 1:
                raise RuntimeError("scancel failed")
            return ""
        if command == "squeue -j 12345 -h -o '%T|%N|%R'":
            return "RUNNING|node001|None\n"
        raise AssertionError(f"unexpected SSH command: {command}")


class _StartFailureAndScancelFailureLauncher(_FailingLauncher):
    def _ssh_text(self, command: str) -> str:
        if command == "scancel 12345":
            raise RuntimeError("scancel failed")
        if command == "squeue -j 12345 -h -o '%T|%N|%R'":
            return "RUNNING|node001|None\n"
        return super()._ssh_text(command)
