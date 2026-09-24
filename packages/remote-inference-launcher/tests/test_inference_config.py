from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from remote_inference_launcher.endpoint_race import EndpointRaceConfig
from remote_inference_launcher.fleet import FleetConfig
from remote_inference_launcher.inference_config import (
    load_inference_config,
    load_inference_launcher,
)
from remote_inference_launcher.local_vllm import LocalVllmConfig, vllm_server_command
from remote_inference_launcher.slurm_vllm import SlurmVllmConfig, render_slurm_vllm_sbatch
from remote_inference_launcher.ssh_vllm import SshVllmConfig, render_ssh_vllm_script


def test_loads_fleet_config_with_endpoint_names_and_memory_zero(tmp_path: Path) -> None:
    config_path = tmp_path / "fleet.yaml"
    config_path.write_text(
        """
kind: fleet
endpoints:
  actor:
    kind: slurm_vllm
    ssh_target: slurm-login
    model: big/model
    python_bin: ~/.venv-vllm/bin/python
    partition: batch-8gpu
    walltime: "12:00:00"
    nodes: 2
    num_gpus: 8
    memory: 0
    cpus_per_task: 32
    distributed_backend: ray
    tensor_parallel_size: 16
  critic:
    kind: local_vllm
    model: critic/model
    python_bin: .venv-vllm/bin/python
    port: 8124
""",
        encoding="utf-8",
    )

    config = load_inference_config(config_path)

    assert isinstance(config, FleetConfig)
    actor = config.endpoints["actor"]
    critic = config.endpoints["critic"]
    assert isinstance(actor, SlurmVllmConfig)
    assert actor.name == "actor"
    assert actor.memory == "0"
    assert actor.distributed_backend == "ray"
    assert isinstance(critic, LocalVllmConfig)
    assert critic.name == "critic"


def test_loads_nested_composite_endpoint_with_parent_default_name(tmp_path: Path) -> None:
    config_path = tmp_path / "fleet.yaml"
    config_path.write_text(
        """
kind: fleet
endpoints:
  actor:
    kind: endpoint_race
    candidates:
      - kind: existing_endpoint
        api_base: http://127.0.0.1:8123/v1
        readiness:
          smoke_test: disabled
""",
        encoding="utf-8",
    )

    config = load_inference_config(config_path)

    assert isinstance(config, FleetConfig)
    actor = config.endpoints["actor"]
    assert isinstance(actor, EndpointRaceConfig)
    assert actor.name == "actor"


def test_load_inference_launcher_passes_planned_run_id(monkeypatch, tmp_path: Path) -> None:
    import remote_inference_launcher.inference_config as inference_config

    captured: dict[str, object] = {}

    class CapturingLocalLauncher:
        def __init__(self, config: LocalVllmConfig, *, run_id: str | None = None) -> None:
            captured["config"] = config
            captured["run_id"] = run_id

    monkeypatch.setattr(inference_config, "LocalVllmLauncher", CapturingLocalLauncher)
    config_path = tmp_path / "local.yaml"
    config_path.write_text("kind: local_vllm\nmodel: vendor/model\n", encoding="utf-8")

    launcher = load_inference_launcher(config_path, run_id="20260603-153012-a3f91c2b")

    assert isinstance(launcher, CapturingLocalLauncher)
    assert captured["run_id"] == "20260603-153012-a3f91c2b"
    assert isinstance(captured["config"], LocalVllmConfig)


def test_load_inference_launcher_applies_single_endpoint_plan_values(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import remote_inference_launcher.inference_config as inference_config

    captured: dict[str, object] = {}

    class CapturingSlurmLauncher:
        def __init__(self, config: SlurmVllmConfig, *, run_id: str | None = None) -> None:
            captured["config"] = config
            captured["run_id"] = run_id

    monkeypatch.setattr(inference_config, "SlurmVllmLauncher", CapturingSlurmLauncher)
    config_path = tmp_path / "slurm.yaml"
    config_path.write_text(
        """
kind: slurm_vllm
ssh_target: cluster
model: vendor/model
partition: gpu
walltime: "01:00:00"
num_gpus: 1
memory: 32G
cpus_per_task: 4
""",
        encoding="utf-8",
    )

    load_inference_launcher(
        config_path,
        run_id="20260603-153012-a3f91c2b",
        endpoint_plan=SimpleNamespace(
            summary_path=tmp_path / "summary.json",
            job_name="ril-default-a3f91c2b",
            out_dir="~/tmp/remote-inference-launcher/vendor-model/default-run",
            local_port=18123,
            remote_port=28123,
        ),
    )

    config = captured["config"]
    assert isinstance(config, SlurmVllmConfig)
    assert captured["run_id"] == "20260603-153012-a3f91c2b"
    assert config.launch_summary_path == str(tmp_path / "summary.json")
    assert config.job_name == "ril-default-a3f91c2b"
    assert config.out_dir == "~/tmp/remote-inference-launcher/vendor-model/default-run"
    assert config.local_port == 18123
    assert config.remote_port == 28123


def test_local_vllm_command_includes_api_key_and_model_options() -> None:
    config = LocalVllmConfig(
        model="vendor/model",
        served_model_name="served",
        python_bin="/venv/bin/python",
        port=8123,
        api_key="secret",
        tensor_parallel_size=2,
        max_model_len=8192,
        extra_args=("--disable-log-requests",),
    )

    command = vllm_server_command(config)

    assert command[:5] == [
        "/venv/bin/python",
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        "vendor/model",
    ]
    assert command[command.index("--api-key") : command.index("--api-key") + 2] == [
        "--api-key",
        "secret",
    ]
    assert command[
        command.index("--tensor-parallel-size") : command.index("--tensor-parallel-size") + 2
    ] == ["--tensor-parallel-size", "2"]
    assert command[-1] == "--disable-log-requests"


def test_ssh_script_records_process_and_serves_openai_endpoint() -> None:
    script = render_ssh_vllm_script(
        SshVllmConfig(
            ssh_target="gpu-host",
            model="vendor/model",
            served_model_name="served",
            remote_port=18817,
            api_key="secret",
            tensor_parallel_size=2,
        ),
        out_dir="/remote/out",
    )

    assert 'echo "$$" > "${PID_PATH}"' in script
    assert "REMOTE_INFERENCE_STATE_PATH=/remote/out/remote-inference-state.json" in script
    assert "write_state_file" in script
    assert "--served-model-name" in script
    assert "--api-key secret" in script
    assert "--tensor-parallel-size 2" in script


def test_slurm_multi_node_ray_script_is_explicit() -> None:
    script = render_slurm_vllm_sbatch(
        SlurmVllmConfig(
            ssh_target="slurm-login",
            model="big/model",
            python_bin="~/.venv-vllm/bin/python",
            partition="batch-8gpu",
            walltime="12:00:00",
            nodes=2,
            num_gpus=8,
            memory="0",
            cpus_per_task=32,
            distributed_backend="ray",
            tensor_parallel_size=16,
            api_key="secret",
        ),
        out_dir="/remote/out",
    )

    assert "#SBATCH --nodes=2" in script
    assert "scontrol show hostnames" in script
    assert "-m ray.scripts.scripts start" in script
    assert "--node-ip-address" in script
    assert "--distributed-executor-backend ray" in script
    assert "--tensor-parallel-size 16" in script
    assert "--api-key secret" in script
