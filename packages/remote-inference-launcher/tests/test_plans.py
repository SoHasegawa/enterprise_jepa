from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from remote_inference_launcher.existing_endpoint import ExistingEndpointConfig
from remote_inference_launcher.local_vllm import LocalVllmConfig
from remote_inference_launcher.plans import (
    build_effective_launch_plan,
    build_effective_launch_plan_from_path,
)
from remote_inference_launcher.slurm_vllm import SlurmVllmConfig
from remote_inference_launcher.summaries import new_run_id
from remote_inference_launcher.validation import validate_config_paths, validation_report_json

RUN_ID = "20260603-153012-a3f91c2b"
CREATED_AT = "2026-06-03T15:30:12Z"


def test_new_run_id_uses_timestamp_and_eight_hex_random_suffix() -> None:
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", new_run_id())


def test_effective_plan_resolves_remote_home_paths(tmp_path: Path) -> None:
    config = tmp_path / "slurm.yaml"
    config.write_text(
        """
kind: slurm_vllm
ssh_target: cluster
model: vendor/model
python_bin: ~/.venv/bin/python
remote_out_dir_root: ~/tmp/remote-inference-launcher
partition: gpu
walltime: "01:00:00"
num_gpus: 1
memory: 32G
cpus_per_task: 4
readiness: {smoke_test: disabled}
""",
        encoding="utf-8",
    )

    plan = build_effective_launch_plan_from_path(
        config,
        run_id=RUN_ID,
        created_at=CREATED_AT,
        remote_home_resolver=lambda _target: "/home/remote-user",
    )

    endpoint = plan.endpoints[0]
    assert endpoint.job_name == "ril-default-a3f91c2b"
    assert endpoint.paths["python_bin"].resolved == "/home/remote-user/.venv/bin/python"
    assert endpoint.paths["python_bin"].expansion == "remote_home"
    assert endpoint.out_dir == (
        f"/home/remote-user/tmp/remote-inference-launcher/vendor-model/default-{RUN_ID}"
    )
    assert endpoint.remote_state_path == endpoint.out_dir + "/remote-inference-state.json"
    assert str(endpoint.summary_path) == (
        f".remote-inference-launcher/runs/{RUN_ID}/endpoints/default/summary.json"
    )


def test_effective_plan_marks_unresolved_remote_home_as_deferred(tmp_path: Path) -> None:
    config = tmp_path / "ssh.yaml"
    config.write_text(
        """
kind: ssh_vllm
ssh_target: cluster
model: vendor/model
python_bin: ~/.venv/bin/python
remote_out_dir_root: ~/tmp/remote-inference-launcher
readiness: {smoke_test: disabled}
""",
        encoding="utf-8",
    )

    plan = build_effective_launch_plan_from_path(
        config,
        run_id=RUN_ID,
        created_at=CREATED_AT,
    )

    endpoint = plan.endpoints[0]
    assert endpoint.paths["python_bin"].resolved == "~/.venv/bin/python"
    assert endpoint.paths["python_bin"].expansion == "deferred_remote_home"
    assert endpoint.out_dir.startswith("~/tmp/remote-inference-launcher/vendor-model/")


def test_effective_plan_rejects_unsupported_remote_home_form(tmp_path: Path) -> None:
    config = tmp_path / "slurm.yaml"
    config.write_text(
        """
kind: slurm_vllm
ssh_target: cluster
model: vendor/model
remote_out_dir_root: ~other/tmp
partition: gpu
walltime: "01:00:00"
num_gpus: 1
memory: 32G
cpus_per_task: 4
readiness: {smoke_test: disabled}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported remote path"):
        build_effective_launch_plan_from_path(config, run_id=RUN_ID, created_at=CREATED_AT)


def test_effective_plan_rejects_relative_remote_path(tmp_path: Path) -> None:
    config = tmp_path / "slurm.yaml"
    config.write_text(
        """
kind: slurm_vllm
ssh_target: cluster
model: vendor/model
remote_out_dir_root: tmp/remote-inference-launcher
partition: gpu
walltime: "01:00:00"
num_gpus: 1
memory: 32G
cpus_per_task: 4
readiness: {smoke_test: disabled}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be absolute or start with '~'"):
        build_effective_launch_plan_from_path(config, run_id=RUN_ID, created_at=CREATED_AT)


def test_effective_plan_generates_unique_values_for_colliding_endpoint_labels(
    tmp_path: Path,
) -> None:
    first_name = "endpoint-" + ("a" * 60) + "-one"
    second_name = "endpoint-" + ("a" * 60) + "-two"
    config = tmp_path / "fleet.yaml"
    config.write_text(
        f"""
kind: fleet
endpoints:
  {first_name}:
    kind: slurm_vllm
    ssh_target: cluster
    model: vendor/model
    partition: gpu
    walltime: "01:00:00"
    num_gpus: 1
    memory: 32G
    cpus_per_task: 4
    readiness: {{smoke_test: disabled}}
  {second_name}:
    kind: slurm_vllm
    ssh_target: cluster
    model: vendor/model
    partition: gpu
    walltime: "01:00:00"
    num_gpus: 1
    memory: 32G
    cpus_per_task: 4
    readiness: {{smoke_test: disabled}}
""",
        encoding="utf-8",
    )

    plan = build_effective_launch_plan_from_path(
        config,
        run_id=RUN_ID,
        created_at=CREATED_AT,
        remote_home_resolver=lambda _target: "/home/remote-user",
    )

    labels = [endpoint.endpoint_label for endpoint in plan.endpoints]
    job_names = [endpoint.job_name for endpoint in plan.endpoints]
    out_dirs = [endpoint.out_dir for endpoint in plan.endpoints]
    state_paths = [endpoint.remote_state_path for endpoint in plan.endpoints]
    summary_paths = [endpoint.summary_path for endpoint in plan.endpoints]
    assert len(set(labels)) == 2
    assert all(len(label) <= 48 for label in labels)
    assert len(set(job_names)) == 2
    assert len(set(out_dirs)) == 2
    assert len(set(state_paths)) == 2
    assert len(set(summary_paths)) == 2


def test_semantic_hash_ignores_generated_values_and_secret_values() -> None:
    first = build_effective_launch_plan(
        LocalVllmConfig(model="vendor/model", api_key="first-secret"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at=CREATED_AT,
    )
    second = build_effective_launch_plan(
        LocalVllmConfig(model="vendor/model", api_key="second-secret"),
        source_config_path="renamed.yaml",
        run_id="20260603-153013-b4e02d9c",
        created_at="2026-06-03T15:30:13Z",
    )

    assert first.semantic_config_hash == second.semantic_config_hash
    assert first.instance_hash != second.instance_hash


def test_existing_endpoint_semantic_hash_includes_normalized_api_base() -> None:
    first = build_effective_launch_plan(
        ExistingEndpointConfig(
            api_base="http://127.0.0.1:8000/v1",
            served_model_name="vendor/model",
        ),
        source_config_path="first.yaml",
        run_id=RUN_ID,
        created_at=CREATED_AT,
        ownership_policy="lease",
    )
    same_with_trailing_slash = build_effective_launch_plan(
        ExistingEndpointConfig(
            api_base="http://127.0.0.1:8000/v1/",
            served_model_name="vendor/model",
        ),
        source_config_path="same.yaml",
        run_id="20260603-153013-b4e02d9c",
        created_at="2026-06-03T15:30:13Z",
        ownership_policy="lease",
    )
    different_url = build_effective_launch_plan(
        ExistingEndpointConfig(
            api_base="http://127.0.0.1:9000/v1",
            served_model_name="vendor/model",
        ),
        source_config_path="different.yaml",
        run_id="20260603-153014-c5d13e8f",
        created_at="2026-06-03T15:30:14Z",
        ownership_policy="lease",
    )

    assert first.semantic_config_hash == same_with_trailing_slash.semantic_config_hash
    assert first.semantic_config_hash != different_url.semantic_config_hash


def test_slurm_semantic_hash_ignores_generated_launch_identity() -> None:
    config = SlurmVllmConfig(
        ssh_target="cluster",
        model="vendor/model",
        partition="gpu",
        walltime="01:00:00",
        num_gpus=1,
        memory="32G",
        cpus_per_task=4,
    )

    first = build_effective_launch_plan(
        config,
        source_config_path="inference.yaml",
        run_id=RUN_ID,
        created_at=CREATED_AT,
    )
    second = build_effective_launch_plan(
        config,
        source_config_path="renamed.yaml",
        run_id="20260603-153013-b4e02d9c",
        created_at="2026-06-03T15:30:13Z",
    )

    assert first.endpoints[0].job_name != second.endpoints[0].job_name
    assert first.endpoints[0].out_dir != second.endpoints[0].out_dir
    assert first.semantic_config_hash == second.semantic_config_hash
    assert first.instance_hash != second.instance_hash


def test_validation_json_includes_effective_plan_generated_values(tmp_path: Path) -> None:
    config = tmp_path / "slurm.yaml"
    config.write_text(
        """
kind: slurm_vllm
ssh_target: cluster
model: vendor/model
partition: gpu
walltime: "01:00:00"
num_gpus: 1
memory: 32G
cpus_per_task: 4
readiness: {smoke_test: disabled}
""",
        encoding="utf-8",
    )

    report = validate_config_paths([config])
    payload = json.loads(validation_report_json(report))
    effective_plan = payload["effective_launch_plans"][0]
    endpoint = effective_plan["endpoints"][0]

    assert effective_plan["schema_version"] == "ril-plan/v1"
    assert effective_plan["semantic_config_hash"].startswith("sha256:")
    assert effective_plan["instance_hash"].startswith("sha256:")
    assert endpoint["job_name"].startswith("ril-default-")
    assert endpoint["summary_path"].endswith("/endpoints/default/summary.json")
    assert endpoint["out_dir"].startswith("~/tmp/remote-inference-launcher/slurm-vllm/")
