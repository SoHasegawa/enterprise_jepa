from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from remote_inference_launcher.cli import main
from remote_inference_launcher.leases import activate_lease, create_pending_lease
from remote_inference_launcher.local_vllm import LocalVllmConfig
from remote_inference_launcher.plans import build_effective_launch_plan
from remote_inference_launcher.pool import (
    acquire_batch,
    acquire_endpoint,
    build_pool_status,
    load_assignments,
    pool_health_signal,
    release_assignment,
    release_batch,
)
from remote_inference_launcher.registry import create_run_registry, record_ready_sessions
from remote_inference_launcher.scheduler_status import (
    partition_recommendations,
    refresh_slurm_scheduler,
)


def test_pool_status_indexes_active_leases_with_capabilities(tmp_path) -> None:
    lease_root, registry_root, pool_root = _ready_pool_roots(tmp_path)

    payload = build_pool_status(
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
        health_depth="cached",
    )

    assert payload["summary"]["healthy_free"] == 1
    assert payload["summary"]["healthy_busy"] == 0
    assert payload["summary"]["acquirable_healthy_free"] == 1
    endpoint = payload["endpoints"][0]
    assert endpoint["endpoint_id"] == "lease:lease-20260603-160000-b64a19ef:actor"
    assert endpoint["assignment_state"] == "free"
    assert endpoint["acquirable"] is True
    assert "not_acquirable_reason" not in endpoint
    assert endpoint["health"]["status"] == "healthy"
    assert endpoint["capabilities"]["served_model_name"] == "vendor/model"
    assert endpoint["capabilities"]["max_model_len"] == 8192
    assert endpoint["capabilities"]["max_num_seqs"] == 16
    assert endpoint["capabilities"]["num_gpus"] == 1


def test_pool_acquire_health_signal_and_release_assignment(tmp_path) -> None:
    lease_root, registry_root, pool_root = _ready_pool_roots(tmp_path)

    handoff = acquire_endpoint(
        owner="benchmark",
        model="vendor/model",
        min_context=4096,
        assignment_ttl="15m",
        health_depth="cached",
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
    )

    assignment_id = str(handoff["assignment_id"])
    assert handoff["schema_version"] == "ril-pool-handoff/v1"
    assert handoff["endpoint_id"] == "lease:lease-20260603-160000-b64a19ef:actor"
    assert handoff["api_base"] == "http://127.0.0.1:8123/v1"
    assert handoff["assignment_expires_at"]
    assert handoff["lease_expires_at"]
    assert load_assignments(pool_root=pool_root)[assignment_id]["status"] == "active"

    health = pool_health_signal(
        assignment_id=assignment_id,
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
    )
    assert health["assignment_id"] == assignment_id
    assert health["status"] == "healthy"

    with pytest.raises(RuntimeError, match="No free endpoint"):
        acquire_endpoint(
            owner="other-benchmark",
            health_depth="cached",
            lease_root=lease_root,
            registry_root=registry_root,
            pool_root=pool_root,
        )
    busy_status = build_pool_status(
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
    )
    busy_endpoint = busy_status["endpoints"][0]
    assert busy_status["summary"]["healthy_busy"] == 1
    assert busy_status["summary"]["acquirable_healthy_free"] == 0
    assert busy_endpoint["acquirable"] is False
    assert busy_endpoint["not_acquirable_reason"] == "assignment_active"

    released = release_assignment(assignment_id=assignment_id, pool_root=pool_root)
    assert released["status"] == "released"
    status = build_pool_status(
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
    )
    assert status["summary"]["healthy_free"] == 1
    assert status["summary"]["acquirable_healthy_free"] == 1


def test_pool_release_assignment_owner_guard_applies_to_explicit_assignment_id(tmp_path) -> None:
    lease_root, registry_root, pool_root = _ready_pool_roots(tmp_path)
    handoff = acquire_endpoint(
        owner="worker-a",
        model="vendor/model",
        assignment_ttl="15m",
        health_depth="cached",
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
    )
    assignment_id = str(handoff["assignment_id"])

    with pytest.raises(RuntimeError, match="owned by worker-a, not worker-b"):
        release_assignment(
            assignment_id=assignment_id,
            owner="worker-b",
            pool_root=pool_root,
        )

    assert load_assignments(pool_root=pool_root)[assignment_id]["status"] == "active"
    released = release_assignment(
        assignment_id=assignment_id,
        owner="worker-a",
        pool_root=pool_root,
    )
    assert released["status"] == "released"


def test_registry_only_status_is_not_acquirable_and_acquire_explains_why(tmp_path) -> None:
    lease_root, registry_root, pool_root = _ready_registry_roots(tmp_path)

    payload = build_pool_status(
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
        health_depth="cached",
    )

    assert payload["summary"]["healthy_free"] == 1
    assert payload["summary"]["acquirable_healthy_free"] == 0
    endpoint = payload["endpoints"][0]
    assert endpoint["source_kind"] == "registry"
    assert endpoint["acquirable"] is False
    assert endpoint["not_acquirable_reason"] == "registry_endpoint_without_lease"

    with pytest.raises(RuntimeError, match="registry_endpoint_without_lease=1"):
        acquire_endpoint(
            owner="benchmark",
            model="vendor/model",
            health_depth="cached",
            lease_root=lease_root,
            registry_root=registry_root,
            pool_root=pool_root,
        )


def test_acquire_error_reports_model_mismatch_predicate(tmp_path) -> None:
    lease_root, registry_root, pool_root = _ready_pool_roots(tmp_path)

    with pytest.raises(RuntimeError, match="model_mismatch=1"):
        acquire_endpoint(
            owner="benchmark",
            model="different/model",
            health_depth="cached",
            lease_root=lease_root,
            registry_root=registry_root,
            pool_root=pool_root,
        )


def test_pool_acquire_batch_handoff_and_release(tmp_path) -> None:
    lease_root, registry_root, pool_root = _ready_pool_roots_with_leases(tmp_path, count=2)

    handoff = acquire_batch(
        owner="benchmark",
        model="vendor/model",
        count=2,
        shard_ids=["shard-a", "shard-b"],
        assignment_ttl="15m",
        health_depth="cached",
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
    )

    assert handoff["schema_version"] == "ril-pool-batch-handoff/v1"
    assert handoff["requested_count"] == 2
    assert handoff["acquired_count"] == 2
    assert handoff["missing_shards"] == []
    assert {item["shard_id"] for item in handoff["assignments"]} == {"shard-a", "shard-b"}
    assert all(item["batch_id"] == handoff["batch_id"] for item in handoff["assignments"])

    status = build_pool_status(
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
    )
    assert status["summary"]["healthy_busy"] == 2
    assert status["summary"]["acquirable_healthy_free"] == 0

    released = release_batch(batch_id=handoff["batch_id"], pool_root=pool_root)
    assert released["released_count"] == 2
    assert {item["status"] for item in released["released"]} == {"released"}


def test_pool_release_batch_owner_guard_applies_to_explicit_assignment_ids(tmp_path) -> None:
    lease_root, registry_root, pool_root = _ready_pool_roots(tmp_path)
    handoff = acquire_endpoint(
        owner="worker-a",
        model="vendor/model",
        assignment_ttl="15m",
        health_depth="cached",
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
    )
    assignment_id = str(handoff["assignment_id"])

    with pytest.raises(RuntimeError, match="owned by worker-a, not worker-b"):
        release_batch(
            assignment_ids=[assignment_id],
            owner="worker-b",
            pool_root=pool_root,
        )

    assert load_assignments(pool_root=pool_root)[assignment_id]["status"] == "active"
    released = release_batch(
        assignment_ids=[assignment_id],
        owner="worker-a",
        pool_root=pool_root,
    )
    assert released["released_count"] == 1
    assert released["released"][0]["status"] == "released"


def test_pool_acquire_batch_rejects_duplicate_shards(tmp_path) -> None:
    lease_root, registry_root, pool_root = _ready_pool_roots_with_leases(tmp_path, count=2)

    with pytest.raises(ValueError, match="Duplicate shard IDs"):
        acquire_batch(
            owner="benchmark",
            model="vendor/model",
            count=2,
            shard_ids=["shard-a", "shard-a"],
            health_depth="cached",
            lease_root=lease_root,
            registry_root=registry_root,
            pool_root=pool_root,
        )


def test_pool_acquire_batch_all_or_nothing_releases_partial_acquire(tmp_path) -> None:
    lease_root, registry_root, pool_root = _ready_pool_roots_with_leases(tmp_path, count=1)

    with pytest.raises(RuntimeError, match="No free endpoint"):
        acquire_batch(
            owner="benchmark",
            model="vendor/model",
            count=2,
            shard_ids=["shard-a", "shard-b"],
            health_depth="cached",
            lease_root=lease_root,
            registry_root=registry_root,
            pool_root=pool_root,
        )

    assignments = load_assignments(pool_root=pool_root)
    assert len(assignments) == 1
    assert next(iter(assignments.values()))["status"] == "released"


def test_pool_cli_acquire_batch_and_release_batch(tmp_path, capsys) -> None:
    lease_root, registry_root, pool_root = _ready_pool_roots_with_leases(tmp_path, count=2)
    shard_file = tmp_path / "shards.json"
    shard_file.write_text(json.dumps({"shards": [{"shard_id": "a"}, {"shard_id": "b"}]}))
    roots = [
        "--lease-root",
        str(lease_root),
        "--registry-root",
        str(registry_root),
        "--pool-root",
        str(pool_root),
    ]

    assert (
        main(
            [
                "pool",
                "acquire-batch",
                *roots,
                "--owner",
                "benchmark",
                "--model",
                "vendor/model",
                "--count",
                "2",
                "--shard-file",
                str(shard_file),
                "--health-depth",
                "cached",
            ]
        )
        == 0
    )
    handoff = json.loads(capsys.readouterr().out)
    assert handoff["acquired_count"] == 2

    assert main(["pool", "release-batch", *roots, handoff["batch_id"], "--format", "json"]) == 0
    released = json.loads(capsys.readouterr().out)
    assert released["released_count"] == 2


def test_pool_cli_status_acquire_release_and_manifest(tmp_path, capsys) -> None:
    lease_root, registry_root, pool_root = _ready_pool_roots(tmp_path)
    roots = [
        "--lease-root",
        str(lease_root),
        "--registry-root",
        str(registry_root),
        "--pool-root",
        str(pool_root),
    ]

    assert main(["pool", "status", *roots, "--format", "json"]) == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["summary"]["healthy_free"] == 1

    assert (
        main(
            [
                "pool",
                "acquire",
                *roots,
                "--owner",
                "benchmark",
                "--model",
                "vendor/model",
                "--health-depth",
                "cached",
            ]
        )
        == 0
    )
    handoff = json.loads(capsys.readouterr().out)
    assignment_id = handoff["assignment_id"]

    assert main(["pool", "manifest", *roots, "--assignment", assignment_id]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["assignment"]["assignment_id"] == assignment_id
    assert manifest["endpoints"][0]["assignment_state"] == "active"

    assert main(["pool", "release", *roots, assignment_id, "--format", "json"]) == 0
    released = json.loads(capsys.readouterr().out)
    assert released["status"] == "released"


def test_slurm_scheduler_refresh_batches_by_ssh_target() -> None:
    calls: list[list[str]] = []

    def fake_runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=_remote_batch_stdout(_slurm_stdout()))

    reports = refresh_slurm_scheduler(
        (
            {
                "endpoint_id": "endpoint-a",
                "ssh_target": "cluster",
                "job_id": "123",
                "expires_at": "2026-06-09T11:00:00Z",
                "slurm": {"pending_duration_seconds": 12.5},
            },
            {
                "endpoint_id": "endpoint-b",
                "ssh_target": "cluster",
                "job_id": "124",
                "expires_at": "2026-06-09T13:00:00Z",
            },
        ),
        command_runner=fake_runner,
        useful_deadline="2026-06-09T11:30:00Z",
    )

    assert len(calls) == 1
    command_text = " ".join(calls[0])
    assert "squeue -j 123,124" in command_text
    assert "sinfo -h" in command_text
    assert "find " not in command_text
    assert "grep" not in command_text
    assert "import torch" not in command_text
    assert "import vllm" not in command_text
    assert reports["endpoint-a"]["slurm_state"] == "RUNNING"
    assert reports["endpoint-a"]["pending_duration_seconds"] == 12.5
    assert reports["endpoint-b"]["slurm_state"] == "PENDING"
    assert reports["endpoint-b"]["estimated_start_at"] == "2026-06-09T12:00:00"
    assert reports["endpoint-b"]["start_after_useful_deadline"] is True


def test_partition_recommendations_use_single_inventory_request_per_target() -> None:
    calls: list[list[str]] = []

    def fake_runner(command: list[str], _timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout=_remote_batch_stdout(_partition_stdout())
        )

    payload = partition_recommendations(
        (
            {
                "endpoint_name": "actor",
                "ssh_target": "cluster",
                "walltime": "00:30:00",
                "num_gpus": 2,
            },
        ),
        command_runner=fake_runner,
    )

    assert len(calls) == 1
    target = payload["targets"]["cluster"]
    recommendation = target["partitions"][0]
    assert recommendation["compatible"] == [{"partition": "gpu", "reasons": []}]
    assert [partition["name"] for partition in target["inventory"]] == ["gpu", "short", "down"]
    assert target["inventory"][0]["states"] == ["idle", "drain"]
    rejected = {item["partition"]: item["reasons"] for item in recommendation["rejected"]}
    assert rejected["short"] == ["walltime_exceeded", "gres_insufficient"]
    assert rejected["down"] == ["partition_not_schedulable"]


def _ready_pool_roots(tmp_path):
    lease_root = tmp_path / "leases"
    registry_root = tmp_path / "runs"
    pool_root = tmp_path / "pool"
    plan = build_effective_launch_plan(
        LocalVllmConfig(
            name="actor",
            model="vendor/model",
            port=8123,
            tensor_parallel_size=1,
            max_model_len=8192,
            max_num_seqs=16,
        ),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=registry_root,
        ownership_policy="lease",
    )
    create_run_registry(plan)
    lease = create_pending_lease(
        plan,
        ttl="12h",
        lease_root=lease_root,
        lease_id="lease-20260603-160000-b64a19ef",
    )
    record_ready_sessions(
        plan.registry_dir,
        {
            "actor": SimpleNamespace(
                name="actor",
                api_base="http://127.0.0.1:8123/v1",
                served_model_name="vendor/model",
                summary_path=str(plan.endpoints[0].summary_path),
                cleanup_command="true",
                backend_kind="local_vllm",
                local_port=8123,
            )
        },
    )
    activate_lease(str(lease["lease_id"]), lease_root=lease_root)
    return lease_root, registry_root, pool_root


def _ready_pool_roots_with_leases(tmp_path, *, count: int):
    lease_root = tmp_path / "leases"
    registry_root = tmp_path / "runs"
    pool_root = tmp_path / "pool"
    for index in range(count):
        run_id = f"20260603-1530{index:02d}-a3f91c2b"
        endpoint_name = f"actor{index}"
        plan = build_effective_launch_plan(
            LocalVllmConfig(
                name=endpoint_name,
                model="vendor/model",
                port=8123 + index,
                tensor_parallel_size=1,
                max_model_len=8192,
                max_num_seqs=16,
            ),
            source_config_path=f"inference-{index}.yaml",
            run_id=run_id,
            created_at="2026-06-03T15:30:12Z",
            registry_root=registry_root,
            ownership_policy="lease",
        )
        create_run_registry(plan)
        lease = create_pending_lease(
            plan,
            ttl="12h",
            lease_root=lease_root,
            lease_id=f"lease-20260603-1600{index:02d}-b64a19ef",
        )
        record_ready_sessions(
            plan.registry_dir,
            {
                endpoint_name: SimpleNamespace(
                    name=endpoint_name,
                    api_base=f"http://127.0.0.1:{8123 + index}/v1",
                    served_model_name="vendor/model",
                    summary_path=str(plan.endpoints[0].summary_path),
                    cleanup_command="true",
                    backend_kind="local_vllm",
                    local_port=8123 + index,
                )
            },
        )
        activate_lease(str(lease["lease_id"]), lease_root=lease_root)
    return lease_root, registry_root, pool_root


def _ready_registry_roots(tmp_path):
    lease_root = tmp_path / "leases"
    registry_root = tmp_path / "runs"
    pool_root = tmp_path / "pool"
    plan = build_effective_launch_plan(
        LocalVllmConfig(
            name="actor",
            model="vendor/model",
            port=8123,
            tensor_parallel_size=1,
            max_model_len=8192,
            max_num_seqs=16,
        ),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=registry_root,
        ownership_policy="owned",
    )
    create_run_registry(plan)
    record_ready_sessions(
        plan.registry_dir,
        {
            "actor": SimpleNamespace(
                name="actor",
                api_base="http://127.0.0.1:8123/v1",
                served_model_name="vendor/model",
                summary_path=str(plan.endpoints[0].summary_path),
                cleanup_command="true",
                backend_kind="local_vllm",
                local_port=8123,
            )
        },
    )
    return lease_root, registry_root, pool_root


def _remote_batch_stdout(stdout: str) -> str:
    return "\n".join(
        [
            "__RIL_PREFLIGHT_CHECK_BEGIN__",
            "index=0",
            "returncode=0",
            "__RIL_PREFLIGHT_STDERR__",
            "__RIL_PREFLIGHT_STDOUT__",
            stdout.strip(),
            "__RIL_PREFLIGHT_CHECK_END__",
            "",
        ]
    )


def _slurm_stdout() -> str:
    return """
__RIL_POOL_SQUEUE__
123|RUNNING|node-a|None|gpu
124|PENDING|(null)|Priority|gpu
__RIL_POOL_START__
123|
124|124 gpu user PENDING 2026-06-09T12:00:00 1 (Priority)
__RIL_POOL_PARTITIONS__
gpu*|idle|01:00:00|gpu:4|2|(null)
"""


def _partition_stdout() -> str:
    return """
gpu|idle|01:00:00|gpu:4|2|(null)
gpu|drain|01:00:00|gpu:4|1|(null)
short|idle|00:10:00|gpu:1|1|(null)
down|down|01:00:00|gpu:4|1|(null)
"""
