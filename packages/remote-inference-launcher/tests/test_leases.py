from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from remote_inference_launcher.cli import main
from remote_inference_launcher.leases import (
    activate_lease,
    attach_lease,
    benchmark_metadata_from_handoff,
    create_pending_lease,
    format_lease_status,
    format_lease_stop_plan,
    lease_env_text,
    parse_ttl,
    read_lease,
    record_lease_ready_timeout,
    recover_lease,
    stop_lease,
    write_lease,
)
from remote_inference_launcher.local_vllm import LocalVllmConfig
from remote_inference_launcher.plans import build_effective_launch_plan
from remote_inference_launcher.registry import (
    create_run_registry,
    read_registry,
    record_endpoint_summary,
    record_failure,
    record_ready_sessions,
)
from remote_inference_launcher.slurm_vllm import SlurmVllmConfig
from remote_inference_launcher.summaries import endpoint_summary


def test_parse_ttl_supports_compact_units() -> None:
    assert parse_ttl("30m").total_seconds() == 1800
    assert parse_ttl("12h").total_seconds() == 43200
    assert parse_ttl("2d").total_seconds() == 172800
    assert parse_ttl("45").total_seconds() == 45


def test_lease_lifecycle_env_attach_metadata_and_stop(tmp_path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        ownership_policy="lease",
    )
    create_run_registry(plan)
    lease = create_pending_lease(
        plan,
        ttl="12h",
        lease_root=tmp_path / "leases",
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
            )
        },
    )

    assert lease["status"] == "pending"
    active = activate_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")
    assert active["status"] == "active"
    assert "status=active" in format_lease_status(active)
    assert "OPENAI_BASE_URL='http://127.0.0.1:8123/v1'" in lease_env_text(
        active,
        endpoint_name="actor",
    )

    handoff = attach_lease(
        str(lease["lease_id"]),
        lease_root=tmp_path / "leases",
        readiness_checker=lambda _endpoint: None,
    )
    metadata = benchmark_metadata_from_handoff(handoff)
    assert handoff["endpoints"]["actor"]["model"] == "vendor/model"
    assert metadata["inference_identity_key"] == "lease:lease-20260603-160000-b64a19ef"
    assert metadata["inference_served_model_name"] == "vendor/model"
    updated = read_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")
    assert updated["attach_count"] == 1

    assert stop_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases") == 0
    stopped = read_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")
    assert stopped["status"] == "stopped"


def test_ready_at_ttl_starts_when_lease_activates(tmp_path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        ownership_policy="lease",
    )
    create_run_registry(plan)
    lease = create_pending_lease(
        plan,
        ttl="12h",
        ttl_start="ready_at",
        lease_root=tmp_path / "leases",
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
            )
        },
    )

    assert lease["expires_at"] == ""
    assert lease["ttl_policy"]["mode"] == "ready_at"
    active = activate_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")

    assert active["ready_at"]
    assert active["expires_at"] > active["ready_at"]


def test_ready_at_ttl_starts_when_read_reconciles_ready_registry(tmp_path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        ownership_policy="lease",
    )
    create_run_registry(plan)
    lease = create_pending_lease(
        plan,
        ttl="12h",
        ttl_start="ready_at",
        lease_root=tmp_path / "leases",
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
            )
        },
    )

    active = read_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")

    assert active["status"] == "active"
    assert active["ready_at"]
    assert active["expires_at"] > active["ready_at"]


def test_read_lease_reconciles_ready_registry_with_endpoint_details(tmp_path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        ownership_policy="lease",
    )
    create_run_registry(plan)
    lease = create_pending_lease(
        plan,
        ttl="12h",
        lease_root=tmp_path / "leases",
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
                job_id="12345",
                logs="/remote/logs",
                local_port=8123,
            )
        },
    )

    active = read_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")
    status = format_lease_status(active)

    assert active["status"] == "active"
    assert active["ready_at"]
    assert active["expires_at"] > active["ready_at"]
    assert active["endpoints"]["actor"]["job_id"] == "12345"
    assert "endpoint=actor" in status
    assert "job_id=12345" in status
    assert "logs=/remote/logs" in status


def test_read_lease_reconciles_failed_registry_for_pending_lease(tmp_path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        ownership_policy="lease",
    )
    create_run_registry(plan)
    lease = create_pending_lease(
        plan,
        ttl="12h",
        lease_root=tmp_path / "leases",
        lease_id="lease-20260603-160000-b64a19ef",
    )
    record_failure(plan.registry_dir, RuntimeError("submission failed"))

    failed = read_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")
    status = format_lease_status(failed)

    assert failed["status"] == "failed"
    assert failed["registry_lifecycle_state"] == "FAILED"
    assert failed["failure"]["code"] == "RuntimeError"
    assert "status=failed" in status
    assert "submission failed" in status


def test_read_lease_reconciles_failed_registry_for_active_lease(tmp_path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        ownership_policy="lease",
    )
    create_run_registry(plan)
    lease = create_pending_lease(
        plan,
        ttl="12h",
        lease_root=tmp_path / "leases",
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
            )
        },
    )
    activate_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")
    record_failure(plan.registry_dir, RuntimeError("controller crashed"))

    failed = read_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")

    assert failed["status"] == "failed"
    assert failed["registry_lifecycle_state"] == "FAILED"
    assert failed["failure"]["message"] == "controller crashed"
    with pytest.raises(RuntimeError, match="not active: failed"):
        attach_lease(
            str(lease["lease_id"]),
            lease_root=tmp_path / "leases",
            readiness_checker=lambda _endpoint: None,
        )


def test_active_lease_expiry_marks_registry_and_endpoint_expired(tmp_path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        ownership_policy="lease",
    )
    create_run_registry(plan)
    lease = create_pending_lease(
        plan,
        ttl="12h",
        lease_root=tmp_path / "leases",
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
            )
        },
    )
    active = activate_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")
    write_lease(
        {**active, "expires_at": "2020-01-01T00:00:00Z"},
        lease_root=tmp_path / "leases",
    )

    expired = read_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")
    registry = read_registry(plan.registry_dir)
    events = [
        json.loads(line)["event"]
        for line in (plan.registry_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert expired["status"] == "expired"
    assert expired["endpoints"]["actor"]["lifecycle_state"] == "EXPIRED"
    assert registry["state"]["lifecycle_state"] == "EXPIRED"
    assert registry["state"]["lease"]["status"] == "expired"
    assert registry["state"]["endpoints"]["actor"]["lifecycle_state"] == "EXPIRED"
    assert "lifecycle_state=EXPIRED" in format_lease_status(expired)
    assert "lease_expired" in events
    with pytest.raises(RuntimeError, match=r"Lease .* is expired"):
        attach_lease(
            str(lease["lease_id"]),
            lease_root=tmp_path / "leases",
            readiness_checker=lambda _endpoint: None,
        )


def test_read_lease_reconciles_queued_slurm_registry_details(tmp_path) -> None:
    plan = build_effective_launch_plan(
        SlurmVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            partition="gpu",
            walltime="00:30:00",
            num_gpus=1,
            memory="32GB",
            cpus_per_task=4,
        ),
        source_config_path="slurm.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        ownership_policy="lease",
    )
    create_run_registry(plan)
    lease = create_pending_lease(
        plan,
        ttl="12h",
        lease_root=tmp_path / "leases",
        lease_id="lease-20260603-160000-b64a19ef",
    )
    record_endpoint_summary(
        plan.endpoints[0].summary_path,
        endpoint_summary(
            endpoint_name="default",
            backend_kind="slurm_vllm",
            lifecycle_state="PENDING",
            run_id=plan.run_id,
            served_model_name="vendor/model",
            model="vendor/model",
            job_id="12345",
            cleanup_command="ssh cluster 'scancel 12345'",
            remote_log_dir="/remote/logs",
            remote_state_path="/remote/logs/state.json",
            resource_attempts=[
                {
                    "partition": "gpu",
                    "job_id": "12345",
                    "latest_slurm_state": "PENDING",
                    "latest_slurm_reason": "Priority",
                    "pending_duration_seconds": 12.5,
                }
            ],
        ),
    )

    queued = read_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")
    status = format_lease_status(queued)

    assert queued["status"] == "queued"
    assert queued["endpoints"]["default"]["job_id"] == "12345"
    assert queued["endpoints"]["default"]["cleanup_command"] == "ssh cluster 'scancel 12345'"
    assert queued["endpoints"]["default"]["slurm"]["latest_reason"] == "Priority"
    assert "status=queued" in status
    assert "slurm_reason=Priority" in status


def test_read_lease_reconciles_terminal_slurm_state_as_expired(tmp_path) -> None:
    plan = build_effective_launch_plan(
        SlurmVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            partition="gpu",
            walltime="00:30:00",
            num_gpus=1,
            memory="32GB",
            cpus_per_task=4,
        ),
        source_config_path="slurm.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        ownership_policy="lease",
    )
    create_run_registry(plan)
    lease = create_pending_lease(
        plan,
        ttl="12h",
        lease_root=tmp_path / "leases",
        lease_id="lease-20260603-160000-b64a19ef",
    )
    record_ready_sessions(
        plan.registry_dir,
        {
            "default": SimpleNamespace(
                name="default",
                api_base="http://127.0.0.1:8123/v1",
                served_model_name="vendor/model",
                summary_path=str(plan.endpoints[0].summary_path),
                cleanup_command="ssh cluster 'scancel 12345'",
                job_id="12345",
                logs="/remote/logs",
            )
        },
    )
    activate_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")
    record_endpoint_summary(
        plan.endpoints[0].summary_path,
        endpoint_summary(
            endpoint_name="default",
            backend_kind="slurm_vllm",
            lifecycle_state="READY",
            run_id=plan.run_id,
            served_model_name="vendor/model",
            model="vendor/model",
            job_id="12345",
            cleanup_command="ssh cluster 'scancel 12345'",
            remote_log_dir="/remote/logs",
            remote_state_path="/remote/logs/state.json",
            resource_attempts=[
                {
                    "partition": "gpu",
                    "job_id": "12345",
                    "latest_slurm_state": "COMPLETED",
                    "latest_slurm_reason": "None",
                }
            ],
        ),
    )

    expired = read_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")
    status = format_lease_status(expired)

    assert expired["status"] == "expired"
    assert expired["registry_lifecycle_state"] == "EXPIRED"
    assert expired["endpoints"]["default"]["lifecycle_state"] == "EXPIRED"
    assert expired["endpoints"]["default"]["slurm"]["latest_state"] == "COMPLETED"
    assert "status=expired" in status
    assert "lifecycle_state=EXPIRED" in status
    with pytest.raises(RuntimeError, match=r"Lease .* is expired"):
        attach_lease(
            str(lease["lease_id"]),
            lease_root=tmp_path / "leases",
            readiness_checker=lambda _endpoint: None,
        )


def test_pending_lease_stop_dry_run_filters_self_registry_cleanup(tmp_path) -> None:
    plan = build_effective_launch_plan(
        SlurmVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            partition="gpu",
            walltime="00:30:00",
            num_gpus=1,
            memory="32GB",
            cpus_per_task=4,
        ),
        source_config_path="slurm.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        ownership_policy="lease",
    )
    create_run_registry(plan)
    lease = create_pending_lease(
        plan,
        ttl="12h",
        lease_root=tmp_path / "leases",
        lease_id="lease-20260603-160000-b64a19ef",
    )

    dry_run = format_lease_stop_plan(str(lease["lease_id"]), lease_root=tmp_path / "leases")

    assert "cleanup_commands=0" in dry_run
    assert "remote-inference-launcher stop" not in dry_run
    assert stop_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases") == 0
    stopped = read_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")
    assert stopped["status"] == "stopped"


def test_lease_cli_stop_dry_run(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    plan = build_effective_launch_plan(
        SlurmVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            partition="gpu",
            walltime="00:30:00",
            num_gpus=1,
            memory="32GB",
            cpus_per_task=4,
        ),
        source_config_path="slurm.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        ownership_policy="lease",
    )
    create_run_registry(plan)
    lease = create_pending_lease(
        plan,
        ttl="12h",
        lease_id="lease-20260603-160000-b64a19ef",
    )

    assert main(["lease", "stop", str(lease["lease_id"]), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "lease_id=lease-20260603-160000-b64a19ef" in output
    assert "cleanup_commands=0" in output


def test_lease_attach_rejects_mismatched_expected_config(tmp_path) -> None:
    expected_config = tmp_path / "expected.yaml"
    expected_config.write_text(
        "kind: local_vllm\nname: actor\nmodel: different/model\n",
        encoding="utf-8",
    )
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        ownership_policy="lease",
    )
    create_run_registry(plan)
    lease = create_pending_lease(
        plan,
        ttl="12h",
        lease_root=tmp_path / "leases",
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
            )
        },
    )
    activate_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")

    with pytest.raises(RuntimeError, match="semantic config hash mismatch"):
        attach_lease(
            str(lease["lease_id"]),
            expect_config=expected_config,
            lease_root=tmp_path / "leases",
            readiness_checker=lambda _endpoint: None,
        )


def test_lease_cli_start_status_env_and_attach(monkeypatch, tmp_path, capsys) -> None:
    import remote_inference_launcher.inference_config as inference_config
    import remote_inference_launcher.leases as leases

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        inference_config,
        "load_inference_launcher",
        lambda _path, **_kwargs: _ExternalLauncher(),
    )
    monkeypatch.setattr(leases, "lightweight_readiness_check", lambda _endpoint: None)
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        "kind: existing_endpoint\n"
        "name: actor\n"
        "api_base: http://127.0.0.1:8123/v1\n"
        "served_model_name: vendor/model\n"
        "readiness: {smoke_test: disabled}\n",
        encoding="utf-8",
    )

    assert main(["lease", "start", "--config", str(config_path), "--ttl", "12h"]) == 0
    start_output = capsys.readouterr().out
    assert "lease_id=lease-" in start_output
    lease_id = next(
        line for line in start_output.splitlines() if line.startswith("lease_id=")
    ).split(
        "=",
        1,
    )[1]
    assert main(["lease", "status", lease_id]) == 0
    assert "status=active" in capsys.readouterr().out
    assert main(["lease", "env", lease_id, "--endpoint", "actor"]) == 0
    assert "OPENAI_MODEL_NAME='vendor/model'" in capsys.readouterr().out
    assert main(["lease", "attach", lease_id, "--format", "json"]) == 0
    handoff = json.loads(capsys.readouterr().out)
    assert handoff["lease_id"] == lease_id
    assert handoff["endpoints"]["actor"]["api_base"] == "http://127.0.0.1:8123/v1"


def test_lease_cli_detach_wait_ready_activates_lease(monkeypatch, tmp_path, capsys) -> None:
    import remote_inference_launcher.controller as controller
    import remote_inference_launcher.leases as leases
    import remote_inference_launcher.registry as registry

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(leases, "lightweight_readiness_check", lambda _endpoint: None)

    def fake_spawn_detached_controller(**kwargs: object) -> SimpleNamespace:
        registry_dir = kwargs["registry_dir"]
        log_path = registry_dir / "controller.log"
        log_path.write_text("ready\n", encoding="utf-8")
        registry.update_controller_pid(registry_dir, os.getpid())
        registry.record_ready_sessions(
            registry_dir,
            {
                "actor": SimpleNamespace(
                    name="actor",
                    api_base="http://127.0.0.1:8123/v1",
                    served_model_name="vendor/model",
                    summary_path="/tmp/summary.json",
                )
            },
        )
        return SimpleNamespace(pid=os.getpid(), log_path=log_path)

    monkeypatch.setattr(controller, "spawn_detached_controller", fake_spawn_detached_controller)
    monkeypatch.setattr(
        controller,
        "wait_for_controller_start",
        lambda registry_dir: registry.read_json(registry_dir / "state.json"),
    )
    monkeypatch.setattr(
        controller,
        "wait_for_ready_state",
        lambda registry_dir, **_kwargs: registry.read_json(registry_dir / "state.json"),
    )
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        "kind: existing_endpoint\n"
        "name: actor\n"
        "api_base: http://127.0.0.1:8123/v1\n"
        "served_model_name: vendor/model\n"
        "readiness: {smoke_test: disabled}\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "lease",
                "start",
                "--config",
                str(config_path),
                "--ttl",
                "12h",
                "--detach",
                "--wait-ready",
            ]
        )
        == 0
    )
    lease_id = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("lease_id=")
    ).split("=", 1)[1]

    assert main(["lease", "status", lease_id]) == 0
    assert "status=active" in capsys.readouterr().out
    assert main(["lease", "attach", lease_id]) == 0
    handoff = json.loads(capsys.readouterr().out)
    assert handoff["schema_version"] == "ril-handoff/v1"
    assert handoff["identity_key"] == f"lease:{lease_id}"


def test_lease_cli_detach_wait_ready_timeout_records_recoverable_status(
    monkeypatch, tmp_path, capsys
) -> None:
    import remote_inference_launcher.controller as controller
    import remote_inference_launcher.registry as registry

    monkeypatch.chdir(tmp_path)
    timeout_values: list[float] = []

    def fake_spawn_detached_controller(**kwargs: object) -> SimpleNamespace:
        registry_dir = kwargs["registry_dir"]
        log_path = registry_dir / "controller.log"
        log_path.write_text("starting\n", encoding="utf-8")
        registry.update_controller_pid(registry_dir, os.getpid())
        return SimpleNamespace(pid=os.getpid(), log_path=log_path)

    def fake_wait_for_ready_state(_registry_dir, *, timeout_seconds: float, **_kwargs):
        timeout_values.append(timeout_seconds)
        raise TimeoutError("still loading model")

    monkeypatch.setattr(controller, "spawn_detached_controller", fake_spawn_detached_controller)
    monkeypatch.setattr(
        controller,
        "wait_for_controller_start",
        lambda registry_dir: registry.read_json(registry_dir / "state.json"),
    )
    monkeypatch.setattr(controller, "wait_for_ready_state", fake_wait_for_ready_state)
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        "kind: existing_endpoint\n"
        "name: actor\n"
        "api_base: http://127.0.0.1:8123/v1\n"
        "served_model_name: vendor/model\n"
        "readiness: {smoke_test: disabled}\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "lease",
                "start",
                "--config",
                str(config_path),
                "--ttl",
                "12h",
                "--detach",
                "--wait-ready",
                "--ready-timeout",
                "1800",
            ]
        )
        == 2
    )
    assert timeout_values == [1800.0]
    assert "lease recover" in capsys.readouterr().err
    lease_files = list((tmp_path / ".remote-inference-launcher" / "leases").glob("*.json"))
    assert len(lease_files) == 1
    lease = read_lease(lease_files[0], lease_root=lease_files[0].parent)
    assert lease["status"] == "ready_timeout"
    assert lease["ready_timeout"]["message"] == "still loading model"


def test_recover_lease_activates_timed_out_ready_endpoint(tmp_path) -> None:
    plan = build_effective_launch_plan(
        SlurmVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            partition="gpu",
            walltime="00:30:00",
            num_gpus=1,
            memory="32GB",
            cpus_per_task=4,
        ),
        source_config_path="slurm.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        ownership_policy="lease",
    )
    create_run_registry(plan)
    lease = create_pending_lease(
        plan,
        ttl="12h",
        ttl_start="ready_at",
        lease_root=tmp_path / "leases",
        lease_id="lease-20260603-160000-b64a19ef",
    )
    record_lease_ready_timeout(
        lease,
        TimeoutError("still loading model"),
        lease_root=tmp_path / "leases",
    )
    record_ready_sessions(
        plan.registry_dir,
        {
            "default": SimpleNamespace(
                name="default",
                api_base="http://127.0.0.1:8123/v1",
                served_model_name="vendor/model",
                summary_path=str(plan.endpoints[0].summary_path),
                cleanup_command="ssh cluster 'scancel 12345'",
                backend_kind="slurm_vllm",
                job_id="12345",
                local_port=8123,
                remote_port=8000,
                logs="/remote/logs",
            )
        },
    )

    pending_recovery = read_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")
    assert pending_recovery["status"] == "ready_timeout"
    assert pending_recovery["registry_lifecycle_state"] == "READY"

    recovered = recover_lease(
        str(lease["lease_id"]),
        expected_model="vendor/model",
        health_depth="cached",
        lease_root=tmp_path / "leases",
    )

    assert recovered["schema_version"] == "ril-lease-recovery/v1"
    assert recovered["lease"]["status"] == "active"
    assert recovered["lease"]["ready_at"]
    assert recovered["lease"]["expires_at"] > recovered["lease"]["ready_at"]
    assert recovered["handoff"]["identity_key"] == "lease:lease-20260603-160000-b64a19ef"
    assert recovered["health"]["default"]["status"] == "healthy"
    active = read_lease(str(lease["lease_id"]), lease_root=tmp_path / "leases")
    assert active["status"] == "active"


class _ExternalLauncher:
    owns_resources = False

    def start(self) -> SimpleNamespace:
        return SimpleNamespace(
            name="actor",
            api_base="http://127.0.0.1:8123/v1",
            served_model_name="vendor/model",
            model="vendor/model",
            summary_path="/tmp/summary.json",
        )

    def stop(self) -> None:
        return None
