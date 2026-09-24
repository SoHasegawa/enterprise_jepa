from __future__ import annotations

import json
import signal
from types import SimpleNamespace

import remote_inference_launcher.registry as registry_module
from remote_inference_launcher.cli import main
from remote_inference_launcher.local_vllm import LocalVllmConfig
from remote_inference_launcher.plans import build_effective_launch_plan
from remote_inference_launcher.registry import (
    cleanup_commands_from_state,
    create_run_registry,
    env_text_from_state,
    format_status,
    read_registry,
    record_endpoint_summary,
    record_ready_sessions,
    resolve_registry_path,
    run_stop,
    stop_controller,
    update_controller_pid,
    update_state,
)
from remote_inference_launcher.slurm_vllm import SlurmVllmConfig
from remote_inference_launcher.summaries import endpoint_summary


def test_registry_writes_plan_state_events_and_secret_free_env(tmp_path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model", api_key="secret"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
    )

    create_run_registry(plan)
    record_ready_sessions(
        plan.registry_dir,
        {
            "actor": SimpleNamespace(
                name="actor",
                api_base="http://127.0.0.1:8123/v1",
                served_model_name="vendor/model",
                model="vendor/model",
                api_key="secret",
                local_port=8123,
                summary_path=str(plan.endpoints[0].summary_path),
                cleanup_command="echo cleanup",
                backend_kind="local_vllm",
            )
        },
    )

    plan_payload = json.loads((plan.registry_dir / "plan.json").read_text(encoding="utf-8"))
    state_payload = json.loads((plan.registry_dir / "state.json").read_text(encoding="utf-8"))
    env_text = (plan.registry_dir / "env.sh").read_text(encoding="utf-8")
    events = (plan.registry_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert plan_payload["schema_version"] == "ril-plan/v1"
    assert state_payload["schema_version"] == "ril-state/v1"
    assert state_payload["lifecycle_state"] == "READY"
    assert state_payload["endpoints"]["actor"]["api_key_set"] is True
    assert "export OPENAI_API_KEY" not in env_text
    assert "OPENAI_BASE_URL='http://127.0.0.1:8123/v1'" in env_text
    assert len(events) == 2


def test_registry_resolution_status_env_and_cleanup_commands(tmp_path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
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
                cleanup_command="echo cleanup",
                job_id="12345",
                logs="/remote/logs",
                local_port=8123,
            )
        },
    )

    assert resolve_registry_path(plan.registry_dir) == plan.registry_dir
    assert resolve_registry_path(plan.endpoints[0].summary_path) == plan.registry_dir
    registry = read_registry(plan.registry_dir)
    status = format_status(registry)
    env_text = env_text_from_state(registry["state"], endpoint_name="actor")

    assert "state=READY" in status
    assert "endpoint=actor" in status
    assert "job_id=12345" in status
    assert "logs=/remote/logs" in status
    assert "cleanup_command=echo cleanup" in status
    assert "OPENAI_MODEL_NAME='vendor/model'" in env_text
    assert cleanup_commands_from_state(registry["state"]) == ["echo cleanup"]


def test_registry_mirrors_pending_slurm_summary_status_and_event(tmp_path) -> None:
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
    )
    create_run_registry(plan)
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
            remote_log_dir="/remote/logs",
            remote_state_path="/remote/logs/state.json",
            resource_attempts=[
                {
                    "candidate_index": 0,
                    "candidate_name": "default",
                    "backend_kind": "slurm_vllm",
                    "ssh_target": "cluster",
                    "partition": "gpu",
                    "job_id": "12345",
                    "latest_slurm_state": "PENDING",
                    "latest_slurm_reason": "Priority",
                    "latest_slurm_diagnostics": "squeue --start unavailable",
                    "pending_duration_seconds": 12.5,
                }
            ],
        ),
    )

    registry = read_registry(plan.registry_dir)
    status = format_status(registry)
    events = [
        json.loads(line)
        for line in (plan.registry_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert "endpoint=default state=PENDING" in status
    assert "job_id=12345" in status
    assert "logs=/remote/logs" in status
    assert "slurm_state=PENDING" in status
    assert "slurm_reason=Priority" in status
    assert events[-1]["event"] == "slurm_job_pending"
    assert events[-1]["payload"]["slurm"]["latest_reason"] == "Priority"


def test_registry_uses_explicit_summary_event_for_early_remote_progress(tmp_path) -> None:
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
    )
    create_run_registry(plan)
    record_endpoint_summary(
        plan.endpoints[0].summary_path,
        endpoint_summary(
            endpoint_name="default",
            backend_kind="slurm_vllm",
            lifecycle_state="SUBMITTING",
            run_id=plan.run_id,
            served_model_name="vendor/model",
            model="vendor/model",
            extra={
                "registry_event": "remote_output_directory_created",
                "registry_event_payload": {
                    "remote_log_dir": "/remote/logs",
                    "remote_state_path": "/remote/logs/state.json",
                },
            },
        ),
    )

    registry = read_registry(plan.registry_dir)
    events = [
        json.loads(line)
        for line in (plan.registry_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert registry["state"]["endpoints"]["default"]["lifecycle_state"] == "SUBMITTING"
    assert events[-1]["event"] == "remote_output_directory_created"
    assert events[-1]["payload"]["remote_log_dir"] == "/remote/logs"
    assert events[-1]["payload"]["remote_state_path"] == "/remote/logs/state.json"


def test_registry_reconciles_terminal_slurm_state_as_endpoint_expired(tmp_path) -> None:
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
    )
    create_run_registry(plan)
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
                    "latest_slurm_state": "TIMEOUT",
                    "latest_slurm_reason": "TimeLimit",
                    "pending_duration_seconds": 12.5,
                }
            ],
        ),
    )

    registry = read_registry(plan.registry_dir)
    status = format_status(registry)
    events = [
        json.loads(line)
        for line in (plan.registry_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert registry["state"]["lifecycle_state"] == "EXPIRED"
    assert registry["state"]["endpoints"]["default"]["lifecycle_state"] == "EXPIRED"
    assert (
        registry["state"]["endpoints"]["default"]["diagnostics"]["expiry_reason"]
        == "slurm_job_terminal"
    )
    assert "state=EXPIRED" in status
    assert "endpoint=default state=EXPIRED" in status
    assert "slurm_state=TIMEOUT" in status
    assert "cleanup_command=ssh cluster 'scancel 12345'" in status
    assert events[-1]["event"] == "endpoint_expired"
    assert events[-1]["payload"]["slurm_state"] == "TIMEOUT"


def test_registry_cli_commands_read_recorded_state(tmp_path, capsys) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
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
                cleanup_command="echo cleanup",
            )
        },
    )

    assert main(["status", str(plan.registry_dir)]) == 0
    assert "state=READY" in capsys.readouterr().out
    assert main(["env", str(plan.registry_dir), "--endpoint", "actor"]) == 0
    assert "OPENAI_BASE_URL='http://127.0.0.1:8123/v1'" in capsys.readouterr().out
    assert main(["cleanup-command", str(plan.registry_dir)]) == 0
    assert capsys.readouterr().out.strip() == "echo cleanup"


def test_registry_status_marks_dead_detached_controller_orphaned(tmp_path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        controller_policy="detached",
    )
    create_run_registry(plan)
    update_controller_pid(plan.registry_dir, 999999999)
    update_state(plan.registry_dir, lifecycle_state="READY")

    registry = read_registry(plan.registry_dir)

    assert registry["state"]["lifecycle_state"] == "ORPHANED"
    assert "state=ORPHANED" in format_status(registry)


def test_registry_logs_command_prints_paths_and_tail(tmp_path, capsys) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        controller_policy="detached",
    )
    create_run_registry(plan)
    (plan.registry_dir / "controller.log").write_text("first\nlast\n", encoding="utf-8")

    assert main(["logs", str(plan.registry_dir), "--tail", "1"]) == 0
    output = capsys.readouterr().out
    assert "controller.log" in output
    assert "last" in output
    assert "first" not in output
    assert main(["logs", str(plan.registry_dir), "--paths-only"]) == 0
    assert capsys.readouterr().out.strip().endswith("controller.log")


def test_registry_stop_force_cleans_orphaned_run(tmp_path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        controller_policy="detached",
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
            )
        },
    )
    update_controller_pid(plan.registry_dir, 999999999)
    update_state(plan.registry_dir, lifecycle_state="ORPHANED")

    assert main(["stop", str(plan.registry_dir), "--force"]) == 0
    registry = read_registry(plan.registry_dir)
    assert registry["state"]["lifecycle_state"] == "STOPPED"


def test_registry_stop_records_cleanup_command_timeout(monkeypatch, tmp_path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
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
                cleanup_command="ssh cluster 'scancel 12345'",
            )
        },
    )

    def timeout_run(command: str, **_kwargs: object) -> object:
        raise registry_module.subprocess.TimeoutExpired(command, timeout=0.01)

    monkeypatch.setattr(registry_module.subprocess, "run", timeout_run)

    assert run_stop(plan.registry_dir, cleanup_timeout_seconds=0.01) == 124
    registry = read_registry(plan.registry_dir)
    events = [
        json.loads(line)
        for line in (plan.registry_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert registry["state"]["lifecycle_state"] == "FAILED"
    assert events[-2]["event"] == "cleanup_command_timeout"
    assert events[-2]["payload"]["returncode"] == 124
    assert events[-1]["event"] == "run_stop_failed"


def test_stop_controller_without_force_does_not_escalate(monkeypatch, tmp_path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        controller_policy="detached",
    )
    create_run_registry(plan)
    update_controller_pid(plan.registry_dir, 4242)
    signals: list[int] = []

    monkeypatch.setattr(registry_module, "pid_alive", lambda _pid: True)
    monkeypatch.setattr(registry_module, "_wait_for_pid_exit", lambda _pid, **_kwargs: False)
    monkeypatch.setattr(
        registry_module.os,
        "kill",
        lambda _pid, signum: signals.append(signum),
    )

    assert stop_controller(plan.registry_dir, force=False) == 1
    assert signals == [signal.SIGTERM]


def test_stop_controller_force_escalates_to_sigkill(monkeypatch, tmp_path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        controller_policy="detached",
    )
    create_run_registry(plan)
    update_controller_pid(plan.registry_dir, 4242)
    signals: list[int] = []
    waits = iter([False, True])

    monkeypatch.setattr(registry_module, "pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        registry_module,
        "_wait_for_pid_exit",
        lambda _pid, **_kwargs: next(waits),
    )
    monkeypatch.setattr(
        registry_module.os,
        "kill",
        lambda _pid, signum: signals.append(signum),
    )

    assert stop_controller(plan.registry_dir, force=True) == 0
    assert signals == [signal.SIGTERM, signal.SIGKILL]
