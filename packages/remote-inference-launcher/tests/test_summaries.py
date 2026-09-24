from __future__ import annotations

import json
from pathlib import Path

import pytest

from remote_inference_launcher.local_vllm import LocalVllmConfig
from remote_inference_launcher.plans import build_effective_launch_plan
from remote_inference_launcher.registry import create_run_registry, read_registry
from remote_inference_launcher.summaries import SummaryWriter, endpoint_summary


def test_summary_writer_refuses_existing_path_without_overwrite(tmp_path: Path) -> None:
    summary_path = tmp_path / "launch-summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")

    writer = SummaryWriter(
        summary_path,
        run_id="run",
        endpoint_name="endpoint",
        backend_kind="existing_endpoint",
    )

    with pytest.raises(FileExistsError, match="Launch summary already exists"):
        writer.reserve()


def test_summary_writer_truncates_existing_path_with_overwrite(tmp_path: Path) -> None:
    summary_path = tmp_path / "launch-summary.json"
    summary_path.write_text(
        '{"lifecycle_state": "OLD", "padding": "xxxxxxxxxx"}\n',
        encoding="utf-8",
    )

    writer = SummaryWriter(
        summary_path,
        run_id="run",
        endpoint_name="endpoint",
        backend_kind="existing_endpoint",
        overwrite=True,
    )

    writer.reserve()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload == {"lifecycle_state": "RESERVED"}


def test_summary_writer_mirrors_registry_endpoint_state(tmp_path: Path) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
    )
    create_run_registry(plan)
    writer = SummaryWriter(
        plan.endpoints[0].summary_path,
        run_id=plan.run_id,
        endpoint_name="actor",
        backend_kind="slurm_vllm",
    )

    writer.write(
        endpoint_summary(
            endpoint_name="actor",
            backend_kind="slurm_vllm",
            lifecycle_state="SUBMITTED",
            run_id=plan.run_id,
            served_model_name="vendor/model",
            job_id="12345",
            job_name="ril-actor-a3f91c2b",
            remote_log_dir="/remote/logs",
            remote_state_path="/remote/logs/remote-inference-state.json",
            resource_attempts=[
                {
                    "partition": "gpu",
                    "job_id": "12345",
                    "latest_slurm_state": "PENDING",
                    "latest_slurm_reason": "Priority",
                    "latest_slurm_diagnostics": "squeue_start: later",
                }
            ],
        )
    )

    registry = read_registry(plan.registry_dir)
    endpoint = registry["state"]["endpoints"]["actor"]
    events = (plan.registry_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()

    assert endpoint["lifecycle_state"] == "SUBMITTED"
    assert endpoint["job_id"] == "12345"
    assert endpoint["logs"] == "/remote/logs"
    assert endpoint["remote_state_path"] == "/remote/logs/remote-inference-state.json"
    assert endpoint["slurm"]["latest_state"] == "PENDING"
    assert endpoint["slurm"]["latest_reason"] == "Priority"
    assert any('"event": "slurm_job_submitted"' in event for event in events)
