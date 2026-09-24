from __future__ import annotations

import json
import signal
from pathlib import Path
from threading import Event

import remote_inference_launcher.controller as controller
import remote_inference_launcher.inference_config as inference_config
from remote_inference_launcher.local_vllm import LocalVllmConfig
from remote_inference_launcher.plans import build_effective_launch_plan
from remote_inference_launcher.registry import create_run_registry, read_json


class _BlockingStartupLauncher:
    def __init__(self, heartbeat_seen: Event) -> None:
        self.heartbeat_seen = heartbeat_seen
        self.stopped = False

    def start(self) -> object:
        if not self.heartbeat_seen.wait(timeout=1.0):
            raise RuntimeError("heartbeat did not run while startup was blocked")
        raise RuntimeError("startup still pending")

    def stop(self) -> None:
        self.stopped = True


def test_controller_heartbeats_while_launcher_start_is_blocked(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = build_effective_launch_plan(
        LocalVllmConfig(name="actor", model="vendor/model"),
        source_config_path="inference.yaml",
        run_id="20260603-153012-a3f91c2b",
        created_at="2026-06-03T15:30:12Z",
        registry_root=tmp_path / "runs",
        controller_policy="detached",
    )
    create_run_registry(plan)
    config_path = tmp_path / "inference.yaml"
    config_path.write_text("kind: local_vllm\nmodel: vendor/model\n", encoding="utf-8")

    heartbeat_seen = Event()
    launcher = _BlockingStartupLauncher(heartbeat_seen)
    heartbeat_count = 0
    original_heartbeat = controller.update_controller_heartbeat

    def count_heartbeat(registry_dir: str | Path, pid: int) -> None:
        nonlocal heartbeat_count
        heartbeat_count += 1
        original_heartbeat(registry_dir, pid)
        if heartbeat_count >= 2:
            heartbeat_seen.set()

    monkeypatch.setattr(controller, "HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(controller, "update_controller_heartbeat", count_heartbeat)
    monkeypatch.setattr(
        inference_config,
        "load_inference_launcher",
        lambda _path, **_kwargs: launcher,
    )
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        status = controller.run_controller(registry_dir=plan.registry_dir, config_path=config_path)
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

    state = read_json(plan.registry_dir / "state.json")
    events = [
        json.loads(line)["event"]
        for line in (plan.registry_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert status == 2
    assert launcher.stopped
    assert heartbeat_count >= 2
    assert state["lifecycle_state"] == "FAILED"
    assert state["controller"]["heartbeat_at"]
    assert state["failure"]["message"] == "startup still pending"
    assert "controller_started" in events
    assert "launcher_loaded" in events
