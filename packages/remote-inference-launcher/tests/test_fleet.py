from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier, Event

import pytest

from remote_inference_launcher.core import InferenceSession
from remote_inference_launcher.fleet import FleetConfig, InferenceFleetLauncher
from remote_inference_launcher.summaries import SummaryWriter


@dataclass
class FakeLauncher:
    name: str
    events: list[str]
    fail: bool = False
    started_event: Event | None = None
    wait_for_start: Event | None = None

    def start(self) -> InferenceSession:
        self.events.append(f"start:{self.name}")
        if self.started_event is not None:
            self.started_event.set()
        if self.wait_for_start is not None:
            self.wait_for_start.wait(timeout=5)
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        return InferenceSession(
            name=self.name,
            api_base=f"http://127.0.0.1/{self.name}/v1",
            served_model_name=self.name,
            local_port=8000,
            remote_port=18000,
            job_id=f"job-{self.name}",
        )

    def stop(self) -> None:
        self.events.append(f"stop:{self.name}")


@dataclass
class BarrierLauncher:
    name: str
    events: list[str]
    barrier: Barrier

    def start(self) -> InferenceSession:
        self.events.append(f"start:{self.name}")
        self.barrier.wait(timeout=5)
        self.events.append(f"ready:{self.name}")
        return InferenceSession(
            name=self.name,
            api_base=f"http://127.0.0.1/{self.name}/v1",
            served_model_name=self.name,
        )

    def stop(self) -> None:
        self.events.append(f"stop:{self.name}")


@dataclass(frozen=True)
class SummaryConfig:
    name: str
    launch_summary_path: str = ""
    overwrite_launch_summary: bool = False


@dataclass
class SummaryReserveLauncher:
    config: SummaryConfig
    events: list[str]

    def reserve_summary(self) -> Path:
        self.events.append(f"reserve:{self.config.name}")
        return SummaryWriter(
            self.config.launch_summary_path,
            run_id=f"run-{self.config.name}",
            endpoint_name=self.config.name,
            backend_kind="fake",
            overwrite=self.config.overwrite_launch_summary,
        ).reserve()

    def start(self) -> InferenceSession:
        self.events.append(f"start:{self.config.name}")
        return InferenceSession(
            name=self.config.name,
            api_base=f"http://127.0.0.1/{self.config.name}/v1",
            served_model_name=self.config.name,
            summary_path=self.config.launch_summary_path,
        )

    def stop(self) -> None:
        self.events.append(f"stop:{self.config.name}")


def test_fleet_starts_launchers_concurrently(monkeypatch) -> None:
    events: list[str] = []
    barrier = Barrier(2)
    launchers = {
        "actor": BarrierLauncher("actor", events, barrier),
        "critic": BarrierLauncher("critic", events, barrier),
    }
    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        lambda config: launchers[config],
    )
    fleet = InferenceFleetLauncher({"actor": "actor", "critic": "critic"})

    sessions = fleet.start()

    assert set(sessions) == {"actor", "critic"}
    assert set(events[:2]) == {"start:actor", "start:critic"}
    assert set(events[2:]) == {"ready:actor", "ready:critic"}


def test_fleet_cleans_up_started_launchers_on_partial_failure(monkeypatch) -> None:
    events: list[str] = []
    actor_started = Event()
    launchers = {
        "actor": FakeLauncher("actor", events, started_event=actor_started),
        "critic": FakeLauncher("critic", events, fail=True, wait_for_start=actor_started),
    }

    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        lambda config: launchers[config],
    )
    fleet = InferenceFleetLauncher({"actor": "actor", "critic": "critic"})

    with pytest.raises(RuntimeError, match="critic failed"):
        fleet.start()

    assert "start:actor" in events
    assert "start:critic" in events
    assert "stop:actor" in events
    assert "stop:critic" in events


def test_fleet_keep_ready_fails_when_no_endpoint_becomes_ready(monkeypatch) -> None:
    events: list[str] = []
    launchers = {
        "actor": FakeLauncher("actor", events, fail=True),
        "critic": FakeLauncher("critic", events, fail=True),
    }
    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        lambda config: launchers[config],
    )
    fleet = InferenceFleetLauncher(
        FleetConfig(
            endpoints={"actor": "actor", "critic": "critic"},
            failure_policy="keep_ready",
        )
    )

    with pytest.raises(RuntimeError, match="did not start any endpoints"):
        fleet.start()


def test_fleet_rejects_best_effort_failure_policy() -> None:
    with pytest.raises(ValueError, match="fail_fast or keep_ready"):
        InferenceFleetLauncher(
            FleetConfig(
                endpoints={"actor": FakeLauncher("actor", [])},
                failure_policy="best_effort",
            )
        )


def test_fleet_env_uses_named_endpoint_prefixes(monkeypatch) -> None:
    events: list[str] = []
    launchers = {
        "actor": FakeLauncher("actor", events),
        "critic": FakeLauncher("critic", events),
    }
    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        lambda config: launchers[config],
    )
    fleet = InferenceFleetLauncher({"actor": "actor", "critic": "critic"})

    fleet.start()

    assert fleet.env() == {
        "INFERENCE_ACTOR_BASE_URL": "http://127.0.0.1/actor/v1",
        "INFERENCE_ACTOR_MODEL": "actor",
        "INFERENCE_CRITIC_BASE_URL": "http://127.0.0.1/critic/v1",
        "INFERENCE_CRITIC_MODEL": "critic",
    }


def test_fleet_event_callback_receives_ready_and_complete_events(monkeypatch) -> None:
    events: list[str] = []
    callbacks: list[str] = []
    launchers = {
        "actor": FakeLauncher("actor", events),
        "critic": FakeLauncher("critic", events),
    }
    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        lambda config: launchers[config],
    )
    fleet = InferenceFleetLauncher(
        {"actor": "actor", "critic": "critic"},
        event_callback=lambda event: callbacks.append(event.event),
    )

    fleet.start()

    assert callbacks.count("endpoint_ready") == 2
    assert callbacks[-1] == "fleet_complete"


def test_fleet_reserves_child_summaries_before_starting_endpoints(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    configs = {
        "actor": SummaryConfig("actor"),
        "critic": SummaryConfig("critic"),
    }

    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        lambda config: SummaryReserveLauncher(config, events),
    )
    fleet = InferenceFleetLauncher(
        FleetConfig(
            endpoints=configs,
            launch_summary_path=str(tmp_path / "fleet-summary.json"),
        )
    )

    fleet.start()

    reserve_positions = [
        index for index, event in enumerate(events) if event.startswith("reserve:")
    ]
    start_positions = [index for index, event in enumerate(events) if event.startswith("start:")]
    assert reserve_positions
    assert start_positions
    assert max(reserve_positions) < min(start_positions)


def test_fleet_child_summary_paths_are_unique_for_similar_endpoint_names(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    child_configs: list[SummaryConfig] = []
    configs = {
        "actor": SummaryConfig("actor"),
        "actor!": SummaryConfig("actor!"),
    }

    def launcher_from_config(config: SummaryConfig) -> SummaryReserveLauncher:
        child_configs.append(config)
        return SummaryReserveLauncher(config, events)

    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        launcher_from_config,
    )
    summary_path = tmp_path / "fleet-summary.json"
    fleet = InferenceFleetLauncher(
        FleetConfig(
            endpoints=configs,
            launch_summary_path=str(summary_path),
            overwrite_launch_summary=True,
        )
    )

    sessions = fleet.start()

    child_summary_paths = {config.launch_summary_path for config in child_configs}
    assert set(sessions) == {"actor", "actor!"}
    assert any(event == "start:actor" for event in events)
    assert any(event == "start:actor!" for event in events)
    assert len(child_summary_paths) == 2
    assert all(Path(path).parent.name == "endpoints" for path in child_summary_paths)
    assert all(Path(path).name.startswith("actor-") for path in child_summary_paths)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["lifecycle_state"] == "READY"


def test_fleet_summary_records_launcher_construction_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def launcher_from_config(_config: object) -> object:
        raise RuntimeError("bad endpoint config")

    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        launcher_from_config,
    )
    summary_path = tmp_path / "fleet-summary.json"
    fleet = InferenceFleetLauncher(
        FleetConfig(
            endpoints={"actor": "actor"},
            launch_summary_path=str(summary_path),
        )
    )

    with pytest.raises(RuntimeError, match="bad endpoint config"):
        fleet.start()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["lifecycle_state"] == "FAILED"
    assert payload["observed_resource_usage"]["failure_message"] == "bad endpoint config"


def test_fleet_summary_preserves_released_incremental_endpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    launchers = {
        "actor": FakeLauncher("actor", events),
        "critic": FakeLauncher("critic", events),
    }
    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        lambda config: launchers[config],
    )
    summary_path = tmp_path / "fleet-summary.json"
    fleet = InferenceFleetLauncher(
        FleetConfig(
            endpoints={"actor": "actor", "critic": "critic"},
            handoff_mode="incremental_ready",
            endpoint_lifetime_policy="per_endpoint",
            launch_summary_path=str(summary_path),
        )
    )

    fleet.start()
    fleet.release("actor")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["endpoints"]["actor"]["lifecycle_state"] == "RELEASED"
    assert payload["endpoints"]["critic"]["lifecycle_state"] == "READY"
    assert payload["endpoints"]["actor"]["local_port"] == 8000
    assert payload["endpoints"]["actor"]["remote_port"] == 18000
    assert payload["endpoints"]["actor"]["job_id"] == "job-actor"
