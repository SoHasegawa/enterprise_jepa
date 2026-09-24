from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pytest

from remote_inference_launcher.config_types import ResourceBudget
from remote_inference_launcher.core import InferenceSession
from remote_inference_launcher.endpoint_race import EndpointRaceConfig, EndpointRaceLauncher


@dataclass
class FakeCandidate:
    name: str
    events: list[str]
    fail: bool = False
    cleanup_command: str = ""

    def start(self) -> InferenceSession:
        self.events.append(f"start:{self.name}")
        if self.fail:
            if self.cleanup_command:
                self._last_summary = {"cleanup_command": self.cleanup_command}
            raise RuntimeError(f"{self.name} failed")
        return InferenceSession(
            name=self.name,
            api_base=f"http://127.0.0.1/{self.name}/v1",
            served_model_name=self.name,
            backend_kind="fake",
            cleanup_command=self.cleanup_command,
        )

    def stop(self) -> None:
        self.events.append(f"stop:{self.name}")
        if self.cleanup_command:
            self._last_summary = {"cleanup_command": self.cleanup_command}


def test_endpoint_race_returns_first_ready_candidate(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    summary_path = tmp_path / "race.json"
    candidates = {
        "bad": FakeCandidate("bad", events, fail=True, cleanup_command="cleanup bad"),
        "good": FakeCandidate("good", events, cleanup_command="cleanup good"),
    }
    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        lambda config: candidates[config],
    )
    launcher = EndpointRaceLauncher(
        EndpointRaceConfig(
            name="logical",
            candidates={"bad": "bad", "good": "good"},
            max_active_candidates=1,
            launch_summary_path=str(summary_path),
        )
    )

    session = launcher.start()

    assert session.name == "logical"
    assert session.api_base == "http://127.0.0.1/good/v1"
    assert events[:3] == ["start:bad", "stop:bad", "start:good"]
    attempts = json.loads(summary_path.read_text(encoding="utf-8"))["resource_attempts"]
    assert [attempt["candidate_name"] for attempt in attempts] == ["bad", "good"]
    assert [attempt["candidate_index"] for attempt in attempts] == [0, 1]
    assert [attempt["final_state"] for attempt in attempts] == ["FAILED", "READY"]
    assert attempts[0]["cleanup_command"] == "cleanup bad"
    assert attempts[0]["cleanup_status"] == "stopped"
    assert attempts[1]["cleanup_command"] == "cleanup good"


def test_endpoint_race_records_cancelled_and_skipped_losers(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    summary_path = tmp_path / "race.json"
    candidates = {
        "first": FakeCandidate("first", events, cleanup_command="cleanup first"),
        "second": FakeCandidate("second", events, cleanup_command="cleanup second"),
        "third": FakeCandidate("third", events, cleanup_command="cleanup third"),
    }
    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        lambda config: candidates[config],
    )
    launcher = EndpointRaceLauncher(
        EndpointRaceConfig(
            name="logical",
            candidates={"first": "first", "second": "second", "third": "third"},
            max_active_candidates=2,
            launch_summary_path=str(summary_path),
        )
    )

    launcher.start()

    attempts = {
        attempt["candidate_name"]: attempt
        for attempt in json.loads(summary_path.read_text(encoding="utf-8"))["resource_attempts"]
    }
    assert attempts["first"]["final_state"] == "READY"
    assert attempts["second"]["final_state"] == "CANCELLED"
    assert attempts["third"]["final_state"] == "SKIPPED"
    assert attempts["second"]["candidate_index"] == 1
    assert attempts["second"]["cleanup_command"] == "cleanup second"
    assert attempts["third"]["candidate_index"] == 2


def test_endpoint_race_returns_without_waiting_for_blocked_loser(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    slow_started = Event()
    release_slow = Event()
    summary_path = tmp_path / "race.json"

    class SlowCandidate(FakeCandidate):
        def start(self) -> InferenceSession:
            events.append("start:slow")
            slow_started.set()
            release_slow.wait(timeout=5)
            return super().start()

    class WinnerCandidate(FakeCandidate):
        def start(self) -> InferenceSession:
            assert slow_started.wait(timeout=1)
            return super().start()

    candidates = {
        "slow": SlowCandidate("slow", events),
        "winner": WinnerCandidate("winner", events),
    }
    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        lambda config: candidates[config],
    )
    launcher = EndpointRaceLauncher(
        EndpointRaceConfig(
            name="logical",
            candidates={"slow": "slow", "winner": "winner"},
            max_active_candidates=2,
            launch_summary_path=str(summary_path),
        )
    )
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(launcher.start)

    try:
        session = future.result(timeout=1)
    finally:
        release_slow.set()
        executor.shutdown(wait=True)
        launcher.stop()

    assert session.api_base == "http://127.0.0.1/winner/v1"
    assert "stop:slow" in events


def test_endpoint_race_marks_uncancelled_loser_cleanup_not_requested(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    slow_started = Event()
    release_slow = Event()
    summary_path = tmp_path / "race.json"

    class SlowCandidate(FakeCandidate):
        def start(self) -> InferenceSession:
            events.append("start:slow")
            slow_started.set()
            release_slow.wait(timeout=5)
            return super().start()

    class WinnerCandidate(FakeCandidate):
        def start(self) -> InferenceSession:
            assert slow_started.wait(timeout=1)
            return super().start()

    candidates = {
        "slow": SlowCandidate("slow", events, cleanup_command="cleanup slow"),
        "winner": WinnerCandidate("winner", events, cleanup_command="cleanup winner"),
    }
    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        lambda config: candidates[config],
    )
    launcher = EndpointRaceLauncher(
        EndpointRaceConfig(
            name="logical",
            candidates={"slow": "slow", "winner": "winner"},
            max_active_candidates=2,
            cancel_losers=False,
            launch_summary_path=str(summary_path),
        )
    )
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(launcher.start)

    try:
        session = future.result(timeout=1)
        events_before_release = list(events)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    finally:
        release_slow.set()
        executor.shutdown(wait=True)
        launcher.stop()

    attempts = {attempt["candidate_name"]: attempt for attempt in payload["resource_attempts"]}
    assert session.api_base == "http://127.0.0.1/winner/v1"
    assert "stop:slow" not in events_before_release
    assert attempts["slow"]["final_state"] == "SUPERSEDED"
    assert attempts["slow"]["cleanup_status"] == "not_requested"


def test_endpoint_race_fails_when_loser_cleanup_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    slow_started = Event()
    release_slow = Event()
    summary_path = tmp_path / "race.json"

    class CleanupFailingSlowCandidate(FakeCandidate):
        def start(self) -> InferenceSession:
            events.append("start:slow")
            slow_started.set()
            release_slow.wait(timeout=5)
            return super().start()

        def stop(self) -> None:
            events.append("stop:slow")
            raise RuntimeError("remote cleanup failed")

    class WinnerCandidate(FakeCandidate):
        def start(self) -> InferenceSession:
            assert slow_started.wait(timeout=1)
            return super().start()

    candidates = {
        "slow": CleanupFailingSlowCandidate("slow", events),
        "winner": WinnerCandidate("winner", events),
    }
    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        lambda config: candidates[config],
    )
    launcher = EndpointRaceLauncher(
        EndpointRaceConfig(
            name="logical",
            candidates={"slow": "slow", "winner": "winner"},
            max_active_candidates=2,
            launch_summary_path=str(summary_path),
        )
    )
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(launcher.start)

    try:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            future.result(timeout=3)
    finally:
        release_slow.set()
        executor.shutdown(wait=True)
        with suppress(RuntimeError):
            launcher.stop()

    attempts = {
        attempt["candidate_name"]: attempt
        for attempt in json.loads(summary_path.read_text(encoding="utf-8"))["resource_attempts"]
    }
    assert attempts["slow"]["final_state"] == "CLEANUP_FAILED"
    assert attempts["slow"]["cleanup_status"] == "failed"
    assert "remote cleanup failed" in attempts["slow"]["cleanup_error"]


def test_endpoint_race_preserves_slurm_winner_fields(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    summary_path = tmp_path / "race.json"

    class SlurmLikeCandidate(FakeCandidate):
        def start(self) -> InferenceSession:
            self.events.append(f"start:{self.name}")
            return InferenceSession(
                name=self.name,
                api_base="http://127.0.0.1:8123/v1",
                served_model_name="served",
                backend_kind="slurm_vllm",
                logs="/remote/logs",
                local_port=8123,
                remote_port=18817,
                job_id="12345",
                node="node001",
                cleanup_command="ssh cluster scancel 12345",
                summary_path="/tmp/slurm-summary.json",
                metadata={"recommended_benchmark_max_parallel": 2},
            )

    candidates = {"slurm": SlurmLikeCandidate("slurm", events)}
    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        lambda config: candidates[config],
    )
    launcher = EndpointRaceLauncher(
        EndpointRaceConfig(
            name="logical",
            candidates={"slurm": "slurm"},
            launch_summary_path=str(summary_path),
        )
    )

    session = launcher.start()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert session.name == "logical"
    assert session.backend_kind == "endpoint_race"
    assert session.logs == "/remote/logs"
    assert session.job_id == "12345"
    assert session.node == "node001"
    assert summary["job_id"] == "12345"
    assert summary["node"] == "node001"
    assert summary["remote_log_dir"] == "/remote/logs"
    assert summary["cleanup_command"] == "ssh cluster scancel 12345"
    assert summary["winner_backend_kind"] == "slurm_vllm"
    assert summary["winner_summary_path"] == "/tmp/slurm-summary.json"
    assert summary["recommended_benchmark_max_parallel"] == 2
    assert summary["benchmark_handoff"]["recommended_max_parallel"] == 2
    assert summary["resource_attempts"][0]["job_id"] == "12345"
    assert summary["resource_attempts"][0]["logs"] == "/remote/logs"


def test_endpoint_race_summary_records_launcher_construction_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "race.json"

    def launcher_from_config(_config: object) -> object:
        raise RuntimeError("bad candidate config")

    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        launcher_from_config,
    )
    launcher = EndpointRaceLauncher(
        EndpointRaceConfig(
            name="logical",
            candidates={"bad": "bad"},
            launch_summary_path=str(summary_path),
        )
    )

    with pytest.raises(RuntimeError, match="bad candidate config"):
        launcher.start()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["lifecycle_state"] == "FAILED"
    assert payload["diagnostics"]["failure_message"] == "bad candidate config"


def test_endpoint_race_budget_records_logical_launch_peak(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    summary_path = tmp_path / "race.json"
    candidates = {"first": FakeCandidate("first", events)}
    monkeypatch.setattr(
        "remote_inference_launcher.inference_config.launcher_from_config",
        lambda config: candidates[config],
    )
    launcher = EndpointRaceLauncher(
        EndpointRaceConfig(
            name="logical",
            candidates={"first": "first"},
            resource_budget=ResourceBudget(max_concurrent_logical_launches=1),
            launch_summary_path=str(summary_path),
        )
    )

    launcher.start()
    ready_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    launcher.stop()
    released_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert ready_payload["observed_resource_usage"]["concurrent_logical_launches"] == 0
    assert ready_payload["observed_resource_usage"]["peak_concurrent_logical_launches"] == 1
    assert released_payload["observed_resource_usage"]["concurrent_logical_launches"] == 0
    assert released_payload["observed_resource_usage"]["peak_concurrent_logical_launches"] == 1


def test_endpoint_race_rejects_allocated_winner_condition() -> None:
    with pytest.raises(ValueError, match="endpoint_ready"):
        EndpointRaceLauncher(
            EndpointRaceConfig(
                candidates={"a": "a"},
                winner_condition="allocated",
            )
        )
