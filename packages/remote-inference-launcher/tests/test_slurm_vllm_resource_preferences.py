from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from threading import Event

import pytest
from slurm_vllm_fakes import TEST_MODEL, _CapacityLogLauncher, base_config

from remote_inference_launcher.config_types import CandidateRaceConfig, QueuePolicy
from remote_inference_launcher.core import InferenceSession
from remote_inference_launcher.slurm_vllm import (
    SlurmPendingTimeoutError,
    SlurmVllmConfig,
    SlurmVllmLauncher,
    _pending_timeout_seconds,
    _resource_attempt_summary,
    _resource_preference_candidates,
    validate_slurm_vllm_config,
    with_slurm_vllm_defaults,
)


class TestSlurmVllmResourcePreferences:
    def test_resource_attempt_summary_keeps_cleanup_command_for_failed_job(self) -> None:
        launcher = SlurmVllmLauncher(base_config())
        launcher._job_id = "12345"
        launcher._pending_duration_seconds = 12.5
        launcher._latest_slurm_state = "PENDING"
        launcher._latest_slurm_reason = "Resources"
        launcher._latest_slurm_diagnostics = "scontrol details"

        attempt = _resource_attempt_summary(
            index=0,
            name="preferred",
            config=launcher.config,
            state="FAILED",
            error=TimeoutError("pending timeout"),
            launcher=launcher,
        )

        assert attempt["job_id"] == "12345"
        assert attempt["cleanup_command"] == "ssh cluster 'scancel 12345'"
        assert attempt["cleanup_status"] == "command_available"
        assert attempt["pending_duration_seconds"] == 12.5
        assert attempt["latest_slurm_state"] == "PENDING"
        assert attempt["latest_slurm_reason"] == "Resources"
        assert attempt["latest_slurm_diagnostics"] == "scontrol details"

    def test_resource_preferences_use_only_listed_candidates_by_default(self) -> None:
        config = base_config(
            name="logical",
            ssh_target="cluster-a",
            partition="preferred",
            resource_preferences=(
                {"name": "fallback", "ssh_target": "cluster-b", "partition": "fallback"},
            ),
        )

        candidates = _resource_preference_candidates(config)

        assert [name for name, _candidate in candidates] == ["fallback"]
        assert candidates[0][1].ssh_target == "cluster-b"

    def test_resource_preferences_include_base_config_as_final_unique_candidate(self) -> None:
        config = base_config(
            name="logical",
            ssh_target="cluster-a",
            partition="preferred",
            include_base_resource_candidate=True,
            resource_preferences=(
                {"name": "fallback", "ssh_target": "cluster-b", "partition": "fallback"},
            ),
        )

        candidates = _resource_preference_candidates(config)

        assert [name for name, _candidate in candidates] == ["fallback", "logical-base"]
        assert [candidate.name for _name, candidate in candidates] == ["fallback", "logical-base"]
        assert candidates[0][1].ssh_target == "cluster-b"
        assert candidates[1][1].ssh_target == "cluster-a"
        assert candidates[1][1].partition == "preferred"
        assert candidates[0][1].job_name == ""
        assert candidates[1][1].job_name == ""
        assert candidates[0][1].out_dir == ""
        assert candidates[1][1].out_dir == ""

    def test_resource_preferences_generate_distinct_candidate_job_names(self) -> None:
        config = base_config(
            name="logical",
            ssh_target="cluster-a",
            partition="preferred",
            resource_preferences=(
                {"name": "first", "partition": "batch-a"},
                {"name": "second", "partition": "batch-b"},
            ),
        )

        candidates = _resource_preference_candidates(config)
        first = with_slurm_vllm_defaults(candidates[0][1], run_id="20260530-aaaaaa")
        second = with_slurm_vllm_defaults(candidates[1][1], run_id="20260530-bbbbbb")

        assert first.name == "first"
        assert second.name == "second"
        assert first.job_name != second.job_name
        assert first.job_name.startswith("ril-first-")
        assert second.job_name.startswith("ril-second-")

    def test_resource_preferences_do_not_duplicate_equivalent_base_config(self) -> None:
        config = base_config(
            name="logical",
            ssh_target="cluster-a",
            partition="preferred",
            include_base_resource_candidate=True,
            resource_preferences=(
                {"name": "preferred", "ssh_target": "cluster-a", "partition": "preferred"},
            ),
        )

        candidates = _resource_preference_candidates(config)

        assert [name for name, _candidate in candidates] == ["preferred"]

    def test_resource_preferences_reject_duplicate_candidate_names(self) -> None:
        config = with_slurm_vllm_defaults(
            base_config(
                resource_preferences=(
                    {"name": "preferred", "partition": "batch-2gpu"},
                    {"name": "preferred", "partition": "batch-4gpu"},
                )
            )
        )

        with pytest.raises(ValueError, match="duplicate candidate name"):
            validate_slurm_vllm_config(config)

    @pytest.mark.parametrize("field_name", ("candidate_race", "readiness", "queue_policy"))
    def test_resource_preferences_reject_parent_only_candidate_fields(
        self,
        field_name: str,
    ) -> None:
        config = base_config(
            resource_preferences=({"name": "preferred", "partition": "batch-a", field_name: {}},)
        )

        with pytest.raises(ValueError, match=f"unsupported fields: {field_name}"):
            _resource_preference_candidates(config)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        (
            ("job_name", "shared-job"),
            ("out_dir", "/remote/shared"),
            ("local_port", 8123),
            ("remote_port", 18817),
            ("head_node_port", 18818),
        ),
    )
    def test_resource_preferences_reject_parent_level_identity(
        self,
        field_name: str,
        value: object,
    ) -> None:
        config = base_config(
            resource_preferences=({"name": "preferred", "partition": "batch-2gpu"},),
            **{field_name: value},
        )

        with pytest.raises(ValueError, match="parent-level"):
            with_slurm_vllm_defaults(config)

    def test_resource_preferences_reject_duplicate_explicit_candidate_identity(self) -> None:
        config = base_config(
            resource_preferences=(
                {"name": "first", "job_name": "shared"},
                {"name": "second", "job_name": "shared"},
            )
        )

        with pytest.raises(ValueError, match="duplicate explicit candidate job_name"):
            _resource_preference_candidates(config)

    def test_candidate_race_rejects_detached_losers(self) -> None:
        config = with_slurm_vllm_defaults(
            base_config(
                resource_preferences=(
                    {"name": "first", "partition": "batch-a"},
                    {"name": "second", "partition": "batch-b"},
                ),
                candidate_race=CandidateRaceConfig(
                    enabled=True,
                    cancel_losers=False,
                ),
            )
        )

        with pytest.raises(ValueError, match="cancel_losers"):
            validate_slurm_vllm_config(config)

    def test_queue_policy_pending_timeout_takes_precedence(self) -> None:
        config = validate_slurm_vllm_config(
            with_slurm_vllm_defaults(
                base_config(
                    queue_timeout_seconds=7200,
                    queue_policy=QueuePolicy(max_pending_seconds=300),
                )
            )
        )

        assert _pending_timeout_seconds(config) == 300

    def test_fallback_on_pending_requires_resource_preferences(self) -> None:
        config = with_slurm_vllm_defaults(
            base_config(
                queue_policy=QueuePolicy(
                    max_pending_seconds=300,
                    fallback_on_pending=True,
                ),
            )
        )

        with pytest.raises(ValueError, match="requires resource_preferences"):
            validate_slurm_vllm_config(config)

    def test_resource_preferences_do_not_fallback_on_pending_without_policy(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        summary_path = tmp_path / "summary.json"
        parent = SlurmVllmLauncher(
            base_config(
                launch_summary_path=str(summary_path),
                resource_preferences=(
                    {"name": "first", "partition": "batch-a"},
                    {"name": "second", "partition": "batch-b"},
                ),
                queue_policy=QueuePolicy(
                    max_pending_seconds=300,
                    fallback_on_pending=False,
                ),
            )
        )
        started: list[str] = []

        class FakeCandidateLauncher:
            def __init__(
                self,
                config: SlurmVllmConfig,
                *,
                _candidate_mode: bool = False,
            ) -> None:
                del _candidate_mode
                self.config = config
                self._job_id = f"job-{config.name}"
                self._pending_duration_seconds = 300.0
                self._latest_slurm_state = "PENDING"
                self._latest_slurm_reason = "Resources"
                self._latest_slurm_diagnostics = "pending diagnostics"

            def start(self) -> InferenceSession:
                started.append(self.config.name)
                if self.config.name == "first":
                    raise SlurmPendingTimeoutError("pending timeout")
                raise AssertionError("pending fallback should not launch the next candidate")

            def stop(self) -> None:
                return None

            def _safe_remote_log_tail(self, *, lines: int = 400) -> str:
                del lines
                return ""

        monkeypatch.setattr(
            "remote_inference_launcher.slurm_vllm.SlurmVllmLauncher",
            FakeCandidateLauncher,
        )

        with pytest.raises(SlurmPendingTimeoutError, match="pending timeout"):
            parent._start_resource_preferences()

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        attempts = payload["resource_attempts"]
        assert started == ["first"]
        assert payload["lifecycle_state"] == "FAILED"
        assert len(attempts) == 1
        assert attempts[0]["candidate_name"] == "first"
        assert attempts[0]["failure_code"] == "slurm_pending_timeout"

    def test_resource_preferences_do_not_fallback_on_generic_start_failure(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        summary_path = tmp_path / "summary.json"
        parent = SlurmVllmLauncher(
            base_config(
                launch_summary_path=str(summary_path),
                resource_preferences=(
                    {"name": "first", "partition": "batch-a"},
                    {"name": "second", "partition": "batch-b"},
                ),
                queue_policy=QueuePolicy(
                    max_pending_seconds=300,
                    fallback_on_pending=True,
                ),
            )
        )
        started: list[str] = []

        class FakeCandidateLauncher:
            def __init__(
                self,
                config: SlurmVllmConfig,
                *,
                _candidate_mode: bool = False,
            ) -> None:
                del _candidate_mode
                self.config = config
                self._job_id = ""
                self._pending_duration_seconds = None
                self._latest_slurm_state = ""
                self._latest_slurm_reason = ""
                self._latest_slurm_diagnostics = ""

            def start(self) -> InferenceSession:
                started.append(self.config.name)
                if self.config.name == "first":
                    raise RuntimeError("bad vLLM environment")
                raise AssertionError("generic failures must not launch fallback candidates")

            def stop(self) -> None:
                return None

            def _safe_remote_log_tail(self, *, lines: int = 400) -> str:
                del lines
                return ""

        monkeypatch.setattr(
            "remote_inference_launcher.slurm_vllm.SlurmVllmLauncher",
            FakeCandidateLauncher,
        )

        with pytest.raises(RuntimeError, match="bad vLLM environment"):
            parent._start_resource_preferences()

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        attempts = payload["resource_attempts"]
        assert started == ["first"]
        assert payload["lifecycle_state"] == "FAILED"
        assert len(attempts) == 1
        assert attempts[0]["candidate_name"] == "first"
        assert attempts[0]["failure_code"] == "slurm_cancelled_or_failed"

    def test_resource_preference_winner_summary_keeps_handoff_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        summary_path = tmp_path / "summary.json"
        parent = SlurmVllmLauncher(base_config(launch_summary_path=str(summary_path)))
        child = _CapacityLogLauncher(base_config(max_model_len=8192, max_num_seqs=4))
        parent._delegate_launcher = child

        parent._adopt_resource_preference_winner(
            InferenceSession(
                name="candidate",
                api_base="http://127.0.0.1:8123/v1",
                served_model_name=TEST_MODEL,
                model=TEST_MODEL,
                local_port=8123,
                remote_port=18817,
                logs="/remote/logs",
                job_id="12345",
                node="node001",
                cleanup_command="ssh cluster 'scancel 12345'",
                backend_kind="slurm_vllm",
                metadata={
                    "remote_state_path": "/remote/state.json",
                    "readiness": {
                        "models_endpoint_ok": True,
                        "smoke_test_ok": True,
                        "smoke_test_kind": "chat_completion",
                    },
                },
            ),
            attempts=[],
            winner_name="preferred",
        )

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["remote_state_path"] == "/remote/state.json"
        assert payload["readiness"]["smoke_test_ok"] is True
        assert payload["vllm_max_concurrency"] == 3.5
        assert payload["recommended_benchmark_max_parallel"] == 3

    def test_resource_preference_race_returns_without_waiting_for_blocked_loser(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        slow_started = Event()
        release_slow = Event()
        stopped: list[str] = []
        summary_path = tmp_path / "summary.json"
        parent = SlurmVllmLauncher(
            base_config(
                launch_summary_path=str(summary_path),
                candidate_race=CandidateRaceConfig(enabled=True, max_active_candidates=2),
            )
        )

        class FakeCandidateLauncher:
            def __init__(
                self,
                config: SlurmVllmConfig,
                *,
                _candidate_mode: bool = False,
            ) -> None:
                del _candidate_mode
                self.config = config
                self._job_id = f"job-{config.name}"
                self._pending_duration_seconds = None
                self._latest_slurm_state = "RUNNING"
                self._latest_slurm_reason = ""

            def start(self) -> InferenceSession:
                if self.config.name == "slow":
                    slow_started.set()
                    release_slow.wait(timeout=5)
                else:
                    assert slow_started.wait(timeout=1)
                return InferenceSession(
                    name=self.config.name,
                    api_base=f"http://127.0.0.1/{self.config.name}/v1",
                    served_model_name=TEST_MODEL,
                    model=TEST_MODEL,
                    logs=f"/remote/{self.config.name}",
                    job_id=self._job_id,
                    node="node001",
                    cleanup_command=f"cleanup {self.config.name}",
                    backend_kind="slurm_vllm",
                    metadata={
                        "remote_state_path": f"/remote/{self.config.name}/state.json",
                        "readiness": {
                            "models_endpoint_ok": True,
                            "smoke_test_ok": True,
                            "smoke_test_kind": "chat_completion",
                        },
                    },
                )

            def stop(self) -> None:
                stopped.append(self.config.name)

            def _safe_remote_log_tail(self, *, lines: int = 400) -> str:
                del lines
                return ""

        monkeypatch.setattr(
            "remote_inference_launcher.slurm_vllm.SlurmVllmLauncher",
            FakeCandidateLauncher,
        )
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            parent._race_resource_preferences,
            (
                ("slow", base_config(name="slow")),
                ("winner", base_config(name="winner")),
            ),
        )

        try:
            session = future.result(timeout=1)
        finally:
            release_slow.set()
            executor.shutdown(wait=True)
            parent.stop()

        assert session.api_base == "http://127.0.0.1/winner/v1"
        assert "slow" in stopped
        assert "winner" in stopped

    def test_resource_preference_race_fails_when_loser_cleanup_fails(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        slow_started = Event()
        release_slow = Event()
        summary_path = tmp_path / "summary.json"
        parent = SlurmVllmLauncher(
            base_config(
                launch_summary_path=str(summary_path),
                candidate_race=CandidateRaceConfig(enabled=True, max_active_candidates=2),
            )
        )

        class FakeCandidateLauncher:
            def __init__(
                self,
                config: SlurmVllmConfig,
                *,
                _candidate_mode: bool = False,
            ) -> None:
                del _candidate_mode
                self.config = config
                self._job_id = f"job-{config.name}"
                self._pending_duration_seconds = None
                self._latest_slurm_state = "RUNNING"
                self._latest_slurm_reason = ""
                self._latest_slurm_diagnostics = ""

            def start(self) -> InferenceSession:
                if self.config.name == "slow":
                    slow_started.set()
                    release_slow.wait(timeout=5)
                else:
                    assert slow_started.wait(timeout=1)
                return InferenceSession(
                    name=self.config.name,
                    api_base=f"http://127.0.0.1/{self.config.name}/v1",
                    served_model_name=TEST_MODEL,
                    model=TEST_MODEL,
                    logs=f"/remote/{self.config.name}",
                    job_id=self._job_id,
                    node="node001",
                    cleanup_command=f"cleanup {self.config.name}",
                    backend_kind="slurm_vllm",
                    metadata={
                        "remote_state_path": f"/remote/{self.config.name}/state.json",
                        "readiness": {
                            "models_endpoint_ok": True,
                            "smoke_test_ok": True,
                            "smoke_test_kind": "chat_completion",
                        },
                    },
                )

            def stop(self) -> None:
                if self.config.name == "slow":
                    raise RuntimeError("scancel failed")

            def _safe_remote_log_tail(self, *, lines: int = 400) -> str:
                del lines
                return ""

        monkeypatch.setattr(
            "remote_inference_launcher.slurm_vllm.SlurmVllmLauncher",
            FakeCandidateLauncher,
        )
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            parent._race_resource_preferences,
            (
                ("slow", base_config(name="slow")),
                ("winner", base_config(name="winner")),
            ),
        )

        try:
            with pytest.raises(RuntimeError, match="cleanup failed"):
                future.result(timeout=3)
        finally:
            release_slow.set()
            executor.shutdown(wait=True)
            with suppress(RuntimeError):
                parent.stop()

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        attempts = {attempt["candidate_name"]: attempt for attempt in payload["resource_attempts"]}
        assert payload["lifecycle_state"] == "FAILED"
        assert attempts["slow"]["final_state"] == "CLEANUP_FAILED"
        assert attempts["slow"]["cleanup_status"] == "failed"
        assert "scancel failed" in attempts["slow"]["cleanup_error"]
