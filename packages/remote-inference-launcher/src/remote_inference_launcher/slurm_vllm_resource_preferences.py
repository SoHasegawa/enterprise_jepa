"""Resource preference expansion and racing for Slurm vLLM launches."""

from __future__ import annotations

import shlex
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from typing import Any

from remote_inference_launcher.config_types import CandidateRaceConfig
from remote_inference_launcher.core import InferenceSession
from remote_inference_launcher.diagnostics import (
    CapacityInfo,
    classify_failure,
    parse_vllm_capacity,
    sanitized_excerpt,
)
from remote_inference_launcher.slurm import scancel_job_command
from remote_inference_launcher.slurm_vllm_config import (
    _RESOURCE_PREFERENCE_OVERRIDE_FIELDS,
    SlurmPendingTimeoutError,
    SlurmVllmConfig,
    _resource_preference_name,
    _validate_resource_preference_identity,
)
from remote_inference_launcher.summaries import endpoint_summary


class SlurmVllmResourcePreferencesMixin:
    def _start_resource_preferences(self) -> InferenceSession:
        self._summary.reserve()
        candidates = slurm_resource_preference_candidates(self._input_config)
        if self.config.candidate_race.enabled:
            return self._race_resource_preferences(candidates)
        attempts: list[dict[str, object]] = []
        last_error: BaseException | None = None
        for index, (candidate_name, candidate_config) in enumerate(candidates):
            launcher = _new_candidate_launcher(candidate_config)
            self._candidate_launchers[candidate_name] = launcher
            try:
                session = launcher.start()
            except BaseException as error:
                last_error = error
                attempts.append(
                    _resource_attempt_summary(
                        index=index,
                        name=candidate_name,
                        config=candidate_config,
                        state="FAILED",
                        error=error,
                        launcher=launcher,
                    )
                )
                if _should_try_next_resource_candidate(error, self.config):
                    continue
                message = str(error)
                self._write_resource_preferences_failed_summary(
                    message,
                    attempts,
                    failure_code=_failure_code_for_exception(
                        self.config,
                        error,
                        text=message,
                    ),
                )
                raise
            attempts.append(
                _resource_attempt_summary(
                    index=index,
                    name=candidate_name,
                    config=candidate_config,
                    state="READY",
                    session=session,
                    winner=True,
                    launcher=launcher,
                )
            )
            self._delegate_launcher = launcher
            self._winner_candidate_name = candidate_name
            return self._adopt_resource_preference_winner(
                session,
                attempts=attempts,
                winner_name=candidate_name,
            )
        message = (
            str(last_error) if last_error is not None else "No resource candidate was launched."
        )
        self._write_resource_preferences_failed_summary(
            message,
            attempts,
            failure_code=(
                _failure_code_for_exception(self.config, last_error, text=message)
                if last_error is not None
                else None
            ),
        )
        raise RuntimeError(f"All Slurm vLLM resource candidates failed: {message}")

    def _race_resource_preferences(
        self,
        candidates: tuple[tuple[str, SlurmVllmConfig], ...],
    ) -> InferenceSession:
        attempts: list[dict[str, object]] = []
        launchers: dict[str, Any] = {}
        candidate_indices = {name: index for index, (name, _config) in enumerate(candidates)}
        pending = list(enumerate(candidates))
        active: dict[Future[InferenceSession], tuple[int, str]] = {}
        executor = ThreadPoolExecutor(
            max_workers=self.config.candidate_race.max_active_candidates,
            thread_name_prefix="remote-inference-slurm-race",
        )
        shutdown_wait = True
        try:
            _submit_slurm_candidate_work(
                executor,
                pending,
                active,
                launchers,
                max_active=self.config.candidate_race.max_active_candidates,
                launch_stagger_seconds=self.config.candidate_race.launch_stagger_seconds,
            )
            self._candidate_launchers = launchers
            while active:
                done, _not_done = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    index, name = active.pop(future)
                    try:
                        session = future.result()
                    except BaseException as error:
                        candidate_config = launchers[name].config
                        attempts.append(
                            _resource_attempt_summary(
                                index=index,
                                name=name,
                                config=candidate_config,
                                state="FAILED",
                                error=error,
                                launcher=launchers[name],
                            )
                        )
                        continue
                    attempts.append(
                        _resource_attempt_summary(
                            index=index,
                            name=name,
                            config=launchers[name].config,
                            state="READY",
                            session=session,
                            winner=True,
                            launcher=launchers[name],
                        )
                    )
                    self._candidate_launchers = launchers
                    cleanup_errors: dict[str, str] = {}
                    if self.config.candidate_race.cancel_losers:
                        cleanup_errors = _stop_slurm_losers(launchers, winner_name=name)
                    recorded_names = {str(attempt.get("candidate_name")) for attempt in attempts}
                    for loser_name, loser in sorted(launchers.items()):
                        if loser_name == name or loser_name in recorded_names:
                            continue
                        cleanup_error = cleanup_errors.get(loser_name)
                        attempts.append(
                            _resource_attempt_summary(
                                index=candidate_indices.get(loser_name, -1),
                                name=loser_name,
                                config=loser.config,
                                state="CLEANUP_FAILED"
                                if cleanup_error
                                else "CANCELLED"
                                if self.config.candidate_race.cancel_losers
                                else "SUPERSEDED",
                                launcher=loser,
                                cleanup_error=cleanup_error,
                                cleanup_status=(
                                    "failed"
                                    if cleanup_error
                                    else "stopped"
                                    if self.config.candidate_race.cancel_losers
                                    else "not_requested"
                                ),
                            )
                        )
                    self._delegate_launcher = launchers[name]
                    self._winner_candidate_name = name
                    if cleanup_errors:
                        message = _cleanup_failure_message(cleanup_errors)
                        adopted = self._adopt_resource_preference_winner(
                            session,
                            attempts=attempts,
                            winner_name=name,
                        )
                        del adopted
                        self._mark_last_summary_failed(message)
                        shutdown_wait = False
                        try:
                            self.stop()
                        except RuntimeError as stop_error:
                            raise RuntimeError(
                                "Slurm vLLM resource candidate cleanup failed and final "
                                f"cleanup also failed: {stop_error}. Original failure: {message}"
                            ) from stop_error
                        raise RuntimeError(message)
                    adopted = self._adopt_resource_preference_winner(
                        session,
                        attempts=attempts,
                        winner_name=name,
                    )
                    shutdown_wait = False
                    return adopted
                _submit_slurm_candidate_work(
                    executor,
                    pending,
                    active,
                    launchers,
                    max_active=self.config.candidate_race.max_active_candidates,
                    launch_stagger_seconds=self.config.candidate_race.launch_stagger_seconds,
                )
                self._candidate_launchers = launchers
        finally:
            executor.shutdown(wait=shutdown_wait, cancel_futures=True)
        message = "No Slurm vLLM resource candidate became ready."
        self._summary.write(
            endpoint_summary(
                endpoint_name=self.config.name,
                backend_kind="slurm_vllm",
                lifecycle_state="FAILED",
                run_id=self._run_id,
                served_model_name=self.config.served_model_name,
                model=self.config.model,
                api_key_set=bool(self.config.api_key),
                failure_code=_classify_failure_if_enabled(
                    self.config,
                    message,
                    fallback="slurm_cancelled_or_failed",
                ),
                failure_message=sanitized_excerpt(message),
                resource_attempts=attempts,
            )
        )
        raise RuntimeError(message)

    def _adopt_resource_preference_winner(
        self,
        session: InferenceSession,
        *,
        attempts: list[dict[str, object]],
        winner_name: str,
    ) -> InferenceSession:
        self._job_id = session.job_id
        self._remote_node = session.node
        self._remote_out_dir = session.logs
        self._selected_local_port = session.local_port
        self._selected_remote_port = session.remote_port
        metadata = dict(session.metadata)
        metadata["resource_attempts"] = attempts
        metadata["winner_candidate"] = winner_name
        adopted = replace(
            session,
            name=self.config.name,
            run_id=self._run_id,
            summary_path=str(self._summary.path),
            metadata=metadata,
        )
        winner_config = (
            self._delegate_launcher.config if self._delegate_launcher is not None else self.config
        )
        winner_log_text = (
            self._delegate_launcher._safe_remote_log_tail(lines=400)
            if self._delegate_launcher is not None
            else ""
        )
        capacity = _parse_capacity_if_enabled(
            winner_config,
            winner_log_text,
            max_model_len=winner_config.max_model_len,
            max_num_seqs=winner_config.max_num_seqs,
        )
        payload = endpoint_summary(
            endpoint_name=self.config.name,
            backend_kind="slurm_vllm",
            lifecycle_state="READY",
            run_id=self._run_id,
            api_base=adopted.api_base,
            served_model_name=adopted.served_model_name,
            model=adopted.model,
            api_key_set=bool(adopted.api_key),
            local_port=adopted.local_port,
            remote_port=adopted.remote_port,
            job_id=adopted.job_id,
            node=adopted.node,
            cleanup_command=adopted.cleanup_command,
            remote_log_dir=adopted.logs,
            remote_state_path=str(metadata.get("remote_state_path") or "") or None,
            capacity=capacity,
            resource_attempts=attempts,
            extra={"winner_candidate": winner_name},
        )
        if isinstance(metadata.get("readiness"), dict):
            payload["readiness"] = metadata["readiness"]
        self._last_summary = payload
        self._summary.write(payload)
        return adopted

    def _write_resource_preferences_failed_summary(
        self,
        message: str,
        attempts: list[dict[str, object]],
        *,
        failure_code: str | None = None,
    ) -> None:
        self._summary.write(
            endpoint_summary(
                endpoint_name=self.config.name,
                backend_kind="slurm_vllm",
                lifecycle_state="FAILED",
                run_id=self._run_id,
                served_model_name=self.config.served_model_name,
                model=self.config.model,
                api_key_set=bool(self.config.api_key),
                failure_code=failure_code
                or _classify_failure_if_enabled(
                    self.config,
                    message,
                    fallback="slurm_cancelled_or_failed",
                ),
                failure_message=sanitized_excerpt(message),
                resource_attempts=attempts,
            )
        )


def _new_candidate_launcher(config: SlurmVllmConfig) -> Any:
    from remote_inference_launcher import slurm_vllm

    return slurm_vllm.SlurmVllmLauncher(config, _candidate_mode=True)


def _should_try_next_resource_candidate(
    error: BaseException,
    config: SlurmVllmConfig,
) -> bool:
    return isinstance(error, SlurmPendingTimeoutError) and config.queue_policy.fallback_on_pending


def _remote_scancel_command(ssh_target: str, job_id: str) -> str:
    if not job_id:
        return ""
    return f"ssh {shlex.quote(ssh_target)} {shlex.quote(scancel_job_command(job_id))}"


def _failure_fallback_for_exception(error: BaseException) -> str:
    if isinstance(error, SlurmPendingTimeoutError):
        return "slurm_pending_timeout"
    if isinstance(error, TimeoutError):
        message = str(error).lower()
        if "port state" in message:
            return "remote_port_not_written"
        if "readiness" in message:
            return "readiness_smoke_failed"
    return "slurm_cancelled_or_failed"


def _failure_code_for_exception(
    config: SlurmVllmConfig,
    error: BaseException,
    *,
    text: str | None = None,
) -> str | None:
    if isinstance(error, SlurmPendingTimeoutError):
        return "slurm_pending_timeout"
    return _classify_failure_if_enabled(
        config,
        str(error) if text is None else text,
        fallback=_failure_fallback_for_exception(error),
    )


def _parse_capacity_if_enabled(
    config: SlurmVllmConfig,
    log_text: str,
    *,
    max_model_len: int | None,
    max_num_seqs: int | None,
) -> CapacityInfo | None:
    if not config.diagnostics.collect_vllm_capacity:
        return None
    return parse_vllm_capacity(
        log_text,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )


def _classify_failure_if_enabled(
    config: SlurmVllmConfig,
    text: str,
    *,
    fallback: str | None = None,
) -> str | None:
    if not config.diagnostics.classify_startup_failures:
        return fallback or "slurm_cancelled_or_failed"
    return classify_failure(text, fallback=fallback)


def slurm_resource_preference_candidates(
    base_config: SlurmVllmConfig,
) -> tuple[tuple[str, SlurmVllmConfig], ...]:
    """Return the concrete Slurm resource candidates launch may attempt."""

    _validate_resource_preference_identity(base_config)
    candidates: list[tuple[str, SlurmVllmConfig]] = []
    seen_candidate_names: set[str] = set()
    base_candidate = replace(
        base_config,
        name=f"{base_config.name}-base",
        job_name="",
        out_dir="",
        local_port=None,
        remote_port=None,
        head_node_port=None,
        resource_preferences=(),
        include_base_resource_candidate=False,
        candidate_race=CandidateRaceConfig(),
        launch_summary_path="",
        overwrite_launch_summary=False,
    )
    for index, preference in enumerate(base_config.resource_preferences):
        unknown = sorted(set(preference) - _RESOURCE_PREFERENCE_OVERRIDE_FIELDS)
        if unknown:
            raise ValueError(
                f"Slurm vLLM resource preference contains unsupported fields: {', '.join(unknown)}."
            )
        candidate_name = _resource_preference_name(preference, index=index)
        if candidate_name in seen_candidate_names:
            raise ValueError(
                f"Slurm vLLM resource_preferences contain duplicate candidate name: "
                f"{candidate_name}."
            )
        seen_candidate_names.add(candidate_name)
        overrides = dict(preference)
        overrides.pop("name", None)
        candidate_values = {
            "name": candidate_name,
            "job_name": "",
            "out_dir": "",
            "local_port": None,
            "remote_port": None,
            "head_node_port": None,
            **overrides,
            "resource_preferences": (),
            "include_base_resource_candidate": False,
            "candidate_race": CandidateRaceConfig(),
            "launch_summary_path": "",
            "overwrite_launch_summary": False,
        }
        candidate = replace(base_config, **candidate_values)
        candidates.append((candidate_name, candidate))
    base_candidate_name = f"{base_config.name}-base"
    if base_config.include_base_resource_candidate and not any(
        _equivalent_resource_candidate(base_candidate, candidate) for _name, candidate in candidates
    ):
        if base_candidate_name in seen_candidate_names:
            raise ValueError(
                f"Slurm vLLM resource_preferences contain duplicate candidate name: "
                f"{base_candidate_name}."
            )
        candidates.append((base_candidate_name, base_candidate))
    _validate_unique_candidate_identity(candidates)
    return tuple(candidates)


def _resource_preference_candidates(
    base_config: SlurmVllmConfig,
) -> tuple[tuple[str, SlurmVllmConfig], ...]:
    return slurm_resource_preference_candidates(base_config)


def _validate_unique_candidate_identity(
    candidates: list[tuple[str, SlurmVllmConfig]],
) -> None:
    seen: dict[tuple[str, str, object], str] = {}
    for candidate_name, candidate in candidates:
        for key in _candidate_identity_keys(candidate):
            prior = seen.get(key)
            if prior is not None:
                field_name, _scope, value = key
                raise ValueError(
                    "Slurm vLLM resource_preferences contain duplicate explicit candidate "
                    f"{field_name}: {value!r} is used by {prior!r} and {candidate_name!r}."
                )
            seen[key] = candidate_name


def _candidate_identity_keys(config: SlurmVllmConfig) -> tuple[tuple[str, str, object], ...]:
    keys: list[tuple[str, str, object]] = []
    if config.local_port is not None:
        keys.append(("local_port", config.local_bind_host, config.local_port))
    for field_name in ("remote_port", "head_node_port", "job_name", "out_dir"):
        value = getattr(config, field_name)
        if value not in {"", None}:
            keys.append((field_name, config.ssh_target, value))
    return tuple(keys)


def _equivalent_resource_candidate(first: SlurmVllmConfig, second: SlurmVllmConfig) -> bool:
    ignored_fields = {
        "name",
        "resource_preferences",
        "include_base_resource_candidate",
        "candidate_race",
        "launch_summary_path",
        "overwrite_launch_summary",
    }
    return all(
        getattr(first, field.name) == getattr(second, field.name)
        for field in dataclass_fields(SlurmVllmConfig)
        if field.name not in ignored_fields
    )


def _submit_slurm_candidate_work(
    executor: ThreadPoolExecutor,
    pending: list[tuple[int, tuple[str, SlurmVllmConfig]]],
    active: dict[Future[InferenceSession], tuple[int, str]],
    launchers: dict[str, Any],
    *,
    max_active: int,
    launch_stagger_seconds: float = 0.0,
) -> None:
    while pending and len(active) < max(1, max_active):
        index, (name, config) = pending.pop(0)
        launcher = _new_candidate_launcher(config)
        launchers[name] = launcher
        if active and launch_stagger_seconds:
            time.sleep(launch_stagger_seconds)
        context = copy_context()
        active[executor.submit(context.run, launcher.start)] = (index, name)


def _stop_slurm_losers(
    launchers: Mapping[str, Any],
    *,
    winner_name: str,
) -> dict[str, str]:
    cleanup_errors: dict[str, str] = {}
    for loser_name, loser in launchers.items():
        if loser_name == winner_name:
            continue
        try:
            loser.stop()
        except Exception as error:
            cleanup_errors[loser_name] = str(error)
    return cleanup_errors


def _cleanup_failure_message(cleanup_errors: Mapping[str, str]) -> str:
    failures = ", ".join(
        f"{name}: {sanitized_excerpt(error, max_chars=300)}"
        for name, error in sorted(cleanup_errors.items())
    )
    return f"Slurm vLLM cleanup failed for non-winning candidate(s): {failures}"


def _resource_attempt_summary(
    *,
    index: int,
    name: str,
    config: SlurmVllmConfig,
    state: str,
    session: InferenceSession | None = None,
    error: BaseException | None = None,
    winner: bool = False,
    launcher: Any | None = None,
    cleanup_error: str | None = None,
    cleanup_status: str | None = None,
) -> dict[str, object]:
    message = str(error) if error is not None else ""
    diagnostic_message = message or str(cleanup_error or "")
    job_id = session.job_id if session is not None else launcher._job_id if launcher else ""
    pending_duration = launcher._pending_duration_seconds if launcher is not None else None
    latest_state = getattr(launcher, "_latest_slurm_state", "") if launcher is not None else ""
    latest_reason = getattr(launcher, "_latest_slurm_reason", "") if launcher is not None else ""
    latest_diagnostics = (
        getattr(launcher, "_latest_slurm_diagnostics", "") if launcher is not None else ""
    )
    cleanup_command = (
        session.cleanup_command
        if session is not None
        else _remote_scancel_command(config.ssh_target, job_id)
        if job_id
        else None
    )
    return {
        "candidate_index": index,
        "candidate_name": name,
        "backend_kind": "slurm_vllm",
        "ssh_target": config.ssh_target,
        "partition": config.partition,
        "job_id": job_id or None,
        "final_state": state,
        "winner": winner,
        "failure_code": (
            "cleanup_failed"
            if cleanup_error
            else _failure_code_for_exception(
                config,
                error,
                text=message,
            )
            if error is not None
            else None
        ),
        "failure_message": sanitized_excerpt(diagnostic_message) if diagnostic_message else None,
        "cleanup_command": cleanup_command,
        "cleanup_status": cleanup_status
        or (
            "failed"
            if cleanup_error
            else "command_available"
            if cleanup_command
            else "not_applicable"
        ),
        "cleanup_error": sanitized_excerpt(cleanup_error) if cleanup_error else None,
        "pending_duration_seconds": pending_duration,
        "latest_slurm_state": latest_state or None,
        "latest_slurm_reason": latest_reason or None,
        "latest_slurm_diagnostics": latest_diagnostics or None,
    }
