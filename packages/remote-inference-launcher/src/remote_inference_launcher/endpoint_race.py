"""First-ready-wins endpoint candidate racing."""

from __future__ import annotations

import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass, field, replace

from remote_inference_launcher.config_types import ResourceBudget
from remote_inference_launcher.core import InferenceLauncher, InferenceSession
from remote_inference_launcher.diagnostics import classify_failure, sanitized_excerpt
from remote_inference_launcher.resource_budget import (
    ResourceBudgetManager,
    ResourceBudgetReservation,
    resource_budget_scope,
)
from remote_inference_launcher.summaries import SummaryWriter, endpoint_summary, new_run_id


@dataclass(frozen=True)
class EndpointRaceConfig:
    """Configuration for one logical endpoint backed by multiple candidates."""

    name: str = "default"
    candidates: Mapping[str, object] = field(default_factory=dict)
    max_active_candidates: int = 1
    launch_stagger_seconds: float = 0.0
    winner_condition: str = "endpoint_ready"
    cancel_losers: bool = True
    resource_budget: ResourceBudget | None = None
    launch_summary_path: str = ""
    overwrite_launch_summary: bool = False


class EndpointRaceLauncher:
    """Launch candidates and return the first endpoint that passes readiness."""

    owns_resources = True

    def __init__(self, config: EndpointRaceConfig, *, run_id: str | None = None) -> None:
        self.config = validate_endpoint_race_config(config)
        self._launchers: dict[str, InferenceLauncher] = {}
        self._winner_name = ""
        self._winner_session: InferenceSession | None = None
        self._run_id = run_id or new_run_id()
        self._summary = SummaryWriter(
            self.config.launch_summary_path,
            run_id=self._run_id,
            endpoint_name=self.config.name,
            backend_kind="endpoint_race",
            overwrite=self.config.overwrite_launch_summary,
        )
        self._attempts: list[dict[str, object]] = []
        self._last_lifecycle_state = ""
        self._candidate_indices = {name: index for index, name in enumerate(self.config.candidates)}
        self._budget_manager = (
            ResourceBudgetManager(self.config.resource_budget)
            if self.config.resource_budget is not None
            else None
        )
        self._logical_budget_token: ResourceBudgetReservation | None = None

    def start(self) -> InferenceSession:  # noqa: C901
        """Return the first ready candidate as this logical endpoint."""

        from remote_inference_launcher.inference_config import launcher_from_config

        self._summary.reserve()
        self._attempts = []
        self._winner_name = ""
        self._winner_session = None
        try:
            self._acquire_logical_budget()
            self._launchers = {
                name: launcher_from_config(config)
                for name, config in self.config.candidates.items()
            }
        except BaseException as error:
            self._release_logical_budget()
            self._write_summary("FAILED", failure_message=str(error))
            raise
        pending_names = list(self._launchers)
        active: dict[Future[InferenceSession], str] = {}
        executor = ThreadPoolExecutor(
            max_workers=self.config.max_active_candidates,
            thread_name_prefix="remote-inference-endpoint-race",
        )
        shutdown_wait = True
        with resource_budget_scope(self._budget_manager):
            try:
                self._fill_active(executor, pending_names, active)
                while active:
                    done, _not_done = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        name = active.pop(future)
                        try:
                            session = future.result()
                        except BaseException as error:
                            cleanup_error = self._stop_candidate(name)
                            cleanup_command = _cleanup_command_for_launcher(self._launchers[name])
                            self._record_attempt_final(
                                name,
                                final_state="CLEANUP_FAILED" if cleanup_error else "FAILED",
                                failure_code=classify_failure(str(error)),
                                failure_message=sanitized_excerpt(str(error)),
                                cleanup_command=cleanup_command,
                                cleanup_status=("failed" if cleanup_error else "stopped"),
                                cleanup_error=sanitized_excerpt(cleanup_error)
                                if cleanup_error
                                else None,
                            )
                            if cleanup_error:
                                message = (
                                    f"Endpoint race candidate {name!r} failed and cleanup also "
                                    f"failed: {cleanup_error}"
                                )
                                self._release_logical_budget()
                                self._write_summary("FAILED", failure_message=message)
                                raise RuntimeError(message) from error
                            self._fill_active(executor, pending_names, active)
                            continue
                        self._winner_name = name
                        self._winner_session = _session_with_name(
                            session,
                            self.config.name,
                            winner_name=name,
                            run_id=self._run_id,
                            summary_path=str(self._summary.path),
                        )
                        shutdown_wait = False
                        self._record_attempt_final(
                            name,
                            final_state="READY",
                            winner=True,
                            backend_kind=getattr(session, "backend_kind", ""),
                            cleanup_command=getattr(session, "cleanup_command", ""),
                            cleanup_status="command_available"
                            if getattr(session, "cleanup_command", "")
                            else "not_applicable",
                            local_port=getattr(session, "local_port", None),
                            remote_port=getattr(session, "remote_port", None),
                            job_id=str(getattr(session, "job_id", "") or "") or None,
                            pid=getattr(session, "pid", None),
                            node=str(getattr(session, "node", "") or "") or None,
                            logs=str(getattr(session, "logs", "") or "") or None,
                            summary_path=str(getattr(session, "summary_path", "") or "") or None,
                        )
                        cleanup_errors: dict[str, str] = {}
                        if self.config.cancel_losers:
                            cleanup_errors = self._stop_losers(except_name=name)
                        self._finalize_non_winning_attempts(
                            winner_name=name,
                            cleanup_errors=cleanup_errors,
                        )
                        self._release_logical_budget()
                        if cleanup_errors:
                            message = _cleanup_failure_message(cleanup_errors)
                            self._write_summary("FAILED", failure_message=message)
                            raise RuntimeError(message)
                        self._write_summary("READY")
                        return self._winner_session
                self._release_logical_budget()
                self._write_summary(
                    "FAILED",
                    failure_message="No endpoint race candidate became ready.",
                )
                raise RuntimeError("No endpoint race candidate became ready.")
            except BaseException as error:
                if self.config.cancel_losers:
                    try:
                        self.stop()
                    except Exception as stop_error:
                        raise RuntimeError(
                            "Endpoint race failed and failed to stop cleanly: "
                            f"{stop_error}. Original failure: {error}"
                        ) from error
                else:
                    self._release_logical_budget()
                raise
            finally:
                executor.shutdown(wait=shutdown_wait, cancel_futures=True)

    def stop(self) -> None:
        """Stop every owned candidate, including the winner."""

        stop_error: Exception | None = None
        for name in reversed(list(self._launchers)):
            try:
                self._launchers[name].stop()
            except Exception as error:
                stop_error = error
        self._release_logical_budget()
        if self._winner_session is not None and self._last_lifecycle_state != "FAILED":
            self._write_summary("RELEASED")
        self._launchers.clear()
        self._winner_session = None
        if stop_error is not None:
            raise stop_error

    def reserve_summary(self):
        """Reserve the logical endpoint race summary before candidate launch."""

        return self._summary.reserve()

    def _fill_active(
        self,
        executor: ThreadPoolExecutor,
        pending_names: list[str],
        active: dict[Future[InferenceSession], str],
    ) -> None:
        while pending_names and len(active) < self.config.max_active_candidates:
            name = pending_names.pop(0)
            if active and self.config.launch_stagger_seconds:
                time.sleep(self.config.launch_stagger_seconds)
            context = copy_context()
            active[executor.submit(context.run, self._launchers[name].start)] = name
            self._record_attempt_start(name)

    def _record_attempt_start(self, name: str) -> None:
        if any(attempt.get("candidate_name") == name for attempt in self._attempts):
            return
        config = self.config.candidates.get(name)
        self._attempts.append(
            {
                "candidate_index": self._candidate_indices.get(name),
                "candidate_name": name,
                "backend_kind": _candidate_backend_kind(config),
                "final_state": "STARTED",
                "winner": False,
                "cleanup_status": "unknown",
                **_candidate_target_fields(config),
            }
        )

    def _record_attempt_final(self, name: str, **fields: object) -> None:
        for attempt in reversed(self._attempts):
            if attempt.get("candidate_name") == name:
                attempt.update(fields)
                attempt.setdefault("winner", False)
                return
        config = self.config.candidates.get(name)
        payload = {
            "candidate_index": self._candidate_indices.get(name),
            "candidate_name": name,
            "backend_kind": _candidate_backend_kind(config),
            "winner": False,
            **_candidate_target_fields(config),
        }
        payload.update(fields)
        self._attempts.append(payload)

    def _acquire_logical_budget(self) -> None:
        if self._budget_manager is None or self._logical_budget_token is not None:
            return
        self._logical_budget_token = self._budget_manager.acquire(logical_launches=1)

    def _release_logical_budget(self) -> None:
        if self._logical_budget_token is None:
            return
        self._logical_budget_token.release()
        self._logical_budget_token = None

    def _finalize_non_winning_attempts(
        self,
        *,
        winner_name: str,
        cleanup_errors: Mapping[str, str] | None = None,
    ) -> None:
        cleanup_errors = dict(cleanup_errors or {})
        recorded_names = {str(attempt.get("candidate_name")) for attempt in self._attempts}
        for name, launcher in self._launchers.items():
            if name == winner_name:
                continue
            existing = _attempt_for_name(self._attempts, name)
            if existing is not None and existing.get("final_state") in {
                "FAILED",
                "READY",
                "CANCELLED",
                "SUPERSEDED",
                "SKIPPED",
            }:
                continue
            cleanup_command = _cleanup_command_for_launcher(launcher)
            cleanup_error = cleanup_errors.get(name)
            final_state = (
                "CLEANUP_FAILED"
                if cleanup_error
                else "CANCELLED"
                if self.config.cancel_losers and name in recorded_names
                else "SUPERSEDED"
                if name in recorded_names
                else "SKIPPED"
            )
            self._record_attempt_final(
                name,
                final_state=final_state,
                cleanup_command=cleanup_command,
                cleanup_status=(
                    "failed"
                    if cleanup_error
                    else "stopped"
                    if final_state == "CANCELLED"
                    else "not_requested"
                    if final_state == "SUPERSEDED"
                    else "not_started"
                ),
                cleanup_error=sanitized_excerpt(cleanup_error) if cleanup_error else None,
            )

    def _stop_candidate(self, name: str) -> str:
        try:
            self._launchers[name].stop()
        except Exception as error:
            return str(error)
        return ""

    def _stop_losers(self, *, except_name: str) -> dict[str, str]:
        cleanup_errors: dict[str, str] = {}
        for name in self._launchers:
            if name != except_name:
                cleanup_error = self._stop_candidate(name)
                if cleanup_error:
                    cleanup_errors[name] = cleanup_error
        return cleanup_errors

    def _write_summary(self, lifecycle_state: str, *, failure_message: str | None = None) -> None:
        self._last_lifecycle_state = lifecycle_state
        session = self._winner_session
        log_dirs = _session_log_dirs(session)
        self._summary.write(
            endpoint_summary(
                endpoint_name=self.config.name,
                backend_kind="endpoint_race",
                lifecycle_state=lifecycle_state,
                run_id=self._run_id,
                api_base=session.api_base if session is not None else "",
                served_model_name=session.served_model_name if session is not None else "",
                model=session.model if session is not None else "",
                api_key_set=bool(session.api_key) if session is not None else False,
                local_port=session.local_port if session is not None else None,
                remote_port=session.remote_port if session is not None else None,
                job_id=session.job_id if session is not None else "",
                pid=session.pid if session is not None else None,
                node=session.node if session is not None else "",
                cleanup_command=session.cleanup_command if session is not None else "",
                remote_log_dir=log_dirs.get("remote_log_dir"),
                local_log_dir=log_dirs.get("local_log_dir"),
                failure_message=failure_message,
                resource_attempts=self._attempts,
                extra={
                    "winner_candidate": self._winner_name or None,
                    **_session_summary_extra(session),
                    "resource_budget": self.config.resource_budget.__dict__
                    if self.config.resource_budget is not None
                    else None,
                    "observed_resource_usage": (
                        self._budget_manager.snapshot().to_dict()
                        if self._budget_manager is not None
                        else None
                    ),
                },
            )
        )


def _session_log_dirs(session: InferenceSession | None) -> dict[str, str | None]:
    if session is None or not session.logs:
        return {}
    metadata = dict(session.metadata or {})
    backend_kind = str(metadata.get("winner_backend_kind") or session.backend_kind or "")
    if backend_kind == "local_vllm":
        return {"local_log_dir": session.logs}
    return {"remote_log_dir": session.logs}


def _session_summary_extra(session: InferenceSession | None) -> dict[str, object]:
    if session is None:
        return {}
    metadata = dict(session.metadata or {})
    winner_backend_kind = str(metadata.get("winner_backend_kind") or session.backend_kind or "")
    extra: dict[str, object] = {
        "winner_backend_kind": winner_backend_kind or None,
        "winner_summary_path": metadata.get("winner_summary_path") or session.summary_path or None,
        "logs": session.logs or None,
    }
    readiness = metadata.get("readiness")
    if isinstance(readiness, dict):
        extra["readiness"] = dict(readiness)
    recommended = metadata.get("recommended_benchmark_max_parallel")
    if isinstance(recommended, int) and recommended > 0:
        extra["recommended_benchmark_max_parallel"] = recommended
        extra["benchmark_handoff"] = {
            "api_base": session.api_base or None,
            "model": session.served_model_name or session.model or None,
            "recommended_max_parallel": recommended,
        }
    return extra


def _cleanup_command_for_launcher(launcher: InferenceLauncher) -> str:
    last_summary = getattr(launcher, "_last_summary", None)
    if isinstance(last_summary, dict):
        return str(last_summary.get("cleanup_command") or "")
    return ""


def _cleanup_failure_message(cleanup_errors: Mapping[str, str]) -> str:
    failures = ", ".join(
        f"{name}: {sanitized_excerpt(error, max_chars=300)}"
        for name, error in sorted(cleanup_errors.items())
    )
    return f"Endpoint race cleanup failed for non-winning candidate(s): {failures}"


def _candidate_backend_kind(config: object) -> str:
    class_name = type(config).__name__
    return {
        "ExistingEndpointConfig": "existing_endpoint",
        "LocalVllmConfig": "local_vllm",
        "SshVllmConfig": "ssh_vllm",
        "SlurmVllmConfig": "slurm_vllm",
        "EndpointRaceConfig": "endpoint_race",
        "FleetConfig": "fleet",
    }.get(class_name, str(getattr(config, "backend_kind", "") or class_name or "unknown"))


def _candidate_target_fields(config: object) -> dict[str, object]:
    if config is None:
        return {}
    fields: dict[str, object] = {}
    for field_name in (
        "ssh_target",
        "partition",
        "api_base",
        "host",
        "local_bind_host",
        "local_port",
        "remote_port",
        "job_name",
        "out_dir",
    ):
        if hasattr(config, field_name):
            value = getattr(config, field_name)
            if value not in {"", None}:
                fields[field_name] = value
    return fields


def _attempt_for_name(
    attempts: list[dict[str, object]],
    name: str,
) -> dict[str, object] | None:
    for attempt in reversed(attempts):
        if attempt.get("candidate_name") == name:
            return attempt
    return None


def validate_endpoint_race_config(config: EndpointRaceConfig) -> EndpointRaceConfig:  # noqa: C901
    """Reject endpoint-race configs that cannot be launched safely."""

    if not config.name.strip():
        raise ValueError("endpoint_race name must be non-empty.")
    if not config.candidates:
        raise ValueError("endpoint_race candidates must be non-empty.")
    if len(set(config.candidates)) != len(config.candidates):
        raise ValueError("endpoint_race candidate names must be unique.")
    if config.max_active_candidates < 1:
        raise ValueError("endpoint_race max_active_candidates must be positive.")
    if config.launch_stagger_seconds < 0:
        raise ValueError("endpoint_race launch_stagger_seconds must be non-negative.")
    if config.winner_condition != "endpoint_ready":
        raise ValueError("endpoint_race winner_condition must be 'endpoint_ready'.")
    normalized = config
    if config.max_active_candidates > len(config.candidates):
        normalized = replace(config, max_active_candidates=len(config.candidates))
    if config.resource_budget is not None:
        for field_name, value in config.resource_budget.__dict__.items():
            if value is not None and value < 1:
                raise ValueError(f"endpoint_race resource_budget.{field_name} must be positive.")
        cap = config.resource_budget.max_active_candidate_attempts
        if cap is not None and normalized.max_active_candidates > cap:
            raise ValueError(
                "endpoint_race max_active_candidates exceeds "
                "resource_budget.max_active_candidate_attempts."
            )
    return normalized


def _session_with_name(
    session: InferenceSession,
    name: str,
    *,
    winner_name: str,
    run_id: str,
    summary_path: str,
) -> InferenceSession:
    metadata = dict(session.metadata)
    metadata["winner_candidate"] = winner_name
    metadata["winner_backend_kind"] = session.backend_kind
    metadata["winner_summary_path"] = session.summary_path
    return InferenceSession(
        name=name,
        api_base=session.api_base,
        served_model_name=session.served_model_name,
        model=session.model,
        api_key=session.api_key,
        local_port=session.local_port,
        remote_port=session.remote_port,
        logs=session.logs,
        pid=session.pid,
        job_id=session.job_id,
        node=session.node,
        cleanup_command=session.cleanup_command,
        backend_kind="endpoint_race",
        run_id=run_id,
        summary_path=summary_path,
        metadata=metadata,
    )
