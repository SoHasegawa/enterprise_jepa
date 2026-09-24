"""Multi-endpoint inference fleet launcher."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from threading import RLock

from remote_inference_launcher.config_types import ResourceBudget
from remote_inference_launcher.core import InferenceLauncher, InferenceSession
from remote_inference_launcher.resource_budget import (
    ResourceBudgetManager,
    ResourceBudgetReservation,
    resource_budget_scope,
)
from remote_inference_launcher.summaries import SummaryWriter, fleet_summary, new_run_id, safe_label


@dataclass(frozen=True)
class FleetConfig:
    """Configuration for multiple named endpoint configs."""

    endpoints: Mapping[str, object]
    name: str = "fleet"
    max_active_launches: int | None = None
    launch_stagger_seconds: float = 0.0
    failure_policy: str = "fail_fast"
    handoff_mode: str = "all_ready"
    endpoint_lifetime_policy: str = "fleet"
    resource_budget: ResourceBudget | None = None
    launch_summary_path: str = ""
    overwrite_launch_summary: bool = False


@dataclass(frozen=True)
class FleetEvent:
    """Machine-readable fleet lifecycle event."""

    event: str
    endpoint_name: str = ""
    session: InferenceSession | None = None
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        session_payload = None
        if self.session is not None:
            session_payload = {
                "name": self.session.name,
                "api_base": self.session.api_base,
                "model": self.session.model,
                "served_model_name": self.session.served_model_name,
                "backend_kind": self.session.backend_kind,
                "run_id": self.session.run_id,
                "summary_path": self.session.summary_path,
                "cleanup_command": self.session.cleanup_command,
                "local_port": self.session.local_port,
                "remote_port": self.session.remote_port,
                "job_id": self.session.job_id,
                "pid": self.session.pid,
                "node": self.session.node,
                "metadata": dict(self.session.metadata),
            }
        return {
            "event": self.event,
            "endpoint_name": self.endpoint_name,
            "session": session_payload,
            "api_base": self.session.api_base if self.session is not None else None,
            "model": self.session.served_model_name if self.session is not None else None,
            "error": self.error,
        }


class InferenceFleetLauncher:
    """Start and stop multiple named inference endpoints as one lifecycle."""

    def __init__(
        self,
        config: FleetConfig | Mapping[str, object],
        *,
        event_callback: Callable[[FleetEvent], None] | None = None,
        run_id: str | None = None,
    ) -> None:
        if isinstance(config, FleetConfig):
            self.config = validate_fleet_config(config)
        else:
            self.config = validate_fleet_config(FleetConfig(endpoints=config))
        if not self.config.endpoints:
            raise ValueError("Inference fleet requires at least one endpoint.")
        self._endpoint_configs = dict(self.config.endpoints)
        self._launchers: dict[str, InferenceLauncher] = {}
        self._sessions: dict[str, InferenceSession] = {}
        self._known_sessions: dict[str, InferenceSession] = {}
        self._failures: dict[str, str] = {}
        self._released: set[str] = set()
        self._events: list[FleetEvent] = []
        self._event_callback = event_callback
        self._lock = RLock()
        self._run_id = run_id or new_run_id()
        self._summary = SummaryWriter(
            self.config.launch_summary_path,
            run_id=self._run_id,
            endpoint_name=self.config.name,
            backend_kind="fleet",
            overwrite=self.config.overwrite_launch_summary,
        )
        self._budget_manager = (
            ResourceBudgetManager(self.config.resource_budget)
            if self.config.resource_budget is not None
            else None
        )
        self._logical_budget_tokens: dict[str, ResourceBudgetReservation] = {}

    def set_event_callback(self, callback: Callable[[FleetEvent], None] | None) -> None:
        """Set a callback invoked when fleet lifecycle events are observed."""

        self._event_callback = callback

    @property
    def owns_resources(self) -> bool:
        """Return whether any started endpoint is owned by this fleet."""

        return any(
            bool(getattr(launcher, "owns_resources", True)) for launcher in self._launchers.values()
        )

    def start(self) -> Mapping[str, InferenceSession]:
        """Start every endpoint concurrently and return sessions by name."""

        try:
            self._prepare_launchers()
            self._reserve_child_summaries()
        except BaseException as error:
            self._write_summary("FAILED", failure_message=str(error), sessions={})
            raise
        futures: dict[Future[InferenceSession], str] = {}
        executor = ThreadPoolExecutor(
            max_workers=self._max_workers(),
            thread_name_prefix="remote-inference-fleet",
        )
        with resource_budget_scope(self._budget_manager):
            try:
                pending = list(self._launchers)
                futures = self._submit_ready_work(executor, pending, active_count=0)
                sessions: dict[str, InferenceSession] = {}
                while futures:
                    done, _not_done = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        name = futures.pop(future)
                        self._handle_completed_start(future, name, sessions)
                    futures.update(
                        self._submit_ready_work(
                            executor,
                            pending,
                            active_count=len(futures),
                        )
                    )
                if not sessions:
                    raise RuntimeError("Inference fleet did not start any endpoints.")
                self._finalize_started_sessions(sessions)
            except BaseException as error:
                self._cancel_pending_starts(futures)
                self._release_logical_budgets()
                self._write_summary(
                    "FAILED",
                    failure_message=str(error),
                    sessions=sessions if "sessions" in locals() else {},
                )
                if self.config.failure_policy == "fail_fast":
                    self._stop_after_start_failure(error)
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        return dict(self._sessions)

    def _handle_completed_start(
        self,
        future: Future[InferenceSession],
        name: str,
        sessions: dict[str, InferenceSession],
    ) -> None:
        self._release_logical_budget(name)
        try:
            session = _session_with_name(future.result(), name)
        except BaseException as error:
            self._record_start_failure(name, error)
            if self.config.failure_policy == "fail_fast":
                raise
            return

        self._record_start_success(name, session, sessions)

    def _record_start_failure(self, name: str, error: BaseException) -> None:
        with self._lock:
            self._failures[name] = str(error)
        self._record_event(
            FleetEvent(
                "endpoint_failed",
                endpoint_name=name,
                error=str(error),
            )
        )

    def _record_start_success(
        self,
        name: str,
        session: InferenceSession,
        sessions: dict[str, InferenceSession],
    ) -> None:
        sessions[name] = session
        with self._lock:
            self._sessions[name] = session
            self._known_sessions[name] = session
            self._failures.pop(name, None)
        self._record_event(
            FleetEvent(
                "endpoint_ready",
                endpoint_name=name,
                session=session,
            )
        )
        if self.config.handoff_mode == "incremental_ready":
            self._write_summary("PARTIAL_READY", sessions=sessions)

    def _finalize_started_sessions(self, sessions: dict[str, InferenceSession]) -> None:
        with self._lock:
            self._sessions = {
                name: sessions[name]
                for name in sorted(sessions)
                if name not in self._released
            }
            self._known_sessions.update({name: sessions[name] for name in sorted(sessions)})
        self._record_event(FleetEvent("fleet_complete"))
        state = "READY" if len(self._sessions) == len(self._launchers) else "PARTIAL_READY"
        self._write_summary(state)

    def stop(self) -> None:
        """Stop every started endpoint in reverse startup order."""

        stop_error: Exception | None = None
        for name in reversed(list(self._launchers)):
            if name in self._released:
                continue
            try:
                self._launchers[name].stop()
            except Exception as error:
                stop_error = error
            else:
                self._released.add(name)
            self._release_logical_budget(name)
        self._launchers.clear()
        with self._lock:
            self._sessions.clear()
        self._write_summary("RELEASED")
        if stop_error is not None:
            raise stop_error

    def reserve_summary(self):
        """Reserve the fleet and child summary paths before launch work starts."""

        self._prepare_launchers()
        self._reserve_child_summaries()
        return self._summary.path

    def release(self, name: str) -> None:
        """Release one endpoint without stopping unrelated fleet endpoints."""

        if self.config.endpoint_lifetime_policy != "per_endpoint":
            raise RuntimeError(
                "Fleet endpoint_lifetime_policy does not allow per-endpoint release."
            )
        if name in self._released:
            return
        if name not in self._launchers:
            raise KeyError(name)
        self._launchers[name].stop()
        with self._lock:
            self._released.add(name)
            self._sessions.pop(name, None)
        self._release_logical_budget(name)
        self._write_summary("PARTIAL_READY")

    def events(self) -> tuple[FleetEvent, ...]:
        """Return lifecycle events observed so far."""

        return tuple(self._events)

    @contextmanager
    def running(self) -> Iterator[Mapping[str, InferenceSession]]:
        """Context manager that starts and stops all endpoints."""

        sessions = self.start()
        try:
            yield sessions
        finally:
            self.stop()

    def __getitem__(self, name: str) -> InferenceSession:
        """Return a started session by name."""

        return self._sessions[name]

    def sessions(self) -> Mapping[str, InferenceSession]:
        """Return started sessions by name."""

        with self._lock:
            return dict(self._sessions)

    def env(self, *, prefix: str = "INFERENCE") -> dict[str, str]:
        """Return named endpoint env values."""

        values: dict[str, str] = {}
        for name, session in self._sessions.items():
            endpoint_prefix = f"{prefix}_{name}"
            values.update(session.env(prefix=endpoint_prefix))
        return values

    def _max_workers(self) -> int:
        if self.config.max_active_launches is None:
            return len(self._launchers)
        return max(1, min(self.config.max_active_launches, len(self._launchers)))

    def _prepare_launchers(self) -> None:
        if self._launchers:
            return
        from remote_inference_launcher.inference_config import launcher_from_config

        summary_path = self._summary.reserve()
        child_paths = _child_summary_paths(summary_path, sorted(self._endpoint_configs))
        self._launchers = {
            name: launcher_from_config(
                _with_child_summary_path(
                    self._endpoint_configs[name],
                    summary_path=str(child_paths[name]),
                    overwrite=self.config.overwrite_launch_summary,
                )
            )
            for name in sorted(self._endpoint_configs)
        }

    def _reserve_child_summaries(self) -> None:
        for name, launcher in sorted(self._launchers.items()):
            reserve_summary = getattr(launcher, "reserve_summary", None)
            if reserve_summary is None:
                continue
            try:
                reserve_summary()
            except FileExistsError as error:
                raise FileExistsError(
                    f"Fleet endpoint {name!r} launch summary path is not available: {error}"
                ) from error

    def _submit_ready_work(
        self,
        executor: ThreadPoolExecutor,
        pending: list[str],
        *,
        active_count: int,
    ) -> dict[Future[InferenceSession], str]:
        submitted: dict[Future[InferenceSession], str] = {}
        while pending and active_count + len(submitted) < self._max_workers():
            name = pending.pop(0)
            if not self._acquire_logical_budget(name):
                pending.insert(0, name)
                break
            if submitted and self.config.launch_stagger_seconds:
                time.sleep(self.config.launch_stagger_seconds)
            context = copy_context()
            submitted[executor.submit(context.run, self._launchers[name].start)] = name
        return submitted

    def _acquire_logical_budget(self, name: str) -> bool:
        if self._budget_manager is None:
            return True
        token = self._budget_manager.acquire(logical_launches=1, block=False)
        if token is None:
            return False
        self._logical_budget_tokens[name] = token
        return True

    def _release_logical_budget(self, name: str) -> None:
        token = self._logical_budget_tokens.pop(name, None)
        if token is not None:
            token.release()

    def _release_logical_budgets(self) -> None:
        for name in list(self._logical_budget_tokens):
            self._release_logical_budget(name)

    def _cancel_pending_starts(self, futures: Mapping[Future[InferenceSession], str]) -> None:
        for future in futures:
            future.cancel()

    def _stop_after_start_failure(self, error: BaseException) -> None:
        try:
            self.stop()
        except Exception as stop_error:
            raise RuntimeError(
                "Inference fleet failed and failed to stop cleanly: "
                f"{stop_error}. Original failure: {error}"
            ) from error

    def _record_event(self, event: FleetEvent) -> None:
        self._events.append(event)
        if self._event_callback is not None:
            self._event_callback(event)

    def _write_summary(
        self,
        lifecycle_state: str,
        *,
        failure_message: str | None = None,
        sessions: Mapping[str, InferenceSession] | None = None,
    ) -> None:
        with self._lock:
            active_sessions = dict(self._known_sessions if sessions is None else sessions)
            failures = dict(self._failures)
            released = set(self._released)
        endpoint_payloads: dict[str, dict[str, object]] = {}
        for name, session in active_sessions.items():
            endpoint_payloads[name] = {
                "endpoint_name": name,
                "api_base": session.api_base,
                "served_model_name": session.served_model_name,
                "model": session.model,
                "backend_kind": session.backend_kind,
                "run_id": session.run_id,
                "summary_path": session.summary_path,
                "cleanup_command": session.cleanup_command,
                "local_port": session.local_port,
                "remote_port": session.remote_port,
                "job_id": session.job_id or None,
                "pid": session.pid,
                "node": session.node or None,
                "logs": session.logs or None,
                "recommended_benchmark_max_parallel": session.metadata.get(
                    "recommended_benchmark_max_parallel"
                ),
                "lifecycle_state": "RELEASED" if name in released else "READY",
            }
        for name, message in sorted(failures.items()):
            endpoint_payloads.setdefault(
                name,
                {
                    "endpoint_name": name,
                    "lifecycle_state": "FAILED",
                    "failure_message": message,
                },
            )
        payload = fleet_summary(
            fleet_name=self.config.name,
            lifecycle_state=lifecycle_state,
            run_id=self._run_id,
            endpoints=endpoint_payloads,
            resource_budget=(
                self.config.resource_budget.__dict__
                if self.config.resource_budget is not None
                else None
            ),
            observed={
                "known_endpoints": len(endpoint_payloads),
                "ready_endpoints": sum(
                    1
                    for endpoint in endpoint_payloads.values()
                    if endpoint.get("lifecycle_state") in {"READY", "RELEASED"}
                ),
                "failed_endpoints": sum(
                    1
                    for endpoint in endpoint_payloads.values()
                    if endpoint.get("lifecycle_state") == "FAILED"
                ),
                "released_endpoints": len(self._released),
                "failure_message": failure_message,
                "handoff_mode": self.config.handoff_mode,
                "failure_policy": self.config.failure_policy,
                "resource_usage": (
                    self._budget_manager.snapshot().to_dict()
                    if self._budget_manager is not None
                    else None
                ),
            },
        )
        self._summary.write(payload)


def _session_with_name(session: InferenceSession, name: str) -> InferenceSession:
    if session.name == name:
        return session
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
        backend_kind=session.backend_kind,
        run_id=session.run_id,
        summary_path=session.summary_path,
        metadata=session.metadata,
    )


def _with_child_summary_path(config: object, *, summary_path: str, overwrite: bool) -> object:
    values: dict[str, object] = {}
    if hasattr(config, "launch_summary_path"):
        values["launch_summary_path"] = summary_path
    if overwrite and hasattr(config, "overwrite_launch_summary"):
        values["overwrite_launch_summary"] = True
    if values and is_dataclass(config):
        return replace(config, **values)
    return config


def _child_summary_paths(fleet_summary_path: Path, endpoint_names: list[str]) -> dict[str, str]:
    root = fleet_summary_path.parent / "endpoints"
    paths: dict[str, str] = {}
    seen_paths: dict[str, str] = {}
    for name in endpoint_names:
        path = str(root / _child_summary_filename(name))
        prior_name = seen_paths.get(path)
        if prior_name is not None:
            raise RuntimeError(
                "Fleet generated duplicate child summary path for endpoints "
                f"{prior_name!r} and {name!r}: {path}"
            )
        paths[name] = path
        seen_paths[path] = name
    return paths


def _child_summary_filename(endpoint_name: str) -> str:
    digest = hashlib.sha256(endpoint_name.encode("utf-8")).hexdigest()[:8]
    return f"{safe_label(endpoint_name)}-{digest}.json"


def validate_fleet_config(config: FleetConfig) -> FleetConfig:  # noqa: C901
    """Reject fleet policies that cannot be launched safely."""

    if not config.name.strip():
        raise ValueError("Fleet name must be non-empty.")
    if not config.endpoints:
        raise ValueError("Fleet endpoints must be non-empty.")
    if config.max_active_launches is not None and config.max_active_launches < 1:
        raise ValueError("Fleet max_active_launches must be positive.")
    if config.launch_stagger_seconds < 0:
        raise ValueError("Fleet launch_stagger_seconds must be non-negative.")
    if config.failure_policy not in {"fail_fast", "keep_ready"}:
        raise ValueError("Fleet failure_policy must be fail_fast or keep_ready.")
    if config.handoff_mode not in {"all_ready", "incremental_ready"}:
        raise ValueError("Fleet handoff_mode must be all_ready or incremental_ready.")
    if config.endpoint_lifetime_policy not in {"fleet", "per_endpoint"}:
        raise ValueError("Fleet endpoint_lifetime_policy must be fleet or per_endpoint.")
    if config.resource_budget is not None:
        for field_name, value in config.resource_budget.__dict__.items():
            if value is not None and value < 1:
                raise ValueError(f"Fleet resource_budget.{field_name} must be positive.")
    return config
