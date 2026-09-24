"""Durable file-backed run registry for remote inference launches."""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from remote_inference_launcher.diagnostics import sanitized_excerpt
from remote_inference_launcher.plans import DEFAULT_REGISTRY_ROOT, EffectiveLaunchPlan
from remote_inference_launcher.session_env import shell_quote

STATE_SCHEMA_VERSION = "ril-state/v1"
EVENT_SCHEMA_VERSION = "ril-event/v1"
DEFAULT_CLEANUP_TIMEOUT_SECONDS = 60.0
TERMINAL_STATES = {"FAILED", "STOPPED", "RELEASED", "ORPHANED", "EXPIRED"}
SLURM_TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


def create_run_registry(plan: EffectiveLaunchPlan) -> None:
    """Create a run registry and write immutable plan plus initial state."""

    registry_dir = plan.registry_dir
    with registry_lock(registry_dir):
        (registry_dir / "endpoints").mkdir(parents=True, exist_ok=True)
        write_json_atomic(registry_dir / "plan.json", plan.to_dict())
        state = initial_state(plan)
        write_json_atomic(registry_dir / "state.json", state)
        append_event_locked(
            registry_dir,
            event="run_planned",
            level="info",
            payload={"source_config_path": plan.source_config_path},
        )


def mark_submitting(registry_dir: str | Path) -> None:
    """Record that launch side effects are starting."""

    update_state(registry_dir, lifecycle_state="SUBMITTING")
    append_event(
        registry_dir,
        event="run_submitting",
        level="info",
        payload={},
    )


def record_ready_sessions(registry_dir: str | Path, sessions: Mapping[str, object]) -> None:
    """Record ready session state and write non-secret env handoff."""

    registry_path = Path(registry_dir)
    with registry_lock(registry_path):
        state = read_state_locked(registry_path)
        endpoints = {
            name: endpoint_state_from_session(session) for name, session in sorted(sessions.items())
        }
        state.update(
            {
                "lifecycle_state": "READY",
                "updated_at": utc_now(),
                "failure": None,
                "endpoints": endpoints,
            }
        )
        write_json_atomic(registry_path / "state.json", state)
        (registry_path / "env.sh").write_text(env_text_from_state(state), encoding="utf-8")
        append_event_locked(
            registry_path,
            event="run_ready",
            level="info",
            payload={"endpoints": sorted(endpoints)},
        )


def record_failure(registry_dir: str | Path, error: BaseException) -> None:
    """Record a failed launch or readiness attempt."""

    failure = {
        "code": type(error).__name__,
        "message": sanitized_excerpt(str(error), max_chars=1000),
    }
    with registry_lock(registry_dir):
        state = read_state_locked(registry_dir)
        state.update(
            {
                "lifecycle_state": "FAILED",
                "updated_at": utc_now(),
                "failure": failure,
            }
        )
        write_json_atomic(Path(registry_dir) / "state.json", state)
        append_event_locked(
            registry_dir,
            event="run_failed",
            level="error",
            payload={"failure": failure},
        )


def record_endpoint_summary(summary_path: str | Path, payload: Mapping[str, object]) -> None:
    """Mirror an endpoint summary payload into its owning registry when possible."""

    if payload.get("backend_kind") == "fleet" and "endpoints" in payload:
        return
    resolved_registry = _registry_from_summary_path(Path(summary_path))
    if resolved_registry is None:
        return
    endpoint_name = str(payload.get("endpoint_name", "") or "")
    if not endpoint_name:
        endpoint_name = _endpoint_name_from_summary_path(Path(summary_path))
    if not endpoint_name:
        return
    lifecycle = str(payload.get("lifecycle_state", "") or "")
    endpoint_update = endpoint_state_from_summary_payload(payload, summary_path=summary_path)
    with registry_lock(resolved_registry):
        state = read_state_locked(resolved_registry)
        endpoints = dict(state.get("endpoints", {}) or {})
        current = dict(endpoints.get(endpoint_name, {}) or {})
        current.update(endpoint_update)
        endpoints[endpoint_name] = current
        state["endpoints"] = endpoints
        if lifecycle == "FAILED":
            state["lifecycle_state"] = "FAILED"
            diagnostics = payload.get("diagnostics", {})
            if isinstance(diagnostics, Mapping):
                state["failure"] = {
                    "code": diagnostics.get("failure_code") or "endpoint_failed",
                    "message": diagnostics.get("failure_message") or "",
                }
        else:
            root_lifecycle = _root_lifecycle_from_endpoint_summary(
                str(state.get("lifecycle_state", "") or ""),
                lifecycle,
                endpoints=endpoints,
            )
            if root_lifecycle:
                state["lifecycle_state"] = root_lifecycle
        state["updated_at"] = utc_now()
        write_json_atomic(resolved_registry / "state.json", state)
        event_payload = _endpoint_summary_event_payload(payload, summary_path=summary_path)
        registry_event_payload = payload.get("registry_event_payload")
        if isinstance(registry_event_payload, Mapping):
            event_payload.update(dict(registry_event_payload))
        append_event_locked(
            resolved_registry,
            event=str(payload.get("registry_event", "") or "")
            or _endpoint_summary_event(lifecycle),
            level="error" if lifecycle == "FAILED" else "info",
            endpoint_name=endpoint_name,
            payload=event_payload,
        )


def _endpoint_summary_event(lifecycle: str) -> str:
    return {
        "SUBMITTED": "slurm_job_submitted",
        "PENDING": "slurm_job_pending",
        "ALLOCATED": "slurm_allocation_acquired",
        "REMOTE_PORT_READY": "remote_port_ready",
        "TUNNEL_READY": "tunnel_ready",
        "READY": "endpoint_ready",
        "FAILED": "endpoint_failed",
        "RELEASED": "endpoint_released",
        "STOPPED": "endpoint_stopped",
    }.get(lifecycle, "endpoint_summary")


def _root_lifecycle_from_endpoint_summary(
    current_lifecycle: str,
    endpoint_lifecycle: str,
    *,
    endpoints: Mapping[str, object],
) -> str:
    if current_lifecycle in TERMINAL_STATES:
        return ""
    if len(endpoints) != 1:
        return ""
    if endpoint_lifecycle in {
        "SUBMITTED",
        "PENDING",
        "ALLOCATED",
        "REMOTE_PORT_READY",
        "TUNNEL_READY",
        "READY",
        "RELEASED",
        "STOPPED",
    }:
        return endpoint_lifecycle
    return ""


def _endpoint_summary_event_payload(
    payload: Mapping[str, object],
    *,
    summary_path: str | Path,
) -> dict[str, object]:
    event_payload: dict[str, object] = {
        "lifecycle_state": payload.get("lifecycle_state"),
        "job_id": payload.get("job_id"),
        "node": payload.get("node"),
        "local_port": payload.get("local_port"),
        "remote_port": payload.get("remote_port"),
        "remote_log_dir": payload.get("remote_log_dir"),
        "remote_state_path": payload.get("remote_state_path"),
        "summary_path": str(summary_path),
    }
    slurm = _slurm_state_from_summary_payload(payload)
    if slurm:
        event_payload["slurm"] = slurm
    return {key: value for key, value in event_payload.items() if value is not None and value != ""}


def resolve_registry_path(value: str | Path, *, root: str | Path = DEFAULT_REGISTRY_ROOT) -> Path:
    """Resolve a run ID, registry directory, or summary path to a registry directory."""

    path = Path(value)
    if path.is_dir() and (path / "state.json").exists():
        return path
    if path.is_file():
        resolved = _registry_from_summary_path(path)
        if resolved is not None:
            return resolved
    if path.name == "summary.json":
        resolved = _registry_from_summary_path(path)
        if resolved is not None:
            return resolved
    run_path = Path(root).expanduser() / str(value)
    if (run_path / "state.json").exists():
        return run_path
    raise FileNotFoundError(f"Run registry not found: {value}")


def read_registry(value: str | Path) -> dict[str, object]:
    """Read plan and state for a registry reference."""

    registry_dir = resolve_registry_path(value)
    reconcile_terminal_endpoint_states(registry_dir)
    reconcile_orphaned_controller(registry_dir)
    return {
        "registry_dir": str(registry_dir),
        "plan": read_json(registry_dir / "plan.json"),
        "state": read_json(registry_dir / "state.json"),
    }


def format_status(registry: Mapping[str, object]) -> str:
    """Return human-readable registry status."""

    state = dict(registry.get("state", {}) or {})
    lines = [
        f"run_id={state.get('run_id', '')}",
        f"registry={registry.get('registry_dir', '')}",
        f"state={state.get('lifecycle_state', '')}",
    ]
    failure = state.get("failure")
    if failure:
        lines.append(f"failure={json.dumps(failure, sort_keys=True)}")
    controller = state.get("controller")
    if isinstance(controller, Mapping):
        lines.append(
            "controller="
            f"mode={controller.get('mode', '')} "
            f"pid={controller.get('pid', '')} "
            f"heartbeat_at={controller.get('heartbeat_at', '')}"
        )
    endpoints = state.get("endpoints", {})
    if isinstance(endpoints, Mapping):
        for name, endpoint in sorted(endpoints.items()):
            if not isinstance(endpoint, Mapping):
                continue
            lines.append(f"endpoint={_format_endpoint_status(name, endpoint)}")
    return "\n".join(lines)


def env_text_from_state(state: Mapping[str, object], *, endpoint_name: str = "") -> str:
    """Return non-secret shell exports for one endpoint or the default endpoint."""

    endpoints = state.get("endpoints", {})
    if not isinstance(endpoints, Mapping) or not endpoints:
        return "# No ready endpoints are recorded for this run.\n"
    selected_name = endpoint_name or ("default" if "default" in endpoints else sorted(endpoints)[0])
    endpoint = endpoints.get(selected_name)
    if not isinstance(endpoint, Mapping):
        raise ValueError(f"Endpoint not found in registry state: {selected_name}")
    values = {
        "OPENAI_BASE_URL": str(endpoint.get("api_base", "") or ""),
        "OPENAI_MODEL_NAME": str(endpoint.get("served_model_name", "") or ""),
        "INFERENCE_LAUNCH_SUMMARY_PATH": str(endpoint.get("summary_path", "") or ""),
    }
    lines = ["#!/usr/bin/env bash", "# Generated by remote-inference-launcher"]
    for name, value in values.items():
        if value:
            lines.append(f"export {name}={shell_quote(value)}")
    lines.append("# OPENAI_API_KEY must be supplied separately when required.")
    return "\n".join(lines) + "\n"


def _format_endpoint_status(name: str, endpoint: Mapping[str, object]) -> str:
    parts = [
        str(name),
        f"state={endpoint.get('lifecycle_state', '')}",
    ]
    for field in (
        "api_base",
        "served_model_name",
        "job_id",
        "local_port",
        "remote_port",
        "node",
        "logs",
        "summary_path",
        "cleanup_command",
        "remote_state_path",
    ):
        value = endpoint.get(field)
        if value not in {"", None}:
            parts.append(f"{field}={value}")
    slurm = endpoint.get("slurm")
    if isinstance(slurm, Mapping):
        for label, field in (
            ("state", "latest_state"),
            ("reason", "latest_reason"),
            ("partition", "partition"),
            ("pending_seconds", "pending_duration_seconds"),
        ):
            value = slurm.get(field)
            if value not in {"", None}:
                parts.append(f"slurm_{label}={value}")
    return " ".join(parts)


def cleanup_commands_from_state(
    state: Mapping[str, object],
    *,
    registry_dir: str | Path | None = None,
) -> list[str]:
    """Return endpoint cleanup commands from registry state."""

    endpoints = state.get("endpoints", {})
    if not isinstance(endpoints, Mapping):
        return []
    commands: list[str] = []
    for endpoint in endpoints.values():
        if isinstance(endpoint, Mapping):
            command = str(endpoint.get("cleanup_command", "") or "")
            if command:
                commands.append(command)
    if registry_dir is not None:
        return [
            command
            for command in commands
            if not _is_self_registry_stop_command(command, registry_dir=Path(registry_dir))
        ]
    return commands


def run_stop(
    registry_dir: str | Path,
    *,
    force: bool = False,
    cleanup_timeout_seconds: float = DEFAULT_CLEANUP_TIMEOUT_SECONDS,
) -> int:
    """Run recorded cleanup commands and update registry state."""

    registry_path = Path(registry_dir)
    state = read_json(registry_path / "state.json")
    commands = cleanup_commands_from_state(state, registry_dir=registry_path)
    status = 0
    update_state(registry_path, lifecycle_state="STOPPING")
    append_event(
        registry_path,
        event="run_stopping",
        level="info",
        payload={"commands": len(commands)},
    )
    controller_status = stop_controller(registry_path, force=force)
    if controller_status != 0:
        status = controller_status
        append_event(
            registry_path,
            event="controller_stop_failed",
            level="error",
            payload={"returncode": controller_status, "force": force},
        )
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                shell=True,
                check=False,
                timeout=cleanup_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            status = 124
            append_event(
                registry_path,
                event="cleanup_command_timeout",
                level="error",
                payload={
                    "command": command,
                    "timeout_seconds": cleanup_timeout_seconds,
                    "returncode": status,
                },
            )
            continue
        if completed.returncode != 0:
            status = completed.returncode
            append_event(
                registry_path,
                event="cleanup_command_failed",
                level="error",
                payload={"command": command, "returncode": completed.returncode},
            )
    final_lifecycle = "STOPPED" if status == 0 else "FAILED"
    update_state(registry_path, lifecycle_state=final_lifecycle)
    append_event(
        registry_path,
        event="run_stopped" if status == 0 else "run_stop_failed",
        level="info" if status == 0 else "error",
        payload={"returncode": status},
    )
    return status


def _is_self_registry_stop_command(command: str, *, registry_dir: Path) -> bool:
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    if len(parts) != 3 or parts[:2] != ["remote-inference-launcher", "stop"]:
        return False
    return Path(parts[2]) == registry_dir


def stop_controller(registry_dir: str | Path, *, force: bool = False) -> int:
    """Terminate a detached controller process when one is recorded."""

    registry_path = Path(registry_dir)
    pid = controller_pid(registry_path)
    if pid is None or pid == os.getpid() or not pid_alive(pid):
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return 0
    except PermissionError:
        return 1
    if _wait_for_pid_exit(pid, timeout_seconds=10.0):
        return 0
    if not force:
        return 1
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return 0
    except PermissionError:
        return 1
    return 0 if _wait_for_pid_exit(pid, timeout_seconds=5.0) else 1


def _wait_for_pid_exit(pid: int, *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.1)
    return not pid_alive(pid)


def initial_state(plan: EffectiveLaunchPlan) -> dict[str, object]:
    """Return the initial registry state for an effective plan."""

    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "run_id": plan.run_id,
        "lifecycle_state": "PLANNED",
        "created_at": plan.created_at,
        "updated_at": plan.created_at,
        "failure": None,
        "controller": {
            "mode": plan.controller_policy,
            "pid": os.getpid() if plan.controller_policy == "foreground" else None,
            "heartbeat_at": None,
        },
        "lease": None,
        "endpoints": {
            endpoint.name: {
                "lifecycle_state": "PLANNED",
                "api_base": "",
                "served_model_name": endpoint.served_model_name,
                "job_id": "",
                "local_port": endpoint.local_port,
                "remote_port": endpoint.remote_port,
                "summary_path": str(endpoint.summary_path),
                "cleanup_command": endpoint.cleanup_command,
            }
            for endpoint in plan.endpoints
        },
    }


def endpoint_state_from_session(session: object) -> dict[str, object]:
    """Return registry endpoint state from a ready session object."""

    return {
        "lifecycle_state": "READY",
        "api_base": str(getattr(session, "api_base", "") or "").rstrip("/"),
        "served_model_name": str(
            getattr(session, "served_model_name", "") or getattr(session, "model", "") or ""
        ),
        "job_id": str(getattr(session, "job_id", "") or ""),
        "local_port": getattr(session, "local_port", None),
        "remote_port": getattr(session, "remote_port", None),
        "summary_path": str(getattr(session, "summary_path", "") or ""),
        "cleanup_command": str(getattr(session, "cleanup_command", "") or ""),
        "backend_kind": str(getattr(session, "backend_kind", "") or ""),
        "node": str(getattr(session, "node", "") or ""),
        "logs": str(getattr(session, "logs", "") or ""),
        "pid": getattr(session, "pid", None),
        "run_id": str(getattr(session, "run_id", "") or ""),
        "api_key_set": bool(str(getattr(session, "api_key", "") or "")),
    }


def endpoint_state_from_summary_payload(
    payload: Mapping[str, object],
    *,
    summary_path: str | Path,
) -> dict[str, object]:
    """Return registry endpoint state from a launch summary payload."""

    state = {
        "lifecycle_state": str(payload.get("lifecycle_state", "") or ""),
        "api_base": str(payload.get("local_api_base", "") or "").rstrip("/"),
        "served_model_name": str(payload.get("served_model_name", "") or ""),
        "job_id": str(payload.get("job_id", "") or ""),
        "local_port": payload.get("local_port"),
        "remote_port": payload.get("remote_port"),
        "summary_path": str(summary_path),
        "cleanup_command": str(payload.get("cleanup_command", "") or ""),
        "backend_kind": str(payload.get("backend_kind", "") or ""),
        "node": str(payload.get("node", "") or ""),
        "logs": str(payload.get("remote_log_dir", "") or payload.get("local_log_dir", "") or ""),
        "run_id": str(payload.get("run_id", "") or ""),
        "api_key_set": bool(payload.get("api_key_set")),
        "remote_state_path": str(payload.get("remote_state_path", "") or ""),
    }
    slurm = _slurm_state_from_summary_payload(payload)
    if slurm:
        state["slurm"] = slurm
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        state["diagnostics"] = dict(diagnostics)
    return state


def _slurm_state_from_summary_payload(payload: Mapping[str, object]) -> dict[str, object]:
    attempts = payload.get("resource_attempts")
    if not isinstance(attempts, list) or not attempts:
        return {}
    latest = attempts[-1]
    if not isinstance(latest, Mapping):
        return {}
    fields = {
        "partition": latest.get("partition"),
        "job_id": latest.get("job_id"),
        "latest_state": latest.get("latest_slurm_state"),
        "latest_reason": latest.get("latest_slurm_reason"),
        "latest_diagnostics": latest.get("latest_slurm_diagnostics"),
        "pending_duration_seconds": latest.get("pending_duration_seconds"),
    }
    return {name: value for name, value in fields.items() if value not in {"", None}}


def update_state(registry_dir: str | Path, **updates: object) -> None:
    """Update state.json under lock."""

    with registry_lock(registry_dir):
        state = read_state_locked(registry_dir)
        state.update(updates)
        state["updated_at"] = utc_now()
        write_json_atomic(Path(registry_dir) / "state.json", state)


def update_controller_pid(registry_dir: str | Path, pid: int) -> None:
    """Record the detached controller PID."""

    registry_path = Path(registry_dir)
    with registry_lock(registry_path):
        state = read_state_locked(registry_path)
        controller = dict(state.get("controller", {}) or {})
        controller.update({"mode": "detached", "pid": pid})
        state["controller"] = controller
        state["updated_at"] = utc_now()
        write_json_atomic(registry_path / "state.json", state)
        (registry_path / "controller.pid").write_text(f"{pid}\n", encoding="utf-8")


def update_controller_heartbeat(registry_dir: str | Path, pid: int) -> None:
    """Write heartbeat.json and mirror heartbeat time into state."""

    registry_path = Path(registry_dir)
    timestamp = utc_now()
    with registry_lock(registry_path):
        state = read_state_locked(registry_path)
        lifecycle = str(state.get("lifecycle_state", "") or "")
        controller = dict(state.get("controller", {}) or {})
        controller.update({"mode": "detached", "pid": pid, "heartbeat_at": timestamp})
        state["controller"] = controller
        state["updated_at"] = timestamp
        write_json_atomic(registry_path / "state.json", state)
        write_json_atomic(
            registry_path / "heartbeat.json",
            {
                "schema_version": "ril-heartbeat/v1",
                "run_id": str(state.get("run_id", "")),
                "pid": pid,
                "timestamp": timestamp,
                "lifecycle_state": lifecycle,
            },
        )


def reconcile_orphaned_controller(registry_dir: str | Path) -> None:
    """Mark a nonterminal detached run orphaned when its controller died."""

    registry_path = Path(registry_dir)
    state = read_json(registry_path / "state.json")
    lifecycle = str(state.get("lifecycle_state", "") or "")
    if lifecycle in TERMINAL_STATES:
        return
    controller = state.get("controller")
    if not isinstance(controller, Mapping) or controller.get("mode") != "detached":
        return
    pid = controller_pid(registry_path, state=state)
    if pid is not None and not pid_alive(pid):
        update_state(registry_path, lifecycle_state="ORPHANED")
        append_event(
            registry_path,
            event="controller_orphaned",
            level="error",
            payload={"pid": pid},
        )


def reconcile_terminal_endpoint_states(registry_dir: str | Path) -> None:
    """Mark endpoints expired when recorded Slurm state proves the job ended."""

    registry_path = Path(registry_dir)
    with registry_lock(registry_path):
        state = read_state_locked(registry_path)
        endpoints = state.get("endpoints", {})
        if not isinstance(endpoints, Mapping):
            return
        updated_endpoints: dict[str, object] = {}
        expired: list[tuple[str, Mapping[str, object], str, str]] = []
        for name, endpoint in sorted(endpoints.items()):
            if not isinstance(endpoint, Mapping):
                updated_endpoints[str(name)] = endpoint
                continue
            updated_endpoint = dict(endpoint)
            terminal_state = _terminal_slurm_state(updated_endpoint)
            lifecycle = str(updated_endpoint.get("lifecycle_state", "") or "")
            if terminal_state and lifecycle not in TERMINAL_STATES:
                reason = _terminal_slurm_reason(updated_endpoint)
                raw_diagnostics = updated_endpoint.get("diagnostics", {})
                diagnostics = dict(raw_diagnostics) if isinstance(raw_diagnostics, Mapping) else {}
                diagnostics.update(
                    {
                        "expiry_reason": "slurm_job_terminal",
                        "slurm_terminal_state": terminal_state,
                        "slurm_terminal_reason": reason,
                    }
                )
                updated_endpoint["diagnostics"] = diagnostics
                updated_endpoint["lifecycle_state"] = "EXPIRED"
                expired.append((str(name), updated_endpoint, terminal_state, reason))
            updated_endpoints[str(name)] = updated_endpoint
        if not expired:
            return
        state["endpoints"] = updated_endpoints
        root_lifecycle = str(state.get("lifecycle_state", "") or "")
        if len(updated_endpoints) == 1 and root_lifecycle not in TERMINAL_STATES:
            state["lifecycle_state"] = "EXPIRED"
        state["updated_at"] = utc_now()
        write_json_atomic(registry_path / "state.json", state)
        for name, endpoint, slurm_state, reason in expired:
            append_event_locked(
                registry_path,
                event="endpoint_expired",
                level="info",
                endpoint_name=name,
                payload={
                    "reason": "slurm_job_terminal",
                    "slurm_state": slurm_state,
                    "slurm_reason": reason,
                    "job_id": endpoint.get("job_id", ""),
                },
            )


def _terminal_slurm_state(endpoint: Mapping[str, object]) -> str:
    slurm = endpoint.get("slurm")
    if not isinstance(slurm, Mapping):
        return ""
    state = str(slurm.get("latest_state", "") or "").upper()
    return state if state in SLURM_TERMINAL_STATES else ""


def _terminal_slurm_reason(endpoint: Mapping[str, object]) -> str:
    slurm = endpoint.get("slurm")
    if not isinstance(slurm, Mapping):
        return ""
    return str(slurm.get("latest_reason", "") or "")


def controller_pid(
    registry_dir: str | Path,
    *,
    state: Mapping[str, object] | None = None,
) -> int | None:
    """Return the recorded controller PID, if any."""

    if state is None:
        state = read_json(Path(registry_dir) / "state.json")
    controller = state.get("controller")
    if isinstance(controller, Mapping):
        try:
            pid = int(controller.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0
        if pid > 0:
            return pid
    pid_path = Path(registry_dir) / "controller.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            return None
        return pid if pid > 0 else None
    return None


def pid_alive(pid: int) -> bool:
    """Return whether a process ID currently exists."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def format_logs(
    registry: Mapping[str, object],
    *,
    tail: int = 80,
    paths_only: bool = False,
) -> str:
    """Return controller and endpoint log paths plus recent controller lines."""

    registry_dir = Path(str(registry.get("registry_dir", "")))
    paths = log_paths(registry)
    if paths_only:
        return "\n".join(str(path) for path in paths) + ("\n" if paths else "")
    lines = ["Log paths:", *[f"  {path}" for path in paths]]
    controller_log = registry_dir / "controller.log"
    if controller_log.exists():
        lines.append("")
        lines.append(f"Last {tail} controller log lines:")
        lines.extend(_tail_lines(controller_log, tail))
    return "\n".join(lines) + "\n"


def log_paths(registry: Mapping[str, object]) -> list[Path]:
    """Return controller and endpoint log paths recorded for a run."""

    registry_dir = Path(str(registry.get("registry_dir", "")))
    paths = [registry_dir / "controller.log"]
    state = dict(registry.get("state", {}) or {})
    endpoints = state.get("endpoints", {})
    if isinstance(endpoints, Mapping):
        for endpoint in endpoints.values():
            if isinstance(endpoint, Mapping):
                log_path = str(endpoint.get("logs", "") or "")
                if log_path:
                    paths.append(Path(log_path))
    return paths


def append_event(
    registry_dir: str | Path,
    *,
    event: str,
    level: str,
    payload: Mapping[str, object],
    endpoint_name: str = "",
) -> None:
    """Append one registry event under lock."""

    with registry_lock(registry_dir):
        append_event_locked(
            registry_dir,
            event=event,
            level=level,
            payload=payload,
            endpoint_name=endpoint_name,
        )


def append_event_locked(
    registry_dir: str | Path,
    *,
    event: str,
    level: str,
    payload: Mapping[str, object],
    endpoint_name: str = "",
) -> None:
    """Append one registry event; caller must hold the registry lock."""

    registry_path = Path(registry_dir)
    event_path = registry_path / "events.jsonl"
    seq = 1
    if event_path.exists():
        with event_path.open(encoding="utf-8") as handle:
            seq = sum(1 for _line in handle) + 1
    state = read_state_locked(registry_path)
    record = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "seq": seq,
        "timestamp": utc_now(),
        "run_id": str(state.get("run_id", "")),
        "endpoint_name": endpoint_name or None,
        "event": event,
        "level": level,
        "payload": dict(payload),
    }
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


@contextmanager
def registry_lock(registry_dir: str | Path):
    """Hold an exclusive lock for a registry directory."""

    registry_path = Path(registry_dir)
    registry_path.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path / "lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_state_locked(registry_dir: str | Path) -> dict[str, object]:
    """Read state.json while the caller holds the registry lock."""

    state_path = Path(registry_dir) / "state.json"
    if not state_path.exists():
        return {}
    return read_json(state_path)


def read_json(path: str | Path) -> dict[str, object]:
    """Read a JSON object from disk."""

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def write_json_atomic(path: str | Path, payload: Mapping[str, object]) -> None:
    """Atomically write one JSON object."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def utc_now() -> str:
    """Return the current UTC timestamp for registry records."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _tail_lines(path: Path, count: int) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if count <= 0:
        return lines
    return lines[-count:]


def _registry_from_summary_path(path: Path) -> Path | None:
    parts = path.parts
    if "endpoints" in parts:
        index = parts.index("endpoints")
        candidate = Path(*parts[:index])
        if (candidate / "state.json").exists():
            return candidate
    for parent in path.parents:
        if (parent / "state.json").exists():
            return parent
    return None


def _endpoint_name_from_summary_path(path: Path) -> str:
    parts = path.parts
    if "endpoints" not in parts:
        return ""
    index = parts.index("endpoints")
    try:
        return parts[index + 1]
    except IndexError:
        return ""
