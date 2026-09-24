"""Reusable endpoint lease records and handoff helpers."""

from __future__ import annotations

import fcntl
import json
import shlex
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from remote_inference_launcher.health import endpoint_health_report
from remote_inference_launcher.plans import EffectiveLaunchPlan
from remote_inference_launcher.readiness import ReadinessConfig, check_openai_readiness
from remote_inference_launcher.registry import (
    append_event,
    cleanup_commands_from_state,
    env_text_from_state,
    read_json,
    read_registry,
    run_stop,
    update_state,
    utc_now,
    write_json_atomic,
)
from remote_inference_launcher.run_state import (
    READY_LIFECYCLES,
    RunStateReconciler,
    lease_endpoints_from_state,
)
from remote_inference_launcher.summaries import new_run_id

LEASE_SCHEMA_VERSION = "ril-lease/v1"
DEFAULT_LEASE_ROOT = ".remote-inference-launcher/leases"
ReadinessChecker = Callable[[Mapping[str, object]], None]
HealthChecker = Callable[[Mapping[str, object], str], dict[str, object]]


def new_lease_id() -> str:
    """Return a unique lease ID."""

    timestamp = new_run_id().rsplit("-", 1)[0]
    return f"lease-{timestamp}-{uuid.uuid4().hex[:8]}"


def create_pending_lease(
    plan: EffectiveLaunchPlan,
    *,
    ttl: str,
    ttl_start: str = "created_at",
    cleanup_on_expiry: bool = False,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
    lease_id: str | None = None,
) -> dict[str, object]:
    """Create a pending lease record for a planned launch."""

    resolved_lease_id = lease_id or new_lease_id()
    created_at = utc_now()
    ttl_delta = parse_ttl(ttl)
    _validate_ttl_start(ttl_start)
    record = {
        "schema_version": LEASE_SCHEMA_VERSION,
        "lease_id": resolved_lease_id,
        "run_id": plan.run_id,
        "registry_dir": str(plan.registry_dir),
        "semantic_config_hash": plan.semantic_config_hash,
        "created_at": created_at,
        "ready_at": None,
        "expires_at": _timestamp_after(created_at, ttl_delta) if ttl_start == "created_at" else "",
        "ttl_policy": {
            "mode": ttl_start,
            "duration_seconds": int(ttl_delta.total_seconds()),
        },
        "status": "pending",
        "cleanup_on_expiry": cleanup_on_expiry,
        "attach_count": 0,
        "last_attached_at": None,
        "endpoints": {},
    }
    write_lease(record, lease_root=lease_root)
    update_state(
        plan.registry_dir,
        lifecycle_state="PLANNED",
        lease={
            "lease_id": resolved_lease_id,
            "status": "pending",
            "expires_at": record["expires_at"],
            "cleanup_on_expiry": cleanup_on_expiry,
        },
    )
    return record


def activate_lease(
    lease_id: str,
    *,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
) -> dict[str, object]:
    """Mark a ready registry lease active."""

    record = read_lease(lease_id, lease_root=lease_root, refresh_expiry=False)
    registry = read_registry(record["registry_dir"])
    endpoints = lease_endpoints_from_state(registry["state"])
    if record.get("status") != "active":
        record = _activated_lease_record(record, endpoints=endpoints)
    else:
        record = {**record, "endpoints": endpoints}
    write_lease(record, lease_root=lease_root)
    update_state(
        record["registry_dir"],
        lifecycle_state="LEASED",
        lease={
            "lease_id": record["lease_id"],
            "status": "active",
            "expires_at": record["expires_at"],
            "cleanup_on_expiry": record["cleanup_on_expiry"],
        },
    )
    return record


def recover_lease(
    value: str | Path,
    *,
    expect_config: str | Path | None = None,
    expected_model: str = "",
    health_depth: str = "models",
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
    health_checker: HealthChecker | None = None,
) -> dict[str, object]:
    """Recover a lease whose parent process timed out before readiness."""

    record = read_lease(value, lease_root=lease_root, refresh_expiry=False)
    if expect_config is not None:
        _validate_expected_config(record, expect_config)
    registry = read_registry(record["registry_dir"])
    state = registry.get("state", {})
    if not isinstance(state, Mapping):
        raise RuntimeError("Lease registry state is not a JSON object.")
    lifecycle = str(state.get("lifecycle_state", "") or "")
    if lifecycle not in READY_LIFECYCLES and not lease_endpoints_from_state(state):
        raise RuntimeError(f"Lease registry is not recoverable yet: {lifecycle or 'unknown'}.")
    endpoints = lease_endpoints_from_state(state)
    if not endpoints:
        raise RuntimeError("Lease registry has no endpoint handoff metadata.")
    checks = _lease_recovery_checks(record, state=state, endpoints=endpoints)
    failed_checks = [check for check in checks if not check.get("ok")]
    if failed_checks:
        raise RuntimeError(f"Lease recovery checks failed: {failed_checks!r}")
    health_reports: dict[str, dict[str, object]] = {}
    for name, endpoint in endpoints.items():
        model = expected_model or str(endpoint.get("served_model_name", "") or "")
        health = (
            health_checker(endpoint, model)
            if health_checker is not None
            else _recovery_health_report(endpoint, name=name, depth=health_depth, model=model)
        )
        health_reports[name] = health
        if health.get("status") != "healthy":
            raise RuntimeError(
                f"Lease endpoint {name} is not healthy: {health.get('reason_code', 'unknown')}"
            )
    active = _activated_lease_record(record, endpoints=endpoints)
    active["recovered_at"] = utc_now()
    active["recovery"] = {
        "checks": checks,
        "health": health_reports,
        "source_status": record.get("status", ""),
        "registry_lifecycle_state": lifecycle,
    }
    write_lease(active, lease_root=lease_root)
    _update_registry_lease_status(active)
    return {
        "schema_version": "ril-lease-recovery/v1",
        "lease": active,
        "handoff": handoff_payload(active),
        "health": health_reports,
    }


def read_lease(
    value: str | Path,
    *,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
    refresh_expiry: bool = True,
) -> dict[str, object]:
    """Read a lease by ID or path, optionally refreshing expiry state."""

    lease_path = resolve_lease_path(value, lease_root=lease_root)
    record = read_json(lease_path)
    record = _reconcile_lease(record, lease_root=lease_path.parent)
    if refresh_expiry and _lease_is_expired(record):
        record = _expired_lease_record(record)
        write_lease(record, lease_root=lease_path.parent)
        _record_registry_lease_expired(record)
    return record


def write_lease(record: Mapping[str, object], *, lease_root: str | Path) -> None:
    """Write a lease record atomically under lock."""

    lease_id = str(record["lease_id"])
    lease_path = Path(lease_root).expanduser() / f"{lease_id}.json"
    with lease_lock(lease_path):
        write_json_atomic(lease_path, record)


def record_lease_ready_timeout(
    record: Mapping[str, object],
    error: BaseException,
    *,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
) -> dict[str, object]:
    """Record a parent-side wait-ready timeout without marking the lease failed."""

    updated = {
        **dict(record),
        "status": "ready_timeout",
        "ready_timeout": {
            "code": type(error).__name__,
            "message": str(error),
            "recorded_at": utc_now(),
        },
    }
    write_lease(updated, lease_root=lease_root)
    update_state(
        updated["registry_dir"],
        lease={
            "lease_id": updated.get("lease_id"),
            "status": "ready_timeout",
            "expires_at": updated.get("expires_at"),
            "cleanup_on_expiry": updated.get("cleanup_on_expiry"),
        },
    )
    append_event(
        updated["registry_dir"],
        event="lease_ready_timeout",
        level="warning",
        payload={"lease_id": updated.get("lease_id"), "message": str(error)},
    )
    return updated


def resolve_lease_path(
    value: str | Path,
    *,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
) -> Path:
    """Resolve a lease ID or lease JSON path."""

    path = Path(value)
    if path.is_file():
        return path
    candidate = Path(lease_root).expanduser() / f"{value}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Lease not found: {value}")


def format_lease_status(record: Mapping[str, object]) -> str:
    """Return human-readable lease status."""

    lines = [
        f"lease_id={record.get('lease_id', '')}",
        f"run_id={record.get('run_id', '')}",
        f"status={record.get('status', '')}",
        f"expires_at={record.get('expires_at', '')}",
        f"registry={record.get('registry_dir', '')}",
        f"attach_count={record.get('attach_count', 0)}",
    ]
    failure = record.get("failure")
    if isinstance(failure, Mapping):
        lines.append(f"failure={json.dumps(dict(failure), sort_keys=True)}")
    for name, endpoint in sorted(_endpoint_records(record).items()):
        display_endpoint = dict(endpoint)
        if _is_self_registry_stop_command(
            str(display_endpoint.get("cleanup_command", "") or ""),
            registry_dir=str(record.get("registry_dir", "") or ""),
        ):
            display_endpoint["cleanup_command"] = ""
        lines.append(f"endpoint={_format_endpoint_status(name, display_endpoint)}")
    if record.get("status") == "expired" and record.get("cleanup_on_expiry"):
        lines.append("cleanup=overdue")
    return "\n".join(lines)


def lease_env_text(record: Mapping[str, object], *, endpoint_name: str = "") -> str:
    """Return non-secret env exports from a lease record."""

    state = {"endpoints": record.get("endpoints", {})}
    return env_text_from_state(state, endpoint_name=endpoint_name)


def attach_lease(
    value: str | Path,
    *,
    expect_config: str | Path | None = None,
    allow_expired: bool = False,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
    readiness_checker: ReadinessChecker | None = None,
) -> dict[str, object]:
    """Attach to an active lease and return the handoff payload."""

    record = read_lease(value, lease_root=lease_root)
    if record["status"] == "expired" and allow_expired:
        record = {**record, "status": "active"}
    elif record["status"] == "expired":
        raise RuntimeError(f"Lease {record['lease_id']} is expired.")
    elif record["status"] != "active":
        raise RuntimeError(f"Lease {record['lease_id']} is not active: {record['status']}")
    if expect_config is not None:
        _validate_expected_config(record, expect_config)
    checker = readiness_checker or lightweight_readiness_check
    for endpoint in _endpoint_records(record).values():
        checker(endpoint)
    updated = {
        **record,
        "attach_count": int(record.get("attach_count", 0)) + 1,
        "last_attached_at": utc_now(),
    }
    write_lease(updated, lease_root=lease_root)
    _update_registry_lease_status(updated)
    return handoff_payload(updated)


def stop_lease(
    value: str | Path,
    *,
    force: bool = False,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
) -> int:
    """Stop a lease by delegating to the recorded registry cleanup commands."""

    record = read_lease(value, lease_root=lease_root, refresh_expiry=False)
    stopping = {**record, "status": "stopping"}
    write_lease(stopping, lease_root=lease_root)
    _update_registry_lease_status(stopping)
    status = run_stop(str(record["registry_dir"]), force=force)
    stopped = {**stopping, "status": "stopped" if status == 0 else "failed"}
    write_lease(stopped, lease_root=lease_root)
    _update_registry_lease_status(stopped)
    return status


def format_lease_stop_plan(
    value: str | Path,
    *,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
) -> str:
    """Return a dry-run cleanup plan for a lease."""

    record = read_lease(value, lease_root=lease_root, refresh_expiry=False)
    registry = read_registry(str(record["registry_dir"]))
    state = registry.get("state", {})
    state_mapping = state if isinstance(state, Mapping) else {}
    controller = state_mapping.get("controller", {})
    commands = cleanup_commands_from_state(
        state_mapping,
        registry_dir=str(record["registry_dir"]),
    )
    lines = [
        f"lease_id={record.get('lease_id', '')}",
        f"lease_status={record.get('status', '')}",
        f"registry={record.get('registry_dir', '')}",
        f"registry_state={state_mapping.get('lifecycle_state', '')}",
    ]
    if isinstance(controller, Mapping):
        lines.append(
            "controller="
            f"mode={controller.get('mode', '')} "
            f"pid={controller.get('pid', '')} "
            f"heartbeat_at={controller.get('heartbeat_at', '')}"
        )
    for name, endpoint in sorted(_endpoint_records(record).items()):
        display_endpoint = dict(endpoint)
        if _is_self_registry_stop_command(
            str(display_endpoint.get("cleanup_command", "") or ""),
            registry_dir=str(record.get("registry_dir", "") or ""),
        ):
            display_endpoint["cleanup_command"] = ""
        lines.append(f"endpoint={_format_endpoint_status(name, display_endpoint)}")
    lines.append(f"cleanup_commands={len(commands)}")
    for command in commands:
        lines.append(f"cleanup_command={command}")
    return "\n".join(lines)


def handoff_payload(record: Mapping[str, object]) -> dict[str, object]:
    """Return machine-readable lease handoff JSON."""

    return {
        "schema_version": "ril-handoff/v1",
        "lease_id": record["lease_id"],
        "run_id": record["run_id"],
        "semantic_config_hash": record["semantic_config_hash"],
        "identity_key": f"lease:{record['lease_id']}",
        "cleanup_owner": "lease",
        "endpoints": {
            name: {
                "api_base": endpoint.get("api_base", ""),
                "model": endpoint.get("served_model_name", ""),
                "served_model_name": endpoint.get("served_model_name", ""),
                "summary_path": endpoint.get("summary_path", ""),
            }
            for name, endpoint in _endpoint_records(record).items()
        },
    }


def benchmark_metadata_from_handoff(handoff: Mapping[str, object]) -> dict[str, object]:
    """Return stable benchmark metadata from a lease handoff payload."""

    endpoints = handoff.get("endpoints", {})
    first_endpoint = next(iter(endpoints.values()), {}) if isinstance(endpoints, Mapping) else {}
    served_model = (
        str(first_endpoint.get("served_model_name", "") or first_endpoint.get("model", ""))
        if isinstance(first_endpoint, Mapping)
        else ""
    )
    lease_id = str(handoff.get("lease_id", "") or "")
    return {
        "inference_lease_id": lease_id,
        "inference_run_id": str(handoff.get("run_id", "") or ""),
        "inference_semantic_config_hash": str(handoff.get("semantic_config_hash", "") or ""),
        "inference_identity_key": f"lease:{lease_id}",
        "inference_cleanup_owner": "lease",
        "inference_served_model_name": served_model,
        "inference_sessions": {
            "endpoints": dict(endpoints) if isinstance(endpoints, Mapping) else {}
        },
    }


def _lease_recovery_checks(
    record: Mapping[str, object],
    *,
    state: Mapping[str, object],
    endpoints: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    plan_hash = str(record.get("semantic_config_hash", "") or "")
    state_lease = state.get("lease", {})
    state_lease_id = (
        str(state_lease.get("lease_id", "") or "") if isinstance(state_lease, Mapping) else ""
    )
    checks: list[dict[str, object]] = [
        {
            "check": "lease_owner_matches_registry",
            "ok": state_lease_id in {"", str(record.get("lease_id", "") or "")},
            "registry_lease_id": state_lease_id,
        },
        {
            "check": "launcher_config_hash_present",
            "ok": bool(plan_hash),
            "semantic_config_hash": plan_hash,
        },
    ]
    for name, endpoint in sorted(endpoints.items()):
        backend_kind = str(endpoint.get("backend_kind", "") or "")
        api_base = str(endpoint.get("api_base", "") or "")
        served_model = str(endpoint.get("served_model_name", "") or "")
        job_id = str(endpoint.get("job_id", "") or "")
        local_port = endpoint.get("local_port")
        checks.extend(
            [
                {
                    "check": "endpoint_url_present",
                    "ok": bool(api_base),
                    "endpoint": name,
                    "api_base": api_base,
                },
                {
                    "check": "model_identity_present",
                    "ok": bool(served_model),
                    "endpoint": name,
                    "served_model_name": served_model,
                },
                {
                    "check": "slurm_job_identity_present",
                    "ok": backend_kind != "slurm_vllm" or bool(job_id),
                    "endpoint": name,
                    "backend_kind": backend_kind,
                    "job_id": job_id,
                },
                {
                    "check": "tunnel_identity_present",
                    "ok": backend_kind != "slurm_vllm" or local_port not in {"", None},
                    "endpoint": name,
                    "backend_kind": backend_kind,
                    "local_port": local_port,
                },
            ]
        )
    return checks


def _recovery_health_report(
    endpoint: Mapping[str, object],
    *,
    name: str,
    depth: str,
    model: str,
) -> dict[str, object]:
    return endpoint_health_report(
        {
            **dict(endpoint),
            "endpoint_id": name,
            "lease_status": "active",
        },
        depth=depth,
        expected_model=model,
    )


def lightweight_readiness_check(endpoint: Mapping[str, object]) -> None:
    """Run a lightweight readiness check for a lease endpoint."""

    api_base = str(endpoint.get("api_base", "") or "")
    model = str(endpoint.get("served_model_name", "") or endpoint.get("model", "") or "")
    if not api_base:
        raise RuntimeError("Lease endpoint is missing api_base.")
    result = check_openai_readiness(
        api_base,
        model=model,
        config=ReadinessConfig(smoke_test="disabled"),
    )
    if not result.ok:
        raise RuntimeError(f"Lease endpoint is not ready: {result.failure_message}")


def parse_ttl(value: str) -> timedelta:
    """Parse a compact TTL such as 30m, 12h, or 2d."""

    stripped = value.strip().lower()
    if not stripped:
        raise ValueError("Lease TTL must be non-empty.")
    unit = stripped[-1]
    number_text = stripped[:-1] if unit.isalpha() else stripped
    if not number_text.isdigit():
        raise ValueError(f"Unsupported lease TTL: {value!r}.")
    amount = int(number_text)
    if amount < 1:
        raise ValueError("Lease TTL must be positive.")
    if unit == "s" or not unit.isalpha():
        return timedelta(seconds=amount)
    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    raise ValueError(f"Unsupported lease TTL unit: {unit!r}.")


def _validate_ttl_start(value: str) -> None:
    if value not in {"created_at", "ready_at"}:
        raise ValueError("Lease ttl_start must be 'created_at' or 'ready_at'.")


def _activated_lease_record(
    record: Mapping[str, object],
    *,
    endpoints: Mapping[str, object],
) -> dict[str, object]:
    ready_at = utc_now()
    return {
        **dict(record),
        "status": "active",
        "ready_at": ready_at,
        "expires_at": _activated_expires_at(record, ready_at=ready_at),
        "endpoints": dict(endpoints),
    }


def _activated_expires_at(record: Mapping[str, object], *, ready_at: str) -> str:
    ttl_policy = record["ttl_policy"]
    if not isinstance(ttl_policy, Mapping):
        raise ValueError("Lease ttl_policy must be a JSON object.")
    mode = str(ttl_policy["mode"])
    duration_seconds = int(ttl_policy["duration_seconds"])
    if duration_seconds < 1:
        raise ValueError("Lease ttl_policy.duration_seconds must be positive.")
    if mode == "ready_at":
        return _timestamp_after(ready_at, timedelta(seconds=duration_seconds))
    if mode == "created_at":
        expires_at = str(record["expires_at"])
        if not expires_at:
            raise ValueError("created_at TTL leases require expires_at.")
        return expires_at
    raise ValueError("Lease ttl_policy.mode must be 'created_at' or 'ready_at'.")


def _validate_expected_config(record: Mapping[str, object], expect_config: str | Path) -> None:
    from remote_inference_launcher.plans import build_effective_launch_plan_from_path

    expected = build_effective_launch_plan_from_path(expect_config, ownership_policy="lease")
    if expected.semantic_config_hash != record["semantic_config_hash"]:
        raise RuntimeError(
            "Lease semantic config hash mismatch: "
            f"expected {expected.semantic_config_hash}, lease {record['semantic_config_hash']}."
        )


def _timestamp_after(timestamp: str, delta: timedelta) -> str:
    return (
        (datetime.fromisoformat(timestamp.replace("Z", "+00:00")) + delta)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _reconcile_lease(
    record: dict[str, object],
    *,
    lease_root: str | Path,
) -> dict[str, object]:
    reconciled = RunStateReconciler().reconcile_lease(record)
    if record.get("status") != "active" and reconciled.get("status") == "active":
        reconciled = _activated_lease_record(
            reconciled,
            endpoints=_endpoint_records(reconciled),
        )
    if reconciled == record:
        return record
    write_lease(reconciled, lease_root=lease_root)
    _update_registry_lease_status(reconciled)
    return reconciled


def _lease_is_expired(record: Mapping[str, object]) -> bool:
    if record.get("status") != "active":
        return False
    expires_at = str(record.get("expires_at", "") or "")
    if not expires_at:
        return False
    expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    return expires <= datetime.now(UTC)


def _expired_lease_record(record: Mapping[str, object]) -> dict[str, object]:
    endpoints = {
        name: _expired_endpoint_record(endpoint)
        for name, endpoint in _endpoint_records(record).items()
    }
    return {
        **dict(record),
        "status": "expired",
        "endpoints": endpoints,
    }


def _expired_endpoint_record(endpoint: Mapping[str, object]) -> dict[str, object]:
    updated = dict(endpoint)
    if str(updated.get("lifecycle_state", "") or "") in {"", "READY", "LEASED"}:
        updated["lifecycle_state"] = "EXPIRED"
    return updated


def _endpoint_records(record: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    endpoints = record.get("endpoints", {})
    if not isinstance(endpoints, Mapping):
        return {}
    return {
        str(name): endpoint for name, endpoint in endpoints.items() if isinstance(endpoint, Mapping)
    }


def _format_endpoint_status(name: str, endpoint: Mapping[str, object]) -> str:
    parts = [name]
    for field in (
        "lifecycle_state",
        "api_base",
        "served_model_name",
        "job_id",
        "local_port",
        "remote_port",
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


def _update_registry_lease_status(record: Mapping[str, object]) -> None:
    registry_dir = str(record.get("registry_dir", "") or "")
    if not registry_dir:
        return
    lifecycle = "LEASED" if record.get("status") == "active" else None
    updates: dict[str, object] = {
        "lease": {
            "lease_id": record.get("lease_id"),
            "status": record.get("status"),
            "expires_at": record.get("expires_at"),
            "cleanup_on_expiry": record.get("cleanup_on_expiry"),
        }
    }
    if lifecycle:
        updates["lifecycle_state"] = lifecycle
    update_state(registry_dir, **updates)


def _record_registry_lease_expired(record: Mapping[str, object]) -> None:
    registry_dir = str(record.get("registry_dir", "") or "")
    if not registry_dir:
        return
    state = read_json(Path(registry_dir) / "state.json")
    registry_endpoints = state.get("endpoints", {})
    endpoints = dict(registry_endpoints) if isinstance(registry_endpoints, Mapping) else {}
    for name, endpoint in _endpoint_records(record).items():
        existing_endpoint = endpoints.get(name, {})
        current = dict(existing_endpoint) if isinstance(existing_endpoint, Mapping) else {}
        current.update(dict(endpoint))
        current["lifecycle_state"] = "EXPIRED"
        endpoints[name] = current
    update_state(
        registry_dir,
        lifecycle_state="EXPIRED",
        endpoints=endpoints,
        lease={
            "lease_id": record.get("lease_id"),
            "status": "expired",
            "expires_at": record.get("expires_at"),
            "cleanup_on_expiry": record.get("cleanup_on_expiry"),
        },
    )
    append_event(
        registry_dir,
        event="lease_expired",
        level="info",
        payload={"lease_id": record.get("lease_id")},
    )


def _is_self_registry_stop_command(command: str, *, registry_dir: str) -> bool:
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    return (
        len(parts) == 3
        and parts[:2] == ["remote-inference-launcher", "stop"]
        and (parts[2] == registry_dir)
    )


@contextmanager
def lease_lock(lease_path: Path):
    """Hold an exclusive lock for one lease file."""

    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = lease_path.with_suffix(lease_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
