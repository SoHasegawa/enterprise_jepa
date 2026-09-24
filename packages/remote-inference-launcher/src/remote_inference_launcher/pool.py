"""Endpoint pool indexing, assignment, health, and manifest helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from remote_inference_launcher.capabilities import capability_matches, capability_payload
from remote_inference_launcher.health import endpoint_health_report
from remote_inference_launcher.leases import (
    DEFAULT_LEASE_ROOT,
    parse_ttl,
    read_lease,
    write_lease,
)
from remote_inference_launcher.plans import DEFAULT_REGISTRY_ROOT
from remote_inference_launcher.pool_assignments import (
    ACTIVE_ASSIGNMENT_STATES,
    _active_assignment,
    _assignment_expires_at,
    _assignment_lock,
    _assignment_root,
    _assignment_state,
    _new_assignment_id,
    _new_batch_id,
    _resolve_batch_shards,
    load_assignments,
    read_assignment,
)
from remote_inference_launcher.registry import (
    read_json,
    read_registry,
    update_state,
    utc_now,
    write_json_atomic,
)
from remote_inference_launcher.remote_execution import CommandRunner
from remote_inference_launcher.scheduler_status import refresh_slurm_scheduler
from remote_inference_launcher.tunnels import DEFAULT_POOL_ROOT, reconnect_slurm_tunnel

POOL_STATUS_SCHEMA_VERSION = "ril-pool-status/v1"
POOL_HANDOFF_SCHEMA_VERSION = "ril-pool-handoff/v1"
POOL_BATCH_HANDOFF_SCHEMA_VERSION = "ril-pool-batch-handoff/v1"
ASSIGNMENT_SCHEMA_VERSION = "ril-assignment/v1"
POOL_MANIFEST_SCHEMA_VERSION = "ril-pool-manifest/v1"
HealthChecker = Callable[[Mapping[str, object]], dict[str, object]]


def build_pool_status(
    *,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
    pool_root: str | Path = DEFAULT_POOL_ROOT,
    health_depth: str = "cached",
    include_registry_endpoints: bool = True,
    scheduler_command_runner: CommandRunner | None = None,
    useful_deadline: str = "",
) -> dict[str, object]:
    """Return a pool-wide status payload."""

    endpoints = load_pool_endpoints(
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
        include_registry_endpoints=include_registry_endpoints,
    )
    if health_depth in {"scheduler", "light", "models", "full"}:
        scheduler = refresh_slurm_scheduler(
            tuple(endpoints),
            command_runner=scheduler_command_runner,
            useful_deadline=useful_deadline,
        )
        endpoints = [_with_scheduler(endpoint, scheduler) for endpoint in endpoints]
    status_depth = _status_health_depth(health_depth)
    endpoints = [
        _with_health(endpoint, depth=status_depth if status_depth else "cached")
        for endpoint in endpoints
    ]
    endpoints = [_with_acquirability(endpoint) for endpoint in endpoints]
    return {
        "schema_version": POOL_STATUS_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "summary": _pool_summary(endpoints),
        "endpoints": endpoints,
    }


def load_pool_endpoints(
    *,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
    pool_root: str | Path = DEFAULT_POOL_ROOT,
    include_registry_endpoints: bool = True,
) -> list[dict[str, object]]:
    """Return normalized pool endpoints from known leases and registries."""

    assignments = load_assignments(pool_root=pool_root)
    endpoints: list[dict[str, object]] = []
    covered: set[tuple[str, str]] = set()
    for lease_path in _lease_paths(lease_root):
        try:
            lease = read_lease(lease_path, lease_root=lease_path.parent)
            registry = read_registry(str(lease.get("registry_dir", "")))
        except (OSError, RuntimeError, ValueError):
            continue
        state = _mapping(registry.get("state"))
        for endpoint_name, endpoint in _endpoint_records(state).items():
            endpoints.append(
                _pool_endpoint(
                    source_kind="lease",
                    endpoint_name=endpoint_name,
                    endpoint=endpoint,
                    registry=registry,
                    lease=lease,
                    assignments=assignments,
                )
            )
            covered.add((str(lease.get("run_id", "") or ""), endpoint_name))
    if include_registry_endpoints:
        for registry_path in _registry_paths(registry_root):
            try:
                registry = read_registry(registry_path)
            except (OSError, RuntimeError, ValueError):
                continue
            state = _mapping(registry.get("state"))
            run_id = str(state.get("run_id", "") or "")
            for endpoint_name, endpoint in _endpoint_records(state).items():
                if (run_id, endpoint_name) in covered:
                    continue
                endpoints.append(
                    _pool_endpoint(
                        source_kind="registry",
                        endpoint_name=endpoint_name,
                        endpoint=endpoint,
                        registry=registry,
                        lease={},
                        assignments=assignments,
                    )
                )
    return sorted(endpoints, key=lambda item: str(item.get("endpoint_id", "")))


def acquire_endpoint(
    *,
    model: str = "",
    min_context: int | None = None,
    min_remaining: str = "",
    owner: str,
    shard_id: str = "",
    batch_id: str = "",
    assignment_ttl: str = "",
    health_depth: str = "full",
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
    pool_root: str | Path = DEFAULT_POOL_ROOT,
    health_checker: HealthChecker | None = None,
) -> dict[str, object]:
    """Atomically acquire a healthy free endpoint from the pool."""

    if not owner:
        raise ValueError("pool acquire requires owner.")
    last_failure = ""
    while True:
        endpoints = load_pool_endpoints(
            lease_root=lease_root,
            registry_root=registry_root,
            pool_root=pool_root,
            include_registry_endpoints=True,
        )
        candidates = [
            endpoint
            for endpoint in endpoints
            if _candidate_matches(
                endpoint,
                model=model,
                min_context=min_context,
                min_remaining=min_remaining,
            )
        ]
        if not candidates:
            message = "No free endpoint satisfies the requested pool constraints."
            failure_summary = _eligibility_failure_summary(
                endpoints,
                model=model,
                min_context=min_context,
                min_remaining=min_remaining,
            )
            if failure_summary:
                message = f"{message} Eligibility failures: {failure_summary}."
            if last_failure:
                message = f"{message} Last failed health reason: {last_failure}."
            raise RuntimeError(message)
        for endpoint in candidates:
            assignment = _create_probe_assignment(
                endpoint,
                owner=owner,
                shard_id=shard_id,
                batch_id=batch_id,
                assignment_ttl=assignment_ttl,
                health_depth=health_depth,
                requested_capabilities={"model": model, "min_context": min_context},
                pool_root=pool_root,
            )
            if not assignment:
                continue
            if health_checker is None:
                health = endpoint_health_report(
                    endpoint,
                    depth=health_depth,
                    assignment=assignment,
                    capability=_mapping(endpoint.get("capabilities")),
                    expected_model=model,
                    min_context=min_context,
                )
            else:
                health = health_checker(endpoint)
            if health.get("status") == "healthy":
                active = _update_assignment_status(
                    str(assignment["assignment_id"]),
                    "active",
                    pool_root=pool_root,
                    health=health,
                )
                return pool_handoff(endpoint, active)
            last_failure = str(health.get("reason_code", "") or "unknown")
            _update_assignment_status(
                str(assignment["assignment_id"]),
                "failed",
                pool_root=pool_root,
                health=health,
            )


def acquire_batch(
    *,
    owner: str,
    shard_ids: list[str],
    model: str = "",
    count: int | None = None,
    min_context: int | None = None,
    min_remaining: str = "",
    assignment_ttl: str = "",
    health_depth: str = "full",
    partial: bool = False,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
    pool_root: str | Path = DEFAULT_POOL_ROOT,
    health_checker: HealthChecker | None = None,
) -> dict[str, object]:
    """Acquire a batch of endpoint assignments for deterministic shards."""

    if not owner:
        raise ValueError("pool acquire-batch requires owner.")
    resolved_shards = _resolve_batch_shards(shard_ids, count=count)
    batch_id = _new_batch_id()
    assignments: list[dict[str, object]] = []
    missing_shards: list[dict[str, object]] = []
    try:
        for shard_id in resolved_shards:
            try:
                assignments.append(
                    acquire_endpoint(
                        model=model,
                        min_context=min_context,
                        min_remaining=min_remaining,
                        owner=owner,
                        shard_id=shard_id,
                        batch_id=batch_id,
                        assignment_ttl=assignment_ttl,
                        health_depth=health_depth,
                        lease_root=lease_root,
                        registry_root=registry_root,
                        pool_root=pool_root,
                        health_checker=health_checker,
                    )
                )
            except (OSError, RuntimeError, ValueError) as error:
                if not partial:
                    raise
                missing_shards.append({"shard_id": shard_id, "error": str(error)})
        if missing_shards and not partial:
            raise RuntimeError("Batch acquisition failed.")
    except Exception:
        for assignment in assignments:
            release_assignment(
                assignment_id=str(assignment.get("assignment_id", "") or ""),
                pool_root=pool_root,
            )
        raise
    return {
        "schema_version": POOL_BATCH_HANDOFF_SCHEMA_VERSION,
        "batch_id": batch_id,
        "owner": owner,
        "requested_count": len(resolved_shards),
        "acquired_count": len(assignments),
        "partial": partial,
        "assignments": assignments,
        "missing_shards": missing_shards,
        "release_command": f"remote-inference-launcher pool release-batch {batch_id}",
    }


def release_assignment(
    *,
    assignment_id: str = "",
    endpoint_id: str = "",
    owner: str = "",
    pool_root: str | Path = DEFAULT_POOL_ROOT,
) -> dict[str, object]:
    """Release a pool assignment without stopping its endpoint."""

    assignment = _resolve_assignment(
        assignment_id=assignment_id,
        endpoint_id=endpoint_id,
        owner=owner,
        pool_root=pool_root,
    )
    if not assignment:
        raise FileNotFoundError("Assignment not found.")
    return _update_assignment_status(
        str(assignment["assignment_id"]),
        "released",
        pool_root=pool_root,
    )


def release_batch(
    *,
    batch_id: str = "",
    assignment_ids: list[str] | None = None,
    owner: str = "",
    pool_root: str | Path = DEFAULT_POOL_ROOT,
) -> dict[str, object]:
    """Release a batch of assignments without stopping endpoints."""

    assignment_ids = assignment_ids or []
    if not batch_id and not assignment_ids:
        raise ValueError("release-batch requires a batch ID or at least one assignment ID.")
    selected: list[dict[str, object]] = []
    assignments = load_assignments(pool_root=pool_root)
    for assignment_id in assignment_ids:
        assignment = read_assignment(assignment_id, pool_root=pool_root)
        _enforce_assignment_owner(assignment, owner=owner, assignment_id=assignment_id)
        selected.append(assignment)
    if batch_id:
        for assignment in assignments.values():
            if assignment.get("batch_id") != batch_id:
                continue
            if owner and assignment.get("owner") != owner:
                continue
            if assignment not in selected:
                selected.append(dict(assignment))
    released: list[dict[str, object]] = []
    for assignment in selected:
        released.append(
            _update_assignment_status(
                str(assignment["assignment_id"]),
                "released",
                pool_root=pool_root,
            )
        )
    return {
        "schema_version": "ril-pool-batch-release/v1",
        "batch_id": batch_id,
        "released_count": len(released),
        "released": released,
    }


def pool_health_signal(
    *,
    endpoint_id: str = "",
    assignment_id: str = "",
    depth: str = "cached",
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
    pool_root: str | Path = DEFAULT_POOL_ROOT,
) -> dict[str, object]:
    """Return the wrapper-facing health signal for one endpoint or assignment."""

    assignment = {}
    if assignment_id:
        assignment = read_assignment(assignment_id, pool_root=pool_root)
        endpoint_id = str(assignment.get("endpoint_id", "") or endpoint_id)
    endpoint = _find_endpoint(
        endpoint_id,
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
    )
    return endpoint_health_report(
        endpoint,
        depth=depth,
        assignment=assignment,
        capability=_mapping(endpoint.get("capabilities")),
    )


def pool_manifest(
    *,
    endpoint_id: str = "",
    assignment_id: str = "",
    available: bool = False,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
    pool_root: str | Path = DEFAULT_POOL_ROOT,
) -> dict[str, object]:
    """Return a benchmark-consumable pool manifest."""

    assignment = {}
    if assignment_id:
        assignment = read_assignment(assignment_id, pool_root=pool_root)
        endpoint_id = str(assignment.get("endpoint_id", "") or endpoint_id)
    endpoints = load_pool_endpoints(
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
        include_registry_endpoints=not available,
    )
    if endpoint_id:
        endpoints = [
            endpoint for endpoint in endpoints if endpoint.get("endpoint_id") == endpoint_id
        ]
    if available:
        endpoints = [endpoint for endpoint in endpoints if _endpoint_acquirability(endpoint)[0]]
    return {
        "schema_version": POOL_MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "assignment": assignment,
        "endpoints": endpoints,
    }


def reconnect_endpoint(
    *,
    endpoint_id: str = "",
    assignment_id: str = "",
    preferred_local_port: int | None = None,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
    pool_root: str | Path = DEFAULT_POOL_ROOT,
    scheduler_command_runner: CommandRunner | None = None,
) -> dict[str, object]:
    """Reconnect a leased Slurm endpoint's local tunnel without stopping Slurm."""

    assignment = {}
    if assignment_id:
        assignment = read_assignment(assignment_id, pool_root=pool_root)
        endpoint_id = str(assignment.get("endpoint_id", "") or endpoint_id)
    endpoint = _find_endpoint(
        endpoint_id,
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
    )
    scheduler = refresh_slurm_scheduler(
        (endpoint,),
        command_runner=scheduler_command_runner,
    ).get(endpoint_id, {})
    if str(scheduler.get("slurm_state", "") or "").upper() in {
        "NOT_FOUND",
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
    }:
        raise RuntimeError(
            "Cannot reconnect tunnel because Slurm job is not running: "
            f"{scheduler.get('slurm_state', '')}."
        )
    record = reconnect_slurm_tunnel(
        endpoint,
        pool_root=pool_root,
        preferred_local_port=preferred_local_port,
    )
    updated_endpoint = {
        **endpoint,
        "api_base": record.get("api_base", ""),
        "local_port": record.get("local_port"),
    }
    _update_endpoint_api_base(updated_endpoint, lease_root=lease_root)
    health = endpoint_health_report(
        updated_endpoint,
        depth="full",
        assignment=assignment,
        capability=_mapping(endpoint.get("capabilities")),
    )
    return {
        "schema_version": "ril-pool-reconnect/v1",
        "endpoint_id": endpoint_id,
        "assignment_id": assignment.get("assignment_id", "") if assignment else "",
        "tunnel": record,
        "health": health,
    }


def pool_handoff(
    endpoint: Mapping[str, object],
    assignment: Mapping[str, object],
) -> dict[str, object]:
    """Return the pool handoff payload for an acquired endpoint."""

    return {
        "schema_version": POOL_HANDOFF_SCHEMA_VERSION,
        "assignment_id": assignment.get("assignment_id", ""),
        "batch_id": assignment.get("batch_id", ""),
        "endpoint_id": endpoint.get("endpoint_id", ""),
        "lease_id": endpoint.get("lease_id", ""),
        "run_id": endpoint.get("run_id", ""),
        "cleanup_owner": endpoint.get("cleanup_owner", ""),
        "api_base": endpoint.get("api_base", ""),
        "served_model_name": endpoint.get("served_model_name", ""),
        "summary_path": endpoint.get("summary_path", ""),
        "capabilities": endpoint.get("capabilities", {}),
        "health": assignment.get("health", endpoint.get("health", {})),
        "owner": assignment.get("owner", ""),
        "shard_id": assignment.get("shard_id", ""),
        "expires_at": assignment.get("expires_at", "") or endpoint.get("expires_at", ""),
        "assignment_expires_at": assignment.get("expires_at", ""),
        "lease_expires_at": endpoint.get("expires_at", ""),
        "release_command": (
            f"remote-inference-launcher pool release {assignment.get('assignment_id', '')}"
            if assignment.get("assignment_id")
            else ""
        ),
    }


def _pool_endpoint(
    *,
    source_kind: str,
    endpoint_name: str,
    endpoint: Mapping[str, object],
    registry: Mapping[str, object],
    lease: Mapping[str, object],
    assignments: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    state = _mapping(registry.get("state"))
    plan = _mapping(registry.get("plan"))
    run_id = str(state.get("run_id", "") or lease.get("run_id", "") or "")
    lease_id = str(lease.get("lease_id", "") or "")
    endpoint_id = (
        f"lease:{lease_id}:{endpoint_name}"
        if source_kind == "lease"
        else f"run:{run_id}:{endpoint_name}"
    )
    plan_endpoint = _plan_endpoint(plan, endpoint_name)
    summary_path = str(endpoint.get("summary_path", "") or plan_endpoint.get("summary_path", ""))
    summary = _read_summary(summary_path)
    active_assignment = _active_assignment(endpoint_id, assignments)
    capabilities = capability_payload(
        endpoint=endpoint,
        plan_endpoint=plan_endpoint,
        summary=summary,
    )
    slurm = _merged_slurm(endpoint, plan_endpoint)
    payload = {
        "endpoint_id": endpoint_id,
        "source_kind": source_kind,
        "lease_id": lease_id,
        "run_id": run_id,
        "endpoint_name": endpoint_name,
        "registry_dir": registry.get("registry_dir", ""),
        "summary_path": summary_path,
        "api_base": endpoint.get("api_base", ""),
        "local_bind_host": plan_endpoint.get("local_bind_host", "127.0.0.1"),
        "local_port": endpoint.get("local_port"),
        "local_port_strategy": plan_endpoint.get("local_port_strategy", ""),
        "remote_port": endpoint.get("remote_port"),
        "backend_kind": endpoint.get("backend_kind", "") or plan_endpoint.get("backend_kind", ""),
        "cleanup_owner": "lease" if source_kind == "lease" else "run",
        "lease_status": lease.get("status", ""),
        "lifecycle_state": endpoint.get("lifecycle_state", ""),
        "expires_at": lease.get("expires_at", ""),
        "assignment_state": _assignment_state(active_assignment),
        "assignment": active_assignment,
        "capabilities": capabilities,
        "scheduler": {},
        "slurm": slurm,
        "ssh_target": plan_endpoint.get("ssh_target", ""),
        "job_id": endpoint.get("job_id", ""),
        "node": endpoint.get("node", ""),
        "logs": endpoint.get("logs", ""),
        "remote_state_path": endpoint.get("remote_state_path", "")
        or plan_endpoint.get("remote_state_path", ""),
    }
    return payload


def _with_health(endpoint: Mapping[str, object], *, depth: str) -> dict[str, object]:
    health = endpoint_health_report(
        endpoint,
        depth=depth,
        assignment=_mapping(endpoint.get("assignment")),
        capability=_mapping(endpoint.get("capabilities")),
    )
    return {**dict(endpoint), "health": health}


def _with_acquirability(endpoint: Mapping[str, object]) -> dict[str, object]:
    acquirable, reason = _endpoint_acquirability(endpoint)
    payload = {**dict(endpoint), "acquirable": acquirable}
    if not acquirable:
        payload["not_acquirable_reason"] = reason
    return payload


def _status_health_depth(health_depth: str) -> str:
    if health_depth == "scheduler":
        return "cached"
    if health_depth == "light":
        return "models"
    return health_depth


def _with_scheduler(
    endpoint: Mapping[str, object],
    scheduler: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    endpoint_id = str(endpoint.get("endpoint_id", "") or "")
    report = scheduler.get(endpoint_id, {})
    if not report:
        return dict(endpoint)
    slurm = dict(_mapping(endpoint.get("slurm")))
    if report.get("slurm_state"):
        slurm["latest_state"] = report.get("slurm_state")
    if report.get("slurm_reason"):
        slurm["latest_reason"] = report.get("slurm_reason")
    return {**dict(endpoint), "scheduler": dict(report), "slurm": slurm}


def _candidate_matches(
    endpoint: Mapping[str, object],
    *,
    model: str,
    min_context: int | None,
    min_remaining: str,
) -> bool:
    acquirable, _reason = _endpoint_acquirability(
        endpoint,
        model=model,
        min_context=min_context,
        min_remaining=min_remaining,
    )
    return acquirable


def _endpoint_acquirability(
    endpoint: Mapping[str, object],
    *,
    model: str = "",
    min_context: int | None = None,
    min_remaining: str = "",
) -> tuple[bool, str]:
    if endpoint.get("source_kind") != "lease":
        return False, "registry_endpoint_without_lease"
    if endpoint.get("lease_status") != "active":
        return False, "lease_not_active"
    if not _endpoint_is_free(endpoint):
        return False, "assignment_active"
    health = _mapping(endpoint.get("health")) or endpoint_health_report(endpoint, depth="cached")
    if health.get("status") != "healthy":
        return False, str(health.get("reason_code", "") or "endpoint_not_healthy")
    ok, reason = capability_matches(
        _mapping(endpoint.get("capabilities")),
        model=model,
        min_context=min_context,
    )
    if not ok:
        return False, reason
    if min_remaining and not _has_min_remaining(endpoint, min_remaining):
        return False, "ttl_below_min_remaining"
    return True, "ok"


def _endpoint_is_free(endpoint: Mapping[str, object]) -> bool:
    assignment = endpoint.get("assignment")
    if not isinstance(assignment, Mapping):
        return True
    return str(assignment.get("status", "") or "") not in ACTIVE_ASSIGNMENT_STATES


def _create_probe_assignment(
    endpoint: Mapping[str, object],
    *,
    owner: str,
    shard_id: str,
    batch_id: str,
    assignment_ttl: str,
    health_depth: str,
    requested_capabilities: Mapping[str, object],
    pool_root: str | Path,
) -> dict[str, object]:
    endpoint_id = str(endpoint.get("endpoint_id", "") or "")
    with _assignment_lock(pool_root):
        assignments = load_assignments(pool_root=pool_root)
        if _active_assignment(endpoint_id, assignments):
            return {}
        assignment_id = _new_assignment_id()
        created_at = utc_now()
        record: dict[str, object] = {
            "schema_version": ASSIGNMENT_SCHEMA_VERSION,
            "assignment_id": assignment_id,
            "endpoint_id": endpoint_id,
            "lease_id": endpoint.get("lease_id", ""),
            "run_id": endpoint.get("run_id", ""),
            "endpoint_name": endpoint.get("endpoint_name", ""),
            "owner": owner,
            "shard_id": shard_id,
            "batch_id": batch_id,
            "status": "probing",
            "created_at": created_at,
            "expires_at": _assignment_expires_at(created_at, assignment_ttl),
            "heartbeat_at": created_at,
            "health_policy": health_depth,
            "requested_capabilities": dict(requested_capabilities),
        }
        write_json_atomic(_assignment_root(pool_root) / f"{assignment_id}.json", record)
        return record


def _update_assignment_status(
    assignment_id: str,
    status: str,
    *,
    pool_root: str | Path,
    health: Mapping[str, object] | None = None,
) -> dict[str, object]:
    with _assignment_lock(pool_root):
        record = read_assignment(assignment_id, pool_root=pool_root)
        record.update({"status": status, "updated_at": utc_now()})
        if health is not None:
            record["health"] = dict(health)
        write_json_atomic(_assignment_root(pool_root) / f"{assignment_id}.json", record)
        return record


def _resolve_assignment(
    *,
    assignment_id: str,
    endpoint_id: str,
    owner: str,
    pool_root: str | Path,
) -> dict[str, object]:
    if assignment_id:
        assignment = read_assignment(assignment_id, pool_root=pool_root)
        _enforce_assignment_owner(assignment, owner=owner, assignment_id=assignment_id)
        return assignment
    for assignment in load_assignments(pool_root=pool_root).values():
        if endpoint_id and assignment.get("endpoint_id") != endpoint_id:
            continue
        if owner and assignment.get("owner") != owner:
            continue
        if assignment.get("status") in ACTIVE_ASSIGNMENT_STATES:
            return dict(assignment)
    return {}


def _enforce_assignment_owner(
    assignment: Mapping[str, object],
    *,
    owner: str,
    assignment_id: str,
) -> None:
    if owner and assignment.get("owner") != owner:
        raise RuntimeError(
            f"Assignment {assignment_id} is owned by "
            f"{assignment.get('owner', '') or 'unknown'}, not {owner}."
        )


def _find_endpoint(
    endpoint_id: str,
    *,
    lease_root: str | Path,
    registry_root: str | Path,
    pool_root: str | Path,
) -> dict[str, object]:
    if not endpoint_id:
        raise ValueError("endpoint_id is required.")
    for endpoint in load_pool_endpoints(
        lease_root=lease_root,
        registry_root=registry_root,
        pool_root=pool_root,
    ):
        if endpoint.get("endpoint_id") == endpoint_id:
            return endpoint
    raise FileNotFoundError(f"Pool endpoint not found: {endpoint_id}")


def _update_endpoint_api_base(
    endpoint: Mapping[str, object],
    *,
    lease_root: str | Path,
) -> None:
    registry_dir = str(endpoint.get("registry_dir", "") or "")
    endpoint_name = str(endpoint.get("endpoint_name", "") or "")
    if registry_dir and endpoint_name:
        state = read_json(Path(registry_dir) / "state.json")
        endpoints = dict(_mapping(state.get("endpoints")))
        current = dict(_mapping(endpoints.get(endpoint_name)))
        current.update(
            {"api_base": endpoint.get("api_base", ""), "local_port": endpoint.get("local_port")}
        )
        endpoints[endpoint_name] = current
        update_state(registry_dir, endpoints=endpoints)
    lease_id = str(endpoint.get("lease_id", "") or "")
    if lease_id:
        lease = read_lease(lease_id, lease_root=lease_root, refresh_expiry=False)
        endpoints = dict(_mapping(lease.get("endpoints")))
        current = dict(_mapping(endpoints.get(endpoint_name)))
        current.update(
            {"api_base": endpoint.get("api_base", ""), "local_port": endpoint.get("local_port")}
        )
        endpoints[endpoint_name] = current
        write_lease({**lease, "endpoints": endpoints}, lease_root=Path(lease_root))


def _pool_summary(endpoints: list[Mapping[str, object]]) -> dict[str, int]:
    summary = {
        "acquirable_healthy_free": 0,
        "healthy_free": 0,
        "healthy_busy": 0,
        "expired": 0,
        "failed": 0,
        "unknown": 0,
        "orphaned": 0,
    }
    for endpoint in endpoints:
        health = _mapping(endpoint.get("health"))
        status = str(health.get("status", "") or "unknown")
        reason = str(health.get("reason_code", "") or "")
        if status == "healthy":
            key = "healthy_free" if _endpoint_is_free(endpoint) else "healthy_busy"
            summary[key] += 1
            if endpoint.get("acquirable") is True:
                summary["acquirable_healthy_free"] += 1
        elif status == "expired":
            summary["expired"] += 1
        elif reason == "registry_orphaned":
            summary["orphaned"] += 1
        elif status == "failed":
            summary["failed"] += 1
        else:
            summary["unknown"] += 1
    return summary


def _eligibility_failure_summary(
    endpoints: list[Mapping[str, object]],
    *,
    model: str,
    min_context: int | None,
    min_remaining: str,
) -> str:
    counts: dict[str, int] = {}
    for endpoint in endpoints:
        acquirable, reason = _endpoint_acquirability(
            endpoint,
            model=model,
            min_context=min_context,
            min_remaining=min_remaining,
        )
        if acquirable:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return ", ".join(f"{reason}={count}" for reason, count in sorted(counts.items()))


def _has_min_remaining(endpoint: Mapping[str, object], min_remaining: str) -> bool:
    expires_at = str(endpoint.get("expires_at", "") or "")
    if not expires_at:
        return False
    try:
        expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return expires >= datetime.now(UTC) + parse_ttl(min_remaining)


def _lease_paths(root: str | Path) -> list[Path]:
    lease_root = Path(root).expanduser()
    if not lease_root.exists():
        return []
    return [path for path in sorted(lease_root.glob("*.json")) if path.is_file()]


def _registry_paths(root: str | Path) -> list[Path]:
    registry_root = Path(root).expanduser()
    if not registry_root.exists():
        return []
    return [
        path
        for path in sorted(registry_root.iterdir())
        if path.is_dir() and (path / "state.json").exists()
    ]


def _endpoint_records(state: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    endpoints = state.get("endpoints", {})
    if not isinstance(endpoints, Mapping):
        return {}
    return {
        str(name): endpoint for name, endpoint in endpoints.items() if isinstance(endpoint, Mapping)
    }


def _plan_endpoint(plan: Mapping[str, object], endpoint_name: str) -> Mapping[str, object]:
    endpoints = plan.get("endpoints", ())
    if not isinstance(endpoints, list | tuple):
        return {}
    for endpoint in endpoints:
        if isinstance(endpoint, Mapping) and endpoint.get("name") == endpoint_name:
            return endpoint
    return {}


def _read_summary(path: str) -> Mapping[str, object]:
    if not path:
        return {}
    summary_path = Path(path).expanduser()
    if not summary_path.exists():
        return {}
    try:
        payload = read_json(summary_path)
    except OSError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _merged_slurm(
    endpoint: Mapping[str, object],
    plan_endpoint: Mapping[str, object],
) -> dict[str, object]:
    slurm = dict(_mapping(endpoint.get("slurm")))
    plan_slurm = _mapping(plan_endpoint.get("slurm"))
    for field in ("partition", "num_gpus", "nodes", "walltime", "memory", "cpus_per_task"):
        if field not in slurm and _is_present(plan_slurm.get(field)):
            slurm[field] = plan_slurm[field]
    return slurm


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _is_present(value: object) -> bool:
    return value is not None and value != ""
