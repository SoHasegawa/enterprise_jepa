"""Derived run and lease state reconciliation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from remote_inference_launcher.registry import read_registry

READY_LIFECYCLES = {"READY", "LEASED"}
QUEUED_LIFECYCLES = {
    "SUBMITTING",
    "SUBMITTED",
    "PENDING",
    "ALLOCATED",
    "REMOTE_PORT_READY",
    "TUNNEL_READY",
}
STOPPED_LIFECYCLES = {"STOPPED", "RELEASED"}


@dataclass(frozen=True)
class RunStateReconciler:
    """Derive user-facing lease state from registry/controller evidence."""

    def reconcile_lease(self, record: Mapping[str, object]) -> dict[str, object]:
        registry_dir = str(record.get("registry_dir", "") or "")
        if not registry_dir:
            return dict(record)
        registry = read_registry(registry_dir)
        state = registry.get("state", {})
        if not isinstance(state, Mapping):
            return dict(record)
        lifecycle = str(state.get("lifecycle_state", "") or "")
        current_status = str(record.get("status", "") or "")
        if lifecycle in READY_LIFECYCLES:
            if current_status in {
                "expired",
                "stopping",
                "stopped",
                "failed",
                "orphaned",
                "ready_timeout",
            }:
                return self._with_registry_state(record, state=state)
            return self._with_registry_state(record, state=state, status="active")
        if lifecycle in QUEUED_LIFECYCLES and current_status in {"pending", "queued"}:
            return self._with_registry_state(record, state=state, status="queued")
        if lifecycle == "FAILED":
            return self._with_registry_state(record, state=state, status="failed")
        if lifecycle == "EXPIRED":
            return self._with_registry_state(record, state=state, status="expired")
        if lifecycle in STOPPED_LIFECYCLES:
            return self._with_registry_state(record, state=state, status="stopped")
        if lifecycle == "ORPHANED":
            return self._with_registry_state(record, state=state, status="orphaned")
        return self._with_registry_state(record, state=state)

    def _with_registry_state(
        self,
        record: Mapping[str, object],
        *,
        state: Mapping[str, object],
        status: str = "",
    ) -> dict[str, object]:
        reconciled: dict[str, object] = {
            **dict(record),
            "registry_lifecycle_state": state.get("lifecycle_state", ""),
        }
        if status:
            reconciled["status"] = status
        endpoints = lease_endpoints_from_state(state)
        if endpoints:
            reconciled["endpoints"] = endpoints
        failure = state.get("failure")
        if isinstance(failure, Mapping):
            reconciled["failure"] = dict(failure)
        return reconciled


def lease_endpoints_from_state(state: object) -> dict[str, dict[str, object]]:
    """Return lease endpoint records derived from registry state."""

    if not isinstance(state, Mapping):
        return {}
    endpoints = state.get("endpoints", {})
    if not isinstance(endpoints, Mapping):
        return {}
    return {
        str(name): _lease_endpoint_record(endpoint)
        for name, endpoint in endpoints.items()
        if isinstance(endpoint, Mapping)
    }


def _lease_endpoint_record(endpoint: Mapping[str, object]) -> dict[str, object]:
    return {
        "api_base": endpoint.get("api_base", ""),
        "served_model_name": endpoint.get("served_model_name", ""),
        "summary_path": endpoint.get("summary_path", ""),
        "lifecycle_state": endpoint.get("lifecycle_state", ""),
        "job_id": endpoint.get("job_id", ""),
        "logs": endpoint.get("logs", ""),
        "node": endpoint.get("node", ""),
        "local_port": endpoint.get("local_port"),
        "remote_port": endpoint.get("remote_port"),
        "backend_kind": endpoint.get("backend_kind", ""),
        "cleanup_command": endpoint.get("cleanup_command", ""),
        "remote_state_path": endpoint.get("remote_state_path", ""),
        "diagnostics": endpoint.get("diagnostics", {}),
        "slurm": endpoint.get("slurm", {}),
    }
