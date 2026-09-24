"""Pool endpoint health checks and wrapper-facing health signals."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping

from remote_inference_launcher.capabilities import capability_matches
from remote_inference_launcher.diagnostics import sanitized_excerpt
from remote_inference_launcher.readiness import ReadinessConfig, check_openai_readiness
from remote_inference_launcher.registry import utc_now

HEALTH_SCHEMA_VERSION = "ril-health-signal/v1"
HEALTH_DEPTHS = {"cached", "local", "models", "full"}


def endpoint_health_report(
    endpoint: Mapping[str, object],
    *,
    depth: str = "cached",
    assignment: Mapping[str, object] | None = None,
    capability: Mapping[str, object] | None = None,
    expected_model: str = "",
    min_context: int | None = None,
    api_key: str = "",
    timeout_seconds: float = 2.0,
) -> dict[str, object]:
    """Return an evidence-backed health report for one pool endpoint."""

    if depth not in HEALTH_DEPTHS:
        raise ValueError(f"Unsupported health depth: {depth}")
    evidence: dict[str, object] = {
        "lease_expired": str(endpoint.get("lease_status", "") or "") == "expired",
        "assignment_expired": _assignment_expired(assignment),
        "local_tunnel_ok": None,
        "models_endpoint_ok": None,
        "smoke_test_ok": None,
        "model_ids": [],
        "slurm_state": _slurm_state(endpoint),
        "latency_seconds": None,
    }
    cached_status, cached_reason = _cached_status(endpoint, assignment=assignment)
    if cached_status != "healthy" or depth == "cached":
        return _report(
            endpoint,
            assignment=assignment,
            status=cached_status,
            reason_code=cached_reason,
            evidence=evidence,
        )

    api_base = str(endpoint.get("api_base", "") or "")
    if not api_base:
        return _missing_api_base_report(endpoint, assignment=assignment, evidence=evidence)

    for probe in (
        lambda: _local_probe_report(
            endpoint,
            assignment=assignment,
            depth=depth,
            api_base=api_base,
            evidence=evidence,
            timeout_seconds=timeout_seconds,
        ),
        lambda: _models_probe_report(
            endpoint,
            assignment=assignment,
            depth=depth,
            api_base=api_base,
            api_key=api_key,
            expected_model=expected_model,
            evidence=evidence,
            timeout_seconds=timeout_seconds,
        ),
        lambda: _capability_probe_report(
            endpoint,
            assignment=assignment,
            capability=capability,
            expected_model=expected_model,
            min_context=min_context,
            evidence=evidence,
        ),
        lambda: _full_probe_report(
            endpoint,
            assignment=assignment,
            depth=depth,
            api_base=api_base,
            api_key=api_key,
            expected_model=expected_model,
            evidence=evidence,
            timeout_seconds=timeout_seconds,
        ),
    ):
        report = probe()
        if report is not None:
            return report
    return _report(
        endpoint,
        assignment=assignment,
        status="healthy",
        reason_code="ok",
        evidence=evidence,
    )


def _missing_api_base_report(
    endpoint: Mapping[str, object],
    *,
    assignment: Mapping[str, object] | None,
    evidence: Mapping[str, object],
) -> dict[str, object]:
    return _report(
        endpoint,
        assignment=assignment,
        status="unknown",
        reason_code="health_check_error",
        evidence={**evidence, "error": "endpoint is missing api_base"},
    )


def _local_probe_report(
    endpoint: Mapping[str, object],
    *,
    assignment: Mapping[str, object] | None,
    depth: str,
    api_base: str,
    evidence: dict[str, object],
    timeout_seconds: float,
) -> dict[str, object] | None:
    if depth not in {"local", "models", "full"}:
        return None
    local_ok, local_error, latency = _check_local_connection(
        api_base,
        timeout_seconds=timeout_seconds,
    )
    evidence.update({"local_tunnel_ok": local_ok, "latency_seconds": latency})
    if local_ok:
        return None
    return _report(
        endpoint,
        assignment=assignment,
        status="unhealthy",
        reason_code="tunnel_dead",
        evidence={**evidence, "error": local_error},
    )


def _models_probe_report(
    endpoint: Mapping[str, object],
    *,
    assignment: Mapping[str, object] | None,
    depth: str,
    api_base: str,
    api_key: str,
    expected_model: str,
    evidence: dict[str, object],
    timeout_seconds: float,
) -> dict[str, object] | None:
    if depth not in {"models", "full"}:
        return None
    models_ok, model_ids, models_error, latency = fetch_openai_model_ids(
        api_base,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    evidence.update(
        {
            "models_endpoint_ok": models_ok,
            "model_ids": list(model_ids),
            "latency_seconds": latency,
        }
    )
    if not models_ok:
        return _report(
            endpoint,
            assignment=assignment,
            status="unhealthy",
            reason_code="models_failed",
            evidence={**evidence, "error": models_error},
        )
    model_to_check = expected_model or str(endpoint.get("served_model_name", "") or "")
    if not model_to_check or model_to_check in model_ids:
        return None
    return _report(
        endpoint,
        assignment=assignment,
        status="unhealthy",
        reason_code="model_mismatch",
        evidence=evidence,
    )


def _capability_probe_report(
    endpoint: Mapping[str, object],
    *,
    assignment: Mapping[str, object] | None,
    capability: Mapping[str, object] | None,
    expected_model: str,
    min_context: int | None,
    evidence: dict[str, object],
) -> dict[str, object] | None:
    if capability is None:
        return None
    ok, reason = capability_matches(
        capability,
        model=expected_model,
        min_context=min_context,
    )
    evidence["capability_ok"] = ok
    if ok:
        return None
    return _report(
        endpoint,
        assignment=assignment,
        status="unhealthy",
        reason_code=reason,
        evidence=evidence,
    )


def _full_probe_report(
    endpoint: Mapping[str, object],
    *,
    assignment: Mapping[str, object] | None,
    depth: str,
    api_base: str,
    api_key: str,
    expected_model: str,
    evidence: dict[str, object],
    timeout_seconds: float,
) -> dict[str, object] | None:
    if depth != "full":
        return None
    readiness = check_openai_readiness(
        api_base,
        model=expected_model or str(endpoint.get("served_model_name", "") or ""),
        api_key=api_key,
        config=ReadinessConfig(timeout_seconds=timeout_seconds),
    )
    evidence["smoke_test_ok"] = readiness.smoke_test_ok
    if readiness.ok:
        return None
    return _report(
        endpoint,
        assignment=assignment,
        status="unhealthy",
        reason_code=readiness.failure_code or "smoke_failed",
        evidence={**evidence, "error": readiness.failure_message or ""},
    )


def fetch_openai_model_ids(
    api_base: str,
    *,
    api_key: str = "",
    timeout_seconds: float = 2.0,
) -> tuple[bool, tuple[str, ...], str, float | None]:
    """Return model ids from an OpenAI-compatible `/models` endpoint."""

    request = urllib.request.Request(api_base.rstrip("/") + "/models")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        return False, (), f"/models returned HTTP {error.code}: {detail}", None
    except (OSError, urllib.error.URLError) as error:
        return False, (), str(error), None
    latency = time.monotonic() - started
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return False, (), f"/models returned non-JSON response: {body[:200]}", latency
    data = parsed.get("data") if isinstance(parsed, dict) else None
    if not isinstance(data, list):
        return False, (), f"/models returned no data list: {body[:200]}", latency
    model_ids = tuple(
        str(item.get("id", ""))
        for item in data
        if isinstance(item, Mapping) and item.get("id") not in {"", None}
    )
    return True, model_ids, "", latency


def _cached_status(
    endpoint: Mapping[str, object],
    *,
    assignment: Mapping[str, object] | None,
) -> tuple[str, str]:
    if _assignment_expired(assignment):
        return "expired", "assignment_expired"
    lease_status = str(endpoint.get("lease_status", "") or "")
    lifecycle = str(endpoint.get("lifecycle_state", "") or "")
    if lease_status == "expired" or lifecycle == "EXPIRED":
        return "expired", "lease_expired"
    if lifecycle == "FAILED" or lease_status == "failed":
        return "failed", "registry_failed"
    if lifecycle == "ORPHANED" or lease_status == "orphaned":
        return "failed", "registry_orphaned"
    slurm = endpoint.get("slurm")
    if isinstance(slurm, Mapping):
        state = str(slurm.get("latest_state", "") or "").upper()
        if state in {
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
            return "failed", "slurm_terminal"
    if lifecycle in {"READY", "LEASED"} and lease_status in {"", "active"}:
        return "healthy", "ok"
    return "unknown", "slurm_unknown" if endpoint.get("job_id") else "health_check_error"


def _assignment_expired(assignment: Mapping[str, object] | None) -> bool:
    if not isinstance(assignment, Mapping):
        return False
    if assignment.get("status") == "expired":
        return True
    expires_at = str(assignment.get("expires_at", "") or "")
    if not expires_at:
        return False
    try:
        expires = _timestamp_seconds(expires_at)
    except ValueError:
        return False
    return expires <= time.time()


def _timestamp_seconds(value: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _check_local_connection(
    api_base: str,
    *,
    timeout_seconds: float,
) -> tuple[bool, str, float | None]:
    parsed = urllib.parse.urlparse(api_base)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False, "api_base has no host", None
    started = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True, "", time.monotonic() - started
    except OSError as error:
        return False, sanitized_excerpt(str(error), max_chars=500), None


def _slurm_state(endpoint: Mapping[str, object]) -> str:
    slurm = endpoint.get("slurm")
    if not isinstance(slurm, Mapping):
        return ""
    return str(slurm.get("latest_state", "") or "")


def _report(
    endpoint: Mapping[str, object],
    *,
    assignment: Mapping[str, object] | None,
    status: str,
    reason_code: str,
    evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "endpoint_id": endpoint.get("endpoint_id", ""),
        "assignment_id": assignment.get("assignment_id", "") if assignment else "",
        "status": status,
        "reason_code": reason_code,
        "recommended_action": _recommended_action(status, reason_code),
        "checked_at": utc_now(),
        "evidence": dict(evidence),
    }


def _recommended_action(status: str, reason_code: str) -> str:
    if status == "healthy":
        return "use_endpoint"
    if reason_code in {"tunnel_dead"}:
        return "reconnect_or_reacquire"
    if status in {"expired", "failed"}:
        return "reacquire"
    return "inspect"
