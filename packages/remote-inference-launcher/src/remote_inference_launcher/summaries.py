"""Machine-readable launch summary helpers."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from remote_inference_launcher.diagnostics import CapacityInfo
from remote_inference_launcher.readiness import ReadinessResult

SUMMARY_ROOT = ".remote-inference-launcher/runs"


def new_run_id() -> str:
    """Return a short inspectable unique run id."""

    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def safe_label(value: str) -> str:
    """Return a filesystem and Slurm-display friendly label."""

    safe = "".join(
        char if char.isascii() and (char.isalnum() or char in {"-", "_"}) else "-"
        for char in value.strip()
    )
    safe = "-".join(part for part in safe.split("-") if part)
    return safe[:64] or "default"


def generated_summary_path(
    *,
    run_id: str,
    endpoint_name: str,
    backend_kind: str,
    root: str | Path = SUMMARY_ROOT,
) -> Path:
    """Return the default non-overwriting local summary path."""

    return Path(root).expanduser() / run_id / safe_label(endpoint_name) / f"{backend_kind}.json"


class SummaryWriter:
    """Reserve and atomically write one launch summary file."""

    def __init__(
        self,
        path: str | Path | None,
        *,
        run_id: str,
        endpoint_name: str,
        backend_kind: str,
        overwrite: bool = False,
    ) -> None:
        self.path = (
            Path(path).expanduser()
            if path
            else generated_summary_path(
                run_id=run_id,
                endpoint_name=endpoint_name,
                backend_kind=backend_kind,
            )
        )
        self.overwrite = overwrite
        self._reserved = False

    def reserve(self) -> Path:
        """Reserve the summary path before launch work starts."""

        if self._reserved:
            return self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT
        if not self.overwrite:
            flags |= os.O_EXCL
        else:
            flags |= os.O_TRUNC
        try:
            fd = os.open(self.path, flags, 0o644)
        except FileExistsError as error:
            raise FileExistsError(
                f"Launch summary already exists: {self.path}. "
                "Pass overwrite_launch_summary=true or --overwrite-launch-summary to replace it."
            ) from error
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"lifecycle_state": "RESERVED"}, handle, indent=2)
            handle.write("\n")
        self._reserved = True
        return self.path

    def write(self, payload: Mapping[str, Any]) -> None:
        """Atomically write the current summary payload."""

        self.reserve()
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                handle.write("\n")
            os.replace(tmp_name, self.path)
            _record_registry_summary(self.path, payload)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)


def _record_registry_summary(path: Path, payload: Mapping[str, Any]) -> None:
    from remote_inference_launcher.registry import record_endpoint_summary

    record_endpoint_summary(path, payload)


def endpoint_summary(
    *,
    endpoint_name: str,
    backend_kind: str,
    lifecycle_state: str,
    run_id: str,
    api_base: str = "",
    served_model_name: str = "",
    model: str = "",
    api_key_set: bool = False,
    local_port: int | None = None,
    remote_port: int | None = None,
    job_id: str = "",
    pid: int | None = None,
    job_name: str = "",
    node: str = "",
    cleanup_command: str = "",
    remote_log_dir: str | None = None,
    remote_state_path: str | None = None,
    local_log_dir: str | None = None,
    readiness: ReadinessResult | None = None,
    capacity: CapacityInfo | None = None,
    failure_code: str | None = None,
    failure_message: str | None = None,
    resource_attempts: list[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared single-endpoint launch summary shape."""

    capacity_payload = capacity.to_summary() if capacity is not None else {}
    recommended = capacity_payload.get("recommended_benchmark_max_parallel")
    payload: dict[str, Any] = {
        "endpoint_name": endpoint_name,
        "backend_kind": backend_kind,
        "lifecycle_state": lifecycle_state,
        "run_id": run_id,
        "job_id": job_id or None,
        "pid": pid,
        "job_name": job_name or None,
        "node": node or None,
        "local_api_base": api_base or None,
        "local_port": local_port,
        "remote_port": remote_port,
        "cleanup_command": cleanup_command or None,
        "model": model or None,
        "served_model_name": served_model_name or None,
        "api_key_set": api_key_set,
        "max_model_len": capacity_payload.get("max_model_len"),
        "max_num_seqs": capacity_payload.get("max_num_seqs"),
        "vllm_max_concurrency": capacity_payload.get("vllm_max_concurrency"),
        "recommended_benchmark_max_parallel": recommended,
        "resource_attempts": list(resource_attempts or ()),
        "remote_log_dir": remote_log_dir,
        "remote_state_path": remote_state_path,
        "local_log_dir": local_log_dir,
        "readiness": readiness.to_summary() if readiness is not None else None,
        "diagnostics": {
            "failure_code": failure_code,
            "failure_message": failure_message,
            "vllm_capacity_excerpt": capacity_payload.get("vllm_capacity_excerpt", ""),
            "kv_cache_tokens": capacity_payload.get("kv_cache_tokens"),
        },
        "benchmark_handoff": {
            "api_base": api_base or None,
            "model": served_model_name or model or None,
            "recommended_max_parallel": recommended,
        },
    }
    if extra:
        payload.update(dict(extra))
    return payload


def fleet_summary(
    *,
    fleet_name: str,
    lifecycle_state: str,
    run_id: str,
    endpoints: Mapping[str, Mapping[str, Any]],
    resource_budget: Mapping[str, Any] | None = None,
    observed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared fleet summary shape."""

    return {
        "endpoint_name": fleet_name,
        "backend_kind": "fleet",
        "lifecycle_state": lifecycle_state,
        "run_id": run_id,
        "endpoints": {name: dict(payload) for name, payload in sorted(endpoints.items())},
        "resource_budget": dict(resource_budget or {}),
        "observed_resource_usage": dict(observed or {}),
    }
