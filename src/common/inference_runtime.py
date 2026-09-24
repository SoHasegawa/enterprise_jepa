"""Shared benchmark-runner adapter for managed inference endpoints."""

from __future__ import annotations

import signal
import subprocess
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

from common.logging_utils import get_logger
from common.network_env import ensure_no_proxy

LOGGER = get_logger(__name__)
HANDOFF_SCHEMA_VERSION = "ril-handoff/v1"


@dataclass(slots=True)
class ManagedInferenceRuntime:
    """Start configured inference for a benchmark run and expose generic env."""

    config_path: Path
    launch_summary_path: Path | None = None
    overwrite_launch_summary: bool = False
    launcher: Any = None
    sessions: dict[str, object] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    plan: Any = None

    def __enter__(self) -> ManagedInferenceRuntime:
        from remote_inference_launcher.session_env import (
            generic_inference_env,
            normalize_sessions,
        )

        self.plan = self._build_effective_plan()
        self.launcher = self._load_launcher()
        try:
            self.sessions = normalize_sessions(self.launcher.start())
        except BaseException as error:
            self._stop_launcher_safely(original_error=error)
            raise
        self.env = generic_inference_env(self.sessions)
        ensure_no_proxy(self.env, _base_urls(self.env))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self.launcher is not None:
            self.launcher.stop()

    def apply_to(self, env: dict[str, str]) -> None:
        """Merge generic inference env into an existing process environment."""

        env.update(self.env)
        ensure_no_proxy(env, _base_urls(self.env))

    def generic_env(self) -> dict[str, str]:
        """Return generic inference env values."""

        return dict(self.env)

    def payload(self) -> dict[str, Any]:
        """Return the machine-readable inference handoff payload."""

        from remote_inference_launcher.session_env import sessions_payload

        sessions = sessions_payload(self.sessions)
        endpoints = dict(sessions.get("endpoints", {}) or {})
        run_ids = sorted(
            {
                str(endpoint.get("run_id", "") or "")
                for endpoint in endpoints.values()
                if isinstance(endpoint, Mapping) and endpoint.get("run_id")
            }
        )
        run_id = run_ids[0] if len(run_ids) == 1 else ""
        semantic_hash = str(getattr(self.plan, "semantic_config_hash", "") or "")
        return {
            "schema_version": HANDOFF_SCHEMA_VERSION,
            "run_id": run_id,
            "semantic_config_hash": semantic_hash,
            "identity_key": _handoff_identity_key(
                run_id=run_id,
                semantic_config_hash=semantic_hash,
            ),
            "cleanup_owner": (
                "external" if getattr(self.launcher, "owns_resources", True) is False else "runtime"
            ),
            "source_config_path": str(self.config_path),
            "endpoints": endpoints,
            "served_model_names": {
                name: _endpoint_served_model(endpoint)
                for name, endpoint in endpoints.items()
                if isinstance(endpoint, Mapping)
            },
            "summary_paths": {
                name: str(endpoint.get("summary_path", "") or "")
                for name, endpoint in endpoints.items()
                if isinstance(endpoint, Mapping)
            },
            "metadata": {
                "api_key_set": any(
                    bool(endpoint.get("api_key_set"))
                    for endpoint in endpoints.values()
                    if isinstance(endpoint, Mapping)
                ),
            },
        }

    def recommended_max_parallel(self) -> int | None:
        """Return the most conservative launcher recommendation across sessions."""

        recommendations: list[int] = []
        for session in self.sessions.values():
            metadata = dict(getattr(session, "metadata", {}) or {})
            value = metadata.get("recommended_benchmark_max_parallel")
            if isinstance(value, int) and value > 0:
                recommendations.append(value)
        if not recommendations:
            return None
        return min(recommendations)

    def _stop_launcher_safely(self, *, original_error: BaseException | None = None) -> None:
        if self.launcher is None:
            return
        try:
            self.launcher.stop()
        except Exception as stop_error:
            LOGGER.exception("Failed to stop inference launcher during cleanup.")
            if original_error is not None:
                raise RuntimeError(
                    "Inference launcher failed and cleanup also failed: "
                    f"{stop_error}. Original failure: {original_error}"
                ) from original_error
            raise

    def _load_launcher(self) -> Any:
        from remote_inference_launcher.inference_config import load_inference_launcher

        if self.launch_summary_path is not None or self.overwrite_launch_summary:
            return load_inference_launcher(
                self.config_path,
                launch_summary_path=(
                    str(self.launch_summary_path) if self.launch_summary_path is not None else None
                ),
                overwrite_launch_summary=self.overwrite_launch_summary,
            )
        return load_inference_launcher(self.config_path)

    def _build_effective_plan(self) -> Any:
        from remote_inference_launcher.plans import build_effective_launch_plan_from_path

        return build_effective_launch_plan_from_path(self.config_path)


def managed_inference(
    config_path: str | Path,
    *,
    launch_summary_path: str | Path | None = None,
    overwrite_launch_summary: bool = False,
) -> ManagedInferenceRuntime:
    """Return a managed inference context for a config path."""

    return ManagedInferenceRuntime(
        Path(config_path),
        launch_summary_path=Path(launch_summary_path) if launch_summary_path is not None else None,
        overwrite_launch_summary=overwrite_launch_summary,
    )


def write_inference_env_file(path: str | Path, env: dict[str, str]) -> None:
    """Write generic inference env values as shell exports."""

    from remote_inference_launcher.session_env import write_env_file

    write_env_file(path, env)


@contextmanager
def translate_termination_signals() -> Iterator[None]:
    """Convert SIGINT/SIGTERM into KeyboardInterrupt for managed cleanup paths."""

    previous_handlers: dict[int, signal.Handlers] = {}

    def handle_signal(signum: int, _frame: object) -> None:
        signame = signal.Signals(signum).name
        raise KeyboardInterrupt(f"received {signame}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)
    except ValueError:
        yield
        return

    try:
        yield
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


def attach_reproducibility_metadata(
    request_config: dict[str, Any],
    *,
    result_paths: Any,
    inference_summary_path: Path | None,
    inference_handoff: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> None:
    """Record generic reproducibility metadata in the benchmark request config."""

    generation_keys = (
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_output_tokens",
        "output_token_budget",
        "max_model_len",
        "context_size",
    )
    generation = {
        key: request_config[key]
        for key in generation_keys
        if key in request_config and request_config[key] is not None
    }
    metadata = dict(request_config.get("reproducibility", {}) or {})
    metadata.update(
        {
            "result_run_id": str(result_paths.run_id),
            "result_dir": str(result_paths.result_dir),
            "config_hash": str(result_paths.config_hash),
            "task_ids": [str(task_id) for task_id in request_config.get("task_ids", []) or []],
            "shard_count": request_config.get("shard_count"),
            "shard_index": request_config.get("shard_index"),
            "run_variants": list(request_config.get("run_variants", []) or []),
            "include_reference": request_config.get("include_reference"),
            "generation": generation,
            "git": _git_fingerprint(repo_root) if repo_root is not None else {},
        }
    )
    if inference_summary_path is not None:
        metadata["inference_launch_summary_path"] = str(inference_summary_path)
    if inference_handoff is not None:
        metadata.update(_benchmark_metadata_from_handoff(inference_handoff))
    if (
        request_config.get("shard_count") is not None
        or request_config.get("shard_index") is not None
    ):
        shard_task_ids = [
            str(task_id)
            for task_id in request_config.get("shard_task_ids")
            or request_config.get("task_ids", [])
            or []
        ]
        metadata["shard"] = {
            "shard_count": request_config.get("shard_count"),
            "shard_index": request_config.get("shard_index"),
            "task_ids": shard_task_ids,
            "source_task_count": request_config.get("shard_source_task_count"),
            "endpoint_name": request_config.get("inference_endpoint_name"),
            "aggregation_key": (
                f"{request_config.get('target')}:{request_config.get('shard_count')}"
            ),
        }
    if request_config.get("resume_skipped_task_ids"):
        metadata["resume"] = {
            "original_task_ids": list(request_config.get("resume_original_task_ids", []) or []),
            "remaining_task_ids": list(request_config.get("resume_remaining_task_ids", []) or []),
            "skipped_task_ids": list(request_config.get("resume_skipped_task_ids", []) or []),
            "source_manifest_paths": list(
                request_config.get("resume_source_manifest_paths", []) or []
            ),
        }
    request_config["reproducibility"] = metadata


def prepare_inference_runtime_config(
    *,
    runtime_config: dict[str, Any],
    env: dict[str, str],
    result_paths: Any,
    inference_summary_path: Path | None,
    inference_runtime: Any,
    warn: Callable[[str], None] | None = None,
    repo_root: Path | None = None,
) -> None:
    """Apply a managed inference runtime to benchmark env and request metadata."""

    inference_runtime.apply_to(env)
    inference_handoff = require_inference_handoff_payload(inference_runtime)
    if inference_summary_path is not None:
        env["BENCHMARK_INFERENCE_LAUNCH_SUMMARY"] = str(inference_summary_path)
        runtime_config["config"]["inference_launch_summary_path"] = str(inference_summary_path)
    runtime_config["config"]["inference_handoff"] = dict(inference_handoff)
    runtime_config["config"]["inference_sessions"] = {
        "endpoints": dict(inference_handoff["endpoints"])
    }
    attach_reproducibility_metadata(
        runtime_config["config"],
        result_paths=result_paths,
        inference_summary_path=inference_summary_path,
        inference_handoff=inference_handoff,
        repo_root=repo_root,
    )
    apply_recommended_max_parallel(
        runtime_config["config"],
        recommended_max_parallel=inference_runtime.recommended_max_parallel(),
        warn=warn,
    )


def apply_recommended_max_parallel(
    request_config: dict[str, Any],
    *,
    recommended_max_parallel: int | None,
    warn: Callable[[str], None] | None = None,
) -> None:
    """Record and report a launcher benchmark parallelism recommendation."""

    configured_max_parallel = request_config.get("max_parallel")
    if recommended_max_parallel is None:
        return

    metadata = dict(request_config.get("reproducibility", {}) or {})
    metadata["launcher_recommended_max_parallel"] = recommended_max_parallel
    request_config["reproducibility"] = metadata

    if configured_max_parallel is None:
        if warn is not None:
            warn(
                "[yellow]Max Parallel was omitted; launcher recommends "
                f"{recommended_max_parallel}. Pass --max-parallel to use that value.[/yellow]"
            )
        return

    if int(configured_max_parallel) > recommended_max_parallel and warn is not None:
        warn(
            "[yellow]Configured Max Parallel exceeds launcher recommendation "
            f"({configured_max_parallel} > {recommended_max_parallel}); "
            "the run may queue or time out.[/yellow]"
        )


def require_inference_handoff_payload(inference_runtime: Any) -> dict[str, Any]:
    """Return a validated machine-readable handoff payload from a runtime."""

    payload_method = getattr(inference_runtime, "payload", None)
    if not callable(payload_method):
        raise RuntimeError("Managed inference runtime did not expose a handoff payload.")
    payload = payload_method()
    if not isinstance(payload, Mapping):
        raise RuntimeError("Managed inference handoff payload must be a mapping.")
    handoff = dict(payload)
    _validate_inference_handoff_payload(handoff)
    return handoff


def _validate_inference_handoff_payload(handoff: Mapping[str, Any]) -> None:
    endpoints = _handoff_endpoint_records(handoff)
    if not str(handoff.get("schema_version", "") or ""):
        raise RuntimeError("Managed inference handoff payload is missing schema_version.")
    if not str(handoff.get("cleanup_owner", "") or ""):
        raise RuntimeError("Managed inference handoff payload is missing cleanup_owner.")
    if not _handoff_identity_key_from_payload(handoff):
        raise RuntimeError("Managed inference handoff payload is missing stable identity fields.")
    if not str(handoff.get("semantic_config_hash", "") or ""):
        raise RuntimeError("Managed inference handoff payload is missing semantic_config_hash.")
    _validate_handoff_endpoints(endpoints)


def _handoff_endpoint_records(handoff: Mapping[str, Any]) -> Mapping[str, Any]:
    endpoints = handoff.get("endpoints")
    if not isinstance(endpoints, Mapping) or not endpoints:
        raise RuntimeError("Managed inference handoff payload must include endpoint sessions.")
    return endpoints


def _validate_handoff_endpoints(endpoints: Mapping[str, Any]) -> None:
    for name, endpoint in endpoints.items():
        if not isinstance(endpoint, Mapping):
            raise RuntimeError(f"Managed inference endpoint {name!r} must be a mapping.")
        _validate_handoff_endpoint(name, endpoint)


def _validate_handoff_endpoint(name: object, endpoint: Mapping[str, Any]) -> None:
    if not str(endpoint.get("api_base", "") or ""):
        raise RuntimeError(f"Managed inference endpoint {name!r} is missing api_base.")
    if not _endpoint_served_model(endpoint):
        raise RuntimeError(f"Managed inference endpoint {name!r} is missing served_model_name.")
    if not str(endpoint.get("summary_path", "") or ""):
        raise RuntimeError(f"Managed inference endpoint {name!r} is missing summary_path.")


def _benchmark_metadata_from_handoff(handoff: Mapping[str, Any]) -> dict[str, Any]:
    endpoints = handoff.get("endpoints", {})
    endpoint_records = dict(endpoints) if isinstance(endpoints, Mapping) else {}
    first_endpoint = next(iter(endpoint_records.values()), {})
    served_model = (
        _endpoint_served_model(first_endpoint) if isinstance(first_endpoint, Mapping) else ""
    )
    metadata = {
        "inference_run_id": str(handoff.get("run_id", "") or ""),
        "inference_semantic_config_hash": str(handoff.get("semantic_config_hash", "") or ""),
        "inference_identity_key": _handoff_identity_key_from_payload(handoff),
        "inference_cleanup_owner": str(handoff.get("cleanup_owner", "") or ""),
        "inference_served_model_name": served_model,
        "inference_sessions": {"endpoints": endpoint_records},
    }
    lease_id = str(handoff.get("lease_id", "") or "")
    if lease_id:
        metadata["inference_lease_id"] = lease_id
    return metadata


def _handoff_identity_key_from_payload(handoff: Mapping[str, Any]) -> str:
    explicit = str(handoff.get("identity_key", "") or "")
    if explicit:
        return explicit
    lease_id = str(handoff.get("lease_id", "") or "")
    if lease_id:
        return f"lease:{lease_id}"
    return _handoff_identity_key(
        run_id=str(handoff.get("run_id", "") or ""),
        semantic_config_hash=str(handoff.get("semantic_config_hash", "") or ""),
    )


def _handoff_identity_key(*, run_id: str, semantic_config_hash: str) -> str:
    if run_id:
        return f"run:{run_id}"
    if semantic_config_hash:
        return f"semantic:{semantic_config_hash}"
    return ""


def _endpoint_served_model(endpoint: Mapping[str, Any]) -> str:
    return str(endpoint.get("served_model_name", "") or endpoint.get("model", "") or "")


def _base_urls(env: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        value
        for name, value in env.items()
        if name == "OPENAI_BASE_URL" or name.endswith("_BASE_URL")
    )


def _git_fingerprint(repo_root: Path) -> dict[str, str]:
    return {
        "commit": _git_output(["git", "rev-parse", "HEAD"], repo_root=repo_root),
        "branch": _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root=repo_root),
        "dirty": "true" if _git_dirty(repo_root) else "false",
    }


def _git_output(command: list[str], *, repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if completed.returncode:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _git_dirty(repo_root: Path) -> bool:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return bool(completed.stdout.strip())
