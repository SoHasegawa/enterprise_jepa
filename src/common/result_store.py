from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.models import BenchmarkRunManifest, BenchmarkRunPaths
from common.storage import default_result_root

DEFAULT_RESULT_ROOT = default_result_root()
RESULT_MANIFEST_FILE_NAME = "manifest.json"
RESULT_DETAIL_FILE_NAME = "detail.json"


def resolve_result_root(explicit_root: Path | None = None) -> Path:
    """Resolve the root directory results are written to."""
    if explicit_root is not None:
        return explicit_root.expanduser().resolve()
    env_value = os.getenv("BENCHMARK_RESULT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return DEFAULT_RESULT_ROOT


def _slugify(value: str, *, fallback: str, max_length: int = 64) -> str:
    """Convert to an ASCII slug that is safe as a path fragment."""
    normalized = "".join(
        char if char.isascii() and (char.isalnum() or char in {"-", "_"}) else "_"
        for char in value.strip()
    )
    collapsed = re.sub(r"_+", "_", normalized).strip("_")
    if not collapsed:
        collapsed = fallback
    return collapsed[:max_length]


def _task_selection_label(task_ids: list[str]) -> str:
    """Summarize a task_ids list into a short label."""
    if not task_ids:
        return "all"
    if len(task_ids) == 1:
        return task_ids[0]
    if len(task_ids) <= 3:
        return "__".join(task_ids)
    return f"{task_ids[0]}+{len(task_ids) - 1}"


def resolve_user_name(explicit_user_name: str | None = None) -> str:
    """Resolve the user name recorded with a result."""
    candidates = [
        explicit_user_name,
        os.getenv("BENCHMARK_USER_NAME"),
        os.getenv("LOGNAME"),
        os.getenv("USER"),
        os.getenv("USERNAME"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str):
            normalized = candidate.strip()
            if normalized:
                return normalized
    try:
        candidate = getpass.getuser().strip()
    except Exception:
        candidate = ""
    return candidate or "unknown"


def _canonicalize_participants(participants: Mapping[str, Any]) -> dict[str, str]:
    """Stringify URL values into a dict with a stable ordering."""
    normalized: dict[str, str] = {}
    for role in sorted(participants):
        value = str(participants[role]).strip()
        if "://" in value:
            value = value.rstrip("/")
        normalized[str(role)] = value
    return normalized


def _compose_result_dir_name(
    *,
    benchmark_name: str,
    executor_name: str,
    target: str,
    task_label: str,
    config_hash: str,
    user_name: str,
    run_id: str,
) -> str:
    """Build a result directory name that is identifiable one level under experiments/."""
    benchmark_slug = _slugify(benchmark_name, fallback="benchmark", max_length=40)
    executor_slug = _slugify(executor_name, fallback="executor", max_length=24)
    target_slug = _slugify(target, fallback="target", max_length=24)
    task_slug = _slugify(task_label, fallback="tasks", max_length=48)
    user_slug = _slugify(user_name, fallback="unknown", max_length=24)
    run_slug = _slugify(run_id, fallback="run", max_length=32)
    return (
        f"bm-{benchmark_slug}"
        f"_ex-{executor_slug}"
        f"_tg-{target_slug}"
        f"_ts-{task_slug}"
        f"_cf-{config_hash}"
        f"_us-{user_slug}"
        f"_rn-{run_slug}"
    )


def _json_default(value: Any) -> Any:
    """Stringify values that are not JSON-serializable, deterministically."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def build_execution_identity(
    *,
    benchmark_name: str,
    executor_name: str,
    request_config: Mapping[str, Any],
    participants: Mapping[str, Any],
    result_root: Path | None = None,
    run_id: str | None = None,
    user_name: str | None = None,
    config_hash: str | None = None,
    created_at_utc: datetime | None = None,
) -> BenchmarkRunPaths:
    """Derive the result directory and run_id from the run configuration."""
    resolved_root = resolve_result_root(result_root)
    created_at = created_at_utc or datetime.now(UTC)
    task_ids = [str(task_id) for task_id in request_config.get("task_ids", []) or []]
    target = str(request_config.get("target") or "default")
    task_label = _task_selection_label(task_ids)
    resolved_user_name = resolve_user_name(user_name)
    normalized_participants = _canonicalize_participants(participants)

    effective_config_hash = config_hash
    if effective_config_hash is None:
        hash_source = {
            "benchmark_name": benchmark_name,
            "executor_name": executor_name,
            "participants": normalized_participants,
            "request_config": dict(request_config),
        }
        effective_config_hash = hashlib.sha1(
            json.dumps(
                hash_source,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=_json_default,
            ).encode("utf-8")
        ).hexdigest()[:12]

    effective_run_id = run_id or f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{effective_config_hash}"
    result_dir = resolved_root / _compose_result_dir_name(
        benchmark_name=benchmark_name,
        executor_name=executor_name,
        target=target,
        task_label=task_label,
        config_hash=effective_config_hash,
        user_name=resolved_user_name,
        run_id=effective_run_id,
    )
    return BenchmarkRunPaths(
        result_root=resolved_root,
        result_dir=result_dir,
        manifest_path=result_dir / RESULT_MANIFEST_FILE_NAME,
        detail_path=result_dir / RESULT_DETAIL_FILE_NAME,
        run_id=effective_run_id,
        user_name=resolved_user_name,
        config_hash=effective_config_hash,
        task_selection_label=task_label,
        created_at_utc=created_at,
    )


def ensure_result_dir(paths: BenchmarkRunPaths) -> None:
    """Create the result directory."""
    paths.result_dir.mkdir(parents=True, exist_ok=True)


def write_json_file(path: Path, payload: Any) -> Path:
    """Write a JSON file as UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path


def write_result_artifacts(
    *,
    detail_payload: Mapping[str, Any],
    manifest: BenchmarkRunManifest,
    paths: BenchmarkRunPaths,
) -> tuple[Path, Path]:
    """Write detail.json and manifest.json."""
    ensure_result_dir(paths)
    detail_path = write_json_file(paths.detail_path, detail_payload)
    manifest_path = write_json_file(paths.manifest_path, manifest)
    return detail_path, manifest_path


def iter_result_manifest_paths(result_root: Path | None = None) -> Iterable[Path]:
    """List the manifest.json files that follow the convention."""
    root = resolve_result_root(result_root)
    if not root.exists():
        return []
    pattern = "bm-*_ex-*_tg-*_ts-*_cf-*_rn-*/manifest.json"
    return sorted(root.glob(pattern), reverse=True)


def load_result_manifest(path: Path) -> BenchmarkRunManifest:
    """Load a manifest.json into its model."""
    return BenchmarkRunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def build_manifest_search_blob(manifest: BenchmarkRunManifest) -> str:
    """Flatten manifest contents into one searchable string."""
    return json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).lower()
