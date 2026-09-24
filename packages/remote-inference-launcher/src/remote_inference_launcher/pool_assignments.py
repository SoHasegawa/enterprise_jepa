"""Assignment record storage helpers for remote inference endpoint pools."""

from __future__ import annotations

import fcntl
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from remote_inference_launcher.leases import parse_ttl
from remote_inference_launcher.registry import read_json, write_json_atomic
from remote_inference_launcher.summaries import new_run_id
from remote_inference_launcher.tunnels import DEFAULT_POOL_ROOT

ACTIVE_ASSIGNMENT_STATES = {"probing", "active"}


def load_assignments(
    *,
    pool_root: str | Path = DEFAULT_POOL_ROOT,
) -> dict[str, dict[str, object]]:
    """Load assignment records, marking stale active records expired."""

    root = _assignment_root(pool_root)
    assignments: dict[str, dict[str, object]] = {}
    if not root.exists():
        return assignments
    for path in sorted(root.glob("*.json")):
        try:
            payload = read_json(path)
        except OSError:
            continue
        if not isinstance(payload, Mapping):
            continue
        record = dict(payload)
        if _assignment_is_expired(record) and record.get("status") in ACTIVE_ASSIGNMENT_STATES:
            record["status"] = "expired"
            write_json_atomic(path, record)
        assignments[str(record.get("assignment_id", "") or path.stem)] = record
    return assignments


def read_assignment(
    assignment_id: str,
    *,
    pool_root: str | Path = DEFAULT_POOL_ROOT,
) -> dict[str, object]:
    """Read one assignment record by id."""

    path = _assignment_root(pool_root) / f"{assignment_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Assignment not found: {assignment_id}")
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Assignment record must be a JSON object: {path}")
    record = dict(payload)
    if _assignment_is_expired(record) and record.get("status") in ACTIVE_ASSIGNMENT_STATES:
        record["status"] = "expired"
        write_json_atomic(path, record)
    return record


def _assignment_expires_at(created_at: str, assignment_ttl: str) -> str:
    if not assignment_ttl:
        return ""
    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    return (created + parse_ttl(assignment_ttl)).isoformat().replace("+00:00", "Z")


def _assignment_is_expired(record: Mapping[str, object]) -> bool:
    expires_at = str(record.get("expires_at", "") or "")
    if not expires_at:
        return False
    try:
        expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return expires <= datetime.now(UTC)


def _assignment_state(assignment: Mapping[str, object]) -> str:
    return str(assignment.get("status", "") or "free") if assignment else "free"


def _active_assignment(
    endpoint_id: str,
    assignments: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    for assignment in assignments.values():
        if assignment.get("endpoint_id") != endpoint_id:
            continue
        if assignment.get("status") in ACTIVE_ASSIGNMENT_STATES:
            return dict(assignment)
    return {}


def _assignment_root(pool_root: str | Path) -> Path:
    root = Path(pool_root).expanduser() / "assignments"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _new_assignment_id() -> str:
    timestamp = new_run_id().rsplit("-", 1)[0]
    return f"assign-{timestamp}-{uuid.uuid4().hex[:8]}"


def _new_batch_id() -> str:
    timestamp = new_run_id().rsplit("-", 1)[0]
    return f"batch-{timestamp}-{uuid.uuid4().hex[:8]}"


def _resolve_batch_shards(shard_ids: list[str], *, count: int | None) -> list[str]:
    if count is not None and count < 1:
        raise ValueError("Batch count must be positive.")
    if not shard_ids:
        if count is None:
            raise ValueError("Batch acquisition requires shard IDs or --count.")
        shard_ids = [f"shard_{index:03d}" for index in range(count)]
    normalized = [str(shard_id) for shard_id in shard_ids]
    duplicates = sorted({shard_id for shard_id in normalized if normalized.count(shard_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate shard IDs in batch request: {', '.join(duplicates)}")
    if count is not None and len(normalized) != count:
        raise ValueError(f"Batch count {count} does not match shard count {len(normalized)}.")
    return normalized


@contextmanager
def _assignment_lock(pool_root: str | Path):
    lock_dir = Path(pool_root).expanduser() / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "assignments.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
