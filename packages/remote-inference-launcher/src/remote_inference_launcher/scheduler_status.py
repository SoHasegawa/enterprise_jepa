"""Batched Slurm scheduler status and partition recommendation helpers."""

from __future__ import annotations

import datetime as dt
import shlex
import subprocess
from collections import defaultdict
from collections.abc import Mapping

from remote_inference_launcher.remote_execution import (
    CommandRunner,
    RemoteCommand,
    RemoteCommandPolicy,
    run_ssh_batch,
)

SCHEDULER_SCHEMA_VERSION = "ril-scheduler-report/v1"
PARTITION_RECOMMENDATION_SCHEMA_VERSION = "ril-partition-recommendation/v1"
_SQUEUE_MARKER = "__RIL_POOL_SQUEUE__"
_START_MARKER = "__RIL_POOL_START__"
_PARTITION_MARKER = "__RIL_POOL_PARTITIONS__"


def refresh_slurm_scheduler(
    endpoints: tuple[Mapping[str, object], ...],
    *,
    command_runner: CommandRunner | None = None,
    policy: RemoteCommandPolicy | None = None,
    useful_deadline: str = "",
) -> dict[str, dict[str, object]]:
    """Refresh Slurm status for endpoints with one SSH batch per target."""

    runner = command_runner or _run_command
    command_policy = policy or RemoteCommandPolicy()
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for endpoint in endpoints:
        ssh_target = str(endpoint.get("ssh_target", "") or "")
        job_id = str(endpoint.get("job_id", "") or "")
        if ssh_target and job_id:
            groups[ssh_target].append(endpoint)
    refreshed: dict[str, dict[str, object]] = {}
    for ssh_target, grouped in sorted(groups.items()):
        job_ids = tuple(str(endpoint.get("job_id", "")) for endpoint in grouped)
        command = RemoteCommand(
            name="slurm-pool-status",
            command=_slurm_pool_status_command(job_ids),
        )
        result = run_ssh_batch(
            ssh_target=ssh_target,
            commands=(command,),
            command_runner=runner,
            policy=command_policy,
        )[0]
        parsed = _parse_slurm_pool_status(result.stdout)
        for endpoint in grouped:
            endpoint_id = str(endpoint.get("endpoint_id", "") or "")
            job_id = str(endpoint.get("job_id", "") or "")
            refreshed[endpoint_id] = _scheduler_report_for_endpoint(
                endpoint,
                ssh_target=ssh_target,
                job_id=job_id,
                job_info=parsed["jobs"].get(job_id, {}),
                start_info=parsed["starts"].get(job_id, ""),
                useful_deadline=useful_deadline,
                command_summary=result.command_summary,
            )
    return refreshed


def partition_recommendations(
    requests: tuple[Mapping[str, object], ...],
    *,
    command_runner: CommandRunner | None = None,
    policy: RemoteCommandPolicy | None = None,
) -> dict[str, object]:
    """Return cheap Slurm partition compatibility recommendations by target."""

    runner = command_runner or _run_command
    command_policy = policy or RemoteCommandPolicy()
    by_target: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for request in requests:
        ssh_target = str(request.get("ssh_target", "") or "")
        if ssh_target:
            by_target[ssh_target].append(request)
    targets: dict[str, object] = {}
    for ssh_target, target_requests in sorted(by_target.items()):
        result = run_ssh_batch(
            ssh_target=ssh_target,
            commands=(
                RemoteCommand(
                    name="slurm-partition-inventory",
                    command=_partition_inventory_command(),
                ),
            ),
            command_runner=runner,
            policy=command_policy,
        )[0]
        partitions = _parse_partition_inventory(result.stdout)
        targets[ssh_target] = {
            "partitions": [
                _recommend_for_request(request, partitions=partitions)
                for request in target_requests
            ],
            "inventory": partitions,
            "command_summary": result.command_summary,
        }
    return {
        "schema_version": PARTITION_RECOMMENDATION_SCHEMA_VERSION,
        "targets": targets,
    }


def _slurm_pool_status_command(job_ids: tuple[str, ...]) -> str:
    quoted_ids = ",".join(shlex.quote(job_id) for job_id in job_ids)
    loop_ids = " ".join(shlex.quote(job_id) for job_id in job_ids)
    return "\n".join(
        [
            f"printf '%s\\n' {shlex.quote(_SQUEUE_MARKER)}",
            f"squeue -j {quoted_ids} -h -o '%i|%T|%N|%R|%P' 2>/dev/null || true",
            f"printf '%s\\n' {shlex.quote(_START_MARKER)}",
            f"for job_id in {loop_ids}; do",
            "  printf '%s|' \"$job_id\"",
            "  squeue --start -j \"$job_id\" -h 2>/dev/null | sed -n '1p'",
            "  printf '\\n'",
            "done",
            f"printf '%s\\n' {shlex.quote(_PARTITION_MARKER)}",
            _partition_inventory_command(),
        ]
    )


def _partition_inventory_command() -> str:
    return "sinfo -h -o '%P|%t|%l|%G|%D|%f' 2>/dev/null || true"


def _parse_slurm_pool_status(output: str) -> dict[str, dict[str, object]]:
    section = ""
    jobs: dict[str, dict[str, object]] = {}
    starts: dict[str, str] = {}
    partitions: list[dict[str, object]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {_SQUEUE_MARKER, _START_MARKER, _PARTITION_MARKER}:
            section = line
            continue
        if section == _SQUEUE_MARKER:
            parsed = _parse_squeue_line(line)
            if parsed:
                jobs[str(parsed["job_id"])] = parsed
        elif section == _START_MARKER:
            job_id, _, start_line = line.partition("|")
            starts[job_id] = start_line.strip()
        elif section == _PARTITION_MARKER:
            partition = _parse_partition_line(line)
            if partition:
                partitions.append(partition)
    return {"jobs": jobs, "starts": starts, "partitions": {"items": partitions}}


def _parse_squeue_line(line: str) -> dict[str, object]:
    parts = line.split("|", maxsplit=4)
    if len(parts) < 2:
        return {}
    return {
        "job_id": parts[0],
        "slurm_state": parts[1],
        "node": parts[2] if len(parts) > 2 else "",
        "slurm_reason": parts[3] if len(parts) > 3 else "",
        "partition": parts[4] if len(parts) > 4 else "",
    }


def _scheduler_report_for_endpoint(
    endpoint: Mapping[str, object],
    *,
    ssh_target: str,
    job_id: str,
    job_info: Mapping[str, object],
    start_info: str,
    useful_deadline: str,
    command_summary: str,
) -> dict[str, object]:
    estimated_start = _estimated_start_from_squeue_start(start_info)
    expires_at = str(endpoint.get("expires_at", "") or "")
    return {
        "schema_version": SCHEDULER_SCHEMA_VERSION,
        "endpoint_id": endpoint.get("endpoint_id", ""),
        "job_id": job_id,
        "ssh_target": ssh_target,
        "slurm_state": job_info.get("slurm_state", "NOT_FOUND"),
        "slurm_reason": job_info.get("slurm_reason", ""),
        "partition": job_info.get("partition", ""),
        "node": job_info.get("node", ""),
        "pending_duration_seconds": _slurm_pending_seconds(endpoint),
        "estimated_start_at": estimated_start,
        "estimated_start_source": "squeue --start" if estimated_start else "",
        "start_after_lease_expiry": _timestamp_after(estimated_start, expires_at),
        "start_after_useful_deadline": _timestamp_after(estimated_start, useful_deadline),
        "raw_excerpt": start_info[:1000],
        "command_summary": command_summary,
    }


def _slurm_pending_seconds(endpoint: Mapping[str, object]) -> float | None:
    slurm = endpoint.get("slurm")
    if not isinstance(slurm, Mapping):
        return None
    value = slurm.get("pending_duration_seconds")
    if value in {"", None}:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _estimated_start_from_squeue_start(line: str) -> str:
    for token in line.replace("|", " ").split():
        if len(token) >= 10 and token[4:5] == "-" and token[7:8] == "-":
            return token
    return ""


def _timestamp_after(first: str, second: str) -> bool | None:
    if not first or not second:
        return None
    first_timestamp = _parse_timestamp(first)
    second_timestamp = _parse_timestamp(second)
    if first_timestamp is None or second_timestamp is None:
        return None
    return first_timestamp > second_timestamp


def _parse_timestamp(value: str) -> dt.datetime | None:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _parse_partition_inventory(output: str) -> list[dict[str, object]]:
    by_name: dict[str, dict[str, object]] = {}
    for line in output.splitlines():
        partition = _parse_partition_line(line)
        if not partition:
            continue
        name = str(partition["name"])
        if name in by_name:
            by_name[name] = _merge_partition(by_name[name], partition)
        else:
            by_name[name] = {**partition, "states": [partition["state"]]}
    return list(by_name.values())


def _parse_partition_line(line: str) -> dict[str, object]:
    parts = line.strip().split("|", maxsplit=5)
    if len(parts) < 3:
        return {}
    name = parts[0].rstrip("*")
    return {
        "name": name,
        "state": parts[1],
        "max_walltime": parts[2],
        "gres": parts[3] if len(parts) > 3 else "",
        "nodes": _parse_int(parts[4]) if len(parts) > 4 else None,
        "features": parts[5] if len(parts) > 5 else "",
    }


def _merge_partition(
    existing: Mapping[str, object],
    incoming: Mapping[str, object],
) -> dict[str, object]:
    states = _merged_values(existing.get("states", ()), incoming.get("state", ""))
    existing_nodes = _parse_int(existing.get("nodes")) or 0
    incoming_nodes = _parse_int(incoming.get("nodes")) or 0
    existing_gres = str(existing.get("gres", "") or "")
    incoming_gres = str(incoming.get("gres", "") or "")
    return {
        **dict(existing),
        "state": _representative_partition_state(states),
        "states": states,
        "max_walltime": _max_walltime(
            str(existing.get("max_walltime", "") or ""),
            str(incoming.get("max_walltime", "") or ""),
        ),
        "gres": _max_gres(existing_gres, incoming_gres),
        "nodes": existing_nodes + incoming_nodes,
        "features": " ".join(
            _merged_values(existing.get("features", ""), incoming.get("features", ""))
        ),
    }


def _merged_values(existing: object, incoming: object) -> list[str]:
    values: list[str] = []
    if isinstance(existing, list | tuple):
        candidates = [*existing, incoming]
    else:
        candidates = [existing, incoming]
    for value in candidates:
        text = str(value or "").strip()
        if not text or text in {"(null)", "none", "N/A"} or text in values:
            continue
        values.append(text)
    return values


def _representative_partition_state(states: list[str]) -> str:
    schedulable = [state for state in states if _partition_state_is_schedulable(state)]
    if schedulable:
        return schedulable[0]
    return states[0] if states else ""


def _max_walltime(first: str, second: str) -> str:
    first_seconds = _walltime_seconds(first)
    second_seconds = _walltime_seconds(second)
    if first_seconds is None:
        return second or first
    if second_seconds is None:
        return first or second
    if first_seconds < 0 or second_seconds < 0:
        return first if first_seconds < 0 else second
    return first if first_seconds >= second_seconds else second


def _max_gres(first: str, second: str) -> str:
    return first if _max_gpus_from_gres(first) >= _max_gpus_from_gres(second) else second


def _recommend_for_request(
    request: Mapping[str, object],
    *,
    partitions: list[dict[str, object]],
) -> dict[str, object]:
    compatible = []
    rejected = []
    for partition in partitions:
        reasons = _partition_rejection_reasons(request, partition)
        if reasons:
            rejected.append({"partition": partition["name"], "reasons": reasons})
        else:
            compatible.append({"partition": partition["name"], "reasons": []})
    return {
        "request": dict(request),
        "compatible": compatible,
        "rejected": rejected,
    }


def _partition_rejection_reasons(
    request: Mapping[str, object],
    partition: Mapping[str, object],
) -> list[str]:
    reasons: list[str] = []
    if not _partition_is_schedulable(partition):
        reasons.append("partition_not_schedulable")
    requested_walltime = str(request.get("walltime", "") or "")
    max_walltime = str(partition.get("max_walltime", "") or "")
    if requested_walltime and max_walltime and not _walltime_fits(requested_walltime, max_walltime):
        reasons.append("walltime_exceeded")
    requested_gpus = _parse_int(request.get("num_gpus")) or 0
    if (
        requested_gpus > 0
        and _max_gpus_from_gres(str(partition.get("gres", "") or "")) < requested_gpus
    ):
        reasons.append("gres_insufficient")
    return reasons


def _partition_is_schedulable(partition: Mapping[str, object]) -> bool:
    states = partition.get("states")
    if isinstance(states, list | tuple):
        return any(_partition_state_is_schedulable(str(state)) for state in states)
    return _partition_state_is_schedulable(str(partition.get("state", "") or ""))


def _partition_state_is_schedulable(state: str) -> bool:
    return state.lower().rstrip("*") in {"idle", "alloc", "mix", "comp", "resv", "plnd", "up"}


def _walltime_fits(requested: str, maximum: str) -> bool:
    max_seconds = _walltime_seconds(maximum)
    requested_seconds = _walltime_seconds(requested)
    if max_seconds is None or requested_seconds is None:
        return True
    return max_seconds < 0 or requested_seconds <= max_seconds


def _walltime_seconds(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"infinite", "unlimited"}:
        return -1
    days = 0
    clock = lowered
    if "-" in clock:
        day_text, _, clock = clock.partition("-")
        parsed_days = _parse_int(day_text)
        if parsed_days is None:
            return None
        days = parsed_days
    parts = clock.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = "0", parts[0], parts[1]
    elif len(parts) == 1:
        hours, minutes, seconds = "0", parts[0], "0"
    else:
        return None
    parsed = [_parse_int(item) for item in (hours, minutes, seconds)]
    if any(item is None for item in parsed):
        return None
    parsed_hours, parsed_minutes, parsed_seconds = (int(item) for item in parsed)
    return days * 86400 + parsed_hours * 3600 + parsed_minutes * 60 + parsed_seconds


def _max_gpus_from_gres(value: str) -> int:
    max_gpus = 0
    for item in value.replace(",", " ").split():
        if item in {"", "(null)", "none", "N/A"}:
            continue
        if not item.startswith("gpu"):
            continue
        count_text = item.rsplit(":", maxsplit=1)[-1] if ":" in item else "1"
        count_text = count_text.split("(", maxsplit=1)[0].split("[", maxsplit=1)[0]
        count = _parse_int(count_text) or 1
        max_gpus = max(max_gpus, count)
    return max_gpus


def _parse_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _run_command(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout_seconds,
    )
