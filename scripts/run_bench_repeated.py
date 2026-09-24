#!/usr/bin/env python3
"""Run ``ejepa bench run`` repeatedly and summarize task, cost, and WM metrics.

The script is intentionally benchmark-agnostic.  It reads the canonical
``detail.json`` written by ``ejepa bench run`` and, when trajectory capture is
enabled, augments the summary with per-task execution time, tool-call counts,
failed tool-call counts, and beam-plan world-model telemetry.

Usage:

    python scripts/run_bench_repeated.py --runs 3 \
      --result-root "$BENCHMARK_HOME/experiments" \
      -- ejepa bench run EnterpriseOps-Gym --executor mcp_react ...

By default the wrapper appends ``--config capture_trajectory=true`` when the
command does not already request trajectory capture, because WM/cost metrics are
not reliably recoverable from ``detail.json`` alone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

COMMON_DETAIL_KEYS = {
    "schema_version",
    "run_id",
    "user_name",
    "status",
    "benchmark_name",
    "benchmark_version",
    "green_agent_version",
    "purple_agent_version",
    "executor_version",
    "executor_name",
    "target",
    "eval_type",
    "task_ids",
    "task_selection_label",
    "config_hash",
    "created_at_utc",
    "completed_at_utc",
    "participants",
    "request_config",
    "details",
    "fatal_error",
    "trajectory_capture",
    "executor_runtime",
    "agentic_eval_mode",
    "agent_replay_mode",
    "agent_replay_state",
    "summary",
    "entropic",
    "dimension_averages",
    "by_category",
    "extension_metrics",
    "original",
    "timing",
    "result_dir",
    "version",
}


@dataclass(frozen=True)
class RunDiscovery:
    result_root: Path | None
    before_dirs: set[Path]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a ejepa bench command repeatedly and summarize performance/cost/WM metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of repeated benchmark runs.")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=None,
        help="Directory where ejepa writes result directories. Used to find each new result.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bench_repeat_summaries"),
        help="Directory for wrapper logs and aggregate JSON.",
    )
    parser.add_argument(
        "--label", default=None, help="Optional label embedded in output filenames."
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first subprocess failure. By default all repeats are attempted.",
    )
    parser.add_argument(
        "--no-auto-capture-trajectory",
        action="store_true",
        help="Do not append --config capture_trajectory=true to the ejepa command.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Do not run a command; summarize existing result directories passed after --.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after --, or result directories when --summarize-only is set.",
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("pass a command after --, or result directories with --summarize-only")
    return args


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def command_has_capture_trajectory(command: Sequence[str]) -> bool:
    joined = "\n".join(command).lower()
    return (
        "capture_trajectory=true" in joined
        or "capture_trajectories=true" in joined
        or "benchmark_capture_trajectory=true" in joined
    )


def effective_command(command: Sequence[str], *, auto_capture_trajectory: bool) -> list[str]:
    result = list(command)
    if auto_capture_trajectory and not command_has_capture_trajectory(result):
        result.extend(["--config", "capture_trajectory=true"])
    return result


def infer_result_root(command: Sequence[str], explicit: Path | None) -> Path | None:
    if explicit is not None:
        return expand_path(explicit)
    for index, arg in enumerate(command):
        if arg == "--result-root" and index + 1 < len(command):
            return expand_path(command[index + 1])
        if arg.startswith("--result-root="):
            return expand_path(arg.split("=", 1)[1])
    benchmark_home = os.getenv("BENCHMARK_HOME")
    if benchmark_home:
        return expand_path(Path(benchmark_home) / "experiments")
    local_results = Path("results/experiments")
    if local_results.exists():
        return expand_path(local_results)
    return None


def result_dirs_under(result_root: Path | None) -> set[Path]:
    if result_root is None or not result_root.exists():
        return set()
    return {
        path.resolve()
        for path in result_root.iterdir()
        if path.is_dir() and (path / "detail.json").exists()
    }


def prepare_discovery(command: Sequence[str], explicit_result_root: Path | None) -> RunDiscovery:
    result_root = infer_result_root(command, explicit_result_root)
    return RunDiscovery(result_root=result_root, before_dirs=result_dirs_under(result_root))


def discover_new_result_dir(discovery: RunDiscovery, *, started_at_monotonic: float) -> Path | None:
    root = discovery.result_root
    if root is None or not root.exists():
        return None
    after = result_dirs_under(root)
    candidates = sorted(
        after - discovery.before_dirs,
        key=lambda path: (path / "detail.json").stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    # If the runner overwrote/reused a directory, fall back to a recently modified detail.json.
    wall_started = time.time() - max(0.0, time.monotonic() - started_at_monotonic) - 2.0
    recent = sorted(
        (path for path in after if (path / "detail.json").stat().st_mtime >= wall_started),
        key=lambda path: (path / "detail.json").stat().st_mtime,
        reverse=True,
    )
    return recent[0] if recent else None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            stripped = stripped[:-1]
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def mean(values: Iterable[float | int | None]) -> float | None:
    nums = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def population_std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def numeric_summary(values: Iterable[Any]) -> dict[str, Any]:
    nums = [value for value in (safe_float(v) for v in values) if value is not None]
    return {
        "count": len(nums),
        "mean": mean(nums),
        "min": min(nums) if nums else None,
        "max": max(nums) if nums else None,
        "std": population_std(nums),
        "values": nums,
    }


def scalar_benchmark_metrics(detail: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in detail.items():
        if key in COMMON_DETAIL_KEYS:
            continue
        if isinstance(value, str | int | float | bool) or value is None:
            metrics[key] = value
    return metrics


def nested_get(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def first_numeric(*values: Any) -> float | None:
    for value in values:
        parsed = safe_float(value)
        if parsed is not None:
            return parsed
    return None


def task_rows_from_detail(detail: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = [
        detail.get("details"),
        detail.get("results"),
        detail.get("task_results"),
        nested_get(detail, ["details", "task_results"]),
        nested_get(detail, ["details", "detail_records"]),
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return [row for row in candidate if isinstance(row, Mapping)]
        if isinstance(candidate, Mapping):
            return [row for row in candidate.values() if isinstance(row, Mapping)]
    return []


def task_id_from_row(row: Mapping[str, Any]) -> str | None:
    for key in ("task_id", "id", "task_idx", "task_index", "name"):
        value = row.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return None


def normalized_total_tasks(
    detail: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> int | None:
    value = first_numeric(
        detail.get("total_tasks"),
        nested_get(detail, ["summary", "total_tasks"]),
        nested_get(detail, ["entropic", "summary", "total_tasks"]),
        nested_get(detail, ["original", "summary", "total_tasks"]),
    )
    if value is not None:
        return int(value)
    return len(rows) if rows else None


def normalized_score_rate(
    detail: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> float | None:
    explicit = first_numeric(
        detail.get("score_rate"),
        detail.get("pass_rate"),
        detail.get("accuracy"),
        nested_get(detail, ["summary", "pass_rate"]),
        nested_get(detail, ["entropic", "summary", "pass_rate"]),
        nested_get(detail, ["original", "scores", "accuracy"]),
    )
    if explicit is not None:
        return explicit / 100.0 if explicit > 1.0 else explicit
    successes = [normalize_success(row) for row in rows]
    successes = [value for value in successes if value is not None]
    if successes:
        return sum(1 for value in successes if value) / len(successes)
    scores = [
        safe_float(first_present(row.get("score"), row.get("crm_reward"), row.get("reward")))
        for row in rows
    ]
    scores = [value for value in scores if value is not None]
    if scores and all(0.0 <= value <= 1.0 for value in scores):
        return sum(scores) / len(scores)
    return None


def normalized_total_score(
    detail: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> float | None:
    explicit = first_numeric(
        detail.get("total_score"),
        nested_get(detail, ["summary", "total_passed"]),
        nested_get(detail, ["entropic", "summary", "total_passed"]),
        nested_get(detail, ["original", "summary", "passed"]),
    )
    if explicit is not None:
        return explicit
    values = [
        safe_float(first_present(row.get("score"), row.get("crm_reward"), row.get("reward")))
        for row in rows
    ]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def normalized_duration_seconds(detail: Mapping[str, Any]) -> float | None:
    return first_numeric(
        detail.get("duration_seconds"),
        nested_get(detail, ["timing", "total_seconds"]),
    )


def infer_from_result_dir(result_dir: Path, marker: str) -> str | None:
    marker_boundary = r"_(?:bm|ex|tg|ts|cf|us|rn)-|$"
    match = re.search(rf"(?:^|_){re.escape(marker)}-(.*?)({marker_boundary})", result_dir.name)
    return match.group(1) if match else None


def normalized_status(detail: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str | None:
    status = detail.get("status")
    if status is not None:
        return str(status)
    if detail.get("fatal_error"):
        return "failed"
    if rows or normalized_score_rate(detail, rows) is not None:
        return "completed"
    return None


def extended_benchmark_metrics(detail: Mapping[str, Any]) -> dict[str, Any]:
    metrics = scalar_benchmark_metrics(detail)
    for prefix, source in (
        ("summary", detail.get("summary")),
        ("entropic_summary", nested_get(detail, ["entropic", "summary"])),
        ("dimension", detail.get("dimension_averages")),
        ("timing", detail.get("timing")),
        ("original_scores", nested_get(detail, ["original", "scores"])),
    ):
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            if isinstance(value, str | int | float | bool) or value is None:
                metrics[f"{prefix}_{key}"] = value
    return metrics


def detail_task_index(detail: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in task_rows_from_detail(detail):
        task_id = task_id_from_row(row)
        if task_id:
            indexed[task_id] = row
    return indexed


def iter_text_parts(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        if isinstance(value.get("text"), str):
            yield value["text"]
        for item in value.values():
            yield from iter_text_parts(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_text_parts(item)


def iter_data_parts(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        data = value.get("data")
        if isinstance(data, Mapping):
            yield data
        for item in value.values():
            yield from iter_data_parts(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_data_parts(item)


def regex_int_sum(pattern: str, text: str) -> int:
    total = 0
    for match in re.finditer(pattern, text):
        try:
            total += int(match.group(1))
        except (IndexError, ValueError):
            continue
    return total


def regex_count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))


def parse_json_text(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def purple_payloads_from_event(event: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    event_type = str(event.get("event_type") or "")
    shell_payload = event.get("payload")
    if event_type in {"ShellProtocolExecRequest", "ShellProtocolExecResult"} and isinstance(
        shell_payload, Mapping
    ):
        kind = "exec_request" if event_type.endswith("ExecRequest") else "exec_result"
        return [
            {
                "_shell_protocol_kind": kind,
                "command": first_present(event.get("command"), shell_payload.get("command")),
                "exit_code": first_present(event.get("exit_code"), shell_payload.get("exit_code")),
            }
        ]

    if event.get("event_type") == "PurpleInternalRecord" and isinstance(
        event.get("payload"), Mapping
    ):
        return [event["payload"]]

    payloads: list[Mapping[str, Any]] = []
    for data in iter_data_parts(event):
        if any(key in data for key in ("metrics", "wm_steps", "wm_strategy", "task_id")):
            payload: dict[str, Any] = {"task_id": data.get("task_id")}
            info: dict[str, Any] = {}
            if isinstance(data.get("metrics"), Mapping):
                info["metrics"] = data["metrics"]
            if data.get("wm_steps") is not None:
                info["wm_steps_count"] = data.get("wm_steps")
            if data.get("wm_strategy") is not None:
                info["wm_strategy"] = data.get("wm_strategy")
            if data.get("wm_backend") is not None:
                info["wm_backend"] = data.get("wm_backend")
            if info:
                payload["info"] = info
            payloads.append(payload)

    for text in iter_text_parts(event):
        decoded = parse_json_text(text)
        if (
            isinstance(decoded, Mapping)
            and decoded.get("source") == "purple_executor"
            and isinstance(decoded.get("payload"), Mapping)
        ):
            payloads.append(decoded["payload"])
            continue
        # Some benchmarks truncate artifact text to BENCHMARK_TRAJECTORY_MAX_TEXT_CHARS, leaving
        # the JSON unterminated. Keep enough raw text to count WM events and action changes.
        if '"source": "purple_executor"' in text and ('"wm_steps"' in text or '"wm_react"' in text):
            task_match = re.search(r'"task_id"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
            payloads.append(
                {"task_id": task_match.group(1) if task_match else None, "_wm_text": text}
            )
    return payloads


def load_trajectory_payloads(path: Path) -> list[Mapping[str, Any]]:
    payloads: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, Mapping):
                payloads.extend(purple_payloads_from_event(event))
    return payloads


def payload_signature(payload: Mapping[str, Any]) -> str:
    if isinstance(payload.get("_wm_text"), str):
        return "wm_text:" + payload["_wm_text"][:512]
    info = payload.get("info")
    if isinstance(info, Mapping):
        try:
            return "info:" + json.dumps(
                {"task_id": payload.get("task_id"), "info": info},
                ensure_ascii=False,
                sort_keys=True,
            )
        except TypeError:
            return "info:" + repr((payload.get("task_id"), info))
    compact_keys = ("task_id", "run_number", "execution_time_ms", "overall_success", "wm_react")
    compact = {key: payload.get(key) for key in compact_keys if key in payload}
    try:
        return json.dumps(compact or payload, ensure_ascii=False, sort_keys=True, default=str)[
            :4096
        ]
    except TypeError:
        return repr(compact or payload)[:4096]


def dedupe_payloads(payloads: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[str] = set()
    unique: list[Mapping[str, Any]] = []
    for payload in payloads:
        signature = payload_signature(payload)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(payload)
    return unique


def task_id_from_trajectory_path(path: Path) -> str:
    return path.stem


def trajectory_paths(result_dir: Path, detail: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for row in task_rows_from_detail(detail):
        raw = first_present(
            row.get("trajectory_file_path"),
            row.get("trajectory_path"),
            row.get("trajectory"),
        )
        if not raw:
            continue
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = result_dir / candidate
        if candidate.exists():
            paths.append(candidate.resolve())
    if paths:
        return sorted(set(paths))
    root = result_dir / "trajectories"
    if not root.exists():
        return []
    return sorted(path.resolve() for path in root.rglob("*.jsonl"))


def result_failed(tool_result: Mapping[str, Any]) -> bool:
    result = tool_result.get("result")
    if not isinstance(result, Mapping):
        return False
    if result.get("success") is False:
        return True
    if result.get("error") not in (None, "", []):
        return True
    nested = result.get("result")
    return isinstance(nested, Mapping) and nested.get("isError") is True


def tool_results_from_flow(flow: Sequence[Any]) -> list[Mapping[str, Any]]:
    return [
        item for item in flow if isinstance(item, Mapping) and item.get("type") == "tool_result"
    ]


def count_ai_tool_calls(flow: Sequence[Any]) -> int:
    total = 0
    for item in flow:
        if not isinstance(item, Mapping) or item.get("type") != "ai_message":
            continue
        calls = item.get("tool_calls")
        if isinstance(calls, list):
            total += len(calls)
    return total


def is_beam_plan_step(step: Mapping[str, Any]) -> bool:
    return step.get("strategy") == "beam_plan" or str(step.get("event") or "").startswith(
        "GYM_BEAM_PLAN"
    )


# Beam steps that spent NO planning cycle. GYM_BEAM_PLAN_FOLLOW (executing an already-cached
# plan) and GYM_BEAM_PLAN_NO_SEED/UNSUPPORTED (nothing to plan for) are not re-plans; counting
# them made the critic trigger -- which follows a cached plan rarely and re-plans often -- look
# cheaper than the interval trigger that re-plans on a fixed cadence.
NON_REPLAN_BEAM_EVENTS = frozenset(
    {
        "GYM_BEAM_PLAN_COOLDOWN",
        "GYM_BEAM_PLAN_FOLLOW",
        "GYM_BEAM_PLAN_NO_SEED",
        "GYM_BEAM_PLAN_UNSUPPORTED",
    }
)


def is_beam_replan(step: Mapping[str, Any]) -> bool:
    # Every step that reaches the re-plan decision reports it explicitly; trust that first and
    # fall back to the event name only for the early returns that carry no ``replanned`` key.
    replanned = step.get("replanned")
    if isinstance(replanned, bool):
        return replanned
    event = str(step.get("event") or "")
    return event.startswith("GYM_BEAM_PLAN") and event not in NON_REPLAN_BEAM_EVENTS


def wm_step_records_from_payload(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = []
    wm = payload.get("wm_react")
    if isinstance(wm, Mapping):
        candidates.append(wm.get("steps"))
    info = payload.get("info")
    if isinstance(info, Mapping):
        candidates.append(info.get("wm_steps"))
    candidates.append(payload.get("wm_steps"))

    steps: list[Mapping[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, list):
            continue
        for item in candidate:
            if not isinstance(item, Mapping):
                continue
            detail = item.get("detail")
            if isinstance(detail, Mapping):
                steps.append(detail)
            else:
                steps.append(item)
    return steps


def wm_metrics_from_text(text: str) -> dict[str, int]:
    return {
        "wm_steps": 0,
        "wm_beam_plan_steps": regex_count(r'"event"\s*:\s*"GYM_BEAM_PLAN', text),
        "wm_beam_planning_count": regex_count(r'"replanned"\s*:\s*true', text),
        "wm_beam_planning_success_count": regex_count(r'"event"\s*:\s*"GYM_BEAM_PLAN"', text),
        "wm_action_change_count": regex_count(r'"override_applied"\s*:\s*true', text),
        "wm_advice_injected_count": regex_count(r'"injected"\s*:\s*true', text),
        "wm_critic_check_count": regex_count(r'"critic"\s*:\s*\{', text),
        "wm_critic_fire_count": regex_count(r'"fires"\s*:\s*true', text),
        "wm_terminal_advice_count": regex_int_sum(
            r'"beam_plan_terminal_advice_count"\s*:\s*(\d+)', text
        ),
        "wm_imagined_plan_step_count": regex_int_sum(r'"imagined_plan_len"\s*:\s*(\d+)', text),
        "wm_beam_llm_call_count": regex_int_sum(r'"beam_llm_calls"\s*:\s*(\d+)', text),
        "wm_world_model_call_count": regex_int_sum(r'"world_model_calls"\s*:\s*(\d+)', text),
        "wm_judge_call_count": regex_int_sum(r'"judge_calls"\s*:\s*(\d+)', text),
        "wm_model_call_count": regex_int_sum(r'"model_calls"\s*:\s*(\d+)', text),
        "wm_beam_refinement_round_count": regex_int_sum(
            r'"beam_refinement_rounds_completed"\s*:\s*(\d+)', text
        ),
        "wm_beam_refinement_score_pass_count": regex_int_sum(
            r'"beam_refinement_score_passes"\s*:\s*(\d+)', text
        ),
        "wm_revision_count": regex_count(r'"strategy"\s*:\s*"revision"', text),
        "wm_reference_count": regex_count(r'"strategy"\s*:\s*"reference"', text),
    }


def wm_metrics_from_payload(payload: Mapping[str, Any]) -> dict[str, int]:
    raw_text = payload.get("_wm_text")
    if isinstance(raw_text, str):
        return wm_metrics_from_text(raw_text)

    info = payload.get("info")
    if isinstance(info, Mapping) and info.get("wm_steps_count") is not None:
        count = int(first_numeric(info.get("wm_steps_count")) or 0)
        is_beam = str(info.get("wm_strategy") or "").lower() == "beam_plan"
        return {
            "wm_steps": count,
            "wm_beam_plan_steps": count if is_beam else 0,
            "wm_beam_planning_count": 0,
            "wm_beam_planning_success_count": 0,
            "wm_action_change_count": 0,
            "wm_advice_injected_count": 0,
            "wm_critic_check_count": 0,
            "wm_critic_fire_count": 0,
            "wm_terminal_advice_count": 0,
            "wm_imagined_plan_step_count": 0,
            "wm_beam_llm_call_count": 0,
            "wm_world_model_call_count": 0,
            "wm_judge_call_count": 0,
            "wm_model_call_count": 0,
            "wm_beam_refinement_round_count": 0,
            "wm_beam_refinement_score_pass_count": 0,
            "wm_revision_count": 0,
            "wm_reference_count": 0,
        }

    steps = wm_step_records_from_payload(payload)
    beam_steps = [step for step in steps if isinstance(step, Mapping) and is_beam_plan_step(step)]
    replans = [step for step in beam_steps if is_beam_replan(step)]
    world_model_calls = 0
    judge_calls = 0
    model_calls = 0
    for step in steps:
        direct_world_calls = int(first_numeric(step.get("world_model_calls")) or 0)
        critic = step.get("critic")
        critic_world_calls = (
            int(first_numeric(critic.get("world_model_calls")) or 0)
            if isinstance(critic, Mapping)
            else 0
        )
        itp_world_calls = int(first_numeric(step.get("itp_i_world_model_calls")) or 0)
        world_model_calls += direct_world_calls + critic_world_calls + itp_world_calls
        judge_calls += int(first_numeric(step.get("judge_calls")) or 0)
        model_calls += int(first_numeric(step.get("model_calls")) or 0)
    return {
        "wm_steps": len(steps),
        "wm_beam_plan_steps": len(beam_steps),
        "wm_beam_planning_count": len(replans),
        "wm_beam_planning_success_count": sum(
            1 for step in replans if step.get("event") == "GYM_BEAM_PLAN"
        ),
        "wm_action_change_count": sum(
            1 for step in beam_steps if step.get("override_applied") is True
        ),
        "wm_advice_injected_count": sum(1 for step in beam_steps if step.get("injected") is True),
        "wm_critic_check_count": sum(
            1 for step in beam_steps if isinstance(step.get("critic"), Mapping)
        ),
        "wm_critic_fire_count": sum(
            1
            for step in beam_steps
            if isinstance(step.get("critic"), Mapping) and step["critic"].get("fires") is True
        ),
        "wm_terminal_advice_count": sum(
            int(step.get("beam_plan_terminal_advice_count") or 0) for step in beam_steps
        ),
        "wm_imagined_plan_step_count": sum(
            int(step.get("imagined_plan_len") or 0) for step in beam_steps
        ),
        "wm_beam_llm_call_count": sum(int(step.get("beam_llm_calls") or 0) for step in beam_steps),
        "wm_world_model_call_count": world_model_calls,
        "wm_judge_call_count": judge_calls,
        "wm_model_call_count": model_calls,
        "wm_beam_refinement_round_count": sum(
            int(first_numeric(step.get("beam_refinement_rounds_completed")) or 0)
            for step in beam_steps
        ),
        "wm_beam_refinement_score_pass_count": sum(
            int(first_numeric(step.get("beam_refinement_score_passes")) or 0) for step in beam_steps
        ),
        "wm_revision_count": sum(
            1
            for step in steps
            if step.get("strategy") == "revision" and step.get("event") == "action_feedback"
        ),
        "wm_reference_count": sum(
            1
            for step in steps
            if step.get("strategy") == "reference" and step.get("event") == "action_feedback"
        ),
    }


def summarize_trajectory(path: Path, detail_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payloads = dedupe_payloads(load_trajectory_payloads(path))
    task_id = str(
        task_id_from_row(detail_row or {})
        or next((p.get("task_id") for p in payloads if p.get("task_id")), None)
        or task_id_from_trajectory_path(path)
    )
    records = [
        payload
        for payload in payloads
        if "execution_time_ms" in payload
        or "tool_results" in payload
        or "tools_used" in payload
        or "wm_react" in payload
        or "wm_steps" in payload
        or "_wm_text" in payload
        or "_shell_protocol_kind" in payload
        or isinstance(payload.get("info"), Mapping)
    ]
    if not records:
        return {
            "task_id": task_id,
            "trajectory_file_path": str(path),
            "trajectory_payload_found": False,
        }

    execution_time_ms = mean(safe_float(payload.get("execution_time_ms")) for payload in records)
    tool_results: list[Mapping[str, Any]] = []
    tool_call_count = 0
    shell_failed_tool_calls = 0
    observed_tool_calls = False
    for payload in records:
        shell_kind = payload.get("_shell_protocol_kind")
        if shell_kind == "exec_request":
            observed_tool_calls = True
            tool_call_count += 1
            continue
        if shell_kind == "exec_result":
            exit_code = first_numeric(payload.get("exit_code"))
            if exit_code is not None and int(exit_code) != 0:
                shell_failed_tool_calls += 1
            continue
        current_results = payload.get("tool_results")
        if isinstance(current_results, list):
            observed_tool_calls = True
            typed_results = [item for item in current_results if isinstance(item, Mapping)]
            tool_results.extend(typed_results)
            tool_call_count += len(typed_results)
            continue
        metrics = nested_get(payload, ["info", "metrics"])
        if isinstance(metrics, Mapping):
            metric_tool_calls = first_numeric(metrics.get("tool_calls"), metrics.get("queries"))
            if metric_tool_calls is not None:
                observed_tool_calls = True
                tool_call_count += int(metric_tool_calls)
                continue
        flow = payload.get("conversation_flow")
        if isinstance(flow, list):
            observed_tool_calls = True
            fallback_results = tool_results_from_flow(flow)
            tool_results.extend(fallback_results)
            tool_call_count += len(fallback_results) or count_ai_tool_calls(flow)
        elif isinstance(payload.get("tools_used"), list):
            observed_tool_calls = True
            tool_call_count += len(payload["tools_used"])

    wm_totals: dict[str, int] = {}
    for payload in records:
        for key, value in wm_metrics_from_payload(payload).items():
            # A2A trajectories often repeat the same final purple telemetry as history,
            # internal text, and Answer.data. Treat records as snapshots, not deltas.
            wm_totals[key] = max(wm_totals.get(key, 0), value)

    failed_tool_calls = None
    if observed_tool_calls:
        failed_tool_calls = shell_failed_tool_calls + sum(
            1 for item in tool_results if result_failed(item)
        )
        if not tool_results:
            failed_tool_calls += sum(
                int(first_numeric(nested_get(payload, ["info", "metrics", "failed_queries"])) or 0)
                for payload in records
            )
    summary: dict[str, Any] = {
        "task_id": task_id,
        "trajectory_file_path": str(path),
        "trajectory_payload_found": True,
        "execution_time_seconds": execution_time_ms / 1000.0
        if execution_time_ms is not None
        else None,
        "tool_calls": tool_call_count if observed_tool_calls else None,
        "failed_tool_calls": failed_tool_calls,
        # Alias retained for paper tables where "unnecessary" is defined operationally.
        "unnecessary_tool_calls": failed_tool_calls,
    }
    summary.update(wm_totals)
    return summary


def normalize_success(row: Mapping[str, Any]) -> bool | None:
    for key in ("purple_overall_success", "success", "passed", "pass"):
        value = row.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "passed", "pass", "success", "1"}:
                return True
            if lowered in {"false", "no", "failed", "fail", "failure", "0"}:
                return False
    entropic = row.get("entropic")
    if isinstance(entropic, Mapping):
        nested = normalize_success(entropic)
        if nested is not None:
            return nested
    return None


def row_execution_time_seconds(row: Mapping[str, Any]) -> float | None:
    seconds = first_numeric(
        nested_get(row, ["timing", "purple_agent_seconds"]),
        nested_get(row, ["timing", "total_seconds"]),
        row.get("execution_time_seconds"),
        row.get("elapsed_seconds"),
        row.get("duration_seconds"),
    )
    if seconds is not None:
        return seconds
    millis = first_numeric(
        row.get("execution_time_ms"),
        nested_get(row, ["statistics", "mean_execution_time_ms"]),
    )
    return millis / 1000.0 if millis is not None else None


def row_tool_calls(row: Mapping[str, Any]) -> int | None:
    value = first_numeric(
        row.get("tool_calls"),
        nested_get(row, ["metrics", "tool_calls"]),
        nested_get(row, ["metrics", "queries"]),
        nested_get(row, ["statistics", "tool_calls"]),
    )
    if value is not None:
        return int(value)
    for key in ("purple_tools_used", "tools_used", "tool_results", "predicted_function_calls"):
        if isinstance(row.get(key), list):
            return len(row[key])
    return None


def row_failed_tool_calls(row: Mapping[str, Any]) -> int | None:
    value = first_numeric(
        row.get("failed_tool_calls"),
        nested_get(row, ["metrics", "failed_tool_calls"]),
        nested_get(row, ["metrics", "invalid_tool_calls"]),
        nested_get(row, ["metrics", "failed_queries"]),
    )
    return int(value) if value is not None else None


def merge_task_detail_metrics(
    row: Mapping[str, Any], trajectory_summary: Mapping[str, Any] | None
) -> dict[str, Any]:
    task_id = str(task_id_from_row(row) or (trajectory_summary or {}).get("task_id") or "")
    merged: dict[str, Any] = {
        "task_id": task_id,
        "score": first_present(
            row.get("score"), row.get("crm_reward"), row.get("reward"), row.get("total_score")
        ),
        "verifier_pass_rate": first_present(
            row.get("verifier_pass_rate"), nested_get(row, ["verification_summary", "pass_rate"])
        ),
        "success": normalize_success(row),
        "reason": row.get("reason"),
        "error": first_present(row.get("error"), row.get("executor_error")),
    }
    fallback_execution_time = row_execution_time_seconds(row)
    fallback_tool_calls = row_tool_calls(row)
    fallback_failed_tool_calls = row_failed_tool_calls(row)
    if fallback_execution_time is not None:
        merged["execution_time_seconds"] = fallback_execution_time
    if fallback_tool_calls is not None:
        merged["tool_calls"] = fallback_tool_calls
    if fallback_failed_tool_calls is not None:
        merged["failed_tool_calls"] = fallback_failed_tool_calls
        merged["unnecessary_tool_calls"] = fallback_failed_tool_calls
    if isinstance(row.get("purple_verification_summary"), Mapping):
        merged["verification_summary"] = dict(row["purple_verification_summary"])
    if isinstance(row.get("purple_tools_used"), list):
        merged["detail_tool_calls"] = len(row["purple_tools_used"])
    if trajectory_summary:
        for key, value in trajectory_summary.items():
            if value is None and merged.get(key) is not None:
                continue
            merged[key] = value
    return merged


def summarize_result_dir(result_dir: Path) -> dict[str, Any]:
    result_dir = expand_path(result_dir)
    detail_path = result_dir / "detail.json"
    if not detail_path.exists():
        raise FileNotFoundError(f"detail.json not found under {result_dir}")
    detail = load_json(detail_path)
    if not isinstance(detail, Mapping):
        raise ValueError(f"{detail_path} must contain a JSON object")

    rows = task_rows_from_detail(detail)
    rows_by_task = detail_task_index(detail)
    summaries_by_task: dict[str, Mapping[str, Any]] = {}
    for path in trajectory_paths(result_dir, detail):
        guessed = task_id_from_trajectory_path(path)
        row = rows_by_task.get(guessed)
        summary = summarize_trajectory(path, row)
        summaries_by_task[str(summary["task_id"])] = summary

    per_task: list[dict[str, Any]] = []
    for task_id, row in rows_by_task.items():
        per_task.append(merge_task_detail_metrics(row, summaries_by_task.get(task_id)))
    for task_id, summary in summaries_by_task.items():
        if task_id not in rows_by_task:
            per_task.append(merge_task_detail_metrics({"task_id": task_id}, summary))
    per_task.sort(key=lambda item: item.get("task_id") or "")

    aggregate = agentic_task_metrics(per_task)

    return {
        "result_dir": str(result_dir),
        "detail_path": str(detail_path),
        "benchmark_name": first_present(
            detail.get("benchmark_name"), infer_from_result_dir(result_dir, "bm")
        ),
        "benchmark_version": first_present(detail.get("benchmark_version"), detail.get("version")),
        "executor_name": first_present(
            detail.get("executor_name"), infer_from_result_dir(result_dir, "ex")
        ),
        "executor_version": detail.get("executor_version"),
        "target": first_present(detail.get("target"), infer_from_result_dir(result_dir, "tg")),
        "status": normalized_status(detail, rows),
        "total_tasks": normalized_total_tasks(detail, rows),
        "total_score": normalized_total_score(detail, rows),
        "score_rate": normalized_score_rate(detail, rows),
        "score_rate_percent": detail.get("score_rate_percent"),
        "duration_seconds": normalized_duration_seconds(detail),
        "fatal_error": detail.get("fatal_error"),
        "final_summary": detail.get("final_summary"),
        "benchmark_metrics": extended_benchmark_metrics(detail),
        "agentic_task_metrics": aggregate,
        "per_task": per_task,
    }


def agentic_task_metrics(per_task: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    critic_checks = sum(int(task.get("wm_critic_check_count") or 0) for task in per_task)
    critic_fires = sum(int(task.get("wm_critic_fire_count") or 0) for task in per_task)
    return {
        "avg_task_execution_time_seconds": mean(
            task.get("execution_time_seconds") for task in per_task
        ),
        "avg_tool_calls": mean(task.get("tool_calls") for task in per_task),
        "avg_unnecessary_tool_calls": mean(task.get("unnecessary_tool_calls") for task in per_task),
        "avg_failed_tool_calls": mean(task.get("failed_tool_calls") for task in per_task),
        "avg_wm_beam_planning_count": mean(task.get("wm_beam_planning_count") for task in per_task),
        "avg_wm_action_change_count": mean(task.get("wm_action_change_count") for task in per_task),
        "avg_wm_critic_fire_count": mean(task.get("wm_critic_fire_count") for task in per_task),
        "avg_wm_critic_check_count": mean(task.get("wm_critic_check_count") for task in per_task),
        # Fire rate over all critic checks, not the mean of per-task rates, so short tasks do
        # not outweigh long ones. This is what decides whether the critic trigger is cheaper
        # than the interval trigger it replaces (break-even = 1/wm-beam-mpc-execute-steps).
        "wm_critic_fire_rate": (critic_fires / critic_checks) if critic_checks else None,
        "avg_wm_imagined_plan_step_count": mean(
            task.get("wm_imagined_plan_step_count") for task in per_task
        ),
        "avg_wm_world_model_call_count": mean(
            task.get("wm_world_model_call_count") for task in per_task
        ),
        "avg_wm_judge_call_count": mean(task.get("wm_judge_call_count") for task in per_task),
        "avg_wm_model_call_count": mean(task.get("wm_model_call_count") for task in per_task),
        "avg_wm_beam_refinement_round_count": mean(
            task.get("wm_beam_refinement_round_count") for task in per_task
        ),
        "avg_wm_beam_refinement_score_pass_count": mean(
            task.get("wm_beam_refinement_score_pass_count") for task in per_task
        ),
        "avg_wm_revision_count": mean(task.get("wm_revision_count") for task in per_task),
        "avg_wm_reference_count": mean(task.get("wm_reference_count") for task in per_task),
        "total_tool_calls": sum(int(task.get("tool_calls") or 0) for task in per_task),
        "total_unnecessary_tool_calls": sum(
            int(task.get("unnecessary_tool_calls") or 0) for task in per_task
        ),
        "total_wm_beam_planning_count": sum(
            int(task.get("wm_beam_planning_count") or 0) for task in per_task
        ),
        "total_wm_action_change_count": sum(
            int(task.get("wm_action_change_count") or 0) for task in per_task
        ),
    }


def aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parsed = [run for run in runs if isinstance(run.get("result_summary"), Mapping)]
    benchmark_metric_keys = sorted(
        {
            key
            for run in parsed
            for key in (run["result_summary"].get("benchmark_metrics") or {})
            if safe_float((run["result_summary"].get("benchmark_metrics") or {}).get(key))
            is not None
        }
    )
    agentic_metric_keys = sorted(
        {
            key
            for run in parsed
            for key in (run["result_summary"].get("agentic_task_metrics") or {})
            if safe_float((run["result_summary"].get("agentic_task_metrics") or {}).get(key))
            is not None
        }
    )

    return {
        "runs_attempted": len(runs),
        "runs_with_result": len(parsed),
        "subprocess_success_count": sum(1 for run in runs if run.get("returncode") == 0),
        "benchmark_status_counts": {
            status: sum(1 for run in parsed if run["result_summary"].get("status") == status)
            for status in sorted({str(run["result_summary"].get("status")) for run in parsed})
        },
        "score_rate": numeric_summary(run["result_summary"].get("score_rate") for run in parsed),
        "duration_seconds": numeric_summary(
            run["result_summary"].get("duration_seconds") for run in parsed
        ),
        "wrapper_elapsed_seconds": numeric_summary(
            run.get("wrapper_elapsed_seconds") for run in runs
        ),
        "benchmark_metrics": {
            key: numeric_summary(
                (run["result_summary"].get("benchmark_metrics") or {}).get(key) for run in parsed
            )
            for key in benchmark_metric_keys
        },
        "agentic_task_metrics": {
            key: numeric_summary(
                (run["result_summary"].get("agentic_task_metrics") or {}).get(key) for run in parsed
            )
            for key in agentic_metric_keys
        },
    }


def run_command(command: Sequence[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return process.wait()


def output_basename(label: str | None) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not label:
        return f"ejepa-bench-repeat-{stamp}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip()).strip("-")
    return f"ejepa-bench-repeat-{safe}-{stamp}" if safe else f"ejepa-bench-repeat-{stamp}"


def summarize_only(paths: Sequence[str]) -> dict[str, Any]:
    runs = []
    for index, raw in enumerate(paths, 1):
        result_dir = expand_path(raw)
        runs.append(
            {
                "run_index": index,
                "returncode": None,
                "result_dir": str(result_dir),
                "result_summary": summarize_result_dir(result_dir),
            }
        )
    return {
        "schema_version": "1.0",
        "created_at_utc": utc_now_iso(),
        "mode": "summarize_only",
        "runs": runs,
        "aggregate": aggregate_runs(runs),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = output_basename(args.label)
    summary_path = output_dir / f"{basename}.json"

    if args.summarize_only:
        summary = summarize_only(args.command)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        print(f"Wrote summary: {summary_path}")
        return 0

    command = effective_command(
        args.command,
        auto_capture_trajectory=not args.no_auto_capture_trajectory,
    )
    result_root = infer_result_root(command, args.result_root)
    runs: list[dict[str, Any]] = []
    for run_index in range(1, args.runs + 1):
        discovery = prepare_discovery(command, args.result_root)
        log_path = output_dir / f"{basename}-run{run_index}.log"
        started_iso = utc_now_iso()
        started_mono = time.monotonic()
        print(f"=== ejepa bench repeat {run_index}/{args.runs} ===")
        print("Command:", " ".join(command))
        returncode = run_command(command, log_path)
        elapsed = time.monotonic() - started_mono
        result_dir = discover_new_result_dir(discovery, started_at_monotonic=started_mono)

        run_record: dict[str, Any] = {
            "run_index": run_index,
            "started_at_utc": started_iso,
            "completed_at_utc": utc_now_iso(),
            "wrapper_elapsed_seconds": elapsed,
            "returncode": returncode,
            "log_path": str(log_path),
            "result_root": str(result_root) if result_root else None,
            "result_dir": str(result_dir) if result_dir else None,
        }
        if result_dir is not None:
            try:
                run_record["result_summary"] = summarize_result_dir(result_dir)
            except Exception as exc:
                run_record["summary_error"] = f"{type(exc).__name__}: {exc}"
        runs.append(run_record)

        summary = {
            "schema_version": "1.0",
            "created_at_utc": utc_now_iso(),
            "mode": "run",
            "runs_requested": args.runs,
            "command": list(args.command),
            "effective_command": command,
            "auto_capture_trajectory": not args.no_auto_capture_trajectory,
            "result_root": str(result_root) if result_root else None,
            "runs": runs,
            "aggregate": aggregate_runs(runs),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )

        result_summary = run_record.get("result_summary") or {}
        agentic = result_summary.get("agentic_task_metrics") or {}
        print(
            "Run summary:",
            {
                "returncode": returncode,
                "result_dir": run_record.get("result_dir"),
                "score_rate": result_summary.get("score_rate"),
                "avg_task_execution_time_seconds": agentic.get("avg_task_execution_time_seconds"),
                "avg_tool_calls": agentic.get("avg_tool_calls"),
                "avg_failed_tool_calls": agentic.get("avg_failed_tool_calls"),
                "avg_wm_beam_planning_count": agentic.get("avg_wm_beam_planning_count"),
                "avg_wm_action_change_count": agentic.get("avg_wm_action_change_count"),
            },
        )
        print(f"Partial summary written: {summary_path}")
        if returncode != 0 and args.stop_on_failure:
            break

    return 0 if all(run.get("returncode") == 0 for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
