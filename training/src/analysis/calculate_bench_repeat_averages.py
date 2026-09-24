"""Average EJEPA-Bench repeat summary metrics per JSON file."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLOB = "results/bench_repeat_summaries/ejepa-bench-repeat-*.json"

RECORD_CONTAINER_KEYS = (
    "records",
    "results",
    "task_records",
    "task_results",
    "tasks",
    "runs",
    "samples",
    "summaries",
    "repeat_summaries",
)

SUCCESS_RATE_KEYS = (
    "success_rate",
    "overall_success_rate",
    "task_success_rate",
    "avg_success_rate",
    "average_success_rate",
)
SUCCESS_BOOL_KEYS = ("success", "successful", "is_success", "passed")
SUCCESS_COUNT_KEYS = (
    ("successful_tasks", "total_tasks"),
    ("num_successful_tasks", "num_tasks"),
    ("successful_runs", "total_runs"),
    ("num_successful", "num_total"),
    ("passed", "total"),
)
INFERENCE_TIME_KEYS = (
    "inference_time_per_task",
    "avg_inference_time_per_task",
    "average_inference_time_per_task",
    "mean_inference_time_per_task",
    "inference_time",
    "inference_time_seconds",
    "total_inference_time",
    "duration",
    "duration_seconds",
    "elapsed_time",
    "elapsed_seconds",
)
TOOL_CALL_KEYS = (
    "average_tool_calls",
    "avg_tool_calls",
    "mean_tool_calls",
    "average_number_of_tool_calls",
    "num_tool_calls",
    "tool_call_count",
    "tool_calls_count",
    "total_tool_calls",
    "tools_called",
    "average_tools_called",
)
UNNECESSARY_TOOL_CALL_KEYS = (
    "average_unnecessary_tool_calls",
    "avg_unnecessary_tool_calls",
    "mean_unnecessary_tool_calls",
    "average_number_of_unnecessary_tool_calls",
    "num_unnecessary_tool_calls",
    "unnecessary_tool_call_count",
    "unnecessary_tool_calls_count",
    "total_unnecessary_tool_calls",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "JSON files to summarize. If omitted, uses "
            f"{DEFAULT_GLOB!r} relative to the repo root."
        ),
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help="Glob to use when no explicit paths are passed.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text table.",
    )
    return parser.parse_args()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def as_number(value: Any) -> float | None:
    if is_number(value):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            try:
                return float(stripped[:-1]) / 100.0
            except ValueError:
                return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def get_first_number(mapping: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in mapping:
            value = as_number(mapping[key])
            if value is not None:
                return value
    return None


def get_first_bool(mapping: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
    return None


def get_ratio(mapping: dict[str, Any], keys: tuple[tuple[str, str], ...]) -> float | None:
    for numerator_key, denominator_key in keys:
        numerator = as_number(mapping.get(numerator_key))
        denominator = as_number(mapping.get(denominator_key))
        if numerator is not None and denominator:
            return numerator / denominator
    return None


def count_list(mapping: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, list):
            return float(len(value))
    return None


def has_target_metric(mapping: dict[str, Any]) -> bool:
    metric_keys = (
        SUCCESS_RATE_KEYS
        + SUCCESS_BOOL_KEYS
        + INFERENCE_TIME_KEYS
        + TOOL_CALL_KEYS
        + UNNECESSARY_TOOL_CALL_KEYS
    )
    return any(key in mapping for key in metric_keys) or any(
        numerator in mapping and denominator in mapping
        for numerator, denominator in SUCCESS_COUNT_KEYS
    )


def as_record_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, dict)]
    return []


def find_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        return []

    if has_target_metric(payload):
        return [payload]

    candidate_records: list[dict[str, Any]] = []
    for key in RECORD_CONTAINER_KEYS:
        records = as_record_list(payload.get(key))
        if records:
            candidate_records.extend(records)

    if candidate_records:
        return candidate_records

    for value in payload.values():
        records = find_records(value)
        if records:
            return records

    return []


def success_rate_for_record(record: dict[str, Any]) -> float | None:
    for extractor in (
        lambda item: get_first_number(item, SUCCESS_RATE_KEYS),
        lambda item: get_first_bool(item, SUCCESS_BOOL_KEYS),
        lambda item: get_ratio(item, SUCCESS_COUNT_KEYS),
    ):
        value = extractor(record)
        if value is not None:
            return value
    return None


def inference_time_for_record(record: dict[str, Any]) -> float | None:
    return get_first_number(record, INFERENCE_TIME_KEYS)


def tool_calls_for_record(record: dict[str, Any]) -> float | None:
    value = get_first_number(record, TOOL_CALL_KEYS)
    if value is not None:
        return value
    return count_list(record, ("tool_calls", "tools_used", "tool_results"))


def unnecessary_tool_calls_for_record(record: dict[str, Any]) -> float | None:
    value = get_first_number(record, UNNECESSARY_TOOL_CALL_KEYS)
    if value is not None:
        return value
    return count_list(record, ("unnecessary_tool_calls",))


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_path(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = find_records(payload)
    metrics = {
        "success_rate": [],
        "inference_time_per_task": [],
        "average_tool_calls": [],
        "average_unnecessary_tool_calls": [],
    }

    for record in records:
        metric_extractors = {
            "success_rate": success_rate_for_record,
            "inference_time_per_task": inference_time_for_record,
            "average_tool_calls": tool_calls_for_record,
            "average_unnecessary_tool_calls": unnecessary_tool_calls_for_record,
        }
        for metric_name, extractor in metric_extractors.items():
            value = extractor(record)
            if value is not None:
                metrics[metric_name].append(value)

    return {
        "path": str(path),
        "records": len(records),
        "averages": {
            metric_name: average(values) for metric_name, values in metrics.items()
        },
        "counts": {metric_name: len(values) for metric_name, values in metrics.items()},
    }


def resolve_paths(args: argparse.Namespace) -> list[Path]:
    if args.paths:
        return args.paths

    pattern = args.glob
    if not Path(pattern).is_absolute():
        pattern = str(ROOT / pattern)
    return [Path(path) for path in sorted(glob.glob(pattern))]


def format_value(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if percent:
        return f"{value:.4%}"
    return f"{value:.6f}"


def print_text(summaries: list[dict[str, Any]]) -> None:
    headers = (
        "file",
        "records",
        "success_rate",
        "inference_time_per_task",
        "avg_tool_calls",
        "avg_unnecessary_tool_calls",
    )
    rows = []
    for summary in summaries:
        averages = summary["averages"]
        rows.append(
            (
                summary["path"],
                str(summary["records"]),
                format_value(averages["success_rate"], percent=True),
                format_value(averages["inference_time_per_task"]),
                format_value(averages["average_tool_calls"]),
                format_value(averages["average_unnecessary_tool_calls"]),
            )
        )

    widths = [
        max(len(str(row[index])) for row in (headers, *rows))
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    if not paths:
        raise SystemExit(f"No JSON files matched {args.glob!r}.")

    summaries = [summarize_path(path) for path in paths]
    if args.json:
        print(json.dumps(summaries, indent=2))
        return

    print_text(summaries)


if __name__ == "__main__":
    main()
