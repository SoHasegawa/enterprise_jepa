#!/usr/bin/env python3
"""Report repeated EJEPA benchmark averages from summary JSON files."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

DEFAULT_INPUT_GLOB = "results/bench_repeat_summaries/ejepa-bench-repeat-*.json"
PASS_SCORE_EPSILON = 1e-9

# Benchmarks whose ``score_rate`` is a mean partial-credit score rather than a task
# success rate, so the success rate must be recomputed from the per-task records.
# Workspace-Bench reports the rubric pass rate averaged over tasks.
RUBRIC_SCORED_BENCHMARKS = {"workspacebench"}

# Benchmarks that publish the task success rate under a different key than
# ``score_rate``. Keyed by :func:`normalized_name`.
#
# * AutomationBench ``score_rate`` is upstream's *partial credit* (fraction of
#   assertions satisfied), so a task meeting 4 of 5 assertions contributes 0.8.
#   The success rate is ``pass_rate`` = ``task_completed_correctly``.
# * crmarenapro grades every answer twice from one run: ``score_rate`` /
#   ``summary_pass_rate`` come from this repo's own pattern-matching grader, while
#   ``original_scores_accuracy`` is upstream Salesforce CRMArena-Pro's own
#   evaluator -- the number comparable to the published leaderboard.
SUCCESS_METRIC_OVERRIDES: dict[str, dict[str, Any]] = {
    "automationbench": {
        "metric": "pass_rate",
        "aggregate_paths": (
            ("aggregate", "benchmark_metrics", "pass_rate"),
            ("aggregate", "pass_rate"),
        ),
        "run_paths": (("benchmark_metrics", "pass_rate"), ("pass_rate",)),
    },
    "crmarenapro": {
        "metric": "original_accuracy",
        "aggregate_paths": (
            ("aggregate", "benchmark_metrics", "original_scores_accuracy"),
            ("aggregate", "benchmark_metrics", "original_scores_accuracy_percent"),
        ),
        "run_paths": (
            ("benchmark_metrics", "original_scores_accuracy"),
            ("benchmark_metrics", "original_scores_accuracy_percent"),
        ),
    },
}
FILTERED_FIELDS = [
    "filtered_tasks",
    "filtered_success_rate",
    "filtered_pass_rate",
]
FIELDS = [
    "file",
    "benchmark",
    "executor",
    "target",
    "runs_with_result",
    "success_metric",
    "success_rate",
    "score_rate",
    "inference_time_per_task_seconds",
    "avg_tool_calls",
    "avg_unnecessary_tool_calls",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print per-file averages for success rate, inference time per task, "
            "tool calls, and unnecessary tool calls. The success_metric column "
            "names which definition each row uses: task_pass_rate for rubric-scored "
            "benchmarks (Workspace-Bench), pass_rate for AutomationBench "
            "(task_completed_correctly, not partial credit), original_accuracy for "
            "crmarenapro (upstream CRMArena-Pro's own grader), else score_rate."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional summary JSON files. Defaults to --input-glob when omitted.",
    )
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help="Glob used when no positional paths are provided.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--exclude-domains",
        nargs="*",
        default=[],
        metavar="DOMAIN",
        help=(
            "Add clearly-labelled filtered_* columns computed over tasks whose "
            "task_id domain prefix (before the first '.') is NOT in this list, e.g. "
            "--exclude-domains marketing finance. The unfiltered success_rate is "
            "always kept, so the headline number never changes silently."
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Significant digits for table and CSV numeric values.",
    )
    return parser.parse_args(argv)


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
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


def mean(values: Iterable[float | None]) -> float | None:
    nums = [value for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def nested_get(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def numeric_summary_mean(value: Any) -> float | None:
    if isinstance(value, Mapping):
        return safe_float(value.get("mean"))
    return safe_float(value)


def normalized_rate(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 100.0 if value > 1.0 else value


def first_summary_mean(
    data: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
    *,
    rate: bool = False,
) -> float | None:
    for path in paths:
        value = numeric_summary_mean(nested_get(data, path))
        if value is not None:
            return normalized_rate(value) if rate else value
    return None


def run_summaries(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    runs = data.get("runs")
    if not isinstance(runs, list):
        return []
    summaries = []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        summary = run.get("result_summary")
        if isinstance(summary, Mapping):
            summaries.append(summary)
    return summaries


def mean_from_run_summaries(
    data: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
    *,
    rate: bool = False,
) -> float | None:
    values = []
    for summary in run_summaries(data):
        for path in paths:
            value = numeric_summary_mean(nested_get(summary, path))
            if value is not None:
                values.append(normalized_rate(value) if rate else value)
                break
    return mean(values)


def normalized_name(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def task_succeeded(record: Mapping[str, Any]) -> bool | None:
    """Whether one per-task record counts as a success, or None when unknown."""
    success = record.get("success")
    if isinstance(success, bool):
        return success
    score = safe_float(record.get("score"))
    if score is None:
        return None
    return score >= 1.0 - PASS_SCORE_EPSILON


def per_task_records(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for path in (("per_task",), ("benchmark_metrics", "per_task")):
        records = nested_get(summary, path)
        if isinstance(records, list):
            return [record for record in records if isinstance(record, Mapping)]
    return []


def rubric_task_pass_rate(data: Mapping[str, Any]) -> float | None:
    """Mean over runs of (#succeeded tasks / #evaluated tasks) from per-task scores."""
    rates = []
    for summary in run_summaries(data):
        outcomes = [task_succeeded(record) for record in per_task_records(summary)]
        known = [outcome for outcome in outcomes if outcome is not None]
        if known:
            rates.append(sum(1 for outcome in known if outcome) / len(known))
    return mean(rates)


def first_metric(
    data: Mapping[str, Any],
    aggregate_paths: Sequence[Sequence[str]],
    run_paths: Sequence[Sequence[str]],
    *,
    rate: bool = False,
) -> float | None:
    value = first_summary_mean(data, aggregate_paths, rate=rate)
    if value is not None:
        return value
    return mean_from_run_summaries(data, run_paths, rate=rate)


def inference_time_per_task(data: Mapping[str, Any]) -> float | None:
    explicit = first_metric(
        data,
        [
            ("aggregate", "agentic_task_metrics", "avg_task_execution_time_seconds"),
            ("aggregate", "benchmark_metrics", "timing_avg_seconds_per_task"),
            ("aggregate", "benchmark_metrics", "avg_seconds_per_task"),
        ],
        [
            ("agentic_task_metrics", "avg_task_execution_time_seconds"),
            ("benchmark_metrics", "timing_avg_seconds_per_task"),
            ("benchmark_metrics", "avg_seconds_per_task"),
        ],
    )
    if explicit is not None:
        return explicit

    run_values = []
    for summary in run_summaries(data):
        duration = first_summary_mean(
            summary,
            [
                ("duration_seconds",),
                ("benchmark_metrics", "duration_seconds"),
                ("benchmark_metrics", "timing_total_seconds"),
            ],
        )
        total_tasks = first_summary_mean(
            summary,
            [("total_tasks",), ("benchmark_metrics", "total_tasks"), ("summary", "total_tasks")],
        )
        if duration is not None and total_tasks and total_tasks > 0:
            run_values.append(duration / total_tasks)
    if run_values:
        return mean(run_values)

    duration = first_summary_mean(
        data,
        [
            ("aggregate", "duration_seconds"),
            ("aggregate", "benchmark_metrics", "duration_seconds"),
            ("aggregate", "benchmark_metrics", "timing_total_seconds"),
        ],
    )
    total_tasks = first_summary_mean(
        data,
        [
            ("aggregate", "benchmark_metrics", "total_tasks"),
            ("aggregate", "benchmark_metrics", "summary_total_tasks"),
        ],
    )
    if duration is not None and total_tasks and total_tasks > 0:
        return duration / total_tasks
    return None


def command_arg_after(data: Mapping[str, Any], flag: str) -> str | None:
    command = data.get("command")
    if not isinstance(command, list):
        return None
    for index, item in enumerate(command):
        if item == flag and index + 1 < len(command):
            return str(command[index + 1])
        prefix = f"{flag}="
        if isinstance(item, str) and item.startswith(prefix):
            return item[len(prefix) :]
    return None


def benchmark_from_command(data: Mapping[str, Any]) -> str | None:
    command = data.get("command")
    if not isinstance(command, list):
        return None
    for index, item in enumerate(command):
        if item == "run" and index + 1 < len(command):
            return str(command[index + 1])
    return None


def target_from_command(data: Mapping[str, Any]) -> str | None:
    command = data.get("command")
    if not isinstance(command, list):
        return None
    for index, item in enumerate(command):
        if item == "--config" and index + 1 < len(command):
            value = str(command[index + 1])
        elif isinstance(item, str) and item.startswith("--config="):
            value = item.split("=", 1)[1]
        else:
            continue
        if value.startswith("target="):
            return value.split("=", 1)[1]
    return None


def first_run_metadata(data: Mapping[str, Any], key: str) -> str | None:
    for summary in run_summaries(data):
        value = summary.get(key)
        if value is not None:
            return str(value)
    return None


def metadata_value(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is not None:
        return str(value)
    return first_run_metadata(data, key)


def runs_with_result(data: Mapping[str, Any]) -> int | None:
    value = first_summary_mean(data, [("aggregate", "runs_with_result")])
    if value is not None:
        return int(value)
    summaries = run_summaries(data)
    return len(summaries) if summaries else None


_SCORED_ASSERTIONS_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+scored assertions")
PASS_SCORE_EPSILON = 1e-9


def task_domain(task_id: Any) -> str:
    """Domain prefix of a task id (``finance.ap_aging_report`` -> ``finance``)."""
    text = str(task_id or "")
    return text.split(".", 1)[0] if "." in text else ""


def unique_per_task(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Per-task records de-duplicated by task_id (last write wins).

    The internal-trajectory artifact is re-emitted cumulatively, so ``per_task``
    can hold several rows per task (1032 rows for a 600-task AutomationBench run);
    averaging the raw list double-counts.
    """
    records = summary.get("per_task")
    if not isinstance(records, list):
        return []
    deduped: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if isinstance(record, Mapping) and record.get("task_id") is not None:
            deduped[str(record["task_id"])] = record
    return list(deduped.values())


def scorable_assertions(record: Mapping[str, Any]) -> int | None:
    """Denominator from ``reason`` ("0/5 scored assertions satisfied"), else None."""
    match = _SCORED_ASSERTIONS_RE.search(str(record.get("reason") or ""))
    return int(match.group(2)) if match else None


def filtered_run_metrics(
    summary: Mapping[str, Any], excluded: set[str]
) -> tuple[float, float, int] | None:
    """(score_rate, pass_rate, kept) for one run, ignoring ``excluded`` domains."""
    records = unique_per_task(summary)
    if not records:
        return None
    scores: list[float] = []
    passes = 0
    for record in records:
        if task_domain(record.get("task_id")).lower() in excluded:
            continue
        if scorable_assertions(record) == 0:
            # Nothing scorable can ever pass; excluding keeps the rate meaningful.
            continue
        score = record.get("score")
        if not isinstance(score, int | float) or isinstance(score, bool):
            continue
        scores.append(float(score))
        if float(score) >= 1.0 - PASS_SCORE_EPSILON:
            passes += 1
    if not scores:
        return None
    return sum(scores) / len(scores), passes / len(scores), len(scores)


def filtered_metrics(data: Mapping[str, Any], excluded: Sequence[str]) -> dict[str, Any] | None:
    """Averaged over runs, or None when no domains are excluded / no per-task data."""
    drop = {str(name).strip().lower() for name in excluded if str(name).strip()}
    if not drop:
        return None
    per_run = [filtered_run_metrics(s, drop) for s in run_summaries(data)]
    per_run = [r for r in per_run if r is not None]
    if not per_run:
        return None
    return {
        "filtered_tasks": round(mean([float(r[2]) for r in per_run]) or 0.0),
        "filtered_success_rate": mean([r[0] for r in per_run]),
        "filtered_pass_rate": mean([r[1] for r in per_run]),
    }


def default_score_rate(data: Mapping[str, Any]) -> float | None:
    """The generic ``score_rate`` average used by benchmarks without an override."""
    return first_metric(
        data,
        [
            ("aggregate", "score_rate"),
            ("aggregate", "benchmark_metrics", "score_rate"),
            ("aggregate", "benchmark_metrics", "pass_rate"),
            ("aggregate", "benchmark_metrics", "summary_pass_rate"),
        ],
        [
            ("score_rate",),
            ("benchmark_metrics", "score_rate"),
            ("benchmark_metrics", "pass_rate"),
            ("benchmark_metrics", "summary_pass_rate"),
        ],
        rate=True,
    )


def resolve_success_rate(data: Mapping[str, Any], benchmark: Any) -> tuple[float | None, str]:
    """(success_rate, metric_name) using the right definition for this benchmark."""
    name = normalized_name(benchmark)

    override = SUCCESS_METRIC_OVERRIDES.get(name)
    if override is not None:
        value = first_metric(data, override["aggregate_paths"], override["run_paths"], rate=True)
        if value is not None:
            return value, override["metric"]
        # The preferred aggregate is absent -- e.g. crmarenapro run with
        # skip_original=true never computes the upstream accuracy. Fall back, but
        # say so in the label rather than reporting a different metric silently.
        fallback = default_score_rate(data)
        if fallback is not None:
            return fallback, f"score_rate (no {override['metric']})"
        return None, override["metric"]

    if name in RUBRIC_SCORED_BENCHMARKS:
        # score_rate averages rubric pass rates, so recount succeeded tasks.
        return rubric_task_pass_rate(data), "task_pass_rate"

    return default_score_rate(data), "score_rate"


def summarize_file(path: Path, excluded: Sequence[str] = ()) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a JSON object")

    benchmark = metadata_value(data, "benchmark_name") or benchmark_from_command(data)
    success_rate, success_metric = resolve_success_rate(data, benchmark)
    filtered = filtered_metrics(data, excluded)

    row = {
        "file": str(path),
        "benchmark": benchmark,
        "executor": metadata_value(data, "executor_name") or command_arg_after(data, "--executor"),
        "target": metadata_value(data, "target") or target_from_command(data),
        "runs_with_result": runs_with_result(data),
        "success_metric": success_metric,
        "success_rate": success_rate,
        # The benchmark's native score_rate is always reported next to the chosen
        # metric, so crmarenapro shows the repo pass rate beside original_accuracy
        # (and AutomationBench its partial-credit rate beside the strict pass rate).
        "score_rate": default_score_rate(data),
        "inference_time_per_task_seconds": inference_time_per_task(data),
        "avg_tool_calls": first_metric(
            data,
            [("aggregate", "agentic_task_metrics", "avg_tool_calls")],
            [("agentic_task_metrics", "avg_tool_calls")],
        ),
        "avg_unnecessary_tool_calls": first_metric(
            data,
            [("aggregate", "agentic_task_metrics", "avg_unnecessary_tool_calls")],
            [("agentic_task_metrics", "avg_unnecessary_tool_calls")],
        ),
    }
    if filtered is not None:
        row.update(filtered)
    return row


def input_paths(args: argparse.Namespace) -> list[Path]:
    if args.paths:
        return sorted(dict.fromkeys(path.expanduser() for path in args.paths))
    matches = glob.glob(args.input_glob)
    return sorted(dict.fromkeys(Path(match).expanduser() for match in matches))


def format_number(value: Any, precision: int) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def active_fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Base columns plus the filtered_* ones, only when some row carries them."""
    extra = [f for f in FILTERED_FIELDS if any(f in row for row in rows)]
    return [*FIELDS, *extra]


def rows_for_display(rows: Sequence[Mapping[str, Any]], precision: int) -> list[dict[str, str]]:
    fields = active_fields(rows)
    return [{field: format_number(row.get(field), precision) for field in fields} for row in rows]


def render_table(rows: Sequence[Mapping[str, Any]], precision: int) -> str:
    fields = active_fields(rows)
    display_rows = rows_for_display(rows, precision)
    widths = {
        field: max(len(field), *(len(row[field]) for row in display_rows)) for field in fields
    }
    header = "  ".join(field.ljust(widths[field]) for field in fields)
    separator = "  ".join("-" * widths[field] for field in fields)
    body = ["  ".join(row[field].ljust(widths[field]) for field in fields) for row in display_rows]
    return "\n".join([header, separator, *body])


def render_csv(rows: Sequence[Mapping[str, Any]], precision: int) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=active_fields(rows), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows_for_display(rows, precision))
    return output.getvalue().rstrip("\n")


def render_json(rows: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(list(rows), indent=2, ensure_ascii=False, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = input_paths(args)
    if not paths:
        print(f"No files matched {args.input_glob!r}", file=sys.stderr)
        return 1

    rows = [summarize_file(path, args.exclude_domains) for path in paths]
    if args.format == "csv":
        print(render_csv(rows, args.precision))
    elif args.format == "json":
        print(render_json(rows))
    else:
        print(render_table(rows, args.precision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
