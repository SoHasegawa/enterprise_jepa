#!/usr/bin/env python3
"""Summarize WM harness task pass rate and per-task latency."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

DEFAULT_INPUT_GLOB = "results/wm_harness_summaries/ejepa-wm-harnesses-*.json"
# Benchmarks whose ``score_rate`` is a mean partial-credit score (Workspace-Bench
# reports the rubric pass rate averaged over tasks), so the task pass rate has to
# be recomputed from the per-task records instead.
RUBRIC_SCORED_BENCHMARKS = {"workspacebench"}
PASS_SCORE_EPSILON = 1e-9

# Benchmarks whose ``score_rate`` is not the task success rate, and the aggregate
# their green agent already publishes instead. Keyed by ``normalized_name``.
#
# * AutomationBench ``score_rate`` is upstream's *partial credit* (fraction of
#   assertions satisfied), so a task scoring 4/5 contributes 0.8. The success rate
#   is ``pass_rate`` = ``task_completed_correctly`` (every scored assertion met).
# * crmarenapro grades each answer twice from one run: ``score_rate`` /
#   ``summary_pass_rate`` come from this repo's own pattern-matching grader, while
#   ``original_scores_accuracy`` is upstream Salesforce CRMArena-Pro's own
#   evaluator -- the number comparable to the published leaderboard.
SUCCESS_METRIC_OVERRIDES: dict[str, dict[str, Any]] = {
    "automationbench": {
        "metric": "pass_rate",
        "rate_paths": (("benchmark_metrics", "pass_rate"), ("pass_rate",)),
        "count_paths": (("benchmark_metrics", "total_passed"), ("total_passed",)),
    },
    "crmarenapro": {
        "metric": "original_accuracy",
        "rate_paths": (("benchmark_metrics", "original_scores_accuracy"),),
        "percent_paths": (("benchmark_metrics", "original_scores_accuracy_percent"),),
        "count_paths": (),
    },
}
FILTERED_FIELDS = [
    "filtered_tasks",
    "filtered_success_rate_pct",
    "filtered_pass_rate_pct",
]
FIELDS = [
    "file",
    "created_at_utc",
    "benchmark",
    "executor",
    "target",
    "harness",
    "status",
    "returncode",
    "total_tasks",
    "succeeded_tasks",
    "success_metric",
    "success_rate_pct",
    "score_rate_pct",
    "latency_per_task_seconds",
    "duration_seconds",
    "wrapper_elapsed_seconds",
    "result_dir",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print one row per usable WM harness run, focused on task success rate "
            "(#succeeded tasks / #total tasks) and latency per task. Rows missing "
            "either metric are skipped. The success_metric column names which "
            "definition each row uses: task_pass_rate for rubric-scored benchmarks "
            "(Workspace-Bench), pass_rate for AutomationBench (task_completed_"
            "correctly, not partial credit), original_accuracy for crmarenapro "
            "(upstream CRMArena-Pro's own grader), else score_rate."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional harness summary JSON files. Defaults to --input-glob when omitted.",
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
        "--precision",
        type=int,
        default=4,
        help="Significant digits for table and CSV numeric values.",
    )
    parser.add_argument(
        "--show-skipped",
        action="store_true",
        help="Print skipped file/row counts to stderr.",
    )
    parser.add_argument(
        "--exclude-domains",
        nargs="*",
        default=[],
        metavar="DOMAIN",
        help=(
            "Add clearly-labelled filtered_* columns computed over tasks whose "
            "task_id domain prefix (before the first '.') is NOT in this list, e.g. "
            "--exclude-domains marketing. The unfiltered success_rate_pct is always "
            "kept, so the headline number never changes silently."
        ),
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


def nested_get(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def first_float(value: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> float | None:
    for path in paths:
        parsed = safe_float(nested_get(value, path))
        if parsed is not None:
            return parsed
    return None


def rate_to_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return value if value > 1.0 else value * 100.0


def normalized_name(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def is_rubric_scored_benchmark(benchmark: Any) -> bool:
    return normalized_name(benchmark) in RUBRIC_SCORED_BENCHMARKS


def success_metric_override(benchmark: Any) -> dict[str, Any] | None:
    """The aggregate to report as the success rate, when ``score_rate`` is not it."""
    return SUCCESS_METRIC_OVERRIDES.get(normalized_name(benchmark))


def per_task_records(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for path in (("per_task",), ("benchmark_metrics", "per_task")):
        records = nested_get(summary, path)
        if isinstance(records, list):
            return [record for record in records if isinstance(record, Mapping)]
    return []


_SCORED_ASSERTIONS_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+scored assertions")


def task_domain(task_id: Any) -> str:
    """Domain prefix of a task id (``finance.ap_aging_report`` -> ``finance``)."""
    text = str(task_id or "")
    return text.split(".", 1)[0] if "." in text else ""


def unique_per_task(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Per-task records de-duplicated by task_id (last write wins).

    The internal-trajectory artifact is re-emitted cumulatively, so ``per_task``
    can hold several rows per task (1002 rows for a 600-task AutomationBench run);
    averaging the raw list double-counts.
    """
    deduped: dict[str, Mapping[str, Any]] = {}
    for record in per_task_records(summary):
        if record.get("task_id") is not None:
            deduped[str(record["task_id"])] = record
    return list(deduped.values())


def scorable_assertions(record: Mapping[str, Any]) -> int | None:
    """Denominator from ``reason`` ("0/5 scored assertions satisfied"), else None."""
    match = _SCORED_ASSERTIONS_RE.search(str(record.get("reason") or ""))
    return int(match.group(2)) if match else None


def filtered_rates(summary: Mapping[str, Any], excluded: Sequence[str]) -> dict[str, Any] | None:
    """Score/pass rate over tasks outside ``excluded`` domains, or None if unusable."""
    drop = {str(name).strip().lower() for name in excluded if str(name).strip()}
    if not drop:
        return None
    scores: list[float] = []
    passes = 0
    for record in unique_per_task(summary):
        if task_domain(record.get("task_id")).lower() in drop:
            continue
        if scorable_assertions(record) == 0:
            # Nothing scorable can ever pass; excluding keeps the rate meaningful.
            continue
        score = safe_float(record.get("score"))
        if score is None:
            continue
        scores.append(score)
        if score >= 1.0 - PASS_SCORE_EPSILON:
            passes += 1
    if not scores:
        return None
    return {
        "filtered_tasks": len(scores),
        "filtered_success_rate_pct": sum(scores) / len(scores) * 100.0,
        "filtered_pass_rate_pct": passes / len(scores) * 100.0,
    }


def task_succeeded(record: Mapping[str, Any]) -> bool | None:
    success = record.get("success")
    if isinstance(success, bool):
        return success
    score = safe_float(record.get("score"))
    if score is None:
        return None
    return score >= 1.0 - PASS_SCORE_EPSILON


def succeeded_task_count(summary: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return (succeeded, evaluated) per-task outcomes, or None when unavailable."""
    outcomes = [task_succeeded(record) for record in per_task_records(summary)]
    known = [outcome for outcome in outcomes if outcome is not None]
    if not known:
        return None
    return sum(1 for outcome in known if outcome), len(known)


def command_arg_after(command: Sequence[Any], flag: str) -> str | None:
    for index, item in enumerate(command):
        if item == flag and index + 1 < len(command):
            return str(command[index + 1])
        if isinstance(item, str) and item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def command_config_value(command: Sequence[Any], key: str) -> str | None:
    for index, item in enumerate(command):
        value: str | None = None
        if item == "--config" and index + 1 < len(command):
            value = str(command[index + 1])
        elif isinstance(item, str) and item.startswith("--config="):
            value = item.split("=", 1)[1]
        if value and value.startswith(f"{key}="):
            return value.split("=", 1)[1]
    return None


def benchmark_from_command(command: Sequence[Any]) -> str | None:
    for index, item in enumerate(command):
        if item == "run" and index + 1 < len(command):
            return str(command[index + 1])
    return None


def base_command(data: Mapping[str, Any]) -> list[Any]:
    command = data.get("base_command")
    return command if isinstance(command, list) else []


def run_command(run: Mapping[str, Any], data: Mapping[str, Any]) -> list[Any]:
    command = run.get("command")
    if isinstance(command, list):
        return command
    return base_command(data)


def file_label(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"^ejepa-wm-harnesses-", "", stem)
    return stem


def latency_per_task(summary: Mapping[str, Any], comparison: Mapping[str, Any]) -> float | None:
    explicit = first_float(
        summary,
        [
            ("agentic_task_metrics", "avg_task_execution_time_seconds"),
            ("benchmark_metrics", "timing_avg_seconds_per_task"),
            ("benchmark_metrics", "avg_seconds_per_task"),
            ("timing", "avg_seconds_per_task"),
        ],
    )
    if explicit is not None:
        return explicit
    explicit = first_float(comparison, [("avg_task_execution_time_seconds",)])
    if explicit is not None:
        return explicit

    duration = first_float(
        summary,
        [
            ("duration_seconds",),
            ("benchmark_metrics", "duration_seconds"),
            ("benchmark_metrics", "timing_total_seconds"),
            ("timing", "total_seconds"),
        ],
    )
    total_tasks = first_float(
        summary,
        [
            ("total_tasks",),
            ("benchmark_metrics", "total_tasks"),
            ("benchmark_metrics", "summary_total_tasks"),
            ("summary", "total_tasks"),
        ],
    )
    if duration is not None and total_tasks and total_tasks > 0:
        return duration / total_tasks
    return None


def resolve_success_rate(
    summary: Mapping[str, Any],
    comparison: Mapping[str, Any],
    benchmark: Any,
    counts: tuple[int, int] | None,
    total_tasks: float | None,
) -> tuple[float, int | None, str] | None:
    """(success_rate_pct, succeeded_tasks, metric_name), or None when unavailable."""
    succeeded = counts[0] if counts is not None else None
    task_count = total_tasks if total_tasks and total_tasks > 0 else None
    if task_count is None and counts is not None and counts[1] > 0:
        task_count = float(counts[1])

    override = success_metric_override(benchmark)
    if override is not None:
        rate = first_float(summary, override["rate_paths"])
        percent = (
            first_float(summary, override["percent_paths"])
            if override.get("percent_paths")
            else None
        )
        if rate is None and percent is None:
            return None
        pct = rate_to_percent(rate) if rate is not None else percent
        passed = first_float(summary, override["count_paths"]) if override["count_paths"] else None
        if passed is None and task_count is not None and pct is not None:
            # crmarenapro publishes the upstream accuracy but no pass count.
            passed = round(pct / 100.0 * task_count)
        return pct, (int(passed) if passed is not None else None), override["metric"]

    if is_rubric_scored_benchmark(benchmark):
        # score_rate averages rubric pass rates, so recount succeeded tasks.
        if succeeded is None or task_count is None:
            return None
        return succeeded / task_count * 100.0, succeeded, "task_pass_rate"

    score_rate = first_float(summary, [("score_rate",), ("benchmark_metrics", "score_rate")])
    if score_rate is None:
        score_rate = safe_float(comparison.get("score_rate"))
    if score_rate is None:
        return None
    return rate_to_percent(score_rate), succeeded, "score_rate"


def run_row(
    path: Path,
    data: Mapping[str, Any],
    run: Mapping[str, Any],
    comparison: Mapping[str, Any],
    excluded: Sequence[str] = (),
) -> dict[str, Any] | None:
    summary = run.get("result_summary")
    summary = summary if isinstance(summary, Mapping) else {}
    command = run_command(run, data)

    latency = latency_per_task(summary, comparison)
    if latency is None:
        return None

    total_tasks = first_float(
        summary,
        [
            ("total_tasks",),
            ("benchmark_metrics", "total_tasks"),
            ("benchmark_metrics", "summary_total_tasks"),
            ("summary", "total_tasks"),
        ],
    )
    benchmark = summary.get("benchmark_name") or benchmark_from_command(command)
    counts = succeeded_task_count(summary)

    resolved = resolve_success_rate(summary, comparison, benchmark, counts, total_tasks)
    if resolved is None:
        return None
    success_rate_pct, succeeded, success_metric = resolved
    # The benchmark's own score_rate is reported alongside whichever metric
    # success_rate_pct uses, so e.g. crmarenapro shows both the upstream
    # original accuracy and the repo pass rate every other benchmark reports.
    native_score = first_float(summary, [("score_rate",), ("benchmark_metrics", "score_rate")])
    if native_score is None:
        native_score = safe_float(comparison.get("score_rate"))

    duration = first_float(
        summary, [("duration_seconds",), ("benchmark_metrics", "duration_seconds")]
    )
    if duration is None:
        duration = safe_float(comparison.get("duration_seconds"))

    row: dict[str, Any] = {
        "file": file_label(path),
        "created_at_utc": data.get("created_at_utc"),
        "benchmark": benchmark,
        "executor": summary.get("executor_name") or command_arg_after(command, "--executor"),
        "target": summary.get("target") or command_config_value(command, "target"),
        "harness": run.get("harness") or comparison.get("harness"),
        "status": summary.get("status") or comparison.get("status"),
        "returncode": run.get("returncode", comparison.get("returncode")),
        "total_tasks": int(total_tasks) if total_tasks is not None else None,
        "succeeded_tasks": succeeded,
        "success_metric": success_metric,
        "success_rate_pct": success_rate_pct,
        "score_rate_pct": rate_to_percent(native_score) if native_score is not None else None,
        "latency_per_task_seconds": latency,
        "duration_seconds": duration,
        "wrapper_elapsed_seconds": safe_float(
            run.get("wrapper_elapsed_seconds", comparison.get("wrapper_elapsed_seconds"))
        ),
        "result_dir": run.get("result_dir") or comparison.get("result_dir"),
    }
    filtered = filtered_rates(summary, excluded)
    if filtered is not None:
        row.update(filtered)
    return row


def summarize_file(path: Path, excluded: Sequence[str] = ()) -> tuple[list[dict[str, Any]], int]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a JSON object")

    runs = data.get("runs")
    runs = runs if isinstance(runs, list) else []
    comparisons = data.get("comparison")
    comparisons = comparisons if isinstance(comparisons, list) else []

    rows = []
    skipped = 0
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            skipped += 1
            continue
        comparison = comparisons[index] if index < len(comparisons) else {}
        comparison = comparison if isinstance(comparison, Mapping) else {}
        row = run_row(path, data, run, comparison, excluded)
        if row is None:
            skipped += 1
        else:
            rows.append(row)
    return rows, skipped


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

    rows: list[dict[str, Any]] = []
    skipped_files = 0
    skipped_rows = 0
    for path in paths:
        file_rows, file_skipped = summarize_file(path, args.exclude_domains)
        skipped_rows += file_skipped
        if file_rows:
            rows.extend(file_rows)
        else:
            skipped_files += 1

    if not rows:
        print("No usable WM harness summaries found.", file=sys.stderr)
        return 1
    if args.show_skipped:
        print(
            f"Skipped {skipped_files} files with no usable rows and {skipped_rows} rows.",
            file=sys.stderr,
        )

    if args.format == "csv":
        print(render_csv(rows, args.precision))
    elif args.format == "json":
        print(render_json(rows))
    else:
        print(render_table(rows, args.precision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
