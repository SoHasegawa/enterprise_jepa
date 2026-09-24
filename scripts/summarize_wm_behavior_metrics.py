#!/usr/bin/env python3
"""Pool the behavioural (non-score, non-latency) metrics into one CSV.

Reads both summary families -- ``results/wm_harness_summaries/*.json`` (one run per
harness) and ``results/bench_repeat_summaries/*.json`` (N repeats of one config) --
and aggregates their per-task metrics per (benchmark, world model, harness).

Success rate and latency are deliberately absent: they are covered by
``summarize_wm_harness_summaries.py`` and the Pareto figure. What is here is
what the agent and the world model *did*: tool calls, world-model calls, advice
injections, re-plans, critic checks/fires, action overrides.

Every column is a mean per task, pooled over all tasks of all matching runs (so a
100-task run does not count the same as a 1-task smoke run), except the ``*_total``
and ``runs`` columns and the pooled ``critic_fire_rate``.

    .venv/bin/python3 scripts/summarize_wm_behavior_metrics.py \\
        --out results/analysis/wm_behavior_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wm_figure_common import (
    REPO_ROOT,
    classify_world_model,
    rollout_mode,
    write_csv,
)

HARNESS_GLOB = "results/wm_harness_summaries/ejepa-wm-harnesses-*.json"
REPEAT_GLOB = "results/bench_repeat_summaries/ejepa-bench-repeat-*.json"

# per-task metric -> CSV column (mean per task)
METRICS = {
    "tool_calls": "avg_tool_calls",
    "failed_tool_calls": "avg_failed_tool_calls",
    "unnecessary_tool_calls": "avg_unnecessary_tool_calls",
    "wm_steps": "avg_wm_steps",
    "wm_world_model_call_count": "avg_wm_calls",
    "wm_advice_injected_count": "avg_advice_injections",
    "wm_terminal_advice_count": "avg_terminal_advice_injections",
    "wm_beam_planning_count": "avg_replans",
    "wm_beam_planning_success_count": "avg_replans_with_plan",
    "wm_beam_plan_steps": "avg_beam_plan_steps",
    "wm_imagined_plan_step_count": "avg_imagined_plan_steps",
    "wm_beam_llm_call_count": "avg_planning_llm_calls",
    "wm_critic_check_count": "avg_critic_checks",
    "wm_critic_fire_count": "avg_critic_fires",
    "wm_action_change_count": "avg_action_overrides",
    "wm_beam_refinement_round_count": "avg_refinement_rounds",
    "wm_beam_refinement_score_pass_count": "avg_refinement_score_passes",
    "wm_revision_count": "avg_revisions",
    "wm_reference_count": "avg_reference_injections",
    "wm_judge_call_count": "avg_judge_calls",
    "wm_model_call_count": "avg_wm_model_calls",
}

KEY_FIELDS = (
    "benchmark",
    "target",
    "world_model",
    "world_model_checkpoint",
    "harness",
    "rollout_mode",
    "config_model_name",
)
COUNT_FIELDS = ("runs", "tasks", "expected_tasks_per_run", "task_completeness", "critic_fire_rate")
CSV_FIELDS = (*KEY_FIELDS, *COUNT_FIELDS, *METRICS.values(), "source_files")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--harness-glob", default=HARNESS_GLOB)
    parser.add_argument("--repeat-glob", default=REPEAT_GLOB)
    parser.add_argument(
        "--min-tasks",
        type=int,
        default=10,
        help="Skip targets whose full task set is smaller than this (sample/smoke targets).",
    )
    parser.add_argument(
        "--min-task-fraction",
        type=float,
        default=0.9,
        help=(
            "Skip runs that evaluated less than this fraction of the target's full "
            "task set -- interrupted sessions (e.g. wow stopped at 14 of 50 tasks)."
        ),
    )
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="Keep interrupted/failed runs (reports them instead of excluding).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "out" / "analysis" / "wm_behavior_metrics.csv",
    )
    return parser.parse_args(argv)


def command_value(command: Sequence[Any], flag: str) -> str | None:
    items = [str(item) for item in command]
    for index, item in enumerate(items):
        if item == flag and index + 1 < len(items):
            return items[index + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def config_value(command: Sequence[Any], key: str) -> str | None:
    items = [str(item) for item in command]
    for index, item in enumerate(items):
        value = None
        if item == "--config" and index + 1 < len(items):
            value = items[index + 1]
        elif item.startswith("--config="):
            value = item.split("=", 1)[1]
        if value and value.startswith(f"{key}="):
            return value.split("=", 1)[1]
    return None


def harness_from_command(command: Sequence[Any]) -> str:
    """Name the harness for repeat summaries, which carry no ``harness`` field."""
    strategy = (command_value(command, "--wm-strategy") or "none").strip().lower()
    if strategy in ("", "none"):
        return "baseline"
    if strategy == "beam_plan":
        trigger = (command_value(command, "--wm-beam-plan-trigger") or "interval").lower()
        return "beam_critic" if trigger == "critic" else "beam_interval"
    return strategy


def world_model_checkpoint(command: Sequence[Any]) -> str | None:
    for flag in ("--wm-ewm-jepa-checkpoint", "--wm-ewm-llm-canonical-event-checkpoint"):
        value = command_value(command, flag)
        if value:
            return Path(value).name
    return command_value(command, "--wm-ewm-model")


def evaluated_tasks(tasks: Sequence[Mapping[str, Any]]) -> int:
    """Unique task ids that produced a score.

    ``per_task`` can hold more rows than the benchmark has tasks: detail rows and
    trajectory-derived rows are merged by id, and a benchmark whose trajectory
    filenames differ from its detail task ids yields one row of each kind (DevOps-Gym
    reports 200 rows for 100 tasks). Counting scored unique ids gives the real
    denominator for completeness.
    """
    scored = {
        str(task.get("task_id")) for task in tasks if isinstance(task.get("score"), (int, float))
    }
    return len(scored) or len({str(task.get("task_id")) for task in tasks})


def completion_problem(run: Mapping[str, Any], summary: Mapping[str, Any]) -> str | None:
    """Why this run should not be treated as a completed session, if so."""
    if summary.get("fatal_error"):
        return "fatal_error"
    status = summary.get("status")
    if status is not None and str(status).lower() != "completed":
        return f"status={status}"
    returncode = run.get("returncode")
    if returncode not in (None, 0):
        return f"returncode={returncode}"
    return None


def iter_runs(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    """Yield ``(key fields, per_task list)`` for every run in both summary families."""
    for glob_pattern, is_harness in ((args.harness_glob, True), (args.repeat_glob, False)):
        for path in sorted(REPO_ROOT.glob(glob_pattern)):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, Mapping):
                continue
            base_command = data.get("base_command") or data.get("effective_command") or []
            for run in data.get("runs") or []:
                if not isinstance(run, Mapping):
                    continue
                summary = run.get("result_summary")
                summary = summary if isinstance(summary, Mapping) else {}
                per_task = summary.get("per_task")
                if not isinstance(per_task, list) or not per_task:
                    continue
                command = run.get("command") if isinstance(run.get("command"), list) else None
                command = command or base_command
                tasks = [row for row in per_task if isinstance(row, Mapping)]
                evaluated = evaluated_tasks(tasks)
                expected = summary.get("total_tasks") or (
                    (summary.get("benchmark_metrics") or {}).get("total_tasks")
                    if isinstance(summary.get("benchmark_metrics"), Mapping)
                    else None
                )
                problem = completion_problem(run, summary)
                # Summaries written before the trajectory metric extraction carry only
                # score/verifier fields. Reporting them as a row of blanks would look
                # like "the harness did nothing"; skip and count them instead.
                if not any(
                    isinstance(task.get(metric), (int, float))
                    for task in tasks
                    for metric in METRICS
                ):
                    yield {"skipped_no_metrics": path.name}
                    continue
                yield {
                    "benchmark": summary.get("benchmark_name") or "unknown",
                    "target": config_value(command, "target") or summary.get("target") or "",
                    "evaluated_tasks": evaluated,
                    "expected_tasks": int(expected) if expected else None,
                    "completion_problem": problem,
                    "world_model": classify_world_model(command) or "none / baseline",
                    "world_model_checkpoint": world_model_checkpoint(command) or "",
                    "harness": (run.get("harness") if is_harness else None)
                    or harness_from_command(command),
                    "rollout_mode": rollout_mode(command),
                    "config_model_name": config_value(command, "model_name") or "",
                    "per_task": tasks,
                    "source_file": path.name,
                }


def filter_complete(
    runs: Sequence[Mapping[str, Any]], args: argparse.Namespace
) -> tuple[list[Mapping[str, Any]], dict[str, list[str]]]:
    """Keep only runs that evaluated (essentially) the whole target task set.

    The full task count comes from the benchmark itself (``total_tasks``); when a
    summary omits it, the largest evaluated count seen for that (benchmark, target)
    stands in. Targets whose full set is smaller than ``--min-tasks`` are sample
    targets and are dropped wholesale.
    """
    observed: dict[tuple[str, str], int] = {}
    for run in runs:
        key = (str(run["benchmark"]), str(run["target"]))
        observed[key] = max(observed.get(key, 0), int(run["evaluated_tasks"]))

    kept: list[Mapping[str, Any]] = []
    excluded: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        key = (str(run["benchmark"]), str(run["target"]))
        expected = run["expected_tasks"] or observed[key]
        label = f"{run['benchmark']}/{run['target']} {run['evaluated_tasks']}/{expected} ({run['source_file']})"
        if run["completion_problem"] and not args.keep_incomplete:
            excluded[run["completion_problem"]].append(label)
            continue
        if expected < args.min_tasks:
            excluded[f"sample target (<{args.min_tasks} tasks)"].append(label)
            continue
        if run["evaluated_tasks"] < args.min_task_fraction * expected:
            excluded[f"partial session (<{args.min_task_fraction:.0%} of tasks)"].append(label)
            continue
        kept.append({**run, "expected_tasks": expected})
    return kept, excluded


def aggregate(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[tuple(str(run[field]) for field in KEY_FIELDS)].append(run)

    rows: list[dict[str, Any]] = []
    for key, members in groups.items():
        tasks = [task for member in members for task in member["per_task"]]
        row: dict[str, Any] = dict(zip(KEY_FIELDS, key, strict=True))
        row["runs"] = len(members)
        row["tasks"] = sum(int(member["evaluated_tasks"]) for member in members)
        expected = [int(member["expected_tasks"]) for member in members]
        row["expected_tasks_per_run"] = max(expected)
        row["task_completeness"] = round(row["tasks"] / sum(expected), 4) if expected else ""
        row["source_files"] = ";".join(sorted({member["source_file"] for member in members}))
        for metric, column in METRICS.items():
            values = [
                float(task[metric]) for task in tasks if isinstance(task.get(metric), (int, float))
            ]
            row[column] = round(statistics.mean(values), 4) if values else ""
        checks = sum(float(task.get("wm_critic_check_count") or 0.0) for task in tasks)
        fires = sum(float(task.get("wm_critic_fire_count") or 0.0) for task in tasks)
        row["critic_fire_rate"] = round(fires / checks, 4) if checks else ""
        rows.append(row)
    rows.sort(key=lambda item: (item["benchmark"], item["world_model"], item["harness"]))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    everything = list(iter_runs(args))
    runs = [run for run in everything if "per_task" in run]
    skipped = [run["skipped_no_metrics"] for run in everything if "skipped_no_metrics" in run]
    if not runs:
        print("No runs with per-task metrics found.", file=sys.stderr)
        return 1
    complete, excluded = filter_complete(runs, args)
    if not complete:
        print("Every run was excluded as incomplete.", file=sys.stderr)
        return 1
    rows = aggregate(complete)
    write_csv(args.out, rows, CSV_FIELDS)
    print(
        f"Wrote {args.out}\n"
        f"{len(rows)} configurations from {len(complete)} complete runs "
        f"({sum(row['tasks'] for row in rows)} tasks pooled)"
    )
    for reason, labels in sorted(excluded.items()):
        print(f"Excluded {len(labels)} run(s) -- {reason}:", file=sys.stderr)
        for label in sorted(labels)[:6]:
            print(f"    {label}", file=sys.stderr)
        if len(labels) > 6:
            print(f"    ... and {len(labels) - 6} more", file=sys.stderr)
    if skipped:
        print(
            f"Skipped {len(skipped)} run(s) whose summaries predate the trajectory "
            f"metric extraction (no tool-call or wm_* fields): "
            + ", ".join(sorted(set(skipped))[:4])
            + (" ..." if len(set(skipped)) > 4 else ""),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
