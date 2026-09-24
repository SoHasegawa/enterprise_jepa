#!/usr/bin/env python3
"""Latency per *policy step* rather than per task.

Per-task wall-clock rewards arms that give up early, so this reports the pooled
ratio

    seconds per step = sum(task execution seconds) / sum(policy steps)

where a policy step is one LLM call of the ReAct loop, i.e. ``tool_calls + 1``
(the final answer). Both harness summaries (``ejepa-wm-harnesses-*.json``) and
repeat summaries (``ejepa-bench-repeat-*.json``) are accepted; give globs or paths.

    uv run python scripts/summarize_per_step_latency.py \
        "results/wm_harness_summaries/ejepa-wm-harnesses-beam-ablation-eops-*.json" \
        results/bench_repeat_summaries/ejepa-bench-repeat-eops-baseline-*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_wm_harness_summaries import first_float, success_metric_override


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("patterns", nargs="+", help="Summary JSON paths or globs.")
    parser.add_argument("--benchmark", help="Keep only runs whose benchmark name contains this.")
    parser.add_argument("--csv", type=Path, help="Write the per-run table here.")
    return parser.parse_args(argv)


def success_of(summary: dict[str, Any]) -> tuple[float | None, str]:
    """(success %, metric name) using the same per-benchmark overrides as the harness
    summarizer (AutomationBench strict pass rate, crmarenapro original accuracy)."""
    override = success_metric_override(summary.get("benchmark_name"))
    if override is not None:
        rate = first_float(summary, override["rate_paths"])
        if rate is not None:
            return 100.0 * rate, override["metric"]
        if override.get("percent_paths"):
            pct = first_float(summary, override["percent_paths"])
            if pct is not None:
                return pct, override["metric"]
    rate = summary.get("score_rate")
    return (None if rate is None else 100.0 * float(rate)), "score_rate"


def label_of(path: Path) -> str:
    name = path.stem
    name = re.sub(r"^ejepa-(wm-harnesses|bench-repeat)-", "", name)
    return re.sub(r"-\d{8}T\d{6}Z$", "", name)


def step_stats(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    secs = [float(t.get("execution_time_seconds") or 0.0) for t in per_task]
    steps = [int(t.get("tool_calls") or 0) + 1 for t in per_task]
    per_task_ratio = [s / n for s, n in zip(secs, steps, strict=True) if n > 0]
    total_steps = sum(steps)
    if sum(secs) <= 0:  # e.g. AutomationBench summaries carry no execution_time_seconds
        return {
            "tasks": len(per_task),
            "s_per_task": None,
            "steps_per_task": statistics.mean(steps) if steps else None,
            "s_per_step_pooled": None,
            "s_per_step_task_mean": None,
            "s_per_step_median": None,
        }
    return {
        "tasks": len(per_task),
        "s_per_task": statistics.mean(secs) if secs else None,
        "steps_per_task": statistics.mean(steps) if steps else None,
        "s_per_step_pooled": (sum(secs) / total_steps) if total_steps else None,
        "s_per_step_task_mean": statistics.mean(per_task_ratio) if per_task_ratio else None,
        "s_per_step_median": statistics.median(per_task_ratio) if per_task_ratio else None,
    }


def rows_from_file(path: Path, benchmark_filter: str | None) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for run in data.get("runs") or []:
        summary = run.get("result_summary") or {}
        per_task = summary.get("per_task") or []
        bench = str(summary.get("benchmark_name") or "")
        if benchmark_filter and benchmark_filter.lower() not in bench.lower():
            continue
        if not per_task:
            continue
        cmd = " ".join(map(str, run.get("command") or data.get("command") or []))
        mp = re.search(r"max_parallel=(\d+)", cmd)
        success_pct, metric = success_of(summary)
        rows.append(
            {
                "label": label_of(path),
                "harness": run.get("harness") or "baseline",
                "benchmark": bench,
                "max_parallel": int(mp.group(1)) if mp else 1,
                "success_pct": success_pct,
                "metric": metric,
                **step_stats(per_task),
                "file": path.name,
            }
        )
    return rows


def fmt(v: Any, d: int = 1) -> str:
    return "-" if v is None else f"{v:.{d}f}"


def main(argv=None) -> int:
    args = parse_args(argv)
    paths: list[Path] = []
    for pattern in args.patterns:
        hits = sorted(glob.glob(pattern))
        paths.extend(Path(h) for h in hits) if hits else paths.append(Path(pattern))
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.is_file():
            rows.extend(rows_from_file(path, args.benchmark))
    if not rows:
        print("no runs with per_task data matched", file=sys.stderr)
        return 1
    print(
        f"{'label':<46s} {'harness':<14s} {'mp':>2s} {'succ%':>6s} {'tasks':>5s} "
        f"{'s/task':>7s} {'steps/task':>10s} {'s/step':>7s} {'s/step(med)':>11s}"
    )
    for r in rows:
        print(
            f"{r['label'][:46]:<46s} {str(r['harness'])[:14]:<14s} {r['max_parallel']:>2d} "
            f"{fmt(r['success_pct']):>6s} {r['tasks']:>5d} {fmt(r['s_per_task']):>7s} "
            f"{fmt(r['steps_per_task']):>10s} {fmt(r['s_per_step_pooled'], 2):>7s} "
            f"{fmt(r['s_per_step_median'], 2):>11s}"
        )
    print(
        "\ns/step = sum(execution seconds) / sum(tool_calls + 1) over all tasks of the run "
        "(pooled); s/step(med) = median over tasks of the per-task ratio."
    )
    if args.csv:
        import csv

        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"-> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
