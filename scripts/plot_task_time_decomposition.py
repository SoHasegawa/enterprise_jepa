#!/usr/bin/env python3
"""Supplementary: where a task's wall clock goes, per configuration.

    task time = agent generation + planning generation + WM inference + environment/tools

Stacked bars, one per (benchmark, world model, harness) configuration. The split
separates the two distinct efficiency claims:

* **Enterprise-JEPA** -- low latency *per simulated transition* (the WM-inference
  segment stays small even when many transitions are simulated).
* **PaN / beam_critic** -- *fewer* planning invocations (the planning-generation
  segment shrinks because re-plans are event-triggered, not on a fixed cadence).

Provenance of each segment is recorded in the CSV:

* ``world_model_seconds`` / ``itp_i_world_model_seconds`` are timed by the
  harness itself (revision, reference, itp_i) -- used directly when present.
* ``open_loop_candidate_stats.generation_seconds`` times open-loop candidate
  generation -- used directly when present.
* Anything not timed is modelled as ``calls x measured latency per call``, taking
  the per-call latency from ``measure_wm_latency_vs_horizon.py`` (``--wm-latency-json``)
  or ``--wm-latency-seconds`` / ``--agent-latency-seconds``.
* Environment/tool time is the residual and is labelled as such.

    .venv/bin/python3 scripts/plot_task_time_decomposition.py \\
        --benchmark EnterpriseOps-Gym \\
        --wm-latency-json results/wm_latency/wm_latency_vs_horizon.json \\
        --agent-latency-seconds 3.2 \\
        --out results/figures/fig_task_time_decomposition.pdf
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wm_figure_common import (
    COST_COMPONENTS,
    DEFAULT_SUMMARY_GLOB,
    INK_MUTED,
    INK_PRIMARY,
    REPO_ROOT,
    SURFACE,
    apply_print_style,
    iter_runs,
    require_matplotlib,
    short_benchmark,
    write_csv,
)

CSV_FIELDS = (
    "benchmark",
    "world_model",
    "harness",
    "rollout_mode",
    "runs",
    "task_seconds",
    "agent_generation_seconds",
    "planning_generation_seconds",
    "wm_inference_seconds",
    "environment_seconds",
    "modelled_overflow_seconds",
    "wm_calls_per_task",
    "wm_seconds_per_call",
    "wm_time_source",
    "planning_llm_calls_per_task",
    "planning_time_source",
    "latent_transitions_per_task",
    "predicted_tokens_per_task",
    "success_rate_pct",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-glob", default=DEFAULT_SUMMARY_GLOB)
    parser.add_argument(
        "--benchmark",
        default="EnterpriseOps-Gym",
        help="Benchmark to decompose (one per figure keeps the bars comparable).",
    )
    parser.add_argument(
        "--wm-latency-json",
        type=Path,
        help="measure_wm_latency_vs_horizon.py output, for per-call WM latency.",
    )
    parser.add_argument(
        "--wm-latency-seconds",
        type=float,
        help="Fallback WM latency per call when neither telemetry nor --wm-latency-json applies.",
    )
    parser.add_argument(
        "--agent-latency-seconds",
        type=float,
        help=(
            "Mean latency of one policy LLM call. Without it, agent generation is "
            "not separated and stays inside the residual."
        ),
    )
    parser.add_argument("--min-tasks", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "out" / "figures" / "fig_task_time_decomposition.pdf",
    )
    parser.add_argument(
        "--width", type=float, default=0.0, help="Inches; 0 scales with the number of bars."
    )
    parser.add_argument("--height", type=float, default=4.6)
    return parser.parse_args(argv)


def wm_latency_table(path: Path | None) -> dict[str, float]:
    """Median seconds per simulated transition, per world model, from the sweep."""
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_model: dict[str, list[float]] = defaultdict(list)
    for row in payload.get("measurements") or []:
        name = row.get("world_model")
        value = row.get("seconds_per_transition")
        if name and isinstance(value, (int, float)):
            per_model[name].append(float(value))
    return {name: statistics.median(values) for name, values in per_model.items()}


def walk_wm_records(result_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield the world-model step records embedded in a run's trajectory files."""
    for path in sorted(result_dir.glob("trajectories/*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            stack: list[Any] = [payload]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    # One record per WM-assisted agent step. The shape differs per
                    # strategy: beam_plan carries ``event``/``critic``, itp_i carries
                    # ``itp_i_*`` fields and no event, revision/reference carry
                    # ``world_model_calls``.
                    if "strategy" in item and (
                        "event" in item
                        or "critic" in item
                        or "itp_i_k" in item
                        or "world_model_calls" in item
                    ):
                        yield item
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)


def measured_components(result_dir: Path, tasks: int) -> dict[str, Any]:
    """Per-task times and counts that the harness already timed for itself."""
    totals = defaultdict(float)
    counts = defaultdict(float)
    for record in walk_wm_records(result_dir):
        counts["wm_steps"] += 1
        for field, bucket in (
            ("world_model_seconds", "wm"),
            ("itp_i_world_model_seconds", "wm"),
        ):
            value = record.get(field)
            if isinstance(value, (int, float)):
                totals[bucket] += float(value)
        advice = record.get("itp_i_total_advice_seconds")
        wm_part = record.get("itp_i_world_model_seconds")
        if isinstance(advice, (int, float)):
            # Everything in the ITP-I advice phase that is not the world model is
            # candidate-action generation by the policy model.
            totals["planning"] += max(0.0, float(advice) - float(wm_part or 0.0))
        stats = record.get("open_loop_candidate_stats")
        if isinstance(stats, dict) and isinstance(stats.get("generation_seconds"), (int, float)):
            totals["planning"] += float(stats["generation_seconds"])
        for field, key in (
            ("beam_llm_calls", "planning_calls"),
            ("itp_i_world_model_calls", "wm_calls"),
            ("world_model_calls", "wm_calls"),
            ("itp_i_action_proposal_calls", "planning_calls"),
        ):
            value = record.get(field)
            if isinstance(value, (int, float)):
                counts[key] += float(value)
        critic = record.get("critic")
        if isinstance(critic, dict) and isinstance(critic.get("world_model_calls"), (int, float)):
            counts["wm_calls"] += float(critic["world_model_calls"])
        plan = record.get("imagined_plan")
        if isinstance(plan, list) and record.get("num_candidates"):
            counts["latent_transitions"] += float(record["num_candidates"]) * max(1, len(plan))
    scale = 1.0 / max(1, tasks)
    counts["latent_transitions"] += counts["wm_calls"]  # each critic/feedback call is one transition
    return {
        "wm_seconds": totals["wm"] * scale,
        "planning_seconds": totals["planning"] * scale,
        "wm_calls": counts["wm_calls"] * scale,
        "planning_calls": counts["planning_calls"] * scale,
        "steps": counts["wm_steps"] * scale,
        "latent_transitions": counts["latent_transitions"] * scale,
    }


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    latency_per_transition = wm_latency_table(args.wm_latency_json)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in iter_runs(args.input_glob):
        if str(run.get("benchmark")) != args.benchmark or run.get("world_model") is None:
            continue
        if not run.get("total_tasks") or int(run["total_tasks"]) < args.min_tasks:
            continue
        grouped[(run["world_model"], str(run["harness"]), run["rollout_mode"])].append(run)

    rows: list[dict[str, Any]] = []
    for (world_model, harness, mode), members in grouped.items():
        task_seconds = statistics.mean(float(m["latency_per_task_seconds"]) for m in members)
        measured = [
            measured_components(Path(m["result_dir"]), int(m["total_tasks"]))
            for m in members
            if m.get("result_dir") and Path(str(m["result_dir"])).is_dir()
        ]
        agg = {
            key: statistics.mean([m[key] for m in measured]) if measured else 0.0
            for key in ("wm_seconds", "planning_seconds", "wm_calls", "planning_calls", "steps", "latent_transitions")
        }
        wm_calls = agg["wm_calls"] or float(
            statistics.mean(
                [float(m.get("wm_calls_per_task") or 0.0) for m in members]
            )
        )
        # WM inference: prefer the harness's own timing, else calls x measured latency.
        if agg["wm_seconds"] > 0:
            wm_seconds, wm_source = agg["wm_seconds"], "harness telemetry"
        else:
            per_call = latency_per_transition.get(world_model) or args.wm_latency_seconds or 0.0
            wm_seconds, wm_source = wm_calls * per_call, (
                "modelled: calls x measured latency" if per_call else "unavailable"
            )
        if agg["planning_seconds"] > 0:
            planning_seconds, planning_source = agg["planning_seconds"], "harness telemetry"
        elif args.agent_latency_seconds:
            planning_seconds = agg["planning_calls"] * args.agent_latency_seconds
            planning_source = "modelled: planning calls x agent latency"
        else:
            planning_seconds, planning_source = 0.0, "folded into residual"
        agent_seconds = (agg["steps"] * args.agent_latency_seconds) if args.agent_latency_seconds else 0.0
        # The residual is environment/tool time. When the modelled segments overshoot
        # the measured task time (a wrong --agent-latency-seconds will do that), record
        # the overshoot instead of silently clamping it away.
        environment = task_seconds - wm_seconds - planning_seconds - agent_seconds
        overflow = max(0.0, -environment)
        environment = max(0.0, environment)
        rows.append(
            {
                "benchmark": args.benchmark,
                "world_model": world_model,
                "harness": harness,
                "rollout_mode": mode,
                "runs": len(members),
                "task_seconds": task_seconds,
                "agent_generation_seconds": agent_seconds,
                "planning_generation_seconds": planning_seconds,
                "wm_inference_seconds": wm_seconds,
                "environment_seconds": environment,
                "modelled_overflow_seconds": overflow,
                "wm_calls_per_task": wm_calls,
                "wm_seconds_per_call": (wm_seconds / wm_calls) if wm_calls else 0.0,
                "wm_time_source": wm_source,
                "planning_llm_calls_per_task": agg["planning_calls"],
                "planning_time_source": planning_source,
                "latent_transitions_per_task": agg["latent_transitions"],
                "predicted_tokens_per_task": None,
                "success_rate_pct": statistics.mean(
                    float(m["success_rate_pct"]) for m in members
                ),
            }
        )
    rows.sort(key=lambda row: row["task_seconds"])
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require_matplotlib()
    rows = build_rows(args)
    if not rows:
        print(f"No runs for benchmark {args.benchmark!r}.", file=sys.stderr)
        return 1

    apply_print_style()
    import matplotlib.pyplot as plt

    width = args.width if args.width else max(6.5, 0.82 * len(rows))
    figure, axes = plt.subplots(figsize=(width, args.height))
    short_wm = {
        "Enterprise-JEPA": "JEPA",
        "LLM-WM (state output)": "LLM-state",
        "LLM-WM (tool output)": "LLM-tool",
    }
    labels = [
        f"{row['harness']}\n{short_wm.get(row['world_model'], row['world_model'])}"
        + (" (open)" if row["rollout_mode"] == "open_loop" else "")
        for row in rows
    ]
    positions = range(len(rows))
    bottoms = [0.0] * len(rows)
    for field, label, color in COST_COMPONENTS:
        values = [float(row.get(field) or 0.0) for row in rows]
        if not any(values):
            continue
        axes.bar(
            list(positions),
            values,
            bottom=bottoms,
            color=color,
            label=label,
            width=0.62,
            linewidth=1.6,
            edgecolor=SURFACE,  # 2px surface gap between stacked segments
        )
        bottoms = [b + v for b, v in zip(bottoms, values, strict=False)]

    for index, row in enumerate(rows):
        axes.text(
            index,
            bottoms[index] * 1.02,
            f"{row['task_seconds']:.0f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK_PRIMARY,
            fontweight="bold",
        )
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, fontsize=7, rotation=40, ha="right", rotation_mode="anchor")
    axes.set_ylabel("Mean time per task (s)")
    axes.set_title(f"Task-time decomposition - {short_benchmark(args.benchmark)}", loc="left")
    axes.grid(axis="x", visible=False)
    axes.legend(loc="upper left", ncol=2)
    overflowing = [row for row in rows if row["modelled_overflow_seconds"] > 0.5]
    if overflowing:
        print(
            "WARNING: modelled segments exceed measured task time for "
            + ", ".join(f"{r['harness']}/{r['world_model']}" for r in overflowing)
            + " -- check --agent-latency-seconds (see modelled_overflow_seconds in the CSV).",
            file=sys.stderr,
        )
    sources = {row["wm_time_source"] for row in rows} | {row["planning_time_source"] for row in rows}
    axes.text(
        0.0,
        -0.42,
        "segment provenance: " + "; ".join(sorted(sources)),
        transform=axes.transAxes,
        fontsize=7,
        color=INK_MUTED,
        ha="left",
        va="top",
    )
    figure.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)
    csv_path = args.out.with_suffix(".csv")
    write_csv(csv_path, rows, CSV_FIELDS)
    print(f"Wrote {args.out}\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
