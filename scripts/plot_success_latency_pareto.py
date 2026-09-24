#!/usr/bin/env python3
"""Panel 2: success-latency Pareto plot, faceted by benchmark.

Built entirely from existing runs (``results/wm_harness_summaries/*.json``) --
no new measurement needed. One point per (world model, harness) configuration:

* x = mean end-to-end task execution time
* y = task success rate (the summarizer's pass-rate rules, so Workspace-Bench
  reports #succeeded/#tasks rather than its mean rubric rate)
* colour = world model, marker = harness (identity never rests on colour alone)

The dashed step line is the Pareto frontier *within each benchmark*. It is drawn
from the data, so a configuration appears on it only where it actually dominates:
where PaN/beam_plan trades success for time, the plot shows it off the frontier.

    .venv/bin/python3 scripts/plot_success_latency_pareto.py \\
        --out results/figures/fig_success_latency_pareto.pdf
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wm_figure_common import (
    DEFAULT_SUMMARY_GLOB,
    GRID,
    HARNESS_MARKERS,
    INK_MUTED,
    INK_SECONDARY,
    REPO_ROOT,
    WORLD_MODEL_COLORS,
    WORLD_MODELS,
    apply_print_style,
    iter_runs,
    pareto_front,
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
    "success_rate_pct",
    "success_rate_stdev",
    "mean_task_seconds",
    "total_tasks",
    "on_pareto_front",
    "summary_files",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-glob", default=DEFAULT_SUMMARY_GLOB)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "out" / "figures" / "fig_success_latency_pareto.pdf",
    )
    parser.add_argument(
        "--benchmarks",
        help="Comma-separated benchmark names to facet (default: all with >=2 configurations).",
    )
    parser.add_argument(
        "--min-configs",
        type=int,
        default=2,
        help="Skip benchmarks with fewer configurations than this.",
    )
    parser.add_argument(
        "--min-tasks",
        type=int,
        default=10,
        help=(
            "Drop runs that evaluated fewer tasks than this. The summaries contain "
            "1-task smoke runs whose success rate (0%% or 100%%) would otherwise be "
            "averaged into the real 80-100 task runs."
        ),
    )
    parser.add_argument("--columns", type=int, default=3, help="Facet columns.")
    parser.add_argument("--width", type=float, default=10.5)
    parser.add_argument("--height-per-row", type=float, default=3.2)
    return parser.parse_args(argv)


def aggregate(rows: Sequence[dict[str, Any]], *, min_tasks: int = 0) -> list[dict[str, Any]]:
    """Average repeated runs of the same (benchmark, world model, harness, mode)."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("world_model") is None:
            continue  # a world model outside the three-way comparison
        if row.get("success_rate_pct") is None or row.get("latency_per_task_seconds") is None:
            continue
        total_tasks = row.get("total_tasks")
        if min_tasks and (total_tasks is None or int(total_tasks) < min_tasks):
            continue  # smoke run, not an evaluation
        key = (
            str(row.get("benchmark")),
            row["world_model"],
            str(row.get("harness")),
            row.get("rollout_mode"),
        )
        groups[key].append(row)

    points: list[dict[str, Any]] = []
    for (benchmark, world_model, harness, mode), members in groups.items():
        successes = [float(m["success_rate_pct"]) for m in members]
        latencies = [float(m["latency_per_task_seconds"]) for m in members]
        points.append(
            {
                "benchmark": benchmark,
                "world_model": world_model,
                "harness": harness,
                "rollout_mode": mode,
                "runs": len(members),
                "success_rate_pct": statistics.mean(successes),
                "success_rate_stdev": (
                    statistics.stdev(successes) if len(successes) > 1 else 0.0
                ),
                "mean_task_seconds": statistics.mean(latencies),
                "total_tasks": members[0].get("total_tasks"),
                "summary_files": ";".join(sorted({Path(m["summary_path"]).name for m in members})),
            }
        )
    return points


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require_matplotlib()
    all_rows = list(iter_runs(args.input_glob))
    points = aggregate(all_rows, min_tasks=args.min_tasks)
    if not points:
        print(f"No usable runs matched {args.input_glob!r}.", file=sys.stderr)
        return 1
    kept = sum(int(p["runs"]) for p in points)
    if kept < len(all_rows):
        print(
            f"Using {kept} of {len(all_rows)} runs "
            f"(dropped runs with <{args.min_tasks} tasks or a world model outside the comparison).",
            file=sys.stderr,
        )

    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        by_benchmark[point["benchmark"]].append(point)
    wanted = (
        [name.strip() for name in args.benchmarks.split(",")]
        if args.benchmarks
        else [
            name
            for name, group in sorted(by_benchmark.items())
            if len(group) >= args.min_configs
        ]
    )
    facets = [name for name in wanted if by_benchmark.get(name)]
    if not facets:
        print("No benchmark has enough configurations to facet.", file=sys.stderr)
        return 1

    apply_print_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = max(1, min(args.columns, len(facets)))
    rows_count = (len(facets) + columns - 1) // columns
    figure, axes_grid = plt.subplots(
        rows_count,
        columns,
        figsize=(args.width, args.height_per_row * rows_count),
        squeeze=False,
    )

    for index, benchmark in enumerate(facets):
        axes = axes_grid[index // columns][index % columns]
        group = by_benchmark[benchmark]
        coords = [(p["mean_task_seconds"], p["success_rate_pct"]) for p in group]
        front = set(pareto_front(coords))
        for i, point in enumerate(group):
            point["on_pareto_front"] = i in front
            axes.scatter(
                point["mean_task_seconds"],
                point["success_rate_pct"],
                s=70 if i in front else 46,
                color=WORLD_MODEL_COLORS.get(point["world_model"], INK_SECONDARY),
                marker=HARNESS_MARKERS.get(point["harness"], "P"),
                edgecolors="#fcfcfb",  # 2px surface ring keeps overlapping marks legible
                linewidths=1.6,
                zorder=4 if i in front else 3,
            )
        # Pareto frontier as a step line (minimise time, maximise success).
        front_points = sorted(
            (coords[i] for i in front), key=lambda item: item[0]
        )
        if len(front_points) > 1:
            step_x: list[float] = []
            step_y: list[float] = []
            for (x, y), (next_x, _) in pairwise(front_points):
                step_x += [x, next_x]
                step_y += [y, y]
            step_x.append(front_points[-1][0])
            step_y.append(front_points[-1][1])
            axes.plot(step_x, step_y, color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=2)

        tasks = {p.get("total_tasks") for p in group if p.get("total_tasks")}
        subtitle = f"n={sorted(tasks)[0]} tasks" if len(tasks) == 1 else ""
        axes.set_title(f"{short_benchmark(benchmark)}   {subtitle}".strip(), loc="left")
        axes.set_xlabel("Mean task time (s)")
        axes.set_ylabel("Task success rate (%)")
        axes.set_xlim(left=0)
        axes.margins(x=0.12, y=0.18)
        axes.text(
            0.99,
            0.02,
            "better ↖",
            transform=axes.transAxes,
            fontsize=7,
            color=INK_MUTED,
            ha="right",
            va="bottom",
        )

    for blank in range(len(facets), rows_count * columns):
        axes_grid[blank // columns][blank % columns].axis("off")

    used_models = [m for m in WORLD_MODELS if any(p["world_model"] == m for p in points)]
    used_harnesses = [h for h in HARNESS_MARKERS if any(p["harness"] == h for p in points)]
    legend_handles = [
        Line2D(
            [], [], color=WORLD_MODEL_COLORS[name], marker="o", linestyle="none",
            markersize=7, label=name,
        )
        for name in used_models
    ] + [
        Line2D(
            [], [], color=INK_SECONDARY, marker=HARNESS_MARKERS[name], linestyle="none",
            markersize=7, label=name,
        )
        for name in used_harnesses
    ] + [
        Line2D([], [], color=INK_MUTED, linestyle="--", linewidth=1.0, label="Pareto frontier")
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(4, len(legend_handles)),
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle(
        "Success vs. end-to-end latency by world model and harness",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, facecolor=figure.get_facecolor(), edgecolor=GRID)
    csv_path = args.out.with_suffix(".csv")
    write_csv(csv_path, sorted(points, key=lambda p: (p["benchmark"], -p["success_rate_pct"])), CSV_FIELDS)
    print(f"Wrote {args.out}\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
