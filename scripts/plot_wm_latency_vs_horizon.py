#!/usr/bin/env python3
"""Panel 1: world-model latency versus rollout horizon.

Reads the JSON written by ``measure_wm_latency_vs_horizon.py`` and draws one line
per world model: x = rollout horizon (simulated transitions per candidate),
y = wall-clock latency per candidate trajectory.

The architectural claim under test: Enterprise-JEPA pays a fixed number of latent
forward passes per transition, while LLM world models additionally decode
autoregressively, so the gap should widen with depth. The right-hand annotation
reports the slope (seconds added per extra simulated transition) for each model,
which is that claim as a single number.

    .venv/bin/python3 scripts/plot_wm_latency_vs_horizon.py \\
        --measurements results/wm_latency/wm_latency_vs_horizon.json \\
        --out results/figures/fig_wm_latency_vs_horizon.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wm_figure_common import (
    INK_MUTED,
    INK_SECONDARY,
    REPO_ROOT,
    WORLD_MODEL_COLORS,
    WORLD_MODELS,
    apply_print_style,
    require_matplotlib,
    write_csv,
)

BACKEND_STYLES = {"transformers": "-", "vllm": "--"}

# What the y axis shows. "per_call" is the whole beam: every candidate trajectory
# rolled out and scored in one world-model pass, which is what a planner actually
# waits for. The other two divide that by the beam width / by the simulated
# transitions, i.e. amortised views of the same measurement.
Y_METRICS = {
    "per_call": (
        "seconds_per_call",
        1.0,
        "Beam-search latency (s)",
    ),
    "per_trajectory": (
        "seconds_per_candidate_trajectory",
        None,
        "Latency per candidate trajectory (s)",
    ),
    "per_transition": (
        "seconds_per_transition",
        None,
        "Latency per simulated transition (s)",
    ),
}

CSV_FIELDS = (
    "world_model",
    "backend",
    "state_cache",
    "candidate_mode",
    "statistic",
    "horizon",
    "candidates",
    "transitions",
    "repeats",
    "seconds_per_candidate_trajectory",
    "seconds_per_call",
    "seconds_per_call_mean",
    "seconds_per_call_median",
    "seconds_per_call_stdev",
    "seconds_per_transition",
    "predicted_tokens_per_call",
    "predicted_tokens_per_transition",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--measurements",
        type=Path,
        nargs="+",
        default=[REPO_ROOT / "out" / "wm_latency" / "wm_latency_vs_horizon.json"],
        help=(
            "One or more measure_wm_latency_vs_horizon.py outputs. Several files are "
            "merged, so legs measured in separate runs (a slow world model on its own) "
            "can be combined into one figure."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "out" / "figures" / "fig_wm_latency_vs_horizon.pdf",
    )
    parser.add_argument(
        "--y-metric",
        choices=tuple(Y_METRICS),
        default="per_call",
        help=(
            "per_call (default): total time to roll out and score the whole beam. "
            "per_trajectory / per_transition: the same number amortised over the "
            "beam width or the simulated transitions."
        ),
    )
    parser.add_argument(
        "--log-y", action="store_true", help="Log-scale latency (use when the gap is >10x)."
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Print the fitted fixed + marginal cost model inside the axes.",
    )
    parser.add_argument(
        "--no-direct-labels",
        action="store_true",
        help=(
            "Drop the per-series labels next to the last point. They overflow the "
            "axes, and savefig's tight bbox then widens the canvas -- which shrinks "
            "the type once the figure is scaled to a column. Use for single-column "
            "output; the legend and the CSV keep identity available."
        ),
    )
    parser.add_argument(
        "--width", type=float, default=5.2, help="Figure width in inches (single column)."
    )
    parser.add_argument("--height", type=float, default=3.4)
    return parser.parse_args(argv)


def load_series(paths: Sequence[Path]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group measurements from one or more sweeps by (world model, serving backend)."""
    series: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("measurements") or []:
            name = row.get("world_model")
            if name:
                series[(name, str(row.get("backend") or "transformers"))].append(row)
    for rows in series.values():
        rows.sort(key=lambda item: item.get("horizon", 0))
    return series


def call_seconds(row: dict[str, Any]) -> float | None:
    """Per-call latency as reported by the sweep (mean by default; older JSON: median)."""
    for field in ("seconds_per_call", "seconds_per_call_mean", "seconds_per_call_median"):
        value = row.get(field)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def fit_cost_model(rows: Sequence[dict[str, Any]]) -> tuple[float, float] | None:
    """Least-squares ``(slope, intercept)`` of per-call latency against horizon.

    The slope is the marginal cost of one extra simulated transition per candidate;
    the intercept is the per-call fixed cost (context/state encoding for JEPA,
    prompt prefill for an LLM world model). Reported separately because they are
    two different claims.
    """
    points = [
        (float(r["horizon"]), float(call_seconds(r)))
        for r in rows
        if r.get("horizon") is not None and call_seconds(r) is not None
    ]
    if len(points) < 2:
        return None
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return slope, mean_y - slope * mean_x


def draw_series(
    axes: Any,
    ordered: Sequence[tuple[str, str]],
    series: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    field: str,
    divisor: Any,
    mixed_backends: bool,
    direct_labels: bool,
) -> None:
    """Draw one line (or a bare dot) per (world model, backend) series."""
    for key in ordered:
        name, backend = key
        rows = series[key]
        color = WORLD_MODEL_COLORS.get(name, INK_SECONDARY)
        label = f"{name} ({backend})" if mixed_backends else name
        xs = [row["horizon"] for row in rows]
        ys = [float(row.get(field) or (call_seconds(row) or 0.0) / divisor(row)) for row in rows]
        # Error bars are the stdev over --repeats calls: the plotted point is a
        # mean, so its spread belongs in the figure rather than only in the CSV.
        errors = [float(row.get("seconds_per_call_stdev") or 0.0) / divisor(row) for row in rows]
        axes.errorbar(
            xs,
            ys,
            yerr=errors if any(errors) else None,
            color=color,
            marker="o" if backend == "transformers" else "s",
            # One measured horizon plots as a bare dot: there is no trend to imply.
            linestyle=BACKEND_STYLES.get(backend, "-") if len(xs) > 1 else "none",
            markersize=5,
            linewidth=2.0,
            elinewidth=1.0,
            capsize=2.5,
            label=label,
            zorder=3,
        )
        # Direct label at the last point: two palette slots sit under 3:1 contrast
        # on the print surface, so identity never rests on colour alone.
        if not direct_labels:
            continue
        axes.annotate(
            label,
            xy=(xs[-1], ys[-1]),
            xytext=(4, 3),
            textcoords="offset points",
            color=color,
            fontsize=7.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def annotate_cost_model(
    axes: Any,
    ordered: Sequence[tuple[str, str]],
    series: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    mixed_backends: bool,
) -> None:
    """Print the fitted ``fixed + marginal`` cost model inside the axes (opt-in)."""
    lines: list[str] = []
    for key in ordered:
        fit = fit_cost_model(series[key])
        if fit is None:
            continue
        slope, intercept = fit
        name, backend = key
        tag = f"{name} ({backend})" if mixed_backends else name
        candidates = series[key][0].get("candidates") or 1
        lines.append(
            f"{tag}: {intercept * 1000:.0f} ms fixed "
            f"+ {slope / candidates * 1000:.2f} ms per simulated transition"
        )
    if not lines:
        return
    # Bottom-left: the top-right corner belongs to the direct labels, which the
    # relief rule requires (two palette slots are under 3:1 on this surface).
    axes.text(
        0.02,
        0.02,
        "\n".join(lines),
        transform=axes.transAxes,
        fontsize=7,
        color=INK_MUTED,
        va="bottom",
        ha="left",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require_matplotlib()
    missing = [path for path in args.measurements if not path.exists()]
    if missing:
        print(
            "No measurements at " + ", ".join(str(path) for path in missing) + "\n"
            "Run scripts/measure_wm_latency_vs_horizon.py first "
            "(no existing benchmark run sweeps the rollout horizon).",
            file=sys.stderr,
        )
        return 1
    series = load_series(args.measurements)
    if not series:
        print("The given files contain no measurements.", file=sys.stderr)
        return 1

    apply_print_style()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(args.width, args.height))
    # Colour stays with the world model; the serving backend is the line style, so
    # a transformers/vLLM pair of the same model reads as one entity, two backends.
    ordered = [key for name in WORLD_MODELS for key in series if key[0] == name]
    ordered += [key for key in series if key not in ordered]
    mixed_backends = len({backend for _, backend in ordered}) > 1

    field, fixed_divisor, y_label = Y_METRICS[args.y_metric]

    def divisor(row: dict[str, Any]) -> float:
        if fixed_divisor is not None:
            return fixed_divisor
        if args.y_metric == "per_trajectory":
            return float(row.get("candidates") or 1)
        return float(row.get("transitions") or 1)

    draw_series(
        axes,
        ordered,
        series,
        field=field,
        divisor=divisor,
        mixed_backends=mixed_backends,
        direct_labels=not args.no_direct_labels,
    )

    horizons = sorted({row["horizon"] for rows in series.values() for row in rows})
    axes.set_xticks(horizons)
    axes.set_xticklabels([str(h) for h in horizons])
    axes.set_xlabel("Rollout horizon")
    axes.set_ylabel(y_label)
    if args.log_y:
        axes.set_yscale("log")
    # Room on the right only when direct labels need it.
    right_pad = 1.06 if args.no_direct_labels else 1.45
    axes.set_xlim(min(horizons) - 0.4, max(horizons) * right_pad)
    axes.grid(axis="both", alpha=0.7)

    if args.annotate:
        annotate_cost_model(axes, ordered, series, mixed_backends=mixed_backends)

    # A single series needs no legend box -- the direct label names it. With more,
    # the legend goes below the axes: the plot area is occupied by the marks and
    # their direct labels, and an inset box would land on top of a series.
    if len(ordered) > 1:
        handles, labels = axes.get_legend_handles_labels()
        # One row of entries is wider than a paper column, and savefig's tight bbox
        # would widen the canvas to fit it -- defeating the requested figure width.
        columns = 1 if args.width < 4.5 else min(3, len(labels))
        rows = -(-len(labels) // columns)
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=columns,
            bbox_to_anchor=(0.5, -0.02),
        )
        figure.tight_layout(rect=(0, min(0.4, 0.06 * rows + 0.02), 1, 1))
    else:
        figure.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)
    csv_path = args.out.with_suffix(".csv")
    write_csv(
        csv_path,
        [row for rows in series.values() for row in rows],
        CSV_FIELDS,
    )
    print(f"Wrote {args.out}\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
