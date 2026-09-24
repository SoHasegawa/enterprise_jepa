#!/usr/bin/env python3
"""How beam width and depth change the world-model latency gap.

At the paper's operating point (8 candidates, horizon 3) the two world models differ
by ~6x, which understates the architectural difference. This figure sweeps both axes:
x = rollout horizon, y = wall-clock for one replan (log), one line per beam width,
blue for Enterprise-JEPA and orange for the vLLM-served state-output LLM world model.

The shapes differ because the costs differ in kind. The LLM world model decodes ~47
tokens per imagined transition, so its cost is linear in candidates x horizon and its
per-transition cost is flat (~20 ms) no matter how the work is arranged. Enterprise-JEPA
rolls out in latent space with candidates on the batch dimension, so widening the beam
is nearly free and each extra step is one predictor pass.

    uv run --with matplotlib python scripts/plot_wm_latency_scaling.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wm_figure_common import (
    INK_MUTED,
    INK_SECONDARY,
    REPO_ROOT,
    apply_print_style,
    require_matplotlib,
    write_csv,
)

JEPA = "Enterprise-JEPA"
LLM = "LLM-WM (state output)"
COLORS = {JEPA: "#2a78d6", LLM: "#eb6834"}
# lighter shade = smaller beam; the eye should read "family, then width"
ALPHA = {8: 0.45, 32: 0.7, 128: 1.0, 512: 1.0}
DASH = {512: (0, (4, 2))}
CSV_FIELDS = (
    "world_model",
    "candidates",
    "horizon",
    "seconds_per_call",
    "predicted_tokens_per_call",
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--glob", default=str(REPO_ROOT / "results/wm_latency/wm_latency_*.json"))
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results/figures/fig_wm_latency_scaling.pdf")
    p.add_argument(
        "--candidates",
        default="8,32,128,512",
        help="Beam widths to draw; earlier one-off sweeps (1, 2, 4) only clutter the figure.",
    )
    p.add_argument("--width", type=float, default=5.6)
    p.add_argument("--height", type=float, default=3.8)
    p.add_argument(
        "--exclude-compiled",
        action="store_true",
        default=True,
        help="Plot the eager JEPA numbers, so both models are shown in their default setup.",
    )
    return p.parse_args(argv)


def load(pattern: str, exclude_compiled: bool, keep: set[int]):
    series: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(pattern)):
        name = Path(path).name
        if exclude_compiled and "compiled" in name:
            continue
        if "tool_output" in name:  # two orders of magnitude away; separate discussion
            continue
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in payload.get("measurements") or []:
            model = row.get("world_model")
            secs = row.get("seconds_per_call_mean") or row.get("seconds_per_call")
            cand, hor = row.get("candidates"), row.get("horizon")
            if model not in (JEPA, LLM) or secs is None or not cand or not hor:
                continue
            if int(cand) not in keep:
                continue
            if row.get("backend") == "transformers" and model == LLM:
                continue  # superseded by the vLLM measurement
            series[(model, int(cand))][int(hor)] = float(secs)
            rows.append(
                {
                    "world_model": model,
                    "candidates": int(cand),
                    "horizon": int(hor),
                    "seconds_per_call": float(secs),
                    "predicted_tokens_per_call": row.get("predicted_tokens_per_call"),
                }
            )
    return series, rows


def main(argv=None) -> int:
    args = parse_args(argv)
    require_matplotlib()
    apply_print_style()
    import matplotlib.pyplot as plt

    keep = {int(c) for c in args.candidates.split(",")}
    series, rows = load(args.glob, args.exclude_compiled, keep)
    fig, axes = plt.subplots(figsize=(args.width, args.height))
    for (model, cand), points in sorted(series.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if len(points) < 2:
            continue
        xs = sorted(points)
        ys = [points[x] for x in xs]
        axes.plot(
            xs,
            ys,
            color=COLORS[model],
            alpha=ALPHA.get(cand, 1.0),
            linestyle=DASH.get(cand, "-"),
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            zorder=3,
        )
        axes.annotate(
            f"{cand}",
            xy=(xs[-1], ys[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            color=COLORS[model],
            va="center",
            fontweight="bold",
        )
    # the number the figure exists to make: the gap at the widest measured beam
    jepa10 = series.get((JEPA, 128), {}).get(10)
    llm10 = series.get((LLM, 128), {}).get(10)
    if jepa10 and llm10:
        axes.annotate(
            f"{llm10 / jepa10:.0f}x at 128 candidates,\nhorizon 10",
            xy=(10, (jepa10 * llm10) ** 0.5),
            xytext=(-6, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_SECONDARY,
            ha="right",
            va="center",
        )
    axes.set_yscale("log")
    axes.set_xticks([1, 2, 3, 4, 6, 8, 10])
    axes.set_xlabel("Rollout horizon $h$ (imagined transitions per candidate)")
    axes.set_ylabel("Beam-search latency per replan (s, log)")
    axes.grid(True, axis="y")
    axes.grid(False, axis="x")
    handles = [
        plt.Line2D([], [], color=COLORS[LLM], marker="o", markersize=3.5, label=LLM),
        plt.Line2D([], [], color=COLORS[JEPA], marker="o", markersize=3.5, label=JEPA),
    ]
    axes.legend(handles=handles, loc="upper left", fontsize=8)
    fig.text(
        0.5,
        -0.04,
        "Line labels are the number of candidate plans scored per replan; cold state, mean of 3 calls.",
        ha="center",
        fontsize=7,
        color=INK_MUTED,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"))
    write_csv(args.out.with_suffix(".csv"), rows, CSV_FIELDS)
    print(f"wrote {args.out} (+ .png, .csv) from {len(rows)} measurements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
