#!/usr/bin/env python3
"""Beam-search latency per world model, HF-Transformers setting vs production serving.

Two panels share one log y-axis: the left panel is the measurement as submitted
(every world model run in-process under HF Transformers, unbatched); the right is
the same protocol with each model on its production stack -- the LLM world models
served by vLLM, Enterprise-JEPA under ``torch.compile`` + CUDA graphs
(``WM_JEPA_COMPILE=1``). Same cold-state protocol, 8 distinct candidate plans, x =
rollout horizon, y = wall-clock for one whole beam (roll out + score all candidates).

    uv run --with matplotlib python scripts/plot_wm_latency_hf_vs_production.py \
        --hf results/wm_latency/wm_latency_jepa.json results/wm_latency/wm_latency_llm_state.json \
             results/wm_latency/wm_latency_tool_output_h1.json \
        --production results/wm_latency/wm_latency_jepa_compiled_h1234.json \
             results/wm_latency/wm_latency_jepa_compiled_h8.json \
             results/wm_latency/wm_latency_llm_state_vllm.json results/wm_latency/wm_latency_llm_state_vllm_h8.json \
             results/wm_latency/wm_latency_tool_output_vllm_h1.json \
        --out results/figures/fig_wm_beam_latency_hf_vs_production.pdf
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
    INK_SECONDARY,
    REPO_ROOT,
    WORLD_MODEL_COLORS,
    WORLD_MODELS,
    apply_print_style,
    require_matplotlib,
    write_csv,
)

# Panel letters only make sense in the two-panel layout; --only renders standalone
# figures whose caption supplies the context instead.
PANELS = (
    ("hf", "(a) HF Transformers"),
    ("production", "(b) Production serving\n(LLM-WM: vLLM; JEPA: compiled)"),
)
STANDALONE_TITLES = {
    "hf": "In-process HF Transformers, unbatched",
    "production": "Production serving: LLM-WM on vLLM, JEPA compiled",
}
CSV_FIELDS = (
    "setting",
    "world_model",
    "backend",
    "horizon",
    "candidates",
    "seconds_per_call",
    "seconds_per_call_stdev",
    "repeats",
    "source",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hf", type=Path, nargs="+", required=True)
    parser.add_argument("--production", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results/figures/fig_wm_beam_latency_hf_vs_production.pdf",
    )
    parser.add_argument("--width", type=float, default=7.0, help="Inches; 7.0 = two columns.")
    parser.add_argument("--height", type=float, default=3.1)
    parser.add_argument(
        "--candidates", type=int, default=8, help="Beam width to plot (one width per figure)."
    )
    parser.add_argument(
        "--only",
        choices=("both", "production", "hf"),
        default="both",
        help=(
            "Render one panel as a standalone figure. The paper puts the production "
            "panel in the main text and the HF-Transformers panel in the appendix."
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["LLM-WM (tool output)"],
        metavar="WORLD_MODEL",
        help="World-model series to leave out (default: the tool-output LLM-WM, an h=1-only dot).",
    )
    return parser.parse_args(argv)


def call_seconds(row: dict[str, Any]) -> float | None:
    for field in ("seconds_per_call", "seconds_per_call_mean", "seconds_per_call_median"):
        value = row.get(field)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def load(
    paths: Sequence[Path],
    setting: str,
    exclude: Sequence[str] = (),
    candidates: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Rows grouped by world model (one series per model; the backend is implied by
    the panel). Duplicate horizons keep the most recent file's value."""
    series: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    csv_rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("measurements") or []:
            name = row.get("world_model")
            secs = call_seconds(row)
            if not name or name in exclude or secs is None or row.get("horizon") is None:
                continue
            # One beam width per figure: series are keyed by world model, so mixing
            # widths silently merges them into a single zig-zagging line.
            if candidates is not None and int(row.get("candidates") or 0) != candidates:
                continue
            series[name][int(row["horizon"])] = row
            csv_rows.append(
                {
                    "setting": setting,
                    "world_model": name,
                    "backend": row.get("backend"),
                    "horizon": row["horizon"],
                    "candidates": row.get("candidates"),
                    "seconds_per_call": secs,
                    "seconds_per_call_stdev": row.get("seconds_per_call_stdev"),
                    "repeats": row.get("repeats"),
                    "source": path.name,
                }
            )
    return {k: [v[h] for h in sorted(v)] for k, v in series.items()}, csv_rows


def draw_panel(axes: Any, series: dict[str, list[dict[str, Any]]], title: str) -> None:
    horizons: set[int] = set()
    for name in WORLD_MODELS:
        rows = series.get(name)
        if not rows:
            continue
        xs = [int(r["horizon"]) for r in rows]
        ys = [call_seconds(r) for r in rows]
        errs = [float(r.get("seconds_per_call_stdev") or 0.0) for r in rows]
        horizons.update(xs)
        axes.errorbar(
            xs,
            ys,
            yerr=errs if any(errs) else None,
            color=WORLD_MODEL_COLORS[name],
            marker="o",
            linestyle="-" if len(xs) > 1 else "none",
            markersize=4.5,
            linewidth=1.8,
            elinewidth=0.9,
            capsize=2,
            label=name,
            zorder=3,
        )
        # direct value label on the last point so the number is readable off the log axis
        axes.annotate(
            f"{ys[-1]:.3g} s",
            xy=(xs[-1], ys[-1]),
            xytext=(4, -2 if name == WORLD_MODELS[0] else 3),
            textcoords="offset points",
            fontsize=7,
            color=INK_SECONDARY,
            ha="left",
            va="center",
        )
    axes.set_title(title, fontsize=9)
    axes.set_xscale("log", base=2)
    ticks = [h for h in (1, 2, 4, 8) if h in horizons] or sorted(horizons)
    axes.set_xticks(ticks)
    axes.set_xticklabels([str(h) for h in ticks])
    axes.minorticks_off()
    axes.set_xlabel("Rollout horizon $h$")
    axes.grid(True, axis="y", which="major")
    axes.grid(False, axis="x")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require_matplotlib()
    apply_print_style()
    import matplotlib.pyplot as plt

    hf, rows_hf = load(args.hf, "hf", args.exclude, args.candidates)
    prod, rows_prod = load(args.production, "production", args.exclude, args.candidates)
    if args.only == "both":
        fig, axes = plt.subplots(1, 2, figsize=(args.width, args.height), sharey=True)
        draw_panel(axes[0], hf, PANELS[0][1])
        draw_panel(axes[1], prod, PANELS[1][1])
        axes = list(axes)
    else:
        series = prod if args.only == "production" else hf
        title = STANDALONE_TITLES[args.only]
        fig, ax = plt.subplots(figsize=(args.width / 2 + 0.6, args.height))
        draw_panel(ax, series, title)
        axes = [ax]
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Beam-search latency (s)")
    for ax in axes:
        ax.set_xlim(0.8, 12)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(len(labels), 1),
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(w_pad=1.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"))
    write_csv(args.out.with_suffix(".csv"), rows_hf + rows_prod, CSV_FIELDS)
    print(f"wrote {args.out} (+ .png, .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
