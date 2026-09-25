"""Shared palette, styling and data extraction for the world-model figures.

Used by ``plot_wm_latency_hf_vs_production.py`` (paper Figures 2 and 4).

Every figure script writes the CSV it plotted next to the image: two palette slots
sit below 3:1 contrast on the print surface, and the data-viz relief rule requires
visible labels or a table view wherever that is true.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_GLOB = "results/wm_harness_summaries/ejepa-wm-harnesses-*.json"

# Validated categorical slots (light/print surface) -- see the data-viz palette
# reference. Slots 1-3 clear the all-pairs CVD and normal-vision floors, which is
# what scatter (panel 2) needs; the 4th slot is only used in the stacked bar,
# whose adjacent pairs also pass.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e3e2dd"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

# World models (color) -- exactly the three the architectural claim contrasts.
WORLD_MODELS = ("Enterprise-JEPA", "LLM-WM (state output)", "LLM-WM (tool output)")
WORLD_MODEL_COLORS = dict(zip(WORLD_MODELS, SERIES[:3], strict=False))

# Harnesses (marker shape) -- secondary encoding, so identity never rests on color.
HARNESS_MARKERS = {
    "itp_i": "o",
    "beam_interval": "s",
    "beam_critic": "D",
    "revision": "^",
    "reference": "v",
}

COST_COMPONENTS = (
    ("wm_inference_seconds", "World-model inference", SERIES[0]),
    ("planning_generation_seconds", "Planning generation (LLM)", SERIES[1]),
    ("agent_generation_seconds", "Agent generation", SERIES[2]),
    ("environment_seconds", "Environment / tools (residual)", SERIES[3]),
)


def apply_print_style() -> None:
    """Paper-facing matplotlib defaults: recessive axes, thin marks, no chartjunk."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "font.size": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Publisher requirement: embed fonts as TrueType (42) rather than
            # matplotlib's default Type 3 (3), which many venues reject and which
            # breaks text search/copy in the accepted PDF.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 6,
        }
    )
    for spine in ("top", "right"):
        plt.rcParams[f"axes.spines.{spine}"] = False


def load_summarizer():
    """Import ``summarize_wm_harness_summaries`` so the pass-rate rules stay in one place."""
    path = REPO_ROOT / "scripts" / "summarize_wm_harness_summaries.py"
    spec = importlib.util.spec_from_file_location("_wm_summarizer", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_world_model(command: Sequence[Any]) -> str | None:
    """Map a ``ejepa bench run`` command to one of :data:`WORLD_MODELS`.

    Returns ``None`` for runs driven by a world model outside the three-way
    comparison (e.g. served zero-shot chat models), so callers can drop them
    rather than inventing a fourth colour.
    """
    text = " ".join(str(item) for item in command)
    if "--wm-ewm-jepa-checkpoint" in text or "--wm-ewm-backend jepa" in text:
        return WORLD_MODELS[0]
    if "llm_tool_output_judge" in text:
        return WORLD_MODELS[2]
    if "--wm-ewm-llm-canonical-event-checkpoint" in text or "llm_canonical" in text:
        return WORLD_MODELS[1]
    return None


def rollout_mode(command: Sequence[Any]) -> str:
    return "open_loop" if "open_loop" in " ".join(str(i) for i in command) else "closed_loop"


def _first(value: Mapping[str, Any], *paths: Sequence[str]) -> Any:
    for path in paths:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping):
                current = None
                break
            current = current.get(key)
        if current is not None:
            return current
    return None


def iter_runs(
    summary_glob: str = DEFAULT_SUMMARY_GLOB,
    *,
    root: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one record per usable harness run found in the WM harness summaries.

    ``success_rate_pct`` follows the summarizer's rules, so rubric-scored benchmarks
    (Workspace-Bench) report the task pass rate rather than the mean rubric rate.
    """
    summarizer = load_summarizer()
    base = root or REPO_ROOT
    for path in sorted(base.glob(summary_glob)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, Mapping):
            continue
        base_command = data.get("base_command") or []
        comparisons = data.get("comparison")
        comparisons = comparisons if isinstance(comparisons, list) else []
        for index, run in enumerate(data.get("runs") or []):
            if not isinstance(run, Mapping):
                continue
            comparison = comparisons[index] if index < len(comparisons) else {}
            comparison = comparison if isinstance(comparison, Mapping) else {}
            row = summarizer.run_row(path, data, run, comparison)
            if row is None:
                continue
            command = run.get("command") if isinstance(run.get("command"), list) else base_command
            summary = run.get("result_summary")
            summary = summary if isinstance(summary, Mapping) else {}
            agentic = summary.get("agentic_task_metrics")
            agentic = agentic if isinstance(agentic, Mapping) else {}
            row.update(
                {
                    "summary_path": str(path),
                    "world_model": classify_world_model(command),
                    "rollout_mode": rollout_mode(command),
                    "result_dir": run.get("result_dir"),
                    "wm_calls_per_task": agentic.get("avg_wm_world_model_call_count"),
                    "beam_plans_per_task": agentic.get("avg_wm_beam_planning_count"),
                    "critic_checks_per_task": agentic.get("avg_wm_critic_check_count"),
                    "critic_fire_rate": agentic.get("wm_critic_fire_rate"),
                    "tool_calls_per_task": agentic.get("avg_tool_calls"),
                    "duration_seconds": _first(summary, ("duration_seconds",)),
                }
            )
            yield row


def pareto_front(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the points that are Pareto-optimal for (minimise x, maximise y)."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    best_y = float("-inf")
    front: list[int] = []
    for index in order:
        _, y = points[index]
        if y > best_y:
            front.append(index)
            best_y = y
    return front


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def short_benchmark(name: Any) -> str:
    """Facet title for a benchmark.

    Names are kept intact: ``WorkBench`` and ``Workspace-Bench`` are different
    benchmarks, so trimming the suffix would silently merge them in the reader's
    head. Only the redundant version tail is dropped.
    """
    return re.sub(r"-2\.0$", " 2.0", str(name or "unknown"))


def require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "matplotlib is required for the figure scripts:\n"
            "  uv pip install --python .venv/bin/python matplotlib"
        ) from exc
