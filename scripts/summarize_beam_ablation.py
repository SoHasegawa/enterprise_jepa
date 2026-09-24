#!/usr/bin/env python3
"""Summarise the beam-search planning ablation produced by scripts/run_beam_ablation.sh.

Recovers (candidates, horizon, execute_steps, rollout mode, repeat) from each run's
label, reports success rate and latency per configuration, the one-factor-at-a-time
marginals against the centre point, the success/latency Pareto front, and a single
recommended configuration per benchmark under an explicit rule:

    take the configuration with the highest success rate; among all configurations
    within --tolerance percentage points of it, take the one with the lowest latency.

Success rates come from summarize_wm_harness_summaries.py, so they use each
benchmark's proper definition (task success on EnterpriseOps-Gym, strict pass rate
rather than partial credit on AutomationBench).
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_GLOB = "results/wm_harness_summaries/ejepa-wm-harnesses-beam-ablation-*.json"
LABEL_RE = re.compile(
    r"beam-ablation-(?P<bench>[a-z0-9]+)(?:-(?P<domain>[a-z]+))?-s(?P<s>\d+)-h(?P<h>\d+)-e(?P<e>\d+)-"
    r"(?P<loop>open|closed)(?:-r(?P<rep>\d+))?"
)
BENCH_NAMES = {"eops": "EnterpriseOps-Gym", "ab": "AutomationBench"}
# AutomationBench runs are one domain each; labels from before the domain segment
# was added were all the `operations` domain.
DEFAULT_DOMAIN = {"ab": "operations"}


def load_summarizer():
    path = Path(__file__).with_name("summarize_wm_harness_summaries.py")
    spec = importlib.util.spec_from_file_location("wm_harness_summarizer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "paths", nargs="*", type=Path, help="Summary JSONs; defaults to --input-glob."
    )
    parser.add_argument("--input-glob", default=DEFAULT_GLOB)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Success-rate tolerance (pp) within which a cheaper config is preferred.",
    )
    parser.add_argument("--csv", type=Path, help="Also write the per-run table to this CSV.")
    return parser.parse_args(argv)


def wm_activity(path: Path) -> dict[str, float | None]:
    """Per-task world-model activity, read straight from the summary's agentic metrics."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for run in data.get("runs") or []:
        summary = run.get("result_summary") or {}
        agentic = summary.get("agentic_task_metrics") or {}
        if agentic:
            # beam_plan does not bump the generic world-model counter; each beam
            # planning event is (at least) one scoring call, so fall back to it.
            wm_calls = agentic.get("avg_wm_world_model_call_count") or agentic.get(
                "avg_wm_beam_planning_count"
            )
            return {
                "wm_calls": wm_calls,
                "replans": agentic.get("avg_wm_beam_planning_count"),
                "tool_calls": agentic.get("avg_tool_calls"),
            }
    return {"wm_calls": None, "replans": None, "tool_calls": None}


def collect(paths: list[Path], summarizer) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in paths:
        match = LABEL_RE.search(path.name)
        if not match:
            continue
        rows, _skipped = summarizer.summarize_file(path)
        if not rows:
            print(f"[skip] {path.name}: no usable row (missing rate or latency)", file=sys.stderr)
            continue
        row = rows[0]
        activity = wm_activity(path)
        runs.append(
            {
                "benchmark": BENCH_NAMES.get(match["bench"], match["bench"]),
                "samples": int(match["s"]),
                "horizon": int(match["h"]),
                "execute": int(match["e"]),
                "loop": match["loop"],
                "rep": int(match["rep"] or 1),
                "domain": match["domain"] or DEFAULT_DOMAIN.get(match["bench"]),
                "tasks": row.get("total_tasks"),
                "success_pct": row.get("success_rate_pct"),
                "metric": row.get("success_metric"),
                "latency_s": row.get("latency_per_task_seconds"),
                "step_s": seconds_per_step(path),
                "wall_s": row.get("wrapper_elapsed_seconds"),
                **activity,
                "file": path.name,
            }
        )
    return runs


def seconds_per_step(path: Path) -> float | None:
    """Pooled seconds per policy step (tool_calls + 1 final answer) -- per-task
    latency rewards early termination, this does not."""
    data = json.loads(path.read_text(encoding="utf-8"))
    secs = steps = 0.0
    for run in data.get("runs") or []:
        for task in (run.get("result_summary") or {}).get("per_task") or []:
            secs += float(task.get("execution_time_seconds") or 0.0)
            steps += int(task.get("tool_calls") or 0) + 1
    return secs / steps if steps and secs > 0 else None  # AutomationBench records no per-task time


def config_key(run: dict[str, Any]) -> tuple[int, int, str]:
    return run["samples"], run["horizon"], run["loop"]


def fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}{suffix}"


def aggregate(runs: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """Mean over repeats per configuration."""
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[config_key(run)].append(run)
    out = {}
    for key, group in grouped.items():
        succ = [r["success_pct"] for r in group if r["success_pct"] is not None]
        lat = [r["latency_s"] for r in group if r["latency_s"] is not None]
        # Pool by task count so several 100-task domains combine into one rate
        # (equal-size repeats reduce to the plain mean).
        weights = [r.get("tasks") or 1 for r in group if r["success_pct"] is not None]
        pooled = (
            sum(v * w for v, w in zip(succ, weights, strict=True)) / sum(weights) if succ else None
        )
        out[key] = {
            "samples": key[0],
            "horizon": key[1],
            "loop": key[2],
            "n_runs": len(group),
            "domains": sorted({r["domain"] for r in group if r.get("domain")}),
            "success_pct": pooled,
            "success_sd": statistics.stdev(succ) if len(succ) > 1 else None,
            "latency_s": statistics.mean(lat) if lat else None,
            "step_s": statistics.mean(
                [r["step_s"] for r in group if r["step_s"] is not None] or [float("nan")]
            ),
            "wm_calls": statistics.mean(
                [r["wm_calls"] for r in group if r["wm_calls"] is not None] or [float("nan")]
            ),
            "replans": statistics.mean(
                [r["replans"] for r in group if r["replans"] is not None] or [float("nan")]
            ),
            "tasks": group[0]["tasks"],
            "metric": group[0]["metric"],
        }
    return out


def pareto(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-dominated set: higher success and lower latency."""
    front = []
    for c in configs:
        if c["success_pct"] is None or c["latency_s"] is None:
            continue
        dominated = any(
            o is not c
            and o["success_pct"] is not None
            and o["latency_s"] is not None
            and o["success_pct"] >= c["success_pct"]
            and o["latency_s"] <= c["latency_s"]
            and (o["success_pct"] > c["success_pct"] or o["latency_s"] < c["latency_s"])
            for o in configs
        )
        if not dominated:
            front.append(c)
    return sorted(front, key=lambda c: -c["success_pct"])


def recommend(configs: list[dict[str, Any]], tolerance: float) -> dict[str, Any] | None:
    scored = [c for c in configs if c["success_pct"] is not None and c["latency_s"] is not None]
    if not scored:
        return None
    best = max(c["success_pct"] for c in scored)
    candidates = [c for c in scored if c["success_pct"] >= best - tolerance]
    return min(candidates, key=lambda c: c["latency_s"])


def label(c: dict[str, Any]) -> str:
    return f"s={c['samples']:<2d} h={c['horizon']} {c['loop']:<6s}"


def find_centre(configs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The OFAT centre: the config sharing exactly two factors with the most others."""
    best: tuple[int, dict[str, Any]] | None = None
    for c in configs:
        share = sum(
            (o["samples"] == c["samples"])
            + (o["horizon"] == c["horizon"])
            + (o["loop"] == c["loop"])
            == 2
            for o in configs
        )
        if best is None or share > best[0]:
            best = (share, c)
    return best[1] if best and best[0] >= 2 else None


def print_ofat_marginals(configs: list[dict[str, Any]]) -> None:
    """Δ success / Δ latency of each single-factor change relative to the centre."""
    cen = find_centre(configs)
    if cen is None:
        return
    print(
        f"\n  one-factor-at-a-time deltas vs centre [{label(cen).strip()}]  "
        "(Δ success pp, Δ latency s/task)"
    )
    factors = ("samples", "horizon", "loop")
    for factor in factors:
        for c in configs:
            if c is cen or any(c[f] != cen[f] for f in factors if f != factor):
                continue
            if c["success_pct"] is None or cen["success_pct"] is None:
                continue
            ds = c["success_pct"] - cen["success_pct"]
            dl = (c["latency_s"] or 0) - (cen["latency_s"] or 0)
            print(
                f"    {factor:<8s} {cen[factor]!s:>11s} -> {c[factor]!s:<11s}  "
                f"{ds:+6.1f} pp   {dl:+7.1f} s"
            )


def report_benchmark(
    bench: str, runs: list[dict[str, Any]], tolerance: float
) -> dict[str, Any] | None:
    agg = aggregate(runs)
    configs = sorted(agg.values(), key=lambda c: (c["loop"], c["samples"], c["horizon"]))
    metric = configs[0]["metric"] if configs else "?"
    print(
        f"\n{'=' * 78}\n{bench}  (success metric: {metric}; tasks per run: {configs[0]['tasks'] if configs else '?'}; runs pooled over repeats and domains)\n{'=' * 78}"
    )
    print(
        f"  {'config':<20s} {'runs':>4s} {'success %':>10s} {'sd':>5s} {'s/task':>8s} {'s/step':>7s} {'WM calls/task':>14s} {'replans/task':>13s}"
    )
    for c in configs:
        print(
            f"  {label(c):<20s} {c['n_runs']:>4d} {fmt(c['success_pct']):>10s} "
            f"{fmt(c['success_sd']):>5s} {fmt(c['latency_s']):>8s} {fmt(c['step_s'], 2):>7s} {fmt(c['wm_calls']):>14s} {fmt(c['replans']):>13s}"
        )

    domains = sorted({r["domain"] for r in runs if r.get("domain")})
    if len(domains) > 1:
        # per-domain breakdown (success % per config, one column per domain)
        print("\n  per-domain success % (pooled column above weights domains by task count):")
        print(f"  {'config':<20s} " + " ".join(f"{d[:11]:>11s}" for d in domains))
        by_cfg: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in runs:
            if r["success_pct"] is not None:
                by_cfg[config_key(r)][r["domain"]].append(r["success_pct"])
        for c in configs:
            cells = by_cfg.get((c["samples"], c["horizon"], c["loop"]), {})
            print(
                f"  {label(c):<20s} "
                + " ".join(
                    f"{fmt(statistics.mean(cells[d])) if cells.get(d) else '-':>11s}"
                    for d in domains
                )
            )

    print_ofat_marginals(configs)

    front = pareto(configs)
    print("\n  Pareto front (success ↑, latency ↓):")
    for c in front:
        print(f"    {label(c)}  {fmt(c['success_pct'])}%  {fmt(c['latency_s'])} s/task")
    rec = recommend(configs, tolerance)
    if rec:
        print(
            f"\n  RECOMMENDED (max success, then min latency within {tolerance:g} pp): "
            f"{label(rec).strip()}  ->  {fmt(rec['success_pct'])}%  {fmt(rec['latency_s'])} s/task"
        )
    return rec


def main(argv=None) -> int:
    args = parse_args(argv)
    summarizer = load_summarizer()
    paths = args.paths or sorted(Path(p) for p in glob.glob(args.input_glob))
    if not paths:
        print(f"No files matched {args.input_glob!r}", file=sys.stderr)
        return 1
    runs = collect(paths, summarizer)
    if not runs:
        print("No ablation runs with a parsable label were found.", file=sys.stderr)
        return 1

    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_bench[run["benchmark"]].append(run)

    recs = {}
    for bench in sorted(by_bench):
        recs[bench] = report_benchmark(bench, by_bench[bench], args.tolerance)

    # --- cross-benchmark suggestion by rank sum --------------------------------
    if len(by_bench) > 1:
        rank_sum: dict[tuple, float] = defaultdict(float)
        seen: dict[tuple, int] = defaultdict(int)
        for bruns in by_bench.values():
            agg = [c for c in aggregate(bruns).values() if c["success_pct"] is not None]
            for rank, c in enumerate(
                sorted(agg, key=lambda c: (-c["success_pct"], c["latency_s"] or 0)), 1
            ):
                rank_sum[config_key(c)] += rank
                seen[config_key(c)] += 1
        common = [k for k in rank_sum if seen[k] == len(by_bench)]
        if common:
            best = min(common, key=lambda k: rank_sum[k])
            print(
                f"\n{'=' * 78}\nSingle configuration with the best rank sum across benchmarks: "
                f"s={best[0]} h={best[1]} {best[2]}  (rank sum {rank_sum[best]:.0f})"
            )

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(runs[0].keys()))
            writer.writeheader()
            writer.writerows(runs)
        print(f"\nper-run table written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
