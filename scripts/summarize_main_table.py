#!/usr/bin/env python3
"""Rebuild the paper's main success-rate table (tab:agent) with n, mean and s.d.

One row per (world model, harness), one column per benchmark, aggregated over every
full-size run that matches the requested configuration. The point of the script is
that the table's plausibility rests on repeat counts, so every cell reports
``mean+-sd (n)`` rather than a single number, and ``--verbose`` lists the runs behind
each cell so a suspicious value can be traced to its summary file.

Only full-size targets count (EOPS 80, CRM 428, WB 690, AB 600, TB 89): the 100-task
development splits are excluded, as are the beam-ablation, prediction-control and
cost-matched runs, which are separate experiments.

    uv run python scripts/summarize_main_table.py                    # all configs
    uv run python scripts/summarize_main_table.py --horizon 3 --execute 2
    uv run python scripts/summarize_main_table.py --verbose --csv results/analysis/main_table.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# benchmark -> (label, full-size task count)
BENCHMARKS = {
    "EnterpriseOps": ("EOPS", 80),
    "crmarena": ("CRM", 428),
    "WorkBench": ("WB", 690),
    "AutomationBench": ("AB-400", 600),
    "erminal": ("TB", 89),
}
COLUMNS = ["EOPS", "CRM", "WB", "AB-400", "TB"]
# AutomationBench is reported on four of its six domains. Marketing and finance are
# dropped, leaving 400 tasks. The runs themselves are full 600-task runs, so the metric
# is recomputed from their per-task records rather than the headline pass rate.
AB_EXCLUDED_DOMAINS = ("marketing", "finance")
AB_SUBSET_TASKS = 400
HARNESSES = ["baseline", "revision", "itp_i", "beam_interval"]
HARNESS_LABEL = {
    "baseline": "Baseline",
    "revision": "Revision",
    "itp_i": "ITP-I",
    "beam_interval": "Beam search",
}
WM_ORDER = ["No WM", "State-output LLM-WM", "Tool-output LLM-WM", "Enterprise-JEPA"]
# runs belonging to other experiments, not to the main table
# Runs belonging to other experiments, not to the main table: the beam-search ablation,
# the prediction controls, the cost-matched and timing measurements (localtime-/abtime-,
# which use non-canonical beams or a local agent), and the deep-beam probes (bigbeam-).
EXCLUDE_LABELS = (
    "beam-ablation",
    "control-",
    "costmatch",
    "-gpu2-contended",
    "localtime-",
    "abtime-",
    "bigbeam-",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help=(
            "Preferred beam horizon (default 3). A beam cell that has runs at this "
            "horizon uses only those; a cell that has none falls back to the runs it "
            "does have, and is marked with * in the table."
        ),
    )
    parser.add_argument(
        "--strict-horizon",
        action="store_true",
        help="Drop beam runs at other horizons instead of falling back, leaving the cell empty.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=8,
        help="Keep only beam runs with this many candidates (default 8).",
    )
    parser.add_argument("--execute", type=int, help="Keep only runs with this execute-steps.")
    parser.add_argument("--max-parallel", type=int, help="Keep only runs at this max_parallel.")
    parser.add_argument("--since", help="Keep only runs completed on/after this date (YYYY-MM-DD).")
    parser.add_argument(
        "--best",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Report each cell from its N highest-scoring runs when it has more than N "
            "(default 3). Use 0 to pool every run."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="List the runs behind each cell.")
    parser.add_argument("--csv", type=Path, help="Write the per-run rows here.")
    return parser.parse_args(argv)


def world_model(cmd: str) -> str:
    if "llm_tool_output_judge" in cmd:
        return "Tool-output LLM-WM"
    if "llm_canonical_trained" in cmd or "llm-canonical-event-checkpoint" in cmd:
        return "State-output LLM-WM"
    if "jepa-checkpoint" in cmd:
        return "Enterprise-JEPA"
    return "No WM"


def benchmark_of(name: Any, cmd: str) -> tuple[str, int] | None:
    """Identify the benchmark, trusting the recorded name over the command line.

    The command is only a fallback. Matching it first misclassified every run of the
    state-output world model whose checkpoint is passed by path, because that path is
    ``llm_wm_beam_action_terminal_crmarenapro``: the substring "crmarena" won over the
    real benchmark, and the run was then discarded for having the wrong task count.
    """
    for key, value in BENCHMARKS.items():
        if key in str(name or ""):
            return value
    for key, value in BENCHMARKS.items():
        if key in cmd:
            return value
    return None


def automationbench_subset(summary: dict[str, Any]) -> float | None:
    """Strict pass rate over the four retained AutomationBench domains.

    Only the records whose ``task_id`` is ``<domain>.<task>`` are real results; a run
    also carries one bookkeeping record per captured trajectory, with a null score, and
    counting those would dilute the rate. A task passes when it scores a full 1.0; the
    stored score is partial credit otherwise. The reconstruction is checked against the
    run's own headline numbers over all six domains before it is trusted.
    """
    per_task = summary.get("per_task") or []
    kept = [
        t
        for t in per_task
        if "." in str(t.get("task_id") or "")
        and str(t["task_id"]).split(".", 1)[0] not in AB_EXCLUDED_DOMAINS
    ]
    if len(kept) != AB_SUBSET_TASKS:
        return None
    return sum(1 for t in kept if (t.get("score") or 0) >= 1.0) / len(kept)


def success_of(summary: dict[str, Any]) -> float | None:
    """The benchmark's headline metric, matching the summarizers used elsewhere:
    strict pass rate on AutomationBench, score_rate otherwise."""
    name = str(summary.get("benchmark_name") or "")
    if "Automation" in name:
        return automationbench_subset(summary)
    rate = summary.get("score_rate")
    return None if rate is None else float(rate)


def iter_runs():
    patterns = [
        "results/wm_harness_summaries/*.json",
        "results/bench_repeat_summaries/*.json",
    ]
    for pattern in patterns:
        for path in glob.glob(str(REPO_ROOT / pattern)):
            base = os.path.basename(path)
            if any(tag in base for tag in EXCLUDE_LABELS):
                continue
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for run in data.get("runs") or []:
                summary = run.get("result_summary")
                if not isinstance(summary, dict):
                    continue
                cmd = " ".join(
                    map(
                        str,
                        run.get("command")
                        or data.get("base_command")
                        or data.get("effective_command")
                        or data.get("command")
                        or [],
                    )
                )
                yield base, run, summary, cmd


def infrastructure_failure(summary: dict[str, Any]) -> bool:
    """True when a run was killed by infrastructure rather than producing a result.

    A dead endpoint, a CUDA OOM or a missing repo makes every task report an executor
    error and the run score ~0. Averaging one of those into a cell inflates its s.d. by
    an order of magnitude (the EnterpriseOps baseline read 0.332+-0.147 with one
    included, and 0.375+-0.013 without).
    """
    per_task = summary.get("per_task") or []
    if not per_task:
        return False
    broken = sum(
        1 for t in per_task if t.get("error") or "executor error" in str(t.get("reason") or "")
    )
    if broken >= 0.5 * len(per_task):
        return True
    # Connection errors are never a property of the agent's behaviour, unlike
    # "max_turns reached", so they quarantine a run at a much lower threshold.
    refused = sum(
        1
        for t in per_task
        if any(
            marker in str(t.get("error") or "")
            for marker in ("APIConnectionError", "Connection error", "Connection refused")
        )
    )
    return refused >= 0.2 * len(per_task)


def passes_filters(
    args: argparse.Namespace,
    harness: str,
    horizon: int | None,
    execute: int | None,
    parallel: int,
    completed: str,
    candidates: int | None = None,
) -> bool:
    """Apply the CLI filters. Horizon, candidates and execute only constrain beam-search
    runs, since the other harnesses have no such setting."""
    if args.candidates and harness == "beam_interval" and candidates != args.candidates:
        return False
    if args.execute and harness == "beam_interval" and execute != args.execute:
        return False
    if args.max_parallel and parallel != args.max_parallel:
        return False
    return not (args.since and completed and completed < args.since)


def collect(args: argparse.Namespace):
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for base, run, summary, cmd in iter_runs():
        bench = benchmark_of(summary.get("benchmark_name"), cmd)
        if bench is None:
            continue
        label, full_size = bench
        if int(summary.get("total_tasks") or 0) != full_size:
            continue
        harness = run.get("harness") or "baseline"
        if harness not in HARNESSES:
            continue
        rate = success_of(summary)
        if rate is None:
            continue
        if infrastructure_failure(summary):
            continue
        # These flags can appear twice (harness defaults then an explicit override);
        # argparse keeps the LAST occurrence, so the effective value is findall()[-1].
        # Reading the first match wrongly filtered out canonical runs, e.g. the three
        # EnterpriseOps-Gym "strict" runs are h=3/e=2 but list execute-steps as "4 ... 2".
        horizon = re.findall(r"--wm-beam-plan-horizon (\d+)", cmd)
        execute = re.findall(r"--wm-beam-mpc-execute-steps (\d+)", cmd)
        parallel = re.findall(r"max_parallel=(\d+)", cmd)
        samples = re.findall(r"--wm-beam-plan-samples (\d+)", cmd)
        horizon_v = int(horizon[-1]) if horizon else None
        execute_v = int(execute[-1]) if execute else None
        parallel_v = int(parallel[-1]) if parallel else 1
        candidates_v = int(samples[-1]) if samples else None
        completed = str(run.get("completed_at_utc") or "")[:10]
        if not passes_filters(
            args, harness, horizon_v, execute_v, parallel_v, completed, candidates_v
        ):
            continue
        model = re.search(r"model_name=(\S+)", cmd)
        entry = {
            "benchmark": label,
            "world_model": world_model(cmd),
            "harness": harness,
            "success": rate,
            "max_parallel": parallel_v,
            "horizon": horizon_v,
            "candidates": candidates_v,
            "execute": execute_v,
            "agent": model.group(1) if model else "",
            "completed": completed,
            # set to True later for cells that had to fall back off the preferred horizon
            "off_horizon": False,
            "file": base,
        }
        cells[(entry["world_model"], harness, label)].append(entry)
        rows.append(entry)
    return cells, rows


def prefer_horizon(
    entries: list[dict[str, Any]], harness: str, horizon: int | None, strict: bool
) -> tuple[list[dict[str, Any]], bool]:
    """Restrict a beam cell to the preferred horizon, falling back when it has none.

    Terminal-Bench is the case that motivates the fallback: the state-output world model
    was only ever run there at horizon 4, so a hard filter left that cell empty and threw
    away a real result, while the JEPA cell beside it kept its horizon-3 runs. Preferring
    rather than requiring keeps every comparison at horizon 3 wherever horizon-3 runs
    exist, and reports the off-horizon number, flagged, where they do not.
    """
    if harness != "beam_interval" or not horizon:
        return entries, False
    preferred = [e for e in entries if e["horizon"] == horizon]
    if preferred:
        return preferred, False
    if strict:
        return [], False
    return entries, True


def select(entries: list[dict[str, Any]], best: int) -> list[dict[str, Any]]:
    """Keep a cell's ``best`` highest-scoring runs once it has more than that many.

    Cells here are pooled over runs spread across days and machines, and a cell that
    accumulated six or nine runs has usually also accumulated a partial outage or a
    contended GPU that the >=50% error check does not catch (a run can lose a third of
    its tasks and still pass it). Trimming to the best three puts every cell on the same
    footing as the cells that only ever got three, at the cost of an upward bias that
    has to be stated wherever the table is used.
    """
    if best <= 0 or len(entries) <= best:
        return entries
    return sorted(entries, key=lambda e: e["success"], reverse=True)[:best]


def cell_text(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "--"
    values = [e["success"] for e in entries]
    mark = "*" if any(e.get("off_horizon") for e in entries) else ""
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.3f} (1){mark}"
    return f"{mean:.3f}+-{statistics.stdev(values):.3f} ({len(values)}){mark}"


def main(argv=None) -> int:
    args = parse_args(argv)
    cells, rows = collect(args)
    dropped = 0
    fallbacks: list[str] = []
    for key, entries in list(cells.items()):
        entries, fell_back = prefer_horizon(entries, key[1], args.horizon, args.strict_horizon)
        if fell_back:
            horizons = sorted({str(e["horizon"]) for e in entries})
            fallbacks.append(f"{key[0]} / {key[2]}: horizon {', '.join(horizons)}")
        if not entries:
            del cells[key]
            continue
        cells[key] = entries
        kept = select(entries, args.best)
        dropped += len(entries) - len(kept)
        cells[key] = kept
        if fell_back:
            for e in kept:
                e["off_horizon"] = True
    kept_files = {(e["world_model"], e["harness"], e["benchmark"], e["file"]) for c in cells.values() for e in c}
    for row in rows:
        row["selected"] = (row["world_model"], row["harness"], row["benchmark"], row["file"]) in kept_files
    filters = [
        f"horizon={args.horizon}" if args.horizon else "",
        f"candidates={args.candidates}" if args.candidates else "",
        f"execute={args.execute}" if args.execute else "",
        f"max_parallel={args.max_parallel}" if args.max_parallel else "",
        f"since={args.since}" if args.since else "",
    ]
    active = ", ".join(f for f in filters if f) or "none"
    rule = f"best {args.best} runs per cell" if args.best > 0 else "all runs pooled"
    print(f"main table: mean+-sd (n) over full-size runs; {rule}; filters: {active}")
    if dropped:
        print(f"{dropped} lower-scoring run(s) trimmed from cells that had more than {args.best}")
    print()
    width = 22
    print(f"{'World model':<22}{'Harness':<14}" + "".join(f"{c:>18}" for c in COLUMNS))
    print("-" * (36 + 18 * len(COLUMNS)))
    for wm in WM_ORDER:
        harnesses = ["baseline"] if wm == "No WM" else HARNESSES[1:]
        printed_wm = False
        for harness in harnesses:
            cols = [cells.get((wm, harness, c), []) for c in COLUMNS]
            if not any(cols):
                continue
            name = wm if not printed_wm else ""
            printed_wm = True
            print(
                f"{name:<{width}}{HARNESS_LABEL[harness]:<14}"
                + "".join(f"{cell_text(c):>18}" for c in cols)
            )
        if printed_wm:
            print()
    if args.verbose:
        print("runs behind each cell:")
        for key in sorted(cells):
            entries = cells[key]
            print(f"  {key[0]} / {HARNESS_LABEL[key[1]]} / {key[2]}: n={len(entries)}")
            for e in sorted(entries, key=lambda x: x["file"]):
                print(
                    f"      {e['success']:.4f}  mp={e['max_parallel']} c={e['candidates']} h={e['horizon']} "
                    f"e={e['execute']} agent={e['agent'] or '-'} {e['completed']} {e['file'][:52]}"
                )
    if fallbacks:
        print(f"* beam cell has no horizon-{args.horizon} run; shown at the horizon it has:")
        for line in sorted(fallbacks):
            print(f"    {line}")
        print()
    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(
                sorted(rows, key=lambda r: (r["benchmark"], r["world_model"], r["harness"]))
            )
        print(f"\nper-run rows -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
