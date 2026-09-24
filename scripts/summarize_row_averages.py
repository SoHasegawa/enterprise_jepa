#!/usr/bin/env python3
"""Average each (world model, harness) row of the main table across benchmarks.

Averaging the raw scores is misleading: WorkBench sits near 0.80 and AutomationBench
near 0.18, so a plain mean measures which benchmarks are in the average, not how much
the world model helps, and its spread is dominated by benchmark difficulty rather than
by run-to-run noise.

This script therefore reports the **paired delta against the no-world-model baseline of
the same benchmark**, macro-averaged with equal weight per benchmark, and separates the
two kinds of uncertainty that get conflated:

* ``se``    sampling error of the average delta, propagated from the per-cell standard
            errors (sd / sqrt(n)) of the cell and of its baseline. This says how well
            the average is pinned down by the repeats we ran.
* ``sd_b``  the spread of the per-benchmark deltas around their mean. This says how
            consistent the effect is across benchmarks, and it is usually the larger and
            more interesting number. It is not an error bar on the mean.

Equal weighting is deliberate. Inverse-variance weighting is more efficient in principle
but with three runs per cell the variance estimates are themselves very noisy, and it
would silently let WorkBench (sd 0.017) outvote AutomationBench (sd 0.016 on a mean four
times smaller). Standardised effects (delta divided by a pooled sd) have the same
problem, worse: dividing by an sd estimated from three points is unstable.

``--common`` restricts every row to the benchmarks that all compared rows cover, so the
rows are averages over the same thing. Without it, a row covering two benchmarks is
compared with one covering five.

    uv run python scripts/summarize_row_averages.py
    uv run python scripts/summarize_row_averages.py --common --relative
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_main_table import (
    COLUMNS,
    HARNESS_LABEL,
    WM_ORDER,
    collect,
    prefer_horizon,
    select,
)
from summarize_main_table import (
    parse_args as table_args,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--common",
        action="store_true",
        help="Restrict every row to the benchmarks covered by all rows that have any data.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Also report each delta as a fraction of its benchmark's baseline.",
    )
    parser.add_argument("--best", type=int, default=3, help="Runs per cell (see main table).")
    parser.add_argument("--horizon", type=int, default=3, help="Preferred beam horizon.")
    parser.add_argument("--candidates", type=int, default=8, help="Required beam candidates.")
    return parser.parse_args(argv)


def cell_stats(entries):
    """(mean, standard error, n). A single run has no measurable error."""
    values = [e["success"] for e in entries]
    if not values:
        return None
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, None, 1
    return mean, statistics.stdev(values) / math.sqrt(len(values)), len(values)


def build_cells(args):
    base = table_args([])
    base.best, base.horizon, base.candidates = args.best, args.horizon, args.candidates
    base.strict_horizon = False
    cells, _ = collect(base)
    out = {}
    for key, entries in cells.items():
        entries, _ = prefer_horizon(entries, key[1], args.horizon, False)
        if not entries:
            continue
        out[key] = cell_stats(select(entries, args.best))
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    cells = build_cells(args)
    baselines = {b: cells.get(("No WM", "baseline", b)) for b in COLUMNS}

    rows = []
    for wm in WM_ORDER:
        if wm == "No WM":
            continue
        for harness in ("revision", "itp_i", "beam_interval"):
            covered = [
                b
                for b in COLUMNS
                if cells.get((wm, harness, b)) is not None and baselines.get(b) is not None
            ]
            if covered:
                rows.append((wm, harness, covered))

    if args.common and rows:
        common = set(rows[0][2])
        for _, _, cov in rows:
            common &= set(cov)
        rows = [(wm, h, [b for b in COLUMNS if b in common]) for wm, h, _ in rows]
        print(f"restricted to the benchmarks every row covers: {', '.join(sorted(common)) or 'none'}")

    print(
        "\npaired delta vs the no-world-model baseline of the same benchmark, "
        "macro-averaged with equal weight\n"
        "  se   = sampling error of the mean delta, from the repeats\n"
        "  sd_b = spread of the per-benchmark deltas (consistency across benchmarks)\n"
    )
    head = f"{'World model':<22}{'Harness':<14}{'mean delta':>12}{'se':>9}{'sd_b':>9}{'benchmarks':>12}"
    print(head)
    print("-" * len(head))
    for wm, harness, covered in rows:
        deltas, variances, parts = [], [], []
        for b in covered:
            cm, cse, cn = cells[(wm, harness, b)]
            bm, bse, bn = baselines[b]
            d = cm - bm
            deltas.append(d)
            # a single-run cell has no error of its own; fall back to the baseline's,
            # which is the only estimate of that benchmark's run-to-run spread we have
            ce = cse if cse is not None else (bse or 0.0)
            be = bse or 0.0
            variances.append(ce * ce + be * be)
            piece = f"{b} {d:+.3f}"
            if args.relative and bm:
                piece += f" ({d / bm:+.0%})"
            if cn == 1:
                piece += "[n=1]"
            parts.append(piece)
        k = len(deltas)
        se = math.sqrt(sum(variances)) / k
        sd_b = statistics.stdev(deltas) if k > 1 else float("nan")
        sd_text = "  --" if k < 2 else f"{sd_b:.3f}"
        print(
            f"{wm:<22}{HARNESS_LABEL[harness]:<14}{statistics.mean(deltas):>+12.3f}"
            f"{se:>9.3f}{sd_text:>9}{k:>12}"
        )
        print(f"{'':36}{'  '.join(parts)}")
    print(
        "\nRead sd_b, not se, when asking whether a world model helps in general: se only "
        "says how precisely we measured the average of these particular benchmarks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
