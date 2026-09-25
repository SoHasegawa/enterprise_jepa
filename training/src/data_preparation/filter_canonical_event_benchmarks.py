#!/usr/bin/env python3
"""Drop rows of selected benchmarks from a canonical_event_with_nudge JSONL.

Why this exists: the JEPA trunk presets that end in `_no_tb` / use CORE_NO_TB_* paths exclude
Terminal-Bench trajectories, but the canonical-event label files were annotated over
EnterpriseOps-Gym + Terminal-Bench + CRMArenaPro. Training classification heads (or the
causal-LM canonical-event target) on rows whose benchmark the trunk never saw during
pre-training mixes an in-distribution result with an out-of-distribution one, and the mix is
invisible in the aggregate metrics. Filtering the labels to the pre-training benchmarks makes
the two stages consistent.

Row-level only: every kept line is copied byte-for-byte from the source, so labels, value
targets and successor links are untouched. Successor recovery still works because it is keyed on
(trajectory_id, interaction_index) WITHIN the file -- but note that dropping a benchmark removes
whole trajectories, not steps from the middle of a kept trajectory, so no successor chain of a
kept row is broken.

    uv run python src/data_preparation/filter_canonical_event_benchmarks.py \
        --exclude Terminal-Bench-2.0 \
        trajectories/canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro_train_examples_cleaned_ensemble_value_scored.jsonl \
        trajectories/canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro_eval_examples_cleaned_ensemble_value_scored.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Substrings removed from the filename stem when a benchmark is dropped, so the output name
# describes its contents instead of inheriting a stem that names a benchmark it no longer has.
STEM_TOKENS: dict[str, tuple[str, ...]] = {
    "Terminal-Bench-2.0": ("terminalbench_2_0_", "_terminalbench_2_0", "terminalbench_2_0"),
    "EnterpriseOps-Gym": ("enterpriseops_gym_", "_enterpriseops_gym", "enterpriseops_gym"),
    "CRMArenaPro": ("crmarenapro_", "_crmarenapro", "crmarenapro"),
}


def default_output_path(source: Path, excluded: list[str]) -> Path:
    stem = source.name
    for benchmark in excluded:
        for token in STEM_TOKENS.get(benchmark, ()):
            if token in stem:
                stem = stem.replace(token, "", 1)
                break
        else:
            # No known token for this benchmark: make the exclusion explicit rather than silent.
            slug = benchmark.lower().replace("-", "").replace(".", "").replace(" ", "")
            stem = stem.replace(".jsonl", f"_no_{slug}.jsonl")
    return source.with_name(stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", type=Path, help="canonical_event_with_nudge JSONL files.")
    parser.add_argument("--exclude", action="append", default=None, required=True,
                        help="Benchmark value to drop (repeatable), matched against the row's "
                             "`benchmark` field, e.g. --exclude Terminal-Bench-2.0.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (only valid with a single source; default derives the "
                             "name from the source by removing the benchmark from its stem).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded = list(args.exclude)
    if args.output is not None and len(args.sources) != 1:
        raise SystemExit("--output takes a single source file")

    for source in args.sources:
        if not source.is_file():
            raise SystemExit(f"missing source: {source}")
        destination = args.output or default_output_path(source, excluded)
        if destination == source:
            raise SystemExit(f"refusing to overwrite the source in place: {source}")
        if destination.exists() and not args.overwrite:
            raise SystemExit(f"{destination} exists; pass --overwrite to replace it")

        kept_by_benchmark: Counter[str] = Counter()
        dropped_by_benchmark: Counter[str] = Counter()
        kept_trajectories: set[str] = set()
        dropped_trajectories: set[str] = set()
        malformed = 0
        with source.open(encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
            for line in reader:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                benchmark = str(row.get("benchmark") or "unknown")
                trajectory = str(row.get("trajectory_id") or "")
                if benchmark in excluded:
                    dropped_by_benchmark[benchmark] += 1
                    dropped_trajectories.add(trajectory)
                    continue
                kept_by_benchmark[benchmark] += 1
                kept_trajectories.add(trajectory)
                writer.write(line if line.endswith("\n") else line + "\n")

        kept = sum(kept_by_benchmark.values())
        dropped = sum(dropped_by_benchmark.values())
        print(f"{source.name}")
        print(f"  -> {destination.name}")
        print(f"     kept {kept} rows ({len(kept_trajectories)} trajectories), "
              f"dropped {dropped} ({len(dropped_trajectories)} trajectories)"
              + (f", {malformed} malformed lines skipped" if malformed else ""))
        for benchmark, count in kept_by_benchmark.most_common():
            print(f"       keep {benchmark:24s} {count:6d}")
        for benchmark, count in dropped_by_benchmark.most_common():
            print(f"       drop {benchmark:24s} {count:6d}")


if __name__ == "__main__":
    main()
