#!/usr/bin/env python3
"""Seeded subsample of a world-model trajectory corpus, written in BOTH formats.

Purpose: the three largest corpora (toucan_enterprise 8.5 GB, dolci 2.2 GB, toolmind 0.9 GB)
dominate extraction time, cache size (jepa_train_examples.jsonl reached 140 GB) and per-rank
RAM. A 25k-trajectory subset keeps their distribution in the mixture at a fraction of the cost.

NOT the same as the raw {name}_25k.jsonl files already in trajectories/ -- those are ShareGPT
{id, system, conversations} records (conversion INPUTS); this samples the converted
world-model {messages: [...]} trajectories the trainers actually read.

Both formats, same stem, so every consumer works without special-casing:
  {out}.jsonl  one trajectory per line -- finetuning_jepa.py streams it (and its
               resolve_streamable_path prefers a .jsonl sibling automatically)
  {out}.json   a single JSON list       -- finetuning.py's load_json path

Sampling is a seeded reservoir over the source .jsonl, so it is deterministic for a given
(source line order, seed, n) and never holds the full corpus in memory.

Usage:
  uv run python src/data_preparation/subsample_world_model_trajectories.py \
      trajectories/toucan_enterprise_world_model_trajectories.jsonl \
      --n 25000 --seed 42 \
      --out trajectories/toucan_enterprise_25k_world_model_trajectories
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help=".jsonl world-model trajectory file to sample from")
    p.add_argument("--n", type=int, default=25000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True,
                   help="output stem; writes {out}.jsonl and {out}.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.source.suffix != ".jsonl":
        raise SystemExit(f"{args.source} must be a .jsonl (stream-sampleable) file; convert first "
                         "with convert_trajectories_to_jsonl.py")
    rng = random.Random(args.seed)
    reservoir: list[str] = []
    total = 0
    with args.source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            if len(reservoir) < args.n:
                reservoir.append(line)
            else:
                j = rng.randrange(total)
                if j < args.n:
                    reservoir[j] = line
            if total % 100_000 == 0:
                print(f"  scanned {total} trajectories...", flush=True)
    if total <= args.n:
        print(f"[warn] source has only {total} trajectories (<= n={args.n}); keeping all.")

    jsonl_path = args.out.with_suffix(".jsonl")
    json_path = args.out.with_suffix(".json")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for line in reservoir:
            handle.write(line + "\n")
    # The .json variant is written by streaming the sampled lines into a list literal --
    # re-serializing 25k parsed objects would double peak memory for identical bytes.
    with json_path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        for index, line in enumerate(reservoir):
            handle.write(line + (",\n" if index + 1 < len(reservoir) else "\n"))
        handle.write("]\n")
    sample_gb = sum(len(l) for l in reservoir) / 2**30
    print(f"{args.source.name}: {total} trajectories -> {len(reservoir)} sampled "
          f"({sample_gb:.2f} GB)\n  -> {jsonl_path}\n  -> {json_path}")


if __name__ == "__main__":
    main()
