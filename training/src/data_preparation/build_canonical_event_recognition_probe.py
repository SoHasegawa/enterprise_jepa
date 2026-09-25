#!/usr/bin/env python3
"""Materialize observation-carrying JSONLs for the canonical-event recognition probe.

Purpose: `--canonical-event-recognition-probe` (finetuning_jepa.py) measures the RECOGNITION
ceiling of the classification heads -- read the heads off the encoded ACTUAL observation
(z_observation) instead of the predicted latent (z_pred) -- to separate two very different
explanations for the accuracy plateau:

  * recognition >> prediction  ->  the frozen encoder CAN carry outcome information when it is
    present in the text; the prediction gap is aleatoric (hidden environment state), and
    prediction accuracy is near its true ceiling.
  * recognition ~= prediction  ->  the pooled-embedding representation itself is the
    bottleneck; attacking it (multi-token latents, unfreezing) has headroom.

The probe needs an ``observation`` field per row, which the labeled canonical JSONLs do not
carry directly. But they carry it INDIRECTLY: row t+1's ``input_history[-1]`` is row t's own
(action, observation, step) -- an alignment verified on the real data (the appended item's
``step`` equals row t's interaction_index + 1 for every consecutive pair, and the action
arguments match). This script copies that observation back onto row t and writes only the rows
that have one (a trajectory's terminal row has no successor, hence no recoverable observation).

KNOWN LIMITATION -- the labeler truncated history observations to 200 characters, so the
recovered observations are 200-char prefixes (tool name + leading output/error text). The
recognition number measured on these is therefore a LOWER BOUND on the true recognition
ceiling: if even truncated observations lift accuracy far above prediction mode, the
aleatoric-gap conclusion holds a fortiori; if they do not, the result is ambiguous and a
full-observation join against the source trajectory files is the upgrade path.

Usage:

    uv run python src/data_preparation/build_canonical_event_recognition_probe.py
    # then, on identical rows:
    #   recognition:  --train-canonical-event-heads-only --canonical-event-recognition-probe \
    #                 --canonical-event-train-jsonl <train_probe> --canonical-event-eval-jsonl <eval_probe>
    #   prediction :  same command WITHOUT --canonical-event-recognition-probe
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAJECTORIES_DIR = REPO_ROOT / "trajectories"
STEM = "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro"
DEFAULT_TRAIN = TRAJECTORIES_DIR / f"{STEM}_train_examples_cleaned_value_scored.jsonl"
DEFAULT_EVAL = TRAJECTORIES_DIR / f"{STEM}_eval_examples_cleaned_value_scored.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-suffix", default="_recognition_probe")
    return parser.parse_args()


def recover_observations(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach each row's observation from its successor's input_history; keep only rows that
    get one. Rows are emitted in their original file order."""
    by_trajectory: dict[str, dict[int, int]] = defaultdict(dict)
    for position, row in enumerate(rows):
        by_trajectory[row["trajectory_id"]][int(row.get("interaction_index") or 0)] = position

    kept: list[dict[str, Any]] = []
    stats = {
        "input_rows": len(rows),
        "no_successor": 0,
        "misaligned_step": 0,
        "empty_observation": 0,
        "kept": 0,
        "kept_by_benchmark": defaultdict(int),
        "observation_char_min": None,
        "observation_char_max": None,
    }
    for row in rows:
        steps = by_trajectory[row["trajectory_id"]]
        index = int(row.get("interaction_index") or 0)
        successor_position = steps.get(index + 1)
        if successor_position is None:
            stats["no_successor"] += 1
            continue
        history = rows[successor_position].get("input_history") or []
        last = history[-1] if history and isinstance(history[-1], dict) else None
        # Alignment guard: the appended history item must be THIS row's step. Verified to hold
        # universally on the current data; fail closed (skip) rather than mispair if it ever
        # stops holding after a regeneration.
        if last is None or last.get("step") != index + 1:
            stats["misaligned_step"] += 1
            continue
        observation = last.get("observation")
        if not isinstance(observation, str) or not observation.strip():
            stats["empty_observation"] += 1
            continue
        out = dict(row)
        out["observation"] = observation  # key read by canonical_event_observation_text()
        kept.append(out)
        stats["kept"] += 1
        stats["kept_by_benchmark"][row.get("benchmark") or "unknown"] += 1
        n = len(observation)
        stats["observation_char_min"] = n if stats["observation_char_min"] is None else min(stats["observation_char_min"], n)
        stats["observation_char_max"] = n if stats["observation_char_max"] is None else max(stats["observation_char_max"], n)
    stats["kept_by_benchmark"] = dict(stats["kept_by_benchmark"])
    return kept, stats


def main() -> None:
    args = parse_args()
    manifest: dict[str, Any] = {
        "observation_source": "successor row input_history[-1].observation (labeler-truncated to ~200 chars)",
        "interpretation": "recognition accuracy on these observations is a LOWER bound on the true recognition ceiling",
    }
    for split, path in (("train", args.train_path), ("eval", args.eval_path)):
        rows = [json.loads(line) for line in path.open() if line.strip()]
        kept, stats = recover_observations(rows)
        out_path = path.with_name(path.stem + args.output_suffix + path.suffix)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest[split] = {"input": str(path), "output": str(out_path), **stats}
        print(
            f"[{split}] {stats['input_rows']} -> {stats['kept']} rows with observations "
            f"(no successor: {stats['no_successor']}, misaligned: {stats['misaligned_step']}, "
            f"empty: {stats['empty_observation']}) -> {out_path.name}"
        )
        print(f"        by benchmark: {stats['kept_by_benchmark']}")
    manifest_path = TRAJECTORIES_DIR / f"{STEM}_recognition_probe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
