#!/usr/bin/env python3
"""Rewrite the canonical_event_with_nudge examples with ensemble majority-vote labels.

The labels shipped in trajectories/canonical_event_with_nudge_..._examples_cleaned.jsonl come
from a SINGLE rater (gpt-5.4-mini). src/data_preparation/ensemble_relabel_canonical_events.py
re-labelled the same transitions with three frontier models and wrote a per-transition consensus
file; this script applies that consensus back onto the example rows, so the trainers can be
pointed at ensemble-labelled data instead.

Join key is `trajectory_id::interaction_index`, the same key ensemble_relabel_canonical_events.py
emits (its `row_key`).

Two things the consensus file does NOT settle, both handled here:

1. COVERAGE. The ensemble run labelled a subset (the sampled transitions), not every row. Rows
   with no consensus entry keep their original gpt-5.4-mini label by default (--unlabeled keep),
   so the corpus does not shrink; `label_source` stays "llm" on those rows, so the provenance
   is visible per row rather than assumed. --unlabeled drop emits only the relabelled subset.

2. TIES. With three raters a field can split 1/1/1. The consensus file resolves those by
   Counter.most_common, which is an arbitrary pick determined by rater order -- it records them
   in `contested_fields` precisely because they are not majority decisions. Default here is
   --tiebreak original: treat the existing gpt-5.4-mini label as a fourth rater, so a 1/1/1 split
   where the original agrees with one rater becomes a real 2/4 majority. A genuine four-way split
   keeps the original label and is counted as `no_majority`. --tiebreak first-rater reproduces
   the consensus file's arbitrary pick; --tiebreak keep-original never breaks ties.

Per-field provenance is written to `label_field_source` on every relabelled row, so a downstream
analysis can exclude tie-broken fields without re-deriving anything.

The value-head targets (`step_score`, `value_target`, ...) are DERIVED from these labels, so they
are deliberately not copied: run src/data_preparation/annotate_step_value_scores.py on the output
of this script to regenerate them. This script refuses to read an already-value-scored input for
that reason.

Usage:
  uv run python src/data_preparation/apply_ensemble_consensus_labels.py
  uv run python src/data_preparation/annotate_step_value_scores.py \
      --train-path trajectories/..._train_examples_cleaned_ensemble.jsonl \
      --eval-path  trajectories/..._eval_examples_cleaned_ensemble.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.canonical_event_schema import (  # noqa: E402
    CANONICAL_EVENT_ALL_FIELDS,
    CANONICAL_EVENT_STATE_FIELDS,
    NUDGE_MULTI_LABEL_FIELDS,
)

TRAJECTORIES = REPO_ROOT / "trajectories"
STEM = "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro"
DEFAULT_TRAIN_INPUT = TRAJECTORIES / f"{STEM}_train_examples_cleaned.jsonl"
DEFAULT_EVAL_INPUT = TRAJECTORIES / f"{STEM}_eval_examples_cleaned.jsonl"
DEFAULT_TRAIN_CONSENSUS = REPO_ROOT / "data/week1/ensemble_labels/ensemble_consensus_labels.jsonl"
DEFAULT_EVAL_CONSENSUS = REPO_ROOT / "data/week1_eval/ensemble_labels/ensemble_consensus_labels.jsonl"

# Value-head fields derived from the labels; stale the moment a label changes.
DERIVED_VALUE_FIELDS = ("step_score", "step_score_contributions", "value_target")


def row_key(row: dict[str, Any]) -> str:
    """Must match ensemble_relabel_canonical_events.row_key."""
    return f"{row.get('trajectory_id')}::{row.get('interaction_index')}"


def normalize(field: str, value: Any) -> Any:
    """Hashable, order-insensitive form for vote counting. The multi-label field is a list whose
    order carries no meaning, so ["a","b"] and ["b","a"] must count as the same vote."""
    if field in NUDGE_MULTI_LABEL_FIELDS:
        if value is None:
            return ()
        if isinstance(value, (list, tuple)):
            return tuple(sorted(str(v) for v in value))
        return (str(value),)
    return value


def denormalize(field: str, value: Any) -> Any:
    if field in NUDGE_MULTI_LABEL_FIELDS:
        return list(value)
    return value


def original_label(row: dict[str, Any], field: str) -> Any:
    container = row.get("canonical_event_state") if field in CANONICAL_EVENT_STATE_FIELDS else row.get("nudge")
    if isinstance(container, dict) and field in container:
        return container[field]
    bundle = row.get("canonical_event_with_nudge") or {}
    for part in ("canonical_event_state", "nudge"):
        section = bundle.get(part)
        if isinstance(section, dict) and field in section:
            return section[field]
    return None


def resolve_field(
    field: str, votes: list[Any], original: Any, tiebreak: str
) -> tuple[Any, str]:
    """(value, provenance) for one field. `votes` are the raters' values in rater order."""
    counts = Counter(normalize(field, v) for v in votes)
    top, count = counts.most_common(1)[0]
    if count >= 2:
        return denormalize(field, top), ("unanimous" if count == len(votes) else "majority")
    # 1/1/1 split.
    if tiebreak == "original":
        original_norm = normalize(field, original)
        if original_norm in counts:
            # The original agrees with one rater -> 2 of 4, a real majority.
            return denormalize(field, original_norm), "original_tiebreak"
        return original, "no_majority"
    if tiebreak == "keep-original":
        return original, "no_majority"
    return denormalize(field, top), "first_rater_tiebreak"  # reproduces the consensus file


def load_consensus(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    by_key: dict[str, dict[str, Any]] = {}
    models: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            by_key[entry["key"]] = entry
            if not models:
                models = list(entry.get("label_models") or [])
    return by_key, models


def relabel_split(
    input_path: Path,
    consensus_path: Path,
    output_path: Path,
    *,
    tiebreak: str,
    unlabeled: str,
    include_votes: bool,
) -> dict[str, Any]:
    consensus, models = load_consensus(consensus_path)
    stats: dict[str, Any] = {
        "input": str(input_path), "consensus": str(consensus_path), "output": str(output_path),
        "raters": models, "tiebreak": tiebreak, "unlabeled_policy": unlabeled,
        "rows_in": 0, "rows_out": 0, "relabelled": 0, "kept_original": 0,
        "consensus_keys": len(consensus), "consensus_keys_unmatched": 0,
        "rows_with_any_change": 0,
        "field_changed": Counter(), "field_provenance": defaultdict(Counter),
    }
    matched: set[str] = set()
    dropped_derived = False

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stats["rows_in"] += 1
            entry = consensus.get(row_key(row))
            if entry is None:
                if unlabeled == "drop":
                    continue
                stats["kept_original"] += 1
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["rows_out"] += 1
                continue
            matched.add(row_key(row))
            votes = entry.get("votes") or {}
            state: dict[str, Any] = {}
            nudge: dict[str, Any] = {}
            provenance: dict[str, str] = {}
            changed = False
            for field in CANONICAL_EVENT_ALL_FIELDS:
                original = original_label(row, field)
                field_votes = (votes.get(field) or {}).get("values")
                if not field_votes:
                    value, source = original, "missing_votes"
                else:
                    value, source = resolve_field(field, field_votes, original, tiebreak)
                provenance[field] = source
                stats["field_provenance"][field][source] += 1
                if normalize(field, value) != normalize(field, original):
                    stats["field_changed"][field] += 1
                    changed = True
                (state if field in CANONICAL_EVENT_STATE_FIELDS else nudge)[field] = value
            if changed:
                stats["rows_with_any_change"] += 1
            # Keep the three label containers consistent -- trainers read different ones.
            row["canonical_event_state"] = dict(sorted(state.items()))
            row["nudge"] = dict(sorted(nudge.items()))
            row["canonical_event_with_nudge"] = {
                "canonical_event_state": dict(sorted(state.items())),
                "nudge": dict(sorted(nudge.items())),
            }
            row["label_source"] = "ensemble_majority"
            row["label_model"] = ",".join(models)
            row["label_models"] = models
            row["label_field_source"] = provenance
            row["label_model_original"] = "gpt-5.4-mini"
            if include_votes:
                row["label_votes"] = votes
            for key in DERIVED_VALUE_FIELDS:
                if key in row:
                    row.pop(key)
                    dropped_derived = True
            stats["relabelled"] += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["rows_out"] += 1

    stats["consensus_keys_unmatched"] = len(set(consensus) - matched)
    stats["dropped_stale_value_fields"] = dropped_derived
    stats["field_changed"] = dict(stats["field_changed"])
    stats["field_provenance"] = {f: dict(c) for f, c in stats["field_provenance"].items()}
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    p.add_argument("--eval-input", type=Path, default=DEFAULT_EVAL_INPUT)
    p.add_argument("--train-consensus", type=Path, default=DEFAULT_TRAIN_CONSENSUS)
    p.add_argument("--eval-consensus", type=Path, default=DEFAULT_EVAL_CONSENSUS)
    p.add_argument("--output-suffix", default="_ensemble")
    p.add_argument(
        "--tiebreak", choices=("original", "first-rater", "keep-original"), default="original",
        help="How to resolve a 1/1/1 three-way split. `original` (default) uses the existing "
             "gpt-5.4-mini label as a fourth vote; `first-rater` reproduces the consensus "
             "file's arbitrary pick; `keep-original` never breaks ties.",
    )
    p.add_argument(
        "--unlabeled", choices=("keep", "drop"), default="keep",
        help="Rows the ensemble did not label: keep the original gpt-5.4-mini label (default) "
             "or drop the row.",
    )
    p.add_argument("--include-votes", action="store_true", help="Embed the per-field raw votes.")
    p.add_argument("--report", type=Path, default=None, help="Write the stats JSON here.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {}
    for split, input_path, consensus_path in (
        ("train", args.train_input, args.train_consensus),
        ("eval", args.eval_input, args.eval_consensus),
    ):
        if not input_path.is_file():
            raise SystemExit(f"missing input: {input_path}")
        if not consensus_path.is_file():
            raise SystemExit(f"missing consensus: {consensus_path}")
        if "_value_scored" in input_path.name:
            raise SystemExit(
                f"{input_path.name} is already value-scored. Relabel the *_cleaned.jsonl file and "
                "re-run annotate_step_value_scores.py -- step_score/value_target are derived from "
                "the labels and would be stale."
            )
        output_path = input_path.with_name(input_path.stem + args.output_suffix + input_path.suffix)
        stats = relabel_split(
            input_path, consensus_path, output_path,
            tiebreak=args.tiebreak, unlabeled=args.unlabeled, include_votes=args.include_votes,
        )
        report[split] = stats
        print(f"[{split}] {stats['rows_in']} in -> {stats['rows_out']} out  "
              f"(relabelled {stats['relabelled']}, kept original {stats['kept_original']}, "
              f"changed {stats['rows_with_any_change']})")
        print(f"         -> {output_path}")
        for field in CANONICAL_EVENT_ALL_FIELDS:
            changed = stats["field_changed"].get(field, 0)
            prov = stats["field_provenance"].get(field, {})
            if stats["relabelled"]:
                print(f"         {field:30s} changed {changed:6d} ({changed/stats['relabelled']:6.1%})  "
                      + "  ".join(f"{k}={v}" for k, v in sorted(prov.items())))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(f"\nreport -> {args.report}")
    print("\nNEXT: regenerate the derived value-head targets on the new files:\n"
          "  uv run python src/data_preparation/annotate_step_value_scores.py \\\n"
          f"      --train-path {report['train']['output']} \\\n"
          f"      --eval-path {report['eval']['output']}")


if __name__ == "__main__":
    main()
