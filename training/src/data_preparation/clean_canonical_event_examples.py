#!/usr/bin/env python3
"""Clean the canonical_event_with_nudge LLM-labeled example JSONL files.

Three defects observed in the materialized examples (see the cleaning manifest this
script writes for the exact counts):

1. **Uninformative labels** -- records where the labeling LLM effectively punted:
   every canonical_event_state field is ``unknown`` except at most ``risk_signal:
   "none"`` (that field's null default). Such a label supervises nothing about the
   transition and only teaches the model to emit ``unknown``. Dropped.

2. **Duplicate inputs** -- records sharing the identical model input (system prompt,
   task prompt, action, input history, previous state). Most duplicate groups carry
   CONFLICTING labels (the labeler was sampled more than once and disagreed), which
   is contradictory supervision. Per group, the single record with the fewest
   ``unknown`` fields is kept (ties -> first occurrence, file order preserved).

3. **Train->eval leakage** -- eval records whose input also appears in the cleaned
   train file are dropped from eval.

Kept records are written verbatim (original JSONL line bytes), so nothing changes
except which lines survive. Outputs are ``<stem>_cleaned.jsonl`` next to the inputs
plus a ``*_cleaned_manifest.json`` documenting what was dropped and why.

Usage:

    uv run python src/data_preparation/clean_canonical_event_examples.py
    # or with explicit paths:
    uv run python src/data_preparation/clean_canonical_event_examples.py \
        --train-path trajectories/..._train_examples.jsonl \
        --eval-path trajectories/..._eval_examples.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAJECTORIES_DIR = REPO_ROOT / "trajectories"
DEFAULT_TRAIN = TRAJECTORIES_DIR / (
    "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro_train_examples.jsonl"
)
DEFAULT_EVAL = TRAJECTORIES_DIR / (
    "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro_eval_examples.jsonl"
)

EVENT_FIELDS = (
    "action_type",
    "error_signature",
    "execution_status",
    "object_type",
    "progress_signal",
    "risk_signal",
    "side_effect_type",
)
NUDGE_FIELDS = ("information_gain", "information_sufficiency", "recommended_abstract_action")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL)
    parser.add_argument(
        "--output-suffix",
        default="_cleaned",
        help="Suffix inserted before .jsonl for the cleaned output files.",
    )
    return parser.parse_args()


def load_lines(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Return (raw_line, parsed_record) pairs so kept records can be re-emitted verbatim."""
    pairs: list[tuple[str, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Malformed JSON at {path}:{line_number}: {exc}") from exc
            pairs.append((stripped, record))
    return pairs


def label_of(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    label = record.get("canonical_event_with_nudge") or {}
    event = label.get("canonical_event_state") or record.get("canonical_event_state") or {}
    nudge = label.get("nudge") or record.get("nudge") or {}
    return (event if isinstance(event, dict) else {}), (nudge if isinstance(nudge, dict) else {})


def event_is_uninformative(event: dict[str, Any]) -> bool:
    """True when the event label supervises nothing: every field is ``unknown``,
    tolerating only ``risk_signal: "none"`` (that field's null default) as 'known'."""
    if not event:
        return True
    for field in EVENT_FIELDS:
        value = str(event.get(field, "unknown"))
        if value == "unknown":
            continue
        if field == "risk_signal" and value == "none":
            continue
        return False
    return True


def unknown_field_count(record: dict[str, Any]) -> int:
    """Total ``unknown`` count across the 11 categorical fields (used to pick the
    best record inside a duplicate group -- fewer unknowns = more informative label)."""
    event, nudge = label_of(record)
    count = sum(1 for field in EVENT_FIELDS if str(event.get(field, "unknown")) == "unknown")
    count += sum(1 for field in NUDGE_FIELDS if str(nudge.get(field, "unknown")) == "unknown")
    missing = nudge.get("missing_information_type") or []
    if isinstance(missing, list) and missing and all(str(item) == "unknown" for item in missing):
        count += 1
    return count


def input_key(record: dict[str, Any]) -> str:
    """The full model input: two records with equal keys produce the identical
    training prompt, so keeping both is pure redundancy (or, with differing labels,
    contradictory supervision)."""
    return json.dumps(
        [
            record.get("system_prompt"),
            record.get("task_prompt"),
            record.get("action"),
            record.get("input_history"),
            record.get("previous_state"),
        ],
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def clean_split(
    pairs: list[tuple[str, dict[str, Any]]],
    *,
    exclude_input_keys: set[str] | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    """Apply the three rules; returns (kept pairs in original order, stats)."""
    stats = {
        "input_records": len(pairs),
        "dropped_uninformative_label": 0,
        "duplicate_input_groups": 0,
        "duplicate_groups_with_conflicting_labels": 0,
        "dropped_duplicate_records": 0,
        "dropped_leaked_records": 0,
    }

    survivors: list[tuple[int, str, dict[str, Any]]] = []
    for index, (raw, record) in enumerate(pairs):
        event, _ = label_of(record)
        if event_is_uninformative(event):
            stats["dropped_uninformative_label"] += 1
            continue
        survivors.append((index, raw, record))

    groups: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
    for entry in survivors:
        groups.setdefault(input_key(entry[2]), []).append(entry)

    kept: list[tuple[int, str, dict[str, Any]]] = []
    for key, members in groups.items():
        if exclude_input_keys is not None and key in exclude_input_keys:
            stats["dropped_leaked_records"] += len(members)
            continue
        if len(members) > 1:
            stats["duplicate_input_groups"] += 1
            labels = {json.dumps(label_of(record), sort_keys=True, default=str) for _, _, record in members}
            if len(labels) > 1:
                stats["duplicate_groups_with_conflicting_labels"] += 1
            stats["dropped_duplicate_records"] += len(members) - 1
        # Fewest unknown fields wins; ties resolve to the earliest occurrence. The kept
        # record is re-anchored at the group's FIRST index so file order is stable.
        best = min(members, key=lambda entry: (unknown_field_count(entry[2]), entry[0]))
        kept.append((members[0][0], best[1], best[2]))

    kept.sort(key=lambda entry: entry[0])
    stats["kept_records"] = len(kept)
    return [(raw, record) for _, raw, record in kept], stats


def write_jsonl(path: Path, pairs: list[tuple[str, dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for raw, _ in pairs:
            handle.write(raw + "\n")


def output_path(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix + path.suffix)


def main() -> None:
    args = parse_args()

    train_pairs = load_lines(args.train_path)
    eval_pairs = load_lines(args.eval_path)

    train_clean, train_stats = clean_split(train_pairs)
    train_keys = {input_key(record) for _, record in train_clean}
    eval_clean, eval_stats = clean_split(eval_pairs, exclude_input_keys=train_keys)

    train_out = output_path(args.train_path, args.output_suffix)
    eval_out = output_path(args.eval_path, args.output_suffix)
    write_jsonl(train_out, train_clean)
    write_jsonl(eval_out, eval_clean)

    manifest = {
        "train": {"input": str(args.train_path), "output": str(train_out), **train_stats},
        "eval": {"input": str(args.eval_path), "output": str(eval_out), **eval_stats},
        "rules": {
            "uninformative_label": (
                "every canonical_event_state field is 'unknown' (risk_signal 'none' tolerated as the null default)"
            ),
            "duplicate_input": (
                "identical (system_prompt, task_prompt, action, input_history, previous_state); "
                "kept the member with the fewest 'unknown' fields, ties -> first occurrence"
            ),
            "leakage": "eval records whose input key also appears in the cleaned train file",
        },
    }
    manifest_path = eval_out.with_name(eval_out.stem.replace("_eval_examples" + args.output_suffix, "") + "_cleaned_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for split, stats, out in (("train", train_stats, train_out), ("eval", eval_stats, eval_out)):
        print(
            f"[{split}] {stats['input_records']} -> {stats['kept_records']} "
            f"(uninformative: -{stats['dropped_uninformative_label']}, "
            f"duplicates: -{stats['dropped_duplicate_records']} across {stats['duplicate_input_groups']} groups "
            f"({stats['duplicate_groups_with_conflicting_labels']} conflicting), "
            f"leaked: -{stats['dropped_leaked_records']}) -> {out}"
        )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
