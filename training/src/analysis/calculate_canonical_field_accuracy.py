"""Calculate per-field categorical classification accuracy for canonical-event
next-state predictions.

Reads an `evaluation.py` metrics JSON (produced with
`--world-model-target canonical_event_state` or
`canonical_event_with_nudge`) and aggregates, over the `world_model_state_eval`
records, the exact-match accuracy of each categorical field in the predicted
canonical event / nudge against the gold label. Every canonical field is
categorical, so exact match is a well-defined classification metric; the
`missing_information_type` list field is matched as a set.

Each record is expected to carry a `canonical_field_comparisons` mapping
(written by `evaluate_world_model_predictions`). Records without it are skipped
and reported as such.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the evaluation metrics JSON file.",
    )
    parser.add_argument(
        "--section",
        default="world_model_state_eval",
        help="Top-level section containing the evaluation records.",
    )
    parser.add_argument(
        "--records-key",
        default="records",
        help="Key inside the section that contains the list of records.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print the result as JSON.",
    )
    return parser.parse_args()


def load_records(json_path: Path, section: str, records_key: str) -> list[dict]:
    payload = json.loads(json_path.read_text())
    section_payload = payload.get(section)
    if not isinstance(section_payload, dict):
        raise ValueError(f"Section '{section}' is missing or is not an object in {json_path}.")
    records = section_payload.get(records_key)
    if not isinstance(records, list):
        raise ValueError(
            f"Key '{records_key}' under section '{section}' is missing or is not a list."
        )
    return [record for record in records if isinstance(record, dict)]


def compute_field_accuracy(records: list[dict]) -> dict[str, Any]:
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    scored_records = 0
    full_matches = 0
    skipped_records = 0

    for record in records:
        comparisons = record.get("canonical_field_comparisons")
        if not isinstance(comparisons, dict) or not comparisons:
            skipped_records += 1
            continue
        scored_records += 1
        all_match = True
        for field, comparison in comparisons.items():
            if not isinstance(comparison, dict):
                continue
            total[field] = total.get(field, 0) + 1
            if comparison.get("match"):
                correct[field] = correct.get(field, 0) + 1
            else:
                all_match = False
        full_matches += int(all_match)

    per_field = {
        field: {
            "accuracy": correct.get(field, 0) / total[field],
            "correct": correct.get(field, 0),
            "count": total[field],
        }
        for field in sorted(total)
    }
    total_correct = sum(correct.values())
    total_count = sum(total.values())
    micro = total_correct / total_count if total_count else None
    macro = (
        sum(per_field[field]["accuracy"] for field in per_field) / len(per_field)
        if per_field
        else None
    )
    return {
        "scored_records": scored_records,
        "skipped_records": skipped_records,
        "micro_field_accuracy": micro,
        "macro_field_accuracy": macro,
        "full_match_rate": full_matches / scored_records if scored_records else None,
        "per_field": per_field,
    }


def main() -> None:
    args = parse_args()
    records = load_records(args.json_path, args.section, args.records_key)
    result = compute_field_accuracy(records)

    if args.as_json:
        print(
            json.dumps(
                {
                    "json_path": str(args.json_path),
                    "section": args.section,
                    "records": len(records),
                    **result,
                },
                indent=2,
            )
        )
        return

    print(f"File: {args.json_path}")
    print(f"Section: {args.section}")
    print(f"Records: {len(records)}")
    print(f"Scored records: {result['scored_records']} (skipped: {result['skipped_records']})")

    def _fmt(value: float | None) -> str:
        return f"{value:.6f}" if value is not None else "n/a"

    print(f"micro_field_accuracy: {_fmt(result['micro_field_accuracy'])}")
    print(f"macro_field_accuracy: {_fmt(result['macro_field_accuracy'])}")
    print(f"full_match_rate: {_fmt(result['full_match_rate'])}")
    for field, summary in result["per_field"].items():
        print(
            f"{field}: accuracy={summary['accuracy']:.6f} "
            f"({summary['correct']}/{summary['count']})"
        )


if __name__ == "__main__":
    main()
