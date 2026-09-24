from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_METRICS = [
    "semantic_equivalence",
    "factual_consistency",
    "intent_alignment",
    "outcome_alignment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate average LLM judge scores from an evaluation metrics JSON file."
        )
    )
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
        "--score-path",
        default="llm_judge.scores",
        help=(
            "Dot-separated path inside each record that contains the metric scores."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to average.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print the result as JSON.",
    )
    return parser.parse_args()


def get_nested(mapping: object, path: list[str]) -> object | None:
    current = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def load_records(
    json_path: Path,
    section: str,
    records_key: str,
) -> list[dict]:
    payload = json.loads(json_path.read_text())
    section_payload = payload.get(section)
    if not isinstance(section_payload, dict):
        raise ValueError(
            f"Section '{section}' is missing or is not an object in {json_path}."
        )

    records = section_payload.get(records_key)
    if not isinstance(records, list):
        raise ValueError(
            f"Key '{records_key}' under section '{section}' is missing or is not a list."
        )

    return [record for record in records if isinstance(record, dict)]


def compute_averages(
    records: list[dict],
    metrics: list[str],
    score_path: list[str],
) -> dict[str, dict[str, float | int | None]]:
    values: dict[str, list[float]] = {metric: [] for metric in metrics}

    for record in records:
        scores = get_nested(record, score_path)
        if not isinstance(scores, dict):
            continue

        for metric in metrics:
            value = scores.get(metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[metric].append(float(value))

    result: dict[str, dict[str, float | int | None]] = {}
    for metric, metric_values in values.items():
        average = (
            sum(metric_values) / len(metric_values) if metric_values else None
        )
        result[metric] = {
            "average": average,
            "count": len(metric_values),
        }
    return result


def main() -> None:
    args = parse_args()
    score_path = args.score_path.split(".")
    records = load_records(args.json_path, args.section, args.records_key)
    averages = compute_averages(records, args.metrics, score_path)

    if args.as_json:
        print(
            json.dumps(
                {
                    "json_path": str(args.json_path),
                    "section": args.section,
                    "records": len(records),
                    "score_path": args.score_path,
                    "metrics": averages,
                },
                indent=2,
            )
        )
        return

    print(f"File: {args.json_path}")
    print(f"Section: {args.section}")
    print(f"Records: {len(records)}")
    print(f"Score path: {args.score_path}")
    for metric, summary in averages.items():
        average = summary["average"]
        count = summary["count"]
        average_text = f"{average:.6f}" if average is not None else "n/a"
        print(f"{metric}: average={average_text} count={count}")


if __name__ == "__main__":
    main()
