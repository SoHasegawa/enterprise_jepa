#!/usr/bin/env python3
"""Classify low-score evaluation records into fine-grained error causes.

The script reads `world_model_state_eval.records` from an evaluation JSON and
uses `src/llm.py` to classify records whose `llm_judge.overall_score` is below
the requested threshold.

The category set is derived from examples in:
`sessions/gymops_tool_output_qwen36_27b/evaluation_metrics_test_gpt-4o-mini.json`
and is intentionally more fine-grained than broad buckets like "semantic
mismatch". In particular, it distinguishes:

- permission errors that were missed
- not-found errors that were missed
- validation / bad-input errors that were missed
- ID / key-field mismatches
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm import LLM


DEFAULT_INPUT = (
    ROOT
    / "sessions"
    / "gymops_tool_output_qwen36_27b"
    / "evaluation_metrics_test_gpt-4o-mini.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "sessions"
    / "gymops_tool_output_qwen36_27b"
    / "evaluation_metrics_test_gpt-4o-mini_low_score_causes.json"
)

CATEGORIES: dict[str, str] = {
    "id_or_key_field_mismatch": (
        "Main problem is wrong IDs or key factual fields: account_id, user_id, case_id, "
        "product_id, location_id, owner_id, email, serial_number, service_id, state, status, priority, channel, etc."
    ),
    "missing_expected_records_or_details": (
        "Prediction is incomplete: empty result when gold has content, missing rows, missing fields, or missing descriptive details."
    ),
    "extra_or_hallucinated_records": (
        "Prediction adds extra rows, extra fields, or unsupported content not present in the gold result."
    ),
    "missed_permission_error": (
        "Gold indicates a permission / authorization / access-denied type error, but prediction misses it or predicts the wrong outcome."
    ),
    "missed_not_found_error": (
        "Gold indicates a not-found / no-match / missing-entity type error, but prediction misses it or predicts the wrong outcome."
    ),
    "missed_validation_error": (
        "Gold indicates invalid input / bad request / schema / required-field / type validation error, but prediction misses it or predicts the wrong outcome."
    ),
    "wrong_error_reason": (
        "Both sides indicate failure, but the predicted failure reason is materially different from the gold reason and does not fit a narrower error family above."
    ),
    "format_or_schema_mismatch": (
        "Prediction shape/format is the main issue: wrong schema, wrong output form, unusable structure, header-like output instead of actual payload, etc."
    ),
    "entity_or_content_mismatch": (
        "Prediction returns a substantially different entity/result overall and the main issue is not best explained by the narrower categories above."
    ),
}

CLASSIFICATION_SCHEMA = {
    "primary_cause": "",
    "secondary_cause": "",
    "confidence": 0.0,
    "rationale": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--llm-method", default="gpt4-mini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_low_score_records(payload: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    eval_block = payload.get("world_model_state_eval")
    if not isinstance(eval_block, dict):
        raise SystemExit("Missing `world_model_state_eval` object.")
    records = eval_block.get("records")
    if not isinstance(records, list):
        raise SystemExit("Missing `world_model_state_eval.records` list.")

    result: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        judge = record.get("llm_judge") or {}
        score = judge.get("overall_score")
        if isinstance(score, (int, float)) and float(score) < threshold:
            item = dict(record)
            item["_record_index"] = idx
            result.append(item)
    return result


def build_system_prompt() -> str:
    category_text = "\n".join(f"- {name}: {desc}" for name, desc in CATEGORIES.items())
    return (
        "You classify the main cause of a low evaluation score for predicted tool output.\n\n"
        "Choose the single best primary cause from the allowed categories below. "
        "Optionally choose one secondary cause if it clearly contributes.\n\n"
        "Allowed categories:\n"
        f"{category_text}\n\n"
        "Decision rules:\n"
        "1. Prefer the narrowest category.\n"
        "2. Use `missed_permission_error` when the gold output indicates lack of permission/access.\n"
        "3. Use `missed_not_found_error` when the gold output indicates missing entity / no match found.\n"
        "4. Use `missed_validation_error` when the gold output indicates invalid input, schema, type, or required-field failure.\n"
        "5. Use `id_or_key_field_mismatch` when the predicted output has mostly the right operation but wrong identifiers or other crucial values.\n"
        "6. Use `wrong_error_reason` only when the failure reason differs materially but does not fit the three specific error families above.\n"
        "7. Use `format_or_schema_mismatch` only when output structure/format is itself the main problem.\n"
        "8. Return JSON only."
    )


def build_user_prompt(record: dict[str, Any]) -> str:
    judge = record.get("llm_judge") or {}
    return (
        f"overall_score: {judge.get('overall_score')}\n\n"
        f"prediction:\n{record.get('prediction') or ''}\n\n"
        f"gold:\n{record.get('gold') or ''}\n\n"
        f"judge_comments:\n{judge.get('comments') or ''}\n\n"
        "Classify why this example scored below threshold."
    )


def normalize_category(value: Any) -> str:
    if not value:
        return ""
    raw = str(value).strip()
    if raw in CATEGORIES:
        return raw
    simplified = raw.lower().replace("-", "_").replace(" ", "_")
    for category in CATEGORIES:
        if simplified == category:
            return category
    return ""


def classify_record(llm: LLM, record: dict[str, Any]) -> dict[str, Any]:
    response = llm.generate_format(
        build_user_prompt(record),
        build_system_prompt(),
        temperature=0.0,
        format="json",
        schema=CLASSIFICATION_SCHEMA,
    )
    if not isinstance(response, dict):
        raise ValueError(f"Expected dict classifier output, got {type(response).__name__}: {response!r}")

    primary = normalize_category(response.get("primary_cause"))
    secondary = normalize_category(response.get("secondary_cause"))
    if not primary:
        raise ValueError(f"Invalid primary cause: {response.get('primary_cause')!r}")
    if secondary == primary:
        secondary = ""
    try:
        confidence = float(response.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "primary_cause": primary,
        "secondary_cause": secondary,
        "confidence": confidence,
        "rationale": str(response.get("rationale", "")).strip(),
        "raw_response": response,
    }


def load_resume(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = load_json(path)
    entries = payload.get("classified_records")
    if not isinstance(entries, list):
        return {}
    cached: dict[int, dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        idx = item.get("record_index")
        classification = item.get("classification")
        if isinstance(idx, int) and isinstance(classification, dict):
            cached[idx] = classification
    return cached


def main() -> None:
    args = parse_args()
    payload = load_json(args.input_path)
    records = get_low_score_records(payload, args.threshold)
    if args.limit is not None:
        records = records[: max(0, args.limit)]

    resume_map = load_resume(args.output) if args.resume else {}
    llm = LLM(args.llm_method)

    primary_counts: Counter[str] = Counter()
    secondary_counts: Counter[str] = Counter()
    classified_records: list[dict[str, Any]] = []

    for idx, record in enumerate(records, start=1):
        record_index = int(record["_record_index"])
        classification = resume_map.get(record_index)
        if classification is None:
            classification = classify_record(llm, record)
        primary_counts[classification["primary_cause"]] += 1
        if classification.get("secondary_cause"):
            secondary_counts[classification["secondary_cause"]] += 1
        classified_records.append(
            {
                "record_index": record_index,
                "overall_score": float((record.get("llm_judge") or {}).get("overall_score")),
                "prediction": record.get("prediction"),
                "gold": record.get("gold"),
                "judge_comments": (record.get("llm_judge") or {}).get("comments"),
                "classification": classification,
            }
        )
        print(
            f"[{idx}/{len(records)}] record_index={record_index} "
            f"score={classified_records[-1]['overall_score']:.3f} "
            f"primary={classification['primary_cause']}",
            flush=True,
        )

    output = {
        "input_path": str(args.input_path),
        "threshold": args.threshold,
        "llm_method": args.llm_method,
        "category_definitions": CATEGORIES,
        "low_score_record_count": len(classified_records),
        "primary_cause_counts": dict(primary_counts.most_common()),
        "secondary_cause_counts": dict(secondary_counts.most_common()),
        "classified_records": classified_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
