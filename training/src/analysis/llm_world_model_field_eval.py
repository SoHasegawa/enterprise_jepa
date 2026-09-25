#!/usr/bin/env python3
"""Per-field next-state evaluation for the LLM-based world model, comparable to the JEPA heads.

The JEPA canonical-event heads are scored per field on
`canonical_event_with_nudge_*.jsonl` (11 fields: 7 canonical_event_state + 4 nudge). The
causal-LM world model trained with `--world-model-target tool_output` does not emit those
fields at all -- it predicts the raw tool output text -- so a like-for-like comparison needs a
bridge, and the bridge has to be measured rather than assumed. This script therefore reports
three layers:

  1. text     Predicted tool output vs the GOLD observation, which is recoverable from the
              eval file itself: step t's observation is stored in step t+1's `input_history`
              (the same trick src/data_preparation/ensemble_relabel_canonical_events.py used
              to build the labels). Exact match plus token precision/recall/F1.
  2. derived  `execution_status` collapsed to failure vs non-failure using the repo's own
              `tool_output_looks_like_failure` -- the same derivation replay uses at inference
              (predict_world_model_feedback), so it needs no annotator and no API budget. This
              is the field the beam-plan critic actually gates on.
  3. annotated  All 11 fields, by re-running the SAME labeling prompt the gold ensemble used
              (`build_labeling_messages`) over the model's predicted observation. Because that
              annotator is itself noisy, the script also annotates the GOLD observation and
              reports it as `annotator_ceiling` -- the score a perfect world model would get
              through this bridge. Read the model number against the ceiling, not against 1.0.

Metric definitions match evaluate_canonical_event_heads in src/finetuning_jepa.py (macro
averages over classes with gold support, per-class precision/recall/F1, `classes_missed`),
so numbers can be placed side by side with a JEPA metrics file via --jepa-metrics.

Examples
--------
# free layers only (text + derived execution_status) over 500 rows
uv run python src/analysis/llm_world_model_field_eval.py \
    --world-model-path checkpoints/llm_wm --limit 500

# full 11-field comparison against a JEPA run, annotated with gpt-5.1
uv run python src/analysis/llm_world_model_field_eval.py \
    --world-model-path checkpoints/llm_wm --limit 300 --annotator gpt-5.1 \
    --jepa-metrics checkpoints/data_jepa_heads_ensemble/canonical_event_training_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.canonical_event_schema import (  # noqa: E402
    CANONICAL_EVENT_SINGLE_LABEL_FIELDS,
    NUDGE_MULTI_LABEL_FIELDS,
)
from src.finetuning import (  # noqa: E402
    DEFAULT_TRAJECTORIES_DIR,
    WorldModelStateExample,
    build_state_prediction_chat_messages,
    dump_json,
    make_blank_state,
    tool_output_looks_like_failure,
)
from src.finetuning_jepa import (  # noqa: E402
    canonical_event_field_value,
    load_jsonl_rows,
    render_action,
    stringify_tool_output,
)

DEFAULT_EVAL_JSONL = (
    DEFAULT_TRAJECTORIES_DIR
    / "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro_eval_examples_cleaned_ensemble_value_scored.jsonl"
)
ALL_FIELDS = tuple(CANONICAL_EVENT_SINGLE_LABEL_FIELDS) + tuple(NUDGE_MULTI_LABEL_FIELDS)


# --------------------------------------------------------------------------------------
# metrics (definitions mirror evaluate_canonical_event_heads)
# --------------------------------------------------------------------------------------
def classification_metrics(
    gold: list[Any], predicted: list[Any], multi_label: bool = False
) -> dict[str, Any]:
    """Accuracy + per-class precision/recall/F1 + macro/weighted averages for one field.

    Macro averages cover only classes with gold support: a category that never occurs in the
    eval split would otherwise contribute a hard 0 and make the number a function of the label
    vocabulary rather than of prediction quality. `classes_missed` lists supported categories
    the model never predicts -- the collapse signal accuracy hides.
    """
    if not gold:
        return {"accuracy": 0.0, "macro_f1": 0.0, "macro_recall": 0.0, "examples": 0, "per_class": {}}

    def as_set(value: Any) -> set[str]:
        if isinstance(value, (list, tuple, set)):
            return {str(item) for item in value}
        return {str(value)}

    exact = 0
    support: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    true_positive: Counter[str] = Counter()
    unparsed = 0
    for gold_value, pred_value in zip(gold, predicted):
        gold_set = as_set(gold_value)
        if pred_value is None:
            unparsed += 1
            pred_set: set[str] = set()
        else:
            pred_set = as_set(pred_value)
        if gold_set == pred_set:
            exact += 1
        for value in gold_set:
            support[value] += 1
        for value in pred_set:
            predicted_counts[value] += 1
        for value in gold_set & pred_set:
            true_positive[value] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    weighted_f1 = 0.0
    missed: list[str] = []
    for value in sorted(set(support) | set(predicted_counts)):
        tp = true_positive[value]
        precision = tp / predicted_counts[value] if predicted_counts[value] else 0.0
        recall = tp / support[value] if support[value] else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[value] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": support[value], "predicted": predicted_counts[value],
        }
        if support[value]:
            f1s.append(f1)
            recalls.append(recall)
            precisions.append(precision)
            weighted_f1 += f1 * support[value]
            if predicted_counts[value] == 0:
                missed.append(value)
    total_support = sum(support.values())
    return {
        "accuracy": exact / len(gold),
        "macro_f1": (sum(f1s) / len(f1s)) if f1s else 0.0,
        "macro_precision": (sum(precisions) / len(precisions)) if precisions else 0.0,
        "macro_recall": (sum(recalls) / len(recalls)) if recalls else 0.0,
        "weighted_f1": (weighted_f1 / total_support) if total_support else 0.0,
        "classes_with_support": len(f1s),
        "classes_missed": missed,
        "unparsed_predictions": unparsed,
        "examples": len(gold),
        "per_class": per_class,
        "accuracy_is_exact_match": multi_label,
    }


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.:/@-]+")


def token_overlap(predicted: str, gold: str) -> dict[str, float]:
    """Bag-of-tokens precision/recall/F1 -- a floor on text agreement that does not reward
    length the way a raw containment check would."""
    pred_tokens = Counter(_TOKEN_PATTERN.findall((predicted or "").lower()))
    gold_tokens = Counter(_TOKEN_PATTERN.findall((gold or "").lower()))
    overlap = sum((pred_tokens & gold_tokens).values())
    pred_total, gold_total = sum(pred_tokens.values()), sum(gold_tokens.values())
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / gold_total if gold_total else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
    }


# --------------------------------------------------------------------------------------
# eval rows
# --------------------------------------------------------------------------------------
def recover_gold_observations(rows: list[dict[str, Any]]) -> dict[int, str]:
    """row index -> the observation THIS row's action produced.

    A canonical-event row stores only its own action; the resulting observation is the last
    `input_history` entry of the next step in the same trajectory. The final step of every
    trajectory therefore has no recoverable observation and is reported separately rather
    than silently scored as an empty string.
    """
    by_trajectory: dict[str, dict[int, int]] = {}
    for index, row in enumerate(rows):
        key = str(row.get("trajectory_id") or "")
        by_trajectory.setdefault(key, {})[int(row.get("interaction_index") or 0)] = index
    observations: dict[int, str] = {}
    for index, row in enumerate(rows):
        key = str(row.get("trajectory_id") or "")
        successor = by_trajectory.get(key, {}).get(int(row.get("interaction_index") or 0) + 1)
        if successor is None:
            continue
        history = rows[successor].get("input_history") or []
        if not history:
            continue
        observation = stringify_tool_output((history[-1] or {}).get("observation") or "").strip()
        if observation:
            observations[index] = observation
    return observations


def build_prompt_messages(row: dict[str, Any], target_mode: str, args: argparse.Namespace) -> list[dict[str, str]]:
    """The same prompt the world model was TRAINED with, filled from a canonical-event row.

    Uses exactly the inputs the JEPA path gets for this row (system prompt, task prompt,
    input_history, previous_state, action), so neither model sees information the other does not.
    """
    previous_state = row.get("previous_state")
    if not isinstance(previous_state, dict):
        previous_state = make_blank_state()
    example = WorldModelStateExample(
        trajectory_id=str(row.get("trajectory_id") or ""),
        trajectory_index=int(row.get("trajectory_index") or 0),
        interaction_index=int(row.get("interaction_index") or 0),
        system_prompt=str(row.get("system_prompt") or ""),
        user_prompt=str(row.get("task_prompt") or ""),
        action=row.get("action"),
        state_history=list(row.get("state_history") or []),
        input_history=list(row.get("input_history") or []),
        previous_state=previous_state,
        state=make_blank_state(),
    )
    return build_state_prediction_chat_messages(
        example,
        target_mode=target_mode,
        include_error_message=False,
        include_stage=False,
        include_input_history=True,
        system_prompt_max_chars=args.system_prompt_max_chars,
        action_max_chars=args.action_max_chars,
    )


# --------------------------------------------------------------------------------------
# annotator bridge (predicted observation -> 11 fields)
# --------------------------------------------------------------------------------------
def annotate_observations(
    rows: list[dict[str, Any]],
    observations: list[str],
    method: str,
    workers: int,
    max_field_chars: int,
) -> list[dict[str, Any] | None]:
    """Label (action, observation) pairs with the gold ensemble's own prompt.

    Reuses build_labeling_messages / parse_and_validate from
    src/data_preparation/ensemble_relabel_canonical_events.py so the bridge asks exactly what
    the gold labels answered; anything else would measure prompt drift instead of the model.
    """
    from src.data_preparation.ensemble_relabel_canonical_events import (
        build_labeling_messages,
        compact_example_payload,
        parse_and_validate,
    )
    from src.finetuning import build_agent_generator

    generator = build_agent_generator(method, max_new_tokens=1024)
    lock_free_generators = [generator]
    if hasattr(generator, "clone_for_parallel_requests"):
        lock_free_generators += [generator.clone_for_parallel_requests() for _ in range(max(0, workers - 1))]

    def annotate(index_and_row: tuple[int, dict[str, Any]]) -> dict[str, Any] | None:
        index, row = index_and_row
        observation = observations[index]
        if not observation:
            return None
        payload_row = dict(row)
        payload_row["observation"] = observation
        payload = compact_example_payload(
            payload_row, context="step", trajectory_steps=None,
            max_input_history_items=4, max_trajectory_steps=0, max_field_chars=max_field_chars,
        )
        worker = lock_free_generators[index % len(lock_free_generators)]
        for attempt in range(3):
            try:
                text = worker.generate_from_messages(build_labeling_messages(payload), temperature=0.0)
                return parse_and_validate(text)
            except Exception as exc:                                   # noqa: BLE001
                if attempt == 2:
                    print(f"[annotate] row {index} failed: {type(exc).__name__}: {exc}", flush=True)
                    return None
                time.sleep(2 * (attempt + 1))
        return None

    results: list[dict[str, Any] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, label in zip(range(len(rows)), pool.map(annotate, list(enumerate(rows)))):
            results[index] = label
    return results


def label_field(label: dict[str, Any] | None, field: str) -> Any:
    if not isinstance(label, dict):
        return None
    return canonical_event_field_value(label, field)


# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-model-path", required=True, help="Fine-tuned causal-LM world-model checkpoint.")
    parser.add_argument("--eval-jsonl", type=Path, default=DEFAULT_EVAL_JSONL,
                        help="canonical_event_with_nudge JSONL the JEPA heads are scored on.")
    parser.add_argument("--limit", type=int, default=500,
                        help="Evaluate the first N rows that have a recoverable gold observation (0 = all).")
    parser.add_argument("--world-model-target", default=None,
                        help="Prompt/target mode. Defaults to the checkpoint's run_summary.json value.")
    parser.add_argument("--annotator", default="none",
                        help="LLM method that maps predicted observations to the 11 fields "
                             "(e.g. gpt-5.1, gemini, claude, vllm:9010/wm_agent). 'none' = skip layer 3.")
    parser.add_argument("--annotator-workers", type=int, default=8)
    parser.add_argument("--annotator-max-field-chars", type=int, default=4000)
    parser.add_argument("--skip-annotator-ceiling", action="store_true",
                        help="Do not annotate the GOLD observations. Halves annotator cost and "
                             "removes the reference the model score should be read against.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batched generations per forward.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--system-prompt-max-chars", type=int, default=0)
    parser.add_argument("--action-max-chars", type=int, default=0)
    parser.add_argument("--jepa-metrics", type=Path, default=None,
                        help="Optional canonical_event_training_metrics.json to print side by side.")
    parser.add_argument("--output", type=Path, default=None, help="Where to write the metrics JSON.")
    parser.add_argument("--dump-predictions", type=Path, default=None,
                        help="Optional JSONL of per-row prompts/predictions/labels for inspection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_mode = args.world_model_target
    if target_mode is None:
        summary_path = Path(args.world_model_path) / "run_summary.json"
        if summary_path.is_file():
            with summary_path.open() as handle:
                summary = json.load(handle)
            target_mode = ((summary.get("training_metrics") or {}).get("world_model_target")
                           or summary.get("world_model_target"))
        target_mode = target_mode or "tool_output"
    print(f"[setup] world_model_target={target_mode!r} (from the checkpoint unless overridden)", flush=True)

    rows = load_jsonl_rows(args.eval_jsonl)
    gold_observations = recover_gold_observations(rows)
    print(f"[setup] {len(rows)} eval rows, {len(gold_observations)} with a recoverable gold observation "
          f"({100 * len(gold_observations) / max(1, len(rows)):.1f}%; the rest are trajectory-final steps)",
          flush=True)
    indices = [index for index in range(len(rows)) if index in gold_observations]
    if args.limit > 0:
        indices = indices[: args.limit]
    subset = [rows[index] for index in indices]
    gold_texts = [gold_observations[index] for index in indices]
    print(f"[setup] evaluating {len(subset)} rows", flush=True)

    from src.finetuning import HFTextGenerator

    generator = HFTextGenerator(
        args.world_model_path, max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code, dtype=args.dtype, device_map=args.device_map,
    )
    messages = [build_prompt_messages(row, target_mode, args) for row in subset]
    predictions: list[str] = []
    started = time.time()
    for start in range(0, len(messages), max(1, args.batch_size)):
        chunk = messages[start : start + max(1, args.batch_size)]
        predictions.extend(generator.generate_from_messages_batch(chunk, temperature=0.0))
        done = min(start + len(chunk), len(messages))
        rate = done / max(1e-9, time.time() - started)
        print(f"  generated {done}/{len(messages)} ({rate:.2f} rows/s)", flush=True)
    predictions = [(text or "").split("</think>")[-1].strip() for text in predictions]

    metrics: dict[str, Any] = {
        "world_model_path": str(args.world_model_path),
        "world_model_target": target_mode,
        "eval_jsonl": str(args.eval_jsonl),
        "evaluated_rows": len(subset),
        "rows_with_recoverable_observation": len(gold_observations),
        "total_rows": len(rows),
        "benchmarks": dict(Counter(str(row.get("benchmark") or "unknown") for row in subset)),
    }

    # ---- layer 1: text ----------------------------------------------------------------
    overlaps = [token_overlap(pred, gold) for pred, gold in zip(predictions, gold_texts)]
    metrics["text"] = {
        "exact_match": sum(1 for p, g in zip(predictions, gold_texts) if p.strip() == g.strip()) / max(1, len(subset)),
        "token_f1": sum(o["f1"] for o in overlaps) / max(1, len(overlaps)),
        "token_precision": sum(o["precision"] for o in overlaps) / max(1, len(overlaps)),
        "token_recall": sum(o["recall"] for o in overlaps) / max(1, len(overlaps)),
        "empty_predictions": sum(1 for p in predictions if not p.strip()),
        "mean_predicted_chars": sum(len(p) for p in predictions) / max(1, len(subset)),
        "mean_gold_chars": sum(len(g) for g in gold_texts) / max(1, len(subset)),
    }

    # ---- layer 2: derived execution status (no annotator) -----------------------------
    gold_execution = [str(canonical_event_field_value(row, "execution_status") or "unknown") for row in subset]
    gold_binary = ["failure" if value == "failure" else "non_failure" for value in gold_execution]
    pred_binary = ["failure" if tool_output_looks_like_failure(pred) else "non_failure" for pred in predictions]
    gold_from_gold_text = ["failure" if tool_output_looks_like_failure(text) else "non_failure" for text in gold_texts]
    metrics["derived_execution_status"] = {
        "model": classification_metrics(gold_binary, pred_binary),
        # Same derivation applied to the GOLD observation: the ceiling this rule can reach,
        # i.e. how much of any gap is the rule rather than the world model. The gold label is
        # LLM-annotated semantics ("did this action fail") while the rule is a textual
        # error-marker check, so this ceiling is well below 1.0 and the composed `model`
        # number mixes two error sources.
        "derivation_ceiling": classification_metrics(gold_binary, gold_from_gold_text),
        # Model error ISOLATED from rule error: does the prediction carry the same
        # rule-detectable failure signal as the gold observation? Reference is the rule applied
        # to gold text, not the annotated label, so a weak rule no longer masks the model.
        "model_vs_derivation": classification_metrics(gold_from_gold_text, pred_binary),
        "majority_baseline": classification_metrics(
            gold_binary, [Counter(gold_binary).most_common(1)[0][0]] * len(gold_binary)
        ),
    }

    # ---- layer 3: annotated 11 fields -------------------------------------------------
    annotated_model: list[dict[str, Any] | None] = []
    annotated_gold: list[dict[str, Any] | None] = []
    if args.annotator and args.annotator.lower() != "none":
        print(f"[annotate] labeling {len(subset)} predicted observations with {args.annotator}", flush=True)
        annotated_model = annotate_observations(
            subset, predictions, args.annotator, args.annotator_workers, args.annotator_max_field_chars
        )
        if not args.skip_annotator_ceiling:
            print(f"[annotate] labeling {len(subset)} GOLD observations (annotator ceiling)", flush=True)
            annotated_gold = annotate_observations(
                subset, gold_texts, args.annotator, args.annotator_workers, args.annotator_max_field_chars
            )
        per_field: dict[str, Any] = {}
        for field in ALL_FIELDS:
            multi = field in NUDGE_MULTI_LABEL_FIELDS
            gold_values = [canonical_event_field_value(row, field) for row in subset]
            keep = [i for i, value in enumerate(gold_values) if value is not None]
            entry = {
                "model": classification_metrics(
                    [gold_values[i] for i in keep],
                    [label_field(annotated_model[i], field) for i in keep],
                    multi_label=multi,
                )
            }
            if annotated_gold:
                entry["annotator_ceiling"] = classification_metrics(
                    [gold_values[i] for i in keep],
                    [label_field(annotated_gold[i], field) for i in keep],
                    multi_label=multi,
                )
            majority = Counter(
                tuple(sorted(v)) if isinstance(v, list) else str(v) for v in (gold_values[i] for i in keep)
            ).most_common(1)
            if majority:
                top = majority[0][0]
                entry["majority_baseline"] = classification_metrics(
                    [gold_values[i] for i in keep],
                    [list(top) if isinstance(top, tuple) else top] * len(keep),
                    multi_label=multi,
                )
            per_field[field] = entry
        metrics["annotated"] = {
            "annotator": args.annotator,
            "annotated_rows": sum(1 for label in annotated_model if label),
            "annotation_failures": sum(1 for label in annotated_model if not label),
            "per_field": per_field,
            "accuracy": {f: per_field[f]["model"]["accuracy"] for f in ALL_FIELDS},
            "macro_f1": {f: per_field[f]["model"]["macro_f1"] for f in ALL_FIELDS},
            "macro_recall": {f: per_field[f]["model"]["macro_recall"] for f in ALL_FIELDS},
        }

    print_report(metrics, args)
    output = args.output or (Path(args.world_model_path) / "llm_world_model_field_eval.json")
    dump_json(output, metrics)
    print(f"\n[done] metrics written to {output}", flush=True)

    if args.dump_predictions:
        with Path(args.dump_predictions).open("w", encoding="utf-8") as handle:
            for position, row in enumerate(subset):
                handle.write(json.dumps({
                    "trajectory_id": row.get("trajectory_id"),
                    "interaction_index": row.get("interaction_index"),
                    "benchmark": row.get("benchmark"),
                    "action": render_action(row.get("action")),
                    "predicted_observation": predictions[position],
                    "gold_observation": gold_texts[position],
                    "gold_fields": {f: canonical_event_field_value(row, f) for f in ALL_FIELDS},
                    "annotated_prediction": annotated_model[position] if annotated_model else None,
                    "annotated_gold": annotated_gold[position] if annotated_gold else None,
                }, ensure_ascii=False) + "\n")
        print(f"[done] per-row dump written to {args.dump_predictions}", flush=True)


def print_report(metrics: dict[str, Any], args: argparse.Namespace) -> None:
    text = metrics["text"]
    print(f"\n=== layer 1: predicted tool output vs gold observation ({metrics['evaluated_rows']} rows) ===")
    print(f"  exact_match={text['exact_match']:.4f}  token_f1={text['token_f1']:.4f} "
          f"(P={text['token_precision']:.4f} R={text['token_recall']:.4f})")
    print(f"  empty predictions={text['empty_predictions']}  "
          f"mean chars pred/gold={text['mean_predicted_chars']:.0f}/{text['mean_gold_chars']:.0f}")

    derived = metrics["derived_execution_status"]
    print("\n=== layer 2: execution_status (failure vs non_failure), no annotator ===")
    print(f"  {'':22s} {'accuracy':>9s} {'macro_f1':>9s} {'failure recall':>15s}")
    for label in ("model", "derivation_ceiling", "model_vs_derivation", "majority_baseline"):
        if label not in derived:
            continue
        entry = derived[label]
        failure_recall = entry["per_class"].get("failure", {}).get("recall", 0.0)
        print(f"  {label:22s} {entry['accuracy']:9.4f} {entry['macro_f1']:9.4f} {failure_recall:15.4f}")

    annotated = metrics.get("annotated")
    if not annotated:
        print("\n(layer 3 skipped: pass --annotator <llm method> for the 11-field comparison)")
        return
    jepa = {}
    if args.jepa_metrics and Path(args.jepa_metrics).is_file():
        with Path(args.jepa_metrics).open() as handle:
            jepa = (json.load(handle).get("eval_metrics") or {})
    print(f"\n=== layer 3: 11 canonical-event fields, annotated by {annotated['annotator']} "
          f"({annotated['annotated_rows']} labeled, {annotated['annotation_failures']} failed) ===")
    header = f"  {'field':30s} {'LLM-WM acc':>11s} {'LLM-WM mF1':>11s} {'ceiling acc':>12s} {'major. acc':>11s}"
    if jepa:
        header += f" {'JEPA acc':>9s} {'JEPA mF1':>9s}"
    print(header)
    for field in ALL_FIELDS:
        entry = annotated["per_field"][field]
        model = entry["model"]
        ceiling = entry.get("annotator_ceiling", {})
        major = entry.get("majority_baseline", {})
        line = (f"  {field:30s} {model['accuracy']:11.4f} {model['macro_f1']:11.4f} "
                f"{ceiling.get('accuracy', float('nan')):12.4f} {major.get('accuracy', float('nan')):11.4f}")
        if jepa:
            line += (f" {(jepa.get('accuracy') or {}).get(field, float('nan')):9.4f}"
                     f" {(jepa.get('macro_f1') or {}).get(field, float('nan')):9.4f}")
        print(line)
    accs = list(annotated["accuracy"].values())
    f1s = list(annotated["macro_f1"].values())
    print(f"  {'MEAN':30s} {sum(accs)/len(accs):11.4f} {sum(f1s)/len(f1s):11.4f}")
    print("\nRead the LLM-WM column against `ceiling acc` (a perfect world model scored through the "
          "same annotator), not against 1.0. The JEPA columns are direct field predictions and pay "
          "no annotation cost.")


if __name__ == "__main__":
    main()
