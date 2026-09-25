#!/usr/bin/env python3
"""Per-class and macro P/R/F1 for canonical-event heads, computed from per-example dumps.

The metrics files written during training keep only MARGINAL class distributions (how often
each category is predicted vs how often it appears in the gold labels). Per-class precision /
recall / F1 and confusion matrices need the JOINT (gold, predicted) pairs, which marginals
cannot recover -- so they used to require another forward pass over the eval split. Runs with
`--canonical-event-dump-predictions` now store those pairs, and this script turns them into any
metric on demand, on CPU, in seconds.

Pass a `.jsonl` dump to compute metrics from the stored (gold, predicted) pairs, or a `.json`
metrics file (`evaluation_metrics_canonical_event.json` from the causal-LM eval, or a JEPA
`canonical_event_training_metrics.json`) to reuse metrics that were already computed with the
same definitions -- useful for a checkpoint whose predictions were not dumped.

Compare several checkpoints by passing one label=path pair each:

    uv run python src/analysis/canonical_event_prediction_report.py \
        ensemble=checkpoints/data_jepa_heads_ensemble/eval_predictions_per_example.jsonl \
        tool_swe=checkpoints/data_jepa_heads_ensemble_tool_swe_all/eval_predictions_per_example.jsonl \
        imb_state=checkpoints/data_jepa_heads_ensemble_tool_swe_all_imb_state/eval_predictions_per_example.jsonl

Macro averages cover only classes with gold support: a category that never occurs in the eval
split would otherwise contribute a hard 0 and make the number a function of label-space size
rather than of prediction quality. Definitions match evaluate_canonical_event_heads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


def field_metrics(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    """Accuracy, per-class P/R/F1 and macro/weighted averages for one field."""
    support: dict[str, int] = {}
    predicted: dict[str, int] = {}
    true_positive: dict[str, int] = {}
    exact = 0
    for gold_value, pred_value in pairs:
        gold_set, pred_set = as_set(gold_value), as_set(pred_value)
        exact += int(gold_set == pred_set)
        for value in gold_set:
            support[value] = support.get(value, 0) + 1
        for value in pred_set:
            predicted[value] = predicted.get(value, 0) + 1
        for value in gold_set & pred_set:
            true_positive[value] = true_positive.get(value, 0) + 1

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    weighted_f1 = 0.0
    missed: list[str] = []
    for value in sorted(set(support) | set(predicted), key=lambda v: (-support.get(v, 0), v)):
        tp = true_positive.get(value, 0)
        pred_count = predicted.get(value, 0)
        gold_count = support.get(value, 0)
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / gold_count if gold_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[value] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": gold_count, "predicted": pred_count,
        }
        if gold_count:
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            weighted_f1 += f1 * gold_count
            if pred_count == 0:
                missed.append(value)
    total_support = sum(support.values())
    majority = max(support.values()) if support else 0
    return {
        "accuracy": exact / len(pairs) if pairs else 0.0,
        "majority_class_rate": majority / len(pairs) if pairs else 0.0,
        "macro_f1": (sum(f1s) / len(f1s)) if f1s else 0.0,
        "macro_precision": (sum(precisions) / len(precisions)) if precisions else 0.0,
        "macro_recall": (sum(recalls) / len(recalls)) if recalls else 0.0,
        "weighted_f1": (weighted_f1 / total_support) if total_support else 0.0,
        "classes_with_support": len(f1s),
        "classes_missed": missed,
        "per_class": per_class,
        "examples": len(pairs),
    }


def load_metrics_json(path: Path) -> dict[str, dict[str, Any]]:
    """Read already-computed per-field metrics instead of recomputing from pairs.

    Accepts either shape, since both are produced with the same definitions:
      * the causal-LM canonical-event eval (`evaluation_metrics_canonical_event.json`) ->
        top-level `per_field`;
      * a JEPA head run (`canonical_event_training_metrics.json`) -> `eval_metrics.per_head`.
    Lets a checkpoint join the comparison without another forward pass when its predictions
    were not dumped, at the cost of not being able to compute NEW metrics for it.
    """
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    fields = payload.get("per_field")
    if fields is None:
        fields = (payload.get("eval_metrics") or {}).get("per_head")
    if fields is None:
        raise SystemExit(f"{path}: no per_field / eval_metrics.per_head block found")
    normalized: dict[str, dict[str, Any]] = {}
    for field, entry in fields.items():
        normalized[field] = {
            "accuracy": entry.get("accuracy", entry.get("exact_set_match", 0.0)),
            "majority_class_rate": entry.get("majority_class_rate", float("nan")),
            "macro_f1": entry.get("macro_f1", 0.0),
            "macro_precision": entry.get("macro_precision", 0.0),
            "macro_recall": entry.get("macro_recall", 0.0),
            "weighted_f1": entry.get("weighted_f1", 0.0),
            "classes_with_support": entry.get("classes_with_support", 0),
            "classes_missed": entry.get("classes_missed", []),
            "per_class": entry.get("per_class", {}),
            "examples": payload.get("n", payload.get("eval_examples", 0)),
        }
    return normalized


def load_dump(path: Path) -> dict[str, list[tuple[Any, Any]]]:
    """field -> [(gold, predicted), ...] in dataset order."""
    pairs: dict[str, list[tuple[Any, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gold, pred = row.get("gold") or {}, row.get("pred") or {}
            for field in gold:
                pairs.setdefault(field, []).append((gold.get(field), pred.get(field)))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dumps", nargs="+", metavar="LABEL=PATH",
                        help="One or more per-example prediction dumps to compare.")
    parser.add_argument("--per-class-fields", default=None,
                        help="Comma-separated fields to print per-class tables for "
                             "(default: every field).")
    parser.add_argument("--min-support", type=int, default=1,
                        help="Hide per-class rows with gold support below this.")
    parser.add_argument("--exclude-classes", default=None,
                        help="Comma-separated class names to drop from the MACRO averages "
                             "(e.g. `unknown` -- an annotation fallback, not a real category).")
    parser.add_argument("--macro-min-support", type=int, default=1,
                        help="Drop classes with gold support below this from the macro averages.")
    parser.add_argument("--class-table", action="store_true",
                        help="Print one F1/Recall table covering every class of every field, "
                             "with per-field macro rows and an overall average row.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON with all metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs: dict[str, dict[str, list[tuple[Any, Any]]]] = {}
    precomputed: dict[str, dict[str, dict[str, Any]]] = {}
    ordered_labels: list[str] = []
    for spec in args.dumps:
        if "=" not in spec:
            raise SystemExit(f"expected LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        ordered_labels.append(label)
        dump_path = Path(path)
        if not dump_path.is_file():
            raise SystemExit(f"missing dump: {dump_path}")
        if dump_path.suffix == ".json":
            precomputed[label] = load_metrics_json(dump_path)
            rows = next(iter(precomputed[label].values()), {}).get("examples", 0)
            print(f"[load] {label}: precomputed metrics for {rows} examples from {dump_path}", flush=True)
        else:
            runs[label] = load_dump(dump_path)
            rows = len(next(iter(runs[label].values()), []))
            print(f"[load] {label}: {rows} examples from {dump_path}", flush=True)

    metrics = {label: {field: field_metrics(pairs) for field, pairs in fields.items()}
               for label, fields in runs.items()}
    metrics.update(precomputed)
    excluded = {name.strip() for name in (args.exclude_classes or "").split(",") if name.strip()}
    if excluded or args.macro_min_support > 1:
        # Recompute the macro averages from per_class under the exclusion rules. Applied
        # identically to every model, and to precomputed metrics too, so the comparison stays
        # like-for-like -- but note this CHANGES the metric definition: classes dropped here
        # (annotation fallbacks such as `unknown`, or classes too rare to estimate) no longer
        # contribute their usually-zero scores, so every number rises.
        dropped: dict[str, list[str]] = {}
        for label, fields_metrics in metrics.items():
            for field, entry in fields_metrics.items():
                kept_f1, kept_recall, kept_precision = [], [], []
                for klass, stats in (entry.get("per_class") or {}).items():
                    support = stats.get("support", 0)
                    if support < max(1, args.macro_min_support) or klass in excluded:
                        if support:
                            dropped.setdefault(field, [])
                            if klass not in dropped[field]:
                                dropped[field].append(klass)
                        continue
                    kept_f1.append(stats["f1"])
                    kept_recall.append(stats["recall"])
                    kept_precision.append(stats["precision"])
                entry["macro_f1"] = (sum(kept_f1) / len(kept_f1)) if kept_f1 else 0.0
                entry["macro_recall"] = (sum(kept_recall) / len(kept_recall)) if kept_recall else 0.0
                entry["macro_precision"] = (sum(kept_precision) / len(kept_precision)) if kept_precision else 0.0
                entry["classes_with_support"] = len(kept_f1)
        note = []
        if excluded:
            note.append(f"excluded classes: {', '.join(sorted(excluded))}")
        if args.macro_min_support > 1:
            note.append(f"support < {args.macro_min_support} dropped")
        print(f"[macro] {'; '.join(note)}")
        for field, names in sorted(dropped.items()):
            print(f"        {field}: dropped {', '.join(names)}")
    # Keep the order the labels were given in, not dump-before-metrics.
    metrics = {label: metrics[label] for label in ordered_labels if label in metrics}
    fields = sorted({field for label in metrics for field in metrics[label]})
    labels = list(metrics)

    for name, key in (("macro-F1", "macro_f1"), ("macro-recall", "macro_recall"),
                      ("macro-precision", "macro_precision"), ("accuracy", "accuracy")):
        print(f"\n=== {name} ===")
        header = f"{'field':30s}" + "".join(f"{label:>16s}" for label in labels)
        if len(labels) > 1:
            header += f"{'best':>12s}"
        print(header)
        for field in fields:
            values = [metrics[label].get(field, {}).get(key, float('nan')) for label in labels]
            line = f"{field:30s}" + "".join(f"{value:16.4f}" for value in values)
            if len(labels) > 1:
                best = max(range(len(values)), key=lambda i: values[i])
                line += f"{labels[best]:>12s}"
            print(line)
        means = [
            sum(metrics[label].get(f, {}).get(key, 0.0) for f in fields) / max(1, len(fields))
            for label in labels
        ]
        print(f"{'MEAN':30s}" + "".join(f"{value:16.4f}" for value in means))

    selected = (args.per_class_fields.split(",") if args.per_class_fields else fields)
    for field in selected:
        print(f"\n=== per-class: {field} ===")
        classes = sorted(
            {c for label in labels for c, stats in metrics[label].get(field, {}).get("per_class", {}).items()
             if stats["support"] >= args.min_support},
            key=lambda c: -max(metrics[label].get(field, {}).get("per_class", {}).get(c, {}).get("support", 0)
                               for label in labels),
        )
        print(f"  {'class':28s} {'support':>8s}" + "".join(f"{label[:14]:>16s}" for label in labels))
        print(f"  {'':28s} {'':>8s}" + "".join(f"{'P / R / F1':>16s}" for _ in labels))
        for klass in classes:
            support = max(metrics[label].get(field, {}).get("per_class", {}).get(klass, {}).get("support", 0)
                          for label in labels)
            cells = []
            for label in labels:
                stats = metrics[label].get(field, {}).get("per_class", {}).get(klass)
                cells.append("        -       " if stats is None else
                             f"{stats['precision']:.2f}/{stats['recall']:.2f}/{stats['f1']:.2f}".rjust(16))
            print(f"  {klass:28s} {support:8d}" + "".join(cells))
        for label in labels:
            missed = metrics[label].get(field, {}).get("classes_missed") or []
            if missed:
                print(f"  [{label}] never predicted (has support): {', '.join(missed)}")

    if args.class_table:
        print("\n=== every class: F1/Recall (support in parentheses) ===")
        width = 17
        print(f"  {'class':38s}" + "".join(f"{label[:width - 1]:>{width}s}" for label in labels))
        all_f1: dict[str, list[float]] = {label: [] for label in labels}
        all_recall: dict[str, list[float]] = {label: [] for label in labels}
        for field in fields:
            print(f"  {field}")
            classes = sorted(
                {c for label in labels
                 for c, stats in metrics[label].get(field, {}).get("per_class", {}).items()
                 if stats.get("support", 0) >= args.min_support},
                key=lambda c: -max(metrics[label].get(field, {}).get("per_class", {})
                                   .get(c, {}).get("support", 0) for label in labels),
            )
            field_f1: dict[str, list[float]] = {label: [] for label in labels}
            field_recall: dict[str, list[float]] = {label: [] for label in labels}
            for klass in classes:
                support = max(metrics[label].get(field, {}).get("per_class", {})
                              .get(klass, {}).get("support", 0) for label in labels)
                cells = []
                for label in labels:
                    stats = metrics[label].get(field, {}).get("per_class", {}).get(klass)
                    if stats is None:
                        cells.append(f"{'-':>{width}s}")
                        continue
                    field_f1[label].append(stats["f1"])
                    field_recall[label].append(stats["recall"])
                    all_f1[label].append(stats["f1"])
                    all_recall[label].append(stats["recall"])
                    cells.append(f"{stats['f1']:.2f}/{stats['recall']:.2f}".rjust(width))
                print(f"    {klass:32s}({support:>4d})" + "".join(cells))
            cells = [
                (f"{sum(field_f1[label]) / len(field_f1[label]):.2f}/"
                 f"{sum(field_recall[label]) / len(field_recall[label]):.2f}").rjust(width)
                if field_f1[label] else f"{'-':>{width}s}"
                for label in labels
            ]
            print(f"    {'-- macro over this field':32s}      " + "".join(cells))
        print()
        cells = [
            (f"{sum(all_f1[label]) / len(all_f1[label]):.3f}/"
             f"{sum(all_recall[label]) / len(all_recall[label]):.3f}").rjust(width)
            if all_f1[label] else f"{'-':>{width}s}"
            for label in labels
        ]
        print(f"  {'AVERAGE over all classes':38s}" + "".join(cells))
        cells = [
            (f"{sum(metrics[label].get(f, {}).get('macro_f1', 0.0) for f in fields) / len(fields):.3f}/"
             f"{sum(metrics[label].get(f, {}).get('macro_recall', 0.0) for f in fields) / len(fields):.3f}"
             ).rjust(width)
            for label in labels
        ]
        print(f"  {'AVERAGE of the 11 field macros':38s}" + "".join(cells))

    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=1, ensure_ascii=False)
        print(f"\n[done] metrics written to {args.output}")


if __name__ == "__main__":
    main()
