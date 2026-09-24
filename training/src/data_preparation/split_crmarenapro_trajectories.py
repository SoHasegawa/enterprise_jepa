import argparse
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GPT55_SOURCE_ROOT = Path(
    "/data/Trajectory/"
    "bm-crmarenapro_ex-baseline_crm_agent_tg-sample_ts-0_2134_cf-f9299b609828_"
    "us-user_rn-20260519T050759Z-f9299b609828/trajectories"
)
DEFAULT_GPT51_SOURCE_ROOT = Path(
    os.environ.get("BENCHMARK_RESULT_ROOT", "results/experiments") + "/"
    "bm-crmarenapro_ex-baseline_crm_agent_tg-sample_ts-all_cf-d696d56c6e95_"
    "us-user_rn-20260701T014852Z-d696d56c6e95/trajectories"
)
DEFAULT_OUTPUT_PATH = ROOT / "trajectories" / "crmarenapro_multi_model_trajectories.jsonl"
DEFAULT_TRAIN_OUTPUT_PATH = ROOT / "trajectories" / "crmarenapro_multi_model_train_trajectories.jsonl"
DEFAULT_TEST_OUTPUT_PATH = ROOT / "trajectories" / "crmarenapro_multi_model_test_trajectories.jsonl"
DEFAULT_SPLIT_MANIFEST_PATH = ROOT / "trajectories" / "crmarenapro_multi_model_trajectory_split_manifest.json"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    model: str
    root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create task-disjoint train/test splits for CRMArenaPro trajectories, stratified by "
            "task outcome, task metadata, observed turn type, and CRM query success/failure counts."
        )
    )
    parser.add_argument("--gpt55-source-root", type=Path, default=DEFAULT_GPT55_SOURCE_ROOT)
    parser.add_argument("--gpt51-source-root", type=Path, default=DEFAULT_GPT51_SOURCE_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--train-output-path", type=Path, default=DEFAULT_TRAIN_OUTPUT_PATH)
    parser.add_argument("--test-output-path", type=Path, default=DEFAULT_TEST_OUTPUT_PATH)
    parser.add_argument("--generated-split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--test-task-count",
        type=int,
        default=None,
        help="Exact number of unique CRMArenaPro tasks to place in test. Overrides --train-ratio.",
    )
    parser.add_argument(
        "--max-records-per-source",
        type=int,
        default=None,
        help="Optional cap per source for smoke tests.",
    )
    parser.add_argument(
        "--allow-missing-outcomes",
        action="store_true",
        help="Keep trajectories whose task has no evaluator result and label them outcome=unknown.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return records


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def dump_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def experiment_root_for(source_root: Path) -> Path:
    return source_root.parent if source_root.name == "trajectories" else source_root


def task_id_aliases(task_id: str) -> set[str]:
    return {task_id, str(int(task_id)) if task_id.isdigit() else task_id}


def normalize_success(value: Any, score: Any = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(score, (int, float)):
        return score > 0
    if isinstance(value, (int, float)):
        return value > 0
    return None


def build_result_lookup(detail: dict[str, Any], manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    result_sources = []
    if isinstance(detail.get("results"), list):
        result_sources.extend(("detail.results", item) for item in detail["results"])
    task_results = ((manifest.get("eval_result") or {}).get("task_results") or [])
    result_sources.extend(("manifest.eval_result.task_results", item) for item in task_results)

    for source_name, item in result_sources:
        task_id = str(item.get("task_idx") or item.get("task_id") or "").strip()
        if not task_id:
            continue
        dataset_reference = item.get("dataset_reference") or {}
        score = item.get("crm_reward", item.get("score"))
        success = normalize_success(item.get("success"), score)
        if success is None:
            success = normalize_success(item.get("score"))
        normalized = {
            "task_id": task_id,
            "task_category": item.get("task_category"),
            "reward_metric": dataset_reference.get("reward_metric") or item.get("eval_func"),
            "dataset_source": dataset_reference.get("source"),
            "dataset_split": dataset_reference.get("split"),
            "dataset_idx": dataset_reference.get("idx"),
            "score": score,
            "total_score": item.get("total_score"),
            "success": success,
            "reason": item.get("reason"),
            "metrics": item.get("metrics") or {},
            "dimension_scores": item.get("dimension_scores") or ((item.get("entropic") or {}).get("dimension_scores") or {}),
            "result_source": source_name,
        }
        for alias in task_id_aliases(task_id):
            lookup.setdefault(alias, normalized)
    return lookup


def parse_task_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        if event.get("event_type") != "Message":
            continue
        payload = event.get("payload") or {}
        for part in payload.get("parts") or []:
            text = part.get("text")
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if parsed.get("type") == "crm_task" or parsed.get("task_id") is not None:
                return parsed
    return {}


def parse_answer_data(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("event_type") != "TaskArtifactUpdateEvent":
            continue
        artifact = ((event.get("event") or {}).get("artifact") or {})
        if artifact.get("name") != "Answer":
            continue
        for part in artifact.get("parts") or []:
            if part.get("kind") == "data" and isinstance(part.get("data"), dict):
                return part["data"]
    return {}


def normalize_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def merge_metrics(*sources: dict[str, Any]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for source in sources:
        for key in ("turns", "tokens", "tokens_estimate", "tool_calls", "queries", "failed_queries"):
            value = normalize_int(source.get(key)) if isinstance(source, dict) else None
            if value is not None and key not in merged:
                merged[key] = value
    return merged


def query_outcome_counts(metrics: dict[str, Any]) -> dict[str, int]:
    queries = max(0, normalize_int(metrics.get("queries")) or 0)
    failed_queries = max(0, normalize_int(metrics.get("failed_queries")) or 0)
    failed_queries = min(failed_queries, queries)
    return {
        "failure": failed_queries,
        "success": max(0, queries - failed_queries),
    }


def observed_turn_type(metrics: dict[str, Any]) -> str:
    turns = normalize_int(metrics.get("turns"))
    if turns == 1:
        return "single_turn"
    if turns is not None and turns > 1:
        return "multi_turn"
    return "unknown"


def collect_source_records(
    *,
    spec: SourceSpec,
    allow_missing_outcomes: bool,
    max_records: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    experiment_root = experiment_root_for(spec.root)
    detail_path = experiment_root / "detail.json"
    manifest_path = experiment_root / "manifest.json"
    detail = load_json(detail_path) if detail_path.exists() else {}
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    result_lookup = build_result_lookup(detail, manifest)
    source_files = sorted(spec.root.glob("*.jsonl"), key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)

    records = []
    skipped = []
    for source_path in source_files:
        if max_records is not None and len(records) >= max_records:
            break
        task_stem = source_path.stem
        try:
            events = load_jsonl(source_path)
        except Exception as exc:
            skipped.append({"source_path": str(source_path), "reason": str(exc)})
            continue

        task_metadata = parse_task_metadata(events)
        answer_data = parse_answer_data(events)
        task_id = str(task_metadata.get("task_id") or task_stem)
        outcome = None
        for alias in task_id_aliases(task_id):
            outcome = result_lookup.get(alias)
            if outcome is not None:
                break
        if outcome is None and not allow_missing_outcomes:
            skipped.append({"source_path": str(source_path), "reason": "missing evaluator result"})
            continue

        outcome = outcome or {
            "task_id": task_id,
            "task_category": task_metadata.get("task_category"),
            "reward_metric": None,
            "dataset_source": None,
            "dataset_split": None,
            "dataset_idx": task_id,
            "score": None,
            "total_score": None,
            "success": None,
            "reason": None,
            "metrics": {},
            "dimension_scores": {},
            "result_source": None,
        }
        answer_metrics = answer_data.get("metrics") if isinstance(answer_data.get("metrics"), dict) else {}
        metrics = merge_metrics(answer_metrics, outcome.get("metrics") or {})
        entropy = task_metadata.get("entropy") or {}
        config = task_metadata.get("config") or {}
        task_category = task_metadata.get("task_category") or outcome.get("task_category") or answer_data.get("category")
        canonical_task_id = str(outcome.get("task_id") or task_id)
        records.append(
            {
                "trajectory_id": f"crmarenapro-{spec.name}-{task_stem}",
                "benchmark": "CRMArenaPro",
                "source": "crmarenapro_a2a_jsonl",
                "source_model": spec.model,
                "source_variant": spec.name,
                "task_id": canonical_task_id,
                "task_stem": task_stem,
                "source_path": str(source_path),
                "task_category": task_category,
                "reward_metric": outcome.get("reward_metric"),
                "dataset_source": outcome.get("dataset_source"),
                "dataset_split": outcome.get("dataset_split"),
                "dataset_idx": outcome.get("dataset_idx"),
                "org_type": config.get("org_type"),
                "max_steps": config.get("max_steps"),
                "entropy": {
                    "drift_level": entropy.get("drift_level"),
                    "rot_level": entropy.get("rot_level"),
                    "drift_mappings": entropy.get("drift_mappings") or [],
                },
                "task_score": outcome.get("score"),
                "task_total_score": outcome.get("total_score"),
                "task_success": outcome.get("success"),
                "task_reason": outcome.get("reason"),
                "dimension_scores": outcome.get("dimension_scores") or {},
                "metrics": metrics,
                "observed_turn_type": observed_turn_type(metrics),
                "query_execution_counts": query_outcome_counts(metrics),
                "answer_data": answer_data,
                "task_metadata": task_metadata,
                "events": events,
            }
        )

    detail_task_ids = [str(item.get("task_idx")) for item in detail.get("results") or [] if item.get("task_idx") is not None]
    manifest_task_ids = [str(item) for item in manifest.get("task_ids") or []]
    file_task_ids = {path.stem for path in source_files}
    metadata = {
        "source_root": str(spec.root),
        "experiment_root": str(experiment_root),
        "detail_path": str(detail_path) if detail_path.exists() else None,
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "benchmark_name": manifest.get("benchmark_name"),
        "benchmark_version": manifest.get("benchmark_version"),
        "run_id": manifest.get("run_id"),
        "score_summary": manifest.get("score_summary") or detail.get("summary"),
        "trajectory_file_count": len(source_files),
        "collected_trajectory_count": len(records),
        "detail_result_count": len(detail_task_ids),
        "manifest_task_id_count": len(manifest_task_ids),
        "missing_trajectory_task_ids": [task_id for task_id in detail_task_ids if task_id not in file_task_ids],
    }
    return records, skipped, metadata


def add_counts(target: dict[str, int], counts: dict[str, int]) -> dict[str, int]:
    for key, value in counts.items():
        target[key] = target.get(key, 0) + value
    return target


def feature_counts_for_record(record: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    def inc(name: str, value: Any, amount: int = 1) -> None:
        normalized = "unknown" if value is None or value == "" else str(value)
        counts[f"{name}={normalized}"] = counts.get(f"{name}={normalized}", 0) + amount

    if record.get("task_success") is True:
        inc("task_outcome", "success")
    elif record.get("task_success") is False:
        inc("task_outcome", "failure")
    else:
        inc("task_outcome", "unknown")
    inc("task_category", record.get("task_category"))
    inc("reward_metric", record.get("reward_metric"))
    inc("dataset_split", record.get("dataset_split"))
    inc("org_type", record.get("org_type"))
    inc("observed_turn_type", record.get("observed_turn_type"))
    entropy = record.get("entropy") or {}
    inc("drift_level", entropy.get("drift_level"))
    inc("rot_level", entropy.get("rot_level"))
    inc("source_variant", record.get("source_variant"))
    for label, amount in (record.get("query_execution_counts") or {}).items():
        if amount:
            inc("query_outcome", label, amount)
    return counts


def task_feature_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        add_counts(counts, feature_counts_for_record(record))
    return counts


def count_error(selected: dict[str, int], target: dict[str, float], weights: dict[str, float]) -> float:
    keys = set(selected) | set(target)
    error = 0.0
    for key in keys:
        diff = selected.get(key, 0) - target.get(key, 0.0)
        error += weights.get(key, 1.0) * diff * diff
    return error


def feature_weight(key: str) -> float:
    if key.startswith("task_outcome="):
        return 6.0
    if key.startswith("task_category="):
        return 5.0
    if key.startswith("reward_metric="):
        return 3.0
    if key.startswith("observed_turn_type="):
        return 3.0
    if key.startswith("query_outcome="):
        return 4.0
    if key.startswith("source_variant="):
        return 2.0
    return 1.0


def select_test_task_ids(
    *,
    grouped_records: dict[str, list[dict[str, Any]]],
    test_task_count: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    task_counts = {task_id: task_feature_counts(records) for task_id, records in grouped_records.items()}
    total_counts: dict[str, int] = {}
    for counts in task_counts.values():
        add_counts(total_counts, counts)

    total_tasks = len(grouped_records)
    target_counts = {key: value * test_task_count / total_tasks for key, value in total_counts.items()}
    weights = {key: feature_weight(key) for key in total_counts}

    remaining = list(task_counts)
    rng.shuffle(remaining)
    selected = []
    selected_counts: dict[str, int] = {}
    while len(selected) < test_task_count:
        best_task_id = None
        best_score = None
        for task_id in remaining:
            candidate_counts = dict(selected_counts)
            add_counts(candidate_counts, task_counts[task_id])
            score = count_error(candidate_counts, target_counts, weights)
            ranking = (score, task_id)
            if best_score is None or ranking < best_score:
                best_score = ranking
                best_task_id = task_id
        if best_task_id is None:
            raise ValueError("Could not select enough test tasks")
        selected.append(best_task_id)
        add_counts(selected_counts, task_counts[best_task_id])
        remaining.remove(best_task_id)
    return sorted(selected, key=lambda task_id: int(task_id) if task_id.isdigit() else task_id)


def count_task_outcomes(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        if record.get("task_success") is True:
            counts["success"] += 1
        elif record.get("task_success") is False:
            counts["failure"] += 1
        else:
            counts["unknown"] += 1
    return dict(sorted(counts.items()))


def count_query_outcomes(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"failure": 0, "success": 0}
    for record in records:
        for key, value in (record.get("query_execution_counts") or {}).items():
            counts[key] = counts.get(key, 0) + value
    return {key: value for key, value in counts.items() if value}


def count_field(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(record.get(field) if record.get(field) is not None else "unknown") for record in records)
    return dict(sorted(counts.items()))


def count_entropy(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        entropy = record.get("entropy") or {}
        counts[f"drift={entropy.get('drift_level') or 'unknown'}|rot={entropy.get('rot_level') or 'unknown'}"] += 1
    return dict(sorted(counts.items()))


def count_by_source(records: list[dict[str, Any]], counter_fn) -> dict[str, dict[str, int]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["source_variant"]].append(record)
    return {source: counter_fn(items) for source, items in sorted(grouped.items())}


def validate_split(train_records: list[dict[str, Any]], test_records: list[dict[str, Any]]) -> None:
    train_task_ids = {record["task_id"] for record in train_records}
    test_task_ids = {record["task_id"] for record in test_records}
    overlap = train_task_ids & test_task_ids
    if overlap:
        preview = ", ".join(sorted(overlap)[:10])
        raise ValueError(f"Train/test task overlap detected: {preview}")


def sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda item: (int(item["task_id"]) if item["task_id"].isdigit() else item["task_id"], item["source_variant"]))


def main() -> None:
    args = parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")

    source_specs = [
        SourceSpec("gpt-5.5-baseline-crm-agent", "GPT-5.5", args.gpt55_source_root),
        SourceSpec("gpt-5.1-baseline-crm-agent", "GPT-5.1", args.gpt51_source_root),
    ]

    records = []
    skipped_records = []
    source_metadata = {}
    for spec in source_specs:
        source_records, source_skipped, metadata = collect_source_records(
            spec=spec,
            allow_missing_outcomes=args.allow_missing_outcomes,
            max_records=args.max_records_per_source,
        )
        records.extend(source_records)
        skipped_records.extend({"source_variant": spec.name, **item} for item in source_skipped)
        source_metadata[spec.name] = metadata

    if not records:
        raise ValueError("No CRMArenaPro trajectories were collected.")

    grouped_records = defaultdict(list)
    for record in records:
        grouped_records[record["task_id"]].append(record)

    task_ids = sorted(grouped_records, key=lambda task_id: int(task_id) if task_id.isdigit() else task_id)
    if args.test_task_count is None:
        test_task_count = max(1, round(len(task_ids) * (1 - args.train_ratio)))
    else:
        test_task_count = args.test_task_count
    if not 0 < test_task_count < len(task_ids):
        raise ValueError(f"--test-task-count must be between 1 and {len(task_ids) - 1}; got {test_task_count}")

    test_task_ids = set(
        select_test_task_ids(
            grouped_records=grouped_records,
            test_task_count=test_task_count,
            seed=args.seed,
        )
    )
    train_task_ids = set(task_ids) - test_task_ids

    train_records = []
    test_records = []
    for record in records:
        split = "test" if record["task_id"] in test_task_ids else "train"
        record["split"] = split
        if split == "test":
            test_records.append(record)
        else:
            train_records.append(record)

    validate_split(train_records, test_records)

    dump_jsonl(args.output_path, sorted_records(records))
    dump_jsonl(args.train_output_path, sorted_records(train_records))
    dump_jsonl(args.test_output_path, sorted_records(test_records))

    manifest = {
        "benchmark": "CRMArenaPro",
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "split_unit": "task_id",
        "stratification_targets": [
            "task_outcome_success_failure",
            "task_category",
            "reward_metric",
            "dataset_split",
            "org_type",
            "entropy_drift_rot_level",
            "observed_turn_type",
            "source_variant",
            "crm_query_success_failure_counts",
        ],
        "stratification_method": "greedy task-level selection matching weighted aggregate metadata and query outcome counts",
        "trajectory_count": len(records),
        "task_id_count": len(task_ids),
        "train_trajectories": len(train_records),
        "test_trajectories": len(test_records),
        "train_task_count": len(train_task_ids),
        "test_task_count": len(test_task_ids),
        "task_ids": task_ids,
        "train_task_ids": sorted(train_task_ids, key=lambda task_id: int(task_id) if task_id.isdigit() else task_id),
        "test_task_ids": sorted(test_task_ids, key=lambda task_id: int(task_id) if task_id.isdigit() else task_id),
        "total_task_outcome_counts": count_task_outcomes(records),
        "train_task_outcome_counts": count_task_outcomes(train_records),
        "test_task_outcome_counts": count_task_outcomes(test_records),
        "total_query_outcome_counts": count_query_outcomes(records),
        "train_query_outcome_counts": count_query_outcomes(train_records),
        "test_query_outcome_counts": count_query_outcomes(test_records),
        "total_source_task_outcome_counts": count_by_source(records, count_task_outcomes),
        "train_source_task_outcome_counts": count_by_source(train_records, count_task_outcomes),
        "test_source_task_outcome_counts": count_by_source(test_records, count_task_outcomes),
        "total_source_query_outcome_counts": count_by_source(records, count_query_outcomes),
        "train_source_query_outcome_counts": count_by_source(train_records, count_query_outcomes),
        "test_source_query_outcome_counts": count_by_source(test_records, count_query_outcomes),
        "total_task_category_counts": count_field(records, "task_category"),
        "train_task_category_counts": count_field(train_records, "task_category"),
        "test_task_category_counts": count_field(test_records, "task_category"),
        "total_reward_metric_counts": count_field(records, "reward_metric"),
        "train_reward_metric_counts": count_field(train_records, "reward_metric"),
        "test_reward_metric_counts": count_field(test_records, "reward_metric"),
        "total_observed_turn_type_counts": count_field(records, "observed_turn_type"),
        "train_observed_turn_type_counts": count_field(train_records, "observed_turn_type"),
        "test_observed_turn_type_counts": count_field(test_records, "observed_turn_type"),
        "total_entropy_counts": count_entropy(records),
        "train_entropy_counts": count_entropy(train_records),
        "test_entropy_counts": count_entropy(test_records),
        "source_metadata": source_metadata,
        "skipped_records": skipped_records,
    }
    dump_json(args.generated_split_manifest, manifest)

    print(f"Wrote {len(records)} trajectories to {args.output_path}")
    print(f"Wrote {len(train_records)} train trajectories to {args.train_output_path}")
    print(f"Wrote {len(test_records)} test trajectories to {args.test_output_path}")
    print(f"Wrote split manifest to {args.generated_split_manifest}")
    print(f"Unique train tasks: {len(train_task_ids)}")
    print(f"Unique test tasks: {len(test_task_ids)}")
    print(f"Total task outcomes: {manifest['total_task_outcome_counts']}")
    print(f"Train task outcomes: {manifest['train_task_outcome_counts']}")
    print(f"Test task outcomes: {manifest['test_task_outcome_counts']}")
    print(f"Total query outcomes: {manifest['total_query_outcome_counts']}")
    print(f"Train query outcomes: {manifest['train_query_outcome_counts']}")
    print(f"Test query outcomes: {manifest['test_query_outcome_counts']}")
    print(f"Skipped records: {len(skipped_records)}")


if __name__ == "__main__":
    main()
