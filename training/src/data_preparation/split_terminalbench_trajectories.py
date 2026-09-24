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
    "bm-Terminal-Bench-2_0_ex-llm_shell_tg-all_ts-all_cf-77cb399f613c_"
    "us-user_rn-20260519T220446Z-77cb399f613c/trajectories"
)
DEFAULT_GPT51_SOURCE_ROOT = Path(
    os.environ.get("BENCHMARK_RESULT_ROOT", "results/experiments") + "/"
    "bm-Terminal-Bench-2_0_ex-llm_shell_tg-all_ts-all_cf-77cb399f613c_"
    "us-user_rn-20260624T214539Z-77cb399f613c/trajectories"
)
DEFAULT_OUTPUT_PATH = ROOT / "trajectories" / "terminalbench_2_0_multi_model_trajectories.jsonl"
DEFAULT_TRAIN_OUTPUT_PATH = ROOT / "trajectories" / "terminalbench_2_0_multi_model_train_trajectories.jsonl"
DEFAULT_TEST_OUTPUT_PATH = ROOT / "trajectories" / "terminalbench_2_0_multi_model_test_trajectories.jsonl"
DEFAULT_SPLIT_MANIFEST_PATH = (
    ROOT / "trajectories" / "terminalbench_2_0_multi_model_trajectory_split_manifest.json"
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    model: str
    root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create task-disjoint train/test splits for Terminal-Bench 2.0 trajectories, "
            "balanced by per-command shell success/failure labels."
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
        help="Exact number of unique Terminal-Bench tasks to place in test. Overrides --train-ratio.",
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
        help="Keep trajectories whose task has no reward in detail.json and label them outcome=unknown.",
    )
    parser.add_argument(
        "--strict-terminalbench-protocol",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files that do not contain Terminal-Bench shell protocol events.",
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
    return {
        task_id,
        task_id.replace("_", "."),
        task_id.replace(".", "_"),
    }


def build_reward_lookup(detail_path: Path) -> dict[str, dict[str, Any]]:
    detail = load_json(detail_path)
    raw_rewards = detail.get("task_rewards") or {}
    lookup = {}
    for task_id, reward_record in raw_rewards.items():
        reward = reward_record.get("reward")
        normalized = {
            "task_id": task_id,
            "reward": reward,
            "success": bool(isinstance(reward, (int, float)) and reward > 0),
            "error": reward_record.get("error"),
        }
        for alias in task_id_aliases(task_id):
            lookup[alias] = normalized

    eval_result = detail.get("eval_result") or {}
    for task_result in eval_result.get("task_results") or []:
        task_id = task_result.get("task_id")
        if not task_id:
            continue
        score = task_result.get("score")
        normalized = {
            "task_id": task_id,
            "reward": score,
            "success": bool(isinstance(score, (int, float)) and score > 0),
            "error": None if task_result.get("reason") == "passed" else task_result.get("reason"),
        }
        for alias in task_id_aliases(task_id):
            lookup.setdefault(alias, normalized)
    return lookup


def normalize_command_label(exit_code: Any) -> str | None:
    if isinstance(exit_code, bool):
        return None
    if isinstance(exit_code, int):
        return "success" if exit_code == 0 else "failure"
    if isinstance(exit_code, str):
        stripped = exit_code.strip()
        if stripped.lstrip("-").isdigit():
            return "success" if int(stripped) == 0 else "failure"
    return None


def command_results_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shell_results = [
        event
        for event in events
        if event.get("event_type") == "ShellProtocolExecResult" and event.get("direction") == "green"
    ]
    if not shell_results:
        shell_results = [
            event
            for event in events
            if event.get("event_type") == "ShellProtocolExecResult"
        ]

    results = []
    for event in shell_results:
        payload = event.get("payload") or {}
        exit_code = event.get("exit_code", payload.get("exit_code"))
        label = normalize_command_label(exit_code)
        if label is None:
            continue
        results.append(
            {
                "sequence": event.get("sequence"),
                "exit_code": exit_code,
                "label": label,
            }
        )
    return results


def count_command_results(command_results: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(result["label"] for result in command_results)
    return {"failure": counts.get("failure", 0), "success": counts.get("success", 0)}


def is_terminalbench_trajectory(events: list[dict[str, Any]]) -> bool:
    for event in events:
        payload = event.get("payload") or {}
        if payload.get("protocol") == "terminal-bench-shell-v1":
            return True

        source = event.get("source") or {}
        if source.get("format") == "terminal-bench-shell-v1":
            return True

        if event.get("event_type") != "TaskArtifactUpdateEvent":
            continue
        artifact = ((event.get("event") or {}).get("artifact") or {})
        for part in artifact.get("parts") or []:
            text = part.get("text")
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if parsed.get("format") == "terminal-bench-shell-v1":
                return True
    return False


def collect_source_records(
    *,
    spec: SourceSpec,
    allow_missing_outcomes: bool,
    strict_terminalbench_protocol: bool,
    max_records: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    experiment_root = experiment_root_for(spec.root)
    detail_path = experiment_root / "detail.json"
    manifest_path = experiment_root / "manifest.json"
    reward_lookup = build_reward_lookup(detail_path) if detail_path.exists() else {}
    detail = load_json(detail_path) if detail_path.exists() else {}
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    source_files = sorted(spec.root.glob("*.jsonl"))

    records = []
    skipped = []
    for source_path in source_files:
        if max_records is not None and len(records) >= max_records:
            break

        task_id = source_path.stem
        try:
            events = load_jsonl(source_path)
        except Exception as exc:
            skipped.append({"source_path": str(source_path), "reason": str(exc)})
            continue

        if strict_terminalbench_protocol and not is_terminalbench_trajectory(events):
            skipped.append(
                {
                    "source_path": str(source_path),
                    "reason": "not a terminal-bench-shell-v1 trajectory",
                }
            )
            continue

        outcome = None
        for alias in task_id_aliases(task_id):
            outcome = reward_lookup.get(alias)
            if outcome is not None:
                break
        if outcome is None and not allow_missing_outcomes:
            skipped.append({"source_path": str(source_path), "reason": "missing reward in detail.json"})
            continue

        command_results = command_results_from_events(events)
        if not command_results:
            skipped.append({"source_path": str(source_path), "reason": "no shell exec_result events"})
            continue

        outcome = outcome or {"task_id": task_id, "reward": None, "success": None, "error": None}
        canonical_task_id = outcome.get("task_id") or task_id
        records.append(
            {
                "trajectory_id": f"terminalbench-2.0-{spec.name}-{task_id}",
                "benchmark": "Terminal-Bench-2.0",
                "source": "terminal_bench_shell_jsonl",
                "source_model": spec.model,
                "source_variant": spec.name,
                "task_id": canonical_task_id,
                "task_stem": task_id,
                "source_path": str(source_path),
                "task_reward": outcome.get("reward"),
                "task_success": outcome.get("success"),
                "task_error": outcome.get("error"),
                "command_execution_results": command_results,
                "command_outcome_counts": count_command_results(command_results),
                "events": events,
            }
        )

    seen_task_aliases = {alias for path in source_files for alias in task_id_aliases(path.stem)}
    detail_task_ids = detail.get("task_ids") or manifest.get("task_ids") or []
    metadata = {
        "source_root": str(spec.root),
        "experiment_root": str(experiment_root),
        "detail_path": str(detail_path) if detail_path.exists() else None,
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "benchmark_name": detail.get("benchmark_name") or manifest.get("benchmark_name"),
        "run_id": detail.get("run_id") or manifest.get("run_id"),
        "total_tasks": detail.get("total_tasks") or manifest.get("total_tasks"),
        "total_score": detail.get("total_score"),
        "score_rate": detail.get("score_rate"),
        "num_passed": detail.get("num_passed"),
        "trajectory_file_count": len(source_files),
        "collected_trajectory_count": len(records),
        "missing_trajectory_task_ids": [
            task_id
            for task_id in detail_task_ids
            if not task_id_aliases(task_id) & seen_task_aliases
        ],
    }
    return records, skipped, metadata


def add_command_counts(target: dict[str, int], counts: dict[str, int]) -> dict[str, int]:
    target["failure"] = target.get("failure", 0) + counts.get("failure", 0)
    target["success"] = target.get("success", 0) + counts.get("success", 0)
    return target


def record_command_counts(record: dict[str, Any]) -> dict[str, int]:
    return record.get("command_outcome_counts") or {"failure": 0, "success": 0}


def task_command_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"failure": 0, "success": 0}
    for record in records:
        add_command_counts(counts, record_command_counts(record))
    return counts


def command_count_error(selected: dict[str, int], target: dict[str, float]) -> float:
    return sum((selected.get(label, 0) - target.get(label, 0.0)) ** 2 for label in ("failure", "success"))


def select_test_task_ids(
    *,
    grouped_records: dict[str, list[dict[str, Any]]],
    test_task_count: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    task_counts = {
        task_id: task_command_counts(records)
        for task_id, records in grouped_records.items()
    }
    total_counts = {"failure": 0, "success": 0}
    for counts in task_counts.values():
        add_command_counts(total_counts, counts)

    total_tasks = len(grouped_records)
    target_counts = {
        label: count * test_task_count / total_tasks
        for label, count in total_counts.items()
    }

    remaining = list(task_counts)
    rng.shuffle(remaining)
    selected = []
    selected_counts = {"failure": 0, "success": 0}
    while len(selected) < test_task_count:
        best_task_id = None
        best_score = None
        for task_id in remaining:
            candidate_counts = dict(selected_counts)
            add_command_counts(candidate_counts, task_counts[task_id])
            score = command_count_error(candidate_counts, target_counts)
            tie_breaker = task_id
            ranking = (score, tie_breaker)
            if best_score is None or ranking < best_score:
                best_score = ranking
                best_task_id = task_id
        if best_task_id is None:
            raise ValueError("Could not select enough test tasks")
        selected.append(best_task_id)
        add_command_counts(selected_counts, task_counts[best_task_id])
        remaining.remove(best_task_id)
    return sorted(selected)


def count_command_outcomes(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"failure": 0, "success": 0}
    for record in records:
        add_command_counts(counts, record_command_counts(record))
    return {key: value for key, value in counts.items() if value}


def count_by_model_and_command_outcome(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["source_variant"]].append(record)
    return {source_variant: count_command_outcomes(items) for source_variant, items in sorted(grouped.items())}


def count_task_rewards(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        if record.get("task_success") is True:
            counts["success"] += 1
        elif record.get("task_success") is False:
            counts["failure"] += 1
        else:
            counts["unknown"] += 1
    return dict(sorted(counts.items()))


def validate_split(train_records: list[dict[str, Any]], test_records: list[dict[str, Any]]) -> None:
    train_task_ids = {record["task_id"] for record in train_records}
    test_task_ids = {record["task_id"] for record in test_records}
    overlap = train_task_ids & test_task_ids
    if overlap:
        preview = ", ".join(sorted(overlap)[:10])
        raise ValueError(f"Train/test task overlap detected: {preview}")


def main() -> None:
    args = parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")

    source_specs = [
        SourceSpec("gpt-5.5-llm-shell", "GPT-5.5", args.gpt55_source_root),
        SourceSpec("gpt-5.1-llm-shell", "GPT-5.1", args.gpt51_source_root),
    ]

    records = []
    skipped_records = []
    source_metadata = {}
    for spec in source_specs:
        source_records, source_skipped, metadata = collect_source_records(
            spec=spec,
            allow_missing_outcomes=args.allow_missing_outcomes,
            strict_terminalbench_protocol=args.strict_terminalbench_protocol,
            max_records=args.max_records_per_source,
        )
        records.extend(source_records)
        skipped_records.extend({"source_variant": spec.name, **item} for item in source_skipped)
        source_metadata[spec.name] = metadata

    if not records:
        raise ValueError("No Terminal-Bench trajectories were collected.")

    grouped_records = defaultdict(list)
    for record in records:
        grouped_records[record["task_id"]].append(record)

    task_ids = sorted(grouped_records)
    if args.test_task_count is None:
        test_task_count = max(1, round(len(task_ids) * (1 - args.train_ratio)))
    else:
        test_task_count = args.test_task_count
    if not 0 < test_task_count < len(task_ids):
        raise ValueError(
            f"--test-task-count must be between 1 and {len(task_ids) - 1}; got {test_task_count}"
        )

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

    dump_jsonl(args.output_path, sorted(records, key=lambda item: (item["task_id"], item["source_variant"])))
    dump_jsonl(args.train_output_path, sorted(train_records, key=lambda item: (item["task_id"], item["source_variant"])))
    dump_jsonl(args.test_output_path, sorted(test_records, key=lambda item: (item["task_id"], item["source_variant"])))

    manifest = {
        "benchmark": "Terminal-Bench-2.0",
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "split_unit": "task_id",
        "stratification_target": "shell_command_exit_code",
        "stratification_labels": {"success": "exit_code == 0", "failure": "exit_code != 0"},
        "stratification_method": "greedy task-level selection matching aggregate command success/failure counts",
        "trajectory_count": len(records),
        "task_id_count": len(task_ids),
        "train_trajectories": len(train_records),
        "test_trajectories": len(test_records),
        "train_task_count": len(train_task_ids),
        "test_task_count": len(test_task_ids),
        "task_ids": task_ids,
        "train_task_ids": sorted(train_task_ids),
        "test_task_ids": sorted(test_task_ids),
        "total_command_outcome_counts": count_command_outcomes(records),
        "train_command_outcome_counts": count_command_outcomes(train_records),
        "test_command_outcome_counts": count_command_outcomes(test_records),
        "total_model_command_outcome_counts": count_by_model_and_command_outcome(records),
        "train_model_command_outcome_counts": count_by_model_and_command_outcome(train_records),
        "test_model_command_outcome_counts": count_by_model_and_command_outcome(test_records),
        "total_task_reward_counts": count_task_rewards(records),
        "train_task_reward_counts": count_task_rewards(train_records),
        "test_task_reward_counts": count_task_rewards(test_records),
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
    print(f"Total command outcomes: {manifest['total_command_outcome_counts']}")
    print(f"Train command outcomes: {manifest['train_command_outcome_counts']}")
    print(f"Test command outcomes: {manifest['test_command_outcome_counts']}")
    print(f"Skipped records: {len(skipped_records)}")


if __name__ == "__main__":
    main()
