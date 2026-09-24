import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps the script usable in minimal envs.
    def tqdm(iterable, **kwargs):
        return iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.generate_enterpriseops_gym_stateful_samples import (  # noqa: E402
    extract_action_batches,
    load_json,
    load_jsonl,
)
from src.generation.generate_enterpriseops_gym_world_model_trajectories import (  # noqa: E402
    add_tool_counts,
    assign_canonical_task_indices,
    build_enterpriseops_split_groups,
    build_fallback_seed,
    build_planned_state_message,
    build_trajectory_stage_labels,
    canonical_task_key,
    canonical_task_key_from_parts,
    count_enterpriseops_tool_execution_results,
    derive_gym_task_config_name,
    domain_from_source_path,
    dump_json,
    dump_records,
    empty_tool_execution_result_counts,
    extract_system_and_user_messages,
    flush_stage_cache,
    initialize_llm,
    load_enterpriseops_gym_seed_lookup,
    load_stage_cache,
    lookup_seed_for_source_file,
    make_default_tool_context,
    parse_domain_filter,
    repair_explicit_tool_error_labels,
    serialize_tool_execution_result_counts,
    summarize_tool_batch,
    summarize_tool_batch_with_semantic_stagnation,
)
from src.data_preparation.world_model_trajectory_cleanup import cleanup_world_model_trajectory  # noqa: E402


DEFAULT_JSONL_SOURCE_DIR = Path(
    "/data/Trajectory/"
    "bm-EnterpriseOps-Gym_ex-mcp_react_tg-hf_dataset_ts-all_cf-cfff05db2ef8_"
    "us-user_rn-20260513T044856Z-cfff05db2ef8/trajectories"
)
DEFAULT_GPT_SOURCE_ROOT = Path("/data/Trajectory/user_enterpriseops_gym/gpt-5")
DEFAULT_QWEN_SOURCE_ROOT = Path("/data/Trajectory/user_enterpriseops_gym/qwen3")
DEFAULT_SEEDS_PATH = ROOT / "trajectories" / "imported_benchmark_seeds.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "trajectories" / "enterpriseops_gym_multi_model_world_model_trajectories.json"
DEFAULT_TRAIN_OUTPUT_PATH = ROOT / "trajectories" / "enterpriseops_gym_multi_model_world_model_train_trajectories.json"
DEFAULT_TEST_OUTPUT_PATH = ROOT / "trajectories" / "enterpriseops_gym_multi_model_world_model_test_trajectories.json"
DEFAULT_SPLIT_MANIFEST_PATH = (
    ROOT / "trajectories" / "enterpriseops_gym_multi_model_world_model_trajectory_split_manifest.json"
)
DEFAULT_STAGE_CACHE = ROOT / "trajectories" / "enterpriseops_gym_world_model_stage_cache.json"
JSONL_SUFFIX = ".jsonl"


SOURCE_MODEL_SPECS = {
    "gpt-5.5": {
        "source": "enterprise_ops_gym_mcp_react_jsonl",
        "source_variant": "gpt-5.5-mcp-react-jsonl",
        "source_model": "GPT-5.5",
    },
    "gpt-5.1": {
        "source": "enterprise_ops_gym_oracle_run",
        "source_variant": "gpt-5.1-oracle-json",
        "source_model": "GPT-5.1",
    },
    "qwen3-8b": {
        "source": "enterprise_ops_gym_oracle_run",
        "source_variant": "qwen3-8b-oracle-json",
        "source_model": "Qwen3-8B",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate combined EnterpriseOps-Gym world-model train/test trajectories "
            "from GPT-5.5 JSONL logs, GPT-5.1 JSON runs, and Qwen3-8B JSON runs."
        )
    )
    parser.add_argument("--jsonl-source-dir", type=Path, default=DEFAULT_JSONL_SOURCE_DIR)
    parser.add_argument("--gpt-source-root", type=Path, default=DEFAULT_GPT_SOURCE_ROOT)
    parser.add_argument("--qwen-source-root", type=Path, default=DEFAULT_QWEN_SOURCE_ROOT)
    parser.add_argument("--seeds-path", type=Path, default=DEFAULT_SEEDS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--train-output-path", type=Path, default=DEFAULT_TRAIN_OUTPUT_PATH)
    parser.add_argument("--test-output-path", type=Path, default=DEFAULT_TEST_OUTPUT_PATH)
    parser.add_argument("--generated-split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH)
    parser.add_argument("--stage-cache", type=Path, default=DEFAULT_STAGE_CACHE)
    parser.add_argument(
        "--llm-method",
        default="qwen3-vllm-9000",
        help="Method passed to src.llm.LLM for stage generation.",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl"],
        default="json",
        help="Serialize trajectories as a JSON list or JSONL.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="",
        help="Optional comma-separated domain filter, for example `csm,itsm,teams`.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-task-count",
        type=int,
        default=200,
        help="Exact number of unique EnterpriseOps-Gym tasks to place in the test split.",
    )
    parser.add_argument(
        "--max-records-per-source",
        type=int,
        default=None,
        help="Optional cap per source set for smoke tests.",
    )
    parser.add_argument(
        "--allow-stage-fallback",
        action="store_true",
        help="Use heuristic stage labels if the configured LLM call fails.",
    )
    parser.add_argument(
        "--heuristic-stage-labels",
        action="store_true",
        help="Skip LLM stage generation and use deterministic heuristic stage labels.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any selected source file cannot be reconstructed.",
    )
    return parser.parse_args()


def task_id_from_source_file(source_file: Path) -> str:
    return canonical_task_key_from_parts(source_file.stem)


def build_trajectory_id(source_path: Path, source_variant: str) -> str:
    return f"enterpriseops-gym-world-model-{source_variant}-{source_path.stem}"


def source_files_under(root: Path, suffix: str) -> list[Path]:
    return sorted(path for path in root.glob(f"**/*{suffix}") if path.is_file())


def first_json_object_from_artifact_text(value: str | None) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_request_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        if event.get("event_type") != "Message":
            continue
        payload = event.get("payload") or {}
        for part in payload.get("parts") or []:
            text = part.get("text")
            parsed = first_json_object_from_artifact_text(text)
            if parsed is not None:
                return parsed
    return {}


def extract_jsonl_internal_record(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in events:
        if event.get("event_type") == "PurpleInternalRecord":
            payload = event.get("payload")
            if isinstance(payload, dict):
                return payload

    for event in events:
        artifact = ((event.get("event") or {}).get("artifact") or {})
        if artifact.get("name") != "internal_trajectory":
            continue
        for part in artifact.get("parts") or []:
            parsed = first_json_object_from_artifact_text(part.get("text"))
            if isinstance(parsed, dict):
                payload = parsed.get("payload")
                if isinstance(payload, dict):
                    records = payload.get("records") or []
                    if records and isinstance(records[0], dict):
                        return records[0]
                return parsed
    return None


def seed_metadata_for_file(
    *,
    source_file: Path,
    seed_lookup: dict[str, Any],
    domain: str | None,
) -> tuple[dict[str, Any], int]:
    seed, task_index = lookup_seed_for_source_file(source_file, seed_lookup)
    if seed is None:
        seed = build_fallback_seed(source_file)
        seed["environment"]["domain"] = domain or seed["environment"].get("domain") or domain_from_source_path(source_file)
        task_index = -1
    elif domain and not (seed.get("environment") or {}).get("domain"):
        seed.setdefault("environment", {})["domain"] = domain
    return seed, task_index


def reconstruct_from_conversation_flow(
    *,
    source_path: Path,
    conversation_flow: list[dict[str, Any]],
    seed: dict[str, Any],
    task_index: int,
    source_spec: dict[str, str],
    gym_task_config_name: str | None,
    llm,
    use_heuristic_stage_fallback: bool,
    stage_cache,
    cache_path: Path,
    cache_tracker: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    system_prompt, user_prompt = extract_system_and_user_messages(conversation_flow)
    action_batches = extract_action_batches(conversation_flow)
    if not action_batches:
        raise ValueError("no reconstructable ai_message batches")

    stage_labels = build_trajectory_stage_labels(
        llm=llm,
        stage_cache=stage_cache,
        cache_path=cache_path,
        cache_tracker=cache_tracker,
        args=args,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        action_batches=action_batches,
        use_heuristic_stage_fallback=use_heuristic_stage_fallback,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    previous_process_state = None
    latest_tool_context = make_default_tool_context()
    total_steps = len(action_batches)

    for step_index, batch in enumerate(action_batches, start=1):
        action_content = batch["action_content"]
        if batch["tool_results"]:
            if use_heuristic_stage_fallback:
                latest_tool_context = summarize_tool_batch(batch["tool_results"])
            else:
                latest_tool_context = summarize_tool_batch_with_semantic_stagnation(
                    llm=llm,
                    stage_cache=stage_cache,
                    cache_path=cache_path,
                    cache_tracker=cache_tracker,
                    args=args,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    action_content=action_content,
                    tool_results=batch["tool_results"],
                )

        messages.append({"role": "action", "content": action_content})
        state_message, previous_process_state = build_planned_state_message(
            system_prompt=system_prompt,
            action_content=action_content,
            latest_tool_context=latest_tool_context,
            previous_process_state=previous_process_state,
            stage_labels=stage_labels,
            step_index=step_index,
            total_steps=total_steps,
        )
        messages.append(state_message)

    task_key = task_id_from_source_file(source_path)
    trajectory = {
        "trajectory_id": build_trajectory_id(source_path, source_spec["source_variant"]),
        "source": source_spec["source"],
        "source_model": source_spec["source_model"],
        "domain": seed["environment"]["domain"],
        "seed_id": seed["seed_id"],
        "task_index": task_index,
        "task_key": task_key,
        "task_stem": task_key,
        "gym_task_config_name": gym_task_config_name,
        "coordination_pattern": seed["coordination_pattern"],
        "seed_source_path": seed["source_record"]["source_path"],
        "source_path": str(source_path),
        "source_variant": source_spec["source_variant"],
        "messages": messages,
    }
    return repair_explicit_tool_error_labels(cleanup_world_model_trajectory(trajectory))


def reconstruct_json_trajectory(
    *,
    source_file: Path,
    source_spec: dict[str, str],
    seed_lookup: dict[str, Any],
    llm,
    use_heuristic_stage_fallback: bool,
    stage_cache,
    cache_path: Path,
    cache_tracker: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_payload = load_json(source_file)
    runs = raw_payload.get("runs") or []
    if not runs:
        raise ValueError("missing runs")

    domain = domain_from_source_path(source_file)
    seed, task_index = seed_metadata_for_file(
        source_file=source_file,
        seed_lookup=seed_lookup,
        domain=domain,
    )
    return reconstruct_from_conversation_flow(
        source_path=source_file,
        conversation_flow=runs[0]["conversation_flow"],
        seed=seed,
        task_index=task_index,
        source_spec=source_spec,
        gym_task_config_name=derive_gym_task_config_name(source_file),
        llm=llm,
        use_heuristic_stage_fallback=use_heuristic_stage_fallback,
        stage_cache=stage_cache,
        cache_path=cache_path,
        cache_tracker=cache_tracker,
        args=args,
    )


def reconstruct_jsonl_trajectory(
    *,
    source_file: Path,
    source_spec: dict[str, str],
    seed_lookup: dict[str, Any],
    llm,
    use_heuristic_stage_fallback: bool,
    stage_cache,
    cache_path: Path,
    cache_tracker: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    events = load_jsonl(source_file)
    request_metadata = extract_request_metadata(events)
    domain = request_metadata.get("domain") or domain_from_source_path(source_file)
    seed, task_index = seed_metadata_for_file(
        source_file=source_file,
        seed_lookup=seed_lookup,
        domain=domain,
    )
    internal_record = extract_jsonl_internal_record(events)
    if internal_record is None:
        raise ValueError("missing PurpleInternalRecord/internal_trajectory")
    conversation_flow = internal_record.get("conversation_flow") or []
    if not conversation_flow:
        raise ValueError("missing conversation_flow")

    return reconstruct_from_conversation_flow(
        source_path=source_file,
        conversation_flow=conversation_flow,
        seed=seed,
        task_index=task_index,
        source_spec=source_spec,
        gym_task_config_name=f"oracle__{domain}__{source_file.stem}.json",
        llm=llm,
        use_heuristic_stage_fallback=use_heuristic_stage_fallback,
        stage_cache=stage_cache,
        cache_path=args.stage_cache,
        cache_tracker=cache_tracker,
        args=args,
    )


def build_task_split_manifest(
    *,
    trajectories: list[dict[str, Any]],
    args: argparse.Namespace,
    source_roots: list[Path],
) -> dict[str, Any]:
    grouped_trajectories: dict[str, list[dict[str, Any]]] = {}
    grouped_source_variants: dict[str, list[str]] = {}
    grouped_task_indices: dict[str, list[int]] = {}
    grouped_task_stems: dict[str, list[str]] = {}

    for trajectory in trajectories:
        group_key = canonical_task_key(trajectory)
        grouped_trajectories.setdefault(group_key, []).append(trajectory)
        grouped_source_variants.setdefault(group_key, [])
        grouped_task_indices.setdefault(group_key, [])
        grouped_task_stems.setdefault(group_key, [])

        task_index = trajectory["task_index"]
        if task_index not in grouped_task_indices[group_key]:
            grouped_task_indices[group_key].append(task_index)
        task_stem = trajectory.get("task_stem")
        if task_stem and task_stem not in grouped_task_stems[group_key]:
            grouped_task_stems[group_key].append(task_stem)
        source_variant = trajectory.get("source_variant")
        if source_variant and source_variant not in grouped_source_variants[group_key]:
            grouped_source_variants[group_key].append(source_variant)

    ordered_group_keys = sorted(grouped_trajectories)
    if args.test_task_count > len(ordered_group_keys):
        raise ValueError(
            f"Requested {args.test_task_count} test tasks, but only "
            f"{len(ordered_group_keys)} unique EnterpriseOps-Gym tasks are available."
        )

    split_index_for_group = {
        group_key: split_index
        for split_index, group_key in enumerate(ordered_group_keys)
    }
    primary_trajectories = []
    additional_variant_records = []
    for group_key in ordered_group_keys:
        task_trajectories = grouped_trajectories[group_key]
        primary_trajectories.append(task_trajectories[0])
        for trajectory in task_trajectories[1:]:
            additional_variant_records.append(
                {
                    "task_index": split_index_for_group[group_key],
                    "trajectory": trajectory,
                }
            )

    split_group_stats = build_enterpriseops_split_groups(
        primary_trajectories=primary_trajectories,
        additional_variant_records=additional_variant_records,
    )
    random_generator = random.Random(args.seed)
    shuffled_indices = list(range(len(ordered_group_keys)))
    random_generator.shuffle(shuffled_indices)
    test_indices = sorted(shuffled_indices[: args.test_task_count])
    test_index_set = set(test_indices)
    train_indices = sorted(index for index in range(len(ordered_group_keys)) if index not in test_index_set)

    train_label_counts = empty_tool_execution_result_counts()
    test_label_counts = empty_tool_execution_result_counts()
    train_record_count = 0
    test_record_count = 0
    for group in split_group_stats["groups"]:
        if group["split_index"] in test_index_set:
            add_tool_counts(test_label_counts, group["label_counts"])
            test_record_count += group["record_count"]
        else:
            add_tool_counts(train_label_counts, group["label_counts"])
            train_record_count += group["record_count"]

    split_manifest = {
        "data_path": str(source_roots[0].resolve()),
        "source_roots": [str(source_root.resolve()) for source_root in source_roots],
        "seeds_path": str(args.seeds_path.resolve()),
        "seed": args.seed,
        "train_ratio": (len(ordered_group_keys) - args.test_task_count) / len(ordered_group_keys),
        "split_unit": "canonical_task_key",
        "trajectory_count": len(ordered_group_keys),
        "task_key_count": len(ordered_group_keys),
        "test_task_count": args.test_task_count,
        "train_trajectories": len(train_indices),
        "test_trajectories": len(test_indices),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "task_keys": ordered_group_keys,
        "train_task_keys": [ordered_group_keys[index] for index in train_indices],
        "test_task_keys": [ordered_group_keys[index] for index in test_indices],
        "task_indices": sorted(
            {task_index for task_indices in grouped_task_indices.values() for task_index in task_indices}
        ),
        "train_task_indices": sorted(
            {
                task_index
                for split_index in train_indices
                for task_index in grouped_task_indices[ordered_group_keys[split_index]]
            }
        ),
        "test_task_indices": sorted(
            {
                task_index
                for split_index in test_indices
                for task_index in grouped_task_indices[ordered_group_keys[split_index]]
            }
        ),
        "stratification_target": "last_tool_execution_result",
        "stratification_labels": [-1, 0, 1],
        "total_label_counts": serialize_tool_execution_result_counts(split_group_stats["total_label_counts"]),
        "train_label_counts": serialize_tool_execution_result_counts(train_label_counts),
        "test_label_counts": serialize_tool_execution_result_counts(test_label_counts),
        "total_record_count": split_group_stats["total_record_count"],
        "train_record_count": train_record_count,
        "test_record_count": test_record_count,
        "grouped_batch_records": {
            str(key): value for key, value in sorted(split_group_stats["grouped_batch_records"].items())
        },
        "unmatched_batch_records": split_group_stats["unmatched_batch_records"],
        "task_key_source_variants": {
            group_key: grouped_source_variants[group_key]
            for group_key in ordered_group_keys
        },
        "task_key_stems": {
            group_key: grouped_task_stems[group_key]
            for group_key in ordered_group_keys
        },
        "multi_variant_task_count": sum(
            len(grouped_trajectories[group_key]) > 1
            for group_key in ordered_group_keys
        ),
    }
    return split_manifest


def collect_source_set(
    *,
    name: str,
    source_files: list[Path],
    source_spec: dict[str, str],
    seed_lookup: dict[str, Any],
    domain_filter: set[str],
    llm,
    use_heuristic_stage_fallback: bool,
    stage_cache,
    cache_path: Path,
    cache_tracker: dict[str, int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    trajectories = []
    skipped_records = []

    progress = tqdm(
        source_files,
        desc=f"Processing {name}",
        unit="file",
        dynamic_ncols=True,
    )
    for source_file in progress:
        if args.max_records_per_source is not None and len(trajectories) >= args.max_records_per_source:
            break

        if (
            source_file.suffix != JSONL_SUFFIX
            and domain_filter
            and domain_from_source_path(source_file) not in domain_filter
        ):
            continue

        try:
            if source_file.suffix == JSONL_SUFFIX:
                trajectory = reconstruct_jsonl_trajectory(
                    source_file=source_file,
                    source_spec=source_spec,
                    seed_lookup=seed_lookup,
                    llm=llm,
                    use_heuristic_stage_fallback=use_heuristic_stage_fallback,
                    stage_cache=stage_cache,
                    cache_path=cache_path,
                    cache_tracker=cache_tracker,
                    args=args,
                )
            else:
                trajectory = reconstruct_json_trajectory(
                    source_file=source_file,
                    source_spec=source_spec,
                    seed_lookup=seed_lookup,
                    llm=llm,
                    use_heuristic_stage_fallback=use_heuristic_stage_fallback,
                    stage_cache=stage_cache,
                    cache_path=cache_path,
                    cache_tracker=cache_tracker,
                    args=args,
                )
        except Exception as exc:
            skipped_records.append(
                {
                    "source_set": name,
                    "source_path": str(source_file),
                    "reason": str(exc),
                }
            )
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(done=len(trajectories), skipped=len(skipped_records))
            continue

        if domain_filter and str(trajectory.get("domain", "")).lower() not in domain_filter:
            continue
        trajectories.append(trajectory)
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(done=len(trajectories), skipped=len(skipped_records))

    return trajectories, skipped_records


def main() -> None:
    args = parse_args()
    domain_filter = parse_domain_filter(args.domains)
    seed_lookup = load_enterpriseops_gym_seed_lookup(args.seeds_path, domain_filter)

    if args.heuristic_stage_labels:
        llm = None
        llm_fallback_reason = "heuristic stage labels requested"
    else:
        llm, llm_fallback_reason = initialize_llm(args)
    stage_cache = load_stage_cache(args.stage_cache)
    cache_tracker = {"pending_writes": 0, "flush_every": 25}
    use_heuristic_stage_fallback = args.heuristic_stage_labels or bool(llm_fallback_reason)

    source_sets = [
        (
            "gpt-5.5-jsonl",
            source_files_under(args.jsonl_source_dir, JSONL_SUFFIX),
            SOURCE_MODEL_SPECS["gpt-5.5"],
        ),
        (
            "gpt-5.1-json",
            source_files_under(args.gpt_source_root, ".json"),
            SOURCE_MODEL_SPECS["gpt-5.1"],
        ),
        (
            "qwen3-8b-json",
            source_files_under(args.qwen_source_root, ".json"),
            SOURCE_MODEL_SPECS["qwen3-8b"],
        ),
    ]

    trajectories: list[dict[str, Any]] = []
    skipped_records: list[dict[str, str]] = []
    source_file_counts = {}
    for name, source_files, source_spec in source_sets:
        source_file_counts[name] = len(source_files)
        source_trajectories, source_skipped_records = collect_source_set(
            name=name,
            source_files=source_files,
            source_spec=source_spec,
            seed_lookup=seed_lookup,
            domain_filter=domain_filter,
            llm=llm,
            use_heuristic_stage_fallback=use_heuristic_stage_fallback,
            stage_cache=stage_cache,
            cache_path=args.stage_cache,
            cache_tracker=cache_tracker,
            args=args,
        )
        trajectories.extend(source_trajectories)
        skipped_records.extend(source_skipped_records)

    flush_stage_cache(stage_cache, args.stage_cache, cache_tracker, force=True)

    if args.strict and skipped_records:
        preview = "\n".join(item["source_path"] for item in skipped_records[:10])
        raise ValueError(f"Found non-reconstructable trajectories:\n{preview}")
    if not trajectories:
        raise ValueError("No EnterpriseOps-Gym trajectories were reconstructed.")

    assign_canonical_task_indices(trajectories)
    source_roots = [args.jsonl_source_dir, args.gpt_source_root, args.qwen_source_root]
    split_manifest = build_task_split_manifest(
        trajectories=trajectories,
        args=args,
        source_roots=source_roots,
    )
    train_task_keys = set(split_manifest["train_task_keys"])
    train_payload = []
    test_payload = []
    for trajectory in trajectories:
        split = "train" if canonical_task_key(trajectory) in train_task_keys else "test"
        trajectory["split"] = split
        if split == "train":
            train_payload.append(trajectory)
        else:
            test_payload.append(trajectory)

    from src.generation.generate_enterpriseops_gym_world_model_trajectories import (  # noqa: E402
        validate_disjoint_enterpriseops_task_splits,
    )

    validate_disjoint_enterpriseops_task_splits(
        train_payload,
        test_payload,
        expected_test_task_count=args.test_task_count,
    )

    dump_records(args.output_path, trajectories, args.output_format)
    dump_records(args.train_output_path, train_payload, args.output_format)
    dump_records(args.test_output_path, test_payload, args.output_format)
    dump_json(
        args.generated_split_manifest,
        {
            **split_manifest,
            "combined_record_count": len(trajectories),
            "train_record_count": len(train_payload),
            "test_record_count": len(test_payload),
            "source_file_counts": source_file_counts,
            "skipped_records": skipped_records,
            "llm_method": args.llm_method,
            "source_models": {
                name: spec["source_model"]
                for name, _, spec in source_sets
            },
        },
    )

    domain_counts = Counter(item["domain"] for item in trajectories)
    source_variant_counts = Counter(item["source_variant"] for item in trajectories)
    train_source_variant_counts = Counter(item["source_variant"] for item in train_payload)
    test_source_variant_counts = Counter(item["source_variant"] for item in test_payload)
    train_label_counts = empty_tool_execution_result_counts()
    test_label_counts = empty_tool_execution_result_counts()
    for trajectory in train_payload:
        add_tool_counts(train_label_counts, count_enterpriseops_tool_execution_results(trajectory))
    for trajectory in test_payload:
        add_tool_counts(test_label_counts, count_enterpriseops_tool_execution_results(trajectory))

    print(f"Wrote {len(trajectories)} trajectories to {args.output_path}")
    print(f"Wrote {len(train_payload)} train trajectories to {args.train_output_path}")
    print(f"Wrote {len(test_payload)} test trajectories to {args.test_output_path}")
    print(f"Wrote split manifest to {args.generated_split_manifest}")
    print(f"Unique train tasks: {len(set(split_manifest['train_task_keys']))}")
    print(f"Unique test tasks: {len(set(split_manifest['test_task_keys']))}")
    print(f"Skipped non-reconstructable files: {len(skipped_records)}")
    if llm_fallback_reason:
        print(f"Stage generation fallback reason: {llm_fallback_reason}")
    print(f"Domain counts: {dict(sorted(domain_counts.items()))}")
    print(f"Source variant counts: {dict(sorted(source_variant_counts.items()))}")
    print(f"Train source variant counts: {dict(sorted(train_source_variant_counts.items()))}")
    print(f"Test source variant counts: {dict(sorted(test_source_variant_counts.items()))}")
    print(f"Train label counts: {serialize_tool_execution_result_counts(train_label_counts)}")
    print(f"Test label counts: {serialize_tool_execution_result_counts(test_label_counts)}")
    if skipped_records:
        print("Sample skipped records:")
        for item in skipped_records[:5]:
            print(f"  {item['source_path']}: {item['reason']}")


if __name__ == "__main__":
    main()
