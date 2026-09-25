import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.generate_enterpriseops_gym_stateful_samples import (  # noqa: E402
    extract_action_batches,
    extract_tool_payload,
    load_json,
    load_jsonl,
)
from src.generation.generate_world_model_trajectories import (  # noqa: E402
    action_uses_tool,
    cache_key,
    classify_action_execution_result,
    empty_tool_execution_result_counts,
    extract_agent_role,
    flush_stage_cache,
    is_explicit_tool_error,
    load_stage_cache,
    make_default_tool_context,
    prepare_llm_environment,
    serialize_tool_execution_result_counts,
    summarize_action_for_stage,
    summarize_text,
)
from src.data_preparation.world_model_trajectory_cleanup import cleanup_world_model_trajectory  # noqa: E402


DEFAULT_SEEDS_PATH = ROOT / "trajectories" / "imported_benchmark_seeds.jsonl"
DEFAULT_SOURCE_ROOT = Path("/data/user/enterprisegym/results/react/gpt-5/teams/oracle/run_1")
DEFAULT_QWEN3_SOURCE_ROOT = Path("/data/user/enterprisegym/results/react/qwen3/teams/oracle/run_1")
DEFAULT_USER_SOURCE_ROOT = Path("/data/Trajectory/user_enterpriseops_gym")
DEFAULT_EXISTING_TRAJECTORY_PATH = (
    ROOT / "trajectories" / "enterpriseops_gym_world_model_mcp_react_trajectories.json"
)
DEFAULT_OUTPUT_PATH = ROOT / "trajectories" / "enterpriseops_gym_world_model_trajectories.json"
DEFAULT_TRAIN_OUTPUT_PATH = ROOT / "trajectories" / "enterpriseops_gym_world_model_train_trajectories.json"
DEFAULT_TEST_OUTPUT_PATH = ROOT / "trajectories" / "enterpriseops_gym_world_model_test_trajectories.json"
DEFAULT_SPLIT_MANIFEST_PATH = ROOT / "trajectories" / "enterpriseops_gym_world_model_trajectory_split_manifest.json"
DEFAULT_STAGE_CACHE = ROOT / "trajectories" / "enterpriseops_gym_world_model_stage_cache.json"
TRAJECTORY_STAGE_PLAN_SCHEMA = {
    "stage_labels": [""],
}
SEMANTIC_STAGNATION_JUDGE_SCHEMA = {
    "semantic_stagnation": False,
    "reason": "",
}
SEMANTIC_STAGNATION_TRIGGER_MARKERS = (
    "404",
    "not found",
    "page not found",
    "does not exist",
    "no such",
    "unable to",
    "cannot ",
    "can't ",
    "could not",
    "no result",
    "no results",
    "no matching",
    "no record",
    "no records",
    "no file",
    "no files",
    "missing",
    "invalid",
    "forbidden",
    "unauthorized",
    "permission denied",
    "access denied",
    "unavailable",
    "unsupported",
    "already exists",
    "duplicate",
)
SEMANTIC_STAGNATION_STATUS_RE = re.compile(r"\b(?:4\d{2}|5\d{2})\b")
TASK_ID_RE = re.compile(r"(task_\d{8}_\d{6}_\d{3}_[0-9a-f]+_[0-9a-f]+)")
RESULTS_STEM_RE = re.compile(r"^results_[^_]+__([^_]+)__")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate EnterpriseOps-Gym trajectories in the lean world-model format "
            "defined in instructions/TRAJECTORIES.md, with three-class tool execution labels."
        )
    )
    parser.add_argument("--seeds-path", type=Path, default=DEFAULT_SEEDS_PATH)
    parser.add_argument(
        "--source-root",
        dest="source_roots",
        action="append",
        type=Path,
        default=None,
        help=(
            "EnterpriseOps-Gym source root. Pass multiple times to include multiple model runs. "
            "Defaults to the gpt-5/qwen3 oracle run_1 roots and /data/Trajectory/user_enterpriseops_gym."
        ),
    )
    parser.add_argument(
        "--existing-trajectory-path",
        dest="existing_trajectory_paths",
        action="append",
        type=Path,
        default=None,
        help=(
            "Prebuilt world-model trajectory JSON/JSONL to include before splitting. "
            "Defaults to the EnterpriseOps-Gym MCP React trajectories when present."
        ),
    )
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
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on the number of matched run files to convert.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument(
        "--test-task-count",
        type=int,
        default=200,
        help="Exact number of unique EnterpriseOps-Gym tasks to place in the test split.",
    )
    parser.add_argument(
        "--reuse-split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_PATH,
        help="Reuse an existing EnterpriseOps-Gym split manifest when it matches the reconstructed label counts.",
    )
    parser.add_argument(
        "--force-new-split",
        action="store_true",
        help="Ignore any existing EnterpriseOps-Gym split manifest and generate a fresh stratified split.",
    )
    parser.add_argument(
        "--max-history-actions",
        type=int,
        default=8,
        help="Number of recent actions to expose to the stage generator.",
    )
    parser.add_argument(
        "--allow-stage-fallback",
        action="store_true",
        help="Use a heuristic stage fallback if the configured LLM call fails.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any JSON file in the selected source roots is unmatched or not reconstructable.",
    )
    return parser.parse_args()


def normalize_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def parse_domain_filter(raw_value: str) -> set[str]:
    if not raw_value.strip():
        return set()
    return {item.strip().lower() for item in raw_value.split(",") if item.strip()}


def resolve_source_roots(raw_source_roots: list[Path] | None) -> list[Path]:
    source_roots = raw_source_roots or [
        DEFAULT_SOURCE_ROOT,
        DEFAULT_QWEN3_SOURCE_ROOT,
        DEFAULT_USER_SOURCE_ROOT,
    ]
    resolved_roots = []
    seen = set()
    for source_root in source_roots:
        normalized = normalize_path(source_root)
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved_roots.append(Path(normalized))
    return resolved_roots


def resolve_existing_trajectory_paths(raw_paths: list[Path] | None) -> list[Path]:
    paths = raw_paths
    if paths is None:
        paths = [DEFAULT_EXISTING_TRAJECTORY_PATH] if DEFAULT_EXISTING_TRAJECTORY_PATH.exists() else []
    resolved_paths = []
    seen = set()
    for path in paths:
        normalized = normalize_path(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved_paths.append(Path(normalized))
    return resolved_paths


def extract_task_id(value: str | Path | None) -> str | None:
    if value is None:
        return None
    match = TASK_ID_RE.search(str(value))
    return match.group(1) if match else None


def canonical_task_key_from_parts(*values: str | Path | None) -> str:
    for value in values:
        task_id = extract_task_id(value)
        if task_id:
            return task_id
    for value in values:
        if value:
            return Path(str(value)).stem
    return "unknown_task"


def canonical_task_key(trajectory: dict) -> str:
    return canonical_task_key_from_parts(
        trajectory.get("task_key"),
        trajectory.get("task_stem"),
        trajectory.get("gym_task_config_name"),
        trajectory.get("source_path"),
        trajectory.get("seed_source_path"),
        trajectory.get("trajectory_id"),
    )


def domain_from_source_path(path: str | Path) -> str:
    stem = Path(path).stem
    match = RESULTS_STEM_RE.match(stem)
    if match:
        return match.group(1).lower()
    for part in reversed(Path(path).parts):
        lowered = part.lower()
        if lowered in {"calendar", "csm", "drive", "email", "hr", "hybrid", "itsm", "teams"}:
            return lowered
    return "unknown"


def load_enterpriseops_gym_seed_lookup(seeds_path: Path, domains: set[str]) -> dict[str, dict]:
    seed_map_by_path = {}
    seed_map_by_stem = {}
    seed_map_by_task_key = {}
    normalized_source_paths = []
    for record in load_jsonl(seeds_path):
        if record.get("source_dataset") != "EnterpriseOps-Gym":
            continue
        domain = (record.get("environment") or {}).get("domain", "").lower()
        if domains and domain not in domains:
            continue
        source_path = (record.get("source_record") or {}).get("source_path")
        if not source_path:
            continue
        normalized = normalize_path(source_path)
        stem = Path(source_path).stem
        if normalized in seed_map_by_path:
            raise ValueError(f"Duplicate EnterpriseOps-Gym seed for source path: {normalized}")
        if stem in seed_map_by_stem:
            raise ValueError(f"Duplicate EnterpriseOps-Gym seed for source stem: {stem}")
        seed_map_by_path[normalized] = record
        seed_map_by_stem[stem] = record
        task_key = canonical_task_key_from_parts(stem, source_path)
        seed_map_by_task_key.setdefault(task_key, record)
        normalized_source_paths.append((normalized, stem))
    task_index_by_stem = {
        stem: index
        for index, (_, stem) in enumerate(sorted(normalized_source_paths))
    }
    return {
        "by_path": seed_map_by_path,
        "by_stem": seed_map_by_stem,
        "by_task_key": seed_map_by_task_key,
        "task_index_by_stem": task_index_by_stem,
    }


def lookup_seed_for_source_file(source_file: Path, seed_lookup: dict[str, dict]) -> tuple[dict | None, int | None]:
    normalized = normalize_path(source_file)
    seed = seed_lookup["by_path"].get(normalized)
    stem = source_file.stem
    if seed is None:
        seed = seed_lookup["by_stem"].get(stem)
    if seed is None:
        seed = seed_lookup["by_task_key"].get(canonical_task_key_from_parts(stem))
    if seed is None:
        return None, None
    seed_source_path = (seed.get("source_record") or {}).get("source_path")
    seed_stem = Path(seed_source_path).stem if seed_source_path else stem
    return seed, seed_lookup["task_index_by_stem"][seed_stem]


def iter_source_files(source_roots: list[Path]) -> list[Path]:
    source_files = []
    seen = set()
    for source_root in source_roots:
        for path in sorted(path for path in source_root.glob("**/*.json") if path.is_file()):
            normalized = normalize_path(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            source_files.append(path)
    return source_files


def slugify_path_part(part: str) -> str:
    slug = "".join(character if character.isalnum() else "_" for character in str(part))
    slug = slug.strip("_")
    return slug or "unknown"


def source_variant_from_path(path: str | Path) -> str:
    resolved_parts = Path(path).resolve().parts
    if "react" in resolved_parts:
        start_index = resolved_parts.index("react") + 1
        variant_parts = resolved_parts[start_index : start_index + 4]
    else:
        variant_parts = resolved_parts[-4:]
    return "-".join(slugify_path_part(part) for part in variant_parts if part) or "unknown-source"


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def dump_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def dump_records(path: Path, records: list[dict], output_format: str):
    if output_format == "jsonl":
        dump_jsonl(path, records)
    else:
        dump_json(path, records)


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def load_existing_trajectories(paths: list[Path], domains: set[str]) -> list[dict]:
    trajectories = []
    for path in paths:
        for trajectory in load_records(path):
            if domains and str(trajectory.get("domain", "")).lower() not in domains:
                continue
            trajectory.setdefault("source_path", str(path))
            trajectory.setdefault("source_variant", source_variant_from_path(path))
            trajectory["task_key"] = canonical_task_key(trajectory)
            trajectory.setdefault("task_stem", trajectory["task_key"])
            trajectories.append(repair_explicit_tool_error_labels(cleanup_world_model_trajectory(trajectory)))
    return trajectories


def assign_canonical_task_indices(trajectories: list[dict]) -> None:
    ordered_task_keys = sorted({canonical_task_key(trajectory) for trajectory in trajectories})
    task_index_by_key = {
        task_key: task_index
        for task_index, task_key in enumerate(ordered_task_keys)
    }
    for trajectory in trajectories:
        task_key = canonical_task_key(trajectory)
        trajectory["task_key"] = task_key
        trajectory["task_index"] = task_index_by_key[task_key]


class FallbackStageLLM:
    def __init__(self, reason: str):
        self.reason = reason

    def generate_format(self, *args, **kwargs):
        raise RuntimeError(self.reason)


def extract_system_and_user_messages(conversation_flow: list[dict]) -> tuple[str, str]:
    system_message = next(
        item["content"]
        for item in conversation_flow
        if item.get("type") == "system_message"
    )
    user_message = next(
        item["content"]
        for item in conversation_flow
        if item.get("type") == "user_message"
    )
    return system_message, user_message


def serialize_full_tool_output(tool_result: dict, parsed: dict) -> str:
    tool_name = tool_result.get("tool_name") or "unknown_tool"
    if parsed["raw_text"]:
        output = parsed["raw_text"]
    else:
        output = parsed["payload"]
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False, sort_keys=True)

    return f"{tool_name}: {output}"


def summarize_tool_batch(tool_results: list[dict]) -> dict:
    if not tool_results:
        return make_default_tool_context()

    full_outputs = []
    failures = []
    for tool_result in tool_results:
        tool_name = tool_result.get("tool_name") or "unknown_tool"
        parsed = extract_tool_payload(tool_result)
        full_outputs.append(serialize_full_tool_output(tool_result, parsed))
        if not parsed["success"]:
            failure_reason = summarize_text(
                parsed["error"] or parsed["raw_text"] or parsed["summary"] or f"{tool_name} failed",
                limit=240,
            )
            failures.append({"tool_name": tool_name, "reason": failure_reason})

    selected_tool = failures[-1]["tool_name"] if failures else (tool_results[-1].get("tool_name") or "unknown_tool")
    joined_output = "\n\n".join(full_outputs)
    error_message = None
    if failures:
        error_message = summarize_text(
            " | ".join(f"{item['tool_name']}: {item['reason']}" for item in failures),
            limit=600,
        )

    return {
        "last_tool_execution_result": -1 if failures else 1,
        "last_tool_name": selected_tool,
        "last_tool_output": joined_output,
        "error_message": error_message,
    }


def state_context_for_message(state_message: dict) -> dict | None:
    content = state_message.get("content")
    if not isinstance(content, dict):
        return None
    nested_context = content.get("state", {}).get("context")
    if isinstance(nested_context, dict):
        return nested_context
    if "last_tool_execution_result" in content:
        return content
    return None


def contains_explicit_error_signal(value) -> bool:
    if not value:
        return False
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    lowered = value.lower()
    if re.search(r"\b(?:4\d{2}|5\d{2})\b", lowered):
        return True
    markers = (
        "error:",
        "api error",
        "mcperror",
        "error code",
        "exception",
        "traceback",
        "input validation error",
        "bad request",
        "failed",
        "failure",
        "internal server error",
        "service unavailable",
        "tool returned no output",
        '"iserror": true',
        '"success": false',
        "success=false",
    )
    return any(marker in lowered for marker in markers)


def tool_context_has_explicit_error(latest_tool_context: dict) -> bool:
    if normalize_tool_execution_label(latest_tool_context.get("last_tool_execution_result")) == -1:
        return True
    return is_explicit_tool_error(latest_tool_context) or any(
        contains_explicit_error_signal(latest_tool_context.get(key))
        for key in ("error_message", "last_tool_output")
    )


def repair_explicit_tool_error_labels(trajectory: dict) -> dict:
    messages = trajectory.get("messages") or []
    for index in range(len(messages) - 1):
        action_message = messages[index]
        state_message = messages[index + 1]
        if action_message.get("role") != "action" or state_message.get("role") != "state":
            continue
        action_content = action_message.get("content")
        if not action_uses_tool(action_content):
            continue
        context = state_context_for_message(state_message)
        if not context:
            continue
        latest_tool_context = {
            "last_tool_execution_result": context.get("last_tool_execution_result"),
            "last_tool_name": context.get("last_tool_name"),
            "last_tool_output": context.get("last_tool_output"),
            "error_message": context.get("error_message"),
        }
        if tool_context_has_explicit_error(latest_tool_context):
            context["last_tool_execution_result"] = -1
    return trajectory


def normalize_tool_execution_label(value) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {-1, 0, 1}:
        return value
    if isinstance(value, float) and value in {-1.0, 0.0, 1.0}:
        return int(value)
    if isinstance(value, str) and value.strip() in {"-1", "0", "1"}:
        return int(value.strip())
    return None


def count_enterpriseops_tool_execution_results(trajectory: dict) -> dict[int, int]:
    counts = empty_tool_execution_result_counts()
    messages = trajectory.get("messages") or []
    for index in range(len(messages) - 1):
        action_message = messages[index]
        state_message = messages[index + 1]
        if action_message.get("role") != "action" or state_message.get("role") != "state":
            continue
        if not action_uses_tool(action_message.get("content")):
            continue
        context = state_context_for_message(state_message)
        if not context:
            continue
        label = normalize_tool_execution_label(context.get("last_tool_execution_result"))
        if label in counts:
            counts[label] += 1
    return counts


def add_tool_counts(target: dict[int, int], counts: dict[int, int]) -> dict[int, int]:
    for label in (-1, 0, 1):
        target[label] = target.get(label, 0) + counts.get(label, 0)
    return target


def build_enterpriseops_split_groups(
    primary_trajectories: list[dict],
    additional_variant_records: list[dict],
) -> dict:
    groups = []
    grouped_batch_records = {}
    unmatched_batch_records = []
    total_label_counts = empty_tool_execution_result_counts()
    total_record_count = 0

    for split_index, trajectory in enumerate(primary_trajectories):
        label_counts = count_enterpriseops_tool_execution_results(trajectory)
        groups.append(
            {
                "split_index": split_index,
                "label_counts": label_counts,
                "record_count": 1,
                "gold_record_count": 1,
                "batch_record_count": 0,
            }
        )
        add_tool_counts(total_label_counts, label_counts)
        total_record_count += 1

    group_index_map = {group["split_index"]: group for group in groups}
    for batch_record in additional_variant_records:
        split_index = batch_record["task_index"]
        trajectory = batch_record["trajectory"]
        if split_index not in group_index_map:
            unmatched_batch_records.append(
                {
                    "task_index": batch_record["task_index"],
                    "split_index": split_index,
                    "trajectory_id": trajectory.get("trajectory_id"),
                }
            )
            continue

        label_counts = count_enterpriseops_tool_execution_results(trajectory)
        group = group_index_map[split_index]
        add_tool_counts(group["label_counts"], label_counts)
        group["record_count"] += 1
        group["batch_record_count"] += 1
        add_tool_counts(total_label_counts, label_counts)
        total_record_count += 1
        grouped_batch_records.setdefault(split_index, []).append(batch_record["task_index"])

    return {
        "groups": groups,
        "total_label_counts": total_label_counts,
        "total_record_count": total_record_count,
        "grouped_batch_records": grouped_batch_records,
        "unmatched_batch_records": unmatched_batch_records,
    }


def semantic_stagnation_triggered(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if SEMANTIC_STAGNATION_STATUS_RE.search(lowered):
        return True
    return any(marker in lowered for marker in SEMANTIC_STAGNATION_TRIGGER_MARKERS)


def build_semantic_stagnation_prompt_payload(
    *,
    system_prompt: str,
    user_prompt: str,
    action_content,
    tool_results: list[dict],
) -> dict | None:
    observed_results = []
    saw_non_success = False
    suspicious = False

    for tool_result in tool_results:
        tool_name = tool_result.get("tool_name") or "unknown_tool"
        parsed = extract_tool_payload(tool_result)
        if not parsed["success"]:
            saw_non_success = True

        observed_output = summarize_text(
            parsed["raw_text"] or parsed["summary"] or parsed["payload"],
            limit=600,
        )
        observed_error = summarize_text(parsed["error"], limit=240)
        suspicious = suspicious or semantic_stagnation_triggered(observed_output) or semantic_stagnation_triggered(
            observed_error
        )
        observed_results.append(
            {
                "tool_name": tool_name,
                "reported_success": bool(parsed["success"]),
                "output": observed_output,
                "error": observed_error,
            }
        )

    if saw_non_success or not suspicious or not observed_results:
        return None

    return {
        "mode": "enterpriseops_gym_semantic_stagnation_judge",
        "system_prompt": summarize_text(system_prompt, limit=1800),
        "user_prompt": summarize_text(user_prompt, limit=1200),
        "action": summarize_text(action_content, limit=800),
        "observed_results": observed_results,
    }


def classify_semantic_stagnation_with_llm(
    *,
    llm,
    stage_cache,
    cache_path: Path,
    cache_tracker: dict,
    args,
    system_prompt: str,
    user_prompt: str,
    action_content,
    tool_results: list[dict],
) -> dict:
    prompt_payload = build_semantic_stagnation_prompt_payload(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        action_content=action_content,
        tool_results=tool_results,
    )
    if prompt_payload is None:
        return {"semantic_stagnation": False, "reason": None}

    key = cache_key(prompt_payload)
    cached = stage_cache.get(key)
    if isinstance(cached, dict) and "semantic_stagnation" in cached:
        return {
            "semantic_stagnation": bool(cached.get("semantic_stagnation")),
            "reason": summarize_text(cached.get("reason"), limit=240),
        }

    system_instruction = (
        "You judge whether a tool batch that was reported as successful should still be treated as "
        "semantic stagnation for trajectory labeling. "
        "Return strict JSON with keys `semantic_stagnation` and `reason`. "
        "Set `semantic_stagnation` to true only when the observed output shows the requested step did not actually "
        "make progress, despite the wrapper saying success. Examples include 404/page not found, missing resource, "
        "invalid identifier, no matching object, access denied text, unsupported operation, or other clearly "
        "error-like responses that leave the task stuck. "
        "If the output reflects a valid successful read/write/search result, return false."
    )
    prompt = (
        "Task context:\n"
        f"- Agent system prompt: {prompt_payload['system_prompt']}\n"
        f"- User task: {prompt_payload['user_prompt']}\n"
        f"- Action content: {prompt_payload['action']}\n"
        f"- Observed tool results: {json.dumps(prompt_payload['observed_results'], ensure_ascii=False)}\n\n"
        "Should this step be labeled as semantic stagnation (`0`) instead of success (`1`)?"
    )

    try:
        judgment = llm.generate_format(
            prompt=prompt,
            system_prompt=system_instruction,
            format="json",
            schema=SEMANTIC_STAGNATION_JUDGE_SCHEMA,
        )
    except Exception:
        if not args.allow_stage_fallback:
            raise
        return {"semantic_stagnation": False, "reason": None}

    judgment_payload = judgment if isinstance(judgment, dict) else {}
    result = {
        "semantic_stagnation": bool(judgment_payload.get("semantic_stagnation")),
        "reason": summarize_text(judgment_payload.get("reason"), limit=240),
    }
    stage_cache[key] = result
    cache_tracker["pending_writes"] += 1
    flush_stage_cache(stage_cache, cache_path, cache_tracker)
    return result


def summarize_tool_batch_with_semantic_stagnation(
    *,
    llm,
    stage_cache,
    cache_path: Path,
    cache_tracker: dict,
    args,
    system_prompt: str,
    user_prompt: str,
    action_content,
    tool_results: list[dict],
) -> dict:
    tool_context = summarize_tool_batch(tool_results)
    semantic_judgment = classify_semantic_stagnation_with_llm(
        llm=llm,
        stage_cache=stage_cache,
        cache_path=cache_path,
        cache_tracker=cache_tracker,
        args=args,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        action_content=action_content,
        tool_results=tool_results,
    )
    if semantic_judgment["semantic_stagnation"]:
        tool_context["semantic_stagnation"] = True
        tool_context["semantic_stagnation_reason"] = semantic_judgment["reason"]
    return tool_context


def build_trajectory_id(source_path: str | Path) -> str:
    source_path = Path(source_path)
    return f"enterpriseops-gym-world-model-{source_variant_from_path(source_path)}-{source_path.stem}"


def derive_gym_task_config_name(source_path: str | Path) -> str | None:
    """Map a gym run-output filename to the corresponding task-config filename.

    Run outputs use the pattern `results_<mode>__<domain>__<task_id>.json`; the
    task configs `evaluate.py` consumes drop the `results_` prefix. Returns None
    if the filename does not match the gym pattern.
    """
    stem = Path(source_path).stem
    if not stem.startswith("results_"):
        return None
    return f"{stem[len('results_'):]}.json"


def build_fallback_seed(source_file: Path) -> dict:
    domain = domain_from_source_path(source_file)
    task_key = canonical_task_key_from_parts(source_file.stem)
    return {
        "seed_id": f"import.enterpriseops_gym_raw.{domain}.{task_key}",
        "environment": {"domain": domain},
        "coordination_pattern": "synthetic_single_agent_replay",
        "source_record": {"source_path": str(source_file)},
    }


def initialize_llm(args):
    prepare_llm_environment(args.llm_method)
    try:
        from src.llm import LLM

        return LLM(args.llm_method), None
    except Exception as exc:
        if not args.allow_stage_fallback:
            raise
        return FallbackStageLLM(str(exc)), str(exc)


def humanize_tool_name(tool_name: str) -> str:
    return str(tool_name).replace("_", " ").strip()


def sentence_case(text: str) -> str:
    if not text:
        return text
    return text[:1].upper() + text[1:]


def unique_ordered(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def normalize_stage_value(value, *, limit: int = 48) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        normalized = value.replace("_", " ").strip()
    elif isinstance(value, (int, float)):
        normalized = str(value)
    else:
        normalized = summarize_text(value, limit=limit)
    return summarize_text(normalized, limit=limit)


def parse_tool_arguments(arguments):
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def extract_stage_targets(arguments: dict) -> list[str]:
    if not isinstance(arguments, dict) or not arguments:
        return []

    targets = []
    first_name = normalize_stage_value(arguments.get("first_name"))
    last_name = normalize_stage_value(arguments.get("last_name"))
    if first_name or last_name:
        targets.append(" ".join(item for item in (first_name, last_name) if item))

    for key in (
        "name",
        "title",
        "team_name",
        "channel_name",
        "group_name",
        "chat_name",
        "meeting_name",
        "webinar_name",
        "event_name",
        "tag_name",
        "serial_number",
        "short_description",
        "subject",
        "email",
        "email_domain",
        "job_title",
        "search_text",
        "query",
        "number",
        "case_number",
        "incident_number",
        "problem_number",
    ):
        value = normalize_stage_value(arguments.get(key))
        if value:
            targets.append(value)

    if not targets:
        for key, value in arguments.items():
            if key.endswith("_id"):
                continue
            normalized = normalize_stage_value(value)
            if normalized:
                targets.append(normalized)
            if len(targets) >= 3:
                break

    return unique_ordered(targets)[:3]


def summarize_tool_call_for_stage(tool_call: dict, *, limit: int = 120) -> str:
    function = tool_call.get("function") or {}
    tool_name = sentence_case(humanize_tool_name(function.get("name") or "tool call"))
    arguments = parse_tool_arguments(function.get("arguments"))
    targets = extract_stage_targets(arguments)
    if not targets:
        return summarize_text(tool_name, limit=limit) or "Complete the active step"
    return summarize_text(f"{tool_name}: {', '.join(targets)}", limit=limit) or tool_name


def ensure_distinct_adjacent_stage_labels(stage_labels: list[str]) -> list[str]:
    distinct_labels = []
    total_steps = len(stage_labels)
    for index, label in enumerate(stage_labels, start=1):
        normalized = summarize_text(label, limit=120) or "Complete the active step"
        if distinct_labels and normalized == distinct_labels[-1]:
            normalized = summarize_text(f"{normalized} ({index}/{total_steps})", limit=120)
        distinct_labels.append(normalized)
    return distinct_labels


def build_stage_labels(action_batches: list[dict]) -> list[str]:
    labels = []
    total_steps = len(action_batches)
    for index, batch in enumerate(action_batches, start=1):
        action_content = batch["action_content"]
        if action_uses_tool(action_content):
            tool_calls = action_content.get("tool_calls") or []
            tool_call_labels = [
                summarize_tool_call_for_stage(tool_call, limit=72)
                for tool_call in tool_calls
            ]
            tool_call_labels = [label for label in tool_call_labels if label]
            if tool_call_labels:
                label = summarize_text("; ".join(tool_call_labels), limit=120)
            else:
                label = summarize_action_for_stage(action_content, limit=120)
        else:
            label = "Deliver final response" if index == total_steps else summarize_action_for_stage(
                action_content,
                limit=120,
            )
        labels.append(label or "Complete the active step")
    return ensure_distinct_adjacent_stage_labels(labels)


def normalize_stage_plan(
    proposed_labels,
    *,
    fallback_labels: list[str],
) -> list[str]:
    if not isinstance(proposed_labels, list) or len(proposed_labels) != len(fallback_labels):
        return fallback_labels

    normalized = []
    for index, fallback_label in enumerate(fallback_labels):
        candidate = summarize_text(proposed_labels[index], limit=120)
        normalized.append(candidate or fallback_label)

    for index in range(1, len(normalized)):
        if normalized[index] == normalized[index - 1]:
            normalized[index] = fallback_labels[index]

    return ensure_distinct_adjacent_stage_labels(normalized)


def build_trajectory_stage_labels(
    *,
    llm,
    stage_cache,
    cache_path: Path,
    cache_tracker: dict,
    args,
    system_prompt: str,
    user_prompt: str,
    action_batches: list[dict],
    use_heuristic_stage_fallback: bool,
) -> list[str]:
    fallback_labels = build_stage_labels(action_batches)
    if use_heuristic_stage_fallback:
        return fallback_labels

    ordered_actions = []
    for step_index, batch in enumerate(action_batches, start=1):
        tool_context = summarize_tool_batch_with_semantic_stagnation(
            llm=llm,
            stage_cache=stage_cache,
            cache_path=cache_path,
            cache_tracker=cache_tracker,
            args=args,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            action_content=batch["action_content"],
            tool_results=batch["tool_results"],
        )
        ordered_actions.append(
            {
                "step": step_index,
                "action": fallback_labels[step_index - 1],
                "tool_output": summarize_text(tool_context.get("last_tool_output"), limit=240),
                "tool_error": summarize_text(
                    tool_context.get("error_message") or tool_context.get("semantic_stagnation_reason"),
                    limit=180,
                ),
            }
        )

    prompt_payload = {
        "mode": "enterpriseops_gym_full_trajectory_stage_plan",
        "system_prompt": summarize_text(system_prompt, limit=2500),
        "user_prompt": summarize_text(user_prompt, limit=1500),
        "ordered_actions": ordered_actions,
    }
    key = cache_key(prompt_payload)
    cached = stage_cache.get(key)
    if isinstance(cached, dict) and "stage_labels" in cached:
        return normalize_stage_plan(
            cached.get("stage_labels"),
            fallback_labels=fallback_labels,
        )

    system_instruction = (
        "You infer a detailed process stage plan for a completed single-agent enterprise task. "
        "Return strict JSON with key `stage_labels`. "
        "`stage_labels` must contain exactly one concise actionable stage label for each action step in order. "
        "Include substeps: when multiple actions contribute to a broader goal, break that goal into smaller stages so each action advances to a distinct next stage. "
        "Do not merge neighboring actions into one stage. "
        "Do not add, remove, or reorder steps. "
        "Keep labels short, grounded in the task, and specific about the entity, object, or outcome when possible."
    )
    prompt = (
        "Task context:\n"
        f"- Agent system prompt: {prompt_payload['system_prompt']}\n"
        f"- User task: {prompt_payload['user_prompt']}\n"
        f"- Ordered actions: {json.dumps(prompt_payload['ordered_actions'], ensure_ascii=False)}\n\n"
        "Generate the full stage plan now."
    )

    try:
        stage_plan = llm.generate_format(
            prompt=prompt,
            system_prompt=system_instruction,
            format="json",
            schema=TRAJECTORY_STAGE_PLAN_SCHEMA,
        )
    except Exception:
        if not args.allow_stage_fallback:
            raise
        return fallback_labels

    stage_plan_payload = stage_plan if isinstance(stage_plan, dict) else {}
    normalized_stage_labels = normalize_stage_plan(
        stage_plan_payload.get("stage_labels"),
        fallback_labels=fallback_labels,
    )
    stage_cache[key] = {
        "stage_labels": normalized_stage_labels,
    }
    cache_tracker["pending_writes"] += 1
    flush_stage_cache(stage_cache, cache_path, cache_tracker)
    return normalized_stage_labels


def build_planned_state_message(
    *,
    system_prompt: str,
    action_content,
    latest_tool_context: dict,
    previous_process_state: dict | None,
    stage_labels: list[str],
    step_index: int,
    total_steps: int,
):
    explicit_tool_error = action_uses_tool(action_content) and tool_context_has_explicit_error(latest_tool_context)
    semantic_stagnation = action_uses_tool(action_content) and bool(latest_tool_context.get("semantic_stagnation"))
    current_label = stage_labels[step_index - 1] if stage_labels else "Complete the active step"

    if explicit_tool_error or semantic_stagnation:
        process_state = {
            "remaining_stages": stage_labels[step_index - 1 :] if stage_labels else [current_label],
            "current_stage": current_label,
        }
    elif step_index >= total_steps:
        process_state = {
            "remaining_stages": [],
            "current_stage": "finished",
        }
    else:
        process_state = {
            "remaining_stages": stage_labels[step_index:],
            "current_stage": stage_labels[step_index],
        }

    if semantic_stagnation and not explicit_tool_error:
        action_execution_result = 0
    else:
        action_execution_result = classify_action_execution_result(
            action_content,
            latest_tool_context,
            previous_process_state,
            process_state,
        )
    state = {
        "state": {
            "agent": {
                "role": extract_agent_role(system_prompt),
            },
            "context": {
                "last_tool_execution_result": action_execution_result,
                "last_tool_name": latest_tool_context.get("last_tool_name"),
                "last_tool_output": latest_tool_context.get("last_tool_output"),
            },
            "process": process_state,
            "relational": {
                "permisson_level": "admin",
            },
            "temporal": {
                "remaining_time": "unbounded",
            },
        }
    }
    if explicit_tool_error and latest_tool_context.get("error_message"):
        state["state"]["context"]["error_message"] = latest_tool_context["error_message"]
    return {
        "role": "state",
        "content": state,
    }, process_state


def reconstruct_trajectory(
    seed: dict,
    *,
    source_path: Path,
    task_index: int,
    llm,
    use_heuristic_stage_fallback: bool,
    stage_cache,
    cache_path: Path,
    cache_tracker: dict,
    args,
) -> dict:
    raw_payload = load_json(source_path)
    runs = raw_payload.get("runs") or []
    if not runs:
        raise ValueError("missing runs")

    run = runs[0]
    conversation_flow = run["conversation_flow"]
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

    trajectory = {
        "trajectory_id": build_trajectory_id(source_path),
        "source": "enterprise_ops_gym_oracle_run",
        "domain": seed["environment"]["domain"],
        "seed_id": seed["seed_id"],
        "task_index": task_index,
        "task_key": canonical_task_key_from_parts(source_path.stem),
        "task_stem": canonical_task_key_from_parts(source_path.stem),
        "gym_task_config_name": derive_gym_task_config_name(source_path),
        "coordination_pattern": seed["coordination_pattern"],
        "seed_source_path": seed["source_record"]["source_path"],
        "source_path": str(source_path),
        "source_variant": source_variant_from_path(source_path),
        "messages": messages,
    }
    return cleanup_world_model_trajectory(trajectory)


def build_enterpriseops_split_manifest(
    trajectories: list[dict],
    source_roots: list[Path],
    args,
) -> dict:
    grouped_trajectories = {}
    grouped_source_variants = {}
    grouped_task_indices = {}
    grouped_task_stems = {}
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
    if args.test_task_count < 0:
        raise ValueError(f"--test-task-count must be non-negative, got {args.test_task_count}")
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
    split_train_ratio = (
        1.0 if not ordered_group_keys else (len(ordered_group_keys) - args.test_task_count) / len(ordered_group_keys)
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
        "seed": args.seed,
        "train_ratio": split_train_ratio,
        "trajectory_count": len(ordered_group_keys),
        "train_trajectories": len(train_indices),
        "test_trajectories": len(test_indices),
        "train_indices": train_indices,
        "test_indices": test_indices,
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
    }
    split_manifest["source_roots"] = [str(source_root.resolve()) for source_root in source_roots]
    split_manifest["seeds_path"] = str(args.seeds_path.resolve())
    split_manifest["split_unit"] = "canonical_task_key"
    split_manifest["test_task_count"] = args.test_task_count
    split_manifest["task_key_count"] = len(ordered_group_keys)
    split_manifest["task_keys"] = ordered_group_keys
    split_manifest["train_task_keys"] = [
        ordered_group_keys[split_index] for split_index in split_manifest["train_indices"]
    ]
    split_manifest["test_task_keys"] = [
        ordered_group_keys[split_index] for split_index in split_manifest["test_indices"]
    ]
    split_manifest["task_indices"] = sorted(
        {task_index for task_indices in grouped_task_indices.values() for task_index in task_indices}
    )
    split_manifest["train_task_indices"] = sorted(
        {
            task_index
            for split_index in split_manifest["train_indices"]
            for task_index in grouped_task_indices[ordered_group_keys[split_index]]
        }
    )
    split_manifest["test_task_indices"] = sorted(
        {
            task_index
            for split_index in split_manifest["test_indices"]
            for task_index in grouped_task_indices[ordered_group_keys[split_index]]
        }
    )
    split_manifest["task_source_variants"] = {
        str(task_index): grouped_source_variants[group_key]
        for group_key in ordered_group_keys
        for task_index in grouped_task_indices[group_key]
    }
    split_manifest["task_key_source_variants"] = {
        group_key: grouped_source_variants[group_key]
        for group_key in ordered_group_keys
    }
    split_manifest["task_key_stems"] = {
        group_key: grouped_task_stems[group_key]
        for group_key in ordered_group_keys
    }
    split_manifest["multi_variant_task_count"] = sum(
        len(grouped_trajectories[group_key]) > 1
        for group_key in ordered_group_keys
    )
    return split_manifest


def validate_disjoint_enterpriseops_task_splits(
    train_payload: list[dict],
    test_payload: list[dict],
    *,
    expected_test_task_count: int,
) -> None:
    train_task_keys = {canonical_task_key(trajectory) for trajectory in train_payload}
    test_task_keys = {canonical_task_key(trajectory) for trajectory in test_payload}
    if len(test_task_keys) != expected_test_task_count:
        raise ValueError(
            f"EnterpriseOps-Gym test split has {len(test_task_keys)} unique tasks, "
            f"expected {expected_test_task_count}."
        )

    overlapping_task_keys = sorted(train_task_keys & test_task_keys)
    if overlapping_task_keys:
        preview = overlapping_task_keys[:10]
        raise ValueError(
            "EnterpriseOps-Gym train/test split leaked overlapping task keys: "
            f"{preview}"
        )

    train_gym_task_names = {
        trajectory.get("gym_task_config_name")
        for trajectory in train_payload
        if trajectory.get("gym_task_config_name")
    }
    test_gym_task_names = {
        trajectory.get("gym_task_config_name")
        for trajectory in test_payload
        if trajectory.get("gym_task_config_name")
    }
    overlapping_task_names = sorted(train_gym_task_names & test_gym_task_names)
    if overlapping_task_names:
        preview = overlapping_task_names[:10]
        raise ValueError(
            "EnterpriseOps-Gym train/test split leaked overlapping gym_task_config_name values: "
            f"{preview}"
        )


def main():
    args = parse_args()
    domain_filter = parse_domain_filter(args.domains)
    source_roots = resolve_source_roots(args.source_roots)
    existing_trajectory_paths = resolve_existing_trajectory_paths(args.existing_trajectory_paths)
    seed_lookup = load_enterpriseops_gym_seed_lookup(args.seeds_path, domain_filter)
    source_files = iter_source_files(source_roots)

    llm, llm_fallback_reason = initialize_llm(args)
    stage_cache = load_stage_cache(args.stage_cache)
    cache_tracker = {"pending_writes": 0, "flush_every": 25}

    trajectories = load_existing_trajectories(existing_trajectory_paths, domain_filter)
    matched_paths = []
    missing_paths = []
    skipped_records = []
    seen_source_paths = {
        normalize_path(trajectory.get("source_path"))
        for trajectory in trajectories
        if trajectory.get("source_path")
    }
    seen_trajectory_ids = {
        trajectory.get("trajectory_id")
        for trajectory in trajectories
        if trajectory.get("trajectory_id")
    }

    for source_file in source_files:
        if normalize_path(source_file) in seen_source_paths:
            continue
        seed, task_index = lookup_seed_for_source_file(source_file, seed_lookup)
        if seed is None:
            if domain_filter and domain_from_source_path(source_file) not in domain_filter:
                missing_paths.append(str(source_file))
                continue
            seed = build_fallback_seed(source_file)
            task_index = -1

        try:
            trajectory = reconstruct_trajectory(
                seed,
                source_path=source_file,
                task_index=task_index,
                llm=llm,
                use_heuristic_stage_fallback=bool(llm_fallback_reason),
                stage_cache=stage_cache,
                cache_path=args.stage_cache,
                cache_tracker=cache_tracker,
                args=args,
            )
        except Exception as exc:
            skipped_records.append(
                {
                    "source_path": str(source_file),
                    "seed_id": seed.get("seed_id"),
                    "reason": str(exc),
                }
            )
            continue

        trajectory = repair_explicit_tool_error_labels(trajectory)
        if trajectory.get("trajectory_id") in seen_trajectory_ids:
            continue
        trajectories.append(trajectory)
        seen_trajectory_ids.add(trajectory.get("trajectory_id"))
        matched_paths.append(str(source_file))

        if args.max_records is not None and len(trajectories) >= args.max_records:
            break

    flush_stage_cache(stage_cache, args.stage_cache, cache_tracker, force=True)

    if args.strict and (missing_paths or skipped_records):
        preview_lines = missing_paths[:10]
        preview_lines.extend(item["source_path"] for item in skipped_records[:10])
        preview = "\n".join(preview_lines)
        raise ValueError(
            "Found EnterpriseOps-Gym run files without matching imported seeds or reconstructable runs:\n"
            f"{preview}"
        )

    if not trajectories:
        raise ValueError(
            f"No EnterpriseOps-Gym trajectories were generated from {source_roots} "
            f"using seeds from {args.seeds_path}."
        )

    assign_canonical_task_indices(trajectories)
    split_manifest = build_enterpriseops_split_manifest(trajectories, source_roots, args)
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
            "matched_run_files": len(matched_paths),
            "existing_trajectory_paths": [str(path) for path in existing_trajectory_paths],
            "existing_trajectory_count": len(trajectories) - len(matched_paths),
            "unmatched_run_files": len(missing_paths),
            "skipped_records": skipped_records,
            "llm_method": args.llm_method,
        },
    )

    domain_counts = Counter(item["domain"] for item in trajectories)
    train_domain_counts = Counter(item["domain"] for item in train_payload)
    test_domain_counts = Counter(item["domain"] for item in test_payload)
    source_variant_counts = Counter(item["source_variant"] for item in trajectories)
    train_source_variant_counts = Counter(item["source_variant"] for item in train_payload)
    test_source_variant_counts = Counter(item["source_variant"] for item in test_payload)

    print(f"Wrote {len(trajectories)} trajectories to {args.output_path}")
    print(f"Wrote {len(train_payload)} train trajectories to {args.train_output_path}")
    print(f"Wrote {len(test_payload)} test trajectories to {args.test_output_path}")
    print(f"Wrote split manifest to {args.generated_split_manifest}")
    if existing_trajectory_paths:
        print(f"Loaded existing trajectories from: {', '.join(str(path) for path in existing_trajectory_paths)}")
    print(f"Matched run files: {len(matched_paths)}")
    print(f"Unmatched run files: {len(missing_paths)}")
    print(f"Skipped non-reconstructable runs: {len(skipped_records)}")
    if llm_fallback_reason:
        print(f"Stage generation fallback reason: {llm_fallback_reason}")
    if domain_counts:
        print("Domain counts:")
        for domain, count in sorted(domain_counts.items()):
            print(f"  {domain}: {count}")
    if train_domain_counts:
        print("Train domain counts:")
        for domain, count in sorted(train_domain_counts.items()):
            print(f"  {domain}: {count}")
    if test_domain_counts:
        print("Test domain counts:")
        for domain, count in sorted(test_domain_counts.items()):
            print(f"  {domain}: {count}")
    if source_variant_counts:
        print("Source variant counts:")
        for source_variant, count in sorted(source_variant_counts.items()):
            print(f"  {source_variant}: {count}")
    if train_source_variant_counts:
        print("Train source variant counts:")
        for source_variant, count in sorted(train_source_variant_counts.items()):
            print(f"  {source_variant}: {count}")
    if test_source_variant_counts:
        print("Test source variant counts:")
        for source_variant, count in sorted(test_source_variant_counts.items()):
            print(f"  {source_variant}: {count}")
    print(f"Train label counts: {split_manifest['train_label_counts']}")
    print(f"Test label counts: {split_manifest['test_label_counts']}")
    if skipped_records:
        print("Sample skipped runs:")
        for item in skipped_records[:5]:
            print(f"  {item['source_path']}: {item['reason']}")


if __name__ == "__main__":
    main()
