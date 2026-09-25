import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preparation.world_model_trajectory_cleanup import cleanup_world_model_trajectory


DEFAULT_GOLD_PATH = ROOT / "trajectories" / "enterprise_arena_gold.json"
DEFAULT_BATCH_GLOB = (
    Path.home()
    / "program"
    / "tools"
    / "EnterpriseLab"
    / "Evaluate"
    / "EnterpriseArena"
    / "trajectories"
    / "batch_trajectories_*.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "trajectories"
DEFAULT_SPLIT_MANIFEST = ROOT / "data" / "user" / "ewm" / "world_model" / "split_manifest.json"

STAGE_SCHEMA = {
    "remaining_stages": [""],
    "current_stage": "",
}
TOOL_EXECUTION_RESULT_LABELS = (-1, 0, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct EnterpriseArena trajectories into world-model action/state JSON.",
    )
    parser.add_argument("--gold-path", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--batch-glob", type=str, default=str(DEFAULT_BATCH_GLOB))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--train-output",
        type=str,
        default="world_model_train_trajectories.json",
    )
    parser.add_argument(
        "--test-output",
        type=str,
        default="world_model_test_trajectories.json",
    )
    parser.add_argument(
        "--generated-split-manifest",
        type=str,
        default="world_model_trajectory_split_manifest.json",
    )
    parser.add_argument(
        "--stage-cache",
        type=str,
        default="world_model_stage_cache.json",
    )
    parser.add_argument(
        "--llm-method",
        default="gemini",
        help="Method passed to src.llm.LLM for stage generation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument(
        "--reuse-split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
        help="Reuse an existing 60/40 split manifest when available for consistent task-index mapping.",
    )
    parser.add_argument(
        "--force-new-split",
        action="store_true",
        help="Ignore any existing split manifest and generate a fresh random split from the gold trajectories.",
    )
    parser.add_argument(
        "--batch-task-index-base",
        type=int,
        default=1,
        help="Base for batch task_index values. The default maps task_index=1 to gold split index 0.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Optional limit for debugging.",
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
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def try_parse_json(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def normalize_message_content(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        parsed = try_parse_json(stripped)
        if isinstance(parsed, dict):
            return parsed
        return stripped
    return json.dumps(value, ensure_ascii=False)


def summarize_text(value, limit=600):
    if value is None:
        return None
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def extract_agent_role(system_prompt: str) -> str:
    if not system_prompt:
        return "single-agent enterprise assistant"
    match = re.search(r"You are (?:an?|the) ([^.\n]+)", system_prompt, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    first_line = system_prompt.splitlines()[0].strip()
    return first_line or "single-agent enterprise assistant"


def humanize_tool_name(tool_name: str) -> str:
    return str(tool_name).replace("_", " ").strip()


def summarize_action_for_stage(action_content, *, limit=180):
    if isinstance(action_content, dict) and action_content.get("tool_calls"):
        names = []
        for tool_call in action_content["tool_calls"]:
            function = tool_call.get("function") or {}
            name = function.get("name")
            if name:
                names.append(humanize_tool_name(name))
        if names:
            if len(names) == 1:
                return f"Execute {names[0]}"
            if len(names) == 2:
                return f"Execute {names[0]} and {names[1]}"
            return f"Execute {', '.join(names[:3])}"
    return summarize_text(action_content, limit=limit)


def load_stage_cache(path: Path):
    if not path.exists():
        return {}
    return load_json(path)


def cache_key(payload) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def flush_stage_cache(stage_cache, cache_path: Path, tracker, force=False):
    if force or tracker["pending_writes"] >= tracker["flush_every"]:
        dump_json(cache_path, stage_cache)
        tracker["pending_writes"] = 0


def empty_tool_execution_result_counts():
    return {label: 0 for label in TOOL_EXECUTION_RESULT_LABELS}


def serialize_tool_execution_result_counts(counts):
    return {str(label): int(counts.get(label, 0)) for label in TOOL_EXECUTION_RESULT_LABELS}


def add_tool_execution_result_counts(target, counts):
    for label in TOOL_EXECUTION_RESULT_LABELS:
        target[label] = target.get(label, 0) + counts.get(label, 0)
    return target


def subtract_tool_execution_result_counts(target, counts):
    for label in TOOL_EXECUTION_RESULT_LABELS:
        target[label] = target.get(label, 0) - counts.get(label, 0)
    return target


def normalize_last_tool_execution_result(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in TOOL_EXECUTION_RESULT_LABELS else None
    if isinstance(value, float):
        if value in {-1.0, 0.0, 1.0}:
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"-1", "0", "1"}:
            return int(stripped)
    return None


def state_message_follows_tool_action(message, next_message):
    content = message.get("content")
    return (
        message.get("role") == "action"
        and next_message.get("role") == "state"
        and isinstance(content, dict)
        and bool(content.get("tool_calls"))
    )


def extract_last_tool_execution_result_from_state(state):
    return normalize_last_tool_execution_result(
        state.get("state", {}).get("context", {}).get("last_tool_execution_result")
    )


def count_trajectory_tool_execution_results(trajectory):
    counts = empty_tool_execution_result_counts()
    messages = trajectory.get("messages", [])
    for cursor in range(len(messages) - 1):
        message = messages[cursor]
        next_message = messages[cursor + 1]
        if not state_message_follows_tool_action(message, next_message):
            continue
        state_content = next_message.get("content")
        if not isinstance(state_content, dict):
            continue
        result = extract_last_tool_execution_result_from_state(state_content)
        if result in counts:
            counts[result] += 1
    return counts


def score_split_assignment(
    train_label_counts,
    target_train_label_counts,
    total_label_counts,
    train_group_count,
    target_train_group_count,
    total_group_count,
    train_record_count,
    target_train_record_count,
    total_record_count,
):
    label_score = 0.0
    for label in TOOL_EXECUTION_RESULT_LABELS:
        denominator = max(1, total_label_counts.get(label, 0))
        label_score += abs(train_label_counts.get(label, 0) - target_train_label_counts.get(label, 0)) / denominator

    group_score = abs(train_group_count - target_train_group_count) / max(1, total_group_count)
    record_score = abs(train_record_count - target_train_record_count) / max(1, total_record_count)
    return label_score + group_score + record_score


def weighted_group_mass(group, total_label_counts):
    return (
        sum(
            group["label_counts"].get(label, 0) / max(1, total_label_counts.get(label, 0))
            for label in TOOL_EXECUTION_RESULT_LABELS
        ),
        group["record_count"],
        sum(group["label_counts"].values()),
    )


def build_candidate_group_split(
    groups,
    *,
    seed,
    total_label_counts,
    total_record_count,
    target_train_group_count,
    target_train_record_count,
    target_train_label_counts,
):
    ordered_groups = list(groups)
    random.Random(seed).shuffle(ordered_groups)
    ordered_groups.sort(
        key=lambda item: weighted_group_mass(item, total_label_counts),
        reverse=True,
    )

    train_groups = []
    test_groups = []
    train_label_counts = empty_tool_execution_result_counts()
    train_record_count = 0
    total_group_count = len(ordered_groups)

    for index, group in enumerate(ordered_groups):
        remaining_groups = total_group_count - index
        remaining_train_slots = target_train_group_count - len(train_groups)
        if remaining_train_slots <= 0:
            test_groups.append(group)
            continue
        if remaining_train_slots >= remaining_groups:
            train_groups.append(group)
            add_tool_execution_result_counts(train_label_counts, group["label_counts"])
            train_record_count += group["record_count"]
            continue

        candidate_train_label_counts = add_tool_execution_result_counts(
            empty_tool_execution_result_counts(),
            train_label_counts,
        )
        add_tool_execution_result_counts(candidate_train_label_counts, group["label_counts"])
        candidate_train_record_count = train_record_count + group["record_count"]

        train_score = score_split_assignment(
            candidate_train_label_counts,
            target_train_label_counts,
            total_label_counts,
            len(train_groups) + 1,
            target_train_group_count,
            total_group_count,
            candidate_train_record_count,
            target_train_record_count,
            total_record_count,
        )
        test_score = score_split_assignment(
            train_label_counts,
            target_train_label_counts,
            total_label_counts,
            len(train_groups),
            target_train_group_count,
            total_group_count,
            train_record_count,
            target_train_record_count,
            total_record_count,
        )

        if train_score <= test_score:
            train_groups.append(group)
            train_label_counts = candidate_train_label_counts
            train_record_count = candidate_train_record_count
        else:
            test_groups.append(group)

    return train_groups, test_groups, train_label_counts, train_record_count


def improve_group_split(
    train_groups,
    test_groups,
    train_label_counts,
    train_record_count,
    *,
    total_label_counts,
    total_record_count,
    target_train_group_count,
    target_train_record_count,
    target_train_label_counts,
):
    best_score = score_split_assignment(
        train_label_counts,
        target_train_label_counts,
        total_label_counts,
        len(train_groups),
        target_train_group_count,
        len(train_groups) + len(test_groups),
        train_record_count,
        target_train_record_count,
        total_record_count,
    )

    improved = True
    while improved:
        improved = False
        for train_index, train_group in enumerate(train_groups):
            for test_index, test_group in enumerate(test_groups):
                candidate_label_counts = add_tool_execution_result_counts(
                    empty_tool_execution_result_counts(),
                    train_label_counts,
                )
                subtract_tool_execution_result_counts(candidate_label_counts, train_group["label_counts"])
                add_tool_execution_result_counts(candidate_label_counts, test_group["label_counts"])
                candidate_record_count = (
                    train_record_count
                    - train_group["record_count"]
                    + test_group["record_count"]
                )
                candidate_score = score_split_assignment(
                    candidate_label_counts,
                    target_train_label_counts,
                    total_label_counts,
                    len(train_groups),
                    target_train_group_count,
                    len(train_groups) + len(test_groups),
                    candidate_record_count,
                    target_train_record_count,
                    total_record_count,
                )
                if candidate_score + 1e-12 >= best_score:
                    continue

                train_groups[train_index], test_groups[test_index] = test_group, train_group
                train_label_counts = candidate_label_counts
                train_record_count = candidate_record_count
                best_score = candidate_score
                improved = True
                break
            if improved:
                break

    return train_groups, test_groups, train_label_counts, train_record_count, best_score


def build_split_groups(gold_trajectories, batch_records, batch_task_index_base):
    groups = []
    grouped_batch_records = {}
    unmatched_batch_records = []
    total_label_counts = empty_tool_execution_result_counts()
    total_record_count = 0

    for gold_index, trajectory in enumerate(gold_trajectories):
        label_counts = count_trajectory_tool_execution_results(trajectory)
        groups.append(
            {
                "split_index": gold_index,
                "label_counts": label_counts,
                "record_count": 1,
                "gold_record_count": 1,
                "batch_record_count": 0,
            }
        )
        add_tool_execution_result_counts(total_label_counts, label_counts)
        total_record_count += 1

    group_index_map = {group["split_index"]: group for group in groups}

    for batch_record in batch_records:
        split_index = batch_record["task_index"] - batch_task_index_base
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

        label_counts = count_trajectory_tool_execution_results(trajectory)
        group = group_index_map[split_index]
        add_tool_execution_result_counts(group["label_counts"], label_counts)
        group["record_count"] += 1
        group["batch_record_count"] += 1
        add_tool_execution_result_counts(total_label_counts, label_counts)
        total_record_count += 1
        grouped_batch_records.setdefault(split_index, []).append(batch_record["task_index"])

    return {
        "groups": groups,
        "total_label_counts": total_label_counts,
        "total_record_count": total_record_count,
        "grouped_batch_records": grouped_batch_records,
        "unmatched_batch_records": unmatched_batch_records,
    }


def build_split(grouped_split_stats, args):
    groups = grouped_split_stats["groups"]
    total_label_counts = grouped_split_stats["total_label_counts"]
    total_record_count = grouped_split_stats["total_record_count"]
    grouped_batch_records = grouped_split_stats["grouped_batch_records"]
    unmatched_batch_records = grouped_split_stats["unmatched_batch_records"]
    gold_trajectory_count = len(groups)

    if (
        not args.force_new_split
        and args.reuse_split_manifest
        and args.reuse_split_manifest.exists()
    ):
        manifest = load_json(args.reuse_split_manifest)
        if (
            manifest.get("trajectory_count") == gold_trajectory_count
            and abs(manifest.get("train_ratio", args.train_ratio) - args.train_ratio) < 1e-9
            and manifest.get("stratification_target") == "last_tool_execution_result"
            and manifest.get("stratification_labels") == list(TOOL_EXECUTION_RESULT_LABELS)
            and manifest.get("total_label_counts") == serialize_tool_execution_result_counts(total_label_counts)
            and manifest.get("total_record_count") == total_record_count
        ):
            return manifest

    if gold_trajectory_count == 0:
        train_indices = []
        test_indices = []
        train_label_counts = empty_tool_execution_result_counts()
        test_label_counts = empty_tool_execution_result_counts()
        train_record_count = 0
        test_record_count = 0
    elif gold_trajectory_count == 1:
        train_indices = [groups[0]["split_index"]]
        test_indices = []
        train_label_counts = dict(groups[0]["label_counts"])
        test_label_counts = empty_tool_execution_result_counts()
        train_record_count = groups[0]["record_count"]
        test_record_count = 0
    else:
        target_train_group_count = int(round(gold_trajectory_count * args.train_ratio))
        target_train_group_count = max(1, min(gold_trajectory_count - 1, target_train_group_count))
        target_train_record_count = round(total_record_count * args.train_ratio)
        target_train_label_counts = {
            label: round(total_label_counts.get(label, 0) * args.train_ratio)
            for label in TOOL_EXECUTION_RESULT_LABELS
        }
        restart_count = 16
        best_assignment = None
        best_score = None

        for restart_index in range(restart_count):
            candidate = build_candidate_group_split(
                groups,
                seed=args.seed + restart_index,
                total_label_counts=total_label_counts,
                total_record_count=total_record_count,
                target_train_group_count=target_train_group_count,
                target_train_record_count=target_train_record_count,
                target_train_label_counts=target_train_label_counts,
            )
            candidate = improve_group_split(
                *candidate,
                total_label_counts=total_label_counts,
                total_record_count=total_record_count,
                target_train_group_count=target_train_group_count,
                target_train_record_count=target_train_record_count,
                target_train_label_counts=target_train_label_counts,
            )
            candidate_train_groups, candidate_test_groups, candidate_train_label_counts, candidate_train_record_count, candidate_score = candidate
            if best_score is None or candidate_score < best_score:
                best_assignment = candidate
                best_score = candidate_score

        train_groups, test_groups, train_label_counts, train_record_count, _ = best_assignment
        train_indices = sorted(group["split_index"] for group in train_groups)
        test_indices = sorted(group["split_index"] for group in test_groups)
        test_label_counts = empty_tool_execution_result_counts()
        test_record_count = 0
        for group in test_groups:
            add_tool_execution_result_counts(test_label_counts, group["label_counts"])
            test_record_count += group["record_count"]

    if gold_trajectory_count <= 1:
        test_label_counts = {
            label: total_label_counts.get(label, 0) - train_label_counts.get(label, 0)
            for label in TOOL_EXECUTION_RESULT_LABELS
        }
        test_record_count = total_record_count - train_record_count

    return {
        "data_path": str(args.gold_path.resolve()),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "trajectory_count": gold_trajectory_count,
        "train_trajectories": len(train_indices),
        "test_trajectories": len(test_indices),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "stratification_target": "last_tool_execution_result",
        "stratification_labels": list(TOOL_EXECUTION_RESULT_LABELS),
        "total_label_counts": serialize_tool_execution_result_counts(total_label_counts),
        "train_label_counts": serialize_tool_execution_result_counts(train_label_counts),
        "test_label_counts": serialize_tool_execution_result_counts(test_label_counts),
        "total_record_count": total_record_count,
        "train_record_count": train_record_count,
        "test_record_count": test_record_count,
        "grouped_batch_records": {str(key): value for key, value in sorted(grouped_batch_records.items())},
        "unmatched_batch_records": unmatched_batch_records,
    }


def prepare_llm_environment(method: str):
    local_vllm_methods = {
        "qwen3-vllm-9000",
        "qwen3-vllm-9001",
        "qwen3-vl-vllm-9002",
    }
    if method in local_vllm_methods and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "local-vllm-placeholder"


def make_default_tool_context():
    return {
        "last_tool_execution_result": 1,
        "last_tool_name": None,
        "last_tool_output": None,
        "error_message": None,
    }


def detect_api_error_text(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "api error",
        "mcperror",
        "error code",
        "traceback",
        "input validation error",
        "bad request",
        "please fix your mistakes",
        "rate limit",
        "maximum context length",
        "internal server error",
        "service unavailable",
        "tool returned no output",
    ]
    if lowered.startswith("error:"):
        return True
    return any(marker in lowered for marker in markers)


def is_explicit_tool_error(latest_tool_context: dict) -> bool:
    candidates = [
        latest_tool_context.get("error_message"),
        latest_tool_context.get("last_tool_output"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if not isinstance(candidate, str):
            candidate = json.dumps(candidate, ensure_ascii=False)
        parsed = try_parse_json(candidate)
        if isinstance(parsed, dict):
            if any(key in parsed for key in ("error", "errors")):
                return True
            status_code = parsed.get("status") or parsed.get("status_code")
            if isinstance(status_code, int) and status_code >= 400:
                return True
            if parsed.get("isError") is True:
                return True
        if detect_api_error_text(candidate):
            return True
    return False


def action_uses_tool(action_content) -> bool:
    return isinstance(action_content, dict) and bool(action_content.get("tool_calls"))


def classify_action_execution_result(
    action_content,
    latest_tool_context: dict,
    previous_process_state: dict | None,
    current_process_state: dict,
) -> int:
    previous_stage = summarize_text(
        (previous_process_state or {}).get("current_stage"),
        limit=120,
    )
    current_stage = summarize_text(current_process_state.get("current_stage"), limit=120)

    if action_uses_tool(action_content) and is_explicit_tool_error(latest_tool_context):
        return -1
    if current_stage != previous_stage:
        return 1
    return 0


def summarize_tool_result(tool_name, raw_output):
    context = {
        "last_tool_execution_result": 1,
        "last_tool_name": tool_name,
        "last_tool_output": summarize_text(raw_output),
        "error_message": None,
    }
    if raw_output is None:
        context["last_tool_execution_result"] = 0
        context["error_message"] = "Tool returned no output."
        return context

    parsed = try_parse_json(raw_output) if isinstance(raw_output, str) else raw_output
    if isinstance(parsed, dict):
        if any(key in parsed for key in ("error", "errors")):
            context["last_tool_execution_result"] = 0
            context["error_message"] = summarize_text(parsed.get("error") or parsed.get("errors"))
            return context
        status_code = parsed.get("status") or parsed.get("status_code")
        if isinstance(status_code, int) and status_code >= 400:
            context["last_tool_execution_result"] = 0
            context["error_message"] = summarize_text(parsed)
            return context

    text = raw_output if isinstance(raw_output, str) else json.dumps(raw_output, ensure_ascii=False)
    if detect_api_error_text(text):
        context["last_tool_execution_result"] = 0
        context["error_message"] = summarize_text(text)

    return context


def make_tool_call_action(tool_name, arguments):
    return {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arguments if arguments is not None else {},
                },
            }
        ]
    }


def normalize_gold_action(message):
    if message.get("tool_calls"):
        return {"tool_calls": message["tool_calls"]}
    return normalize_message_content(message.get("content"))


def normalize_batch_action(step):
    step_type = step.get("step_type")
    if step_type == "thought":
        parsed = try_parse_json(step.get("content"))
        if isinstance(parsed, dict) and parsed.get("thought"):
            return str(parsed["thought"]).strip()
        return normalize_message_content(step.get("content"))
    if step_type == "action":
        if step.get("tool_used"):
            return make_tool_call_action(step["tool_used"], step.get("tool_input"))
        return normalize_message_content(step.get("content"))
    return None


def normalize_batch_observation(step):
    tool_name = step.get("tool_used") or "unknown_tool"
    raw_output = step.get("tool_output") or step.get("content")
    return summarize_tool_result(tool_name, raw_output)


def prompt_stage_model(
    llm,
    stage_cache,
    cache_path,
    cache_tracker,
    args,
    *,
    system_prompt,
    user_prompt,
    action_history,
    latest_tool_context,
    previous_process_state,
    is_final_action,
):
    recent_actions = action_history[-args.max_history_actions :]
    recent_action_summaries = [summarize_action_for_stage(item, limit=180) for item in recent_actions]
    prompt_payload = {
        "system_prompt": summarize_text(system_prompt, limit=2500),
        "user_prompt": summarize_text(user_prompt, limit=1500),
        "recent_actions": recent_action_summaries,
        "latest_tool_context": latest_tool_context,
        "previous_process_state": previous_process_state,
        "is_final_action": is_final_action,
    }
    key = cache_key(prompt_payload)
    cached = stage_cache.get(key)
    if cached:
        return cached

    system_instruction = (
        "You infer the current process stage for a single-agent enterprise task. "
        "Return strict JSON with keys `remaining_stages` and `current_stage`. "
        "`remaining_stages` must be an ordered list of short actionable stages from the present moment onward. "
        "`current_stage` must exactly match one item in `remaining_stages`. "
        "Keep stage names concise and grounded in the task and recent actions. "
        "If previous stages are provided, you must reuse only those exact stage names. "
        "Do not rename stages, paraphrase them, or invent new stages. "
        "`remaining_stages` must be a contiguous suffix of the previous `remaining_stages`."
    )
    prompt = (
        "Task context:\n"
        f"- Agent system prompt: {prompt_payload['system_prompt']}\n"
        f"- User task: {prompt_payload['user_prompt']}\n"
        f"- Recent actions: {json.dumps(prompt_payload['recent_actions'], ensure_ascii=False)}\n"
        f"- Latest tool context: {json.dumps(prompt_payload['latest_tool_context'], ensure_ascii=False)}\n"
        f"- Is final action: {json.dumps(prompt_payload['is_final_action'], ensure_ascii=False)}\n"
        f"- Previous process state: {json.dumps(prompt_payload['previous_process_state'], ensure_ascii=False)}\n\n"
        "Generate the current stage plan now."
    )

    try:
        stage = llm.generate_format(
            prompt=prompt,
            system_prompt=system_instruction,
            format="json",
            schema=STAGE_SCHEMA,
        )
    except Exception:
        if not args.allow_stage_fallback:
            raise
        stage = None

    if not stage and args.allow_stage_fallback:
        if previous_process_state and previous_process_state.get("remaining_stages"):
            stage = previous_process_state
        else:
            last_action = recent_action_summaries[-1] if recent_action_summaries else summarize_text(user_prompt, limit=80)
            stage = {
                "remaining_stages": [
                    "Complete the active step",
                    "Verify the result",
                    "Deliver the final response",
                ],
                "current_stage": summarize_text(last_action, limit=80) or "Complete the active step",
            }
            if stage["current_stage"] not in stage["remaining_stages"]:
                stage["remaining_stages"][0] = stage["current_stage"]

    if not stage:
        raise RuntimeError("Stage generation returned no data.")

    remaining_stages = [
        summarize_text(item, limit=120)
        for item in stage.get("remaining_stages", [])
        if summarize_text(item, limit=120)
    ]
    current_stage = summarize_text(stage.get("current_stage"), limit=120)
    previous_remaining_stages = [
        summarize_text(item, limit=120)
        for item in (previous_process_state or {}).get("remaining_stages", [])
        if summarize_text(item, limit=120)
    ]
    previous_current_stage = summarize_text(
        (previous_process_state or {}).get("current_stage"),
        limit=120,
    )

    if is_final_action and latest_tool_context["last_tool_execution_result"] != 0:
        normalized_stage = {
            "remaining_stages": [],
            "current_stage": "finished",
        }
        stage_cache[key] = normalized_stage
        cache_tracker["pending_writes"] += 1
        flush_stage_cache(stage_cache, cache_path, cache_tracker)
        return normalized_stage

    if previous_remaining_stages:
        if current_stage not in previous_remaining_stages:
            current_stage = next(
                (item for item in remaining_stages if item in previous_remaining_stages),
                None,
            )
        if current_stage not in previous_remaining_stages:
            current_stage = (
                previous_current_stage
                if previous_current_stage in previous_remaining_stages
                else previous_remaining_stages[0]
            )
        remaining_stages = previous_remaining_stages[previous_remaining_stages.index(current_stage) :]
    else:
        if not remaining_stages:
            remaining_stages = [current_stage or "Complete the active step"]
        if not current_stage or current_stage not in remaining_stages:
            current_stage = remaining_stages[0]

    normalized_stage = {
        "remaining_stages": remaining_stages,
        "current_stage": current_stage,
    }
    stage_cache[key] = normalized_stage
    cache_tracker["pending_writes"] += 1
    flush_stage_cache(stage_cache, cache_path, cache_tracker)
    return normalized_stage


def build_state_message(
    llm,
    stage_cache,
    cache_path,
    cache_tracker,
    args,
    *,
    system_prompt,
    user_prompt,
    action_history,
    latest_tool_context,
    previous_process_state,
    is_final_action,
):
    current_action = action_history[-1] if action_history else None
    process_state = prompt_stage_model(
        llm=llm,
        stage_cache=stage_cache,
        cache_path=cache_path,
        cache_tracker=cache_tracker,
        args=args,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        action_history=action_history,
        latest_tool_context=latest_tool_context,
        previous_process_state=previous_process_state,
        is_final_action=is_final_action,
    )
    action_execution_result = classify_action_execution_result(
        current_action,
        latest_tool_context,
        previous_process_state,
        process_state,
    )
    explicit_tool_error = action_uses_tool(current_action) and is_explicit_tool_error(latest_tool_context)
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


def reconstruct_gold_trajectory(
    trajectory,
    *,
    gold_index,
    split,
    llm,
    stage_cache,
    cache_path,
    cache_tracker,
    args,
):
    messages = trajectory["messages"]
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    reconstructed = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    assistant_positions = [index for index, message in enumerate(messages) if message.get("role") == "assistant"]
    last_tool_context = make_default_tool_context()
    action_history = []
    previous_process_state = None

    for position_index, message_index in enumerate(assistant_positions):
        if message_index < 2:
            continue
        action_content = normalize_gold_action(messages[message_index])
        if not action_content:
            continue

        next_assistant_index = (
            assistant_positions[position_index + 1]
            if position_index + 1 < len(assistant_positions)
            else len(messages)
        )
        tool_messages = [
            message
            for message in messages[message_index + 1 : next_assistant_index]
            if message.get("role") == "tool"
        ]
        if tool_messages:
            latest_tool = tool_messages[-1]
            last_tool_context = summarize_tool_result(
                latest_tool.get("name") or "unknown_tool",
                latest_tool.get("content"),
            )

        action_history.append(action_content)
        reconstructed.append({"role": "action", "content": action_content})
        state_message, previous_process_state = build_state_message(
            llm=llm,
            stage_cache=stage_cache,
            cache_path=cache_path,
            cache_tracker=cache_tracker,
            args=args,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            action_history=action_history,
            latest_tool_context=last_tool_context,
            previous_process_state=previous_process_state,
            is_final_action=position_index == len(assistant_positions) - 1,
        )
        reconstructed.append(state_message)

    return cleanup_world_model_trajectory(
        {
            "trajectory_id": f"gold-{gold_index}",
            "source": "enterprise_arena_gold",
            "split": split,
            "task_index": gold_index,
            "messages": reconstructed,
        }
    )


def should_skip_batch_task(task):
    error_text = (task.get("error") or "").lower()
    return "rate limit" in error_text or "maximum context length" in error_text


def reconstruct_batch_trajectory(
    task,
    *,
    split,
    system_prompt,
    llm,
    stage_cache,
    cache_path,
    cache_tracker,
    args,
):
    user_prompt = task["query"]
    reconstructed = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    action_history = []
    last_tool_context = make_default_tool_context()
    previous_process_state = None
    trajectory_steps = task.get("trajectory") or []
    action_step_positions = [
        index
        for index, step in enumerate(trajectory_steps)
        if step.get("step_type") in {"thought", "action"}
    ]

    for position_index, step_index in enumerate(action_step_positions):
        step = trajectory_steps[step_index]
        action_content = normalize_batch_action(step)
        if not action_content:
            continue

        next_action_index = (
            action_step_positions[position_index + 1]
            if position_index + 1 < len(action_step_positions)
            else len(trajectory_steps)
        )
        observation_steps = [
            item
            for item in trajectory_steps[step_index + 1 : next_action_index]
            if item.get("step_type") == "observation"
        ]
        if observation_steps:
            last_tool_context = normalize_batch_observation(observation_steps[-1])

        action_history.append(action_content)
        reconstructed.append({"role": "action", "content": action_content})
        state_message, previous_process_state = build_state_message(
            llm=llm,
            stage_cache=stage_cache,
            cache_path=cache_path,
            cache_tracker=cache_tracker,
            args=args,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            action_history=action_history,
            latest_tool_context=last_tool_context,
            previous_process_state=previous_process_state,
            is_final_action=False,
        )
        reconstructed.append(state_message)

    final_answer = (task.get("final_answer") or "").strip()
    if final_answer:
        final_action_content = normalize_message_content(final_answer)
        action_history.append(final_action_content)
        reconstructed.append({"role": "action", "content": final_action_content})
        state_message, previous_process_state = build_state_message(
            llm=llm,
            stage_cache=stage_cache,
            cache_path=cache_path,
            cache_tracker=cache_tracker,
            args=args,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            action_history=action_history,
            latest_tool_context=last_tool_context,
            previous_process_state=previous_process_state,
            is_final_action=task.get("status") != "error" or not task.get("error"),
        )
        reconstructed.append(state_message)

    if task.get("status") == "error" and task.get("error"):
        last_tool_context = {
            "last_tool_execution_result": 0,
            "last_tool_name": last_tool_context.get("last_tool_name"),
            "last_tool_output": summarize_text(task["error"]),
            "error_message": summarize_text(task["error"]),
        }
        error_action_content = normalize_message_content(task["error"])
        action_history.append(error_action_content)
        reconstructed.append({"role": "action", "content": error_action_content})
        state_message, previous_process_state = build_state_message(
            llm=llm,
            stage_cache=stage_cache,
            cache_path=cache_path,
            cache_tracker=cache_tracker,
            args=args,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            action_history=action_history,
            latest_tool_context=last_tool_context,
            previous_process_state=previous_process_state,
            is_final_action=True,
        )
        reconstructed.append(state_message)

    return cleanup_world_model_trajectory(
        {
            "trajectory_id": f"batch-{task['task_index']}",
            "source": "enterprise_arena_batch",
            "split": split,
            "task_index": task["task_index"],
            "messages": reconstructed,
        }
    )


def batch_split_for_task(task_index, split_manifest, batch_task_index_base):
    split_index = task_index - batch_task_index_base
    train_indices = set(split_manifest["train_indices"])
    return "train" if split_index in train_indices else "test"


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / args.stage_cache

    gold_trajectories = load_json(args.gold_path.resolve())

    stage_cache = load_stage_cache(cache_path)
    cache_tracker = {"pending_writes": 0, "flush_every": 25}
    prepare_llm_environment(args.llm_method)
    from src.llm import LLM
    llm = LLM(args.llm_method)

    reconstructed_gold_records = []
    reconstructed_batch_records = []

    default_system_prompt = gold_trajectories[0]["messages"][0]["content"]

    for gold_index, trajectory in tqdm(enumerate(gold_trajectories), desc="Processing gold trajectories", total=len(gold_trajectories)):
        reconstructed = reconstruct_gold_trajectory(
            trajectory,
            gold_index=gold_index,
            split=None,
            llm=llm,
            stage_cache=stage_cache,
            cache_path=cache_path,
            cache_tracker=cache_tracker,
            args=args,
        )
        reconstructed_gold_records.append((gold_index, reconstructed))
        if args.max_trajectories and len(reconstructed_gold_records) >= args.max_trajectories:
            break

    processed_batch = 0
    skipped_batch = []
    if not args.max_trajectories:
        for batch_path in sorted(glob.glob(args.batch_glob)):
            batch_data = load_json(Path(batch_path))
            for task in tqdm(batch_data.get("tasks", []), desc="Processing batch tasks", total=len(batch_data.get("tasks", []))):
                if should_skip_batch_task(task):
                    skipped_batch.append(
                        {
                            "task_index": task.get("task_index"),
                            "reason": summarize_text(task.get("error")),
                        }
                    )
                    continue
                reconstructed = reconstruct_batch_trajectory(
                    task,
                    split=None,
                    system_prompt=default_system_prompt,
                    llm=llm,
                    stage_cache=stage_cache,
                    cache_path=cache_path,
                    cache_tracker=cache_tracker,
                    args=args,
                )
                reconstructed_batch_records.append(
                    {
                        "task_index": task["task_index"],
                        "trajectory": reconstructed,
                    }
                )
                processed_batch += 1

    split_group_stats = build_split_groups(
        gold_trajectories=[trajectory for _, trajectory in reconstructed_gold_records],
        batch_records=reconstructed_batch_records,
        batch_task_index_base=args.batch_task_index_base,
    )
    split_manifest = build_split(split_group_stats, args)
    train_indices = set(split_manifest["train_indices"])

    train_payload = []
    test_payload = []

    for gold_index, reconstructed in reconstructed_gold_records:
        split = "train" if gold_index in train_indices else "test"
        reconstructed["split"] = split
        if split == "train":
            train_payload.append(reconstructed)
        else:
            test_payload.append(reconstructed)

    for batch_record in reconstructed_batch_records:
        split = batch_split_for_task(
            batch_record["task_index"],
            split_manifest=split_manifest,
            batch_task_index_base=args.batch_task_index_base,
        )
        batch_record["trajectory"]["split"] = split
        if split == "train":
            train_payload.append(batch_record["trajectory"])
        else:
            test_payload.append(batch_record["trajectory"])

    train_output_path = output_dir / args.train_output
    test_output_path = output_dir / args.test_output
    generated_split_manifest_path = output_dir / args.generated_split_manifest

    dump_json(train_output_path, train_payload)
    dump_json(test_output_path, test_payload)
    flush_stage_cache(stage_cache, cache_path, cache_tracker, force=True)
    dump_json(
        generated_split_manifest_path,
        {
            **split_manifest,
            "batch_task_index_base": args.batch_task_index_base,
            "gold_train_records": sum(item["source"] == "enterprise_arena_gold" for item in train_payload),
            "gold_test_records": sum(item["source"] == "enterprise_arena_gold" for item in test_payload),
            "batch_train_records": sum(item["source"] == "enterprise_arena_batch" for item in train_payload),
            "batch_test_records": sum(item["source"] == "enterprise_arena_batch" for item in test_payload),
            "processed_batch_records": processed_batch,
            "skipped_batch_records": skipped_batch,
            "llm_method": args.llm_method,
        },
    )

    print(f"Wrote train trajectories to {train_output_path}")
    print(f"Wrote test trajectories to {test_output_path}")
    print(f"Wrote split manifest to {generated_split_manifest_path}")
    print(f"Train label counts: {split_manifest['train_label_counts']}")
    print(f"Test label counts: {split_manifest['test_label_counts']}")
    print(f"Stage cache entries: {len(stage_cache)}")


if __name__ == "__main__":
    main()
