"""Convert TOUCAN multi-turn enterprise trajectories into the lean world-model
trajectory format consumed by `src/finetuning.py`.

Mirrors the EnterpriseOps-Gym generator's structure: each TOUCAN record becomes
a list of `{system, user, action, state, ...}` messages, where `action` carries
OpenAI-style `tool_calls` and `state` carries an EWM state dict (six aspects:
agent / context / process / relational / temporal). Multi-turn user prompts in
the source are preserved as interleaved `user` messages so downstream training
sees the full conversational structure.

State labelling here is heuristic by default — there is no LLM-driven stage
generator, since TOUCAN's enterprise-domain field is empty and stage induction
would be guesswork. Pass `--llm-method` to opt in to the same stage-cache
machinery used by the EnterpriseOps-Gym generator.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random  # noqa: E402

from src.generation.generate_world_model_trajectories import (  # noqa: E402
    extract_agent_role,
    summarize_text,
)
from src.data_preparation.world_model_trajectory_cleanup import cleanup_world_model_trajectory  # noqa: E402


DEFAULT_TOUCAN_PATH = (
    Path.home() / "program" / "tools" / "Toucan" / "enterprise_trajectories_multi_turn_e5.jsonl"
)
DEFAULT_OUTPUT_PATH = ROOT / "trajectories" / "toucan_world_model_trajectories.json"
DEFAULT_TRAIN_OUTPUT_PATH = ROOT / "trajectories" / "toucan_world_model_train_trajectories.json"
DEFAULT_TEST_OUTPUT_PATH = ROOT / "trajectories" / "toucan_world_model_test_trajectories.json"
DEFAULT_SPLIT_MANIFEST_PATH = (
    ROOT / "trajectories" / "toucan_world_model_trajectory_split_manifest.json"
)

CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}

TOOL_DECLARE_PATTERN = re.compile(
    r"<\|im_system\|>\s*tool_declare\s*<\|im_middle\|>\s*(\[.*\])\s*(?:<\|im_end\|>\s*)?$",
    re.DOTALL,
)

EXPLICIT_FAILURE_MARKERS = (
    "an error occurred when calling tool",
    "mcperror",
    "traceback (most recent call last)",
    "exception:",
    "error:",
    "failed:",
)
STAGNATION_MARKERS = (
    "no result",
    "no results",
    "not found",
    # Chinese "not found" phrasings that appear verbatim in Toucan tool outputs; these are
    # data patterns matched against observations, not prose.
    "未找到",
    "没有找到",
    "未能找到",
    "no matching",
    "nothing found",
    "empty result",
    "no records",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate TOUCAN trajectories in the lean world-model format defined "
            "in instructions/TRAJECTORIES.md."
        )
    )
    parser.add_argument("--toucan-path", type=Path, default=DEFAULT_TOUCAN_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--train-output-path", type=Path, default=DEFAULT_TRAIN_OUTPUT_PATH)
    parser.add_argument("--test-output-path", type=Path, default=DEFAULT_TEST_OUTPUT_PATH)
    parser.add_argument(
        "--generated-split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH
    )
    parser.add_argument(
        "--min-confidence",
        choices=("high", "medium", "low"),
        default="high",
        help="Lowest TOUCAN enterprise-label confidence bucket to include.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on the number of TOUCAN records to convert.",
    )
    parser.add_argument(
        "--require-tool-calls",
        action="store_true",
        help="Skip TOUCAN records that contain no tool calls (no action/state to learn from).",
    )
    parser.add_argument(
        "--min-action-count",
        type=int,
        default=1,
        help=(
            "Drop converted trajectories with fewer than this many action/observation pairs. "
            "The default of 1 removes single-turn chat records that contain no tool call at "
            "all (19,000 of 150,000 in toucan_enterprise.jsonl -- one human turn plus one gpt "
            "answer, no transition to learn from). Pass 2 for the stricter multi-round "
            "definition used by the toucan_1_5m_multiturn extract, or 0 to keep everything."
        ),
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=2000,
        help="Truncate `last_tool_output` text written into state messages.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument(
        "--reuse-split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_PATH,
        help="Reuse an existing TOUCAN split manifest when it matches the reconstructed counts.",
    )
    parser.add_argument(
        "--force-new-split",
        action="store_true",
        help="Ignore any existing TOUCAN split manifest and generate a fresh stratified split.",
    )
    return parser.parse_args()


def parse_tool_declarations(system_content: str) -> list[dict[str, Any]]:
    """Pull tool schemas out of TOUCAN's `<|im_system|>tool_declare<|im_middle|>` block."""
    if not system_content:
        return []
    match = TOOL_DECLARE_PATTERN.search(system_content)
    raw = match.group(1) if match else None
    if raw is None:
        # Fallback: find the first JSON-array bracket and try to parse it.
        bracket_start = system_content.find("[")
        if bracket_start == -1:
            return []
        raw = system_content[bracket_start:]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def build_system_prompt(tool_declarations: list[dict[str, Any]]) -> str:
    lines = [
        "You are an enterprise operations assistant.",
        "Use the available tools to fulfil the user's request, and return clear answers when finished.",
    ]
    if tool_declarations:
        lines.append("")
        lines.append("Available tools:")
        for entry in tool_declarations:
            fn = entry.get("function", {}) if isinstance(entry, dict) else {}
            name = fn.get("name", "")
            description = (fn.get("description") or "").strip()
            if not name:
                continue
            description_summary = summarize_text(description, limit=240)
            lines.append(f"- {name}: {description_summary}")
    return "\n".join(lines)


def parse_function_call_arguments(raw_arguments: Any) -> dict[str, Any]:
    if not raw_arguments:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {"__raw_arguments__": raw_arguments}
        return parsed if isinstance(parsed, dict) else {"__value__": parsed}
    return {"__value__": raw_arguments}


def classify_function_result(content: str) -> tuple[int, str]:
    """Map a TOUCAN `function`-role content blob to (last_tool_execution_result, error_text)."""
    text = (content or "").strip()
    if not text:
        return 0, ""
    lowered = text.lower()
    for marker in EXPLICIT_FAILURE_MARKERS:
        if marker in lowered:
            return -1, text
    for marker in STAGNATION_MARKERS:
        if marker in lowered:
            return 0, ""
    return 1, ""


def build_action_message(function_call: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "action",
        "content": {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": function_call.get("name", ""),
                        "arguments": parse_function_call_arguments(
                            function_call.get("arguments")
                        ),
                    },
                }
            ]
        },
    }


def build_state_message(
    *,
    system_prompt: str,
    last_tool_name: str,
    last_tool_output: str,
    execution_result: int,
    error_message: str,
    step_index: int,
    total_steps: int,
    max_output_chars: int,
) -> dict[str, Any]:
    is_final = step_index >= total_steps
    state = {
        "agent": {"role": extract_agent_role(system_prompt) or "Toucan agent"},
        "context": {
            "last_tool_execution_result": execution_result,
            "last_tool_name": last_tool_name,
            "last_tool_output": summarize_text(last_tool_output, limit=max_output_chars),
        },
        "process": {
            "remaining_stages": [] if is_final else ["finalize"],
            "current_stage": "finalize" if is_final else "execute",
            "completed_stages": [],
            "blocking_conditions": [],
        },
        "relational": {"permission_level": "agent"},
        "temporal": {
            "step_index": step_index,
            "total_steps": total_steps,
            "is_final_action": is_final,
        },
    }
    if execution_result == -1 and error_message:
        state["context"]["error_message"] = summarize_text(error_message, limit=800)
    return {"role": "state", "content": {"state": state}}


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def strip_tag_blocks(text: str, pattern: re.Pattern) -> str:
    return pattern.sub("", text or "").strip()


def convert_record_conversations(
    record: dict[str, Any],
    *,
    require_tool_calls: bool,
    max_output_chars: int,
) -> dict[str, Any] | None:
    """Convert a raw TOUCAN/ShareGPT record -- {id, system, conversations:[{from, value}]} --
    into the lean world-model trajectory format.

    This is the shape of trajectories/toucan_enterprise*.jsonl, which differs from the
    `record["trajectory"]["messages"]` shape convert_record() handles: actions are
    <tool_call> blocks inside a `gpt` turn and observations are <tool_response> blocks in the
    following `human` turn, rather than assistant.function_call / function-role messages.

    A `gpt` turn may batch several <tool_call> blocks (parallel calls) and the answering
    `human` turn several <tool_response> blocks; they are zipped positionally. Calls with no
    matching response (the conversation was truncated mid-execution -- 6.4%% of calls in
    toucan_enterprise.jsonl) are dropped rather than paired with a fabricated observation.
    """
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None
    system_prompt = record.get("system") or ""

    body: list[dict[str, Any]] = []
    action_step_count = 0
    pending_calls: list[dict[str, Any]] = []

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        speaker = turn.get("from")
        value = turn.get("value") or ""

        if speaker == "gpt":
            calls = []
            for raw in TOOL_CALL_RE.findall(value):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and parsed.get("name"):
                    calls.append(parsed)
            narration = strip_tag_blocks(value, TOOL_CALL_RE)
            if narration and not calls:
                body.append({"role": "assistant", "content": narration})
            pending_calls = calls
            continue

        if speaker == "human":
            responses = TOOL_RESPONSE_RE.findall(value)
            if not responses:
                text = strip_tag_blocks(value, TOOL_RESPONSE_RE)
                if text:
                    body.append({"role": "user", "content": text})
                pending_calls = []
                continue
            for call, response_text in zip(pending_calls, responses):
                execution_result, error_text = classify_function_result(response_text)
                body.append(build_action_message(call))
                action_step_count += 1
                body.append(
                    build_state_message(
                        system_prompt=system_prompt,
                        last_tool_name=call.get("name", ""),
                        last_tool_output=response_text,
                        execution_result=execution_result,
                        error_message=error_text,
                        step_index=action_step_count,
                        total_steps=0,          # patched below, as in convert_record()
                        max_output_chars=max_output_chars,
                    )
                )
            pending_calls = []

    if require_tool_calls and action_step_count == 0:
        return None
    if not body:
        return None

    for entry in body:
        if entry.get("role") == "state":
            content = entry["content"]["state"]
            content["temporal"]["total_steps"] = action_step_count
            content["temporal"]["is_final_action"] = (
                content["temporal"]["step_index"] >= action_step_count
            )
            if content["temporal"]["is_final_action"]:
                content["process"]["remaining_stages"] = []
                content["process"]["current_stage"] = "finalize"

    messages = [{"role": "system", "content": system_prompt}] + body
    trajectory = {
        "trajectory_id": str(record.get("id") or record.get("record_index") or ""),
        "messages": messages,
        "action_count": action_step_count,
    }
    return cleanup_world_model_trajectory(trajectory)


def convert_record(
    record: dict[str, Any],
    *,
    require_tool_calls: bool,
    max_output_chars: int,
) -> dict[str, Any] | None:
    raw_messages = record.get("trajectory", {}).get("messages") or []
    if not raw_messages:
        return None

    system_msg = next((m for m in raw_messages if m.get("role") == "system"), None)
    tool_declarations = parse_tool_declarations(system_msg.get("content", "") if system_msg else "")
    system_prompt = build_system_prompt(tool_declarations)

    pending_function_calls: list[dict[str, Any]] = []
    body: list[dict[str, Any]] = []
    action_step_count = 0

    for msg in raw_messages:
        role = msg.get("role")
        if role == "system":
            continue
        if role == "user":
            content = (msg.get("content") or "").strip()
            if content:
                body.append({"role": "user", "content": content})
            continue
        if role == "assistant":
            function_call = msg.get("function_call")
            content = (msg.get("content") or "").strip()
            if function_call:
                pending_function_calls.append(function_call)
            elif content:
                body.append({"role": "assistant", "content": content})
            continue
        if role == "function":
            if not pending_function_calls:
                # Stray tool result with no matching call — skip rather than crash.
                continue
            function_call = pending_function_calls.pop(0)
            result_text = msg.get("content", "") or ""
            execution_result, error_text = classify_function_result(result_text)
            body.append(build_action_message(function_call))
            action_step_count += 1
            body.append(
                build_state_message(
                    system_prompt=system_prompt,
                    last_tool_name=function_call.get("name", ""),
                    last_tool_output=result_text,
                    execution_result=execution_result,
                    error_message=error_text,
                    step_index=action_step_count,
                    total_steps=0,  # patched after the loop
                    max_output_chars=max_output_chars,
                )
            )
            continue
        # Unknown role — skip silently.

    if require_tool_calls and action_step_count == 0:
        return None

    # Drop unmatched calls (the conversation was truncated mid-execution); they
    # would otherwise produce action messages with no observable outcome.
    # Patch total_steps now that we know it.
    total_steps = action_step_count
    for entry in body:
        if entry.get("role") == "state":
            content = entry["content"]["state"]
            content["temporal"]["total_steps"] = total_steps
            content["temporal"]["is_final_action"] = (
                content["temporal"]["step_index"] >= total_steps
            )
            if content["temporal"]["is_final_action"]:
                content["process"]["remaining_stages"] = []
                content["process"]["current_stage"] = "finalize"

    if not body:
        return None

    label = record.get("enterprise_label") or {}
    trajectory = {
        "trajectory_id": f"toucan-world-model-{record.get('uuid') or record.get('record_index')}",
        "source": "toucan_multi_turn_e5",
        "domain": "toucan",
        "seed_id": f"import.toucan.{record.get('uuid') or record.get('record_index')}",
        "subset_name": record.get("subset_name", ""),
        "record_index": record.get("record_index"),
        "uuid": record.get("uuid"),
        "enterprise_label_confidence": label.get("confidence"),
        "enterprise_label_probability": label.get("probability_enterprise"),
        "requested_mcp_servers": record.get("requested_mcp_servers") or [],
        "matched_mcp_servers": record.get("matched_mcp_servers") or [],
        "target_tools": record.get("target_tools") or [],
        "messages": [{"role": "system", "content": system_prompt}] + body,
    }
    return cleanup_world_model_trajectory(trajectory)


def iter_toucan_records(path: Path, min_confidence: str):
    threshold = CONFIDENCE_RANK.get(min_confidence, 3)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = record.get("enterprise_label")
            # toucan_enterprise*.jsonl is pre-filtered to enterprise applications upstream and
            # carries no enterprise_label; only apply the confidence gate to records that have
            # one, otherwise every such record would be silently dropped.
            if isinstance(label, dict) and label:
                if CONFIDENCE_RANK.get(label.get("confidence", "low"), 0) < threshold:
                    continue
            yield record


def build_toucan_split(trajectories: list[dict], args) -> dict:
    """Stratified train/test split keyed by `enterprise_label_confidence`.

    The TOUCAN dataset doesn't expose enterprise domains, so we stratify on the
    classifier confidence bucket so each bucket appears in train and test in
    roughly the requested ratio. The manifest is reproducible by seed.
    """
    if (
        not args.force_new_split
        and args.reuse_split_manifest
        and args.reuse_split_manifest.exists()
    ):
        with args.reuse_split_manifest.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("trajectory_count") == len(trajectories)
            and manifest.get("train_ratio") == args.train_ratio
        ):
            return manifest

    rng = random.Random(args.seed)
    bucketed: dict[str, list[str]] = {}
    for trajectory in trajectories:
        bucket = trajectory.get("enterprise_label_confidence") or "unknown"
        bucketed.setdefault(bucket, []).append(trajectory["trajectory_id"])

    train_ids: list[str] = []
    test_ids: list[str] = []
    bucket_counts: dict[str, dict[str, int]] = {}
    for bucket, ids in bucketed.items():
        rng.shuffle(ids)
        cutoff = max(1, int(round(len(ids) * args.train_ratio))) if len(ids) > 1 else len(ids)
        bucket_train = ids[:cutoff]
        bucket_test = ids[cutoff:]
        train_ids.extend(bucket_train)
        test_ids.extend(bucket_test)
        bucket_counts[bucket] = {
            "total": len(ids),
            "train": len(bucket_train),
            "test": len(bucket_test),
        }

    label_counts: Counter = Counter()
    for trajectory in trajectories:
        for message in trajectory["messages"]:
            if message.get("role") != "state":
                continue
            label = (
                message["content"]
                .get("state", {})
                .get("context", {})
                .get("last_tool_execution_result")
            )
            label_counts[label] += 1

    return {
        "trajectory_count": len(trajectories),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "stratification_target": "enterprise_label_confidence",
        "bucket_counts": bucket_counts,
        "label_counts": {str(k): v for k, v in sorted(label_counts.items(), key=lambda kv: str(kv[0]))},
        "train_ids": sorted(train_ids),
        "test_ids": sorted(test_ids),
    }


def main():
    args = parse_args()
    if not args.toucan_path.exists():
        raise SystemExit(f"TOUCAN file not found: {args.toucan_path}")

    trajectories: list[dict] = []
    skipped = 0
    skipped_below_min = 0
    for index, record in enumerate(iter_toucan_records(args.toucan_path, args.min_confidence)):
        if args.max_records is not None and len(trajectories) >= args.max_records:
            break
        try:
            converter = (
                convert_record_conversations
                if isinstance(record.get("conversations"), list)
                else convert_record
            )
            converted = converter(
                record,
                require_tool_calls=args.require_tool_calls,
                max_output_chars=args.max_output_chars,
            )
        except (KeyError, TypeError, ValueError) as exc:
            skipped += 1
            print(f"  skipped record_index={record.get('record_index')}: {exc}")
            continue
        if converted is None:
            skipped += 1
            continue
        if converted.get("action_count", 0) < args.min_action_count:
            skipped_below_min += 1
            continue
        trajectories.append(converted)

    if not trajectories:
        raise SystemExit(
            "No TOUCAN trajectories were produced; check --min-confidence and --require-tool-calls."
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(trajectories, handle, ensure_ascii=False, indent=2)
    print(
        f"Wrote {len(trajectories)} TOUCAN trajectories to {args.output_path} "
        f"(skipped {skipped} unconvertible, {skipped_below_min} below "
        f"--min-action-count={args.min_action_count})"
    )

    split_payload = build_toucan_split(trajectories, args)
    train_id_set = set(split_payload.get("train_ids", []))
    test_id_set = set(split_payload.get("test_ids", []))

    train_records = []
    test_records = []
    for trajectory in trajectories:
        traj_id = trajectory["trajectory_id"]
        if traj_id in test_id_set:
            test_records.append(trajectory)
        elif traj_id in train_id_set:
            train_records.append(trajectory)
        else:
            train_records.append(trajectory)

    args.train_output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.train_output_path.open("w", encoding="utf-8") as handle:
        json.dump(train_records, handle, ensure_ascii=False, indent=2)
    with args.test_output_path.open("w", encoding="utf-8") as handle:
        json.dump(test_records, handle, ensure_ascii=False, indent=2)
    args.generated_split_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.generated_split_manifest.open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, ensure_ascii=False, indent=2)
    print(
        f"Train: {len(train_records)} → {args.train_output_path}\n"
        f"Test:  {len(test_records)} → {args.test_output_path}\n"
        f"Split manifest → {args.generated_split_manifest}"
    )


if __name__ == "__main__":
    main()
