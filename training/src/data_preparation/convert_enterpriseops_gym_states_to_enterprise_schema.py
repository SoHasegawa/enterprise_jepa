#!/usr/bin/env python3
"""Convert EnterpriseOps-Gym state messages to an enterprise state schema.

The converted state expands the O/P/R/C/H shorthand into descriptive keys and
intentionally omits temporal state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm import ClousedSourceLLM  # noqa: E402


DEFAULT_TRAIN_INPUT = ROOT / "trajectories" / "enterpriseops_gym_multi_model_world_model_train_trajectories.json"
DEFAULT_TEST_INPUT = ROOT / "trajectories" / "enterpriseops_gym_multi_model_world_model_test_trajectories.json"
DEFAULT_TRAIN_OUTPUT = (
    ROOT / "trajectories" / "enterpriseops_gym_multi_model_world_model_enterprise_state_train_trajectories.json"
)
DEFAULT_TEST_OUTPUT = (
    ROOT / "trajectories" / "enterpriseops_gym_multi_model_world_model_enterprise_state_test_trajectories.json"
)
DEFAULT_CACHE_PATH = ROOT / "trajectories" / "enterpriseops_gym_multi_model_enterprise_state_cache.json"


ENTERPRISE_STATE_SCHEMA = {
    "outcome": {
        "status": "",
        "summary": "",
        "failure_category": "",
        "recoverable": True,
    },
    "objects_artifacts": {
        "objects": [
            {
                "type": "",
                "id": "",
                "name": "",
                "status": "",
                "fields": {},
            }
        ],
        "created": [""],
        "updated": [""],
        "deleted": [""],
        "found": [""],
    },
    "process_state": {
        "stage": "",
        "completed_requirements": [""],
        "remaining_requirements": [""],
        "owner": "",
        "queue": "",
        "approval_status": "",
        "blockers": [""],
    },
    "relational_state": {
        "permissions": [{}],
        "dependencies": [{}],
        "assignments": [{}],
        "communication_links": [{}],
    },
    "constraints": {
        "satisfied": [""],
        "violated": [""],
        "active_constraints": [""],
        "access_control": [{}],
        "sla": [{}],
    },
    "history_context": {
        "salient_facts": [""],
        "prior_decisions": [""],
        "unresolved_assumptions": [""],
        "last_tool_events": [
            {
                "tool_name": "",
                "status": "",
                "operation": "",
                "summary": "",
                "error": "",
            }
        ],
    },
}

VALID_OUTCOME_STATUSES = {"success", "partial_success", "no_progress", "failure"}
VALID_FAILURE_CATEGORIES = {
    "none",
    "validation_error",
    "not_found",
    "permission_denied",
    "duplicate",
    "policy_violation",
    "schema_error",
    "empty_result",
    "semantic_mismatch",
    "unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert state messages in EnterpriseOps-Gym multi-model train/test trajectories "
            "to an expanded enterprise state schema without temporal state."
        )
    )
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_OUTPUT)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument(
        "--llm-methods",
        default="gpt5,claude,gemini",
        help="Comma-separated provider order passed to src.llm.ClousedSourceLLM.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-chars", type=int, default=5000)
    parser.add_argument("--max-history-messages", type=int, default=6)
    parser.add_argument("--limit-trajectories", type=int, default=None)
    parser.add_argument(
        "--state-mode",
        choices=["diff", "full"],
        default="diff",
        help="Emit the first state as full schema and later states as diffs by default.",
    )
    parser.add_argument(
        "--splits",
        choices=["train", "test", "both"],
        default="both",
        help="Which input files to convert.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse cached state conversions from --cache-path.",
    )
    parser.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        help="Emit a conservative non-LLM conversion if the LLM call fails.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def summarize_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    value = " ".join(value.split()).strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def compact_json(value: Any, limit: int) -> str:
    return summarize_text(json.dumps(value, ensure_ascii=False, sort_keys=True), limit)


def load_cache(path: Path, resume: bool) -> dict[str, Any]:
    if not resume or not path.exists():
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def build_llm(args: argparse.Namespace) -> ClousedSourceLLM:
    methods = [item.strip() for item in args.llm_methods.split(",") if item.strip()]
    return ClousedSourceLLM(methods=methods or None)


def cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def extract_state_payload(state_message: dict[str, Any]) -> dict[str, Any]:
    content = state_message.get("content")
    if not isinstance(content, dict):
        return {}
    nested = content.get("state")
    if isinstance(nested, dict):
        return nested
    return content


def extract_context(state_message: dict[str, Any]) -> dict[str, Any]:
    state = extract_state_payload(state_message)
    context = state.get("context") if isinstance(state, dict) else None
    if isinstance(context, dict):
        return context
    if isinstance(state, dict):
        return {
            "last_tool_execution_result": state.get("last_tool_execution_result"),
            "last_tool_name": state.get("last_tool_name"),
            "last_tool_output": state.get("last_tool_output"),
            "error_message": state.get("error_message"),
        }
    return {}


def extract_process(state_message: dict[str, Any]) -> dict[str, Any]:
    state = extract_state_payload(state_message)
    process = state.get("process") if isinstance(state, dict) else None
    return process if isinstance(process, dict) else {}


def status_from_label(label: Any) -> str:
    try:
        numeric = int(label)
    except Exception:
        return "no_progress"
    if numeric == 1:
        return "success"
    if numeric == -1:
        return "failure"
    return "no_progress"


def infer_failure_category(context: dict[str, Any]) -> str:
    text = " ".join(
        str(context.get(key) or "")
        for key in ("error_message", "last_tool_output")
    ).lower()
    if not text:
        return "unknown"
    if "permission" in text or "unauthorized" in text or "forbidden" in text or "access denied" in text:
        return "permission_denied"
    if "not found" in text or "no matching" in text or "no result" in text or "404" in text:
        return "not_found"
    if "duplicate" in text or "already exists" in text:
        return "duplicate"
    if "invalid tool arguments" in text or "required" in text or "validation" in text or "400" in text:
        return "validation_error"
    if "schema" in text or "nonetype" in text:
        return "schema_error"
    if "empty" in text or "no output" in text:
        return "empty_result"
    return "unknown"


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set)):
        return all(is_empty_value(item) for item in value)
    if isinstance(value, dict):
        return all(is_empty_value(item) for item in value.values())
    return False


def normalize_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict) and not is_empty_value(item)]


def merge_enterprise_state_sections(normalized: dict[str, Any], state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    for key in normalized:
        if isinstance(state.get(key), dict):
            normalized[key].update(state[key])


def normalize_outcome(outcome: dict[str, Any], context: dict[str, Any]) -> None:
    if outcome.get("status") not in VALID_OUTCOME_STATUSES:
        outcome["status"] = status_from_label(context.get("last_tool_execution_result"))
    if outcome.get("failure_category") not in VALID_FAILURE_CATEGORIES:
        outcome["failure_category"] = "none" if outcome["status"] == "success" else infer_failure_category(context)
    if not outcome.get("summary"):
        outcome["summary"] = summarize_text(context.get("last_tool_output"), 320)
    outcome["recoverable"] = bool(outcome.get("recoverable", outcome["status"] != "success"))


def normalize_objects_artifacts(objects: dict[str, Any]) -> None:
    objects["objects"] = normalize_dict_list(objects.get("objects"))
    for key in ("created", "updated", "deleted", "found"):
        objects[key] = normalize_string_list(objects.get(key))


def normalize_process_state(proc: dict[str, Any], process: dict[str, Any], fallback_stage: str | None) -> None:
    if not proc.get("stage"):
        proc["stage"] = str(process.get("current_stage") or "")
    if fallback_stage and str(proc.get("stage") or "").strip().lower() == "finished":
        proc["stage"] = fallback_stage
    if not proc.get("remaining_requirements"):
        proc["remaining_requirements"] = normalize_string_list(process.get("remaining_stages"))
    for key in ("completed_requirements", "remaining_requirements", "blockers"):
        proc[key] = normalize_string_list(proc.get(key))
    if fallback_stage and not proc["remaining_requirements"]:
        proc["remaining_requirements"] = [fallback_stage]


def normalize_relational_state(relational: dict[str, Any]) -> None:
    for key in ("permissions", "dependencies", "assignments", "communication_links"):
        relational[key] = normalize_dict_list(relational.get(key))


def normalize_constraints(constraints: dict[str, Any]) -> None:
    for key in ("satisfied", "violated", "active_constraints"):
        constraints[key] = normalize_string_list(constraints.get(key))
    for key in ("access_control", "sla"):
        constraints[key] = normalize_dict_list(constraints.get(key))


def normalize_history_context(history: dict[str, Any], context: dict[str, Any], status: str) -> None:
    for key in ("salient_facts", "prior_decisions", "unresolved_assumptions"):
        history[key] = normalize_string_list(history.get(key))
    history["last_tool_events"] = normalize_dict_list(history.get("last_tool_events"))
    if history["last_tool_events"]:
        return
    history["last_tool_events"] = [
        {
            "tool_name": str(context.get("last_tool_name") or ""),
            "status": status,
            "operation": "",
            "summary": summarize_text(context.get("last_tool_output"), 240),
            "error": str(context.get("error_message") or "") if status != "success" else "",
        }
    ]


def is_failed_final_step(context: dict[str, Any], process: dict[str, Any]) -> bool:
    return (
        status_from_label(context.get("last_tool_execution_result")) != "success"
        and str(process.get("current_stage") or "").strip().lower() == "finished"
    )


def previous_non_finished_stage(messages: list[dict[str, Any]], message_index: int) -> str:
    for index in range(message_index - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "state":
            continue
        process = extract_process(message)
        stage = str(process.get("current_stage") or "").strip()
        if stage and stage.lower() != "finished":
            return stage
    action = messages[message_index - 1].get("content") if message_index > 0 else None
    if isinstance(action, dict):
        tool_calls = action.get("tool_calls") or []
        names = [
            ((call.get("function") or {}).get("name") or "").replace("_", " ")
            for call in tool_calls
            if isinstance(call, dict)
        ]
        names = [name for name in names if name]
        if names:
            return "; ".join(names)
    return "retry or repair failed action"


def normalize_enterprise_state(
    state: dict[str, Any],
    context: dict[str, Any],
    process: dict[str, Any],
    *,
    fallback_stage: str | None = None,
) -> dict[str, Any]:
    normalized = copy.deepcopy(ENTERPRISE_STATE_SCHEMA)
    merge_enterprise_state_sections(normalized, state)

    outcome = normalized["outcome"]
    normalize_outcome(outcome, context)

    normalize_objects_artifacts(normalized["objects_artifacts"])
    normalize_process_state(normalized["process_state"], process, fallback_stage)
    normalize_relational_state(normalized["relational_state"])
    normalize_constraints(normalized["constraints"])
    normalize_history_context(normalized["history_context"], context, outcome["status"])

    return normalized


def build_system_prompt() -> str:
    return (
        "You convert EnterpriseOps-Gym trajectory observations into a compact enterprise world state. "
        "Return strict JSON only. Use expanded key names, not single-letter shorthand: "
        "`objects_artifacts`, `process_state`, `relational_state`, `constraints`, and `history_context`. "
        "Do not include temporal state or a `T` key. Do not copy raw tool output verbatim. "
        "This is a world-model prediction target produced before real tool execution, so do not include concrete "
        "IDs, names, emails, URLs, timestamps, or other specific values that are only available from tool outputs. "
        "You may use specific values only when they appear in the system prompt, user prompt, or action arguments. "
        "Do not restate requirements, policies, permissions, or facts that are explicitly present in the system "
        "prompt or user prompt unless the current action newly satisfies, violates, or changes them. "
        "Otherwise use placeholders such as `<created_calendar_id>`, `<matched_user>`, or descriptive generic facts. "
        "Extract only planning-relevant facts, object references, permissions, dependencies, process progress, "
        "constraints, errors, and unresolved assumptions. Ground every field in the supplied trajectory context."
    )


def build_prompt(
    *,
    trajectory: dict[str, Any],
    message_index: int,
    messages: list[dict[str, Any]],
    max_output_chars: int,
    max_history_messages: int,
) -> str:
    state_message = messages[message_index]
    action_message = messages[message_index - 1] if message_index > 0 else {}
    context = extract_context(state_message)
    process = extract_process(state_message)
    history_start = max(0, message_index - max_history_messages)
    recent_messages = messages[history_start : message_index + 1]

    payload = {
        "trajectory_metadata": {
            "trajectory_id": trajectory.get("trajectory_id"),
            "domain": trajectory.get("domain"),
            "source": trajectory.get("source"),
            "source_model": trajectory.get("source_model"),
            "task_key": trajectory.get("task_key"),
        },
        "system_prompt": messages[0].get("content") if messages and messages[0].get("role") == "system" else "",
        "user_task": messages[1].get("content") if len(messages) > 1 and messages[1].get("role") == "user" else "",
        "current_action": action_message.get("content"),
        "existing_state_context": context,
        "existing_process_state": process,
        "recent_messages": recent_messages,
    }
    return (
        "Convert the current state message into this JSON schema:\n"
        f"{json.dumps(ENTERPRISE_STATE_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        "Schema meanings:\n"
        "- objects_artifacts: tickets, docs, orders, invoices, contracts, opportunities, incidents, events, files, users, or other business artifacts.\n"
        "- process_state: stage, queue, approval status, owner, blockers, completed and remaining requirements.\n"
        "- relational_state: permissions, dependencies, team ownership, assignments, communication links.\n"
        "- constraints: policy, compliance, SLA, budget, access control, validation constraints.\n"
        "- history_context: messages, prior decisions, latent intent, unresolved assumptions, and compact per-tool events.\n"
        "Temporal information must be omitted even if present in the source observation.\n\n"
        "Prediction constraints:\n"
        "- Do not include concrete IDs, names, emails, URLs, timestamps, event IDs, etags, or opaque tokens that are only visible in tool outputs.\n"
        "- It is OK to use specific values from the system prompt, user task, or current action arguments.\n"
        "- Do not duplicate system/user-prompt requirements as `salient_facts`, `active_constraints`, or `satisfied` items unless the tool result shows a new state change about them.\n"
        "- Prefer deltas: what became true, false, blocked, created, updated, linked, or still unresolved after this action.\n"
        "- For tool-output-only values, use stable placeholders or generic descriptions instead.\n"
        "- If the current/final tool action failed, the process stage must remain on the failed/current action, not `finished`.\n\n"
        "Use one of outcome.status: success, partial_success, no_progress, failure.\n"
        "Use one of outcome.failure_category: none, validation_error, not_found, permission_denied, duplicate, "
        "policy_violation, schema_error, empty_result, semantic_mismatch, unknown.\n\n"
        "Source trajectory context:\n"
        f"{compact_json(payload, max_output_chars)}"
    )


def heuristic_enterprise_state(
    state_message: dict[str, Any],
    *,
    fallback_stage: str | None = None,
) -> dict[str, Any]:
    context = extract_context(state_message)
    process = extract_process(state_message)
    status = status_from_label(context.get("last_tool_execution_result"))
    failure_category = "none" if status == "success" else infer_failure_category(context)
    return normalize_enterprise_state(
        {
            "outcome": {
                "status": status,
                "summary": summarize_text(context.get("last_tool_output"), 320),
                "failure_category": failure_category,
                "recoverable": status != "success",
            },
            "process_state": {
                "stage": str(process.get("current_stage") or ""),
                "remaining_requirements": normalize_string_list(process.get("remaining_stages")),
                "completed_requirements": [],
                "blockers": [summarize_text(context.get("error_message"), 200)] if context.get("error_message") else [],
            },
            "history_context": {
                "salient_facts": [summarize_text(context.get("last_tool_output"), 240)] if context.get("last_tool_output") else [],
                "last_tool_events": [
                    {
                        "tool_name": str(context.get("last_tool_name") or ""),
                        "status": status,
                        "operation": "",
                        "summary": summarize_text(context.get("last_tool_output"), 240),
                        "error": str(context.get("error_message") or "") if status != "success" else "",
                    }
                ],
            },
        },
        context,
        process,
        fallback_stage=fallback_stage,
    )


def diff_values(previous: Any, current: Any) -> Any:
    if previous == current:
        return None
    if isinstance(previous, dict) and isinstance(current, dict):
        diff = {}
        for key, value in current.items():
            nested = diff_values(previous.get(key), value)
            if nested is not None:
                diff[key] = nested
        return diff or None
    return current


def make_state_diff(previous_state: dict[str, Any] | None, current_state: dict[str, Any]) -> dict[str, Any]:
    if previous_state is None:
        return {
            "mode": "full",
            "schema": "enterprise_ops_objects_process_relational_constraints_history_v1",
            **current_state,
        }
    delta = diff_values(previous_state, current_state) or {}
    return {
        "mode": "diff",
        "schema": "enterprise_ops_objects_process_relational_constraints_history_v1",
        "diff_from_previous_state": delta,
    }


def convert_state_message(
    *,
    llm: ClousedSourceLLM,
    trajectory: dict[str, Any],
    messages: list[dict[str, Any]],
    message_index: int,
    previous_enterprise_state: dict[str, Any] | None,
    cache: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    state_message = messages[message_index]
    context = extract_context(state_message)
    process = extract_process(state_message)
    fallback_stage = previous_non_finished_stage(messages, message_index) if is_failed_final_step(context, process) else None
    prompt_payload = {
        "trajectory_id": trajectory.get("trajectory_id"),
        "message_index": message_index,
        "action": messages[message_index - 1].get("content") if message_index > 0 else None,
        "state": state_message.get("content"),
        "state_mode": "full_enterprise_state_before_optional_diff",
        "fallback_stage": fallback_stage,
    }
    key = cache_key(prompt_payload)
    cached = cache.get(key) if args.resume else None
    if isinstance(cached, dict):
        enterprise_state = cached
    else:
        prompt = build_prompt(
            trajectory=trajectory,
            message_index=message_index,
            messages=messages,
            max_output_chars=args.max_output_chars,
            max_history_messages=args.max_history_messages,
        )
        try:
            response = llm.generate_format(
                prompt=prompt,
                system_prompt=build_system_prompt(),
                temperature=args.temperature,
                format="json",
                schema=ENTERPRISE_STATE_SCHEMA,
            )
            if not isinstance(response, dict):
                raise ValueError(f"Expected dict, got {type(response).__name__}")
            enterprise_state = response
        except Exception:
            if not args.allow_heuristic_fallback:
                raise
            enterprise_state = heuristic_enterprise_state(state_message, fallback_stage=fallback_stage)
        enterprise_state = normalize_enterprise_state(enterprise_state, context, process, fallback_stage=fallback_stage)
        cache[key] = enterprise_state

    converted = copy.deepcopy(state_message)
    if args.state_mode == "diff":
        output_state = make_state_diff(previous_enterprise_state, enterprise_state)
    else:
        output_state = enterprise_state
    converted["content"] = {"state": output_state}
    return converted, enterprise_state


def convert_trajectory(
    trajectory: dict[str, Any],
    llm: ClousedSourceLLM,
    cache: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    converted = copy.deepcopy(trajectory)
    messages = trajectory.get("messages") or []
    converted_messages = []
    previous_enterprise_state = None
    for index, message in enumerate(messages):
        if message.get("role") == "state":
            converted_message, previous_enterprise_state = convert_state_message(
                llm=llm,
                trajectory=trajectory,
                messages=messages,
                message_index=index,
                previous_enterprise_state=previous_enterprise_state,
                cache=cache,
                args=args,
            )
            converted_messages.append(converted_message)
        else:
            converted_messages.append(copy.deepcopy(message))
    converted["messages"] = converted_messages
    converted["state_schema"] = "enterprise_ops_objects_process_relational_constraints_history_v1"
    converted["state_schema_omits"] = ["temporal_state"]
    return converted


def convert_file(
    input_path: Path,
    output_path: Path,
    llm: ClousedSourceLLM,
    cache: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    trajectories = load_json(input_path)
    if not isinstance(trajectories, list):
        raise ValueError(f"Expected JSON list in {input_path}")
    if args.limit_trajectories is not None:
        trajectories = trajectories[: max(0, args.limit_trajectories)]

    converted = []
    total = len(trajectories)
    for index, trajectory in enumerate(trajectories, start=1):
        converted.append(convert_trajectory(trajectory, llm, cache, args))
        if index % 10 == 0 or index == total:
            print(f"{input_path.name}: converted {index}/{total}", flush=True)
            dump_json(args.cache_path, cache)

    dump_json(output_path, converted)
    dump_json(args.cache_path, cache)
    print(f"Wrote {len(converted)} trajectories to {output_path}")


def main() -> None:
    args = parse_args()
    cache = load_cache(args.cache_path, args.resume)
    llm = build_llm(args)

    if args.splits in {"train", "both"}:
        convert_file(args.train_input, args.train_output, llm, cache, args)
    if args.splits in {"test", "both"}:
        convert_file(args.test_input, args.test_output, llm, cache, args)


if __name__ == "__main__":
    main()
