import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TRAJECTORIES_DIR = ROOT / "trajectories"

DEFAULT_TERMINALBENCH_INPUTS = {
    "all": TRAJECTORIES_DIR / "terminalbench_2_0_multi_model_trajectories.jsonl",
    "train": TRAJECTORIES_DIR / "terminalbench_2_0_multi_model_train_trajectories.jsonl",
    "test": TRAJECTORIES_DIR / "terminalbench_2_0_multi_model_test_trajectories.jsonl",
}
DEFAULT_CRMARENAPRO_INPUTS = {
    "all": TRAJECTORIES_DIR / "crmarenapro_multi_model_trajectories.jsonl",
    "train": TRAJECTORIES_DIR / "crmarenapro_multi_model_train_trajectories.jsonl",
    "test": TRAJECTORIES_DIR / "crmarenapro_multi_model_test_trajectories.jsonl",
}
DEFAULT_TERMINALBENCH_OUTPUTS = {
    "all": TRAJECTORIES_DIR / "terminalbench_2_0_multi_model_world_model_trajectories.json",
    "train": TRAJECTORIES_DIR / "terminalbench_2_0_multi_model_world_model_train_trajectories.json",
    "test": TRAJECTORIES_DIR / "terminalbench_2_0_multi_model_world_model_test_trajectories.json",
}
DEFAULT_CRMARENAPRO_OUTPUTS = {
    "all": TRAJECTORIES_DIR / "crmarenapro_multi_model_world_model_trajectories.json",
    "train": TRAJECTORIES_DIR / "crmarenapro_multi_model_world_model_train_trajectories.json",
    "test": TRAJECTORIES_DIR / "crmarenapro_multi_model_world_model_test_trajectories.json",
}
DEFAULT_MANIFEST_PATH = TRAJECTORIES_DIR / "aligned_world_model_trajectory_manifest.json"
CRM_RESPOND_RE = re.compile(r"<respond>\s*(.*?)\s*</respond>", re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize TerminalBench and CRMArenaPro trajectories in EnterpriseOps-Gym world-model messages format."
    )
    parser.add_argument(
        "--benchmark",
        choices=["terminalbench", "crmarenapro", "all"],
        default="all",
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--keep-raw-events",
        action="store_true",
        help="Retain raw event logs in the materialized records. Disabled by default for interpretability.",
    )
    return parser.parse_args()


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


def compact_record(record: dict[str, Any], *, keep_raw_events: bool) -> dict[str, Any]:
    if keep_raw_events:
        return dict(record)
    return {key: value for key, value in record.items() if key != "events"}


def terminalbench_system_prompt() -> str:
    return (
        "You are a terminal task agent operating in a shell environment. "
        "Use bash commands to inspect files, run programs, edit artifacts, and complete the user task."
    )


def terminalbench_task_instruction(events: list[dict[str, Any]]) -> str:
    for event in events:
        if event.get("event_type") != "ShellProtocolTask":
            continue
        payload = event.get("payload") or {}
        instruction = payload.get("instruction")
        if isinstance(instruction, str):
            return instruction
    for event in events:
        payload = event.get("payload") or {}
        parts = payload.get("parts") if isinstance(payload, dict) else None
        if not isinstance(parts, list):
            continue
        for part in parts:
            text = part.get("text") if isinstance(part, dict) else None
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if parsed.get("kind") == "task" and isinstance(parsed.get("instruction"), str):
                return parsed["instruction"]
    return ""


def terminalbench_exec_pairs(events: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs = []
    pending_requests: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("event_type")
        if event_type == "ShellProtocolExecRequest" and event.get("direction") == "inbound":
            pending_requests.append(event)
            continue
        if event_type != "ShellProtocolExecResult" or event.get("direction") != "green":
            continue
        if not pending_requests:
            continue
        pairs.append((pending_requests.pop(0), event))
    return pairs


def terminalbench_final_output(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("event_type") != "ShellProtocolFinal":
            continue
        payload = event.get("payload") or {}
        output = payload.get("output")
        if isinstance(output, str):
            return output
    return ""


def normalize_terminalbench_trajectory(record: dict[str, Any], trajectory_index: int, *, keep_raw_events: bool) -> dict[str, Any]:
    events = record.get("events") or []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": terminalbench_system_prompt()},
        {"role": "user", "content": terminalbench_task_instruction(events)},
    ]
    action_state_pairs = 0
    for request_event, result_event in terminalbench_exec_pairs(events):
        request_payload = request_event.get("payload") or {}
        result_payload = result_event.get("payload") or {}
        command = request_payload.get("command") or request_event.get("command") or ""
        timeout = request_payload.get("timeout")
        exit_code = result_event.get("exit_code", result_payload.get("exit_code"))
        label = 1 if exit_code == 0 else -1
        stdout = result_payload.get("stdout") or ""
        stderr = result_payload.get("stderr") or ""
        output_parts = []
        if stdout:
            output_parts.append(f"stdout:\n{stdout}")
        if stderr:
            output_parts.append(f"stderr:\n{stderr}")
        output = "\n\n".join(output_parts) or f"exit_code: {exit_code}"
        arguments = {"command": command}
        if timeout is not None:
            arguments["timeout"] = timeout
        messages.append(
            {
                "role": "action",
                "content": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "execute_bash", "arguments": arguments},
                        }
                    ]
                },
            }
        )
        messages.append(
            {
                "role": "state",
                "content": {
                    "state": {
                        "agent": {"role": "terminal task agent"},
                        "context": {
                            "last_tool_execution_result": label,
                            "last_tool_name": "execute_bash",
                            "last_tool_output": output,
                            "error_message": output if label != 1 else "",
                        },
                        "process": {
                            "remaining_stages": [],
                            "current_stage": "continue" if label == 1 else "repair failed shell command",
                        },
                        "relational": {},
                        "temporal": {},
                    }
                },
            }
        )
        action_state_pairs += 1
    final_output = terminalbench_final_output(events)
    if final_output:
        messages.append({"role": "assistant", "content": final_output})

    normalized = compact_record(record, keep_raw_events=keep_raw_events)
    normalized.setdefault("trajectory_id", f"terminalbench-2.0-{trajectory_index}")
    normalized["messages"] = messages
    normalized["ewm_input_format"] = "system_user_action_state_v1"
    normalized["alignment_granularity"] = "tool_level"
    normalized["alignment_action_state_pairs"] = action_state_pairs
    return normalized


def parse_crm_task_user_prompt(task_metadata: dict[str, Any]) -> str:
    prompt = task_metadata.get("prompt") or ""
    required_context = task_metadata.get("required_context") or ""
    persona = task_metadata.get("persona") or ""
    parts = []
    if prompt:
        parts.append(f"Question: {prompt}")
    if required_context:
        parts.append(f"Context:\n{required_context}")
    if persona:
        parts.append(f"Persona: {persona}")
    return "\n\n".join(parts)


def crm_system_prompt() -> str:
    return (
        "You are an expert Salesforce CRM assistant with database access. "
        "Use CRM queries and available context to answer the user task."
    )


def parse_internal_trajectory(record: dict[str, Any]) -> dict[str, Any] | None:
    for event in record.get("events") or []:
        if event.get("event_type") != "TaskArtifactUpdateEvent":
            continue
        artifact = ((event.get("event") or {}).get("artifact") or {})
        if artifact.get("name") != "internal_trajectory":
            continue
        for part in artifact.get("parts") or []:
            text = part.get("text")
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if parsed.get("format") == "crmarenapro_react" or (parsed.get("payload") or {}).get("messages"):
                return parsed
    return None


def parse_purple_internal_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = []
    for event in record.get("events") or []:
        if event.get("event_type") != "PurpleInternalMessage":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        message = dict(payload)
        message.setdefault("role", event.get("role"))
        message.setdefault("sequence", event.get("sequence"))
        source = event.get("source")
        if isinstance(source, dict):
            message.setdefault("source", source)
        messages.append(message)
    messages.sort(key=lambda message: message.get("sequence") if isinstance(message.get("sequence"), int) else 10**12)
    return messages


def crm_action_name(tool_name: Any) -> str:
    name = str(tool_name or "crm_tool").strip() or "crm_tool"
    if name == "execute":
        return "execute_crm_sql"
    if name == "describe":
        return "describe_crm_table"
    if name == "respond":
        return "respond"
    return name


def stringify_tool_result(tool_message: dict[str, Any]) -> str:
    result = tool_message.get("result")
    if result is not None:
        if isinstance(result, dict) and result.get("success") is False and result.get("error"):
            return str(result.get("error"))
        return json.dumps(result, ensure_ascii=False, sort_keys=True)
    content = tool_message.get("content")
    if content is None:
        return ""
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, sort_keys=True)


def crm_tool_success(tool_message: dict[str, Any]) -> bool:
    result = tool_message.get("result")
    if isinstance(result, dict) and isinstance(result.get("success"), bool):
        return bool(result["success"])
    content = stringify_tool_result(tool_message).lower()
    return not any(marker in content for marker in ("sql error", "syntax error", "exception", "traceback", "permission denied"))


def extract_respond_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        if not isinstance(content, str):
            continue
        match = CRM_RESPOND_RE.search(content)
        if match:
            return match.group(1).strip()
    return ""


def crm_messages_from_internal_messages(record: dict[str, Any], internal_messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    system_prompt = next(
        (message.get("content") for message in internal_messages if message.get("role") == "system" and isinstance(message.get("content"), str)),
        crm_system_prompt(),
    )
    user_prompt = next(
        (message.get("content") for message in internal_messages if message.get("role") == "user" and isinstance(message.get("content"), str)),
        parse_crm_task_user_prompt(record.get("task_metadata") or {}),
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    action_state_pairs = 0
    for tool_message in internal_messages:
        if tool_message.get("role") != "tool":
            continue
        name = crm_action_name(tool_message.get("name"))
        content = tool_message.get("content") or ""
        success = crm_tool_success(tool_message)
        output = stringify_tool_result(tool_message)
        messages.append(
            {
                "role": "action",
                "content": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": {"query": content} if name == "execute_crm_sql" else {"input": content},
                            },
                        }
                    ]
                },
            }
        )
        messages.append(
            {
                "role": "state",
                "content": {
                    "state": {
                        "agent": {"role": "salesforce crm assistant"},
                        "context": {
                            "last_tool_execution_result": 1 if success else -1,
                            "last_tool_name": name,
                            "last_tool_output": output,
                            "error_message": output if not success else "",
                        },
                        "process": {
                            "remaining_stages": [],
                            "current_stage": "continue" if success else "repair failed CRM query",
                        },
                        "relational": {
                            "task_category": record.get("task_category"),
                            "org_type": record.get("org_type"),
                        },
                        "temporal": {},
                    }
                },
            }
        )
        action_state_pairs += 1
    final_answer = extract_respond_text(internal_messages) or (record.get("answer_data") or {}).get("answer")
    if final_answer:
        messages.append({"role": "assistant", "content": str(final_answer)})
    return messages, action_state_pairs


def crm_messages_from_internal(record: dict[str, Any], internal: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    payload = internal.get("payload") or {}
    internal_messages = payload.get("messages") or []
    return crm_messages_from_internal_messages(record, internal_messages)


def crm_fallback_messages(record: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    task_metadata = record.get("task_metadata") or {}
    answer_data = record.get("answer_data") or {}
    task_success = record.get("task_success")
    metrics = record.get("metrics") or {}
    query_counts = record.get("query_execution_counts") or {}
    output = {
        "answer": answer_data.get("answer"),
        "task_success": task_success,
        "task_score": record.get("task_score"),
        "task_total_score": record.get("task_total_score"),
        "task_reason": record.get("task_reason"),
        "metrics": metrics,
        "query_execution_counts": query_counts,
    }
    if task_success is True:
        label = 1
        stage = "task completed"
    elif task_success is False:
        label = -1
        stage = "task failed final evaluator"
    else:
        label = 0
        stage = "task outcome unknown"
    messages = [
        {"role": "system", "content": crm_system_prompt()},
        {"role": "user", "content": parse_crm_task_user_prompt(task_metadata)},
        {
            "role": "action",
            "content": {
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "run_crm_agent",
                            "arguments": {
                                "task_id": record.get("task_id"),
                                "task_category": record.get("task_category"),
                                "reward_metric": record.get("reward_metric"),
                            },
                        },
                    }
                ]
            },
        },
        {
            "role": "state",
            "content": {
                "state": {
                    "agent": {"role": "salesforce crm assistant"},
                    "context": {
                        "last_tool_execution_result": label,
                        "last_tool_name": "run_crm_agent",
                        "last_tool_output": json.dumps(output, ensure_ascii=False, sort_keys=True),
                        "error_message": "" if label == 1 else json.dumps(output, ensure_ascii=False, sort_keys=True),
                    },
                    "process": {"remaining_stages": [], "current_stage": stage},
                    "relational": {
                        "task_category": record.get("task_category"),
                        "org_type": record.get("org_type"),
                        "reward_metric": record.get("reward_metric"),
                    },
                    "temporal": {},
                }
            },
        },
    ]
    if answer_data.get("answer") is not None:
        messages.append({"role": "assistant", "content": str(answer_data.get("answer"))})
    return messages, 1


def normalize_crmarenapro_trajectory(record: dict[str, Any], trajectory_index: int, *, keep_raw_events: bool) -> dict[str, Any]:
    purple_messages = parse_purple_internal_messages(record)
    if any(message.get("role") == "tool" for message in purple_messages):
        messages, action_state_pairs = crm_messages_from_internal_messages(record, purple_messages)
        granularity = "tool_level_purple_internal"
    else:
        internal = parse_internal_trajectory(record)
        if internal is not None:
            messages, action_state_pairs = crm_messages_from_internal(record, internal)
            granularity = "tool_level_artifact"
        else:
            messages, action_state_pairs = crm_fallback_messages(record)
            granularity = "trajectory_level_fallback"
    normalized = compact_record(record, keep_raw_events=keep_raw_events)
    normalized.setdefault("trajectory_id", f"crmarenapro-{trajectory_index}")
    normalized["messages"] = messages
    normalized["ewm_input_format"] = "system_user_action_state_v1"
    normalized["alignment_granularity"] = granularity
    normalized["alignment_action_state_pairs"] = action_state_pairs
    return normalized


def materialize_file(
    *,
    benchmark: str,
    split_name: str,
    input_path: Path,
    output_path: Path,
    keep_raw_events: bool,
) -> dict[str, Any]:
    records = load_jsonl(input_path)
    normalized_records = []
    granularity_counts: dict[str, int] = {}
    pair_count = 0
    for index, record in enumerate(records):
        if benchmark == "terminalbench":
            normalized = normalize_terminalbench_trajectory(record, index, keep_raw_events=keep_raw_events)
        elif benchmark == "crmarenapro":
            normalized = normalize_crmarenapro_trajectory(record, index, keep_raw_events=keep_raw_events)
        else:
            raise ValueError(f"unknown benchmark: {benchmark}")
        granularity = normalized.get("alignment_granularity", "unknown")
        granularity_counts[granularity] = granularity_counts.get(granularity, 0) + 1
        pair_count += int(normalized.get("alignment_action_state_pairs") or 0)
        normalized_records.append(normalized)
    dump_json(output_path, normalized_records)
    return {
        "benchmark": benchmark,
        "split": split_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "trajectory_count": len(normalized_records),
        "alignment_granularity_counts": granularity_counts,
        "action_state_pair_count": pair_count,
        "kept_raw_events": keep_raw_events,
    }


def main() -> None:
    args = parse_args()
    jobs = []
    if args.benchmark in {"terminalbench", "all"}:
        for split_name, input_path in DEFAULT_TERMINALBENCH_INPUTS.items():
            jobs.append(("terminalbench", split_name, input_path, DEFAULT_TERMINALBENCH_OUTPUTS[split_name]))
    if args.benchmark in {"crmarenapro", "all"}:
        for split_name, input_path in DEFAULT_CRMARENAPRO_INPUTS.items():
            jobs.append(("crmarenapro", split_name, input_path, DEFAULT_CRMARENAPRO_OUTPUTS[split_name]))

    summaries = []
    for benchmark, split_name, input_path, output_path in jobs:
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        summary = materialize_file(
            benchmark=benchmark,
            split_name=split_name,
            input_path=input_path,
            output_path=output_path,
            keep_raw_events=args.keep_raw_events,
        )
        summaries.append(summary)
        print(
            f"Wrote {summary['trajectory_count']} {benchmark} {split_name} trajectories to {output_path} "
            f"({summary['action_state_pair_count']} action/state pairs; {summary['alignment_granularity_counts']})"
        )
    dump_json(
        args.manifest_path,
        {
            "ewm_input_format": "system_user_action_state_v1",
            "raw_events_kept": args.keep_raw_events,
            "outputs": summaries,
        },
    )
    print(f"Wrote manifest to {args.manifest_path}")


if __name__ == "__main__":
    main()
