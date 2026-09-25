import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preparation.import_benchmark_sources import (  # noqa: E402
    CREATE_PREFIXES,
    DELIVER_PREFIXES,
    READ_PREFIXES,
    UPDATE_PREFIXES,
    artifact_from_tool,
    unique_ordered,
)


DEFAULT_PILOT_SEEDS_PATH = ROOT / "trajectories" / "enterpriseops_gym_pilot_seeds.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "trajectories" / "enterpriseops_gym_stateful_domain_trajectories.json"
DOMAIN_ORDER = ["csm", "itsm", "email", "drive", "teams"]

INTERESTING_RESULT_KEYS = {
    "account_id",
    "active",
    "call_id",
    "call_record_id",
    "case_id",
    "case_number",
    "channel",
    "comment_id",
    "contact_id",
    "contract_id",
    "coverage_hours",
    "created_at",
    "delegate_email",
    "description",
    "draft_id",
    "email",
    "entitlement_id",
    "event_id",
    "file_id",
    "file_name",
    "filter_id",
    "incident_id",
    "installed_product_id",
    "label_id",
    "location_id",
    "message",
    "message_id",
    "name",
    "permission_id",
    "priority",
    "product_id",
    "reply_id",
    "serial_number",
    "service_id",
    "service_offering_id",
    "short_description",
    "sla_definition_id",
    "state",
    "status",
    "subject",
    "team_id",
    "team_name",
    "thread_id",
    "title",
    "townhall_id",
    "user_id",
    "webinar_id",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate one stateful EnterpriseOps-Gym trajectory for each of "
            "csm, itsm, email, drive, and teams from the pilot imported seeds."
        )
    )
    parser.add_argument("--pilot-seeds-path", type=Path, default=DEFAULT_PILOT_SEEDS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def summarize_text(value, limit=300):
    if value is None:
        return None
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    value = " ".join(value.split()).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def try_parse_json(text):
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def looks_like_error_text(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "api error",
        "error:",
        "exception",
        "invalid ",
        "failed",
        "traceback",
    )
    return any(marker in lowered for marker in markers)


def load_selected_seeds(path: Path):
    pilot_records = load_jsonl(path)
    selected = {}
    for domain in DOMAIN_ORDER:
        for record in pilot_records:
            if record["environment"]["domain"] == domain:
                selected[domain] = record
                break
        if domain not in selected:
            raise ValueError(f"No pilot seed found for domain {domain}")
    return [selected[domain] for domain in DOMAIN_ORDER]


def convert_tool_calls(tool_calls):
    converted = []
    for tool_call in tool_calls:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": tool_call.get("args") or {},
                },
            }
        )
    return converted


def normalize_ai_action(ai_message):
    tool_calls = ai_message.get("tool_calls") or []
    if tool_calls:
        return {"tool_calls": convert_tool_calls(tool_calls)}

    content = (ai_message.get("content") or "").strip()
    if content:
        return content
    return None


def is_scalar(value):
    return isinstance(value, (str, int, float, bool)) or value is None


def flatten_record_signals(value):
    signals = []

    def walk(node):
        if isinstance(node, dict):
            for key, child in node.items():
                if is_scalar(child):
                    if key in INTERESTING_RESULT_KEYS or key.endswith("_id"):
                        signals.append(f"{key}={child}")
                else:
                    walk(child)
        elif isinstance(node, list):
            for item in node[:8]:
                walk(item)

    walk(value)
    return unique_ordered(signals)


def extract_tool_payload(tool_result):
    result_wrapper = tool_result.get("result") or {}
    success = bool(result_wrapper.get("success"))
    error = result_wrapper.get("error")
    payload = result_wrapper.get("result") or {}

    text_chunks = []
    if isinstance(payload, dict):
        for content_item in payload.get("content") or []:
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                text_chunks.append(text)
        structured_content = payload.get("structuredContent")
    else:
        structured_content = None

    if error and not text_chunks:
        if isinstance(error, str):
            text_chunks.append(error)
        else:
            text_chunks.append(json.dumps(error, ensure_ascii=False))

    raw_text = "\n".join(text_chunks).strip()
    if raw_text and looks_like_error_text(raw_text):
        success = False
    parsed_text = try_parse_json(raw_text)
    parsed_payload = parsed_text if parsed_text is not None else structured_content
    if parsed_payload is None:
        parsed_payload = payload if payload else raw_text

    if isinstance(parsed_payload, (dict, list)):
        record_signals = flatten_record_signals(parsed_payload)
    else:
        record_signals = []

    summary = summarize_text(record_signals[:8] if record_signals else raw_text or error or payload, limit=240)
    return {
        "success": success,
        "error": error,
        "payload": parsed_payload,
        "raw_text": raw_text,
        "record_signals": record_signals,
        "summary": summary,
    }


def humanize_tool_name(tool_name: str) -> str:
    return tool_name.replace("_", " ")


def action_summary(action_content):
    if isinstance(action_content, dict) and action_content.get("tool_calls"):
        names = [call["function"]["name"] for call in action_content["tool_calls"]]
        preview = ", ".join(humanize_tool_name(name) for name in names[:3])
        if len(names) > 3:
            preview += ", and more"
        return f"Run {preview}."
    return summarize_text(action_content, limit=180)


def build_stage_labels(action_contents):
    labels = []
    for index, action_content in enumerate(action_contents):
        if isinstance(action_content, dict) and action_content.get("tool_calls"):
            names = [call["function"]["name"] for call in action_content["tool_calls"]]
            if len(names) == 1:
                labels.append(f"Execute {humanize_tool_name(names[0])}")
            else:
                joined = ", ".join(humanize_tool_name(name) for name in names[:3])
                if len(names) > 3:
                    joined += ", and more"
                labels.append(f"Execute {joined}")
        else:
            label = "Deliver final response" if index == len(action_contents) - 1 else action_summary(action_content)
            labels.append(label)
    return labels


def build_initial_state(seed: dict, source_path: str, stage_labels: list[str]) -> dict:
    state_hooks = seed["state_hooks"]
    candidate_agents = seed["candidate_agents"]
    active_agent = candidate_agents[0]
    return {
        "identify": {
            "coordination_pattern": seed["coordination_pattern"],
            "candidate_agents": candidate_agents,
            "active_agent": active_agent,
            "engaged_agents": [active_agent["agent_id"]],
            "stakeholders": state_hooks["identify"]["stakeholders"],
            "named_entities": state_hooks["identify"]["named_entities"],
            "observed_entities": [],
        },
        "artifacts": {
            "durable_objects": state_hooks["artifacts"]["durable_objects"],
            "pending_reads": state_hooks["artifacts"]["read"],
            "pending_creates": state_hooks["artifacts"]["create"],
            "pending_updates": state_hooks["artifacts"]["update"],
            "pending_deliveries": state_hooks["artifacts"]["deliver"],
            "observed_records": [],
            "created_records": [],
            "updated_records": [],
            "delivered_records": [],
        },
        "process": {
            "preconditions": state_hooks["process"]["preconditions"],
            "current_stage": stage_labels[0] if stage_labels else "Analyze request",
            "completed_stages": [],
            "remaining_stages": stage_labels,
            "target_states": state_hooks["process"]["target_states"],
            "blocking_conditions": state_hooks["process"]["blocking_conditions"],
        },
        "relational": {
            "ownership": state_hooks["relational"]["ownership"],
            "permissions": state_hooks["relational"]["permissions"],
            "dependencies": state_hooks["relational"]["dependencies"],
            "approvals": state_hooks["relational"]["approvals"],
            "active_links": [],
        },
        "context": {
            "facts": state_hooks["context"]["facts"],
            "domains": state_hooks["context"]["domains"],
            "communication_channels": state_hooks["context"]["communication_channels"],
            "last_tool_name": None,
            "last_tool_names": [],
            "last_tool_execution_result": None,
            "last_tool_output": None,
            "latest_action_summary": None,
            "tool_result_summaries": [],
            "source_seed_id": seed["seed_id"],
            "source_path": source_path,
        },
        "temporal": {
            "absolute_dates": state_hooks["temporal"]["absolute_dates"],
            "relative_constraints": state_hooks["temporal"]["relative_constraints"],
            "sla_signals": state_hooks["temporal"]["sla_signals"],
            "sequencing_constraints": state_hooks["temporal"]["sequencing_constraints"],
            "step_index": 0,
            "total_steps": len(stage_labels),
            "is_final_action": False,
        },
    }


def infer_operation_bucket(tool_name: str) -> str:
    lowered = tool_name.lower()
    if lowered.startswith(CREATE_PREFIXES):
        return "create"
    if lowered.startswith(UPDATE_PREFIXES):
        return "update"
    if lowered.startswith(DELIVER_PREFIXES):
        return "deliver"
    if lowered.startswith(READ_PREFIXES):
        return "read"
    return "read"


def active_agent_for_step(seed: dict, *, is_final_action: bool, has_tool_calls: bool) -> dict:
    candidate_agents = seed["candidate_agents"]
    if has_tool_calls and len(candidate_agents) >= 2:
        return candidate_agents[1]
    if is_final_action and len(candidate_agents) >= 3:
        return candidate_agents[2]
    if len(candidate_agents) >= 2:
        return candidate_agents[1]
    return candidate_agents[0]


def engaged_agents_for_step(seed: dict, active_agent: dict, *, is_final_action: bool) -> list[str]:
    candidate_agents = seed["candidate_agents"]
    engaged = []
    if candidate_agents:
        engaged.append(candidate_agents[0]["agent_id"])
    engaged.append(active_agent["agent_id"])
    if is_final_action and len(candidate_agents) >= 3:
        engaged.append(candidate_agents[2]["agent_id"])
    if seed["state_hooks"]["relational"]["approvals"] and len(candidate_agents) >= 3:
        engaged.append(candidate_agents[2]["agent_id"])
    return unique_ordered(engaged)


def update_pending_artifacts(state: dict, artifact: str, bucket: str):
    pending_key_by_bucket = {
        "read": "pending_reads",
        "create": "pending_creates",
        "update": "pending_updates",
        "deliver": "pending_deliveries",
    }
    pending_key = pending_key_by_bucket.get(bucket)
    if not pending_key:
        return
    state["artifacts"][pending_key] = [
        item for item in state["artifacts"][pending_key] if item != artifact
    ]


def add_records_for_bucket(state: dict, bucket: str, record_signals: list[str], artifact: str):
    list_key_by_bucket = {
        "read": "observed_records",
        "create": "created_records",
        "update": "updated_records",
        "deliver": "delivered_records",
    }
    list_key = list_key_by_bucket[bucket]
    values = record_signals[:] if record_signals else [artifact]
    state["artifacts"][list_key] = unique_ordered(state["artifacts"][list_key] + values)


def update_relational_links(state: dict, tool_name: str, record_signals: list[str]):
    relation_terms = ("assign", "permission", "member", "link", "group", "owner", "contact")
    if not any(term in tool_name for term in relation_terms):
        return

    link_summary = tool_name
    if record_signals:
        link_summary += ": " + ", ".join(record_signals[:4])
    state["relational"]["active_links"] = unique_ordered(
        state["relational"]["active_links"] + [link_summary]
    )


def update_state_for_step(
    previous_state: dict,
    seed: dict,
    action_content,
    stage_label: str,
    tool_results: list[dict],
    step_index: int,
    total_steps: int,
):
    state = copy.deepcopy(previous_state)
    has_tool_calls = isinstance(action_content, dict) and bool(action_content.get("tool_calls"))
    is_final_action = step_index == total_steps
    active_agent = active_agent_for_step(
        seed,
        is_final_action=is_final_action,
        has_tool_calls=has_tool_calls,
    )

    state["identify"]["active_agent"] = active_agent
    state["identify"]["engaged_agents"] = engaged_agents_for_step(
        seed,
        active_agent,
        is_final_action=is_final_action,
    )

    tool_names = []
    tool_result_summaries = []
    last_tool_execution_result = state["context"]["last_tool_execution_result"]
    last_tool_name = state["context"]["last_tool_name"]
    last_tool_output = state["context"]["last_tool_output"]
    observed_entities = state["identify"]["observed_entities"][:]
    additional_blockers = []

    for tool_result in tool_results:
        tool_name = tool_result["tool_name"]
        tool_names.append(tool_name)
        bucket = infer_operation_bucket(tool_name)
        artifact = artifact_from_tool(tool_name)
        parsed = extract_tool_payload(tool_result)
        tool_result_summaries.append(f"{tool_name}: {parsed['summary']}")

        update_pending_artifacts(state, artifact, bucket)
        add_records_for_bucket(state, bucket, parsed["record_signals"], artifact)
        update_relational_links(state, tool_name, parsed["record_signals"])

        observed_entities = unique_ordered(observed_entities + parsed["record_signals"])
        last_tool_execution_result = 1 if parsed["success"] else 0
        last_tool_name = tool_name
        last_tool_output = summarize_text(parsed["summary"], limit=240)
        if not parsed["success"]:
            additional_blockers.append(
                summarize_text(parsed["error"] or parsed["raw_text"] or f"{tool_name} failed", limit=180)
            )

    state["identify"]["observed_entities"] = observed_entities
    state["process"]["current_stage"] = stage_label
    state["process"]["completed_stages"] = state["process"]["completed_stages"] + [stage_label]
    state["process"]["remaining_stages"] = (
        state["process"]["remaining_stages"][1:] if state["process"]["remaining_stages"] else []
    )
    if is_final_action and not additional_blockers and last_tool_execution_result != 0:
        state["process"]["blocking_conditions"] = []
        state["artifacts"]["pending_reads"] = []
        state["artifacts"]["pending_creates"] = []
        state["artifacts"]["pending_updates"] = []
        state["artifacts"]["pending_deliveries"] = []
    elif additional_blockers:
        state["process"]["blocking_conditions"] = unique_ordered(
            state["process"]["blocking_conditions"] + additional_blockers
        )

    state["context"]["last_tool_name"] = last_tool_name
    state["context"]["last_tool_names"] = (
        tool_names if tool_names else state["context"]["last_tool_names"]
    )
    state["context"]["last_tool_execution_result"] = last_tool_execution_result
    state["context"]["last_tool_output"] = last_tool_output
    state["context"]["latest_action_summary"] = action_summary(action_content)
    state["context"]["tool_result_summaries"] = (
        tool_result_summaries
        if tool_result_summaries
        else state["context"]["tool_result_summaries"]
    )

    state["temporal"]["step_index"] = step_index
    state["temporal"]["total_steps"] = total_steps
    state["temporal"]["is_final_action"] = is_final_action
    return state


def extract_action_batches(conversation_flow):
    action_batches = []
    index = 0
    while index < len(conversation_flow):
        item = conversation_flow[index]
        if item["type"] != "ai_message":
            index += 1
            continue

        action_content = normalize_ai_action(item)
        if action_content is None:
            index += 1
            continue

        next_index = index + 1
        tool_results = []
        while next_index < len(conversation_flow) and conversation_flow[next_index]["type"] != "ai_message":
            if conversation_flow[next_index]["type"] == "tool_result":
                tool_results.append(conversation_flow[next_index])
            next_index += 1

        action_batches.append(
            {
                "action_content": action_content,
                "tool_results": tool_results,
            }
        )
        index = next_index

    return action_batches


def default_trajectory_id(seed: dict) -> str:
    source_path = seed["source_record"].get("source_path")
    if source_path:
        return f"enterpriseops-gym-{Path(source_path).stem}"
    return f"enterpriseops-gym-{seed['seed_id']}"


def reconstruct_seed_trajectory(
    seed: dict,
    *,
    trajectory_id: str | None = None,
    selection_basis: str | None = None,
):
    source_path = Path(seed["source_record"]["source_path"])
    raw_payload = load_json(source_path)
    conversation_flow = raw_payload["runs"][0]["conversation_flow"]
    system_message = next(item for item in conversation_flow if item["type"] == "system_message")
    user_message = next(item for item in conversation_flow if item["type"] == "user_message")

    action_batches = extract_action_batches(conversation_flow)
    action_contents = [batch["action_content"] for batch in action_batches]
    stage_labels = build_stage_labels(action_contents)
    initial_state = build_initial_state(seed, str(source_path), stage_labels)

    messages = [
        {"role": "system", "content": system_message["content"]},
        {"role": "user", "content": user_message["content"]},
    ]

    current_state = initial_state
    total_steps = len(action_batches)
    for step_number, batch in enumerate(action_batches, start=1):
        messages.append({"role": "action", "content": batch["action_content"]})
        current_state = update_state_for_step(
            current_state,
            seed,
            batch["action_content"],
            stage_labels[step_number - 1],
            batch["tool_results"],
            step_number,
            total_steps,
        )
        messages.append({"role": "state", "content": {"state": current_state}})

    return {
        "trajectory_id": trajectory_id or default_trajectory_id(seed),
        "source": "enterprise_ops_gym_oracle_run",
        "selection_basis": selection_basis or "first_domain_record_from_enterpriseops_gym_pilot_seeds",
        "domain": seed["environment"]["domain"],
        "seed_id": seed["seed_id"],
        "coordination_pattern": seed["coordination_pattern"],
        "source_path": str(source_path),
        "gym_task_config_name": _derive_gym_task_config_name(source_path),
        "initial_state": initial_state,
        "messages": messages,
    }


def _derive_gym_task_config_name(source_path: str | Path) -> str | None:
    """See `generate_enterpriseops_gym_world_model_trajectories.derive_gym_task_config_name`."""
    stem = Path(source_path).stem
    if not stem.startswith("results_"):
        return None
    return f"{stem[len('results_'):]}.json"


def main():
    args = parse_args()
    selected_seeds = load_selected_seeds(args.pilot_seeds_path)
    trajectories = [reconstruct_seed_trajectory(seed) for seed in selected_seeds]
    dump_json(args.output_path, trajectories)
    print(f"Wrote {len(trajectories)} trajectories to {args.output_path}")
    for trajectory in trajectories:
        print(f"  {trajectory['domain']}: {trajectory['source_path']}")


if __name__ == "__main__":
    main()
