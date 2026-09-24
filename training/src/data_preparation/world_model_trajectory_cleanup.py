from __future__ import annotations

from copy import deepcopy
from typing import Any


def is_tool_action_content(action_content: Any) -> bool:
    return isinstance(action_content, dict) and bool(action_content.get("tool_calls"))


def message_is_tool_action(message: dict[str, Any]) -> bool:
    return message.get("role") == "action" and is_tool_action_content(message.get("content"))


def _copy_followup_state_sections(
    merged_state: dict[str, Any],
    followup_state: dict[str, Any],
) -> None:
    for key in ("agent", "process", "relational", "temporal"):
        if key in followup_state:
            merged_state[key] = deepcopy(followup_state[key])


def _merge_last_tool_result(
    merged_context: dict[str, Any],
    followup_context: dict[str, Any],
) -> None:
    current_result = merged_context.get("last_tool_execution_result")
    followup_result = followup_context.get("last_tool_execution_result")

    if current_result != -1 and followup_result == 1:
        merged_context["last_tool_execution_result"] = 1
    elif current_result is None and followup_result in {-1, 0, 1}:
        merged_context["last_tool_execution_result"] = followup_result


def _copy_missing_context_fields(
    merged_context: dict[str, Any],
    followup_context: dict[str, Any],
) -> None:
    for key in ("last_tool_name", "last_tool_output", "error_message"):
        if not merged_context.get(key) and followup_context.get(key) is not None:
            merged_context[key] = deepcopy(followup_context[key])


def merge_followup_state_into_tool_state(
    tool_state_message: dict[str, Any],
    followup_state_message: dict[str, Any],
) -> dict[str, Any]:
    merged_content = tool_state_message.get("content")
    followup_content = followup_state_message.get("content")
    if not isinstance(merged_content, dict) or not isinstance(followup_content, dict):
        return deepcopy(tool_state_message)

    merged_state = merged_content.get("state")
    followup_state = followup_content.get("state")
    if not isinstance(merged_state, dict) or not isinstance(followup_state, dict):
        return deepcopy(tool_state_message)

    merged = deepcopy(tool_state_message)
    merged_state = merged["content"]["state"]
    _copy_followup_state_sections(merged_state, followup_state)

    merged_context = merged_state.setdefault("context", {})
    followup_context = followup_state.get("context")
    if not isinstance(followup_context, dict):
        return merged

    _merge_last_tool_result(merged_context, followup_context)
    _copy_missing_context_fields(merged_context, followup_context)

    return merged


def _merge_followup_into_latest_tool_state(
    cleaned: list[dict[str, Any]],
    latest_tool_state_index: int | None,
    next_message: dict[str, Any] | None,
) -> None:
    if next_message and next_message.get("role") == "state" and latest_tool_state_index is not None:
        cleaned[latest_tool_state_index] = merge_followup_state_into_tool_state(
            cleaned[latest_tool_state_index],
            next_message,
        )


def _state_followup_advance(next_message: dict[str, Any] | None) -> int:
    return 2 if next_message and next_message.get("role") == "state" else 1


def cleanup_world_model_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    latest_tool_state_index: int | None = None
    cursor = 0

    while cursor < len(messages):
        message = deepcopy(messages[cursor])
        next_message = messages[cursor + 1] if cursor + 1 < len(messages) else None

        if message.get("role") == "action" and not is_tool_action_content(message.get("content")):
            _merge_followup_into_latest_tool_state(cleaned, latest_tool_state_index, next_message)
            message["role"] = "assistant"
            cleaned.append(message)
            cursor += _state_followup_advance(next_message)
            continue

        cleaned.append(message)
        if message.get("role") == "state" and len(cleaned) >= 2 and message_is_tool_action(cleaned[-2]):
            latest_tool_state_index = len(cleaned) - 1
        cursor += 1

    return cleaned


def cleanup_world_model_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    cleaned = deepcopy(trajectory)
    cleaned["messages"] = cleanup_world_model_messages(trajectory.get("messages") or [])
    return cleaned
