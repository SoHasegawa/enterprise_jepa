"""Schema-based canonical event state (+ epistemic nudge) — the ``canonical_nudge`` WM target.

Vendored verbatim from the EWM ``canonical`` branch
(``ewm/src/data_preparation/canonical_event_state.py``). Self-contained (stdlib only):
frozen categorical vocabularies + pure validator/inference functions, deliberately
schema-as-vocabulary rather than pydantic. A canonical event state is a flat JSON object
of 7 required categorical fields (``execution_status``/``error_signature``/``action_type``/
``object_type``/``side_effect_type``/``progress_signal``/``risk_signal``); the ``nudge`` adds
forward-looking ``information_sufficiency``/``missing_information_type``/``information_gain``/
``recommended_abstract_action``. The combined ``{canonical_event_state, nudge}`` payload is
the target the ``ewm_imagined`` backend predicts under ``WM_STATE=canonical_nudge``.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

EXECUTION_STATUS_VALUES = {"success", "failure", "partial", "no_op", "unknown"}
ERROR_SIGNATURE_VALUES = {
    "none",
    "not_found",
    "permission_denied",
    "invalid_argument",
    "dependency_missing",
    "test_failed",
    "timeout",
    "policy_risk",
    "parse_error",
    "runtime_error",
    "unknown",
}
ACTION_TYPE_VALUES = {
    "read",
    "search",
    "create",
    "update",
    "delete",
    "run",
    "test",
    "communicate",
    "clarify",
    "validate",
    "unknown",
}
OBJECT_TYPE_VALUES = {
    "file",
    "database_row",
    "ticket",
    "customer",
    "quote",
    "account",
    "package",
    "process",
    "message",
    "calendar",
    "case",
    "permission",
    "comment",
    "label",
    "repository",
    "branch",
    "unknown",
}
SIDE_EFFECT_TYPE_VALUES = {
    "none",
    "retrieved",
    "modified",
    "created",
    "deleted",
    "sent",
    "installed",
    "validated",
    "failed_validation",
    "executed",
    "unknown",
}
PROGRESS_SIGNAL_VALUES = {"positive", "negative", "neutral", "unknown"}
RISK_SIGNAL_VALUES = {
    "none",
    "policy_risk",
    "confidentiality_risk",
    "destructive_action",
    "irreversible_action",
}
UNCERTAINTY_LEVEL_VALUES = {"low", "medium", "high", "unknown"}
UNCERTAINTY_REASON_VALUES = {
    "none",
    "ambiguous_observation",
    "missing_context",
    "hidden_verifier",
    "heuristic_inference",
    "unknown",
}
INFORMATION_SUFFICIENCY_VALUES = {"sufficient", "insufficient", "unknown"}
MISSING_INFORMATION_TYPE_VALUES = {
    "none",
    "object_id",
    "schema",
    "policy",
    "file_context",
    "test_result",
    "user_intent",
    "dependency",
    "current_state",
    "permission",
    "unknown",
}
INFORMATION_GAIN_VALUES = {"high", "medium", "low", "negative", "unknown"}
RECOMMENDED_ABSTRACT_ACTION_VALUES = {
    "proceed",
    "search",
    "inspect",
    "retrieve",
    "validate",
    "clarify",
    "avoid",
    "rollback",
    "finalize",
    "unknown",
}

EVENT_STATE_SCHEMA_VERSION = "canonical_event_state_v1"
NUDGE_SCHEMA_VERSION = "ewm_nudge_v1"
CANONICAL_EVENT_WITH_NUDGE_SCHEMA_VERSION = "canonical_event_with_nudge_v1"
REQUIRED_CATEGORICAL_FIELDS = {
    "execution_status": EXECUTION_STATUS_VALUES,
    "error_signature": ERROR_SIGNATURE_VALUES,
    "action_type": ACTION_TYPE_VALUES,
    "object_type": OBJECT_TYPE_VALUES,
    "side_effect_type": SIDE_EFFECT_TYPE_VALUES,
    "progress_signal": PROGRESS_SIGNAL_VALUES,
    "risk_signal": RISK_SIGNAL_VALUES,
}
OPTIONAL_CATEGORICAL_FIELDS = {
    "uncertainty_level": UNCERTAINTY_LEVEL_VALUES,
    "uncertainty_reason": UNCERTAINTY_REASON_VALUES,
}
CATEGORICAL_FIELDS = {**REQUIRED_CATEGORICAL_FIELDS, **OPTIONAL_CATEGORICAL_FIELDS}
REQUIRED_EVENT_STATE_FIELDS = tuple(REQUIRED_CATEGORICAL_FIELDS.keys())
OPTIONAL_EVENT_STATE_FIELDS = tuple(OPTIONAL_CATEGORICAL_FIELDS.keys())
EVENT_STATE_FIELDS = (*REQUIRED_EVENT_STATE_FIELDS, *OPTIONAL_EVENT_STATE_FIELDS)
NUDGE_CATEGORICAL_FIELDS = {
    "information_sufficiency": INFORMATION_SUFFICIENCY_VALUES,
    "information_gain": INFORMATION_GAIN_VALUES,
    "recommended_abstract_action": RECOMMENDED_ABSTRACT_ACTION_VALUES,
}
NUDGE_LIST_FIELDS = {
    "missing_information_type": MISSING_INFORMATION_TYPE_VALUES,
}
NUDGE_FIELDS = (*NUDGE_CATEGORICAL_FIELDS.keys(), *NUDGE_LIST_FIELDS.keys())
CANONICAL_EVENT_WITH_NUDGE_FIELDS = ("canonical_event_state", "nudge")

READ_PREFIXES = ("get", "list", "find", "retrieve", "read", "cat", "head", "tail", "grep", "sed")
SEARCH_PREFIXES = ("search", "query", "locate", "rg", "find")
CREATE_PREFIXES = ("create", "add", "register", "upload", "copy", "fork", "mkdir", "touch")
UPDATE_PREFIXES = ("update", "patch", "modify", "edit", "write", "move", "rename", "chmod", "chown")
DELETE_PREFIXES = ("delete", "remove", "archive", "rm", "drop", "truncate")
COMMUNICATE_PREFIXES = ("send", "reply", "publish", "post", "message", "email")
VALIDATE_TOKENS = ("pytest", "unittest", "npm test", "cargo test", "go test", "make test", "test")
INSTALL_TOKENS = ("pip install", "apt install", "apt-get install", "npm install", "cargo install")
DESTRUCTIVE_TOKENS = ("rm ", "rm -", "drop ", "truncate", "delete", "shred", "mkfs", "dd ")


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def compact_text(value: Any, limit: int = 4096) -> str:
    text = coerce_text(value)
    text = " ".join(text.split()).strip().lower()
    return text[:limit]


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def normalize_tool_call(raw_action: Any) -> dict[str, Any]:
    if isinstance(raw_action, dict) and "tool_calls" in raw_action:
        calls = raw_action.get("tool_calls") or []
        raw_action = calls[-1] if calls else {}
    if isinstance(raw_action, list):
        raw_action = raw_action[-1] if raw_action else {}
    if isinstance(raw_action, str):
        parsed = parse_jsonish(raw_action.strip())
        if parsed is not raw_action:
            return normalize_tool_call(parsed)
        return {"name": "text_action", "arguments": {"text": raw_action}}
    if not isinstance(raw_action, dict):
        return {"name": "unknown", "arguments": {}}
    if "function" in raw_action and isinstance(raw_action["function"], dict):
        raw_action = raw_action["function"]
    name = str(raw_action.get("name") or raw_action.get("tool_name") or "unknown").strip()
    arguments = raw_action.get("arguments") or raw_action.get("args") or {}
    if isinstance(arguments, str):
        arguments = parse_jsonish(arguments)
    return {"name": name, "arguments": arguments}


def state_context(raw_state: Any) -> dict[str, Any]:
    if isinstance(raw_state, str):
        raw_state = parse_jsonish(raw_state)
    if not isinstance(raw_state, dict):
        return {}
    body = raw_state.get("state") if isinstance(raw_state.get("state"), dict) else raw_state
    context = body.get("context") if isinstance(body, dict) else None
    if isinstance(context, dict):
        return context
    if isinstance(body, dict) and any(key in body for key in ("last_tool_execution_result", "last_tool_name", "last_tool_output")):
        return body
    delta = body.get("diff_from_previous_state") if isinstance(body, dict) and isinstance(body.get("diff_from_previous_state"), dict) else {}
    outcome = delta.get("outcome") if isinstance(delta.get("outcome"), dict) else body.get("outcome") if isinstance(body, dict) else {}
    if not isinstance(outcome, dict):
        outcome = {}
    status = str(outcome.get("status") or "").lower()
    label = None
    if status in {"success", "succeeded", "completed"}:
        label = 1
    elif status in {"failure", "failed", "error", "exception"}:
        label = -1
    elif status in {"partial_success", "partial", "no_progress", "blocked"}:
        label = 0
    summary = outcome.get("summary") or ""
    return {
        "last_tool_execution_result": label,
        "last_tool_name": None,
        "last_tool_output": summary,
        "error_message": summary if label in {-1, 0} else "",
    }


def _iter_nested_values(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _iter_nested_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_values(item)


def state_indicates_task_completion(raw_state: Any) -> bool:
    if isinstance(raw_state, str):
        raw_state = parse_jsonish(raw_state)
    if not isinstance(raw_state, dict):
        return False

    body = raw_state.get("state") if isinstance(raw_state.get("state"), dict) else raw_state
    relational = body.get("relational") if isinstance(body, dict) else None
    task_completion = relational.get("task_completion") if isinstance(relational, dict) else None
    if isinstance(task_completion, dict) and isinstance(task_completion.get("success"), bool):
        return bool(task_completion["success"])

    text = compact_text(body)
    failure_markers = (
        "task failed",
        "failed final evaluator",
        "task_success false",
        "unsolved",
        "unresolved",
        "remaining unsolved",
        "remaining problems",
    )
    if any(marker in text for marker in failure_markers):
        return False

    process = body.get("process") if isinstance(body, dict) else None
    if isinstance(process, dict):
        remaining = process.get("remaining_stages") or process.get("remaining_requirements")
        if isinstance(remaining, list) and remaining:
            return False
        current_stage = compact_text(process.get("current_stage") or "")
        if any(marker in current_stage for marker in ("task completed", "task succeeded", "completed final evaluator")):
            return True

    for key, value in _iter_nested_values(body):
        normalized_key = str(key).lower()
        if normalized_key in {
            "task_success",
            "task_completed",
            "final_success",
            "all_verifiers_passed",
            "verifiers_passed",
            "verifier_success",
        } and value is True:
            return True
        if normalized_key in {"task_success", "task_completed", "final_success"} and value is False:
            return False

    if any(marker in text for marker in ("all verifiers passed", "task completed", "task succeeded", "final evaluator passed")):
        return True
    return False


def normalize_execution_status(label: Any, output_text: str) -> str:
    if isinstance(label, bool):
        return "success" if label else "failure"
    if isinstance(label, int):
        if label == 1:
            return "success"
        if label == -1:
            return "failure"
        if label == 0:
            return "no_op"
    if isinstance(label, str):
        stripped = label.strip().lower()
        if stripped in {"1", "success", "succeeded", "ok", "completed"}:
            return "success"
        if stripped in {"-1", "failure", "failed", "error", "exception"}:
            return "failure"
        if stripped in {"0", "stagnation", "no_progress", "blocked", "no_op"}:
            return "no_op"
    if any(marker in output_text for marker in ("error", "failed", "traceback", "not found", "permission denied")):
        return "failure"
    return "unknown"


def infer_error_signature(output_text: str, execution_status: str, action_type: str) -> str:
    if execution_status == "success":
        return "none"
    if any(marker in output_text for marker in ("not found", "no such file", "404", "does not exist", "no matching")):
        return "not_found"
    if any(marker in output_text for marker in ("permission denied", "forbidden", "unauthorized", "access denied")):
        return "permission_denied"
    if any(marker in output_text for marker in ("invalid", "bad request", "validation error", "parse error", "syntaxerror")):
        return "invalid_argument"
    if any(marker in output_text for marker in ("module not found", "command not found", "no module named", "missing", "dependency")):
        return "dependency_missing"
    if action_type in {"test", "validate"} and execution_status != "success":
        return "test_failed"
    if any(marker in output_text for marker in ("timed out", "timeout", "command timed out")):
        return "timeout"
    if any(marker in output_text for marker in ("policy", "confidential", "privacy")):
        return "policy_risk"
    if any(marker in output_text for marker in ("jsondecode", "could not parse", "parse")):
        return "parse_error"
    if any(marker in output_text for marker in ("traceback", "exception", "runtimeerror")):
        return "runtime_error"
    return "unknown" if execution_status != "success" else "none"


def first_word(value: str) -> str:
    return re.split(r"[\s_./-]+", value.strip().lower(), maxsplit=1)[0] if value.strip() else ""


def infer_action_type(name: str, arguments: Any) -> str:
    name_l = name.lower()
    args_l = compact_text(arguments)
    command = ""
    if isinstance(arguments, dict):
        command = str(arguments.get("command") or "").lower()
    combined = f"{name_l} {args_l} {command}"
    token = first_word(command or name_l)
    if any(test_token in combined for test_token in VALIDATE_TOKENS):
        return "test"
    if token in SEARCH_PREFIXES or name_l.startswith(SEARCH_PREFIXES):
        return "search"
    if token in READ_PREFIXES or name_l.startswith(READ_PREFIXES):
        return "read"
    if token in CREATE_PREFIXES or name_l.startswith(CREATE_PREFIXES):
        return "create"
    if token in UPDATE_PREFIXES or name_l.startswith(UPDATE_PREFIXES):
        return "update"
    if token in DELETE_PREFIXES or name_l.startswith(DELETE_PREFIXES):
        return "delete"
    if name_l.startswith(COMMUNICATE_PREFIXES):
        return "communicate"
    if "validate" in name_l or "verify" in name_l or "check" in name_l:
        return "validate"
    if name_l in {"execute_bash", "bash", "shell", "run"} or command:
        return "run"
    return "unknown"


def infer_object_type(name: str, arguments: Any, output_text: str) -> str:
    text = f"{name} {compact_text(arguments)} {output_text}".lower()
    if any(token in text for token in ("ticket", "incident", "issue")):
        return "ticket"
    if "case" in text:
        return "case"
    if any(token in text for token in ("customer", "contact")):
        return "customer"
    if "quote" in text:
        return "quote"
    if "account" in text:
        return "account"
    if any(token in text for token in ("calendar", "event")):
        return "calendar"
    if "permission" in text or "acl" in text:
        return "permission"
    if "comment" in text or "reply" in text:
        return "comment"
    if "label" in text:
        return "label"
    if "repository" in text or "repo" in text:
        return "repository"
    if "branch" in text:
        return "branch"
    if any(token in text for token in ("pip install", "apt install", "npm install", "package")):
        return "package"
    if any(token in text for token in ("process", "pid", "pytest", "test", "server")):
        return "process"
    if any(token in text for token in ("message", "email", "chat", "channel")):
        return "message"
    if any(token in text for token in ("file", "/app/", ".py", ".json", ".txt", ".c", ".cpp", "directory")):
        return "file"
    if any(token in text for token in ("select ", "insert ", "update ", "database", "sql")):
        return "database_row"
    return "unknown"


def infer_side_effect_type(action_type: str, execution_status: str, name: str, arguments: Any, output_text: str) -> str:
    text = f"{name} {compact_text(arguments)} {output_text}".lower()
    if action_type in {"test", "validate"}:
        return "validated" if execution_status == "success" else "failed_validation"
    if any(token in text for token in INSTALL_TOKENS):
        return "installed" if execution_status == "success" else "none"
    if execution_status != "success":
        return "none"
    if action_type in {"read", "search"}:
        return "retrieved"
    if action_type == "create":
        return "created"
    if action_type == "update":
        return "modified"
    if action_type == "delete":
        return "deleted"
    if action_type == "communicate":
        return "sent"
    if action_type == "run":
        return "executed"
    return "unknown"


def infer_progress_signal(action_type: str, side_effect_type: str, execution_status: str) -> str:
    if execution_status == "failure":
        return "negative"
    if side_effect_type == "failed_validation":
        return "negative"
    if side_effect_type in {"created", "modified", "deleted", "sent", "installed", "validated"}:
        return "positive"
    if side_effect_type in {"retrieved", "executed", "none"}:
        return "neutral"
    if execution_status == "no_op":
        return "neutral"
    return "unknown"


def infer_risk_signal(action_type: str, name: str, arguments: Any, output_text: str) -> str:
    text = f"{name} {compact_text(arguments)} {output_text}".lower()
    if any(token in text for token in ("confidential", "secret", "private", "pii", "ssn")):
        return "confidentiality_risk"
    if action_type == "delete" or any(token in text for token in DESTRUCTIVE_TOKENS):
        return "destructive_action"
    if any(token in text for token in ("archive", "publish", "send", "transfer", "grant")):
        return "irreversible_action"
    if "policy" in text or "compliance" in text:
        return "policy_risk"
    return "none"


def infer_uncertainty(execution_status: str, error_signature: str, side_effect_type: str, output_text: str) -> tuple[str, str]:
    if execution_status in {"success", "failure"} and error_signature != "unknown" and side_effect_type != "unknown":
        return "low", "none"
    if not output_text:
        return "high", "missing_context"
    if execution_status == "unknown" or side_effect_type == "unknown":
        return "medium", "heuristic_inference"
    return "medium", "ambiguous_observation"


def infer_nudge_state(event_state: dict[str, Any], *, task_completed: bool = False) -> dict[str, Any]:
    event = validate_event_state(event_state)
    action_type = event["action_type"]
    error_signature = event["error_signature"]
    progress_signal = event["progress_signal"]
    risk_signal = event["risk_signal"]
    side_effect_type = event["side_effect_type"]
    execution_status = event["execution_status"]

    information_sufficiency = "sufficient"
    missing_information_type = ["none"]
    information_gain = "low"
    recommended_abstract_action = "proceed"

    if task_completed:
        information_sufficiency = "sufficient"
        missing_information_type = ["none"]
        information_gain = "low"
        recommended_abstract_action = "finalize"
    elif risk_signal == "confidentiality_risk":
        information_sufficiency = "insufficient"
        missing_information_type = ["policy", "permission"]
        information_gain = "high"
        recommended_abstract_action = "retrieve"
    elif risk_signal == "policy_risk":
        information_sufficiency = "insufficient"
        missing_information_type = ["policy"]
        information_gain = "high"
        recommended_abstract_action = "retrieve"
    elif risk_signal in {"destructive_action", "irreversible_action"}:
        information_sufficiency = "insufficient"
        missing_information_type = ["current_state", "permission"]
        information_gain = "medium"
        recommended_abstract_action = "avoid"
    elif error_signature == "not_found":
        information_sufficiency = "insufficient"
        missing_information_type = ["object_id"]
        information_gain = "high"
        recommended_abstract_action = "search"
    elif error_signature == "dependency_missing":
        information_sufficiency = "insufficient"
        missing_information_type = ["dependency"]
        information_gain = "high"
        recommended_abstract_action = "inspect"
    elif error_signature == "permission_denied":
        information_sufficiency = "insufficient"
        missing_information_type = ["permission"]
        information_gain = "medium"
        recommended_abstract_action = "retrieve"
    elif error_signature in {"invalid_argument", "parse_error"}:
        information_sufficiency = "insufficient"
        missing_information_type = ["schema"]
        information_gain = "high"
        recommended_abstract_action = "inspect"
    elif error_signature == "test_failed" or side_effect_type == "failed_validation":
        information_sufficiency = "insufficient"
        missing_information_type = ["test_result"]
        information_gain = "high"
        recommended_abstract_action = "inspect"
    elif execution_status in {"failure", "partial", "no_op"}:
        information_sufficiency = "insufficient"
        missing_information_type = ["current_state"]
        information_gain = "high"
        recommended_abstract_action = "inspect"
    elif action_type in {"read", "search"}:
        information_gain = "high"
    elif action_type in {"test", "validate"}:
        information_gain = "high"
        recommended_abstract_action = "proceed" if progress_signal != "negative" else "inspect"
    elif action_type in {"update", "delete", "communicate"}:
        information_gain = "low"
    elif action_type == "run":
        information_gain = "medium"
        recommended_abstract_action = "validate" if progress_signal == "positive" else "inspect"
    elif action_type == "clarify":
        information_gain = "high"

    if progress_signal == "unknown" and recommended_abstract_action == "proceed":
        information_sufficiency = "unknown"
        missing_information_type = ["unknown"]
        information_gain = "unknown"
        recommended_abstract_action = "inspect"
    if progress_signal == "negative" and recommended_abstract_action == "proceed":
        information_sufficiency = "insufficient"
        missing_information_type = ["current_state"]
        information_gain = "high"
        recommended_abstract_action = "inspect"

    return validate_nudge_state(
        {
            "information_sufficiency": information_sufficiency,
            "missing_information_type": missing_information_type,
            "information_gain": information_gain,
            "recommended_abstract_action": recommended_abstract_action,
        }
    )


def canonical_event_from_action_state(action: Any, state: Any, *, include_uncertainty: bool = False) -> dict[str, str]:
    tool_call = normalize_tool_call(action)
    context = state_context(state)
    output_text = compact_text(context.get("last_tool_output") or context.get("error_message") or "")
    execution_status = normalize_execution_status(context.get("last_tool_execution_result"), output_text)
    action_type = infer_action_type(tool_call["name"], tool_call["arguments"])
    error_signature = infer_error_signature(output_text, execution_status, action_type)
    object_type = infer_object_type(tool_call["name"], tool_call["arguments"], output_text)
    side_effect_type = infer_side_effect_type(
        action_type,
        execution_status,
        tool_call["name"],
        tool_call["arguments"],
        output_text,
    )
    progress_signal = infer_progress_signal(action_type, side_effect_type, execution_status)
    risk_signal = infer_risk_signal(action_type, tool_call["name"], tool_call["arguments"], output_text)
    event_state = {
        "execution_status": execution_status,
        "error_signature": error_signature,
        "action_type": action_type,
        "object_type": object_type,
        "side_effect_type": side_effect_type,
        "progress_signal": progress_signal,
        "risk_signal": risk_signal,
    }
    if include_uncertainty:
        uncertainty_level, uncertainty_reason = infer_uncertainty(
            execution_status,
            error_signature,
            side_effect_type,
            output_text,
        )
        event_state.update(
            {
                "uncertainty_level": uncertainty_level,
                "uncertainty_reason": uncertainty_reason,
            }
        )
    return validate_event_state(event_state)


def validate_event_state(event_state: dict[str, Any]) -> dict[str, str]:
    if not isinstance(event_state, dict):
        raise ValueError("canonical event state must be a JSON object")
    event_state = dict(event_state)
    event_state.pop("schema_version", None)
    extra_fields = sorted(set(event_state) - set(EVENT_STATE_FIELDS))
    if extra_fields:
        raise ValueError(f"canonical event state contains unsupported fields: {extra_fields}")
    normalized = {}
    for field, allowed_values in REQUIRED_CATEGORICAL_FIELDS.items():
        value = event_state.get(field)
        if not isinstance(value, str) or value not in allowed_values:
            raise ValueError(f"{field} must be one of {sorted(allowed_values)}, got {value!r}")
        normalized[field] = value
    present_optional = [field for field in OPTIONAL_CATEGORICAL_FIELDS if field in event_state]
    if present_optional and set(present_optional) != set(OPTIONAL_CATEGORICAL_FIELDS):
        raise ValueError("uncertainty fields must be provided together or omitted together")
    for field in present_optional:
        allowed_values = OPTIONAL_CATEGORICAL_FIELDS[field]
        value = event_state.get(field)
        if not isinstance(value, str) or value not in allowed_values:
            raise ValueError(f"{field} must be one of {sorted(allowed_values)}, got {value!r}")
        normalized[field] = value
    return normalized


def canonical_event_json(event_state: dict[str, Any]) -> str:
    return json.dumps(validate_event_state(event_state), ensure_ascii=False, sort_keys=True)


def validate_nudge_state(nudge_state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(nudge_state, dict):
        raise ValueError("nudge state must be a JSON object")
    nudge_state = dict(nudge_state)
    nudge_state.pop("schema_version", None)
    extra_fields = sorted(set(nudge_state) - set(NUDGE_FIELDS))
    if extra_fields:
        raise ValueError(f"nudge state contains unsupported fields: {extra_fields}")
    normalized: dict[str, Any] = {}
    for field, allowed_values in NUDGE_CATEGORICAL_FIELDS.items():
        value = nudge_state.get(field)
        if not isinstance(value, str) or value not in allowed_values:
            raise ValueError(f"{field} must be one of {sorted(allowed_values)}, got {value!r}")
        normalized[field] = value
    for field, allowed_values in NUDGE_LIST_FIELDS.items():
        values = nudge_state.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{field} must be a non-empty list of {sorted(allowed_values)}")
        normalized_values = []
        for value in values:
            if not isinstance(value, str) or value not in allowed_values:
                raise ValueError(f"{field} values must be one of {sorted(allowed_values)}, got {value!r}")
            if value not in normalized_values:
                normalized_values.append(value)
        if "none" in normalized_values and len(normalized_values) > 1:
            raise ValueError(f"{field} cannot combine 'none' with other values")
        normalized[field] = normalized_values
    return normalized


def canonical_event_with_nudge_from_action_state(
    action: Any,
    state: Any,
    *,
    include_uncertainty: bool = False,
) -> dict[str, Any]:
    event_state = canonical_event_from_action_state(
        action,
        state,
        include_uncertainty=include_uncertainty,
    )
    return validate_canonical_event_with_nudge(
        {
            "canonical_event_state": event_state,
            "nudge": infer_nudge_state(
                event_state,
                task_completed=state_indicates_task_completion(state),
            ),
        }
    )


def validate_canonical_event_with_nudge(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("canonical event with nudge must be a JSON object")
    payload = dict(payload)
    payload.pop("schema_version", None)
    extra_fields = sorted(set(payload) - set(CANONICAL_EVENT_WITH_NUDGE_FIELDS))
    if extra_fields:
        raise ValueError(f"canonical event with nudge contains unsupported fields: {extra_fields}")
    return {
        "canonical_event_state": validate_event_state(payload.get("canonical_event_state")),
        "nudge": validate_nudge_state(payload.get("nudge")),
    }


def canonical_event_with_nudge_json(payload: dict[str, Any]) -> str:
    return json.dumps(validate_canonical_event_with_nudge(payload), ensure_ascii=False, sort_keys=True)


def flatten_canonical_fields(payload: Any, *, include_nudge: bool) -> dict[str, Any]:
    """Flatten a canonical payload to a ``{dotted_field: value}`` mapping.

    Accepts either the nested ``{"canonical_event_state": ..., "nudge": ...}`` form
    (``canonical_event_with_nudge`` target) or a bare event-state dict
    (``canonical_event_state`` target). It is deliberately lenient so it can be
    applied to raw, possibly-malformed model predictions as well as validated
    gold labels: missing/misshaped sections simply yield fewer keys, so an absent
    predicted field compares unequal to the gold value rather than raising.
    """
    if not isinstance(payload, dict):
        return {}
    if include_nudge:
        event = payload.get("canonical_event_state")
        nudge = payload.get("nudge")
    else:
        event = payload
        nudge = None
    flat: dict[str, Any] = {}
    if isinstance(event, dict):
        for field in EVENT_STATE_FIELDS:
            if field in event:
                flat[f"canonical_event_state.{field}"] = event[field]
    if include_nudge and isinstance(nudge, dict):
        for field in NUDGE_FIELDS:
            if field in nudge:
                flat[f"nudge.{field}"] = nudge[field]
    return flat


def canonical_value_matches(gold_value: Any, predicted_value: Any) -> bool:
    """Categorical equality for a single canonical field.

    List-valued fields (e.g. ``missing_information_type``) are compared as sets so
    ordering does not affect the match; scalar categorical fields use exact equality.
    """
    if isinstance(gold_value, list):
        if not isinstance(predicted_value, list):
            return False
        return sorted(str(v) for v in gold_value) == sorted(str(v) for v in predicted_value)
    return gold_value == predicted_value


def canonical_field_matches(
    gold_payload: Any,
    predicted_payload: Any,
    *,
    include_nudge: bool,
) -> dict[str, dict[str, Any]]:
    """Per-field categorical comparison of a predicted vs. gold canonical payload.

    The gold payload defines the scored field set (it is expected to be a
    validated/complete label). Each entry reports the gold value, the predicted
    value (``None`` when the prediction omits the field), and whether they match.
    """
    gold_flat = flatten_canonical_fields(gold_payload, include_nudge=include_nudge)
    predicted_flat = flatten_canonical_fields(predicted_payload, include_nudge=include_nudge)
    comparisons: dict[str, dict[str, Any]] = {}
    for field, gold_value in gold_flat.items():
        predicted_value = predicted_flat.get(field)
        comparisons[field] = {
            "gold": gold_value,
            "predicted": predicted_value,
            "match": canonical_value_matches(gold_value, predicted_value),
        }
    return comparisons


def event_state_value_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {field: Counter() for field in CATEGORICAL_FIELDS}
    for record in records:
        event = validate_event_state(record["canonical_event_state"])
        for field in CATEGORICAL_FIELDS:
            if field in event:
                counts[field][event[field]] += 1
    return {field: dict(sorted(counter.items())) for field, counter in counts.items() if counter}
