"""Symbolic dependency helpers for open-loop action plans.

Plans may refer to values that are only available after earlier real tool calls,
for example ``{"user_id": "$step1.sys_id"}`` or ``"$vars.assignee_id"`` from a
prior step's ``bind`` selector. This module validates those dependencies without
inventing concrete identifiers and resolves them from observed tool outputs when
the corresponding execution history exists.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

STEP_REF_RE = re.compile(r"^\$step(?P<step>\d+)(?P<path>(?:\.[A-Za-z_][A-Za-z0-9_]*)+)$")
VARS_REF_RE = re.compile(r"^\$vars\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")

_ENTITY_WORDS = (
    "account",
    "asset",
    "case",
    "channel",
    "customer",
    "device",
    "file",
    "group",
    "incident",
    "issue",
    "location",
    "record",
    "request",
    "team",
    "ticket",
    "user",
)
_ASSIGNEE_FIELDS = {"assigned_to", "assignee", "owner", "reporter", "requester"}


@dataclass(frozen=True)
class SymbolicReference:
    consumer_step: int
    argument_path: tuple[str, ...]
    raw: str
    producer_step: int | None
    source_path: tuple[str, ...]
    variable_name: str | None = None
    destination_argument: str = ""
    producer_tool: str = ""
    consumer_tool: str = ""
    producer_type: str = "unknown"
    destination_type: str = "unknown"
    valid: bool = True
    reason: str = ""


def _tool_calls(action: Any) -> list[dict[str, Any]]:
    if isinstance(action, list):
        return [call for item in action for call in _tool_calls(item)]
    if not isinstance(action, dict):
        return []
    if isinstance(action.get("tool_calls"), list):
        return [call for item in action["tool_calls"] for call in _tool_calls(item)]
    function = action.get("function")
    if isinstance(function, dict) and function.get("name"):
        return [{"name": function["name"], "arguments": function.get("arguments", {})}]
    name = action.get("name") or action.get("tool") or action.get("tool_name")
    if not name:
        return []
    return [{"name": name, "arguments": action.get("arguments", action.get("args", {}))}]


def _first_tool_name(action: Any) -> str:
    calls = _tool_calls(action)
    return str(calls[0].get("name") or "") if calls else ""


def _argument_values(action: Any) -> list[Any]:
    return [call.get("arguments", {}) for call in _tool_calls(action)]


def _iter_symbolic_values(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, str):
        if STEP_REF_RE.match(value) or VARS_REF_RE.match(value):
            yield path, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_symbolic_values(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_symbolic_values(item, (*path, str(index)))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9]+", text.lower()))


def semantic_type_from_name(name: str) -> str:
    """Infer a coarse entity type from a tool, field, or argument name."""
    lowered = str(name or "").lower()
    if lowered in _ASSIGNEE_FIELDS:
        return "user"
    tokens = _tokens(lowered)
    for entity in _ENTITY_WORDS:
        if entity in tokens or lowered.endswith((f"_{entity}_id", f"{entity}_id")):
            return entity
    if lowered in {"id", "sys_id", "uuid"} or lowered.endswith("_id"):
        return "unknown"
    return "unknown"


def _field_semantic_type(path: tuple[str, ...]) -> str:
    for item in reversed(path):
        inferred = semantic_type_from_name(item)
        if inferred != "unknown":
            return inferred
    return "unknown"


def _types_compatible(producer_type: str, destination_type: str) -> bool:
    if producer_type == "unknown" or destination_type == "unknown":
        return True
    if producer_type == destination_type:
        return True
    return destination_type == "record"


def _parse_ref(raw: str) -> tuple[int | None, tuple[str, ...], str | None]:
    step_match = STEP_REF_RE.match(raw)
    if step_match:
        step = int(step_match.group("step"))
        path = tuple(part for part in step_match.group("path").split(".") if part)
        return step, path, None
    var_match = VARS_REF_RE.match(raw)
    if var_match:
        return None, (), var_match.group("name")
    return None, (), None


def _broken_reference_dict(reference: SymbolicReference) -> dict[str, Any]:
    return {
        "consumer_step": reference.consumer_step,
        "argument_path": list(reference.argument_path),
        "reference": reference.raw,
        "reason": reference.reason,
        "producer_tool": reference.producer_tool,
        "consumer_tool": reference.consumer_tool,
        "producer_type": reference.producer_type,
        "destination_type": reference.destination_type,
    }


def _validate_variable_reference(
    *,
    action_plan: list[Any],
    consumer_index: int,
    consumer_tool: str,
    path: tuple[str, ...],
    raw: str,
    variable_name: str,
    bound_vars: dict[str, int],
) -> SymbolicReference:
    producer_step = bound_vars.get(variable_name)
    reason = ""
    if producer_step is None:
        reason = "variable_undefined"
    elif producer_step >= consumer_index:
        reason = "producer_must_precede_consumer"
    producer_tool = (
        _first_tool_name(action_plan[producer_step - 1])
        if producer_step and 1 <= producer_step <= len(action_plan)
        else ""
    )
    producer_type = semantic_type_from_name(producer_tool)
    if producer_type == "unknown":
        producer_type = semantic_type_from_name(variable_name)
    destination_arg = path[-1] if path else ""
    destination_type = semantic_type_from_name(destination_arg)
    if not reason and not _types_compatible(producer_type, destination_type):
        reason = "dependency_type_mismatch"
    return SymbolicReference(
        consumer_step=consumer_index,
        argument_path=path,
        raw=raw,
        producer_step=producer_step,
        source_path=(),
        variable_name=variable_name,
        destination_argument=destination_arg,
        producer_tool=producer_tool,
        consumer_tool=consumer_tool,
        producer_type=producer_type,
        destination_type=destination_type,
        valid=not reason,
        reason=reason,
    )


def _validate_step_reference(
    *,
    action_plan: list[Any],
    consumer_index: int,
    consumer_tool: str,
    path: tuple[str, ...],
    raw: str,
    producer_step: int | None,
    source_path: tuple[str, ...],
) -> SymbolicReference:
    reason = ""
    if producer_step is None:
        reason = "invalid_reference_syntax"
    elif producer_step >= consumer_index:
        reason = "producer_must_precede_consumer"
    elif producer_step < 1 or producer_step > len(action_plan):
        reason = "producer_step_out_of_range"
    producer_tool = (
        _first_tool_name(action_plan[producer_step - 1])
        if producer_step and 1 <= producer_step <= len(action_plan)
        else ""
    )
    producer_type = semantic_type_from_name(producer_tool)
    if producer_type == "unknown":
        producer_type = _field_semantic_type(source_path)
    destination_arg = path[-1] if path else ""
    destination_type = semantic_type_from_name(destination_arg)
    if not reason and not _types_compatible(producer_type, destination_type):
        reason = "dependency_type_mismatch"
    return SymbolicReference(
        consumer_step=consumer_index,
        argument_path=path,
        raw=raw,
        producer_step=producer_step,
        source_path=source_path,
        destination_argument=destination_arg,
        producer_tool=producer_tool,
        consumer_tool=consumer_tool,
        producer_type=producer_type,
        destination_type=destination_type,
        valid=not reason,
        reason=reason,
    )


def validate_symbolic_references(action_plan: list[Any]) -> dict[str, Any]:
    """Validate symbolic references inside a sampled plan.

    Validation is intentionally semantic, not schema-complete: every ``$stepN`` producer must
    precede the consumer, every reference must parse, and inferred entity types must be
    compatible when both sides are known.
    """
    references: list[SymbolicReference] = []
    broken: list[dict[str, Any]] = []
    bound_vars: dict[str, int] = {}
    for producer_index, action in enumerate(action_plan, start=1):
        bind = action.get("bind") if isinstance(action, dict) else None
        if isinstance(bind, dict):
            for name, selector in bind.items():
                if isinstance(selector, dict) and str(selector.get("field") or "").strip():
                    bound_vars[str(name)] = producer_index

    for consumer_index, action in enumerate(action_plan, start=1):
        consumer_tool = _first_tool_name(action)
        for args in _argument_values(action):
            for path, raw in _iter_symbolic_values(args):
                producer_step, source_path, variable_name = _parse_ref(raw)
                if variable_name is not None:
                    reference = _validate_variable_reference(
                        action_plan=action_plan,
                        consumer_index=consumer_index,
                        consumer_tool=consumer_tool,
                        path=path,
                        raw=raw,
                        variable_name=variable_name,
                        bound_vars=bound_vars,
                    )
                else:
                    reference = _validate_step_reference(
                        action_plan=action_plan,
                        consumer_index=consumer_index,
                        consumer_tool=consumer_tool,
                        path=path,
                        raw=raw,
                        producer_step=producer_step,
                        source_path=source_path,
                    )
                references.append(reference)
                if reference.reason:
                    broken.append(_broken_reference_dict(reference))
    return {
        "valid": not broken,
        "references": [reference.__dict__ for reference in references],
        "valid_reference_count": sum(1 for reference in references if reference.valid),
        "broken_references": broken,
    }


def _flatten_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        records.append(value)
        for item in value.values():
            records.extend(_flatten_records(item))
    elif isinstance(value, list):
        for item in value:
            records.extend(_flatten_records(item))
    return records


def _record_matches(record: dict[str, Any], match: dict[str, Any]) -> bool:
    return all(record.get(key) == expected for key, expected in match.items())


def _select_bound_value(observation: Any, selector: dict[str, Any]) -> tuple[bool, Any, str]:
    field = str(selector.get("field") or "").strip()
    if not field:
        return False, None, "missing_bind_field"
    match = selector.get("match") if isinstance(selector.get("match"), dict) else {}
    candidates = [
        record
        for record in _flatten_records(observation)
        if not match or _record_matches(record, match)
    ]
    values = [record[field] for record in candidates if field in record]
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    if len(unique) == 1:
        return True, unique[0], ""
    if not unique:
        return False, None, "binding_not_found"
    return False, None, "binding_ambiguous"


def _find_path_value(value: Any, path: tuple[str, ...]) -> tuple[bool, Any, str]:
    current = value
    for part in path:
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list):
            if part.isdigit() and int(part) < len(current):
                current = current[int(part)]
                continue
            matches = [_find_path_value(item, (part,)) for item in current]
            found = [item[1] for item in matches if item[0]]
            unique = []
            for item in found:
                if item not in unique:
                    unique.append(item)
            if len(unique) == 1:
                current = unique[0]
                continue
            return False, None, "reference_ambiguous" if unique else "reference_not_found"
        return _find_unique_key(value, path[-1])
    return True, current, ""


def _find_unique_key(value: Any, key: str) -> tuple[bool, Any, str]:
    matches = []
    for record in _flatten_records(value):
        if key in record:
            matches.append(record[key])
    unique = []
    for item in matches:
        if item not in unique:
            unique.append(item)
    if len(unique) == 1:
        return True, unique[0], ""
    if not unique:
        return False, None, "reference_not_found"
    return False, None, "reference_ambiguous"


def _bindings_from_steps(
    plan_steps: list[dict[str, Any]], observations: list[Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    variables: dict[str, Any] = {}
    unresolved: list[dict[str, Any]] = []
    for index, step in enumerate(plan_steps[: len(observations)], start=1):
        bind = step.get("bind") if isinstance(step, dict) else None
        if not isinstance(bind, dict):
            continue
        for name, selector in bind.items():
            if not isinstance(selector, dict):
                unresolved.append({"step": index, "variable": str(name), "reason": "invalid_bind"})
                continue
            ok, value, reason = _select_bound_value(observations[index - 1], selector)
            if ok:
                variables[str(name)] = value
            else:
                unresolved.append({"step": index, "variable": str(name), "reason": reason})
    return variables, unresolved


def _resolve_symbolic_scalar(
    value: str,
    path: tuple[str, ...],
    *,
    observations: list[Any],
    variables: dict[str, Any],
    bindings: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
) -> Any:
    producer_step, source_path, variable_name = _parse_ref(value)
    if variable_name is not None:
        if variable_name in variables:
            bindings.append(
                {
                    "path": list(path),
                    "reference": value,
                    "source": "vars",
                    "name": variable_name,
                }
            )
            return variables[variable_name]
        unresolved.append({"path": list(path), "reference": value, "reason": "variable_unresolved"})
        return value
    if producer_step is None:
        return value
    if producer_step < 1 or producer_step > len(observations):
        unresolved.append(
            {
                "path": list(path),
                "reference": value,
                "reason": "producer_output_missing",
            }
        )
        return value
    ok, found, reason = _find_path_value(observations[producer_step - 1], source_path)
    if ok:
        bindings.append(
            {
                "path": list(path),
                "reference": value,
                "source": f"step{producer_step}",
                "field_path": list(source_path),
            }
        )
        return found
    unresolved.append({"path": list(path), "reference": value, "reason": reason})
    return value


def _resolve_value(
    value: Any,
    path: tuple[str, ...],
    *,
    observations: list[Any],
    variables: dict[str, Any],
    bindings: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
) -> Any:
    if isinstance(value, str):
        return _resolve_symbolic_scalar(
            value,
            path,
            observations=observations,
            variables=variables,
            bindings=bindings,
            unresolved=unresolved,
        )
    if isinstance(value, dict):
        return {
            key: _resolve_value(
                item,
                (*path, str(key)),
                observations=observations,
                variables=variables,
                bindings=bindings,
                unresolved=unresolved,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_value(
                item,
                (*path, str(index)),
                observations=observations,
                variables=variables,
                bindings=bindings,
                unresolved=unresolved,
            )
            for index, item in enumerate(value)
        ]
    return value


def _resolve_action_node(
    node: Any,
    *,
    observations: list[Any],
    variables: dict[str, Any],
    bindings: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
) -> None:
    if isinstance(node, list):
        for item in node:
            _resolve_action_node(
                item,
                observations=observations,
                variables=variables,
                bindings=bindings,
                unresolved=unresolved,
            )
        return
    if not isinstance(node, dict):
        return
    if isinstance(node.get("tool_calls"), list):
        for item in node["tool_calls"]:
            _resolve_action_node(
                item,
                observations=observations,
                variables=variables,
                bindings=bindings,
                unresolved=unresolved,
            )
        return
    function = node.get("function") if isinstance(node.get("function"), dict) else None
    if function is not None:
        if isinstance(function.get("arguments"), dict):
            function["arguments"] = _resolve_value(
                function["arguments"],
                (),
                observations=observations,
                variables=variables,
                bindings=bindings,
                unresolved=unresolved,
            )
        return
    if any(key in node for key in ("name", "tool", "tool_name")) and isinstance(
        node.get("arguments"), dict
    ):
        node["arguments"] = _resolve_value(
            node["arguments"],
            (),
            observations=observations,
            variables=variables,
            bindings=bindings,
            unresolved=unresolved,
        )


def resolve_symbolic_references(
    action: Any,
    *,
    observations: list[Any],
    plan_steps: list[dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve symbolic references in one action from actual prior observations.

    The returned detail records unresolved bindings/references. Callers must not execute an action
    while ``detail["unresolved"]`` is non-empty.
    """
    plan_steps = plan_steps or []
    variables, unresolved = _bindings_from_steps(plan_steps, observations)
    resolved = copy.deepcopy(action)
    bindings: list[dict[str, Any]] = []
    _resolve_action_node(
        resolved,
        observations=observations,
        variables=variables,
        bindings=bindings,
        unresolved=unresolved,
    )

    return resolved, {"bindings": bindings, "unresolved": unresolved}


def has_symbolic_references(value: Any) -> bool:
    if isinstance(value, str):
        return bool(STEP_REF_RE.match(value) or VARS_REF_RE.match(value))
    if isinstance(value, dict):
        return any(has_symbolic_references(item) for item in value.values())
    if isinstance(value, list):
        return any(has_symbolic_references(item) for item in value)
    return False


__all__ = [
    "STEP_REF_RE",
    "VARS_REF_RE",
    "has_symbolic_references",
    "resolve_symbolic_references",
    "semantic_type_from_name",
    "validate_symbolic_references",
]
