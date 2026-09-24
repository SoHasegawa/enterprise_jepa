"""Single-step EWM prediction — the core the MCP ``predict_state`` tool wraps.

Kept free of any ``fastmcp`` import so it is unit-testable with a stub generator and
without the server's optional serving dependencies. The MCP wrapper in
:mod:`ejepa_wm.server.ewm_predict` only adds the tool registration + a cached generator.

The Enterprise World Model is a **per-step** model: it predicts the outcome of the *next*
action conditioned on the *current* state. So this core (a) only ever scores the leading
concrete action(s) — dependent/placeholder-laden later steps are dropped with a warning so
the agent re-predicts each step once its inputs are known — and (b) can reconstruct the
current enterprise state from a forwarded ``conversation_flow`` (matching the imagined-mode
backend) instead of relying on the agent to hand-assemble a state dict.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from ejepa_wm.backends import _ewm_finetuning as ft
from ejepa_wm.backends import _ewm_runtime as ewm
from ejepa_wm.backends._ewm_generators import normalize_planned_calls
from ejepa_wm.backends._ewm_runtime import (
    WM_STATE_BINARY_ERROR,
    WM_STATES,
    predict_wm_feedback,
)

logger = logging.getLogger(__name__)

# Angle-bracket placeholders the agent leaves for values it does not have yet, e.g.
# ``"<to_fill_after_creation>"`` / ``"<primary_calendar_id>"``. Their presence means the
# action is not a concrete next step and the WM cannot meaningfully score it.
_PLACEHOLDER_RE = re.compile(r"<[^<>]{1,80}>")


def default_wm_state(env: dict[str, str] | None = None) -> str:
    """Server default for the WM_STATE mode (``WM_STATE`` env, else ``binary_error``)."""
    source = env if env is not None else os.environ
    value = (source.get("WM_STATE") or "").strip().lower()
    return value if value in WM_STATES else WM_STATE_BINARY_ERROR


def _prompt_from_flow(conversation_flow: list[dict[str, Any]], etype: str) -> str:
    for event in conversation_flow or []:
        if isinstance(event, dict) and event.get("type") == etype:
            return str(event.get("content", "") or "")
    return ""


def _collect_placeholders(obj: Any, out: list[str]) -> None:
    if isinstance(obj, str):
        out.extend(_PLACEHOLDER_RE.findall(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_placeholders(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_placeholders(value, out)


def _select_evaluable_action(
    action: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Pick the leading **concrete** call(s) to score and explain what was dropped.

    EWM scores one step against the current state. If the agent passes a multi-step plan,
    only the leading prefix of calls with fully-resolved (placeholder-free) arguments is
    predicted; later/dependent calls are dropped with a warning telling the agent to
    re-predict each once its inputs are known.
    """
    action = action or []
    warnings: list[str] = []
    if not action:
        return action, warnings

    concrete_prefix: list[dict[str, Any]] = []
    for call in action:
        found: list[str] = []
        _collect_placeholders(call, found)
        if found:
            break
        concrete_prefix.append(call)

    evaluated = concrete_prefix or action[:1]  # always score at least the first call

    leftover_placeholders: list[str] = []
    _collect_placeholders(evaluated, leftover_placeholders)
    if leftover_placeholders:
        warnings.append(
            "The action still contains unresolved placeholder argument(s) "
            f"{sorted(set(leftover_placeholders))}; this prediction is unreliable. Run the "
            "prerequisite step(s) first, then call predict_state with concrete values."
        )

    dropped = len(action) - len(evaluated)
    if dropped > 0:
        warnings.append(
            f"predict_state scores ONE step against the current state. {dropped} later call(s) "
            "were not scored because they depend on earlier results (and/or use placeholders). "
            f"Only the first {len(evaluated)} concrete action(s) were predicted — re-call "
            "predict_state for each subsequent step after you have its real inputs."
        )
    elif len(action) > 1:
        warnings.append(
            f"Note: the {len(action)} calls were scored as a single step. If they are "
            "sequential/dependent rather than issued together, predict them one at a time."
        )
    return evaluated, warnings


def run_predict_state(
    generator: Any,
    *,
    system_prompt: str = "",
    user_prompt: str = "",
    action: list[dict[str, Any]],
    previous_state: dict[str, Any] | None = None,
    state_history: list[dict[str, Any]] | None = None,
    wm_state: str | None = None,
    interaction_index: int = 0,
    conversation_flow: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ask the EWM ``generator`` to predict the outcome of the next ``action``.

    Inputs are the required EWM inputs; output is the EWM prediction. For
    ``wm_state='binary_error'`` the ``state`` carries the predicted binary tool-execution
    result plus the error message — the agent can treat this tool like any other (email,
    calendar) tool, calling it once per consequential step.

    If ``conversation_flow`` is given, the current enterprise state, multi-step history and
    (absent explicit values) the prompts are reconstructed from it — the same way the
    imagined-mode backend derives state — so the per-step prediction is state-aware without
    the agent hand-assembling a state dict.
    """
    mode = (wm_state or default_wm_state()).strip().lower()
    if mode not in WM_STATES:
        raise ValueError(f"wm_state must be one of {WM_STATES}; got {wm_state!r}")

    # (b) Reconstruct state + prompts from a forwarded conversation flow (explicit args win).
    if conversation_flow:
        flow_state, flow_history = ewm.enterprise_state_from_flow(conversation_flow)
        if previous_state is None:
            previous_state = flow_state
        if state_history is None:
            state_history = flow_history
        if not system_prompt:
            system_prompt = _prompt_from_flow(conversation_flow, "system_message")
        if not user_prompt:
            user_prompt = _prompt_from_flow(conversation_flow, "user_message")

    # (guard) score only the leading concrete step; warn about dropped/placeholder calls.
    evaluated_action, warnings = _select_evaluable_action(action or [])

    prev = previous_state if previous_state else ft.make_blank_state()
    feedback = predict_wm_feedback(
        generator,
        system_prompt,
        user_prompt,
        prev,
        normalize_planned_calls(evaluated_action),
        state_history or [],
        mode,
        interaction_index,
    )[0]

    result = {
        "wm_state": mode,
        "state": feedback.get("predicted_state"),
        "success": 1 if feedback.get("predicted_success") else 0,
        "error_message": feedback.get("predicted_error_message") or "",
        "current_stage": feedback.get("predicted_current_stage"),
        "remaining_stages": feedback.get("predicted_remaining_stages"),
        "tool_output": feedback.get("predicted_tool_output") or "",
        # canonical_nudge mode also carries the categorical event state + epistemic nudge.
        "canonical_event_state": feedback.get("predicted_canonical_event"),
        "nudge": feedback.get("predicted_nudge"),
        "raw_prediction": feedback.get("raw_prediction") or "",
        "parse_error": feedback.get("parse_error"),
        "evaluated_action": evaluated_action,
    }
    if warnings:
        result["warnings"] = warnings
    return result


__all__ = ["default_wm_state", "run_predict_state"]
