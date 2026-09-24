"""Standalone MCP server exposing the single-step EWM prediction as an MCP tool.

Follows the reference FastMCP servers in this repo —
the benchmark MCP servers under ``assets/*/purple-executors/mcp_react/server`` and
``assets/EnterpriseOps-Gym/purple-executors/mcp_react_ewm/server/ewm/mcp_server.py`` — a
FastMCP server speaking streamable-HTTP at ``/mcp`` with ``stateless_http=True`` and the
``fastmcp==2.6.1`` pin (see ``requirements.txt``). Those reference servers return plain
``application/json`` (which the EnterpriseOps-Gym ``benchmark/mcp_client.py`` parses with
``response.json()``); under fastmcp 2.6.1 stateless mode does that by default, while newer
fastmcp needs ``json_response=True`` — :func:`_build_http_app` sets it so both behave the
same. It wraps a *finer* primitive than the imagined-execution server:
instead of running a whole imagined execution, it exposes one world-model call —
``predict_state`` — so the agent (client-side) keeps any imagined-rollout loop and consults
the EWM the same way it consults email/calendar tools.

The server runs the EWM via either backend, selected by env (see
:func:`ejepa_wm.backends._ewm_generators.resolve_ewm_backend`):

* ``EWM_WORLD_MODEL_METHOD=vllm/<model>`` (+ ``WM_VLLM_BASE_URL`` / ``WM_VLLM_SERVER_PORT``)
  -> proxy to an OpenAI-compatible (vLLM) endpoint; or
* ``EWM_WORLD_MODEL_PATH=/path/to/checkpoint`` (or ``EWM_WORLD_MODEL_METHOD=transformers``)
  -> load a local HuggingFace checkpoint in-process.

The default WM_STATE mode is read from ``WM_STATE`` (``binary_error`` if unset).

Run:  ``MCP_PORT=12072 python -m ejepa_wm.server.ewm_predict``  (see ``run_local.sh``).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from fastmcp import FastMCP

from ejepa_wm.backends._ewm_generators import build_ewm_generator, resolve_ewm_backend
from ejepa_wm.server._predict import default_wm_state, run_predict_state

logger = logging.getLogger(__name__)

# `stateless_http` is a FastMCP() kwarg in 2.x but moved to http_app() in 3.x.
# Support both so the same server runs under the iems-pinned fastmcp==2.6.1 and newer.
try:
    mcp = FastMCP(name="enterpriseops_ewm_predict", stateless_http=True)
except TypeError:  # pragma: no cover - depends on fastmcp version
    mcp = FastMCP(name="enterpriseops_ewm_predict")


def _build_http_app() -> Any:
    """Build the streamable-HTTP ASGI app, forcing **plain-JSON** responses.

    Simple HTTP MCP clients — notably the EnterpriseOps-Gym benchmark ``MCPClient``, which
    parses every reply with ``response.json()`` — cannot read Server-Sent Events, so the
    server must answer ``application/json`` rather than the streamable-HTTP default of
    ``text/event-stream``. ``json_response`` / ``stateless_http`` live on ``http_app()`` in
    fastmcp 3.x (and ``stateless_http`` on the constructor in 2.x); try the richest kwargs
    and degrade gracefully so the same module runs under fastmcp 2.6.1 and newer.
    """
    for kwargs in (
        {"path": "/mcp", "json_response": True, "stateless_http": True},
        {"path": "/mcp", "json_response": True},
        {"path": "/mcp"},
    ):
        try:
            return mcp.http_app(**kwargs)
        except TypeError:  # pragma: no cover - depends on fastmcp version
            continue
    return mcp.http_app(path="/mcp")


_GENERATOR: Any = None


def _get_generator() -> Any:
    """Build the EWM generator once (lazy so import — and tests — don't load a model)."""
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = build_ewm_generator()
    return _GENERATOR


@mcp.tool()
def predict_state(
    system_prompt: str,
    user_prompt: str,
    action: list[dict[str, Any]],
    previous_state: dict[str, Any] | None = None,
    state_history: list[dict[str, Any]] | None = None,
    wm_state: str | None = None,
    interaction_index: int = 0,
    conversation_flow: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Look ahead with the Enterprise World Model — predict an action's outcome, don't run it.

    Call this to FORESEE the outcome of a SINGLE next action WITHOUT executing it, before any
    consequential or hard-to-undo step (create / update / delete / send / share / pay). The
    world model predicts whether that one action would succeed given the CURRENT state, and
    why it might fail.

    PER CALL — one concrete action:
    * Pass only ONE action with CONCRETE, fully-resolved arguments. Do NOT pass a multi-step
      plan in a single call, and do NOT use placeholders (e.g. ``"<id_from_previous_step>"``)
      — values that depend on an earlier step make the prediction meaningless. (If you pass
      more than one call, only the leading concrete action(s) are scored and ``warnings`` says
      what was skipped.)
    * If ``success: 0``, read ``error_message``, fix the action (or choose another), and
      predict again; execute the real tool only once it predicts ``success: 1``.

    MULTI-STEP LOOKAHEAD: if an ``imagine_trajectory`` tool is available, PREFER it for looking
    several steps ahead — it runs the whole imagined rollout for you in one call. Use
    ``predict_state`` for a quick check of a SINGLE concrete action's outcome. (If no rollout
    tool is available you may still chain ``predict_state`` manually: take the returned ``state``
    and pass it as ``previous_state`` on the next call with your next imagined action, repeating
    without executing anything; the first call needs no ``previous_state`` — the current real
    state is supplied for you.)

    Args:
        system_prompt: the task system prompt you are operating under (optional; derived from
            ``conversation_flow``/your context when omitted).
        user_prompt: the user query / task you are solving (likewise optional).
        action: the SINGLE next (real or imagined) tool call, e.g.
            ``[{"name": "create_calendar", "args": {...}}]`` (NOT executed).
        previous_state: the state to predict against. Omit on the first/grounding call (the
            current real state is supplied); for a chained imagined step, pass the ``state``
            returned by the previous call.
        state_history: optional recent ``[{interaction_index, state}]`` history.
        wm_state: prediction mode — ``binary_error`` (default), ``binary_error_stage``,
            ``tool_output``, or ``canonical_nudge`` (schema-based categorical event state +
            epistemic nudge). Falls back to the server's ``WM_STATE`` env default.
        interaction_index: step index in the task (default ``0``).
        conversation_flow: optional running flow (``system_message`` / ``user_message`` /
            ``ai_message`` / ``tool_result`` events) the current state/prompts are rebuilt from.

    Returns:
        ``{wm_state, state, success, error_message, current_stage, remaining_stages,
        tool_output, canonical_event_state, nudge, raw_prediction, parse_error,
        evaluated_action[, warnings]}`` — ``success``
        is ``1`` (predicted to succeed) or ``0`` (predicted to fail; see ``error_message``).
        ``canonical_event_state`` / ``nudge`` are populated only in ``canonical_nudge`` mode.
        ``state`` is the predicted next state — pass it as ``previous_state`` to chain the next
        imagined step. ``evaluated_action`` is the call(s) actually scored.
    """
    return run_predict_state(
        _get_generator(),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        action=action,
        previous_state=previous_state,
        state_history=state_history,
        wm_state=wm_state,
        interaction_index=interaction_index,
        conversation_flow=conversation_flow,
    )


@mcp.tool()
def generate(messages: list[dict[str, Any]], temperature: float = 0.0) -> dict[str, Any]:
    """Raw EWM chat-completion passthrough (the vLLM/transformers world model).

    Used by the imagined-trajectory rollout: the client-side loop (agent + WM) drives the
    rollout and calls this once per world-model step/candidate to reach a *remote* EWM as an
    MCP service, then assembles the imagined trajectory and injects it into the agent prompt.
    Unlike ``predict_state`` (which builds the per-step WM prompt and parses the result), this
    returns the model's raw text so the caller's rollout helpers can build the WM prompt
    exactly as they do in-process.

    Args:
        messages: OpenAI-style chat messages (``[{"role", "content"}, ...]``).
        temperature: sampling temperature (default ``0.0``).

    Returns:
        ``{"text": <raw model output>}``.
    """
    return {"text": _get_generator().generate_from_messages(messages, temperature=float(temperature))}


@mcp.tool()
def info() -> dict[str, Any]:
    """Report the resolved EWM backend configuration (no model is loaded)."""
    try:
        backend = resolve_ewm_backend().info()
    except Exception as exc:
        backend = {"error": str(exc)}
    return {
        "server": "enterpriseops_ewm_predict",
        "tool": "predict_state",
        "wm_state_default": default_wm_state(),
        **backend,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        _build_http_app(),
        host=os.getenv("MCP_HOST", "0.0.0.0"),
        port=int(os.getenv("MCP_PORT", "12072")),
    )
