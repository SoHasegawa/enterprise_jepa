"""World-Model-guided ReAct loop for WorkBench (``ejepa_wm``-backed).

This is the WorkBench analogue of EnterpriseOps-Gym's ``mcp_react/wm_react.py``
and CRMArenaPro's ``mcp_react/wm_react.py``. WorkBench has no MCP layer (like
CRMArenaPro), and unlike either reference its baseline agent loop
(``src.evals.agent.run_agent``) is a **free function**, not a class with an
overridable decision-point method. So instead of subclassing an orchestrator
or agent class, this module reimplements ``run_agent``'s exact ReAct loop
control flow (message construction, ``parse_action``, tool dispatch,
iteration/time limits, ``Final Answer`` termination) verbatim, substituting
the single ``call_llm(...)`` decision point with a ``ejepa_wm``-guided step.
When ``WM_STRATEGY`` is unset/``none`` (or the World Model fails to build),
:func:`run_single_task_with_wm` delegates straight to the upstream
``src.evals.inference._run_single_task`` unmodified, so the executor is always
runnable without any WM configured.

The World Model itself is the shared, pluggable :mod:`ejepa_wm` package
(``src/ejepa_wm``); this module only wires it into WorkBench's ReAct loop via
the generic ``conversation_flow`` event contract (``system_message`` /
``user_message`` / ``ai_message`` / ``tool_result``). WM behavior is entirely
env-driven (``WM_STRATEGY`` / ``WM_BACKEND`` / ``WM_N`` / the ``EWM_*`` /
``WM_EWM_MCP_URL`` / ``WM_EWM_JEPA_CHECKPOINT`` knobs), exactly like the gym
and CRMArenaPro executors.

Seven strategies are supported (selected by ``WM_STRATEGY``):

* ``selection`` (pairs well with ``WM_BACKEND=ewm_predict``) -- sample
  ``WM_N`` candidate actions from the policy LLM, let the WM rank them, and
  commit the best one.
* ``prompt_injection`` (pairs with ``WM_BACKEND=ewm_imagined`` or ``llm``) --
  ask the WM to ``advise`` and inject the returned guidance block before the
  next action.
* ``itp_i`` (pairs with ``WM_BACKEND=ewm_imagined``) -- Imagine-Then-Plan
  inference: ask the WM to imagine a trajectory and inject it for policy
  reflection before the next action.
* ``revision`` / ``reference`` -- predict feedback for the proposed action and
  either revise immediately or expose the prediction at the following decision
  step.
* ``beam_plan`` (auto-selects ``WM_BACKEND=ewm_imagined``; requires a JEPA
  canonical-event checkpoint, ``WM_EWM_JEPA_CHECKPOINT``) -- JEPA MPC
  lookahead: every ``WM_BEAM_MPC_EXECUTE_STEPS`` turns the WM proposes
  candidate actions per horizon step, scores the imagined outcomes with the
  canonical-event heads (LLM-free), and hands the agent a plan to follow; the
  step-0 action can override the agent's baseline when it beats it by more
  than ``WM_BEAM_PLAN_SCORE_MARGIN``. With any non-JEPA backend it degrades to
  plain ReAct. See
  :meth:`ejepa_wm.backends.ewm_imagined.EwmImaginedWorldModel.beam_plan_step`.
* ``hier_latent_cem`` (auto-selects ``WM_BACKEND=ewm_imagined``; same JEPA
  canonical-event checkpoint requirement as ``beam_plan``) -- a second
  JEPA-only MPC lookahead: one LLM call proposes a handful of diverse anchor
  actions, then the world model CEM-refines thousands of continuous
  latent-action trajectories around them (LLM-free, decode-free) and only
  decodes the converged best plan. With any non-JEPA backend it degrades to
  plain ReAct. See
  :meth:`ejepa_wm.backends.ewm_imagined.EwmImaginedWorldModel.hier_latent_cem_step`.

``WM_STRATEGY`` unset / ``none`` -> plain ``run_agent`` behavior (no WM), so
the executor is always runnable without an EWM server.

WM guidance only wraps the ReAct text-parsing loop (``run_agent``); a task
requesting ``structured_outputs=True`` (native tool-calling) is not
WM-guided -- :func:`run_single_task_with_wm` delegates to the plain
``_run_single_task`` for those, matching the scope of the ported strategies.

NOTE on scope: the bundled EWM world model is fine-tuned on EnterpriseOps-Gym
tool outcomes. Pointing it at WorkBench's calendar/email/analytics/CRM/
project-management actions is an intentional **out-of-domain / over-fit
probe** -- it is expected to guide gym well and WorkBench poorly, which is the
point of wiring it here (mirroring the same caveat in CRMArenaPro's
``wm_react.py``).
"""

from __future__ import annotations

import contextvars
import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The model_name a WM chat_fn call should use, scoped per in-flight task via
# contextvars so concurrent `asyncio.to_thread` workers (each processing a
# different task, in principle with a different model_name) don't race on a
# shared mutable global. `asyncio.to_thread` copies the calling coroutine's
# context into the new thread, so a `.set()` done just before dispatching a
# task is visible only to that task's thread.
_active_model_name: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "workbench_wm_model_name", default=None
)

# One world model per TASK, not per process. The JEPA MPC state
# (``_beam_imagined_plan``, ``_beam_plan_cursor``, the critic counters) lives on
# the world-model instance and ``reset_episode()`` wipes all of it, so a single
# shared instance cannot serve two tasks at once: with ``config.max_parallel > 1``
# one task's reset destroyed the other's live plan and the injected advice could
# describe a different task's records -- silently, with no error. The multi-GB
# generator is still built once per process and shared (``WM_SHARE_MODEL_WEIGHTS``),
# so per-task construction stays cheap.
_wm_build_failed = False


def _ensure_ejepa_wm_on_path() -> None:
    """Put the repo's ``src`` dir on ``sys.path`` so ``ejepa_wm`` is importable.

    Layout: ``benchmarks/assets/WorkBench/purple-executors/mcp_react/wm_react.py``,
    so ``parents[4]`` is the benchmarks repo root and ``parents[4]/src`` holds
    ``ejepa_wm``.
    """
    src = Path(__file__).resolve().parents[4] / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _make_chat_fn(agent_module: Any):
    """A ``ejepa_wm.ChatFn`` (role-dict messages -> reply text) over WorkBench's own LLM client.

    WorkBench's ``call_llm(model_name, system_prompt, human_msg, temperature)``
    expects a single system + single user message, so a multi-message WM
    prompt is folded: leading ``system`` messages become the system prompt,
    everything else is joined into one user message.
    """

    def chat_fn(messages: list[dict[str, str]]) -> str:
        model_name = _active_model_name.get()
        if not model_name:
            raise RuntimeError("wm_react: no active model_name for WM chat_fn")
        system_chunks = [m.get("content", "") for m in messages if m.get("role") == "system"]
        other_chunks = [m.get("content", "") for m in messages if m.get("role") != "system"]
        system_prompt = "\n\n".join(c for c in system_chunks if c) or "You are a helpful assistant."
        human_msg = "\n\n".join(c for c in other_chunks if c) or "Continue."
        try:
            return agent_module.call_llm(model_name, system_prompt, human_msg, 0.0)
        except Exception as exc:
            logger.warning("wm_react: WM chat_fn call failed: %s", exc)
            return ""

    return chat_fn


def _get_world_model(agent_module: Any) -> tuple[Any, Any]:
    """Build a fresh World Model for ONE task from the ``WM_*`` env vars.

    Per task, so each task owns its MPC/episode state -- see the note on
    ``_wm_build_failed``. Model weights are shared process-wide by the
    ``ejepa_wm`` generator cache, so this does not reload the checkpoint.
    """
    global _wm_build_failed
    if _wm_build_failed:
        return None, None
    _ensure_ejepa_wm_on_path()
    # These executors build one world model per TASK so that concurrent tasks
    # (``config.max_parallel > 1``) cannot share episode state. Share the model
    # weights process-wide so that costs one checkpoint load, not one per task.
    os.environ.setdefault("WM_SHARE_MODEL_WEIGHTS", "1")
    wm = None
    wm_config = None
    try:
        from ejepa_wm import build_world_model, wm_config_from_env

        wm_config = wm_config_from_env()
        if wm_config.strategy != "none":
            wm = build_world_model(wm_config, chat_fn=_make_chat_fn(agent_module))
    except Exception as exc:  # ejepa_wm missing / WM build failed -> plain ReAct
        # Latch the failure: it is a configuration problem, identical for every
        # task, so retrying (and re-logging) it 100+ times is pure noise.
        _wm_build_failed = True
        logger.warning("wm_react: WM unavailable (%s); using plain ReAct", exc)
        return None, None
    logger.info(
        "WorkBench mcp_react: wm=%s strategy=%s backend=%s n=%s",
        "on" if wm is not None else "off",
        getattr(wm_config, "strategy", "none"),
        getattr(wm_config, "backend", "noop"),
        getattr(wm_config, "n", 1),
    )
    return wm, wm_config


def _tool_dict(agent_module: Any, tool: Any) -> dict[str, Any]:
    """Render one WorkBench ``Tool`` as a gym-style tool dict for ``conversation_flow``."""
    schema = agent_module.tool_to_openai_schema(tool)["function"]
    return {
        "name": schema["name"],
        "description": schema.get("description", ""),
        "inputSchema": schema.get("parameters", {}),
    }


def _tool_calls_from_action(agent_module: Any, action: str, action_input: Any) -> list[dict[str, Any]]:
    if action in (agent_module.FINAL_ANSWER, agent_module.PARSE_ERROR):
        return []
    args = action_input if isinstance(action_input, dict) else {"input": str(action_input)}
    return [{"name": action, "args": args}]


def _ai_event(agent_module: Any, response_text: str, action: str, action_input: Any) -> dict[str, Any]:
    return {
        "type": "ai_message",
        "content": response_text,
        "tool_calls": _tool_calls_from_action(agent_module, action, action_input),
    }


def _action_summary(agent_module: Any, response_text: str) -> str:
    action, action_input = agent_module.parse_action(response_text)
    if action and action != agent_module.PARSE_ERROR:
        return f"{action}: {str(action_input)[:160]}"
    return response_text[:160]


def _response_with_call(call: dict[str, Any]) -> str:
    """Rebuild a raw JSON-blob response string that WorkBench's own ``parse_action``
    parses back into the WM-overridden call, so a beam/hier_cem override flows through
    the normal action-dispatch path unmodified."""
    name = str(call.get("name", "") or "")
    args = call.get("arguments") or call.get("args") or {}
    blob = json.dumps({"action": name, "action_input": args})
    return f"Thought: World-model override.\nAction:\n```\n{blob}\n```"


def _beam_or_hier_step(
    *,
    strategy: str,
    agent_module: Any,
    wm: Any,
    wm_config: Any,
    system_prompt: str,
    human_msg: str,
    conversation_flow: list[dict[str, Any]],
    model_name: str,
    temperature: float,
    iteration: int,
    wm_steps: list[dict[str, Any]],
) -> str:
    """``beam_plan`` / ``hier_latent_cem``: JEPA MPC lookahead, advisory by default."""
    method_name = "beam_plan_step" if strategy == "beam_plan" else "hier_latent_cem_step"
    injection_method = "beam_injection_text" if strategy == "beam_plan" else "hier_injection_text"
    support_check = "supports_beam_plan" if strategy == "beam_plan" else "supports_hier_cem"

    step: dict[str, Any] = {"iteration": iteration, "strategy": strategy, "backend": wm_config.backend}
    # Guard: both JEPA MPC strategies need the ewm_imagined + canonical-event backend;
    # otherwise behave exactly like the plain no-WM agent (one sample, no override).
    supported = hasattr(wm, method_name) and bool(getattr(wm, support_check, lambda: False)())
    if not supported:
        step["fallback"] = True
        wm_steps.append(step)
        return agent_module.call_llm(model_name, system_prompt, human_msg, temperature)

    # (a) Show the cached imagined plan to the agent BEFORE it acts (transient only --
    # not persisted to the scratchpad, so the WM always re-plans from the clean flow).
    injection = getattr(wm, injection_method)()
    guided_human_msg = f"{human_msg}\n\n{injection}" if injection else human_msg
    # (b) Sample the baseline next action once.
    response = agent_module.call_llm(model_name, system_prompt, guided_human_msg, temperature)
    # (c) seed_calls = the baseline response's tool call(s).
    action, action_input = agent_module.parse_action(response)
    seed_calls = [
        {"name": tc["name"], "arguments": tc["args"]}
        for tc in _tool_calls_from_action(agent_module, action, action_input)
    ]
    user_query = next(
        (e.get("content", "") for e in conversation_flow if e.get("type") == "user_message"), ""
    )
    # (d) One MPC decision (re-plan / follow / override). Never crashes the run.
    try:
        method = getattr(wm, method_name)
        res = method(conversation_flow, seed_calls=seed_calls, user_query=user_query)
        detail = res.detail or {}
        step.update({"injected_chars": len(injection or ""), "detail": detail})
    except Exception as exc:
        logger.warning("wm_react: %s failed (%s); baseline action", method_name, exc)
        step.update({"error": str(exc), "fallback": True})
        wm_steps.append(step)
        return response
    wm_steps.append(step)
    # (e) Override the baseline action with the plan's step-0 calls when it won.
    if detail.get("override_applied") and detail.get("calls"):
        override = _response_with_call(detail["calls"][0])
        logger.info(
            "wm_react: %s overrode baseline (%s)",
            strategy,
            detail.get("override_reason") or detail.get("reason"),
        )
        return override
    return response


def _selection_step(
    *,
    agent_module: Any,
    wm: Any,
    wm_config: Any,
    system_prompt: str,
    human_msg: str,
    conversation_flow: list[dict[str, Any]],
    model_name: str,
    temperature: float,
    iteration: int,
    wm_steps: list[dict[str, Any]],
) -> str:
    """``selection``: sample ``WM_N`` candidates, let the WM rank them."""
    n = max(1, int(wm_config.n))
    temp = max(temperature, 0.7)  # diversity for ranking
    candidates = [agent_module.call_llm(model_name, system_prompt, human_msg, temp) for _ in range(n)]
    candidates = [c for c in candidates if c] or [
        agent_module.call_llm(model_name, system_prompt, human_msg, temperature)
    ]
    events = []
    for candidate in candidates:
        action, action_input = agent_module.parse_action(candidate)
        events.append(_ai_event(agent_module, candidate, action, action_input))
    idx = 0
    step = {
        "iteration": iteration,
        "strategy": "selection",
        "backend": wm_config.backend,
        "num_candidates": len(candidates),
    }
    try:
        sel = wm.select(conversation_flow, events)
        if 0 <= sel.index < len(candidates):
            idx = sel.index
        step.update(
            {
                "index": idx,
                "candidates": [_action_summary(agent_module, c) for c in candidates],
                "chosen": _action_summary(agent_module, candidates[idx]),
                "detail": sel.detail,
            }
        )
    except Exception as exc:
        logger.warning("wm_react: select failed (%s); candidate 0", exc)
        step.update({"index": idx, "error": str(exc), "fallback": True})
    wm_steps.append(step)
    return candidates[idx]


def _prompt_injection_step(
    *,
    agent_module: Any,
    wm: Any,
    wm_config: Any,
    system_prompt: str,
    human_msg: str,
    conversation_flow: list[dict[str, Any]],
    model_name: str,
    temperature: float,
    iteration: int,
    wm_steps: list[dict[str, Any]],
    state_history: list[str] | None = None,
) -> str:
    """``prompt_injection`` / ``itp_i``: ask the WM to advise, inject, sample once."""
    strategy = wm_config.strategy
    step = {"iteration": iteration, "strategy": strategy, "backend": wm_config.backend}
    advice_text = ""
    try:
        advice = wm.advise(conversation_flow, history=state_history)
        advice_text = advice.text or ""
        step.update({"injected_chars": len(advice_text), "detail": advice.detail})
        if state_history is not None:
            current_state = (advice.detail or {}).get("current_state")
            if current_state:
                state_history.append(str(current_state))
        if strategy == "itp_i":
            step["itp_i_policy_reflect_calls"] = 1
    except Exception as exc:
        logger.warning("wm_react: advise failed (%s); no injection", exc)
        step.update({"error": str(exc), "fallback": True})
    wm_steps.append(step)
    guided_human_msg = f"{human_msg}\n\n{advice_text}" if advice_text else human_msg
    return agent_module.call_llm(model_name, system_prompt, guided_human_msg, temperature)


def _wm_step(
    *,
    agent_module: Any,
    wm: Any,
    wm_config: Any,
    system_prompt: str,
    human_msg: str,
    conversation_flow: list[dict[str, Any]],
    model_name: str,
    temperature: float,
    iteration: int,
    wm_steps: list[dict[str, Any]],
    pending_reference: list[str] | None = None,
    state_history: list[str] | None = None,
) -> str:
    """Produce the next agent response, consulting the WM per ``WM_STRATEGY``."""
    strategy = wm_config.strategy
    if pending_reference is None:
        pending_reference = [""]
    kwargs = {
        "agent_module": agent_module,
        "wm": wm,
        "wm_config": wm_config,
        "system_prompt": system_prompt,
        "human_msg": human_msg,
        "conversation_flow": conversation_flow,
        "model_name": model_name,
        "temperature": temperature,
        "iteration": iteration,
        "wm_steps": wm_steps,
    }

    if strategy in ("beam_plan", "hier_latent_cem"):
        return _beam_or_hier_step(strategy=strategy, **kwargs)
    if strategy == "selection":
        return _selection_step(**kwargs)
    if strategy in {"prompt_injection", "itp_i"}:
        return _prompt_injection_step(state_history=state_history, **kwargs)
    if strategy in {"revision", "reference"}:
        pending = pending_reference[0] if strategy == "reference" else ""
        guided_human = f"{human_msg}\n\n{pending}" if pending else human_msg
        response = agent_module.call_llm(
            model_name, system_prompt, guided_human, temperature
        )
        action, action_input = agent_module.parse_action(response)
        seed_calls = [
            {"name": call["name"], "arguments": call["args"]}
            for call in _tool_calls_from_action(agent_module, action, action_input)
        ]
        step = {
            "iteration": iteration,
            "strategy": strategy,
            "backend": wm_config.backend,
            "seed_calls": seed_calls,
        }
        try:
            result = wm.action_feedback(
                conversation_flow,
                seed_calls=seed_calls,
                user_query=next(
                    (
                        event.get("content", "")
                        for event in conversation_flow
                        if event.get("type") == "user_message"
                    ),
                    "",
                ),
                mode=strategy,
            )
            step["detail"] = result.detail or {}
        except Exception as exc:
            logger.warning("wm_react: %s feedback failed (%s); baseline action", strategy, exc)
            step.update({"error": str(exc), "fallback": True})
            pending_reference[0] = ""
            wm_steps.append(step)
            return response
        wm_steps.append(step)
        if strategy == "reference":
            pending_reference[0] = result.text or ""
            return response
        if result.text:
            return agent_module.call_llm(
                model_name,
                system_prompt,
                f"{human_msg}\n\n{result.text}",
                temperature,
            )
        return response
    return agent_module.call_llm(model_name, system_prompt, human_msg, temperature)


def _run_agent_with_wm(
    *,
    agent_module: Any,
    wm: Any,
    wm_config: Any,
    model_name: str,
    tools: list[Any],
    task: str,
    datetime_prefix: str,
    act_without_confirmation: bool,
    max_iterations: int = 20,
    max_execution_time: float = 600.0,
    temperature: float = 0,
) -> tuple[Any, list[dict[str, Any]]]:
    """WM-guided reimplementation of ``src.evals.agent.run_agent``'s ReAct loop.

    Mirrors the upstream control flow (message construction, action parsing,
    tool dispatch, iteration/time limits, ``Final Answer`` termination)
    verbatim; the only difference is that each turn's next-action LLM call
    goes through :func:`_wm_step` instead of a bare ``call_llm``. Returns
    ``(AgentResult, wm_steps)``.
    """
    system_prompt = agent_module.build_system_prompt(tools, datetime_prefix, act_without_confirmation)
    tool_map = {t.name: t for t in tools}
    scratchpad = ""
    steps: list[tuple[str, dict[str, str]]] = []
    trace: list[Any] = []
    wm_steps: list[dict[str, Any]] = []
    pending_reference = [""]
    state_history: list[str] = []
    conversation_flow: list[dict[str, Any]] = [
        {"type": "tools", "tools": [_tool_dict(agent_module, t) for t in tools]},
        {"type": "system_message", "content": system_prompt},
        {"type": "user_message", "content": task},
    ]
    if hasattr(wm, "reset_episode"):
        wm.reset_episode()  # clear beam_plan/hier_latent_cem MPC state between tasks

    start_time = time.time()
    for iteration in range(max_iterations):
        if time.time() - start_time > max_execution_time:
            return (
                agent_module.AgentResult(
                    output=agent_module.AGENT_STOPPED_MESSAGE, intermediate_steps=steps, trace=trace
                ),
                wm_steps,
            )

        human_msg = task + "\n\nThought:" + scratchpad
        response_text = _wm_step(
            agent_module=agent_module,
            wm=wm,
            wm_config=wm_config,
            system_prompt=system_prompt,
            human_msg=human_msg,
            conversation_flow=conversation_flow,
            model_name=model_name,
            temperature=temperature,
            iteration=iteration,
            wm_steps=wm_steps,
            pending_reference=pending_reference,
            state_history=state_history,
        )
        action, action_input = agent_module.parse_action(response_text)
        conversation_flow.append(_ai_event(agent_module, response_text, action, action_input))

        if action == agent_module.FINAL_ANSWER:
            output = action_input if isinstance(action_input, str) else json.dumps(action_input)
            trace.append(
                agent_module.TraceStep(
                    llm_input=human_msg,
                    llm_output=response_text,
                    action=action,
                    action_input=action_input,
                    observation="",
                )
            )
            return (
                agent_module.AgentResult(output=output, intermediate_steps=steps, trace=trace),
                wm_steps,
            )

        if action == agent_module.PARSE_ERROR:
            observation = action_input if isinstance(action_input, str) else str(action_input)
            trace.append(
                agent_module.TraceStep(
                    llm_input=human_msg,
                    llm_output=response_text,
                    action=action,
                    action_input=action_input,
                    observation=observation,
                )
            )
            conversation_flow.append(
                {"type": "tool_result", "tool_name": "parse_error", "result": observation}
            )
            scratchpad += response_text + f"\nObservation: {observation}\nThought:"
            continue

        if action not in tool_map:
            observation = f"Tool '{action}' not found. Available tools: {', '.join(tool_map.keys())}"
        else:
            tool = tool_map[action]
            if isinstance(action_input, dict):
                str_input = {k: str(v) for k, v in action_input.items()}
                observation = str(tool(**str_input))
            else:
                observation = str(tool(str(action_input)))
        conversation_flow.append({"type": "tool_result", "tool_name": action, "result": observation})

        trace.append(
            agent_module.TraceStep(
                llm_input=human_msg,
                llm_output=response_text,
                action=action,
                action_input=action_input,
                observation=observation,
            )
        )
        steps.append((action, action_input if isinstance(action_input, dict) else {"input": str(action_input)}))
        scratchpad += response_text + f"\nObservation: {observation}\nThought:"

    return (
        agent_module.AgentResult(
            output=agent_module.AGENT_STOPPED_MESSAGE, intermediate_steps=steps, trace=trace
        ),
        wm_steps,
    )


def run_single_task_with_wm(
    index: int,
    task: str,
    model_name: str,
    tools: list[Any],
    datetime_prefix: str,
    act_without_confirmation: bool,
    structured_outputs: bool,
    inference_module: Any,
    agent_module: Any,
) -> dict[str, Any]:
    """Drop-in, WM-aware replacement for ``inference_module._run_single_task``.

    Same signature order as the upstream private function (plus the two
    already-resolved modules at the end) and the same return shape
    (``task``/``function_calls``/``full_response``/``error``/``trace``/
    ``_index``), plus a ``wm_steps`` telemetry field. Delegates straight to the
    upstream function when WM is disabled/unavailable or ``structured_outputs``
    is requested (WM guidance only wraps the ReAct text-parsing loop).
    """
    wm, wm_config = _get_world_model(agent_module)
    wm_enabled = wm is not None and getattr(wm_config, "strategy", "none") != "none"
    if not wm_enabled or structured_outputs:
        result = inference_module._run_single_task(
            index, task, model_name, tools, datetime_prefix, act_without_confirmation, structured_outputs
        )
        result.setdefault("wm_steps", [])
        return result

    token = _active_model_name.set(model_name)
    task_start = time.time()
    error = ""
    function_calls: list[str] = []
    response: Any = ""
    all_traces: list[dict[str, Any]] = []
    wm_steps: list[dict[str, Any]] = []
    try:
        result, wm_steps = _run_agent_with_wm(
            agent_module=agent_module,
            wm=wm,
            wm_config=wm_config,
            model_name=model_name,
            tools=tools,
            task=task,
            datetime_prefix=datetime_prefix,
            act_without_confirmation=act_without_confirmation,
        )
        response = result
        all_traces = [dataclasses.asdict(s) for s in result.trace]
        for tool_name, tool_input in result.intermediate_steps:
            function_calls.append(
                inference_module.convert_intermediate_step_to_function_call(tool_name, tool_input)
            )
        if result.output == agent_module.AGENT_STOPPED_MESSAGE:
            error = result.output
    except Exception as exc:
        if inference_module._is_context_window_error(exc):
            logger.warning("wm_react: context window exceeded with task: %s", task)
            error = "Context window exceeded"
        else:
            logger.exception("wm_react: unexpected error with task: %s", task)
            error = str(exc)
    finally:
        _active_model_name.reset(token)
        inference_module.reset_state()

    elapsed = time.time() - task_start
    logger.info(
        "wm_react task done in %.1fs | function_calls=%s | error=%s", elapsed, function_calls, error or "none"
    )

    return {
        "task": task,
        "function_calls": function_calls,
        "full_response": str(response),
        "error": error,
        "trace": all_traces,
        "wm_steps": wm_steps,
        "_index": index,
    }


__all__ = ["run_single_task_with_wm"]
