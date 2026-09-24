"""World-Model-guided ReAct loop for AutomationBench (``ejepa_wm``-backed).

Structured after EnterpriseOps-Gym's ``purple-executors/mcp_react/wm_react.py``:
one orchestrator owns the ReAct loop and dispatches per strategy, the policy LLM is
handed to ``ejepa_wm`` as ``chat_fn`` so every world model works through the same
contract, and this module adds no per-backend code. AutomationBench has no upstream
ReAct orchestrator to subclass (upstream drives a ``verifiers`` StatefulToolEnv), so
the loop lives here, but its shape, telemetry and ``conversation_flow`` events match
EnterpriseOps-Gym so the shared harnesses and metric extraction work unchanged.

Strategies (``--wm-strategy`` / ``WM_STRATEGY``):

* ``selection`` -- sample ``WM_N`` candidate actions, the WM picks one.
* ``prompt_injection`` (alias ``imagined``) -- WM guidance injected transiently.
* ``itp_i`` -- Imagine-Then-Plan: the WM imagines a K-step trajectory, the policy
  reflects on it before issuing its real tool call.
* ``revision`` / ``reference`` -- predicted outcome of the proposed action, applied
  immediately or one step late.
* ``beam_plan`` -- JEPA MPC lookahead (``--wm-beam-plan-trigger interval|critic``).
* ``hier_latent_cem`` -- hierarchical latent-action CEM MPC.
* unset / ``none`` -- plain ReAct (the no-WM baseline arm).

World models are selected by the standard flags, not by anything here:

* Enterprise-JEPA: ``--wm-ewm-jepa-checkpoint <dir> --wm-jepa-observation-backend
  canonical_event``
* LLM-WM (state output): ``--wm-llm-ewm-mode llm_canonical_trained
  --wm-ewm-llm-canonical-event-checkpoint <dir>``
* LLM-WM (tool output) / agent world model: ``--wm-llm-ewm-mode
  llm_tool_output_judge`` (+ ``WM_QWEN_AGENTWORLD=1`` and ``--wm-ewm-model`` for a
  served agent-world model).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _ensure_ejepa_wm_on_path() -> None:
    """Put the repo's ``src`` dir on sys.path so ``ejepa_wm`` is importable."""
    src = Path(__file__).resolve().parents[4] / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


class _PolicyChatFn:
    """Expose the policy client to ``ejepa_wm``, including native vLLM ``n=k`` sampling."""

    def __init__(self, llm_client: Any) -> None:
        self._llm_client = llm_client

    def __call__(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        return self._llm_client.complete(
            messages, temperature=temperature, response_format=response_format
        )

    def generate_samples(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        num_samples: int = 1,
        response_format: dict[str, Any] | None = None,
    ) -> list[str]:
        """``n=k`` in ONE request so open-loop beam sampling shares a single prefill."""
        return self._llm_client.complete_samples(
            messages,
            temperature=temperature,
            num_samples=num_samples,
            response_format=response_format,
        )


def _make_chat_fn(llm_client: Any) -> _PolicyChatFn:
    """Return the uniform policy-LLM bridge consumed by ``ejepa_wm`` backends."""
    return _PolicyChatFn(llm_client)


def _calls_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise an assistant message's tool calls to ``{name, arguments}``."""
    normalized: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        raw_args = function.get("arguments")
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                arguments = {}
        else:
            arguments = dict(raw_args or {})
        if name:
            normalized.append({"name": name, "arguments": arguments})
    return normalized


def _ai_event(message: dict[str, Any]) -> dict[str, Any]:
    """A ``conversation_flow`` ai_message event for one candidate response."""
    return {
        "type": "ai_message",
        "content": str(message.get("content") or ""),
        "tool_calls": [
            {"name": call["name"], "args": call["arguments"]}
            for call in _calls_from_message(message)
        ],
    }


def _message_with_calls(message: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild an assistant message keeping its content but replacing its tool calls."""
    return {
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": [
            {
                "id": f"wm_{index}",
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                },
            }
            for index, call in enumerate(calls)
        ],
    }


SYSTEM_PROMPT = (
    "You are a workflow automation agent operating over simulated SaaS applications. "
    "Execute the requested work using the available tools. Do not ask clarifying "
    "questions -- use the information provided and make reasonable assumptions. "
    "Search before you write so you act on real record ids, and stop calling tools "
    "once the requested state changes are committed."
)


class AutomationBenchWmReactAgent:
    """ReAct loop over AutomationBench tools with the shared ``ejepa_wm`` hook."""

    def __init__(
        self,
        *,
        llm_client: Any,
        binding: Any,
        user_prompt: str,
        task_name: str = "",
        domain: str = "automationbench",
        max_turns: int = 50,
    ) -> None:
        _ensure_ejepa_wm_on_path()
        # These executors build one world model per TASK so that concurrent tasks
        # (``config.max_parallel > 1``) cannot share episode state. Share the model
        # weights process-wide so that costs one checkpoint load, not one per task.
        os.environ.setdefault("WM_SHARE_MODEL_WEIGHTS", "1")
        from ejepa_wm import build_world_model, wm_config_from_env

        self.llm_client = llm_client
        self.binding = binding
        self.user_prompt = user_prompt
        self.task_name = task_name
        self.domain = domain
        self.max_turns = max(1, int(max_turns))
        self.available_tools = list(binding.schemas)

        self.wm_config = wm_config_from_env()
        self.strategy = self.wm_config.strategy
        self.wm_n = self.wm_config.n
        try:
            self.wm = build_world_model(self.wm_config, chat_fn=_make_chat_fn(llm_client))
        except ValueError as exc:
            logger.warning("wm_react: WM build failed (%s); running plain ReAct", exc)
            from ejepa_wm import WMConfig
            from ejepa_wm import build_world_model as _build

            self.wm_config = WMConfig(strategy="none")
            self.strategy = "none"
            self.wm = _build(self.wm_config)

        self._wm_enabled = self.strategy != "none" and getattr(self.wm, "name", "") != "noop"
        if self.strategy == "selection":
            self._wm_enabled = self._wm_enabled and self.wm_n > 1

        self._wm_steps: list[dict[str, Any]] = []
        self._state_history: list[str] = []
        self._pending_reference = ""
        logger.info(
            "wm_react: strategy=%s backend=%s enabled=%s n=%d tools=%d",
            self.strategy,
            self.wm_config.backend,
            self._wm_enabled,
            self.wm_n,
            len(self.available_tools),
        )

    # -- per-step response strategies --------------------------------------
    async def _plain_response(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return await self.llm_client.invoke_with_tools(messages, self.available_tools)

    async def _select_response(
        self,
        messages: list[dict[str, Any]],
        conversation_flow: list[dict[str, Any]],
        iteration: int,
    ) -> dict[str, Any]:
        """selection: sample WM_N candidates, let the WM pick one."""
        if not self._wm_enabled:
            return await self._plain_response(messages)
        sampled = await asyncio.gather(
            *[self._plain_response(messages) for _ in range(self.wm_n)],
            return_exceptions=True,
        )
        candidates = [item for item in sampled if not isinstance(item, Exception)]
        if not candidates:
            raise next(item for item in reversed(sampled) if isinstance(item, Exception))
        if len(candidates) == 1:
            return candidates[0]
        events = [_ai_event(candidate) for candidate in candidates]
        result = await asyncio.get_running_loop().run_in_executor(
            None, self.wm.select, conversation_flow, events
        )
        index = result.index if 0 <= result.index < len(candidates) else 0
        self._wm_steps.append(
            {
                "iteration": iteration,
                "strategy": self.strategy,
                "n_candidates": len(candidates),
                "chosen_idx": index,
                **(result.detail or {}),
            }
        )
        return candidates[index]

    async def _inject_response(
        self,
        messages: list[dict[str, Any]],
        conversation_flow: list[dict[str, Any]],
        iteration: int,
    ) -> dict[str, Any]:
        """prompt_injection / itp_i: WM guidance injected transiently for this step."""
        if not self._wm_enabled:
            return await self._plain_response(messages)
        result = await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.wm.advise(conversation_flow, history=self._state_history)
        )
        detail = result.detail or {}
        state = detail.get("current_state")
        if state is not None:
            self._state_history.append(str(state))
        self._wm_steps.append(
            {
                "iteration": iteration,
                "strategy": self.strategy,
                "injected": bool(result.text),
                **({"itp_i_policy_reflect_calls": 1} if self.strategy == "itp_i" else {}),
                **detail,
            }
        )
        if not result.text:
            return await self._plain_response(messages)
        # Transient: handed to the policy this step but never appended to the
        # transcript, so the WM always re-scores a clean flow.
        guided = [*messages, {"role": "user", "content": result.text}]
        return await self._plain_response(guided)

    async def _action_feedback_response(
        self,
        messages: list[dict[str, Any]],
        conversation_flow: list[dict[str, Any]],
        iteration: int,
    ) -> dict[str, Any]:
        """revision / reference: predicted outcome of the proposed action."""
        mode = self.strategy
        if mode == "reference" and self._pending_reference:
            response = await self._plain_response(
                [*messages, {"role": "user", "content": self._pending_reference}]
            )
        else:
            response = await self._plain_response(messages)
        seed_calls = _calls_from_message(response)
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.wm.action_feedback(
                    conversation_flow,
                    seed_calls=seed_calls,
                    user_query=self.user_prompt,
                    mode=mode,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - a WM failure must not kill the step
            logger.warning("wm_react[%s]: action feedback failed (%s); baseline action", mode, exc)
            self._wm_steps.append(
                {"iteration": iteration, "strategy": mode, "error": str(exc), "fallback": True}
            )
            if mode == "reference":
                self._pending_reference = ""
            return response
        detail = result.detail or {}
        self._wm_steps.append(
            {
                "iteration": iteration,
                "strategy": mode,
                "injected": bool(result.text),
                **detail,
            }
        )
        if mode == "reference":
            self._pending_reference = result.text or ""
            return response
        if not result.text:
            return response
        revised = [*messages, {"role": "user", "content": result.text}]
        return await self._plain_response(revised)

    async def _mpc_response(
        self,
        messages: list[dict[str, Any]],
        conversation_flow: list[dict[str, Any]],
        iteration: int,
    ) -> dict[str, Any]:
        """beam_plan / hier_latent_cem: show the cached plan, then one MPC cycle."""
        step_method = getattr(
            self.wm,
            "beam_plan_step" if self.strategy == "beam_plan" else "hier_latent_cem_step",
            None,
        )
        if not self._wm_enabled or step_method is None:
            return await self._plain_response(messages)
        loop = asyncio.get_running_loop()
        injection_method = getattr(
            self.wm,
            "beam_injection_text" if self.strategy == "beam_plan" else "hier_injection_text",
            None,
        )
        injection = ""
        if injection_method is not None:
            injection = str(await loop.run_in_executor(None, injection_method) or "")
        guided = [*messages, {"role": "user", "content": injection}] if injection else messages
        response = await self._plain_response(guided)
        seed_calls = _calls_from_message(response)
        result = await loop.run_in_executor(
            None,
            lambda: step_method(
                conversation_flow, seed_calls=seed_calls, user_query=self.user_prompt
            ),
        )
        detail = dict(result.detail or {})
        self._wm_steps.append({"iteration": iteration, "strategy": self.strategy, **detail})
        if detail.get("override_applied") and detail.get("calls"):
            logger.info(
                "wm_react[it=%d]: %s overrode baseline (%s)",
                iteration,
                self.strategy,
                detail.get("override_reason"),
            )
            return _message_with_calls(response, detail["calls"])
        return response

    async def _next_response(
        self,
        messages: list[dict[str, Any]],
        conversation_flow: list[dict[str, Any]],
        iteration: int,
    ) -> dict[str, Any]:
        if self.strategy in {"beam_plan", "hier_latent_cem"}:
            return await self._mpc_response(messages, conversation_flow, iteration)
        if self._wm_enabled and self.strategy in {"revision", "reference"}:
            return await self._action_feedback_response(messages, conversation_flow, iteration)
        if self._wm_enabled and self.strategy in {"prompt_injection", "itp_i"}:
            return await self._inject_response(messages, conversation_flow, iteration)
        return await self._select_response(messages, conversation_flow, iteration)

    # -- main loop (mirrors EnterpriseOps-Gym's WmReactOrchestrator.execute) -
    async def execute(self) -> dict[str, Any]:
        reset = getattr(self.wm, "reset_episode", None)
        if reset is not None:
            reset()  # clear per-episode MPC state between tasks

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.user_prompt},
        ]
        conversation_flow: list[dict[str, Any]] = [
            {"type": "task_metadata", "domain": self.domain, "task_name": self.task_name},
            # Carry the tool catalogue in the flow (not a build_world_model arg) so WM
            # backends that need it can build a training-matched prompt.
            {"type": "tools", "tools": self.available_tools},
            {"type": "system_message", "content": SYSTEM_PROMPT},
            {"type": "user_message", "content": self.user_prompt},
        ]
        tool_calls: list[dict[str, Any]] = []
        tools_used: list[str] = []
        model_calls = 0
        error: str | None = None
        final_response = ""

        try:
            for iteration in range(self.max_turns):
                logger.info(
                    "--- Iteration %d (wm_react strategy=%s) ---",
                    iteration + 1,
                    self.strategy,
                )
                response = await self._next_response(messages, conversation_flow, iteration)
                model_calls += 1
                messages.append(response)
                calls = _calls_from_message(response)
                conversation_flow.append(_ai_event(response))

                if not calls:
                    final_response = str(response.get("content") or "")
                    logger.info("No tool calls requested. Task complete.")
                    break

                for index, call in enumerate(calls):
                    outcome = self.binding.call(call["name"], call["arguments"])
                    record = {
                        "name": call["name"],
                        "arguments": call["arguments"],
                        **outcome,
                    }
                    tool_calls.append(record)
                    if call["name"] not in tools_used:
                        tools_used.append(call["name"])
                    observation = json.dumps(outcome, ensure_ascii=False, default=str)[:6000]
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": _tool_call_id(response, index),
                            "content": observation,
                        }
                    )
                    conversation_flow.append(
                        {
                            "type": "tool_result",
                            "tool_name": call["name"],
                            "result": outcome,
                        }
                    )
                    self._state_history.append(observation)
            else:
                error = f"max_turns ({self.max_turns}) reached"
                final_response = str(messages[-1].get("content") or "")
        except Exception as exc:  # noqa: BLE001 - reported to Green, never raised through A2A
            error = f"{type(exc).__name__}: {exc}"
            logger.error("wm_react loop failed: %s", exc)

        return {
            "final_state": self.binding.final_state(),
            "final_response": final_response,
            "tool_calls": tool_calls,
            "tools_used": tools_used,
            "num_tool_calls": len(tool_calls),
            "num_model_calls": model_calls,
            "steps": model_calls,
            "wm_strategy": self.strategy,
            "wm_backend": self.wm_config.backend,
            "wm_steps": self._wm_steps,
            "tool_source": self.binding.source,
            "error": error,
        }


def _tool_call_id(response: dict[str, Any], index: int) -> str:
    calls = response.get("tool_calls") or []
    if index < len(calls) and isinstance(calls[index], dict):
        return str(calls[index].get("id") or f"call_{index}")
    return f"call_{index}"


async def run_single_task_with_wm(
    *,
    payload: dict[str, Any],
    llm_client: Any,
    binding: Any,
) -> dict[str, Any]:
    """Run one AutomationBench task end to end and return the result payload."""
    agent = AutomationBenchWmReactAgent(
        llm_client=llm_client,
        binding=binding,
        user_prompt=str(payload.get("prompt") or ""),
        task_name=str(payload.get("task_name") or payload.get("task_id") or ""),
        domain=str(payload.get("domain") or "automationbench"),
        max_turns=int(payload.get("max_turns") or 50),
    )
    return await agent.execute()
