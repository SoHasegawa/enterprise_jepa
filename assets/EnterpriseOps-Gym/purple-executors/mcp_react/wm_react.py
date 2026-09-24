"""World-Model–guided ReAct orchestrator (ejepa_wm-backed).

Drop-in alternative to the upstream ``react`` orchestrator. The World Model is provided by the
shared, pluggable :mod:`ejepa_wm` package (``src/ejepa_wm``); this module only wires it into the
ReAct loop. The strategy and backend are read from the environment by
``ejepa_wm.wm_config_from_env`` (the ``ejepa`` CLI exports these from ``--wm-strategy/--wm-backend/
--wm-model/--wm-n``):

* ``selection`` (a.k.a. legacy ``best_of_n``) — *action selection*. Each step samples ``WM_N``
  candidate actions and the WM picks one (``WorldModel.select``). With ``WM_N <= 1`` or the
  ``noop`` backend it degrades to plain ReAct (the no-WM baseline).
* ``prompt_injection`` — *policy guidance*. Before each step the WM produces a guidance block
  (``WorldModel.advise``) injected transiently into the prompt (not persisted to the transcript).
* ``itp_i`` — *Imagine-Then-Plan implicit feedback baseline*. The unchanged policy selects an
  adaptive K, a generative WM predicts one K-step trajectory, and the policy reflects on that
  hypothetical trajectory before issuing its real tool-bound action. No policy training is used.
* ``beam_plan`` — *JEPA MPC lookahead* (requires ``ewm_imagined`` + a JEPA canonical-event
  checkpoint). Every ``WM_BEAM_MPC_EXECUTE_STEPS`` steps the WM proposes ``m`` candidate actions
  per horizon step, scores the ``m*n`` imagined outcomes with the canonical-event heads
  (LLM-free) and hands the agent a plan to follow; the step-0 action can override the agent's
  baseline when it beats it by more than ``WM_BEAM_PLAN_SCORE_MARGIN``. With any other backend
  it degrades to plain ReAct. See :meth:`ejepa_wm.backends.ewm_imagined.EwmImaginedWorldModel.beam_plan_step`.
* ``hier_latent_cem`` — *JEPA hierarchical latent-action CEM MPC* (same requirements as
  ``beam_plan``). ONE LLM call proposes a handful of diverse anchor actions; the world model then
  samples thousands of continuous latent-action trajectories around them and CEM-refines a
  per-family Gaussian proposal toward the highest-scoring region — LLM-free, decode-free. Same
  advisory-by-default / confidence-gated-injection / cooldown semantics as ``beam_plan`` (shared
  knobs: ``WM_BEAM_MPC_EXECUTE_STEPS``, ``WM_BEAM_PLAN_HARD_OVERRIDE``). See
  :meth:`ejepa_wm.backends.ewm_imagined.EwmImaginedWorldModel.hier_latent_cem_step`.
* ``none`` — no WM (baseline arm of the with/without-WM evaluation).

Backends (``WM_BACKEND``): ``served`` (served OpenAI-compatible model as LLM-as-WM),
``llm`` (generic, uses the policy LLM), ``ewm_predict`` / ``ewm_imagined`` (the Enterprise
World Model — single-step feasibility prediction and agent+EWM imagined-trajectory
planning, incl. WM_STATE / K_CONTROLLER / ACTION_OPTIMIZER), ``noop``. The policy LLM is wired in as
``chat_fn`` so every backend — including the EWM modes — works through the one uniform
``ejepa_wm`` contract; this orchestrator adds no per-backend code.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from orchestrators.react import ReactOrchestrator

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
        self._vllm_sampler: Any = None

    @staticmethod
    def _convert_messages(messages: List[Dict[str, str]]) -> List[Any]:
        converted: List[Any] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
        return converted

    def _model(self) -> Any:
        model = getattr(self._llm_client, "llm", None)
        if model is None and hasattr(self._llm_client, "_initialize_llm"):
            self._llm_client._initialize_llm()
            model = getattr(self._llm_client, "llm", None)
        if model is None:
            raise RuntimeError("llm_client has no initialized `.llm` chat model")
        return model

    def __call__(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        response_format: Dict[str, Any] | None = None,
    ) -> str:
        model = self._model()
        model_kwargs: Dict[str, Any] = {}
        if temperature is not None:
            model_kwargs["temperature"] = temperature
        if response_format is not None:
            model_kwargs["response_format"] = response_format
        if model_kwargs:
            model = model.bind(**model_kwargs)
        response = model.invoke(self._convert_messages(messages))
        return str(getattr(response, "content", "") or "")

    def _get_vllm_sampler(self) -> Any:
        if self._vllm_sampler is not None:
            return self._vllm_sampler
        provider = str(getattr(self._llm_client, "provider", "") or "").lower()
        base_url = str(
            getattr(self._llm_client, "custom_api_endpoint", "") or ""
        ).strip()
        model = str(getattr(self._llm_client, "model", "") or "").strip()
        if provider != "vllm" or not base_url or not model:
            raise NotImplementedError

        from ejepa_wm.backends._ewm_runtime import EwmGenerator

        max_tokens = int(getattr(self._llm_client, "max_tokens", 512) or 512)
        self._vllm_sampler = EwmGenerator(
            model,
            base_url,
            str(getattr(self._llm_client, "api_key", "") or "not-needed"),
            max_new_tokens=min(max_tokens, 512),
        )
        return self._vllm_sampler

    def generate_samples(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        num_samples: int = 1,
        response_format: Dict[str, Any] | None = None,
    ) -> List[str]:
        """Use the policy vLLM endpoint directly so SSoT needs one shared ``n=k`` request."""
        return self._get_vllm_sampler().generate_samples(
            messages,
            temperature=temperature,
            num_samples=num_samples,
            response_format=response_format,
        )


def _make_chat_fn(llm_client: Any) -> _PolicyChatFn:
    """Return the uniform policy-LLM bridge consumed by ``ejepa_wm`` backends."""
    return _PolicyChatFn(llm_client)


def _ai_event(response: Any) -> Dict[str, Any]:
    """A ``conversation_flow`` ai_message event for one candidate (WM input)."""
    return {
        "type": "ai_message",
        "content": response.content,
        "tool_calls": [
            {"name": tc["name"], "args": tc["args"]} for tc in (response.tool_calls or [])
        ],
    }


class WmReactOrchestrator(ReactOrchestrator):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Tolerate (and ignore) extra orchestrator_kwargs the executor may pass.
        self._task_domain = str(kwargs.pop("domain", "") or "").strip()
        super().__init__(*args, **kwargs)

        _ensure_ejepa_wm_on_path()
        # These executors build one world model per TASK so that concurrent tasks
        # (``config.max_parallel > 1``) cannot share episode state. Share the model
        # weights process-wide so that costs one checkpoint load, not one per task.
        os.environ.setdefault("WM_SHARE_MODEL_WEIGHTS", "1")
        from ejepa_wm import build_world_model, wm_config_from_env

        self.wm_config = wm_config_from_env()
        self.strategy = self.wm_config.strategy
        self.wm_n = self.wm_config.n
        try:
            # Uniform, executor-agnostic build: pass the policy LLM as chat_fn so EVERY
            # backend works through the same contract (llm / ewm_imagined consult it;
            # served / ewm_predict / noop ignore it). All WM modes —
            # including the EWM imagined-trajectory and predict backends — are handled
            # inside ejepa_wm via the `selection` / `prompt_injection` strategies; this
            # orchestrator stays a thin wiring layer.
            self.wm = build_world_model(
                self.wm_config, chat_fn=_make_chat_fn(self.llm_client)
            )
        except ValueError as exc:
            # e.g. backend='llm' without a chat_fn — degrade to plain ReAct.
            logger.warning("wm_react: WM build failed (%s); running plain ReAct", exc)
            from ejepa_wm import WMConfig, build_world_model as _b

            self.wm_config = WMConfig(strategy="none")
            self.strategy = "none"
            self.wm = _b(self.wm_config)

        self._wm_enabled = self.strategy != "none" and getattr(self.wm, "name", "") != "noop"
        if self.strategy == "selection":
            self._wm_enabled = self._wm_enabled and self.wm_n > 1

        self._wm_steps: List[Dict[str, Any]] = []
        self._state_history: List[str] = []
        self._pending_reference = ""

        temp = getattr(self.llm_client, "temperature", None)
        if self.strategy == "selection" and self._wm_enabled and (temp is None or float(temp) == 0.0):
            logger.warning(
                "wm_react: WM_N=%d but sampling temperature is %s — candidates will be "
                "identical; set ENTERPRISEOPS_LLM_TEMPERATURE>0 for diversity.",
                self.wm_n, temp,
            )
        logger.info(
            "wm_react: strategy=%s backend=%s enabled=%s n=%d",
            self.strategy, self.wm_config.backend, self._wm_enabled, self.wm_n,
        )

    # -- per-step response strategies --------------------------------------
    async def _select_response(
        self, messages: List[Any], conversation_flow: List[Dict[str, Any]], iteration: int
    ) -> Any:
        """selection: sample WM_N candidates, let the WM pick one, return that AIMessage."""
        if not self._wm_enabled:
            return await self.llm_client.invoke_with_tools(messages, self.available_tools)

        sampled = await asyncio.gather(
            *[self.llm_client.invoke_with_tools(messages, self.available_tools)
              for _ in range(self.wm_n)],
            return_exceptions=True,
        )
        candidates = [c for c in sampled if not isinstance(c, Exception)]
        if not candidates:
            raise next(c for c in reversed(sampled) if isinstance(c, Exception))
        if len(candidates) == 1:
            return candidates[0]

        events = [_ai_event(c) for c in candidates]
        result = await asyncio.get_event_loop().run_in_executor(
            None, self.wm.select, conversation_flow, events
        )
        idx = result.index if 0 <= result.index < len(candidates) else 0
        self._wm_steps.append({"iteration": iteration, "n_candidates": len(candidates),
                               "chosen_idx": idx, **result.detail})
        logger.info("wm_react[it=%d]: chose candidate %d/%d", iteration, idx, len(candidates))
        return candidates[idx]

    async def _inject_response(
        self, messages: List[Any], conversation_flow: List[Dict[str, Any]], iteration: int
    ) -> Any:
        """prompt_injection: ask the WM for guidance, inject it transiently, sample once."""
        if not self._wm_enabled:
            return await self.llm_client.invoke_with_tools(messages, self.available_tools)

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.wm.advise(conversation_flow, history=self._state_history)
        )
        state = (result.detail or {}).get("current_state")
        if state is not None:
            self._state_history.append(str(state))
        self._wm_steps.append({
            "iteration": iteration,
            "injected": bool(result.text),
            **({"itp_i_policy_reflect_calls": 1} if self.strategy == "itp_i" else {}),
            **(result.detail or {}),
        })
        if not result.text:
            return await self.llm_client.invoke_with_tools(messages, self.available_tools)
        # Feedback is transient: passed to the policy this step but NOT appended to
        # `messages`/`conversation_flow`, so the WM always re-scores the clean flow.
        guided = list(messages) + [HumanMessage(content=result.text)]
        return await self.llm_client.invoke_with_tools(guided, self.available_tools)

    async def _action_feedback_response(
        self, messages: List[Any], conversation_flow: List[Dict[str, Any]], iteration: int
    ) -> Any:
        """Run the revision or one-step-delayed reference action-feedback harness."""
        mode = self.strategy
        loop = asyncio.get_event_loop()
        if mode == "reference":
            guided = (
                list(messages) + [HumanMessage(content=self._pending_reference)]
                if self._pending_reference
                else messages
            )
            response = await self.llm_client.invoke_with_tools(guided, self.available_tools)
        else:
            response = await self.llm_client.invoke_with_tools(messages, self.available_tools)
        seed_calls = [
            {"name": call["name"], "arguments": call["args"]}
            for call in (response.tool_calls or [])
        ]
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.wm.action_feedback(
                    conversation_flow,
                    seed_calls=seed_calls,
                    user_query=self.config.user_prompt,
                    mode=mode,
                ),
            )
            detail = result.detail or {}
            self._wm_steps.append(
                {
                    "iteration": iteration,
                    "strategy": mode,
                    "injected": bool(result.text),
                    "detail": detail,
                }
            )
        except Exception as exc:
            logger.warning("wm_react[%s]: action feedback failed (%s); baseline action", mode, exc)
            self._wm_steps.append(
                {"iteration": iteration, "strategy": mode, "error": str(exc), "fallback": True}
            )
            if mode == "reference":
                self._pending_reference = ""
            return response
        if mode == "reference":
            self._pending_reference = result.text or ""
            return response
        if not result.text:
            return response
        # Do not append the unexecuted assistant tool-call message: OpenAI chat protocols
        # require a tool result after such a message. The feedback itself serializes the seed.
        revised_messages = list(messages) + [HumanMessage(content=result.text)]
        return await self.llm_client.invoke_with_tools(revised_messages, self.available_tools)

    def _ai_message_with_calls(self, response: Any, calls: List[Dict[str, Any]]) -> Any:
        """Rebuild an AIMessage keeping its content but replacing its tool calls (used when
        the beam overrides the agent's baseline action)."""
        tool_calls = [
            {"name": c.get("name", ""), "args": c.get("arguments", {}) or {}, "id": f"beam_{i}", "type": "tool_call"}
            for i, c in enumerate(calls)
        ]
        return AIMessage(content=getattr(response, "content", "") or "", tool_calls=tool_calls)

    async def _beam_plan_response(
        self, messages: List[Any], conversation_flow: List[Dict[str, Any]], iteration: int
    ) -> Any:
        """beam_plan: show the cached imagined plan to the agent, sample the baseline action,
        then let the JEPA WM re-plan / override per the MPC cadence."""
        if not self._wm_enabled or not hasattr(self.wm, "beam_plan_step"):
            return await self.llm_client.invoke_with_tools(messages, self.available_tools)
        loop = asyncio.get_event_loop()
        injection = await loop.run_in_executor(None, self.wm.beam_injection_text)
        guided = list(messages) + [HumanMessage(content=injection)] if injection else messages
        response = await self.llm_client.invoke_with_tools(guided, self.available_tools)
        seed_calls = [{"name": tc["name"], "arguments": tc["args"]} for tc in (response.tool_calls or [])]
        result = await loop.run_in_executor(
            None,
            lambda: self.wm.beam_plan_step(
                conversation_flow, seed_calls=seed_calls, user_query=self.config.user_prompt
            ),
        )
        detail = result.detail or {}
        self._wm_steps.append({"iteration": iteration, **detail})
        if detail.get("override_applied") and detail.get("calls"):
            logger.info("wm_react[it=%d]: beam_plan overrode baseline (%s)", iteration, detail.get("override_reason"))
            return self._ai_message_with_calls(response, detail["calls"])
        return response

    async def _hier_cem_response(
        self, messages: List[Any], conversation_flow: List[Dict[str, Any]], iteration: int
    ) -> Any:
        """hier_latent_cem: same MPC shape as ``_beam_plan_response``, but the imagined
        trajectory comes from the hierarchical latent-action CEM instead of a discrete
        LLM-proposed candidate pool."""
        if not self._wm_enabled or not hasattr(self.wm, "hier_latent_cem_step"):
            return await self.llm_client.invoke_with_tools(messages, self.available_tools)
        loop = asyncio.get_event_loop()
        injection = await loop.run_in_executor(None, self.wm.hier_injection_text)
        guided = list(messages) + [HumanMessage(content=injection)] if injection else messages
        response = await self.llm_client.invoke_with_tools(guided, self.available_tools)
        seed_calls = [{"name": tc["name"], "arguments": tc["args"]} for tc in (response.tool_calls or [])]
        result = await loop.run_in_executor(
            None,
            lambda: self.wm.hier_latent_cem_step(
                conversation_flow, seed_calls=seed_calls, user_query=self.config.user_prompt
            ),
        )
        detail = result.detail or {}
        self._wm_steps.append({"iteration": iteration, **detail})
        if detail.get("override_applied") and detail.get("calls"):
            logger.info("wm_react[it=%d]: hier_latent_cem overrode baseline (%s)", iteration, detail.get("reason"))
            return self._ai_message_with_calls(response, detail["calls"])
        return response

    async def _next_response(
        self, messages: List[Any], conversation_flow: List[Dict[str, Any]], iteration: int
    ) -> Any:
        if self.strategy == "beam_plan":
            return await self._beam_plan_response(messages, conversation_flow, iteration)
        if self.strategy == "hier_latent_cem":
            return await self._hier_cem_response(messages, conversation_flow, iteration)
        if self._wm_enabled and self.strategy in {"revision", "reference"}:
            return await self._action_feedback_response(messages, conversation_flow, iteration)
        if self._wm_enabled and self.strategy in {"prompt_injection", "itp_i"}:
            return await self._inject_response(messages, conversation_flow, iteration)
        return await self._select_response(messages, conversation_flow, iteration)

    # -- main loop (mirrors ReactOrchestrator.execute, with strategy dispatch) --
    async def execute(self) -> Dict[str, Any]:
        if hasattr(self.wm, "reset_episode"):
            self.wm.reset_episode()  # clear beam_plan MPC state between tasks
        self._pending_reference = ""
        messages = [
            SystemMessage(content=self.config.system_prompt),
            HumanMessage(content=self.config.user_prompt),
        ]
        conversation_flow = [
            {"type": "task_metadata", "domain": self._task_domain},
            # Carry the tool catalog in the flow (not a build_world_model arg) so WM backends
            # that need it (ewm_imagined's text-ReAct rollout) can build a training-matched
            # system prompt, while build_world_model(config, chat_fn) stays uniform.
            {"type": "tools", "tools": self.available_tools},
            {"type": "system_message", "content": self.config.system_prompt},
            {"type": "user_message", "content": self.config.user_prompt},
        ]
        tools_used: List[str] = []
        tool_results: List[Dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            logger.info(f"\n--- Iteration {iteration + 1} (wm_react strategy={self.strategy}) ---")

            response = await self._next_response(messages, conversation_flow, iteration)
            messages.append(response)

            usage_metadata = response.usage_metadata if hasattr(response, "usage_metadata") else {}
            response_metadata = response.response_metadata if hasattr(response, "response_metadata") else {}
            conversation_flow.append(
                {
                    "type": "ai_message",
                    "content": response.content,
                    "usage_metadata": usage_metadata,
                    "response_metadata": response_metadata,
                    "tool_calls": [
                        {"name": tc["name"], "args": tc["args"]} for tc in (response.tool_calls or [])
                    ],
                }
            )

            if not response.tool_calls or len(response.tool_calls) == 0:
                logger.info("No tool calls requested. Task complete.")
                break

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                exec_result = await self._execute_tool_call(tool_name, tool_args)
                tool_result = exec_result["result"]
                target_gym = exec_result["gym_server"]

                if tool_name not in tools_used:
                    tools_used.append(tool_name)
                tool_results.append(
                    {
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": tool_result,
                        "gym_server": target_gym,
                    }
                )
                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_result.get("result", {})),
                        tool_call_id=tool_call.get("id", ""),
                    )
                )
                conversation_flow.append(
                    {
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "result": tool_result,
                        "gym_server": target_gym,
                    }
                )

        return {
            "final_response": messages[-1].content if messages else "",
            "conversation_flow": conversation_flow,
            "tools_used": tools_used,
            "tool_results": tool_results,
            "messages": messages,
        }

    def get_result_metadata(self) -> Dict[str, Any]:
        return {
            "wm_react": {
                "enabled": self._wm_enabled,
                "strategy": self.strategy,
                "backend": self.wm_config.backend,
                "n": self.wm_n,
                "state_history": self._state_history if self.strategy == "prompt_injection" else None,
                "agent_call_count": (
                    getattr(self.wm, "agent_call_count", None)
                    if self.strategy in ("itp_i", "beam_plan", "hier_latent_cem") else None
                ),
                "steps": self._wm_steps,
            }
        }
