"""World-Model–guided ReAct agent for CRMArenaPro (``ejepa_wm``-backed).

This mirrors EnterpriseOps-Gym's ``mcp_react/wm_react.py`` orchestrator, but the gym
reaches tools over **MCP servers** while CRMArenaPro has no MCP layer — so this agent
reaches tools through CRMArenaPro's **native CRM-SQL path** (``CRMDatabase``) reused
verbatim from the ``baseline_crm_agent`` executor.

The world model itself is the shared, pluggable :mod:`ejepa_wm` package (``src/ejepa_wm``);
this module only *wires it into* the CRM ReAct loop via the generic ``conversation_flow``
event contract (``system_message`` / ``user_message`` / ``ai_message`` / ``tool_result``).
WM behavior is entirely env-driven (``WM_STRATEGY`` / ``WM_BACKEND`` / ``WM_N`` / the
``EWM_*`` / ``WM_EWM_MCP_URL`` knobs), exactly like the gym executor.

Seven strategies are supported (selected by ``WM_STRATEGY``):

* ``selection`` (pairs well with ``WM_BACKEND=ewm_predict``) — sample ``WM_N`` candidate
  actions from the policy LLM, let the WM rank them, and commit the best one;
* ``prompt_injection`` (pairs with ``WM_BACKEND=ewm_imagined`` or ``llm``) — ask the WM
  to ``advise`` and inject the returned guidance block before the next action;
* ``itp_i`` (pairs with ``WM_BACKEND=ewm_imagined``) — Imagine-Then-Plan inference:
  ask the WM to imagine a trajectory and inject that trajectory for policy reflection;
* ``revision`` / ``reference`` — predict feedback for the proposed action and either
  revise immediately or expose the prediction at the following decision step;
* ``beam_plan`` (auto-selects ``WM_BACKEND=ewm_imagined``; requires a JEPA canonical-event
  checkpoint) — JEPA MPC lookahead. Every ``WM_BEAM_MPC_EXECUTE_STEPS`` turns the WM
  proposes ``m`` candidate actions per horizon step, scores the imagined outcomes with the
  canonical-event heads (LLM-free) and hands the agent a plan to follow; the step-0 action
  can override the agent's baseline when it beats it by more than
  ``WM_BEAM_PLAN_SCORE_MARGIN``. With any non-JEPA backend it degrades to plain ReAct. See
  :meth:`ejepa_wm.backends.ewm_imagined.EwmImaginedWorldModel.beam_plan_step`.
* ``hier_latent_cem`` (auto-selects ``WM_BACKEND=ewm_imagined``; same JEPA canonical-event
  checkpoint requirement as ``beam_plan``) — a second JEPA-only MPC lookahead, same
  cadence/advisory/override contract as ``beam_plan``, but it plans via a hierarchical
  latent-action CEM (``WM_HIER_CEM_*`` knobs) instead of discrete LLM-proposed per-horizon-step
  candidates: ONE LLM call proposes a handful of diverse anchor actions, then the world model
  CEM-refines thousands of continuous latent-action trajectories around them (LLM-free,
  decode-free) and only decodes the converged best plan. With any non-JEPA backend it degrades
  to plain ReAct. See
  :meth:`ejepa_wm.backends.ewm_imagined.EwmImaginedWorldModel.hier_latent_cem_step`.

``WM_STRATEGY`` unset / ``none`` ⇒ this falls back to the plain ``baseline_crm_agent``
behavior (no WM), so the executor is always runnable without an EWM server.

NOTE on scope: the bundled EWM world model is fine-tuned on EnterpriseOps-Gym tool
outcomes. Pointing it at CRM-SQL actions is an intentional **out-of-domain / over-fit
probe** — it is expected to guide gym well and CRMArenaPro poorly, which is the point of
wiring it here.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from agent import (
    SYSTEM_PROMPT,
    CRMDatabase,
    _is_empty_answer,
    _uses_max_completion_tokens,
)

# Reuse the baseline CRM tool layer unchanged (CRMDatabase, SYSTEM_PROMPT, the ReAct
# action parser, the provider clients, artifact helpers). The baseline executor dir is
# put on sys.path by executor.py before this module is imported.
from agent import (  # type: ignore[import-not-found]
    Agent as BaselineAgent,
)

logger = logging.getLogger(__name__)


def _ensure_ejepa_wm_on_path() -> None:
    """Put the repo's ``src`` dir on ``sys.path`` so ``ejepa_wm`` is importable.

    Layout: ``benchmarks/assets/crmarenapro/purple-executors/mcp_react/wm_react.py``
    so parents[4] is the repo root and ``parents[4]/src`` holds ``ejepa_wm``.
    """
    src = Path(__file__).resolve().parents[4] / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


# The CRM "tools" advertised to ejepa_wm. Only ``ewm_imagined`` consumes this (to embed a
# tool catalog in its imagined-rollout system prompt); ``ewm_predict`` / ``llm`` ignore
# it. Shapes mirror gym tool dicts so the same backends accept them unchanged.
_CRM_TOOLS: list[dict[str, Any]] = [
    {
        "name": "execute",
        "description": "Run a read-only SQL query (SELECT/WITH) against the CRM database.",
        "inputSchema": {
            "type": "object",
            "properties": {"content": {"type": "string", "description": "SQL query"}},
            "required": ["content"],
        },
    },
    {
        "name": "describe",
        "description": "Return the columns and row count of a CRM table.",
        "inputSchema": {
            "type": "object",
            "properties": {"content": {"type": "string", "description": "table name"}},
            "required": ["content"],
        },
    },
    {
        "name": "respond",
        "description": "Submit the final concise answer to the task.",
        "inputSchema": {
            "type": "object",
            "properties": {"content": {"type": "string", "description": "final answer"}},
            "required": ["content"],
        },
    },
]


class WmReactAgent(BaselineAgent):
    """``baseline_crm_agent`` ReAct loop + a per-turn ``ejepa_wm`` world-model hook."""

    def __init__(self) -> None:
        super().__init__()
        _ensure_ejepa_wm_on_path()
        self.wm = None
        self._wm_steps: list[dict[str, Any]] = []
        self._pending_reference = ""
        self._wm_state_history: list[str] = []
        try:
            # These executors build one world model per TASK so that concurrent tasks
            # (``config.max_parallel > 1``) cannot share episode state. Share the model
            # weights process-wide so that costs one checkpoint load, not one per task.
            os.environ.setdefault("WM_SHARE_MODEL_WEIGHTS", "1")
            from ejepa_wm import build_world_model, wm_config_from_env

            self.wm_config = wm_config_from_env()
            if self.wm_config.strategy != "none":
                # Uniform contract: build_world_model(config, chat_fn). The CRM tool catalog
                # is carried in the conversation_flow as a `tools` event (see run()), so
                # ewm_imagined builds its rollout prompt without an executor-specific arg.
                self.wm = build_world_model(self.wm_config, chat_fn=self._wm_chat_fn)
        except Exception as exc:  # ejepa_wm missing / WM build failed → plain ReAct
            logger.warning("mcp_react: WM unavailable (%s); using plain ReAct", exc)
            self.wm = None
        strat = getattr(getattr(self, "wm_config", None), "strategy", "none")
        backend = getattr(getattr(self, "wm_config", None), "backend", "noop")
        logger.info(
            "crmarenapro mcp_react: wm=%s strategy=%s backend=%s n=%s",
            "on" if self.wm is not None else "off", strat, backend,
            getattr(getattr(self, "wm_config", None), "n", 1),
        )

    # ------------------------------------------------------------------ WM glue

    def _wm_enabled(self) -> bool:
        return self.wm is not None and getattr(self.wm_config, "strategy", "none") != "none"

    @staticmethod
    def _traj_log_chars() -> int:
        """Max chars of the imagined trajectory to log per step (0 = unlimited)."""
        try:
            return int(os.getenv("WM_TRAJECTORY_LOG_CHARS", "0"))
        except ValueError:
            return 0

    def _action_summary(self, content: str) -> str:
        """Short '<type>: <content>' summary of a candidate response, for wm_steps logging."""
        action = self._extract_action(content)
        if action.get("type"):
            return f"{action['type']}: {str(action.get('content', ''))[:160]}"
        return content[:160]

    def _trajectory_payload(self, task_id: str, category: str) -> dict[str, Any]:
        """Augment the baseline internal_trajectory with WM telemetry under ``info``.

        The crmarenapro green recorder copies ``payload.info`` verbatim, so nesting the WM
        strategy/backend and per-step ``wm_steps`` (incl. each step's imagined trajectory)
        here makes EWM's contribution observable in the captured trajectory.
        """
        payload = super()._trajectory_payload(task_id, category)
        payload["executor"] = "mcp_react"
        info = payload.setdefault("payload", {}).setdefault("info", {})
        info.update({
            "executor": "mcp_react",
            "wm_strategy": getattr(self.wm_config, "strategy", "none"),
            "wm_backend": getattr(self.wm_config, "backend", "noop"),
            "wm_enabled": self._wm_enabled(),
            "wm_agent_call_count": (
                getattr(self.wm, "agent_call_count", None)
                if getattr(self.wm_config, "strategy", "none")
                in ("itp_i", "beam_plan", "hier_latent_cem")
                else None
            ),
            "wm_steps": list(self._wm_steps),
        })
        return payload

    def _wm_chat_fn(self, messages: list[dict[str, str]]) -> str:
        """``ejepa_wm.ChatFn``: OpenAI-style messages -> reply text (synchronous)."""
        return self._sync_complete(messages, self.temperature)

    def _sync_complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        """One-shot completion via the same provider client the agent uses.

        Synchronous (the underlying SDK calls are blocking) and deliberately does NOT
        record to ``self.trajectory`` — it backs WM candidate sampling / WM-internal
        rollouts, which should not pollute the visible transcript.
        """
        try:
            if self.provider == "anthropic":
                system_chunks: list[str] = []
                conv: list[dict[str, str]] = []
                for msg in messages:
                    role = msg.get("role")
                    if role == "system":
                        system_chunks.append(msg.get("content", ""))
                    elif role in {"user", "assistant"}:
                        conv.append({"role": role, "content": msg.get("content", "")})
                if not conv:
                    conv = [{"role": "user", "content": "Continue."}]
                resp = self.anthropic_client.messages.create(
                    model=self.model,
                    system="\n\n".join(system_chunks) if system_chunks else None,
                    messages=conv,
                    temperature=temperature,
                    max_tokens=2048,
                )
                return "".join(
                    b.text for b in resp.content if getattr(b, "type", None) == "text"
                )
            args: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if _uses_max_completion_tokens(self.model, self.base_url):
                args["max_completion_tokens"] = 2048
            else:
                args["max_tokens"] = 2048
            resp = self.openai_client.chat.completions.create(**args)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("mcp_react: WM sync completion failed: %s", exc)
            return ""

    def _ai_event(self, content: str) -> dict[str, Any]:
        """Build an ``ai_message`` conversation_flow event from a raw LLM response."""
        action = self._extract_action(content)
        tool_calls: list[dict[str, Any]] = []
        if action.get("type") and action.get("content") is not None:
            tool_calls = [{"name": action["type"], "args": {"content": action["content"]}}]
        return {"type": "ai_message", "content": content, "tool_calls": tool_calls}

    def _response_with_calls(self, calls: list[dict[str, Any]], strategy: str = "beam_plan") -> str:
        """Rebuild a raw ReAct response *string* that ``_extract_action`` parses back into the
        WM-overridden call (this executor commits actions from response TEXT, not dict/AIMessage
        objects, so the beam/hier_cem override is rendered as the CRM
        ``<execute>/<describe>/<respond>`` tag protocol). Uses the first call — the CRM loop acts
        on one tool call per turn. Shared by both JEPA MPC strategies (``beam_plan`` /
        ``hier_latent_cem``); ``strategy`` only affects the logged thought text."""
        call = (calls or [{}])[0]
        name = str(call.get("name", "") or "").strip()
        content = str((call.get("arguments", {}) or {}).get("content", "") or "")
        return f"<thought>World-model {strategy} override.</thought>\n<{name}>{content}</{name}>"

    async def _wm_step(
        self, messages: list[dict[str, str]], conversation_flow: list[dict[str, Any]]
    ) -> str:
        """Produce the next agent response, consulting the WM per ``WM_STRATEGY``."""
        strategy = self.wm_config.strategy

        if strategy == "beam_plan":
            step = {
                "step": self.metrics["turns"], "strategy": "beam_plan",
                "backend": self.wm_config.backend,
            }
            # Guard: beam_plan needs the JEPA canonical-event backend; otherwise behave
            # exactly like the plain no-WM agent (one sample, no override).
            if not hasattr(self.wm, "beam_plan_step") or not self.wm.supports_beam_plan():
                step["fallback"] = True
                self._wm_steps.append(step)
                return await self._call_llm(messages)
            # (a) Show the cached imagined plan to the agent BEFORE it acts (transient only —
            # NOT appended to `messages`, so the WM always re-plans from the clean flow).
            injection = self.wm.beam_injection_text()
            guided = [*messages, {"role": "user", "content": injection}] if injection else messages
            # (b) Sample the baseline next action once.
            response = await self._call_llm(guided)
            # (c) seed_calls = the baseline response's tool call(s) (see _ai_event's shape).
            seed_calls = [
                {"name": tc["name"], "arguments": tc["args"]}
                for tc in self._ai_event(response)["tool_calls"]
            ]
            user_query = next(
                (e.get("content", "") for e in conversation_flow if e.get("type") == "user_message"), ""
            )
            # (d) One MPC decision (re-plan / follow / override). Never crashes the run.
            try:
                res = self.wm.beam_plan_step(
                    conversation_flow, seed_calls=seed_calls, user_query=user_query
                )
                detail = res.detail or {}
                step.update({"injected_chars": len(injection or ""), "detail": detail})
            except Exception as exc:
                logger.warning("mcp_react: beam_plan_step failed (%s); baseline action", exc)
                step.update({"error": str(exc), "fallback": True})
                self._wm_steps.append(step)
                return response
            # (e) telemetry appended above.
            self._wm_steps.append(step)
            # (f) Override the baseline action with the beam's step-0 calls when it won.
            if detail.get("override_applied") and detail.get("calls"):
                override = self._response_with_calls(detail["calls"])
                logger.info("mcp_react: beam_plan overrode baseline (%s)", detail.get("override_reason"))
                self.trajectory.append(
                    {"role": "assistant", "content": override,
                     "wm": {"strategy": "beam_plan", "override": True, "reason": detail.get("override_reason")}}
                )
                return override
            return response

        if strategy == "hier_latent_cem":
            step = {
                "step": self.metrics["turns"], "strategy": "hier_latent_cem",
                "backend": self.wm_config.backend,
            }
            # Guard: hier_latent_cem needs the JEPA canonical-event backend; otherwise behave
            # exactly like the plain no-WM agent (one sample, no override). Same MPC shape as
            # beam_plan above, but the imagined trajectory comes from the hierarchical
            # latent-action CEM instead of a discrete LLM-proposed candidate pool.
            if not hasattr(self.wm, "hier_latent_cem_step") or not self.wm.supports_hier_cem():
                step["fallback"] = True
                self._wm_steps.append(step)
                return await self._call_llm(messages)
            # (a) Show the cached imagined plan to the agent BEFORE it acts (transient only —
            # NOT appended to `messages`, so the WM always re-plans from the clean flow).
            injection = self.wm.hier_injection_text()
            guided = [*messages, {"role": "user", "content": injection}] if injection else messages
            # (b) Sample the baseline next action once.
            response = await self._call_llm(guided)
            # (c) seed_calls = the baseline response's tool call(s) (see _ai_event's shape).
            seed_calls = [
                {"name": tc["name"], "arguments": tc["args"]}
                for tc in self._ai_event(response)["tool_calls"]
            ]
            user_query = next(
                (e.get("content", "") for e in conversation_flow if e.get("type") == "user_message"), ""
            )
            # (d) One MPC decision (re-plan / follow / override). Never crashes the run.
            try:
                res = self.wm.hier_latent_cem_step(
                    conversation_flow, seed_calls=seed_calls, user_query=user_query
                )
                detail = res.detail or {}
                step.update({"injected_chars": len(injection or ""), "detail": detail})
            except Exception as exc:
                logger.warning("mcp_react: hier_latent_cem_step failed (%s); baseline action", exc)
                step.update({"error": str(exc), "fallback": True})
                self._wm_steps.append(step)
                return response
            # (e) telemetry appended above.
            self._wm_steps.append(step)
            # (f) Override the baseline action with the CEM plan's step-0 calls when confident.
            if detail.get("override_applied") and detail.get("calls"):
                override = self._response_with_calls(detail["calls"], strategy="hier_latent_cem")
                logger.info("mcp_react: hier_latent_cem overrode baseline (%s)", detail.get("reason"))
                self.trajectory.append(
                    {"role": "assistant", "content": override,
                     "wm": {"strategy": "hier_latent_cem", "override": True, "reason": detail.get("reason")}}
                )
                return override
            return response

        if strategy == "selection":
            n = max(1, int(self.wm_config.n))
            temp = max(self.temperature, 0.7)  # diversity for ranking
            candidates = [self._sync_complete(messages, temp) for _ in range(n)]
            candidates = [c for c in candidates if c] or [self._sync_complete(messages, self.temperature)]
            events = [self._ai_event(c) for c in candidates]
            idx = 0
            step: dict[str, Any] = {
                "step": self.metrics["turns"], "strategy": "selection",
                "backend": self.wm_config.backend, "num_candidates": len(candidates),
            }
            try:
                sel = self.wm.select(conversation_flow, events)
                if 0 <= sel.index < len(candidates):
                    idx = sel.index
                step.update({
                    "index": idx,
                    "candidates": [self._action_summary(c) for c in candidates],
                    "chosen": self._action_summary(candidates[idx]),
                    "detail": sel.detail,
                })
            except Exception as exc:
                logger.warning("mcp_react: select failed (%s); candidate 0", exc)
                step.update({"index": idx, "error": str(exc), "fallback": True})
            self._wm_steps.append(step)
            chosen = candidates[idx]
            self.trajectory.append(
                {"role": "assistant", "content": chosen,
                 "wm": {"strategy": "selection", "index": idx, "num_candidates": len(candidates)}}
            )
            return chosen

        if strategy in {"revision", "reference"}:
            pending = self._pending_reference if strategy == "reference" else ""
            guided = [*messages, {"role": "user", "content": pending}] if pending else messages
            response = await self._call_llm(guided)
            seed_calls = [
                {"name": call["name"], "arguments": call["args"]}
                for call in self._ai_event(response)["tool_calls"]
            ]
            step = {
                "step": self.metrics["turns"],
                "strategy": strategy,
                "backend": self.wm_config.backend,
                "seed_calls": seed_calls,
            }
            try:
                result = self.wm.action_feedback(
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
                logger.warning("mcp_react: %s feedback failed (%s); baseline action", strategy, exc)
                step.update({"error": str(exc), "fallback": True})
                self._wm_steps.append(step)
                self._pending_reference = ""
                return response
            self._wm_steps.append(step)
            if strategy == "reference":
                self._pending_reference = result.text or ""
                return response
            if result.text:
                return await self._call_llm(
                    [*messages, {"role": "user", "content": result.text}]
                )
            return response

        if strategy in {"prompt_injection", "itp_i"}:
            advice_text = ""
            step = {
                "step": self.metrics["turns"], "strategy": strategy,
                "backend": self.wm_config.backend,
            }
            try:
                advice = self.wm.advise(conversation_flow, history=self._wm_state_history)
                advice_text = advice.text or ""
                # Capture the actual imagined trajectory EWM returned (predicted states per
                # imagined step), not just its length. Cap via WM_TRAJECTORY_LOG_CHARS
                # (0 = unlimited) to bound trajectory size.
                cap = self._traj_log_chars()
                step.update({
                    "injected_chars": len(advice_text),
                    "imagined_trajectory": (advice_text if cap <= 0 else advice_text[:cap]),
                    "imagined_trajectory_truncated": cap > 0 and len(advice_text) > cap,
                    "detail": advice.detail,
                })
                current_state = (advice.detail or {}).get("current_state")
                if current_state:
                    self._wm_state_history.append(str(current_state))
                if strategy == "itp_i":
                    step["itp_i_policy_reflect_calls"] = 1
            except Exception as exc:
                logger.warning("mcp_react: advise failed (%s); no injection", exc)
                step.update({"error": str(exc), "fallback": True})
            self._wm_steps.append(step)
            if advice_text:
                # Inject transiently (not persisted to the running message history).
                augmented = [*messages, {"role": "user", "content": advice_text}]
                return await self._call_llm(augmented)
            return await self._call_llm(messages)

        return await self._call_llm(messages)

    async def _final_response_from_history(
        self,
        messages: list[dict[str, str]],
        *,
        reason: str,
    ) -> tuple[str, str]:
        """Use the final reserved turn for synthesis only; no tools are available."""
        self.metrics["forced_final_responses"] = (
            int(self.metrics.get("forced_final_responses", 0)) + 1
        )
        final_prompt = (
            "You have no tool calls remaining. Use only the real observations and schema "
            "information already shown above. Return exactly one concise final answer in "
            "<respond>...</respond>. Do not call <execute> or <describe>."
        )
        response = await self._call_llm([*messages, {"role": "user", "content": final_prompt}])
        action = self._extract_action(response)
        if action["type"] == "respond" and not _is_empty_answer(action["content"]):
            final_answer = str(action["content"])
        else:
            final_answer = self._fallback_answer(response)
        self.trajectory.append(
            {
                "role": "metadata",
                "event_type": "finalization_guard",
                "reason": reason,
                "response_action": action.get("type"),
                "forced_final_response": True,
            }
        )
        return final_answer, response

    # --------------------------------------------------------------------- run

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        # No WM (or WM build failed) → identical to baseline_crm_agent.
        if not self._wm_enabled():
            return await super().run(message, updater)

        self.reset_metrics()
        self._wm_steps = []
        self._pending_reference = ""
        self._wm_state_history = []
        task = self._parse_task(get_message_text(message))
        task_id = task.get("task_id", "unknown")
        category = task.get("category", "unknown")

        # Degenerate cases (no key / DB missing) carry no WM benefit → defer to baseline.
        if not self.api_key:
            return await super().run(message, updater)

        await updater.update_status(
            TaskState.working, new_agent_text_message(f"Processing (WM): {category}")
        )

        org_type = task.get("config", {}).get("org_type", "b2b")
        db = CRMDatabase(org_type=org_type)
        try:
            tables = db.get_tables()
            if not db.available or not tables:
                db.close()
                return await super().run(message, updater)

            system_msg = SYSTEM_PROMPT + f"\n\n## Current Database Tables\n{', '.join(tables)}"
            entropy = task.get("entropy", {})
            if entropy.get("drift_level") and entropy["drift_level"] != "none":
                system_msg += (
                    f"\n\n⚠️ SCHEMA DRIFT ACTIVE ({entropy['drift_level']}): "
                    "Column names may have been renamed! Use <describe> to verify before querying!"
                )

            user_content = f"Question: {task['prompt']}"
            if task.get("context"):
                user_content += f"\n\nContext:\n{task['context']}"
            if task.get("optional_context"):
                user_content += f"\n\nDomain Info:\n{task['optional_context'][:1500]}"

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ]
            conversation_flow: list[dict[str, Any]] = [
                # Carry the tool catalog in the flow (not a build_world_model arg) so WM
                # backends that need it (ewm_imagined's text-ReAct rollout) build a
                # training-matched prompt, while build_world_model(config, chat_fn) stays uniform.
                {"type": "tools", "tools": _CRM_TOOLS},
                {"type": "system_message", "content": system_msg},
                {"type": "user_message", "content": user_content},
            ]

            if hasattr(self.wm, "reset_episode"):
                self.wm.reset_episode()  # clear beam_plan/hier_latent_cem MPC state between tasks

            final_answer: str | None = None
            response = ""
            for turn in range(self.max_turns):
                self.metrics["turns"] += 1
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Turn {turn + 1}/{self.max_turns} (WM)"),
                )

                if turn >= self.max_turns - 1:
                    final_answer, response = await self._final_response_from_history(
                        messages, reason="reserved_final_turn"
                    )
                    break

                response = await self._wm_step(messages, conversation_flow)
                action = self._extract_action(response)
                conversation_flow.append(self._ai_event(response))

                if action["type"] == "execute" and action["content"]:
                    self.metrics["tool_calls"] += 1
                    self.metrics["queries"] += 1
                    result = db.execute_query(action["content"])
                    if result["success"]:
                        obs = f"Result ({result['count']} rows): {json.dumps(result['data'][:8], default=str)}"
                    else:
                        obs = f"SQL Error: {result['error']}"
                        self.metrics["failed_queries"] += 1
                    self.trajectory.append(
                        {"role": "tool", "name": "execute", "content": action["content"], "result": result}
                    )
                    conversation_flow.append(
                        {"type": "tool_result", "tool_name": "execute", "result": result, "gym_server": "crm-sqlite"}
                    )
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"[Observation: {obs}]"})

                elif action["type"] == "describe" and action["content"]:
                    self.metrics["tool_calls"] += 1
                    result = db.describe_table(action["content"])
                    if result["success"]:
                        cols = [f"{c['name']}" for c in result["columns"]]
                        obs = f"{result['table']} ({result['row_count']} rows): {', '.join(cols)}"
                    else:
                        obs = f"Error: {result['error']}"
                    self.trajectory.append(
                        {"role": "tool", "name": "describe", "content": action["content"], "result": result}
                    )
                    conversation_flow.append(
                        {"type": "tool_result", "tool_name": "describe", "result": result, "gym_server": "crm-sqlite"}
                    )
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"[Schema: {obs}]"})

                elif action["type"] == "respond" and action["content"]:
                    if _is_empty_answer(action["content"]):
                        messages.append({"role": "assistant", "content": response})
                        messages.append({
                            "role": "user",
                            "content": (
                                "Do not final-answer with None/unknown. Use <execute> to query the CRM "
                                "database or <describe> to inspect schema, then provide a concrete answer."
                            ),
                        })
                    else:
                        final_answer = action["content"]
                        break
                else:
                    if turn >= self.max_turns - 2:
                        final_answer = self._fallback_answer(response)
                        break
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Please use <execute> for SQL, <describe> for schema, "
                            "or <respond> for your final answer."
                        ),
                    })

            if not final_answer:
                final_answer = self._fallback_answer(response)

            self.metrics["failed_queries"] = db.failed_queries
            logger.info("Task %s WM answer: %s", task_id, str(final_answer)[:100])

            await self._add_internal_trajectory_artifact(
                updater, task_id=str(task_id), category=str(category)
            )
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=final_answer)),
                    Part(root=DataPart(data={
                        "task_id": task_id,
                        "category": category,
                        "answer": final_answer,
                        "full_response": response[:1000],
                        "metrics": self.metrics,
                        "wm_strategy": self.wm_config.strategy,
                        "wm_backend": self.wm_config.backend,
                        "wm_steps": len(self._wm_steps),
                    })),
                ],
                name="Answer",
            )
        finally:
            db.close()
