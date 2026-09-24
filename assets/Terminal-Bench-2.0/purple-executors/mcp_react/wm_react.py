"""World-Model–guided shell agent for Terminal-Bench-2.0 (``ejepa_wm``-backed).

Terminal-Bench analogue of EnterpriseOps-Gym's ``mcp_react`` orchestrator and of the
CRMArenaPro ``mcp_react`` executor. The gym reaches tools over MCP servers; Terminal-Bench
has no MCP layer, so this reuses the sibling ``llm_shell`` executor's shell-command tool
loop verbatim and bolts the shared :mod:`ejepa_wm` world-model hook onto its single decision
point (``_next_action``).

``llm_shell`` drives a *per-step* A2A protocol (green runs each command and returns the
``exec_result``), so there is no in-process loop to wrap. Instead the ``conversation_flow``
that ``ejepa_wm`` consumes is reconstructed from ``session.history`` on every step.

Strategies (env ``WM_STRATEGY``), identical to the other ``mcp_react`` executors:

* ``selection`` (+ ``WM_BACKEND=ewm_predict``) — sample ``WM_N`` candidate commands, let the
  WM rank them, commit the best;
* ``prompt_injection`` (+ ``ewm_imagined`` / ``llm``) — inject the WM's guidance block before
  the next command;
* ``itp_i`` (+ ``ewm_imagined``) — Imagine-Then-Plan inference: ask the WM to imagine a
  trajectory and inject it for policy reflection before the next command;
* ``revision`` / ``reference`` — predict feedback for the proposed shell command and either
  revise immediately or expose the prediction at the following decision step;
* ``beam_plan`` (+ ``ewm_imagined`` with a JEPA canonical-event checkpoint) — JEPA MPC
  lookahead: every ``WM_BEAM_MPC_EXECUTE_STEPS`` steps the WM proposes ``m`` candidate
  commands per horizon step, scores the imagined outcomes with the canonical-event heads
  (LLM-free) and can override the agent's baseline command when it beats it by more than
  ``WM_BEAM_PLAN_SCORE_MARGIN``; degrades to plain ``llm_shell`` with any other backend;
* ``hier_latent_cem`` (+ ``ewm_imagined`` with a JEPA canonical-event checkpoint) — a second
  JEPA-only MPC lookahead, same cadence/override contract as ``beam_plan`` but planning via
  hierarchical latent-action CEM (``WM_HIER_CEM_*`` knobs) instead of discrete LLM-proposed
  per-horizon-step candidates; degrades to plain ``llm_shell`` with any other backend;
* unset / ``none`` — plain ``llm_shell`` behavior (no WM), so the executor always runs.

NOTE: the bundled EWM is fine-tuned on EnterpriseOps-Gym tool outcomes; scoring shell
commands with it is an intentional out-of-domain (over-fit) probe, not an expected win.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InternalError, InvalidParamsError, Part, TaskState, TextPart
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

logger = logging.getLogger(__name__)

# Load the sibling llm_shell executor module under a distinct name (both packages have an
# executor.py, so a plain ``import executor`` would collide). We reuse its tool loop,
# LLM-client plumbing, session type, and prompt wholesale.
_LLM_SHELL_PATH = Path(__file__).resolve().parent.parent / "llm_shell" / "executor.py"
_spec = importlib.util.spec_from_file_location("tb_llm_shell_base", _LLM_SHELL_PATH)
assert _spec is not None and _spec.loader is not None
base = importlib.util.module_from_spec(_spec)
sys.modules["tb_llm_shell_base"] = base
_spec.loader.exec_module(base)


def _ensure_ejepa_wm_on_path() -> None:
    src = Path(__file__).resolve().parents[4] / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


# The single shell "tool" advertised to ejepa_wm (only ewm_imagined consumes the catalog).
_SHELL_TOOLS: list[dict[str, Any]] = [
    {
        "name": "run_shell",
        "description": "Run one shell command in the task container and observe its output.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "shell command"},
                "timeout": {"type": "integer", "description": "seconds, 1-300"},
            },
            "required": ["command"],
        },
    }
]


class WmShellExecutor(base.LlmShellExecutor):  # type: ignore[misc, name-defined]
    """``llm_shell`` shell agent + a ``ejepa_wm`` world-model hook at ``_next_action``."""

    def __init__(self) -> None:
        super().__init__()
        _ensure_ejepa_wm_on_path()
        self.wm = None
        try:
            from ejepa_wm import build_world_model, wm_config_from_env

            self.wm_config = wm_config_from_env()
            if self.wm_config.strategy != "none":
                # Uniform contract: build_world_model(config, chat_fn). The shell tool catalog
                # is carried in the conversation_flow as a `tools` event (see _conversation_flow),
                # so ewm_imagined builds its rollout prompt without an executor-specific arg.
                self.wm = build_world_model(self.wm_config, chat_fn=self._wm_chat_fn)
        except Exception as exc:  # ejepa_wm missing / build failed → plain llm_shell
            logger.warning("mcp_react(tb): WM unavailable (%s); plain shell agent", exc)
            self.wm = None
        strat = getattr(getattr(self, "wm_config", None), "strategy", "none")
        backend = getattr(getattr(self, "wm_config", None), "backend", "noop")
        logger.info(
            "terminal-bench mcp_react: wm=%s strategy=%s backend=%s n=%s",
            "on" if self.wm is not None else "off", strat, backend,
            getattr(getattr(self, "wm_config", None), "n", 1),
        )

    # ------------------------------------------------------------------ helpers

    def _wm_enabled(self) -> bool:
        return self.wm is not None and getattr(self.wm_config, "strategy", "none") != "none"

    @staticmethod
    def _env_temperature() -> float:
        try:
            return float(os.getenv("TERMINAL_BENCH_LLM_TEMPERATURE", "0"))
        except ValueError:
            return 0.0

    @staticmethod
    def _traj_log_chars() -> int:
        """Max chars of the imagined trajectory to log per step (0 = unlimited)."""
        try:
            return int(os.getenv("WM_TRAJECTORY_LOG_CHARS", "0"))
        except ValueError:
            return 0

    def _complete_text(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        """Raw LLM completion text for ``messages`` (reuses llm_shell's client plumbing)."""
        config = self._llm_config
        kwargs = dict(base._completion_kwargs(config))
        if temperature is not None and not base._is_gpt5_family(config.model):
            kwargs["temperature"] = temperature
        use_azure_sdk = config.provider == "azure" and os.getenv(
            "TERMINAL_BENCH_LLM_USE_HTTP", ""
        ).strip().lower() not in {"1", "true", "yes"}
        if use_azure_sdk:
            from openai import AzureOpenAI

            client = AzureOpenAI(
                api_key=config.api_key,
                azure_endpoint=str(config.base_url).rstrip("/"),
                api_version=config.api_version,
            )
            completion = client.chat.completions.create(
                model=config.deployment, messages=messages, **kwargs
            )
            payload = completion.model_dump(mode="json")
        else:
            body: dict[str, Any] = {"messages": messages, **kwargs}
            if config.provider != "azure":
                body["model"] = config.model
            with httpx.Client(timeout=180.0) as client:
                resp = client.post(
                    base._chat_completions_url(config),
                    headers=base._chat_completions_headers(config),
                    json=body,
                )
                if resp.is_error:
                    raise RuntimeError(
                        f"LLM request failed ({resp.status_code}): {base._http_error_detail(resp)}"
                    )
                payload = resp.json()
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"Unexpected LLM response: {payload}")
        return base._message_content_to_text(choices[0].get("message") or {})

    def _wm_chat_fn(self, messages: list[dict[str, str]]) -> str:
        """``ejepa_wm.ChatFn``: OpenAI-style messages -> reply text."""
        try:
            return self._complete_text(messages)
        except Exception as exc:
            logger.warning("mcp_react(tb): WM chat_fn failed: %s", exc)
            return ""

    def _sample_action(self, session, temperature: float) -> dict[str, Any]:
        """Sample one candidate action (parsed protocol object) at a given temperature."""
        messages = base._build_chat_messages(session)
        text = self._complete_text(messages, temperature=temperature)
        try:
            return base._extract_json_object(text)
        except Exception:
            return {"kind": "final", "output": text[:200]}

    def _action_with_injection(self, session, advice_text: str) -> dict[str, Any]:
        messages = base._build_chat_messages(session)
        if advice_text:
            messages = [*messages, {"role": "user", "content": advice_text}]
        text = self._complete_text(messages)
        return base._extract_json_object(text)

    @staticmethod
    def _ai_event(action: dict[str, Any]) -> dict[str, Any]:
        tool_calls: list[dict[str, Any]] = []
        if str(action.get("kind")) == "exec_request" and action.get("command"):
            tool_calls = [{
                "name": "run_shell",
                "args": {"command": action.get("command"), "timeout": action.get("timeout", 30)},
            }]
        return {"type": "ai_message", "content": json.dumps(action, ensure_ascii=False), "tool_calls": tool_calls}

    def _conversation_flow(self, session) -> list[dict[str, Any]]:
        """Reconstruct the generic ejepa_wm conversation_flow from session.history."""
        flow: list[dict[str, Any]] = [
            # Carry the tool catalog in the flow (not a build_world_model arg) so WM backends
            # that need it (ewm_imagined's text-ReAct rollout) build a training-matched prompt,
            # while build_world_model(config, chat_fn) stays uniform.
            {"type": "tools", "tools": _SHELL_TOOLS},
            {"type": "system_message", "content": base.SYSTEM_PROMPT},
            {"type": "user_message", "content": f"Task instruction:\n{session.instruction}"},
        ]
        for entry in session.history:
            role = str(entry.get("role"))
            content = str(entry.get("content", ""))
            if role == "assistant":
                try:
                    action = json.loads(content)
                except json.JSONDecodeError:
                    action = {"kind": "raw", "output": content}
                flow.append(self._ai_event(action))
            elif role == "tool":
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    result = {"raw": content}
                flow.append({
                    "type": "tool_result", "tool_name": "run_shell",
                    "result": result, "gym_server": "shell",
                })
        return flow

    def _finalize_action(self, session, action: dict[str, Any]) -> dict[str, Any]:
        """Validate/normalize an action and append it to history (mirrors llm_shell)."""
        kind = str(action.get("kind", ""))
        if kind == "exec_request":
            command = action.get("command")
            timeout = action.get("timeout", 30)
            if not isinstance(command, str) or not command.strip():
                raise ValueError("exec_request requires non-empty command")
            if not isinstance(timeout, int):
                timeout = 30
            timeout = max(1, min(int(timeout), 300))
            normalized = {"kind": "exec_request", "command": command.strip(), "timeout": timeout}
            session.history.append({"role": "assistant", "content": json.dumps(normalized)})
            return normalized
        if kind == "final":
            return {"kind": "final", "output": str(action.get("output", ""))}
        raise ValueError(f"LLM returned unsupported action: {action}")

    # ----------------------------------------------------------------- execute

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Same protocol handling as ``llm_shell`` but records WM telemetry.

        Overridden (rather than inherited) so the captured ``internal_trajectory`` artifact
        includes ``wm_strategy`` / ``wm_backend`` and the per-step ``wm_steps`` recorded by
        ``_next_action`` — otherwise the world-model decisions are dropped on the floor.
        """
        if context.current_task:
            task = context.current_task
        elif context.message:
            task = new_task(context.message)
        else:
            raise ServerError(error=InvalidParamsError(message="No message provided"))
        if not context.message:
            raise ServerError(error=InvalidParamsError(message="No message provided"))

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        try:
            request_text = base._extract_request_text(context.message.parts)
            payload = base._decode_payload(request_text)
            kind = str(payload.get("kind", ""))
            context_id = task.context_id or task.id

            if kind == "task":
                instruction = payload.get("instruction")
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError("task payload requires non-empty instruction")
                self._sessions[context_id] = base._Session(instruction=instruction.strip())
                response_payload = self._next_action(self._sessions[context_id])
            elif kind == "exec_result":
                session = self._sessions.get(context_id)
                if session is None:
                    raise ValueError("exec_result received without an active session")
                sanitized = base._sanitize_exec_result_payload(payload)
                session.history.append(
                    {"role": "tool", "content": json.dumps(sanitized, ensure_ascii=False)}
                )
                response_payload = self._next_action(session)
            else:
                raise ValueError(f"Unsupported payload kind: {kind}")

            session = self._sessions.get(context_id)
            if session is not None and session.history:
                internal_trajectory = {
                    "executor": "mcp_react",
                    "format": "terminal-bench-shell-v1",
                    "payload": {
                        "info": {
                            "steps": session.steps,
                            "provider": self._llm_config.provider,
                            "deployment": self._llm_config.deployment,
                            "api_version": self._llm_config.api_version,
                            "wm_strategy": getattr(self.wm_config, "strategy", "none"),
                            "wm_backend": getattr(self.wm_config, "backend", "noop"),
                            "wm_enabled": self._wm_enabled(),
                            "wm_agent_call_count": (
                                getattr(self.wm, "agent_call_count", None)
                                if getattr(self.wm_config, "strategy", "none")
                                in ("itp_i", "beam_plan", "hier_latent_cem")
                                else None
                            ),
                            # Nested under info so the green trajectory recorder preserves it
                            # (it copies info verbatim but drops unknown top-level keys).
                            "wm_steps": list(getattr(session, "wm_steps", [])),
                        },
                        "messages": list(session.history),
                    },
                }
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=json.dumps(internal_trajectory, ensure_ascii=False)))],
                    name=base.INTERNAL_TRAJECTORY_ARTIFACT_NAME,
                )
            await updater.add_artifact(parts=[Part(root=TextPart(text=json.dumps(response_payload)))])
            await updater.complete()
        except Exception as exc:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"mcp_react execution failed: {exc}", task.context_id, task.id),
                final=True,
            )
            raise ServerError(error=InternalError(message=str(exc))) from exc

    # ------------------------------------------------------------- WM decision

    def _next_action(self, session) -> dict[str, Any]:
        if not self._wm_enabled():
            return super()._next_action(session)

        # Episode reset: the first _next_action of a task (kind == "task") sees an empty
        # history; clear any per-episode beam_plan MPC state carried over from a prior task.
        if not session.history and hasattr(self.wm, "reset_episode"):
            self.wm.reset_episode()

        session.steps += 1
        if session.steps > base._max_steps():
            return {"kind": "final", "output": "step limit reached"}
        if not hasattr(session, "wm_steps"):
            session.wm_steps = []  # type: ignore[attr-defined]
        if not hasattr(session, "wm_state_history"):
            session.wm_state_history = []  # type: ignore[attr-defined]

        flow = self._conversation_flow(session)
        strategy = self.wm_config.strategy
        # Always record the WM attempt + outcome (incl. failures), so an empty wm_steps can
        # never be confused with "EWM ran but did nothing". A gym-trained backend that
        # cannot process a non-gym flow shows up here as {"error": ...} + fallback=True.
        wm_step: dict[str, Any] = {
            "step": session.steps, "strategy": strategy, "backend": self.wm_config.backend,
        }
        try:
            if strategy == "selection":
                n = max(1, int(self.wm_config.n))
                temp = max(self._env_temperature(), 0.7)  # diversity for ranking
                candidates = [self._sample_action(session, temp) for _ in range(n)]
                events = [self._ai_event(a) for a in candidates]
                idx = 0
                sel = self.wm.select(flow, events)
                if 0 <= sel.index < len(candidates):
                    idx = sel.index
                wm_step.update({
                    "n": len(candidates),
                    "index": idx,
                    "candidates": [
                        c.get("command") if c.get("kind") == "exec_request" else c.get("kind")
                        for c in candidates
                    ],
                    "chosen": candidates[idx].get("command") or candidates[idx].get("kind"),
                    "detail": sel.detail,
                })
                action = candidates[idx]
            elif strategy in {"revision", "reference"}:
                pending = str(getattr(session, "wm_pending_reference", "") or "")
                action = self._action_with_injection(
                    session, pending if strategy == "reference" else ""
                )
                seed_calls = [
                    {"name": call["name"], "arguments": call["args"]}
                    for call in (self._ai_event(action).get("tool_calls") or [])
                ]
                result = self.wm.action_feedback(
                    flow,
                    seed_calls=seed_calls,
                    user_query=session.instruction,
                    mode=strategy,
                )
                wm_step.update(
                    {
                        "injected_chars": (
                            len(pending) if strategy == "reference" else len(result.text or "")
                        ),
                        "seed_calls": seed_calls,
                        "detail": result.detail or {},
                    }
                )
                if strategy == "reference":
                    session.wm_pending_reference = result.text or ""
                elif result.text:
                    action = self._action_with_injection(session, result.text)
            elif strategy in {"prompt_injection", "itp_i"}:
                advice = self.wm.advise(
                    flow, history=session.wm_state_history  # type: ignore[attr-defined]
                )
                text = advice.text or ""
                # Capture the actual imagined trajectory EWM returned (its predicted states
                # per imagined step) so the result is observable, not just its length. Cap
                # via WM_TRAJECTORY_LOG_CHARS (0 = unlimited) to bound trajectory size.
                cap = self._traj_log_chars()
                wm_step.update({
                    "injected_chars": len(text),
                    "imagined_trajectory": (text if cap <= 0 else text[:cap]),
                    "imagined_trajectory_truncated": cap > 0 and len(text) > cap,
                    "detail": advice.detail,
                })
                current_state = (advice.detail or {}).get("current_state")
                if current_state:
                    session.wm_state_history.append(str(current_state))  # type: ignore[attr-defined]
                if strategy == "itp_i":
                    wm_step["itp_i_policy_reflect_calls"] = 1
                action = self._action_with_injection(session, text)
            elif strategy == "beam_plan":
                # JEPA MPC lookahead. Degrade to the plain agent when the backend can't beam.
                if not hasattr(self.wm, "beam_plan_step") or not self.wm.supports_beam_plan():
                    wm_step.update({"event": "unsupported", "fallback": True})
                    action = base._call_llm(session, self._llm_config)
                else:
                    # (b) transient injection of the cached imagined plan before proposing;
                    # (c) sample the baseline command once (injection is not persisted).
                    injection = self.wm.beam_injection_text()
                    action = self._action_with_injection(session, injection)
                    # (d) the baseline next tool call: the single run_shell command.
                    seed_calls = [
                        {"name": tc["name"], "arguments": tc["args"]}
                        for tc in (self._ai_event(action).get("tool_calls") or [])
                    ]
                    # (e) one MPC decision over the reconstructed flow.
                    result = self.wm.beam_plan_step(
                        flow, seed_calls=seed_calls, user_query=session.instruction
                    )
                    detail = result.detail or {}
                    # (f) per-step WM telemetry.
                    wm_step.update({
                        "injected_chars": len(injection or ""),
                        "seed_calls": seed_calls,
                        "override_applied": bool(detail.get("override_applied")),
                        "detail": detail,
                    })
                    # (g) apply an override by mapping the chosen call back to an exec_request.
                    calls = detail.get("calls") or []
                    if detail.get("override_applied") and calls:
                        args = calls[0].get("arguments", {}) or {}
                        action = {
                            "kind": "exec_request",
                            "command": args.get("command", ""),
                            "timeout": args.get("timeout", 30),
                        }
            elif strategy == "hier_latent_cem":
                # Second JEPA MPC lookahead (hierarchical latent-action CEM); same contract as
                # beam_plan above. Degrade to the plain agent when the backend can't hier-cem.
                if not hasattr(self.wm, "hier_latent_cem_step") or not self.wm.supports_hier_cem():
                    wm_step.update({"event": "unsupported", "fallback": True})
                    action = base._call_llm(session, self._llm_config)
                else:
                    # (b) transient injection of the cached imagined plan before proposing;
                    # (c) sample the baseline command once (injection is not persisted).
                    injection = self.wm.hier_injection_text()
                    action = self._action_with_injection(session, injection)
                    # (d) the baseline next tool call: the single run_shell command.
                    seed_calls = [
                        {"name": tc["name"], "arguments": tc["args"]}
                        for tc in (self._ai_event(action).get("tool_calls") or [])
                    ]
                    # (e) one MPC decision over the reconstructed flow.
                    result = self.wm.hier_latent_cem_step(
                        flow, seed_calls=seed_calls, user_query=session.instruction
                    )
                    detail = result.detail or {}
                    # (f) per-step WM telemetry.
                    wm_step.update({
                        "injected_chars": len(injection or ""),
                        "seed_calls": seed_calls,
                        "override_applied": bool(detail.get("override_applied")),
                        "detail": detail,
                    })
                    # (g) apply an override by mapping the chosen call back to an exec_request.
                    calls = detail.get("calls") or []
                    if detail.get("override_applied") and calls:
                        args = calls[0].get("arguments", {}) or {}
                        action = {
                            "kind": "exec_request",
                            "command": args.get("command", ""),
                            "timeout": args.get("timeout", 30),
                        }
            else:
                wm_step = None  # type: ignore[assignment]
                action = base._call_llm(session, self._llm_config)
        except Exception as exc:
            logger.warning("mcp_react(tb): WM step failed (%s); falling back to plain action", exc)
            wm_step["error"] = str(exc)
            wm_step["fallback"] = True
            action = base._call_llm(session, self._llm_config)

        if wm_step is not None:
            session.wm_steps.append(wm_step)  # type: ignore[attr-defined]
        return self._finalize_action(session, action)
