"""World-model-assisted orchestrator for EnterpriseOps-Gym replay.

Plugs our agent and world-model generators into the gym's BenchmarkExecutor /
VerifierEngine pipeline so held-out gym tasks can be scored with the same
verifiers (`database_state`, `response_checker`) that `evaluate.py` uses.

The orchestrator subclasses the gym's `AgentOrchestrator` so it can route tool
calls through `_execute_tool_call`, but it does not use `self.llm_client` —
agent decisions come from our `agent_generator`, and the world-model loop
(revision / imagined-rollout / react_wm-foresight) uses `world_model_generator`.

Six modes selectable via `mode`:
- "baseline":          agent generator only, no world-model interposition.
- "revision":          world-model predicts outcome and the agent revises tool calls.
- "imagined":          world-model rolls out an imagined trajectory before committing.
- "react_wm":          after turn 0, build state from rolling history, roll
                       the world model forward `fixed_k` steps, and inject
                       `[World-model foresight] ... [End foresight]` into the
                       chat history before the next agent call.
- "react_wm_decide_k": same as react_wm but the agent itself picks K via a
                       one-shot prompt at each turn.
- "react_wm_rl_k":     same as react_wm but a separately trained K-controller
                       (small local LM with an RL-trained K-head) picks K per
                       turn while the action policy remains unchanged.
"""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


logger = logging.getLogger(__name__)


def _render_state_text_for_k_controller(
    *,
    current_state: dict[str, Any],
    state_history: list[dict[str, Any]],
    input_history: list[dict[str, Any]],
    include_world_model_history: bool,
    system_prompt_max_chars: int,
    task: Any,
) -> str:
    """Build the K-controller's state text from the current orchestrator state.

    The text shape MUST match what
    :func:`src.itp.training.train_adaptive_k.build_state_text_from_example`
    produces at Stage I/II training time, so the K-head sees the same prompt
    format at inference time.
    """
    from src.finetuning import WorldModelStateExample, make_blank_state_like
    from src.itp.training.train_adaptive_k import build_state_text_from_example

    user_prompt = task.user_messages[-1] if getattr(task, "user_messages", None) else ""
    example = WorldModelStateExample(
        trajectory_id=str(getattr(task, "trajectory_index", 0)),
        trajectory_index=int(getattr(task, "trajectory_index", 0)),
        interaction_index=len(state_history or []),
        system_prompt=getattr(task, "system_prompt", "") or "",
        user_prompt=user_prompt,
        action="",
        state_history=list(state_history or []),
        input_history=list(input_history or []),
        previous_state=current_state,
        state=make_blank_state_like(current_state),
    )
    return build_state_text_from_example(
        example,
        include_input_history=include_world_model_history,
        system_prompt_max_chars=system_prompt_max_chars,
    )


def build_react_tool_descriptions_from_gym(available_tools: list[dict[str, Any]]) -> str:
    """Render gym MCP tool dicts into the descriptive block our react prompt expects.

    Mirrors the shape produced by `evaluation.build_react_tool_descriptions` for
    LangChain BaseTool objects so the same react system prompt template works
    with gym tools.
    """
    sections = []
    for tool in available_tools:
        name = tool.get("name", "unknown")
        description = tool.get("description", "")
        input_schema = tool.get("inputSchema") or tool.get("input_schema") or {}
        schema_text = (
            json.dumps(input_schema, ensure_ascii=False, indent=2)
            if input_schema
            else "{}"
        )
        sections.append(
            f"- {name}: {description}\n  arguments schema: {schema_text}"
        )
    return "\n".join(sections)


def build_world_model_assisted_orchestrator_class():
    """Build the orchestrator class.

    Wrapped in a factory because the gym's `AgentOrchestrator` only imports
    cleanly when the gym package is on `sys.path` — callers should arrange that
    before invoking this builder.
    """
    from orchestrators.base import AgentOrchestrator

    class WorldModelAssistedOrchestrator(AgentOrchestrator):
        def __init__(
            self,
            *args,
            agent_generator: Any,
            world_model_generator: Any,
            mode: str = "baseline",
            world_model_target: str = "state",
            include_error_message_in_target: bool = False,
            include_stage_in_target: bool = False,
            include_world_model_history: bool = False,
            internal_thinking_max_iterations: int = 0,
            imagined_trajectory_max_steps: int = 0,
            imagined_trajectory_rollouts: int = 1,
            imagined_rollout_temperature: float = 0.7,
            imagined_trajectory_selection_strategy: str = "llm_judge",
            imagined_trajectory_observation_source: str = "world_model",
            imagined_trajectory_candidate_actions: int = 3,
            imagined_trajectory_top_k: int = 3,
            revision_lookahead_steps: int = 1,
            revision_imagined_rollouts: int = 1,
            revision_rollout_temperature: float = 0.7,
            final_answer_f1_threshold: float = 0.5,
            agent_max_observation_chars: int = 2000,
            agent_replay_history_budget_chars: int = 60000,
            gym_tool_call_timeout_seconds: float = 45.0,
            state_history_size: int = 8,
            system_prompt_max_chars: int = 0,
            action_max_chars: int = 0,
            initial_state: dict[str, Any] | None = None,
            react_wm_k_steps: int = 0,
            react_wm_kmax: int = 3,
            react_wm_foresight_temperature: float = 0.0,
            react_wm_foresight_observation_source: str = "world_model",
            k_controller: Any | None = None,
            **kwargs,
        ):
            options = WorldModelAssistedOrchestratorOptions.from_kwargs(
                kwargs,
                options,
            )
            super().__init__(*args, **kwargs)
            from src.itp.react_wm import REACT_WM_MODES

            allowed_modes = {"baseline", "revision", "imagined", *REACT_WM_MODES}
            if mode not in allowed_modes:
                raise ValueError(
                    f"Unknown orchestrator mode: {mode}. Expected one of {sorted(allowed_modes)}."
                )
            self.agent_generator = agent_generator
            self.world_model_generator = world_model_generator
            self.mode = options.mode
            self.world_model_target = options.world_model_target
            self.include_error_message_in_target = options.include_error_message_in_target
            self.include_stage_in_target = options.include_stage_in_target
            self.include_world_model_history = options.include_world_model_history
            self.internal_thinking_max_iterations = (
                options.internal_thinking_max_iterations
            )
            self.state_history_size = max(0, int(state_history_size))
            self.system_prompt_max_chars = max(0, int(system_prompt_max_chars))
            self.action_max_chars = max(0, int(action_max_chars))
            self.initial_state = initial_state
            self.react_wm_k_steps = max(0, int(react_wm_k_steps))
            self.react_wm_kmax = max(0, int(react_wm_kmax))
            self.react_wm_foresight_temperature = max(0.0, float(react_wm_foresight_temperature))
            self.react_wm_foresight_observation_source = (
                react_wm_foresight_observation_source
                if react_wm_foresight_observation_source in {"world_model", "none"}
                else "world_model"
            )
            self.k_controller = k_controller
            if mode == "react_wm_rl_k" and k_controller is None:
                raise ValueError(
                    "mode=react_wm_rl_k requires a k_controller (see src.itp.k_controller.KController)."
                )
            self._extra_metadata: dict[str, Any] = {}
            self._react_wm_records: list[dict[str, Any]] = []

        def get_result_metadata(self) -> dict[str, Any]:
            return dict(self._extra_metadata)

        async def execute(self) -> dict[str, Any]:
            from src.finetuning import (
                TaskTrajectory,
                append_state_history,
                append_world_model_input_history,
                configure_replay_limits,
                emit_progress,
                is_tool_output_target,
                make_actual_world_model_history_entry_from_results,
                make_blank_state_like,
                normalize_tool_call,
                preview_tool_calls,
                tool_calls_equal,
            )
            from src.evaluation import (
                build_imagined_trajectory_prompt_message,
                build_internal_thinking_messages,
                build_react_action_messages,
                build_react_system_prompt,
                imagine_revision_rollouts,
                imagine_trajectory_candidates,
                parse_agent_decision,
                predict_world_model_feedback,
                summarize_state_for_planning,
                update_state_from_actual_execution,
            )
            from src.itp.k_decider import decide_k_via_agent
            from src.itp.react_wm import (
                REACT_WM_MODES,
                build_foresight_user_message,
                build_world_model_foresight,
            )

            configure_replay_limits(
                observation_chars=self.agent_max_observation_chars,
                history_budget_chars=self.agent_replay_history_budget_chars,
            )

            react_system_prompt = build_react_system_prompt(
                build_react_tool_descriptions_from_gym(self.available_tools)
            )
            user_query = self.config.user_prompt or ""

            conversation: list[dict[str, Any]] = [
                {"role": "system", "content": self.config.system_prompt or ""},
                {"role": "user", "content": user_query},
            ]
            conversation_flow: list[dict[str, Any]] = [
                {"type": "system_message", "content": self.config.system_prompt or ""},
                {"type": "user_message", "content": user_query},
            ]
            tools_used: list[str] = []
            tool_results: list[dict[str, Any]] = []

            current_state = make_blank_state_like(self.initial_state)
            if self.initial_state is not None:
                current_state = self.initial_state
            current_state_history = append_state_history(
                [], current_state, max_items=self.state_history_size
            )
            current_input_history: list[dict[str, Any]] = []
            task = TaskTrajectory(
                trajectory_index=0,
                system_prompt=self.config.system_prompt or "",
                user_messages=[user_query],
                steps=[],
                final_answer="",
                initial_state=current_state,
            )

            steps_taken = 0
            final_answer = ""
            internal_iterations_used = 0
            wm_revisions = 0
            wm_imagined_rollouts = 0
            wm_revision_step_details: list[dict[str, Any]] = []
            imagined_rollout_records: list[dict[str, Any]] = []
            emit_progress(
                "GYM_TASK_START",
                mode=self.mode,
                query=user_query,
            )
            supports_native_tool_calling = getattr(
                self.agent_generator,
                "supports_native_tool_calling",
                None,
            )
            if callable(supports_native_tool_calling):
                use_native_tool_calling = bool(supports_native_tool_calling())
            else:
                use_native_tool_calling = hasattr(self.agent_generator, "invoke_with_tools")

            while steps_taken < self.max_iterations:
                planning_conversation = conversation
                if self.mode in REACT_WM_MODES and steps_taken > 0:
                    if self.mode == "react_wm":
                        k_for_turn = int(self.react_wm_k_steps)
                        k_source = "fixed"
                        k_decision_meta: dict[str, Any] = {"fixed_k": k_for_turn}
                    elif self.mode == "react_wm_decide_k":
                        k_for_turn = decide_k_via_agent(
                            agent_generator=self.agent_generator,
                            conversation=conversation,
                            state=current_state,
                            kmax=int(self.react_wm_kmax),
                            fallback_k=0,
                        )
                        k_source = "agent_prompt"
                        k_decision_meta = {"kmax": int(self.react_wm_kmax)}
                    elif self.mode == "react_wm_rl_k":
                        k_state_text = _render_state_text_for_k_controller(
                            current_state=current_state,
                            state_history=current_state_history,
                            input_history=current_input_history,
                            include_world_model_history=self.include_world_model_history,
                            system_prompt_max_chars=self.system_prompt_max_chars,
                            task=task,
                        )
                        try:
                            decision = self.k_controller.decide_k(k_state_text)
                            k_for_turn = int(decision.k)
                            k_decision_meta = {
                                "kmax": int(self.react_wm_kmax),
                                "logits": decision.logits,
                                "raw": decision.raw,
                            }
                        except Exception as exc:  # noqa: BLE001
                            # Log the full traceback so misconfigurations of
                            # the K-controller checkpoint (missing files,
                            # ABI / version mismatches, etc.) don't get
                            # silently swallowed into an opaque per-task
                            # error string downstream.
                            logger.exception("k_controller.decide_k failed: %s", exc)
                            k_for_turn = 0
                            k_decision_meta = {
                                "kmax": int(self.react_wm_kmax),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        k_source = "k_controller"
                    else:  # pragma: no cover - guarded by allowed_modes
                        k_for_turn = 0
                        k_source = "unknown"
                        k_decision_meta = {}

                    emit_progress(
                        "REACT_WM_DECIDED_K",
                        mode=self.mode,
                        step=steps_taken,
                        k=k_for_turn,
                        k_source=k_source,
                        decision=k_decision_meta,
                    )
                    if k_for_turn > 0:
                        foresight_text, rollout_records = build_world_model_foresight(
                            agent_generator=self.agent_generator,
                            world_model_generator=self.world_model_generator,
                            task=task,
                            conversation=conversation,
                            previous_state=current_state,
                            state_history=current_state_history,
                            input_history=current_input_history,
                            react_system_prompt=react_system_prompt,
                            k_steps=k_for_turn,
                            world_model_target=self.world_model_target,
                            include_error_message_in_target=self.include_error_message_in_target,
                            include_stage_in_target=self.include_stage_in_target,
                            include_world_model_history=self.include_world_model_history,
                            start_interaction_index=steps_taken,
                            state_history_size=self.state_history_size,
                            system_prompt_max_chars=self.system_prompt_max_chars,
                            action_max_chars=self.action_max_chars,
                            foresight_action_temperature=self.react_wm_foresight_temperature,
                            foresight_observation_source=self.react_wm_foresight_observation_source,
                            selection_strategy=self.imagined_trajectory_selection_strategy,
                            candidate_action_count=self.imagined_trajectory_candidate_actions,
                            top_k=self.imagined_trajectory_top_k,
                            rollout_temperature=self.imagined_rollout_temperature,
                        )
                        if foresight_text:
                            planning_conversation = conversation + [
                                build_foresight_user_message(foresight_text)
                            ]
                        emit_progress(
                            "REACT_WM_FORESIGHT",
                            mode=self.mode,
                            step=steps_taken,
                            k=k_for_turn,
                            foresight=foresight_text,
                            rollout_steps=len(rollout_records),
                        )
                        self._react_wm_records.append(
                            {
                                "step_index": steps_taken,
                                "mode": self.mode,
                                "k": k_for_turn,
                                "k_source": k_source,
                                "k_decision_meta": k_decision_meta,
                                "foresight_text": foresight_text,
                                "rollout_records": rollout_records,
                            }
                        )
                    else:
                        self._react_wm_records.append(
                            {
                                "step_index": steps_taken,
                                "mode": self.mode,
                                "k": 0,
                                "k_source": k_source,
                                "k_decision_meta": k_decision_meta,
                                "foresight_text": "",
                                "rollout_records": [],
                            }
                        )
                if self.mode == "imagined":
                    imagined_steps, candidate_rollouts, imagined_selection = imagine_trajectory_candidates(
                        agent_generator=self.agent_generator,
                        world_model_generator=self.world_model_generator,
                        task=task,
                        conversation=conversation,
                        previous_state=current_state,
                        react_system_prompt=react_system_prompt,
                        max_imagined_steps=self.imagined_trajectory_max_steps,
                        world_model_target=self.world_model_target,
                        include_error_message_in_target=self.include_error_message_in_target,
                        include_stage_in_target=self.include_stage_in_target,
                        include_world_model_history=self.include_world_model_history,
                        start_interaction_index=steps_taken,
                        num_rollouts=self.imagined_trajectory_rollouts,
                        rollout_temperature=self.imagined_rollout_temperature,
                        selection_strategy=self.imagined_trajectory_selection_strategy,
                        observation_source=self.imagined_trajectory_observation_source,
                        candidate_action_count=self.imagined_trajectory_candidate_actions,
                        top_k=self.imagined_trajectory_top_k,
                        state_history=current_state_history,
                        input_history=current_input_history,
                        state_history_size=self.state_history_size,
                        system_prompt_max_chars=self.system_prompt_max_chars,
                        action_max_chars=self.action_max_chars,
                    )
                    if imagined_steps:
                        planning_conversation = conversation + [
                            build_imagined_trajectory_prompt_message(imagined_steps)
                        ]
                        wm_imagined_rollouts += 1
                        imagined_rollout_records.append(
                            {
                                "step_index": steps_taken,
                                "starting_state": summarize_state_for_planning(current_state),
                                "imagined_steps": imagined_steps,
                                "candidate_rollouts": candidate_rollouts,
                                "selection": imagined_selection,
                                "observation_source": self.imagined_trajectory_observation_source,
                                "candidate_action_count": self.imagined_trajectory_candidate_actions,
                                "top_k": self.imagined_trajectory_top_k,
                            }
                        )

                response_content = ""
                response_usage_metadata: dict[str, Any] = {}
                response_metadata: dict[str, Any] = {}
                planned_calls: list[dict[str, Any]] = []
                if use_native_tool_calling:
                    response = self.agent_generator.invoke_with_tools(
                        planning_conversation,
                        self.available_tools,
                    )
                    response_content = str(getattr(response, "content", "") or "")
                    response_usage_metadata = getattr(response, "usage_metadata", {}) or {}
                    response_metadata = getattr(response, "response_metadata", {}) or {}
                    for idx, tool_call in enumerate(getattr(response, "tool_calls", []) or []):
                        planned_calls.append(
                            {
                                "id": tool_call.get("id") or f"call_{steps_taken}_{idx}",
                                "name": str(tool_call.get("name", "")).strip(),
                                "arguments": tool_call.get("args", {}),
                            }
                        )
                    if not planned_calls:
                        final_answer = response_content
                        emit_progress(
                            "GYM_AGENT_FINAL_ANSWER",
                            mode=self.mode,
                            step=steps_taken,
                            final_answer=final_answer,
                        )
                        conversation.append({"role": "assistant", "content": final_answer})
                        conversation_flow.append(
                            {
                                "type": "ai_message",
                                "content": final_answer,
                                "usage_metadata": response_usage_metadata,
                                "response_metadata": response_metadata,
                                "tool_calls": [],
                            }
                        )
                        break
                else:
                    raw_decision = self.agent_generator.generate_from_messages(
                        build_react_action_messages(
                            planning_conversation,
                            current_query=user_query,
                            system_prompt=react_system_prompt,
                        )
                    )
                    raw_decision = raw_decision.split("</think>\n", 1)[-1].strip()
                    try:
                        decision = parse_agent_decision(raw_decision)
                    except Exception as exc:
                        logger.warning(f"unparseable agent decision: {exc}")
                        emit_progress(
                            "GYM_AGENT_ACTION_PARSE_ERROR",
                            mode=self.mode,
                            step=steps_taken,
                            error=str(exc),
                            raw_action=raw_decision,
                        )
                        final_answer = ""
                        break

                    if "final_answer" in decision:
                        final_answer = str(decision["final_answer"])
                        emit_progress(
                            "GYM_AGENT_FINAL_ANSWER",
                            mode=self.mode,
                            step=steps_taken,
                            final_answer=final_answer,
                        )
                        conversation.append({"role": "assistant", "content": final_answer})
                        conversation_flow.append(
                            {"type": "ai_message", "content": final_answer, "tool_calls": []}
                        )
                        break

                    if "clarify" in decision:
                        emit_progress(
                            "GYM_AGENT_CLARIFY",
                            mode=self.mode,
                            step=steps_taken,
                            clarify=decision.get("clarify", ""),
                        )
                        final_answer = decision.get("clarify", "")
                        break

                    planned_calls = [
                        normalize_tool_call(call)
                        for call in decision.get("tool_calls", [])
                    ]
                    if not planned_calls:
                        emit_progress(
                            "GYM_AGENT_EMPTY_ACTION",
                            mode=self.mode,
                            step=steps_taken,
                            raw_action=raw_decision,
                        )
                        final_answer = ""
                        break
                emit_progress(
                    "GYM_AGENT_ACTION",
                    mode=self.mode,
                    step=steps_taken,
                    tool_calls=preview_tool_calls(planned_calls),
                )

                internal_feedbacks: list[dict[str, Any]] = []
                if self.mode == "revision" and self.internal_thinking_max_iterations > 0:
                    target_is_tool_output = is_tool_output_target(self.world_model_target)
                    use_lookahead_rollouts = (
                        self.revision_lookahead_steps > 1
                        or self.revision_imagined_rollouts > 1
                    )
                    original_planned_calls = [dict(call) for call in planned_calls]
                    iter1_predicted_success: bool | None = None
                    iter1_error_message = ""
                    revision_loop_outcome = "no_iterations"
                    for iteration in range(1, self.internal_thinking_max_iterations + 1):
                        emit_progress(
                            "GYM_REVISION_ITERATION_START",
                            mode=self.mode,
                            step=steps_taken,
                            iteration=iteration,
                            planned_calls=preview_tool_calls(planned_calls),
                        )
                        revision_rollouts: list[dict[str, Any]] = []
                        if use_lookahead_rollouts:
                            revision_rollouts = imagine_revision_rollouts(
                                agent_generator=self.agent_generator,
                                world_model_generator=self.world_model_generator,
                                task=task,
                                conversation=conversation,
                                previous_state=current_state,
                                react_system_prompt=react_system_prompt,
                                initial_planned_calls=planned_calls,
                                lookahead_steps=self.revision_lookahead_steps,
                                num_rollouts=self.revision_imagined_rollouts,
                                world_model_target=self.world_model_target,
                                include_error_message_in_target=self.include_error_message_in_target,
                                include_stage_in_target=self.include_stage_in_target,
                                include_world_model_history=self.include_world_model_history,
                                start_interaction_index=steps_taken,
                                rollout_temperature=self.revision_rollout_temperature,
                                state_history=current_state_history,
                                input_history=current_input_history,
                                state_history_size=self.state_history_size,
                                system_prompt_max_chars=self.system_prompt_max_chars,
                                action_max_chars=self.action_max_chars,
                            )
                            feedbacks = (
                                revision_rollouts[0].get("first_step_feedbacks", [])
                                if revision_rollouts
                                else []
                            )
                            if not feedbacks:
                                feedbacks = predict_world_model_feedback(
                                    self.world_model_generator,
                                    task,
                                    current_state,
                                    planned_calls,
                                    interaction_index=steps_taken,
                                    world_model_target=self.world_model_target,
                                    include_error_message_in_target=self.include_error_message_in_target,
                                    include_stage_in_target=self.include_stage_in_target,
                                    include_world_model_history=self.include_world_model_history,
                                    state_history=current_state_history,
                                    input_history=current_input_history,
                                )
                        else:
                            feedbacks = predict_world_model_feedback(
                                self.world_model_generator,
                                task,
                                current_state,
                                planned_calls,
                                interaction_index=steps_taken,
                                world_model_target=self.world_model_target,
                                include_error_message_in_target=self.include_error_message_in_target,
                                include_stage_in_target=self.include_stage_in_target,
                                include_world_model_history=self.include_world_model_history,
                                state_history=current_state_history,
                                input_history=current_input_history,
                                system_prompt_max_chars=self.system_prompt_max_chars,
                                action_max_chars=self.action_max_chars,
                            )
                        internal_feedbacks = feedbacks
                        internal_iterations_used += 1
                        if iteration == 1:
                            iter1_predicted_success = bool(
                                feedbacks
                                and all(fb.get("predicted_success", False) for fb in feedbacks)
                            )
                            iter1_error_message = " | ".join(
                                msg
                                for msg in (
                                    (fb.get("predicted_error_message") or "").strip()
                                    for fb in feedbacks
                                    if not fb.get("predicted_success", True)
                                )
                                if msg
                            )
                        if use_lookahead_rollouts and revision_rollouts:
                            all_rollouts_success = all(
                                rollout.get("all_predicted_success")
                                for rollout in revision_rollouts
                            )
                        else:
                            all_rollouts_success = all(
                                fb.get("predicted_success", False) for fb in feedbacks
                            )
                        if not target_is_tool_output and all_rollouts_success:
                            revision_loop_outcome = "wm_predicted_success"
                            break
                        raw_revision = self.agent_generator.generate_from_messages(
                            build_internal_thinking_messages(
                                conversation,
                                planned_calls,
                                feedbacks,
                                iteration=iteration,
                                max_iterations=self.internal_thinking_max_iterations,
                                world_model_target=self.world_model_target,
                                revision_rollouts=revision_rollouts,
                            )
                        )
                        emit_progress(
                            "GYM_REVISION_RAW",
                            mode=self.mode,
                            step=steps_taken,
                            iteration=iteration,
                            feedbacks=feedbacks,
                            raw_revision=raw_revision,
                        )
                        try:
                            revised_decision = parse_agent_decision(raw_revision)
                        except Exception as exc:
                            logger.warning(f"unparseable revision: {exc}")
                            emit_progress(
                                "GYM_REVISION_PARSE_ERROR",
                                mode=self.mode,
                                step=steps_taken,
                                iteration=iteration,
                                error=str(exc),
                            )
                            revision_loop_outcome = "agent_unparseable_revision"
                            break
                        revised_calls = [
                            normalize_tool_call(call)
                            for call in revised_decision.get("tool_calls", [])
                        ]
                        if not revised_calls:
                            emit_progress(
                                "GYM_REVISION_EMPTY_ACTION",
                                mode=self.mode,
                                step=steps_taken,
                                iteration=iteration,
                            )
                            revision_loop_outcome = "agent_empty_revision"
                            break
                        emit_progress(
                            "GYM_REVISION_ACTION",
                            mode=self.mode,
                            step=steps_taken,
                            iteration=iteration,
                            revised_calls=preview_tool_calls(revised_calls),
                        )
                        if target_is_tool_output and tool_calls_equal(planned_calls, revised_calls):
                            planned_calls = revised_calls
                            revision_loop_outcome = "agent_kept_calls"
                            break
                        if tool_calls_equal(planned_calls, revised_calls):
                            revision_loop_outcome = "agent_kept_calls"
                            break
                        wm_revisions += 1
                        planned_calls = revised_calls
                        revision_loop_outcome = "iterations_exhausted"

                    wm_revision_step_details.append(
                        {
                            "step_index": steps_taken,
                            "original_calls": original_planned_calls,
                            "executed_calls": list(planned_calls),
                            "iter1_predicted_success": iter1_predicted_success,
                            "iter1_predicted_error_message": iter1_error_message,
                            "calls_changed": not tool_calls_equal(original_planned_calls, planned_calls),
                            "iterations": iteration,
                            "loop_outcome": revision_loop_outcome,
                            "revision_rollouts": revision_rollouts,
                        }
                    )

                planned_calls = [
                    {
                        **normalize_tool_call(call),
                        "id": str(call.get("id", f"call_{steps_taken}_{idx}")),
                    }
                    for idx, call in enumerate(planned_calls)
                ]
                conversation.append(
                    {
                        "role": "assistant",
                        "content": response_content or "",
                        "tool_calls": [
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": call.get("arguments", {}),
                                },
                            }
                            for call in planned_calls
                        ],
                    }
                )
                conversation_flow.append(
                    {
                        "type": "ai_message",
                        "content": response_content or "",
                        "usage_metadata": response_usage_metadata,
                        "response_metadata": response_metadata,
                        "tool_calls": [
                            {
                                "id": call["id"],
                                "name": call["name"],
                                "args": call.get("arguments", {}),
                            }
                            for call in planned_calls
                        ],
                    }
                )

                execution_results: list[dict[str, Any]] = []
                for call in planned_calls:
                    tool_name = call["name"]
                    tool_args = call.get("arguments", {})
                    emit_progress(
                        "GYM_TOOL_CALL_START",
                        mode=self.mode,
                        step=steps_taken,
                        tool_name=tool_name,
                        arguments=tool_args,
                        timeout_seconds=self.gym_tool_call_timeout_seconds,
                    )
                    try:
                        exec_result = await asyncio.wait_for(
                            self._execute_tool_call(tool_name, tool_args),
                            timeout=self.gym_tool_call_timeout_seconds,
                        )
                    except TimeoutError:
                        logger.error(
                            "tool execution timed out for %s after %.1fs",
                            tool_name,
                            self.gym_tool_call_timeout_seconds,
                        )
                        exec_result = {
                            "result": {
                                "success": False,
                                "error": (
                                    f"Timed out after "
                                    f"{self.gym_tool_call_timeout_seconds:.1f}s"
                                ),
                                "result": {},
                            },
                            "gym_server": None,
                        }
                        emit_progress(
                            "GYM_TOOL_CALL_TIMEOUT",
                            mode=self.mode,
                            step=steps_taken,
                            tool_name=tool_name,
                            arguments=tool_args,
                            timeout_seconds=self.gym_tool_call_timeout_seconds,
                        )
                    except Exception as exc:
                        logger.error(f"tool execution failed for {tool_name}: {exc}")
                        exec_result = {
                            "result": {"success": False, "error": str(exc), "result": {}},
                            "gym_server": None,
                        }
                    tool_result = exec_result.get("result", {})
                    target_gym = exec_result.get("gym_server")
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
                    raw_success = tool_result.get("success") if isinstance(tool_result, dict) else None
                    raw_payload = tool_result.get("result", tool_result) if isinstance(tool_result, dict) else tool_result
                    raw_is_error = (
                        isinstance(raw_payload, dict) and raw_payload.get("isError") is True
                    )
                    tool_success = bool(raw_success) and not raw_is_error if raw_success is not None else not raw_is_error
                    payload = raw_payload
                    payload_text = (
                        json.dumps(payload, ensure_ascii=False, default=str)
                        if not isinstance(payload, str)
                        else payload
                    )
                    conversation.append(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "tool_call_id": call["id"],
                            "content": payload_text,
                        }
                    )
                    conversation_flow.append(
                        {
                            "type": "tool_result",
                            "tool_name": tool_name,
                            "result": tool_result,
                            "gym_server": target_gym,
                        }
                    )
                    execution_results.append(
                        {
                            "requested_name": tool_name,
                            "resolved_name": tool_name,
                            "success": tool_success,
                            "content": payload_text,
                            "tool_call": call,
                            "raw_result": tool_result,
                        }
                    )
                    emit_progress(
                        "GYM_TOOL_RESULT",
                        mode=self.mode,
                        step=steps_taken,
                        tool_name=tool_name,
                        arguments=tool_args,
                        gym_server=target_gym,
                        result=payload_text,
                    )

                if internal_feedbacks:
                    current_state = update_state_from_actual_execution(
                        previous_state=current_state,
                        execution_results=execution_results,
                        predicted_feedbacks=internal_feedbacks,
                        trust_predicted_state=True,
                    )
                elif self.mode == "revision":
                    executed_feedbacks = predict_world_model_feedback(
                        self.world_model_generator,
                        task,
                        current_state,
                        planned_calls,
                        interaction_index=steps_taken,
                        world_model_target=self.world_model_target,
                        include_error_message_in_target=self.include_error_message_in_target,
                        include_stage_in_target=self.include_stage_in_target,
                        include_world_model_history=self.include_world_model_history,
                        state_history=current_state_history,
                        input_history=current_input_history,
                        system_prompt_max_chars=self.system_prompt_max_chars,
                        action_max_chars=self.action_max_chars,
                    )
                    current_state = update_state_from_actual_execution(
                        previous_state=current_state,
                        execution_results=execution_results,
                        predicted_feedbacks=executed_feedbacks,
                        trust_predicted_state=bool(executed_feedbacks),
                    )
                else:
                    current_state = update_state_from_actual_execution(
                        previous_state=current_state,
                        execution_results=execution_results,
                        predicted_feedbacks=None,
                        trust_predicted_state=False,
                    )

                current_state_history = append_state_history(
                    current_state_history, current_state, max_items=self.state_history_size
                )
                current_input_history = append_world_model_input_history(
                    current_input_history,
                    make_actual_world_model_history_entry_from_results(
                        step=steps_taken + 1,
                        tool_calls=planned_calls,
                        execution_results=execution_results,
                    ),
                )
                steps_taken += 1

            self._extra_metadata = {
                "world_model_mode": self.mode,
                "world_model_target": self.world_model_target,
                "include_stage_in_target": self.include_stage_in_target,
                "include_world_model_history": self.include_world_model_history,
                "steps_taken": steps_taken,
                "internal_thinking_iterations": internal_iterations_used,
                "wm_revisions_applied": wm_revisions,
                "wm_imagined_rollouts": wm_imagined_rollouts,
                "wm_revision_step_details": wm_revision_step_details,
                "imagined_rollout_records": imagined_rollout_records,
                "imagined_trajectory_rollouts": self.imagined_trajectory_rollouts,
                "imagined_rollout_temperature": self.imagined_rollout_temperature,
                "imagined_trajectory_selection_strategy": self.imagined_trajectory_selection_strategy,
                "imagined_trajectory_observation_source": self.imagined_trajectory_observation_source,
                "revision_lookahead_steps": self.revision_lookahead_steps,
                "revision_imagined_rollouts": self.revision_imagined_rollouts,
                "revision_rollout_temperature": self.revision_rollout_temperature,
                "state_history_size": self.state_history_size,
                "system_prompt_max_chars": self.system_prompt_max_chars,
                "action_max_chars": self.action_max_chars,
                "react_wm_k_steps": self.react_wm_k_steps,
                "react_wm_kmax": self.react_wm_kmax,
                "react_wm_foresight_temperature": self.react_wm_foresight_temperature,
                "react_wm_foresight_observation_source": self.react_wm_foresight_observation_source,
                "react_wm_records": self._react_wm_records,
            }
            emit_progress(
                "GYM_TASK_END",
                mode=self.mode,
                steps_taken=steps_taken,
                final_answer=final_answer,
                tools_used=tools_used,
                metadata=self._extra_metadata,
            )

            return {
                "final_response": final_answer,
                "conversation_flow": conversation_flow,
                "tools_used": tools_used,
                "tool_results": tool_results,
                "messages": conversation,
            }

    return WorldModelAssistedOrchestrator
