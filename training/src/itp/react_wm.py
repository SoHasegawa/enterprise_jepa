"""Per-turn world-model foresight for the ``react_wm`` family of replay modes.

The ``react_wm`` orchestrator does three things on every agent turn after
turn 0:

1. Build a fresh state from the rolling history.
2. Roll the local trained world model forward ``K`` steps to produce an
   imagined trajectory.
3. Inject the imagined trajectory into the chat history as a
   ``HumanMessage`` clearly labelled ``[World-model foresight] ... [End
   foresight]`` so the next LLM call (the action policy) can condition its
   tool decision on the foresight.

In ewm-enterprisearena the state is already maintained per turn by the
orchestrator (``current_state`` / ``current_state_history`` /
``current_input_history``), so we do not need a separate runtime state
builder. The world-model rollout reuses ewm's
:func:`src.evaluation.predict_world_model_feedback` so the foresight is
produced via exactly the same prompt the world model saw at training time
(``--world-model-target`` selects the state encoding: binary,
structured ``state``, or raw ``tool_output``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


REACT_WM_MODES: tuple[str, ...] = (
    "react_wm",
    "react_wm_decide_k",
    "react_wm_rl_k",
)


FORESIGHT_OPEN_TAG = "[World-model foresight]"
FORESIGHT_CLOSE_TAG = "[End foresight]"


@dataclass
class ReactWMConfig:
    """Per-task configuration for ``react_wm*`` modes.

    Only the ``mode`` and the chosen K source are required; everything else
    inherits from the orchestrator's existing world-model knobs.
    """

    mode: str
    fixed_k: int = 0
    kmax: int = 3
    k_controller: Any | None = None
    decide_k_via_agent_prompt: bool = False
    foresight_action_temperature: float = 0.0
    foresight_observation_source: str = "world_model"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in REACT_WM_MODES:
            raise ValueError(
                f"Unknown react_wm mode: {self.mode!r}. "
                f"Expected one of {REACT_WM_MODES}."
            )
        self.fixed_k = max(0, int(self.fixed_k))
        self.kmax = max(0, int(self.kmax))
        if self.mode == "react_wm" and self.fixed_k <= 0:
            # Allow fixed_k=0 (i.e. degenerate to react), but warn loudly so it
            # doesn't silently look like a config bug.
            logger.warning(
                "react_wm with fixed_k=0 will not inject any foresight; "
                "the run will be indistinguishable from baseline react."
            )
        if self.mode == "react_wm_rl_k" and self.k_controller is None:
            raise ValueError(
                "react_wm_rl_k requires a k_controller instance "
                "(see src.itp.k_controller.KController)."
            )
        if self.mode == "react_wm_decide_k":
            self.decide_k_via_agent_prompt = True


def build_foresight_user_message(foresight_text: str) -> dict[str, str]:
    """Wrap an imagined trajectory in the canonical foresight envelope.

    The label format ``[World-model foresight]`` / ``[End foresight]`` matches
    what the original ``react_wm`` orchestrator (in the source repo) injects
    before each non-first agent turn.
    """
    body = foresight_text.strip() or "(no foresight produced)"
    return {
        "role": "user",
        "content": f"{FORESIGHT_OPEN_TAG}\n{body}\n{FORESIGHT_CLOSE_TAG}",
    }


def _summarize_predicted_state(
    predicted_state: Any, world_model_target: str
) -> str:
    """Render the world model's predicted state into a foresight-step body."""
    from src.finetuning import (
        WORLD_MODEL_TARGET_STATE,
        WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
        WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY,
        WORLD_MODEL_TARGET_TOOL_OUTPUT,
        canonicalize_world_model_target,
    )

    target = canonicalize_world_model_target(world_model_target)
    if target in {
        WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
        WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY,
    } or target == WORLD_MODEL_TARGET_TOOL_OUTPUT:
        return json.dumps(predicted_state, ensure_ascii=False, default=str)
    if target == WORLD_MODEL_TARGET_STATE:
        return json.dumps(predicted_state, ensure_ascii=False, indent=2, default=str)
    return json.dumps(predicted_state, ensure_ascii=False, default=str)


def _summarize_feedback_for_foresight(
    feedback: dict[str, Any],
    world_model_target: str,
    *,
    char_budget: int = 800,
) -> str:
    """Render a single per-step WM feedback as compact foresight text."""
    from src.finetuning import (
        WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
        WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY,
        WORLD_MODEL_TARGET_TOOL_OUTPUT,
        canonicalize_world_model_target,
    )

    target = canonicalize_world_model_target(world_model_target)
    predicted_success = feedback.get("predicted_success")
    predicted_error = (feedback.get("predicted_error_message") or "").strip()
    predicted_tool_output = (feedback.get("predicted_tool_output") or "").strip()
    predicted_state = feedback.get("predicted_state")

    if target in {
        WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
        WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY,
    }:
        label = "success" if predicted_success else "failure"
        body = f"predicted={label}"
        if predicted_error:
            body += f", error={predicted_error}"
        if feedback.get("predicted_current_stage"):
            body += f", stage={feedback['predicted_current_stage']}"
    elif target == WORLD_MODEL_TARGET_TOOL_OUTPUT:
        if predicted_tool_output:
            body = f"tool_output={predicted_tool_output}"
        else:
            body = f"tool_output=(empty); predicted_success={predicted_success}"
    else:
        body = (
            f"predicted_success={predicted_success}"
            + (f"; error={predicted_error}" if predicted_error else "")
            + "; predicted_state="
            + (
                _summarize_predicted_state(predicted_state, world_model_target)
                if predicted_state is not None
                else "(none)"
            )
        )

    if char_budget > 0 and len(body) > char_budget:
        body = body[: char_budget - 3].rstrip() + "..."
    return body


def _render_foresight_lines_from_topk_steps(
    imagined_steps: list[dict[str, Any]],
    world_model_target: str,
    *,
    summary_char_budget: int = 800,
) -> list[str]:
    """Render a beam's imagined steps into the canonical foresight lines.

    Mirrors the per-step text produced by the greedy single-rollout path so the
    injected ``[World-model foresight] ... [End foresight]`` block has the same
    shape regardless of which search strategy generated it.
    """
    lines: list[str] = []
    for step in imagined_steps:
        offset = step.get("imagined_step")
        prefix = f"Step+{offset}" if offset is not None else "Step"
        if "final_answer" in step:
            lines.append(
                f"{prefix}: agent would emit a final_answer -- no further "
                f"tool calls imagined."
            )
            continue
        if "clarify" in step:
            lines.append(
                f"{prefix}: agent would emit a clarify -- no further tool "
                f"calls imagined."
            )
            continue
        if step.get("parse_error"):
            lines.append(
                f"{prefix}: (agent failed to produce a parseable imagined "
                f"action; stopping foresight rollout)"
            )
            continue
        if step.get("error") == "empty_tool_calls":
            lines.append(
                f"{prefix}: (agent emitted no tool calls; stopping foresight)"
            )
            continue
        planned_calls = step.get("tool_calls") or []
        action_summary = ", ".join(
            f"{call.get('name', '')}({json.dumps(call.get('arguments', {}), ensure_ascii=False, default=str)})"
            for call in planned_calls
        )
        if summary_char_budget > 0 and len(action_summary) > summary_char_budget:
            action_summary = action_summary[: summary_char_budget - 3].rstrip() + "..."
        feedbacks = step.get("predicted_feedback") or []
        foresight_body = _summarize_feedback_for_foresight(
            feedbacks[-1] if feedbacks else {},
            world_model_target,
            char_budget=summary_char_budget,
        )
        if step.get("repeated_tool_call_loop"):
            foresight_body += " (repeated tool call -- likely a loop)"
        lines.append(f"{prefix}: action={action_summary} | {foresight_body}")
    return lines


def _foresight_topk_search(
    *,
    agent_generator: Any,
    world_model_generator: Any,
    task: Any,
    conversation: list[dict[str, Any]],
    previous_state: dict[str, Any],
    state_history: list[dict[str, Any]],
    input_history: list[dict[str, Any]],
    react_system_prompt: str,
    k_steps: int,
    world_model_target: str,
    include_error_message_in_target: bool,
    include_stage_in_target: bool,
    include_world_model_history: bool,
    start_interaction_index: int,
    state_history_size: int,
    system_prompt_max_chars: int,
    action_max_chars: int,
    foresight_observation_source: str,
    summary_char_budget: int,
    candidate_action_count: int,
    top_k: int,
    rollout_temperature: float,
) -> tuple[str, list[dict[str, Any]]]:
    """Beam-search variant of the K-step foresight rollout.

    Depth is ``k_steps``; ``candidate_action_count`` actions are proposed per
    beam per step and pruned to the ``top_k`` best partial trajectories. The
    foresight text is rendered from the single best beam; ``rollout_records``
    contains that beam's per-step records (each carrying the ``topk_*`` scoring
    fields) for orchestrator metadata.
    """
    from src.evaluation import (
        _generate_with_optional_temperature,
        append_imagined_observation_for_planning,
        build_react_action_batch_messages,
        build_react_think_messages,
        full_world_model_state_for_agent,
        imagined_trajectory_system_prompt,
        parse_agent_candidate_decisions,
        predict_world_model_feedback,
        score_topk_imagined_step,
        summarize_state_for_planning,
    )
    from src.finetuning import (
        WORLD_MODEL_INPUT_HISTORY_SIZE,
        append_state_history,
        append_world_model_input_history,
        json_compact,
        make_imagined_world_model_history_entry,
        normalize_tool_call,
        parse_thought_payload,
        sanitize_state_content,
        state_is_finished,
        strip_model_thinking_output,
    )

    candidate_action_count = max(1, int(candidate_action_count))
    top_k = max(1, int(top_k))
    user_query = task.user_messages[-1] if getattr(task, "user_messages", None) else ""
    imagined_react_system_prompt = imagined_trajectory_system_prompt(react_system_prompt)
    root_state = sanitize_state_content(previous_state)
    beams: list[dict[str, Any]] = [
        {
            "branch_id": "0",
            "depth": 0,
            "conversation": [dict(message) for message in conversation],
            "state": root_state,
            "state_history": append_state_history(
                list(state_history or []), root_state, max_items=state_history_size
            ),
            "input_history": list(input_history or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:],
            "imagined_steps": [],
            "seen_signatures": {},
            "score": 0.0,
            "terminal": False,
            "rollout_temperature": 0.0,
        }
    ]

    for imagined_index in range(int(k_steps)):
        expanded: list[dict[str, Any]] = []
        for branch in beams:
            if branch.get("terminal"):
                expanded.append(branch)
                continue

            branch_conversation = [dict(message) for message in branch["conversation"]]
            raw_thought = _generate_with_optional_temperature(
                agent_generator,
                build_react_think_messages(
                    branch_conversation,
                    current_query=user_query,
                    system_prompt=imagined_react_system_prompt,
                ),
                temperature=0.0,
            )
            raw_thought = strip_model_thinking_output(raw_thought)
            thought_payload = parse_thought_payload(raw_thought)
            thought_conversation = branch_conversation + [
                {"role": "assistant", "content": json.dumps(thought_payload, ensure_ascii=False)}
            ]

            raw_action_batch = _generate_with_optional_temperature(
                agent_generator,
                build_react_action_batch_messages(
                    thought_conversation,
                    current_query=user_query,
                    system_prompt=imagined_react_system_prompt,
                    candidate_action_count=candidate_action_count,
                ),
                temperature=rollout_temperature if candidate_action_count > 1 else 0.0,
            )
            raw_action_batch = strip_model_thinking_output(raw_action_batch)
            try:
                candidate_decisions = parse_agent_candidate_decisions(
                    raw_action_batch, expected_count=candidate_action_count
                )
                batch_parse_error = None
            except Exception as exc:  # noqa: BLE001
                candidate_decisions = []
                batch_parse_error = str(exc)

            seen_candidate_signatures: set[str] = set()
            for candidate_index in range(candidate_action_count):
                temperature = rollout_temperature if candidate_action_count > 1 else 0.0
                if candidate_index < len(candidate_decisions):
                    decision, raw_action = candidate_decisions[candidate_index]
                    parse_error = None
                else:
                    decision = None
                    raw_action = raw_action_batch
                    parse_error = batch_parse_error or (
                        f"Batched action response returned {len(candidate_decisions)} "
                        f"candidate(s), expected {candidate_action_count}."
                    )
                child_id = f"{branch['branch_id']}.{candidate_index}"
                base_step = {
                    "imagined_step": imagined_index + 1,
                    "topk_branch_id": child_id,
                    "topk_parent_id": branch.get("branch_id"),
                    "topk_candidate_index": candidate_index,
                    "thought": thought_payload,
                    "raw_action": raw_action,
                }

                if decision is None:
                    step = {
                        **base_step,
                        "parse_error": parse_error,
                        "topk_score_delta": 0.0,
                        "topk_score_reasons": [f"parse_error=0:{parse_error}"],
                        "topk_total_score": branch["score"],
                    }
                    expanded.append(
                        {
                            **branch,
                            "branch_id": child_id,
                            "depth": imagined_index + 1,
                            "imagined_steps": branch["imagined_steps"] + [step],
                            "terminal": True,
                            "rollout_temperature": temperature,
                        }
                    )
                    continue

                if "final_answer" in decision:
                    step_score, score_reasons = score_topk_imagined_step(
                        previous_state=branch["state"],
                        predicted_state=branch["state"],
                        feedbacks=[],
                        repeated_tool_call_count=0,
                        final_answer=True,
                    )
                    step = {
                        **base_step,
                        "final_answer": decision["final_answer"],
                        "predicted_state": full_world_model_state_for_agent(branch["state"]),
                        "predicted_state_summary": summarize_state_for_planning(branch["state"]),
                        "topk_score_delta": step_score,
                        "topk_score_reasons": score_reasons,
                        "topk_total_score": branch["score"] + step_score,
                    }
                    expanded.append(
                        {
                            **branch,
                            "branch_id": child_id,
                            "depth": imagined_index + 1,
                            "imagined_steps": branch["imagined_steps"] + [step],
                            "score": branch["score"] + step_score,
                            "terminal": True,
                            "rollout_temperature": temperature,
                        }
                    )
                    continue

                if "clarify" in decision:
                    step = {
                        **base_step,
                        "clarify": decision["clarify"],
                        "predicted_state": full_world_model_state_for_agent(branch["state"]),
                        "predicted_state_summary": summarize_state_for_planning(branch["state"]),
                        "topk_score_delta": 0.0,
                        "topk_score_reasons": ["clarify=0"],
                        "topk_total_score": branch["score"],
                    }
                    expanded.append(
                        {
                            **branch,
                            "branch_id": child_id,
                            "depth": imagined_index + 1,
                            "imagined_steps": branch["imagined_steps"] + [step],
                            "terminal": True,
                            "rollout_temperature": temperature,
                        }
                    )
                    continue

                planned_calls = [
                    normalize_tool_call(call) for call in decision.get("tool_calls", [])
                ]
                if not planned_calls:
                    step = {
                        **base_step,
                        "error": "empty_tool_calls",
                        "topk_score_delta": 0.0,
                        "topk_score_reasons": ["empty_tool_calls=0"],
                        "topk_total_score": branch["score"],
                    }
                    expanded.append(
                        {
                            **branch,
                            "branch_id": child_id,
                            "depth": imagined_index + 1,
                            "imagined_steps": branch["imagined_steps"] + [step],
                            "terminal": True,
                            "rollout_temperature": temperature,
                        }
                    )
                    continue

                signature = json_compact(planned_calls)
                local_duplicate_candidate = signature in seen_candidate_signatures
                seen_candidate_signatures.add(signature)
                seen_signatures = dict(branch["seen_signatures"])
                repeated_tool_call_count = seen_signatures.get(signature, 0) + 1
                seen_signatures[signature] = repeated_tool_call_count
                if local_duplicate_candidate:
                    repeated_tool_call_count = max(repeated_tool_call_count, 2)

                if repeated_tool_call_count > 1:
                    step_score, score_reasons = score_topk_imagined_step(
                        previous_state=branch["state"],
                        predicted_state=branch["state"],
                        feedbacks=[],
                        repeated_tool_call_count=repeated_tool_call_count,
                    )
                    step = {
                        **base_step,
                        "tool_calls": planned_calls,
                        "repeated_tool_call_loop": True,
                        "repeated_tool_call_count": repeated_tool_call_count,
                        "predicted_feedback": [],
                        "predicted_state": full_world_model_state_for_agent(branch["state"]),
                        "predicted_state_summary": summarize_state_for_planning(branch["state"]),
                        "observation_source": foresight_observation_source,
                        "topk_score_delta": step_score,
                        "topk_score_reasons": score_reasons,
                        "topk_total_score": branch["score"] + step_score,
                    }
                    expanded.append(
                        {
                            **branch,
                            "branch_id": child_id,
                            "depth": imagined_index + 1,
                            "imagined_steps": branch["imagined_steps"] + [step],
                            "score": branch["score"] + step_score,
                            "terminal": True,
                            "rollout_temperature": temperature,
                        }
                    )
                    continue

                feedbacks: list[dict[str, Any]] = []
                predicted_state = branch["state"]
                if foresight_observation_source == "world_model":
                    feedbacks = predict_world_model_feedback(
                        world_model_generator,
                        task,
                        branch["state"],
                        planned_calls,
                        interaction_index=start_interaction_index + imagined_index,
                        world_model_target=world_model_target,
                        include_error_message_in_target=include_error_message_in_target,
                        include_stage_in_target=include_stage_in_target,
                        include_world_model_history=include_world_model_history,
                        state_history=branch["state_history"],
                        input_history=branch["input_history"],
                        system_prompt_max_chars=system_prompt_max_chars,
                        action_max_chars=action_max_chars,
                    )
                    predicted_state = (
                        feedbacks[-1].get("predicted_state") if feedbacks else branch["state"]
                    )

                step_score, score_reasons = score_topk_imagined_step(
                    previous_state=branch["state"],
                    predicted_state=predicted_state,
                    feedbacks=feedbacks,
                    repeated_tool_call_count=repeated_tool_call_count,
                )
                next_score = branch["score"] + step_score
                next_input_history = append_world_model_input_history(
                    branch["input_history"],
                    make_imagined_world_model_history_entry(
                        imagined_step=imagined_index + 1,
                        action=planned_calls,
                        state=predicted_state,
                    ),
                )
                next_conversation = [dict(message) for message in thought_conversation]
                append_imagined_observation_for_planning(
                    next_conversation,
                    planned_calls,
                    feedbacks,
                    predicted_state,
                    foresight_observation_source,
                )
                next_state_history = branch["state_history"]
                if predicted_state is not None:
                    next_state_history = append_state_history(
                        next_state_history, predicted_state, max_items=state_history_size
                    )
                terminal = state_is_finished(predicted_state)
                step = {
                    **base_step,
                    "tool_calls": planned_calls,
                    "repeated_tool_call_loop": False,
                    "repeated_tool_call_count": repeated_tool_call_count,
                    "predicted_feedback": feedbacks,
                    "predicted_state": full_world_model_state_for_agent(predicted_state),
                    "predicted_state_summary": (
                        summarize_state_for_planning(predicted_state)
                        if predicted_state
                        else None
                    ),
                    "observation_source": foresight_observation_source,
                    "topk_score_delta": step_score,
                    "topk_score_reasons": score_reasons,
                    "topk_total_score": next_score,
                }
                expanded.append(
                    {
                        "branch_id": child_id,
                        "depth": imagined_index + 1,
                        "conversation": next_conversation,
                        "state": predicted_state,
                        "state_history": next_state_history,
                        "input_history": next_input_history,
                        "imagined_steps": branch["imagined_steps"] + [step],
                        "seen_signatures": seen_signatures,
                        "score": next_score,
                        "terminal": terminal,
                        "rollout_temperature": temperature,
                    }
                )

        if not expanded:
            break
        expanded.sort(
            key=lambda item: (
                float(item.get("score", 0.0)),
                int(item.get("depth", 0)),
                0 if item.get("terminal") else 1,
            ),
            reverse=True,
        )
        beams = expanded[:top_k]
        if all(branch.get("terminal") for branch in beams):
            break

    if not beams:
        return "", []
    winner = beams[0]
    winning_steps = winner.get("imagined_steps") or []
    foresight_lines = _render_foresight_lines_from_topk_steps(
        winning_steps, world_model_target, summary_char_budget=summary_char_budget
    )
    return "\n".join(foresight_lines), winning_steps


def build_world_model_foresight(
    *,
    agent_generator: Any,
    world_model_generator: Any,
    task: Any,
    conversation: list[dict[str, Any]],
    previous_state: dict[str, Any],
    state_history: list[dict[str, Any]],
    input_history: list[dict[str, Any]],
    react_system_prompt: str,
    k_steps: int,
    world_model_target: str,
    include_error_message_in_target: bool,
    include_stage_in_target: bool,
    include_world_model_history: bool,
    start_interaction_index: int,
    state_history_size: int,
    system_prompt_max_chars: int,
    action_max_chars: int,
    foresight_action_temperature: float = 0.0,
    foresight_observation_source: str = "world_model",
    summary_char_budget: int = 800,
    selection_strategy: str = "greedy",
    candidate_action_count: int = 1,
    top_k: int = 1,
    rollout_temperature: float = 0.7,
) -> tuple[str, list[dict[str, Any]]]:
    """Roll the world model forward ``k_steps`` and return foresight text.

    The agent is used in a "think-then-act" loop to propose a candidate action
    at each imagined step (lightweight; we only need a plausible action so the
    world model has something to condition on). The world model is then called
    via :func:`src.evaluation.predict_world_model_feedback` to imagine the
    resulting state. The chain is unrolled ``k_steps`` times.

    When ``selection_strategy == "topk_search"`` the chain is instead unrolled
    as a beam search of depth ``k_steps``: at each imagined step the agent
    proposes ``candidate_action_count`` candidate actions, each is scored with
    :func:`src.evaluation.score_topk_imagined_step`, and the ``top_k``
    highest-scoring partial trajectories are retained for the next step. The
    foresight text is rendered from the single best beam. This mirrors the
    ``imagined`` mode's :func:`src.evaluation.imagine_trajectory_topk_search`
    but uses the controller-/config-chosen ``k_steps`` as the search depth.
    Any other ``selection_strategy`` keeps the original single-rollout
    ("greedy") behaviour.

    Returns ``(foresight_text, rollout_records)`` where:

    * ``foresight_text`` is what should be wrapped in
      ``[World-model foresight] ... [End foresight]`` and injected into the
      planning conversation before the next agent turn.
    * ``rollout_records`` is a serialisable list of per-step dicts for
      orchestrator metadata (mirrors ``imagined_rollout_records`` shape).
    """
    if k_steps <= 0:
        return "", []

    if str(selection_strategy) == "topk_search":
        return _foresight_topk_search(
            agent_generator=agent_generator,
            world_model_generator=world_model_generator,
            task=task,
            conversation=conversation,
            previous_state=previous_state,
            state_history=state_history,
            input_history=input_history,
            react_system_prompt=react_system_prompt,
            k_steps=k_steps,
            world_model_target=world_model_target,
            include_error_message_in_target=include_error_message_in_target,
            include_stage_in_target=include_stage_in_target,
            include_world_model_history=include_world_model_history,
            start_interaction_index=start_interaction_index,
            state_history_size=state_history_size,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
            foresight_observation_source=foresight_observation_source,
            summary_char_budget=summary_char_budget,
            candidate_action_count=candidate_action_count,
            top_k=top_k,
            rollout_temperature=rollout_temperature,
        )

    # Imports are local to avoid a hard cycle at package import time.
    from src.evaluation import (
        _generate_with_optional_temperature,
        build_react_action_messages,
        build_react_think_messages,
        full_world_model_state_for_agent,
        imagined_trajectory_system_prompt,
        parse_agent_decision,
        predict_world_model_feedback,
        summarize_state_for_planning,
    )
    from src.finetuning import (
        WORLD_MODEL_INPUT_HISTORY_SIZE,
        append_state_history,
        append_world_model_input_history,
        make_imagined_world_model_history_entry,
        normalize_tool_call,
        parse_thought_payload,
        sanitize_state_content,
        state_is_finished,
        strip_model_thinking_output,
    )

    imagined_conversation = [dict(message) for message in conversation]
    imagined_react_system_prompt = imagined_trajectory_system_prompt(react_system_prompt)
    imagined_state = sanitize_state_content(previous_state)
    imagined_state_history = append_state_history(
        list(state_history or []), imagined_state, max_items=state_history_size
    )
    imagined_input_history = list(input_history or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:]
    foresight_lines: list[str] = []
    rollout_records: list[dict[str, Any]] = []

    user_query = task.user_messages[-1] if getattr(task, "user_messages", None) else ""

    for step_offset in range(k_steps):
        # 1) propose a thought for the imagined state
        raw_thought = _generate_with_optional_temperature(
            agent_generator,
            build_react_think_messages(
                imagined_conversation,
                current_query=user_query,
                system_prompt=imagined_react_system_prompt,
            ),
            temperature=foresight_action_temperature,
        )
        raw_thought = strip_model_thinking_output(raw_thought)
        thought_payload = parse_thought_payload(raw_thought)
        imagined_conversation.append(
            {"role": "assistant", "content": json.dumps(thought_payload, ensure_ascii=False)}
        )

        # 2) propose tool calls
        raw_action = _generate_with_optional_temperature(
            agent_generator,
            build_react_action_messages(
                imagined_conversation,
                current_query=user_query,
                system_prompt=imagined_react_system_prompt,
            ),
            temperature=foresight_action_temperature,
        )
        raw_action = strip_model_thinking_output(raw_action)
        try:
            decision = parse_agent_decision(raw_action)
        except Exception as exc:  # noqa: BLE001
            logger.debug("react_wm foresight: unparseable imagined action: %s", exc)
            rollout_records.append(
                {
                    "step_offset": step_offset + 1,
                    "thought": thought_payload,
                    "raw_action": raw_action,
                    "parse_error": str(exc),
                }
            )
            foresight_lines.append(
                f"Step+{step_offset + 1}: (agent failed to produce a parseable imagined action; "
                f"stopping foresight rollout)"
            )
            break

        if "final_answer" in decision or "clarify" in decision:
            stop_reason = "final_answer" if "final_answer" in decision else "clarify"
            rollout_records.append(
                {
                    "step_offset": step_offset + 1,
                    "thought": thought_payload,
                    stop_reason: decision.get(stop_reason, ""),
                    "predicted_state": full_world_model_state_for_agent(imagined_state),
                    "predicted_state_summary": summarize_state_for_planning(imagined_state),
                }
            )
            foresight_lines.append(
                f"Step+{step_offset + 1}: agent would emit a {stop_reason} -- no further "
                f"tool calls imagined."
            )
            break

        planned_calls = [
            normalize_tool_call(call) for call in decision.get("tool_calls", [])
        ]
        if not planned_calls:
            foresight_lines.append(
                f"Step+{step_offset + 1}: (agent emitted no tool calls; stopping foresight)"
            )
            rollout_records.append(
                {
                    "step_offset": step_offset + 1,
                    "thought": thought_payload,
                    "raw_action": raw_action,
                    "error": "empty_tool_calls",
                }
            )
            break

        # 3) ask the world model to imagine the resulting state
        feedbacks: list[dict[str, Any]] = []
        predicted_state = None
        if foresight_observation_source == "world_model":
            feedbacks = predict_world_model_feedback(
                world_model_generator,
                task,
                imagined_state,
                planned_calls,
                interaction_index=start_interaction_index + step_offset,
                world_model_target=world_model_target,
                include_error_message_in_target=include_error_message_in_target,
                include_stage_in_target=include_stage_in_target,
                include_world_model_history=include_world_model_history,
                state_history=imagined_state_history,
                input_history=imagined_input_history,
                system_prompt_max_chars=system_prompt_max_chars,
                action_max_chars=action_max_chars,
            )
            if feedbacks:
                predicted_state = feedbacks[-1].get("predicted_state")
                imagined_input_history = append_world_model_input_history(
                    imagined_input_history,
                    make_imagined_world_model_history_entry(
                        imagined_step=step_offset + 1,
                        action=planned_calls,
                        state=predicted_state,
                    ),
                )

        action_summary = ", ".join(
            f"{call.get('name', '')}({json.dumps(call.get('arguments', {}), ensure_ascii=False, default=str)})"
            for call in planned_calls
        )
        if len(action_summary) > summary_char_budget:
            action_summary = action_summary[: summary_char_budget - 3].rstrip() + "..."
        foresight_body = _summarize_feedback_for_foresight(
            feedbacks[-1] if feedbacks else {},
            world_model_target,
            char_budget=summary_char_budget,
        )
        foresight_lines.append(
            f"Step+{step_offset + 1}: action={action_summary} | {foresight_body}"
        )
        rollout_records.append(
            {
                "step_offset": step_offset + 1,
                "thought": thought_payload,
                "tool_calls": planned_calls,
                "predicted_feedback": feedbacks,
                "predicted_state": full_world_model_state_for_agent(predicted_state),
                "predicted_state_summary": (
                    summarize_state_for_planning(predicted_state)
                    if predicted_state
                    else None
                ),
                "observation_source": foresight_observation_source,
            }
        )

        # 4) advance the imagined state for the next foresight step
        imagined_conversation.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"foresight_{step_offset}_{idx}",
                        "type": "function",
                        "function": {
                            "name": call.get("name", ""),
                            "arguments": call.get("arguments", {}),
                        },
                    }
                    for idx, call in enumerate(planned_calls)
                ],
            }
        )
        if predicted_state is not None:
            imagined_state = predicted_state
            imagined_state_history = append_state_history(
                imagined_state_history, imagined_state, max_items=state_history_size
            )
            if state_is_finished(predicted_state):
                foresight_lines.append(
                    f"(Imagined trajectory predicted task completion at step "
                    f"+{step_offset + 1}; stopping foresight.)"
                )
                break

    return "\n".join(foresight_lines), rollout_records


__all__ = [
    "FORESIGHT_OPEN_TAG",
    "FORESIGHT_CLOSE_TAG",
    "REACT_WM_MODES",
    "ReactWMConfig",
    "build_foresight_user_message",
    "build_world_model_foresight",
]
