"""World-model-assisted orchestrator for EnterpriseOps-Gym replay.

Plugs our agent and world-model generators into the gym's BenchmarkExecutor /
VerifierEngine pipeline so held-out gym tasks can be scored with the same
verifiers (`database_state`, `response_checker`) that `evaluate.py` uses.

The orchestrator subclasses the gym's `AgentOrchestrator` so it can route tool
calls through `_execute_tool_call`, but it does not use `self.llm_client` —
agent decisions come from our `agent_generator`, and the world-model loop
(revision / imagined-rollout) uses `world_model_generator`.

Modes selectable via `mode`:
- "baseline": agent generator only, no world-model interposition.
- "latent_guided": sample candidate actions and choose the one with lowest JEPA latent goal cost.
- "revision": world-model predicts outcome and the agent revises tool calls.
- "imagined": world-model rolls out an imagined trajectory before committing.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


logger = logging.getLogger(__name__)


def build_react_tool_descriptions_from_gym(available_tools: list[dict[str, Any]]) -> str:
    """Render gym MCP tool dicts into the descriptive block our react prompt expects.

    Mirrors the shape produced by `finetuning.build_react_tool_descriptions` for
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


# latent_guided planning: when True, sample an independent candidate-action pool
# per horizon step (each still proposed open-loop from the current state) instead
# of slicing one flat pool into sliding-window plans. Toggle to False to A/B
# against the legacy flat-pool behavior.
LATENT_PLAN_PER_STEP_POOLS_DEFAULT = True

_IMAGINED_PLAN_STATE_FIELDS = (
    "execution_status", "progress_signal", "side_effect_type", "error_signature",
    "information_sufficiency", "recommended_abstract_action", "missing_information_type",
)


def render_imagined_plan_message(
    remaining_plan: list[dict[str, Any]],
    *,
    terminal_advice: bool = False,
    terminal_advice_threshold: float = 0.75,
    state_fields: tuple[str, ...] = _IMAGINED_PLAN_STATE_FIELDS,
) -> dict[str, str]:
    """Render an alternating (predicted action -> predicted resulting state) trajectory as a
    single advisory user-turn message, shared by beam_plan and hier_latent_cem so both modes'
    injected guidance is visually identical to the agent regardless of which planner produced
    it. `calls` entries must already be flat {"name","arguments"} dicts (normalize_tool_call)."""
    plan_lines = [
        "World-model imagined rollout for the next steps (predicted action -> predicted resulting "
        "state). Use it as guidance, not ground truth:"
    ]
    finish_steps: list[tuple[int, float]] = []
    for plan_i, plan_entry in enumerate(remaining_plan, 1):
        calls_text = ", ".join(
            f"{c.get('name')}({json.dumps(c.get('arguments', {}), ensure_ascii=False)[:100]})"
            for c in (plan_entry.get("calls") or [])
        ) or "(no-op)"
        plan_lines.append(f"  step {plan_i} action: {calls_text}")
        predicted_state = plan_entry.get("predicted_state") or {}
        state_text = ", ".join(
            f"{field}={predicted_state[field]}"
            for field in state_fields
            if predicted_state.get(field) not in (None, "", [], ["none"], "none", "unknown")
        )
        terminal_prob = plan_entry.get("terminal_probability")
        if isinstance(terminal_prob, (int, float)):
            if state_text:
                state_text = f"{state_text}, terminal_probability={terminal_prob:.2f}"
            else:
                state_text = f"terminal_probability={terminal_prob:.2f}"
            if terminal_advice and terminal_prob >= terminal_advice_threshold:
                finish_steps.append((plan_i, float(terminal_prob)))
        if state_text:
            plan_lines.append(f"  step {plan_i} predicted state: {state_text}")
    if finish_steps:
        step_i, prob = max(finish_steps, key=lambda item: item[1])
        plan_lines.append(
            f"Terminal advisory: the world model predicts the task may be ready to finish after "
            f"step {step_i} (P(done)={prob:.2f}). If the actual state confirms every requirement "
            "is satisfied, stop calling tools and provide the final answer."
        )
    plan_lines.append(
        "Guidance: these predicted states are the world model's guesses, NOT confirmed results. "
        "Do NOT give a final answer or give up prematurely. A predicted 'finalize' only means the "
        "task MAY be near done -- first verify from the actual state that every requirement is met. "
        "If you are blocked, uncertain, or a step failed, examine the current state and try a "
        "different action or approach before concluding."
    )
    return {"role": "user", "content": "\n".join(plan_lines)}


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
            latent_plan_samples: int = 10,
            latent_plan_elites: int = 3,
            latent_plan_iters: int = 3,
            latent_plan_horizon: int = 5,
            latent_mpc_execute_steps: int = 1,
            latent_plan_temperature: float = 0.7,
            latent_plan_score_margin: float = 0.0,
            latent_plan_diversity_multiplier: int = 1,
            latent_plan_hard_override: bool = False,
            latent_plan_goal_mode: str = "final",
            gate_flat_score_ratio: float = 1.0,
            imagined_rollout_mode: str = "closed_loop",
            beam_plan_trigger: str = "interval",
            beam_plan_critic_failure_prob: float = 0.3,
            beam_plan_critic_stall_prob: float = 0.7,
            beam_plan_critic_min_score: float | None = None,
            beam_plan_critic_max_quiet_steps: int = 0,
            beam_plan_terminal_advice: bool = False,
            beam_plan_terminal_advice_threshold: float = 0.75,
            hier_cem_anchors: int = 8,
            hier_cem_samples: int = 256,
            hier_cem_elites: int = 16,
            hier_cem_iters: int = 3,
            hier_cem_horizon: int = 5,
            hier_cem_init_std: float = 0.2,
            hier_cem_min_std: float = 0.02,
            hier_cem_smoothing: float = 1.0,
            hier_cem_min_elite_agreement: float = 0.5,
            hier_cem_decode_strategy: str = "nearest_anchor",
            hier_cem_decode_max_new_tokens: int = 96,
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
            initial_canonical_observation: dict[str, Any] | None = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            if mode not in {"baseline", "latent_guided", "revision", "imagined", "beam_plan", "hier_latent_cem"}:
                raise ValueError(f"Unknown orchestrator mode: {mode}")
            self.agent_generator = agent_generator
            self.world_model_generator = world_model_generator
            self.mode = mode
            self.world_model_target = world_model_target
            self.include_error_message_in_target = include_error_message_in_target
            self.include_stage_in_target = include_stage_in_target
            self.include_world_model_history = include_world_model_history
            self.internal_thinking_max_iterations = internal_thinking_max_iterations
            self.imagined_trajectory_max_steps = imagined_trajectory_max_steps
            self.imagined_trajectory_rollouts = imagined_trajectory_rollouts
            self.imagined_rollout_temperature = imagined_rollout_temperature
            self.imagined_trajectory_selection_strategy = imagined_trajectory_selection_strategy
            self.imagined_trajectory_observation_source = imagined_trajectory_observation_source
            self.latent_plan_samples = max(1, int(latent_plan_samples))
            self.latent_plan_elites = max(1, int(latent_plan_elites))
            self.latent_plan_iters = max(1, int(latent_plan_iters))
            self.latent_plan_horizon = max(1, int(latent_plan_horizon))
            self.latent_mpc_execute_steps = max(1, int(latent_mpc_execute_steps))
            self.latent_plan_temperature = float(latent_plan_temperature)
            self.latent_plan_score_margin = max(0.0, float(latent_plan_score_margin))
            self.latent_plan_diversity_multiplier = max(1, int(latent_plan_diversity_multiplier))
            self.latent_plan_hard_override = bool(latent_plan_hard_override)
            if latent_plan_goal_mode not in {"final", "next_subgoal"}:
                raise ValueError(f"Unknown latent_plan_goal_mode: {latent_plan_goal_mode}")
            self.latent_plan_goal_mode = latent_plan_goal_mode
            self.gate_flat_score_ratio = max(0.0, float(gate_flat_score_ratio))
            if imagined_rollout_mode not in {"closed_loop", "open_loop"}:
                raise ValueError(f"Unknown imagined_rollout_mode: {imagined_rollout_mode}")
            # open_loop: beam_plan proposes every candidate plan in ONE agent call and scores
            # them in ONE batched world-model pass (see _beam_plan_open_loop); the `imagined`
            # mode reads the same setting from src.finetuning's module global.
            self.imagined_rollout_mode = imagined_rollout_mode
            if beam_plan_trigger not in {"interval", "critic"}:
                raise ValueError(f"Unknown beam_plan_trigger: {beam_plan_trigger}")
            # `interval`: re-plan every latent_mpc_execute_steps steps (historical).
            # `critic`: score the agent's own action first (one world-model forward) and only
            # plan when that says the action is bad -- see _beam_plan_critic.
            self.beam_plan_trigger = beam_plan_trigger
            self.beam_plan_critic_failure_prob = float(beam_plan_critic_failure_prob)
            self.beam_plan_critic_stall_prob = float(beam_plan_critic_stall_prob)
            self.beam_plan_critic_min_score = (
                None if beam_plan_critic_min_score is None else float(beam_plan_critic_min_score)
            )
            self.beam_plan_critic_max_quiet_steps = max(0, int(beam_plan_critic_max_quiet_steps))
            self.beam_plan_terminal_advice = bool(beam_plan_terminal_advice)
            self.beam_plan_terminal_advice_threshold = min(
                1.0, max(0.0, float(beam_plan_terminal_advice_threshold))
            )
            self.hier_cem_anchors = max(1, int(hier_cem_anchors))
            self.hier_cem_samples = max(1, int(hier_cem_samples))
            self.hier_cem_elites = max(1, int(hier_cem_elites))
            self.hier_cem_iters = max(1, int(hier_cem_iters))
            self.hier_cem_horizon = max(1, int(hier_cem_horizon))
            self.hier_cem_init_std = float(hier_cem_init_std)
            self.hier_cem_min_std = float(hier_cem_min_std)
            self.hier_cem_smoothing = float(hier_cem_smoothing)
            self.hier_cem_min_elite_agreement = float(hier_cem_min_elite_agreement)
            if hier_cem_decode_strategy not in {"nearest_anchor", "learned_decoder"}:
                raise ValueError(f"Unknown hier_cem_decode_strategy: {hier_cem_decode_strategy}")
            self.hier_cem_decode_strategy = hier_cem_decode_strategy
            self.hier_cem_decode_max_new_tokens = max(1, int(hier_cem_decode_max_new_tokens))
            self.imagined_trajectory_candidate_actions = max(1, int(imagined_trajectory_candidate_actions))
            self.imagined_trajectory_top_k = max(1, int(imagined_trajectory_top_k))
            self.revision_lookahead_steps = revision_lookahead_steps
            self.revision_imagined_rollouts = revision_imagined_rollouts
            self.revision_rollout_temperature = revision_rollout_temperature
            self.final_answer_f1_threshold = final_answer_f1_threshold
            self.agent_max_observation_chars = agent_max_observation_chars
            self.agent_replay_history_budget_chars = agent_replay_history_budget_chars
            self.gym_tool_call_timeout_seconds = max(
                1.0, float(gym_tool_call_timeout_seconds)
            )
            self.state_history_size = max(0, int(state_history_size))
            self.system_prompt_max_chars = max(0, int(system_prompt_max_chars))
            self.action_max_chars = max(0, int(action_max_chars))
            self.initial_state = initial_state
            self.initial_canonical_observation = initial_canonical_observation
            self._extra_metadata: dict[str, Any] = {}
            self._agent_call_count = 0          # LLM/agent generations this task (for beam_plan benchmarking)
            self._beam_plan_records: list[dict[str, Any]] = []
            self._beam_recent_tool_names: list[tuple] = []  # executed tool-name sets per step (anti-repetition)
            self._beam_imagined_plan: list[dict[str, Any]] = []  # cached imagined trajectory (MPC)
            self._beam_plan_cursor: int = 0                       # steps executed since the last re-plan attempt
            self._beam_has_planned: bool = False   # forces a first re-plan attempt; after that the cooldown alone governs
            self._beam_quiet_steps: int = 0         # consecutive non-firing critic checks
            self._beam_critic_checks: int = 0       # critic evaluations this task (denominator of the fire rate)
            self._beam_critic_fires: int = 0        # critic evaluations that escalated to planning
            self._beam_pending_terminal_advice: dict[str, Any] | None = None
            self._beam_terminal_advice_count: int = 0
            self._hier_cem_records: list[dict[str, Any]] = []
            self._hier_imagined_plan: list[dict[str, Any]] = []   # cached imagined trajectory (MPC)
            self._hier_plan_cursor: int = 0                       # steps executed since the last re-plan attempt
            self._hier_has_planned: bool = False   # forces a first re-plan attempt; after that the cooldown alone governs

        def _set_beam_terminal_advisory(
            self,
            *,
            probability: Any,
            source: str,
            step_index: int,
        ) -> bool:
            if not self.beam_plan_terminal_advice:
                return False
            if not isinstance(probability, (int, float)):
                return False
            prob = float(probability)
            if prob < self.beam_plan_terminal_advice_threshold:
                return False
            self._beam_pending_terminal_advice = {
                "source": source,
                "step_index": step_index,
                "terminal_probability": prob,
                "threshold": self.beam_plan_terminal_advice_threshold,
            }
            self._beam_terminal_advice_count += 1
            return True

        def _beam_plan_critic(
            self,
            *,
            seed_step: dict[str, Any] | None,
            system_prompt_text: str,
            user_query: str,
            imagined_history: list[dict[str, Any]],
            score_cfg: Any,
        ) -> dict[str, Any]:
            """Should we spend a planning cycle on THIS step? One world-model forward, no LLM.

            The interval trigger re-plans every --latent-mpc-execute-steps steps whether or not
            anything is wrong, so the planning cost (an agent call plus the rollout) is paid on
            every step in amortized terms. This asks the cheaper question first: score the action
            the agent ALREADY produced -- that generation is sunk cost -- as a one-step plan, and
            only escalate to full planning when the prediction says the action is bad. Escalation
            then does the expensive thing properly: revise the current action AND look several
            steps ahead. Amortized cost becomes `critic + p * planning`, so the fire rate p is
            the lever, and it is logged per step (fired or not) so it can be measured and the
            thresholds calibrated from a real run.

            Fires when the world model vetoes the action (catastrophic tail: the score config's
            P(failure)/P(deleted) limits), when P(execution_status=failure) reaches
            --beam-plan-critic-failure-prob, when the action looks like it will not advance the
            task (1 - P(progress_signal=positive) reaches --beam-plan-critic-stall-prob), or when
            the step score falls below --beam-plan-critic-min-score.
            """
            from src.finetuning import emit_progress

            if not seed_step:
                return {"fires": True, "reason": "no_seed_action", "world_model_calls": 0}
            try:
                scored = self.world_model_generator.score_action_plans_canonical_event(
                    system_prompt=system_prompt_text, user_prompt=str(user_query),
                    input_history=imagined_history, action_plans=[[seed_step]],
                    score_config=score_cfg,
                )
                record = scored[0] if scored else {}
                probs = (record.get("per_step_field_probs") or [{}])[0] or {}
                per_step = record.get("per_step") or []
                score = float(per_step[0].get("score", 0.0)) if per_step else float(record.get("score", 0.0))
                failure_prob = float(probs.get("execution_status", {}).get("failure", 0.0))
                positive_prob = float(probs.get("progress_signal", {}).get("positive", 0.0))
                stall_prob = 1.0 - positive_prob
                terminal_probs = record.get("per_step_terminal_prob") or []
                terminal_prob = terminal_probs[0] if terminal_probs else record.get("terminal_probability")
                terminal_prob = float(terminal_prob) if isinstance(terminal_prob, (int, float)) else None
                vetoed = bool(record.get("vetoed"))
                reasons: list[str] = []
                if vetoed:
                    flat = "; ".join(
                        ", ".join(entry.get("reasons", [])) for entry in (record.get("veto_reasons") or [])
                    )
                    reasons.append(f"veto({flat})" if flat else "veto")
                if failure_prob >= self.beam_plan_critic_failure_prob:
                    reasons.append(f"P(failure)={failure_prob:.2f}>={self.beam_plan_critic_failure_prob:.2f}")
                if stall_prob >= self.beam_plan_critic_stall_prob:
                    reasons.append(f"P(no_progress)={stall_prob:.2f}>={self.beam_plan_critic_stall_prob:.2f}")
                if self.beam_plan_critic_min_score is not None and score < self.beam_plan_critic_min_score:
                    reasons.append(f"score={score:.3f}<{self.beam_plan_critic_min_score:.3f}")
                return {
                    "fires": bool(reasons),
                    "reason": " | ".join(reasons) if reasons else "action_looks_good",
                    "score": score,
                    "vetoed": vetoed,
                    "failure_prob": failure_prob,
                    "stall_prob": stall_prob,
                    "terminal_probability": terminal_prob,
                    "terminal_advice": (
                        terminal_prob is not None
                        and self.beam_plan_terminal_advice
                        and terminal_prob >= self.beam_plan_terminal_advice_threshold
                    ),
                    "predicted_state": record.get("predicted_state"),
                    "world_model_calls": 1,
                }
            except Exception as exc:                                  # noqa: BLE001
                # A critic malfunction must not silently disable planning: escalate instead.
                emit_progress("GYM_BEAM_PLAN_CRITIC_ERROR", mode=self.mode, error=str(exc))
                return {"fires": True, "reason": f"critic_error:{type(exc).__name__}", "world_model_calls": 0}

        def _beam_plan_open_loop(
            self,
            *,
            beam_cfg: Any,
            score_cfg: Any,
            seed_step: dict[str, Any] | None,
            system_prompt_text: str,
            user_query: str,
            imagined_history: list[dict[str, Any]],
            tool_names: list[str],
            state_text: str,
            wrap_step: Any,
            step_tool_names: Any,
        ) -> dict[str, Any]:
            """Open-loop beam planning: ONE agent call for all m plans, ONE batched world-model
            pass for every step of every plan.

            Closed-loop beam planning spends `horizon` agent calls and `horizon` world-model
            calls per re-plan (one per depth, each conditioned on the previously chosen step).
            Here the agent emits m COMPLETE n-step plans up front -- actions never see a
            predicted state, so nothing serializes -- and score_action_plans_canonical_event
            rolls all m plans forward in one batched latent rollout, chaining each plan's own
            states within the plan. Cost: 1 + 1 calls instead of 2*horizon.

            Returns the same bookkeeping the closed-loop depth loop produces, so the
            confidence gate, override logic, injection and records downstream are unchanged:
              imagined_plan, per_depth_scores, per_depth_normalized, first_num_candidates,
              first_best, beam_calls, first_recommend_calls_candidate,
              first_recommend_reason_candidate.
            """
            # These live in src.finetuning and are imported locally in execute() too -- this
            # method is called from there but must not rely on that scope.
            from src.beam_plan import build_single_plan_prompt, parse_single_plan
            from src.finetuning import emit_progress, normalize_tool_call, sample_many

            empty = {
                "imagined_plan": [], "per_depth_scores": [], "per_depth_normalized": [],
                "first_num_candidates": 0, "first_best": {}, "beam_calls": 1,
                "first_recommend_calls_candidate": None,
                "first_recommend_reason_candidate": "no_candidates",
            }

            # k samples of a ONE-plan prompt rather than one response listing k plans: same
            # candidate set, but the samples share one prefill and decode concurrently (n=k on
            # vLLM/OpenAI, num_return_sequences on local HF) instead of emitting every plan on
            # a single serial decode stream. sample_many reports how many requests that took.
            prompt = build_single_plan_prompt(
                system_prompt_text, state_text, beam_cfg, tool_names=tool_names or None
            )
            raw_samples, agent_requests = sample_many(
                self.agent_generator,
                [
                    {"role": "system", "content": "You output ONLY a JSON array of tool-call steps. No prose, no markdown."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.latent_plan_temperature,
                num_samples=beam_cfg.num_candidates,
            )
            self._agent_call_count += agent_requests
            empty["beam_calls"] = agent_requests
            candidate_plans: list[list[dict[str, Any]]] = []
            seen_plan_signatures: set[str] = set()
            unparsed = 0
            for raw in raw_samples:
                raw_text = raw if isinstance(raw, str) else str(getattr(raw, "content", raw) or "")
                parsed = parse_single_plan(raw_text, beam_cfg)
                if not parsed:
                    unparsed += 1
                    continue
                steps = [wrapped for wrapped in (wrap_step(step) for step in parsed) if wrapped]
                if not steps:
                    unparsed += 1
                    continue
                # Sampling repeats itself; a duplicate plan would only re-spend world-model
                # compute on a candidate already in the set.
                signature = json.dumps(steps, ensure_ascii=False, sort_keys=True, default=str)
                if signature in seen_plan_signatures:
                    continue
                seen_plan_signatures.add(signature)
                candidate_plans.append(steps)
            if not candidate_plans:
                emit_progress(
                    "GYM_BEAM_PLAN_NO_CANDIDATES", mode=self.mode, depth=0, open_loop=True,
                    samples=len(raw_samples), agent_requests=agent_requests,
                    raw_response_head=(raw_samples[0][:400] if raw_samples else ""),
                )
                return empty

            # The seed (the agent's own action) is scored as a one-step plan at index 0, exactly
            # as depth 0 does in closed-loop mode, so the override margin gate below compares
            # like with like. It is never selected as the trajectory to inject: a one-step plan
            # cannot reach the horizon, and "keep the agent's action" is expressed by declining
            # to override, not by injecting a one-step plan.
            seed_offset = 0
            if seed_step:
                candidate_plans = [[seed_step]] + candidate_plans
                seed_offset = 1
            scored = self.world_model_generator.score_action_plans_canonical_event(
                system_prompt=system_prompt_text, user_prompt=str(user_query),
                input_history=imagined_history, action_plans=candidate_plans, score_config=score_cfg,
            )
            by_index = {int(record.get("plan_index", record.get("index", -1))): record for record in scored}
            seed_record = by_index.get(0) if seed_offset else None

            # Same anti-repetition penalty as closed-loop, applied to each plan's FIRST step
            # (the only step that could be executed now).
            repeat_penalty_scale = 0.75 * max(1, int(self.latent_plan_diversity_multiplier))
            path_names = list(self._beam_recent_tool_names)

            def adjusted(record: dict[str, Any]) -> float:
                plan = record.get("plan") or []
                names = step_tool_names(plan[0]) if plan else tuple()
                return float(record.get("score", 0.0)) - repeat_penalty_scale * path_names.count(names)

            def first_step_adjusted(record: dict[str, Any]) -> float:
                """Depth-0 score, penalised -- comparable across plans of different lengths,
                unlike the discounted-sum trajectory score."""
                per_step = record.get("per_step") or []
                plan = record.get("plan") or []
                names = step_tool_names(plan[0]) if plan else tuple()
                head = float(per_step[0].get("score", 0.0)) if per_step else 0.0
                return head - repeat_penalty_scale * path_names.count(names)

            generated = [
                record for record in scored
                if int(record.get("plan_index", -1)) >= seed_offset
                and not record.get("vetoed", False)
                and (record.get("plan") or [])
            ]
            if not generated:
                emit_progress(
                    "GYM_BEAM_PLAN_VETOED", mode=self.mode, depth=0, open_loop=True,
                    num_candidates=len(candidate_plans), agent_requests=agent_requests,
                )
                return empty
            winner = max(generated, key=adjusted)

            winner_plan = winner.get("plan") or []
            per_step = winner.get("per_step") or []
            per_step_states = winner.get("per_step_predicted_state") or []
            per_step_terminal = winner.get("per_step_terminal_prob") or []
            imagined_plan: list[dict[str, Any]] = []
            per_depth_scores: list[float] = []
            for depth, step in enumerate(winner_plan):
                calls = [
                    normalize_tool_call(call)
                    for call in (step.get("tool_calls", []) if isinstance(step, dict) else [])
                ]
                calls = [call for call in calls if str(call.get("name", "")).strip()]
                step_score = float(per_step[depth].get("score", 0.0)) if depth < len(per_step) else 0.0
                per_depth_scores.append(step_score)
                imagined_plan.append({
                    "calls": calls,
                    "score": step_score,
                    "reason": winner.get("reason"),
                    "predicted_state": (
                        per_step_states[depth] if depth < len(per_step_states) else winner.get("predicted_state")
                    ),
                    "terminal_probability": (
                        float(per_step_terminal[depth]) if depth < len(per_step_terminal) else None
                    ),
                })

            # One softmax over the candidate PLANS replaces closed-loop's per-depth softmaxes.
            # The downstream gate averages per_depth_normalized, so repeating the winner's
            # trajectory-level normalized score per depth keeps that average on the same scale
            # (probability mass among ~m candidates) that gate_flat_score_ratio is tuned for.
            winner_normalized = float(winner.get("normalized_score") or 0.0)
            per_depth_normalized = [winner_normalized] * max(1, len(imagined_plan))

            # Override margin, decided on depth-0 scores exactly as closed-loop does.
            first_calls = list(imagined_plan[0]["calls"]) if imagined_plan else []
            if not seed_record:
                recommend_calls = first_calls or None
                recommend_reason = "beam_best_beats_seed" if first_calls else "no_candidates"
            else:
                margin = max(0.0, float(self.latent_plan_score_margin))
                if first_step_adjusted(winner) - first_step_adjusted(seed_record) > margin:
                    recommend_calls = first_calls or None
                    recommend_reason = "beam_best_beats_seed"
                else:
                    recommend_calls = None
                    recommend_reason = "kept_baseline_below_margin"

            emit_progress(
                "GYM_BEAM_PLAN_OPEN_LOOP", mode=self.mode,
                num_candidate_plans=len(candidate_plans), plan_lengths=[len(p) for p in candidate_plans],
                samples_requested=beam_cfg.num_candidates, agent_requests=agent_requests,
                duplicate_samples=len(raw_samples) - len(candidate_plans) - unparsed,
                unparsed_samples=unparsed, world_model_calls=1,
                winner_plan_index=winner.get("plan_index"), winner_score=winner.get("score"),
                winner_normalized_score=winner_normalized, winner_plan_len=len(winner_plan),
                recommend_reason=recommend_reason,
            )
            return {
                "imagined_plan": imagined_plan,
                "per_depth_scores": per_depth_scores,
                "per_depth_normalized": per_depth_normalized,
                "first_num_candidates": len(candidate_plans),
                "first_best": winner,
                "beam_calls": agent_requests,
                "first_recommend_calls_candidate": recommend_calls,
                "first_recommend_reason_candidate": recommend_reason,
            }

        def get_result_metadata(self) -> dict[str, Any]:
            return dict(self._extra_metadata)

        async def execute(self) -> dict[str, Any]:
            from src.finetuning import (
                TaskTrajectory,
                build_imagined_trajectory_prompt_message,
                build_internal_thinking_messages,
                build_react_action_messages,
                build_react_system_prompt,
                configure_replay_limits,
                emit_progress,
                imagine_revision_rollouts,
                imagine_trajectory_candidates,
                is_tool_output_target,
                append_state_history,
                append_world_model_input_history,
                make_actual_world_model_history_entry_from_results,
                make_blank_state_like,
                normalize_tool_call,
                parse_agent_decision,
                parse_jsonish,
                preview_tool_calls,
                predict_world_model_feedback,
                summarize_state_for_planning,
                strip_code_fence,
                tool_calls_equal,
                update_state_from_actual_execution,
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
            latent_plan_records: list[dict[str, Any]] = []
            emit_progress(
                "GYM_TASK_START",
                mode=self.mode,
                query=user_query,
            )
            use_native_tool_calling = hasattr(self.agent_generator, "invoke_with_tools")

            def text_list(value: Any) -> list[str]:
                if isinstance(value, list):
                    return [str(item) for item in value if str(item).strip()]
                if isinstance(value, tuple):
                    return [str(item) for item in value if str(item).strip()]
                if isinstance(value, str) and value.strip():
                    return [value.strip()]
                return []

            stage_plan_observation = (
                dict(self.initial_canonical_observation)
                if isinstance(self.initial_canonical_observation, dict)
                else None
            )
            stage_plan_stages = (
                stage_plan_observation.get("stages")
                if isinstance(stage_plan_observation, dict)
                and isinstance(stage_plan_observation.get("stages"), dict)
                else {}
            )
            stage_plan_remaining = text_list(stage_plan_stages.get("remaining_stages"))
            stage_plan_completed = text_list(stage_plan_stages.get("completed_stages"))
            stage_plan_current = str(stage_plan_stages.get("current_stage") or "").strip()
            if stage_plan_current and stage_plan_current.lower() not in {"finished", "unknown"}:
                if stage_plan_current not in stage_plan_remaining:
                    stage_plan_remaining = [stage_plan_current] + stage_plan_remaining
            if stage_plan_observation and stage_plan_remaining:
                stage_plan_observation = dict(stage_plan_observation)
                stage_plan_observation["stages"] = {
                    "current_stage": stage_plan_remaining[0],
                    "remaining_stages": stage_plan_remaining,
                    "completed_stages": stage_plan_completed,
                }

            def rewrite_stage_plan_observation(completed_count: int) -> dict[str, Any] | None:
                if not stage_plan_observation or not stage_plan_remaining:
                    return None
                completed_count = max(0, min(int(completed_count), len(stage_plan_remaining)))
                completed = list(stage_plan_completed)
                for stage in stage_plan_remaining[:completed_count]:
                    if stage not in completed:
                        completed.append(stage)
                remaining = stage_plan_remaining[completed_count:]
                current = remaining[0] if remaining else "finished"
                payload = dict(stage_plan_observation)
                payload["tool_outcome"] = {
                    "success": True,
                    "label": 1,
                    "error_message": "",
                    "summary": (
                        "Stage-plan target observation. "
                        f"Completed {completed_count} planned stage(s); current stage is {current}."
                    ),
                }
                payload["stages"] = {
                    "current_stage": current,
                    "remaining_stages": remaining,
                    "completed_stages": completed,
                }
                evidence = text_list(payload.get("evidence"))
                payload["evidence"] = evidence + [
                    f"Completed stage count: {completed_count}",
                    f"Current target stage: {current}",
                ]
                return payload

            final_goal_observation = rewrite_stage_plan_observation(len(stage_plan_remaining))

            def build_next_subgoal(step_index: int) -> dict[str, Any] | None:
                return rewrite_stage_plan_observation(step_index + 1)

            def build_subgoal_query(subgoal: dict[str, Any] | None) -> str:
                if not subgoal:
                    return user_query
                stages = subgoal.get("stages") if isinstance(subgoal.get("stages"), dict) else {}
                completed = text_list(stages.get("completed_stages"))
                remaining = text_list(stages.get("remaining_stages"))
                current_completed = completed[-1] if completed else "the current stage"
                next_stage = str(stages.get("current_stage") or "finished")
                return (
                    f"{user_query}\n\n"
                    "For this next action, optimize only the immediate subgoal before continuing the full task.\n"
                    f"Immediate subgoal: complete stage `{current_completed}` and advance to `{next_stage}`.\n"
                    f"Remaining stages after this subgoal: {json.dumps(remaining, ensure_ascii=False)}.\n"
                    "Choose the next tool action that best completes this immediate subgoal."
                )

            while steps_taken < self.max_iterations:
                active_subgoal = (
                    build_next_subgoal(steps_taken)
                    if self.mode == "latent_guided" and self.latent_plan_goal_mode == "next_subgoal"
                    else None
                )
                active_goal_observation = (
                    active_subgoal
                    if active_subgoal
                    else final_goal_observation
                    if self.mode == "latent_guided" and self.latent_plan_goal_mode == "final"
                    else None
                )
                active_goal_text = (
                    json.dumps(active_goal_observation, ensure_ascii=False, sort_keys=True)
                    if active_goal_observation
                    else None
                )
                action_query = build_subgoal_query(active_subgoal)
                planning_conversation = conversation
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

                if self.mode == "beam_plan" and self._beam_pending_terminal_advice:
                    advice = self._beam_pending_terminal_advice
                    planning_conversation = planning_conversation + [{
                        "role": "user",
                        "content": (
                            "Terminal advisory: the world model predicted that the previous step may "
                            f"have completed the task (P(done)={advice['terminal_probability']:.2f}, "
                            f"threshold={advice['threshold']:.2f}). If the actual state confirms every "
                            "requirement is satisfied, stop calling tools and provide the final answer."
                        ),
                    }]
                    self._beam_pending_terminal_advice = None

                if self.mode == "beam_plan" and self._beam_imagined_plan and self._beam_plan_cursor < len(self._beam_imagined_plan):
                    # Make the world-model imagined ROLLOUT visible to the agent -- an alternating
                    # (predicted action -> predicted resulting state) sequence -- so it can follow it
                    # without re-invoking the world model every step (re-plan on the MPC cadence).
                    remaining_plan = self._beam_imagined_plan[self._beam_plan_cursor:]
                    planning_conversation = planning_conversation + [
                        render_imagined_plan_message(
                            remaining_plan,
                            terminal_advice=self.beam_plan_terminal_advice,
                            terminal_advice_threshold=self.beam_plan_terminal_advice_threshold,
                            state_fields=(
                                "execution_status", "progress_signal", "side_effect_type",
                                "error_signature", "information_sufficiency",
                            ),
                        )
                    ]

                if self.mode == "hier_latent_cem" and self._hier_imagined_plan and self._hier_plan_cursor < len(self._hier_imagined_plan):
                    # Same advisory injection as beam_plan, but the trajectory came from the
                    # hierarchical latent-action CEM (src/hierarchical_action_sampling.py) instead
                    # of a discrete LLM-proposed candidate pool.
                    remaining_plan = self._hier_imagined_plan[self._hier_plan_cursor:]
                    planning_conversation = planning_conversation + [render_imagined_plan_message(remaining_plan)]

                if active_subgoal and use_native_tool_calling:
                    planning_conversation = planning_conversation + [
                        {"role": "user", "content": action_query}
                    ]

                response_content = ""
                response_usage_metadata: dict[str, Any] = {}
                response_metadata: dict[str, Any] = {}
                planned_calls: list[dict[str, Any]] = []
                if use_native_tool_calling:
                    self._agent_call_count += 1
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
                    self._agent_call_count += 1
                    raw_decision = self.agent_generator.generate_from_messages(
                        build_react_action_messages(
                            planning_conversation,
                            current_query=action_query,
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
                should_run_latent_guidance = (
                    self.mode == "latent_guided"
                    and hasattr(self.world_model_generator, "score_action_plans")
                    and (self.latent_plan_goal_mode != "next_subgoal" or active_subgoal is not None)
                )
                if self.mode == "latent_guided" and self.latent_plan_goal_mode == "next_subgoal" and active_subgoal is None:
                    latent_plan_records.append(
                        {
                            "step_index": steps_taken,
                            "goal_mode": self.latent_plan_goal_mode,
                            "active_subgoal": None,
                            "override_applied": False,
                            "override_reason": "missing_active_subgoal",
                            "objective": "skip_latent_guidance_without_subgoal",
                        }
                    )
                    emit_progress(
                        "GYM_LATENT_PLAN_SKIPPED",
                        mode=self.mode,
                        step=steps_taken,
                        reason="missing_active_subgoal",
                        goal_mode=self.latent_plan_goal_mode,
                    )
                if should_run_latent_guidance:
                    samples = max(1, int(self.latent_plan_samples))
                    horizon = max(1, int(self.latent_plan_horizon))
                    diversity_multiplier = max(1, int(self.latent_plan_diversity_multiplier))
                    score_margin = max(0.0, float(self.latent_plan_score_margin))
                    per_step_pools = getattr(
                        self, "latent_plan_per_step_pools", LATENT_PLAN_PER_STEP_POOLS_DEFAULT
                    )
                    def action_key(action: dict[str, Any]) -> str:
                        return json.dumps(action, ensure_ascii=False, sort_keys=True, default=str)

                    def action_tool_names(action: dict[str, Any]) -> list[str]:
                        names: list[str] = []
                        for call in action.get("tool_calls", []) or []:
                            if isinstance(call, dict):
                                name = str(call.get("name", "")).strip()
                                if name:
                                    names.append(name)
                        return names

                    def pool_summary(pool: list[dict[str, Any]]) -> dict[str, Any]:
                        keys = [action_key(action) for action in pool]
                        tool_names = sorted(
                            {name for action in pool for name in action_tool_names(action)}
                        )
                        return {
                            "pool_size": len(pool),
                            "unique_actions": len(set(keys)),
                            "unique_tool_names": tool_names,
                            "unique_tool_count": len(tool_names),
                            "unique_action_ratio": (len(set(keys)) / len(keys)) if keys else 0.0,
                        }

                    def valid_normalized_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
                        normalized_calls: list[dict[str, Any]] = []
                        for call in calls:
                            if not call:
                                continue
                            normalized = normalize_tool_call(call)
                            name = str(normalized.get("name", "")).strip()
                            if not name:
                                continue
                            normalized["name"] = name
                            normalized_calls.append(normalized)
                        return normalized_calls

                    seed_action = {"tool_calls": valid_normalized_calls(planned_calls)}
                    seed_tool_names = action_tool_names(seed_action)
                    seed_action_key = ""
                    if seed_action["tool_calls"]:
                        seed_action_key = json.dumps(
                            seed_action,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        )

                    def action_from_candidate(candidate: Any) -> dict[str, Any] | None:
                        try:
                            raw_decision = (
                                candidate
                                if isinstance(candidate, str)
                                else json.dumps(candidate, ensure_ascii=False)
                            )
                            candidate_decision = parse_agent_decision(raw_decision)
                            normalized_calls = valid_normalized_calls(
                                candidate_decision.get("tool_calls", [])
                            )
                            return {"tool_calls": normalized_calls} if normalized_calls else None
                        except Exception:
                            return None

                    def action_has_different_tool_name(action: dict[str, Any]) -> bool:
                        names = action_tool_names(action)
                        return bool(names) and names != seed_tool_names

                    def action_has_same_tool_name_different_args(action: dict[str, Any]) -> bool:
                        names = action_tool_names(action)
                        return bool(names) and names == seed_tool_names and action_key(action) != seed_action_key

                    def request_actions(count: int, diversity_mode: str) -> list[dict[str, Any]]:
                        count = max(1, int(count))
                        try:
                            messages = build_react_action_messages(
                                planning_conversation,
                                current_query=action_query,
                                system_prompt=react_system_prompt,
                            )
                            messages = [dict(message) for message in messages]
                            if diversity_mode == "different_tool_name":
                                diversity_instruction = (
                                    "Use tool names different from the current planned action when valid. "
                                    f"Current planned tool-name sequence: {json.dumps(seed_tool_names, ensure_ascii=False)}. "
                                    "At least one tool name in each candidate should differ from that sequence."
                                )
                            elif diversity_mode == "same_tool_name_different_args":
                                diversity_instruction = (
                                    "Use the same tool-name sequence as the current planned action, but change the arguments materially. "
                                    f"Required tool-name sequence: {json.dumps(seed_tool_names, ensure_ascii=False)}. "
                                    "Do not copy the current action's arguments."
                                )
                            else:
                                diversity_instruction = (
                                    "Use different plausible tools or materially different arguments when useful."
                                )
                            messages[-1]["content"] = (
                                f"{messages[-1]['content']}\n\n"
                                f"Generate {count} candidate next actions for the same state. "
                                "Each candidate must be a valid action object using either "
                                "{\"action\": \"<tool_name>\", \"action_input\": {...}} or "
                                "{\"tool_calls\": [{\"name\": \"<tool_name>\", \"arguments\": {...}}]}. "
                                f"{diversity_instruction} "
                                "Do not include final answers, explanations, duplicate candidates, or empty tool names. "
                                "Return JSON only as {\"candidates\": [<candidate_action>, ...]}."
                            )
                            raw_candidate = self.agent_generator.generate_from_messages(
                                messages,
                                temperature=self.latent_plan_temperature,
                            )
                            raw_candidate = raw_candidate.split("</think>\n", 1)[-1].strip()
                            parsed = parse_jsonish(strip_code_fence(raw_candidate))
                            if isinstance(parsed, dict) and "candidates" in parsed:
                                candidates = parsed.get("candidates") or []
                            elif isinstance(parsed, list):
                                candidates = parsed
                            else:
                                candidates = [parsed]
                            actions: list[dict[str, Any]] = []
                            for candidate in candidates:
                                action = action_from_candidate(candidate)
                                if action is not None:
                                    actions.append(action)
                            return actions
                        except Exception as exc:
                            logger.debug(
                                "latent-guided %s action sampling failed: %s",
                                diversity_mode,
                                exc,
                            )
                            return []

                    def sample_actions(count: int) -> list[dict[str, Any]]:
                        count = max(1, int(count))
                        if not seed_tool_names:
                            return request_actions(count, "generic")
                        different_target = count // 2
                        same_target = count - different_target
                        actions: list[dict[str, Any]] = []
                        seen: set[str] = set()

                        def add_matching(
                            candidates: list[dict[str, Any]],
                            predicate: Any,
                            limit: int,
                        ) -> int:
                            added = 0
                            for action in candidates:
                                if added >= limit:
                                    break
                                if not predicate(action):
                                    continue
                                key = action_key(action)
                                if key in seen:
                                    continue
                                seen.add(key)
                                actions.append(action)
                                added += 1
                            return added

                        different_added = 0
                        if different_target:
                            different_added += add_matching(
                                request_actions(different_target, "different_tool_name"),
                                action_has_different_tool_name,
                                different_target,
                            )
                        same_added = 0
                        if same_target:
                            same_added += add_matching(
                                request_actions(same_target, "same_tool_name_different_args"),
                                action_has_same_tool_name_different_args,
                                same_target,
                            )
                        if different_added < different_target:
                            different_added += add_matching(
                                request_actions(different_target - different_added, "different_tool_name"),
                                action_has_different_tool_name,
                                different_target - different_added,
                            )
                        if same_added < same_target:
                            same_added += add_matching(
                                request_actions(same_target - same_added, "same_tool_name_different_args"),
                                action_has_same_tool_name_different_args,
                                same_target - same_added,
                            )
                        if len(actions) < count:
                            fallback = request_actions(count - len(actions), "generic")
                            for action in fallback:
                                key = action_key(action)
                                if key in seen:
                                    continue
                                seen.add(key)
                                actions.append(action)
                                if len(actions) >= count:
                                    break
                        return actions

                    def gather_pool(
                        target: int, seed_calls: list[dict[str, Any]] | None = None
                    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                        """Collect a distinct candidate next-action pool and diagnostics."""
                        target = max(1, int(target))
                        seen: set[str] = set()
                        pool: list[dict[str, Any]] = []
                        diagnostics: dict[str, Any] = {
                            "target": target,
                            "sampling_rounds": 0,
                            "requested_candidates": 0,
                            "sampled_candidates": 0,
                            "duplicate_candidates": 0,
                            "accepted_different_tool_name": 0,
                            "accepted_same_tool_name_different_args": 0,
                            "accepted_other": 0,
                            "seed_included": bool(seed_calls),
                        }

                        def add(calls: list[dict[str, Any]]) -> None:
                            if len(pool) >= target:
                                return
                            normalized_calls = valid_normalized_calls(calls)
                            if not normalized_calls:
                                return
                            action = {"tool_calls": normalized_calls}
                            key = action_key(action)
                            if key in seen:
                                diagnostics["duplicate_candidates"] += 1
                                return
                            seen.add(key)
                            pool.append(action)
                            if action_has_different_tool_name(action):
                                diagnostics["accepted_different_tool_name"] += 1
                            elif action_has_same_tool_name_different_args(action):
                                diagnostics["accepted_same_tool_name_different_args"] += 1
                            else:
                                diagnostics["accepted_other"] += 1

                        if seed_calls:
                            add(seed_calls)
                        while len(pool) < target:
                            remaining = target - len(pool)
                            request_count = max(remaining, remaining * diversity_multiplier)
                            diagnostics["sampling_rounds"] += 1
                            diagnostics["requested_candidates"] += request_count
                            sampled_actions = sample_actions(request_count)
                            diagnostics["sampled_candidates"] += len(sampled_actions)
                            if not sampled_actions:
                                break
                            previous_count = len(pool)
                            for sampled in sampled_actions:
                                add(sampled.get("tool_calls", []))
                            if len(pool) == previous_count:
                                break
                        diagnostics.update(pool_summary(pool))
                        return pool, diagnostics

                    pool_diagnostics: list[dict[str, Any]] = []
                    if per_step_pools:
                        step_pools = []
                        for step in range(horizon):
                            pool, diagnostics = gather_pool(
                                samples, seed_calls=planned_calls if step == 0 else None
                            )
                            diagnostics["step"] = step
                            pool_diagnostics.append(diagnostics)
                            step_pools.append(pool)
                        last_non_empty: list[dict[str, Any]] = []
                        for step in range(horizon):
                            if step_pools[step]:
                                last_non_empty = step_pools[step]
                            else:
                                step_pools[step] = last_non_empty
                        action_pool = [action for pool in step_pools for action in pool]
                    else:
                        action_pool, diagnostics = gather_pool(samples * horizon, seed_calls=planned_calls)
                        diagnostics["step"] = "flat"
                        pool_diagnostics.append(diagnostics)
                        step_pools = [action_pool for _ in range(horizon)]

                    if action_pool:
                        def pool_action(step: int, index: int) -> dict[str, Any]:
                            pool = step_pools[step] or action_pool
                            return pool[index % len(pool)]

                        def make_plan(offset: int = 0) -> list[dict[str, Any]]:
                            return [pool_action(step, offset + step) for step in range(horizon)]

                        seed_plan = make_plan(0)
                        candidate_plans = [seed_plan] + [make_plan(i) for i in range(1, samples)]
                        cem_iterations: list[dict[str, Any]] = []
                        elite_count = min(self.latent_plan_elites, len(candidate_plans))
                        scored_plans: list[dict[str, Any]] = []
                        for cem_iter in range(self.latent_plan_iters):
                            scored_plans = self.world_model_generator.score_action_plans(
                                system_prompt=self.config.system_prompt or "",
                                user_prompt=user_query,
                                input_history=current_input_history,
                                action_plans=candidate_plans,
                                goal_text_override=active_goal_text,
                            )
                            elites = scored_plans[:elite_count]
                            elite_scores = [elite.get("score") for elite in elites]
                            cem_iterations.append(
                                {
                                    "iteration": cem_iter + 1,
                                    "best_score": elites[0].get("score") if elites else None,
                                    "elite_scores": elite_scores,
                                    "elite_score_spread": (
                                        max(elite_scores) - min(elite_scores)
                                        if elite_scores
                                        and all(isinstance(score, (int, float)) for score in elite_scores)
                                        else None
                                    ),
                                }
                            )
                            if cem_iter == self.latent_plan_iters - 1:
                                break
                            next_plans = [list(elite.get("plan", [])) for elite in elites]
                            cursor = 0
                            mutation_samples = sample_actions(
                                max(0, samples - len(next_plans)) * diversity_multiplier
                            )
                            while len(next_plans) < samples and elites:
                                base = list(elites[cursor % len(elites)].get("plan", []))
                                if base:
                                    mutate_at = cursor % len(base)
                                    sampled = mutation_samples.pop(0) if mutation_samples else None
                                    if sampled is None:
                                        sampled = pool_action(mutate_at, cursor)
                                    if sampled is not None:
                                        base[mutate_at] = sampled
                                next_plans.append(base or make_plan(cursor))
                                cursor += 1
                            if seed_plan and next_plans:
                                next_plans[0] = seed_plan
                            candidate_plans = next_plans[: samples]

                        if scored_plans:
                            best_plan = list(scored_plans[0].get("plan", []))
                            execute_count = min(self.latent_mpc_execute_steps, len(best_plan))
                            selected_calls: list[dict[str, Any]] = []
                            for action in best_plan[:execute_count]:
                                selected_calls.extend(action.get("tool_calls", []))
                            selected_calls = [normalize_tool_call(call) for call in selected_calls]

                            baseline_score = None
                            baseline_rank = None
                            for rank, scored_plan in enumerate(scored_plans, start=1):
                                plan = scored_plan.get("plan") or []
                                if plan and action_key(plan[0]) == seed_action_key:
                                    baseline_score = scored_plan.get("score")
                                    baseline_rank = rank
                                    break
                            selected_score = scored_plans[0].get("score")
                            score_improvement = None
                            if isinstance(baseline_score, (int, float)) and isinstance(selected_score, (int, float)):
                                score_improvement = float(baseline_score) - float(selected_score)
                            should_override = bool(selected_calls)
                            override_reason = "selected"
                            if score_improvement is not None and score_improvement <= score_margin:
                                should_override = False
                                override_reason = "score_margin_not_met"
                            elif not selected_calls:
                                should_override = False
                                override_reason = "empty_selected_calls"
                            if should_override:
                                planned_calls = selected_calls

                            top_scores = [
                                plan.get("score")
                                for plan in scored_plans[: min(5, len(scored_plans))]
                                if isinstance(plan.get("score"), (int, float))
                            ]
                            selected_action = {"tool_calls": selected_calls}
                            latent_plan_record = {
                                "step_index": steps_taken,
                                "num_candidates": len(candidate_plans),
                                "horizon": self.latent_plan_horizon,
                                "mpc_execute_steps": execute_count,
                                "iterations": cem_iterations,
                                "selected_score": selected_score,
                                "baseline_seed_score": baseline_score,
                                "baseline_seed_rank": baseline_rank,
                                "score_improvement_over_seed": score_improvement,
                                "score_margin_required": score_margin,
                                "goal_mode": self.latent_plan_goal_mode,
                                "active_subgoal": active_subgoal,
                                "override_applied": should_override,
                                "override_reason": override_reason,
                                "selected_differs_from_seed": (
                                    action_key(selected_action) != seed_action_key if selected_calls else False
                                ),
                                "top_score_spread": max(top_scores) - min(top_scores) if top_scores else None,
                                "top_scores_all_equal": len(set(top_scores)) == 1 if top_scores else None,
                                "candidate_pool_diagnostics": pool_diagnostics,
                                "objective": (
                                    "minimize_next_subgoal_latent_mse"
                                    if active_subgoal
                                    else "minimize_terminal_latent_goal_mse"
                                ),
                                "ranked_plans": scored_plans[: min(5, len(scored_plans))],
                            }
                            latent_plan_records.append(latent_plan_record)
                            emit_progress(
                                "GYM_LATENT_PLAN_SELECTED",
                                mode=self.mode,
                                step=steps_taken,
                                num_candidates=len(candidate_plans),
                                horizon=self.latent_plan_horizon,
                                iterations=self.latent_plan_iters,
                                selected_score=selected_score,
                                baseline_seed_score=baseline_score,
                                score_improvement=score_improvement,
                                score_margin=score_margin,
                                override_applied=should_override,
                                goal_mode=self.latent_plan_goal_mode,
                                active_subgoal=active_subgoal,
                                selected_calls=preview_tool_calls(planned_calls),
                            )

                if self.mode == "beam_plan" and planned_calls:
                    # Beam-search MPC. On a RE-PLAN step, one LLM call per horizon step proposes m
                    # candidate next actions; the world model scores them (canonical-event heads,
                    # LLM-free), the best per step is kept, forming an imagined trajectory that is
                    # cached and shown to the agent. For the next --latent-mpc-execute-steps steps the
                    # agent follows that visible plan WITHOUT re-invoking the world model. Any failure
                    # falls back to the baseline action so planning never crashes the run.
                    try:
                        from src.beam_plan import BeamPlanConfig, build_step_candidates_prompt, parse_action_candidates
                        from src.canonical_event_scoring import CanonicalEventScoreConfig

                        beam_cfg = BeamPlanConfig(
                            num_candidates=self.latent_plan_samples,
                            horizon=self.latent_plan_horizon,
                            top_k=self.imagined_trajectory_top_k,
                            gate_flat_score_ratio=self.gate_flat_score_ratio,
                        )

                        def _wrap_step(step: dict[str, Any]) -> dict[str, Any] | None:
                            # Accept several LLM shapes: {"tool_calls":[...]}, {"name","arguments"},
                            # {"function":{"name","arguments"}}, {"tool"/"tool_name": ...}. Normalize
                            # all to the training-action format {"tool_calls":[{function:{name,args}}]}.
                            if not isinstance(step, dict):
                                return None
                            if isinstance(step.get("tool_calls"), list) and step["tool_calls"]:
                                calls = []
                                for call in step["tool_calls"]:
                                    norm = normalize_tool_call(call)
                                    name = str(norm.get("name", "")).strip()
                                    if name:
                                        calls.append({"type": "function", "function": {"name": name, "arguments": norm.get("arguments", {})}})
                                return {"tool_calls": calls} if calls else None
                            function = step.get("function") if isinstance(step.get("function"), dict) else None
                            source = function or step
                            name = str(source.get("name") or step.get("tool") or step.get("tool_name") or "").strip()
                            if not name:
                                return None
                            arguments = source.get("arguments") or step.get("args") or step.get("arguments") or {}
                            return {"tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}

                        tool_names = []
                        for tool in (self.available_tools or []):
                            tname = tool.get("name") if isinstance(tool, dict) else None
                            if not tname and isinstance(tool, dict):
                                tname = (tool.get("function") or {}).get("name")
                            if tname:
                                tool_names.append(str(tname))
                        system_prompt_text = str(self.config.system_prompt or "")
                        execute_steps = max(1, int(self.latent_mpc_execute_steps))
                        # The cooldown (self._beam_plan_cursor < execute_steps) applies whether or not
                        # the LAST re-plan produced a plan to inject. A rejected re-plan (flat/similar
                        # trajectory scores) is near-certain to look just as flat one step later --
                        # nothing about the environment changed drastically in a single agent step -- so
                        # immediately re-invoking the world model there just re-spends horizon LLM calls
                        # on the same answer. Only the "ran out of cached plan before the cooldown
                        # elapsed" edge case (execute_steps > horizon) forces an early re-plan.
                        # Built before the re-plan decision because the critic trigger scores the
                        # seed action, and reused by the planner after it.
                        seed_step = _wrap_step({"tool_calls": planned_calls})
                        # For planning, up-weight progress and make non-positive progress a penalty
                        # (a stagnant/no-progress step -- e.g. a redundant read -- should score lower).
                        score_cfg = CanonicalEventScoreConfig()
                        score_cfg.weights["progress_signal"] = 1.2
                        score_cfg.utilities["progress_signal"]["neutral"] = -0.3
                        score_cfg.utilities["progress_signal"]["negative"] = -1.0
                        imagined_history = list(current_input_history or [])
                        critic_result: dict[str, Any] | None = None
                        plan_exhausted = bool(self._beam_imagined_plan) and self._beam_plan_cursor >= len(self._beam_imagined_plan)
                        if self.beam_plan_trigger == "critic":
                            # Ask the world model whether THIS action needs help (one forward, no
                            # LLM) instead of re-planning on a fixed cadence.
                            critic_result = self._beam_plan_critic(
                                seed_step=seed_step, system_prompt_text=system_prompt_text,
                                user_query=user_query, imagined_history=imagined_history,
                                score_cfg=score_cfg,
                            )
                            self._beam_critic_checks += 1
                            quiet_exceeded = (
                                self.beam_plan_critic_max_quiet_steps > 0
                                and self._beam_quiet_steps >= self.beam_plan_critic_max_quiet_steps
                            )
                            need_replan = bool(critic_result["fires"]) or quiet_exceeded
                            if need_replan:
                                self._beam_critic_fires += 1
                                self._beam_quiet_steps = 0
                            else:
                                self._beam_quiet_steps += 1
                            emit_progress(
                                "GYM_BEAM_PLAN_CRITIC", mode=self.mode, step=steps_taken,
                                fires=need_replan, reason=critic_result.get("reason"),
                                score=critic_result.get("score"),
                                failure_prob=critic_result.get("failure_prob"),
                                stall_prob=critic_result.get("stall_prob"),
                                terminal_probability=critic_result.get("terminal_probability"),
                                terminal_advice=critic_result.get("terminal_advice"),
                                quiet_steps=self._beam_quiet_steps, quiet_exceeded=quiet_exceeded,
                                fire_rate=(self._beam_critic_fires / max(1, self._beam_critic_checks)),
                                selected_calls=preview_tool_calls(planned_calls),
                            )
                            if not need_replan:
                                # Cheapest path: the agent's action stands, nothing else is spent.
                                terminal_advice_set = self._set_beam_terminal_advisory(
                                    probability=critic_result.get("terminal_probability"),
                                    source="critic_seed_action",
                                    step_index=steps_taken,
                                )
                                self._beam_plan_records.append({
                                    "step_index": steps_taken,
                                    "replanned": False,
                                    "trigger": "critic",
                                    "critic": critic_result,
                                    "critic_checks": self._beam_critic_checks,
                                    "critic_fires": self._beam_critic_fires,
                                    "terminal_advice_set": terminal_advice_set,
                                    "agent_calls": preview_tool_calls(planned_calls),
                                })
                        else:
                            need_replan = not self._beam_has_planned or self._beam_plan_cursor >= execute_steps or plan_exhausted
                        if not need_replan and self._beam_imagined_plan:
                            # Follow the visible imagined trajectory: do NOT call the world model or
                            # override -- the agent's own (imagined-plan-informed) decision stands.
                            self._beam_plan_cursor += 1
                            emit_progress(
                                "GYM_BEAM_PLAN_FOLLOW", mode=self.mode, step=steps_taken,
                                plan_cursor=self._beam_plan_cursor, plan_len=len(self._beam_imagined_plan),
                                selected_calls=preview_tool_calls(planned_calls),
                            )
                        elif not need_replan:
                            # Last re-plan was rejected (below the trajectory-confidence gate) -- ride
                            # out the rest of this cooldown window with no world-model call and no
                            # injected guidance; the agent decides on its own, same cadence as if a
                            # confident plan HAD been injected.
                            self._beam_plan_cursor += 1
                            emit_progress(
                                "GYM_BEAM_PLAN_COOLDOWN", mode=self.mode, step=steps_taken,
                                plan_cursor=self._beam_plan_cursor, execute_steps=execute_steps,
                                selected_calls=preview_tool_calls(planned_calls),
                            )
                        else:
                            first_recommend_calls_candidate = None
                            first_recommend_reason_candidate = "no_candidates"
                            first_num_candidates = 0
                            first_best: dict[str, Any] = {}
                            beam_calls = 0
                            per_depth_scores: list[float] = []
                            per_depth_normalized: list[float] = []
                            # Anti-repetition: penalize a candidate whose TOOL-NAME set was already
                            # executed this task (breaks same-tool loops even when the args are tweaked).
                            # Scaled by --latent-plan-diversity-multiplier; margin from --latent-plan-score-margin.
                            repeat_penalty_scale = 0.75 * max(1, int(self.latent_plan_diversity_multiplier))
                            override_margin = max(0.0, float(self.latent_plan_score_margin))
                            path_names = list(self._beam_recent_tool_names)
                            imagined_plan: list[dict[str, Any]] = []

                            def _step_tool_names(step: dict[str, Any]) -> tuple:
                                if not isinstance(step, dict):
                                    return tuple()
                                names = []
                                for call in (step.get("tool_calls") or []):
                                    if isinstance(call, dict):
                                        name = str((call.get("function") or {}).get("name") or call.get("name", "")).strip()
                                        if name:
                                            names.append(name)
                                return tuple(sorted(names))

                            def _adjusted_rank(records: list[dict[str, Any]]) -> list[tuple]:
                                ranked = []
                                for record in records:
                                    plan = record.get("plan") or []
                                    names = _step_tool_names(plan[0]) if plan else tuple()
                                    penalty = repeat_penalty_scale * path_names.count(names)
                                    ranked.append((float(record.get("score", 0.0)) - penalty, record, names))
                                ranked.sort(key=lambda item: -item[0])
                                return ranked

                            if self.imagined_rollout_mode == "open_loop":
                                # ONE agent call for every candidate plan + ONE batched world-model pass,
                                # instead of one agent call and one world-model call per horizon depth.
                                # Produces the same bookkeeping the depth loop below does.
                                _open = self._beam_plan_open_loop(
                                    beam_cfg=beam_cfg,
                                    score_cfg=score_cfg,
                                    seed_step=seed_step,
                                    system_prompt_text=system_prompt_text,
                                    user_query=user_query,
                                    imagined_history=imagined_history,
                                    tool_names=tool_names,
                                    state_text=json.dumps(
                                        imagined_history[-self.state_history_size:] if self.state_history_size else imagined_history,
                                        ensure_ascii=False, default=str,
                                    )[:8000],
                                    wrap_step=_wrap_step,
                                    step_tool_names=_step_tool_names,
                                )
                                imagined_plan = _open["imagined_plan"]
                                per_depth_scores = _open["per_depth_scores"]
                                per_depth_normalized = _open["per_depth_normalized"]
                                first_num_candidates = _open["first_num_candidates"]
                                first_best = _open["first_best"]
                                beam_calls = _open["beam_calls"]
                                first_recommend_calls_candidate = _open["first_recommend_calls_candidate"]
                                first_recommend_reason_candidate = _open["first_recommend_reason_candidate"]
                            else:
                                for depth in range(max(1, int(self.latent_plan_horizon))):
                                    state_text = json.dumps(
                                        imagined_history[-self.state_history_size:] if self.state_history_size else imagined_history,
                                        ensure_ascii=False, default=str,
                                    )[:8000]
                                    prompt = build_step_candidates_prompt(
                                        system_prompt_text, state_text, beam_cfg, depth, tool_names=tool_names or None
                                    )
                                    self._agent_call_count += 1
                                    beam_calls += 1
                                    raw = self.agent_generator.generate_from_messages(
                                        [
                                            {"role": "system", "content": "You output ONLY a JSON array of tool-call actions. No prose, no markdown."},
                                            {"role": "user", "content": prompt},
                                        ],
                                        temperature=self.latent_plan_temperature,
                                    )
                                    raw_text = raw if isinstance(raw, str) else str(getattr(raw, "content", raw) or "")
                                    actions = parse_action_candidates(raw_text, beam_cfg)
                                    candidate_steps = [w for w in (_wrap_step(a) for a in actions) if w]
                                    if depth == 0 and seed_step:  # always include the baseline action as a candidate
                                        candidate_steps = [seed_step] + candidate_steps
                                    if not candidate_steps:
                                        emit_progress(
                                            "GYM_BEAM_PLAN_NO_CANDIDATES", mode=self.mode, step=steps_taken, depth=depth,
                                            num_parsed=len(actions), raw_response_head=raw_text[:400],
                                        )
                                        break
                                    candidate_plans = [[step] for step in candidate_steps]
                                    scored = self.world_model_generator.score_action_plans_canonical_event(
                                        system_prompt=system_prompt_text, user_prompt=str(user_query),
                                        input_history=imagined_history, action_plans=candidate_plans, score_config=score_cfg,
                                    )
                                    ranked = _adjusted_rank(scored)
                                    best_entry = next((entry for entry in ranked if not entry[1].get("vetoed", False) and entry[1].get("plan")), None)
                                    if best_entry is None:
                                        emit_progress(
                                            "GYM_BEAM_PLAN_VETOED", mode=self.mode, step=steps_taken, depth=depth,
                                            num_candidates=len(candidate_plans),
                                        )
                                        break
                                    best_adj_score, best, best_names = best_entry
                                    best_step = best["plan"][0]
                                    if depth == 0:
                                        first_num_candidates = len(candidate_plans)
                                        first_best = best
                                        # Margin gate: the top candidate must beat the seed (candidate 0) by more
                                        # than override_margin before we recommend replacing the agent's choice.
                                        # This comparison can only be made at depth 0 (the seed is only defined
                                        # for the CURRENT immediate action) -- but whether we ACT on it is decided
                                        # after the full trajectory is built, below.
                                        seed_entry = next((entry for entry in ranked if entry[1].get("plan_index") == 0), None)
                                        seed_adj_score = seed_entry[0] if seed_entry else None
                                        beats_seed = best.get("plan_index") != 0 and (
                                            seed_adj_score is None or best_adj_score - seed_adj_score > override_margin
                                        )
                                        if beats_seed:
                                            new_calls = [
                                                normalize_tool_call(call)
                                                for call in (best_step.get("tool_calls", []) if isinstance(best_step, dict) else [])
                                            ]
                                            first_recommend_calls_candidate = [c for c in new_calls if str(c.get("name", "")).strip()] or None
                                            first_recommend_reason_candidate = "beam_best_beats_seed"
                                            recommend_step = best_step
                                        else:
                                            first_recommend_calls_candidate = None
                                            first_recommend_reason_candidate = (
                                                "seed_is_best" if best.get("plan_index") == 0 else "kept_baseline_below_margin"
                                            )
                                            recommend_step = seed_step if isinstance(seed_step, dict) else best_step
                                        best_step = recommend_step  # lookahead follows the recommended trajectory
                                    path_names.append(_step_tool_names(best_step))
                                    chosen_calls = [
                                        normalize_tool_call(call)
                                        for call in (best_step.get("tool_calls", []) if isinstance(best_step, dict) else [])
                                    ]
                                    chosen_calls = [c for c in chosen_calls if str(c.get("name", "")).strip()]
                                    per_depth_scores.append(float(best.get("score") or 0.0))
                                    per_depth_normalized.append(float(best.get("normalized_score") or 0.0))
                                    terminal_probs = best.get("per_step_terminal_prob") or []
                                    imagined_plan.append({
                                        "calls": chosen_calls,
                                        "score": best.get("score"),
                                        "reason": best.get("reason"),
                                        "predicted_state": best.get("predicted_state"),
                                        "terminal_probability": (
                                            float(terminal_probs[0]) if terminal_probs else best.get("terminal_probability")
                                        ),
                                    })
                                    imagined_history = imagined_history + [{"action": best_step, "observation": ""}]
                                    # No per-step short-circuit here -- gating happens AFTER the full trajectory
                                    # is built (see below): a flat-looking first step can still sit on a
                                    # trajectory that clearly separates from alternatives once the horizon plays
                                    # out, and per-step gating would throw that information away unseen.

                            # Fix 1 (trajectory-level) -- gate on the FULL generated trajectory's aggregate
                            # score/normalized-score, not the isolated first-step candidate spread. The
                            # confidence check and the override decision both move here, after generation.
                            horizon = max(1, int(self.latent_plan_horizon))
                            full_horizon_reached = len(imagined_plan) == horizon
                            trajectory_score = sum(
                                (score_cfg.gamma**index) * step_score for index, step_score in enumerate(per_depth_scores)
                            )
                            trajectory_avg_normalized = (
                                sum(per_depth_normalized) / len(per_depth_normalized) if per_depth_normalized else 0.0
                            )
                            trajectory_confidence_gate = beam_cfg.gate_flat_score_ratio / max(first_num_candidates, 1)
                            beam_confident = full_horizon_reached and trajectory_avg_normalized >= trajectory_confidence_gate

                            if beam_confident and first_recommend_calls_candidate:
                                first_recommend_calls = first_recommend_calls_candidate
                                first_override_reason = "beam_best_beats_seed"
                            elif first_num_candidates == 0:
                                first_recommend_calls = None
                                first_override_reason = "no_candidates"
                            elif not beam_confident:
                                first_recommend_calls = None
                                first_override_reason = "below_confidence_gate"
                            else:
                                first_recommend_calls = None
                                first_override_reason = first_recommend_reason_candidate

                            # Fix 2: only force the agent's action when hard-override is explicitly on.
                            override_applied = False
                            if self.latent_plan_hard_override and first_recommend_calls:
                                planned_calls = first_recommend_calls
                                override_applied = True
                            # Cross-replan anti-repetition memory tracks the EXECUTED action (now known,
                            # since the hard-override decision above is final) -- deferred to here rather
                            # than depth 0, since it depends on the trajectory-level confidence gate.
                            self._beam_recent_tool_names.append(_step_tool_names({"tool_calls": planned_calls}))
                            self._beam_has_planned = True  # a re-plan attempt happened; the cooldown alone governs from here
                            # Fix 1: inject the imagined trajectory ONLY when the beam was confident.
                            # A flat/low-confidence plan (78.8% of steps in the prior run) is more likely
                            # to nudge the agent toward generic-success safe reads than to help, so we
                            # withhold it and let the agent proceed on its own.
                            if beam_confident and imagined_plan:
                                self._beam_imagined_plan = imagined_plan
                                if self.latent_plan_hard_override and override_applied:
                                    self._set_beam_terminal_advisory(
                                        probability=imagined_plan[0].get("terminal_probability"),
                                        source="beam_hard_override",
                                        step_index=steps_taken,
                                    )
                                # Under hard-override step 0 is executed now, so advise from step 1. In
                                # advisory mode the agent kept its own action, so step 0 (the recommended
                                # next action) is still unseen -- show the whole plan from step 0.
                                self._beam_plan_cursor = 1 if (self.latent_plan_hard_override and override_applied) else 0
                                injected = True
                            else:
                                self._beam_imagined_plan = []
                                self._beam_plan_cursor = 0
                                injected = False
                            # Fix 3: log the ACTUAL injected trajectory (calls + predicted states) so the
                            # guidance quality can be audited directly on the next run.
                            def _truncate_args(arguments: Any, limit: int = 300) -> Any:
                                try:
                                    text = json.dumps(arguments, ensure_ascii=False, default=str)
                                except (TypeError, ValueError):
                                    text = str(arguments)
                                return arguments if len(text) <= limit else text[:limit] + "...(truncated)"

                            plan_preview = [
                                {
                                    "calls": [
                                        {"name": c.get("name"), "arguments": _truncate_args(c.get("arguments"))}
                                        for c in (entry.get("calls") or [])
                                    ],
                                    "score": entry.get("score"),
                                    "reason": entry.get("reason"),
                                    "predicted_state": entry.get("predicted_state"),
                                    "terminal_probability": entry.get("terminal_probability"),
                                }
                                for entry in imagined_plan
                            ]
                            self._beam_plan_records.append({
                                "step_index": steps_taken,
                                "replanned": True,
                                "num_candidates": first_num_candidates,
                                "beam_llm_calls": beam_calls,
                                "plan_mode": self.imagined_rollout_mode,
                                "trigger": self.beam_plan_trigger,
                                "critic": critic_result,
                                "critic_checks": self._beam_critic_checks,
                                "critic_fires": self._beam_critic_fires,
                                "imagined_plan_len": len(imagined_plan),
                                "best_score": first_best.get("score"),
                                "best_normalized_score": first_best.get("normalized_score"),
                                "best_reason": first_best.get("reason"),
                                # best_score/best_normalized_score/best_reason are depth-0's OWN values
                                # (kept for diagnostics); the gate itself now uses the trajectory-level
                                # aggregates below, not these.
                                "trajectory_score": trajectory_score,
                                "trajectory_avg_normalized_score": trajectory_avg_normalized,
                                "full_horizon_reached": full_horizon_reached,
                                "confidence_gate": trajectory_confidence_gate,
                                "beam_confident": beam_confident,
                                "advisory": not self.latent_plan_hard_override,
                                "hard_override": self.latent_plan_hard_override,
                                "injected": injected,
                                "override_applied": override_applied,
                                "override_reason": first_override_reason,
                                "recommended_calls": preview_tool_calls(first_recommend_calls) if first_recommend_calls else None,
                                "agent_calls": preview_tool_calls(planned_calls),
                                "imagined_plan": plan_preview,
                            })
                            emit_progress(
                                "GYM_BEAM_PLAN", mode=self.mode, step=steps_taken,
                                num_candidates=first_num_candidates, beam_llm_calls=beam_calls,
                                plan_mode=self.imagined_rollout_mode,
                                imagined_plan_len=len(imagined_plan), trajectory_score=trajectory_score,
                                trajectory_avg_normalized_score=trajectory_avg_normalized,
                                full_horizon_reached=full_horizon_reached,
                                beam_confident=beam_confident, injected=injected,
                                override_applied=override_applied, override_reason=first_override_reason,
                                selected_calls=preview_tool_calls(planned_calls),
                            )
                    except Exception as exc:  # noqa: BLE001 -- never let planning crash execution
                        emit_progress("GYM_BEAM_PLAN_ERROR", mode=self.mode, step=steps_taken, error=str(exc))

                if self.mode == "hier_latent_cem" and planned_calls:
                    # Hierarchical latent-action CEM MPC (src/hierarchical_action_sampling.py):
                    # ONE LLM call opens the search space with K diverse anchor actions; the world
                    # model then samples thousands of CONTINUOUS latent-action trajectories around
                    # them and CEM-refines a per-family Gaussian proposal toward the
                    # highest-scoring region -- LLM-free, decode-free (recursive rollout, NOT the
                    # Fast-LeWM action-prefix predictor). Only the converged best trajectory is
                    # decoded (nearest-anchor) and shown to the agent. Any failure falls back to
                    # the baseline action so planning never crashes the run.
                    try:
                        from src.canonical_event_scoring import CanonicalEventScoreConfig
                        from src.hierarchical_action_sampling import HierarchicalCEMConfig, hierarchical_cem_plan

                        tool_names = []
                        for tool in (self.available_tools or []):
                            tname = tool.get("name") if isinstance(tool, dict) else None
                            if not tname and isinstance(tool, dict):
                                tname = (tool.get("function") or {}).get("name")
                            if tname:
                                tool_names.append(str(tname))
                        system_prompt_text = str(self.config.system_prompt or "")
                        execute_steps = max(1, int(self.latent_mpc_execute_steps))
                        # Same cooldown-vs-rejection distinction as beam_plan: a rejected re-plan
                        # (below hier_cem_min_elite_agreement) shouldn't immediately re-trigger the CEM
                        # on the very next step -- ride out the rest of the cooldown autonomously and
                        # only re-plan once execute_steps have actually elapsed (or the cached plan, if
                        # any, runs out first).
                        hier_plan_exhausted = bool(self._hier_imagined_plan) and self._hier_plan_cursor >= len(self._hier_imagined_plan)
                        need_replan = not self._hier_has_planned or self._hier_plan_cursor >= execute_steps or hier_plan_exhausted
                        if not need_replan and self._hier_imagined_plan:
                            # Follow the visible imagined trajectory: do NOT call the world model or
                            # override -- the agent's own (imagined-plan-informed) decision stands.
                            self._hier_plan_cursor += 1
                            emit_progress(
                                "GYM_HIER_CEM_FOLLOW", mode=self.mode, step=steps_taken,
                                plan_cursor=self._hier_plan_cursor, plan_len=len(self._hier_imagined_plan),
                                selected_calls=preview_tool_calls(planned_calls),
                            )
                        elif not need_replan:
                            # Last re-plan was rejected -- no world-model call, no injected guidance,
                            # same cadence as if a confident plan had been injected.
                            self._hier_plan_cursor += 1
                            emit_progress(
                                "GYM_HIER_CEM_COOLDOWN", mode=self.mode, step=steps_taken,
                                plan_cursor=self._hier_plan_cursor, execute_steps=execute_steps,
                                selected_calls=preview_tool_calls(planned_calls),
                            )
                        else:
                            imagined_history = list(current_input_history or [])
                            gen = self.world_model_generator
                            context_text = gen._context_text(system_prompt_text, str(user_query))
                            current_state_text = gen._current_state_text(
                                system_prompt_text, str(user_query), imagined_history
                            )

                            def _hier_llm_generate(prompt: str) -> str:
                                self._agent_call_count += 1
                                raw = self.agent_generator.generate_from_messages(
                                    [
                                        {"role": "system", "content": "You output ONLY a JSON array of tool-call actions. No prose, no markdown."},
                                        {"role": "user", "content": prompt},
                                    ],
                                    temperature=self.latent_plan_temperature,
                                )
                                return raw if isinstance(raw, str) else str(getattr(raw, "content", raw) or "")

                            # For planning, up-weight progress and make non-positive progress a
                            # penalty (a stagnant/no-progress step -- e.g. a redundant read --
                            # should score lower). Same tweak as the beam_plan mode above.
                            score_cfg = CanonicalEventScoreConfig()
                            score_cfg.weights["progress_signal"] = 1.2
                            score_cfg.utilities["progress_signal"]["neutral"] = -0.3
                            score_cfg.utilities["progress_signal"]["negative"] = -1.0
                            cem_cfg = HierarchicalCEMConfig(
                                num_llm_anchors=self.hier_cem_anchors,
                                num_samples=self.hier_cem_samples,
                                num_elites=self.hier_cem_elites,
                                num_iters=self.hier_cem_iters,
                                horizon=self.hier_cem_horizon,
                                init_std=self.hier_cem_init_std,
                                min_std=self.hier_cem_min_std,
                                smoothing=self.hier_cem_smoothing,
                                min_elite_agreement=self.hier_cem_min_elite_agreement,
                                top_k=self.imagined_trajectory_top_k,
                                max_input_length=gen.max_input_length,
                                max_action_length=gen.max_action_length,
                                decode_strategy=self.hier_cem_decode_strategy,
                                decode_max_new_tokens=self.hier_cem_decode_max_new_tokens,
                                score_config=score_cfg,
                            )
                            result = hierarchical_cem_plan(
                                model=gen.model, tokenizer=gen.tokenizer, vocab=gen.canonical_event_vocab,
                                context_text=context_text, current_state_text=current_state_text,
                                llm_generate=_hier_llm_generate, config=cem_cfg, tool_names=tool_names or None,
                                tool_vocab=getattr(gen, "tool_vocab", None) or None,
                            )
                            confident = bool(result.get("confident"))
                            imagined_plan = []
                            for entry in (result.get("imagined_plan") or []):
                                calls = [c for c in (normalize_tool_call(c) for c in (entry.get("calls") or [])) if str(c.get("name", "")).strip()]
                                imagined_plan.append({
                                    "calls": calls, "score": entry.get("score"), "reason": entry.get("reason"),
                                    "predicted_state": entry.get("predicted_state"),
                                    # Preserved so runs with --hier-cem-decode-strategy learned_decoder can be
                                    # audited: was this step actually decoded by the trained action decoder, or
                                    # did it fall back to nearest_anchor (decode invalid/absent)? Dropping these
                                    # silently made that unanswerable from past logs.
                                    "family": entry.get("family"),
                                    "decode_source": entry.get("decode_source"),
                                    "decode_distance": entry.get("decode_distance"),
                                    "decode_anchor_index": entry.get("decode_anchor_index"),
                                })
                            override_applied = False
                            recommended_calls = imagined_plan[0]["calls"] if imagined_plan else None
                            # Advisory by default: the agent's own action always executes unless
                            # --latent-plan-hard-override is set (shared with beam_plan).
                            if self.latent_plan_hard_override and confident and recommended_calls:
                                planned_calls = recommended_calls
                                override_applied = True
                            self._hier_has_planned = True  # a re-plan attempt happened; the cooldown alone governs from here
                            # Confidence-gated injection: only show a plan the CEM actually
                            # converged on (elite_agreement >= hier_cem_min_elite_agreement).
                            if confident and imagined_plan:
                                self._hier_imagined_plan = imagined_plan
                                self._hier_plan_cursor = 1 if (self.latent_plan_hard_override and override_applied) else 0
                                injected = True
                            else:
                                self._hier_imagined_plan = []
                                self._hier_plan_cursor = 0
                                injected = False
                            decode_source_counts: dict[str, int] = {}
                            for entry in imagined_plan:
                                source = entry.get("decode_source") or "unknown"
                                decode_source_counts[source] = decode_source_counts.get(source, 0) + 1
                            self._hier_cem_records.append({
                                "step_index": steps_taken,
                                "replanned": True,
                                "num_anchors": result.get("num_anchors"),
                                "num_llm_calls": result.get("num_llm_calls"),
                                "imagined_plan_len": len(imagined_plan),
                                "best_score": result.get("score"),
                                "elite_agreement": result.get("elite_agreement"),
                                "vetoed": result.get("vetoed"),
                                "reason": result.get("reason"),
                                "confident": confident,
                                "advisory": not self.latent_plan_hard_override,
                                "hard_override": self.latent_plan_hard_override,
                                "injected": injected,
                                "override_applied": override_applied,
                                "recommended_calls": preview_tool_calls(recommended_calls) if recommended_calls else None,
                                "agent_calls": preview_tool_calls(planned_calls),
                                "family_distribution": result.get("family_distribution"),
                                # Per-record rollup: how many of the imagined_plan's steps were actually
                                # decoded by the learned action decoder vs. fell back to nearest_anchor --
                                # e.g. {"learned_decoder": 3, "nearest_anchor": 2}. Always {} when
                                # --hier-cem-decode-strategy is the nearest_anchor default (no fallback to
                                # report, every step already went straight to nearest_anchor).
                                "decode_source_counts": decode_source_counts,
                                "imagined_plan": imagined_plan,
                            })
                            emit_progress(
                                "GYM_HIER_CEM_PLAN", mode=self.mode, step=steps_taken,
                                num_anchors=result.get("num_anchors"), imagined_plan_len=len(imagined_plan),
                                best_score=result.get("score"), elite_agreement=result.get("elite_agreement"),
                                confident=confident, injected=injected, override_applied=override_applied,
                                decode_source_counts=decode_source_counts,
                                selected_calls=preview_tool_calls(planned_calls),
                            )
                    except Exception as exc:  # noqa: BLE001 -- never let planning crash execution
                        emit_progress("GYM_HIER_CEM_ERROR", mode=self.mode, step=steps_taken, error=str(exc))

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
                "latent_plan_records": latent_plan_records,
                "latent_plan_samples": self.latent_plan_samples,
                "latent_plan_elites": self.latent_plan_elites,
                "latent_plan_iters": self.latent_plan_iters,
                "latent_plan_horizon": self.latent_plan_horizon,
                "latent_mpc_execute_steps": self.latent_mpc_execute_steps,
                "latent_plan_temperature": self.latent_plan_temperature,
                "latent_plan_score_margin": self.latent_plan_score_margin,
                "latent_plan_diversity_multiplier": self.latent_plan_diversity_multiplier,
                "latent_plan_goal_mode": self.latent_plan_goal_mode,
                "gate_flat_score_ratio": self.gate_flat_score_ratio,
                "stage_plan_observation": stage_plan_observation,
                "final_goal_observation": final_goal_observation,
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
                "agent_call_count": self._agent_call_count,
                "beam_plan_records": self._beam_plan_records,
                "beam_plan_trigger": self.beam_plan_trigger,
                # p, the amortization lever: how often the critic escalated to planning.
                "beam_plan_critic_checks": self._beam_critic_checks,
                "beam_plan_critic_fires": self._beam_critic_fires,
                "beam_plan_critic_fire_rate": (
                    self._beam_critic_fires / self._beam_critic_checks if self._beam_critic_checks else None
                ),
                "beam_plan_terminal_advice": self.beam_plan_terminal_advice,
                "beam_plan_terminal_advice_threshold": self.beam_plan_terminal_advice_threshold,
                "beam_plan_terminal_advice_count": self._beam_terminal_advice_count,
                "hier_cem_records": self._hier_cem_records,
                "hier_cem_anchors": self.hier_cem_anchors,
                "hier_cem_samples": self.hier_cem_samples,
                "hier_cem_elites": self.hier_cem_elites,
                "hier_cem_iters": self.hier_cem_iters,
                "hier_cem_horizon": self.hier_cem_horizon,
                "hier_cem_min_elite_agreement": self.hier_cem_min_elite_agreement,
                "hier_cem_decode_strategy": self.hier_cem_decode_strategy,
                "hier_cem_decode_max_new_tokens": self.hier_cem_decode_max_new_tokens,
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
