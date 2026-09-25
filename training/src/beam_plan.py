"""Beam-search / lookahead planning for agentic execution with minimal LLM calls.

Every planning cycle (each n real steps) costs ONE LLM call: the LLM proposes m
open-loop n-step tool-call plans (with symbolic `$stepK.field` references for values
not yet known). The world model then rolls all m plans forward in latent space and
scores every step with the canonical-event classification heads -- zero further LLM
calls. Candidates share one batched WM forward pass per horizon step, so the cost is
n batched passes, not m*n. The agent is handed the top-k trajectories with reasons,
plus a gate flag telling it whether the plan set was confident enough to trust; only
the winning first action is materialized to executable JSON.

    m actions * n steps  ->  m*n world-model PREDICTIONS (cheap, no LLM)
                         ->  1 LLM call (all skeletons) + 0 during rollout
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from src.canonical_event_scoring import CanonicalEventScoreConfig, logits_to_field_probs_batched, rank_trajectories


@dataclass
class BeamPlanConfig:
    num_candidates: int = 8            # m: candidate plans requested from the LLM
    horizon: int = 5                   # n: lookahead steps per plan
    top_k: int = 3
    max_input_length: int = 2048
    max_action_length: int = 512
    temperature: float = 1.0           # head-logit temperature for scoring calibration
    gate_flat_score_ratio: float = 1.0  # call LLM again if best normalized_score < ratio/m
    score_config: CanonicalEventScoreConfig = field(default_factory=CanonicalEventScoreConfig)


SKELETON_PROMPT = """You are planning tool actions for an agent. From the CURRENT STATE, propose {m} DISTINCT candidate plans, each a sequence of up to {n} tool calls to try next.

OUTPUT FORMAT: return ONLY a JSON array (no prose, no markdown fences) of exactly {m} plans.
Each plan is a JSON array of steps. Each step is {{"name": "<tool>", "arguments": {{...}}}}.
Example: [[{{"name":"toolA","arguments":{{"x":1}}}}],[{{"name":"toolB","arguments":{{}}}},{{"name":"toolC","arguments":{{"id":"$step1.id"}}}}]]

Rules:
- For any argument whose value depends on an earlier step's result, use a symbolic reference string like "$step1.field" instead of guessing a concrete value.
- Make the {m} plans genuinely different strategies, not paraphrases.{tools}

CONTEXT:
{context}

CURRENT STATE:
{state}

JSON array of {m} plans:"""


def build_skeleton_prompt(
    context_text: str, current_state_text: str, config: BeamPlanConfig, tool_names: list[str] | None = None
) -> str:
    tools = ("\n- Use ONLY these tools: " + ", ".join(tool_names)) if tool_names else ""
    return SKELETON_PROMPT.format(
        m=config.num_candidates, n=config.horizon, context=context_text, state=current_state_text, tools=tools
    )


STEP_CANDIDATES_PROMPT = """Propose {m} candidate next tool calls to try from the CURRENT STATE (lookahead step {step}), in TWO groups:

GROUP A ({n_diff} actions): each MUST use a DIFFERENT tool name -- explore different kinds of next move. No two actions in this group may share a tool.
GROUP B ({n_same} actions): all use the SAME single tool (the one you judge most likely correct next), but each with DIFFERENT arguments -- explore parameterizations of that tool. Do NOT return the same arguments twice.

OUTPUT FORMAT: return ONLY a JSON array (no prose, no markdown) of all {m} actions, Group A first then Group B.
Each action is {{"name": "<tool>", "arguments": {{...}}}}.
Example: [{{"name":"toolA","arguments":{{"x":1}}}}, {{"name":"toolB","arguments":{{}}}}, {{"name":"toolB","arguments":{{"y":2}}}}]

Rules:
- For a value that depends on an earlier step's result, use a symbolic reference like "$step1.field".{tools}

CONTEXT:
{context}

CURRENT STATE (history so far):
{state}

JSON array of {m} actions:"""


SINGLE_PLAN_PROMPT = """You are planning tool actions for an agent. From the CURRENT STATE, propose ONE plan: a sequence of up to {n} tool calls to try next.

OUTPUT FORMAT: return ONLY a JSON array (no prose, no markdown fences) of up to {n} steps.
Each step is {{"name": "<tool>", "arguments": {{...}}}}.
Example: [{{"name":"toolA","arguments":{{"x":1}}}},{{"name":"toolB","arguments":{{"id":"$step1.id"}}}}]

Rules:
- For any argument whose value depends on an earlier step's result, use a symbolic reference string like "$step1.field" instead of guessing a concrete value.
- Commit to your single best strategy; do not hedge across alternatives.{tools}

CONTEXT:
{context}

CURRENT STATE:
{state}

JSON array of up to {n} steps:"""


def build_single_plan_prompt(
    context_text: str, current_state_text: str, config: BeamPlanConfig, tool_names: list[str] | None = None
) -> str:
    """Prompt for ONE n-step plan, sampled k times in parallel to build the candidate set.

    Cheaper than build_skeleton_prompt's "give me m plans in one response": the m plans are
    the same output tokens either way, but k sampled responses decode CONCURRENTLY after a
    single shared prefill, whereas one response emits all m plans on one serial decode stream.
    Diversity comes from sampling temperature instead of an instruction to differ.
    """
    tools = ("\n- Use ONLY these tools: " + ", ".join(tool_names)) if tool_names else ""
    return SINGLE_PLAN_PROMPT.format(n=config.horizon, context=context_text, state=current_state_text, tools=tools)


def build_step_candidates_prompt(
    context_text: str, state_text: str, config: BeamPlanConfig, step: int, tool_names: list[str] | None = None
) -> str:
    """Prompt for ONE horizon step: ask the LLM for m alternative NEXT tool calls (a flat JSON
    array), split into ceil(m/2) different-tool candidates (breadth) and floor(m/2) same-tool
    different-argument candidates (parameterization). A flat array of m actions is followed far
    more reliably than m full nested multi-step plans."""
    n_diff = config.num_candidates - config.num_candidates // 2  # ceil(m/2)
    n_same = config.num_candidates // 2                          # floor(m/2)
    tools = ("\n- Use ONLY these tools: " + ", ".join(tool_names)) if tool_names else ""
    return STEP_CANDIDATES_PROMPT.format(
        m=config.num_candidates, n_diff=n_diff, n_same=n_same, step=step + 1,
        context=context_text, state=state_text, tools=tools,
    )


def parse_single_plan(text: str, config: BeamPlanConfig) -> list[dict[str, Any]]:
    """Parse ONE plan (the reply to build_single_plan_prompt) into its list of step dicts.

    Deliberately NOT parse_plans: that reads a top-level array of step dicts as N one-step
    plans, which is right for the m-plans-in-one-response prompt and wrong here, where the
    array IS the single plan's step sequence. Tolerates a code fence, surrounding prose,
    `{"steps": [...]}`, a lone step dict, and a nested [[...]] (first plan wins).
    """
    if not text:
        return []
    stripped = _strip_code_fence(text)
    parsed: Any = None
    for candidate in (stripped, _extract_json_array(stripped)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            break
        except (ValueError, TypeError):
            parsed = None
    if isinstance(parsed, dict):
        parsed = parsed["steps"] if isinstance(parsed.get("steps"), list) else [parsed]
    if not isinstance(parsed, list):
        return []
    if parsed and isinstance(parsed[0], list):      # model nested it as a list of plans
        parsed = parsed[0]
    return [step for step in parsed if isinstance(step, dict)][: config.horizon]


def parse_action_candidates(text: str, config: BeamPlanConfig) -> list[dict[str, Any]]:
    """Parse a flat JSON array of candidate actions (one horizon step). Reuses the tolerant
    plan parser and takes each plan's first step; returns up to m action dicts."""
    actions: list[dict[str, Any]] = []
    for plan in parse_plans(text, config):
        if plan:
            actions.append(plan[0])
    return actions[: config.num_candidates]


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _extract_json_array(text: str) -> str | None:
    """Return the first balanced [...] substring (JSON array) in `text`, ignoring brackets
    inside strings. Lets us recover the plan array even when the model wraps it in prose."""
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_plans(text: str, config: BeamPlanConfig) -> list[list[dict[str, Any]]]:
    """Parse the LLM response into up to m plans, each a list of action-step dicts. Tolerant of
    code fences, surrounding prose, a top-level {"plans": [...]}, and single plan/step; returns
    [] on failure. Step dicts are left as-is here and normalized by the caller."""
    if not text:
        return []
    stripped = _strip_code_fence(text)
    parsed: Any = None
    for candidate in (stripped, _extract_json_array(stripped)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            break
        except (ValueError, TypeError):
            parsed = None
    if isinstance(parsed, dict):
        parsed = parsed["plans"] if isinstance(parsed.get("plans"), list) else [parsed]
    if not isinstance(parsed, list):
        return []
    plans: list[list[dict[str, Any]]] = []
    for entry in parsed:
        if isinstance(entry, dict):  # a single-step plan
            entry = [entry]
        if not isinstance(entry, list):
            continue
        steps = [step for step in entry if isinstance(step, dict)][: config.horizon]
        if steps:
            plans.append(steps)
    return plans[: config.num_candidates]


@torch.no_grad()
def rollout_and_score(
    model: Any,
    tokenizer: Any,
    vocab: dict[str, list[str]],
    context_text: str,
    current_state_text: str,
    plans: list[list[dict[str, Any]]],
    config: BeamPlanConfig,
    goal_text: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Roll each plan forward in latent space (recursive imagined rollout, batched over the
    m candidates so each horizon step is ONE forward pass) and score with the canonical-event
    heads. Returns (top_k, all_scored) from rank_trajectories."""
    from src.finetuning_jepa import render_action

    net = getattr(model, "module", model)
    net.eval()
    device = next(net.parameters()).device
    num = len(plans)
    if num == 0:
        return [], []

    def encode(texts: list[str]) -> torch.Tensor:
        tokens = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=config.max_input_length, add_special_tokens=True,
        )
        latent, _ = net.encode_latent_and_logits(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
        return latent

    z_context = encode([context_text] * num)
    z_current = encode([current_state_text] * num)  # imagined-state anchor, updated recursively
    z_goal = encode([goal_text] * num) if (goal_text and getattr(net, "goal_conditioning", False)) else None

    trajectories: list[list[dict[str, dict[str, float]]]] = [[] for _ in range(num)]
    terminal_prob_steps: list[list[float]] = [[] for _ in range(num)]
    terminal_enabled = bool(getattr(net, "terminal_head_enabled", False))
    for step in range(config.horizon):
        active = [i for i in range(num) if step < len(plans[i])]
        if not active:
            break
        action_texts = [render_action(plans[i][step]) if step < len(plans[i]) else " " for i in range(num)]
        z_action = encode([text or " " for text in action_texts])
        goal_arg = z_goal if getattr(net, "goal_conditioning", False) else None
        # predict_latent_with_state: checkpoints trained with --canonical-event-head-inputs
        # state read the transformer predictor's hidden state h_t, not the latent concat.
        # z_state is None for other head-input modes / the mlp predictor, and ignored there.
        z_pred, _, z_state = net.predict_latent_with_state(z_current, z_action, z_context, goal_arg)
        logits = net.predict_canonical_event_logits(z_current, z_action, z_context, z_pred, z_state)  # {field: [m, C]}
        # One host transfer per field instead of one per probability scalar (each of those is a
        # device sync; hundreds per step at m plans x 11 fields).
        probs_rows = logits_to_field_probs_batched(logits, vocab, temperature=config.temperature)
        terminal_probs = None
        if terminal_enabled:
            terminal_logits = net.predict_terminal_logit(z_current, z_action, z_context, z_pred)
            terminal_probs = torch.sigmoid(terminal_logits.detach().float()).cpu().tolist()
        for i in active:
            trajectories[i].append(probs_rows[i])
            if terminal_probs is not None:
                terminal_prob_steps[i].append(float(terminal_probs[i]))
        z_current = z_pred  # imagined next state feeds the next step
    top, all_scored = rank_trajectories(trajectories, config.score_config, config.top_k)
    for record in all_scored:
        index = int(record.get("index", -1))
        probs = terminal_prob_steps[index] if 0 <= index < len(terminal_prob_steps) else []
        record["per_step_terminal_prob"] = probs
        record["terminal_probability"] = probs[-1] if probs else None
    return top, all_scored


def should_call_llm(top: list[dict[str, Any]], config: BeamPlanConfig) -> tuple[bool, str]:
    """Uncertainty gate: fire the (single, batched) LLM proposal call only when the plan set
    is untrustworthy -- no candidates, all vetoed, or no candidate stands out."""
    if not top:
        return True, "no candidate plans"
    if all(record["vetoed"] for record in top):
        return True, "all top candidates vetoed"
    best = max((record["normalized_score"] for record in top), default=0.0)
    threshold = config.gate_flat_score_ratio / max(config.num_candidates, 1)
    if best < threshold:
        return True, f"flat scores (best normalized={best:.2f} < {threshold:.2f})"
    return False, "confident"


def beam_plan(
    model: Any,
    tokenizer: Any,
    vocab: dict[str, list[str]],
    context_text: str,
    current_state_text: str,
    llm_generate: Callable[[str], str],
    config: BeamPlanConfig | None = None,
    goal_text: str | None = None,
) -> dict[str, Any]:
    """One planning cycle. Exactly ONE LLM call (skeleton proposal); the m*n outcome
    predictions and scoring are LLM-free. Returns the top-k trajectories (with reasons and
    the materialized winning first action) plus the gate decision for the agent.
    """
    config = config or BeamPlanConfig()
    prompt = build_skeleton_prompt(context_text, current_state_text, config)
    raw = llm_generate(prompt)  # <-- the only LLM call in the cycle
    plans = parse_plans(raw, config)
    top, all_scored = rollout_and_score(
        model, tokenizer, vocab, context_text, current_state_text, plans, config, goal_text
    )
    for record in top:
        plan = plans[record["index"]] if record["index"] < len(plans) else []
        first = plan[0] if plan else None
        record["plan"] = plan
        record["first_action"] = first
        # "Materialize" = deterministic render of only the winning first action to executable JSON.
        record["first_action_json"] = json.dumps(first, ensure_ascii=False) if first is not None else None
    gate, gate_reason = should_call_llm(top, config)
    return {
        "top": top,
        "all": all_scored,
        "num_llm_calls": 1,
        "should_call_llm": gate,
        "gate_reason": gate_reason,
        "num_candidates_parsed": len(plans),
    }
