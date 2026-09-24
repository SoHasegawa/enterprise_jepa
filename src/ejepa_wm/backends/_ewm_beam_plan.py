"""Beam-search / lookahead planning helpers (prompt building + tolerant parsing).

Inference-only **port** of the EWM repo's ``src/beam_plan.py`` — the subset the
mcp_react ``beam_plan`` MPC controller actually uses: :class:`BeamPlanConfig`, the
per-horizon-step candidate prompt, and the tolerant JSON-array parser. Self-contained
(no external EWM checkout on ``sys.path``).

Only the **per-step flat-candidate** path is ported (one LLM call per horizon step for
``m`` flat next-actions), because that is what the EnterpriseOps-Gym orchestrator's
``beam_plan`` block runs — models follow "give me m alternative next tool calls" far
more reliably than "give me m full nested multi-step plans". The idealized single-call
``beam_plan()`` / ``rollout_and_score()`` functions from the source are intentionally
NOT ported: they require a decoder net seam (``encode_latent_and_logits``) the
inference JEPA generator in :mod:`_ewm_jepa` does not expose; scoring instead goes
through :meth:`_ewm_jepa.JepaEwmGenerator.score_action_plans_canonical_event`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ejepa_wm.backends._ewm_canonical_event_scoring import CanonicalEventScoreConfig


@dataclass
class BeamPlanConfig:
    num_candidates: int = 8  # m: candidate actions requested from the LLM per horizon step
    horizon: int = 5  # n: lookahead steps per plan
    top_k: int = 3
    max_input_length: int = 2048
    max_action_length: int = 512
    temperature: float = 1.0  # head-logit temperature for scoring calibration
    gate_flat_score_ratio: float = 1.5  # call LLM again if best normalized_score < ratio/m
    score_config: CanonicalEventScoreConfig = field(default_factory=CanonicalEventScoreConfig)


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
    context_text: str,
    current_state_text: str,
    config: BeamPlanConfig,
    tool_names: list[str] | None = None,
) -> str:
    """Prompt for a single agent call that proposes ``m`` DISTINCT ``n``-step open-loop plans
    (with symbolic ``$stepK.field`` refs for values not yet known). Used by the batched
    decode+judge path: one LLM call yields all rollouts, which the world model then rolls
    forward in latent space with no further LLM calls.

    Superseded by :func:`build_single_plan_prompt` + ``sample_many`` for beam_plan's own
    open-loop driver (``_beam_plan_open_loop``): m plans in one response is m times the output
    tokens on ONE serial decode stream, whereas m *sampled* single-plan responses decode
    concurrently after a single shared prefill. Kept here (unused by beam_plan itself) as a
    fallback shape for callers whose backend has no ``generate_samples``/parallel-request seam
    at all and would rather pay one big serial call than issue m separate small ones."""
    tools = ("\n- Use ONLY these tools: " + ", ".join(tool_names)) if tool_names else ""
    return SKELETON_PROMPT.format(
        m=config.num_candidates,
        n=config.horizon,
        context=context_text,
        state=current_state_text,
        tools=tools,
    )


DIVERSITY_SLOT_INSTRUCTIONS = (
    (
        "best_overall",
        "Use the most direct, high-confidence action sequence that covers the task requirements.",
    ),
    (
        "different_first_tool",
        "Start with a materially different useful tool from the most obvious first action, then pursue "
        "a valid route to the same goal.",
    ),
    (
        "read_or_verify_first",
        "Start by reading, searching, listing, or verifying state before making a mutation, if that "
        "is plausible for the task.",
    ),
    (
        "write_or_update_first",
        "If the current state contains the required identifiers, start with a justified create, update, "
        "assign, or other progress-making mutation.",
    ),
    (
        "alternate_entity_order",
        "Handle entities or subrequirements in a different order from the most obvious plan while "
        "still aiming to finish the same task.",
    ),
    (
        "minimal_plan",
        "Use the shortest viable plan. Avoid redundant verification unless the state is ambiguous.",
    ),
    (
        "recovery_plan",
        "Use a robust recovery-oriented plan that accounts for missing IDs, lookup failures, or "
        "partial prior progress before proceeding.",
    ),
    (
        "dependency_first",
        "Resolve the most important prerequisite or dependency first, then use its result through "
        "symbolic references in later actions.",
    ),
    (
        "unfinished_requirement_first",
        "Prioritize a task requirement that appears not yet satisfied, especially one other plausible "
        "plans might postpone or overlook.",
    ),
    (
        "relationship_first",
        "Inspect or establish the key relationship between relevant entities before handling their "
        "remaining attributes or status changes.",
    ),
    (
        "reversible_first",
        "Prefer a conservative, reversible first action that reduces uncertainty while preserving a "
        "clear route to completion.",
    ),
    (
        "completion_check",
        "Treat the task as potentially near completion: identify the strongest remaining completion "
        "condition and verify or satisfy it without repeating completed work.",
    ),
)


SINGLE_PLAN_PROMPT = """You are generating ONE candidate action trajectory for beam search. From the CURRENT STATE, propose a sequence of up to {n} tool calls to try next.

OUTPUT FORMAT:
{output_format}
Each step is {{"name": "<tool>", "arguments": {{...}}}}.

Rules:
- For any argument whose value depends on an earlier step's result, use a symbolic reference string like "$step1.field" instead of guessing a concrete value.
- If a lookup can return multiple records, add a bind selector on that step and reference it with "$vars.name"; for example {{"name":"find_user","arguments":{{"name":"Ethan Well"}},"bind":{{"ethan_user_id":{{"field":"sys_id","match":{{"name":"Ethan Well"}}}}}}}} followed by {{"name":"assign_incident","arguments":{{"user_id":"$vars.ethan_user_id"}}}}.
- Do not list alternatives inside the plan. Return only ONE single plan.
- The selected diversity option must materially affect the first useful action, subgoal order, or information-gathering strategy; a paraphrase does not count as a different trajectory.
- Do not invent entity IDs, record identifiers, or tool arguments merely to appear different.
- If the selected diversity option is impossible, use the closest feasible interpretation while preserving task correctness.{tools}

CONTEXT:
{context}

CURRENT STATE:
{state}

{diversity_procedure}

DIVERSITY OPTIONS:
{diversity_options}

{final_cue}"""


SSOT_DIVERSITY_INSTRUCTIONS = """SSoT DIVERSITY PROCEDURE:
This procedure is required.
1. Before selecting any action, generate a unique 16-character mixed-case alphanumeric random string. Do not derive it from the task, state, option labels, or a familiar example.
2. Use ALL characters of that string as the randomness source. Map it to one of the {slot_count} options with a sum-mod calculation: sum the ASCII values and select option 1 + (sum modulo {slot_count}).
3. Condition the entire action trajectory on the selected option. Do not default to option 1 or the most familiar plan after selecting another feasible option.
4. Emit the random string and selected option before the steps in the required JSON object. Do not emit the calculation or any reasoning.
5. The top-level JSON value must be an object. Do not return a bare array."""


PLAIN_OUTPUT_FORMAT = """Return ONLY a JSON array (no prose or markdown fences) of up to {n} steps.
Example: [{{"name":"toolA","arguments":{{"x":1}}}},{{"name":"toolB","arguments":{{"id":"$step1.id"}}}}]"""


SSOT_OUTPUT_FORMAT = """Return ONLY one JSON object (no prose or markdown fences) with keys in this exact order:
{{"random_seed":"<16 mixed-case alphanumeric characters>","diversity_slot":<integer 1-{slot_count}>,"steps":[...]}}
The steps array contains up to {n} tool calls. The diversity_slot must be the option derived from random_seed. Never return a top-level array in SSoT mode."""


def _format_diversity_options(slot_count: int) -> str:
    count = max(1, int(slot_count))
    lines: list[str] = []
    for index in range(count):
        name, instruction = DIVERSITY_SLOT_INSTRUCTIONS[index % len(DIVERSITY_SLOT_INSTRUCTIONS)]
        lines.append(f"{index + 1}. {name}: {instruction}")
    return "\n".join(lines)


def build_single_plan_prompt(
    context_text: str,
    current_state_text: str,
    config: BeamPlanConfig,
    tool_names: list[str] | None = None,
    *,
    ssot_diversity: bool = False,
) -> str:
    """Prompt for ONE ``n``-step plan with a diversity-option menu.

    Open-loop beam planning samples this SAME prompt ``m`` times through ``sample_many``. vLLM's
    n=k path can prefill the shared prompt once while the diversity menu gives stochastic decodes
    several valid plan styles to spread across.
    """
    tools = ("\n- Use ONLY these tools: " + ", ".join(tool_names)) if tool_names else ""
    slot_count = max(1, int(config.num_candidates))
    if ssot_diversity:
        output_format = SSOT_OUTPUT_FORMAT.format(n=config.horizon, slot_count=slot_count)
        diversity_procedure = SSOT_DIVERSITY_INSTRUCTIONS.format(slot_count=slot_count)
        final_cue = (
            'Return the JSON object now. It must start with {"random_seed": and contain '
            '"diversity_slot" before "steps":'
        )
    else:
        output_format = PLAIN_OUTPUT_FORMAT.format(n=config.horizon)
        diversity_procedure = (
            "DIVERSITY PROCEDURE:\n"
            f"- Select exactly one option from 1-{slot_count} before planning. Sample among the "
            "options rather than always choosing option 1.\n"
            "- Use the selected option to shape the trajectory, but output only the JSON plan."
        )
        final_cue = f"JSON array of up to {config.horizon} steps:"
    return SINGLE_PLAN_PROMPT.format(
        n=config.horizon,
        context=context_text,
        state=current_state_text,
        tools=tools,
        diversity_options=_format_diversity_options(slot_count),
        diversity_procedure=diversity_procedure,
        output_format=output_format,
        final_cue=final_cue,
    )


def build_step_candidates_prompt(
    context_text: str,
    state_text: str,
    config: BeamPlanConfig,
    step: int,
    tool_names: list[str] | None = None,
) -> str:
    """Prompt for ONE horizon step: ask the LLM for m alternative NEXT tool calls (a flat JSON
    array), split into ceil(m/2) different-tool candidates (breadth) and floor(m/2) same-tool
    different-argument candidates (parameterization). A flat array of m actions is followed far
    more reliably than m full nested multi-step plans. The split is prompt-enforced, not
    structurally guaranteed; the cross-step anti-repetition penalty is the backstop against loops."""
    n_diff = config.num_candidates - config.num_candidates // 2  # ceil(m/2)
    n_same = config.num_candidates // 2  # floor(m/2)
    tools = ("\n- Use ONLY these tools: " + ", ".join(tool_names)) if tool_names else ""
    return STEP_CANDIDATES_PROMPT.format(
        m=config.num_candidates,
        n_diff=n_diff,
        n_same=n_same,
        step=step + 1,
        context=context_text,
        state=state_text,
        tools=tools,
    )


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


def parse_action_candidates(text: str, config: BeamPlanConfig) -> list[dict[str, Any]]:
    """Parse a flat JSON array of candidate actions (one horizon step). Reuses the tolerant
    plan parser and takes each plan's first step; returns up to m action dicts."""
    actions: list[dict[str, Any]] = []
    for plan in parse_plans(text, config):
        if plan:
            actions.append(plan[0])
    return actions[: config.num_candidates]


def parse_single_plan(text: str, config: BeamPlanConfig) -> list[dict[str, Any]]:
    """Parse ONE plan (the reply to :func:`build_single_plan_prompt`) into its list of step
    dicts.

    Deliberately NOT :func:`parse_plans`: that reads a top-level array of step dicts as m
    one-step plans, which is right for the "m plans in one response" prompt and wrong here,
    where the array IS the single plan's step sequence. Tolerates a code fence, surrounding
    prose, ``{"steps": [...]}``, a lone step dict, and a nested ``[[...]]`` (first plan wins).
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
    if parsed and isinstance(parsed[0], list):  # model nested it as a list of plans
        parsed = parsed[0]
    return [step for step in parsed if isinstance(step, dict)][: config.horizon]


__all__ = [
    "BeamPlanConfig",
    "build_single_plan_prompt",
    "build_skeleton_prompt",
    "build_step_candidates_prompt",
    "parse_action_candidates",
    "parse_plans",
    "parse_single_plan",
]
