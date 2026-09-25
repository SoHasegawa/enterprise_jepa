"""LLM-text canonical-event world model for ``beam_plan``/critic.

The current EWM LLM target is a reduced JSON schema: the five fields consumed by beam
scoring plus a binary ``terminal`` flag. A causal LM emits one category per field, not
calibrated classifier probabilities, so this adapter assigns probability 1.0 to the
generated category and 0.0 to the other categories before calling the shared
canonical-event scorer.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ejepa_wm.backends._canonical_event_state import (
    ERROR_SIGNATURE_VALUES,
    EXECUTION_STATUS_VALUES,
    INFORMATION_SUFFICIENCY_VALUES,
    PROGRESS_SIGNAL_VALUES,
    SIDE_EFFECT_TYPE_VALUES,
)
from ejepa_wm.backends._ewm_finetuning import (
    WORLD_MODEL_INPUT_HISTORY_SIZE,
    normalize_world_model_input_history_text,
    parse_jsonish,
    strip_code_fence,
    strip_model_thinking_output,
)

logger = logging.getLogger(__name__)

# Reduced LLM target schema from the EWM repo's `src/finetuning.py`:
# five beam-scored fields plus a binary terminal flag.
CANONICAL_EVENT_BEAM_TARGET_FIELDS: tuple[str, ...] = (
    "execution_status",
    "progress_signal",
    "information_sufficiency",
    "error_signature",
    "side_effect_type",
)
CANONICAL_EVENT_TERMINAL_FIELD = "terminal"
CANONICAL_EVENT_TERMINAL_VALUES: tuple[str, str] = ("not_finished", "finished")
CANONICAL_EVENT_LLM_TARGET_FIELDS: tuple[str, ...] = (
    CANONICAL_EVENT_BEAM_TARGET_FIELDS + (CANONICAL_EVENT_TERMINAL_FIELD,)
)

# Preserve the value order used by the EWM LLM training prompt, not arbitrary set order.
CANONICAL_EVENT_VOCAB: Dict[str, List[str]] = {
    "execution_status": ["failure", "no_op", "partial", "success", "unknown"],
    "progress_signal": ["negative", "neutral", "positive", "unknown"],
    "information_sufficiency": ["insufficient", "sufficient", "unknown"],
    "error_signature": [
        "dependency_missing", "invalid_argument", "none", "not_found", "parse_error",
        "permission_denied", "policy_risk", "runtime_error", "test_failed", "timeout", "unknown",
    ],
    "side_effect_type": [
        "created", "deleted", "executed", "failed_validation", "installed", "modified", "none",
        "retrieved", "sent", "unknown", "validated",
    ],
    CANONICAL_EVENT_TERMINAL_FIELD: list(CANONICAL_EVENT_TERMINAL_VALUES),
}
assert set(CANONICAL_EVENT_VOCAB["execution_status"]) == EXECUTION_STATUS_VALUES
assert set(CANONICAL_EVENT_VOCAB["progress_signal"]) == PROGRESS_SIGNAL_VALUES
assert set(CANONICAL_EVENT_VOCAB["information_sufficiency"]) == INFORMATION_SUFFICIENCY_VALUES
assert set(CANONICAL_EVENT_VOCAB["error_signature"]) == ERROR_SIGNATURE_VALUES
assert set(CANONICAL_EVENT_VOCAB["side_effect_type"]) == SIDE_EFFECT_TYPE_VALUES
FIELD_ORDER: tuple[str, ...] = CANONICAL_EVENT_LLM_TARGET_FIELDS
MISSING_INFO_FIELD = "missing_information_type"
MISSING_INFO_VOCAB: List[str] = []

_SYSTEM_PROMPT_LINES = [
    "You are an enterprise world model. Given the system prompt, user task, recent history, "
    "and the current action, predict the resulting annotated state.",
    "Return ONLY a JSON object with exactly these keys, in this order, and no other text:",
]
for _field in CANONICAL_EVENT_BEAM_TARGET_FIELDS:
    _SYSTEM_PROMPT_LINES.append(f"- {_field}: one of [{', '.join(CANONICAL_EVENT_VOCAB[_field])}]")
_SYSTEM_PROMPT_LINES.append(
    f"- {CANONICAL_EVENT_TERMINAL_FIELD}: one of [{', '.join(CANONICAL_EVENT_TERMINAL_VALUES)}]; "
    "use finished only when this action is the final step that completes or ends the task"
)
CANONICAL_EVENT_SYSTEM_PROMPT = "\n".join(_SYSTEM_PROMPT_LINES)
del _SYSTEM_PROMPT_LINES, _field

def build_canonical_event_prompt(
    system_prompt: str, user_prompt: str, input_history: List[Dict[str, Any]], action: Any,
) -> List[Dict[str, str]]:
    """Chat messages matching the checkpoint's training format exactly (verified against
    ``test_prompt_completion.jsonl``): ``Previous state`` is always the literal placeholder
    ``{}`` in every training example (unused by this target), so it is hardcoded, not computed.
    """
    action_text = action if isinstance(action, str) else json.dumps(action, indent=2, ensure_ascii=False)
    history_text = normalize_world_model_input_history_text(
        list(input_history or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:]
    )
    user_content = (
        f"System prompt:\n{system_prompt}\n\n"
        f"User prompt:\n{user_prompt}\n\n"
        "Previous state:\n{}\n\n"
        "Recent action/observation history (oldest to newest; input only, not part of the target):\n"
        f"{history_text}\n\n"
        f"Action:\n{action_text}\n\n"
        "Predict the annotated state as JSON. /no_think"
    )
    return [
        {"role": "system", "content": CANONICAL_EVENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _default_canonical_event() -> Dict[str, Any]:
    """Neutral fallback when a generation cannot be parsed at all."""
    result = {field: "unknown" for field in CANONICAL_EVENT_BEAM_TARGET_FIELDS}
    result[CANONICAL_EVENT_TERMINAL_FIELD] = CANONICAL_EVENT_TERMINAL_VALUES[0]
    return result


def parse_canonical_event_completion(text: str) -> Dict[str, Any]:
    """Best-effort parse of a reduced canonical-event completion.

    The prompt asks for flat JSON, but this accepts older nested canonical-event shapes too so
    zero-shot general LLMs can still be scored when they add ``canonical_event_state`` wrappers.
    Missing or invalid fields default to neutral values; ``terminal`` defaults to
    ``not_finished`` to avoid false finish advice.
    """
    cleaned = strip_model_thinking_output(strip_code_fence(text or ""))
    try:
        parsed = parse_jsonish(cleaned)
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        return _default_canonical_event()
    source: Dict[str, Any] = parsed
    nested = parsed.get("canonical_event_state")
    if isinstance(nested, dict):
        source = {**nested, **{k: v for k, v in parsed.items() if k != "canonical_event_state"}}
        nudge = parsed.get("nudge")
        if isinstance(nudge, dict):
            source.update(nudge)
    result = _default_canonical_event()
    for field in CANONICAL_EVENT_BEAM_TARGET_FIELDS:
        value = source.get(field)
        if isinstance(value, str) and value in CANONICAL_EVENT_VOCAB[field]:
            result[field] = value
    terminal = source.get(CANONICAL_EVENT_TERMINAL_FIELD)
    if isinstance(terminal, bool):
        result[CANONICAL_EVENT_TERMINAL_FIELD] = (
            CANONICAL_EVENT_TERMINAL_VALUES[1] if terminal else CANONICAL_EVENT_TERMINAL_VALUES[0]
        )
    elif isinstance(terminal, str):
        normalized = terminal.strip().lower()
        aliases = {"done": "finished", "complete": "finished", "completed": "finished"}
        normalized = aliases.get(normalized, normalized)
        if normalized in CANONICAL_EVENT_TERMINAL_VALUES:
            result[CANONICAL_EVENT_TERMINAL_FIELD] = normalized
    return result


def canonical_event_completion_text(event: Dict[str, Any]) -> str:
    """Serialize the reduced target in the same fixed order used for LLM training."""
    ordered = {field: event.get(field, "unknown") for field in CANONICAL_EVENT_BEAM_TARGET_FIELDS}
    ordered[CANONICAL_EVENT_TERMINAL_FIELD] = event.get(
        CANONICAL_EVENT_TERMINAL_FIELD, CANONICAL_EVENT_TERMINAL_VALUES[0]
    )
    return json.dumps(ordered, ensure_ascii=False)


def canonical_event_to_field_probs(event: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Convert one generated category per field into one-hot scorer probabilities."""
    probs: Dict[str, Dict[str, float]] = {}
    for field, categories in CANONICAL_EVENT_VOCAB.items():
        fallback = CANONICAL_EVENT_TERMINAL_VALUES[0] if field == CANONICAL_EVENT_TERMINAL_FIELD else "unknown"
        value = event.get(field, fallback)
        if value not in categories:
            value = fallback
        probs[field] = {category: (1.0 if category == value else 0.0) for category in categories}
    return probs


def terminal_probability_from_event(event: Dict[str, Any]) -> float:
    return 1.0 if event.get(CANONICAL_EVENT_TERMINAL_FIELD) == CANONICAL_EVENT_TERMINAL_VALUES[1] else 0.0


__all__ = [
    "CANONICAL_EVENT_BEAM_TARGET_FIELDS",
    "CANONICAL_EVENT_LLM_TARGET_FIELDS",
    "CANONICAL_EVENT_SYSTEM_PROMPT",
    "CANONICAL_EVENT_TERMINAL_FIELD",
    "CANONICAL_EVENT_TERMINAL_VALUES",
    "CANONICAL_EVENT_VOCAB",
    "FIELD_ORDER",
    "LlmCanonicalEventGenerator",
    "ServedLlmCanonicalEventGenerator",
    "build_canonical_event_prompt",
    "canonical_event_completion_text",
    "canonical_event_to_field_probs",
    "parse_canonical_event_completion",
    "terminal_probability_from_event",
]


class LlmCanonicalEventGenerator:
    """Loads a fine-tuned causal LM and scores ``beam_plan`` candidate plans with it, matching
    :class:`_ewm_jepa.JepaEwmGenerator`'s ``score_action_plans_canonical_event`` contract.

    Local in-process ``transformers`` inference (no vLLM server needed) — imported lazily so
    importing this module never pulls ``torch``/``transformers`` in on its own, mirroring
    ``_ewm_jepa``/``_ewm_k_controller``.
    """

    canonical_event_available = True

    def __init__(
        self,
        checkpoint: str,
        *,
        max_new_tokens: int = 128,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        device: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ejepa_wm.backends._ewm_jepa import resolve_torch_dtype

        self._torch = torch
        self.checkpoint = checkpoint
        self.max_new_tokens = max_new_tokens
        resolved_dtype = resolve_torch_dtype(dtype)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, trust_remote_code=trust_remote_code, dtype=resolved_dtype,
        ).to(self.device)
        self.model.eval()
    def _prompt_text(self, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        )

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        torch = self._torch
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **encoded, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        completions = []
        for row in range(output_ids.shape[0]):
            new_tokens = output_ids[row, encoded["input_ids"].shape[1] :]
            completions.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return completions


    def _generate_message_batch(self, messages_batch: List[List[Dict[str, str]]]) -> List[str]:
        prompts = [self._prompt_text(messages, add_generation_prompt=True) for messages in messages_batch]
        return self._generate_batch(prompts)

    def score_action_plans_canonical_event(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        input_history: List[Dict[str, Any]],
        action_plans: List[List[Any]],
        goal_text_override: Optional[str] = None,
        score_config: Any = None,
    ) -> List[Dict[str, Any]]:
        """Score candidate action plans using generated reduced canonical-event labels.

        The LLM produces a single category for each field, so probabilities are one-hot by
        construction. This matches the current EWM LLM target and avoids pretending that token
        likelihoods are calibrated class probabilities.
        """
        return _score_action_plans_with_canonical_event_generator(
            generate_message_batch=self._generate_message_batch,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_history=input_history,
            action_plans=action_plans,
            score_config=score_config,
        )


class ServedLlmCanonicalEventGenerator:
    """OpenAI-compatible canonical-event LLM scorer for served trained or zero-shot models."""

    canonical_event_available = True

    def __init__(self, generator: Any, *, mode: str = "llm_canonical_zeroshot", max_workers: int = 8) -> None:
        self.generator = generator
        self.mode = mode
        self.max_workers = max(1, int(max_workers))

    def _generate_message_batch(self, messages_batch: List[List[Dict[str, str]]]) -> List[str]:
        from ejepa_wm.backends import _ewm_runtime as ewm

        return ewm.generate_many(
            self.generator,
            messages_batch,
            [0.0] * len(messages_batch),
            max_workers=self.max_workers,
        )

    def score_action_plans_canonical_event(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        input_history: List[Dict[str, Any]],
        action_plans: List[List[Any]],
        goal_text_override: Optional[str] = None,
        score_config: Any = None,
    ) -> List[Dict[str, Any]]:
        return _score_action_plans_with_canonical_event_generator(
            generate_message_batch=self._generate_message_batch,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_history=input_history,
            action_plans=action_plans,
            score_config=score_config,
        )


def _score_action_plans_with_canonical_event_generator(
    *,
    generate_message_batch: Any,
    system_prompt: str,
    user_prompt: str,
    input_history: List[Dict[str, Any]],
    action_plans: List[List[Any]],
    score_config: Any = None,
) -> List[Dict[str, Any]]:
    from ejepa_wm.backends._ewm_canonical_event_scoring import (
        CanonicalEventScoreConfig,
        finalize_scored_plans,
    )

    if not action_plans:
        return []
    config = score_config or CanonicalEventScoreConfig()
    num_plans = len(action_plans)
    trajectories: List[List[Dict[str, Dict[str, float]]]] = [[] for _ in range(num_plans)]
    terminal_prob_steps: List[List[float]] = [[] for _ in range(num_plans)]
    histories: List[List[Dict[str, Any]]] = [list(input_history or []) for _ in range(num_plans)]
    max_len = max(len(plan) for plan in action_plans)
    for step in range(max_len):
        active = [i for i, plan in enumerate(action_plans) if step < len(plan)]
        if not active:
            break
        messages_batch = [
            build_canonical_event_prompt(
                system_prompt, user_prompt, histories[i], action_plans[i][step],
            )
            for i in active
        ]
        raw_completions = generate_message_batch(messages_batch)
        events = [parse_canonical_event_completion(text) for text in raw_completions]
        probs_rows = [canonical_event_to_field_probs(event) for event in events]
        for local_index, plan_index in enumerate(active):
            event = events[local_index]
            trajectories[plan_index].append(probs_rows[local_index])
            terminal_prob_steps[plan_index].append(terminal_probability_from_event(event))
            histories[plan_index] = histories[plan_index] + [
                {
                    "step": len(histories[plan_index]) + 1,
                    "action": action_plans[plan_index][step],
                    "observation": canonical_event_completion_text(event),
                }
            ]
    scored = finalize_scored_plans(
        trajectories,
        action_plans,
        config,
        task_text=user_prompt,
        input_history=input_history,
    )
    for record in scored:
        index = int(record.get("plan_index", record.get("index", -1)))
        probs = terminal_prob_steps[index] if 0 <= index < len(terminal_prob_steps) else []
        record["per_step_terminal_prob"] = probs
        record["terminal_probability"] = probs[-1] if probs else None
    return scored
