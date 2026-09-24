"""One-shot prompt-based K decider used by the ``react_wm_decide_k`` mode.

The variant lets the agent's own LLM (e.g. GPT-5.1) pick the foresight depth
``K`` for the next turn via a tightly-scoped prompt, without any extra trained
model — the ``react_wm_decide_k`` mode.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


_DECIDE_K_SYSTEM_PROMPT = (
    "You are an enterprise-operations agent's foresight planner. "
    "Given the current chat context and the agent's options, decide how many "
    "steps to look ahead with a learned world model BEFORE you pick the next "
    "tool call.\n\n"
    "Output a single JSON object with the key `k` whose value is an integer "
    "between 0 and {kmax} inclusive.\n"
    "* k=0 means: do NOT call the world model -- act immediately.\n"
    "* k>=1 means: ask the world model to imagine the next k steps and use "
    "that foresight to inform the next tool call.\n\n"
    "Pick a larger k when the task is long-horizon, irreversible, or has "
    "subtle multi-step dependencies; pick a smaller k when the next call is "
    "obvious or cheap to retry."
)


_DECIDE_K_USER_TEMPLATE = (
    "Conversation so far:\n{conversation}\n\n"
    "Most recent observed state summary:\n{state_summary}\n\n"
    "Respond with JSON ONLY in the form {{\"k\": <int>}}. /no_think"
)


_K_PATTERN = re.compile(r'"k"\s*:\s*(-?\d+)')


def _render_recent_conversation(
    conversation: list[dict[str, Any]], max_chars: int = 4000
) -> str:
    """Render the last few messages compactly so the decider sees recent context."""
    parts: list[str] = []
    for message in conversation[-12:]:
        role = message.get("role", "?")
        content = message.get("content")
        if isinstance(content, (dict, list)):
            content_text = json.dumps(content, ensure_ascii=False, default=str)
        else:
            content_text = str(content or "")
        parts.append(f"[{role}] {content_text}")
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _render_state_summary(state: Any) -> str:
    if state is None:
        return "(no state)"
    try:
        return json.dumps(state, ensure_ascii=False, default=str)[:2000]
    except Exception:  # noqa: BLE001
        return str(state)[:2000]


def _parse_k_response(raw: str, kmax: int) -> int | None:
    """Best-effort extract of an integer ``k`` value in [0, kmax]."""
    raw = raw.strip()
    if not raw:
        return None
    raw = raw.split("</think>", 1)[-1].strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw).rstrip("`").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "k" in obj:
            return max(0, min(int(obj["k"]), kmax))
    except Exception:  # noqa: BLE001
        pass
    match = _K_PATTERN.search(raw)
    if match:
        try:
            return max(0, min(int(match.group(1)), kmax))
        except Exception:  # noqa: BLE001
            return None
    bare = re.search(r"\b(\d+)\b", raw)
    if bare:
        try:
            return max(0, min(int(bare.group(1)), kmax))
        except Exception:  # noqa: BLE001
            return None
    return None


def decide_k_via_agent(
    *,
    agent_generator: Any,
    conversation: list[dict[str, Any]],
    state: Any,
    kmax: int,
    fallback_k: int = 0,
) -> int:
    """Ask the action LLM how many steps to look ahead next turn.

    Returns an integer in ``[0, kmax]``. Falls back to ``fallback_k`` on any
    parse failure so the runtime stays robust to flaky LLM outputs.
    """
    from src.evaluation import _generate_with_optional_temperature

    kmax = max(0, int(kmax))
    if kmax == 0:
        return 0

    messages = [
        {
            "role": "system",
            "content": _DECIDE_K_SYSTEM_PROMPT.format(kmax=kmax),
        },
        {
            "role": "user",
            "content": _DECIDE_K_USER_TEMPLATE.format(
                conversation=_render_recent_conversation(conversation),
                state_summary=_render_state_summary(state),
            ),
        },
    ]
    try:
        raw = _generate_with_optional_temperature(
            agent_generator, messages, temperature=0.0
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("decide_k_via_agent generation failed: %s", exc)
        return max(0, min(fallback_k, kmax))
    parsed = _parse_k_response(raw, kmax)
    if parsed is None:
        logger.debug("decide_k_via_agent: unparseable response %r", raw[:200])
        return max(0, min(fallback_k, kmax))
    return parsed


__all__ = ["decide_k_via_agent"]
