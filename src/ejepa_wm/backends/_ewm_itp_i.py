"""Prompt and parsing helpers for the training-free ITP-I inference harness.

The control flow follows the released Imagine-Then-Plan ITP-I evaluator: choose an
adaptive lookahead K, ask the world model for one K-step imagined trajectory, then let
the policy reflect on that trajectory as implicit feedback before acting.  The prompt
structure is adapted from loyiv/ITP (MIT); this module does not implement its policy
training.
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_lookahead(raw: str, *, max_k: int, fallback: int = 1) -> int:
    """Parse the first integer and clamp it to ``[0, max_k]`` like ITP's evaluator."""
    match = re.search(r"-?\d+", raw or "")
    value = int(match.group()) if match else min(max_k, fallback)
    return max(0, min(max_k, value))


def decision_messages(task: str, history: str, *, max_k: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Choose how many future interaction steps a world model should imagine "
                "before the next real action. Harder, uncertain, or risky states merit more "
                f"lookahead. Return exactly one integer from 0 through {max_k}, with no other text."
            ),
        },
        {
            "role": "user",
            "content": f"Task:\n{task}\n\nCurrent interaction history:\n{history or '(none)'}",
        },
    ]


def imagination_messages(
    task: str,
    history: str,
    tools: list[Any],
    *,
    k: int,
) -> list[dict[str, str]]:
    tool_names = [
        str(tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", ""))
        for tool in tools
    ]
    catalog = ", ".join(name for name in tool_names if name) or "(not provided)"
    return [
        {
            "role": "system",
            "content": (
                "You are a world model simulating an interactive tool environment. Predict "
                "plausible future agent actions and the environment observations they cause. "
                "Do not claim these imagined actions have actually happened."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\n{task}\n\nAvailable tool names:\n{catalog}\n\n"
                f"Observed interaction history:\n{history or '(none)'}\n\n"
                f"Imagine the next {k} action-observation steps. Number the steps, include "
                "important arguments and likely outputs/errors, and remain concise."
            ),
        },
    ]


def canonical_action_step_messages(
    task: str,
    history: str,
    tools: list[Any],
    *,
    imagined_prefix: str,
    step: int,
    k: int,
) -> list[dict[str, str]]:
    """Ask the policy for the next action in an alternating policy/JEPA rollout.

    Unlike a generative LLM world model, JEPA predicts the next canonical state only after it
    receives a proposed action. Later actions are conditioned on earlier imagined canonical
    states, matching ITP's step-by-step mental-sandbox interaction.
    """
    return [
        {
            "role": "system",
            "content": (
                "Propose a hypothetical tool-action trajectory for a world model to simulate. "
                "Return only a JSON array. Do not claim any action was executed."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\n{task}\n\nAvailable tools (JSON):\n"
                f"{json.dumps(tools, ensure_ascii=False, default=str)}\n\n"
                f"Observed interaction history:\n{history or '(none)'}\n\n"
                f"Imagined trajectory so far:\n{imagined_prefix or '(none)'}\n\n"
                f"Propose hypothetical action {step} of {k}. Return exactly one JSON object: "
                '{"name":"<tool>","arguments":{...}}. Use only available tools. Do not '
                "invent identifiers that are absent from the real or imagined state."
            ),
        },
    ]
def reflection_guidance(foresight: str, *, k: int) -> str:
    if k <= 0:
        imagined = "No lookahead was requested for this state."
    else:
        imagined = foresight.strip() or "The world model returned no usable foresight."
    return (
        "[ITP_I_IMPLICIT_WORLD_MODEL_FEEDBACK]\n"
        f"Adaptive lookahead: K={k}\n"
        f"Imagined trajectory (hypothetical, not executed):\n{imagined}\n\n"
        "Reflect on the task, the real interaction history, and this hypothetical trajectory. "
        "Identify progress, risks, contradictions, and the next bottleneck. Then choose the next "
        "real action using the available tools. Do not blindly copy the first imagined action, "
        "and do not treat imagined outputs as observed facts.\n"
        "[/ITP_I_IMPLICIT_WORLD_MODEL_FEEDBACK]"
    )
