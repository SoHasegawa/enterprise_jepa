"""Core contracts for the pluggable World Model (WM).

A World Model is a benchmark-agnostic helper that an agent/executor consults at its
decision points. Usage *strategies* are selected by ``WMConfig.strategy``:

* ``prompt_injection`` — the WM produces a short guidance block (:meth:`WorldModel.advise`)
  that the policy prompt is augmented with (transiently) to steer the next action.
* ``itp_i`` — the training-free inference harness from Imagine-Then-Plan: the existing
  policy adaptively selects a lookahead depth, a generative WM imagines that many steps,
  and the existing policy reflects on the imagined trajectory before its real action.
* ``revision`` — predict the proposed action's canonical next state, then ask the policy to
  proceed with or revise that action before execution.
* ``reference`` — predict the selected action's canonical next state and expose that prediction
  to the policy only at its following decision step.
* ``selection`` — at each step the agent samples ``n`` candidate actions and the WM picks
  one of them (:meth:`WorldModel.select`). (Historically called ``best_of_n``.)
* ``none`` — the :class:`~ejepa_wm.backends.noop.NoopWorldModel` baseline: ``advise`` returns
  ``""`` and ``select`` returns ``0`` (first candidate / plain agent behaviour). Used as the
  "without-WM" arm of the with/without-WM evaluation.

The WM *backend* (how it actually decides) is independent of the strategy and is swappable:
an LLM, a served OpenAI-compatible endpoint, the JEPA net, etc. Backends live in
``ejepa_wm.backends`` and are constructed by :func:`ejepa_wm.factory.build_world_model`.

Methods are synchronous and free of any heavy/async dependency so the module imports cleanly
in any benchmark; async callers (e.g. ``wm_react``) wrap them in ``run_in_executor``.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# A provider-agnostic chat function: takes OpenAI-style messages, returns the reply text.
ChatFn = Callable[[list[dict[str, str]]], str]

# Canonical strategy names (+ legacy alias).
# ``beam_plan`` is the JEPA-only MPC lookahead strategy (see backends.ewm_imagined): every
# few steps the WM scores m*n imagined actions with the canonical-event heads and hands the
# agent a plan to follow. It requires the ewm_imagined + JEPA canonical-event backend; with
# any other backend an executor should fall back to plain agent behaviour.
# ``hier_latent_cem`` is the same JEPA-only MPC lookahead, but the discrete "LLM proposes m
# candidates per horizon step" search is replaced by a hierarchical latent-action
# Cross-Entropy Method: one LLM call opens the search space with K anchors, then thousands of
# continuous latent-action trajectories are sampled/refined LLM-free (see
# backends._ewm_hier_cem). Same requirements/fallback as beam_plan.
STRATEGIES = (
    "none",
    "prompt_injection",
    "selection",
    "itp_i",
    "revision",
    "reference",
    "beam_plan",
    "hier_latent_cem",
)
_STRATEGY_ALIASES = {
    "best_of_n": "selection",
    "itp-i": "itp_i",
    "itpi": "itp_i",
    "imagine_then_plan": "itp_i",
}


def normalize_strategy(value: Any) -> str:
    """Map raw strategy text to a canonical name (``best_of_n`` -> ``selection``)."""
    s = (str(value or "none")).strip().lower()
    s = _STRATEGY_ALIASES.get(s, s)
    return s if s in STRATEGIES else "none"


def norm_state(name: Any) -> str:
    """Whitespace-insensitive, lowercased state key."""
    return re.sub(r"\s+", "", str(name)).lower()


@dataclass
class WMConfig:
    """Resolved World-Model configuration (from CLI args / ``WM_*`` env)."""

    strategy: str = "none"  # none | prompt_injection | selection
    backend: str = "noop"  # noop | llm | served | ewm_predict | ewm_imagined
    model: str | None = None  # WM model id (LLM backends)
    n: int = 1  # candidates sampled per step (selection)
    target_state: str | None = None
    inject_template: str = "current"  # current | history | distribution
    timeout: float = 600.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.strategy = normalize_strategy(self.strategy)
        self.backend = (self.backend or "noop").strip().lower()
        self.n = max(1, int(self.n or 1))


@dataclass
class AdviseResult:
    """Outcome of :meth:`WorldModel.advise`."""

    text: str = ""  # guidance block ("" => inject nothing this step)
    detail: dict[str, Any] = field(default_factory=dict)  # telemetry (state, scores, ...)


@dataclass
class SelectResult:
    """Outcome of :meth:`WorldModel.select`."""

    index: int = 0  # chosen candidate index
    detail: dict[str, Any] = field(default_factory=dict)  # telemetry (per-candidate scores)


class WorldModel(ABC):
    """The plug-in contract every WM backend implements."""

    name: str = "base"

    def __init__(self, config: WMConfig) -> None:
        self.config = config

    @abstractmethod
    def advise(
        self, conversation_flow: list[dict[str, Any]], *, history: list[str] | None = None
    ) -> AdviseResult:
        """Return guidance for the next step (``prompt_injection`` strategy)."""

    @abstractmethod
    def select(
        self, conversation_flow: list[dict[str, Any]], candidate_events: list[dict[str, Any]]
    ) -> SelectResult:
        """Pick one of ``candidate_events`` (``selection`` strategy)."""


def render_flow(conversation_flow: list[dict[str, Any]], *, max_chars: int = 1200) -> str:
    """Render a ``conversation_flow`` into compact text for an LLM-as-WM prompt."""
    lines: list[str] = []
    for ev in conversation_flow or []:
        etype = ev.get("type")
        if etype == "system_message":
            lines.append(f"[system] {ev.get('content', '')}")
        elif etype == "user_message":
            lines.append(f"[user] {ev.get('content', '')}")
        elif etype == "ai_message":
            content = ev.get("content", "")
            calls = ev.get("tool_calls") or []
            call_txt = "; ".join(f"{c.get('name')}({c.get('args')})" for c in calls)
            lines.append(f"[assistant] {content}" + (f" -> tools: {call_txt}" if call_txt else ""))
        elif etype == "tool_result":
            res = ev.get("result")
            lines.append(f"[tool:{ev.get('tool_name')}] {str(res)[:max_chars]}")
    return "\n".join(lines)


def candidate_text(event: dict[str, Any]) -> str:
    """One-line summary of a candidate action for the selection prompt."""
    content = (event.get("content") or "").strip()
    calls = event.get("tool_calls") or []
    if calls:
        call_txt = "; ".join(f"{c.get('name')}({c.get('args')})" for c in calls)
        return (content + " " if content else "") + f"CALL {call_txt}"
    return content or "(no action / final answer)"


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
