"""LLM-as-World-Model backend.

Given a provider-agnostic chat callable (:data:`~ejepa_wm.base.ChatFn`), this backend uses a
language model to (a) advise the agent (``prompt_injection``) and (b) pick the best of K
candidate actions (``selection``). It is the backend behind the "served model as both agent and
WM" example; that chat callable is built in :mod:`ejepa_wm.backends.served`.

All prompt text is loaded from the centralized store (``prompts/llm/*.md``).
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ejepa_wm.base import (
    AdviseResult,
    ChatFn,
    SelectResult,
    WMConfig,
    WorldModel,
    candidate_text,
    render_flow,
)
from ejepa_wm.prompts import load_prompt

logger = logging.getLogger(__name__)

_INT_RE = re.compile(r"-?\d+")


class LlmWorldModel(WorldModel):
    """Uses an injected chat function as the World Model."""

    name = "llm"

    def __init__(self, config: WMConfig, chat_fn: ChatFn) -> None:
        super().__init__(config)
        self._chat = chat_fn

    def advise(
        self, conversation_flow: list[dict[str, Any]], *, history: list[str] | None = None
    ) -> AdviseResult:
        hist = ""
        if self.config.inject_template == "history" and history:
            hist = "\nStates visited so far: " + " -> ".join(history) + "\n"
        prompt = load_prompt("llm", "advise", flow=render_flow(conversation_flow), history=hist)
        try:
            text = (self._chat([{"role": "user", "content": prompt}]) or "").strip()
        except Exception as exc:  # provider failure → no guidance this step
            logger.warning("ejepa_wm.llm: advise call failed (%s); no guidance this step", exc)
            return AdviseResult(text="", detail={"backend": "llm", "error": str(exc)})
        return AdviseResult(text=text, detail={"backend": "llm"})

    def select(
        self, conversation_flow: list[dict[str, Any]], candidate_events: list[dict[str, Any]]
    ) -> SelectResult:
        n = len(candidate_events)
        if n <= 1:
            return SelectResult(index=0, detail={"backend": "llm", "n_candidates": n})
        listing = "\n".join(f"[{i}] {candidate_text(ev)}" for i, ev in enumerate(candidate_events))
        prompt = load_prompt(
            "llm", "select", flow=render_flow(conversation_flow), n=n, candidates=listing
        )
        try:
            reply = self._chat([{"role": "user", "content": prompt}]) or ""
        except Exception as exc:
            logger.warning("ejepa_wm.llm: select call failed (%s); falling back to candidate 0", exc)
            return SelectResult(index=0, detail={"backend": "llm", "error": str(exc)})
        idx = self._parse_index(reply, n)
        return SelectResult(
            index=idx, detail={"backend": "llm", "n_candidates": n, "raw": reply.strip()[:200]}
        )

    @staticmethod
    def _parse_index(reply: str, n: int) -> int:
        """Extract a 0-based index from the model reply; clamp to range, default 0."""
        match = _INT_RE.search(reply or "")
        if not match:
            return 0
        idx = int(match.group())
        return idx if 0 <= idx < n else 0
