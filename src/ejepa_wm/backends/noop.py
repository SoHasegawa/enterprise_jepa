"""The no-WM baseline backend (``strategy=none``)."""
from __future__ import annotations

from typing import Any

from ejepa_wm.base import AdviseResult, SelectResult, WorldModel


class NoopWorldModel(WorldModel):
    """Returns no guidance and always picks the first candidate.

    This is the "without-WM" arm of the with/without-WM evaluation: it runs through the same
    code path as a real WM so the only difference between arms is the model itself.
    """

    name = "noop"

    def advise(
        self, conversation_flow: list[dict[str, Any]], *, history: list[str] | None = None
    ) -> AdviseResult:
        return AdviseResult(text="", detail={"backend": "noop"})

    def select(
        self, conversation_flow: list[dict[str, Any]], candidate_events: list[dict[str, Any]]
    ) -> SelectResult:
        return SelectResult(index=0, detail={"backend": "noop", "n_candidates": len(candidate_events)})
