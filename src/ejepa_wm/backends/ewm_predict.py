"""``ewm_predict`` World-Model backend — direct single-step binary+error feasibility.

**No MCP server.** The Enterprise World Model runs in-process via the vLLM or transformers
backend (:func:`~ejepa_wm.backends._ewm_generators.build_ewm_generator`) and predicts the
binary tool-execution result (+ error message) of a candidate action against the enterprise
state reconstructed from the conversation flow.

It plugs into the ``selection`` strategy (``wm_react``): each step the agent samples ``WM_N``
candidate actions and this backend scores each with the WM, picking one the WM predicts will
succeed. This is the "EWM checks the feasibility of each tool call" use — one concrete step at
a time, state-aware — without exposing EWM as an MCP tool the agent must remember to call.

Selected via ``WM_STRATEGY=selection WM_BACKEND=ewm_predict`` (with ``WM_N>1`` and a sampling
temperature so the candidates differ). The world model is configured exactly like the imagined
backend and the EWM MCP server: ``EWM_WORLD_MODEL_METHOD=vllm/<model>`` (+ ``WM_VLLM_SERVER_PORT``
/ ``WM_VLLM_BASE_URL``) or a local checkpoint via ``EWM_WORLD_MODEL_PATH``. ``WM_STATE`` selects
the target mode (``binary_error`` default | ``binary_error_stage`` | ``tool_output``).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from ejepa_wm.backends import _ewm_runtime as ewm
from ejepa_wm.backends._ewm_generators import build_ewm_generator, normalize_planned_calls
from ejepa_wm.base import AdviseResult, SelectResult, WMConfig, WorldModel

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _prompt_from_flow(conversation_flow: list[dict[str, Any]], etype: str) -> str:
    for event in conversation_flow or []:
        if isinstance(event, dict) and event.get("type") == etype:
            return str(event.get("content", "") or "")
    return ""


class EwmPredictWorldModel(WorldModel):
    """selection backend: pick the candidate action the EWM predicts will succeed."""

    name = "ewm_predict"

    def __init__(self, config: WMConfig) -> None:
        super().__init__(config)
        self.generator = build_ewm_generator()
        self.wm_state = (os.getenv("WM_STATE") or ewm.WM_STATE_BINARY_ERROR).strip().lower()
        if self.wm_state not in ewm.WM_STATES:
            logger.warning(
                "ewm_predict: unknown WM_STATE=%s; using %s",
                self.wm_state, ewm.WM_STATE_BINARY_ERROR,
            )
            self.wm_state = ewm.WM_STATE_BINARY_ERROR
        self.state_history_size = _env_int("WM_STATE_HISTORY_SIZE", 3)
        logger.info("ewm_predict: wm_state=%s history=%d", self.wm_state, self.state_history_size)

    def _predict(
        self, conversation_flow: list[dict[str, Any]], planned_calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        previous_state, state_history = ewm.enterprise_state_from_flow(
            conversation_flow, max_items=self.state_history_size
        )
        feedback = ewm.predict_wm_feedback(
            self.generator,
            _prompt_from_flow(conversation_flow, "system_message"),
            _prompt_from_flow(conversation_flow, "user_message"),
            previous_state,
            normalize_planned_calls(planned_calls),
            state_history,
            self.wm_state,
            interaction_index=max(0, len(state_history) - 1),
        )
        return feedback[0]

    def select(
        self, conversation_flow: list[dict[str, Any]], candidate_events: list[dict[str, Any]]
    ) -> SelectResult:
        predictions: list[dict[str, Any]] = []
        for event in candidate_events or []:
            calls = event.get("tool_calls") or []
            if not calls:
                # A final-answer candidate has no action to score; leave it unranked.
                predictions.append({"action": "final_answer", "predicted_success": None})
                continue
            try:
                feedback = self._predict(conversation_flow, calls)
                predictions.append(
                    {
                        "tool": normalize_planned_calls(calls)[0]["name"],
                        "predicted_success": bool(feedback.get("predicted_success")),
                        "error_message": feedback.get("predicted_error_message") or "",
                        "raw_prediction": feedback.get("raw_prediction") or "",
                    }
                )
            except Exception as exc:  # a transient WM error must not fail the step
                logger.warning("ewm_predict: candidate prediction failed (%s)", exc)
                predictions.append({"predicted_success": None, "error": str(exc)})

        # Prefer the first candidate the WM predicts will succeed; else keep the first.
        index = next(
            (i for i, p in enumerate(predictions) if p.get("predicted_success") is True), 0
        )
        logger.info(
            "ewm_predict: chose candidate %d/%d (wm_state=%s)",
            index, len(predictions), self.wm_state,
        )
        return SelectResult(
            index=index,
            detail={"backend": "ewm_predict", "wm_state": self.wm_state, "candidates": predictions},
        )

    def advise(self, conversation_flow, *, history=None) -> AdviseResult:
        # ewm_predict is a selection backend; it scores candidate actions rather than
        # producing a prompt-injection guidance block.
        return AdviseResult(
            text="", detail={"backend": "ewm_predict", "unsupported": "advise"}
        )


__all__ = ["EwmPredictWorldModel"]
