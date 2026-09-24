"""Regression test for a real bug found while auditing hard-override telemetry: beam_plan_step
(and hier_latent_cem_step) rebind the local ``seed_norm`` variable to the WM's recommended calls
when ``override_applied`` fires, then reused that SAME variable for BOTH ``detail["calls"]``
(correct -- it's what actually executes) AND ``detail["agent_calls"]`` (wrong -- this must stay
the agent's own original proposal, or the telemetry can never show whether an override actually
changed the executed action; it always reads as a no-op post-hoc, indistinguishable from a
genuine coincidental match). No torch: score_action_plans_canonical_event is faked."""
from __future__ import annotations

import json

from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel
from ejepa_wm.base import WMConfig


class _FakeJepaGenerator:
    """Stands in for JepaEwmGenerator: only score_action_plans_canonical_event is exercised by
    beam_plan_step's horizon loop for this test's single-depth plan."""

    def score_action_plans_canonical_event(self, *, system_prompt, user_prompt, input_history, action_plans, score_config=None):
        records = []
        for index, plan in enumerate(action_plans):
            is_alt = index != 0  # plan_index 0 is always the seed (the agent's own action)
            records.append({
                "plan": plan,
                "plan_index": index,
                "score": 5.0 if is_alt else 0.1,
                "normalized_score": 1.0 if is_alt else 0.1,
                "vetoed": False,
                "reason": "fake",
                "predicted_state": {},
            })
        return records


def _make_model(*, hard_override: bool) -> EwmImaginedWorldModel:
    def chat_fn(messages) -> str:
        # The "diverse candidates" LLM call for beam_plan's horizon step: always propose ONE
        # alternative action, genuinely different from the seed.
        return json.dumps([{"name": "create_ticket", "arguments": {"priority": "high"}}])

    model = EwmImaginedWorldModel(WMConfig(), chat_fn=chat_fn)
    model._wm = _FakeJepaGenerator()
    model.beam_hard_override = hard_override
    model.beam_horizon = 1  # single depth: only need one scored round to decide
    return model


def _run_beam_plan_step(model):
    conversation_flow = [
        {"type": "system_message", "content": "You are an agent."},
        {"type": "user_message", "content": "Do the task."},
    ]
    seed_calls = [{"name": "list_labels", "arguments": {}}]
    result = model.beam_plan_step(conversation_flow, seed_calls=seed_calls, user_query="Do the task.")
    return result.detail


def test_hard_override_replaces_calls_but_preserves_agent_calls():
    detail = _run_beam_plan_step(_make_model(hard_override=True))
    assert detail["override_applied"] is True
    assert detail["calls"] == [{"name": "create_ticket", "arguments": {"priority": "high"}}]
    # The bug: agent_calls used to be reassigned to the SAME value as calls whenever an
    # override fired, making it impossible to tell the override changed anything at all.
    assert detail["agent_calls"] == [{"name": "list_labels", "arguments": {}}]
    assert detail["agent_calls"] != detail["calls"]


def test_advisory_mode_never_overrides_and_agent_calls_matches_calls():
    detail = _run_beam_plan_step(_make_model(hard_override=False))
    assert detail["override_applied"] is False
    # In advisory mode calls == agent_calls by construction (the agent's own action always
    # executes) -- this is the ONE case where the two fields legitimately coincide.
    assert detail["calls"] == [{"name": "list_labels", "arguments": {}}]
    assert detail["agent_calls"] == detail["calls"]
