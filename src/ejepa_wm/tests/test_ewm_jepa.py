"""Tests for the JEPA world-model path in the ``ewm_imagined`` replay backend.

torch/transformers (and a real checkpoint + GPU) are needed to actually run the
JEPA net, so these tests exercise everything *around* it without loading torch:

* the polymorphic ``predict_feedback`` seam in ``_ewm_runtime.predict_wm_feedback``
  (a fake world model that exposes ``predict_feedback`` short-circuits the text path); and
* the ``ewm_imagined`` → JEPA generator wiring (a fake ``_ewm_jepa`` module is
  injected so the whole imagined-trajectory rollout runs against a stub JEPA WM).

The pure ported helpers in ``_ewm_jepa`` are covered separately, guarded by
``importorskip('torch')`` since the module imports torch at top.
"""
from __future__ import annotations

import sys
import types

import pytest

from ejepa_wm.backends import _ewm_runtime as ewm
from ejepa_wm.factory import build_world_model, wm_config_from_env

# --- the polymorphic predict_feedback seam (no torch) ---------------------------


class _FakeStructuredWM:
    """A world model that answers via the structured ``predict_feedback`` seam."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict_feedback(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "tool_calls": kwargs["planned_calls"],
                "predicted_success": True,
                "predicted_state": None,
                "predicted_tool_output": "imagined tool output",
                "predicted_error_message": "",
                "world_model_backend": "fake_jepa",
            }
        ]


def test_predict_wm_feedback_prefers_predict_feedback():
    wm = _FakeStructuredWM()
    calls = [{"name": "search_users", "arguments": {"q": "bob"}}]
    out = ewm.predict_wm_feedback(
        wm,
        "SYS",
        "USER",
        {"success": True},           # previous_state
        calls,                       # planned_calls
        [{"step": 0, "state": {}}],  # state_history
        "binary_error",              # wm_state
        interaction_index=2,
    )
    assert out[0]["world_model_backend"] == "fake_jepa"
    assert out[0]["predicted_tool_output"] == "imagined tool output"
    # The ejepa_wm positional args are forwarded as keywords to predict_feedback.
    fwd = wm.calls[0]
    assert fwd["system_prompt"] == "SYS"
    assert fwd["user_prompt"] == "USER"
    assert fwd["planned_calls"] == calls
    assert fwd["state_history"] == [{"step": 0, "state": {}}]
    assert fwd["interaction_index"] == 2


def test_predict_wm_feedback_text_path_when_no_predict_feedback():
    """A generator with only generate_from_messages still uses the text WM path."""

    class _TextWM:
        def __init__(self) -> None:
            self.seen = 0

        def generate_from_messages(self, messages, temperature: float = 0.0) -> str:
            self.seen += 1
            return "1"  # binary_error success

    wm = _TextWM()
    out = ewm.predict_wm_feedback(
        wm, "SYS", "USER", ewm.blank_state(), [{"name": "t", "arguments": {}}], [], "binary_error"
    )
    assert wm.seen == 1
    assert out[0]["predicted_success"] is True


# --- ewm_imagined wiring to the JEPA generator (no torch, fake module) ----------


class _FakeJepaGenerator:
    """Stand-in for ``_ewm_jepa.JepaEwmGenerator`` — records construction + calls."""

    last_instance: "_FakeJepaGenerator | None" = None

    def __init__(self, model_path, **kwargs) -> None:
        self.model_path = model_path
        self.kwargs = kwargs
        self.feedback_calls: list[dict] = []
        self.imagined_observation_backend_resolved = "success"
        type(self).last_instance = self

    def predict_feedback(self, **kwargs):
        self.feedback_calls.append(kwargs)
        return [
            {
                "tool_calls": kwargs["planned_calls"],
                "predicted_success": True,
                "predicted_success_probability": 0.9,
                "predicted_state": None,
                "predicted_tool_output": "[jepa] predicted success",
                "predicted_error_message": "",
                "world_model_backend": "text_leworldmodel_jepa_success_classifier",
            }
        ]


@pytest.fixture()
def fake_jepa_module(monkeypatch):
    """Inject a torch-free fake ``ejepa_wm.backends._ewm_jepa`` so the lazy import in
    ``ewm_imagined`` resolves to our stub instead of pulling in torch."""
    mod = types.ModuleType("ejepa_wm.backends._ewm_jepa")
    mod.JepaEwmGenerator = _FakeJepaGenerator
    monkeypatch.setitem(sys.modules, "ejepa_wm.backends._ewm_jepa", mod)
    _FakeJepaGenerator.last_instance = None
    return mod


def _chat_fn_factory():
    """A policy LLM that proposes one thought then one tool call (then would stop)."""
    state = {"n": 0}

    def chat_fn(messages):
        state["n"] += 1
        # Odd call = THINK NODE, even call = ACTION NODE (think precedes action).
        if state["n"] % 2 == 1:
            return '{"thought": "look up the user before acting"}'
        return '{"action": "search_users", "action_input": {"q": "bob"}}'

    return chat_fn


def test_factory_selects_jepa_when_checkpoint_set(monkeypatch, fake_jepa_module):
    monkeypatch.setenv("WM_STRATEGY", "imagined")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/jepa/ckpt")
    monkeypatch.setenv("WM_IMAGINED_MAX_STEPS", "1")
    cfg = wm_config_from_env()
    assert cfg.strategy == "prompt_injection"
    assert cfg.backend == "ewm_imagined"

    wm = build_world_model(cfg, chat_fn=_chat_fn_factory())
    # The JEPA generator was constructed with our checkpoint + resolved defaults.
    inst = _FakeJepaGenerator.last_instance
    assert inst is not None
    assert inst.model_path == "/fake/jepa/ckpt"
    assert inst.kwargs["imagined_observation_backend"] == "auto"


def test_jepa_imagined_advise_runs_rollout(monkeypatch, fake_jepa_module):
    monkeypatch.setenv("WM_STRATEGY", "imagined")
    monkeypatch.setenv("WM_EWM_BACKEND", "jepa")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/jepa/ckpt")
    monkeypatch.setenv("WM_IMAGINED_MAX_STEPS", "1")
    monkeypatch.setenv("WM_JEPA_OBSERVATION_BACKEND", "success")

    wm = build_world_model(wm_config_from_env(), chat_fn=_chat_fn_factory())

    # Under the executor-agnostic contract the tool catalog travels in the flow.
    flow = [
        {"type": "system_message", "content": "You are an enterprise ops agent."},
        {"type": "tools", "tools": [{"name": "search_users", "description": "look up users", "inputSchema": {}}]},
        {"type": "user_message", "content": "Find user bob and disable his account."},
    ]
    result = wm.advise(flow)

    # One imagined step was produced, driven by the JEPA world model's prediction.
    assert result.detail["imagined_step_count"] == 1
    assert "[IMAGINED_TRAJECTORY_FOR_PLANNING_ONLY]" in result.text
    assert "search_users" in result.text
    inst = _FakeJepaGenerator.last_instance
    assert len(inst.feedback_calls) == 1
    assert inst.feedback_calls[0]["planned_calls"][0]["name"] == "search_users"


def test_backend_jepa_requires_checkpoint(monkeypatch, fake_jepa_module):
    monkeypatch.setenv("WM_STRATEGY", "imagined")
    monkeypatch.setenv("WM_EWM_BACKEND", "jepa")
    monkeypatch.delenv("WM_EWM_JEPA_CHECKPOINT", raising=False)
    with pytest.raises(ValueError, match="WM_EWM_JEPA_CHECKPOINT"):
        build_world_model(wm_config_from_env(), chat_fn=_chat_fn_factory())


# --- pure ported helpers (need torch only to import the module) -----------------


def test_jepa_pure_helpers():
    pytest.importorskip("torch")
    from ejepa_wm.backends import _ewm_jepa as jepa

    # strip_model_thinking_output removes reasoning + residual special tokens.
    assert jepa.strip_model_thinking_output("<think>hmm</think>ok") == "ok"
    assert jepa.strip_model_thinking_output("a<eos>", special_tokens=["<eos>"]) == "a"

    # canonical-event label reconstruction maps execution_status onto the ternary.
    state = jepa.reconstruct_state_from_canonical_event_labels(
        {"execution_status": "failure", "error_signature": "timeout"}, tool_name="search_users"
    )
    ctx = state["state"]["context"]
    assert ctx["last_tool_execution_result"] == -1
    assert ctx["last_tool_name"] == "search_users"
    assert "timeout" in ctx["error_message"]

    obs = jepa.build_canonical_event_observation_payload({"execution_status": "success"})
    assert obs["tool_outcome"]["success"] is True

    # render_raw_replay_history renders {step, state} entries with observation=state.
    rendered = jepa.render_raw_replay_history([{"step": 0, "state": {"ok": True}}], max_chars=200)
    assert "ok" in rendered
