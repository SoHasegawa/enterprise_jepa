"""Tests for the ``hier_latent_cem`` strategy — the hierarchical latent-action CEM MPC
controller in ``ewm_imagined.py``. Torch-free: a fake ``_ewm_hier_cem`` module (mirroring
``fake_jepa_module`` in ``test_ewm_beam_plan.py``) stands in for the real
``hierarchical_cem_plan``, so the cooldown/follow/replan state machine, confidence gating,
and advisory/hard-override semantics are exercised without loading torch.
"""
from __future__ import annotations

import sys
import types

import pytest

from ejepa_wm.base import STRATEGIES, normalize_strategy
from ejepa_wm.factory import wm_config_from_env


def test_hier_latent_cem_is_a_known_strategy():
    assert "hier_latent_cem" in STRATEGIES
    assert normalize_strategy("hier_latent_cem") == "hier_latent_cem"


def test_hier_latent_cem_defaults_to_ewm_imagined_backend(monkeypatch):
    monkeypatch.setenv("WM_STRATEGY", "hier_latent_cem")
    monkeypatch.delenv("WM_BACKEND", raising=False)
    cfg = wm_config_from_env()
    assert cfg.strategy == "hier_latent_cem"
    assert cfg.backend == "ewm_imagined"


class _FakeJepaGenerator:
    """Stand-in for a JepaEwmGenerator exposing what hier_latent_cem_step needs: model/
    tokenizer/canonical_event_vocab + the text-assembly helpers `_ewm_jepa` provides."""

    def __init__(self, *, canonical_event_available=True):
        self.model = object()
        self.tokenizer = object()
        self.canonical_event_vocab = {}
        self.canonical_event_available = canonical_event_available
        self.max_input_length = 2048
        self.max_action_length = 512

    def _context_text(self, system_prompt, user_prompt):
        return f"ctx:{system_prompt}|{user_prompt}"

    def _current_state_text(self, system_prompt, user_prompt, history):
        return f"state:{system_prompt}|{user_prompt}|{len(history)}"


@pytest.fixture()
def fake_hier_cem_module(monkeypatch):
    """Inject a torch-free fake ``_ewm_hier_cem`` so the lazy import in
    ``hier_latent_cem_step`` resolves to a scripted ``hierarchical_cem_plan``."""
    mod = types.ModuleType("ejepa_wm.backends._ewm_hier_cem")

    class HierarchicalCEMConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    mod.HierarchicalCEMConfig = HierarchicalCEMConfig
    mod.hierarchical_cem_plan = None  # set per-test
    monkeypatch.setitem(sys.modules, "ejepa_wm.backends._ewm_hier_cem", mod)
    return mod


def _confident_result(tool_name="good_tool"):
    return {
        "num_llm_calls": 1, "num_anchors": 3, "confident": True, "elite_agreement": 0.9,
        "vetoed": False, "score": 5.0, "reason": "good",
        "imagined_plan": [
            {"calls": [{"type": "function", "function": {"name": tool_name, "arguments": {"x": 1}}}],
             "score": 5.0, "reason": "good", "predicted_state": {"execution_status": "success"}},
            {"calls": [{"type": "function", "function": {"name": tool_name, "arguments": {"x": 2}}}],
             "score": 4.0, "reason": "good", "predicted_state": {"execution_status": "success"}},
        ],
        "family_distribution": {tool_name: 1.0},
    }


def _unconfident_result():
    return {
        "num_llm_calls": 1, "num_anchors": 3, "confident": False, "elite_agreement": 0.2,
        "vetoed": False, "score": 1.0, "reason": "flat", "imagined_plan": [], "family_distribution": {},
    }


def _make_hier_wm(monkeypatch, fake_hier_cem_module, **env):
    monkeypatch.setenv("WM_STRATEGY", "hier_latent_cem")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/ckpt")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    # Avoid constructing a real JepaEwmGenerator (needs torch): inject a fake _ewm_jepa module
    # whose JepaEwmGenerator returns our fake generator.
    jepa_mod = types.ModuleType("ejepa_wm.backends._ewm_jepa")
    jepa_mod.JepaEwmGenerator = lambda *a, **k: _FakeJepaGenerator()
    monkeypatch.setitem(sys.modules, "ejepa_wm.backends._ewm_jepa", jepa_mod)

    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    return EwmImaginedWorldModel(wm_config_from_env(), lambda messages: "[]")


_FLOW = [
    {"type": "system_message", "content": "you are an agent"},
    {"type": "user_message", "content": "do the task"},
    {"type": "tools", "tools": [{"name": "good_tool"}, {"name": "seed_tool"}]},
]


def test_hier_cem_supported_when_jepa_canonical_event_available(monkeypatch, fake_hier_cem_module):
    wm = _make_hier_wm(monkeypatch, fake_hier_cem_module)
    assert wm.supports_hier_cem() is True


def test_hier_cem_unsupported_without_canonical_event_heads(monkeypatch, fake_hier_cem_module):
    jepa_mod = types.ModuleType("ejepa_wm.backends._ewm_jepa")
    jepa_mod.JepaEwmGenerator = lambda *a, **k: _FakeJepaGenerator(canonical_event_available=False)
    monkeypatch.setenv("WM_STRATEGY", "hier_latent_cem")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/ckpt")
    monkeypatch.setitem(sys.modules, "ejepa_wm.backends._ewm_jepa", jepa_mod)
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    wm = EwmImaginedWorldModel(wm_config_from_env(), lambda messages: "[]")
    assert wm.supports_hier_cem() is False
    res = wm.hier_latent_cem_step(_FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}])
    assert res.detail["event"] == "GYM_HIER_CEM_UNSUPPORTED"


def test_hier_cem_no_seed_is_noop(monkeypatch, fake_hier_cem_module):
    wm = _make_hier_wm(monkeypatch, fake_hier_cem_module)
    res = wm.hier_latent_cem_step(_FLOW, seed_calls=[])
    assert res.detail["event"] == "GYM_HIER_CEM_NO_SEED"


def test_hier_cem_advisory_recommends_without_override(monkeypatch, fake_hier_cem_module):
    fake_hier_cem_module.hierarchical_cem_plan = lambda **kwargs: _confident_result()
    wm = _make_hier_wm(monkeypatch, fake_hier_cem_module, WM_BEAM_MPC_EXECUTE_STEPS=2)
    res = wm.hier_latent_cem_step(_FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t")
    d = res.detail
    assert d["event"] == "GYM_HIER_CEM_PLAN" and d["replanned"] is True
    assert d["confident"] is True and d["injected"] is True and d["advisory"] is True
    assert d["override_applied"] is False
    assert d["calls"][0]["name"] == "seed_tool"                 # advisory: agent's action stands
    assert d["recommended_calls"][0]["name"] == "good_tool"     # what the CEM would recommend
    assert d["imagined_plan_len"] == 2


def test_hier_cem_hard_override_forces_recommendation(monkeypatch, fake_hier_cem_module):
    fake_hier_cem_module.hierarchical_cem_plan = lambda **kwargs: _confident_result()
    wm = _make_hier_wm(monkeypatch, fake_hier_cem_module, WM_BEAM_PLAN_HARD_OVERRIDE=1, WM_BEAM_MPC_EXECUTE_STEPS=2)
    res = wm.hier_latent_cem_step(_FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t")
    d = res.detail
    assert d["hard_override"] is True and d["advisory"] is False
    assert d["override_applied"] is True
    assert d["calls"][0]["name"] == "good_tool"


def test_hier_cem_unconfident_withholds_injection(monkeypatch, fake_hier_cem_module):
    fake_hier_cem_module.hierarchical_cem_plan = lambda **kwargs: _unconfident_result()
    wm = _make_hier_wm(monkeypatch, fake_hier_cem_module)
    res = wm.hier_latent_cem_step(_FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t")
    d = res.detail
    assert d["confident"] is False and d["injected"] is False
    assert d["override_applied"] is False
    assert wm.hier_injection_text() == ""


def test_hier_cem_injection_then_follow_then_cooldown_then_replan(monkeypatch, fake_hier_cem_module):
    calls = {"n": 0}

    def plan(**kwargs):
        calls["n"] += 1
        return _confident_result() if calls["n"] == 1 else _unconfident_result()

    fake_hier_cem_module.hierarchical_cem_plan = plan
    wm = _make_hier_wm(monkeypatch, fake_hier_cem_module, WM_BEAM_MPC_EXECUTE_STEPS=2)
    seed = [{"name": "seed_tool", "arguments": {}}]

    r1 = wm.hier_latent_cem_step(_FLOW, seed_calls=seed, user_query="t")
    assert r1.detail["event"] == "GYM_HIER_CEM_PLAN" and r1.detail["injected"] is True
    assert "good_tool" in wm.hier_injection_text()

    # imagined_plan has 2 steps and execute_steps=2, so the plan is followed for 2 steps
    # (r2, r3) before the cache is exhausted and cursor>=execute_steps both trigger a re-plan.
    r2 = wm.hier_latent_cem_step(_FLOW, seed_calls=seed, user_query="t")
    assert r2.detail["event"] == "GYM_HIER_CEM_FOLLOW"
    assert r2.detail["calls"][0]["name"] == "seed_tool"

    r3 = wm.hier_latent_cem_step(_FLOW, seed_calls=seed, user_query="t")
    assert r3.detail["event"] == "GYM_HIER_CEM_FOLLOW"

    r4 = wm.hier_latent_cem_step(_FLOW, seed_calls=seed, user_query="t")
    assert r4.detail["event"] == "GYM_HIER_CEM_PLAN"          # cadence elapsed -> re-plan
    assert r4.detail["confident"] is False and r4.detail["injected"] is False

    r5 = wm.hier_latent_cem_step(_FLOW, seed_calls=seed, user_query="t")
    assert r5.detail["event"] == "GYM_HIER_CEM_COOLDOWN"      # rejected replan -> cooldown, no re-trigger


def test_reset_episode_clears_hier_cem_state(monkeypatch, fake_hier_cem_module):
    fake_hier_cem_module.hierarchical_cem_plan = lambda **kwargs: _confident_result()
    wm = _make_hier_wm(monkeypatch, fake_hier_cem_module)
    wm.hier_latent_cem_step(_FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t")
    assert wm.hier_injection_text() != ""
    wm.reset_episode()
    assert wm.hier_injection_text() == ""


def test_hier_cem_error_falls_back_to_baseline(monkeypatch, fake_hier_cem_module):
    def boom(**kwargs):
        raise RuntimeError("cem exploded")

    fake_hier_cem_module.hierarchical_cem_plan = boom
    wm = _make_hier_wm(monkeypatch, fake_hier_cem_module)
    res = wm.hier_latent_cem_step(_FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t")
    assert res.detail["event"] == "GYM_HIER_CEM_ERROR"
    assert res.detail["calls"][0]["name"] == "seed_tool"
