"""Training-free ITP-I strategy tests (no model or network dependency)."""

from __future__ import annotations

from ejepa_wm.backends._ewm_itp_i import parse_lookahead
from ejepa_wm.base import STRATEGIES, WMConfig, normalize_strategy
from ejepa_wm.factory import wm_config_from_env

FLOW = [
    {"type": "tools", "tools": [{"name": "find_record"}, {"name": "update_record"}]},
    {"type": "system_message", "content": "Use tools safely."},
    {"type": "user_message", "content": "Update the requested record."},
    {"type": "tool_result", "tool_name": "find_record", "result": {"id": "sample-id"}},
]


class _FakeWorldModel:
    def __init__(self) -> None:
        self.calls = []

    def generate_from_messages(self, messages, temperature=0.0):
        self.calls.append((messages, temperature))
        return "1. update_record(id=sample-id) -> update succeeds"


def _clear_modes(monkeypatch):
    for name in (
        "WM_EWM_JEPA_CHECKPOINT",
        "WM_EWM_BACKEND",
        "WM_LLM_EWM_MODE",
        "WM_EWM_LLM_CANONICAL_EVENT_CHECKPOINT",
        "WM_EWM_MCP_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_itp_i_is_known_and_defaults_to_imagined_backend(monkeypatch):
    monkeypatch.setenv("WM_STRATEGY", "itp-i")
    assert "itp_i" in STRATEGIES
    assert normalize_strategy("imagine_then_plan") == "itp_i"
    cfg = wm_config_from_env()
    assert cfg.strategy == "itp_i"
    assert cfg.backend == "ewm_imagined"


def test_parse_lookahead_clamps_and_falls_back():
    assert parse_lookahead("K = 3", max_k=5) == 3
    assert parse_lookahead("99", max_k=5) == 5
    assert parse_lookahead("unclear", max_k=5) == 1


def test_itp_i_adaptive_k_imagines_once_and_returns_reflection(monkeypatch):
    _clear_modes(monkeypatch)
    monkeypatch.setenv("WM_ITP_MAX_K", "5")
    policy_calls = []

    def policy(messages, **kwargs):
        policy_calls.append((messages, kwargs))
        return "2"

    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    model = EwmImaginedWorldModel(
        WMConfig(strategy="itp_i", backend="ewm_imagined"), chat_fn=policy
    )
    fake_wm = _FakeWorldModel()
    model._itp_world_model = fake_wm
    result = model.advise(FLOW)

    assert len(policy_calls) == 1
    assert len(fake_wm.calls) == 1
    assert result.detail["itp_i_k"] == 2
    assert result.detail["itp_i_world_model_calls"] == 1
    assert "hypothetical, not executed" in result.text
    assert "Do not blindly copy" in result.text
    assert "update_record" in fake_wm.calls[0][0][1]["content"]


def test_itp_i_fixed_zero_skips_both_decision_and_world_model(monkeypatch):
    _clear_modes(monkeypatch)
    monkeypatch.setenv("WM_ITP_FIXED_K", "0")
    policy_calls = []

    def policy(messages, **kwargs):
        policy_calls.append((messages, kwargs))
        return "5"

    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    model = EwmImaginedWorldModel(
        WMConfig(strategy="itp_i", backend="ewm_imagined"), chat_fn=policy
    )
    fake_wm = _FakeWorldModel()
    model._itp_world_model = fake_wm
    result = model.advise(FLOW)

    assert policy_calls == []
    assert fake_wm.calls == []
    assert result.detail["itp_i_k"] == 0
    assert "No lookahead was requested" in result.text


def test_itp_i_jepa_rolls_policy_plan_into_canonical_state_feedback(monkeypatch):
    _clear_modes(monkeypatch)
    policy_messages = []
    policy_replies = iter(
        [
            "2",
            '{"name":"find_record","arguments":{}}',
            '{"name":"update_record","arguments":{"id":"sample-id"}}',
        ]
    )

    def policy(messages, **_kwargs):
        policy_messages.append(messages)
        return next(policy_replies)

    class _FakeJepa:
        canonical_event_available = True

        def __init__(self):
            self.action_plan_calls = []

        def score_action_plans_canonical_event(self, **kwargs):
            self.action_plan_calls.append(kwargs["action_plans"])
            plan_len = len(kwargs["action_plans"][0])
            states = [
                {
                    "execution_status": "success",
                    "progress_signal": "positive",
                    "information_sufficiency": "sufficient",
                },
                {
                    "execution_status": "success",
                    "progress_signal": "positive",
                    "side_effect_type": "record_updated",
                },
            ][:plan_len]
            return [
                {
                    "per_step_predicted_state": states,
                    "per_step_terminal_prob": [0.2, 0.9][:plan_len],
                }
            ]

    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    model = EwmImaginedWorldModel(
        WMConfig(strategy="itp_i", backend="ewm_imagined"), chat_fn=policy
    )
    fake_jepa = _FakeJepa()
    model._wm = fake_jepa
    model._itp_uses_jepa_canonical = True
    model._itp_world_model = None

    result = model.advise(FLOW)

    assert result.detail["itp_i_k"] == 2
    assert result.detail["itp_i_world_model_kind"] == "jepa_canonical_event"
    assert result.detail["itp_i_action_proposal_calls"] == 2
    assert result.detail["itp_i_world_model_calls"] == 2
    assert [len(plans[0]) for plans in fake_jepa.action_plan_calls] == [1, 2]
    assert "information_sufficiency=sufficient" in policy_messages[2][1]["content"]
    assert "execution_status=success" in result.text
    assert "information_sufficiency=sufficient" in result.text
