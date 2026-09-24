"""Canonical-state revision and delayed-reference harness tests."""

from __future__ import annotations

from ejepa_wm.base import WMConfig
from ejepa_wm.factory import wm_config_from_env

FLOW = [
    {"type": "tools", "tools": [{"name": "update_record"}]},
    {"type": "system_message", "content": "Use tools safely."},
    {"type": "user_message", "content": "Update the record."},
]


class _FakeCanonicalWm:
    canonical_event_available = True

    def __init__(self) -> None:
        self.calls = []

    def score_action_plans_canonical_event(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "score": -0.4,
                "vetoed": True,
                "per_step_predicted_state": [
                    {
                        "execution_status": "failure",
                        "progress_signal": "negative",
                        "error_signature": "permission_denied",
                    }
                ],
                "per_step_terminal_prob": [0.1],
            }
        ]


class _FakeToolOutputJudgeWm:
    canonical_event_available = True

    def score_action_plans_canonical_event(self, **_kwargs):
        return [
            {
                "score": 0.2,
                "vetoed": False,
                "per_step_predicted_state": [
                    {"execution_status": "success", "progress_signal": "positive"}
                ],
                "per_step_terminal_prob": [0.8],
                "per_step_predicted_tool_output": ['{"status":"ok","record":"sample-id"}'],
                "per_step": [
                    {
                        "judge": {
                            "can_proceed": True,
                            "failure_score": 0.05,
                            "progress_score": 0.9,
                            "enough_to_finish": True,
                            "finish_score": 0.8,
                            "reason": "The required record would be updated.",
                        }
                    }
                ],
            }
        ]


def _model(strategy: str):
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    model = EwmImaginedWorldModel(
        WMConfig(strategy=strategy, backend="ewm_imagined"), chat_fn=lambda _messages: ""
    )
    model._wm = _FakeCanonicalWm()
    return model


def test_revision_and_reference_default_to_imagined_backend() -> None:
    for strategy in ("revision", "reference"):
        config = wm_config_from_env({"WM_STRATEGY": strategy})
        assert config.strategy == strategy
        assert config.backend == "ewm_imagined"


def test_revision_feedback_requests_proceed_or_revise() -> None:
    model = _model("revision")
    result = model.action_feedback(
        FLOW,
        seed_calls=[{"name": "update_record", "arguments": {"id": "sample-id"}}],
        mode="revision",
    )

    assert "PROCEED" in result.text and "REVISE" in result.text
    assert "execution_status=failure" in result.text
    assert result.detail["world_model_calls"] == 1


def test_reference_feedback_is_labeled_for_the_following_step() -> None:
    model = _model("reference")
    result = model.action_feedback(
        FLOW,
        seed_calls=[{"name": "update_record", "arguments": {"id": "sample-id"}}],
        mode="reference",
    )

    assert "PREVIOUS_ACTION_REFERENCE" in result.text
    assert "actual tool result" in result.text
    assert result.detail["predicted_state"]["error_signature"] == "permission_denied"


def test_tool_output_revision_exposes_prediction_and_same_agent_judgment() -> None:
    model = _model("revision")
    model._wm = _FakeToolOutputJudgeWm()

    result = model.action_feedback(
        FLOW,
        seed_calls=[{"name": "update_record", "arguments": {"id": "sample-id"}}],
        mode="revision",
    )

    assert "predicted tool output" in result.text
    assert '"record":"sample-id"' in result.text
    assert "Same-agent critic assessment" in result.text
    assert "can_proceed=True" in result.text
    assert result.detail["prediction_kind"] == "tool_output"
    assert result.detail["judge_calls"] == 1
    assert result.detail["model_calls"] == 2


def test_canonical_feedback_records_single_llm_or_jepa_prediction_call() -> None:
    result = _model("revision").action_feedback(
        FLOW,
        seed_calls=[{"name": "update_record", "arguments": {"id": "sample-id"}}],
        mode="revision",
    )

    assert result.detail["prediction_kind"] == "canonical_state"
    assert result.detail["judge_calls"] == 0
    assert result.detail["model_calls"] == 1
