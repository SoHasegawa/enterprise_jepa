"""Tests for the pure-Python pieces of :mod:`_ewm_llm_canonical_event`."""
from __future__ import annotations

import json

from ejepa_wm.backends import _canonical_event_state as ces
from ejepa_wm.backends import _ewm_llm_canonical_event as llmwm


def test_system_prompt_matches_reduced_beam_terminal_training_format():
    expected = (
        "You are an enterprise world model. Given the system prompt, user task, recent "
        "history, and the current action, predict the resulting annotated state.\n"
        "Return ONLY a JSON object with exactly these keys, in this order, and no other text:\n"
        "- execution_status: one of [failure, no_op, partial, success, unknown]\n"
        "- progress_signal: one of [negative, neutral, positive, unknown]\n"
        "- information_sufficiency: one of [insufficient, sufficient, unknown]\n"
        "- error_signature: one of [dependency_missing, invalid_argument, none, not_found, "
        "parse_error, permission_denied, policy_risk, runtime_error, test_failed, timeout, "
        "unknown]\n"
        "- side_effect_type: one of [created, deleted, executed, failed_validation, installed, "
        "modified, none, retrieved, sent, unknown, validated]\n"
        "- terminal: one of [not_finished, finished]; use finished only when this action is the "
        "final step that completes or ends the task"
    )
    assert llmwm.CANONICAL_EVENT_SYSTEM_PROMPT == expected


def test_vocab_matches_beam_scored_value_sets():
    mapping = {
        "execution_status": ces.EXECUTION_STATUS_VALUES,
        "progress_signal": ces.PROGRESS_SIGNAL_VALUES,
        "information_sufficiency": ces.INFORMATION_SUFFICIENCY_VALUES,
        "error_signature": ces.ERROR_SIGNATURE_VALUES,
        "side_effect_type": ces.SIDE_EFFECT_TYPE_VALUES,
    }
    for field, values in mapping.items():
        assert set(llmwm.CANONICAL_EVENT_VOCAB[field]) == values
    assert tuple(llmwm.CANONICAL_EVENT_VOCAB["terminal"]) == llmwm.CANONICAL_EVENT_TERMINAL_VALUES
    assert llmwm.FIELD_ORDER == llmwm.CANONICAL_EVENT_LLM_TARGET_FIELDS


def test_build_canonical_event_prompt_matches_training_shape():
    messages = llmwm.build_canonical_event_prompt(
        "system text", "user task text",
        [{"step": 1, "action": {"name": "x"}, "observation": "obs"}],
        {"name": "create_x", "arguments": {"a": 1}},
    )
    assert messages[0] == {"role": "system", "content": llmwm.CANONICAL_EVENT_SYSTEM_PROMPT}
    content = messages[1]["content"]
    assert content.startswith("System prompt:\nsystem text\n\nUser prompt:\nuser task text\n\n")
    assert "Previous state:\n{}\n\n" in content
    assert (
        "Recent action/observation history (oldest to newest; input only, not part of the "
        'target):\n[{"action": {"name": "x"}, "observation": "obs", "step": 1}]\n\n'
    ) in content
    assert content.endswith(
        'Action:\n{\n  "name": "create_x",\n  "arguments": {\n    "a": 1\n  }\n}\n\n'
        'Predict the annotated state as JSON. /no_think'
    )


def test_build_canonical_event_prompt_truncates_history_to_window_size():
    history = [{"step": i, "action": f"a{i}", "observation": f"o{i}"} for i in range(20)]
    messages = llmwm.build_canonical_event_prompt("s", "u", history, "action")
    content = messages[1]["content"]
    kept = json.loads(content.split("target):\n")[1].split("\n\nAction:")[0])
    from ejepa_wm.backends._ewm_finetuning import WORLD_MODEL_INPUT_HISTORY_SIZE
    assert len(kept) == WORLD_MODEL_INPUT_HISTORY_SIZE
    assert kept[-1]["step"] == 19


def test_parse_canonical_event_completion_happy_path():
    text = (
        '{"execution_status": "success", "progress_signal": "positive", '
        '"information_sufficiency": "sufficient", "error_signature": "none", '
        '"side_effect_type": "retrieved", "terminal": "finished"}'
    )
    event = llmwm.parse_canonical_event_completion(text)
    assert event["execution_status"] == "success"
    assert event["terminal"] == "finished"
    assert llmwm.canonical_event_completion_text(event) == text


def test_parse_canonical_event_completion_strips_code_fence_and_thinking():
    inner = '{"execution_status": "failure", "terminal": "not_finished"}'
    text = "<think>reasoning here</think>\n```json\n" + inner + "\n```"
    event = llmwm.parse_canonical_event_completion(text)
    assert event["execution_status"] == "failure"
    assert event["terminal"] == "not_finished"


def test_parse_canonical_event_completion_defaults_on_garbage():
    event = llmwm.parse_canonical_event_completion("not json at all")
    assert event["execution_status"] == "unknown"
    assert event["progress_signal"] == "unknown"
    assert event["terminal"] == "not_finished"


def test_parse_canonical_event_completion_rejects_out_of_vocab_values():
    text = '{"execution_status": "definitely_not_a_real_category", "terminal": "maybe"}'
    event = llmwm.parse_canonical_event_completion(text)
    assert event["execution_status"] == "unknown"
    assert event["terminal"] == "not_finished"


def test_parse_canonical_event_completion_accepts_nested_legacy_shape():
    text = json.dumps({
        "canonical_event_state": {"execution_status": "success", "progress_signal": "positive"},
        "nudge": {"information_sufficiency": "sufficient"},
        "terminal": True,
    })
    event = llmwm.parse_canonical_event_completion(text)
    assert event["execution_status"] == "success"
    assert event["progress_signal"] == "positive"
    assert event["information_sufficiency"] == "sufficient"
    assert event["terminal"] == "finished"


def test_canonical_event_completion_text_orders_fields_and_fills_defaults():
    text = llmwm.canonical_event_completion_text({"execution_status": "success"})
    parsed = json.loads(text)
    assert list(parsed.keys()) == list(llmwm.CANONICAL_EVENT_LLM_TARGET_FIELDS)
    assert parsed["execution_status"] == "success"
    assert parsed["progress_signal"] == "unknown"
    assert parsed["terminal"] == "not_finished"


def test_canonical_event_to_field_probs_is_one_hot_with_terminal():
    event = llmwm.parse_canonical_event_completion(
        '{"execution_status": "failure", "progress_signal": "negative", '
        '"information_sufficiency": "insufficient", "error_signature": "runtime_error", '
        '"side_effect_type": "failed_validation", "terminal": "finished"}'
    )
    probs = llmwm.canonical_event_to_field_probs(event)
    assert probs["execution_status"]["failure"] == 1.0
    assert probs["execution_status"]["success"] == 0.0
    assert probs["terminal"]["finished"] == 1.0
    assert llmwm.terminal_probability_from_event(event) == 1.0


class _ServedCanonicalStub:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []

    def generate_from_messages(self, messages, temperature=0.0):
        self.messages.append(messages)
        return self.replies.pop(0)


def test_served_canonical_event_scores_with_one_hot_probs_and_terminal():
    good = json.dumps({
        "execution_status": "success",
        "progress_signal": "positive",
        "information_sufficiency": "sufficient",
        "error_signature": "none",
        "side_effect_type": "modified",
        "terminal": "finished",
    })
    bad = json.dumps({
        "execution_status": "failure",
        "progress_signal": "negative",
        "information_sufficiency": "insufficient",
        "error_signature": "runtime_error",
        "side_effect_type": "failed_validation",
        "terminal": "not_finished",
    })
    gen = llmwm.ServedLlmCanonicalEventGenerator(_ServedCanonicalStub([good, bad]))
    scored = gen.score_action_plans_canonical_event(
        system_prompt="sys",
        user_prompt="task",
        input_history=[],
        action_plans=[[{"name": "good"}], [{"name": "bad"}]],
    )
    assert scored[0]["plan_index"] == 0
    assert scored[0]["per_step_field_probs"][0]["execution_status"]["success"] == 1.0
    assert scored[0]["terminal_probability"] == 1.0
    assert scored[1]["per_step_field_probs"][0]["execution_status"]["failure"] == 1.0
