"""Unit tests for the EWM predict MCP server's core (no model load, no fastmcp needed)."""
from __future__ import annotations

import json

import pytest

from ejepa_wm.backends._ewm_generators import (
    BACKEND_TRANSFORMERS,
    BACKEND_VLLM,
    EwmGenerator,
    resolve_ewm_backend,
)
from ejepa_wm.server._predict import default_wm_state, run_predict_state


class StubGenerator:
    """Returns canned WM text, recording the messages it was asked to complete."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list = []

    def generate_from_messages(self, messages, temperature: float = 0.0) -> str:
        self.calls.append(messages)
        return self.reply


# --- backend resolution (pure, no model loaded) ---------------------------------


def test_resolve_defaults_to_vllm():
    r = resolve_ewm_backend({})
    assert r.kind == BACKEND_VLLM
    assert r.model == "gymops_world_model"
    assert r.base_url == "http://127.0.0.1:9000/v1"


def test_resolve_vllm_method_strips_prefix_and_uses_port():
    r = resolve_ewm_backend(
        {"EWM_WORLD_MODEL_METHOD": "vllm/my_wm", "EWM_VLLM_SERVER_PORT": "9100"}
    )
    assert (r.kind, r.model) == (BACKEND_VLLM, "my_wm")
    assert r.base_url == "http://127.0.0.1:9100/v1"


def test_resolve_vllm_base_url_override():
    r = resolve_ewm_backend({"WM_VLLM_BASE_URL": "http://gpu:8000/v1/"})
    assert r.kind == BACKEND_VLLM
    assert r.base_url == "http://gpu:8000/v1"  # trailing slash trimmed


def test_resolve_transformers_via_model_path():
    r = resolve_ewm_backend({"EWM_WORLD_MODEL_PATH": "/models/wm_ckpt"})
    assert r.kind == BACKEND_TRANSFORMERS
    assert r.model_path == "/models/wm_ckpt"


def test_resolve_transformers_method_requires_path():
    with pytest.raises(ValueError):
        resolve_ewm_backend({"EWM_WORLD_MODEL_METHOD": "transformers"})


def test_resolve_vllm_method_wins_over_model_path():
    # An explicit vllm/ method takes precedence even if a stray path is set.
    r = resolve_ewm_backend(
        {"EWM_WORLD_MODEL_METHOD": "vllm/x", "EWM_WORLD_MODEL_PATH": "/models/x"}
    )
    assert r.kind == BACKEND_VLLM


def test_build_vllm_generator_is_ewmgenerator():
    from ejepa_wm.backends._ewm_generators import build_ewm_generator

    gen = build_ewm_generator(resolve_ewm_backend({"EWM_WORLD_MODEL_METHOD": "vllm/foo"}))
    assert isinstance(gen, EwmGenerator)


# --- predict_state output contract (stubbed generator) --------------------------


def test_predict_binary_error_success():
    out = run_predict_state(
        StubGenerator("1"),
        system_prompt="sys",
        user_prompt="do it",
        action=[{"name": "create_calendar", "args": {"name": "Team"}}],
        wm_state="binary_error",
    )
    assert out["wm_state"] == "binary_error"
    assert out["success"] == 1
    assert out["error_message"] == ""
    assert out["state"] is not None  # compact predicted state populated for binary modes


def test_predict_binary_error_failure_carries_message():
    out = run_predict_state(
        StubGenerator("0,disk is full"),
        system_prompt="sys",
        user_prompt="do it",
        action=[{"name": "upload", "args": {}}],
        wm_state="binary_error",
    )
    assert out["success"] == 0
    assert "disk is full" in out["error_message"]


def test_predict_binary_error_stage_mode():
    reply = json.dumps(
        {"success": 1, "current_stage": "create", "remaining_stages": ["share"]}
    )
    out = run_predict_state(
        StubGenerator(reply),
        system_prompt="sys",
        user_prompt="do it",
        action=[{"name": "create_calendar", "args": {}}],
        wm_state="binary_error_stage",
    )
    assert out["success"] == 1
    assert out["current_stage"] == "create"
    assert out["remaining_stages"] == ["share"]


def test_predict_tool_output_mode():
    out = run_predict_state(
        StubGenerator("Calendar created with id cal_42"),
        system_prompt="sys",
        user_prompt="do it",
        action=[{"name": "create_calendar", "args": {}}],
        wm_state="tool_output",
    )
    assert out["tool_output"] == "Calendar created with id cal_42"
    assert out["raw_prediction"] == "Calendar created with id cal_42"


def test_predict_canonical_nudge_mode():
    reply = json.dumps(
        {
            "canonical_event_state": {
                "execution_status": "failure",
                "error_signature": "not_found",
                "action_type": "read",
                "object_type": "ticket",
                "side_effect_type": "none",
                "progress_signal": "negative",
                "risk_signal": "none",
            },
            "nudge": {
                "information_sufficiency": "insufficient",
                "missing_information_type": ["object_id"],
                "information_gain": "high",
                "recommended_abstract_action": "search",
            },
        }
    )
    out = run_predict_state(
        StubGenerator(reply),
        system_prompt="sys",
        user_prompt="do it",
        action=[{"name": "get_ticket", "args": {}}],
        wm_state="canonical_nudge",
    )
    # execution_status drives success; error falls back to the categorical error_signature.
    assert out["success"] == 0
    assert out["error_message"] == "not_found"
    assert out["canonical_event_state"]["execution_status"] == "failure"
    assert out["nudge"]["recommended_abstract_action"] == "search"
    assert out["nudge"]["missing_information_type"] == ["object_id"]
    # canonical mode carries no raw tool output.
    assert out["tool_output"] == ""


def test_predict_canonical_nudge_success_has_no_error():
    reply = json.dumps(
        {
            "canonical_event_state": {
                "execution_status": "success",
                "error_signature": "none",
                "action_type": "read",
                "object_type": "ticket",
                "side_effect_type": "retrieved",
                "progress_signal": "neutral",
                "risk_signal": "none",
            },
            "nudge": {
                "information_sufficiency": "sufficient",
                "missing_information_type": ["none"],
                "information_gain": "high",
                "recommended_abstract_action": "proceed",
            },
        }
    )
    out = run_predict_state(
        StubGenerator(reply),
        system_prompt="sys",
        user_prompt="do it",
        action=[{"name": "get_ticket", "args": {}}],
        wm_state="canonical_nudge",
    )
    assert out["success"] == 1
    assert out["error_message"] == ""
    assert out["parse_error"] is None
    assert out["nudge"]["recommended_abstract_action"] == "proceed"


def test_default_wm_state_accepts_canonical_nudge():
    assert default_wm_state({"WM_STATE": "canonical_nudge"}) == "canonical_nudge"


def test_predict_drops_dependent_calls_with_placeholders():
    # The misuse the trajectory showed: a whole plan, later calls referencing not-yet-known ids.
    gen = StubGenerator("1")
    out = run_predict_state(
        gen,
        system_prompt="sys",
        user_prompt="do it",
        action=[
            {"name": "create_calendar", "args": {"summary": "Beta"}},
            {"name": "get_calendar_list", "args": {"maxResults": 250}},
            {"name": "insert_acl_rule", "args": {"calendarId": "<to_fill_after_creation>"}},
            {"name": "create_event", "args": {"start": {"dateTime": "2025-11-<XX>"}}},
        ],
        wm_state="binary_error",
    )
    # only the leading concrete calls are scored; the dependent ones are dropped.
    names = [c["name"] for c in out["evaluated_action"]]
    assert names == ["create_calendar", "get_calendar_list"]
    assert any("re-call predict_state" in w.lower() for w in out["warnings"])


def test_predict_warns_when_first_call_has_placeholder():
    out = run_predict_state(
        StubGenerator("1"),
        system_prompt="s",
        user_prompt="u",
        action=[{"name": "insert_acl_rule", "args": {"calendarId": "<unknown>"}}],
        wm_state="binary_error",
    )
    assert out["evaluated_action"]  # still scores the single call
    assert any("placeholder" in w.lower() for w in out["warnings"])


def test_predict_single_concrete_action_has_no_warnings():
    out = run_predict_state(
        StubGenerator("1"),
        system_prompt="s",
        user_prompt="u",
        action=[{"name": "create_calendar", "args": {"summary": "X"}}],
        wm_state="binary_error",
    )
    assert "warnings" not in out
    assert out["success"] == 1


def test_predict_reconstructs_state_and_prompts_from_conversation_flow():
    flow = [
        {"type": "system_message", "content": "SYS-FROM-FLOW"},
        {"type": "user_message", "content": "USER-FROM-FLOW"},
        {"type": "ai_message", "content": "", "tool_calls": [{"name": "create_calendar", "args": {}}]},
        {"type": "tool_result", "tool_name": "create_calendar", "result": {"success": True}},
    ]

    class CapturingGen:
        def __init__(self):
            self.messages = None

        def generate_from_messages(self, messages, temperature: float = 0.0):
            self.messages = messages
            return "1"

    gen = CapturingGen()
    out = run_predict_state(
        gen,
        action=[{"name": "insert_acl_rule", "args": {"calendarId": "cal_1", "role": "writer"}}],
        wm_state="binary_error",
        conversation_flow=flow,
    )
    assert out["success"] == 1
    # prompts were derived from the flow and reached the WM prompt.
    blob = json.dumps(gen.messages)
    assert "SYS-FROM-FLOW" in blob and "USER-FROM-FLOW" in blob


def test_predict_state_chains_as_previous_state():
    # Agent-driven imagined rollout: the `state` a call returns must be reusable as the
    # `previous_state` of the next call (no error), so the agent can chain hypothetical steps.
    gen = StubGenerator("1")
    s1 = run_predict_state(
        gen, system_prompt="s", user_prompt="u",
        action=[{"name": "create_calendar", "args": {"summary": "X"}}], wm_state="binary_error",
    )
    assert s1["state"] is not None
    s2 = run_predict_state(
        gen, system_prompt="s", user_prompt="u",
        action=[{"name": "insert_acl_rule", "args": {"calendarId": "c"}}],
        previous_state=s1["state"], wm_state="binary_error",
    )
    assert s2["parse_error"] is None and s2["success"] == 1


def test_predict_rejects_unknown_wm_state():
    with pytest.raises(ValueError):
        run_predict_state(
            StubGenerator("1"),
            system_prompt="s",
            user_prompt="u",
            action=[],
            wm_state="bogus",
        )


def test_default_wm_state_env():
    assert default_wm_state({}) == "binary_error"
    assert default_wm_state({"WM_STATE": "tool_output"}) == "tool_output"
    assert default_wm_state({"WM_STATE": "nonsense"}) == "binary_error"


# --- shared tool-call normalizer (args-key + namespace prefix) ------------------


def test_normalize_planned_calls_fixes_args_key_and_prefix():
    from ejepa_wm.backends._ewm_generators import normalize_planned_calls

    # candidate/agent shape uses "args"; gpt-5.1 namespaces the name.
    out = normalize_planned_calls(
        [{"name": "functions.create_calendar", "args": {"summary": "X"}}]
    )
    assert out == [{"name": "create_calendar", "arguments": {"summary": "X"}}]
    # already-normalized openai shape passes through.
    out2 = normalize_planned_calls(
        [{"function": {"name": "insert_acl_rule", "arguments": {"role": "writer"}}}]
    )
    assert out2 == [{"name": "insert_acl_rule", "arguments": {"role": "writer"}}]


# --- direct (no-MCP) ewm_predict selection backend ------------------------------


def test_ewm_predict_backend_selects_feasible_candidate(monkeypatch):
    from ejepa_wm import WMConfig, build_world_model

    # build_world_model -> EwmPredictWorldModel constructs a vLLM EwmGenerator (no network).
    wm = build_world_model(WMConfig(strategy="selection", backend="ewm_predict", n=3))
    assert wm.name == "ewm_predict"

    # Stub the per-candidate WM call: only `create_event` is predicted to succeed.
    def fake_predict(conversation_flow, planned_calls):
        name = planned_calls[0]["name"]
        ok = name == "create_event"
        return {
            "predicted_success": ok,
            "predicted_error_message": "" if ok else f"{name} would fail",
            "raw_prediction": "1" if ok else "0",
        }

    monkeypatch.setattr(wm, "_predict", fake_predict)
    flow = [{"type": "user_message", "content": "do it"}]
    candidates = [
        {"type": "ai_message", "tool_calls": [{"name": "insert_acl_rule", "args": {}}]},
        {"type": "ai_message", "tool_calls": [{"name": "create_event", "args": {}}]},
    ]
    result = wm.select(flow, candidates)
    assert result.index == 1  # picked the feasible candidate
    assert result.detail["candidates"][0]["predicted_success"] is False
    assert result.detail["candidates"][1]["predicted_success"] is True


# --- imagined-over-MCP: raw generate client + backend wiring --------------------


def test_mcp_ewm_generator_extracts_text(monkeypatch):
    from ejepa_wm.backends._ewm_generators import McpEwmGenerator

    g = McpEwmGenerator("http://localhost:12072")
    assert g.url == "http://localhost:12072/mcp"  # endpoint appended once
    g._initialized = True  # skip the handshake

    # FastMCP structuredContent path
    monkeypatch.setattr(g, "_post", lambda *_args, **_kw: {"result": {"structuredContent": {"text": "1"}}})
    assert g.generate_from_messages([{"role": "user", "content": "x"}]) == "1"

    # content-block JSON path ({"text": ...} embedded in a text block)
    monkeypatch.setattr(
        g, "_post",
        lambda *_args, **_kw: {"result": {"content": [{"type": "text", "text": '{"text": "0,disk full"}'}]}},
    )
    assert g.generate_from_messages([{"role": "user", "content": "x"}]) == "0,disk full"


def test_mcp_ewm_generator_url_with_explicit_endpoint():
    from ejepa_wm.backends._ewm_generators import McpEwmGenerator

    # already includes /mcp -> not doubled
    assert McpEwmGenerator("http://h:1/mcp").url == "http://h:1/mcp"


def test_ewm_imagined_uses_mcp_generator_when_url_set(monkeypatch):
    monkeypatch.setenv("WM_EWM_MCP_URL", "http://localhost:12072")
    from ejepa_wm import WMConfig
    from ejepa_wm.backends._ewm_generators import McpEwmGenerator
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    wm = EwmImaginedWorldModel(
        WMConfig(strategy="prompt_injection", backend="ewm_imagined"), chat_fn=lambda m: ""
    )
    assert isinstance(wm._wm, McpEwmGenerator)  # WM routed over MCP; agent loop stays local


def test_ewm_imagined_uses_direct_generator_without_url(monkeypatch):
    monkeypatch.delenv("WM_EWM_MCP_URL", raising=False)
    from ejepa_wm import WMConfig
    from ejepa_wm.backends._ewm_generators import McpEwmGenerator
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    wm = EwmImaginedWorldModel(
        WMConfig(strategy="prompt_injection", backend="ewm_imagined"), chat_fn=lambda m: ""
    )
    assert not isinstance(wm._wm, McpEwmGenerator)  # in-process vLLM generator


def test_ewm_predict_backend_falls_back_to_first_when_none_feasible(monkeypatch):
    from ejepa_wm import WMConfig, build_world_model

    wm = build_world_model(WMConfig(strategy="selection", backend="ewm_predict", n=2))
    monkeypatch.setattr(
        wm, "_predict", lambda f, c: {"predicted_success": False, "predicted_error_message": "no"}
    )
    candidates = [
        {"type": "ai_message", "tool_calls": [{"name": "a", "args": {}}]},
        {"type": "ai_message", "tool_calls": [{"name": "b", "args": {}}]},
    ]
    assert wm.select([], candidates).index == 0
