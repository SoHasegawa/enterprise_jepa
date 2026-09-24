"""Tests for Qwen-AgentWorld-only world-model adaptation."""
from __future__ import annotations

import pytest

from ejepa_wm.backends._ewm_llm_tool_output_judge import LlmToolOutputJudgeGenerator
from ejepa_wm.backends._ewm_qwen_agentworld import (
    is_qwen_agentworld_model,
    render_qwen_agentworld_system_prompt,
    resolve_qwen_agentworld_domain,
)
from ejepa_wm.backends._ewm_runtime import EwmGenerator


@pytest.mark.parametrize(
    ("benchmark", "expected"),
    [
        ("EnterpriseOps-Gym", "mcp"),
        ("crmarenapro", "mcp"),
        ("WorkBench", "mcp"),
        ("Workspace-Bench", "mcp"),
        ("wow", "mcp"),
        ("Terminal-Bench-2.0", "terminal"),
        ("DevOps-Gym", "swe"),
    ],
)
def test_requested_benchmark_domain_mapping(benchmark, expected):
    assert resolve_qwen_agentworld_domain(benchmark) == expected


def test_model_detection_is_qwen_specific():
    assert is_qwen_agentworld_model("Qwen/Qwen-AgentWorld-35B-A3B")
    assert is_qwen_agentworld_model("/models/qwen_agentworld_35b")
    assert not is_qwen_agentworld_model("Qwen/Qwen3.5-35B-A3B")


def test_domain_override_is_validated():
    assert resolve_qwen_agentworld_domain("wow", override="terminal") == "terminal"
    with pytest.raises(ValueError, match="WM_QWEN_AGENTWORLD_DOMAIN"):
        resolve_qwen_agentworld_domain("wow", override="web")


def test_official_mcp_template_receives_tool_contract():
    prompt = render_qwen_agentworld_system_prompt(
        "mcp",
        tool_context="TOOL CONTRACT: create_record(id: string)",
        benchmark_name="EnterpriseOps-Gym",
    )

    assert "Tool World Model" in prompt
    assert "TOOL CONTRACT: create_record" in prompt
    assert "{tool_definitions}" not in prompt
    assert "{demonstrations}" not in prompt
    assert "EJEPA Benchmark Integration (adapted)" in prompt


def test_qwen_terminal_messages_adapt_run_shell_to_keystrokes():
    wm = LlmToolOutputJudgeGenerator(
        object(),
        object(),
        qwen_agentworld=True,
        benchmark_name="Terminal-Bench-2.0",
        world_model_temperature=0.6,
    )
    messages = wm._tool_output_messages(
        system_prompt="run_shell returns an exec_result JSON payload",
        user_prompt="inspect the project",
        input_history=[],
        action={
            "tool_calls": [
                {
                    "function": {
                        "name": "run_shell",
                        "arguments": {"command": "ls -la", "timeout": 30},
                    }
                }
            ]
        },
    )

    assert "Terminal World Model" in messages[0]["content"]
    assert "run_shell returns an exec_result JSON payload" in messages[0]["content"]
    assert "Current terminal action" in messages[1]["content"]
    assert '"keystrokes": "ls -la\\n"' in messages[1]["content"]
    assert '"duration": 30.0' in messages[1]["content"]
    assert "/no_think" not in messages[1]["content"]


def test_ewm_generator_sends_qwen_sampling_parameters(monkeypatch):
    bodies = []
    generator = EwmGenerator(
        "Qwen/Qwen-AgentWorld-35B-A3B",
        "http://localhost:8000/v1",
        max_new_tokens=32768,
        top_p=0.95,
        top_k=20,
    )
    monkeypatch.setattr(
        generator,
        "_post_chat_completions",
        lambda body: bodies.append(body) or {"choices": [{"message": {"content": "ok"}}]},
    )
    monkeypatch.setattr(generator, "_warn_if_prefix_caching_inactive", lambda: None)

    assert generator.generate_from_messages(
        [{"role": "user", "content": "simulate"}], temperature=0.6
    ) == "ok"
    assert bodies[0]["temperature"] == pytest.approx(0.6)
    assert bodies[0]["top_p"] == pytest.approx(0.95)
    assert bodies[0]["top_k"] == 20
    assert bodies[0]["max_tokens"] == 32768


def test_qwen_model_auto_selects_tool_output_mode(monkeypatch):
    for name in (
        "WM_EWM_BACKEND",
        "WM_LLM_EWM_MODE",
        "WM_EWM_JEPA_CHECKPOINT",
        "WM_EWM_LLM_CANONICAL_EVENT_CHECKPOINT",
        "WM_EWM_MCP_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")
    monkeypatch.setenv("WM_EWM_MODEL", "Qwen/Qwen-AgentWorld-35B-A3B")
    monkeypatch.setenv("BENCHMARK_NAME", "EnterpriseOps-Gym")

    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel
    from ejepa_wm.factory import wm_config_from_env

    wm = EwmImaginedWorldModel(wm_config_from_env(), lambda messages: "")
    assert wm.llm_ewm_mode == "llm_tool_output_judge"
    assert wm.qwen_agentworld is True
    assert wm._wm.qwen_agentworld is True
    assert wm._wm.benchmark_name == "EnterpriseOps-Gym"
    assert wm._wm.world_model_temperature == pytest.approx(0.6)
    assert wm._wm.world_model_generator.max_new_tokens == 32768
    assert wm._wm.world_model_generator.top_p == pytest.approx(0.95)
    assert wm._wm.world_model_generator.top_k == 20
