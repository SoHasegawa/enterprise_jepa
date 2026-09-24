"""Tests for the ported `mcp_react` purple-executors on crmarenapro / Terminal-Bench-2.0.

Both reuse their sibling executor's proven runtime via importlib and only reframe
the system prompt into an MCP-style ReAct tool catalog, keeping the exact action
grammar each green already parses. These tests check discovery + that the reuse
and prompt override actually take effect.

Note: the upstream monorepo also carried a Terminal-Bench world-model injection test
here. It asserted an executor shape (`_base`, `_wm`, `_to_conversation_flow`) that the
executor no longer has, and driving the current `_next_action` through a stub world
model reaches a live LLM call, so it was dropped rather than rewritten. The injection
path itself is covered by tests/test_mcp_react_wm_harnesses.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "assets"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_executor(path: Path, unique_name: str):
    spec = importlib.util.spec_from_file_location(unique_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


def test_mcp_react_is_discovered_for_both_benchmarks() -> None:
    from ejepa_cli.cli import _discover_purple_executor_dirs

    for benchmark in ("crmarenapro", "Terminal-Bench-2.0"):
        discovered = _discover_purple_executor_dirs(ASSETS / benchmark)
        assert "mcp_react" in discovered, f"{benchmark} should discover mcp_react"


def test_terminal_bench_mcp_react_reuses_llm_shell_with_react_prompt() -> None:
    # a2a + common.client_utils are importable in the shared venv, so this loads.
    path = ASSETS / "Terminal-Bench-2.0" / "purple-executors" / "mcp_react" / "executor.py"
    module = _load_executor(path, "tb_mcp_react_under_test")

    assert hasattr(module, "build_executor")
    # Reuses llm_shell's executor class (single source of truth for the protocol).
    assert module.wm_module.base.LlmShellExecutor.__name__ == "LlmShellExecutor"
    # The world-model wrapper subclasses it rather than reimplementing the loop.
    assert issubclass(module.WmShellExecutor, module.wm_module.base.LlmShellExecutor)
    # The terminal-bench-shell-v1 grammar the green agent requires is preserved.
    assert '"kind":"exec_request"' in module.wm_module.base.SYSTEM_PROMPT
    assert '"kind":"final"' in module.wm_module.base.SYSTEM_PROMPT


def test_crmarenapro_mcp_react_reuses_baseline_with_react_prompt() -> None:
    # baseline_crm_agent/agent.py imports anthropic/openai (only in the crmarenapro
    # purple venv); skip cleanly where those aren't installed.
    pytest.importorskip("anthropic")
    pytest.importorskip("openai")
    path = ASSETS / "crmarenapro" / "purple-executors" / "mcp_react" / "executor.py"
    module = _load_executor(path, "crm_mcp_react_under_test")

    assert hasattr(module, "build_executor")
    assert module._base.Agent.__name__ == "Agent"
    # MCP/ReAct framing applied, original action grammar preserved for the parser.
    assert "MCP-style tool catalog" in module._base.SYSTEM_PROMPT
    assert all(tag in module._base.SYSTEM_PROMPT for tag in ("<execute>", "<describe>", "<respond>"))
    assert "SYSTEM_PROMPT" in module._base.Agent.run.__code__.co_names


def test_crmarenapro_mcp_react_world_model_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("anthropic")
    pytest.importorskip("openai")
    monkeypatch.delenv("WM_STRATEGY", raising=False)
    path = ASSETS / "crmarenapro" / "purple-executors" / "mcp_react" / "executor.py"
    module = _load_executor(path, "crm_mcp_react_wm_under_test")

    # CRM messages -> ejepa_wm conversation_flow events, in order.
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Question: q"},
        {"role": "assistant", "content": "<execute>SELECT 1</execute>"},
        {"role": "user", "content": "[Observation: 1 row]"},
    ]
    flow = module.McpReactAgent._to_conversation_flow(msgs)
    assert [e["type"] for e in flow] == [
        "tools",
        "system_message",
        "user_message",
        "ai_message",
        "tool_result",
    ]
    assert flow[0]["tools"], "tool catalog should be non-empty"

    # No WM_STRATEGY configured -> plain ReAct (no world model built).
    wm, strategy = module._build_world_model()
    assert wm is None
    assert strategy == "none"
