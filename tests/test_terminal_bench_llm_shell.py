from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXECUTOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "Terminal-Bench-2.0"
    / "purple-executors"
    / "llm_shell"
    / "executor.py"
)


def _load_executor_module():
    spec = importlib.util.spec_from_file_location("tb_llm_shell_executor", EXECUTOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Required for @dataclass with postponed annotations on Python 3.11.
    sys.modules["tb_llm_shell_executor"] = module
    spec.loader.exec_module(module)
    return module


def test_azure_config_uses_deployment_url(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_executor_module()
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://example.openai.azure.com/openai/v1/",
    )
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.5")
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    monkeypatch.delenv("TERMINAL_BENCH_LLM_MODEL", raising=False)

    config = mod._resolve_llm_config()
    assert config.provider == "azure"
    assert config.deployment == "gpt-5.5"
    assert config.api_version == mod.DEFAULT_AZURE_API_VERSION
    url = mod._chat_completions_url(config)
    assert "deployments/gpt-5.5/chat/completions" in url
    assert "api-version=" in url
    headers = mod._chat_completions_headers(config)
    assert headers["api-key"] == "test-key"
    assert "Authorization" not in headers


def test_azure_preferred_over_openai_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_executor_module()
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.5")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    config = mod._resolve_llm_config()
    assert config.provider == "azure"
    assert config.api_key == "azure-key"


def test_parse_json_object_tolerates_trailing_text() -> None:
    mod = _load_executor_module()
    payload = mod._parse_json_object(
        '{"kind":"exec_request","command":"ls -la","timeout":30}\nI will list files next.',
        source="test",
    )
    assert payload["kind"] == "exec_request"
    assert payload["command"] == "ls -la"


def test_parse_json_object_tolerates_markdown_fence() -> None:
    mod = _load_executor_module()
    payload = mod._parse_json_object(
        '```json\n{"kind":"final","output":"done"}\n```',
        source="test",
    )
    assert payload["kind"] == "final"


def test_gpt5_body_uses_max_completion_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_executor_module()
    config = mod._LlmConfig(
        provider="azure",
        api_key="k",
        model="gpt-5.5",
        deployment="gpt-5.5",
        base_url="https://example.openai.azure.com/",
        api_version="2025-04-01-preview",
    )
    body = mod._build_chat_body(config, [{"role": "user", "content": "hi"}])
    assert "max_completion_tokens" in body
    assert body["max_completion_tokens"] == mod.DEFAULT_GPT5_MAX_COMPLETION_TOKENS
    assert "temperature" not in body
    assert "max_tokens" not in body
    assert "model" not in body


def test_tool_history_becomes_user_message() -> None:
    mod = _load_executor_module()
    session = mod._Session(
        instruction="fix git",
        history=[
            {
                "role": "tool",
                "content": '{"kind":"exec_result","exit_code":0,"stdout":"ok","stderr":""}',
            }
        ],
    )
    messages = mod._build_chat_messages(session)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "user"
    assert "Shell command result" in messages[2]["content"]
