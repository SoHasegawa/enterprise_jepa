from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EXECUTOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "Terminal-Bench-2.0"
    / "purple-executors"
    / "terminus_2"
    / "executor.py"
)
PARSERS_PATH = EXECUTOR_PATH.parent / "parsers.py"


def _load_module(path: Path, name: str):
    executor_dir = str(path.parent)
    if executor_dir not in sys.path:
        sys.path.insert(0, executor_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_keystrokes_to_shell_command() -> None:
    parsers = _load_module(PARSERS_PATH, "tb_terminus_parsers")
    cmd = parsers.keystrokes_to_exec("ls -la\n")
    assert cmd is not None
    assert cmd.command == "ls -la"
    assert cmd.timeout == 30


def test_duration_0_1_maps_to_min_exec_timeout() -> None:
    parsers = _load_module(PARSERS_PATH, "tb_terminus_parsers_duration")
    assert parsers.duration_to_exec_timeout(0.1, "ls -la") == 30
    assert parsers.duration_to_exec_timeout(0.1, "pwd") == 30


def test_slow_command_gets_longer_timeout() -> None:
    parsers = _load_module(PARSERS_PATH, "tb_terminus_parsers_slow")
    timeout = parsers.duration_to_exec_timeout(
        1.0, "apt-get update && apt-get install -y build-essential"
    )
    assert timeout >= 180


def test_convert_json_response_to_exec_queue() -> None:
    parsers = _load_module(PARSERS_PATH, "tb_terminus_parsers2")
    parser = parsers.get_parser("json")
    response = json.dumps(
        {
            "analysis": "list files",
            "plan": "run ls",
            "commands": [
                {"keystrokes": "pwd\n", "duration": 0.1},
                {"keystrokes": "ls\n", "duration": 0.1},
            ],
            "task_complete": False,
        }
    )
    parsed = parsers.convert_parse_result(parser.parse_response(response))
    assert len(parsed.commands) == 2
    assert parsed.commands[0].command == "pwd"
    assert parsed.commands[0].timeout >= 30
    assert parsed.commands[1].command == "ls"
    assert parsed.commands[1].timeout >= 30
    assert parsed.is_task_complete is False


def test_gpt5_litellm_kwargs_omit_temperature() -> None:
    mod = _load_module(EXECUTOR_PATH, "tb_terminus_executor_gpt5")
    config = mod._LiteLlmConfig(
        litellm_model="azure/gpt-5.5",
        api_key="k",
        provider="azure",
        deployment="gpt-5.5",
        temperature=0.7,
    )
    kwargs = mod._litellm_completion_kwargs(config)
    assert "temperature" not in kwargs
    assert "max_completion_tokens" in kwargs
    assert "max_tokens" not in kwargs


def test_non_gpt5_litellm_kwargs_include_temperature() -> None:
    mod = _load_module(EXECUTOR_PATH, "tb_terminus_executor_gpt4")
    config = mod._LiteLlmConfig(
        litellm_model="gpt-4.1",
        api_key="k",
        provider="openai",
        temperature=0.7,
    )
    kwargs = mod._litellm_completion_kwargs(config)
    assert kwargs["temperature"] == 0.7
    assert "max_tokens" in kwargs


def test_azure_litellm_config(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module(EXECUTOR_PATH, "tb_terminus_executor")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://example.openai.azure.com/openai/v1/",
    )
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.5")
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

    config = mod._resolve_litellm_config()
    assert config.provider == "azure"
    assert config.litellm_model == "azure/gpt-5.5"
    assert config.api_version == mod.DEFAULT_AZURE_API_VERSION
    assert config.api_base == "https://example.openai.azure.com/"


def test_openai_compatible_litellm_model_prefixes_huggingface_style_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module(EXECUTOR_PATH, "tb_terminus_executor_vllm")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "Qwen/Qwen3.5-27B")

    config = mod._resolve_litellm_config()
    assert config.provider == "openai"
    assert config.litellm_model == "openai/Qwen/Qwen3.5-27B"
    assert config.api_base == "http://127.0.0.1:8000/v1"
