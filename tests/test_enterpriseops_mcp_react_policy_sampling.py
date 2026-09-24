"""Tests for automatic shared-prompt sampling through EnterpriseOps policy vLLM."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from ejepa_wm.backends._ewm_runtime import EwmGenerator, sample_many
from ejepa_wm.backends.ewm_imagined import _ChatFnAgent

REPO_ROOT = Path(__file__).resolve().parents[1]
WM_REACT_PATH = (
    REPO_ROOT / "assets" / "EnterpriseOps-Gym" / "purple-executors" / "mcp_react" / "wm_react.py"
)


def _load_wm_react(monkeypatch):
    orchestrators = ModuleType("orchestrators")
    react = ModuleType("orchestrators.react")
    react.ReactOrchestrator = type("ReactOrchestrator", (), {})
    messages = ModuleType("langchain_core.messages")

    class Message:
        def __init__(self, content=""):
            self.content = content

    messages.AIMessage = Message
    messages.HumanMessage = Message
    messages.SystemMessage = Message
    messages.ToolMessage = Message
    monkeypatch.setitem(sys.modules, "orchestrators", orchestrators)
    monkeypatch.setitem(sys.modules, "orchestrators.react", react)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages)

    spec = importlib.util.spec_from_file_location(
        "enterpriseops_wm_react_sampling_test", WM_REACT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ssot_uses_policy_vllm_as_one_n_sample_request(monkeypatch) -> None:
    wm_react = _load_wm_react(monkeypatch)
    request_bodies = []

    def fake_post(_self, body):
        request_bodies.append(body)
        return {
            "choices": [{"message": {"content": f"plan-{index}"}} for index in range(body["n"])]
        }

    monkeypatch.setattr(EwmGenerator, "_post_chat_completions", fake_post)
    policy_client = SimpleNamespace(
        provider="vllm",
        model="wm_agent",
        api_key="not-needed",
        custom_api_endpoint="http://127.0.0.1:9010/v1",
        max_tokens=4096,
        llm=None,
    )
    generator = _ChatFnAgent(wm_react._make_chat_fn(policy_client))

    texts, requests_issued = sample_many(
        generator,
        [{"role": "user", "content": "generate a plan"}],
        temperature=1.0,
        num_samples=7,
        response_format={"type": "json_object"},
    )

    assert requests_issued == 1
    assert texts == [f"plan-{index}" for index in range(7)]
    assert len(request_bodies) == 1
    assert request_bodies[0]["n"] == 7
    assert request_bodies[0]["model"] == "wm_agent"
    assert request_bodies[0]["response_format"] == {"type": "json_object"}


def test_non_vllm_policy_keeps_existing_sampling_fallback(monkeypatch) -> None:
    wm_react = _load_wm_react(monkeypatch)
    policy_client = SimpleNamespace(
        provider="anthropic",
        model="policy-model",
        api_key="secret",
        custom_api_endpoint=None,
        max_tokens=4096,
        llm=None,
    )

    chat_fn = wm_react._make_chat_fn(policy_client)

    try:
        chat_fn.generate_samples([], num_samples=2)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("non-vLLM policy unexpectedly used the direct vLLM sampler")
