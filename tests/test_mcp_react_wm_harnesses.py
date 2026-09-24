from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "assets"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"Cannot load: {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeAdviceWM:
    agent_call_count = 1

    def __init__(self) -> None:
        self.history_arg: list[str] | None = None

    def advise(self, _flow: list[dict[str, Any]], *, history: list[str] | None = None) -> Any:
        self.history_arg = history
        return types.SimpleNamespace(
            text="ITP ADVICE", detail={"current_state": "imagined-state"}
        )


class _FinalTurnCRMDatabase:
    executed_queries: ClassVar[list[str]] = []
    available = True

    def __init__(self, org_type: str = "b2b") -> None:
        self.org_type = org_type
        self.failed_queries = 0

    def get_tables(self) -> list[str]:
        return ["Lead"]

    def execute_query(self, query: str) -> dict[str, Any]:
        self.executed_queries.append(query)
        return {"success": True, "count": 1, "data": [{"answer": "observed"}]}

    def describe_table(self, _table_name: str) -> dict[str, Any]:
        raise AssertionError("final reserved turn should not call describe")

    def close(self) -> None:
        pass


class _FinalTurnBaselineAgent:
    def __init__(self) -> None:
        self.api_key = "test-key"
        self.metrics = {"tokens": 0, "tool_calls": 0, "queries": 0, "turns": 0}
        self.max_turns = 2
        self.temperature = 0.0
        self.trajectory: list[dict[str, Any]] = []
        self.final_messages: list[dict[str, str]] = []

    def reset_metrics(self) -> None:
        self.metrics = {"tokens": 0, "tool_calls": 0, "queries": 0, "turns": 0}
        self.trajectory = []

    def _parse_task(self, input_text: str) -> dict[str, Any]:
        data = json.loads(input_text)
        return {
            "task_id": data["task_id"],
            "category": data["task_category"],
            "prompt": data["prompt"],
            "context": "",
            "optional_context": "",
            "config": {},
            "entropy": {},
        }

    def _extract_action(self, response: str) -> dict[str, Any]:
        action = {"type": None, "content": None}
        for name in ("execute", "describe", "respond"):
            match = re.search(rf"<{name}>(.*?)</{name}>", response, re.DOTALL | re.IGNORECASE)
            if match:
                action["type"] = name
                action["content"] = match.group(1).strip()
                break
        return action

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        self.final_messages = messages
        return "<respond>observed</respond>"

    def _fallback_answer(self, response: str) -> str:
        return response

    def _trajectory_payload(self, _task_id: str, _category: str) -> dict[str, Any]:
        return {"payload": {"info": {}}, "executor": "baseline_crm_agent"}

    async def _add_internal_trajectory_artifact(self, updater, **_kwargs) -> None:
        await updater.add_artifact(parts=[], name="internal_trajectory")


class _ArtifactCaptureUpdater:
    def __init__(self) -> None:
        self.artifacts: list[dict[str, Any]] = []

    async def update_status(self, *_args, **_kwargs) -> None:
        pass

    async def add_artifact(self, *, parts, name: str = "", **_kwargs) -> None:
        self.artifacts.append({"name": name, "parts": parts})


def test_workbench_itp_i_advises_with_state_history() -> None:
    wm_react = _load_module(
        ASSETS / "WorkBench" / "purple-executors" / "mcp_react" / "wm_react.py",
        "workbench_wm_react_harness_test",
    )
    calls: list[str] = []
    fake_agent_module = types.SimpleNamespace(
        call_llm=lambda _model, _system, human, _temperature: calls.append(human) or "ACTION"
    )
    fake_wm = _FakeAdviceWM()
    wm_steps: list[dict[str, Any]] = []
    state_history: list[str] = []

    response = wm_react._wm_step(
        agent_module=fake_agent_module,
        wm=fake_wm,
        wm_config=types.SimpleNamespace(strategy="itp_i", backend="noop", n=1),
        system_prompt="system",
        human_msg="human",
        conversation_flow=[{"type": "user_message", "content": "task"}],
        model_name="model",
        temperature=0.0,
        iteration=0,
        wm_steps=wm_steps,
        state_history=state_history,
    )

    assert response == "ACTION"
    assert calls == ["human\n\nITP ADVICE"]
    assert fake_wm.history_arg is state_history
    assert state_history == ["imagined-state"]
    assert wm_steps[0]["strategy"] == "itp_i"
    assert wm_steps[0]["itp_i_policy_reflect_calls"] == 1


@pytest.mark.parametrize(
    ("benchmark", "module_name"),
    [
        ("Terminal-Bench-2.0", "terminal_bench_wm_react_harness_test"),
    ],
)
def test_shell_mcp_react_itp_i_injects_advice(
    benchmark: str, module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("a2a")
    monkeypatch.setenv("WM_STRATEGY", "itp_i")
    monkeypatch.setenv("WM_BACKEND", "noop")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("LLM_MODEL", "dummy-model")
    wm_react = _load_module(
        ASSETS / benchmark / "purple-executors" / "mcp_react" / "wm_react.py",
        module_name,
    )
    executor = wm_react.WmShellExecutor()
    fake_wm = _FakeAdviceWM()
    executor.wm = fake_wm

    seen_messages: list[list[dict[str, str]]] = []

    def _fake_complete(_self, messages: list[dict[str, str]], temperature=None) -> str:
        seen_messages.append(messages)
        return '{"kind":"final","output":"done"}'

    monkeypatch.setattr(type(executor), "_complete_text", _fake_complete)

    session = wm_react.base._Session(instruction="inspect the workspace")
    action = executor._next_action(session)

    assert action == {"kind": "final", "output": "done"}
    assert seen_messages[-1][-1]["content"] == "ITP ADVICE"
    assert fake_wm.history_arg is session.wm_state_history
    assert session.wm_state_history == ["imagined-state"]
    assert session.wm_steps[0]["strategy"] == "itp_i"
    assert session.wm_steps[0]["itp_i_policy_reflect_calls"] == 1


def test_crmarenapro_itp_i_injects_advice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("a2a")
    monkeypatch.delenv("WM_STRATEGY", raising=False)

    fake_agent = types.ModuleType("agent")
    fake_agent.SYSTEM_PROMPT = "system"
    fake_agent.CRMDatabase = object
    fake_agent._is_empty_answer = lambda _value: False
    fake_agent._uses_max_completion_tokens = lambda _model, _base_url: False

    class _BaselineAgent:
        def __init__(self) -> None:
            self.metrics = {"turns": 1}
            self.temperature = 0.0
            self.trajectory: list[dict[str, Any]] = []

        async def _call_llm(self, messages: list[dict[str, str]]) -> str:
            self.seen_messages = messages
            return "<respond>done</respond>"

        def _trajectory_payload(self, _task_id: str, _category: str) -> dict[str, Any]:
            return {"payload": {"info": {}}}

        def reset_metrics(self) -> None:
            self.metrics = {"turns": 0}

    fake_agent.Agent = _BaselineAgent
    monkeypatch.setitem(sys.modules, "agent", fake_agent)

    wm_react = _load_module(
        ASSETS / "crmarenapro" / "purple-executors" / "mcp_react" / "wm_react.py",
        "crmarenapro_wm_react_harness_test",
    )
    agent = wm_react.WmReactAgent()
    fake_wm = _FakeAdviceWM()
    agent.wm = fake_wm
    agent.wm_config = types.SimpleNamespace(strategy="itp_i", backend="noop", n=1)

    response = asyncio.run(
        agent._wm_step(
            [{"role": "user", "content": "Question: q"}],
            [{"type": "user_message", "content": "Question: q"}],
        )
    )

    assert response == "<respond>done</respond>"
    assert agent.seen_messages[-1]["content"] == "ITP ADVICE"
    assert fake_wm.history_arg is agent._wm_state_history
    assert agent._wm_state_history == ["imagined-state"]
    assert agent._wm_steps[0]["strategy"] == "itp_i"
    assert agent._wm_steps[0]["itp_i_policy_reflect_calls"] == 1


def test_crmarenapro_wm_reserves_final_turn_for_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("a2a")
    monkeypatch.delenv("WM_STRATEGY", raising=False)
    _FinalTurnCRMDatabase.executed_queries = []

    fake_agent = types.ModuleType("agent")
    fake_agent.SYSTEM_PROMPT = "system"
    fake_agent.CRMDatabase = _FinalTurnCRMDatabase
    fake_agent._is_empty_answer = lambda value: not str(value or "").strip()
    fake_agent._uses_max_completion_tokens = lambda _model, _base_url: False
    fake_agent.Agent = _FinalTurnBaselineAgent
    monkeypatch.setitem(sys.modules, "agent", fake_agent)

    wm_react = _load_module(
        ASSETS / "crmarenapro" / "purple-executors" / "mcp_react" / "wm_react.py",
        "crmarenapro_wm_react_final_turn_test",
    )
    agent = wm_react.WmReactAgent()
    agent.wm = object()
    agent.wm_config = types.SimpleNamespace(strategy="itp_i", backend="noop", n=1)

    wm_step_calls = 0

    async def _fake_wm_step(_messages, _flow) -> str:
        nonlocal wm_step_calls
        wm_step_calls += 1
        return "<execute>SELECT answer FROM Lead</execute>"

    agent._wm_step = _fake_wm_step

    from a2a.types import Message, Part, Role, TextPart

    message = Message(
        role=Role.user,
        parts=[
            Part(
                root=TextPart(
                    text=json.dumps({
                        "task_id": "task-1",
                        "task_category": "unit",
                        "prompt": "answer from CRM",
                    })
                )
            )
        ],
        message_id="msg-1",
    )
    updater = _ArtifactCaptureUpdater()

    asyncio.run(agent.run(message, updater))

    assert wm_step_calls == 1
    assert _FinalTurnCRMDatabase.executed_queries == ["SELECT answer FROM Lead"]
    assert agent.final_messages[-1]["content"].startswith("You have no tool calls remaining.")
    assert agent.metrics["forced_final_responses"] == 1
    assert any(
        item.get("event_type") == "finalization_guard" for item in agent.trajectory
    )
    answer_artifact = next(item for item in updater.artifacts if item["name"] == "Answer")
    assert answer_artifact["parts"][0].root.text == "observed"
