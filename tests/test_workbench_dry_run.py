from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tomllib
import types
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import pytest

from tests.module_loading import load_module


def _load_workbench_green_module():
    """Load the WorkBench Green agent module directly from assets/."""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "WorkBench"
        / "green"
        / "workbench_green_agent.py"
    )
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    sys.modules.setdefault(
        "uvicorn",
        types.SimpleNamespace(Config=object, Server=object),
    )
    spec = importlib.util.spec_from_file_location("workbench_green_agent", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_workbench_mcp_react_executor_module():
    """Load the mcp_react executor module directly from assets/ (no openai/pandas needed)."""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "WorkBench"
        / "purple-executors"
        / "mcp_react"
        / "executor.py"
    )
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("workbench_mcp_react_executor", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_workbench_mcp_react_runtime_module():
    """Load the mcp_react runtime module directly from assets/."""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "WorkBench"
        / "purple-executors"
        / "mcp_react"
        / "runtime.py"
    )
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("workbench_mcp_react_runtime", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


workbench_green = _load_workbench_green_module()
workbench_mcp_react = _load_workbench_mcp_react_executor_module()


def test_longest_100_target_is_balanced_and_loadable(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = workbench_green.TaskLoader()
    task_ids_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "WorkBench"
        / "green"
        / "tasks"
        / "task_ids.toml"
    )
    with task_ids_path.open("rb") as handle:
        ids = tomllib.load(handle)["longest_100_balanced"]
    domains = Counter(task_id.rsplit("_", 1)[0] for task_id in ids)
    assert len(ids) == len(set(ids)) == 100
    assert sorted(domains.values()) == [15, 17, 17, 17, 17, 17]

    monkeypatch.setattr(loader, "_load_all_datasets", lambda: [{"id": task_id} for task_id in ids])
    assert [task["id"] for task in loader.load_tasks("longest_100_balanced")] == ids


# executor.py eagerly loads the sibling wm_react.py at import time (it needs no external
# WorkBench repo to do so -- only `ejepa_wm`, which lives in this repo).
workbench_wm_react = workbench_mcp_react._wm_react_module


class _FakeAgentModule:
    """Minimal stand-in for `src.evals.agent`'s constants used by wm_react's pure helpers."""

    FINAL_ANSWER = "Final Answer"
    PARSE_ERROR = "__parse_error__"


def test_validate_request_requires_agent_role_target_and_model_name() -> None:
    agent = workbench_green.WorkBenchGreenAgent()

    missing_role = types.SimpleNamespace(
        participants={}, config={"target": "sample", "model_name": "claude-sonnet-4.6"}
    )
    valid, message = agent.validate_request(missing_role)
    assert valid is False
    assert "agent" in message

    missing_target = types.SimpleNamespace(
        participants={"agent": "http://purple"}, config={"model_name": "claude-sonnet-4.6"}
    )
    valid, message = agent.validate_request(missing_target)
    assert valid is False
    assert "target" in message

    missing_model = types.SimpleNamespace(
        participants={"agent": "http://purple"}, config={"target": "sample"}
    )
    valid, message = agent.validate_request(missing_model)
    assert valid is False
    assert "model_name" in message

    ok = types.SimpleNamespace(
        participants={"agent": "http://purple"},
        config={"target": "sample", "model_name": "claude-sonnet-4.6"},
    )
    assert agent.validate_request(ok) == (True, "ok")


def test_validate_request_rejects_bad_tool_selection() -> None:
    agent = workbench_green.WorkBenchGreenAgent()

    request = types.SimpleNamespace(
        participants={"agent": "http://purple"},
        config={"target": "sample", "model_name": "claude-sonnet-4.6", "tool_selection": "bogus"},
    )
    valid, message = agent.validate_request(request)
    assert valid is False
    assert "tool_selection" in message


def test_build_purple_payload_omits_ground_truth() -> None:
    task = {
        "id": "sample_0000",
        "task": "Delete my last email from nadia",
        "outcome": ['email.delete_email.func(email_id="00000479")'],
        "domains": ["email"],
        "target": "sample",
    }

    payload = json.loads(
        workbench_green._build_purple_payload(
            task,
            {"model_name": "claude-sonnet-4.6", "tool_selection": "domains"},
        )
    )

    assert payload["task"] == task["task"]
    assert payload["domains"] == ["email"]
    assert payload["model_name"] == "claude-sonnet-4.6"
    assert payload["tool_selection"] == "domains"
    assert "outcome" not in payload


def test_parse_purple_outcome_handles_malformed_response() -> None:
    empty = workbench_green._parse_purple_outcome("")
    assert empty["function_calls"] == []
    assert empty["error"] == "empty response"

    not_json = workbench_green._parse_purple_outcome("not json")
    assert not_json["function_calls"] == []
    assert not_json["error"] == "invalid JSON response"

    ok = workbench_green._parse_purple_outcome(
        json.dumps({"function_calls": ['a.b.func(x="1")'], "error": "", "full_response": "done"})
    )
    assert ok["function_calls"] == ['a.b.func(x="1")']
    assert ok["error"] == ""
    assert ok["full_response"] == "done"


def test_run_eval_dry_run_writes_result_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Updater:
        def __init__(self) -> None:
            self.artifacts: dict[str, str] = {}

        async def update_status(self, *args, **kwargs) -> None:
            return None

        async def add_artifact(self, *, parts, name: str) -> None:
            self.artifacts[name] = parts[0].root.text

    agent = workbench_green.WorkBenchGreenAgent()
    agent._task_loader.load_tasks = lambda *args, **kwargs: [
        {
            "id": "sample_0000",
            "task": "Delete my last email from nadia",
            "outcome": ['email.delete_email.func(email_id="00000479")'],
            "domains": ["email"],
            "target": "sample",
        }
    ]

    async def _fake_evaluate_task(**kwargs) -> dict:
        task = kwargs["task"]
        return {
            "task_id": task["id"],
            "predicted_text": json.dumps(
                {"function_calls": task["outcome"], "full_response": "done", "error": ""}
            ),
            "score_value": 1.0,
            "side_effects_value": False,
            "task_error": None,
            "task_result": {"task_id": task["id"], "score": 1.0, "unwanted_side_effects": False},
            "detail_record": {
                "task_id": task["id"],
                "score": 1.0,
                "unwanted_side_effects": False,
                "reason": "offline dry run",
            },
        }

    monkeypatch.setattr(agent, "_evaluate_task", _fake_evaluate_task)
    monkeypatch.setenv("BENCHMARK_RESULT_ROOT", os.fspath(tmp_path / "results"))
    monkeypatch.setenv("BENCHMARK_USER_NAME", "tester")

    request = types.SimpleNamespace(
        participants={"agent": "http://127.0.0.1:8080/"},
        config={
            "target": "sample",
            "model_name": "claude-sonnet-4.6",
            "task_ids": ["sample_0000"],
        },
    )
    updater = _Updater()

    asyncio.run(agent.run_eval(request, updater))

    eval_result = json.loads(updater.artifacts["EvaluationResult"])
    assert eval_result["total_tasks"] == 1
    assert eval_result["score_rate"] == pytest.approx(1.0)
    assert eval_result["task_results"][0]["task_id"] == "sample_0000"

    detail_path = Path(updater.artifacts["EvaluationDetailFile"])
    assert detail_path.exists()
    detail_payload = json.loads(detail_path.read_text(encoding="utf-8"))
    assert detail_payload["details"][0]["task_id"] == "sample_0000"
    assert detail_payload["avg_unwanted_side_effects_rate"] == pytest.approx(0.0)


def test_executor_parse_payload_requires_task_and_model_name() -> None:
    with pytest.raises(ValueError, match="task"):
        workbench_mcp_react._parse_payload(json.dumps({"model_name": "claude-sonnet-4.6"}))

    with pytest.raises(ValueError, match="model_name"):
        workbench_mcp_react._parse_payload(json.dumps({"task": "do something"}))

    payload = workbench_mcp_react._parse_payload(
        json.dumps({"task": "do something", "model_name": "claude-sonnet-4.6"})
    )
    assert payload["task"] == "do something"
    assert payload["model_name"] == "claude-sonnet-4.6"


def test_executor_configures_vllm_route_without_stripping_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ModelConfig(NamedTuple):
        model_id: str
        supports_temperature: bool
        provider: str = "openrouter"

    class Route(NamedTuple):
        model_id: str
        base_url: str
        api_key: str
        provider: str
        supports_temperature: bool

    delegated: list[str] = []

    def original_resolve_route(model_name: str) -> Route:
        delegated.append(model_name)
        return Route("upstream-model", "https://example.invalid/v1", "key", "openrouter", True)

    fake_agent_module = types.SimpleNamespace(
        MODEL_REGISTRY={},
        ModelConfig=ModelConfig,
        Route=Route,
        resolve_route=original_resolve_route,
        _PROVIDER_BASE_URLS={},
        _PROVIDER_API_KEY_ENV={},
    )

    monkeypatch.setenv("WORKBENCH_VLLM_BASE_URL", "http://127.0.0.1:9001/v1/")
    monkeypatch.setenv("WORKBENCH_VLLM_MODEL", "Qwen/Qwen3.5-27B")
    monkeypatch.setenv("WORKBENCH_VLLM_MODEL_NAME", "local-qwen")

    assert workbench_mcp_react._configure_vllm_agent_route(fake_agent_module) is True

    route = fake_agent_module.resolve_route("local-qwen")
    assert route == Route("Qwen/Qwen3.5-27B", "http://127.0.0.1:9001/v1", "EMPTY", "vllm", True)
    assert fake_agent_module.MODEL_REGISTRY["local-qwen"] == ModelConfig(
        "Qwen/Qwen3.5-27B",
        True,
        "vllm",
    )
    assert fake_agent_module.resolve_route("Qwen/Qwen3.5-27B") == route
    assert fake_agent_module.MODEL_REGISTRY["Qwen/Qwen3.5-27B"] == ModelConfig(
        "Qwen/Qwen3.5-27B",
        True,
        "vllm",
    )

    assert fake_agent_module.resolve_route("claude-sonnet-4.6").model_id == "upstream-model"
    assert delegated == ["claude-sonnet-4.6"]


def test_executor_registers_vllm_served_model_as_model_name_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ModelConfig(NamedTuple):
        model_id: str
        supports_temperature: bool
        provider: str = "openrouter"

    class Route(NamedTuple):
        model_id: str
        base_url: str
        api_key: str
        provider: str
        supports_temperature: bool

    def original_resolve_route(model_name: str) -> Route:
        raise AssertionError(f"unexpected upstream route lookup for {model_name}")

    fake_agent_module = types.SimpleNamespace(
        MODEL_REGISTRY={},
        ModelConfig=ModelConfig,
        Route=Route,
        resolve_route=original_resolve_route,
        _PROVIDER_BASE_URLS={},
        _PROVIDER_API_KEY_ENV={},
    )

    monkeypatch.setenv("WORKBENCH_VLLM_BASE_URL", "http://127.0.0.1:9012/v1")
    monkeypatch.setenv("WORKBENCH_VLLM_API_KEY", "EMPTY")
    monkeypatch.setenv("WORKBENCH_VLLM_MODEL", "wm_agent3")
    monkeypatch.delenv("WORKBENCH_VLLM_MODEL_NAME", raising=False)

    assert workbench_mcp_react._configure_vllm_agent_route(fake_agent_module) is True

    expected_route = Route("wm_agent3", "http://127.0.0.1:9012/v1", "EMPTY", "vllm", True)
    assert fake_agent_module.resolve_route("local-vllm") == expected_route
    assert fake_agent_module.resolve_route("wm_agent3") == expected_route
    assert fake_agent_module.MODEL_REGISTRY["local-vllm"] == ModelConfig(
        "wm_agent3",
        True,
        "vllm",
    )
    assert fake_agent_module.MODEL_REGISTRY["wm_agent3"] == ModelConfig(
        "wm_agent3",
        True,
        "vllm",
    )


def _load_workbench_mcp_react_runtime():
    """Load the WorkBench mcp_react executor runtime module directly from assets/."""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "WorkBench"
        / "purple-executors"
        / "mcp_react"
        / "runtime.py"
    )
    return load_module("workbench_mcp_react_runtime", module_path)


def test_runtime_accepts_workbench_vllm_env_as_llm_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbench_mcp_react_runtime = _load_workbench_mcp_react_runtime()
    for env_name in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "WORKBENCH_VLLM_BASE_URL",
        "WORKBENCH_VLLM_API_KEY",
        "WORKBENCH_VLLM_MODEL",
        "WORKBENCH_VLLM_MODEL_NAME",
    ):
        monkeypatch.delenv(env_name, raising=False)

    assert "OPENROUTER_API_KEY" in workbench_mcp_react_runtime._missing_env_vars()

    monkeypatch.setenv("WORKBENCH_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    assert workbench_mcp_react_runtime._missing_env_vars() == []


def test_wm_react_tool_calls_from_action_ignores_terminal_actions() -> None:
    agent_module = _FakeAgentModule()

    assert workbench_wm_react._tool_calls_from_action(agent_module, "Final Answer", "done") == []
    assert workbench_wm_react._tool_calls_from_action(agent_module, "__parse_error__", "oops") == []

    calls = workbench_wm_react._tool_calls_from_action(
        agent_module, "email.delete_email", {"email_id": "00000479"}
    )
    assert calls == [{"name": "email.delete_email", "args": {"email_id": "00000479"}}]

    string_input_calls = workbench_wm_react._tool_calls_from_action(
        agent_module, "email.search_emails", "nadia"
    )
    assert string_input_calls == [{"name": "email.search_emails", "args": {"input": "nadia"}}]


def test_wm_react_ai_event_wraps_tool_calls() -> None:
    agent_module = _FakeAgentModule()

    event = workbench_wm_react._ai_event(
        agent_module, "raw response text", "email.delete_email", {"email_id": "1"}
    )
    assert event == {
        "type": "ai_message",
        "content": "raw response text",
        "tool_calls": [{"name": "email.delete_email", "args": {"email_id": "1"}}],
    }

    final_event = workbench_wm_react._ai_event(agent_module, "done", "Final Answer", "done")
    assert final_event["tool_calls"] == []


def test_wm_react_response_with_call_is_a_parseable_json_blob() -> None:
    response = workbench_wm_react._response_with_call(
        {"name": "email.delete_email", "arguments": {"email_id": "1"}}
    )
    assert "```" in response
    blob_text = response.split("```")[1].strip()
    blob = json.loads(blob_text)
    assert blob == {"action": "email.delete_email", "action_input": {"email_id": "1"}}


def test_run_single_task_with_wm_delegates_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With WM_STRATEGY unset, the WM-aware wrapper is a pure pass-through."""
    # The world model is built per task now, so there is no cached instance to clear --
    # only the latched "build failed" flag, which an earlier test may have set.
    monkeypatch.setattr(workbench_wm_react, "_wm_build_failed", False)
    monkeypatch.delenv("WM_STRATEGY", raising=False)

    calls: list[tuple] = []

    def fake_run_single_task(
        index,
        task,
        model_name,
        tools,
        datetime_prefix,
        act_without_confirmation,
        structured_outputs,
    ):
        calls.append((index, task, model_name, structured_outputs))
        return {
            "task": task,
            "function_calls": ["stub.call.func()"],
            "full_response": "ok",
            "error": "",
            "trace": [],
            "_index": index,
        }

    fake_inference_module = types.SimpleNamespace(_run_single_task=fake_run_single_task)

    result = workbench_wm_react.run_single_task_with_wm(
        0,
        "Delete my last email from nadia",
        "claude-sonnet-4.6",
        [],
        "Today is a test day.",
        False,
        False,
        fake_inference_module,
        _FakeAgentModule(),
    )

    assert calls == [(0, "Delete my last email from nadia", "claude-sonnet-4.6", False)]
    assert result["function_calls"] == ["stub.call.func()"]
    assert result["wm_steps"] == []


def test_world_model_is_built_per_task_not_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each task must get its own world model so `max_parallel > 1` is safe.

    The JEPA MPC state (`_beam_imagined_plan`, `_beam_plan_cursor`, critic counters)
    lives on the world-model instance and `reset_episode()` wipes it, so a shared
    instance let one task destroy another's live plan -- silently.
    """
    monkeypatch.setattr(workbench_wm_react, "_wm_build_failed", False)
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")

    built: list[object] = []

    class _FakeWm:
        def __init__(self) -> None:
            self.plan: list[str] = []

    def fake_build(config, chat_fn=None):
        wm = _FakeWm()
        built.append(wm)
        return wm

    fake_ejepa_wm = types.SimpleNamespace(
        build_world_model=fake_build,
        wm_config_from_env=lambda: types.SimpleNamespace(
            strategy="beam_plan", backend="ewm_imagined", n=1
        ),
    )
    monkeypatch.setitem(sys.modules, "ejepa_wm", fake_ejepa_wm)

    first, first_config = workbench_wm_react._get_world_model(object())
    second, _ = workbench_wm_react._get_world_model(object())

    assert first is not None and second is not None
    assert first is not second, "two tasks must not share one world-model instance"
    assert len(built) == 2
    assert first_config.strategy == "beam_plan"

    # Episode state on one instance must be invisible to the other.
    first.plan.append("task-a-step")
    assert second.plan == []


def test_world_model_build_failure_is_latched(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken WM config must not be retried (and re-logged) on all 100+ tasks."""
    monkeypatch.setattr(workbench_wm_react, "_wm_build_failed", False)
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")

    attempts: list[int] = []

    def exploding_config():
        attempts.append(1)
        raise ValueError("bad checkpoint")

    monkeypatch.setitem(
        sys.modules,
        "ejepa_wm",
        types.SimpleNamespace(
            build_world_model=lambda *a, **k: None, wm_config_from_env=exploding_config
        ),
    )

    assert workbench_wm_react._get_world_model(object()) == (None, None)
    assert workbench_wm_react._get_world_model(object()) == (None, None)
    assert len(attempts) == 1, "the failed build must be attempted once, then latched"
