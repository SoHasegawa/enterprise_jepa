from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from a2a.server.agent_execution import AgentExecutor
from a2a.utils.errors import ServerError

from common.green_executor import BaseGreenAgent, BenchmarkGreenExecutor
from common.models import EvalRequest, RuntimeFeedbackRequest, RuntimeFeedbackResponse
from common.purple_router import PurpleExecutorRegistry


class _FakeGreenAgent(BaseGreenAgent):
    def __init__(self, *, fail_eval: bool = False) -> None:
        self.fail_eval = fail_eval
        self.eval_requests: list[EvalRequest] = []
        self.feedback_requests: list[RuntimeFeedbackRequest] = []

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        return ("target" in request.config, "ok" if "target" in request.config else "missing")

    def validate_runtime_feedback_request(
        self,
        request: RuntimeFeedbackRequest,
    ) -> tuple[bool, str]:
        return (request.target == "sample", "ok")

    async def run_eval(self, request: EvalRequest, updater) -> None:
        self.eval_requests.append(request)
        if self.fail_eval:
            raise RuntimeError("eval failed")
        await updater.update_status("running", "eval")

    async def run_runtime_feedback(
        self,
        request: RuntimeFeedbackRequest,
        updater,
    ) -> RuntimeFeedbackResponse:
        del updater
        self.feedback_requests.append(request)
        return RuntimeFeedbackResponse(
            target=request.target,
            task_id=request.task_id,
            generation=request.generation,
            answer=request.answer,
            score=0.5,
            reward=0.5,
            all_passed=False,
            precision=0.5,
            recall=0.5,
            matched_count=1,
            detected_count=2,
            labeled_count=2,
            matching_mode="unit",
            reason="partial",
            eval_result={"ok": False},
        )


class _FakeTaskUpdater:
    instances: ClassVar[list[_FakeTaskUpdater]] = []

    def __init__(self, event_queue, task_id: str, context_id: str) -> None:
        self.event_queue = event_queue
        self.task_id = task_id
        self.context_id = context_id
        self.statuses: list[tuple[object, object]] = []
        self.artifacts: list[dict[str, object]] = []
        self.completed = False
        _FakeTaskUpdater.instances.append(self)

    async def update_status(self, state, message, final: bool = False) -> None:
        self.statuses.append((state, message, final))

    async def add_artifact(self, **kwargs) -> None:
        self.artifacts.append(kwargs)

    async def complete(self) -> None:
        self.completed = True


class _FakeEventQueue:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def enqueue_event(self, event) -> None:
        self.events.append(event)


def _context(payload: dict[str, object]):
    return SimpleNamespace(
        current_task=SimpleNamespace(id="task-1", context_id="ctx-1"),
        message=None,
        get_user_input=lambda: json.dumps(payload),
    )


@pytest.fixture(autouse=True)
def fake_task_updater(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeTaskUpdater.instances.clear()
    monkeypatch.setattr("common.green_executor.TaskUpdater", _FakeTaskUpdater)


def test_base_green_agent_runtime_feedback_defaults() -> None:
    request = RuntimeFeedbackRequest(target="t", task_id="1", generation=1, answer="a")

    class _Agent(BaseGreenAgent):
        async def run_eval(self, request: EvalRequest, updater) -> None:
            del request, updater

        def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
            del request
            return True, "ok"

    agent = _Agent()
    assert agent.validate_runtime_feedback_request(request) == (
        False,
        "runtime feedback is not supported by this Green agent",
    )


def test_benchmark_green_executor_runs_eval_request() -> None:
    agent = _FakeGreenAgent()
    executor = BenchmarkGreenExecutor(agent)
    event_queue = _FakeEventQueue()
    payload = {"participants": {"agent": "http://127.0.0.1:8000/"}, "config": {"target": "sample"}}

    asyncio.run(executor.execute(_context(payload), event_queue))

    assert len(agent.eval_requests) == 1
    assert event_queue.events[0].id == "task-1"
    assert _FakeTaskUpdater.instances[-1].completed is True
    assert _FakeTaskUpdater.instances[-1].artifacts


def test_benchmark_green_executor_runs_runtime_feedback_request() -> None:
    agent = _FakeGreenAgent()
    executor = BenchmarkGreenExecutor(agent)
    event_queue = _FakeEventQueue()
    payload = {
        "kind": "runtime_feedback",
        "target": "sample",
        "task_id": "task-a",
        "generation": 2,
        "answer": "ok",
    }

    asyncio.run(executor.execute(_context(payload), event_queue))

    assert len(agent.feedback_requests) == 1
    artifact_payload = _FakeTaskUpdater.instances[-1].artifacts[-1]
    assert artifact_payload["name"] == "runtime_feedback_response"


def test_benchmark_green_executor_reports_invalid_and_failed_requests() -> None:
    executor = BenchmarkGreenExecutor(_FakeGreenAgent())
    with pytest.raises(ServerError):
        asyncio.run(
            executor.execute(_context({"participants": {}, "config": {}}), _FakeEventQueue())
        )

    failing = BenchmarkGreenExecutor(_FakeGreenAgent(fail_eval=True))
    with pytest.raises(ServerError):
        asyncio.run(
            failing.execute(
                _context(
                    {
                        "participants": {"agent": "http://127.0.0.1:8000/"},
                        "config": {"target": "sample"},
                    }
                ),
                _FakeEventQueue(),
            )
        )

    assert _FakeTaskUpdater.instances[-1].statuses[-1][2] is True


def test_benchmark_green_executor_cancel_is_unsupported() -> None:
    with pytest.raises(ServerError):
        asyncio.run(BenchmarkGreenExecutor(_FakeGreenAgent()).cancel(object(), _FakeEventQueue()))


def test_purple_executor_registry_loads_runtime_and_executor(tmp_path: Path) -> None:
    executor_dir = tmp_path / "purple-executors" / "demo"
    executor_dir.mkdir(parents=True)
    marker = tmp_path / "runtime-marker.txt"
    runtime_path = executor_dir / "runtime.py"
    runtime_path.write_text(
        "\n".join(
            [
                "from contextlib import contextmanager",
                "@contextmanager",
                "def maybe_manage_executor_runtime():",
                f"    open({str(marker)!r}, 'a', encoding='utf-8').write('enter\\n')",
                "    try:",
                "        yield",
                "    finally:",
                f"        open({str(marker)!r}, 'a', encoding='utf-8').write('exit\\n')",
            ]
        ),
        encoding="utf-8",
    )
    (executor_dir / "executor.py").write_text(
        "\n".join(
            [
                "from a2a.server.agent_execution import AgentExecutor",
                "class DemoExecutor(AgentExecutor):",
                "    async def execute(self, context, event_queue):",
                "        pass",
                "    async def cancel(self, request, event_queue):",
                "        return None",
                "def build_executor():",
                "    return DemoExecutor()",
            ]
        ),
        encoding="utf-8",
    )

    registry = PurpleExecutorRegistry(benchmark_dir=tmp_path, module_prefix="unit")
    executor = registry.get_executor("demo")

    assert isinstance(executor, AgentExecutor)
    assert registry.get_executor("demo") is executor
    assert marker.read_text(encoding="utf-8") == "enter\n"


def test_purple_executor_registry_reports_bad_executor_modules(tmp_path: Path) -> None:
    registry = PurpleExecutorRegistry(benchmark_dir=tmp_path, module_prefix="unit")
    with pytest.raises(FileNotFoundError, match="Executor not found"):
        registry.executor_path("missing")

    executor_dir = tmp_path / "purple-executors" / "bad"
    executor_dir.mkdir(parents=True)
    (executor_dir / "executor.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(AttributeError, match="build_executor"):
        registry.get_executor("bad")
