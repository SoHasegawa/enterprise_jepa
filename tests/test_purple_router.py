import asyncio
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, MessageSendParams, Part, Role, TextPart

from common.purple_protocol import build_purple_request_text
from common.purple_router import RoutedPurpleExecutor


def _context_with_text(text: str) -> RequestContext:
    message = Message(
        kind="message",
        role=Role.user,
        parts=[Part(TextPart(kind="text", text=text))],
        messageId="message-1",
        taskId="task-1",
        contextId="context-1",
    )
    return RequestContext(request=MessageSendParams(message=message))


class CaptureExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.seen_context: RequestContext | None = None

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        self.seen_context = context

    async def cancel(self, request: RequestContext, event_queue: EventQueue):
        return None


class CaptureRegistry:
    def __init__(self, executor: CaptureExecutor) -> None:
        self.executor = executor
        self.executor_names: list[str] = []

    def get_executor(self, executor_name: str) -> AgentExecutor:
        self.executor_names.append(executor_name)
        return self.executor


class FailingRegistry:
    def get_executor(self, executor_name: str) -> AgentExecutor:
        raise AssertionError(f"local executor should not be used: {executor_name}")


def test_routed_purple_executor_dispatches_to_requested_local_executor() -> None:
    capture_executor = CaptureExecutor()
    registry = CaptureRegistry(capture_executor)
    router = RoutedPurpleExecutor(registry=registry, default_executor="default")
    envelope = build_purple_request_text("plain task text", {"executor": "local"})
    context = _context_with_text(envelope)
    original_message = context.message
    assert original_message is not None

    asyncio.run(router.execute(context, EventQueue()))

    assert registry.executor_names == ["local"]
    assert capture_executor.seen_context is not None
    routed_message = capture_executor.seen_context.message
    assert routed_message is not None
    assert routed_message.message_id == original_message.message_id
    assert routed_message.task_id == original_message.task_id
    assert routed_message.context_id == original_message.context_id
    assert routed_message.parts[0].root.text == "plain task text"


def test_routed_purple_executor_forwards_to_executor_endpoint(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_send_message_with_files(**kwargs):
        calls.append(kwargs)
        return {"status": "completed", "response": "forwarded answer"}

    monkeypatch.setattr(
        "common.purple_router.send_message_with_files",
        fake_send_message_with_files,
    )

    router = RoutedPurpleExecutor(registry=FailingRegistry(), default_executor="local")
    envelope = build_purple_request_text(
        "task for external executor",
        {
            "executor": "llm",
            "executor_endpoint": "http://executor.example:8080",
        },
    )
    event_queue = EventQueue()

    asyncio.run(router.execute(_context_with_text(envelope), event_queue))

    assert len(calls) == 1
    assert calls[0]["message"] == "task for external executor"
    assert calls[0]["file_payloads"] == []
    assert calls[0]["base_url"] == "http://executor.example:8080"
    assert calls[0]["context_id"] is None
