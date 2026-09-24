from __future__ import annotations

import importlib.util
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    FilePart,
    FileWithBytes,
    InternalError,
    InvalidParamsError,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from common.client_utils import create_message_with_files, send_message_with_files
from common.logging_utils import get_logger
from common.purple_protocol import parse_purple_request_text

LOGGER = get_logger(__name__)


def _extract_text_and_files(parts: list[Part]) -> tuple[str, list[FileWithBytes]]:
    chunks: list[str] = []
    files: list[FileWithBytes] = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, FilePart):
            files.append(part.root.file)
    request_text = "\n".join(chunks).strip()
    if not request_text:
        raise ValueError("No text part found in request message")
    return request_text, files


class PurpleExecutorRegistry:
    """Lazy loader for benchmark-local Purple executors and runtimes."""

    def __init__(self, *, benchmark_dir: Path, module_prefix: str) -> None:
        self._benchmark_dir = benchmark_dir.resolve()
        self._module_prefix = module_prefix
        self._exit_stack = ExitStack()
        self._executors: dict[str, AgentExecutor] = {}
        self._runtime_started: set[str] = set()

    def executor_path(self, executor_name: str) -> Path:
        executor_path = self._benchmark_dir / "purple-executors" / executor_name / "executor.py"
        if not executor_path.exists():
            raise FileNotFoundError(f"Executor not found: {executor_path}")
        return executor_path

    def executor_dir(self, executor_name: str) -> Path:
        return self.executor_path(executor_name).parent

    def _load_module(self, *, module_name: str, path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create module spec: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _ensure_runtime(self, executor_name: str) -> None:
        if executor_name in self._runtime_started:
            return
        runtime_path = self.executor_dir(executor_name) / "runtime.py"
        if not runtime_path.exists():
            self._runtime_started.add(executor_name)
            return

        module = self._load_module(
            module_name=f"{self._module_prefix}_executor_runtime_{executor_name}",
            path=runtime_path,
        )
        runtime_factory = getattr(module, "maybe_manage_executor_runtime", None)
        if runtime_factory is None:
            context = nullcontext()
        elif callable(runtime_factory):
            context = runtime_factory()
        else:
            raise AttributeError(
                f"maybe_manage_executor_runtime() is not callable in {runtime_path}"
            )
        self._exit_stack.enter_context(context)
        self._runtime_started.add(executor_name)

    def get_executor(self, executor_name: str) -> AgentExecutor:
        if executor_name not in self._executors:
            self._ensure_runtime(executor_name)
            executor_path = self.executor_path(executor_name)
            module = self._load_module(
                module_name=f"{self._module_prefix}_executor_{executor_name}",
                path=executor_path,
            )
            builder = getattr(module, "build_executor", None)
            if not callable(builder):
                raise AttributeError(f"build_executor() not found in {executor_path}")
            executor = builder()
            if not isinstance(executor, AgentExecutor):
                LOGGER.debug(
                    "Executor %s is not an AgentExecutor instance: %r", executor_name, executor
                )
            self._executors[executor_name] = executor
        return self._executors[executor_name]


class _RequestContextWithTaskText:
    def __init__(
        self,
        original: RequestContext,
        *,
        task_text: str,
        file_payloads: list[FileWithBytes],
    ) -> None:
        self._original = original
        self.current_task = original.current_task
        replacement_message = create_message_with_files(
            text=task_text,
            file_payloads=file_payloads,
            context_id=original.message.context_id if original.message else None,
        )
        self.message = (
            original.message.model_copy(update={"parts": replacement_message.parts})
            if original.message
            else replacement_message
        )

    def get_user_input(self) -> str:
        return task_text if (task_text := self.message.parts[0].root.text) else ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class RoutedPurpleExecutor(AgentExecutor):
    """Purple executor that dispatches each request by envelope metadata."""

    def __init__(self, *, registry: PurpleExecutorRegistry, default_executor: str) -> None:
        self._registry = registry
        self._default_executor = default_executor

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.message:
            raise ServerError(error=InvalidParamsError(message="No message provided"))

        try:
            request_text, file_payloads = _extract_text_and_files(context.message.parts)
            envelope = parse_purple_request_text(request_text)
            if envelope is not None and envelope.get("executor_endpoint"):
                await self._forward_to_executor_endpoint(
                    context=context,
                    event_queue=event_queue,
                    task_text=str(envelope["task_context"]),
                    file_payloads=file_payloads,
                    executor_endpoint=str(envelope["executor_endpoint"]),
                )
                return

            executor_name = (
                str(envelope.get("executor"))
                if envelope is not None and envelope.get("executor")
                else self._default_executor
            )
            task_text = str(envelope["task_context"]) if envelope is not None else request_text
            executor = self._registry.get_executor(executor_name)
            routed_context = _RequestContextWithTaskText(
                context,
                task_text=task_text,
                file_payloads=file_payloads,
            )
            LOGGER.info("Routing Purple request to local executor=%s", executor_name)
            await executor.execute(routed_context, event_queue)
        except ServerError:
            raise
        except Exception as exc:
            raise ServerError(error=InternalError(message=str(exc))) from exc

    async def _forward_to_executor_endpoint(
        self,
        *,
        context: RequestContext,
        event_queue: EventQueue,
        task_text: str,
        file_payloads: list[FileWithBytes],
        executor_endpoint: str,
    ) -> None:
        if context.current_task:
            task = context.current_task
        elif context.message:
            task = new_task(context.message)
        else:
            raise ServerError(error=InvalidParamsError(message="No message provided"))

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        try:
            LOGGER.info("Forwarding Purple request to executor endpoint=%s", executor_endpoint)
            response = await send_message_with_files(
                message=task_text,
                file_payloads=file_payloads,
                base_url=executor_endpoint,
                context_id=None,
            )
            if response.get("status", "completed") != "completed":
                raise RuntimeError(f"{executor_endpoint} responded with: {response}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=str(response.get("response", ""))))]
            )
            await updater.complete()
        except Exception as exc:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    f"Executor endpoint forwarding failed: {exc}",
                    task.context_id,
                    task.id,
                ),
                final=True,
            )
            raise ServerError(error=InternalError(message=str(exc))) from exc

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
