import asyncio
from abc import ABC, abstractmethod

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
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
from pydantic import ValidationError

from common.logging_utils import get_logger
from common.models import EvalRequest, RuntimeFeedbackRequest, RuntimeFeedbackResponse

LOGGER = get_logger(__name__)


class BaseGreenAgent(ABC):
    """Abstract base class that benchmark-specific Green implementations follow."""

    @abstractmethod
    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        """Run an evaluation request to completion."""

    @abstractmethod
    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate an incoming evaluation request."""

    def validate_runtime_feedback_request(
        self,
        request: RuntimeFeedbackRequest,
    ) -> tuple[bool, str]:
        """Validate a runtime-feedback request; unsupported by default."""
        del request
        return False, "runtime feedback is not supported by this Green agent"

    async def run_runtime_feedback(
        self,
        request: RuntimeFeedbackRequest,
        updater: TaskUpdater,
    ) -> RuntimeFeedbackResponse:
        """Serve a runtime-feedback request; unsupported by default."""
        del request, updater
        await asyncio.sleep(0)
        raise NotImplementedError("runtime feedback is not supported by this Green agent")


class BenchmarkGreenExecutor(AgentExecutor):
    """Thin executor that connects a Green agent to an A2A server."""

    def __init__(self, green_agent: BaseGreenAgent) -> None:
        """Take and hold the evaluation logic itself."""
        self._agent = green_agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Dispatch an A2A request to Green by request type."""
        task_query = context.get_user_input()
        try:
            request = EvalRequest.model_validate_json(task_query)
            request_type = "eval"
            ok, message = self._agent.validate_request(request)
        except ValidationError:
            try:
                request = RuntimeFeedbackRequest.model_validate_json(task_query)
                request_type = "runtime_feedback"
                ok, message = self._agent.validate_runtime_feedback_request(request)
            except ValidationError as exc:
                raise ServerError(error=InvalidParamsError(message=exc.json())) from exc

        if not ok:
            raise ServerError(error=InvalidParamsError(message=message))

        if context.current_task:
            task = context.current_task
        elif context.message:
            task = new_task(context.message)
        else:
            raise ServerError(error=InvalidParamsError(message="No message provided"))

        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                (
                    f"Starting benchmark evaluation.\n{request.model_dump_json()}"
                    if request_type == "eval"
                    else f"Starting runtime feedback.\n{request.model_dump_json()}"
                ),
                context_id=task.context_id,
                task_id=task.id,
            ),
        )

        try:
            if request_type == "eval":
                await self._agent.run_eval(request, updater)
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=f"{task.id}: benchmark evaluation completed"))]
                )
            else:
                runtime_feedback = await self._agent.run_runtime_feedback(request, updater)
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=runtime_feedback.model_dump_json()))],
                    name="runtime_feedback_response",
                )
            await updater.complete()
        except Exception as exc:
            LOGGER.error("Green execution failed: %s", exc)
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Agent error: {exc}", task.context_id, task.id),
                final=True,
            )
            raise ServerError(error=InternalError(message=str(exc))) from exc

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        """Report explicitly that cancellation is unsupported."""
        raise ServerError(error=UnsupportedOperationError())
