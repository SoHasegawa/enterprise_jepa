from pathlib import Path
import sys

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Task,
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
    new_task,
)

EXECUTOR_DIR = Path(__file__).resolve().parent
EXECUTOR_DIR_STR = str(EXECUTOR_DIR)
if EXECUTOR_DIR_STR not in sys.path:
    sys.path.insert(0, EXECUTOR_DIR_STR)

from agent import Agent


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}


class Executor(AgentExecutor):
    def __init__(self):
        self.agents: dict[str, Agent] = {} # context_id to agent instance

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(message=f"Task {task.id} already processed (state: {task.status.state})"))

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self.agents.get(context_id)
        if not agent:
            agent = Agent()
            self.agents[context_id] = agent

        updater = TaskUpdater(event_queue, task.id, context_id)

        await updater.start_work()
        try:
            await agent.run(msg, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as e:
            print(f"Task failed with agent error: {e}")
            await updater.failed(new_agent_text_message(f"Agent error: {e}", context_id=context_id, task_id=task.id))
        finally:
            # Green starts a new A2A conversation per task (new context_id). Without
            # eviction, each Agent keeps an OpenAI/httpx client until the process hits
            # the OS file-descriptor limit (~1024), causing Errno 24 after ~1000 tasks.
            released = self.agents.pop(context_id, None)
            if released is not None:
                released.close()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())


def build_executor() -> AgentExecutor:
    return Executor()
