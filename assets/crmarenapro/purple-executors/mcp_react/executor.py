"""CRMArenaPro purple executor: WM-guided ReAct over native CRM-SQL tools.

This is the CRMArenaPro analogue of EnterpriseOps-Gym's ``mcp_react`` executor. The gym
variant runs a ReAct loop over **MCP tool servers** with an optional ``ejepa_wm`` world
model; CRMArenaPro has no MCP layer, so this executor reuses the ``baseline_crm_agent``
CRM-SQL tool loop and bolts the same shared :mod:`ejepa_wm` world-model hook onto it (see
``wm_react.WmReactAgent``).

With ``WM_STRATEGY`` unset / ``none`` it behaves exactly like ``baseline_crm_agent``.
Set ``WM_STRATEGY=selection WM_BACKEND=ewm_predict`` (+ ``WM_EWM_MCP_URL`` / ``WM_N``)
to have the EWM world model rank candidate actions, or
``WM_STRATEGY=prompt_injection WM_BACKEND=ewm_imagined`` to inject imagined-trajectory
guidance. See this directory's ``README.md`` for the full env-var list.
"""
import importlib.util
import sys
from pathlib import Path

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InvalidRequestError,
    TaskState,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

EXECUTOR_DIR = Path(__file__).resolve().parent
# The CRM tool layer (CRMDatabase, SYSTEM_PROMPT, provider clients, the ReAct action
# parser) is reused verbatim from the sibling baseline_crm_agent executor. Put its dir on
# sys.path *before* loading wm_react (which does ``from agent import ...``).
BASELINE_DIR = EXECUTOR_DIR.parent / "baseline_crm_agent"
for _d in (str(BASELINE_DIR), str(EXECUTOR_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# Load the sibling wm_react.py under a unique module name. Both this benchmark and the
# Terminal-Bench mcp_react ship a ``wm_react.py``; a bare ``import wm_react`` would collide
# in any shared process (e.g. the test suite). importlib with a unique name avoids that.
_spec = importlib.util.spec_from_file_location(
    "crmarenapro_mcp_react_wm", EXECUTOR_DIR / "wm_react.py"
)
wm_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = wm_module
_spec.loader.exec_module(wm_module)
WmReactAgent = wm_module.WmReactAgent


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class Executor(AgentExecutor):
    def __init__(self):
        self.agents: dict[str, WmReactAgent] = {}  # context_id -> agent instance

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(
                message=f"Task {task.id} already processed (state: {task.status.state})"))

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self.agents.get(context_id)
        if not agent:
            agent = WmReactAgent()
            self.agents[context_id] = agent

        updater = TaskUpdater(event_queue, task.id, context_id)
        await updater.start_work()
        try:
            await agent.run(msg, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as exc:  # noqa: BLE001 — mirror baseline: report, don't crash the server
            await updater.failed(
                new_agent_text_message(f"Agent error: {exc}", context_id=context_id, task_id=task.id))
        finally:
            # Green opens a new A2A context per task; evict to release the HTTP/LLM
            # client and avoid leaking file descriptors over long runs.
            released = self.agents.pop(context_id, None)
            if released is not None:
                released.close()
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())


def build_executor() -> AgentExecutor:
    return Executor()
