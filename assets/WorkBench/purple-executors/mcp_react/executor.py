"""Purple executor for WorkBench (`mcp_react`).

WorkBench has no Docker, no MCP protocol, and no external service of any
kind: tasks run against 26 plain Python tool functions operating on in-memory
pandas DataFrames, isolated per-thread (`src/tools/state.py`). This executor
imports the upstream repo's own `src.evals.agent` / `src.evals.inference`
modules **in-process** and runs the real ReAct (or native tool-calling) agent
loop against the real tools -- genuinely real execution, just not over a
network protocol, since there is no MCP server here to be "genuine" about
(building one purely for the name would add a protocol boundary and new
state-isolation logic for zero functional benefit over WorkBench's own
already-correct in-process design).

When ``WM_STRATEGY`` is set, tasks are additionally routed through the
sibling ``wm_react.py``, which wraps WorkBench's own ReAct loop with the
shared, pluggable ``ejepa_wm`` World Model (including the JEPA-based Enterprise
World Model backend, matching how ``EnterpriseOps-Gym``'s and
``crmarenapro``'s ``mcp_react`` executors wire it in) -- see that module's
docstring for the supported strategies. With ``WM_STRATEGY`` unset it behaves
exactly as before (no WM).

Expected upstream repository:
    https://github.com/olly-styles/WorkBench

The location of that upstream repository can be specified through one of:

- The ``WORKBENCH_REPO_PATH`` environment variable
- A symlink named ``WorkBench`` placed directly under this benchmark
  directory (handy for local development)
- ``${BENCHMARK_HOME}/repos/WorkBench``

A credential matching your ``model_name`` (``OPENAI_API_KEY``/``ANTHROPIC_API_KEY``/
``GEMINI_API_KEY`` for that model's native provider, or ``OPENROUTER_API_KEY`` as the
fallback/catch-all) must already be exported in the environment this Purple process
runs in -- WorkBench's own `resolve_route()` reads them directly via `os.environ`.
This venv also needs the upstream repo's own runtime dependencies installed
(``uv sync --extra mcp_react``), since `src.evals.agent` is imported with this
interpreter.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

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

_EXECUTOR_DIR = Path(__file__).resolve().parent
_DEFAULT_REPO_LINK = _EXECUTOR_DIR.parents[1] / "WorkBench"  # assets/WorkBench/WorkBench (symlink)
INTERNAL_TRAJECTORY_ARTIFACT_NAME = "internal_trajectory"
_DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
_DEFAULT_VLLM_MODEL_NAME = "local-vllm"
_VLLM_BASE_URL_ENV = "WORKBENCH_VLLM_BASE_URL"
_VLLM_API_KEY_ENV = "WORKBENCH_VLLM_API_KEY"
_VLLM_MODEL_ENV = "WORKBENCH_VLLM_MODEL"
_VLLM_MODEL_NAME_ENV = "WORKBENCH_VLLM_MODEL_NAME"

_inference_module: Any = None
_agent_module: Any = None

# Load the sibling wm_react.py under a unique module name. Several benchmarks in this repo
# ship a `wm_react.py`; a bare `import wm_react` would collide in any shared process (e.g.
# the test suite). This module itself doesn't need the external WorkBench repo at import
# time (only `ejepa_wm`, which lives in this repo), so it's safe to load eagerly here.
_wm_react_spec = importlib.util.spec_from_file_location(
    "workbench_mcp_react_wm", _EXECUTOR_DIR / "wm_react.py"
)
assert _wm_react_spec is not None and _wm_react_spec.loader is not None
_wm_react_module = importlib.util.module_from_spec(_wm_react_spec)
sys.modules[_wm_react_spec.name] = _wm_react_module
_wm_react_spec.loader.exec_module(_wm_react_module)


def _candidate_repo_paths() -> list[Path]:
    """Return the list of candidate paths where the upstream repository may live."""
    candidates: list[Path] = []
    explicit = os.getenv("WORKBENCH_REPO_PATH")
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())

    if _DEFAULT_REPO_LINK.exists():
        candidates.append(_DEFAULT_REPO_LINK.resolve())

    benchmark_home = os.getenv("BENCHMARK_HOME")
    if benchmark_home:
        candidates.append((Path(benchmark_home) / "repos" / "WorkBench").expanduser().resolve())

    return candidates


def resolve_repo_path() -> Path:
    """Return the upstream repository root (first candidate that exists)."""
    for candidate in _candidate_repo_paths():
        if (candidate / "src" / "evals" / "agent.py").exists():
            return candidate
    raise RuntimeError(
        "WorkBench repository not found. Set WORKBENCH_REPO_PATH, place a symlink at "
        "assets/WorkBench/WorkBench, or clone https://github.com/olly-styles/WorkBench "
        "into ${BENCHMARK_HOME}/repos/."
    )


def _load_inference_module() -> Any:
    """Import the upstream repo's own `src.evals.inference` in-process.

    Safe against name collision with this repo's own `src/` (see
    ``workbench_green_agent.py``): the executor never imports `common.*`, so
    it never needs this repo's `src/` on its own path at all.

    WorkBench's own `src/tools/state.py` loads its sandbox CSVs via paths
    relative to the process's current working directory, matching how its own
    CLI is always invoked from the repo root. This process's cwd is the
    benchmarks repo, not WorkBench, so we `chdir` into the resolved repo once
    before the first task (the loaded pandas snapshot is cached process-wide
    afterward, so this only needs to happen once).
    """
    global _inference_module
    if _inference_module is None:
        repo_path = resolve_repo_path()
        repo_path_str = str(repo_path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)
        os.chdir(repo_path)
        import src.evals.inference as inference_module

        _inference_module = inference_module
    return _inference_module


def _load_agent_module() -> Any:
    """Import the upstream repo's own `src.evals.agent` in-process (WM wiring only).

    `_load_inference_module()` must run first (it resolves the repo path,
    puts it on `sys.path`, and `chdir`s into it); `run_mcp_react_task_with_trajectory`
    always calls it first for that reason.
    """
    global _agent_module
    if _agent_module is None:
        import src.evals.agent as agent_module

        _configure_vllm_agent_route(agent_module)
        _agent_module = agent_module
    return _agent_module


def _vllm_env_configured() -> bool:
    return any(
        os.getenv(name, "").strip()
        for name in (
            _VLLM_BASE_URL_ENV,
            _VLLM_API_KEY_ENV,
            _VLLM_MODEL_ENV,
            _VLLM_MODEL_NAME_ENV,
        )
    )


def _configure_vllm_agent_route(agent_module: Any) -> bool:
    """Register a local vLLM route in WorkBench's upstream model router.

    Upstream WorkBench only accepts hard-coded MODEL_REGISTRY keys and routes
    them to hard-coded provider base URLs. This opt-in patch adds a local
    OpenAI-compatible endpoint without editing the upstream checkout. We wrap
    resolve_route instead of relying only on a provider entry because upstream
    strips provider prefixes from direct-provider model IDs, while vLLM served
    model names may legitimately contain slashes.
    """
    if not _vllm_env_configured():
        return False

    base_url = (os.getenv(_VLLM_BASE_URL_ENV, "").strip() or _DEFAULT_VLLM_BASE_URL).rstrip("/")
    model_name = os.getenv(_VLLM_MODEL_NAME_ENV, "").strip() or _DEFAULT_VLLM_MODEL_NAME
    model_id = os.getenv(_VLLM_MODEL_ENV, "").strip() or model_name
    api_key = os.getenv(_VLLM_API_KEY_ENV, "").strip() or "EMPTY"

    model_aliases = [model_name]
    if model_id not in model_aliases:
        model_aliases.append(model_id)

    for alias in model_aliases:
        agent_module.MODEL_REGISTRY[alias] = agent_module.ModelConfig(model_id, True, "vllm")
    agent_module._PROVIDER_BASE_URLS["vllm"] = base_url
    agent_module._PROVIDER_API_KEY_ENV["vllm"] = _VLLM_API_KEY_ENV

    original_resolve_route = getattr(agent_module, "_mas_workbench_original_resolve_route", None)
    if original_resolve_route is None:
        original_resolve_route = agent_module.resolve_route
        agent_module._mas_workbench_original_resolve_route = original_resolve_route

    def resolve_route_with_vllm(requested_model_name: str) -> Any:
        config = agent_module.MODEL_REGISTRY.get(requested_model_name)
        if config is not None and config.provider == "vllm":
            return agent_module.Route(
                config.model_id,
                base_url,
                api_key,
                "vllm",
                config.supports_temperature,
            )
        return original_resolve_route(requested_model_name)

    agent_module.resolve_route = resolve_route_with_vllm
    return True


def _parse_payload(request_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(request_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Request body must be JSON produced by the WorkBench Green agent") from exc
    if not isinstance(payload, dict):
        raise ValueError("Request payload must be a JSON object")
    if not payload.get("task"):
        raise ValueError("Payload must include a non-empty 'task'")
    if not payload.get("model_name"):
        raise ValueError("Payload must include 'model_name'")
    return payload


def _build_datetime_prefix(inference_module: Any) -> str:
    """Reproduce WorkBench's own `generate_results()` datetime prefix verbatim.

    Ground truth is generated against this exact fixed "now"; if upstream
    ever changes this construction it must be mirrored here too.
    """
    now = inference_module.HARDCODED_CURRENT_TIME
    return (
        f"Today's date is {now.strftime('%A')}, {now.date()} "
        f"and the current time is {now.time()}. "
        "Remember the current date and time when completing tasks. "
        "Meetings must not start before 9am or end after 6pm."
    )


def _resolve_tools(inference_module: Any, payload: dict[str, Any]) -> list[Any]:
    tool_selection = str(payload.get("tool_selection") or "all")
    if tool_selection == "domains":
        domains = payload.get("domains") or []
        if not isinstance(domains, list):
            raise ValueError("'domains' must be a list when tool_selection='domains'")
        return inference_module.get_toolkits([str(d) for d in domains])
    return inference_module.get_toolkits(list(inference_module._TOOLKIT_MAP))


async def run_mcp_react_task(request_text: str) -> str:
    response_text, _internal_trajectory = await run_mcp_react_task_with_trajectory(request_text)
    return response_text


async def run_mcp_react_task_with_trajectory(
    request_text: str,
) -> tuple[str, dict[str, Any] | None]:
    """Run one task through WorkBench's own in-process agent loop.

    Returns ``(response_text, internal_trajectory)``; ``response_text`` is the
    JSON outcome envelope Green parses for scoring (``function_calls``,
    ``full_response``, ``error``), ``internal_trajectory`` is the full
    per-step LLM/tool trace Green may persist when trajectory capture is
    enabled. Ground truth is never seen by this executor -- Green scores the
    returned ``function_calls`` itself.
    """
    wm_steps: list[dict[str, Any]] = []
    try:
        payload = _parse_payload(request_text)
        inference_module = _load_inference_module()
        agent_module = _load_agent_module()
        tools = _resolve_tools(inference_module, payload)
        datetime_prefix = _build_datetime_prefix(inference_module)

        result = await asyncio.to_thread(
            _wm_react_module.run_single_task_with_wm,
            0,
            str(payload["task"]),
            str(payload["model_name"]),
            tools,
            datetime_prefix,
            bool(payload.get("act_without_confirmation", False)),
            bool(payload.get("structured_outputs", False)),
            inference_module,
            agent_module,
        )
        outcome: dict[str, Any] = {
            "function_calls": list(result.get("function_calls") or []),
            "full_response": result.get("full_response"),
            "error": result.get("error") or "",
        }
        trace = result.get("trace") or []
        wm_steps = list(result.get("wm_steps") or [])
    except Exception as exc:
        outcome = {"function_calls": [], "full_response": None, "error": str(exc)}
        trace = []

    internal_trajectory: dict[str, Any] | None = None
    if trace or wm_steps:
        internal_trajectory = {
            "source": "purple_executor",
            "executor": "mcp_react",
            "payload": {
                "steps": [
                    asdict(step) if hasattr(step, "__dataclass_fields__") else step
                    for step in trace
                ],
                "wm_steps": wm_steps,
            },
        }

    response_text = json.dumps(outcome, ensure_ascii=False, default=str)
    return response_text, internal_trajectory


def _extract_request_text_from_parts(parts: list[Part]) -> str:
    chunks: list[str] = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
    request_text = "\n".join(chunks).strip()
    if not request_text:
        raise ValueError("No text part found in request message")
    return request_text


class McpReactExecutor(AgentExecutor):
    """Purple executor that runs WorkBench's own agent loop in-process."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task:
            task = context.current_task
        elif context.message:
            task = new_task(context.message)
        else:
            raise ServerError(error=InvalidParamsError(message="No message provided"))

        if not context.message:
            raise ServerError(error=InvalidParamsError(message="No message provided"))

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        try:
            request_text = _extract_request_text_from_parts(context.message.parts)
            response_text, internal_trajectory = await run_mcp_react_task_with_trajectory(
                request_text
            )
            if internal_trajectory is not None:
                await updater.add_artifact(
                    parts=[
                        Part(
                            root=TextPart(
                                text=json.dumps(
                                    internal_trajectory, ensure_ascii=False, default=str
                                )
                            )
                        )
                    ],
                    name=INTERNAL_TRAJECTORY_ARTIFACT_NAME,
                )
            await updater.add_artifact(parts=[Part(root=TextPart(text=response_text))])
            await updater.complete()
        except Exception as exc:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    f"WorkBench mcp_react execution failed: {exc}",
                    task.context_id,
                    task.id,
                ),
                final=True,
            )
            raise ServerError(error=InternalError(message=str(exc))) from exc

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())


def build_executor() -> AgentExecutor:
    return McpReactExecutor()


__all__ = ["build_executor", "resolve_repo_path", "run_mcp_react_task"]
