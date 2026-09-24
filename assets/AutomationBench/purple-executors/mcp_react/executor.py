# ruff: noqa: E402

"""AutomationBench purple executor: WM-guided tool-calling agent.

AutomationBench analogue of EnterpriseOps-Gym's ``mcp_react`` executor. Green sends
one task (trigger text + initial world state + per-task tool allow-list, never the
assertions); this executor binds upstream AutomationBench's own tools to a fresh
``WorldState``, runs the ReAct loop in the sibling ``wm_react.py`` -- which adds the
shared, pluggable ``ejepa_wm`` world model -- and returns the final world state plus
its trajectory for Green to score.

With ``WM_STRATEGY`` unset it is the plain no-WM baseline. Set the standard
``--wm-*`` flags for a world model:

* Enterprise-JEPA: ``--wm-strategy beam_plan --wm-ewm-jepa-checkpoint <dir>
  --wm-jepa-observation-backend canonical_event``
* LLM-WM (state output): ``--wm-strategy beam_plan --wm-llm-ewm-mode
  llm_canonical_trained --wm-ewm-llm-canonical-event-checkpoint <dir>``
* LLM-WM (tool output) / agent world model: ``--wm-llm-ewm-mode
  llm_tool_output_judge`` (``WM_QWEN_AGENTWORLD=1`` for the agent-world model)

Policy LLM configuration (OpenAI-compatible endpoint):
``AUTOMATIONBENCH_LLM_MODEL`` / ``_API_ENDPOINT`` / ``_API_KEY`` /
``_TEMPERATURE`` / ``_MAX_TOKENS``, falling back to ``LLM_MODEL`` /
``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``OPENAI_*``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TaskState, TextPart
from a2a.utils import new_agent_text_message

_EXECUTOR_DIR = Path(__file__).resolve().parent
SRC_DIR = _EXECUTOR_DIR.parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.logging_utils import configure_logging, get_logger

configure_logging()
LOGGER = get_logger(__name__)

INTERNAL_TRAJECTORY_ARTIFACT_NAME = "internal_trajectory"


def _load_sibling(module_name: str, filename: str) -> Any:
    """Load a sibling module under a unique name.

    Several benchmarks in this repo ship a ``wm_react.py`` / ``tools.py``; a bare
    import would collide in a shared process.
    """
    spec = importlib.util.spec_from_file_location(module_name, _EXECUTOR_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_tools_module = _load_sibling("automationbench_mcp_react_tools", "tools.py")
_wm_react_module = _load_sibling("automationbench_mcp_react_wm", "wm_react.py")


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


class PolicyLlmClient:
    """Minimal OpenAI-compatible chat client with tool calling.

    Upstream AutomationBench's own clients are ``verifiers``-coupled, so this
    executor talks to the endpoint directly -- the same choice the other
    ``mcp_react`` executors make when upstream has no reusable client.
    """

    def __init__(self) -> None:
        self.model = _env(
            "AUTOMATIONBENCH_LLM_MODEL", "LLM_MODEL", "OPENAI_MODEL_NAME", default=""
        )
        self.base_url = _env(
            "AUTOMATIONBENCH_LLM_API_ENDPOINT",
            "AUTOMATIONBENCH_LLM_BASE_URL",
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
            default="http://127.0.0.1:8000/v1",
        )
        self.api_key = _env(
            "AUTOMATIONBENCH_LLM_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", default="EMPTY"
        )
        self.temperature = float(_env("AUTOMATIONBENCH_LLM_TEMPERATURE", default="0") or 0)
        self.max_tokens = int(_env("AUTOMATIONBENCH_LLM_MAX_TOKENS", default="4096") or 4096)
        self.timeout = float(_env("AUTOMATIONBENCH_LLM_TIMEOUT", default="600") or 600)
        if not self.model:
            raise RuntimeError(
                "AutomationBench policy model not configured. Set "
                "AUTOMATIONBENCH_LLM_MODEL (or LLM_MODEL) and the matching endpoint."
            )
        from openai import AsyncOpenAI, OpenAI

        self._async = AsyncOpenAI(
            base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
        )
        self._sync = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    # -- policy interface used by the ReAct loop ---------------------------
    async def invoke_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        response = await self._async.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools or None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        message = response.choices[0].message
        return {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in (message.tool_calls or [])
            ],
        }

    # -- chat_fn interface used by ejepa_wm ----------------------------------
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else float(temperature),
            "max_tokens": self.max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = self._sync.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def complete_samples(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        num_samples: int = 1,
        response_format: dict[str, Any] | None = None,
    ) -> list[str]:
        """``n=k`` in one request; falls back to k requests if the server refuses."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": self.max_tokens,
            "n": max(1, int(num_samples)),
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            response = self._sync.chat.completions.create(**kwargs)
            return [choice.message.content or "" for choice in response.choices]
        except Exception:  # noqa: BLE001 - endpoint may not support n>1
            return [
                self.complete(messages, temperature=temperature, response_format=response_format)
                for _ in range(max(1, int(num_samples)))
            ]


def _parse_payload(request_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(request_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Request body must be the JSON payload produced by the AutomationBench Green agent"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Request payload must be a JSON object")
    if not payload.get("prompt"):
        raise ValueError("Request payload is missing 'prompt'")
    return payload


class AutomationBenchMcpReactExecutor(AgentExecutor):
    """A2A executor running one AutomationBench task per request."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.submit()
        await updater.start_work()

        request_text = context.get_user_input()
        try:
            payload = _parse_payload(request_text)
        except ValueError as exc:
            await updater.update_status(
                TaskState.failed, new_agent_text_message(f"Invalid request: {exc}")
            )
            return

        task_id = str(payload.get("task_id") or payload.get("task_name") or "unknown")
        toolset = str(payload.get("toolset") or "limited_zapier")
        LOGGER.info(
            "AutomationBench task %s (toolset=%s, max_turns=%s)",
            task_id,
            toolset,
            payload.get("max_turns"),
        )

        try:
            binding = _tools_module.build_binding(
                initial_state=dict(payload.get("initial_state") or {}),
                zapier_tools=list(payload.get("zapier_tools") or []),
                toolset=toolset,
                source=str(payload.get("source") or ""),
            )
            llm_client = PolicyLlmClient()
            result = await _wm_react_module.run_single_task_with_wm(
                payload=payload, llm_client=llm_client, binding=binding
            )
        except Exception as exc:  # noqa: BLE001 - reported to Green as a failed task
            LOGGER.error("AutomationBench task %s failed: %s", task_id, exc)
            result = {
                "final_state": {},
                "tool_calls": [],
                "num_tool_calls": 0,
                "num_model_calls": 0,
                "wm_steps": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

        await updater.add_artifact(
            parts=[
                Part(
                    root=TextPart(
                        text=json.dumps(
                            {
                                "source": "purple_executor",
                                "executor": "mcp_react",
                                "format": "automationbench_wm_react",
                                "task_id": task_id,
                                "wm_strategy": result.get("wm_strategy"),
                                "wm_backend": result.get("wm_backend"),
                                "tool_source": result.get("tool_source"),
                                "tool_calls": result.get("tool_calls"),
                                "wm_steps": result.get("wm_steps"),
                            },
                            ensure_ascii=False,
                            default=str,
                        )
                    )
                )
            ],
            name=INTERNAL_TRAJECTORY_ARTIFACT_NAME,
        )
        await updater.update_status(
            TaskState.completed,
            new_agent_text_message(json.dumps(result, ensure_ascii=False, default=str)),
            final=True,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("AutomationBench mcp_react executor does not support cancel")


def build_executor() -> AgentExecutor:
    return AutomationBenchMcpReactExecutor()
