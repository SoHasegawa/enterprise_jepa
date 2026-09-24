"""Standalone MCP server exposing the ``mcp_react`` executor as an MCP tool.

A FastMCP server speaking streamable-HTTP at ``/mcp``. It wraps the **in-process ``mcp_react``
execution core** (``executor.py``), which delegates to the upstream
EnterpriseOps-Gym ``BenchmarkExecutor`` with the ``react`` / ``wm_react``
orchestrator. Those reach out to:

- the EnterpriseOps-Gym **tool MCP servers** (per the task's ``gym_servers_config``), and
- (for ``WM_STRATEGY=imagined``) the **EWM world-model server** (vLLM).

Those run as separate services already, so there is no backend subprocess to
start here.

Orchestrator/strategy selection is the same as the A2A executor — driven by the
environment (``executor.py`` reads it): ``ENTERPRISEOPS_ORCHESTRATOR`` (``react``
default | ``wm_react`` | ...), and for ``wm_react`` the ``WM_STRATEGY`` /
``WM_*`` knobs. LLM credentials come from ``ENTERPRISEOPS_LLM_*`` (see the
executor docstring).

Run:  ``MCP_PORT=12072 python server/mcp_react/mcp_server.py``  (see ``run_local.sh``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

# mcp_react/ holds executor.py + the orchestrator modules (wm_react.py, wm_ewm.py,
# k_controller.py). Put it on the path so `import executor` (and its sibling
# orchestrators) resolve to this package.
_PKG_DIR = Path(__file__).resolve().parents[2]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import executor  # noqa: E402

# `stateless_http` is a FastMCP() kwarg in 2.x but moved to http_app() in 3.x.
# Support both so the same server runs under the iems-pinned fastmcp==2.6.1 and
# newer fastmcp.
try:
    mcp = FastMCP(name="enterpriseops_mcp_react", stateless_http=True)
    _HTTP_APP_KWARGS: dict[str, Any] = {}
except TypeError:
    mcp = FastMCP(name="enterpriseops_mcp_react")
    _HTTP_APP_KWARGS = {"stateless_http": True}


@mcp.tool()
async def run_mcp_react_execution(task: dict[str, Any]) -> dict[str, Any]:
    """Run one EnterpriseOps-Gym task through the ``mcp_react`` executor and
    return the verifier outcome.

    The orchestrator is chosen by ``ENTERPRISEOPS_ORCHESTRATOR`` (``react`` by
    default, ``wm_react`` for the World-Model strategies); for ``wm_react`` the
    ``WM_STRATEGY`` / ``WM_*`` env knobs apply, identical to the A2A executor.

    Args:
        task: the task payload (same shape the EnterpriseOps-Gym Green agent
            sends). Requires ``system_prompt``, ``user_prompt`` and
            ``gym_servers_config``; optional ``verifiers``, ``task_id``,
            ``domain``, ``mode``, ``number_of_runs``, ``selected_tools``,
            ``restricted_tools``, ``reset_database_between_runs``,
            ``max_iterations``, ``orchestrator``.

    Returns:
        ``{"outcome": {...}, "internal_trajectory": {...}}`` where ``outcome``
        carries ``overall_success`` / ``verification_summary`` /
        ``verification_results`` / ``tools_used`` / ``final_response`` /
        ``statistics`` / ``runs`` / ``executor_runtime``.
    """
    response_text, internal_trajectory = await executor.run_mcp_react_task_with_trajectory(
        json.dumps(task)
    )
    return {
        "outcome": json.loads(response_text),
        "internal_trajectory": internal_trajectory,
    }


@mcp.tool()
def info() -> dict[str, Any]:
    """Report the resolved ``mcp_react`` configuration (orchestrator + LLM)."""
    orchestrator = os.getenv("ENTERPRISEOPS_ORCHESTRATOR") or "react"
    runtime = executor._build_executor_runtime_payload_from_env()
    payload: dict[str, Any] = {
        "executor": "mcp_react",
        "orchestrator": orchestrator,
        "gym_repo_path": os.getenv("ENTERPRISEOPS_GYM_REPO_PATH"),
        "llm_models": (runtime or {}).get("llm_models", []),
    }
    if orchestrator == "wm_react":
        strategy = (os.getenv("WM_STRATEGY") or "best_of_n").strip().lower()
        payload["wm_strategy"] = strategy
        if strategy == "imagined":
            payload["wm_state"] = os.getenv("WM_STATE") or "binary_error"
            payload["action_optimizer"] = os.getenv("ACTION_OPTIMIZER") or "none"
            payload["k_controller"] = os.getenv("K_CONTROLLER") or "static"
            payload["ewm_model"] = os.getenv("WM_EWM_MODEL") or "gymops_world_model"
            payload["wm_vllm_base_url"] = os.getenv("WM_VLLM_BASE_URL") or (
                f"http://127.0.0.1:{os.getenv('WM_VLLM_SERVER_PORT') or os.getenv('VLLM_SERVER_PORT') or '9000'}/v1"
            )
        elif strategy in {"best_of_n", "prompt_injection"}:
            payload["wm_scorer_url"] = os.getenv("WM_SCORER_URL")
    return payload


if __name__ == "__main__":
    import uvicorn

    app = mcp.http_app(path="/mcp", **_HTTP_APP_KWARGS)
    uvicorn.run(
        app,
        host=os.getenv("MCP_HOST", "0.0.0.0"),
        port=int(os.getenv("MCP_PORT", "12072")),
    )
