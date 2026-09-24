"""Purple executor for EnterpriseOps-Gym (MCP + ReAct + SQL verifiers).

Calls into the upstream `EnterpriseOps-Gym/benchmark/executor.BenchmarkExecutor` so that
LLM tool calling and SQL-verifier execution complete in a single request.

Expected upstream repository:
    https://github.com/ServiceNow/EnterpriseOps-Gym

The location of that upstream repository can be specified through one of:

- The ``ENTERPRISEOPS_GYM_REPO_PATH`` environment variable
- A symlink named ``EnterpriseOps-Gym`` placed directly under this benchmark directory
  (handy for local development)
- ``${BENCHMARK_HOME}/repos/EnterpriseOps-Gym``

LLM credentials are provided either via ``ENTERPRISEOPS_LLM_CONFIG_FILE`` (path to a JSON
file) or via the individual environment variables ``ENTERPRISEOPS_LLM_PROVIDER`` /
``ENTERPRISEOPS_LLM_MODEL`` / ``ENTERPRISEOPS_LLM_API_KEY`` (and optional siblings). When
using the ``planner_react`` or ``decomposing`` orchestrators, also set
``ENTERPRISEOPS_PLANNER_LLM_CONFIG_FILE``.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import re
import sys
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

SRC_DIR = Path(__file__).resolve().parents[4] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# This executor's own directory, so sibling modules (e.g. the ``wm_react``
# orchestrator) are importable by bare module name via importlib.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


_DEFAULT_REPO_LINK = (
    Path(__file__).resolve().parents[2] / "EnterpriseOps-Gym"
)  # assets/EnterpriseOps-Gym/EnterpriseOps-Gym (symlink)
INTERNAL_TRAJECTORY_ARTIFACT_NAME = "internal_trajectory"
_USER_STATIC_TOKEN_PATTERN = re.compile(
    r"\('(?P<user_id>[^']+)'\s*,\s*'[^']*'\s*,\s*'[^']*'\s*,\s*'[^']*'\s*,"
    r"\s*'[^']*'\s*,\s*'(?P<static_token>[^']+)'",
)
_VERIFIER_USER_ID_PATTERN = re.compile(r"\buser_id\s*=\s*'(?P<user_id>[^']+)'", re.IGNORECASE)


def _candidate_repo_paths() -> list[Path]:
    """Return the list of candidate paths where the upstream repository may live."""
    candidates: list[Path] = []
    explicit = os.getenv("ENTERPRISEOPS_GYM_REPO_PATH")
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())

    if _DEFAULT_REPO_LINK.exists():
        candidates.append(_DEFAULT_REPO_LINK.resolve())

    benchmark_home = os.getenv("BENCHMARK_HOME")
    if benchmark_home:
        candidates.append(
            (Path(benchmark_home) / "repos" / "EnterpriseOps-Gym").expanduser().resolve()
        )

    candidates.append(
        (Path(__file__).resolve().parents[5] / "EnterpriseOps-Gym").resolve()
    )
    return candidates


def _resolve_repo_path() -> Path:
    """Return the upstream repository root (first candidate that exists)."""
    for candidate in _candidate_repo_paths():
        if (candidate / "benchmark" / "executor.py").exists():
            return candidate
    raise RuntimeError(
        "EnterpriseOps-Gym repository not found. Set ENTERPRISEOPS_GYM_REPO_PATH, place a "
        "symlink at assets/EnterpriseOps-Gym/EnterpriseOps-Gym, or clone "
        "https://github.com/ServiceNow/EnterpriseOps-Gym into ${BENCHMARK_HOME}/repos/."
    )


def _ensure_upstream_on_path() -> Path:
    repo_path = _resolve_repo_path()
    repo_path_str = str(repo_path)
    if repo_path_str not in sys.path:
        sys.path.insert(0, repo_path_str)
    return repo_path


_ORCHESTRATOR_MODULES = {
    "react": ("orchestrators.react", "ReactOrchestrator"),
    "wm_react": ("wm_react", "WmReactOrchestrator"),
    "planner_react": ("orchestrators.planner_react", "PlannerReactOrchestrator"),
    "decomposing": ("orchestrators.decomposing_planner", "DecomposingPlannerOrchestrator"),
}


def _load_llm_config_dict() -> dict[str, Any]:
    """Build a dict for the upstream `LLMConfig` from a JSON file or environment variables."""
    config_file = os.getenv("ENTERPRISEOPS_LLM_CONFIG_FILE") or os.getenv(
        "ENTERPRISEOPS_GYM_LLM_CONFIG_FILE"
    )
    if config_file:
        path = Path(config_file).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"LLM config file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    provider = (
        os.getenv("ENTERPRISEOPS_LLM_PROVIDER")
        or os.getenv("LLM_PROVIDER")
        or ("anthropic" if os.getenv("ANTHROPIC_API_KEY") else None)
    )
    model = (
        os.getenv("ENTERPRISEOPS_LLM_MODEL")
        or os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL_NAME")
        or os.getenv("INFERENCE_DEFAULT_MODEL")
        or os.getenv("ANTHROPIC_MODEL")
    )
    api_key = (
        os.getenv("ENTERPRISEOPS_LLM_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
    )
    if provider == "vllm" and not api_key:
        api_key = "not-needed"
    if not provider or not model or not api_key:
        raise RuntimeError(
            "EnterpriseOps-Gym LLM config not provided. Set ENTERPRISEOPS_LLM_CONFIG_FILE or "
            "ENTERPRISEOPS_LLM_PROVIDER + ENTERPRISEOPS_LLM_MODEL + ENTERPRISEOPS_LLM_API_KEY."
        )

    payload: dict[str, Any] = {
        "llm_provider": provider,
        "llm_model": model,
        "llm_api_key": api_key,
    }
    optional_env_to_field = {
        "ENTERPRISEOPS_LLM_API_ENDPOINT": "llm_api_endpoint",
        "OPENAI_BASE_URL": "llm_api_endpoint",
        "INFERENCE_DEFAULT_BASE_URL": "llm_api_endpoint",
        "ANTHROPIC_BASE_URL": "llm_api_endpoint",
        "ENTERPRISEOPS_LLM_API_VERSION": "llm_api_version",
        "ENTERPRISEOPS_LLM_REGION": "llm_region",
        "ENTERPRISEOPS_LLM_TEMPERATURE": "temperature",
        "ENTERPRISEOPS_LLM_MAX_TOKENS": "max_tokens",
        "ENTERPRISEOPS_LLM_TOP_P": "top_p",
        "ENTERPRISEOPS_LLM_EFFORT": "effort",
        "ENTERPRISEOPS_LLM_EXTRA_BODY": "extra_body",
    }
    for env_name, field in optional_env_to_field.items():
        value = os.getenv(env_name)
        if value is None or value == "":
            continue
        if field in payload:
            continue
        if field in {"temperature", "top_p"}:
            payload[field] = float(value)
        elif field == "max_tokens":
            payload[field] = int(value)
        elif field == "extra_body":
            payload[field] = json.loads(value)
        else:
            payload[field] = value
    return payload


def _load_planner_llm_config_dict() -> dict[str, Any] | None:
    config_file = os.getenv("ENTERPRISEOPS_PLANNER_LLM_CONFIG_FILE")
    if not config_file:
        return None
    path = Path(config_file).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Planner LLM config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _static_tokens_by_user(seed_database_file: str) -> dict[str, str]:
    """Extract user static tokens from the SQL seed dump.

    The upstream MCP servers authenticate by matching ``x-access-token`` against
    ``users.static_token``. Some bundled/sample tasks carry placeholder tokens,
    so we repair those from the per-task seed SQL before requests reach MCP.
    """
    path = Path(seed_database_file)
    if not path.exists():
        return {}

    tokens: dict[str, str] = {}
    for match in _USER_STATIC_TOKEN_PATTERN.finditer(path.read_text(encoding="utf-8")):
        tokens[match.group("user_id")] = match.group("static_token")
    return tokens


def _infer_task_user_id(verifiers: list[dict[str, Any]]) -> str | None:
    for verifier in verifiers:
        if not isinstance(verifier, dict):
            continue
        validation_config = verifier.get("validation_config") or {}
        query = validation_config.get("query")
        if not isinstance(query, str):
            continue
        match = _VERIFIER_USER_ID_PATTERN.search(query)
        if match:
            return match.group("user_id")
    return None


def _normalize_context_access_token(
    entry: dict[str, Any],
    *,
    verifiers: list[dict[str, Any]],
) -> dict[str, Any]:
    context = entry.get("context")
    if not isinstance(context, dict):
        return entry

    seed_database_file = entry.get("seed_database_file")
    if not isinstance(seed_database_file, str) or not seed_database_file:
        return entry

    tokens = _static_tokens_by_user(seed_database_file)
    if not tokens:
        return entry

    current_token = context.get("x-access-token") or context.get("access-token")
    if current_token in tokens.values():
        return entry

    task_user_id = _infer_task_user_id(verifiers)
    replacement = tokens.get(task_user_id or "") or next(iter(tokens.values()))
    updated_entry = dict(entry)
    updated_context = dict(context)
    updated_context["x-access-token"] = replacement
    updated_entry["context"] = updated_context
    return updated_entry


def _parse_mcp_port_remap() -> dict[str, str]:
    """Optional ``old=new`` host-port remap for gym MCP server URLs.

    Read from ``ENTERPRISEOPS_MCP_PORT_REMAP`` (comma-separated, e.g. ``8008=8010``).
    The task's ``gym_servers_config`` carries fixed ``mcp_server_url`` ports from the
    dataset; when a required host port is already taken by an unrelated service and
    the matching domain server is reachable on a different port, this remaps the URL
    so the gym connects to the right place without editing the (often HF-sourced,
    non-local) task rows. Empty/unset → no remap.
    """
    raw = (os.getenv("ENTERPRISEOPS_MCP_PORT_REMAP") or "").strip()
    remap: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        old, new = (part.strip() for part in pair.split("=", 1))
        if old.isdigit() and new.isdigit():
            remap[old] = new
    return remap


def _remap_mcp_server_url(url: Any, remap: dict[str, str]) -> Any:
    """Rewrite the host port in an ``mcp_server_url`` per ``remap`` (leaves others as-is)."""
    if not remap or not isinstance(url, str):
        return url
    match = re.search(r":(\d+)(?=/|$)", url)
    if match and match.group(1) in remap:
        return url[: match.start(1)] + remap[match.group(1)] + url[match.end(1) :]
    return url


def _resolve_seed_database_paths(
    gym_servers_config: list[dict[str, Any]],
    *,
    repo_path: Path,
    verifiers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rewrite every `seed_database_file` to an absolute path under ``repo_path``.

    Upstream's ``create_database_from_file`` opens the seed SQL via ``os.path.exists`` /
    ``open`` against the current working directory, but our Purple agent runs from the
    benchmarks workdir rather than the upstream repo root. Without this rewrite,
    upstream silently fails to seed the per-run database and every MCP/verifier call
    goes out with ``x-database-id: None``.
    """
    port_remap = _parse_mcp_port_remap()
    resolved: list[dict[str, Any]] = []
    for entry in gym_servers_config:
        if not isinstance(entry, dict):
            resolved.append(entry)
            continue

        new_entry = dict(entry)
        raw = new_entry.get("seed_database_file")
        if isinstance(raw, str) and raw.strip():
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = repo_path / raw
            new_entry["seed_database_file"] = str(candidate)
        if port_remap and "mcp_server_url" in new_entry:
            new_entry["mcp_server_url"] = _remap_mcp_server_url(new_entry["mcp_server_url"], port_remap)
        new_entry = _normalize_context_access_token(new_entry, verifiers=verifiers)
        resolved.append(new_entry)
    return resolved


def _build_benchmark_config(payload: dict[str, Any], *, repo_path: Path) -> Any:
    """Construct an upstream `BenchmarkConfig` instance from the request payload."""
    from benchmark.models import BenchmarkConfig  # type: ignore[import-not-found]

    verifiers = list(payload.get("verifiers") or [])
    gym_servers_config = _resolve_seed_database_paths(
        list(payload.get("gym_servers_config") or []),
        repo_path=repo_path,
        verifiers=verifiers,
    )

    return BenchmarkConfig(
        system_prompt=str(payload.get("system_prompt", "")),
        user_prompt=str(payload.get("user_prompt", "")),
        verifiers=verifiers,
        number_of_runs=int(payload.get("number_of_runs", 1) or 1),
        gym_servers_config=gym_servers_config,
        mcp_endpoint=str(payload.get("mcp_endpoint") or "/mcp"),
        selected_tools=list(payload.get("selected_tools") or []) or None,
        restricted_tools=list(payload.get("restricted_tools") or []) or None,
        reset_database_between_runs=bool(payload.get("reset_database_between_runs", True)),
    )


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce the upstream `runs` array to the single outcome dict that Green expects."""
    if not runs:
        return {
            "overall_success": False,
            "verification_summary": {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0},
            "verification_results": {},
            "tools_used": [],
            "final_response": None,
            "error": "no runs were executed",
        }

    total_overall = sum(1 for r in runs if r.get("overall_success"))
    overall_success = total_overall == len(runs)
    summary_total = 0
    summary_passed = 0
    last_verification_results: dict[str, Any] = {}
    last_tools_used: list[str] = []
    last_final_response: Any = None
    error: str | None = None

    for run in runs:
        run_summary = run.get("verification_summary") or {}
        summary_total += int(run_summary.get("total", 0) or 0)
        summary_passed += int(run_summary.get("passed", 0) or 0)
        if run.get("verification_results"):
            last_verification_results = run["verification_results"]
        if run.get("tools_used"):
            last_tools_used = list(run["tools_used"])
        if "model_response" in run:
            last_final_response = run["model_response"]
        if run.get("error"):
            error = str(run["error"])

    failed = max(summary_total - summary_passed, 0)
    pass_rate = summary_passed / summary_total if summary_total else 0.0

    return {
        "overall_success": overall_success,
        "verification_summary": {
            "total": summary_total,
            "passed": summary_passed,
            "failed": failed,
            "pass_rate": pass_rate,
        },
        "verification_results": last_verification_results,
        "tools_used": last_tools_used,
        "final_response": last_final_response,
        "error": error,
    }


def _build_executor_runtime_payload(
    *,
    llm_config_dict: dict[str, Any],
    planner_config_dict: dict[str, Any] | None,
    config_source: str,
    planner_source: str | None,
) -> dict[str, Any]:
    """Build the `executor_runtime` payload Green will forward to the CLI."""
    models: list[dict[str, str]] = []
    notes: list[str] = []

    chat_model = (llm_config_dict.get("llm_model") or "").strip()
    if chat_model:
        models.append(
            {
                "role": "chat",
                "model_name": chat_model,
                "source": config_source,
                "status": "configured",
            }
        )

    if planner_config_dict is not None:
        planner_model = (planner_config_dict.get("llm_model") or "").strip()
        if planner_model:
            models.append(
                {
                    "role": "planner",
                    "model_name": planner_model,
                    "source": planner_source or "ENTERPRISEOPS_PLANNER_LLM_CONFIG_FILE",
                    "status": "configured",
                }
            )

    if not models:
        notes.append("Could not determine the LLM model name from the executor config.")

    return {
        "schema_version": "1.0",
        "llm_models": models,
        "notes": notes,
    }


def _build_executor_runtime_payload_from_env() -> dict[str, Any] | None:
    try:
        llm_config_dict = _load_llm_config_dict()
        planner_dict = _load_planner_llm_config_dict()
    except Exception:
        return None
    config_source = (
        f"environment:ENTERPRISEOPS_LLM_CONFIG_FILE={os.getenv('ENTERPRISEOPS_LLM_CONFIG_FILE')}"
        if os.getenv("ENTERPRISEOPS_LLM_CONFIG_FILE")
        else "environment:ENTERPRISEOPS/ANTHROPIC LLM variables"
    )
    planner_source = (
        f"environment:ENTERPRISEOPS_PLANNER_LLM_CONFIG_FILE="
        f"{os.getenv('ENTERPRISEOPS_PLANNER_LLM_CONFIG_FILE')}"
        if planner_dict is not None
        else None
    )
    return _build_executor_runtime_payload(
        llm_config_dict=llm_config_dict,
        planner_config_dict=planner_dict,
        config_source=config_source,
        planner_source=planner_source,
    )


async def _execute_task(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the upstream BenchmarkExecutor for a single task.

    Returns a tuple of ``(upstream_result, executor_runtime_payload)`` so the caller can
    forward the LLM model info to the Green agent without reaching back into the
    upstream `LLMConfig`.
    """
    repo_path = _ensure_upstream_on_path()

    executor_module = importlib.import_module("benchmark.executor")
    BenchmarkExecutor = getattr(executor_module, "BenchmarkExecutor")
    LLMConfig = getattr(importlib.import_module("benchmark.models"), "LLMConfig")

    # Orchestrator selection (first non-empty wins):
    #   1. ENTERPRISEOPS_ORCHESTRATOR env (explicit hard override, e.g. from wm_run.sh)
    #   2. wm_react when a World Model is requested (WM_STRATEGY set & != none; ejepa exports
    #      this from --wm-strategy). This intentionally beats payload.orchestrator so a WM
    #      engages even when the run config pins orchestrator=react (e.g. run.sh eval-opsgym-80).
    #   3. payload.orchestrator
    #   4. react (default)
    _wm_strategy = (os.getenv("WM_STRATEGY") or "").strip().lower()
    _wm_requested = bool(_wm_strategy) and _wm_strategy != "none"
    orchestrator_name = str(
        os.getenv("ENTERPRISEOPS_ORCHESTRATOR")
        or ("wm_react" if _wm_requested else "")
        or payload.get("orchestrator")
        or "react"
    )
    if orchestrator_name not in _ORCHESTRATOR_MODULES:
        raise ValueError(
            f"Unsupported orchestrator: {orchestrator_name!r}. "
            "Choose react, wm_react, planner_react, or decomposing."
        )
    module_name, class_name = _ORCHESTRATOR_MODULES[orchestrator_name]
    orchestrator_class = getattr(importlib.import_module(module_name), class_name)

    benchmark_config = _build_benchmark_config(payload, repo_path=repo_path)
    llm_config_dict = _load_llm_config_dict()
    config_source = (
        f"environment:ENTERPRISEOPS_LLM_CONFIG_FILE={os.getenv('ENTERPRISEOPS_LLM_CONFIG_FILE')}"
        if os.getenv("ENTERPRISEOPS_LLM_CONFIG_FILE")
        else "environment:ENTERPRISEOPS_LLM_PROVIDER/_MODEL/_API_KEY"
    )
    llm_config = LLMConfig(**llm_config_dict)

    orchestrator_kwargs: dict[str, Any] = {}
    planner_dict = _load_planner_llm_config_dict()
    planner_source: str | None = None
    if planner_dict is not None:
        planner_source = (
            f"environment:ENTERPRISEOPS_PLANNER_LLM_CONFIG_FILE="
            f"{os.getenv('ENTERPRISEOPS_PLANNER_LLM_CONFIG_FILE')}"
        )
        orchestrator_kwargs["planner_llm_config"] = LLMConfig(**planner_dict)

    max_iterations = payload.get("max_iterations")
    if max_iterations is not None:
        orchestrator_kwargs.setdefault("max_iterations", int(max_iterations))
    if payload.get("domain") and orchestrator_name == "wm_react":
        # Only WmReactOrchestrator accepts/pops a `domain` kwarg. The upstream
        # AgentOrchestrator.__init__ (react/planner_react/decomposing) has a fixed signature
        # with no **kwargs, so passing `domain` there is a hard TypeError.
        orchestrator_kwargs.setdefault("domain", payload.get("domain"))

    executor = BenchmarkExecutor(
        benchmark_config,
        llm_config=llm_config,
        orchestrator_class=orchestrator_class,
        orchestrator_kwargs=orchestrator_kwargs,
        config_path=str(repo_path / "config.json"),
    )
    result = await executor.execute_benchmark()
    runtime_payload = _build_executor_runtime_payload(
        llm_config_dict=llm_config_dict,
        planner_config_dict=planner_dict,
        config_source=config_source,
        planner_source=planner_source,
    )
    return result, runtime_payload


def _build_outcome_payload(
    payload: dict[str, Any],
    result: dict[str, Any],
    *,
    executor_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    statistics = result.get("statistics") or {}
    runs = list(result.get("runs") or [])
    summary = _summarize_runs(runs)

    outcome: dict[str, Any] = {
        "task_id": payload.get("task_id"),
        "domain": payload.get("domain"),
        "mode": payload.get("mode"),
        "overall_success": summary["overall_success"],
        "verification_summary": summary["verification_summary"],
        "verification_results": summary["verification_results"],
        "tools_used": summary["tools_used"],
        "final_response": summary["final_response"],
        "error": summary["error"],
        "statistics": statistics,
        "runs": runs,
    }
    if executor_runtime is not None:
        outcome["executor_runtime"] = executor_runtime
    return outcome


def _extract_request_text_from_parts(parts: list[Part]) -> str:
    chunks: list[str] = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
    request_text = "\n".join(chunks).strip()
    if not request_text:
        raise ValueError("No text part found in request message")
    return request_text


def _parse_payload(request_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(request_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Request body must be JSON produced by the EnterpriseOps-Gym Green agent"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Request payload must be a JSON object")
    if not payload.get("system_prompt") or not payload.get("user_prompt"):
        raise ValueError("Payload must include system_prompt and user_prompt")
    if not payload.get("gym_servers_config"):
        raise ValueError("Payload must include gym_servers_config")
    return payload


async def run_mcp_react_task(request_text: str) -> str:
    response_text, _internal_trajectory = await run_mcp_react_task_with_trajectory(request_text)
    return response_text


async def run_mcp_react_task_with_trajectory(request_text: str) -> tuple[str, dict[str, Any]]:
    payload = _parse_payload(request_text)
    try:
        result, executor_runtime = await _execute_task(payload)
        outcome = _build_outcome_payload(payload, result, executor_runtime=executor_runtime)
    except Exception as exc:
        executor_runtime = _build_executor_runtime_payload_from_env()
        outcome = {
            "task_id": payload.get("task_id"),
            "domain": payload.get("domain"),
            "mode": payload.get("mode"),
            "overall_success": False,
            "verification_summary": {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0},
            "verification_results": {},
            "tools_used": [],
            "final_response": None,
            "error": str(exc),
            "statistics": {},
            "runs": [],
        }
        if executor_runtime is not None:
            outcome["executor_runtime"] = executor_runtime
        result = {"runs": [], "statistics": {}, "error": str(exc)}

    response_text = json.dumps(outcome, ensure_ascii=False, default=str)
    internal_trajectory = {
        "source": "purple_executor",
        "executor": "mcp_react",
        "format": "enterpriseops_upstream_result",
        "task_id": payload.get("task_id"),
        "payload": {
            "info": {
                "domain": payload.get("domain"),
                "mode": payload.get("mode"),
                "orchestrator": payload.get("orchestrator") or "react",
                "selected_tools": payload.get("selected_tools") or [],
                "restricted_tools": payload.get("restricted_tools") or [],
                "overall_success": outcome.get("overall_success"),
                "verification_summary": outcome.get("verification_summary"),
            },
            "messages": [
                {"role": "system", "content": payload.get("system_prompt", "")},
                {"role": "user", "content": payload.get("user_prompt", "")},
                {"role": "assistant", "content": outcome.get("final_response")},
            ],
            "records": list(result.get("runs") or []),
            "response": outcome,
        },
    }
    return response_text, internal_trajectory


class McpReactEnterpriseOpsExecutor(AgentExecutor):
    """Purple executor that delegates to the upstream EnterpriseOps-Gym BenchmarkExecutor."""

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
            await updater.add_artifact(
                parts=[
                    Part(
                        root=TextPart(
                            text=json.dumps(internal_trajectory, ensure_ascii=False, default=str)
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
                    f"EnterpriseOps-Gym mcp_react execution failed: {exc}",
                    task.context_id,
                    task.id,
                ),
                final=True,
            )
            raise ServerError(error=InternalError(message=str(exc))) from exc

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())


def build_executor() -> AgentExecutor:
    return McpReactEnterpriseOpsExecutor()


__all__ = ["build_executor", "run_mcp_react_task"]
