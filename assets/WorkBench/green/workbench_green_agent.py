# ruff: noqa: E402

"""Green Agent for WorkBench.

Upstream repository: https://github.com/olly-styles/WorkBench
Paper:               https://arxiv.org/abs/2405.00823

WorkBench evaluates a single agent against realistic workplace tasks over 26
read/write tools across calendar, email, analytics, project management, CRM,
and company-directory domains. The Green Agent forwards each task's text (and
declared domains) to the Purple Agent; the `mcp_react` executor runs
WorkBench's own ReAct/native tool-calling agent loop in-process and returns
the predicted action list. Green then scores the prediction against ground
truth using WorkBench's own `is_correct`/`has_side_effects` functions
(imported in-process from the resolved WorkBench repository) -- this
benchmark needs no LLM judge, so unlike some other wrapped benchmarks Green
owns scoring directly.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Part, TaskState, TextPart
from a2a.utils import new_agent_text_message

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.green_executor import BaseGreenAgent, BenchmarkGreenExecutor
from common.logging_utils import configure_logging, get_logger
from common.models import BenchmarkRunManifest, EvalRequest, EvalResult
from common.purple_client import PurpleClient
from common.result_store import build_execution_identity, ensure_result_dir, write_result_artifacts
from common.trajectory import (
    build_trajectory_capture_summary,
    capture_trajectory_enabled,
    trajectory_root_for_result,
    write_task_trajectory,
)
from common.uvicorn_utils import reserve_tcp_listener, run_uvicorn_with_socket, write_port_file
from common.versioning import load_component_version
from task_loader import TaskLoader, resolve_repo_path

configure_logging()
LOGGER = get_logger(__name__)

_BENCHMARK_NAME = "WorkBench"
_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
INTERNAL_TRAJECTORY_ARTIFACT_NAME = "internal_trajectory"

_evaluation_module: Any = None


def _load_evaluation_module() -> Any:
    """Import the upstream repo's own `src.evals.evaluation` in-process (scoring only).

    Safe against name collision with this repo's own `src/` (added above for
    `common.*`): this repo's `src/` directory has no module literally named
    `src` (it exposes `common`, `ejepa_cli`, etc. as siblings), so adding
    WorkBench's repo root to `sys.path` and importing `src.evals.evaluation`
    resolves unambiguously to WorkBench's own package.

    WorkBench's own `src/tools/state.py` loads its sandbox CSVs via paths
    relative to the process's current working directory (e.g.
    ``data/processed/calendar_events.csv``), matching how its own CLI is
    always invoked from the repo root. This process's cwd is the benchmarks
    repo, not WorkBench, so we `chdir` into the resolved repo once before the
    first scoring call (cheap, and the loaded pandas snapshot is cached
    process-wide afterward, so this only needs to happen once).
    """
    global _evaluation_module
    if _evaluation_module is None:
        repo_path = resolve_repo_path()
        repo_path_str = str(repo_path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)
        os.chdir(repo_path)
        import src.evals.evaluation as evaluation_module

        _evaluation_module = evaluation_module
    return _evaluation_module


def _resolve_green_agent_version() -> str:
    version = load_component_version(
        Path(__file__).resolve().parent,
        env_name="BENCHMARK_GREEN_VERSION",
    )
    if version is None:
        raise RuntimeError("WorkBench green agent version is not configured")
    return version


def _resolve_purple_agent_version() -> str | None:
    return load_component_version(
        _BENCHMARK_DIR / "purple",
        env_name="BENCHMARK_PURPLE_VERSION",
    )


def _resolve_executor_version(executor_name: str) -> str | None:
    return load_component_version(
        _BENCHMARK_DIR / "purple-executors" / executor_name,
        env_name="BENCHMARK_EXECUTOR_VERSION",
    )


def _build_purple_payload(task: dict[str, Any], request_config: dict[str, Any]) -> str:
    """Build the JSON envelope consumed by the Purple `mcp_react` executor.

    Ground truth (``task["outcome"]``) is intentionally never sent to Purple --
    Green scores the prediction itself once it comes back.
    """
    payload: dict[str, Any] = {
        "task": task["task"],
        "domains": task["domains"],
        "model_name": request_config["model_name"],
        "tool_selection": str(request_config.get("tool_selection") or "all"),
        "structured_outputs": bool(request_config.get("structured_outputs", False)),
        "act_without_confirmation": bool(request_config.get("act_without_confirmation", False)),
    }
    return json.dumps(payload, ensure_ascii=False)


def _parse_purple_outcome(predicted_text: str) -> dict[str, Any]:
    """Parse the JSON result returned by Purple into a dict."""
    if not predicted_text or not predicted_text.strip():
        return {"function_calls": [], "error": "empty response", "full_response": predicted_text}
    try:
        payload = json.loads(predicted_text)
    except json.JSONDecodeError:
        return {"function_calls": [], "error": "invalid JSON response", "full_response": predicted_text}
    if not isinstance(payload, dict):
        return {
            "function_calls": [],
            "error": "response was not a JSON object",
            "full_response": predicted_text,
        }
    function_calls = payload.get("function_calls")
    if not isinstance(function_calls, list):
        function_calls = []
    return {
        "function_calls": [str(a) for a in function_calls],
        "error": str(payload.get("error") or ""),
        "full_response": payload.get("full_response"),
    }


def _score_task(
    *,
    function_calls: list[str],
    ground_truth_outcome: list[str],
    error: str,
) -> tuple[float, bool, str]:
    """Score a prediction using WorkBench's own state-diff evaluator."""
    evaluation_module = _load_evaluation_module()
    correct = evaluation_module.is_correct(function_calls, ground_truth_outcome, error)
    side_effects = evaluation_module.has_side_effects(function_calls, correct)
    score = 1.0 if correct else 0.0
    reason = "correct" if correct else "incorrect"
    if side_effects:
        reason += "; unwanted side effects detected"
    if error:
        reason += f"; executor error: {error}"
    return score, side_effects, reason


def _detail_record(
    *,
    task: dict[str, Any],
    outcome: dict[str, Any],
    score: float,
    side_effects: bool,
    reason: str,
    error: str | None,
) -> dict[str, Any]:
    return {
        "task_id": task.get("id"),
        "target": task.get("target"),
        "task_text": task.get("task"),
        "domains": task.get("domains"),
        "ground_truth_outcome": task.get("outcome"),
        "predicted_function_calls": outcome.get("function_calls"),
        "predicted_full_response": outcome.get("full_response"),
        "executor_error": outcome.get("error") or None,
        "score": score,
        "unwanted_side_effects": side_effects,
        "reason": reason,
        "error": error,
    }


def _artifact_text(artifact: dict[str, Any]) -> str:
    parts = artifact.get("parts")
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "\n".join(chunks).strip()


def _internal_trajectory_events_from_artifacts(
    artifacts: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not artifacts:
        return []

    events: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.get("name") != INTERNAL_TRAJECTORY_ARTIFACT_NAME:
            continue
        text = _artifact_text(artifact)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"payload": {"text": text}}
        events.append(
            {
                "sequence": 0,
                "direction": "purple_internal",
                "event_type": "PurpleInternalTrajectoryMetadata",
                "source": {
                    "executor": payload.get("executor"),
                    "task_id": payload.get("task_id"),
                },
                "payload": payload.get("payload", payload),
            }
        )
    return events


class WorkBenchGreenAgent(BaseGreenAgent):
    """Green Agent for the WorkBench benchmark."""

    def __init__(self) -> None:
        self._required_roles = {"agent"}
        self._task_loader = TaskLoader()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing = self._required_roles - set(request.participants.keys())
        if missing:
            return False, f"Missing roles: {sorted(missing)}"
        if "target" not in request.config:
            return False, "config.target is required"

        model_name = request.config.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            return False, "config.model_name is required"

        tool_selection = request.config.get("tool_selection")
        if tool_selection is not None and tool_selection not in {"all", "domains"}:
            return False, "config.tool_selection must be 'all' or 'domains'"

        task_ids = request.config.get("task_ids")
        if task_ids is not None and not isinstance(task_ids, list):
            return False, "config.task_ids must be a list when provided"

        max_parallel = request.config.get("max_parallel")
        if max_parallel is not None:
            try:
                if int(max_parallel) < 1:
                    return False, "config.max_parallel must be >= 1"
            except (TypeError, ValueError):
                return False, "config.max_parallel must be an integer"

        return True, "ok"

    async def _orchestrate_with_trajectory(
        self,
        participants: dict[str, Any],
        goal: str,
        request_config: dict[str, Any] | None = None,
        capture_trajectory: bool = False,
    ) -> dict[str, Any]:
        client = PurpleClient()
        return await client.send_message_with_trajectory(
            goal,
            [],
            str(participants["agent"]),
            new_conversation=True,
            request_config=request_config,
            capture_trajectory=capture_trajectory,
        )

    async def _evaluate_task(
        self,
        *,
        task: dict[str, Any],
        participants: dict[str, Any],
        request_config: dict[str, Any],
        capture_trajectory: bool = False,
        trajectory_root: Path | None = None,
    ) -> dict[str, Any]:
        predicted: str | None = None
        outcome: dict[str, Any] = {}
        score = 0.0
        side_effects = False
        reason = "task not executed"
        error: str | None = None
        trajectory_file_path: Path | None = None
        trajectory_event_count = 0
        effective_capture_trajectory = capture_trajectory and trajectory_root is not None

        try:
            LOGGER.info("Processing task %s", task["id"])
            goal = _build_purple_payload(task, request_config)
            purple_result = await self._orchestrate_with_trajectory(
                participants,
                goal,
                request_config=request_config,
                capture_trajectory=effective_capture_trajectory,
            )
            predicted = str(purple_result.get("response", ""))
            trajectory_events = list(purple_result.get("trajectory", []))
            trajectory_events.extend(
                _internal_trajectory_events_from_artifacts(purple_result.get("artifacts", []))
            )
            trajectory_event_count = len(trajectory_events)
            if effective_capture_trajectory and trajectory_events:
                trajectory_file_path = write_task_trajectory(
                    trajectory_root=trajectory_root,
                    task_id=str(task.get("id") or "unknown"),
                    events=trajectory_events,
                )
            outcome = _parse_purple_outcome(predicted or "")
            score, side_effects, reason = _score_task(
                function_calls=outcome["function_calls"],
                ground_truth_outcome=task["outcome"],
                error=outcome["error"],
            )
            task_result = {
                "task_id": task["id"],
                "score": score,
                "unwanted_side_effects": side_effects,
                "reason": reason,
            }
        except Exception as exc:
            error = str(exc)
            LOGGER.error("Task %s failed: %s", task.get("id"), exc)
            task_result = {"task_id": task.get("id", "unknown"), "score": 0.0, "error": error}

        result = {
            "task_id": task.get("id", "unknown"),
            "predicted_text": predicted,
            "score_value": score,
            "side_effects_value": side_effects,
            "task_error": error,
            "task_result": task_result,
            "detail_record": _detail_record(
                task=task,
                outcome=outcome,
                score=score,
                side_effects=side_effects,
                reason=reason,
                error=error,
            ),
        }
        if trajectory_file_path is not None:
            result["detail_record"]["trajectory_file_path"] = str(trajectory_file_path)
            result["detail_record"]["trajectory_event_count"] = trajectory_event_count
        return result

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        benchmark_name = os.getenv("BENCHMARK_NAME", _BENCHMARK_NAME)
        executor_name = str(
            request.config.get("executor")
            or os.getenv("BENCHMARK_EXECUTOR")
            or "mcp_react"
        )

        started_at = datetime.datetime.now(datetime.UTC)
        target = str(request.config["target"])
        requested_ids: list[str] | None = request.config.get("task_ids") or None
        max_parallel = int(request.config.get("max_parallel", 1))
        request_config = dict(request.config)

        tasks = self._task_loader.load_tasks(target, requested_ids)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"=== Starting WorkBench evaluation: {len(tasks)} tasks "
                f"(target={target!r}, model={request_config.get('model_name')!r}, "
                f"max_parallel={max_parallel}) ==="
            ),
        )

        participants_str = {role: str(url) for role, url in request.participants.items()}
        result_paths = build_execution_identity(
            benchmark_name=benchmark_name,
            executor_name=executor_name,
            request_config=request_config,
            participants=participants_str,
            result_root=Path(os.getenv("BENCHMARK_RESULT_ROOT")) if os.getenv("BENCHMARK_RESULT_ROOT") else None,
            run_id=os.getenv("BENCHMARK_RUN_ID"),
            config_hash=os.getenv("BENCHMARK_CONFIG_HASH"),
            created_at_utc=started_at,
        )
        ensure_result_dir(result_paths)
        capture_trajectory = capture_trajectory_enabled(request_config)
        trajectory_root = (
            trajectory_root_for_result(result_paths.result_dir) if capture_trajectory else None
        )

        total_score = 0.0
        total_side_effects = 0
        task_results: list[dict[str, Any]] = []
        detail_records: list[dict[str, Any]] = []
        fatal_error: str | None = None
        status = "completed"
        futures: list[asyncio.Task[Any]] = []

        try:
            semaphore = asyncio.Semaphore(max_parallel)

            async def _bounded(idx: int, task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
                async with semaphore:
                    return idx, await self._evaluate_task(
                        task=task,
                        participants=request.participants,
                        request_config=request_config,
                        capture_trajectory=capture_trajectory,
                        trajectory_root=trajectory_root,
                    )

            futures = [asyncio.create_task(_bounded(i, t)) for i, t in enumerate(tasks)]
            ordered_results: list[dict[str, Any] | None] = [None] * len(tasks)
            ordered_details: list[dict[str, Any] | None] = [None] * len(tasks)

            for fut in asyncio.as_completed(futures):
                idx, outcome = await fut
                ordered_results[idx] = outcome["task_result"]
                ordered_details[idx] = outcome["detail_record"]
                total_score += float(outcome["score_value"])
                if outcome["side_effects_value"]:
                    total_side_effects += 1

                preview = (outcome["predicted_text"] or "")[:160]
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"=== Task {outcome['task_id']} | score={outcome['score_value']} | "
                        f"side_effects={outcome['side_effects_value']} | "
                        f"output_preview: {preview!r} ==="
                    ),
                )

            task_results = [r for r in ordered_results if r is not None]
            detail_records = [d for d in ordered_details if d is not None]
        except Exception as exc:
            fatal_error = str(exc)
            status = "failed"
            for f in futures:
                f.cancel()
            await asyncio.gather(*futures, return_exceptions=True)
            LOGGER.error("Benchmark run failed: %s", exc)
            raise
        finally:
            completed_at = datetime.datetime.now(datetime.UTC)
            score_rate = total_score / len(tasks) if tasks else 0.0
            side_effects_rate = total_side_effects / len(tasks) if tasks else 0.0
            summary = (
                f"Score: {total_score}, Score Rate: {score_rate:.2%}, "
                f"Unwanted Side Effects Rate: {side_effects_rate:.2%}"
            )

            eval_result = EvalResult(
                target=target,
                total_tasks=len(tasks),
                total_score=total_score,
                score_rate=score_rate,
                task_results=task_results,
            )

            detail_payload = {
                "schema_version": "1.0",
                "run_id": result_paths.run_id,
                "user_name": result_paths.user_name,
                "status": status,
                "benchmark_name": benchmark_name,
                "benchmark_version": os.getenv("BENCHMARK_VERSION") or None,
                "green_agent_version": _resolve_green_agent_version(),
                "purple_agent_version": _resolve_purple_agent_version(),
                "executor_version": _resolve_executor_version(executor_name),
                "executor_name": executor_name,
                "target": target,
                "task_ids": [t["id"] for t in tasks],
                "task_selection_label": result_paths.task_selection_label,
                "config_hash": result_paths.config_hash,
                "created_at_utc": started_at.isoformat(),
                "completed_at_utc": completed_at.isoformat(),
                "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
                "participants": participants_str,
                "request_config": request_config,
                "total_tasks": len(tasks),
                "total_score": total_score,
                "score_rate": score_rate,
                "score_rate_percent": f"{score_rate:.2%}",
                "total_unwanted_side_effects": total_side_effects,
                "avg_unwanted_side_effects_rate": side_effects_rate,
                "avg_unwanted_side_effects_rate_percent": f"{side_effects_rate:.2%}",
                "final_summary": summary,
                "fatal_error": fatal_error,
                "trajectory_capture": build_trajectory_capture_summary(
                    enabled=capture_trajectory,
                    trajectory_root=trajectory_root,
                ),
                "details": detail_records,
            }

            manifest = BenchmarkRunManifest(
                run_id=result_paths.run_id,
                user_name=result_paths.user_name,
                status=status,
                benchmark_name=benchmark_name,
                benchmark_version=os.getenv("BENCHMARK_VERSION") or None,
                green_agent_version=_resolve_green_agent_version(),
                purple_agent_version=_resolve_purple_agent_version(),
                executor_version=_resolve_executor_version(executor_name),
                executor_name=executor_name,
                target=target,
                task_ids=[t["id"] for t in tasks],
                task_selection_label=result_paths.task_selection_label,
                config_hash=result_paths.config_hash,
                created_at_utc=started_at,
                completed_at_utc=completed_at,
                duration_seconds=round((completed_at - started_at).total_seconds(), 3),
                result_dir=result_paths.result_dir,
                detail_file_path=result_paths.detail_path,
                participants=participants_str,
                request_config=request_config,
                score_summary=summary,
                eval_result=eval_result,
                fatal_error=fatal_error,
            )

            detail_path, manifest_path = write_result_artifacts(
                detail_payload=detail_payload,
                manifest=manifest,
                paths=result_paths,
            )

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=eval_result.model_dump_json()))],
                name="EvaluationResult",
            )
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=str(detail_path)))],
                name="EvaluationDetailFile",
            )
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=str(manifest_path)))],
                name="EvaluationManifestFile",
            )
            LOGGER.info("Detail: %s  Manifest: %s", detail_path, manifest_path)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"=== Evaluation complete. {summary} ==="),
            )


def build_agent_card(host: str, port: int, card_url: str | None) -> AgentCard:
    skill = AgentSkill(
        id="workbench_green",
        name="workbench_green",
        description="WorkBench benchmark orchestrator (workplace tool-use tasks)",
        tags=["workbench", "benchmark", "green", "tool-use"],
        examples=['{"target":"sample","model_name":"claude-sonnet-4.6"}'],
    )
    return AgentCard(
        name="workbench_green",
        description="Green agent for the WorkBench benchmark.",
        url=card_url or f"http://{host}:{port}/",
        version=_resolve_green_agent_version(),
        default_input_modes=["text", "text/plain", "application/json"],
        default_output_modes=["text", "text/plain", "application/json"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the WorkBench Green Agent.")
    parser.add_argument("--version", action="version", version=_resolve_green_agent_version())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file")
    parser.add_argument("--card-url")
    args = parser.parse_args()

    port_file = Path(args.port_file).expanduser().resolve() if args.port_file else None
    listener, actual_port = reserve_tcp_listener(args.host, args.port)
    write_port_file(port_file, actual_port)

    agent = WorkBenchGreenAgent()
    executor = BenchmarkGreenExecutor(agent)
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=build_agent_card(args.host, actual_port, args.card_url),
        http_handler=request_handler,
    )
    run_uvicorn_with_socket(
        server.build(),
        host=args.host,
        port=actual_port,
        listener=listener,
    )


if __name__ == "__main__":
    main()
