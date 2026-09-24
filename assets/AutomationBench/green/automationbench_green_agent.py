# ruff: noqa: E402

"""Green Agent for AutomationBench.

Upstream repository: https://github.com/zapier/AutomationBench
Paper:               https://arxiv.org/abs/2604.18934

AutomationBench evaluates cross-application business workflow orchestration over
REST-style / Zapier-style tools against a simulated SaaS world: 600 public tasks
across Sales, Marketing, Operations, Support, Finance and HR, plus 200 ``simple``
foundational tasks. Every task ships a trigger message, an initial world state, a
per-task tool allow-list and a list of assertions over the *final* state.

Division of labour in this wrapper:

* Green loads tasks (bundled sample, or upstream's own ``get_combined_dataset``),
  forwards only the trigger text plus the initial state and tool allow-list, and
  **keeps the assertions** -- Purple never sees ground truth.
* Purple's ``mcp_react`` executor runs the tool-calling loop against its own
  ``WorldState`` instance and returns the final state plus its trajectory.
* Green scores that final state with upstream's own rubric functions
  (``partial_credit`` / ``task_completed_correctly``), falling back to a bundled
  evaluator for the offline sample. No LLM judge is involved.

The reported ``score`` is ``partial_credit`` (0..1, the upstream primary reward);
``task_completed_correctly`` is aggregated separately as the strict pass rate.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
import time
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

from scoring import PARTIAL_CREDIT_KEY, PASS_KEY, score_task
from task_loader import TaskLoader, ensure_upstream_importable

from common.green_executor import BaseGreenAgent, BenchmarkGreenExecutor
from common.logging_utils import configure_logging, get_logger
from common.models import BenchmarkRunManifest, EvalRequest, EvalResult
from common.purple_client import PurpleClient
from common.result_store import (
    build_execution_identity,
    ensure_result_dir,
    write_result_artifacts,
)
from common.trajectory import (
    build_trajectory_capture_summary,
    capture_trajectory_enabled,
    trajectory_root_for_result,
    write_task_trajectory,
)
from common.uvicorn_utils import reserve_tcp_listener, run_uvicorn_with_socket, write_port_file
from common.versioning import load_component_version

configure_logging()
LOGGER = get_logger(__name__)

_BENCHMARK_NAME = "AutomationBench"
_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
INTERNAL_TRAJECTORY_ARTIFACT_NAME = "internal_trajectory"
_TOOLSETS = ("api", "zapier", "limited_zapier")


def _resolve_green_agent_version() -> str:
    version = load_component_version(
        Path(__file__).resolve().parent, env_name="BENCHMARK_GREEN_VERSION"
    )
    if version is None:
        raise RuntimeError("AutomationBench green agent version is not configured")
    return version


def _resolve_purple_agent_version() -> str | None:
    return load_component_version(
        _BENCHMARK_DIR / "purple", env_name="BENCHMARK_PURPLE_VERSION"
    )


def _resolve_executor_version(executor_name: str) -> str | None:
    return load_component_version(
        _BENCHMARK_DIR / "purple-executors" / executor_name,
        env_name="BENCHMARK_EXECUTOR_VERSION",
    )


def _build_purple_payload(task: dict[str, Any], request_config: dict[str, Any]) -> str:
    """Everything Purple needs to run the task -- and nothing from the answer key."""
    payload = {
        "task_id": task["id"],
        "task_name": task["name"],
        "domain": task["domain"],
        "prompt": task["prompt"],
        "initial_state": task["initial_state"],
        "zapier_tools": task["zapier_tools"],
        # Purple binds upstream tools only for upstream tasks: the bundled `sample`
        # world is a hand-written fixture that upstream's WorldState schema rejects.
        "source": str(task.get("source") or "upstream_dataset"),
        "toolset": str(request_config.get("toolset") or "limited_zapier"),
        "max_turns": int(request_config.get("max_turns") or 50),
    }
    model_name = request_config.get("model_name")
    if isinstance(model_name, str) and model_name.strip():
        payload["model_name"] = model_name
    return json.dumps(payload, ensure_ascii=False)


def _parse_purple_outcome(text: str) -> dict[str, Any]:
    """Purple returns JSON with the final world state and its own telemetry."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"final_state": {}, "parse_error": str(exc), "raw": text[:2000]}
    if not isinstance(payload, dict):
        return {"final_state": {}, "parse_error": "purple payload is not an object"}
    return {
        "final_state": payload.get("final_state") or {},
        "tool_calls": payload.get("tool_calls") or [],
        "num_tool_calls": payload.get("num_tool_calls"),
        "num_model_calls": payload.get("num_model_calls"),
        "steps": payload.get("steps"),
        "final_response": payload.get("final_response") or "",
        "executor_error": payload.get("error"),
        "wm_steps": payload.get("wm_steps") or [],
        "parse_error": None,
    }


def _internal_trajectory_events_from_artifacts(artifacts: list[Any]) -> list[dict[str, Any]]:
    """Lift the executor's internal trajectory artifact into Green's JSONL events."""
    events: list[dict[str, Any]] = []
    for artifact in artifacts or []:
        name = getattr(artifact, "name", None) or (
            artifact.get("name") if isinstance(artifact, dict) else None
        )
        if name != INTERNAL_TRAJECTORY_ARTIFACT_NAME:
            continue
        parts = getattr(artifact, "parts", None) or (
            artifact.get("parts") if isinstance(artifact, dict) else []
        )
        for part in parts or []:
            root = getattr(part, "root", part)
            text = getattr(root, "text", None) or (
                root.get("text") if isinstance(root, dict) else None
            )
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            events.append(
                {
                    "direction": "purple_internal",
                    "event_type": "PurpleInternalTrajectory",
                    "payload": payload,
                }
            )
    return events


class AutomationBenchGreenAgent(BaseGreenAgent):
    """Green Agent for the AutomationBench benchmark."""

    def __init__(self) -> None:
        self._required_roles = {"agent"}
        self._task_loader = TaskLoader()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing = self._required_roles - set(request.participants.keys())
        if missing:
            return False, f"Missing roles: {sorted(missing)}"
        if "target" not in request.config:
            return False, "config.target is required"

        toolset = request.config.get("toolset")
        if toolset is not None and toolset not in _TOOLSETS:
            return False, f"config.toolset must be one of {list(_TOOLSETS)}"

        task_ids = request.config.get("task_ids")
        if task_ids is not None and not isinstance(task_ids, list):
            return False, "config.task_ids must be a list when provided"

        for key in ("max_turns", "max_parallel", "task_limit"):
            value = request.config.get(key)
            if value is None:
                continue
            try:
                if int(value) < 0:
                    return False, f"config.{key} must be >= 0"
            except (TypeError, ValueError):
                return False, f"config.{key} must be an integer"
        return True, "ok"

    async def _evaluate_task(
        self,
        *,
        task: dict[str, Any],
        participants: dict[str, Any],
        request_config: dict[str, Any],
        prefer_upstream_scorer: bool,
        capture_trajectory: bool = False,
        trajectory_root: Path | None = None,
    ) -> dict[str, Any]:
        outcome: dict[str, Any] = {}
        scores: dict[str, Any] = {}
        error: str | None = None
        trajectory_file_path: Path | None = None
        trajectory_event_count = 0
        effective_capture = capture_trajectory and trajectory_root is not None
        # Wall-clock per task. AutomationBench previously reported only a run-level
        # duration_seconds, which makes per-task latency analysis (and any comparison
        # restricted to a subset of tasks) impossible; every other benchmark in this
        # repo records execution_time_seconds per task.
        started_at = time.perf_counter()

        try:
            LOGGER.info("Processing task %s", task["id"])
            client = PurpleClient()
            purple_result = await client.send_message_with_trajectory(
                _build_purple_payload(task, request_config),
                [],
                str(participants["agent"]),
                new_conversation=True,
                request_config=request_config,
                capture_trajectory=effective_capture,
            )
            outcome = _parse_purple_outcome(str(purple_result.get("response", "")))
            events = list(purple_result.get("trajectory", []))
            events.extend(
                _internal_trajectory_events_from_artifacts(purple_result.get("artifacts", []))
            )
            trajectory_event_count = len(events)
            if effective_capture and events:
                trajectory_file_path = write_task_trajectory(
                    trajectory_root=trajectory_root,
                    task_id=str(task["id"]),
                    events=events,
                )
            scores = score_task(
                assertions=task["assertions"],
                initial_state=task["initial_state"],
                final_state=outcome.get("final_state") or {},
                prefer_upstream=prefer_upstream_scorer,
            )
            error = outcome.get("executor_error") or outcome.get("parse_error")
        except Exception as exc:  # noqa: BLE001 - one task must not abort the run
            error = str(exc)
            LOGGER.error("Task %s failed: %s", task.get("id"), exc)
            scores = {
                PARTIAL_CREDIT_KEY: 0.0,
                PASS_KEY: 0.0,
                "assertions_total": len(task["assertions"]),
            }

        score = float(scores.get(PARTIAL_CREDIT_KEY) or 0.0)
        passed = float(scores.get(PASS_KEY) or 0.0)
        detail_record = {
            "task_id": task["id"],
            "task_name": task["name"],
            "domain": task["domain"],
            "question": task["prompt"],
            "score": score,
            "partial_credit": score,
            "task_completed_correctly": passed,
            "assertions_total": scores.get("assertions_total"),
            "assertions_passed": scores.get("assertions_passed"),
            "assertions_scored": scores.get("assertions_scored"),
            "assertion_results": scores.get("assertion_results") or [],
            "scorer": scores.get("scorer"),
            "unsupported_assertions": scores.get("unsupported_assertions") or [],
            "tool_calls": outcome.get("num_tool_calls"),
            "model_calls": outcome.get("num_model_calls"),
            "steps": outcome.get("steps"),
            "failed_tool_calls": sum(
                1 for call in (outcome.get("tool_calls") or []) if call.get("error")
            ),
            "purple_tools_used": sorted(
                {
                    str(call.get("name"))
                    for call in (outcome.get("tool_calls") or [])
                    if call.get("name")
                }
            ),
            "purple_final_response": outcome.get("final_response"),
            "reason": _reason_text(scores, error),
            "error": error,
            "execution_time_seconds": round(time.perf_counter() - started_at, 3),
        }
        if trajectory_file_path is not None:
            detail_record["trajectory_file_path"] = str(trajectory_file_path)
            detail_record["trajectory_event_count"] = trajectory_event_count

        return {
            "task_id": task["id"],
            "score_value": score,
            "passed_value": passed,
            "task_result": {
                "task_id": task["id"],
                "score": score,
                "task_completed_correctly": passed,
                "reason": detail_record["reason"],
            },
            "detail_record": detail_record,
        }

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        benchmark_name = os.getenv("BENCHMARK_NAME", _BENCHMARK_NAME)
        executor_name = str(
            request.config.get("executor") or os.getenv("BENCHMARK_EXECUTOR") or "mcp_react"
        )
        started_at = datetime.datetime.now(datetime.UTC)
        target = str(request.config["target"])
        request_config = dict(request.config)
        requested_ids: list[str] | None = request_config.get("task_ids") or None
        max_parallel = max(1, int(request_config.get("max_parallel") or 1))

        tasks = self._task_loader.load_tasks(target, requested_ids)
        task_limit = int(request_config.get("task_limit") or 0)
        if task_limit > 0:
            tasks = tasks[:task_limit]

        # The bundled sample is scored offline; domain targets prefer upstream's rubric.
        prefer_upstream_scorer = target != "sample"
        if prefer_upstream_scorer:
            ensure_upstream_importable()

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"=== Starting AutomationBench evaluation: {len(tasks)} tasks "
                f"(target={target!r}, toolset={request_config.get('toolset', 'limited_zapier')!r}, "
                f"max_turns={request_config.get('max_turns', 50)}, max_parallel={max_parallel}) ==="
            ),
        )

        participants_str = {role: str(url) for role, url in request.participants.items()}
        result_paths = build_execution_identity(
            benchmark_name=benchmark_name,
            executor_name=executor_name,
            request_config=request_config,
            participants=participants_str,
            result_root=(
                Path(os.getenv("BENCHMARK_RESULT_ROOT"))
                if os.getenv("BENCHMARK_RESULT_ROOT")
                else None
            ),
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
        total_passed = 0
        task_results: list[dict[str, Any]] = []
        detail_records: list[dict[str, Any]] = []
        fatal_error: str | None = None
        status = "completed"
        futures: list[asyncio.Task[Any]] = []

        try:
            semaphore = asyncio.Semaphore(max_parallel)

            async def _bounded(index: int, task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
                async with semaphore:
                    return index, await self._evaluate_task(
                        task=task,
                        participants=request.participants,
                        request_config=request_config,
                        prefer_upstream_scorer=prefer_upstream_scorer,
                        capture_trajectory=capture_trajectory,
                        trajectory_root=trajectory_root,
                    )

            futures = [asyncio.create_task(_bounded(i, t)) for i, t in enumerate(tasks)]
            ordered_results: list[dict[str, Any] | None] = [None] * len(tasks)
            ordered_details: list[dict[str, Any] | None] = [None] * len(tasks)

            for future in asyncio.as_completed(futures):
                index, outcome = await future
                ordered_results[index] = outcome["task_result"]
                ordered_details[index] = outcome["detail_record"]
                total_score += float(outcome["score_value"])
                total_passed += int(outcome["passed_value"] == 1.0)
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"=== Task {outcome['task_id']} | "
                        f"partial_credit={outcome['score_value']:.2f} | "
                        f"passed={bool(outcome['passed_value'])} | "
                        f"assertions={outcome['detail_record'].get('assertions_passed')}/"
                        f"{outcome['detail_record'].get('assertions_scored')} ==="
                    ),
                )

            task_results = [item for item in ordered_results if item is not None]
            detail_records = [item for item in ordered_details if item is not None]
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            fatal_error = str(exc)
            status = "failed"
            for future in futures:
                future.cancel()
            await asyncio.gather(*futures, return_exceptions=True)
            LOGGER.error("Benchmark run failed: %s", exc)
            raise
        finally:
            completed_at = datetime.datetime.now(datetime.UTC)
            score_rate = total_score / len(tasks) if tasks else 0.0
            pass_rate = total_passed / len(tasks) if tasks else 0.0
            summary = (
                f"Avg Partial Credit: {score_rate:.2%}, Pass Rate: {pass_rate:.2%} "
                f"({total_passed}/{len(tasks)} tasks)"
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
                "task_ids": [task["id"] for task in tasks],
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
                "total_passed": total_passed,
                "pass_rate": pass_rate,
                "pass_rate_percent": f"{pass_rate:.2%}",
                "final_summary": summary,
                "fatal_error": fatal_error,
                "trajectory_capture": build_trajectory_capture_summary(
                    enabled=capture_trajectory, trajectory_root=trajectory_root
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
                task_ids=[task["id"] for task in tasks],
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
                detail_payload=detail_payload, manifest=manifest, paths=result_paths
            )
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=eval_result.model_dump_json()))],
                name="EvaluationResult",
            )
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=str(detail_path)))], name="EvaluationDetailFile"
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


def _reason_text(scores: dict[str, Any], error: str | None) -> str:
    if error:
        return f"error: {error}"
    passed = scores.get("assertions_passed")
    scored = scores.get("assertions_scored")
    if passed is None or scored is None:
        return f"partial_credit={float(scores.get(PARTIAL_CREDIT_KEY) or 0.0):.2f}"
    return f"{passed}/{scored} scored assertions satisfied"


def build_agent_card(host: str, port: int, card_url: str | None) -> AgentCard:
    skill = AgentSkill(
        id="automationbench_green",
        name="automationbench_green",
        description=(
            "AutomationBench orchestrator (cross-app business workflow automation)"
        ),
        tags=["automationbench", "benchmark", "green", "workflow", "tool-use"],
        examples=['{"target":"sample","toolset":"limited_zapier"}'],
    )
    return AgentCard(
        name="automationbench_green",
        description="Green agent for the AutomationBench benchmark.",
        url=card_url or f"http://{host}:{port}/",
        version=_resolve_green_agent_version(),
        default_input_modes=["text", "text/plain", "application/json"],
        default_output_modes=["text", "text/plain", "application/json"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AutomationBench Green Agent.")
    parser.add_argument("--version", action="version", version=_resolve_green_agent_version())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file")
    parser.add_argument("--card-url")
    args = parser.parse_args()

    port_file = Path(args.port_file).expanduser().resolve() if args.port_file else None
    listener, actual_port = reserve_tcp_listener(args.host, args.port)
    write_port_file(port_file, actual_port)

    executor = BenchmarkGreenExecutor(AutomationBenchGreenAgent())
    request_handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore()
    )
    server = A2AStarletteApplication(
        agent_card=build_agent_card(args.host, actual_port, args.card_url),
        http_handler=request_handler,
    )
    run_uvicorn_with_socket(
        server.build(), host=args.host, port=actual_port, listener=listener
    )


if __name__ == "__main__":
    main()
