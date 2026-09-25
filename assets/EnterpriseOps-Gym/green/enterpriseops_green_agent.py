# ruff: noqa: E402

"""Green Agent for EnterpriseOps-Gym.

Upstream repository: https://github.com/ServiceNow/EnterpriseOps-Gym
Official dataset:    https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym

Tasks require live MCP servers and verify the final environment state via SQL verifiers. The
Green Agent forwards the task's system_prompt, user_prompt, selected_tools, gym_servers_config,
and verifiers to the Purple Agent (the `mcp_react` executor). The `overall_success` and
`verifier_pass_rate` returned by Purple are then used as the scoring signal.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
from pathlib import Path
import sys
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
# Test-time self-evolution belongs to an agent scaffold that is not part of this
# repository. These no-op stands-in keep the green agent's
# code path intact and reject a run that explicitly asks for self-evolution.
_SELF_EVOLVE_AVAILABLE = False


class _NoSelfEvolveAdapter:
    supported_executor_names = ("mcp_react",)


def build_benchmark_self_evolution_detail(**_: Any) -> dict[str, Any]:
    return {}


def build_benchmark_self_evolve_session_from_adapter(*_: Any, **__: Any) -> Any:
    raise RuntimeError("self-evolve support is unavailable in this environment")


def build_inter_task_self_evolution_detail(**_: Any) -> dict[str, Any]:
    return {}


def build_score_summary(total_score: float, score_rate: float, *, baseline_score: float | None = None) -> str:
    del baseline_score
    return f"Score: {total_score}, Score Rate: {score_rate:.2%}"


def build_self_evolve_harness_from_adapter(*_: Any, **__: Any) -> Any:
    raise RuntimeError("self-evolve support is unavailable in this environment")


def resolve_max_benchmark_self_evolution_cycles(request_config: dict[str, Any]) -> int:
    return int(request_config.get("max_benchmark_self_evolution_cycles") or 0)


def resolve_max_self_evolutions(request_config: dict[str, Any]) -> int:
    return int(request_config.get("max_self_evolutions") or 0)

async def run_sequential_task_sequence(*_: Any, **__: Any) -> Any:
    raise RuntimeError("self-evolve support is unavailable in this environment")


def validate_self_evolve_config(
    *,
    request_config: dict[str, Any],
    executor_name: str,
    supported_executor_names: list[str] | tuple[str, ...],
) -> tuple[bool, str]:
    del executor_name, supported_executor_names
    if int(request_config.get("max_benchmark_self_evolution_cycles") or 0) > 0:
        return False, "self-evolve support is unavailable in this environment"
    if int(request_config.get("max_self_evolutions") or 0) > 0:
        return False, "self-evolve support is unavailable in this environment"
    return True, "ok"


def build_self_evolve_adapter() -> Any:
    return _NoSelfEvolveAdapter()
from task_loader import (
    TaskLoader,
    is_hf_corpus_target,
    is_supported_domain,
    is_supported_mode,
    resolve_hf_domains,
    resolve_hf_modes,
    resolve_hf_target_config,
)

configure_logging()
LOGGER = get_logger(__name__)

_BENCHMARK_NAME = "EnterpriseOps-Gym"
_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
INTERNAL_TRAJECTORY_ARTIFACT_NAME = "internal_trajectory"
SELF_EVOLVE_ADAPTER = build_self_evolve_adapter()


def _resolve_green_agent_version() -> str:
    version = load_component_version(
        Path(__file__).resolve().parent,
        env_name="BENCHMARK_GREEN_VERSION",
    )
    if version is None:
        raise RuntimeError("EnterpriseOps-Gym green agent version is not configured")
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
    """Build the JSON envelope consumed by the Purple `mcp_react` executor."""
    payload: dict[str, Any] = {
        "task_id": task.get("id"),
        "domain": task.get("domain"),
        "mode": task.get("mode"),
        "system_prompt": task.get("system_prompt", ""),
        "user_prompt": task.get("user_prompt", ""),
        "selected_tools": task.get("selected_tools") or [],
        "restricted_tools": task.get("restricted_tools") or [],
        "gym_servers_config": task.get("gym_servers_config") or [],
        "verifiers": task.get("verifiers") or [],
        "mcp_endpoint": task.get("mcp_endpoint") or "/mcp",
        "number_of_runs": int(task.get("number_of_runs") or 1),
        "reset_database_between_runs": bool(task.get("reset_database_between_runs", True)),
    }

    orchestrator = request_config.get("orchestrator")
    if isinstance(orchestrator, str) and orchestrator.strip():
        payload["orchestrator"] = orchestrator.strip()

    max_iterations = request_config.get("max_iterations")
    if max_iterations is not None:
        payload["max_iterations"] = int(max_iterations)

    return json.dumps(payload, ensure_ascii=False)


def _parse_purple_outcome(predicted_text: str) -> dict[str, Any]:
    """Parse the JSON result returned by Purple into a dict."""
    if not predicted_text or not predicted_text.strip():
        return {"overall_success": False, "raw_response": predicted_text}
    try:
        payload = json.loads(predicted_text)
    except json.JSONDecodeError:
        return {"overall_success": False, "raw_response": predicted_text}
    if not isinstance(payload, dict):
        return {"overall_success": False, "raw_response": predicted_text}
    return payload


def _score_from_outcome(outcome: dict[str, Any]) -> tuple[float, float, str]:
    """Derive (success_score, verifier_pass_rate, reason) from Purple's verdict payload."""
    overall_success = bool(outcome.get("overall_success", False))
    verification_summary = outcome.get("verification_summary") or {}
    total = int(verification_summary.get("total", 0) or 0)
    passed = int(verification_summary.get("passed", 0) or 0)
    pass_rate = float(verification_summary.get("pass_rate") or (passed / total if total else 0.0))

    score = 1.0 if overall_success else 0.0
    if total == 0:
        reason = "no verifiers reported"
    elif overall_success:
        reason = f"all {total} verifier(s) passed"
    else:
        failed = max(total - passed, 0)
        reason = f"{failed}/{total} verifier(s) failed (pass_rate={pass_rate:.2%})"
    error = outcome.get("error")
    if error:
        reason = f"{reason}; executor error: {error}"
    return score, pass_rate, reason


def _detail_record(
    *,
    task: dict[str, Any],
    predicted: str | None,
    outcome: dict[str, Any],
    score: float,
    pass_rate: float,
    reason: str,
    error: str | None,
) -> dict[str, Any]:
    return {
        "task_id": task.get("id"),
        "domain": task.get("domain"),
        "mode": task.get("mode"),
        "user_prompt": task.get("user_prompt"),
        "selected_tools": task.get("selected_tools"),
        "verifier_count": len(task.get("verifiers") or []),
        "purple_overall_success": bool(outcome.get("overall_success", False)),
        "purple_verification_summary": outcome.get("verification_summary"),
        "purple_verification_results": outcome.get("verification_results"),
        "purple_tools_used": outcome.get("tools_used"),
        "purple_final_response": outcome.get("final_response"),
        "purple_raw_payload": predicted,
        "score": score,
        "verifier_pass_rate": pass_rate,
        "reason": reason,
        "error": error,
    }


def _merge_executor_runtime(
    aggregate: dict[str, Any] | None,
    payload: Any,
) -> dict[str, Any] | None:
    """Merge an executor_runtime payload reported by Purple into the per-run aggregate."""
    if not isinstance(payload, dict):
        return aggregate

    new_models = payload.get("llm_models")
    new_notes = payload.get("notes")
    if aggregate is None:
        aggregate = {"schema_version": "1.0", "llm_models": [], "notes": []}

    if isinstance(new_models, list):
        seen = {(m.get("role"), m.get("model_name")) for m in aggregate["llm_models"]}
        for entry in new_models:
            if not isinstance(entry, dict):
                continue
            key = (entry.get("role"), entry.get("model_name"))
            if key in seen:
                continue
            seen.add(key)
            aggregate["llm_models"].append(entry)

    if isinstance(new_notes, list):
        for note in new_notes:
            if isinstance(note, str) and note and note not in aggregate["notes"]:
                aggregate["notes"].append(note)

    return aggregate


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
        data = part.get("data")
        if data is not None:
            chunks.append(json.dumps(data, ensure_ascii=False))
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

        source = {
            "executor": payload.get("executor"),
            "format": payload.get("format"),
            "task_id": payload.get("task_id"),
        }
        native_payload = payload.get("payload", payload)
        events.append(
            {
                "sequence": None,
                "direction": "purple_internal",
                "event_type": "PurpleInternalTrajectoryMetadata",
                "source": source,
                "payload": {
                    "info": native_payload.get("info")
                    if isinstance(native_payload, dict)
                    else None,
                },
            }
        )

        messages = (
            native_payload.get("messages") if isinstance(native_payload, dict) else None
        )
        if isinstance(messages, list):
            for idx, message in enumerate(messages):
                if not isinstance(message, dict):
                    continue
                events.append(
                    {
                        "sequence": idx,
                        "direction": "purple_internal",
                        "event_type": "PurpleInternalMessage",
                        "source": source,
                        "role": message.get("role"),
                        "payload": message,
                    }
                )

        records = (
            native_payload.get("records") if isinstance(native_payload, dict) else None
        )
        if isinstance(records, list):
            for idx, record in enumerate(records):
                events.append(
                    {
                        "sequence": idx,
                        "direction": "purple_internal",
                        "event_type": "PurpleInternalRecord",
                        "source": source,
                        "payload": record,
                    }
                )
    return events


class EnterpriseOpsGymGreenAgent(BaseGreenAgent):
    """Green Agent for the EnterpriseOps-Gym benchmark."""

    def __init__(self) -> None:
        self._required_roles = {"agent"}
        self._task_loader = TaskLoader()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing = self._required_roles - set(request.participants.keys())
        if missing:
            return False, f"Missing roles: {sorted(missing)}"
        if "target" not in request.config:
            return False, "config.target is required"

        target = str(request.config["target"])
        if is_hf_corpus_target(target):
            hf_cfg = resolve_hf_target_config(target, dict(request.config)) or {}
            domains = resolve_hf_domains(hf_cfg)
            if not domains:
                return (
                    False,
                    f"target {target!r} requires domains: set config.domains, "
                    "config.domain, or ENTERPRISEOPS_HF_DOMAIN (e.g. calendar)",
                )
            for domain in domains:
                if not is_supported_domain(domain):
                    return False, f"Unsupported HF domain: {domain!r}"
            for mode in resolve_hf_modes(hf_cfg):
                if not is_supported_mode(mode):
                    return False, f"Unsupported HF mode: {mode!r}"

        raw_modes = request.config.get("modes")
        if raw_modes is not None:
            if not isinstance(raw_modes, list):
                return False, "config.modes must be a list when provided"
            for item in raw_modes:
                if not isinstance(item, str) or not item.strip():
                    return False, "config.modes entries must be non-empty strings"

        raw_domains = request.config.get("domains")
        if raw_domains is not None:
            if not isinstance(raw_domains, list):
                return False, "config.domains must be a list when provided"
            for item in raw_domains:
                if not isinstance(item, str) or not item.strip():
                    return False, "config.domains entries must be non-empty strings"

        single_mode = request.config.get("mode")
        if single_mode is not None and (not isinstance(single_mode, str) or not single_mode.strip()):
            return False, "config.mode must be a non-empty string when provided"

        single_domain = request.config.get("domain")
        if single_domain is not None and (
            not isinstance(single_domain, str) or not single_domain.strip()
        ):
            return False, "config.domain must be a non-empty string when provided"

        if "max_tasks_per_domain" in request.config:
            value = request.config["max_tasks_per_domain"]
            if value is not None:
                try:
                    if int(value) < 1:
                        return False, "config.max_tasks_per_domain must be >= 1 when set"
                except (TypeError, ValueError):
                    return False, "config.max_tasks_per_domain must be an integer or null"

        orchestrator = request.config.get("orchestrator")
        if orchestrator is not None:
            if not isinstance(orchestrator, str) or orchestrator not in {
                "react",
                "planner_react",
                "decomposing",
            }:
                return (
                    False,
                    "config.orchestrator must be one of: react, planner_react, decomposing",
                )

        max_iterations = request.config.get("max_iterations")
        if max_iterations is not None:
            try:
                if int(max_iterations) < 1:
                    return False, "config.max_iterations must be >= 1"
            except (TypeError, ValueError):
                return False, "config.max_iterations must be an integer"

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

        is_valid_self_evolve, self_evolve_message = validate_self_evolve_config(
            request_config=dict(request.config),
            executor_name=str(
                request.config.get("executor")
                or os.getenv("BENCHMARK_EXECUTOR")
                or "mcp_react"
            ),
            supported_executor_names=SELF_EVOLVE_ADAPTER.supported_executor_names,
        )
        if not is_valid_self_evolve:
            return False, self_evolve_message

        return True, "ok"

    async def _orchestrate(
        self,
        participants: dict[str, Any],
        goal: str,
        request_config: dict[str, Any] | None = None,
    ) -> str:
        client = PurpleClient()
        return await client.send_message(
            goal,
            [],
            str(participants["agent"]),
            new_conversation=True,
            request_config=request_config,
        )

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
        request_text: str | None = None,
        request_config: dict[str, Any],
        capture_trajectory: bool = False,
        trajectory_root: Path | None = None,
        trajectory_label: str | None = None,
    ) -> dict[str, Any]:
        predicted: str | None = None
        outcome: dict[str, Any] = {}
        score = 0.0
        pass_rate = 0.0
        reason = "task not executed"
        error: str | None = None
        trajectory_file_path: Path | None = None
        trajectory_event_count = 0
        effective_capture_trajectory = capture_trajectory and trajectory_root is not None

        try:
            LOGGER.info("Processing task %s", task["id"])
            goal = request_text or _build_purple_payload(task, request_config)
            purple_result = await self._orchestrate_with_trajectory(
                participants,
                goal,
                request_config=request_config,
                capture_trajectory=effective_capture_trajectory,
            )
            predicted = str(purple_result.get("response", ""))
            trajectory_events = list(purple_result.get("trajectory", []))
            trajectory_events.extend(
                _internal_trajectory_events_from_artifacts(
                    purple_result.get("artifacts", [])
                )
            )
            trajectory_event_count = len(trajectory_events)
            if effective_capture_trajectory and trajectory_events:
                trajectory_file_path = write_task_trajectory(
                    trajectory_root=trajectory_root,
                    task_id=str(task.get("id") or "unknown"),
                    events=trajectory_events,
                    label=trajectory_label,
                )
            outcome = _parse_purple_outcome(predicted or "")
            score, pass_rate, reason = _score_from_outcome(outcome)
            task_result = {
                "task_id": task["id"],
                "score": score,
                "verifier_pass_rate": pass_rate,
                "reason": reason,
            }
        except Exception as exc:
            error = str(exc)
            LOGGER.error("Task %s failed: %s", task.get("id"), exc)
            exc_outputs = getattr(exc, "outputs", {})
            trajectory_events = list(exc_outputs.get("events", []))
            trajectory_events.extend(
                _internal_trajectory_events_from_artifacts(
                    exc_outputs.get("artifacts", [])
                )
            )
            trajectory_event_count = len(trajectory_events)
            if effective_capture_trajectory and trajectory_events:
                trajectory_file_path = write_task_trajectory(
                    trajectory_root=trajectory_root,
                    task_id=str(task.get("id") or "unknown"),
                    events=trajectory_events,
                    label=trajectory_label,
                )
            task_result = {
                "task_id": task.get("id", "unknown"),
                "score": 0.0,
                "error": error,
            }

        result = {
            "task_id": task.get("id", "unknown"),
            "predicted_text": predicted,
            "score_value": score,
            "verifier_pass_rate": pass_rate,
            "task_error": error,
            "task_result": task_result,
            "purple_outcome": outcome,
            "detail_record": _detail_record(
                task=task,
                predicted=predicted,
                outcome=outcome,
                score=score,
                pass_rate=pass_rate,
                reason=reason,
                error=error,
            ),
        }
        if trajectory_file_path is not None:
            result["detail_record"]["trajectory_file_path"] = str(trajectory_file_path)
            result["detail_record"]["trajectory_event_count"] = trajectory_event_count
        return result

    @staticmethod
    def _build_self_evolve_request_text(*, task: dict[str, Any], session, request_config: dict[str, Any]) -> str:
        return session.build_request_text(
            goal_text=_build_purple_payload(task, request_config),
            task_id=str(task.get("id") or ""),
            task=task,
        )

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

        tasks = self._task_loader.load_tasks(
            target,
            requested_ids,
            hf_config=dict(request.config),
        )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"=== Starting EnterpriseOps-Gym evaluation: {len(tasks)} tasks "
                f"(target={target!r}, max_parallel={max_parallel}) ==="
            ),
        )

        participants_str = {role: str(url) for role, url in request.participants.items()}
        result_paths = build_execution_identity(
            benchmark_name=benchmark_name,
            executor_name=executor_name,
            request_config=dict(request.config),
            participants=participants_str,
            result_root=Path(os.getenv("BENCHMARK_RESULT_ROOT")) if os.getenv("BENCHMARK_RESULT_ROOT") else None,
            run_id=os.getenv("BENCHMARK_RUN_ID"),
            config_hash=os.getenv("BENCHMARK_CONFIG_HASH"),
            created_at_utc=started_at,
        )
        ensure_result_dir(result_paths)
        capture_trajectory = capture_trajectory_enabled(dict(request.config))
        trajectory_root = (
            trajectory_root_for_result(result_paths.result_dir)
            if capture_trajectory
            else None
        )

        total_score = 0.0
        total_pass_rate = 0.0
        task_results: list[dict[str, Any]] = []
        detail_records: list[dict[str, Any]] = []
        executor_runtime: dict[str, Any] | None = None
        baseline_total_score: float | None = None
        self_evolution_detail: dict[str, Any] | None = None
        benchmark_self_evolution_detail: dict[str, Any] | None = None
        fatal_error: str | None = None
        status = "completed"
        futures: list[asyncio.Task[Any]] = []

        try:
            request_config = dict(request.config)
            max_benchmark_cycles = resolve_max_benchmark_self_evolution_cycles(request_config)
            max_self_evolutions = resolve_max_self_evolutions(request_config)

            async def _evaluate_with_request_text(
                task: dict[str, Any],
                goal_text: str,
                *,
                phase_label: str,
            ) -> dict[str, Any]:
                return await self._evaluate_task(
                    task=task,
                    participants=request.participants,
                    request_text=goal_text,
                    request_config=request_config,
                    capture_trajectory=capture_trajectory,
                    trajectory_root=trajectory_root,
                    trajectory_label=phase_label,
                )

            if max_benchmark_cycles > 0:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        "=== Starting benchmark-cycle self-evolve evaluation of "
                        f"{len(tasks)} tasks (max_cycles={max_benchmark_cycles}). ==="
                    ),
                )
                session = build_benchmark_self_evolve_session_from_adapter(
                    SELF_EVOLVE_ADAPTER,
                    request_config=request_config,
                    result_dir=result_paths.result_dir,
                )
                cycle_history: list[dict[str, Any]] = []
                stop_reason = "max_cycles_reached"

                async def _record_cycle_task(
                    task: dict[str, Any],
                    task_outcome: dict[str, Any],
                    goal_text: str,
                ) -> None:
                    session.record_task(
                        task=task,
                        goal_text=goal_text,
                        predicted=task_outcome.get("predicted_text"),
                        score=float(task_outcome.get("score_value") or 0.0),
                        reason=task_outcome.get("detail_record", {}).get("reason"),
                        error=task_outcome.get("task_error"),
                    )

                for pass_index in range(max_benchmark_cycles + 1):
                    session.reset_records()
                    generation_before_run = session.evolutions_applied
                    phase_label = f"pass {pass_index + 1}"
                    current_run = await run_sequential_task_sequence(
                        tasks=tasks,
                        updater=updater,
                        evaluate_task=lambda task, goal_text, phase_label=phase_label: (
                            _evaluate_with_request_text(task, goal_text, phase_label=phase_label)
                        ),
                        request_text_builder=lambda task: self._build_self_evolve_request_text(
                            task=task,
                            session=session,
                            request_config=request_config,
                        ),
                        phase_label=phase_label,
                        task_observer=_record_cycle_task,
                    )
                    current_total_score = float(current_run["total_score"])
                    total_score = current_total_score
                    task_results = list(current_run["task_results"])
                    detail_records = list(current_run["detail_records"])
                    total_pass_rate = sum(
                        float(record.get("verifier_pass_rate") or 0.0)
                        for record in detail_records
                    )
                    if baseline_total_score is None:
                        baseline_total_score = current_total_score

                    current_score_rate = float(current_run["score_rate"])
                    cycle_entry: dict[str, Any] = {
                        "pass_index": pass_index,
                        "phase_label": phase_label,
                        "pack_generation_before_run": generation_before_run,
                        "pack_generation_after_run": session.evolutions_applied,
                        "total_score": current_total_score,
                        "score_rate": current_score_rate,
                        "score_rate_percent": f"{current_score_rate:.2%}",
                        "task_results": task_results,
                        "details": detail_records,
                    }
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            f"=== [{phase_label}] completed. Score: {current_total_score}, "
                            f"Score Rate: {current_score_rate:.2%} ==="
                        ),
                    )

                    if current_score_rate >= 1.0:
                        stop_reason = "score_100_reached"
                        cycle_entry["stop_reason"] = stop_reason
                        cycle_history.append(cycle_entry)
                        break

                    if pass_index >= max_benchmark_cycles:
                        stop_reason = "max_cycles_reached"
                        cycle_entry["stop_reason"] = stop_reason
                        cycle_history.append(cycle_entry)
                        break

                    evolution_event = session.maybe_evolve(
                        after_task_id=str(tasks[-1].get("id") or f"pass_{pass_index + 1}"),
                    )
                    cycle_entry["evolution_event"] = evolution_event
                    cycle_entry["pack_generation_after_evolution"] = session.evolutions_applied
                    cycle_history.append(cycle_entry)

                    if not evolution_event:
                        stop_reason = "evolution_skipped"
                        cycle_entry["stop_reason"] = stop_reason
                        break

                    if evolution_event.get("status") == "completed":
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(
                                "=== "
                                f"[{phase_label}] benchmark self-evolve "
                                f"{evolution_event['generation']} completed. "
                                f"Strategies: {', '.join(evolution_event.get('strategies', [])) or 'none'} ==="
                            ),
                        )
                        continue

                    stop_reason = "evolution_failed"
                    cycle_entry["stop_reason"] = stop_reason
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            "=== "
                            f"[{phase_label}] benchmark self-evolve "
                            f"{evolution_event['generation']} failed: "
                            f"{evolution_event.get('error', 'unknown error')} ==="
                        ),
                    )
                    break

                benchmark_self_evolution_detail = build_benchmark_self_evolution_detail(
                    session=session,
                    cycle_history=cycle_history,
                    baseline_total_score=baseline_total_score,
                    final_total_score=total_score,
                    task_count=len(tasks),
                    final_details=detail_records,
                    stop_reason=stop_reason,
                )
            elif max_self_evolutions > 0:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        "=== Starting baseline + self-evolve evaluation of "
                        f"{len(tasks)} tasks (max_self_evolutions={max_self_evolutions}). ==="
                    ),
                )
                harness = build_self_evolve_harness_from_adapter(
                    SELF_EVOLVE_ADAPTER,
                    request_config=request_config,
                    result_dir=result_paths.result_dir,
                )
                baseline_run = await run_sequential_task_sequence(
                    tasks=tasks,
                    updater=updater,
                    evaluate_task=lambda task, goal_text: _evaluate_with_request_text(
                        task,
                        goal_text,
                        phase_label="baseline",
                    ),
                    request_text_builder=lambda task: self._build_self_evolve_request_text(
                        task=task,
                        session=harness.baseline_session,
                        request_config=request_config,
                    ),
                    phase_label="baseline",
                )
                baseline_total_score = float(baseline_run["total_score"])
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        "=== [baseline] completed. "
                        f"Score: {baseline_total_score}, "
                        f"Score Rate: {baseline_run['score_rate']:.2%} ==="
                    ),
                )

                async def _after_evolved_task(
                    task: dict[str, Any],
                    task_outcome: dict[str, Any],
                    goal_text: str,
                ) -> dict[str, Any] | None:
                    harness.evolved_session.record_task(
                        task=task,
                        goal_text=goal_text,
                        predicted=task_outcome.get("predicted_text"),
                        score=float(task_outcome.get("score_value") or 0.0),
                        reason=task_outcome.get("detail_record", {}).get("reason"),
                        error=task_outcome.get("task_error"),
                    )
                    return harness.evolved_session.maybe_evolve(
                        after_task_id=str(task_outcome["task_id"]),
                    )

                evolved_run = await run_sequential_task_sequence(
                    tasks=tasks,
                    updater=updater,
                    evaluate_task=lambda task, goal_text: _evaluate_with_request_text(
                        task,
                        goal_text,
                        phase_label="evolved",
                    ),
                    request_text_builder=lambda task: self._build_self_evolve_request_text(
                        task=task,
                        session=harness.evolved_session,
                        request_config=request_config,
                    ),
                    phase_label="evolved",
                    after_task_callback=_after_evolved_task,
                )
                total_score = float(evolved_run["total_score"])
                task_results = list(evolved_run["task_results"])
                detail_records = list(evolved_run["detail_records"])
                total_pass_rate = sum(
                    float(record.get("verifier_pass_rate") or 0.0)
                    for record in detail_records
                )
                self_evolution_detail = build_inter_task_self_evolution_detail(
                    harness=harness,
                    baseline_total_score=baseline_total_score,
                    baseline_run=baseline_run,
                    evolved_total_score=total_score,
                    evolved_run=evolved_run,
                )
            else:
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
                    total_pass_rate += float(outcome["verifier_pass_rate"])

                    purple_outcome = outcome.get("purple_outcome") or {}
                    executor_runtime = _merge_executor_runtime(
                        executor_runtime,
                        purple_outcome.get("executor_runtime"),
                    )

                    preview = (outcome["predicted_text"] or "")[:160]
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            f"=== Task {outcome['task_id']} | score={outcome['score_value']} | "
                            f"verifier_pass={outcome['verifier_pass_rate']:.2%} | "
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
            avg_verifier_pass_rate = total_pass_rate / len(tasks) if tasks else 0.0
            summary = build_score_summary(
                total_score,
                score_rate,
                baseline_score=baseline_total_score,
            )
            summary = (
                f"{summary}, Avg Verifier Pass Rate: {avg_verifier_pass_rate:.2%}"
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
                "request_config": dict(request.config),
                "total_tasks": len(tasks),
                "total_score": total_score,
                "score_rate": score_rate,
                "score_rate_percent": f"{score_rate:.2%}",
                "avg_verifier_pass_rate": avg_verifier_pass_rate,
                "avg_verifier_pass_rate_percent": f"{avg_verifier_pass_rate:.2%}",
                "final_summary": summary,
                "fatal_error": fatal_error,
                "executor_runtime": executor_runtime,
                "trajectory_capture": build_trajectory_capture_summary(
                    enabled=capture_trajectory,
                    trajectory_root=trajectory_root,
                ),
                "details": detail_records,
            }
            if self_evolution_detail is not None:
                detail_payload["self_evolution"] = self_evolution_detail
            if benchmark_self_evolution_detail is not None:
                detail_payload["benchmark_self_evolution"] = benchmark_self_evolution_detail

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
                request_config=dict(request.config),
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
        id="enterpriseops_green",
        name="enterpriseops_green",
        description="EnterpriseOps-Gym benchmark orchestrator (MCP + SQL verifiers)",
        tags=["enterpriseops", "benchmark", "green", "mcp", "agentic"],
        examples=['{"target":"sample"}'],
    )
    return AgentCard(
        name="enterpriseops_green",
        description="Green agent for the EnterpriseOps-Gym benchmark.",
        url=card_url or f"http://{host}:{port}/",
        version=_resolve_green_agent_version(),
        default_input_modes=["text", "text/plain", "application/json"],
        default_output_modes=["text", "text/plain", "application/json"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EnterpriseOps-Gym Green Agent.")
    parser.add_argument("--version", action="version", version=_resolve_green_agent_version())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file")
    parser.add_argument("--card-url")
    args = parser.parse_args()

    port_file = Path(args.port_file).expanduser().resolve() if args.port_file else None
    listener, actual_port = reserve_tcp_listener(args.host, args.port)
    write_port_file(port_file, actual_port)

    agent = EnterpriseOpsGymGreenAgent()
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
