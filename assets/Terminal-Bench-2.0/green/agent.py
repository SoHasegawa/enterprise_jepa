# ruff: noqa: E402

"""Terminal-Bench 2.0 Green Agent orchestration (Harbor + Docker + A2A shell protocol)."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from harbor.environments.base import BaseEnvironment
from harbor.models.task.id import LocalTaskId
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.tasks.client import TaskClient
from harbor.verifier.verifier import (
    AddTestsDirError,
    DownloadVerifierDirError,
    RewardFileEmptyError,
    RewardFileNotFoundError,
    Verifier,
    VerifierOutputParseError,
)
from pydantic import ValidationError

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.client_utils import INTERNAL_TRAJECTORY_ARTIFACT_NAME
from common.models import BenchmarkRunManifest, BenchmarkRunPaths, EvalRequest, EvalResult
from common.executor_runtime import resolve_executor_runtime_payload
from common.result_store import build_execution_identity, ensure_result_dir, write_result_artifacts
from common.trajectory import (
    build_trajectory_capture_summary,
    capture_trajectory_enabled,
    redact_trajectory_payload,
    trajectory_root_for_result,
    write_task_trajectory,
)
from common.versioning import load_component_version
from dood_environment import DoodDockerEnvironment
from messenger import Messenger
from task_loader import TaskLoader

logger = logging.getLogger(__name__)

_BENCHMARK_NAME = "Terminal-Bench-2.0"
_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("TERMINAL_BENCH_WORKSPACE", os.environ.get("WORKSPACE", "/tmp/tb-workspace")))
TASKS_DIR = WORKSPACE / "tasks"
TRIALS_DIR = WORKSPACE / "trials"


def _resolve_green_agent_version() -> str:
    version = load_component_version(
        Path(__file__).resolve().parent,
        env_name="BENCHMARK_GREEN_VERSION",
    )
    if version is None:
        raise RuntimeError("Terminal-Bench-2.0 green agent version is not configured")
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


def _assert_docker_available() -> None:
    """Fail fast when the Docker daemon is unreachable (common on fresh dev machines)."""
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Docker CLI not found. Install Docker and ensure `docker` is on PATH."
        ) from exc

    if result.returncode == 0:
        return

    detail = (result.stderr or result.stdout or "docker info failed").strip()
    if "permission denied" in detail.lower():
        raise RuntimeError(
            "Docker permission denied for user "
            f"{os.getenv('USER', 'unknown')!r}. Add yourself to the docker group, "
            "then log out and back in:\n"
            "  sudo usermod -aG docker $USER\n"
            "Verify with: docker ps\n"
            f"Original error: {detail}"
        )
    raise RuntimeError(f"Docker is not available: {detail}")


def _list_all_tasks(task_repo_dir: Path) -> list[str]:
    return sorted(
        entry.name
        for entry in task_repo_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


def _get_task_names(config: dict[str, Any], task_loader: TaskLoader) -> list[str]:
    if "tasks" in config:
        tasks = config["tasks"]
        if tasks == "all":
            names = task_loader.list_all_task_names()
        elif isinstance(tasks, list):
            names = [str(name) for name in tasks]
        else:
            raise ValueError("'tasks' must be a list of task names or 'all'")
    elif "task" in config:
        names = [str(config["task"])]
    elif "task_ids" in config and config["task_ids"]:
        names = [str(task_id) for task_id in config["task_ids"]]
    elif "target" in config:
        names = task_loader.resolve_task_names(
            target=str(config["target"]),
            requested_ids=None,
        )
    else:
        raise ValueError(
            "Config must contain 'task', 'tasks', 'task_ids', or 'target'"
        )

    exclude = config.get("exclude", [])
    if exclude:
        exclude_set = {str(name) for name in exclude}
        names = [name for name in names if name not in exclude_set]
    return names


def _get_shard_task_names(task_names: list[str], config: dict[str, Any]) -> list[str]:
    num_shards = int(config.get("num_shards", 1))
    shard_index = int(config.get("shard_index", 0))
    if num_shards < 1:
        raise ValueError("'num_shards' must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("'shard_index' must be in [0, num_shards)")
    return task_names[shard_index::num_shards]


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    if any(
        key in config
        for key in ("task", "tasks", "task_ids", "target")
    ):
        return config

    nested = config.get("assessment_config")
    if isinstance(nested, dict):
        return {**config, **nested}
    return config


def _missing_task_config_message(config: dict[str, Any]) -> str:
    keys = ", ".join(sorted(config.keys())) or "(none)"
    return (
        "Config must contain 'task', 'tasks', 'task_ids', or 'target' "
        f"(optionally under 'assessment_config'); received keys: {keys}"
    )


def _read_text_tail(path: Path, max_chars: int = 4000) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace").strip()
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return f"...\n{text[-max_chars:]}"


def _format_verifier_failure(
    task_name: str,
    error: Exception,
    trial_paths: TrialPaths,
) -> str:
    message = f"Verifier failed for {task_name}: {error}"
    test_output = _read_text_tail(trial_paths.test_stdout_path)
    if test_output:
        return f"{message}\n\ntest.sh output:\n{test_output}"
    return message


def _encode_protocol_message(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


def _decode_protocol_message(message: str) -> dict[str, Any]:
    """Parse purple protocol JSON; tolerate extra text from merged A2A parts."""
    stripped = message.strip()
    if not stripped:
        return {"kind": "final", "output": ""}

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(stripped, index)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("kind") in {
            "exec_request",
            "final",
        }:
            return payload

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return {"kind": "final", "output": message}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object payload, got: {payload!r}")
    return payload


def _format_exec_result(exit_code: int, stdout: str, stderr: str) -> str:
    return (
        f"exit_code={exit_code}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


def _reward_score(rewards: dict[str, Any] | None) -> float:
    if not rewards:
        return 0.0
    reward = rewards.get("reward")
    if isinstance(reward, (int, float)):
        return float(reward)
    return 1.0 if rewards else 0.0


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
    return events


def _resolve_result_paths(
    *,
    benchmark_name: str,
    executor_name: str,
    request_config: dict[str, Any],
    participants: dict[str, str],
    started_at: datetime.datetime,
) -> BenchmarkRunPaths:
    """Align result paths with ejepa CLI (BENCHMARK_RESULT_* env)."""
    result_paths = build_execution_identity(
        benchmark_name=benchmark_name,
        executor_name=executor_name,
        request_config=request_config,
        participants=participants,
        result_root=Path(os.getenv("BENCHMARK_RESULT_ROOT"))
        if os.getenv("BENCHMARK_RESULT_ROOT")
        else None,
        run_id=os.getenv("BENCHMARK_RUN_ID"),
        config_hash=os.getenv("BENCHMARK_CONFIG_HASH"),
        user_name=os.getenv("BENCHMARK_USER_NAME"),
        created_at_utc=started_at,
    )
    result_dir_env = os.getenv("BENCHMARK_RESULT_DIR")
    if result_dir_env:
        expected_dir = Path(result_dir_env).expanduser().resolve()
        if result_paths.result_dir != expected_dir:
            logger.warning(
                "Result dir mismatch: computed=%s env=%s; using env path",
                result_paths.result_dir,
                expected_dir,
            )
            from common.result_store import RESULT_DETAIL_FILE_NAME, RESULT_MANIFEST_FILE_NAME

            result_paths = BenchmarkRunPaths(
                result_root=result_paths.result_root,
                result_dir=expected_dir,
                manifest_path=expected_dir / RESULT_MANIFEST_FILE_NAME,
                detail_path=expected_dir / RESULT_DETAIL_FILE_NAME,
                run_id=result_paths.run_id,
                user_name=result_paths.user_name,
                config_hash=result_paths.config_hash,
                task_selection_label=result_paths.task_selection_label,
                created_at_utc=result_paths.created_at_utc,
            )
    return result_paths


def _shell_protocol_event(
    *,
    sequence: int,
    direction: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    kind = str(payload.get("kind", ""))
    record: dict[str, Any] = {
        "sequence": sequence,
        "direction": direction,
        "event_type": f"ShellProtocol{kind.title().replace('_', '')}",
        "payload": payload,
    }
    if kind == "exec_request":
        record["command"] = payload.get("command")
    elif kind == "exec_result":
        record["exit_code"] = payload.get("exit_code")
    return redact_trajectory_payload(record)


class TerminalBenchTaskError(RuntimeError):
    """Task failure that still carries a saved trajectory path when capture is enabled."""

    def __init__(
        self,
        message: str,
        *,
        trajectory_info: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.trajectory_info = trajectory_info


def _append_task_failure_event(
    trajectory_events: list[dict[str, Any]],
    error: BaseException,
) -> None:
    trajectory_events.append(
        redact_trajectory_payload(
            {
                "sequence": len(trajectory_events),
                "direction": "green",
                "event_type": "TaskFailed",
                "payload": {
                    "error": str(error),
                    "error_type": type(error).__name__,
                },
            }
        )
    )


def _persist_task_trajectory(
    *,
    trajectory_root: Path,
    task_id: str,
    trajectory_events: list[dict[str, Any]],
    task_error: BaseException | None = None,
) -> dict[str, Any] | None:
    """Write trajectory JSONL for a task, including partial events on failure."""
    if task_error is not None:
        _append_task_failure_event(trajectory_events, task_error)

    trajectory_file_path = write_task_trajectory(
        trajectory_root=trajectory_root,
        task_id=task_id,
        events=trajectory_events,
    )
    if trajectory_file_path is None:
        return None
    return {
        "trajectory_file_path": str(trajectory_file_path),
        "trajectory_event_count": len(trajectory_events),
    }


class Agent:
    required_roles: list[str] = ["agent"]
    _docker_lock = asyncio.Lock()

    def __init__(self, exec_sessions: dict[str, BaseEnvironment]):
        self.messenger = Messenger()
        self.task_client = TaskClient()
        self.task_loader = TaskLoader(benchmark_dir=_BENCHMARK_DIR)
        self.exec_sessions = exec_sessions

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {sorted(missing_roles)}"

        config = _normalize_config(dict(request.config))
        try:
            _get_task_names(config, self.task_loader)
        except (ValueError, FileNotFoundError) as exc:
            return False, str(exc)

        exclude = config.get("exclude", [])
        if exclude and not isinstance(exclude, list):
            return False, "'exclude' must be a list of task names"
        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        TASKS_DIR.mkdir(parents=True, exist_ok=True)
        TRIALS_DIR.mkdir(parents=True, exist_ok=True)

        input_text = get_message_text(message)
        try:
            request = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as exc:
            await updater.reject(new_agent_text_message(f"Invalid request: {exc}"))
            return

        config = _normalize_config(dict(request.config))
        try:
            task_names = _get_task_names(config, self.task_loader)
        except (ValueError, FileNotFoundError) as exc:
            await updater.reject(new_agent_text_message(str(exc)))
            return

        shard_task_names = _get_shard_task_names(task_names, config)
        try:
            _assert_docker_available()
        except RuntimeError as exc:
            await updater.reject(new_agent_text_message(str(exc)))
            return

        agent_url = str(request.participants["agent"])
        oracle = bool(config.get("oracle", False))
        num_shards = int(config.get("num_shards", 1))
        shard_index = int(config.get("shard_index", 0))
        benchmark_name = os.getenv("BENCHMARK_NAME", _BENCHMARK_NAME)
        executor_name = str(
            config.get("executor")
            or os.getenv("BENCHMARK_EXECUTOR")
            or "llm_shell"
        )
        started_at = datetime.datetime.now(datetime.UTC)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting Terminal-Bench 2.0: {len(shard_task_names)} task(s) "
                f"on shard {shard_index + 1}/{num_shards}"
                f"{' (oracle)' if oracle else ''}"
            ),
        )

        participants_str = {role: str(url) for role, url in request.participants.items()}
        result_paths = _resolve_result_paths(
            benchmark_name=benchmark_name,
            executor_name=executor_name,
            request_config=dict(request.config),
            participants=participants_str,
            started_at=started_at,
        )
        ensure_result_dir(result_paths)
        logger.info("Writing results to %s", result_paths.result_dir)
        capture_trajectory = capture_trajectory_enabled(dict(request.config))
        trajectory_root = (
            trajectory_root_for_result(result_paths.result_dir)
            if capture_trajectory
            else None
        )

        task_repo_dir = self.task_loader.task_repo_dir()
        task_rewards: dict[str, Any] = {}
        detail_records: list[dict[str, Any]] = []
        task_results: list[dict[str, Any]] = []

        for task_num, task_name in enumerate(shard_task_names, start=1):
            tag = f"[{task_num}/{len(shard_task_names)}]"
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"{tag} Starting task: {task_name}"),
            )
            try:
                rewards, trajectory_info = await self._run_single_task(
                    task_name=task_name,
                    task_repo_dir=task_repo_dir,
                    agent_url=agent_url,
                    updater=updater,
                    oracle=oracle,
                    tag=tag,
                    capture_trajectory=capture_trajectory,
                    trajectory_root=trajectory_root,
                )
                task_rewards[task_name] = rewards
                score = _reward_score(rewards)
                passed = score > 0
                task_results.append(
                    {
                        "task_id": task_name,
                        "score": score,
                        "reason": "passed" if passed else "failed",
                    }
                )
                detail_record: dict[str, Any] = {
                    "task_id": task_name,
                    "score": score,
                    "passed": passed,
                    "rewards": rewards,
                }
                if trajectory_info:
                    detail_record.update(trajectory_info)
                detail_records.append(detail_record)
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"{tag} Completed task: {task_name} — score={score}"
                    ),
                )
            except TerminalBenchTaskError as exc:
                logger.error("Task %s failed: %s", task_name, exc, exc_info=True)
                task_rewards[task_name] = {"error": str(exc)}
                task_results.append(
                    {"task_id": task_name, "score": 0.0, "error": str(exc)}
                )
                detail_record = {
                    "task_id": task_name,
                    "score": 0.0,
                    "passed": False,
                    "error": str(exc),
                }
                if exc.trajectory_info:
                    detail_record.update(exc.trajectory_info)
                detail_records.append(detail_record)
            except Exception as exc:
                logger.error("Task %s failed: %s", task_name, exc, exc_info=True)
                task_rewards[task_name] = {"error": str(exc)}
                task_results.append(
                    {"task_id": task_name, "score": 0.0, "error": str(exc)}
                )
                detail_records.append(
                    {
                        "task_id": task_name,
                        "score": 0.0,
                        "passed": False,
                        "error": str(exc),
                    }
                )
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"{tag} Failed task: {task_name} — {exc}"),
                )

        total_score = sum(
            _reward_score(r if isinstance(r, dict) else None)
            for r in task_rewards.values()
            if isinstance(r, dict) and "error" not in r
        )
        num_tasks = len(shard_task_names)
        num_passed = sum(
            1
            for r in task_rewards.values()
            if isinstance(r, dict) and "error" not in r and _reward_score(r) > 0
        )
        score_rate = (total_score / num_tasks) if num_tasks else 0.0
        summary = (
            f"Completed {num_tasks} task(s): {num_passed} passed, "
            f"score {total_score}/{num_tasks} ({score_rate:.0%})"
        )

        completed_at = datetime.datetime.now(datetime.UTC)

        eval_result = EvalResult(
            target=str(config.get("target") or "default"),
            total_tasks=num_tasks,
            total_score=total_score,
            score_rate=score_rate,
            task_results=task_results,
        )
        detail_payload = {
            "schema_version": "1.0",
            "run_id": result_paths.run_id,
            "status": "completed",
            "benchmark_name": benchmark_name,
            "benchmark_version": os.getenv("BENCHMARK_VERSION"),
            "green_agent_version": _resolve_green_agent_version(),
            "purple_agent_version": _resolve_purple_agent_version(),
            "executor_version": _resolve_executor_version(executor_name),
            "executor_name": executor_name,
            "target": str(config.get("target") or "default"),
            "task_ids": shard_task_names,
            "created_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "duration_seconds": round(
                (completed_at - started_at).total_seconds(), 3
            ),
            "participants": participants_str,
            "request_config": dict(request.config),
            "total_tasks": num_tasks,
            "total_score": total_score,
            "score_rate": score_rate,
            "num_passed": num_passed,
            "final_summary": summary,
            "task_rewards": task_rewards,
            "details": detail_records,
            "trajectory_capture": build_trajectory_capture_summary(
                enabled=capture_trajectory,
                trajectory_root=trajectory_root,
            ),
            "executor_runtime": resolve_executor_runtime_payload(
                benchmark_name=benchmark_name,
                executor_name=executor_name,
                request_config=dict(request.config),
                result_dir=result_paths.result_dir,
                environ=os.environ,
            ),
        }
        manifest = BenchmarkRunManifest(
            run_id=result_paths.run_id,
            user_name=result_paths.user_name,
            status="completed",
            benchmark_name=benchmark_name,
            benchmark_version=os.getenv("BENCHMARK_VERSION"),
            green_agent_version=_resolve_green_agent_version(),
            purple_agent_version=_resolve_purple_agent_version(),
            executor_version=_resolve_executor_version(executor_name),
            executor_name=executor_name,
            target=str(config.get("target") or "default"),
            task_ids=shard_task_names,
            task_selection_label=result_paths.task_selection_label,
            config_hash=result_paths.config_hash,
            created_at_utc=started_at,
            completed_at_utc=completed_at,
            duration_seconds=round((completed_at - started_at).total_seconds(), 3),
            result_dir=result_paths.result_dir,
            detail_file_path=result_paths.detail_path,
            benchmark_dir=_BENCHMARK_DIR,
            assets_root=_BENCHMARK_DIR.parent,
            participants=participants_str,
            request_config=dict(request.config),
            score_summary=summary,
            eval_result=eval_result,
        )
        detail_path, manifest_path = write_result_artifacts(
            detail_payload=detail_payload,
            manifest=manifest,
            paths=result_paths,
        )
        logger.info("Detail: %s  Manifest: %s", detail_path, manifest_path)

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary)),
                Part(
                    root=DataPart(
                        data={
                            "score": total_score,
                            "max_score": num_tasks,
                            "pass_rate": (num_passed / num_tasks * 100)
                            if num_tasks
                            else 0,
                            "task_rewards": task_rewards,
                            "detail_file_path": str(detail_path),
                            "manifest_file_path": str(manifest_path),
                        }
                    )
                ),
                Part(root=TextPart(text=eval_result.model_dump_json())),
            ],
            name="EvaluationResult",
        )

    async def _run_single_task(
        self,
        *,
        task_name: str,
        task_repo_dir: Path,
        agent_url: str,
        updater: TaskUpdater,
        oracle: bool = False,
        tag: str = "",
        capture_trajectory: bool = False,
        trajectory_root: Path | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        async with self._docker_lock:
            return await self._run_single_task_locked(
                task_name=task_name,
                task_repo_dir=task_repo_dir,
                agent_url=agent_url,
                updater=updater,
                oracle=oracle,
                tag=tag,
                capture_trajectory=capture_trajectory,
                trajectory_root=trajectory_root,
            )

    async def _run_single_task_locked(
        self,
        *,
        task_name: str,
        task_repo_dir: Path,
        agent_url: str,
        updater: TaskUpdater,
        oracle: bool = False,
        tag: str = "",
        capture_trajectory: bool = False,
        trajectory_root: Path | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        session_token = uuid4().hex
        session_id = uuid4().hex
        environment: BaseEnvironment | None = None
        trajectory_events: list[dict[str, Any]] = []
        effective_capture_trajectory = capture_trajectory and trajectory_root is not None
        rewards: dict[str, Any] = {}
        task_error: BaseException | None = None
        trajectory_info: dict[str, Any] | None = None

        try:
            task_id = LocalTaskId(path=task_repo_dir / task_name)
            batch = await self.task_client.download_tasks(
                [task_id], output_dir=TASKS_DIR
            )
            task = Task(batch.paths[0])

            trial_dir = TRIALS_DIR / session_id
            trial_paths = TrialPaths(trial_dir=trial_dir)
            trial_paths.mkdir()

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"{tag} {task_name}: Starting environment..."),
            )
            environment = DoodDockerEnvironment(
                environment_dir=task.paths.environment_dir,
                environment_name=f"terminal-bench-{task_name}",
                session_id=session_id,
                trial_paths=trial_paths,
                task_env_config=task.config.environment,
                logger=logger,
            )
            build_timeout = task.config.environment.build_timeout_sec
            await asyncio.wait_for(
                environment.start(force_build=False),
                timeout=build_timeout,
            )

            self.exec_sessions[session_token] = environment

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"{tag} {task_name}: Agent working..."),
            )
            agent_timeout = int(task.config.agent.timeout_sec)

            if oracle:
                await self._run_oracle_locally(
                    task=task,
                    environment=environment,
                    timeout=agent_timeout,
                    capture_trajectory=effective_capture_trajectory,
                    trajectory_events=trajectory_events,
                )
            else:
                self.messenger.reset()
                await self._run_agent_session(
                    instruction=task.instruction,
                    environment=environment,
                    agent_url=agent_url,
                    updater=updater,
                    timeout=agent_timeout,
                    tag=f"{tag} {task_name}",
                    capture_trajectory=effective_capture_trajectory,
                    trajectory_events=trajectory_events,
                )

            self.exec_sessions.pop(session_token, None)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"{tag} {task_name}: Verifying..."),
            )
            verifier = Verifier(
                task=task,
                trial_paths=trial_paths,
                environment=environment,
                logger=logger,
            )
            verifier_timeout = task.config.verifier.timeout_sec
            try:
                verifier_result = await asyncio.wait_for(
                    verifier.verify(),
                    timeout=verifier_timeout,
                )
                rewards = verifier_result.rewards or {}
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    _format_verifier_failure(
                        task_name,
                        RuntimeError(f"Verifier timed out after {verifier_timeout}s"),
                        trial_paths,
                    )
                ) from exc
            except (
                AddTestsDirError,
                DownloadVerifierDirError,
                RewardFileEmptyError,
                RewardFileNotFoundError,
                VerifierOutputParseError,
            ) as exc:
                raise RuntimeError(
                    _format_verifier_failure(task_name, exc, trial_paths)
                ) from exc

        except Exception as exc:
            task_error = exc

        finally:
            if effective_capture_trajectory and trajectory_root is not None:
                trajectory_info = _persist_task_trajectory(
                    trajectory_root=trajectory_root,
                    task_id=task_name,
                    trajectory_events=trajectory_events,
                    task_error=task_error,
                )
            self.exec_sessions.pop(session_token, None)
            if environment:
                try:
                    await environment.stop(delete=False)
                except Exception:
                    pass

        if task_error is not None:
            raise TerminalBenchTaskError(
                str(task_error),
                trajectory_info=trajectory_info,
            ) from task_error
        return rewards, trajectory_info

    async def _run_oracle_locally(
        self,
        *,
        task: Task,
        environment: BaseEnvironment,
        timeout: int,
        capture_trajectory: bool = False,
        trajectory_events: list[dict[str, Any]] | None = None,
    ) -> str:
        solve_path = task.paths.solve_path
        if not solve_path.exists():
            raise FileNotFoundError(f"Solution script not found: {solve_path}")

        await environment.upload_dir(
            source_dir=task.paths.solution_dir,
            target_dir="/solution",
        )
        command = f"timeout {timeout}s bash /solution/solve.sh"
        if capture_trajectory and trajectory_events is not None:
            trajectory_events.append(
                _shell_protocol_event(
                    sequence=len(trajectory_events),
                    direction="green",
                    payload={
                        "kind": "oracle_exec",
                        "command": command,
                    },
                )
            )
        result = await environment.exec(
            command,
            timeout_sec=timeout + 15,
        )
        logger.info(
            "Oracle solve finished: task=%s exit_code=%s",
            task.name,
            result.return_code,
        )
        if capture_trajectory and trajectory_events is not None:
            trajectory_events.append(
                _shell_protocol_event(
                    sequence=len(trajectory_events),
                    direction="green",
                    payload={
                        "kind": "exec_result",
                        "exit_code": result.return_code,
                        "stdout": result.stdout or "",
                        "stderr": result.stderr or "",
                    },
                )
            )
        return _format_exec_result(
            result.return_code, result.stdout or "", result.stderr or ""
        )

    async def _run_agent_session(
        self,
        *,
        instruction: str,
        environment: BaseEnvironment,
        agent_url: str,
        updater: TaskUpdater,
        timeout: int,
        tag: str,
        capture_trajectory: bool = False,
        trajectory_events: list[dict[str, Any]] | None = None,
    ) -> str:
        start = time.monotonic()
        transcript: list[str] = []
        outbound = _encode_protocol_message(
            {
                "kind": "task",
                "protocol": "terminal-bench-shell-v1",
                "instruction": instruction,
            }
        )
        new_conversation = True

        while True:
            elapsed = int(time.monotonic() - start)
            remaining = max(1, timeout - elapsed)
            outbound_payload = json.loads(outbound)
            if capture_trajectory and trajectory_events is not None:
                trajectory_events.append(
                    _shell_protocol_event(
                        sequence=len(trajectory_events),
                        direction="outbound",
                        payload=outbound_payload,
                    )
                )

            if capture_trajectory and trajectory_events is not None:
                purple_outputs = await self.messenger.talk_to_agent_with_trajectory(
                    message=outbound,
                    url=agent_url,
                    new_conversation=new_conversation,
                    timeout=remaining,
                    capture_trajectory=True,
                )
                response = str(purple_outputs.get("response", ""))
                trajectory_events.extend(purple_outputs.get("events", []))
                trajectory_events.extend(
                    _internal_trajectory_events_from_artifacts(
                        purple_outputs.get("artifacts", [])
                    )
                )
            else:
                response = await self.messenger.talk_to_agent(
                    message=outbound,
                    url=agent_url,
                    new_conversation=new_conversation,
                    timeout=remaining,
                )
            new_conversation = False
            transcript.append(f"green->agent\n{outbound}")
            transcript.append(f"agent->green\n{response}")

            payload = _decode_protocol_message(response)
            kind = payload["kind"]
            if capture_trajectory and trajectory_events is not None:
                trajectory_events.append(
                    _shell_protocol_event(
                        sequence=len(trajectory_events),
                        direction="inbound",
                        payload=payload,
                    )
                )

            if kind == "final":
                return "\n\n".join(transcript)

            if kind != "exec_request":
                raise RuntimeError(f"Unexpected agent payload: {payload}")

            command = payload.get("command")
            if not isinstance(command, str) or not command.strip():
                raise RuntimeError(f"Invalid exec_request: {payload}")

            command_timeout = payload.get("timeout", 30)
            if not isinstance(command_timeout, int):
                raise RuntimeError(f"Invalid timeout in exec_request: {payload}")
            command_timeout = max(1, min(command_timeout, 300))

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"{tag}: $ {command}"),
            )
            try:
                result = await environment.exec(command, timeout_sec=command_timeout)
            except RuntimeError as exc:
                message = str(exc)
                if "Command timed out after" not in message:
                    raise
                result_payload = {
                    "kind": "exec_result",
                    "exit_code": 124,
                    "stdout": "",
                    "stderr": message,
                    "timed_out": True,
                    "timeout_sec": command_timeout,
                }
                if capture_trajectory and trajectory_events is not None:
                    trajectory_events.append(
                        _shell_protocol_event(
                            sequence=len(trajectory_events),
                            direction="green",
                            payload=result_payload,
                        )
                    )
                outbound = _encode_protocol_message(result_payload)
                continue
            if capture_trajectory and trajectory_events is not None:
                trajectory_events.append(
                    _shell_protocol_event(
                        sequence=len(trajectory_events),
                        direction="green",
                        payload={
                            "kind": "exec_result",
                            "exit_code": result.return_code,
                            "stdout": result.stdout or "",
                            "stderr": result.stderr or "",
                        },
                    )
                )
            outbound = _encode_protocol_message(
                {
                    "kind": "exec_result",
                    "exit_code": result.return_code,
                    "stdout": result.stdout or "",
                    "stderr": result.stderr or "",
                }
            )
