"""
Entropic CRMArena Green Agent

Implements CRM agent evaluation with Schema Drift, Context Rot,
and 7-Dimension Scoring.

Uses centralized configuration from shared.config.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
import tomllib
from typing import Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, HttpUrl, ValidationError, Field
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
SRC_DIR_STR = str(SRC_DIR)
if SRC_DIR_STR not in sys.path:
    sys.path.insert(0, SRC_DIR_STR)

from common.executor_runtime import resolve_executor_runtime_payload
from common.models import BenchmarkRunManifest, EvalResult
from common.result_store import build_execution_identity, ensure_result_dir, write_result_artifacts
from common.trajectory import (
    build_trajectory_capture_summary,
    capture_trajectory_enabled,
    trajectory_root_for_result,
    write_task_trajectory,
)
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


def build_self_evolve_score_summary(total_score: float, score_rate: float, *, baseline_score: float | None = None) -> str:
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
from messenger import Messenger
from shared.config import settings

# Import CRM modules (these will be in the crm/ directory)
from crm.tasks import TaskLoader, CRMTask, TASK_CATEGORIES
from crm.splits import is_split_target, resolve_split_task_ids
from crm.entropy import EntropyEngine, DriftLevel, RotLevel
from crm.evaluator import CRMArenaEvaluator
from crm.scorer import SevenDimensionScorer, AgentMetrics

logger = logging.getLogger(__name__)
_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
INTERNAL_TRAJECTORY_ARTIFACT_NAME = "internal_trajectory"
SELF_EVOLVE_ADAPTER = build_self_evolve_adapter()


def _as_bool(value: Any) -> bool:
    """Coerce a config value (bool/str/int) to bool; None -> False."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _load_component_version(component_dir: Path, env_name: str) -> str | None:
    return load_component_version(component_dir, env_name=env_name)


def _build_score_summary(aggregated: dict[str, Any]) -> str:
    summary = aggregated.get("summary", {})
    parts = [
        f"Total Tasks: {summary.get('total_tasks', 0)}",
        f"Passed: {summary.get('total_passed', 0)}",
        f"Pass Rate: {float(summary.get('pass_rate', 0.0)):.2%}",
        f"Avg Score: {float(summary.get('avg_score', 0.0)):.1f}",
    ]
    original = aggregated.get("original")
    if isinstance(original, dict):
        scores = original.get("scores", {})
        parts.append(f"Original Accuracy: {float(scores.get('accuracy_percent', 0.0)):.1f}%")
    return ", ".join(parts)


def _build_manifest_task_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert CRMArenaPro-native task results into the generic CLI table schema."""
    task_results: list[dict[str, Any]] = []
    for result in results:
        dataset_reference = result.get("dataset_reference")
        if not isinstance(dataset_reference, dict):
            dataset_reference = {}

        crm_reward = result.get("crm_reward", 0)
        success = bool(result.get("success"))
        task_result: dict[str, Any] = {
            "task_id": str(result.get("task_idx") or "unknown"),
            "score": crm_reward,
            "eval_func": dataset_reference.get("reward_metric") or "crmarenapro",
            "reason": "correct" if success else "incorrect",
            "task_category": result.get("task_category"),
            "total_score": result.get("total_score"),
        }
        if result.get("error"):
            task_result["error"] = result["error"]
            task_result["reason"] = None
        task_results.append(task_result)
    return task_results


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

        steps = native_payload.get("steps") if isinstance(native_payload, dict) else None
        if isinstance(steps, list):
            for idx, step in enumerate(steps):
                events.append(
                    {
                        "sequence": idx,
                        "direction": "purple_internal",
                        "event_type": "PurpleInternalStep",
                        "source": source,
                        "payload": step,
                    }
                )
    return events


def _write_benchmark_artifacts(
    *,
    request: "EvalRequest",
    aggregated: dict[str, Any],
    started_at_utc: datetime,
    completed_at_utc: datetime,
) -> None:
    benchmark_name = os.getenv("BENCHMARK_NAME") or _BENCHMARK_DIR.name
    executor_name = str(
        request.config.get("executor")
        or os.getenv("BENCHMARK_EXECUTOR")
        or "baseline_crm_agent"
    )
    participants = {role: str(url) for role, url in request.participants.items()}
    result_paths = build_execution_identity(
        benchmark_name=benchmark_name,
        executor_name=executor_name,
        request_config=request.config,
        participants=participants,
        result_root=Path(os.getenv("BENCHMARK_RESULT_ROOT")) if os.getenv("BENCHMARK_RESULT_ROOT") else None,
        run_id=os.getenv("BENCHMARK_RUN_ID"),
        config_hash=os.getenv("BENCHMARK_CONFIG_HASH"),
        created_at_utc=started_at_utc,
    )
    ensure_result_dir(result_paths)

    summary = aggregated.get("summary", {})
    total_tasks = int(summary.get("total_tasks", 0) or 0)
    total_passed = int(summary.get("total_passed", 0) or 0)
    score_rate = float(summary.get("pass_rate", 0.0) or 0.0)
    avg_score = float(summary.get("avg_score", 0.0) or 0.0)
    task_results = _build_manifest_task_results(list(aggregated.get("results", [])))
    eval_result = EvalResult(
        target=str(request.config.get("target") or "default"),
        total_tasks=total_tasks,
        total_score=avg_score,
        score_rate=score_rate,
        task_results=task_results,
    )
    runtime_payload = resolve_executor_runtime_payload(
        benchmark_name=benchmark_name,
        executor_name=executor_name,
        request_config=request.config,
        result_dir=result_paths.result_dir,
        environ=os.environ,
    )
    aggregated["executor_runtime"] = runtime_payload
    aggregated["result_dir"] = str(result_paths.result_dir)
    capture_trajectory = capture_trajectory_enabled(request.config)
    trajectory_root = trajectory_root_for_result(result_paths.result_dir) if capture_trajectory else None
    aggregated["trajectory_capture"] = build_trajectory_capture_summary(
        enabled=capture_trajectory,
        trajectory_root=trajectory_root,
    )

    manifest = BenchmarkRunManifest(
        run_id=result_paths.run_id,
        user_name=result_paths.user_name,
        status="completed",
        benchmark_name=benchmark_name,
        benchmark_version=os.getenv("BENCHMARK_VERSION"),
        green_agent_version=_load_component_version(_BENCHMARK_DIR / "green", "BENCHMARK_GREEN_VERSION"),
        purple_agent_version=_load_component_version(_BENCHMARK_DIR / "purple", "BENCHMARK_PURPLE_VERSION"),
        executor_version=_load_component_version(
            _BENCHMARK_DIR / "purple-executors" / executor_name,
            "BENCHMARK_EXECUTOR_VERSION",
        ),
        executor_name=executor_name,
        target=str(request.config.get("target") or "default"),
        task_ids=[str(task_id) for task_id in request.config.get("task_ids", []) or []],
        task_selection_label=result_paths.task_selection_label,
        config_hash=result_paths.config_hash,
        created_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        duration_seconds=(completed_at_utc - started_at_utc).total_seconds(),
        result_dir=result_paths.result_dir,
        detail_file_path=result_paths.detail_path,
        benchmark_dir=_BENCHMARK_DIR,
        assets_root=_BENCHMARK_DIR.parent,
        participants=participants,
        request_config=dict(request.config),
        score_summary=_build_score_summary(aggregated),
        eval_result=eval_result,
    )
    write_result_artifacts(detail_payload=aggregated, manifest=manifest, paths=result_paths)
    logger.info(
        "Wrote result artifacts to %s (%s/%s tasks passed)",
        result_paths.result_dir,
        total_passed,
        total_tasks,
    )


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


class AssessmentConfig(BaseModel):
    """
    Configuration for CRMArena assessment.
    
    Only task selection parameters are configurable from leaderboard.
    Adversarial testing parameters are HARDCODED for consistent evaluation.
    """
    # Task selection (configurable)
    target: Optional[str] = Field(
        None,
        description="Named task split (crmarena_100_test / crmarena_train / crmarena_valid)",
    )
    task_ids: Optional[list[str]] = Field(None, description="Specific task IDs to run")
    task_offset: int = Field(
        0,
        description="Start index into the resolved split (for sharding a split across jobs)",
    )
    task_categories: Optional[list[str]] = Field(None, description="Filter by task categories")
    task_percentage: float = Field(
        default_factory=lambda: settings.assessment.task_percentage,
        description="Percentage of tasks to sample (1-100)"
    )
    task_limit: Optional[int] = Field(
        default_factory=lambda: settings.assessment.task_limit,
        description="Maximum number of tasks"
    )
    
    # =========================================================================
    # HARDCODED ADVERSARIAL PARAMETERS (not configurable from leaderboard)
    # These ensure consistent, reproducible adversarial robustness testing
    # =========================================================================
    
    # Entropy settings - HARDCODED for adversarial testing
    drift_level: str = Field(
        default="medium",
        description="Schema drift: medium (HARDCODED for adversarial testing)"
    )
    rot_level: str = Field(
        default="medium",
        description="Context rot: medium (HARDCODED for adversarial testing)"
    )
    
    # Evaluation settings - HARDCODED for consistent benchmarking
    max_steps: int = Field(
        default=10,
        description="Maximum agent turns per task (HARDCODED)"
    )
    timeout: int = Field(
        default=300,
        description="Timeout per task in seconds (HARDCODED)"
    )
    org_type: str = Field(
        default="b2b",
        description="Organization type: b2b (HARDCODED for CRMArenaPro B2B split)"
    )
    
    # Original mode compatibility (both run by default)
    skip_original: bool = Field(
        default_factory=lambda: settings.assessment.skip_original,
        description="Skip original CRMArena-Pro scoring (default: False = run both)"
    )


class Agent:
    """
    Entropic CRMArena Green Agent.
    
    Evaluates CRM agents with adversarial robustness testing:
    - Schema Drift: Randomly renames database columns
    - Context Rot: Injects distractor records into results
    - 7D Scoring: Multi-dimensional evaluation
    """
    
    # Required participant role
    required_roles: list[str] = ["agent"]
    required_config_keys: list[str] = []  # All config has defaults

    def __init__(self):
        self.messenger = Messenger()
        self.task_loader: Optional[TaskLoader] = None
        self.entropy_engine: Optional[EntropyEngine] = None
        self.evaluator: Optional[CRMArenaEvaluator] = None
        self.scorer = SevenDimensionScorer()
        self.results: list[dict[str, Any]] = []
        
        # Original mode components (initialized in _initialize_components)
        self.original_evaluator = None
        self.original_scorer = None

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate the assessment request."""
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing required participant roles: {missing_roles}"

        # Note: drift_level, rot_level, max_steps, timeout, org_type are HARDCODED
        # and not validated from config (they're ignored in _parse_config)
        
        # Validate task categories if specified
        config = request.config
        categories = config.get("task_categories")
        if categories:
            invalid = [c for c in categories if c not in TASK_CATEGORIES]
            if invalid:
                return False, f"Invalid task categories: {invalid}"

        max_parallel = config.get("max_parallel")
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
                or "baseline_crm_agent"
            ),
            supported_executor_names=SELF_EVOLVE_ADAPTER.supported_executor_names,
        )
        if not is_valid_self_evolve:
            return False, self_evolve_message

        return True, "ok"

    @staticmethod
    def _resolve_target_task_ids(target: Optional[str]) -> Optional[list[str]]:
        """Map a named ``--config target`` to its task-id list (green/tasks/task_ids.toml).

        Lets CRMArenaPro expose fixed test splits (e.g. ``world_model_test`` — the 428
        held-out B2B task ids from ewm/trajectories/crmarenapro_multi_model_world_model_
        test_trajectories.json) the same way EnterpriseOps-Gym/Terminal-Bench do, resolving
        to explicit dataset-idx task_ids the loader fetches via ``get_task_by_idx``. Returns
        ``None`` for the default/``sample`` target or an unknown name, leaving the existing
        ``task_limit`` sampling behaviour intact.
        """
        name = (target or "").strip()
        if not name or name in {"sample", "default"}:
            return None
        path = Path(__file__).resolve().parent / "tasks" / "task_ids.toml"
        if not path.exists():
            return None
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        ids = raw.get(name)
        if not isinstance(ids, list) or not ids:
            return None
        return [str(task_id) for task_id in ids]

    def _parse_config(self, config: dict[str, Any]) -> AssessmentConfig:
        """
        Parse assessment configuration.

        Only task selection parameters are accepted from config.
        Adversarial parameters (drift_level, rot_level, max_steps, timeout, org_type)
        are HARDCODED and cannot be overridden for consistent evaluation.
        """
        # Leaderboard-replication mode: disable the adversarial perturbations so
        # the run is as close as possible to the official CRMArena-Pro setting
        # (no schema drift, no context rot, higher interaction budget). This is
        # the ONLY sanctioned way to turn off the otherwise-hardcoded adversarial
        # params; the default (flag absent/false) keeps the strict entropic eval.
        leaderboard_mode = _as_bool(config.get("leaderboard_mode"))

        if leaderboard_mode:
            drift_level = "none"
            rot_level = "none"
            # Official harness allows a larger tool-call budget than the strict
            # entropic default of 10; overridable via config.
            max_steps = int(config.get("max_steps") or 20)
            timeout = int(config.get("timeout") or 600)
            logger.info(
                "Leaderboard mode: drift=none, rot=none, "
                f"max_steps={max_steps}, timeout={timeout} (adversarial perturbations OFF)"
            )
        else:
            # Log if caller tried to override hardcoded params (for awareness)
            hardcoded_overrides = []
            if config.get("drift_level") and config.get("drift_level") != "medium":
                hardcoded_overrides.append(f"drift_level={config.get('drift_level')}")
            if config.get("rot_level") and config.get("rot_level") != "medium":
                hardcoded_overrides.append(f"rot_level={config.get('rot_level')}")
            if config.get("max_steps") and config.get("max_steps") != 10:
                hardcoded_overrides.append(f"max_steps={config.get('max_steps')}")
            if config.get("org_type") and config.get("org_type") != "b2b":
                hardcoded_overrides.append(f"org_type={config.get('org_type')}")

            if hardcoded_overrides:
                logger.warning(
                    f"Ignoring config overrides for hardcoded params: {hardcoded_overrides}. "
                    "Using: drift_level=medium, rot_level=medium, max_steps=10, org_type=b2b. "
                    "Pass leaderboard_mode=true to disable the adversarial perturbations."
                )
            drift_level, rot_level, max_steps, timeout = "medium", "medium", 10, 300

        return AssessmentConfig(
            # Configurable task selection params
            target=config.get("target"),
            task_ids=config.get("task_ids"),
            task_offset=int(config.get("task_offset") or 0),
            task_categories=config.get("task_categories"),
            task_percentage=config.get("task_percentage", 5.0),
            task_limit=config.get("task_limit"),
            skip_original=config.get("skip_original", False),
            # Adversarial params: hardcoded unless leaderboard_mode is set.
            drift_level=drift_level,
            rot_level=rot_level,
            max_steps=max_steps,
            timeout=timeout,
            org_type="b2b",
        )

    def _initialize_components(self, config: AssessmentConfig):
        """Initialize evaluation components."""
        # Task loader
        self.task_loader = TaskLoader(org_type=config.org_type)
        
        # Entropy engine for drift/rot (Entropic mode)
        drift = DriftLevel(config.drift_level) if config.drift_level != "none" else DriftLevel.NONE
        rot = RotLevel(config.rot_level) if config.rot_level != "none" else RotLevel.NONE
        self.entropy_engine = EntropyEngine(drift_level=drift, rot_level=rot)
        
        # Entropic evaluator (always initialized)
        self.evaluator = CRMArenaEvaluator()
        
        # Original mode components (initialized by default unless skip_original=True)
        if not config.skip_original:
            from original import OriginalEvaluator, OriginalScorer
            self.original_evaluator = OriginalEvaluator()
            self.original_scorer = OriginalScorer()
            logger.info("Original CRMArena-Pro scoring ENABLED (both modes will run)")
        else:
            self.original_evaluator = None
            self.original_scorer = None
            logger.info("Original CRMArena-Pro scoring DISABLED (entropic only)")

    def _get_tasks(self, config: AssessmentConfig) -> list[CRMTask]:
        """
        Get tasks based on configuration.
        
        Task selection priority:
        1. target: Named fixed split (crmarena_100_test / crmarena_train /
           crmarena_valid) — mirrors EnterpriseOps-Gym's opsgym_80_test design;
           train/valid are derived test-excluded (see crm/splits.py). With
           task_limit, the split is truncated deterministically (corpus order).
        2. task_ids: Run specific task IDs if provided
        3. task_categories: Filter by categories if provided
        4. task_limit: Run up to N tasks if provided
        5. Default: Run ALL tasks (full 2,140 task benchmark)
        """
        if config.target and is_split_target(config.target):
            corpus_ids = [row["idx"] for row in self.task_loader.dataset]
            split_ids = resolve_split_task_ids(config.target, corpus_ids)
            full = len(split_ids)
            # Range window for sharding: task_offset selects the start, task_limit
            # the count (0 = to the end). benchmark.toml defaults task_limit=1, so
            # pass task_limit=0 to run a whole split/shard end.
            offset = max(0, int(config.task_offset or 0))
            if offset or (config.task_limit and config.task_limit > 0):
                end = offset + config.task_limit if (config.task_limit and config.task_limit > 0) else full
                split_ids = split_ids[offset:end]
                logger.info(
                    f"Split '{config.target}' window [{offset}:{end}] -> "
                    f"{len(split_ids)}/{full} tasks (task_offset={offset}, task_limit={config.task_limit})"
                )
            tasks = []
            for task_id in split_ids:
                task = self.task_loader.get_task_by_idx(str(task_id))
                if task:
                    tasks.append(task)
            if len(tasks) != len(split_ids):
                raise RuntimeError(
                    f"split {config.target}: resolved {len(tasks)}/{len(split_ids)} tasks"
                )
            logger.info(f"Running {len(tasks)} tasks from split '{config.target}'")
            return tasks

        if config.target and config.target not in {"default", "sample"}:
            logger.warning(f"Unknown target '{config.target}'; falling back to task selection params")

        if config.task_ids:
            # Specific task IDs requested
            tasks = []
            for task_id in config.task_ids:
                task = self.task_loader.get_task_by_idx(task_id)
                if task:
                    tasks.append(task)
            logger.info(f"Running {len(tasks)} specific tasks by ID")
            return tasks
        
        elif config.task_categories:
            # Filter by categories
            tasks = self.task_loader.load_tasks(
                categories=config.task_categories,
                limit=config.task_limit
            )
            logger.info(f"Running {len(tasks)} tasks from categories: {config.task_categories}")
            return tasks
        
        elif config.task_limit:
            # task_limit provided - run limited sample for quick testing
            all_tasks = self.task_loader.load_tasks()
            random.seed(42)  # Reproducibility
            sample_size = min(config.task_limit, len(all_tasks))
            tasks = random.sample(all_tasks, sample_size)
            logger.info(f"Running {len(tasks)} tasks (task_limit={config.task_limit})")
            return tasks
        
        else:
            # No limit specified - run ALL tasks (full benchmark)
            all_tasks = self.task_loader.load_tasks()
            logger.info(f"Running ALL {len(all_tasks)} tasks (full benchmark)")
            return all_tasks

    async def _evaluate_single_task(
        self,
        task: CRMTask,
        purple_agent_url: str,
        config: AssessmentConfig,
        updater: TaskUpdater,
        capture_trajectory: bool = False,
        trajectory_root: Path | None = None,
        request_text: str | None = None,
        request_config: dict[str, Any] | None = None,
        trajectory_label: str | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a single task with the purple agent.
        
        Sends the task prompt to the purple agent and evaluates the response.
        Tracks metrics for all 7 dimensions of scoring.
        """
        task_start_time = time.time()
        
        # Build task context with entropy applied
        context_start = time.time()
        task_context = self._build_task_context(task, config)
        context_time = time.time() - context_start
        
        # Initialize metric tracking
        conversation_turns = 0
        total_tokens_estimate = 0
        tool_calls_detected = 0
        invalid_tool_calls = 0
        queries_detected = 0
        errors_encountered = 0
        errors_recovered = 0
        purple_agent_time = 0.0  # Track time spent waiting for Purple Agent
        trajectory_file_path: Path | None = None
        trajectory_events: list[dict[str, Any]] = []
        effective_capture_trajectory = capture_trajectory and trajectory_root is not None
        
        # Optimal turns heuristic based on task complexity
        optimal_turns = self._estimate_optimal_turns(task)
        
        # Send task to purple agent
        try:
            # First turn - send task
            conversation_turns += 1
            turn_start = time.time()
            if effective_capture_trajectory:
                purple_outputs = await self.messenger.talk_to_agent_with_trajectory(
                    message=request_text or json.dumps(task_context),
                    url=purple_agent_url,
                    new_conversation=True,
                    timeout=config.timeout,
                    request_config=request_config,
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
                    message=request_text or json.dumps(task_context),
                    url=purple_agent_url,
                    new_conversation=True,
                    timeout=config.timeout,
                    request_config=request_config,
                )
            turn_time = time.time() - turn_start
            purple_agent_time += turn_time
            logger.info(f"[TIMING] Task {task.idx} Turn 1: {turn_time:.2f}s (Purple Agent)")
            
            # Track metrics from response
            response_metrics = self._parse_response_metrics(response)
            total_tokens_estimate += response_metrics.get("tokens", 0)
            tool_calls_detected += response_metrics.get("tool_calls", 0)
            invalid_tool_calls += response_metrics.get("invalid_tool_calls", 0)
            queries_detected += response_metrics.get("queries", 0)
            
            # Multi-turn: Allow agent to request more info or make tool calls
            max_turns = min(config.max_steps, 10)
            while conversation_turns < max_turns:
                # Check if agent needs more turns
                needs_continuation = self._check_needs_continuation(response)
                if not needs_continuation:
                    break
                
                conversation_turns += 1
                try:
                    turn_start = time.time()
                    if effective_capture_trajectory:
                        purple_outputs = await self.messenger.talk_to_agent_with_trajectory(
                            message="Continue processing. Provide your final answer.",
                            url=purple_agent_url,
                            new_conversation=False,
                            timeout=config.timeout,
                            request_config=request_config,
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
                            message="Continue processing. Provide your final answer.",
                            url=purple_agent_url,
                            new_conversation=False,  # Continue conversation
                            timeout=config.timeout,
                            request_config=request_config,
                        )
                    turn_time = time.time() - turn_start
                    purple_agent_time += turn_time
                    logger.info(f"[TIMING] Task {task.idx} Turn {conversation_turns}: {turn_time:.2f}s (Purple Agent)")
                    
                    # Track additional metrics
                    turn_metrics = self._parse_response_metrics(response)
                    total_tokens_estimate += turn_metrics.get("tokens", 0)
                    tool_calls_detected += turn_metrics.get("tool_calls", 0)
                    queries_detected += turn_metrics.get("queries", 0)
                    
                except Exception as e:
                    errors_encountered += 1
                    logger.warning(f"Turn {conversation_turns} error: {e}")
                    break
            
            # Parse agent response
            agent_answer = self._extract_answer(response)
            
            # ================================================================
            # ENTROPIC EVALUATION (always runs)
            # ================================================================
            eval_start = time.time()
            eval_result = self.evaluator.evaluate(
                proposed_answer=agent_answer,
                gt_answer=task.answer,
                reward_metric=task.reward_metric,
                task_name=task.task,
            )
            eval_time = time.time() - eval_start
            
            crm_reward = eval_result.get("reward", 0)
            task_completed = True
            final_state = "success" if crm_reward > 0 else "failed"
            
            # Create metrics for 7D scoring with REAL tracked values
            metrics = AgentMetrics(
                task_completed=task_completed,
                crm_reward=crm_reward,
                drift_level=self._drift_level_to_int(config.drift_level),
                drift_percentage=self.entropy_engine.get_drift_percentage() if self.entropy_engine else 0,
                rot_level=self._rot_level_to_int(config.rot_level),
                # Token tracking
                total_tokens=total_tokens_estimate,
                # Query tracking
                queries_executed=queries_detected,
                queries_failed=0,
                # Error tracking
                errors_encountered=errors_encountered,
                errors_recovered=errors_recovered,
                final_state=final_state,
                # Trajectory tracking (TES)
                actual_turns=conversation_turns,
                optimal_turns=optimal_turns,
                # Hallucination tracking
                total_tool_calls=tool_calls_detected,
                invalid_tool_calls=invalid_tool_calls,
                malformed_tool_calls=0,
            )
            
            # Calculate 7D score (Entropic)
            score_start = time.time()
            score_result = self.scorer.score(task.idx, task.task, metrics)
            score_time = time.time() - score_start
            
            # ================================================================
            # ORIGINAL EVALUATION (runs by default unless skip_original=True)
            # ================================================================
            original_result = None
            original_eval_time = 0
            if self.original_evaluator is not None:
                original_eval_start = time.time()
                original_eval_result = self.original_evaluator.evaluate(
                    proposed_answer=agent_answer,
                    gt_answer=task.answer,
                    reward_metric=task.reward_metric,
                    task_name=task.task,
                )
                original_eval_time = time.time() - original_eval_start
                
                # Record in original scorer
                self.original_scorer.score(
                    task_idx=task.idx,
                    task_name=task.task,
                    reward=original_eval_result.get("reward", 0),
                    parsed_answer=original_eval_result.get("parsed_answer", []),
                    gt_answer=task.answer if isinstance(task.answer, list) else [task.answer],
                    metrics=original_eval_result.get("metrics"),
                )
                
                original_result = {
                    "reward": original_eval_result.get("reward", 0),
                    "parsed_answer": original_eval_result.get("parsed_answer", []),
                }
                logger.info(f"  Original mode: reward={original_result['reward']}")
            
            # Calculate total task time
            task_total_time = time.time() - task_start_time
            green_agent_time = task_total_time - purple_agent_time
            if effective_capture_trajectory and trajectory_events:
                trajectory_file_path = write_task_trajectory(
                    trajectory_root=trajectory_root,
                    task_id=str(task.idx),
                    events=trajectory_events,
                    label=trajectory_label,
                )
            
            # Log timing breakdown
            logger.info(f"[TIMING] Task {task.idx} COMPLETE:")
            logger.info(f"  ├─ Purple Agent: {purple_agent_time:.2f}s ({purple_agent_time/task_total_time*100:.1f}%)")
            logger.info(f"  ├─ Green Agent:  {green_agent_time:.2f}s ({green_agent_time/task_total_time*100:.1f}%)")
            logger.info(f"  │   ├─ Context build: {context_time:.3f}s")
            logger.info(f"  │   ├─ Entropic eval: {eval_time:.3f}s")
            if original_eval_time > 0:
                logger.info(f"  │   ├─ Original eval: {original_eval_time:.3f}s")
            logger.info(f"  │   └─ Scoring:       {score_time:.3f}s")
            logger.info(f"  └─ TOTAL: {task_total_time:.2f}s")
            
            result = {
                "task_idx": task.idx,
                "task_category": task.task,
                # Original question/task from CRMArenaPro dataset
                "task_query": task.query,
                "dataset_reference": {
                    "source": "Salesforce/CRMArenaPro",
                    "split": "b2b",
                    "idx": task.idx,
                    "reward_metric": task.reward_metric,
                },
                # Entropic scores
                "entropic": {
                    "crm_reward": crm_reward,
                    "total_score": score_result.total_score,
                    "dimension_scores": score_result.dimension_breakdown,
                    "success": crm_reward > 0,
                },
                # Keep legacy fields for backward compatibility
                "crm_reward": crm_reward,
                "total_score": score_result.total_score,
                "dimension_scores": score_result.dimension_breakdown,
                "agent_answer": agent_answer[:500] if agent_answer else None,
                "success": crm_reward > 0,
                # Include tracking metrics in output for transparency
                "metrics": {
                    "turns": conversation_turns,
                    "tokens_estimate": total_tokens_estimate,
                    "tool_calls": tool_calls_detected,
                    "queries": queries_detected,
                },
                # Timing breakdown
                "timing": {
                    "total_seconds": round(task_total_time, 2),
                    "purple_agent_seconds": round(purple_agent_time, 2),
                    "green_agent_seconds": round(green_agent_time, 2),
                    "purple_agent_percent": round(purple_agent_time/task_total_time*100, 1) if task_total_time > 0 else 0,
                }
            }
            if trajectory_file_path is not None:
                result["trajectory_file_path"] = str(trajectory_file_path)
                result["trajectory_event_count"] = len(trajectory_events)
            
            # Add original mode result if available
            if original_result is not None:
                result["original"] = original_result
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task.idx} evaluation failed: {e}")
            if effective_capture_trajectory and trajectory_events:
                trajectory_file_path = write_task_trajectory(
                    trajectory_root=trajectory_root,
                    task_id=str(task.idx),
                    events=trajectory_events,
                    label=trajectory_label,
                )
            return {
                "task_idx": task.idx,
                "task_category": task.task,
                "task_query": task.query,
                "dataset_reference": {
                    "source": "Salesforce/CRMArenaPro",
                    "split": "b2b",
                    "idx": task.idx,
                    "reward_metric": task.reward_metric,
                },
                "crm_reward": 0,
                "total_score": 0,
                "dimension_scores": {},
                "error": str(e),
                "success": False,
                "trajectory_file_path": str(trajectory_file_path) if trajectory_file_path else None,
                "trajectory_event_count": len(trajectory_events),
            }

    @staticmethod
    def _build_self_evolve_task_record(task: CRMTask) -> dict[str, Any]:
        return {
            "id": str(task.idx),
            "task_id": str(task.idx),
            "query": task.query,
            "task_category": task.task,
            "reward_metric": task.reward_metric,
            "persona": task.persona,
        }
    
    def _estimate_optimal_turns(self, task: CRMTask) -> int:
        """Estimate optimal number of turns based on task complexity."""
        # Simple tasks: 1 turn
        # Complex multi-hop: 2-3 turns
        simple_tasks = ["knowledge_qa", "named_entity_disambiguation", "lead_qualification"]
        complex_tasks = ["monthly_trend_analysis", "conversion_rate_comprehension", "handle_time"]
        
        if task.task in simple_tasks:
            return 1
        elif task.task in complex_tasks:
            return 3
        else:
            return 2
    
    def _parse_response_metrics(self, response: str) -> dict[str, int]:
        """
        Parse metrics from agent response.
        
        Agents can optionally include metrics in their response:
        {"metrics": {"tokens": 500, "tool_calls": 3, "queries": 2}}
        
        Otherwise, estimate from response characteristics.
        """
        metrics = {
            "tokens": 0,
            "tool_calls": 0,
            "invalid_tool_calls": 0,
            "queries": 0,
        }
        
        if not response:
            return metrics
        
        # Try to parse explicit metrics from response
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "metrics" in data:
                agent_metrics = data["metrics"]
                metrics["tokens"] = agent_metrics.get("tokens", 0)
                metrics["tool_calls"] = agent_metrics.get("tool_calls", 0)
                metrics["queries"] = agent_metrics.get("queries", 0)
                return metrics
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Estimate tokens from response length (~4 chars per token)
        metrics["tokens"] = len(response) // 4
        
        # Detect tool calls by common patterns
        tool_patterns = ["SELECT", "FROM", "WHERE", "query(", "search(", "get_", "find_"]
        for pattern in tool_patterns:
            metrics["tool_calls"] += response.upper().count(pattern)
        
        # Detect SQL queries
        sql_patterns = ["SELECT", "INSERT", "UPDATE", "DELETE"]
        for pattern in sql_patterns:
            metrics["queries"] += response.upper().count(pattern)
        
        return metrics
    
    def _check_needs_continuation(self, response: str) -> bool:
        """Check if agent response indicates it needs more turns."""
        if not response:
            return False
        
        # If agent gave a final answer, don't continue
        if "<respond>" in response.lower() or '"answer"' in response.lower():
            return False
        
        # Check for explicit continuation signals (be conservative!)
        continuation_signals = [
            "TOOL_CALL",
            "ACTION:",
            "<execute>",
            "<describe>",
        ]
        
        response_lower = response.lower()
        for signal in continuation_signals:
            if signal.lower() in response_lower:
                return True
        
        return False

    def _build_task_context(self, task: CRMTask, config: AssessmentConfig) -> dict[str, Any]:
        """Build task context with optional entropy transformations."""
        prompt = task.query
        required_context = task.get_required_context()
        
        # Apply Schema Drift - modify column names in prompt and context
        if config.drift_level != "none" and self.entropy_engine:
            prompt, drift_applied = self._apply_schema_drift(prompt, config.drift_level)
            required_context = self._apply_drift_to_context(required_context, config.drift_level)
            logger.info(f"Schema drift applied: {drift_applied} modifications")
        
        # Apply Context Rot - inject distractor information
        if config.rot_level != "none" and self.entropy_engine:
            required_context = self._apply_context_rot(required_context, config.rot_level)
            logger.info(f"Context rot applied at level: {config.rot_level}")
        
        context = {
            "type": "crm_task",
            "task_id": task.idx,
            "task_category": task.task,
            "prompt": prompt,
            "persona": task.persona,
            "required_context": required_context,
            "config": {
                "org_type": config.org_type,
                "max_steps": config.max_steps,
            }
        }
        
        # Add entropy metadata if active
        if config.drift_level != "none" or config.rot_level != "none":
            context["entropy"] = {
                "drift_level": config.drift_level,
                "rot_level": config.rot_level,
                "drift_mappings": self.entropy_engine.state.drift_mappings if self.entropy_engine else [],
                "note": "Schema/context has been modified for robustness testing"
            }
        
        return context
    
    def _apply_schema_drift(self, text: str, drift_level: str) -> tuple[str, int]:
        """Apply schema drift by replacing column names with synonyms."""
        if not text or drift_level == "none":
            return text, 0
        
        # Column name mappings based on drift level
        drift_mappings = {
            "low": {
                "Status": "CaseStatus",
                "OwnerId": "AssignedTo",
                "AccountId": "CustomerRef",
            },
            "medium": {
                "Status": "StatusCode",
                "OwnerId": "AssignedAgent",
                "AccountId": "ClientId",
                "ContactId": "PersonRef",
                "Subject": "Title",
                "Description": "Details",
            },
            "high": {
                "Status": "st_code",
                "OwnerId": "own_ref",
                "AccountId": "acct_id",
                "ContactId": "cont_ref",
                "Subject": "subj",
                "Description": "desc",
                "Priority": "pri_level",
                "CreatedDate": "create_dt",
                "CaseNumber": "ticket_num",
            },
        }
        
        mappings = drift_mappings.get(drift_level, {})
        modifications = 0
        
        for original, drifted in mappings.items():
            if original in text:
                text = text.replace(original, drifted)
                modifications += 1
        
        return text, modifications
    
    def _apply_drift_to_context(self, context: str, drift_level: str) -> str:
        """Apply schema drift to the required context."""
        if not context:
            return context
        drifted, _ = self._apply_schema_drift(context, drift_level)
        return drifted
    
    def _apply_context_rot(self, context: str, rot_level: str) -> str:
        """Inject distractor information into the context."""
        if not context or rot_level == "none":
            return context
        
        # Distractor templates based on rot level
        distractors = {
            "low": [
                "\n\n[Note: Some records may have been updated recently. Verify timestamps.]",
            ],
            "medium": [
                "\n\n[System Notice: Database migration in progress. Some field names may vary.]",
                "\n\n[Info: Legacy records from previous CRM system included for reference.]",
            ],
            "high": [
                "\n\n[Warning: Multiple customer records with similar names exist. Verify IDs carefully.]",
                "\n\n[Notice: Archived cases from 2019-2020 included. Filter by date if needed.]",
                "\n\n[Alert: Some account records are marked as duplicates pending merge.]",
            ],
        }
        
        import random
        rot_items = distractors.get(rot_level, [])
        if rot_items:
            random.seed(hash(context) % 2**32)  # Reproducible
            num_distractors = {"low": 1, "medium": 2, "high": 3}.get(rot_level, 1)
            selected = random.sample(rot_items, min(num_distractors, len(rot_items)))
            context = context + "".join(selected)
        
        return context

    def _extract_answer(self, response: str) -> str:
        """
        Extract the answer from agent response.
        
        Handles multiple response formats:
        1. Plain text answer
        2. JSON with "answer" field
        3. Text + JSON (common A2A artifact format)
        """
        if not response:
            return ""
        
        response = response.strip()
        
        # Check if response contains JSON (TextPart + DataPart concatenated)
        # Pattern: "Answer text\n{json...}"
        json_start = response.find('\n{')
        if json_start > 0:
            # Extract text before JSON as the primary answer
            text_part = response[:json_start].strip()
            json_part = response[json_start:].strip()
            
            # Try to get "answer" from JSON if it exists
            try:
                data = json.loads(json_part)
                if isinstance(data, dict) and "answer" in data:
                    # Use JSON answer if it matches text_part (validation)
                    json_answer = data["answer"]
                    if json_answer and isinstance(json_answer, str):
                        return json_answer.strip()
            except json.JSONDecodeError:
                pass
            
            # Return text part if JSON parsing fails
            return text_part
        
        # Try to parse entire response as JSON
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                # Priority: answer > response > text > first string value
                for key in ["answer", "response", "text", "result"]:
                    if key in data and data[key]:
                        return str(data[key]).strip()
                # Fallback to first string value
                for v in data.values():
                    if isinstance(v, str) and v:
                        return v.strip()
            return str(data)
        except json.JSONDecodeError:
            pass
        
        # Return as-is if no special handling needed
        return response

    def _drift_level_to_int(self, level: str) -> int:
        """Convert drift level string to int."""
        return {"none": 0, "low": 1, "medium": 2, "high": 3}.get(level, 0)

    def _rot_level_to_int(self, level: str) -> int:
        """Convert rot level string to int."""
        return {"none": 0, "low": 1, "medium": 2, "high": 3}.get(level, 0)

    def _create_aggregated_results(
        self,
        config: AssessmentConfig,
        purple_agent_id: str,
    ) -> dict[str, Any]:
        """Create aggregated results artifact (AgentBeats-compatible format)."""
        total_tasks = len(self.results)
        total_passed = sum(1 for r in self.results if r.get("crm_reward", 0) > 0)
        
        # Calculate averages (Entropic mode)
        avg_score = sum(r.get("total_score", 0) for r in self.results) / total_tasks if total_tasks > 0 else 0
        pass_rate = total_passed / total_tasks if total_tasks > 0 else 0
        
        # Dimension averages (Entropic mode)
        dimension_avgs = {}
        for result in self.results:
            for dim, score in result.get("dimension_scores", {}).items():
                if dim not in dimension_avgs:
                    dimension_avgs[dim] = []
                dimension_avgs[dim].append(score)
        
        dimension_averages = {
            dim: sum(scores) / len(scores) if scores else 0
            for dim, scores in dimension_avgs.items()
        }
        
        # Category breakdown (Entropic mode)
        by_category = {}
        for result in self.results:
            cat = result.get("task_category", "unknown")
            if cat not in by_category:
                by_category[cat] = {"count": 0, "passed": 0, "total_score": 0}
            by_category[cat]["count"] += 1
            by_category[cat]["passed"] += 1 if result.get("crm_reward", 0) > 0 else 0
            by_category[cat]["total_score"] += result.get("total_score", 0)
        
        for cat in by_category:
            count = by_category[cat]["count"]
            by_category[cat]["pass_rate"] = by_category[cat]["passed"] / count if count > 0 else 0
            by_category[cat]["avg_score"] = by_category[cat]["total_score"] / count if count > 0 else 0
        
        # Build base result with Entropic scores
        aggregated = {
            "participants": {
                "agent": purple_agent_id,
            },
            "results": self.results,
            # Entropic mode summary (primary)
            "entropic": {
                "summary": {
                    "pass_rate": round(pass_rate, 3),
                    "total_tasks": total_tasks,
                    "total_passed": total_passed,
                    "avg_score": round(avg_score, 1),
                },
                "dimension_averages": {k: round(v, 1) for k, v in dimension_averages.items()},
                "by_category": by_category,
            },
            # Legacy fields for backward compatibility
            "summary": {
                "pass_rate": round(pass_rate, 3),
                "total_tasks": total_tasks,
                "total_passed": total_passed,
                "avg_score": round(avg_score, 1),
            },
            "dimension_averages": {k: round(v, 1) for k, v in dimension_averages.items()},
            "by_category": by_category,
            "extension_metrics": {
                "drift_level": config.drift_level,
                "rot_level": config.rot_level,
                "org_type": config.org_type,
                "skip_original": config.skip_original,
            },
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",  # Updated version for dual-mode support
        }
        
        # Add Original mode scores if available
        if self.original_scorer is not None:
            original_scores = self.original_scorer.to_dict()
            aggregated["original"] = {
                "scores": original_scores.get("scores", {}),
                "summary": original_scores.get("summary", {}),
                "by_category": original_scores.get("by_category", {}),
                "by_metric_type": original_scores.get("by_metric_type", {}),
            }
            logger.info(f"Original mode accuracy: {original_scores.get('scores', {}).get('accuracy_percent', 0):.1f}%")
        
        return aggregated

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Run the CRMArena assessment.
        
        Main entry point called by the A2A executor.
        """
        started_at_utc = datetime.now(timezone.utc)
        assessment_start_time = time.time()
        logger.info("=" * 60)
        logger.info("[TIMING] ASSESSMENT STARTED")
        logger.info("=" * 60)
        
        input_text = get_message_text(message)

        # Parse and validate request
        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        # Parse config
        config = self._parse_config(request.config)
        
        # Get purple agent URL
        purple_agent_url = str(request.participants["agent"])
        
        # Initialize components
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Initializing CRMArena evaluation components...")
        )
        
        init_start = time.time()
        try:
            self._initialize_components(config)
        except Exception as e:
            await updater.reject(new_agent_text_message(f"Failed to initialize: {e}"))
            return
        init_time = time.time() - init_start
        logger.info(f"[TIMING] Component initialization: {init_time:.3f}s")
        
        # Get tasks
        task_load_start = time.time()
        tasks = self._get_tasks(config)
        total_tasks = len(tasks)
        task_load_time = time.time() - task_load_start
        logger.info(f"[TIMING] Task loading ({total_tasks} tasks): {task_load_time:.3f}s")
        
        if total_tasks == 0:
            await updater.reject(new_agent_text_message("No tasks found matching configuration"))
            return

        result_paths = build_execution_identity(
            benchmark_name=os.getenv("BENCHMARK_NAME") or _BENCHMARK_DIR.name,
            executor_name=str(
                request.config.get("executor")
                or os.getenv("BENCHMARK_EXECUTOR")
                or "baseline_crm_agent"
            ),
            request_config=request.config,
            participants={role: str(url) for role, url in request.participants.items()},
            result_root=Path(os.getenv("BENCHMARK_RESULT_ROOT")) if os.getenv("BENCHMARK_RESULT_ROOT") else None,
            run_id=os.getenv("BENCHMARK_RUN_ID"),
            config_hash=os.getenv("BENCHMARK_CONFIG_HASH"),
            created_at_utc=started_at_utc,
        )
        ensure_result_dir(result_paths)
        capture_trajectory = capture_trajectory_enabled(request.config)
        trajectory_root = (
            trajectory_root_for_result(result_paths.result_dir)
            if capture_trajectory
            else None
        )
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting assessment with {total_tasks} tasks. "
                f"Drift: {config.drift_level}, Rot: {config.rot_level}"
            )
        )
        
        # Run evaluation for each task
        self.results = []
        total_purple_time = 0.0
        total_green_time = 0.0
        self_evolution_detail: dict[str, Any] | None = None
        benchmark_self_evolution_detail: dict[str, Any] | None = None
        request_config = dict(request.config)
        max_benchmark_cycles = resolve_max_benchmark_self_evolution_cycles(request_config)
        max_self_evolutions = resolve_max_self_evolutions(request_config)
        aggregated: dict[str, Any] | None = None
        
        # Reset original scorer for fresh run
        if self.original_scorer is not None:
            self.original_scorer.reset()

        async def _evaluate_phase_task(
            *,
            task: CRMTask,
            phase_label: str | None = None,
            session=None,
        ) -> tuple[dict[str, Any], str, dict[str, Any]]:
            task_context = self._build_task_context(task, config)
            goal_text = json.dumps(task_context, ensure_ascii=False)
            request_text = goal_text
            task_record = self._build_self_evolve_task_record(task)
            if session is not None:
                request_text = session.build_request_text(
                    goal_text=goal_text,
                    task_id=str(task.idx),
                    task=task_record,
                )
            result = await self._evaluate_single_task(
                task,
                purple_agent_url,
                config,
                updater,
                capture_trajectory=capture_trajectory,
                trajectory_root=trajectory_root,
                request_text=request_text,
                request_config=request_config,
                trajectory_label=phase_label,
            )
            return result, goal_text, task_record

        if max_benchmark_cycles > 0:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    "Starting benchmark-cycle self-evolve evaluation "
                    f"(tasks={total_tasks}, max_cycles={max_benchmark_cycles})"
                )
            )
            session = build_benchmark_self_evolve_session_from_adapter(
                SELF_EVOLVE_ADAPTER,
                request_config=request_config,
                result_dir=result_paths.result_dir,
            )
            cycle_history: list[dict[str, Any]] = []
            baseline_total_score: float | None = None
            stop_reason = "max_cycles_reached"

            for pass_index in range(max_benchmark_cycles + 1):
                if self.original_scorer is not None:
                    self.original_scorer.reset()
                self.results = []
                session.reset_records()
                generation_before_run = session.evolutions_applied
                phase_label = f"pass {pass_index + 1}"
                phase_purple_time = 0.0
                phase_green_time = 0.0

                for task_index, task in enumerate(tasks):
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            f"[{phase_label}] [{task_index + 1}/{total_tasks}] "
                            f"Evaluating task {task.idx} ({task.task})"
                        )
                    )
                    result, goal_text, task_record = await _evaluate_phase_task(
                        task=task,
                        phase_label=phase_label,
                        session=session,
                    )
                    self.results.append(result)
                    session.record_task(
                        task=task_record,
                        goal_text=goal_text,
                        predicted=result.get("agent_answer"),
                        score=float(result.get("total_score") or 0.0),
                        reason=None,
                        error=result.get("error"),
                    )
                    if "timing" in result:
                        phase_purple_time += result["timing"].get("purple_agent_seconds", 0)
                        phase_green_time += result["timing"].get("green_agent_seconds", 0)
                    task_status = "✓" if result.get("crm_reward", 0) > 0 else "✗"
                    timing_info = ""
                    if "timing" in result:
                        timing_info = f" ({result['timing']['total_seconds']:.1f}s)"
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            f"[{phase_label}] [{task_index + 1}/{total_tasks}] {task_status} "
                            f"Task {task.idx}: score={result.get('total_score', 0):.1f}{timing_info}"
                        )
                    )

                aggregated = self._create_aggregated_results(config, purple_agent_url)
                total_purple_time = phase_purple_time
                total_green_time = phase_green_time
                current_total_score = float(
                    aggregated["summary"].get("avg_score", 0.0) or 0.0
                )
                current_score_rate = float(
                    aggregated["summary"].get("pass_rate", 0.0) or 0.0
                )
                if baseline_total_score is None:
                    baseline_total_score = current_total_score

                cycle_entry: dict[str, Any] = {
                    "pass_index": pass_index,
                    "phase_label": phase_label,
                    "pack_generation_before_run": generation_before_run,
                    "pack_generation_after_run": session.evolutions_applied,
                    "total_score": current_total_score,
                    "score_rate": current_score_rate,
                    "score_rate_percent": f"{current_score_rate:.2%}",
                    "task_results": list(self.results),
                    "details": list(self.results),
                }

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

                evolution_event = session.maybe_evolve(after_task_id=str(tasks[-1].idx))
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
                            f"[{phase_label}] benchmark self-evolve "
                            f"{evolution_event['generation']} completed. "
                            f"Strategies: {', '.join(evolution_event.get('strategies', [])) or 'none'}"
                        )
                    )
                    continue

                stop_reason = "evolution_failed"
                cycle_entry["stop_reason"] = stop_reason
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"[{phase_label}] benchmark self-evolve "
                        f"{evolution_event['generation']} failed: "
                        f"{evolution_event.get('error', 'unknown error')}"
                    )
                )
                break

            aggregated = aggregated or self._create_aggregated_results(config, purple_agent_url)
            benchmark_self_evolution_detail = build_benchmark_self_evolution_detail(
                session=session,
                cycle_history=cycle_history,
                baseline_total_score=baseline_total_score,
                final_total_score=float(aggregated["summary"].get("avg_score", 0.0) or 0.0),
                task_count=len(tasks),
                final_details=list(self.results),
                stop_reason=stop_reason,
            )
        elif max_self_evolutions > 0:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    "Starting baseline + self-evolve evaluation "
                    f"(tasks={total_tasks}, max_self_evolutions={max_self_evolutions})"
                )
            )
            harness = build_self_evolve_harness_from_adapter(
                SELF_EVOLVE_ADAPTER,
                request_config=request_config,
                result_dir=result_paths.result_dir,
            )

            if self.original_scorer is not None:
                self.original_scorer.reset()
            baseline_results: list[dict[str, Any]] = []
            baseline_purple_time = 0.0
            baseline_green_time = 0.0
            for task_index, task in enumerate(tasks):
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"[baseline] [{task_index + 1}/{total_tasks}] Evaluating task {task.idx} ({task.task})"
                    )
                )
                result, _, _ = await _evaluate_phase_task(
                    task=task,
                    phase_label="baseline",
                    session=harness.baseline_session,
                )
                baseline_results.append(result)
                if "timing" in result:
                    baseline_purple_time += result["timing"].get("purple_agent_seconds", 0)
                    baseline_green_time += result["timing"].get("green_agent_seconds", 0)
            self.results = baseline_results
            baseline_aggregated = self._create_aggregated_results(config, purple_agent_url)
            baseline_total_score = float(
                baseline_aggregated["summary"].get("avg_score", 0.0) or 0.0
            )
            baseline_score_rate = float(
                baseline_aggregated["summary"].get("pass_rate", 0.0) or 0.0
            )
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    "[baseline] completed. "
                    f"Score: {baseline_total_score}, Score Rate: {baseline_score_rate:.2%}"
                )
            )

            if self.original_scorer is not None:
                self.original_scorer.reset()
            evolved_results: list[dict[str, Any]] = []
            evolved_purple_time = 0.0
            evolved_green_time = 0.0
            for task_index, task in enumerate(tasks):
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"[evolved] [{task_index + 1}/{total_tasks}] Evaluating task {task.idx} ({task.task})"
                    )
                )
                result, goal_text, task_record = await _evaluate_phase_task(
                    task=task,
                    phase_label="evolved",
                    session=harness.evolved_session,
                )
                evolved_results.append(result)
                harness.evolved_session.record_task(
                    task=task_record,
                    goal_text=goal_text,
                    predicted=result.get("agent_answer"),
                    score=float(result.get("total_score") or 0.0),
                    reason=None,
                    error=result.get("error"),
                )
                if "timing" in result:
                    evolved_purple_time += result["timing"].get("purple_agent_seconds", 0)
                    evolved_green_time += result["timing"].get("green_agent_seconds", 0)
                if task_index < len(tasks) - 1:
                    evolution_event = harness.evolved_session.maybe_evolve(
                        after_task_id=str(task.idx)
                    )
                    if evolution_event:
                        if evolution_event.get("status") == "completed":
                            await updater.update_status(
                                TaskState.working,
                                new_agent_text_message(
                                    "[evolved] inter test-time self-evolve "
                                    f"{evolution_event['generation']} completed after {task.idx}. "
                                    f"Strategies: {', '.join(evolution_event.get('strategies', [])) or 'none'}"
                                )
                            )
                        else:
                            await updater.update_status(
                                TaskState.working,
                                new_agent_text_message(
                                    "[evolved] inter test-time self-evolve "
                                    f"{evolution_event['generation']} failed after {task.idx}: "
                                    f"{evolution_event.get('error', 'unknown error')}"
                                )
                            )

            self.results = evolved_results
            aggregated = self._create_aggregated_results(config, purple_agent_url)
            total_purple_time = evolved_purple_time
            total_green_time = evolved_green_time
            final_total_score = float(aggregated["summary"].get("avg_score", 0.0) or 0.0)
            final_score_rate = float(aggregated["summary"].get("pass_rate", 0.0) or 0.0)
            self_evolution_detail = build_inter_task_self_evolution_detail(
                harness=harness,
                baseline_total_score=baseline_total_score,
                baseline_run={
                    "score_rate": baseline_score_rate,
                    "task_results": list(baseline_results),
                    "detail_records": list(baseline_results),
                },
                evolved_total_score=final_total_score,
                evolved_run={
                    "score_rate": final_score_rate,
                    "task_results": list(evolved_results),
                    "detail_records": list(evolved_results),
                },
            )
        else:
            max_parallel = max(1, int((request_config or {}).get("max_parallel") or 1))

            async def _run_one(index: int, task: Any) -> tuple[int, dict[str, Any]]:
                """Evaluate one task and emit its start/finish status lines."""
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"[{index+1}/{total_tasks}] Evaluating task {task.idx} ({task.task})"
                    )
                )
                result = await self._evaluate_single_task(
                    task,
                    purple_agent_url,
                    config,
                    updater,
                    capture_trajectory=capture_trajectory,
                    trajectory_root=trajectory_root,
                    request_config=request_config,
                )
                task_status = "✓" if result.get("crm_reward", 0) > 0 else "✗"
                timing_info = ""
                if "timing" in result:
                    timing_info = f" ({result['timing']['total_seconds']:.1f}s)"
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"[{index+1}/{total_tasks}] {task_status} Task {task.idx}: "
                        f"score={result.get('total_score', 0):.1f}{timing_info}"
                    )
                )
                return index, result

            if max_parallel > 1:
                # Green opens a fresh A2A context per task and Purple builds one
                # world model per context, so tasks do not share agent or WM state.
                logger.info(
                    f"Evaluating {total_tasks} tasks with max_parallel={max_parallel}"
                )
                semaphore = asyncio.Semaphore(max_parallel)

                async def _bounded(index: int, task: Any) -> tuple[int, dict[str, Any]]:
                    async with semaphore:
                        return await _run_one(index, task)

                gathered = await asyncio.gather(
                    *(_bounded(i, task) for i, task in enumerate(tasks))
                )
                # Restore dataset order: completion order is nondeterministic under
                # concurrency, and the aggregation/detail records must stay aligned.
                ordered = [result for _, result in sorted(gathered, key=lambda pair: pair[0])]
            else:
                ordered = []
                for i, task in enumerate(tasks):
                    _, result = await _run_one(i, task)
                    ordered.append(result)

            for result in ordered:
                self.results.append(result)
                if "timing" in result:
                    total_purple_time += result["timing"].get("purple_agent_seconds", 0)
                    total_green_time += result["timing"].get("green_agent_seconds", 0)

            aggregated = self._create_aggregated_results(config, purple_agent_url)
        
        # Calculate total time
        assessment_total_time = time.time() - assessment_start_time
        overhead_time = assessment_total_time - total_purple_time - total_green_time
        
        # Log final timing summary
        logger.info("=" * 60)
        logger.info("[TIMING] ASSESSMENT COMPLETE - SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Total tasks:        {total_tasks}")
        logger.info(f"  Total time:         {assessment_total_time:.2f}s")
        logger.info(f"  ├─ Purple Agent:    {total_purple_time:.2f}s ({total_purple_time/assessment_total_time*100:.1f}%)")
        logger.info(f"  ├─ Green Agent:     {total_green_time:.2f}s ({total_green_time/assessment_total_time*100:.1f}%)")
        logger.info(f"  └─ Overhead:        {overhead_time:.2f}s ({overhead_time/assessment_total_time*100:.1f}%)")
        logger.info(f"  Avg time per task:  {assessment_total_time/total_tasks:.2f}s")
        logger.info("=" * 60)
        
        # Add timing to aggregated results
        aggregated["timing"] = {
            "total_seconds": round(assessment_total_time, 2),
            "purple_agent_seconds": round(total_purple_time, 2),
            "green_agent_seconds": round(total_green_time, 2),
            "avg_seconds_per_task": round(assessment_total_time / total_tasks, 2),
            "purple_agent_percent": round(total_purple_time/assessment_total_time*100, 1) if assessment_total_time > 0 else 0,
        }
        if self_evolution_detail is not None:
            aggregated["self_evolution"] = self_evolution_detail
        if benchmark_self_evolution_detail is not None:
            aggregated["benchmark_self_evolution"] = benchmark_self_evolution_detail
        
        # Calculate summary text with both scoring modes
        summary_text = (
            f"Assessment Complete\n"
            f"==================\n"
            f"Tasks: {aggregated['summary']['total_tasks']}\n"
        )
        
        # Add Original mode scores if available
        if "original" in aggregated:
            original_acc = aggregated["original"]["scores"].get("accuracy_percent", 0)
            summary_text += (
                f"\n--- Original CRMArena-Pro Mode ---\n"
                f"Accuracy: {original_acc:.1f}%\n"
            )
        
        # Add Entropic mode scores
        summary_text += (
            f"\n--- Entropic Mode (7D Scoring) ---\n"
            f"Passed: {aggregated['summary']['total_passed']}\n"
            f"Pass Rate: {aggregated['summary']['pass_rate']:.1%}\n"
            f"Avg Score: {aggregated['summary']['avg_score']:.1f}\n"
            f"\nTiming:\n"
            f"  Total: {assessment_total_time:.1f}s\n"
            f"  Purple Agent: {total_purple_time:.1f}s ({total_purple_time/assessment_total_time*100:.0f}%)\n"
            f"  Green Agent: {total_green_time:.1f}s ({total_green_time/assessment_total_time*100:.0f}%)\n"
            f"\nDimension Averages:\n"
        )
        for dim, score in aggregated.get("dimension_averages", {}).items():
            summary_text += f"  {dim}: {score:.1f}\n"

        completed_at_utc = datetime.now(timezone.utc)
        _write_benchmark_artifacts(
            request=request,
            aggregated=aggregated,
            started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
        )
        
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary_text)),
                Part(root=DataPart(data=aggregated))
            ],
            name="CRMArena Assessment Results",
        )
