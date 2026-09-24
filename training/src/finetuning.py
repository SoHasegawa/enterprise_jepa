#!/usr/bin/env python3
"""Fine-tune an EnterpriseOps-Gym world model from reconstructed state trajectories.

Each supervised example uses the system prompt, user prompt, previous state,
and action as input, then predicts the resulting state or selected outcome
field. Runtime evaluation lives in `src/evaluation.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import inspect
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from src.data_preparation.canonical_event_state import (
    NUDGE_CATEGORICAL_FIELDS,
    NUDGE_LIST_FIELDS,
    REQUIRED_CATEGORICAL_FIELDS as CANONICAL_EVENT_REQUIRED_CATEGORICAL_FIELDS,
    canonical_event_from_action_state,
    canonical_event_json,
    canonical_event_with_nudge_from_action_state,
    canonical_event_with_nudge_json,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps --help usable in minimal envs.
    def tqdm(iterable, **kwargs):
        return iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORIES_DIR = REPO_ROOT / "trajectories"
ENTERPRISEOPS_GYM_TRAIN_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "enterpriseops_gym_multi_model_world_model_train_trajectories.json"
)
ENTERPRISEOPS_GYM_EVAL_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "enterpriseops_gym_multi_model_world_model_test_trajectories.json"
)
ENTERPRISEOPS_GYM_ENTERPRISE_STATE_TRAIN_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "enterpriseops_gym_multi_model_world_model_enterprise_state_train_trajectories.json"
)
ENTERPRISEOPS_GYM_ENTERPRISE_STATE_EVAL_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "enterpriseops_gym_multi_model_world_model_enterprise_state_test_trajectories.json"
)
TERMINALBENCH_2_0_TRAIN_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "terminalbench_2_0_multi_model_world_model_train_trajectories.json"
)
TERMINALBENCH_2_0_EVAL_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "terminalbench_2_0_multi_model_world_model_test_trajectories.json"
)
CRMARENAPRO_TRAIN_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "crmarenapro_multi_model_world_model_train_trajectories.json"
)
CRMARENAPRO_EVAL_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "crmarenapro_multi_model_world_model_test_trajectories.json"
)
DEFAULT_TRAIN_DATA_PATH = ENTERPRISEOPS_GYM_TRAIN_DATA_PATH
DEFAULT_EVAL_DATA_PATH = ENTERPRISEOPS_GYM_EVAL_DATA_PATH
DEFAULT_ENTERPRISEOPS_GYM_TASK_SPLIT_MANIFEST = (
    DEFAULT_TRAJECTORIES_DIR / "enterpriseops_gym_80_test_task_split.json"
)
DEFAULT_ENTERPRISEOPS_GYM_REPO_PATH = REPO_ROOT / "EnterpriseOps-Gym"
DEFAULT_ENTERPRISEOPS_GYM_TASK_CONFIGS_DIR = (
    DEFAULT_TRAJECTORIES_DIR / "enterpriseops_gym_task_configs"
)
TRAJECTORY_DATASET_PRESETS: dict[str, tuple[list[Path], list[Path]]] = {
    "enterpriseops_gym": (
        [ENTERPRISEOPS_GYM_TRAIN_DATA_PATH],
        [ENTERPRISEOPS_GYM_EVAL_DATA_PATH],
    ),
    "enterpriseops_gym_enterprise_state": (
        [ENTERPRISEOPS_GYM_ENTERPRISE_STATE_TRAIN_DATA_PATH],
        [ENTERPRISEOPS_GYM_ENTERPRISE_STATE_EVAL_DATA_PATH],
    ),
    "terminalbench_2_0": (
        [TERMINALBENCH_2_0_TRAIN_DATA_PATH],
        [TERMINALBENCH_2_0_EVAL_DATA_PATH],
    ),
    "crmarenapro": (
        [CRMARENAPRO_TRAIN_DATA_PATH],
        [CRMARENAPRO_EVAL_DATA_PATH],
    ),
    "enterpriseops_gym_terminalbench_2_0": (
        [ENTERPRISEOPS_GYM_TRAIN_DATA_PATH, TERMINALBENCH_2_0_TRAIN_DATA_PATH],
        [ENTERPRISEOPS_GYM_EVAL_DATA_PATH, TERMINALBENCH_2_0_EVAL_DATA_PATH],
    ),
    "enterpriseops_gym_crmarenapro": (
        [ENTERPRISEOPS_GYM_TRAIN_DATA_PATH, CRMARENAPRO_TRAIN_DATA_PATH],
        [ENTERPRISEOPS_GYM_EVAL_DATA_PATH, CRMARENAPRO_EVAL_DATA_PATH],
    ),
    "terminalbench_2_0_crmarenapro": (
        [TERMINALBENCH_2_0_TRAIN_DATA_PATH, CRMARENAPRO_TRAIN_DATA_PATH],
        [TERMINALBENCH_2_0_EVAL_DATA_PATH, CRMARENAPRO_EVAL_DATA_PATH],
    ),
    "enterpriseops_gym_terminalbench_2_0_crmarenapro": (
        [
            ENTERPRISEOPS_GYM_TRAIN_DATA_PATH,
            TERMINALBENCH_2_0_TRAIN_DATA_PATH,
            CRMARENAPRO_TRAIN_DATA_PATH,
        ],
        [
            ENTERPRISEOPS_GYM_EVAL_DATA_PATH,
            TERMINALBENCH_2_0_EVAL_DATA_PATH,
            CRMARENAPRO_EVAL_DATA_PATH,
        ],
    ),
}
CSM_MCP_SERVER_NAME = "sn-csm-server"
CSM_MCP_DEFAULT_URL = "http://localhost:8001"
CSM_MCP_OVERRIDE_URL = "http://localhost:8010"
TOOL_CALL_OPEN_TAG = "<tool_call>"
TOOL_CALL_CLOSE_TAG = "</tool_call>"
PROMPT_SECTION_SYSTEM = "System prompt:\n"
PROMPT_SECTION_USER = "\n\nUser prompt:\n"
PROMPT_SECTION_PREVIOUS_STATE = "\n\nPrevious state:\n"
PROMPT_SECTION_ACTION = "\n\nAction:\n"
MISSING_ARGUMENT_ROOT = "<root>"
WORLD_MODEL_OUTPUT_SEPARATOR = "\n---\n"
IMAGINED_TOOL_OUTPUT_HEADER = "[IMAGINED_TOOL_OUTPUT_FROM_WORLD_MODEL]\n"
IMAGINED_OBSERVATION_HEADER = "Imagined observation based on world-model prediction:\n"
TASK_COMPLETION_JUDGE_KEYS = (
    "requirement_coverage",
    "accuracy",
    "completeness",
    "usefulness",
)
STATE_MATCH_JUDGE_KEYS = (
    "field_accuracy",
)
OUTCOME_ERROR_MATCH_JUDGE_KEYS = (
    "label_alignment",
    "error_alignment",
    "semantic_consistency",
)
TOOL_OUTPUT_MATCH_JUDGE_KEYS = (
    "semantic_equivalence",
    "factual_consistency",
    "intent_alignment",
    "outcome_alignment",
)
TOOL_OUTPUT_JUDGE_CHAR_LIMIT = 4000
OUTCOME_LABELS = (-1, 0, 1)
UNKNOWN_OUTCOME_LABEL = -99
DEFAULT_STATE_HISTORY_SIZE = 3
WORLD_MODEL_INPUT_HISTORY_SIZE = 8
WORLD_MODEL_HISTORY_OBSERVATION_CHARS = 200
WORLD_MODEL_TARGET_STATE = "state"
WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY = "tool_execution_result_ternary"
WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY = "tool_execution_result_binary"
WORLD_MODEL_TARGET_TOOL_OUTPUT = "tool_output"
WORLD_MODEL_TARGET_CANONICAL_EVENT_STATE = "canonical_event_state"
WORLD_MODEL_TARGET_CANONICAL_EVENT_WITH_NUDGE = "canonical_event_with_nudge"
LEGACY_WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY = "tool_execution_result_success_failure"


def canonicalize_world_model_target(target_mode: str) -> str:
    if target_mode == LEGACY_WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY:
        return WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY
    return target_mode


def is_tool_execution_result_target(target_mode: str) -> bool:
    return canonicalize_world_model_target(target_mode) in {
        WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
        WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY,
    }


def is_tool_output_target(target_mode: str) -> bool:
    return canonicalize_world_model_target(target_mode) == WORLD_MODEL_TARGET_TOOL_OUTPUT


def is_canonical_event_state_target(target_mode: str) -> bool:
    return canonicalize_world_model_target(target_mode) == WORLD_MODEL_TARGET_CANONICAL_EVENT_STATE


def is_canonical_event_with_nudge_target(target_mode: str) -> bool:
    return canonicalize_world_model_target(target_mode) == WORLD_MODEL_TARGET_CANONICAL_EVENT_WITH_NUDGE


def tool_output_looks_like_failure(text: str) -> bool:
    """Heuristic check for whether a predicted `last_tool_output` denotes a failed call.

    Used by the agent-replay loop when the world model only emits raw tool-output
    text (target_mode == `tool_output`). Conservative: only flags failure on strong
    markers, so genuine API responses that happen to mention error-adjacent
    keywords pass through as success.
    """
    if text is None:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    startswith_markers = (
        "error:",
        "error code",
        "exception:",
        "traceback",
        "fatal:",
        "failed:",
        "failure:",
        "mcperror",
    )
    if any(lowered.startswith(marker) for marker in startswith_markers):
        return True
    exact_markers = {
        "invalid_product_id",
        "invalid_account_id",
        "invalid_contract_id",
        "invalid_entitlement_id",
        "invalid_case_id",
        "invalid_user_id",
        "invalid_contact_id",
        "invalid_location_id",
    }
    if lowered in exact_markers:
        return True
    head = lowered[:300]
    inline_markers = (
        "api error",
        "validation error",
        "invalid_",
        "not found",
        "cannot be null",
        "must not be null",
        "unable to parse",
        "input should be a valid",
        "tool '",
        '"error":',
        "'error':",
        '"status": "error"',
        "status_code: 4",
        "status_code: 5",
    )
    if any(marker in head for marker in inline_markers):
        return True
    return False


def tool_execution_result_label_values(target_mode: str) -> tuple[int, ...]:
    if canonicalize_world_model_target(target_mode) == WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY:
        return (0, 1)
    return OUTCOME_LABELS


@dataclass
class WorldModelExample:
    trajectory_index: int
    interaction_index: int
    system_prompt: str
    task_query: str
    context_messages: list[dict[str, Any]]
    tool_call: dict[str, Any]
    tool_name: str
    tool_arguments: Any
    tool_response_name: str
    tool_response_content: str
    tool_success: bool
    tool_error_message: str


@dataclass
class WorldModelStateExample:
    trajectory_id: str
    trajectory_index: int
    interaction_index: int
    system_prompt: str
    user_prompt: str
    action: Any
    state_history: list[dict[str, Any]]
    input_history: list[dict[str, Any]]
    previous_state: dict[str, Any] | None
    state: dict[str, Any]
    error_payload: str = ""
    tool_output: str = ""
    # Optional pre-computed target label (e.g. an LLM-labeled
    # canonical_event_with_nudge payload) carried through from a materialized
    # examples file so evaluation scores against the same gold used for training
    # instead of re-deriving it from `action`/`state`.
    canonical_label: dict[str, Any] | None = None


@dataclass
class TrajectoryOutcomeStats:
    trajectory: dict[str, Any]
    source_path: str
    source_index: int
    success_count: int
    failure_count: int
    example_count: int


@dataclass
class TaskStep:
    interaction_index: int
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]


@dataclass
class TaskTrajectory:
    trajectory_index: int
    system_prompt: str
    user_messages: list[str]
    steps: list[TaskStep]
    final_answer: str
    gym_task_config_name: str | None = None
    initial_state: dict[str, Any] | None = None
    source_path: str | None = None


@dataclass
class SupervisedDataCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []
        outcome_labels: list[int] = []
        forward_outcome_label = all("outcome_label" in feature for feature in features)
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
            if forward_outcome_label:
                outcome_labels.append(int(feature["outcome_label"]))

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if forward_outcome_label:
            batch["outcome_label"] = torch.tensor(outcome_labels, dtype=torch.long)
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an EWM world model from benchmark trajectories.")
    parser.add_argument("--model", nargs="?", default="Qwen/Qwen3-4B", help="Base Hugging Face model for SFT.")
    parser.add_argument(
        "--trajectory-dataset",
        choices=sorted(TRAJECTORY_DATASET_PRESETS.keys()),
        default="enterpriseops_gym",
        help=(
            "Trajectory dataset preset. `enterpriseops_gym` (default) uses "
            "EnterpriseOps-Gym world-model trajectories; `terminalbench_2_0` and "
            "`crmarenapro` use materialized world-model trajectories with the same "
            "system/user/action/state message format; "
            "`enterpriseops_gym_terminalbench_2_0_crmarenapro` trains on all three. Explicit "
            "`--train-data-path` / `--eval-data-path` override the preset."
        ),
    )
    parser.add_argument("--train-data-path", type=Path, nargs="+", default=None)
    parser.add_argument("--eval-data-path", type=Path, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default="data", help="Directory for checkpoints and training artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for training reproducibility.")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="SFT max sequence length.")
    parser.add_argument("--min-completion-tokens", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=float, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-fast-mamba-kernels", action="store_true")
    parser.add_argument("--debug-one-batch", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--disable-chat-template", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-bias", default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--lora-target-modules", default="auto")
    parser.add_argument("--lora-modules-to-save", default="")
    parser.add_argument("--state-history-size", type=int, default=DEFAULT_STATE_HISTORY_SIZE)
    parser.add_argument("--include-world-model-history", action="store_true")
    parser.add_argument("--world-model-system-prompt-max-chars", type=int, default=0)
    parser.add_argument("--world-model-action-max-chars", type=int, default=0)
    parser.add_argument(
        "--world-model-target",
        default=WORLD_MODEL_TARGET_STATE,
        choices=[
            WORLD_MODEL_TARGET_STATE,
            WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY,
            WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
            WORLD_MODEL_TARGET_TOOL_OUTPUT,
            WORLD_MODEL_TARGET_CANONICAL_EVENT_STATE,
            WORLD_MODEL_TARGET_CANONICAL_EVENT_WITH_NUDGE,
            LEGACY_WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
        ],
    )
    parser.add_argument("--oversample-minority-outcomes", action="store_true")
    parser.add_argument("--oversample-target-ratio", type=float, default=0.5)
    parser.add_argument("--oversample-max-multiplier", type=float, default=10.0)
    parser.add_argument("--include-error-message-in-target", action="store_true")
    parser.add_argument("--include-stage-in-target", action="store_true")
    parser.add_argument(
        "--outcome-balance-loss",
        default="effective_num_loss",
        choices=["none", "inverse_frequency_loss", "effective_num_loss"],
    )
    parser.add_argument("--outcome-balance-beta", type=float, default=0.999)
    args = parser.parse_args()
    args.world_model_target = canonicalize_world_model_target(args.world_model_target)
    preset_train, preset_eval = TRAJECTORY_DATASET_PRESETS[args.trajectory_dataset]
    if args.train_data_path is None:
        args.train_data_path = list(preset_train)
    elif isinstance(args.train_data_path, Path):
        args.train_data_path = [args.train_data_path]
    if args.eval_data_path is None:
        args.eval_data_path = list(preset_eval)
    elif isinstance(args.eval_data_path, Path):
        args.eval_data_path = [args.eval_data_path]
    return args


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


CRM_TOOL_TAG_PATTERN = re.compile(r"<(execute|describe)>\s*(.*?)\s*</\1>", re.DOTALL | re.IGNORECASE)
ENTERPRISE_STATE_OMITTED_FIELDS = (
    "objects_artifacts",
    "relational_state",
    "constraints",
)


def state_body_from_any(raw_state: Any) -> dict[str, Any]:
    if isinstance(raw_state, str):
        try:
            raw_state = parse_jsonish(raw_state)
        except Exception:
            return {}
    if not isinstance(raw_state, dict):
        return {}
    body = raw_state.get("state")
    return body if isinstance(body, dict) else raw_state


def enterprise_state_delta_from_any(raw_state: Any) -> dict[str, Any]:
    body = state_body_from_any(raw_state)
    delta = body.get("diff_from_previous_state")
    return delta if isinstance(delta, dict) else body


def last_enterprise_tool_event(raw_state: Any) -> dict[str, Any]:
    for container in (enterprise_state_delta_from_any(raw_state), state_body_from_any(raw_state)):
        history = container.get("history_context")
        if not isinstance(history, dict):
            continue
        events = history.get("last_tool_events")
        if isinstance(events, list) and events and isinstance(events[-1], dict):
            return events[-1]
    return {}


def enterprise_status_to_label(status: Any) -> int | None:
    if not isinstance(status, str):
        return None
    normalized = status.strip().lower()
    if normalized in {"success", "succeeded", "ok", "completed"}:
        return 1
    if normalized in {"stagnation", "blocked", "partial"}:
        return 0
    if normalized in {"failure", "failed", "error", "exception"}:
        return -1
    return None


def infer_enterprise_outcome_label(outcome: dict[str, Any], event: dict[str, Any]) -> int | None:
    label = enterprise_status_to_label(outcome.get("status"))
    if label is not None:
        return label
    label = enterprise_status_to_label(event.get("status"))
    if label is not None:
        return label
    failure_category = str(outcome.get("failure_category") or "").strip().lower()
    if failure_category and failure_category not in {"none", "unknown", "n/a"}:
        return -1
    summary = str(outcome.get("summary") or event.get("summary") or "").strip().lower()
    if any(marker in summary for marker in ("successfully", "created", "updated", "retrieved", "listed", "sent")):
        return 1
    if any(marker in summary for marker in ("failed", "error", "violat", "blocked", "abort")):
        return -1
    return None


def state_context_from_any(raw_state: Any) -> dict[str, Any]:
    body = state_body_from_any(raw_state)
    if not body:
        return {}

    context = body.get("context")
    if isinstance(context, dict):
        return context

    # Compact tool-result states store tool metadata at the root.
    if any(key in body for key in ("last_tool_execution_result", "last_tool_name", "last_tool_output")):
        return body

    delta = enterprise_state_delta_from_any(raw_state)
    outcome = delta.get("outcome") if isinstance(delta.get("outcome"), dict) else body.get("outcome")
    if not isinstance(outcome, dict):
        outcome = {}
    event = last_enterprise_tool_event(raw_state)
    label = infer_enterprise_outcome_label(outcome, event)

    summary = outcome.get("summary") or event.get("summary") or ""
    error = event.get("error") or (summary if label in {-1, 0} else "")
    synthesized: dict[str, Any] = {
        "last_tool_execution_result": label,
        "last_tool_name": event.get("tool_name"),
        "last_tool_output": summary,
        "error_message": error,
    }
    return {key: value for key, value in synthesized.items() if value is not None}


def state_process_from_any(raw_state: Any) -> dict[str, Any]:
    body = state_body_from_any(raw_state)
    if not body:
        return {}
    process = body.get("process")
    if isinstance(process, dict):
        return process

    if any(key in body for key in ("current_stage", "remaining_stages", "completed_stages")):
        return {
            "current_stage": body.get("current_stage"),
            "remaining_stages": body.get("remaining_stages"),
            "completed_stages": body.get("completed_stages"),
        }

    delta = enterprise_state_delta_from_any(raw_state)
    process_state = delta.get("process_state") if isinstance(delta.get("process_state"), dict) else body.get("process_state")
    if not isinstance(process_state, dict):
        return {}
    return {
        "current_stage": process_state.get("stage"),
        "remaining_stages": process_state.get("remaining_requirements"),
        "completed_stages": process_state.get("completed_requirements"),
    }


def state_current_stage(raw_state: Any) -> Any:
    return state_process_from_any(raw_state).get("current_stage")


def state_remaining_stages(raw_state: Any) -> Any:
    return state_process_from_any(raw_state).get("remaining_stages")


def state_is_finished(raw_state: Any) -> bool:
    stage = state_current_stage(raw_state)
    return isinstance(stage, str) and stage.strip().lower() == "finished"


def is_enterprise_state_payload(raw_state: Any) -> bool:
    body = state_body_from_any(raw_state)
    schema = body.get("schema")
    return (
        schema == "enterprise_ops_objects_process_relational_constraints_history_v1"
        or "process_state" in body
        or "diff_from_previous_state" in body
    )




def load_jsonl(path: Path) -> list[Any]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return records


def load_trajectory_records(path: Path) -> list[dict[str, Any]]:
    loaded = load_jsonl(path) if path.suffix == ".jsonl" else load_json(path)
    if not isinstance(loaded, list):
        raise SystemExit(f"Expected a list of trajectories in {path}")
    if not all(isinstance(record, dict) for record in loaded):
        raise SystemExit(f"Expected all trajectory records in {path} to be JSON objects")
    return loaded


def terminalbench_system_prompt() -> str:
    return (
        "You are a terminal task agent operating in a shell environment. "
        "Use bash commands to inspect files, run programs, edit artifacts, and complete the user task."
    )


def terminalbench_task_instruction(events: list[dict[str, Any]]) -> str:
    for event in events:
        if event.get("event_type") != "ShellProtocolTask":
            continue
        payload = event.get("payload") or {}
        instruction = payload.get("instruction")
        if isinstance(instruction, str):
            return instruction
    for event in events:
        payload = event.get("payload") or {}
        parts = payload.get("parts") if isinstance(payload, dict) else None
        if not isinstance(parts, list):
            continue
        for part in parts:
            text = part.get("text") if isinstance(part, dict) else None
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if parsed.get("kind") == "task" and isinstance(parsed.get("instruction"), str):
                return parsed["instruction"]
    return ""


def terminalbench_exec_pairs(events: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs = []
    pending_requests: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("event_type")
        if event_type == "ShellProtocolExecRequest" and event.get("direction") == "inbound":
            pending_requests.append(event)
            continue
        if event_type != "ShellProtocolExecResult" or event.get("direction") != "green":
            continue
        if not pending_requests:
            continue
        pairs.append((pending_requests.pop(0), event))
    return pairs


def normalize_terminalbench_trajectory(record: dict[str, Any], trajectory_index: int) -> dict[str, Any]:
    events = record.get("events")
    if not isinstance(events, list):
        return record

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": terminalbench_system_prompt()},
        {"role": "user", "content": terminalbench_task_instruction(events)},
    ]
    for request_event, result_event in terminalbench_exec_pairs(events):
        request_payload = request_event.get("payload") or {}
        result_payload = result_event.get("payload") or {}
        command = request_payload.get("command") or request_event.get("command") or ""
        timeout = request_payload.get("timeout")
        exit_code = result_event.get("exit_code", result_payload.get("exit_code"))
        label = 1 if exit_code == 0 else -1
        stdout = result_payload.get("stdout") or ""
        stderr = result_payload.get("stderr") or ""
        output_parts = []
        if stdout:
            output_parts.append(f"stdout:\n{stdout}")
        if stderr:
            output_parts.append(f"stderr:\n{stderr}")
        output = "\n\n".join(output_parts)
        if not output:
            output = f"exit_code: {exit_code}"

        arguments = {"command": command}
        if timeout is not None:
            arguments["timeout"] = timeout
        action = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "execute_bash",
                        "arguments": arguments,
                    },
                }
            ]
        }
        state = {
            "state": {
                "agent": {"role": "terminal task agent"},
                "context": {
                    "last_tool_execution_result": label,
                    "last_tool_name": "execute_bash",
                    "last_tool_output": output,
                    "error_message": output if label != 1 else "",
                },
                "process": {
                    "remaining_stages": [],
                    "current_stage": "continue" if label == 1 else "repair failed shell command",
                },
                "relational": {},
                "temporal": {},
            }
        }
        messages.append({"role": "action", "content": action})
        messages.append({"role": "state", "content": state})

    final_output = ""
    for event in reversed(events):
        if event.get("event_type") != "ShellProtocolFinal":
            continue
        payload = event.get("payload") or {}
        if isinstance(payload.get("output"), str):
            final_output = payload["output"]
            break
    if final_output:
        messages.append({"role": "assistant", "content": final_output})

    normalized = dict(record)
    normalized.setdefault("trajectory_id", f"terminalbench-2.0-{trajectory_index}")
    normalized["messages"] = messages
    normalized["ewm_input_format"] = "system_user_action_state_v1"
    return normalized


def normalize_loaded_trajectory(record: dict[str, Any], trajectory_index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Trajectory records must be JSON objects.")
    if "messages" not in record and record.get("benchmark") == "Terminal-Bench-2.0":
        return normalize_terminalbench_trajectory(record, trajectory_index)
    if "messages" not in record and isinstance(record.get("events"), list):
        return normalize_terminalbench_trajectory(record, trajectory_index)
    normalized = dict(record)
    normalized.setdefault("ewm_input_format", "system_user_action_state_v1")
    return normalized


def normalize_loaded_trajectories(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_loaded_trajectory(record, index) for index, record in enumerate(records)]


def override_enterpriseops_gym_mcp_urls(raw_config: dict[str, Any]) -> dict[str, Any]:
    gym_servers = raw_config.get("gym_servers_config")
    if isinstance(gym_servers, list):
        for server in gym_servers:
            if not isinstance(server, dict):
                continue
            if server.get("mcp_server_name") != CSM_MCP_SERVER_NAME:
                continue
            if server.get("mcp_server_url") == CSM_MCP_DEFAULT_URL:
                server["mcp_server_url"] = CSM_MCP_OVERRIDE_URL

    if raw_config.get("mcp_server_name") == CSM_MCP_SERVER_NAME and raw_config.get("mcp_server_url") == CSM_MCP_DEFAULT_URL:
        raw_config["mcp_server_url"] = CSM_MCP_OVERRIDE_URL

    return raw_config


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")





def validate_disjoint_task_sets(
    train_trajectories: list[dict[str, Any]],
    test_trajectories: list[dict[str, Any]],
    *,
    dataset_name: str,
) -> None:
    def prompt_pair_key(trajectory: dict[str, Any]) -> tuple[str, str]:
        system_prompt = ""
        user_prompt = ""
        for message in trajectory.get("messages") or []:
            role = message.get("role")
            if role == "system" and not system_prompt:
                system_prompt = message.get("content") or ""
            elif role == "user" and not user_prompt:
                user_prompt = message.get("content") or ""
            if system_prompt and user_prompt:
                break
        return system_prompt, user_prompt

    train_task_indices = {trajectory.get("task_index") for trajectory in train_trajectories}
    test_task_indices = {trajectory.get("task_index") for trajectory in test_trajectories}
    overlapping_task_indices = sorted(
        task_index
        for task_index in (train_task_indices & test_task_indices)
        if task_index is not None
    )
    if overlapping_task_indices:
        preview = overlapping_task_indices[:10]
        raise SystemExit(
            f"{dataset_name} train/eval inputs overlap on task_index values: {preview}"
        )

    train_gym_task_names = {
        trajectory.get("gym_task_config_name")
        for trajectory in train_trajectories
        if trajectory.get("gym_task_config_name")
    }
    test_gym_task_names = {
        trajectory.get("gym_task_config_name")
        for trajectory in test_trajectories
        if trajectory.get("gym_task_config_name")
    }
    overlapping_task_names = sorted(train_gym_task_names & test_gym_task_names)
    if overlapping_task_names:
        preview = overlapping_task_names[:10]
        raise SystemExit(
            f"{dataset_name} train/eval inputs overlap on gym_task_config_name values: {preview}"
        )

    train_prompt_pairs = {prompt_pair_key(trajectory) for trajectory in train_trajectories}
    test_prompt_pairs = {prompt_pair_key(trajectory) for trajectory in test_trajectories}
    overlapping_prompt_pairs = train_prompt_pairs & test_prompt_pairs
    if overlapping_prompt_pairs:
        _, sample_user = next(iter(overlapping_prompt_pairs))
        raise SystemExit(
            f"{dataset_name} train/eval inputs overlap on (system prompt, user prompt) pairs. "
            f"Sample user prompt: {(sample_user or '')[:200]}"
        )


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def preview_text(value: Any, limit: int = 320) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            value = str(value)
    value = " ".join(value.split()).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def preview_tool_calls(tool_calls: list[dict[str, Any]], limit: int = 4) -> str:
    preview = []
    for call in tool_calls[:limit]:
        name = call.get("name", "")
        arguments = call.get("arguments", {})
        preview.append(f"{name}({preview_text(arguments, limit=120)})")
    suffix = ""
    if len(tool_calls) > limit:
        suffix = f" +{len(tool_calls) - limit} more"
    return "; ".join(preview) + suffix


def emit_progress(label: str, /, **fields: Any) -> None:
    show_imagined_progress = os.environ.get("EWM_SHOW_IMAGINED_PROGRESS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not show_imagined_progress and (
        label.startswith("IMAGINED_")
        or (label == "GYM_AGENT_FINAL_ANSWER" and fields.get("mode") == "imagined")
    ):
        return

    rendered_fields = []
    for key, value in fields.items():
        if value is None:
            continue
        rendered_fields.append(f"{key}={preview_text(value, limit=280)}")
    suffix = " " + " | ".join(rendered_fields) if rendered_fields else ""
    print(f"[{label}]{suffix}", flush=True)


def summarize_exception_group(exc: BaseException, max_items: int = 5, prefix: str = "") -> str:
    if not isinstance(exc, BaseExceptionGroup):
        label = prefix.rstrip(".") + ": " if prefix else ""
        return f"{label}{type(exc).__name__}: {exc}"

    parts: list[str] = []
    for index, sub_exc in enumerate(exc.exceptions[:max_items], start=1):
        item_prefix = f"{prefix}{index}."
        if isinstance(sub_exc, BaseExceptionGroup):
            parts.append(summarize_exception_group(sub_exc, max_items=max_items, prefix=item_prefix))
        else:
            parts.append(f"{item_prefix.rstrip('.')}: {type(sub_exc).__name__}: {sub_exc}")
    if len(exc.exceptions) > max_items:
        parts.append(f"{prefix}... {len(exc.exceptions) - max_items} more")
    return "; ".join(parts) if parts else f"{type(exc).__name__}: {exc}"


def strip_code_fence(text: str) -> str:
    value = text.strip()
    fence = re.match(r"^```(?:json|python|text)?\s*(.*?)\s*```$", value, re.DOTALL)
    return fence.group(1).strip() if fence else value


def strip_action_wrappers(text: str) -> str:
    """Strip code fences and `<tool_call>` / `<answer>` style wrappers from agent output."""
    value = strip_code_fence(text).strip()
    for opener, closer in ((TOOL_CALL_OPEN_TAG, TOOL_CALL_CLOSE_TAG), ("<answer>", "</answer>")):
        if opener in value and closer in value:
            start = value.find(opener) + len(opener)
            end = value.rfind(closer)
            if end > start:
                value = value[start:end].strip()
    if value.startswith(TOOL_CALL_OPEN_TAG):
        value = value[len(TOOL_CALL_OPEN_TAG):].strip()
    if value.endswith(TOOL_CALL_CLOSE_TAG):
        value = value[: -len(TOOL_CALL_CLOSE_TAG)].strip()
    return value


def _try_parse_json_blob(blob: str) -> Any:
    """Best-effort parse of a single JSON-ish blob, mirroring graph_final_localM.parse_json."""
    candidate = blob.strip()
    if not candidate:
        return None
    attempts = [
        candidate,
        candidate.replace("None", "null").replace("True", "true").replace("False", "false"),
    ]
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(attempt)
        except (ValueError, SyntaxError):
            pass
    # Final resort: replace single quotes with double quotes (graph_final_localM strategy).
    try:
        return json.loads(candidate.replace("'", '"'))
    except json.JSONDecodeError:
        return None


def iter_balanced_json_objects(text: str) -> Iterable[Any]:
    """Yield each top-level balanced `{...}` or `[...]` block parsed as JSON.

    Tracks brace depth while respecting string literals so that braces inside
    quoted strings do not corrupt the depth counter. Skips blobs that fail to
    parse. Useful when a local LM emits multiple JSON candidates concatenated
    by newlines (arguments dict + action dict + truncated trailing object).
    """
    cursor = 0
    n = len(text)
    while cursor < n:
        opener = text[cursor]
        if opener not in "{[":
            cursor += 1
            continue
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        string_quote = ""
        escape = False
        end = cursor
        balanced = False
        while end < n:
            ch = text[end]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == string_quote:
                    in_string = False
            else:
                if ch in "\"'":
                    in_string = True
                    string_quote = ch
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        balanced = True
                        break
            end += 1
        if not balanced:
            return
        blob = text[cursor : end + 1]
        parsed = _try_parse_json_blob(blob)
        if parsed is not None:
            yield parsed
        cursor = end + 1


_AGENT_DECISION_KEYS = ("action", "tool_calls", "final_answer", "name", "function")
_THOUGHT_FIELD_PATTERN = re.compile(
    r'["\']thought["\']\s*:\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,}\n][^,}\n]*)',
    re.DOTALL,
)


def parse_thought_payload(raw_text: str) -> dict[str, Any]:
    """Extract a `{"thought": ...}` dict from a free-form thought response.

    Strategies, in order:
      1. Strip wrappers and try the strict JSON parser; accept the result if it
         already contains a `thought` key.
      2. Walk balanced `{...}` blobs (same scanner as `parse_agent_decision`)
         and return the first one with a `thought` key — handles outputs that
         splice multiple JSON candidates together.
      3. Regex-extract the `thought` field from broken JSON (e.g. unquoted
         values like `{"thought": happens.`); strip surrounding quotes if any.
      4. Fall back to wrapping the cleaned raw text as the thought.
    """
    cleaned = strip_action_wrappers(raw_text).strip()
    if not cleaned:
        return {"thought": ""}

    parsed: Any = None
    try:
        parsed = parse_jsonish(cleaned)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and "thought" in parsed:
        return {"thought": str(parsed.get("thought") or "")}

    for candidate in iter_balanced_json_objects(cleaned):
        if isinstance(candidate, dict) and "thought" in candidate:
            return {"thought": str(candidate.get("thought") or "")}

    match = _THOUGHT_FIELD_PATTERN.search(cleaned)
    if match:
        value = match.group(1).strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            try:
                value = json.loads('"' + value[1:-1].replace('"', '\\"') + '"')
            except json.JSONDecodeError:
                value = value[1:-1]
        return {"thought": value.strip()}

    return {"thought": cleaned}


def _is_agent_decision_shaped(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in value for key in _AGENT_DECISION_KEYS)
    if isinstance(value, list) and value:
        return all(
            isinstance(item, dict) and any(key in item for key in _AGENT_DECISION_KEYS)
            for item in value
        )
    return False


def parse_jsonish(text: str) -> Any:
    text = strip_code_fence(text)
    candidates = [text]
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidates.append(text[first_bracket : last_bracket + 1])
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        for attempt in (
            candidate,
            candidate.replace("None", "null").replace("True", "true").replace("False", "false"),
        ):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(attempt)
                except Exception:
                    continue
    raise ValueError(f"Unable to parse JSON from model output: {text[:200]}")


def omit_low_signal_enterprise_state_fields(state: dict[str, Any]) -> dict[str, Any]:
    if not is_enterprise_state_payload(state):
        return state

    body = state_body_from_any(state)
    for field in ENTERPRISE_STATE_OMITTED_FIELDS:
        body.pop(field, None)

    diff = body.get("diff_from_previous_state")
    if isinstance(diff, dict):
        for field in ENTERPRISE_STATE_OMITTED_FIELDS:
            diff.pop(field, None)

    return state


def sanitize_state_content(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = parse_jsonish(value)
    if not isinstance(value, dict):
        raise ValueError(f"Expected state content to be a dict, got {type(value).__name__}")
    sanitized = json.loads(json.dumps(value, ensure_ascii=False))
    sanitized = omit_low_signal_enterprise_state_fields(sanitized)
    context = sanitized.get("state", {}).get("context")
    if isinstance(context, dict):
        context.pop("last_tool_output", None)
    return sanitized


def normalize_state_text(value: Any) -> str:
    if value is None:
        return "null"
    return json.dumps(sanitize_state_content(value), ensure_ascii=False, sort_keys=True)


def normalize_state_history_text(
    state_history: list[dict[str, Any]],
    current_interaction_index: int,
) -> str:
    entries: list[dict[str, Any]] = []
    start_interaction_index = current_interaction_index - len(state_history)
    for offset, state in enumerate(state_history):
        entries.append(
            {
                "interaction_index": start_interaction_index + offset,
                "state": sanitize_state_content(state),
            }
        )
    return json.dumps(entries, ensure_ascii=False, sort_keys=True)


def build_state_context_input_text(example: WorldModelStateExample) -> str:
    if example.state_history:
        return (
            "Recent state history (oldest to newest; last item is the previous state):\n"
            + normalize_state_history_text(example.state_history, example.interaction_index)
        )
    if example.previous_state is None:
        return "Previous state:\nnull"
    return "Previous state:\n" + normalize_state_text(example.previous_state)


def truncate_world_model_history_observation(value: Any) -> str:
    return stringify_tool_output(value)[:WORLD_MODEL_HISTORY_OBSERVATION_CHARS]


def compact_action_for_world_model_history(action: Any) -> Any:
    if isinstance(action, dict) and "tool_calls" in action:
        action = action.get("tool_calls") or []
    if isinstance(action, list):
        normalized = [normalize_tool_call(call) for call in action]
        return normalized[0] if len(normalized) == 1 else normalized
    if isinstance(action, str):
        stripped = action.strip()
        if stripped:
            try:
                parsed = parse_jsonish(stripped)
            except Exception:
                return stripped
            return compact_action_for_world_model_history(parsed)
        return ""
    if isinstance(action, dict) and ("name" in action or "function" in action):
        return normalize_tool_call(action)
    return action


def append_world_model_input_history(
    history: list[dict[str, Any]],
    entry: dict[str, Any] | None,
    max_items: int = WORLD_MODEL_INPUT_HISTORY_SIZE,
) -> list[dict[str, Any]]:
    if max_items <= 0:
        return []
    if not entry:
        return list(history)[-max_items:]
    return (list(history) + [entry])[-max_items:]


def make_actual_world_model_history_entry(
    step: int,
    action: Any,
    observation: Any,
) -> dict[str, Any]:
    return {
        "step": step,
        "action": compact_action_for_world_model_history(action),
        "observation": truncate_world_model_history_observation(observation),
    }


def make_actual_world_model_history_entry_from_results(
    step: int,
    tool_calls: list[dict[str, Any]],
    execution_results: list[dict[str, Any]],
) -> dict[str, Any]:
    observations = [
        truncate_world_model_history_observation(result.get("content", ""))
        for result in execution_results
    ]
    observation: Any = observations[0] if len(observations) == 1 else observations
    return {
        "step": step,
        "action": compact_action_for_world_model_history(tool_calls),
        "observation": observation,
    }


def make_imagined_world_model_history_entry(
    imagined_step: int,
    action: Any,
    state: Any,
) -> dict[str, Any]:
    return {
        "imagined step": imagined_step,
        "action": compact_action_for_world_model_history(action),
        "state": sanitize_state_content(state) if state is not None else None,
    }


def normalize_world_model_input_history_text(history: list[dict[str, Any]]) -> str:
    return json.dumps(history[-WORLD_MODEL_INPUT_HISTORY_SIZE:], ensure_ascii=False, sort_keys=True)


def append_state_history(
    state_history: list[dict[str, Any]],
    state: dict[str, Any],
    max_items: int = DEFAULT_STATE_HISTORY_SIZE,
) -> list[dict[str, Any]]:
    if max_items <= 0:
        return []
    return (list(state_history) + [sanitize_state_content(state)])[-max_items:]


def extract_state_error_payload(raw_state: Any) -> str:
    if isinstance(raw_state, str):
        try:
            raw_state = parse_jsonish(raw_state)
        except Exception:
            return ""
    context = state_context_from_any(raw_state)
    if not context:
        return ""
    error_message = context.get("error_message")
    if isinstance(error_message, str) and error_message.strip():
        return error_message.strip()
    last_tool_output = context.get("last_tool_output")
    if last_tool_output is None:
        return ""
    if isinstance(last_tool_output, str):
        return last_tool_output.strip()
    return json.dumps(last_tool_output, ensure_ascii=False)


def extract_state_tool_output(raw_state: Any) -> str:
    """Return `state.context.last_tool_output` as a string.

    Falls back to `error_message` when `last_tool_output` is missing so the
    `tool_output` target still has training signal on rows where the upstream
    pipeline only persisted the error message.
    """
    if isinstance(raw_state, str):
        try:
            raw_state = parse_jsonish(raw_state)
        except Exception:
            return ""
    context = state_context_from_any(raw_state)
    if not context:
        return ""
    last_tool_output = context.get("last_tool_output")
    if isinstance(last_tool_output, str) and last_tool_output.strip():
        return last_tool_output.strip()
    if last_tool_output is not None and not isinstance(last_tool_output, str):
        return json.dumps(last_tool_output, ensure_ascii=False)
    error_message = context.get("error_message")
    if isinstance(error_message, str) and error_message.strip():
        return error_message.strip()
    return ""


def label_should_include_error_message(label: int | None) -> bool:
    return label in {-1, 0}


def format_tool_execution_result_target(
    label: int,
    error_payload: str,
    include_error_message: bool,
    include_stage: bool = False,
    current_stage: Any = None,
    remaining_stages: Any = None,
) -> str:
    if include_stage:
        payload = {
            "success": label == 1,
            "last_tool_execution_result": label,
            "error_message": (
                error_payload
                if include_error_message and label_should_include_error_message(label)
                else ""
            ),
            "current_stage": current_stage,
            "remaining_stages": remaining_stages if remaining_stages is not None else [],
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if (
        include_error_message
        and label_should_include_error_message(label)
        and error_payload
    ):
        return f"{label},{error_payload}"
    return str(label)


def blank_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: blank_like(child) for key, child in value.items()}
    if isinstance(value, list):
        return []
    if isinstance(value, bool):
        return False
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return 0
    return ""


def make_blank_state() -> dict[str, Any]:
    return {
        "state": {
            "agent": {
                "role": "",
            },
            "context": {
                "last_tool_execution_result": 0,
                "last_tool_name": None,
            },
            "process": {
                "remaining_stages": [],
                "current_stage": "",
            },
            "relational": {
                "permisson_level": "",
            },
            "temporal": {
                "remaining_time": "",
            },
        }
    }


def make_enterprise_state_blank() -> dict[str, Any]:
    return {
        "state": {
            "mode": "full",
            "schema": "enterprise_ops_objects_process_relational_constraints_history_v1",
            "outcome": {
                "status": "unknown",
                "summary": "",
                "failure_category": "none",
                "recoverable": True,
            },
            "process_state": {
                "stage": "",
                "completed_requirements": [],
                "remaining_requirements": [],
                "owner": "",
                "queue": "",
                "approval_status": "",
                "blockers": [],
            },
            "history_context": {
                "salient_facts": [],
                "last_tool_events": [],
            },
        }
    }


def make_tool_execution_prediction_state(
    predicted_success: bool,
    predicted_result: int,
    error_message: str,
    current_stage: Any = None,
    remaining_stages: Any = None,
) -> dict[str, Any]:
    return {
        "success": predicted_success,
        "last_tool_execution_result": predicted_result,
        "error_message": error_message or "",
        "current_stage": current_stage,
        "remaining_stages": remaining_stages if remaining_stages is not None else [],
    }


def is_compact_tool_execution_state(raw_state: Any) -> bool:
    body = state_body_from_any(raw_state)
    return isinstance(body, dict) and any(
        key in body
        for key in (
            "success",
            "last_tool_execution_result",
            "error_message",
            "current_stage",
            "remaining_stages",
        )
    ) and "context" not in body and "process" not in body


def make_blank_state_like(reference_state: Any | None) -> dict[str, Any]:
    if is_enterprise_state_payload(reference_state):
        return make_enterprise_state_blank()
    if isinstance(reference_state, dict) and any(
        key in reference_state for key in ("last_tool_execution_result", "last_tool_name", "last_tool_output")
    ):
        return blank_like(reference_state)
    return make_blank_state()


def normalize_last_tool_execution_result(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in {-1, 0, 1}:
            return value
        return None
    if isinstance(value, float):
        if value in {-1.0, 0.0, 1.0}:
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"-1", "0", "1"}:
            return int(stripped)
    return None


def normalize_tool_execution_result_for_target(value: Any, target_mode: str) -> int | None:
    normalized = normalize_last_tool_execution_result(value)
    if normalized is None:
        return None
    if canonicalize_world_model_target(target_mode) == WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY:
        return 1 if normalized == 1 else 0
    return normalized


def encode_outcome_label(value: Any, target_mode: str = WORLD_MODEL_TARGET_STATE) -> int:
    normalized = normalize_tool_execution_result_for_target(value, target_mode)
    return normalized if normalized is not None else UNKNOWN_OUTCOME_LABEL


def extract_last_tool_execution_result_from_state(state: dict[str, Any]) -> int | None:
    context = state_context_from_any(state)
    return normalize_last_tool_execution_result(context.get("last_tool_execution_result"))


def normalize_tool_call(raw_call: Any) -> dict[str, Any]:
    if raw_call is None:
        return {"name": "", "arguments": {}}
    if isinstance(raw_call, dict) and "function" in raw_call:
        raw_call = raw_call["function"]
    if not isinstance(raw_call, dict):
        return {"name": str(raw_call), "arguments": {}}
    name = str(raw_call.get("name", "")).strip()
    arguments = raw_call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = parse_jsonish(arguments)
        except Exception:
            arguments = arguments.strip()
    return {"name": name, "arguments": arguments}


def _canonical_tool_call(call: dict[str, Any]) -> tuple[str, str]:
    """Stable (name, json_args) tuple for equality comparison across revisions."""
    name = str(call.get("name", ""))
    arguments = call.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return (name, arguments.strip())
    try:
        canonical_args = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        canonical_args = str(arguments)
    return (name, canonical_args)


def tool_calls_equal(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    if len(left) != len(right):
        return False
    return [_canonical_tool_call(call) for call in left] == [
        _canonical_tool_call(call) for call in right
    ]


def to_openai_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted = []
    for call in calls:
        normalized = normalize_tool_call(call)
        formatted.append(
            {
                "type": "function",
                "function": {
                    "name": normalized["name"],
                    "arguments": normalized["arguments"],
                },
            }
        )
    return formatted


def render_message(message: dict[str, Any]) -> str:
    role = message.get("role", "unknown")
    if "tool_calls" in message:
        return f"ASSISTANT_TOOL_CALLS: {json_compact([normalize_tool_call(call) for call in message['tool_calls']])}"
    if role == "tool":
        return f"TOOL[{message.get('name', '')}]: {message.get('content', '')}"
    return f"{role.upper()}: {message.get('content', '')}"


def render_messages(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "(empty)"
    return "\n\n".join(render_message(message) for message in messages)


def tool_response_is_error(content: Any) -> bool:
    return str(content).lstrip().startswith("Error:")


def infer_execution_result_success(result: dict[str, Any]) -> bool:
    explicit_success = result.get("success")
    raw_result = result.get("raw_result")

    if isinstance(raw_result, dict):
        raw_success = raw_result.get("success")
        raw_payload = raw_result.get("result")
        raw_error = raw_result.get("error")
        is_error = False
        if isinstance(raw_payload, dict):
            is_error = raw_payload.get("isError") is True
        if raw_success is not None:
            return bool(raw_success) and not is_error
        if raw_error:
            return False
        if is_error:
            return False

    if explicit_success is not None:
        return bool(explicit_success)

    content = result.get("content", "")
    return not tool_response_is_error(content)


def build_binary_world_model_target(example: WorldModelExample) -> str:
    if example.tool_success:
        return "1"
    error_message = example.tool_error_message or example.tool_response_content
    return f"0\n{error_message}".strip()


def build_world_model_target(example: WorldModelExample, target_mode: str) -> str:
    if target_mode == "tool_success_binary":
        return build_binary_world_model_target(example)
    return example.tool_response_content


def parse_binary_world_model_prediction(text: str) -> dict[str, Any]:
    value = strip_code_fence(text).strip()
    if not value:
        return {"success": None, "error_message": ""}
    if value[0] in "[{":
        try:
            parsed = parse_jsonish(value)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            label = None
            for key in ("last_tool_execution_result", "outcome_label", "success"):
                if key in parsed:
                    label = normalize_last_tool_execution_result(parsed.get(key))
                    if label is not None:
                        break
            process = parsed.get("process") if isinstance(parsed.get("process"), dict) else {}
            return {
                "success": label,
                "error_message": str(parsed.get("error_message") or parsed.get("error") or "").strip(),
                "current_stage": parsed.get("current_stage", process.get("current_stage")),
                "remaining_stages": parsed.get("remaining_stages", process.get("remaining_stages")),
            }
    match = re.match(r"^(-1|0|1)(?:[,\s]+(.*))?$", value, re.DOTALL)
    if match:
        return {
            "success": int(match.group(1)),
            "error_message": (match.group(2) or "").strip(),
        }
    if value.startswith("-1"):
        return {
            "success": -1,
            "error_message": value[2:].lstrip(", \t\n").strip(),
        }
    if value[0] in {"0", "1"}:
        return {
            "success": int(value[0]),
            "error_message": value[1:].lstrip(", \t\n").strip(),
        }
    return {"success": None, "error_message": value}


def parse_tool_execution_result_binary_prediction(text: str) -> int | None:
    return parse_tool_execution_result_prediction(
        text,
        target_mode=WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
    )


def parse_tool_execution_result_prediction(text: str, target_mode: str) -> int | None:
    parsed = parse_binary_world_model_prediction(text)
    return normalize_tool_execution_result_for_target(parsed.get("success"), target_mode=target_mode)


def apply_chat_template_or_fallback(
    tokenizer: Any | None,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    disable_chat_template: bool = False,
) -> str:
    if tokenizer is not None and not disable_chat_template and getattr(tokenizer, "chat_template", None):
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        # Some chat templates (e.g. Qwen3/Qwen3.6 and Gemma 4) branch on an
        # `enable_thinking` kwarg, but it is forwarded via `**kwargs` rather
        # than appearing in the formal signature. Detect by searching the
        # template text so we actually disable reasoning traces instead of
        # silently ignoring the extra kwarg.
        chat_template_text = getattr(tokenizer, "chat_template", "") or ""
        if "enable_thinking" in chat_template_text:
            kwargs["enable_thinking"] = False
        parameters = inspect.signature(tokenizer.apply_chat_template).parameters
        if "messages" in parameters:
            kwargs["messages"] = messages
        else:
            kwargs["conversation"] = messages
        return tokenizer.apply_chat_template(**kwargs)
    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    if add_generation_prompt:
        rendered.append("ASSISTANT:")
    return "\n\n".join(rendered)


def strip_model_thinking_output(
    text: str,
    *,
    special_tokens: Iterable[str] | None = None,
) -> str:
    """Remove model-emitted reasoning traces while preserving the final answer.

    Handles:
    - Qwen-style `<think> ... </think>`
    - Gemma thought channels: `<|channel>thought ... <channel|>`
    - Residual special tokens from raw decoding when we intentionally decode
      with `skip_special_tokens=False` to detect Gemma thought channels.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    had_model_turn_prefix = cleaned.startswith("<|turn>model\n") or cleaned.startswith("<turn|>model\n")
    cleaned = re.sub(
        r"<\|channel>thought\s*.*?<channel\|>\s*",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if special_tokens:
        for token in sorted({token for token in special_tokens if token}, key=len, reverse=True):
            cleaned = cleaned.replace(token, "")

    if had_model_turn_prefix and cleaned.startswith("model\n"):
        cleaned = cleaned[len("model\n") :]

    return cleaned.strip()


def common_prefix_length(left: list[int], right: list[int]) -> int:
    prefix_length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        prefix_length += 1
    return prefix_length





def build_split_indices(total_count: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(total_count))
    random.Random(seed).shuffle(indices)
    split_at = int(total_count * train_ratio)
    split_at = max(1, min(total_count - 1, split_at))
    train_indices = sorted(indices[:split_at])
    test_indices = sorted(indices[split_at:])
    return train_indices, test_indices


def state_message_follows_tool_action(message: dict[str, Any], next_message: dict[str, Any]) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "action"
        and next_message.get("role") == "state"
        and isinstance(content, dict)
        and bool(content.get("tool_calls"))
    )


def extract_last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        content = message.get("content")
        if message.get("role") == "assistant" and isinstance(content, str):
            return content
        if message.get("role") == "action" and isinstance(content, str):
            return content
    return ""


def count_trajectory_outcomes(trajectory: dict[str, Any]) -> tuple[int, int]:
    success_count = 0
    failure_count = 0
    messages = trajectory.get("messages", [])
    for cursor in range(len(messages) - 1):
        message = messages[cursor]
        next_message = messages[cursor + 1]
        if not state_message_follows_tool_action(message, next_message):
            continue
        current_state = sanitize_state_content(next_message.get("content"))
        result = extract_last_tool_execution_result_from_state(current_state)
        if result == 1:
            success_count += 1
        elif result in {-1, 0}:
            failure_count += 1
    return success_count, failure_count


def build_trajectory_outcome_stats(
    trajectories: list[dict[str, Any]],
    source_path: Path,
) -> list[TrajectoryOutcomeStats]:
    stats: list[TrajectoryOutcomeStats] = []
    for index, trajectory in enumerate(trajectories):
        success_count, failure_count = count_trajectory_outcomes(trajectory)
        stats.append(
            TrajectoryOutcomeStats(
                trajectory=trajectory,
                source_path=str(source_path),
                source_index=index,
                success_count=success_count,
                failure_count=failure_count,
                example_count=success_count + failure_count,
            )
        )
    return stats


def choose_train_ratio(
    explicit_ratio: float | None,
    primary_count: int,
    secondary_count: int,
) -> float:
    if explicit_ratio is not None:
        if not 0.0 < explicit_ratio < 1.0:
            raise SystemExit("--train-ratio must be between 0 and 1.")
        return explicit_ratio
    total_count = primary_count + secondary_count
    if total_count <= 1:
        return 1.0
    inferred_ratio = primary_count / total_count
    if inferred_ratio <= 0.0 or inferred_ratio >= 1.0:
        return 0.6
    return inferred_ratio


def split_trajectory_stats_stratified(
    stats: list[TrajectoryOutcomeStats],
    train_ratio: float,
    seed: int,
) -> tuple[list[TrajectoryOutcomeStats], list[TrajectoryOutcomeStats]]:
    if not stats:
        return [], []
    if len(stats) == 1:
        return list(stats), []

    rng = random.Random(seed)
    shuffled_stats = list(stats)
    rng.shuffle(shuffled_stats)
    shuffled_stats.sort(
        key=lambda item: (
            item.example_count,
            item.failure_count,
            item.success_count,
        ),
        reverse=True,
    )

    total_count = len(shuffled_stats)
    target_train_count = int(round(total_count * train_ratio))
    target_train_count = max(1, min(total_count - 1, target_train_count))
    total_success = sum(item.success_count for item in shuffled_stats)
    total_failure = sum(item.failure_count for item in shuffled_stats)
    target_train_success = round(total_success * train_ratio)
    target_train_failure = round(total_failure * train_ratio)

    train_stats: list[TrajectoryOutcomeStats] = []
    test_stats: list[TrajectoryOutcomeStats] = []
    train_success = 0
    train_failure = 0

    for index, item in enumerate(shuffled_stats):
        remaining_items = total_count - index
        remaining_train_slots = target_train_count - len(train_stats)
        if remaining_train_slots <= 0:
            test_stats.append(item)
            continue
        if remaining_train_slots >= remaining_items:
            train_stats.append(item)
            train_success += item.success_count
            train_failure += item.failure_count
            continue

        train_score = (
            abs((train_success + item.success_count) - target_train_success)
            + abs((train_failure + item.failure_count) - target_train_failure)
            + abs((len(train_stats) + 1) - target_train_count)
        )
        test_score = (
            abs(train_success - target_train_success)
            + abs(train_failure - target_train_failure)
            + abs(len(train_stats) - target_train_count)
        )
        if train_score <= test_score:
            train_stats.append(item)
            train_success += item.success_count
            train_failure += item.failure_count
        else:
            test_stats.append(item)

    return train_stats, test_stats


def extract_examples_and_tasks(
    trajectories: list[dict[str, Any]],
    indices: list[int],
) -> tuple[list[WorldModelExample], list[TaskTrajectory]]:
    examples: list[WorldModelExample] = []
    tasks: list[TaskTrajectory] = []

    for trajectory_index in indices:
        trajectory = trajectories[trajectory_index]
        messages = trajectory.get("messages", [])
        system_prompt = next((msg.get("content", "") for msg in messages if msg.get("role") == "system"), "")
        user_messages = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
        final_answer = ""
        steps: list[TaskStep] = []
        interaction_index = 0
        cursor = 0

        while cursor < len(messages):
            message = messages[cursor]

            if message.get("role") == "assistant" and "tool_calls" in message:
                raw_tool_calls = message.get("tool_calls") or []
                tool_calls = [normalize_tool_call(call) for call in raw_tool_calls]
                tool_messages: list[dict[str, Any]] = []
                look_ahead = cursor + 1
                while look_ahead < len(messages) and messages[look_ahead].get("role") == "tool":
                    tool_messages.append(messages[look_ahead])
                    look_ahead += 1

                context = messages[: cursor + 1]
                paired_results = []
                for call_index, tool_call in enumerate(tool_calls):
                    tool_message = tool_messages[call_index] if call_index < len(tool_messages) else {}
                    tool_name = tool_message.get("name", tool_call["name"])
                    tool_content = tool_message.get("content", "")
                    tool_success = not tool_response_is_error(tool_content)
                    examples.append(
                        WorldModelExample(
                            trajectory_index=trajectory_index,
                            interaction_index=interaction_index,
                            system_prompt=system_prompt,
                            task_query=user_messages[-1] if user_messages else "",
                            context_messages=context,
                            tool_call=tool_call,
                            tool_name=tool_call["name"],
                            tool_arguments=tool_call["arguments"],
                            tool_response_name=tool_name,
                            tool_response_content=tool_content,
                            tool_success=tool_success,
                            tool_error_message=tool_content if not tool_success else "",
                        )
                    )
                    paired_results.append({"name": tool_name, "content": tool_content})

                steps.append(
                    TaskStep(
                        interaction_index=interaction_index,
                        tool_calls=tool_calls,
                        tool_results=paired_results,
                    )
                )
                interaction_index += 1
                cursor = look_ahead
                continue

            if (
                cursor == len(messages) - 1
                and message.get("role") == "assistant"
                and "tool_calls" not in message
            ):
                final_answer = message.get("content", "")

            cursor += 1

        tasks.append(
            TaskTrajectory(
                trajectory_index=trajectory_index,
                system_prompt=system_prompt,
                user_messages=user_messages,
                steps=steps,
                final_answer=final_answer,
                initial_state=make_blank_state_like(
                    next(
                        (msg.get("content") for msg in messages if msg.get("role") == "state"),
                        None,
                    )
                ),
            )
        )

    return examples, tasks


@lru_cache(maxsize=4096)
def enterpriseops_raw_completion_metadata(source_path: str) -> dict[str, Any]:
    path = Path(source_path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if "PurpleInternalTrajectoryMetadata" not in line:
                    continue
                event = json.loads(line)
                if event.get("event_type") != "PurpleInternalTrajectoryMetadata":
                    continue
                info = ((event.get("payload") or {}).get("info") or {})
                if not isinstance(info, dict):
                    continue
                summary = info.get("verification_summary") if isinstance(info.get("verification_summary"), dict) else {}
                success = info.get("overall_success")
                if success is None and summary:
                    success = summary.get("failed") == 0 and summary.get("passed") == summary.get("total")
                return {
                    "success": bool(success) if isinstance(success, bool) else success,
                    "source": "enterpriseops_raw_metadata",
                    "verification_summary": summary,
                    "overall_success": info.get("overall_success"),
                }
    except Exception:
        return {}
    return {}


def trajectory_task_completion_metadata(trajectory: dict[str, Any]) -> dict[str, Any]:
    if isinstance(trajectory.get("task_success"), bool):
        metadata = {
            "success": trajectory.get("task_success"),
            "source": "trajectory_task_success",
        }
        for key in ("task_reward", "task_score", "task_total_score", "task_error", "task_reason"):
            if key in trajectory:
                metadata[key] = trajectory.get(key)
        return metadata

    for key in ("overall_success", "task_completed", "final_success"):
        if isinstance(trajectory.get(key), bool):
            return {"success": trajectory.get(key), "source": f"trajectory_{key}"}

    source_path = trajectory.get("source_path") or trajectory.get("seed_source_path")
    if isinstance(source_path, str) and "EnterpriseOps-Gym" in source_path:
        return enterpriseops_raw_completion_metadata(source_path)
    return {}


def attach_task_completion_to_state(state: dict[str, Any], completion: dict[str, Any]) -> dict[str, Any]:
    if not completion:
        return state
    updated = json.loads(json.dumps(state, ensure_ascii=False))
    body = updated.setdefault("state", {})
    if isinstance(body, dict):
        relational = body.setdefault("relational", {})
        if isinstance(relational, dict):
            relational["task_completion"] = completion
    return updated


def extract_state_examples(
    trajectories: list[dict[str, Any]],
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
) -> list[WorldModelStateExample]:
    examples: list[WorldModelStateExample] = []
    state_history_size = max(0, state_history_size)
    for trajectory_index, trajectory in enumerate(trajectories):
        messages = trajectory.get("messages", [])
        system_prompt = next((msg.get("content", "") for msg in messages if msg.get("role") == "system"), "")
        user_prompt = next((msg.get("content", "") for msg in messages if msg.get("role") == "user"), "")
        interaction_index = 0
        previous_state = None
        state_history: list[dict[str, Any]] = []
        input_history: list[dict[str, Any]] = []

        pair_cursors = [
            cursor
            for cursor in range(len(messages) - 1)
            if state_message_follows_tool_action(messages[cursor], messages[cursor + 1])
        ]
        last_pair_cursor = pair_cursors[-1] if pair_cursors else None
        completion_metadata = trajectory_task_completion_metadata(trajectory)

        for cursor in pair_cursors:
            message = messages[cursor]
            next_message = messages[cursor + 1]
            raw_state_content = next_message.get("content")
            current_state = sanitize_state_content(raw_state_content)
            if cursor == last_pair_cursor:
                current_state = attach_task_completion_to_state(current_state, completion_metadata)
            error_payload = extract_state_error_payload(raw_state_content)
            tool_output = extract_state_tool_output(raw_state_content)
            examples.append(
                WorldModelStateExample(
                    trajectory_id=str(trajectory.get("trajectory_id", trajectory_index)),
                    trajectory_index=trajectory_index,
                    interaction_index=interaction_index,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    action=message.get("content"),
                    state_history=list(state_history[-state_history_size:]) if state_history_size else [],
                    input_history=list(input_history[-WORLD_MODEL_INPUT_HISTORY_SIZE:]),
                    previous_state=None,
                    state=current_state,
                    error_payload=error_payload,
                    tool_output=tool_output,
                )
            )
            input_history = append_world_model_input_history(
                input_history,
                make_actual_world_model_history_entry(
                    step=interaction_index + 1,
                    action=message.get("content"),
                    observation=tool_output,
                ),
            )
            state_history.append(current_state)
            interaction_index += 1

    return examples


def extract_replay_tasks_from_state_trajectories(trajectories: list[dict[str, Any]]) -> list[TaskTrajectory]:
    tasks: list[TaskTrajectory] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        messages = trajectory.get("messages", [])
        system_prompt = next((msg.get("content", "") for msg in messages if msg.get("role") == "system"), "")
        user_messages = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
        final_answer = extract_last_assistant_text(messages)
        first_state = next(
            (msg.get("content") for msg in messages if msg.get("role") == "state"),
            None,
        )
        tasks.append(
            TaskTrajectory(
                trajectory_index=trajectory_index,
                system_prompt=system_prompt,
                user_messages=user_messages,
                steps=[],
                final_answer=final_answer,
                gym_task_config_name=resolve_gym_task_config_name(trajectory),
                initial_state=make_blank_state_like(first_state),
                source_path=trajectory.get("source_path") or trajectory.get("seed_source_path"),
            )
        )
    return tasks


def resolve_gym_task_config_name(trajectory: dict[str, Any]) -> str | None:
    """Pull the gym task-config filename off a trajectory dict.

    Newly generated EnterpriseOps-Gym trajectories carry an explicit
    `gym_task_config_name` field. For trajectories produced before that field was
    added, derive it from `task_stem` / `source_path` (gym run outputs use the
    pattern `results_<mode>__<domain>__<task_id>.json`).
    """
    explicit = trajectory.get("gym_task_config_name")
    domain = trajectory.get("domain")
    if explicit:
        if (
            isinstance(domain, str)
            and domain
            and domain != "unknown"
            and "__unknown__" in explicit
        ):
            return explicit.replace("__unknown__", f"__{domain}__", 1)
        return explicit
    stem = trajectory.get("task_stem")
    if not stem:
        source_path = trajectory.get("source_path")
        if source_path:
            stem = Path(source_path).stem
    if not stem or not stem.startswith("results_"):
        return None
    return f"{stem[len('results_'):]}.json"


def task_jsonl_name_from_path_like(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if path.suffix == ".jsonl":
        return path.name
    stem = path.stem
    task_stem = stem.split("__")[-1]
    if task_stem.startswith("task_"):
        return f"{task_stem}.jsonl"
    return None


def task_jsonl_name_from_replay_task(task: TaskTrajectory) -> str | None:
    return (
        task_jsonl_name_from_path_like(task.source_path)
        or task_jsonl_name_from_path_like(task.gym_task_config_name)
    )


def load_gym_task_split_test_jsonl_names(manifest_path: Path) -> list[str]:
    payload = load_json(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in EnterpriseOps-Gym task split manifest: {manifest_path}")
    test_paths = payload.get("test")
    if not isinstance(test_paths, list):
        raise ValueError(f"Expected `test` list in EnterpriseOps-Gym task split manifest: {manifest_path}")
    names = [Path(str(path)).name for path in test_paths]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate JSONL task names in EnterpriseOps-Gym test split manifest: {manifest_path}")
    return names


def replay_task_has_unknown_gym_config(task: TaskTrajectory) -> bool:
    return bool(task.gym_task_config_name and "__unknown__" in task.gym_task_config_name)


def filter_replay_tasks_by_gym_task_split(
    tasks: list[TaskTrajectory],
    manifest_path: Path,
) -> tuple[list[TaskTrajectory], dict[str, Any]]:
    test_jsonl_names = load_gym_task_split_test_jsonl_names(manifest_path)
    task_by_jsonl_name: dict[str, TaskTrajectory] = {}
    duplicate_task_names: list[str] = []
    for task in tasks:
        jsonl_name = task_jsonl_name_from_replay_task(task)
        if not jsonl_name:
            continue
        existing = task_by_jsonl_name.get(jsonl_name)
        if existing is not None:
            duplicate_task_names.append(jsonl_name)
            if should_replace_replay_task_for_split(existing, task):
                task_by_jsonl_name[jsonl_name] = task
            continue
        task_by_jsonl_name[jsonl_name] = task

    selected_tasks: list[TaskTrajectory] = []
    missing_jsonl_names: list[str] = []
    for jsonl_name in test_jsonl_names:
        task = task_by_jsonl_name.get(jsonl_name)
        if task is None:
            missing_jsonl_names.append(jsonl_name)
            continue
        selected_tasks.append(task)

    selected_domain_counts: dict[str, int] = {}
    for task in selected_tasks:
        domain = "unknown"
        if task.gym_task_config_name and "__" in task.gym_task_config_name:
            parts = task.gym_task_config_name.split("__")
            if len(parts) >= 3:
                domain = parts[1]
        selected_domain_counts[domain] = selected_domain_counts.get(domain, 0) + 1

    metadata = {
        "manifest_path": str(manifest_path),
        "requested_test_tasks": len(test_jsonl_names),
        "matched_test_tasks": len(selected_tasks),
        "selected_domain_counts": dict(sorted(selected_domain_counts.items())),
        "missing_test_tasks": missing_jsonl_names,
        "duplicate_available_task_names": sorted(set(duplicate_task_names)),
    }
    return selected_tasks, metadata


def should_replace_replay_task_for_split(existing: TaskTrajectory, candidate: TaskTrajectory) -> bool:
    return replay_task_has_unknown_gym_config(existing) and not replay_task_has_unknown_gym_config(candidate)


def summarize_example_outcomes(examples: list[WorldModelStateExample]) -> dict[str, int]:
    success_count = 0
    stagnation_count = 0
    error_count = 0
    unknown_count = 0
    for example in examples:
        outcome = extract_last_tool_execution_result_from_state(example.state)
        if outcome == 1:
            success_count += 1
        elif outcome == 0:
            stagnation_count += 1
        elif outcome == -1:
            error_count += 1
        else:
            unknown_count += 1
    return {
        "success_examples": success_count,
        "stagnation_examples": stagnation_count,
        "error_examples": error_count,
        "failure_examples": stagnation_count + error_count,
        "unknown_examples": unknown_count,
    }


def summarize_outcome_label_counts(
    labels: Iterable[Any],
    target_mode: str = WORLD_MODEL_TARGET_STATE,
) -> dict[str, int]:
    label_values = tool_execution_result_label_values(target_mode)
    counts = {str(label): 0 for label in label_values}
    unknown_count = 0
    for value in labels:
        normalized = normalize_tool_execution_result_for_target(value, target_mode)
        if normalized is None:
            unknown_count += 1
        else:
            counts[str(normalized)] += 1
    counts["unknown"] = unknown_count
    return counts


def build_outcome_loss_balance_metadata(
    outcome_labels: list[int],
    strategy: str,
    beta: float,
    target_mode: str,
) -> tuple[dict[int, float] | None, dict[str, Any]]:
    label_values = tool_execution_result_label_values(target_mode)
    class_counts = summarize_outcome_label_counts(outcome_labels, target_mode=target_mode)
    known_total = sum(class_counts[str(label)] for label in label_values)
    metadata = {
        "strategy": strategy,
        "applied_via": "per_row_class_weighted_loss" if strategy != "none" else "disabled",
        "beta": beta if strategy == "effective_num_loss" else None,
        "label_values": list(label_values),
        "class_counts": class_counts,
        "class_weights": {str(label): 1.0 for label in label_values},
        "known_examples": known_total,
        "class_weight_min": 1.0,
        "class_weight_max": 1.0,
    }
    if strategy == "none" or known_total == 0:
        return None, metadata

    raw_weights: dict[int, float] = {}
    for label in label_values:
        count = class_counts[str(label)]
        if count <= 0:
            continue
        if strategy == "inverse_frequency_loss":
            raw_weights[label] = 1.0 / count
        elif strategy == "effective_num_loss":
            if not 0.0 <= beta < 1.0:
                raise ValueError("--outcome-balance-beta must be in [0.0, 1.0).")
            raw_weights[label] = 1.0 if beta <= 0.0 else (1.0 - beta) / (1.0 - (beta ** count))
        else:
            raise ValueError(f"Unsupported outcome balance strategy: {strategy}")

    normalizer_denominator = sum(
        class_counts[str(label)] * raw_weights[label]
        for label in label_values
        if label in raw_weights
    )
    normalizer = (known_total / normalizer_denominator) if normalizer_denominator > 0 else 1.0
    class_weights = {
        label: (raw_weights[label] * normalizer if label in raw_weights else 0.0)
        for label in label_values
    }
    metadata["class_weights"] = {str(label): class_weights[label] for label in label_values}
    nonzero_weights = [w for w in class_weights.values() if w > 0]
    metadata["class_weight_min"] = min(nonzero_weights) if nonzero_weights else 1.0
    metadata["class_weight_max"] = max(nonzero_weights) if nonzero_weights else 1.0
    return class_weights, metadata


def oversample_rows_by_outcome(
    rows: list[dict[str, Any]],
    *,
    target_mode: str,
    target_ratio: float = 1.0,
    max_multiplier: float = 10.0,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_values = tool_execution_result_label_values(target_mode)
    indices_by_label: dict[int, list[int]] = {label: [] for label in label_values}
    for index, row in enumerate(rows):
        try:
            label = int(row.get("outcome_label", UNKNOWN_OUTCOME_LABEL))
        except (TypeError, ValueError):
            continue
        if label in indices_by_label:
            indices_by_label[label].append(index)

    counts_before = {label: len(items) for label, items in indices_by_label.items()}
    metadata: dict[str, Any] = {
        "applied": False,
        "target_ratio": target_ratio,
        "max_multiplier": max_multiplier,
        "label_values": list(label_values),
        "counts_before": {str(label): counts_before[label] for label in label_values},
        "counts_after": {str(label): counts_before[label] for label in label_values},
        "rows_before": len(rows),
        "rows_after": len(rows),
    }

    populated_counts = [count for count in counts_before.values() if count > 0]
    if len(populated_counts) < 2:
        metadata["reason"] = "fewer_than_two_known_classes"
        return rows, metadata
    if max_multiplier <= 1.0:
        metadata["reason"] = "max_multiplier_disables_duplication"
        return rows, metadata

    majority_count = max(counts_before.values())
    target_count = max(1, int(round(majority_count * target_ratio)))
    rng = random.Random(seed)
    augmented_indices: list[int] = list(range(len(rows)))
    counts_after = dict(counts_before)
    capped_classes: list[int] = []

    for label in label_values:
        current = counts_before[label]
        if current == 0 or current >= target_count:
            continue
        deficit = target_count - current
        cap = int(current * max_multiplier) - current
        if cap <= 0:
            continue
        if deficit > cap:
            deficit = cap
            capped_classes.append(label)
        chosen = [rng.choice(indices_by_label[label]) for _ in range(deficit)]
        augmented_indices.extend(chosen)
        counts_after[label] = current + deficit

    if len(augmented_indices) == len(rows):
        metadata["reason"] = "no_minority_classes_below_target"
        return rows, metadata

    rng.shuffle(augmented_indices)
    augmented_rows = [rows[i] for i in augmented_indices]
    metadata.update(
        {
            "applied": True,
            "majority_count": majority_count,
            "target_count_per_class": target_count,
            "counts_after": {str(label): counts_after[label] for label in label_values},
            "rows_after": len(augmented_rows),
            "rows_added": len(augmented_rows) - len(rows),
            "capped_classes": [str(label) for label in capped_classes],
        }
    )
    return augmented_rows, metadata


def maybe_apply_outcome_loss_balance(
    dataset: Any,
    strategy: str,
    beta: float,
    target_mode: str,
) -> tuple[Any, dict[int, float] | None, dict[str, Any]]:
    if "outcome_label" not in dataset.column_names:
        return (
            dataset,
            None,
            {"strategy": "none", "reason": "missing_outcome_label_column"},
        )
    class_weights, metadata = build_outcome_loss_balance_metadata(
        list(dataset["outcome_label"]),
        strategy=strategy,
        beta=beta,
        target_mode=target_mode,
    )
    return dataset, class_weights, metadata


def build_state_prediction_chat_messages(
    example: WorldModelStateExample,
    target_mode: str = "state",
    include_error_message: bool = False,
    include_stage: bool = False,
    include_input_history: bool = False,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> list[dict[str, str]]:
    target_mode = canonicalize_world_model_target(target_mode)
    system_prompt_text = truncate_for_replay(example.system_prompt or "", system_prompt_max_chars)
    action_text = (
        example.action
        if isinstance(example.action, str)
        else json.dumps(example.action, indent=2, ensure_ascii=False)
    )
    action_text = truncate_for_replay(action_text, action_max_chars)
    if is_canonical_event_state_target(target_mode) or is_canonical_event_with_nudge_target(target_mode):
        state_context_text = "Previous canonical event state:\nnull"
    else:
        state_context_text = build_state_context_input_text(example)
    if include_input_history and example.input_history:
        state_context_text += (
            "\n\nRecent action/observation history (oldest to newest; input only, not part of the target):\n"
            + normalize_world_model_input_history_text(example.input_history)
        )
    if target_mode == WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY:
        if include_error_message:
            system_instruction = (
                "You are a world model for task trajectories. "
                "Given the system prompt, task prompt, recent state history, and current action, "
                "predict the action outcome label. "
                "Return `1` for success. For `0` (stagnation) or `-1` (explicit tool failure), "
                "append the error message or API response after the label separated by a comma, "
                "for example `-1,API Error: ...`."
            )
            user_instruction = (
                "Predict the action outcome label. Return `1` for success, or `0,<error>` / "
                "`-1,<error>` when the action does not succeed. /no_think"
            )
        else:
            system_instruction = (
                "You are a world model for task trajectories. "
                "Given the system prompt, task prompt, recent state history, and current action, "
                "predict the action outcome label. "
                "Return only `1` for success, `0` for stagnation, or `-1` for explicit tool failure."
            )
            user_instruction = (
                "Predict the action outcome label. Return only `1`, `0`, or `-1`. /no_think"
            )
        return [
            {
                "role": "system",
                "content": system_instruction,
            },
            {
                "role": "user",
                "content": (
                    f"System prompt:\n{system_prompt_text}\n\n"
                    f"Task prompt:\n{example.user_prompt}\n\n"
                    f"{state_context_text}\n\n"
                    f"Action:\n{action_text}\n\n"
                    f"{user_instruction}"
                ),
            },
        ]
    if target_mode == WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY:
        if include_stage:
            system_instruction = (
                "You are a world model for task trajectories. "
                "Given the system prompt, task prompt, recent state history, and current action, "
                "predict whether the action succeeds and the resulting process stage. "
                "Return JSON only with keys `success` (boolean), `last_tool_execution_result` "
                "(1 for success, 0 for failure), `error_message`, `current_stage`, and "
                "`remaining_stages`. Treat stagnation and explicit tool failure both as `0`."
            )
            user_instruction = (
                "Predict the binary action outcome and process stage as JSON. Use an empty "
                "`error_message` for success; include the error or API response when the action "
                "does not succeed. /no_think"
            )
        elif include_error_message:
            system_instruction = (
                "You are a world model for task trajectories. "
                "Given the system prompt, task prompt, recent state history, and current action, "
                "predict whether the action succeeds. "
                "Return `1` for success. Otherwise return `0` followed by a comma and the error "
                "message or API response, for example `0,API Error: ...`. "
                "Treat stagnation and explicit tool failure both as `0`."
            )
            user_instruction = (
                "Predict whether the action succeeds. Return `1` for success or `0,<error>` "
                "when it does not succeed. /no_think"
            )
        else:
            system_instruction = (
                "You are a world model for task trajectories. "
                "Given the system prompt, task prompt, recent state history, and current action, "
                "predict whether the action succeeds. "
                "Return only `1` for success or `0` for failure. "
                "Treat stagnation and explicit tool failure as `0`."
            )
            user_instruction = (
                "Predict whether the action succeeds. Return only `1` or `0`. /no_think"
            )
        return [
            {
                "role": "system",
                "content": system_instruction,
            },
            {
                "role": "user",
                "content": (
                    f"System prompt:\n{system_prompt_text}\n\n"
                    f"Task prompt:\n{example.user_prompt}\n\n"
                    f"{state_context_text}\n\n"
                    f"Action:\n{action_text}\n\n"
                    f"{user_instruction}"
                ),
            },
        ]
    if target_mode == WORLD_MODEL_TARGET_CANONICAL_EVENT_STATE:
        field_list = ", ".join(CANONICAL_EVENT_REQUIRED_CATEGORICAL_FIELDS.keys())
        return [
            {
                "role": "system",
                "content": (
                    "You are a world model for task trajectories. "
                    "Given the system prompt, task prompt, recent state history, and current action, "
                    "predict the categorical action-effect event state. Return JSON only. "
                    "Every field must be categorical; do not include free-text summaries, evidence, or explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"System prompt:\n{system_prompt_text}\n\n"
                    f"Task prompt:\n{example.user_prompt}\n\n"
                    f"{state_context_text}\n\n"
                    f"Action:\n{action_text}\n\n"
                    "Predict canonical_event_state JSON with exactly these required categorical fields: "
                    f"{field_list}. Do not include schema_version. Do not include uncertainty fields unless explicitly requested. /no_think"
                ),
            },
        ]
    if target_mode == WORLD_MODEL_TARGET_CANONICAL_EVENT_WITH_NUDGE:
        event_field_list = ", ".join(CANONICAL_EVENT_REQUIRED_CATEGORICAL_FIELDS.keys())
        nudge_field_list = ", ".join([*NUDGE_CATEGORICAL_FIELDS.keys(), *NUDGE_LIST_FIELDS.keys()])
        missing_info_values = ", ".join(sorted(NUDGE_LIST_FIELDS["missing_information_type"]))
        return [
            {
                "role": "system",
                "content": (
                    "You are a transition critic for task trajectories. "
                    "Given the system prompt, task prompt, recent state history, and current action, "
                    "predict the categorical action-effect event state and a categorical epistemic nudge. "
                    "Return JSON only. Every field must be categorical; do not include free-text summaries, "
                    "search targets, evidence, or explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"System prompt:\n{system_prompt_text}\n\n"
                    f"Task prompt:\n{example.user_prompt}\n\n"
                    f"{state_context_text}\n\n"
                    f"Action:\n{action_text}\n\n"
                    "Predict canonical_event_with_nudge JSON with exactly these top-level fields: "
                    "canonical_event_state, nudge. Do not include schema_version anywhere. "
                    "canonical_event_state must include exactly these required categorical fields: "
                    f"{event_field_list}. It must not include uncertainty fields. "
                    "nudge must include exactly these categorical fields: "
                    f"{nudge_field_list}. recommended_abstract_action must be the best next recovery/progress action. "
                    "Use finalize only when the observation gives explicit task-completion evidence, such as relational.task_completion.success=true, final verifier success, all required verifiers passing, task_success=true, or equivalent benchmark success. "
                    "If the action failed, errored, had negative progress, or left problems unresolved, do not recommend proceed/finalize. "
                    "missing_information_type must be a non-empty list using only: "
                    f"{missing_info_values}. /no_think"
                ),
            },
        ]
    if target_mode == WORLD_MODEL_TARGET_TOOL_OUTPUT:
        return [
            {
                "role": "system",
                "content": (
                    "You are a world model for task trajectories. "
                    "Given the system prompt, task prompt, recent state history, and current action, "
                    "predict the raw tool output (the data the tool would return, including "
                    "API responses on success and error messages on failure). "
                    "Return only the tool output text, exactly as the tool would return it."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"System prompt:\n{system_prompt_text}\n\n"
                    f"Task prompt:\n{example.user_prompt}\n\n"
                    f"{state_context_text}\n\n"
                    f"Action:\n{action_text}\n\n"
                    "Predict the tool output. /no_think"
                ),
            },
        ]
    return [
        {
            "role": "system",
            "content": (
                "You are a world model for task trajectories. "
                "Given the system prompt, task prompt, recent state history, and current action, predict the resulting state. "
                "Return JSON only. Preserve the state schema used by the previous state. "
                "For enterprise_ops_objects_process_relational_constraints_history_v1 states, predict outcome, process_state, and history_context fields in that schema; do not output objects_artifacts, relational_state, or constraints. "
                "For legacy states, exclude `state.context.last_tool_output` from the output."
            ),
        },
        {
            "role": "user",
            "content": (
                f"System prompt:\n{system_prompt_text}\n\n"
                f"Task prompt:\n{example.user_prompt}\n\n"
                f"{state_context_text}\n\n"
                f"Action:\n{action_text}\n\n"
                "Predict the resulting state as JSON. /no_think"
            ),
        },
    ]


def build_state_prediction_completion_messages(
    example: WorldModelStateExample,
    target_mode: str = "state",
    include_error_message: bool = False,
    include_stage: bool = False,
) -> list[dict[str, str]]:
    target_mode = canonicalize_world_model_target(target_mode)
    if is_tool_execution_result_target(target_mode):
        target = normalize_tool_execution_result_for_target(
            extract_last_tool_execution_result_from_state(example.state),
            target_mode=target_mode,
        )
        if target is None:
            raise ValueError(
                "Missing `state.context.last_tool_execution_result` for tool-result target example."
            )
        content = format_tool_execution_result_target(
            target,
            example.error_payload,
            include_error_message=include_error_message,
            include_stage=(
                include_stage
                and target_mode == WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY
            ),
            current_stage=state_current_stage(example.state),
            remaining_stages=state_remaining_stages(example.state),
        )
        return [{"role": "assistant", "content": content}]
    if is_canonical_event_state_target(target_mode):
        return [
            {
                "role": "assistant",
                "content": canonical_event_json(
                    canonical_event_from_action_state(example.action, example.state)
                ),
            }
        ]
    if is_canonical_event_with_nudge_target(target_mode):
        return [
            {
                "role": "assistant",
                "content": canonical_event_with_nudge_json(
                    canonical_event_with_nudge_from_action_state(example.action, example.state)
                ),
            }
        ]
    if is_tool_output_target(target_mode):
        return [{"role": "assistant", "content": example.tool_output or ""}]
    return [
        {
            "role": "assistant",
            "content": normalize_state_text(example.state),
        }
    ]


def build_state_prediction_prompt_completion_row(
    example: WorldModelStateExample,
    target_mode: str = "state",
    include_error_message: bool = False,
    include_stage: bool = False,
    include_input_history: bool = False,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> dict[str, Any]:
    target_mode = canonicalize_world_model_target(target_mode)
    return {
        "trajectory_id": example.trajectory_id,
        "trajectory_index": example.trajectory_index,
        "interaction_index": example.interaction_index,
        "prompt": build_state_prediction_chat_messages(
            example,
            target_mode=target_mode,
            include_error_message=include_error_message,
            include_stage=include_stage,
            include_input_history=include_input_history,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
        ),
        "completion": build_state_prediction_completion_messages(
            example,
            target_mode=target_mode,
            include_error_message=include_error_message,
            include_stage=include_stage,
        ),
        "outcome_label": encode_outcome_label(
            extract_last_tool_execution_result_from_state(example.state),
            target_mode=target_mode,
        ),
    }


def build_world_model_chat_messages(
    example: WorldModelExample,
    target_mode: str = "response_content",
) -> list[dict[str, str]]:
    if target_mode == "tool_success_binary":
        return [
            {
                "role": "system",
                "content": (
                    "You are an enterprise world model that predicts whether a tool call will succeed. "
                    "Use only the trajectory system prompt and the tool_calls. "
                    "Return `1` only if the tool call succeeds. If it fails, return `0` on the first line "
                    "and the tool error message after it. Do not add any explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Trajectory system prompt:\n{example.system_prompt}\n\n"
                    f"tool_calls:\n{json.dumps([example.tool_call], indent=2, ensure_ascii=False)}\n\n"
                    "Predict whether the tool call succeeds. /no_think"
                ),
            },
        ]

    return [
        {
            "role": "system",
            "content": (
                "You are an enterprise world model that predicts the result of a tool call. "
                "Return only the tool response content with no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task query:\n{example.task_query}\n\n"
                f"Conversation so far:\n{render_messages(example.context_messages)}\n\n"
                f"Tool call:\n{json.dumps(example.tool_call, indent=2, ensure_ascii=False)}\n\n"
                "Predict the tool response content. /no_think"
            ),
        },
    ]


def build_world_model_completion_messages(
    example: WorldModelExample,
    target_mode: str = "response_content",
) -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": build_world_model_target(example, target_mode=target_mode),
        }
    ]


def build_world_model_prompt_completion_row(
    example: WorldModelExample,
    target_mode: str = "response_content",
) -> dict[str, Any]:
    return {
        "trajectory_index": example.trajectory_index,
        "interaction_index": example.interaction_index,
        "tool_name": example.tool_name,
        "prompt": build_world_model_chat_messages(example, target_mode=target_mode),
        "completion": build_world_model_completion_messages(example, target_mode=target_mode),
    }


def build_world_model_prompt(
    example: WorldModelExample,
    tokenizer: Any | None = None,
    disable_chat_template: bool = False,
    target_mode: str = "response_content",
) -> str:
    return apply_chat_template_or_fallback(
        tokenizer,
        build_world_model_chat_messages(example, target_mode=target_mode),
        add_generation_prompt=True,
        disable_chat_template=disable_chat_template,
    )


def build_world_model_training_text(
    example: WorldModelExample,
    tokenizer: Any | None,
    eos_token: str,
    disable_chat_template: bool = False,
    target_mode: str = "response_content",
) -> str:
    prompt = build_world_model_prompt(
        example,
        tokenizer=tokenizer,
        disable_chat_template=disable_chat_template,
        target_mode=target_mode,
    )
    return prompt + build_world_model_target(example, target_mode=target_mode) + (eos_token or "")





_REPLAY_LIMITS: dict[str, int] = {
    "observation_chars": 2000,
    "history_budget_chars": 60000,
}


def configure_replay_limits(observation_chars: int, history_budget_chars: int) -> None:
    """Update the soft caps used by `filter_messages_for_react_replay`.

    `observation_chars` limits each tool observation. `history_budget_chars` is
    a soft cap on the total character count of the prompt sent to the agent;
    when exceeded, the oldest tool/assistant turns are dropped while keeping
    the system prompt and the very first user query.
    """
    _REPLAY_LIMITS["observation_chars"] = max(0, int(observation_chars))
    _REPLAY_LIMITS["history_budget_chars"] = max(0, int(history_budget_chars))


def truncate_for_replay(text: str, char_limit: int) -> str:
    if char_limit <= 0 or len(text) <= char_limit:
        return text
    keep_head = max(char_limit - 200, char_limit // 2)
    head = text[:keep_head]
    return f"{head}... [truncated {len(text) - keep_head} chars to fit context]"





def require_training_stack():
    #try:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    # except ImportError as exc:
    #     raise SystemExit(
    #         "Missing training dependencies. Install `torch`, `transformers`, and `datasets` before running this script."
    #     ) from exc
    return torch, Dataset, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


def resolve_vocab_size(config: Any) -> int:
    """Return the text-decoder vocab size, regardless of model topology.

    Multimodal configs (e.g. `Qwen3_5Config`) put `vocab_size` under
    `config.text_config`; flat causal-LM configs expose it at the top level.
    """
    direct = getattr(config, "vocab_size", None)
    if direct is not None:
        return int(direct)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "vocab_size", None) is not None:
        return int(text_config.vocab_size)
    raise AttributeError(
        f"Could not resolve vocab_size from {type(config).__name__}: "
        "neither `config.vocab_size` nor `config.text_config.vocab_size` is set."
    )


def resolve_text_generation_model_class(
    model_path: str,
    *,
    trust_remote_code: bool,
    causal_lm_class: Any,
) -> Any:
    """Pick the right HF auto-class for `model_path`.

    Multimodal Qwen3 models (e.g. `Qwen/Qwen3.6-27B`) ship with both a vision
    encoder and a text decoder under a single `*ForConditionalGeneration`
    architecture, so `AutoModelForCausalLM` refuses to load them. For
    text-only fine-tuning we still want to drive them through the standard
    causal-LM loop, so we fall back to `AutoModelForImageTextToText` whose
    forward accepts the same `(input_ids, attention_mask, labels)` triple
    when no image inputs are provided.

    `model_path` may also point at a PEFT adapter directory (containing
    `adapter_config.json` instead of `config.json`) — in that case we resolve
    the base model from `base_model_name_or_path` and inspect *that* config.
    """
    from transformers import AutoConfig, AutoModelForImageTextToText

    config_source = model_path
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.is_file():
        with adapter_config_path.open("r", encoding="utf-8") as handle:
            adapter_cfg = json.load(handle)
        base_model = adapter_cfg.get("base_model_name_or_path")
        if base_model:
            config_source = base_model
    config = AutoConfig.from_pretrained(config_source, trust_remote_code=trust_remote_code)
    architectures = list(getattr(config, "architectures", None) or [])
    has_vision_config = getattr(config, "vision_config", None) is not None
    is_multimodal = has_vision_config or any(
        arch.endswith("ForConditionalGeneration") for arch in architectures
    )
    return AutoModelForImageTextToText if is_multimodal else causal_lm_class





def stringify_tool_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)








def resolve_torch_dtype(torch_module: Any, dtype_name: str | None) -> Any:
    if not dtype_name or dtype_name == "auto":
        return "auto"
    if not hasattr(torch_module, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch_module, dtype_name)


def ensure_peft_weight_converter_compatibility() -> None:
    """Bridge PEFT 0.19 adapter conversion calls to newer Transformers APIs."""
    try:
        import peft.utils.transformers_weight_conversion as transformers_weight_conversion
    except Exception:
        return

    weight_converter = getattr(transformers_weight_conversion, "WeightConverter", None)
    if weight_converter is None or getattr(weight_converter, "_ewm_peft_compat_init", False):
        return

    try:
        init_signature = inspect.signature(weight_converter.__init__)
    except (TypeError, ValueError):
        return
    if "distributed_operation" in init_signature.parameters:
        return

    original_init = weight_converter.__init__

    def compatible_init(
        self,
        source_patterns,
        target_patterns,
        operations,
        *,
        distributed_operation=None,
        quantization_operation=None,
        **kwargs,
    ):
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(
                f"{weight_converter.__name__}.__init__() got an unexpected keyword argument "
                f"{unexpected!r}"
            )
        original_init(self, source_patterns, target_patterns, operations)
        self.distributed_operation = distributed_operation
        self.quantization_operation = quantization_operation

    compatible_init.__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("source_patterns", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("target_patterns", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("operations", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("distributed_operation", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("quantization_operation", inspect.Parameter.KEYWORD_ONLY, default=None),
        ]
    )
    weight_converter.__init__ = compatible_init
    weight_converter._ewm_peft_compat_init = True

    original_convert = transformers_weight_conversion.convert_peft_adapter_state_dict_for_transformers

    def compatible_convert_peft_adapter_state_dict_for_transformers(
        model,
        peft_config,
        adapter_state_dict,
        adapter_name="default",
    ):
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        if model_type == "nemotron_h" and any(
            ".mixer." in key and ".lora_" in key and key.startswith("base_model.model.")
            for key in adapter_state_dict
        ):
            return adapter_state_dict
        return original_convert(
            model=model,
            peft_config=peft_config,
            adapter_state_dict=adapter_state_dict,
            adapter_name=adapter_name,
        )

    transformers_weight_conversion.convert_peft_adapter_state_dict_for_transformers = (
        compatible_convert_peft_adapter_state_dict_for_transformers
    )


def parse_csv_arg(raw_value: str) -> list[str]:
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def default_lora_target_modules(model_name: str) -> list[str] | str:
    lowered = model_name.lower()
    if "nemotron-3-nano" in lowered or "nemotron_3_nano" in lowered:
        return [
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
            "in_proj",
            "out_proj",
        ]
    if "gemma-4" in lowered or "gemma4" in lowered:
        return [
            "q_proj.linear",
            "k_proj.linear",
            "v_proj.linear",
            "o_proj.linear",
            "gate_proj.linear",
            "up_proj.linear",
            "down_proj.linear",
        ]
    if "qwen" in lowered:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


def build_lora_config(args: argparse.Namespace) -> Any | None:
    if not args.use_lora:
        return None
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise SystemExit(
            "LoRA was requested with `--use-lora`, but `peft` is not installed. Install `peft` and rerun."
        ) from exc

    target_modules: list[str] | str = (
        default_lora_target_modules(args.model)
        if args.lora_target_modules == "auto"
        else parse_csv_arg(args.lora_target_modules)
    )
    modules_to_save = parse_csv_arg(args.lora_modules_to_save) or None
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )


def resolve_training_device_map(torch_module: Any) -> Any:
    if not torch_module.cuda.is_available():
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {"": local_rank}


def is_single_process_multi_gpu(torch_module: Any) -> bool:
    if not torch_module.cuda.is_available():
        return False
    world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE")
        or os.environ.get("WORLD_SIZE")
        or "1"
    )
    return world_size <= 1 and torch_module.cuda.device_count() > 1


def maybe_disable_nemotron_fast_mamba_kernels(
    model: Any,
    *,
    torch_module: Any,
    gradient_checkpointing: bool,
    allow_fast_mamba_kernels: bool,
) -> dict[str, Any]:
    """Force Nemotron's remote-code Mamba blocks onto the PyTorch path when
    the fused Triton kernels are known to be brittle for training.

    The current `modeling_nemotron_h.py` remote module exposes a
    `config.use_mamba_kernels` flag but dispatches through the module-global
    `is_fast_path_available` instead. We therefore patch that global directly.
    """
    config = getattr(model, "config", None)
    metadata: dict[str, Any] = {
        "model_type": getattr(config, "model_type", None),
        "applied": False,
        "reasons": [],
        "patched_modules": [],
        "config_use_mamba_kernels_before": getattr(config, "use_mamba_kernels", None),
        "effective_use_mamba_kernels": getattr(config, "use_mamba_kernels", None),
    }
    if metadata["model_type"] != "nemotron_h":
        return metadata

    reasons: list[str] = []
    if gradient_checkpointing:
        reasons.append("gradient_checkpointing")
    if is_single_process_multi_gpu(torch_module):
        reasons.append("single_process_multi_gpu")
    metadata["reasons"] = reasons

    if allow_fast_mamba_kernels:
        metadata["skipped_reason"] = "allow_fast_mamba_kernels_flag"
        return metadata
    if not reasons:
        return metadata

    patched_modules: list[str] = []
    seen_module_names: set[str] = set()
    for module_obj in model.modules():
        module_name = type(module_obj).__module__
        if module_name in seen_module_names:
            continue
        seen_module_names.add(module_name)
        remote_module = sys.modules.get(module_name)
        if remote_module is None or not hasattr(remote_module, "is_fast_path_available"):
            continue
        setattr(remote_module, "is_fast_path_available", False)
        patched_modules.append(module_name)

    if hasattr(config, "use_mamba_kernels"):
        config.use_mamba_kernels = False

    metadata["applied"] = bool(patched_modules)
    metadata["patched_modules"] = patched_modules
    metadata["effective_use_mamba_kernels"] = False
    print(
        "[runtime-kernel-override] "
        + json.dumps(
            {
                "model_type": metadata["model_type"],
                "reasons": reasons,
                "patched_modules": patched_modules,
                "effective_use_mamba_kernels": False,
            },
            ensure_ascii=False,
        )
    )
    return metadata


def resolve_inference_device_map(torch_module: Any, override: str | None = None) -> Any:
    """Pick a `device_map` for inference-time `from_pretrained`.

    With no override, returns:
    - `None` when CUDA is unavailable (load on CPU).
    - A per-rank `{"": local_rank}` pin when running under torchrun
      (`WORLD_SIZE`/`LOCAL_WORLD_SIZE` > 1). DDP replicates the model anyway,
      so each rank pins to its own GPU and avoids cross-rank sharding fights.
    - `"auto"` when running single-process on multiple GPUs, so a too-large
      model (e.g. Qwen3.6-27B in bf16) is sharded across all visible GPUs
      instead of trying to fit on `cuda:0` alone.
    - `"cuda"` when only one GPU is visible.

    An explicit `override` short-circuits the heuristic. Strings like `cuda:0`
    or `cpu` are passed through unchanged; `auto`/`balanced`/`sequential` map
    to HF accelerate's named strategies.
    """
    if override:
        return override
    if not torch_module.cuda.is_available():
        return None
    world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE")
        or os.environ.get("WORLD_SIZE")
        or "1"
    )
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return {"": local_rank}
    if torch_module.cuda.device_count() > 1:
        return "auto"
    return "cuda"


GEMMA4_EMPTY_THOUGHT_PREFIX = "<|channel>thought\n<channel|>"


def tokenizer_looks_like_gemma4(tokenizer: Any) -> bool:
    name_parts = [
        getattr(tokenizer, "name_or_path", ""),
        type(tokenizer).__name__,
    ]
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    lowered = " ".join(str(part).lower() for part in name_parts)
    return ("gemma-4" in lowered or "gemma4" in lowered) and "channel" in chat_template


def maybe_mask_gemma4_empty_thought_prefix(
    tokenizer: Any,
    prompt_ids: list[int],
    completion_ids: list[int],
) -> tuple[list[int], list[int]]:
    if not tokenizer_looks_like_gemma4(tokenizer):
        return prompt_ids, completion_ids
    prefix_ids = tokenizer.encode(GEMMA4_EMPTY_THOUGHT_PREFIX, add_special_tokens=False)
    if not prefix_ids:
        return prompt_ids, completion_ids
    if prompt_ids[-len(prefix_ids) :] == prefix_ids:
        return prompt_ids, completion_ids
    if completion_ids[: len(prefix_ids)] == prefix_ids:
        return prompt_ids + prefix_ids, completion_ids[len(prefix_ids) :]
    return prompt_ids + prefix_ids, completion_ids


def build_prompt_completion_dataset(
    dataset_class: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_seq_length: int,
    min_completion_tokens: int,
    disable_chat_template: bool = False,
) -> tuple[Any, dict[str, int]]:
    dataset = dataset_class.from_list(rows)

    def preprocess(example: dict[str, Any]) -> dict[str, Any]:
        prompt_messages = [dict(message) for message in example["prompt"]]
        completion_messages = [dict(message) for message in example["completion"]]

        prompt_text = apply_chat_template_or_fallback(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
            disable_chat_template=disable_chat_template,
        )
        full_text = apply_chat_template_or_fallback(
            tokenizer,
            prompt_messages + completion_messages,
            add_generation_prompt=False,
            disable_chat_template=disable_chat_template,
        )

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_length = common_prefix_length(prompt_ids, full_ids)
        completion_ids = full_ids[prompt_length:]
        prompt_ids, completion_ids = maybe_mask_gemma4_empty_thought_prefix(
            tokenizer,
            prompt_ids[:prompt_length],
            completion_ids,
        )
        full_ids = prompt_ids + completion_ids
        prompt_length = len(prompt_ids)

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": [-100] * prompt_length + completion_ids,
            "prompt_length": prompt_length,
            "completion_length": len(completion_ids),
            "full_length": len(full_ids),
            "outcome_label": int(example.get("outcome_label", UNKNOWN_OUTCOME_LABEL)),
        }

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    original_size = len(dataset)
    dataset = dataset.filter(
        lambda example: example["full_length"] <= max_seq_length
        and example["completion_length"] >= min_completion_tokens
    )
    filtered_size = len(dataset)
    if filtered_size == 0:
        raise ValueError(
            "No valid SFT examples remain after filtering. Increase --max-seq-length "
            "or reduce the prompt size."
        )

    stats = {
        "original_examples": original_size,
        "kept_examples": filtered_size,
        "dropped_examples": original_size - filtered_size,
    }
    return dataset.remove_columns(["prompt_length", "completion_length", "full_length"]), stats


def summarize_trainable_parameters(model: Any) -> dict[str, Any]:
    total_parameters = 0
    trainable_parameters = 0
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total_parameters += count
        if parameter.requires_grad:
            trainable_parameters += count
            if len(trainable_names) < 50:
                trainable_names.append(name)
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_ratio": (trainable_parameters / total_parameters if total_parameters else 0.0),
        "trainable_parameter_name_sample": trainable_names,
    }


def summarize_parameter_gradients(model: Any, torch_module: Any) -> dict[str, Any]:
    grad_sq_sum = None
    grad_abs_max = 0.0
    tensors_with_grad = 0
    tensors_with_nonzero_grad = 0
    none_grad_trainable_tensors = 0
    nonzero_grad_name_sample: list[str] = []
    none_grad_name_sample: list[str] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        if grad is None:
            none_grad_trainable_tensors = record_missing_gradient(
                name,
                none_grad_trainable_tensors,
                none_grad_name_sample,
            )
            continue
        grad = grad.detach()
        tensors_with_grad += 1
        grad_float = grad.float()
        sq_sum = grad_float.pow(2).sum()
        grad_sq_sum = sq_sum if grad_sq_sum is None else grad_sq_sum + sq_sum
        abs_max = float(grad_float.abs().max().detach().cpu()) if grad.numel() else 0.0
        grad_abs_max = max(grad_abs_max, abs_max)
        if abs_max > 0.0:
            tensors_with_nonzero_grad += 1
            if len(nonzero_grad_name_sample) < 20:
                nonzero_grad_name_sample.append(name)

    total_grad_norm = 0.0
    if grad_sq_sum is not None:
        total_grad_norm = float(torch_module.sqrt(grad_sq_sum).detach().cpu())
    return {
        "trainable_tensors_with_grad": tensors_with_grad,
        "trainable_tensors_with_nonzero_grad": tensors_with_nonzero_grad,
        "trainable_tensors_without_grad": none_grad_trainable_tensors,
        "grad_norm": total_grad_norm,
        "grad_abs_max": grad_abs_max,
        "nonzero_grad_name_sample": nonzero_grad_name_sample,
        "none_grad_name_sample": none_grad_name_sample,
    }


def record_missing_gradient(name: str, count: int, sample: list[str]) -> int:
    if len(sample) < 20:
        sample.append(name)
    return count + 1


def validate_tokenized_dataset(dataset: Any, vocab_size: int) -> dict[str, Any]:
    max_input_id = -1
    min_input_id = 10**18
    max_label_id = -1
    min_label_id = 10**18

    for index in range(len(dataset)):
        row = dataset[index]
        input_ids = row["input_ids"]
        labels = row["labels"]

        row_max_input = max(input_ids)
        row_min_input = min(input_ids)
        max_input_id = max(max_input_id, row_max_input)
        min_input_id = min(min_input_id, row_min_input)
        if row_min_input < 0 or row_max_input >= vocab_size:
            raise ValueError(
                f"Out-of-range input id at dataset row {index}: "
                f"min={row_min_input}, max={row_max_input}, vocab_size={vocab_size}"
            )

        non_ignored_labels = [label for label in labels if label != -100]
        if not non_ignored_labels:
            raise ValueError(f"Dataset row {index} has no supervised completion tokens.")
        row_max_label = max(non_ignored_labels)
        row_min_label = min(non_ignored_labels)
        max_label_id = max(max_label_id, row_max_label)
        min_label_id = min(min_label_id, row_min_label)
        if row_min_label < 0 or row_max_label >= vocab_size:
            raise ValueError(
                f"Out-of-range label id at dataset row {index}: "
                f"min={row_min_label}, max={row_max_label}, vocab_size={vocab_size}"
            )

    return {
        "min_input_id": min_input_id,
        "max_input_id": max_input_id,
        "min_label_id": min_label_id,
        "max_label_id": max_label_id,
    }


def build_outcome_balanced_trainer_class(
    base_trainer_class: Any,
    torch_module: Any,
    class_weights: dict[int, float] | None = None,
) -> Any:
    """Trainer subclass that applies a per-row class-balanced loss.

    Each row's mean cross-entropy over its non-ignored answer tokens is multiplied
    by `class_weights[outcome_label]` before the batch is averaged. The sampler is
    left untouched, so training order is the standard shuffled order.
    """
    nn = torch_module.nn
    weights_lookup: dict[int, float] | None = None
    if class_weights:
        weights_lookup = {int(label): float(weight) for label, weight in class_weights.items()}

    class OutcomeBalancedTrainer(base_trainer_class):
        def _compute_base_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool,
            num_items_in_batch: Any = None,
        ) -> Any:
            if num_items_in_batch is None:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)
            try:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            except TypeError:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            inputs.pop("sample_weight", None)
            outcome_labels = inputs.pop("outcome_label", None)

            if (
                weights_lookup is None
                or outcome_labels is None
                or "labels" not in inputs
            ):
                return self._compute_base_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            flat_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.shape)

            mask = (shift_labels != -100).to(flat_loss.dtype)
            per_row_token_count = mask.sum(dim=1).clamp(min=1.0)
            per_row_loss = (flat_loss * mask).sum(dim=1) / per_row_token_count

            row_weight_values = [
                weights_lookup.get(int(label.item()), 1.0)
                for label in outcome_labels
            ]
            row_weights = torch_module.tensor(
                row_weight_values,
                dtype=per_row_loss.dtype,
                device=per_row_loss.device,
            )

            loss = (per_row_loss * row_weights).mean()

            if return_outputs:
                outputs.loss = loss
                return loss, outputs
            return loss

    return OutcomeBalancedTrainer


def debug_one_batch(
    torch_module: Any,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    collator = SupervisedDataCollator(tokenizer)
    batch_size = min(args.per_device_train_batch_size, len(train_dataset))
    if batch_size <= 0:
        raise ValueError("Cannot run --debug-one-batch with an empty training dataset.")

    features = [train_dataset[index] for index in range(batch_size)]
    batch = collator(features)
    model_device = next(model.parameters()).device
    batch = {key: value.to(model_device) for key, value in batch.items()}
    model_batch = {key: value for key, value in batch.items() if key in {"input_ids", "attention_mask", "labels"}}

    labels = model_batch["labels"]
    non_ignored_mask = labels != -100
    non_ignored_labels = labels[non_ignored_mask]
    debug = {
        "mode": "debug_one_batch",
        "model_device": str(model_device),
        "trainable_parameters": summarize_trainable_parameters(model),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "batch_size": batch_size,
        "input_shape": list(model_batch["input_ids"].shape),
        "attention_mask_shape": list(model_batch["attention_mask"].shape),
        "labels_shape": list(labels.shape),
        "input_id_min": int(model_batch["input_ids"].min().detach().cpu()),
        "input_id_max": int(model_batch["input_ids"].max().detach().cpu()),
        "non_ignored_label_count": int(non_ignored_mask.sum().detach().cpu()),
        "label_id_min": int(non_ignored_labels.min().detach().cpu()),
        "label_id_max": int(non_ignored_labels.max().detach().cpu()),
        "vocab_size": resolve_vocab_size(model.config),
        "embedding_rows": int(model.get_input_embeddings().weight.shape[0]),
        "lm_head_rows": int(model.get_output_embeddings().weight.shape[0]),
        "dtype": str(next(model.parameters()).dtype),
        "fp16": bool(args.fp16),
        "bf16": bool(args.bf16),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
    }
    if all("outcome_label" in feature for feature in features):
        debug["outcome_labels"] = [int(feature["outcome_label"]) for feature in features]
    if all("sample_weight" in feature for feature in features):
        sample_weights = [float(feature["sample_weight"]) for feature in features]
        debug["sample_weights"] = sample_weights
        debug["sample_weight_min"] = min(sample_weights)
        debug["sample_weight_max"] = max(sample_weights)

    model.zero_grad(set_to_none=True)
    model.train()

    try:
        outputs = model(**model_batch)
        loss = outputs.loss
        debug["forward_ok"] = True
        debug["loss"] = float(loss.detach().cpu())
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
        loss.backward()
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
        debug["backward_ok"] = True
        debug["gradients"] = summarize_parameter_gradients(model, torch_module)
    except Exception as exc:
        debug["forward_ok"] = debug.get("forward_ok", False)
        debug["backward_ok"] = False
        debug["error_type"] = type(exc).__name__
        debug["error_message"] = str(exc)
        raise RuntimeError(json.dumps(debug, ensure_ascii=False, indent=2)) from exc
    finally:
        model.zero_grad(set_to_none=True)

    return debug





def train_world_model(
    args: argparse.Namespace,
    train_examples: list[WorldModelStateExample],
    eval_examples: list[WorldModelStateExample],
) -> dict[str, Any]:
    torch, Dataset, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed = require_training_stack()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    lora_config = build_lora_config(args)

    model_cls = resolve_text_generation_model_class(
        args.model,
        trust_remote_code=args.trust_remote_code,
        causal_lm_class=AutoModelForCausalLM,
    )
    model = model_cls.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=resolve_torch_dtype(torch, args.dtype),
        device_map=resolve_training_device_map(torch),
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    runtime_kernel_overrides = maybe_disable_nemotron_fast_mamba_kernels(
        model,
        torch_module=torch,
        gradient_checkpointing=args.gradient_checkpointing,
        allow_fast_mamba_kernels=args.allow_fast_mamba_kernels,
    )
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_rows = [
        build_state_prediction_prompt_completion_row(
            example,
            target_mode=args.world_model_target,
            include_error_message=args.include_error_message_in_target,
            include_stage=args.include_stage_in_target,
            include_input_history=args.include_world_model_history,
            system_prompt_max_chars=args.world_model_system_prompt_max_chars,
            action_max_chars=args.world_model_action_max_chars,
        )
        for example in train_examples
    ]
    eval_rows = [
        build_state_prediction_prompt_completion_row(
            example,
            target_mode=args.world_model_target,
            include_error_message=args.include_error_message_in_target,
            include_stage=args.include_stage_in_target,
            include_input_history=args.include_world_model_history,
            system_prompt_max_chars=args.world_model_system_prompt_max_chars,
            action_max_chars=args.world_model_action_max_chars,
        )
        for example in eval_examples
    ]

    if args.oversample_minority_outcomes:
        train_rows, oversample_metadata = oversample_rows_by_outcome(
            train_rows,
            target_mode=args.world_model_target,
            target_ratio=args.oversample_target_ratio,
            max_multiplier=args.oversample_max_multiplier,
            seed=args.seed,
        )
        print(f"[oversample] {json.dumps(oversample_metadata, ensure_ascii=False)}")
    else:
        oversample_metadata = {"applied": False, "reason": "flag_disabled"}

    train_dataset, train_dataset_stats = build_prompt_completion_dataset(
        Dataset,
        tokenizer,
        train_rows,
        max_seq_length=args.max_seq_length,
        min_completion_tokens=args.min_completion_tokens,
        disable_chat_template=args.disable_chat_template,
    )
    eval_dataset = None
    eval_dataset_stats = None
    if eval_rows:
        eval_dataset, eval_dataset_stats = build_prompt_completion_dataset(
            Dataset,
            tokenizer,
            eval_rows,
            max_seq_length=args.max_seq_length,
            min_completion_tokens=args.min_completion_tokens,
            disable_chat_template=args.disable_chat_template,
        )
    model_vocab_size = resolve_vocab_size(model.config)
    train_dataset_validation = validate_tokenized_dataset(train_dataset, model_vocab_size)
    eval_dataset_validation = (
        validate_tokenized_dataset(eval_dataset, model_vocab_size) if eval_dataset is not None else None
    )
    train_dataset, class_loss_weights, outcome_loss_balance = maybe_apply_outcome_loss_balance(
        train_dataset,
        strategy=args.outcome_balance_loss,
        beta=args.outcome_balance_beta,
        target_mode=args.world_model_target,
    )
    print(f"[outcome-balance-loss] {json.dumps(outcome_loss_balance, ensure_ascii=False)}")
    train_outcome_counts = summarize_outcome_label_counts(
        train_dataset["outcome_label"],
        target_mode=args.world_model_target,
    )
    eval_outcome_counts = (
        summarize_outcome_label_counts(
            eval_dataset["outcome_label"],
            target_mode=args.world_model_target,
        )
        if eval_dataset is not None
        else summarize_outcome_label_counts([], target_mode=args.world_model_target)
    )

    training_kwargs = dict(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",
        save_strategy="steps",
        report_to=[],
        remove_unused_columns=False,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=1.0,
        optim="adamw_torch",
    )
    if "gradient_checkpointing_kwargs" in inspect.signature(TrainingArguments.__init__).parameters:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    training_args = TrainingArguments(**training_kwargs)
    if lora_config is not None:
        from peft import get_peft_model

        model = get_peft_model(model, lora_config)
        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainable_parameter_summary = summarize_trainable_parameters(model)
    print(f"[trainable-parameters] {json.dumps(trainable_parameter_summary, ensure_ascii=False)}")
    if trainable_parameter_summary["trainable_parameters"] <= 0:
        raise ValueError(
            "No trainable parameters found after model/LoRA setup. "
            "Check --use-lora and --lora-target-modules for this model."
        )

    if args.debug_one_batch:
        try:
            debug_metrics = debug_one_batch(
                torch,
                model,
                tokenizer,
                train_dataset,
                args,
            )
        except RuntimeError as exc:
            try:
                error_payload = json.loads(str(exc))
            except json.JSONDecodeError:
                raise
            metrics = {
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
                "train_dataset_stats": train_dataset_stats,
                "eval_dataset_stats": eval_dataset_stats,
                "train_dataset_validation": train_dataset_validation,
                "eval_dataset_validation": eval_dataset_validation,
                "world_model_target": args.world_model_target,
                "outcome_balance_loss": args.outcome_balance_loss,
                "outcome_loss_balance": outcome_loss_balance,
                "train_outcome_counts": train_outcome_counts,
                "eval_outcome_counts": eval_outcome_counts,
                "attn_implementation": args.attn_implementation,
                "runtime_kernel_overrides": runtime_kernel_overrides,
                "trainable_parameters": trainable_parameter_summary,
                "debug_one_batch": error_payload,
            }
            dump_json(args.output_dir / "debug_one_batch.json", metrics)
            raise SystemExit(json.dumps(metrics, indent=2, ensure_ascii=False))

        metrics = {
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "train_dataset_stats": train_dataset_stats,
            "eval_dataset_stats": eval_dataset_stats,
            "train_dataset_validation": train_dataset_validation,
            "eval_dataset_validation": eval_dataset_validation,
            "world_model_target": args.world_model_target,
            "outcome_balance_loss": args.outcome_balance_loss,
            "outcome_loss_balance": outcome_loss_balance,
            "oversample_outcomes": oversample_metadata,
            "train_outcome_counts": train_outcome_counts,
            "eval_outcome_counts": eval_outcome_counts,
            "attn_implementation": args.attn_implementation,
            "runtime_kernel_overrides": runtime_kernel_overrides,
            "trainable_parameters": trainable_parameter_summary,
            "debug_one_batch": debug_metrics,
        }
        dump_json(args.output_dir / "debug_one_batch.json", metrics)
        return metrics

    TrainerClass = build_outcome_balanced_trainer_class(
        Trainer, torch, class_weights=class_loss_weights
    )
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedDataCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    eval_metrics = trainer.evaluate() if eval_dataset is not None else {}
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics = {
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "train_dataset_stats": train_dataset_stats,
        "eval_dataset_stats": eval_dataset_stats,
        "train_dataset_validation": train_dataset_validation,
        "eval_dataset_validation": eval_dataset_validation,
        "world_model_target": args.world_model_target,
        "outcome_balance_loss": args.outcome_balance_loss,
        "outcome_loss_balance": outcome_loss_balance,
        "train_outcome_counts": train_outcome_counts,
        "eval_outcome_counts": eval_outcome_counts,
        "attn_implementation": args.attn_implementation,
        "runtime_kernel_overrides": runtime_kernel_overrides,
        "trainable_parameters": trainable_parameter_summary,
        "use_lora": args.use_lora,
        "lora_config": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "bias": args.lora_bias,
            "target_modules": (
                default_lora_target_modules(args.model)
                if args.lora_target_modules == "auto"
                else parse_csv_arg(args.lora_target_modules)
            ) if args.use_lora else None,
            "modules_to_save": parse_csv_arg(args.lora_modules_to_save) if args.use_lora else None,
        },
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    dump_json(args.output_dir / "training_metrics.json", metrics)
    return metrics





def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_trajectories: list[dict[str, Any]] = []
    for path in args.train_data_path:
        train_trajectories.extend(normalize_loaded_trajectories(load_trajectory_records(path)))

    eval_trajectories: list[dict[str, Any]] = []
    for path in args.eval_data_path:
        eval_trajectories.extend(normalize_loaded_trajectories(load_trajectory_records(path)))

    train_split_stats = build_trajectory_outcome_stats(
        train_trajectories, ",".join(str(p) for p in args.train_data_path)
    )
    eval_split_stats = build_trajectory_outcome_stats(
        eval_trajectories, ",".join(str(p) for p in args.eval_data_path)
    )
    train_trajectories = [item.trajectory for item in train_split_stats]
    eval_trajectories = [item.trajectory for item in eval_split_stats]

    train_examples = extract_state_examples(
        train_trajectories,
        state_history_size=args.state_history_size,
    )
    eval_examples = extract_state_examples(
        eval_trajectories,
        state_history_size=args.state_history_size,
    )
    train_outcome_summary = summarize_example_outcomes(train_examples)
    eval_outcome_summary = summarize_example_outcomes(eval_examples)

    train_prompt_completion_rows = [
        build_state_prediction_prompt_completion_row(
            example,
            target_mode=args.world_model_target,
            include_error_message=args.include_error_message_in_target,
            include_stage=args.include_stage_in_target,
            include_input_history=args.include_world_model_history,
            system_prompt_max_chars=args.world_model_system_prompt_max_chars,
            action_max_chars=args.world_model_action_max_chars,
        )
        for example in train_examples
    ]
    eval_prompt_completion_rows = [
        build_state_prediction_prompt_completion_row(
            example,
            target_mode=args.world_model_target,
            include_error_message=args.include_error_message_in_target,
            include_stage=args.include_stage_in_target,
            include_input_history=args.include_world_model_history,
            system_prompt_max_chars=args.world_model_system_prompt_max_chars,
            action_max_chars=args.world_model_action_max_chars,
        )
        for example in eval_examples
    ]

    split_manifest = {
        "source_data_paths": [str(p) for p in args.train_data_path]
        + [str(p) for p in args.eval_data_path],
        "seed": args.seed,
        "split_strategy": "use_provided_train_eval_files",
        "world_model_target": args.world_model_target,
        "state_history_size": args.state_history_size,
        "include_error_message_in_target": args.include_error_message_in_target,
        "include_stage_in_target": args.include_stage_in_target,
        "include_world_model_history": args.include_world_model_history,
        "source_trajectory_count": len(train_split_stats) + len(eval_split_stats),
        "train_trajectory_count": len(train_trajectories),
        "eval_trajectory_count": len(eval_trajectories),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "train_example_outcomes": train_outcome_summary,
        "eval_example_outcomes": eval_outcome_summary,
        "train_prompt_completion_examples": len(train_prompt_completion_rows),
        "eval_prompt_completion_examples": len(eval_prompt_completion_rows),
    }
    dump_json(args.output_dir / "split_manifest.json", split_manifest)
    dump_jsonl(args.output_dir / "train_examples.jsonl", (asdict(example) for example in train_examples))
    dump_jsonl(args.output_dir / "eval_examples.jsonl", (asdict(example) for example in eval_examples))
    dump_jsonl(args.output_dir / "train_prompt_completion.jsonl", train_prompt_completion_rows)
    dump_jsonl(args.output_dir / "eval_prompt_completion.jsonl", eval_prompt_completion_rows)

    training_metrics = train_world_model(args, train_examples, eval_examples)
    summary = {"training_metrics": training_metrics}
    dump_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
