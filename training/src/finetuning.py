#!/usr/bin/env python3
"""Train and evaluate a world model from reconstructed state trajectories.

Each supervised example uses the system prompt, user prompt, and action as
input, and predicts the resulting state. `last_tool_output` is removed from the
target state because it is too variable for useful stage-focused prediction.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import ast
import datetime
import gc
import hashlib
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
import torch
import transformers
import datasets

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as `uv run python src/finetuning.py ...` only `src/` lands on sys.path, so the repo-wide
# `from src.<x> import ...` convention needs the root added explicitly (other scripts already
# do this; this module previously had no top-level `src.` import and so never needed it).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_TRAJECTORIES_DIR = REPO_ROOT / "trajectories"
DEFAULT_TRAIN_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "world_model_train_trajectories.json"
DEFAULT_EVAL_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "world_model_test_trajectories.json"
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
DEFAULT_ENTERPRISEOPS_GYM_TASK_SPLIT_MANIFEST = (
    DEFAULT_TRAJECTORIES_DIR / "enterpriseops_gym_80_test_task_split.json"
)
TERMINALBENCH_2_0_LLM_SHELL_TRAIN_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "terminalbench_2_0_llm_shell_train_trajectories.json"
)
TERMINALBENCH_2_0_LLM_SHELL_EVAL_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "terminalbench_2_0_llm_shell_test_trajectories.json"
)
CRMARENAPRO_BASELINE_CRM_AGENT_TRAIN_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "crmarenapro_baseline_crm_agent_train_results.json"
)
CRMARENAPRO_BASELINE_CRM_AGENT_EVAL_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "crmarenapro_baseline_crm_agent_test_results.json"
)
TOUCAN_TRAIN_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "toucan_world_model_train_trajectories.json"
)
TOUCAN_EVAL_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "toucan_world_model_test_trajectories.json"
)
# ADP (Agent Data Protocol) benchmarks, shared verbatim with the JEPA trainer via
# src/adp_datasets.py so both world models can be trained/compared on the same corpora.
# Phase-1 use (`--world-model-target tool_output`) is what these are for here: ADP carries
# real tool observations at a scale the enterprise benchmarks do not.
from src.canonical_event_schema import (  # noqa: E402
    CANONICAL_EVENT_ALLOWED_VALUES,
    DEFAULT_CANONICAL_EVENT_EVAL_JSONL,
    DEFAULT_CANONICAL_EVENT_TRAIN_JSONL,
    NUDGE_MULTI_LABEL_FIELDS,
    split_canonical_labels,
)
from src.adp_datasets import (  # noqa: E402
    ADP_ALL_EVAL_DATA_PATHS,
    ADP_ALL_TRAIN_DATA_PATHS,
    TOUCAN_ENTERPRISE_DATA_PATH,
    ADP_NO_TERMINAL_EVAL_DATA_PATHS,
    ADP_NO_TERMINAL_TRAIN_DATA_PATHS,
    TOUCAN_1_5M_MULTITURN_DATA_PATH,
    ADP_TRAJECTORY_DATASETS,
    ADP_TRAJECTORY_PATHS,
    WEB_BROWSING_TRAJECTORY_PATHS,
    ADP_ALL_25K_DATA_PATHS,
    ADP_NO_TERMINAL_25K_DATA_PATHS,
    WEB_BROWSING_25K_PATHS,
    subsampled_25k_path,
    enterprise_tool_calling_paths,
)

TRAJECTORY_DATASET_PRESETS: dict[str, tuple[list[Path], list[Path]]] = {
    "enterprise_arena": ([DEFAULT_TRAIN_DATA_PATH], [DEFAULT_EVAL_DATA_PATH]),
    "enterpriseops_gym": (
        [ENTERPRISEOPS_GYM_TRAIN_DATA_PATH],
        [ENTERPRISEOPS_GYM_EVAL_DATA_PATH],
    ),
    "enterpriseops_gym_enterprise_state": (
        [ENTERPRISEOPS_GYM_ENTERPRISE_STATE_TRAIN_DATA_PATH],
        [ENTERPRISEOPS_GYM_ENTERPRISE_STATE_EVAL_DATA_PATH],
    ),
    # `toucan` = the enterprise-filtered extract; the legacy curated split is `toucan_curated`.
    # Loaded from the single combined toucan_enterprise_world_model_trajectories.json -- the
    # train/test split files are no longer used by the preset.
    "toucan": ([TOUCAN_ENTERPRISE_DATA_PATH], [TOUCAN_ENTERPRISE_DATA_PATH]),
    "toucan_enterprise": ([TOUCAN_ENTERPRISE_DATA_PATH], [TOUCAN_ENTERPRISE_DATA_PATH]),
    "toucan_curated": ([TOUCAN_TRAIN_DATA_PATH], [TOUCAN_EVAL_DATA_PATH]),
    "terminalbench_2_0_llm_shell": (
        [TERMINALBENCH_2_0_LLM_SHELL_TRAIN_DATA_PATH],
        [TERMINALBENCH_2_0_LLM_SHELL_EVAL_DATA_PATH],
    ),
    "crmarenapro_baseline_crm_agent": (
        [CRMARENAPRO_BASELINE_CRM_AGENT_TRAIN_DATA_PATH],
        [CRMARENAPRO_BASELINE_CRM_AGENT_EVAL_DATA_PATH],
    ),
    "all": (
        [
            DEFAULT_TRAIN_DATA_PATH,
            ENTERPRISEOPS_GYM_TRAIN_DATA_PATH,
            TOUCAN_TRAIN_DATA_PATH,
            TERMINALBENCH_2_0_LLM_SHELL_TRAIN_DATA_PATH,
            CRMARENAPRO_BASELINE_CRM_AGENT_TRAIN_DATA_PATH,
        ],
        [
            DEFAULT_EVAL_DATA_PATH,
            ENTERPRISEOPS_GYM_EVAL_DATA_PATH,
            TOUCAN_EVAL_DATA_PATH,
            TERMINALBENCH_2_0_LLM_SHELL_EVAL_DATA_PATH,
            CRMARENAPRO_BASELINE_CRM_AGENT_EVAL_DATA_PATH,
        ],
    ),
    # Mirrors finetuning_jepa.py's `all_no_terminalbench`: the enterprise benchmarks plus
    # every ADP corpus, with the terminal DOMAIN removed -- both Terminal-Bench itself and the
    # terminal-centric ADP subset (nemotron_terminal_corpus), since leaving the latter in would
    # reintroduce exactly the distribution the ablation is meant to remove. Note this drops
    # `enterprise_arena` too, matching the JEPA trainer's CORE set (gym + CRMArenaPro).
    "all_no_terminalbench": (
        [
            ENTERPRISEOPS_GYM_TRAIN_DATA_PATH,
            CRMARENAPRO_BASELINE_CRM_AGENT_TRAIN_DATA_PATH,
            *ADP_NO_TERMINAL_TRAIN_DATA_PATHS,
        ],
        [
            ENTERPRISEOPS_GYM_EVAL_DATA_PATH,
            CRMARENAPRO_BASELINE_CRM_AGENT_EVAL_DATA_PATH,
            *ADP_NO_TERMINAL_EVAL_DATA_PATHS,
        ],
    ),
    # --- ADP presets (same inventory as the JEPA trainer) -------------------------------
    # Each ADP benchmark is one combined file with no official split, so train and eval
    # reuse it; pass --train-data-path/--eval-data-path for a held-out split.
    "adp_all": (list(ADP_ALL_TRAIN_DATA_PATHS), list(ADP_ALL_EVAL_DATA_PATHS)),
    "adp_no_terminal": (list(ADP_NO_TERMINAL_TRAIN_DATA_PATHS), list(ADP_NO_TERMINAL_EVAL_DATA_PATHS)),
    # Enterprise benchmarks + every ADP corpus -- the phase-1 pretraining mixture.
    "enterprise_and_adp": (
        [
            DEFAULT_TRAIN_DATA_PATH,
            ENTERPRISEOPS_GYM_TRAIN_DATA_PATH,
            TERMINALBENCH_2_0_LLM_SHELL_TRAIN_DATA_PATH,
            CRMARENAPRO_BASELINE_CRM_AGENT_TRAIN_DATA_PATH,
            *ADP_ALL_TRAIN_DATA_PATHS,
        ],
        [
            DEFAULT_EVAL_DATA_PATH,
            ENTERPRISEOPS_GYM_EVAL_DATA_PATH,
            TERMINALBENCH_2_0_LLM_SHELL_EVAL_DATA_PATH,
            CRMARENAPRO_BASELINE_CRM_AGENT_EVAL_DATA_PATH,
            *ADP_ALL_EVAL_DATA_PATHS,
        ],
    ),
    "toucan_1_5m_multiturn": ([TOUCAN_1_5M_MULTITURN_DATA_PATH], [TOUCAN_1_5M_MULTITURN_DATA_PATH]),
    # See finetuning_jepa.py: same coverage, heavy corpora swapped for 25k seeded subsets.
    "adp_all_25k": (list(ADP_ALL_25K_DATA_PATHS), list(ADP_ALL_25K_DATA_PATHS)),
    "adp_no_terminal_25k": (list(ADP_NO_TERMINAL_25K_DATA_PATHS), list(ADP_NO_TERMINAL_25K_DATA_PATHS)),
    "all_no_terminalbench_25k": (
        [
            ENTERPRISEOPS_GYM_TRAIN_DATA_PATH,
            CRMARENAPRO_BASELINE_CRM_AGENT_TRAIN_DATA_PATH,
            *ADP_NO_TERMINAL_25K_DATA_PATHS,
        ],
        [
            ENTERPRISEOPS_GYM_EVAL_DATA_PATH,
            CRMARENAPRO_BASELINE_CRM_AGENT_EVAL_DATA_PATH,
            *ADP_NO_TERMINAL_25K_DATA_PATHS,
        ],
    ),
    # See finetuning_jepa.py: downstream-targeted mixtures, with/without the SWE/code arm.
    **{
        _preset_name: (
            [ENTERPRISEOPS_GYM_TRAIN_DATA_PATH, CRMARENAPRO_BASELINE_CRM_AGENT_TRAIN_DATA_PATH,
             *enterprise_tool_calling_paths(include_swe=_swe, subsampled=_sub)],
            [ENTERPRISEOPS_GYM_EVAL_DATA_PATH, CRMARENAPRO_BASELINE_CRM_AGENT_EVAL_DATA_PATH,
             *enterprise_tool_calling_paths(include_swe=_swe, subsampled=_sub)],
        )
        for _preset_name, _swe, _sub in (
            ("enterprise_tool_calling", False, False),
            ("enterprise_tool_calling_plus_swe", True, False),
            ("enterprise_tool_calling_25k", False, True),
            ("enterprise_tool_calling_plus_swe_25k", True, True),
        )
    },
}

# One preset per individual ADP benchmark, e.g. --trajectory-dataset toolmind.
for _name in ADP_TRAJECTORY_DATASETS:
    TRAJECTORY_DATASET_PRESETS.setdefault(
        _name, ([ADP_TRAJECTORY_PATHS[_name]], [ADP_TRAJECTORY_PATHS[_name]])
    )
    _sub = subsampled_25k_path(_name)
    if _sub != ADP_TRAJECTORY_PATHS[_name]:
        TRAJECTORY_DATASET_PRESETS.setdefault(f"{_name}_25k", ([_sub], [_sub]))
del _name
DEFAULT_ENTERPRISE_RUNNER = (
    Path.home()
    / "program"
    / "tools"
    / "EnterpriseLab"
    / "Evaluate"
    / "EnterpriseArena"
    / "Interactive_mcp_localM.py"
)
API_AGENT_MODEL_METHODS = {"gemini", "claude", "vllm/nemotron3-nano-4B-BF16", "vllm/qwen3-8b", "vllm/gymops_world_model"}


def looks_like_vllm_agent_model(model_path: str) -> bool:
    """True for `vllm/<served-model-name>` and `vllm:<port>/<served-model-name>`.

    Any vLLM-served model routes to the src.llm wrapper (LLMTextGenerator) rather than
    needing an entry in API_AGENT_MODEL_METHODS, so a freshly served checkpoint works
    without editing this file. The optional `:<port>` selects the local port (default
    9000); $VLLM_ENDPOINT overrides the whole URL.
    """
    prefix = str(model_path).split("/", 1)[0]
    return prefix.split(":", 1)[0] == "vllm" and "/" in str(model_path)
OPENAI_AGENT_MODEL_ALIASES = {
    "gpt5": "gpt-5",
    "gpt-5": "gpt-5",
    "openai/gpt5": "gpt-5",
    "openai:gpt5": "gpt-5",
    "openai/gpt-5": "gpt-5",
    "openai:gpt-5": "gpt-5",
    "gpt5.1": "gpt-5.1",
    "gpt-5.1": "gpt-5.1",
    "openai/gpt5.1": "gpt-5.1",
    "openai:gpt5.1": "gpt-5.1",
    "openai/gpt-5.1": "gpt-5.1",
    "openai:gpt-5.1": "gpt-5.1",
}
LLM_AGENT_MODEL_ALIASES: dict[str, str] = {}
CSM_MCP_SERVER_NAME = "sn-csm-server"
CSM_MCP_DEFAULT_URL = "http://localhost:8001"
CSM_MCP_OVERRIDE_URL = "http://localhost:8001"
HR_MCP_SERVER_NAME = "sn-hr-internal"
HR_MCP_DEFAULT_URL = "http://localhost:8008"
HR_MCP_OVERRIDE_URL = "http://localhost:8010"
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
WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY = "tool_execution_result_success_failure"
WORLD_MODEL_TARGET_TOOL_OUTPUT = "tool_output"
# Phase-2 target: canonical_event_with_nudge annotation, emitted as JSON. For the LLM
# world model this is intentionally narrowed to the fields used by beam planning:
# action_type for action semantics, the five scored fields, plus a binary terminal field
# aligned with the JEPA terminal predictor.
WORLD_MODEL_TARGET_CANONICAL_EVENT = "canonical_event_with_nudge"
CANONICAL_EVENT_BEAM_TARGET_FIELDS: tuple[str, ...] = (
    "action_type",
    "execution_status",
    "progress_signal",
    "information_sufficiency",
    "error_signature",
    "side_effect_type",
)
CANONICAL_EVENT_TERMINAL_FIELD = "terminal"
CANONICAL_EVENT_TERMINAL_VALUES: tuple[str, str] = ("not_finished", "finished")
CANONICAL_EVENT_LLM_TARGET_FIELDS: tuple[str, ...] = (
    CANONICAL_EVENT_BEAM_TARGET_FIELDS + (CANONICAL_EVENT_TERMINAL_FIELD,)
)
LEGACY_WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY = "tool_execution_result_binary"


def canonicalize_world_model_target(target_mode: str) -> str:
    if target_mode == LEGACY_WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY:
        return WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY
    return target_mode


def is_tool_execution_result_target(target_mode: str) -> bool:
    return canonicalize_world_model_target(target_mode) in {
        WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
        WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY,
    }


def is_tool_output_target(target_mode: str) -> bool:
    return canonicalize_world_model_target(target_mode) == WORLD_MODEL_TARGET_TOOL_OUTPUT


def is_canonical_event_target(target_mode: str) -> bool:
    return canonicalize_world_model_target(target_mode) == WORLD_MODEL_TARGET_CANONICAL_EVENT


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
    previous_state: dict[str, Any]
    state: dict[str, Any]
    error_payload: str = ""
    tool_output: str = ""


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
    initial_canonical_observation: dict[str, Any] | None = None
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
    parser = argparse.ArgumentParser(description="Train/evaluate the EnterpriseArena world model.")
    parser.add_argument("--model", nargs="?", default="Qwen/Qwen3-4B", help="Base Hugging Face model for SFT and, by default, agent evaluation.")
    parser.add_argument(
        "--agent-model",
        help=(
            "Optional model for tool-call generation. Defaults to `model`. "
            "Supports HF checkpoints, the legacy LLM-wrapper methods "
            "(`gpt5`, `gpt-5.1`, `gemini`, `claude`, `vllm/<served-name>`, "
            "`vllm:<port>/<served-name>`), direct OpenAI SDK "
            "models, Azure OpenAI deployments, and Gemini API models. OpenAI "
            "models can be passed either bare (e.g. `gpt-4o`, `gpt-4o-mini`, "
            "`gpt-5`, `gpt-5.1`, `gpt-5-mini`, `o1-mini`, `o3-mini`) or with an explicit "
            "`openai/<model>` prefix when the bare name might collide with "
            "another path. Azure OpenAI uses `azureopenai/<deployment>` and "
            "reads `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and "
            "`AZURE_OPENAI_API_VERSION` (unless the endpoint is already an "
            "`.../openai/v1/` base URL). Gemini uses `gemini/<model>` or bare "
            "versioned names like `gemini-2.5-pro`, and reads "
            "`GEMINI_API_KEY`."
        ),
    )
    parser.add_argument("--world-model-path", help="Evaluate an existing fine-tuned checkpoint instead of training.")
    parser.add_argument(
        "--world-model-method",
        default=None,
        help=(
            "Override the world-model generator backend. Same syntax as "
            "`--agent-model`: HF path (default), `openai/<model>`, "
            "`vllm/<served-name>` or `vllm:<port>/<served-name>` "
            "to hit a local vLLM (port 9000 by default, or $VLLM_ENDPOINT), "
            "`azureopenai/<deployment>`, `gemini/<model>`, or an entry in "
            "API_AGENT_MODEL_METHODS. When set, world-model inference goes "
            "through the same dispatch as the agent generator instead of "
            "loading weights locally via `HFTextGenerator`."
        ),
    )
    parser.add_argument(
        "--trajectory-dataset",
        choices=sorted(TRAJECTORY_DATASET_PRESETS.keys()),
        default="enterprise_arena",
        help=(
            "Trajectory dataset preset. `enterprise_arena` (default) uses "
            "trajectories/world_model_{train,test}_trajectories.json. `enterpriseops_gym` uses the "
            "EnterpriseOps-Gym variant, `enterpriseops_gym_enterprise_state` uses the diff-style "
            "EnterpriseOps-Gym enterprise-state schema, `toucan` uses the TOUCAN multi-turn variant, "
            "`terminalbench_2_0_llm_shell` uses Terminal-Bench shell trajectories, "
            "`crmarenapro_baseline_crm_agent` uses normalized CRMArenaPro CRM result trajectories, "
            "and `all` concatenates all major enterprise sources. "
            "ADP: `adp_all` selects every ADP benchmark, `adp_no_terminal` drops the "
            "terminal/shell corpora, `enterprise_and_adp` mixes the enterprise benchmarks with "
            "all of ADP, and each ADP benchmark is selectable by its own name (e.g. `toolmind`, "
            "`toucan_1_5m_multiturn`, `swe-smith`) -- the same inventory the JEPA trainer uses, "
            "from src/adp_datasets.py. ADP files have no official split, so their presets reuse "
            "one file for train and eval; pass explicit paths for a held-out split. "
            "Explicit `--train-data-path` / `--eval-data-path` override the preset."
        ),
    )
    parser.add_argument(
        "--train-data-path",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more train trajectory JSON paths. Pass multiple paths to concatenate "
            "(e.g. EnterpriseArena + EnterpriseOps-Gym + TOUCAN + Terminal-Bench + CRMArenaPro). Defaults to the paths implied "
            "by `--trajectory-dataset`."
        ),
    )
    parser.add_argument(
        "--eval-data-path",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more eval trajectory JSON paths. Pass multiple paths to concatenate. "
            "Defaults to the paths implied by `--trajectory-dataset`."
        ),
    )
    parser.add_argument(
        "--skip-web-trajectories",
        action="store_true",
        help=(
            "Skip loading the web-browsing ADP trajectory files (agenttuning_mind2web, "
            "go-browse-wa, mind2web, nnetnav-live, nnetnav-wa, synatra, mini-coder, "
            "coderforge_preview, nemotron_terminal_corpus), even if they were selected via "
            "--trajectory-dataset (e.g. all/adp_all) or explicit --train-data-path/"
            "--eval-data-path. Same flag, same file set, and same filtering point as "
            "src/finetuning_jepa.py, so the two trainers select identical corpora for a given "
            "preset. Use this to cut load time/memory when these large datasets aren't needed."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default="data", help="Directory for checkpoints and metrics.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for training reproducibility.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Deprecated and ignored. The provided train/eval trajectory files are consumed directly without reshuffling.",
    )
    parser.add_argument("--max-seq-length", type=int, default=8192, help="SFT max sequence length.")
    parser.add_argument(
        "--min-completion-tokens",
        type=int,
        default=1,
        help="Minimum number of completion tokens kept for prompt-completion SFT examples.",
    )
    parser.add_argument(
        "--tokenize-num-proc",
        type=int,
        default=0,
        help=(
            "Worker processes for the tokenization map. 0 (default) auto-picks "
            "min(8, cpu_count - 2); 1 forces in-process tokenization (the batched "
            "fast tokenizer still threads internally)."
        ),
    )
    parser.add_argument(
        "--tokenize-batch-size",
        type=int,
        default=500,
        help="Rows per batched tokenizer call during the tokenization map.",
    )
    parser.add_argument(
        "--tokenize-prefilter-chars-per-token",
        type=float,
        default=12.0,
        help=(
            "Drop rows whose raw character count exceeds --max-seq-length times this bound "
            "before rendering/encoding them (they would be dropped by the length filter "
            "anyway). 0 disables the pre-filter."
        ),
    )
    parser.add_argument(
        "--tokenized-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the tokenized-dataset cache (default: <output-dir>/tokenized_cache). "
            "Rank 0 tokenizes once and the other ranks memory-map the result."
        ),
    )
    parser.add_argument(
        "--no-tokenized-cache",
        action="store_true",
        help="Disable the tokenized-dataset cache and tokenize in every process.",
    )
    parser.add_argument(
        "--rebuild-tokenized-cache",
        action="store_true",
        help="Ignore any existing tokenized-dataset cache entry and rebuild it.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=5, help="Number of SFT epochs.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="SFT learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="SFT weight decay.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1, help="Per-device eval batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="SFT gradient accumulation.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Trainer logging cadence.")
    parser.add_argument("--eval-steps", type=int, default=100, help="Trainer eval cadence when eval data exists.")
    parser.add_argument("--save-steps", type=int, default=100, help="Trainer checkpoint cadence.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Maximum retained checkpoints.")
    parser.add_argument("--resume-from-checkpoint", help="Optional checkpoint to resume SFT from.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loaders.")
    parser.add_argument(
        "--allow-fast-mamba-kernels",
        action="store_true",
        help=(
            "Keep Nemotron-style fused Mamba kernels enabled during training even "
            "when gradient checkpointing or single-process multi-GPU training would "
            "otherwise force the safer PyTorch fallback."
        ),
    )
    parser.add_argument("--skip-training", action="store_true", help="Skip SFT and only run evaluation.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip offline evaluation.")
    parser.add_argument(
        "--debug-one-batch",
        action="store_true",
        help="Build the first training batch, run one forward/backward pass, dump debug stats, and exit training early.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation cap for inference-time decoding.")
    parser.add_argument(
        "--record-replay-trajectories",
        action="store_true",
        help=(
            "Record the per-task agent-replay conversations (one record per task per mode) "
            "and dump them to trajectories/<mode_name>_replay_trajectories.json. Off by default; "
            "the resulting file can be large for long runs because tool outputs are kept verbatim."
        ),
    )
    parser.add_argument(
        "--agent-max-observation-chars",
        type=int,
        default=2000,
        help=(
            "Per-tool-output character cap applied when the conversation is rendered into a "
            "replay prompt. Tool outputs longer than this are truncated with a marker so the "
            "input prompt stays well under the model's context window."
        ),
    )
    parser.add_argument(
        "--agent-replay-history-budget-chars",
        type=int,
        default=60000,
        help=(
            "Soft total-character budget for the replay prompt. When exceeded, the oldest "
            "thought/action/observation turns are dropped (system prompt and first user query "
            "are always preserved) and a `[CONTEXT TRIMMED]` notice is inserted. Roughly 4 chars "
            "per token, so 60000 chars ≈ 15k tokens — leaves room for output and overhead in a "
            "40k context window."
        ),
    )
    parser.add_argument("--dtype", default="auto", help="Model dtype for loading and training. Example: auto, bfloat16, float16, float32.")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend used when loading the model.",
    )
    parser.add_argument(
        "--inference-device-map",
        default=None,
        help=(
            "device_map passed to HF `from_pretrained` for inference-time loading "
            "(world model + HF agent). Defaults to `auto` (shard across all visible "
            "GPUs) when running outside torchrun, and to `cuda:LOCAL_RANK` when "
            "inside torchrun. Set explicitly to `auto`, `balanced`, `cpu`, or a "
            "single device like `cuda:0` to override."
        ),
    )
    parser.add_argument("--disable-chat-template", action="store_true", help="Disable tokenizer chat-template formatting even if the model provides one.")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA adapter finetuning via PEFT.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")
    parser.add_argument("--lora-bias", default="none", choices=["none", "all", "lora_only"], help="PEFT LoRA bias mode.")
    parser.add_argument(
        "--lora-target-modules",
        default="auto",
    )
    parser.add_argument(
        "--lora-modules-to-save",
        default="",
        help="Optional comma-separated module names to keep trainable alongside LoRA adapters.",
    )
    parser.add_argument(
        "--world-model-eval-dump-predictions",
        type=Path,
        default=None,
        help=(
            "Where to write per-example predicted/gold canonical-event labels "
            "(default: <output-dir>/eval_predictions_per_example.jsonl). Same format as the "
            "JEPA head eval's dump, so src/analysis/canonical_event_prediction_report.py can "
            "compare LLM and JEPA checkpoints without re-running either."
        ),
    )
    parser.add_argument(
        "--world-model-eval-batch-size",
        type=int,
        default=8,
        help="Rows per batched generation in the canonical-event generative eval.",
    )
    parser.add_argument("--world-model-eval-samples", type=int, default=2000, help="How many test examples to evaluate for world-model prediction. Set <=0 to evaluate every test example; state targets use one LLM judge call per evaluated example.")
    parser.add_argument(
        "--state-history-size",
        type=int,
        default=DEFAULT_STATE_HISTORY_SIZE,
        help=(
            "Number of prior state messages to include in the world-model input. "
            "This is useful when the target state is a diff rather than a full snapshot."
        ),
    )
    parser.add_argument(
        "--include-world-model-history",
        action="store_true",
        help=(
            "Add an input-only history list to world-model prompts. Actual execution history "
            "uses `step`, `action`, and a 200-character `observation`; imagined history uses "
            "`imagined step`, `action`, and predicted `state`. The world-model target/output "
            "is unchanged."
        ),
    )
    parser.add_argument(
        "--world-model-system-prompt-max-chars",
        type=int,
        default=0,
        help=(
            "Optional character cap for the trajectory system prompt inside world-model "
            "prompts. 0 keeps the full prompt."
        ),
    )
    parser.add_argument(
        "--world-model-action-max-chars",
        type=int,
        default=0,
        help=(
            "Optional character cap for the action payload inside world-model prompts. "
            "0 keeps the full action."
        ),
    )
    parser.add_argument(
        "--world-model-target",
        default=WORLD_MODEL_TARGET_STATE,
        choices=[
            WORLD_MODEL_TARGET_STATE,
            WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY,
            WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
            WORLD_MODEL_TARGET_TOOL_OUTPUT,
            WORLD_MODEL_TARGET_CANONICAL_EVENT,
            LEGACY_WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY,
        ],
        help=(
            "World-model training target. `state` predicts the resulting state from the "
            "system prompt, user prompt, previous state, and action. "
            "`tool_execution_result_ternary` predicts `1` for success, `0` for stagnation, "
            "and `-1` for explicit tool failure. "
            "`tool_execution_result_success_failure` predicts `1` for success and collapses "
            "both stagnation and explicit tool failure into `0`. "
            "`tool_output` predicts the raw `state.context.last_tool_output` text "
            "(falls back to `error_message` when last_tool_output is missing) — useful for "
            "world-model rollouts that need realistic tool responses. "
            "`canonical_event_with_nudge` predicts the reduced beam-search annotated state "
            "as JSON: action_type, the five scored canonical-event fields, plus binary terminal. Labels are "
            "read from the canonical-event JSONL files rather than trajectory JSON. "
            "`tool_execution_result_binary` is kept as a legacy alias for the three-class mode."
        ),
    )
    parser.add_argument(
        "--oversample-minority-outcomes",
        action="store_true",
        help=(
            "Duplicate rows whose `outcome_label` belongs to under-represented classes so the "
            "trainer sees a more balanced distribution. Applied only to training rows; the eval "
            "set keeps its natural distribution."
        ),
    )
    parser.add_argument(
        "--oversample-target-ratio",
        type=float,
        default=0.5,
        help=(
            "After oversampling, the target count for each minority class is "
            "`majority_count * oversample_target_ratio`. 1.0 fully equalizes; 0.5 only halves the gap."
        ),
    )
    parser.add_argument(
        "--oversample-max-multiplier",
        type=float,
        default=10.0,
        help=(
            "Cap on how many duplicates a single minority row can spawn, to prevent extreme "
            "overfitting on a tiny minority set. Set to 1.0 to disable duplication entirely."
        ),
    )
    parser.add_argument(
        "--include-error-message-in-target",
        action="store_true",
        help=(
            "For tool-execution-result targets, append the error message (or API response) after "
            "the label using a comma when the gold label is `0` (stagnation) or `-1` (failure). "
            "Successful labels remain as plain `1`. Example target: `-1,API Error: xxx`."
        ),
    )
    parser.add_argument(
        "--include-stage-in-target",
        action="store_true",
        help=(
            "For the binary tool-execution-result target, return structured JSON with the "
            "binary outcome plus `error_message`, `current_stage`, and `remaining_stages`."
        ),
    )
    parser.add_argument(
        "--outcome-balance-loss",
        default="effective_num_loss",
        choices=["none", "inverse_frequency_loss", "effective_num_loss"],
        help=(
            "Class-balance strategy converted into weighted training-sampler probabilities using "
            "`last_tool_execution_result` labels. `effective_num_loss` is the recommended "
            "default to reduce majority-class collapse without changing the model loss."
        ),
    )
    parser.add_argument(
        "--outcome-balance-beta",
        type=float,
        default=0.999,
        help="Beta used by `effective_num_loss`. Values closer to 1.0 increase minority upweighting.",
    )
    parser.add_argument("--max-agent-tasks", type=int, default=100, help="Maximum held-out tasks used for offline agent evaluation.")
    parser.add_argument(
        "--agent-max-steps",
        type=int,
        default=15,
        help="Maximum number of tool-use decision steps per task for offline replay and EnterpriseArena task export.",
    )
    parser.add_argument(
        "--internal-thinking-max-iters",
        type=int,
        default=3,
        help="Maximum number of world-model-guided internal-thinking revisions before each actual tool execution.",
    )
    parser.add_argument(
        "--agent-draft-model",
        default=None,
        help=(
            "Optional HF draft model for speculative (assisted) decoding of the agent model, "
            "e.g. Qwen/Qwen3-0.6B when the agent is a larger Qwen3. Only applies to local HF "
            "agent backends and only on batch-1 generations (a Transformers restriction)."
        ),
    )
    parser.add_argument(
        "--world-model-draft-model",
        default=None,
        help="Optional HF draft model for speculative decoding of the (HF) world model.",
    )
    parser.add_argument(
        "--prompt-lookup-tokens",
        type=int,
        default=0,
        help=(
            "Enable prompt-lookup decoding with this candidate length (e.g. 10) for local HF "
            "agent/world-model backends. Drafts continuations by matching n-grams already in "
            "the prompt -- effective here because tool-call JSON and predicted states copy "
            "long spans from the context. Ignored when a draft model is set; batch-1 only. "
            "0 disables."
        ),
    )
    parser.add_argument(
        "--llm-batch-parallelism",
        type=int,
        default=8,
        help=(
            "Max concurrent requests when batching LLM calls against API/vLLM backends "
            "(local HF backends batch inside a single generate call instead)."
        ),
    )
    parser.add_argument(
        "--imagined-single-call-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate each imagined step's thought AND action in one LLM call instead of the "
            "two-call think-then-act sequence. --no-imagined-single-call-step restores the "
            "original two-call behaviour for A/B comparison."
        ),
    )
    parser.add_argument(
        "--imagined-rollout-mode",
        choices=("closed_loop", "open_loop"),
        default="closed_loop",
        help=(
            "closed_loop (default): each imagined step's action is chosen after seeing the "
            "world model's predicted state for the previous step (lockstep-batched across "
            "rollouts). open_loop: ONE agent call proposes all rollouts' full action "
            "sequences up front (beam_plan-style skeletons with $stepK.field references); "
            "the world model then fills in each plan's state chain, batched across plans "
            "per step. Cheapest LLM budget (1 agent call + max-steps batched WM calls) but "
            "actions cannot react to predicted failures mid-trajectory. Falls back to "
            "closed_loop for a step if the plan payload cannot be parsed."
        ),
    )
    parser.add_argument(
        "--sample-temperature-ladder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Spread the k planning samples over a temperature ladder (one near-greedy sample "
            "plus increasingly exploratory ones) instead of drawing all k at one temperature. "
            "Reduces duplicate plans, but forfeits the single-request n=k path since a request "
            "carries one sampling config -- enable prefix caching on the server first."
        ),
    )
    parser.add_argument(
        "--sample-temperature-ladder-max",
        type=float,
        default=1.2,
        help="Upper bound of the ladder. Above ~1.2 malformed actions truncate plans.",
    )
    parser.add_argument(
        "--beam-plan-trigger",
        choices=("interval", "critic"),
        default="interval",
        help=(
            "When beam_plan spends a planning cycle. `interval` (default) re-plans every "
            "--latent-mpc-execute-steps steps regardless of need. `critic` first scores the "
            "action the agent already produced with the world model (ONE latent forward, no LLM "
            "call) and only plans -- revising the current action AND looking several steps ahead "
            "-- when that score says the action is bad. Amortized cost becomes "
            "critic + fire_rate * planning, and the fire rate is logged per step "
            "(beam_plan_critic_fire_rate) so it can be measured and the thresholds calibrated."
        ),
    )
    parser.add_argument(
        "--beam-plan-critic-failure-prob",
        type=float,
        default=0.3,
        help="critic trigger: plan when the predicted P(execution_status=failure) reaches this.",
    )
    parser.add_argument(
        "--beam-plan-critic-stall-prob",
        type=float,
        default=0.7,
        help=(
            "critic trigger: plan when the action looks like it will not advance the task, i.e. "
            "1 - P(progress_signal=positive) reaches this."
        ),
    )
    parser.add_argument(
        "--beam-plan-critic-min-score",
        type=float,
        default=None,
        help=(
            "critic trigger: also plan when the predicted step score falls below this. Left "
            "unset by default because the scale is checkpoint-specific -- read the per-step "
            "`score` in the GYM_BEAM_PLAN_CRITIC events from a run first."
        ),
    )
    parser.add_argument(
        "--beam-plan-critic-max-quiet-steps",
        type=int,
        default=0,
        help=(
            "critic trigger safety valve: force a planning cycle after this many consecutive "
            "non-firing steps, so a mis-calibrated critic cannot disable lookahead for a whole "
            "episode. 0 (default) is purely event-driven."
        ),
    )
    parser.add_argument(
        "--beam-plan-terminal-advice",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In beam_plan mode, use the JEPA terminal head during critic/plan scoring to add "
            "a prompt advisory when P(done) is high, encouraging the agent to stop once the "
            "actual state confirms all requirements are satisfied."
        ),
    )
    parser.add_argument(
        "--beam-plan-terminal-advice-threshold",
        type=float,
        default=0.75,
        help="P(done) threshold for --beam-plan-terminal-advice.",
    )
    parser.add_argument(
        "--imagined-parallel-rollouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Advance all --imagined-trajectory-rollouts chains in lockstep, batching each "
            "step's agent and world-model generations across rollouts. "
            "--no-imagined-parallel-rollouts restores sequential rollouts."
        ),
    )
    parser.add_argument(
        "--imagined-trajectory-max-steps",
        type=int,
        default=3,
        help="Maximum number of imagined world-model rollout steps to add as planning context before each actual step.",
    )
    parser.add_argument(
        "--imagined-trajectory-rollouts",
        type=int,
        default=1,
        help=(
            "Number of imagined trajectories to generate per planning step. "
            "Default 1 preserves the original behavior."
        ),
    )
    parser.add_argument(
        "--imagined-rollout-temperature",
        type=float,
        default=0.7,
        help=(
            "Temperature used for imagined rollout generation when "
            "`--imagined-trajectory-rollouts` > 1. The first rollout is deterministic; "
            "the rest use this temperature."
        ),
    )
    parser.add_argument(
        "--imagined-trajectory-selection-strategy",
        choices=("first", "llm_judge", "topk_search"),
        default="llm_judge",
        help=(
            "How to choose among imagined trajectories. "
            "`llm_judge` uses an LLM reranker over independent rollouts; `first` keeps "
            "the first rollout; `topk_search` repeatedly expands candidate actions, "
            "scores partial trajectories with world-model states, and keeps top-k."
        ),
    )
    parser.add_argument(
        "--imagined-trajectory-candidate-actions",
        type=int,
        default=3,
        help="Number of candidate actions to sample per partial trajectory for `topk_search`.",
    )
    parser.add_argument(
        "--imagined-trajectory-top-k",
        type=int,
        default=3,
        help="Number of partial imagined trajectories retained after each `topk_search` step.",
    )
    parser.add_argument(
        "--imagined-trajectory-observation-source",
        choices=("world_model", "none"),
        default="world_model",
        help=(
            "What imagined observations to inject between imagined steps. "
            "`world_model` uses predicted tool outputs or state from the world model; "
            "`none` generates lookahead steps without any imagined tool results."
        ),
    )
    parser.add_argument(
        "--revision-lookahead-steps",
        type=int,
        default=1,
        help=(
            "Depth of multi-step lookahead used by the `revision` strategy. "
            "Default 1 preserves the original single-step behavior. With K>1 the "
            "world model rolls forward K steps starting from the agent's planned "
            "calls (agent picks subsequent steps based on imagined state) before "
            "the agent decides whether to revise."
        ),
    )
    parser.add_argument(
        "--revision-imagined-rollouts",
        type=int,
        default=1,
        help=(
            "Number of parallel imagined rollouts the agent generates per "
            "revision iteration. Default 1 preserves the original single-rollout "
            "behavior. N>1 runs N rollouts (the first deterministic, the rest "
            "sampled with `--revision-rollout-temperature`) and surfaces all of "
            "them to the agent before it commits to a revision."
        ),
    )
    parser.add_argument(
        "--revision-rollout-temperature",
        type=float,
        default=0.7,
        help=(
            "Temperature used for the agent's imagined-rollout decisions when "
            "`--revision-imagined-rollouts` > 1. Only the first rollout is "
            "deterministic; the remaining rollouts use this temperature for "
            "diversity."
        ),
    )
    parser.add_argument(
        "--final-answer-f1-threshold",
        type=float,
        default=0.35,
        help="LLM-judge task-completion score threshold used to count a replayed task as completed.",
    )
    parser.add_argument("--prepare-enterprisearena-tasks", action="store_true", help="Write held-out tasks in Interactive_mcp_localM.py JSON format.")
    parser.add_argument("--run-enterprisearena", action="store_true", help="Run the external Interactive_mcp_localM.py helper with the held-out tasks file.")
    parser.add_argument("--enterprise-runner", type=Path, default=DEFAULT_ENTERPRISE_RUNNER, help="Path to Interactive_mcp_localM.py.")
    parser.add_argument("--enterprise-mcp-config", type=Path, help="Optional MCP config passed to Interactive_mcp_localM.py.")
    parser.add_argument("--enterprise-output-trajectories", type=Path, help="Optional output path passed to Interactive_mcp_localM.py.")
    parser.add_argument(
        "--gym-task-configs",
        type=Path,
        default=None,
        help=(
            "Folder of EnterpriseOps-Gym task config JSONs (the same files "
            "evaluate.py consumes). When set, agent replay routes through the "
            "gym BenchmarkExecutor + verifier engine instead of the EnterpriseArena "
            "MCP path. Each held-out trajectory must carry a `gym_task_config_name` "
            "(automatically populated by the EnterpriseOps-Gym trajectory generators)."
        ),
    )
    parser.add_argument(
        "--gym-repo-path",
        type=Path,
        default=Path.home() / "program" / "tools" / "EnterpriseOps-Gym",
        help="Path to the EnterpriseOps-Gym repo (used when --gym-task-configs is set).",
    )
    parser.add_argument(
        "--gym-task-split-manifest",
        type=Path,
        default=DEFAULT_ENTERPRISEOPS_GYM_TASK_SPLIT_MANIFEST,
        help=(
            "Train/test JSON manifest for EnterpriseOps-Gym JSONL task files. "
            "When --gym-task-configs is set, benchmark replay evaluates only the "
            "manifest's `test` JSONL tasks, preserving manifest order."
        ),
    )
    parser.add_argument(
        "--no-gym-task-split-manifest",
        action="store_true",
        help=(
            "Disable EnterpriseOps-Gym task split filtering even when "
            "--gym-task-configs is set. Use this to evaluate the full held-out "
            "task set from --eval-data-path / --trajectory-dataset."
        ),
    )
    args = parser.parse_args()
    if args.no_gym_task_split_manifest:
        args.gym_task_split_manifest = None
    args.world_model_target = canonicalize_world_model_target(args.world_model_target)
    # Record whether the paths were user-supplied BEFORE the preset fills them in: the
    # canonical-event target reads JSONL label files, not the trajectory JSON a preset points
    # at, so it must be able to tell "user chose these files" from "preset default".
    args.train_data_path_explicit = args.train_data_path is not None
    args.eval_data_path_explicit = args.eval_data_path is not None
    preset_train, preset_eval = TRAJECTORY_DATASET_PRESETS[args.trajectory_dataset]
    if args.train_data_path is None:
        args.train_data_path = list(preset_train)
    elif isinstance(args.train_data_path, Path):
        args.train_data_path = [args.train_data_path]
    if args.eval_data_path is None:
        args.eval_data_path = list(preset_eval)
    elif isinstance(args.eval_data_path, Path):
        args.eval_data_path = [args.eval_data_path]
    # Applied AFTER the preset expands and to explicit paths alike -- same order and same
    # WEB_BROWSING_TRAJECTORY_PATHS set as src/finetuning_jepa.py, so both trainers end up with
    # the same corpora for the same (preset, flag) pair.
    if args.skip_web_trajectories:
        _web = WEB_BROWSING_TRAJECTORY_PATHS | WEB_BROWSING_25K_PATHS
        args.train_data_path = [p for p in args.train_data_path if p not in _web]
        args.eval_data_path = [p for p in args.eval_data_path if p not in _web]
    if args.model == "Qwen/Qwen3-4B" and args.max_seq_length == 16384:
        args.max_seq_length = 16384
    return args


def load_json(path: Path) -> Any:
    # Parse from BYTES, not text: these files are UTF-8 with non-ASCII content, so decoding to a
    # Python str first widens every character to 4 bytes (measured: a 0.84 GB file became a
    # 3.34 GB str). json.loads decodes internally and releases that buffer immediately.
    return json.loads(path.read_bytes())


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

    # TerminalBench-style states store tool metadata at the root.
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


def coerce_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def build_stage_plan_canonical_observation(
    raw_state: Any,
    *,
    user_prompt: str = "",
    system_prompt: str = "",
) -> dict[str, Any] | None:
    process = state_process_from_any(raw_state)
    current_stage = str(process.get("current_stage") or "").strip()
    remaining_stages = coerce_text_list(process.get("remaining_stages"))
    completed_stages = coerce_text_list(process.get("completed_stages"))
    if not current_stage and remaining_stages:
        current_stage = remaining_stages[0]
    if current_stage and current_stage.lower() not in {"finished", "unknown"}:
        if current_stage not in remaining_stages:
            remaining_stages = [current_stage] + remaining_stages
    if not current_stage and not remaining_stages and not completed_stages:
        return None
    task = (user_prompt or system_prompt or "complete the requested task").strip()
    return {
        "schema": "ewm_canonical_observation_v1",
        "tool_outcome": {
            "success": True,
            "label": 1,
            "error_message": "",
            "summary": "Stage-plan observation for task progress.",
        },
        "stages": {
            "current_stage": current_stage or (remaining_stages[0] if remaining_stages else "finished"),
            "remaining_stages": remaining_stages,
            "completed_stages": completed_stages,
        },
        "evidence": [task] if task else [],
    }


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


def crm_tool_calls_from_assistant(content: str) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for match_index, match in enumerate(CRM_TOOL_TAG_PATTERN.finditer(content or "")):
        tool_name = match.group(1).strip().lower()
        tool_input = match.group(2).strip()
        if tool_name == "execute":
            arguments = {"query": tool_input}
        elif tool_name == "describe":
            arguments = {"object_name": tool_input}
        else:
            arguments = {"input": tool_input}
        tool_calls.append(
            {
                "id": f"crm_call_{match_index}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            }
        )
    return tool_calls


def crm_observation_to_state(observation: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    tool_name = None
    if tool_calls:
        function = tool_calls[-1].get("function") or {}
        tool_name = function.get("name")
    lowered = (observation or "").lower()
    result = -1 if "[error" in lowered or lowered.startswith("error") else 1
    return {
        "state": {
            "agent": {"role": "crm_agent"},
            "context": {
                "last_tool_execution_result": result,
                "last_tool_name": tool_name,
                "last_tool_output": observation,
            },
            "process": {
                "remaining_stages": [],
                "current_stage": "tool_observation",
            },
            "relational": {},
            "temporal": {},
        }
    }


def normalize_crmarenapro_result_trajectory(record: dict[str, Any], trajectory_index: int) -> dict[str, Any]:
    raw_messages = record.get("traj") or []
    if not isinstance(raw_messages, list):
        return record

    messages: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(raw_messages):
        message = raw_messages[cursor]
        if not isinstance(message, dict):
            cursor += 1
            continue
        role = message.get("role")
        content = message.get("content", "")
        if role == "assistant" and isinstance(content, str):
            tool_calls = crm_tool_calls_from_assistant(content)
            next_message = raw_messages[cursor + 1] if cursor + 1 < len(raw_messages) else None
            next_is_observation = (
                isinstance(next_message, dict)
                and next_message.get("role") == "user"
                and isinstance(next_message.get("content"), str)
                and next_message.get("content", "").lstrip().startswith("[Observation:")
            )
            if tool_calls and next_is_observation:
                messages.append({"role": "action", "content": {"tool_calls": tool_calls}})
                messages.append(
                    {
                        "role": "state",
                        "content": crm_observation_to_state(next_message.get("content", ""), tool_calls),
                    }
                )
                cursor += 2
                continue
        if role in {"system", "user", "assistant", "action", "state"}:
            messages.append({"role": role, "content": content})
        cursor += 1

    normalized = dict(record)
    normalized.setdefault("trajectory_id", record.get("task_id", trajectory_index))
    normalized["messages"] = messages
    normalized["source_format"] = "crmarenapro_result"
    return normalized


def normalize_loaded_trajectory(record: dict[str, Any], trajectory_index: int) -> dict[str, Any]:
    if isinstance(record.get("messages"), list):
        return record
    if isinstance(record.get("traj"), list):
        return normalize_crmarenapro_result_trajectory(record, trajectory_index)
    return record


def normalize_loaded_trajectories(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_loaded_trajectory(record, index) for index, record in enumerate(records)]


def override_enterpriseops_gym_mcp_urls(raw_config: dict[str, Any]) -> dict[str, Any]:
    endpoint_overrides = {
        CSM_MCP_SERVER_NAME: (CSM_MCP_DEFAULT_URL, CSM_MCP_OVERRIDE_URL),
        HR_MCP_SERVER_NAME: (HR_MCP_DEFAULT_URL, HR_MCP_OVERRIDE_URL),
    }

    def apply_override(server: dict[str, Any]) -> None:
        override = endpoint_overrides.get(server.get("mcp_server_name"))
        if override is None:
            return
        default_url, override_url = override
        if server.get("mcp_server_url") == default_url:
            server["mcp_server_url"] = override_url

    gym_servers = raw_config.get("gym_servers_config")
    if isinstance(gym_servers, list):
        for server in gym_servers:
            if isinstance(server, dict):
                apply_override(server)

    apply_override(raw_config)

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


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --- distributed data-prep gating ------------------------------------------------------------
# Loading the raw trajectory JSON is the memory bottleneck for the large presets: `all_no_
# terminalbench` is ~18.5 GB of JSON on disk, which expands several-fold as nested Python
# objects. Doing that identically in every torchrun process multiplies it by the world size and
# OOMs the host long before training starts. Mirrors the same gate in src/finetuning_jepa.py:
# rank 0 parses and extracts, writes the compact examples, everyone else waits and reads those.


def distributed_is_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def distributed_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return distributed_rank() == 0


def setup_distributed(timeout_minutes: int = 120) -> bool:
    """Init the process group early so the data-prep barrier can be taken before the Trainer
    builds its own. Returns whether this is a distributed run.

    The long timeout matters: rank 0 can spend well over the 30-minute default collective
    timeout parsing a large preset while the other ranks sit at the barrier, and the timeout
    firing kills the whole job.
    """
    if not distributed_is_enabled():
        return False
    if torch.cuda.is_available():
        torch.cuda.set_device(distributed_local_rank())
    if not torch.distributed.is_initialized():
        kwargs: dict[str, Any] = {
            "backend": "nccl" if torch.cuda.is_available() else "gloo",
            "timeout": datetime.timedelta(minutes=timeout_minutes),
        }
        if torch.cuda.is_available():
            # Pin the rank->device mapping explicitly. Without it NCCL infers the device and
            # warns that a wrong guess "can potentially cause a hang".
            kwargs["device_id"] = torch.device(f"cuda:{distributed_local_rank()}")
        torch.distributed.init_process_group(**kwargs)
    return True


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _resident_gb() -> float:
    try:
        return int(open("/proc/self/statm").read().split()[1]) * 4096 / 2**30
    except OSError:  # not Linux
        return 0.0


@dataclass
class ExtractedSplit:
    """Per-file extraction result. Deliberately holds NO trajectory references -- the raw
    trajectories are the memory hog and are freed as soon as a file is extracted."""
    examples: list[WorldModelStateExample]
    tasks: list[TaskTrajectory]
    source_rows: list[dict[str, Any]]
    trajectory_count: int


def extract_split_streaming(
    paths: list[Path], args: argparse.Namespace
) -> dict[Path, ExtractedSplit]:
    """Load and extract ONE trajectory file at a time, freeing each before the next.

    Holding every file's parsed trajectories at once is what exhausts the host: measured on this
    corpus a file expands to ~3.3x its on-disk size as resident Python objects, with a ~7.4x
    transient spike while json is building them (8.34 GB of TOUCAN -> 27.7 GB resident, 61.5 GB
    peak). Extracting per file caps the peak at the largest single file instead of the sum, and
    what accumulates is the examples, which are far smaller.

    Each DISTINCT path is extracted once and returned keyed by path, so a file listed in both
    --train-data-path and --eval-data-path (every ADP preset does this) is parsed once.

    `trajectory_index` is left FILE-LOCAL here (extract_state_examples and
    extract_replay_tasks_from_state_trajectories both number with enumerate()); gather_split
    re-bases it per split, which reproduces the numbering the previous concatenate-then-extract
    code produced -- each split numbered from 0 in path order.
    """
    results: dict[Path, ExtractedSplit] = {}
    for path in dict.fromkeys(paths):  # de-duplicate, preserve order
        size_gb = path.stat().st_size / 2**30 if path.exists() else 0.0
        loaded = load_json(path)
        if not isinstance(loaded, list):
            raise SystemExit(f"Expected a list of trajectories in {path}")
        trajectories = normalize_loaded_trajectories(loaded)
        del loaded
        stats = build_trajectory_outcome_stats(trajectories, path)
        kept = [item.trajectory for item in stats]
        examples = extract_state_examples(kept, state_history_size=args.state_history_size)
        tasks = extract_replay_tasks_from_state_trajectories(kept)
        results[path] = ExtractedSplit(
            examples=examples,
            tasks=tasks,
            # Flattened now so the TrajectoryOutcomeStats (which reference the trajectories)
            # can be dropped with them.
            source_rows=[
                {
                    "source_path": item.source_path,
                    "source_index": item.source_index,
                    "success_count": item.success_count,
                    "failure_count": item.failure_count,
                    "example_count": item.example_count,
                }
                for item in stats
            ],
            trajectory_count=len(kept),
        )
        del trajectories, stats, kept
        gc.collect()
        print(f"[data] {path.name}: {size_gb:.2f} GB on disk -> {len(examples)} examples "
              f"(resident {_resident_gb():.1f} GB)", flush=True)
    return results


def gather_split(
    paths: list[Path], extracted: dict[Path, ExtractedSplit]
) -> tuple[list[WorldModelStateExample], list[TaskTrajectory], list[dict[str, Any]], int]:
    """Concatenate the per-file extraction results for one split, in the listed path order.

    Re-bases the file-local `trajectory_index` onto a per-split running offset, reproducing the
    numbering of the previous concatenate-then-extract code (each split numbered from 0). The
    rebase must COPY rather than mutate: a path listed in both splits shares one ExtractedSplit,
    and the two splits need different offsets. dataclasses.replace is shallow, so the copies
    share every field value and cost only the object headers.
    """
    examples: list[WorldModelStateExample] = []
    tasks: list[TaskTrajectory] = []
    source_rows: list[dict[str, Any]] = []
    trajectory_count = 0
    for path in paths:
        split = extracted[path]
        offset = trajectory_count
        examples.extend(
            example if offset == 0 else replace(example, trajectory_index=example.trajectory_index + offset)
            for example in split.examples
        )
        tasks.extend(
            task if offset == 0 else replace(task, trajectory_index=task.trajectory_index + offset)
            for task in split.tasks
        )
        source_rows.extend(split.source_rows)
        trajectory_count += split.trajectory_count
    return examples, tasks, source_rows, trajectory_count


def rebuild_task_trajectory(row: dict[str, Any]) -> TaskTrajectory:
    """Reconstruct a TaskTrajectory from its asdict() form (nested TaskStep list)."""
    steps = [TaskStep(**step) for step in row.get("steps") or []]
    return TaskTrajectory(**{**row, "steps": steps})


def dump_imagined_trajectories(mode_name: str, payload: list[dict[str, Any]]) -> Path:
    path = DEFAULT_TRAJECTORIES_DIR / f"{mode_name}_trajectories.json"
    dump_json(path, payload)
    return path


def dump_replay_trajectories(mode_name: str, payload: list[dict[str, Any]]) -> Path:
    """Persist the actual replay conversations (per task) for one execution mode."""
    path = DEFAULT_TRAJECTORIES_DIR / f"{mode_name}_replay_trajectories.json"
    dump_json(path, payload)
    return path


def _extract_tool_call_sequence_from_conversation_flow(
    conversation_flow: list[dict[str, Any]] | None,
) -> list[list[str]]:
    sequence: list[list[str]] = []
    for message in conversation_flow or []:
        if not isinstance(message, dict):
            continue
        if message.get("type") != "ai_message":
            continue
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            continue
        names = [call.get("name", "") for call in tool_calls if isinstance(call, dict)]
        if names:
            sequence.append(names)
    return sequence


def build_agent_replay_strategy_records(
    replay_eval: dict[str, Any],
    strategy: str,
) -> list[dict[str, Any]]:
    mode_payload = replay_eval.get(strategy) or {}
    task_records = mode_payload.get("task_records") or []
    records: list[dict[str, Any]] = []

    for task_record in task_records:
        if "result" in task_record:
            result = task_record.get("result") or {}
            benchmark_config = result.get("benchmark_config") or {}
            runs = result.get("runs") or []
            statistics = result.get("statistics") or {}
            base = {
                "strategy": strategy,
                "trajectory_index": task_record.get("trajectory_index"),
                "gym_task_config_name": task_record.get("gym_task_config_name"),
                "task_query": benchmark_config.get("user_prompt"),
                "statistics": statistics,
            }
            for run_index, run in enumerate(runs, start=1):
                record = dict(base)
                record.update(
                    {
                        "run_index": run_index,
                        "completed": run.get("overall_success"),
                        "verification_summary": run.get("verification_summary"),
                        "steps_taken": run.get("steps_taken"),
                        "internal_thinking_iterations": run.get("internal_thinking_iterations"),
                        "wm_revisions_applied": run.get("wm_revisions_applied"),
                        "wm_imagined_rollouts": run.get("wm_imagined_rollouts"),
                        "wm_revision_step_details": run.get("wm_revision_step_details") or [],
                        "imagined_rollout_records": run.get("imagined_rollout_records") or [],
                        "model_response": run.get("model_response"),
                        "tool_call_sequence": _extract_tool_call_sequence_from_conversation_flow(
                            run.get("conversation_flow")
                        ),
                    }
                )
                records.append(record)
            continue

        records.append(
            {
                "strategy": strategy,
                "trajectory_index": task_record.get("trajectory_index"),
                "completed": task_record.get("completed"),
                "failure_reason": task_record.get("failure_reason"),
                "final_answer_score": task_record.get("final_answer_score"),
                "final_answer_evaluation": task_record.get("final_answer_evaluation"),
                "tool_steps_taken": task_record.get("tool_steps_taken"),
                "tool_calls_taken": task_record.get("tool_calls_taken"),
                "internal_thinking_iterations": task_record.get("internal_thinking_iterations"),
                "imagined_rollouts_used": task_record.get("imagined_rollouts_used"),
                "wm_predicted_failures": task_record.get("wm_predicted_failures"),
                "wm_predicted_successes": task_record.get("wm_predicted_successes"),
                "wm_triggered_revisions": task_record.get("wm_triggered_revisions"),
                "wm_no_op_revisions": task_record.get("wm_no_op_revisions"),
                "wm_revision_effectiveness": task_record.get("wm_revision_effectiveness"),
                "wm_revision_step_details": task_record.get("wm_revision_step_details") or [],
                "imagined_rollout_records": task_record.get("imagined_rollout_records") or [],
            }
        )
    return records


def dump_agent_replay_strategy_records(output_dir: Path, replay_eval: dict[str, Any]) -> None:
    for strategy in ("revision", "imagined"):
        records = build_agent_replay_strategy_records(replay_eval, strategy)
        dump_json(output_dir / f"agent_replay_{strategy}_records.json", records)
        dump_jsonl(output_dir / f"agent_replay_{strategy}_records.jsonl", records)


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
    for opener, closer in (("<tool_call>", "</tool_call>"), ("<answer>", "</answer>")):
        if opener in value and closer in value:
            start = value.find(opener) + len(opener)
            end = value.rfind(closer)
            if end > start:
                value = value[start:end].strip()
    if value.startswith("<tool_call>"):
        value = value[len("<tool_call>"):].strip()
    if value.endswith("</tool_call>"):
        value = value[: -len("</tool_call>")].strip()
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
        target_mode=WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY,
    )


def parse_tool_execution_result_prediction(text: str, target_mode: str) -> int | None:
    parsed = parse_binary_world_model_prediction(text)
    return normalize_tool_execution_result_for_target(parsed.get("success"), target_mode=target_mode)


def load_tokenizer_with_repair(
    model_path: str | Path,
    trust_remote_code: bool = False,
    auto_tokenizer_class: Any | None = None,
) -> Any:
    """AutoTokenizer.from_pretrained with one known checkpoint defect repaired.

    Some saved checkpoints write `extra_special_tokens` as a LIST in tokenizer_config.json
    (a transformers save-side quirk), and loading then dies inside
    _set_model_specific_special_tokens with "'list' object has no attribute 'keys'". Those
    tokens are already registered as added tokens in tokenizer.json, so overriding the field
    with an empty mapping loads the same tokenizer without touching the checkpoint on disk.

    Every tokenizer load in this module goes through here: the defect is a property of the
    checkpoint, so it breaks training, evaluation and replay alike, and fixing it at one
    call site only moves the crash.
    """
    if auto_tokenizer_class is None:
        from transformers import AutoTokenizer as auto_tokenizer_class      # noqa: N813
    try:
        return auto_tokenizer_class.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except AttributeError:
        tokenizer = auto_tokenizer_class.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, extra_special_tokens={}
        )
        print(f"[tokenizer] {model_path}: repaired malformed extra_special_tokens while loading "
              "(checkpoint left unchanged).", flush=True)
        return tokenizer


def build_chat_template_renderer(
    tokenizer: Any | None,
    disable_chat_template: bool = False,
) -> Any:
    """Resolve the chat-template call convention ONCE and return a fast renderer.

    The per-call work below (`inspect.signature`, a substring scan of the whole template) costs
    tens of microseconds; at 700k+ rows x 2 renders per row that alone is minutes of every
    tokenization pass. Hot loops should hold on to the returned closure instead of calling
    apply_chat_template_or_fallback() per row.
    """
    if tokenizer is None or disable_chat_template or not getattr(tokenizer, "chat_template", None):

        def render_fallback(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
            rendered = [f"{message['role'].upper()}: {message['content']}" for message in messages]
            if add_generation_prompt:
                rendered.append("ASSISTANT:")
            return "\n\n".join(rendered)

        return render_fallback

    base_kwargs: dict[str, Any] = {"tokenize": False}
    # Some chat templates (e.g. Qwen3/Qwen3.6 and Gemma 4) branch on an
    # `enable_thinking` kwarg, but it is forwarded via `**kwargs` rather
    # than appearing in the formal signature. Detect by searching the
    # template text so we actually disable reasoning traces instead of
    # silently ignoring the extra kwarg.
    if "enable_thinking" in (getattr(tokenizer, "chat_template", "") or ""):
        base_kwargs["enable_thinking"] = False
    parameters = inspect.signature(tokenizer.apply_chat_template).parameters
    messages_key = "messages" if "messages" in parameters else "conversation"

    def render(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
        return tokenizer.apply_chat_template(
            add_generation_prompt=add_generation_prompt,
            **{messages_key: messages},
            **base_kwargs,
        )

    return render


# Keyed by id(tokenizer); the tokenizer itself is kept in the value so the id cannot be
# recycled by another object while the entry is live.
_CHAT_TEMPLATE_RENDERER_CACHE: dict[tuple[int, bool], tuple[Any, Any]] = {}


def apply_chat_template_or_fallback(
    tokenizer: Any | None,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    disable_chat_template: bool = False,
) -> str:
    cache_key = (id(tokenizer), bool(disable_chat_template))
    cached = _CHAT_TEMPLATE_RENDERER_CACHE.get(cache_key)
    if cached is None:
        renderer = build_chat_template_renderer(tokenizer, disable_chat_template)
        _CHAT_TEMPLATE_RENDERER_CACHE[cache_key] = (tokenizer, renderer)
    else:
        renderer = cached[1]
    return renderer(messages, add_generation_prompt=add_generation_prompt)


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


def build_task_completion_judge_prompt() -> str:
    return (
        "You are evaluating how well an AI agent completed the assigned task. "
        "Focus on the final outcome and how well it addresses the original requirements.\n\n"
        "**TASK COMPLETION EVALUATION CRITERIA:**\n"
        "- requirement_coverage: Did the agent address all aspects of the task? (0.0-1.0)\n"
        "- accuracy: Is the information and analysis factually correct? (0.0-1.0)\n"
        "- completeness: Is the response thorough and comprehensive? (0.0-1.0)\n"
        "- usefulness: Is the final result practically valuable to the user? (0.0-1.0)\n\n"
        "Compare the final response against:\n"
        "- Original task requirements\n"
        "- Expected deliverables\n"
        "- Quality of information provided\n"
        "- Practical utility for the user\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        '  "requirement_coverage": float,\n'
        '  "accuracy": float,\n'
        '  "completeness": float,\n'
        '  "usefulness": float,\n'
        '  "comments": "Brief explanation"\n'
        "}"
    )


def build_outcome_error_match_judge_prompt() -> str:
    return (
        "You are evaluating whether a predicted tool-execution outcome matches the gold outcome.\n\n"
        "Each outcome has a label and (when the label denotes failure or stagnation) an error message "
        "or API response that explains what went wrong. The label is `1` for success, `0` for "
        "stagnation, or `-1` for explicit tool failure. When the label is `1` (success), the error "
        "message is empty and only the label needs to match.\n\n"
        "EVALUATION CRITERIA:\n"
        "- label_alignment: Does the predicted label match the gold label? (0.0-1.0)\n"
        "- error_alignment: Does the predicted error message describe the same problem as the gold "
        "error message? Ignore harmless wording, formatting, identifier, and timestamp differences. "
        "(0.0-1.0)\n"
        "- semantic_consistency: Overall, does the prediction convey the same outcome and the same "
        "underlying reason as the gold? (0.0-1.0)\n\n"
        "Set match=true ONLY when the label is correct AND, if the label is `0` or `-1`, the error "
        "message captures the same problem semantically. If the gold label is `1`, judge match purely "
        "on label correctness.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        '  "label_alignment": float,\n'
        '  "error_alignment": float,\n'
        '  "semantic_consistency": float,\n'
        '  "match": boolean,\n'
        '  "comments": "Brief explanation"\n'
        "}"
    )


def build_tool_output_match_judge_prompt() -> str:
    return (
        "You are evaluating whether a predicted tool-execution output (a free-form natural-language "
        "or JSON-stringified payload returned by an enterprise tool call) matches the gold output.\n\n"
        "Judge semantic equivalence rather than strict string equality. Ignore harmless wording, "
        "formatting, key ordering, whitespace, identifier, and timestamp differences. Numeric values "
        "and named entities that drive downstream behavior must still agree.\n\n"
        "EVALUATION CRITERIA:\n"
        "- semantic_equivalence: Do the two outputs convey the same information? (0.0-1.0)\n"
        "- factual_consistency: Are the load-bearing facts (IDs, counts, statuses, names, key fields) "
        "consistent between prediction and gold? (0.0-1.0)\n"
        "- intent_alignment: Does the predicted output answer the same question / serve the same "
        "purpose for the action that produced it? (0.0-1.0)\n"
        "- outcome_alignment: Do the two outputs imply the same success/failure outcome of the tool "
        "call (e.g., both report success, or both surface the same kind of error)? (0.0-1.0)\n\n"
        "Set match=true only when the predicted output is a strong semantic match overall and "
        "preserves the load-bearing facts a downstream agent would rely on.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        '  "semantic_equivalence": float,\n'
        '  "factual_consistency": float,\n'
        '  "intent_alignment": float,\n'
        '  "outcome_alignment": float,\n'
        '  "match": boolean,\n'
        '  "comments": "Brief explanation"\n'
        "}"
    )


def build_state_match_judge_prompt() -> str:
    return (
        "You are evaluating whether a predicted enterprise task state matches the target state.\n\n"
        "Score at FIELD LEVEL, not as one whole-schema score. The user message provides a "
        "`Fields to score` object whose keys are target field paths. Return exactly one score for "
        "each listed field path. Judge semantic equivalence rather than strict string equality; "
        "ignore harmless wording, formatting, key ordering, and minor paraphrases. Numeric values, "
        "IDs, named entities, stage status, success/failure labels, and unresolved "
        "requirements must preserve the same operational meaning.\n\n"
        "For each field:\n"
        "- score: 1.0 means the predicted field is semantically correct for that target field.\n"
        "- score: 0.5 means partially correct or too vague to safely drive downstream behavior.\n"
        "- score: 0.0 means missing, contradictory, or operationally wrong.\n"
        "- match: true only when score is at least 0.8.\n\n"
        "overall_score must be the arithmetic mean of the per-field scores you returned. "
        "overall_match must be true only when every listed field has match=true.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        '  "field_scores": {\n'
        '    "state.aspect.field": {"score": float, "match": boolean, "comments": "Brief explanation"}\n'
        "  },\n"
        '  "overall_score": float,\n'
        '  "overall_match": boolean,\n'
        '  "comments": "Brief explanation"\n'
        "}"
    )


@lru_cache(maxsize=1)
def get_task_completion_judge_client() -> Any:
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("The `openai` package is required for LLM-judge final-answer evaluation.") from exc

    return openai.OpenAI()


def evaluate_final_answer_quality(
    task_description: str,
    final_response: str,
    ground_truth_answer: str,
    execution_trajectory: list[dict[str, Any]] | None = None,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    if not final_response.strip():
        zero_scores = {key: 0.0 for key in TASK_COMPLETION_JUDGE_KEYS}
        return {
            "scores": zero_scores,
            "overall_score": 0.0,
            "comments": "No final answer provided",
            "raw_response": {},
        }

    user_parts = [
        "Task description:\n" + (task_description or ""),
        "Agent's final response:\n" + final_response,
    ]
    if execution_trajectory:
        user_parts.append(
            "Full execution trajectory:\n"
            + json.dumps(execution_trajectory, ensure_ascii=False, indent=2)
        )
    if ground_truth_answer:
        user_parts.append("Expected answer:\n" + ground_truth_answer)

    messages = [
        {"role": "system", "content": build_task_completion_judge_prompt()},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]

    try:
        client = get_task_completion_judge_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        raw_response = parse_jsonish(strip_code_fence(content))
        if not isinstance(raw_response, dict):
            raise ValueError("Expected a JSON object from task completion judge.")
        scores = {
            key: float(raw_response.get(key, 0.0))
            for key in TASK_COMPLETION_JUDGE_KEYS
        }
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        return {
            "scores": scores,
            "overall_score": overall_score,
            "comments": str(raw_response.get("comments", "")),
            "raw_response": raw_response,
        }
    except Exception as exc:
        zero_scores = {key: 0.0 for key in TASK_COMPLETION_JUDGE_KEYS}
        return {
            "scores": zero_scores,
            "overall_score": 0.0,
            "comments": f"Error: {exc}",
            "raw_response": {},
        }


def evaluate_outcome_error_match_quality(
    system_prompt: str,
    user_prompt: str,
    previous_state: dict[str, Any],
    action: Any,
    predicted_label: int | None,
    predicted_error_message: str,
    gold_label: int | None,
    gold_error_message: str,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    zero_scores = {key: 0.0 for key in OUTCOME_ERROR_MATCH_JUDGE_KEYS}
    if predicted_label is None:
        return {
            "scores": zero_scores,
            "overall_score": 0.0,
            "match": False,
            "comments": "No predicted label parsed",
            "raw_response": {},
        }

    action_text = (
        action
        if isinstance(action, str)
        else json.dumps(action, ensure_ascii=False, indent=2)
    )
    messages = [
        {"role": "system", "content": build_outcome_error_match_judge_prompt()},
        {
            "role": "user",
            "content": (
                "System prompt:\n"
                + (system_prompt or "")
                + "\n\nUser prompt:\n"
                + (user_prompt or "")
                + "\n\nPrevious state:\n"
                + normalize_state_text(previous_state)
                + "\n\nAction:\n"
                + action_text
                + f"\n\nPredicted label: {predicted_label}"
                + f"\nPredicted error/response: {predicted_error_message or '(empty)'}"
                + f"\n\nGold label: {gold_label}"
                + f"\nGold error/response: {gold_error_message or '(empty)'}"
            ),
        },
    ]

    try:
        client = get_task_completion_judge_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        raw_response = parse_jsonish(strip_code_fence(content))
        if not isinstance(raw_response, dict):
            raise ValueError("Expected a JSON object from outcome error match judge.")
        scores = {
            key: float(raw_response.get(key, 0.0))
            for key in OUTCOME_ERROR_MATCH_JUDGE_KEYS
        }
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        return {
            "scores": scores,
            "overall_score": overall_score,
            "match": bool(raw_response.get("match", False)),
            "comments": str(raw_response.get("comments", "")),
            "raw_response": raw_response,
        }
    except Exception as exc:
        return {
            "scores": zero_scores,
            "overall_score": 0.0,
            "match": False,
            "comments": f"Error: {exc}",
            "raw_response": {},
        }


def flatten_state_fields(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        fields: dict[str, Any] = {}
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            fields.update(flatten_state_fields(value[key], child_prefix))
        return fields
    return {prefix: value} if prefix else {"$": value}


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def normalize_state_field_judge_response(
    raw_response: dict[str, Any],
    target_fields: dict[str, Any],
) -> dict[str, Any]:
    raw_field_scores = raw_response.get("field_scores", {})
    if not isinstance(raw_field_scores, dict):
        raw_field_scores = {}

    field_scores: dict[str, dict[str, Any]] = {}
    for field_path in target_fields:
        raw_entry = raw_field_scores.get(field_path, {})
        if isinstance(raw_entry, dict):
            score = clamp_score(raw_entry.get("score", 0.0))
            comments = str(raw_entry.get("comments", ""))
            match = bool(raw_entry.get("match", score >= 0.8))
        else:
            score = clamp_score(raw_entry)
            comments = ""
            match = score >= 0.8
        field_scores[field_path] = {
            "score": score,
            "match": match,
            "comments": comments,
        }

    if field_scores:
        overall_score = sum(item["score"] for item in field_scores.values()) / len(field_scores)
        overall_match = all(item["match"] for item in field_scores.values())
    else:
        overall_score = 0.0
        overall_match = False

    return {
        "scores": {"field_accuracy": overall_score},
        "field_scores": field_scores,
        "overall_score": overall_score,
        "match": bool(raw_response.get("overall_match", overall_match)) and overall_match,
        "comments": str(raw_response.get("comments", "")),
        "raw_response": raw_response,
    }


def evaluate_state_match_quality(
    system_prompt: str,
    user_prompt: str,
    previous_state: dict[str, Any],
    action: Any,
    predicted_state_text: str,
    target_state_text: str,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    try:
        target_state = sanitize_state_content(parse_jsonish(target_state_text))
        target_fields = flatten_state_fields(target_state)
    except Exception:
        target_fields = {}

    if not predicted_state_text.strip():
        raw_response = {
            "field_scores": {
                field_path: {
                    "score": 0.0,
                    "match": False,
                    "comments": "No predicted state provided",
                }
                for field_path in target_fields
            },
            "overall_match": False,
            "comments": "No predicted state provided",
        }
        return normalize_state_field_judge_response(raw_response, target_fields)

    try:
        predicted_state = sanitize_state_content(parse_jsonish(predicted_state_text))
        predicted_fields = flatten_state_fields(predicted_state)
    except Exception:
        predicted_fields = {}

    action_text = (
        action
        if isinstance(action, str)
        else json.dumps(action, ensure_ascii=False, indent=2)
    )
    messages = [
        {"role": "system", "content": build_state_match_judge_prompt()},
        {
            "role": "user",
            "content": (
                "System prompt:\n"
                + (system_prompt or "")
                + "\n\nUser prompt:\n"
                + (user_prompt or "")
                + "\n\nPrevious state:\n"
                + normalize_state_text(previous_state)
                + "\n\nAction:\n"
                + action_text
                + "\n\nFields to score:\n"
                + json.dumps(target_fields, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n\nPredicted field values:\n"
                + json.dumps(predicted_fields, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n\nTarget field values:\n"
                + json.dumps(target_fields, ensure_ascii=False, indent=2, sort_keys=True)
            ),
        },
    ]

    try:
        client = get_task_completion_judge_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        raw_response = parse_jsonish(strip_code_fence(content))
        if not isinstance(raw_response, dict):
            raise ValueError("Expected a JSON object from state field match judge.")
        return normalize_state_field_judge_response(raw_response, target_fields)
    except Exception as exc:
        raw_response = {
            "field_scores": {
                field_path: {
                    "score": 0.0,
                    "match": False,
                    "comments": f"Error: {exc}",
                }
                for field_path in target_fields
            },
            "overall_match": False,
            "comments": f"Error: {exc}",
        }
        return normalize_state_field_judge_response(raw_response, target_fields)

def evaluate_tool_output_match_quality(
    system_prompt: str,
    user_prompt: str,
    previous_state: dict[str, Any],
    action: Any,
    predicted_tool_output: str,
    target_tool_output: str,
    model: str = "gpt-4o",
    char_limit: int = TOOL_OUTPUT_JUDGE_CHAR_LIMIT,
) -> dict[str, Any]:
    zero_scores = {key: 0.0 for key in TOOL_OUTPUT_MATCH_JUDGE_KEYS}
    if not predicted_tool_output.strip():
        return {
            "scores": zero_scores,
            "overall_score": 0.0,
            "match": False,
            "comments": "No predicted tool output provided",
            "raw_response": {},
        }

    action_text = (
        action
        if isinstance(action, str)
        else json.dumps(action, ensure_ascii=False, indent=2)
    )
    messages = [
        {"role": "system", "content": build_tool_output_match_judge_prompt()},
        {
            "role": "user",
            "content": (
                "System prompt:\n"
                + (system_prompt or "")
                + "\n\nUser prompt:\n"
                + (user_prompt or "")
                + "\n\nPrevious state:\n"
                + truncate_for_replay(normalize_state_text(previous_state), char_limit)
                + "\n\nAction:\n"
                + truncate_for_replay(action_text, char_limit)
                + "\n\nPredicted tool output:\n"
                + truncate_for_replay(predicted_tool_output, char_limit)
                + "\n\nGold tool output:\n"
                + truncate_for_replay(target_tool_output, char_limit)
            ),
        },
    ]

    try:
        client = get_task_completion_judge_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        raw_response = parse_jsonish(strip_code_fence(content))
        if not isinstance(raw_response, dict):
            raise ValueError("Expected a JSON object from tool output match judge.")
        scores = {
            key: float(raw_response.get(key, 0.0))
            for key in TOOL_OUTPUT_MATCH_JUDGE_KEYS
        }
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        return {
            "scores": scores,
            "overall_score": overall_score,
            "match": bool(raw_response.get("match", False)),
            "comments": str(raw_response.get("comments", "")),
            "raw_response": raw_response,
        }
    except Exception as exc:
        return {
            "scores": zero_scores,
            "overall_score": 0.0,
            "match": False,
            "comments": f"Error: {exc}",
            "raw_response": {},
        }


def missing_argument_paths(predicted: Any, gold: Any, prefix: str = "") -> list[str]:
    if isinstance(gold, dict):
        if not isinstance(predicted, dict):
            return [prefix.rstrip(".") or "<root>"]
        missing: list[str] = []
        for key, gold_value in gold.items():
            child_prefix = f"{prefix}{key}"
            if key not in predicted:
                missing.append(child_prefix)
            else:
                missing.extend(missing_argument_paths(predicted[key], gold_value, f"{child_prefix}."))
        return missing
    if isinstance(gold, list):
        if not isinstance(predicted, list) or len(predicted) < len(gold):
            return [prefix.rstrip(".") or "<root>"]
        missing: list[str] = []
        for index, gold_value in enumerate(gold):
            missing.extend(missing_argument_paths(predicted[index], gold_value, f"{prefix}{index}."))
        return missing
    if predicted != gold:
        return [prefix.rstrip(".") or "<root>"]
    return []


def tool_calls_match(predicted_calls: list[dict[str, Any]], gold_calls: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    if len(predicted_calls) != len(gold_calls):
        return False, [f"count:{len(predicted_calls)}!={len(gold_calls)}"]
    failures: list[str] = []
    for index, gold_call in enumerate(gold_calls):
        pred_call = predicted_calls[index]
        if pred_call.get("name") != gold_call.get("name"):
            failures.append(f"name[{index}]")
            continue
        failures.extend(missing_argument_paths(pred_call.get("arguments"), gold_call.get("arguments"), prefix=f"args[{index}]."))
    return not failures, failures


def normalize_agent_action_name(action: str) -> str:
    return re.sub(r"[\s_-]+", " ", action).strip().lower()


def parse_malformed_final_answer_decision(cleaned: str) -> dict[str, Any] | None:
    action_match = re.search(
        r"[\"']action[\"']\s*:\s*[\"']([^\"']+)[\"']",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not action_match:
        return None
    if normalize_agent_action_name(action_match.group(1)) not in {"final answer", "final", "answer"}:
        return None

    input_match = re.search(
        r"[\"']action_input[\"']\s*:",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not input_match:
        return {"final_answer": ""}

    cursor = input_match.end()
    while cursor < len(cleaned) and cleaned[cursor].isspace():
        cursor += 1
    if cursor >= len(cleaned):
        return {"final_answer": ""}

    quote = cleaned[cursor] if cleaned[cursor] in {'\"', "'"} else ""
    if quote:
        tail = cleaned[cursor + 1 :].strip()
        closing = re.match(rf"(?s)(.*){re.escape(quote)}\s*}}\s*$", tail)
        if closing:
            answer = closing.group(1)
        else:
            answer = tail
            if answer.endswith("}"):
                answer = answer[:-1].rstrip()
            if answer.endswith(quote):
                answer = answer[:-1]
        return {"final_answer": answer}

    tail = cleaned[cursor:].strip()
    if tail.endswith("}"):
        tail = tail[:-1].rstrip()
    try:
        parsed_tail = parse_jsonish(tail)
    except Exception:
        parsed_tail = tail
    if isinstance(parsed_tail, dict):
        parsed_tail = json.dumps(parsed_tail, ensure_ascii=False)
    return {"final_answer": str(parsed_tail)}


def infer_tool_call_from_argument_object(payload: dict[str, Any]) -> dict[str, Any] | None:
    keys = set(payload.keys())
    if {"calendarId", "start", "end", "summary"}.issubset(keys):
        if "eventId" in keys:
            return {"tool_calls": [normalize_tool_call({"name": "patch_event", "arguments": payload})]}
        return {"tool_calls": [normalize_tool_call({"name": "create_event", "arguments": payload})]}
    return None


def parse_agent_decision(raw_text: str) -> dict[str, Any]:
    cleaned = strip_action_wrappers(raw_text)

    parsed: Any = None
    try:
        parsed = parse_jsonish(cleaned)
    except ValueError:
        parsed = None

    if not _is_agent_decision_shaped(parsed):
        first_parseable: Any = None
        for candidate in iter_balanced_json_objects(cleaned):
            if _is_agent_decision_shaped(candidate):
                parsed = candidate
                break
            if first_parseable is None:
                first_parseable = candidate
        else:
            if not _is_agent_decision_shaped(parsed) and first_parseable is not None:
                parsed = first_parseable

    if parsed is None:
        recovered = parse_malformed_final_answer_decision(cleaned)
        if recovered is not None:
            return recovered
        raise ValueError(f"Unable to parse JSON from model output: {raw_text[:200]}")

    if isinstance(parsed, list):
        return {"tool_calls": [normalize_tool_call(item) for item in parsed]}
    if isinstance(parsed, dict):
        if "action" in parsed:
            action = str(parsed.get("action", "")).strip()
            normalized_action = normalize_agent_action_name(action)
            action_input = parsed.get("action_input", {})
            if normalized_action in {"final answer", "final", "answer"}:
                if isinstance(action_input, dict):
                    action_input = json.dumps(action_input, ensure_ascii=False)
                return {"final_answer": str(action_input)}
            if normalized_action == "clarify":
                question = action_input.get("question", "") if isinstance(action_input, dict) else str(action_input)
                return {"clarify": question}
            return {"tool_calls": [normalize_tool_call({"name": action, "arguments": action_input})]}
        if "tool_calls" in parsed:
            tool_calls = parsed["tool_calls"] or []
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]
            return {"tool_calls": [normalize_tool_call(item) for item in tool_calls]}
        if "final_answer" in parsed:
            return {"final_answer": str(parsed["final_answer"])}
        if "name" in parsed or "function" in parsed:
            return {"tool_calls": [normalize_tool_call(parsed)]}
        inferred = infer_tool_call_from_argument_object(parsed)
        if inferred is not None:
            return inferred
    raise ValueError(f"Unsupported next-action payload: {raw_text[:200]}")


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


def canonical_event_llm_target(labels: dict[str, Any], *, terminal: bool | None = None) -> dict[str, Any]:
    """Reduce a full canonical-event label row to the causal-LM target schema.

    Beam planning consumes action_type plus five scored next-state fields. The LLM world
    model is trained to emit exactly those fields plus a binary terminal flag, so generation
    capacity is not spent on weak/unconsumed labels such as recommended_abstract_action or
    missing_information_type.
    """
    target = {field: labels[field] for field in CANONICAL_EVENT_BEAM_TARGET_FIELDS if field in labels}
    if terminal is not None:
        target[CANONICAL_EVENT_TERMINAL_FIELD] = (
            CANONICAL_EVENT_TERMINAL_VALUES[1] if terminal else CANONICAL_EVENT_TERMINAL_VALUES[0]
        )
    return target


def format_canonical_event_target(labels: dict[str, Any]) -> str:
    """Serialize the reduced canonical-event target deterministically."""
    ordered = {field: labels[field] for field in CANONICAL_EVENT_LLM_TARGET_FIELDS if field in labels}
    return json.dumps(ordered, ensure_ascii=False, sort_keys=False)


def build_canonical_event_instruction() -> str:
    """System prompt listing the reduced beam-planning fields and terminal flag."""
    lines = [
        "You are an enterprise world model. Given the system prompt, user task, recent "
        "history, and the current action, predict the resulting annotated state.",
        "Return ONLY a JSON object with exactly these keys, in this order, and no other text:",
    ]
    for field in CANONICAL_EVENT_BEAM_TARGET_FIELDS:
        allowed = ", ".join(CANONICAL_EVENT_ALLOWED_VALUES[field])
        lines.append(f"- {field}: one of [{allowed}]")
    allowed_terminal = ", ".join(CANONICAL_EVENT_TERMINAL_VALUES)
    lines.append(
        f"- {CANONICAL_EVENT_TERMINAL_FIELD}: one of [{allowed_terminal}]; "
        "use finished only when this action is the final step that completes or ends the task"
    )
    return "\n".join(lines)


def extract_canonical_event_examples(
    paths: list[Path],
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
) -> list[WorldModelStateExample]:
    """Load canonical-event JSONL label rows into WorldModelStateExample.

    The source JSONL still carries the full 11-field canonical_event_with_nudge label. For
    the causal-LM canonical target we keep only the fields consumed by beam planning and add
    ``terminal`` by marking the last interaction_index in each trajectory as finished. Rows
    missing any required beam-planning field are skipped.
    """
    state_history_size = max(0, state_history_size)
    skipped = 0
    loaded: list[tuple[dict[str, Any], dict[str, Any], int, str]] = []
    max_interaction_index: dict[str, int] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                labels = split_canonical_labels(row)
                if labels is None or any(field not in labels for field in CANONICAL_EVENT_BEAM_TARGET_FIELDS):
                    skipped += 1
                    continue
                trajectory_id = str(row.get("trajectory_id", row_index))
                interaction_index = int(row.get("interaction_index", 0) or 0)
                max_interaction_index[trajectory_id] = max(
                    interaction_index, max_interaction_index.get(trajectory_id, interaction_index)
                )
                loaded.append((row, labels, interaction_index, trajectory_id))

    examples: list[WorldModelStateExample] = []
    for row, labels, interaction_index, trajectory_id in loaded:
        history = row.get("state_history") or []
        reduced_labels = canonical_event_llm_target(
            labels,
            terminal=interaction_index == max_interaction_index.get(trajectory_id, interaction_index),
        )
        examples.append(
            WorldModelStateExample(
                trajectory_id=trajectory_id,
                trajectory_index=int(row.get("trajectory_index", 0) or 0),
                interaction_index=interaction_index,
                system_prompt=row.get("system_prompt") or "",
                user_prompt=row.get("task_prompt") or row.get("user_prompt") or "",
                action=row.get("action"),
                state_history=list(history[-state_history_size:]) if state_history_size else [],
                input_history=list((row.get("input_history") or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:]),
                previous_state=row.get("previous_state") or {},
                state=reduced_labels,
                error_payload=str(row.get("observed_error_payload") or ""),
                tool_output=str(row.get("observed_tool_output") or ""),
            )
        )
    if skipped:
        print(f"canonical-event loader: skipped {skipped} row(s) with missing/unparseable labels")
    return examples

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

        for cursor in range(len(messages) - 1):
            message = messages[cursor]
            next_message = messages[cursor + 1]
            if not state_message_follows_tool_action(message, next_message):
                continue
            raw_state_content = next_message.get("content")
            current_state = sanitize_state_content(raw_state_content)
            error_payload = extract_state_error_payload(raw_state_content)
            tool_output = extract_state_tool_output(raw_state_content)
            if previous_state is None:
                previous_state = make_blank_state_like(current_state)
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
                    previous_state=previous_state,
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
            previous_state = current_state
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
        user_prompt = user_messages[-1] if user_messages else ""
        tasks.append(
            TaskTrajectory(
                trajectory_index=trajectory_index,
                system_prompt=system_prompt,
                user_messages=user_messages,
                steps=[],
                final_answer=final_answer,
                gym_task_config_name=resolve_gym_task_config_name(trajectory),
                initial_state=make_blank_state_like(first_state),
                initial_canonical_observation=build_stage_plan_canonical_observation(
                    first_state,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                ),
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
            if replay_task_has_unknown_gym_config(existing) and not replay_task_has_unknown_gym_config(task):
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
            raw_weights[label] = 1.0 if beta == 0.0 else (1.0 - beta) / (1.0 - (beta ** count))
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
    state_context_text = build_state_context_input_text(example)
    if include_input_history and example.input_history:
        state_context_text += (
            "\n\nRecent action/observation history (oldest to newest; input only, not part of the target):\n"
            + normalize_world_model_input_history_text(example.input_history)
        )
    if target_mode == WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY:
        if include_error_message:
            system_instruction = (
                "You are a world model for enterprise task trajectories. "
                "Given the system prompt, user prompt, recent state history, and current action, "
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
                "You are a world model for enterprise task trajectories. "
                "Given the system prompt, user prompt, recent state history, and current action, "
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
                    f"User prompt:\n{example.user_prompt}\n\n"
                    f"{state_context_text}\n\n"
                    f"Action:\n{action_text}\n\n"
                    f"{user_instruction}"
                ),
            },
        ]
    if target_mode == WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY:
        if include_stage:
            system_instruction = (
                "You are a world model for enterprise task trajectories. "
                "Given the system prompt, user prompt, recent state history, and current action, "
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
                "You are a world model for enterprise task trajectories. "
                "Given the system prompt, user prompt, recent state history, and current action, "
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
                "You are a world model for enterprise task trajectories. "
                "Given the system prompt, user prompt, recent state history, and current action, "
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
                    f"User prompt:\n{example.user_prompt}\n\n"
                    f"{state_context_text}\n\n"
                    f"Action:\n{action_text}\n\n"
                    f"{user_instruction}"
                ),
            },
        ]
    if target_mode == WORLD_MODEL_TARGET_TOOL_OUTPUT:
        return [
            {
                "role": "system",
                "content": (
                    "You are a world model for enterprise task trajectories. "
                    "Given the system prompt, user prompt, recent state history, and current action, "
                    "predict the raw tool output (the data the tool would return, including "
                    "API responses on success and error messages on failure). "
                    "Return only the tool output text, exactly as the tool would return it."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"System prompt:\n{system_prompt_text}\n\n"
                    f"User prompt:\n{example.user_prompt}\n\n"
                    f"{state_context_text}\n\n"
                    f"Action:\n{action_text}\n\n"
                    "Predict the tool output. /no_think"
                ),
            },
        ]
    if is_canonical_event_target(target_mode):
        return [
            {"role": "system", "content": build_canonical_event_instruction()},
            {
                "role": "user",
                "content": (
                    f"System prompt:\n{system_prompt_text}\n\n"
                    f"User prompt:\n{example.user_prompt}\n\n"
                    f"{state_context_text}\n\n"
                    f"Action:\n{action_text}\n\n"
                    "Predict the annotated state as JSON. /no_think"
                ),
            },
        ]
    return [
        {
            "role": "system",
            "content": (
                "You are a world model for enterprise task trajectories. "
                "Given the system prompt, user prompt, recent state history, and current action, predict the resulting state. "
                "Return JSON only. Preserve the state schema used by the previous state. "
                "For enterprise_ops_objects_process_relational_constraints_history_v1 states, predict outcome, process_state, and history_context fields in that schema; do not output objects_artifacts, relational_state, or constraints. "
                "For legacy states, exclude `state.context.last_tool_output` from the output."
            ),
        },
        {
            "role": "user",
            "content": (
                f"System prompt:\n{system_prompt_text}\n\n"
                f"User prompt:\n{example.user_prompt}\n\n"
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
    if is_tool_output_target(target_mode):
        return [{"role": "assistant", "content": example.tool_output or ""}]
    if is_canonical_event_target(target_mode):
        # `.state` holds the reduced beam-field-plus-terminal annotation for canonical-event examples.
        return [{"role": "assistant", "content": format_canonical_event_target(example.state)}]
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


def build_agent_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are evaluating whether an agent chooses the correct next action in an enterprise tool-use task. "
                "Return JSON only. If the task requires a tool next, return "
                '{"tool_calls": [{"name": "<tool_name>", "arguments": {...}}]}. '
                'If the task is complete, return {"final_answer": "<answer>"}.'
            ),
        },
        {
            "role": "user",
            "content": f"Conversation (/no_think):\n{render_messages(messages)}\n\nNext action JSON:",
        },
    ]


def get_tool_schema_safe_local(tool: Any) -> dict[str, Any]:
    try:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is None:
            return {}
        if isinstance(args_schema, dict):
            return args_schema
        if hasattr(args_schema, "model_json_schema"):
            return args_schema.model_json_schema()
        if hasattr(args_schema, "schema"):
            return args_schema.schema()
    except Exception:
        return {}
    return {}


def build_react_tool_descriptions(tools: list[Any]) -> str:
    parts: list[str] = []
    for tool in tools:
        schema = get_tool_schema_safe_local(tool)
        description = getattr(tool, "description", "") or ""
        chunk = [f"Tool: {tool.name}", f"Description: {description}"]
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
        if properties:
            chunk.append("Parameters:")
            for name, info in properties.items():
                param_type = info.get("type", "string") if isinstance(info, dict) else "string"
                marker = "required" if name in required else "optional"
                chunk.append(f"- {name} ({param_type}, {marker})")
        parts.append("\n".join(chunk))
    return "\n\n".join(parts)


def build_react_system_prompt(tool_descriptions: str) -> str:
    return (
        "You are a ReAct reasoning agent. Follow this EXACT format:\n\n"
        f"AVAILABLE TOOLS:\n{tool_descriptions}\n\n"
        "RESPONSE FORMAT:\n"
        "Step 1: THINK NODE\n"
        "- Perform granular thinking.\n"
        "- Think about what you need to do next based on previous actions.\n"
        '- For THINK NODE return JSON: {"thought": "..."}\n\n'
        "Step 2: ACTION NODE\n"
        "- Output the best action.\n"
        "- For TOOL CALL return JSON:\n"
        '{"action": "<tool_name>", "action_input": {"param1": "value1"}}\n\n'
        "- For FINAL ANSWER return JSON:\n"
        '{"action": "Final Answer", "action_input": "Your complete response"}\n\n'
        "CRITICAL RULES:\n"
        "1. Use exact tool names.\n"
        "2. Use exact parameter names.\n"
        "3. Return JSON only.\n"
        "4. Call tools before answering when information is needed.\n"
        "5. Do not guess.\n"
        "6. Provide the complete final answer when done."
    )


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


def _trim_replay_messages_to_budget(
    messages: list[dict[str, str]],
    char_budget: int,
) -> list[dict[str, str]]:
    """Drop the oldest tool/assistant turns until the prompt fits the budget.

    Keeps the system message (if any) and the first user message intact so the
    agent always retains the task context, then re-adds the most recent turns
    backwards until the budget is exhausted.
    """
    if char_budget <= 0:
        return messages
    total = sum(len(msg.get("content", "")) for msg in messages)
    if total <= char_budget:
        return messages

    pinned: list[dict[str, str]] = []
    rest: list[dict[str, str]] = []
    seen_first_user = False
    for msg in messages:
        role = msg.get("role")
        if role == "system" and len(pinned) == 0:
            pinned.append(msg)
        elif role == "user" and not seen_first_user:
            pinned.append(msg)
            seen_first_user = True
        else:
            rest.append(msg)

    used = sum(len(msg.get("content", "")) for msg in pinned)
    kept_tail: list[dict[str, str]] = []
    for msg in reversed(rest):
        size = len(msg.get("content", ""))
        if used + size > char_budget and kept_tail:
            break
        kept_tail.append(msg)
        used += size
    kept_tail.reverse()

    if len(rest) > len(kept_tail):
        notice = {
            "role": "user",
            "content": (
                f"[CONTEXT TRIMMED] Dropped {len(rest) - len(kept_tail)} earlier "
                "thought/action/observation turns to fit the context window."
            ),
        }
        kept_tail.insert(0, notice)
    return pinned + kept_tail


def filter_messages_for_react_replay(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, str]]:
    observation_limit = _REPLAY_LIMITS["observation_chars"]
    history_budget = _REPLAY_LIMITS["history_budget_chars"]
    filtered: list[dict[str, str]] = []
    system_added = False

    for message in messages:
        role = message.get("role")
        if role == "system":
            content = str(message.get("content", ""))
            if content.startswith("[INTERNAL_WORLD_MODEL_THINKING]"):
                continue
            if not system_added:
                filtered.append({"role": "system", "content": system_prompt})
                system_added = True
            continue
        if role == "user":
            filtered.append({"role": "user", "content": str(message.get("content", ""))})
            continue
        if role == "assistant":
            if message.get("tool_calls"):
                tool_call = message["tool_calls"][0]
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_args = tool_call.get("function", {}).get("arguments", {})
                filtered.append(
                    {
                        "role": "assistant",
                        "content": f"Action: Using {tool_name} with args: {json.dumps(tool_args, ensure_ascii=False)}",
                    }
                )
            elif message.get("content"):
                filtered.append({"role": "assistant", "content": str(message.get("content"))})
            continue
        if role == "tool":
            tool_output = stringify_tool_output(message.get("content"))
            tool_output = truncate_for_replay(tool_output, observation_limit)
            filtered.append(
                {
                    "role": "user",
                    "content": (
                        f"Observation: Tool '{message.get('name', 'unknown')}' returned: "
                        f"{tool_output}"
                    ),
                }
            )

    if not system_added:
        filtered.insert(0, {"role": "system", "content": system_prompt})
    return _trim_replay_messages_to_budget(filtered, history_budget)


def build_conversation_summary_for_replay(messages: list[dict[str, Any]]) -> str:
    summary_parts: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("content"):
            summary_parts.append(f"Thought/Action: {message['content']}")
        elif role == "assistant" and message.get("tool_calls"):
            tool_call = message["tool_calls"][0]
            summary_parts.append(
                f"Action: {tool_call.get('function', {}).get('name', 'unknown')}("
                f"{json.dumps(tool_call.get('function', {}).get('arguments', {}), ensure_ascii=False)})"
            )
        elif role == "tool":
            summary_parts.append(
                f"Observation: Tool {message.get('name', 'unknown')} returned "
                f"{stringify_tool_output(message.get('content'))}"
            )
    return "\n".join(summary_parts[-5:])


IMAGINED_TRAJECTORY_AGENT_POLICY = """

IMAGINED TRAJECTORY POLICY
You are generating hypothetical next steps for planning only. Your goal is to explore useful possible next actions, not to finish early.
Prefer information-gathering and validation actions before irreversible updates.
Do not produce a final answer unless all required task conditions are explicitly satisfied by observed or imagined evidence.
When a world-model prediction is uncertain, incomplete, generic, or lacks concrete IDs/counts, treat it as non-authoritative and continue exploring.
For each step, identify the unresolved requirement and choose one action that would reduce uncertainty.
Prefer lookups that recover concrete IDs, labels, permissions, cases, files, events, or existing records.
Avoid repeating the same tool call with the same arguments.
If a tool call failed, try one alternative formulation or prerequisite lookup before giving up.
Do not assume absence from a predicted empty result unless the query used verified IDs and narrow filters.
Do not claim completion inside imagined trajectories unless every required mutation has been accounted for.
Return a tool call unless all requirements are explicitly satisfied, at least two distinct lookup/update strategies have failed, or the next step requires real user clarification.
If unsure, explore with a read-only lookup.
"""


def imagined_trajectory_system_prompt(system_prompt: str) -> str:
    if IMAGINED_TRAJECTORY_AGENT_POLICY.strip() in system_prompt:
        return system_prompt
    return (system_prompt or "").rstrip() + IMAGINED_TRAJECTORY_AGENT_POLICY


def build_react_think_messages(
    messages: list[dict[str, Any]],
    current_query: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    filtered_messages = filter_messages_for_react_replay(messages, system_prompt)
    conversation_summary = build_conversation_summary_for_replay(messages)
    think_prompt = (
        f"Current Query: {current_query}\n"
        f"Conversation Summary: {conversation_summary}\n\n"
        "Status: Analyzing task...\n\n"
        'Think: What should I do next based on the current query and progress to complete the task?\n'
        'Respond in JSON format: {"thought": "your reasoning here"}'
    )
    return filtered_messages + [{"role": "user", "content": think_prompt}]


def build_react_action_messages(
    messages: list[dict[str, Any]],
    current_query: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    filtered_messages = filter_messages_for_react_replay(messages, system_prompt)
    conversation_summary = build_conversation_summary_for_replay(messages)
    action_prompt = (
        f"Current Query: {current_query}\n"
        f"Conversation Summary: {conversation_summary}\n\n"
        "Based on your previous thought, select and execute the most appropriate action.\n"
        "Return JSON only in one of the allowed ACTION NODE formats. "
        "Do not return tool arguments by themselves; always include the tool name using "
        "{\"action\": \"<tool_name>\", \"action_input\": {...}} or {\"tool_calls\": [...]}."
    )
    return filtered_messages + [{"role": "user", "content": action_prompt}]


# Tier-1 planning speedup: produce all K candidate actions in a single LLM call
# instead of K sequential temperature-sampled generations. Toggle to False to
# A/B benchmark against the original sequential proposal path.
SINGLE_CALL_CANDIDATE_PROPOSAL_DEFAULT = True

# Tier-1 imagined-rollout speedups (set from CLI flags in main()):
#   IMAGINED_SINGLE_CALL_STEP    thought + action in ONE generation per imagined step
#                                instead of the two-call think-then-act sequence.
#   IMAGINED_PARALLEL_ROLLOUTS   advance all --imagined-trajectory-rollouts chains in
#                                lockstep, batching each step's agent and world-model
#                                generations across rollouts (see generate_many).
IMAGINED_SINGLE_CALL_STEP = True
IMAGINED_PARALLEL_ROLLOUTS = True

# Open-loop planning (beam_plan-style skeletons for the text world model):
#   IMAGINED_ROLLOUT_MODE == "open_loop" generates ALL N rollouts' action sequences in
#   ONE agent call -- actions are committed without seeing any predicted state -- and the
#   world model then fills in the state chain per plan, batched across plans at each step.
#   Call budget: 1 agent generation + max_steps batched WM generations, vs the closed-loop
#   lockstep's ~2*max_steps batched generations. The trade: no mid-trajectory reaction to
#   predicted failures. "closed_loop" (default) keeps the previous behaviour.
IMAGINED_ROLLOUT_MODE = "closed_loop"


def build_react_open_loop_plan_messages(
    messages: list[dict[str, Any]],
    current_query: str,
    system_prompt: str,
    max_steps: int,
) -> list[dict[str, str]]:
    """Ask for ONE open-loop plan (used k times in parallel, one plan per sample).

    Preferred over asking a single response for k plans: k plans in one response is k times
    the output tokens on ONE serial decode stream, whereas k sampled responses decode
    concurrently after a single shared prefill -- same candidate set, ~k times less wall clock.
    Diversity comes from sampling temperature rather than from an instruction to differ.
    """
    filtered_messages = filter_messages_for_react_replay(messages, system_prompt)
    conversation_summary = build_conversation_summary_for_replay(messages)
    plan_prompt = (
        f"Current Query: {current_query}\n"
        f"Conversation Summary: {conversation_summary}\n\n"
        f"Plan ahead WITHOUT executing anything: propose ONE plan of up to {max_steps} next "
        "actions to complete the task.\n"
        'Return ONE JSON object only: {"strategy": "<one-line rationale>", '
        '"steps": [<action>, ...]}.\n'
        "Each <action> uses an allowed ACTION NODE format: "
        '{"action": "<tool_name>", "action_input": {...}} or {"tool_calls": [...]}; '
        'the plan may END with {"final_answer": "..."} when the task would be complete.\n'
        "Rules:\n"
        '- For any argument whose value depends on an earlier step\'s result, use a symbolic '
        'reference string like "$step1.field" instead of guessing a concrete value.\n'
        "- Do not return tool arguments by themselves; always include the tool name."
    )
    return filtered_messages + [{"role": "user", "content": plan_prompt}]


def normalize_open_loop_plan_step(step: Any) -> dict[str, Any] | None:
    """One plan step -> the decision shape used by the imagined-rollout machinery
    ({"tool_calls": [...]} or {"final_answer": ...}), or None if unusable."""
    if not isinstance(step, dict):
        return None
    try:
        decision = parse_agent_decision(json.dumps(step, ensure_ascii=False))
    except Exception:                                             # noqa: BLE001
        return None
    if "clarify" in decision:
        # An open-loop plan cannot pause for clarification; treat it as unusable so the
        # plan truncates here rather than imagining a question nobody will answer.
        return None
    return decision


def parse_open_loop_plans(raw_text: str, num_plans: int, max_steps: int) -> list[dict[str, Any]]:
    """Tolerantly parse the one-call plans payload into [{"strategy", "steps"}, ...].

    Accepts {"plans": [...]}, a bare JSON array of plans, and plans given either as
    {"strategy", "steps"} objects or bare step arrays. Unparseable plans/steps are
    dropped (the caller falls back to closed-loop when nothing survives)."""
    cleaned = strip_action_wrappers(raw_text)
    payload: Any = None
    try:
        payload = parse_jsonish(cleaned)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        # Either the multi-plan envelope {"plans": [...]} or a single plan {"steps": [...]}
        # (what build_react_open_loop_plan_messages asks each sample for).
        payload = payload.get("plans") if "plans" in payload else [payload]
    if not isinstance(payload, list):
        for candidate in iter_balanced_json_objects(cleaned):
            if isinstance(candidate, dict) and isinstance(candidate.get("plans"), list):
                payload = candidate["plans"]
                break
            if isinstance(candidate, dict) and isinstance(candidate.get("steps"), list):
                payload = [candidate]
                break
    if not isinstance(payload, list):
        return []

    plans: list[dict[str, Any]] = []
    for plan in payload[:num_plans]:
        if isinstance(plan, dict):
            strategy = str(plan.get("strategy") or "").strip()
            raw_steps = plan.get("steps")
        else:
            strategy, raw_steps = "", plan
        if not isinstance(raw_steps, list):
            continue
        steps: list[dict[str, Any]] = []
        for raw_step in raw_steps[:max_steps]:
            decision = normalize_open_loop_plan_step(raw_step)
            if decision is None:
                break
            steps.append(decision)
            if "final_answer" in decision:
                break
        if steps:
            plans.append({"strategy": strategy, "steps": steps})
    return plans


def build_react_step_messages(
    messages: list[dict[str, Any]],
    current_query: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Single-call variant of think-then-act: one generation returns the thought AND the
    action in one JSON object. Halves the per-imagined-step LLM call count; toggle with
    --no-imagined-single-call-step to A/B against the two-call sequence."""
    filtered_messages = filter_messages_for_react_replay(messages, system_prompt)
    conversation_summary = build_conversation_summary_for_replay(messages)
    step_prompt = (
        f"Current Query: {current_query}\n"
        f"Conversation Summary: {conversation_summary}\n\n"
        "Think about what to do next based on the current query and progress, then select "
        "and execute the most appropriate action.\n"
        "Return ONE JSON object only, containing BOTH your reasoning and the action:\n"
        '{"thought": "your reasoning here", "action": "<tool_name>", "action_input": {...}}\n'
        "Instead of action/action_input you may use one of the other allowed ACTION NODE "
        'formats alongside "thought": {"thought": "...", "tool_calls": [...]}, '
        '{"thought": "...", "final_answer": "..."}, or {"thought": "...", "clarify": "..."}. '
        "Do not return tool arguments by themselves; always include the tool name."
    )
    return filtered_messages + [{"role": "user", "content": step_prompt}]


def build_react_candidate_actions_messages(
    messages: list[dict[str, Any]],
    current_query: str,
    system_prompt: str,
    candidate_count: int,
) -> list[dict[str, str]]:
    """Ask the agent for several distinct candidate actions in one generation.

    Mirrors build_react_action_messages but requests a JSON object
    ``{"candidates": [<action>, ...]}`` whose elements each use a normal ACTION
    NODE format. This replaces K sequential action generations with a single
    call, which is the dominant cost in top-k imagined search.
    """
    filtered_messages = filter_messages_for_react_replay(messages, system_prompt)
    conversation_summary = build_conversation_summary_for_replay(messages)
    action_prompt = (
        f"Current Query: {current_query}\n"
        f"Conversation Summary: {conversation_summary}\n\n"
        f"Based on your previous thought, propose {candidate_count} DISTINCT candidate next "
        "actions, ordered best first. Make them meaningfully different (different tools or "
        "different arguments), not paraphrases of one another.\n"
        'Return JSON only, exactly in the form {"candidates": [<action>, <action>, ...]} '
        f"with at most {candidate_count} elements. Each <action> must use one of the allowed "
        'ACTION NODE formats, e.g. {"action": "<tool_name>", "action_input": {...}}, '
        '{"tool_calls": [...]}, {"final_answer": "..."}, or {"clarify": "..."}. '
        "Do not nest another candidates list inside an action."
    )
    return filtered_messages + [{"role": "user", "content": action_prompt}]


def parse_candidate_actions(raw_text: str, candidate_count: int) -> list[str]:
    """Split a single multi-candidate response into per-candidate raw action strings.

    Each returned string is suitable for ``parse_agent_decision``. Falls back to
    treating the whole response as a single candidate so callers always get at
    least one action to try even when the structured list cannot be recovered.
    """
    cleaned = strip_code_fence(raw_text)
    try:
        parsed: Any = parse_jsonish(cleaned)
    except Exception:
        parsed = None

    candidates: list[Any] = []
    if isinstance(parsed, dict):
        for key in ("candidates", "actions", "options"):
            value = parsed.get(key)
            if isinstance(value, list) and value:
                candidates = value
                break
        if not candidates:
            candidates = [parsed]  # a single action object returned directly
    elif isinstance(parsed, list) and parsed:
        # A bare list is only K candidates when every element is action-shaped;
        # otherwise it is a single tool_calls list (one action).
        if all(isinstance(item, dict) and _is_agent_decision_shaped(item) for item in parsed):
            candidates = parsed
        else:
            candidates = [parsed]

    raw_actions: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = candidate if isinstance(candidate, str) else json.dumps(candidate, ensure_ascii=False)
        text = text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        raw_actions.append(text)
        if len(raw_actions) >= max(1, candidate_count):
            break

    if not raw_actions:
        fallback = cleaned.strip()
        if fallback:
            raw_actions.append(fallback)
    return raw_actions


def build_agent_prompt(
    messages: list[dict[str, Any]],
    tokenizer: Any | None = None,
    disable_chat_template: bool = False,
) -> str:
    return apply_chat_template_or_fallback(
        tokenizer,
        build_agent_chat_messages(messages),
        add_generation_prompt=True,
        disable_chat_template=disable_chat_template,
    )


def build_world_model_reflection(predictions: list[dict[str, str]]) -> dict[str, Any]:
    lines = []
    for prediction in predictions:
        lines.append(f"{prediction['name']}: {prediction['content']}")
    return {
        "role": "system",
        "content": "[WORLD_MODEL_PREDICTION_FOR_PLANNING_ONLY]\n" + "\n\n".join(lines),
    }


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


def require_llm_class():
    try:
        from src.llm import LLM
    except ImportError:
        try:
            from src.llm import LLM
        except ImportError as exc:
            print(exc)
            raise SystemExit(
                "API-backed agent models require src/llm.py and its dependencies to be importable."
            ) from exc
    return LLM


def require_enterprise_runtime():
    try:
        from contextlib import AsyncExitStack
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError as exc:
        raise SystemExit(
            "Actual MCP execution evaluation requires `langchain-mcp-adapters` and its runtime dependencies."
        ) from exc
    return AsyncExitStack, MultiServerMCPClient, load_mcp_tools


def stringify_tool_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def resolve_enterprise_mcp_config(
    explicit_path: Path | None,
    enterprise_runner: Path,
) -> Path | None:
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None
    default_path = enterprise_runner.parent / "mcp_config_http.json"
    return default_path if default_path.exists() else None


def full_world_model_state_for_agent(predicted_state: Any) -> Any:
    if predicted_state is None:
        return None
    try:
        return sanitize_state_content(predicted_state)
    except Exception:
        return predicted_state


def world_model_state_observation_payload(
    predicted_state: Any,
    predicted_error_message: str | None = None,
    raw_prediction: str | None = None,
) -> dict[str, Any]:
    return {
        "predicted_state": full_world_model_state_for_agent(predicted_state),
        "predicted_state_summary": (
            summarize_state_for_planning(predicted_state)
            if isinstance(predicted_state, dict)
            else None
        ),
        "predicted_error_message": predicted_error_message or None,
        "raw_world_model_prediction": raw_prediction or None,
    }


def _summarize_revision_rollouts(
    rollouts: list[dict[str, Any]] | None,
    observation_cap: int,
) -> str:
    """Render a compact human-readable summary of multi-step / multi-rollout lookaheads."""
    if not rollouts:
        return ""
    sections: list[str] = []
    rollout_count = len(rollouts)
    success_count = sum(1 for r in rollouts if r.get("all_predicted_success"))
    failure_count = sum(1 for r in rollouts if r.get("any_predicted_failure"))
    sections.append(
        f"Imagined rollouts: {rollout_count} total — "
        f"{success_count} predicted fully successful, "
        f"{failure_count} predicted at least one failure."
    )
    for rollout in rollouts:
        idx = rollout.get("rollout_index")
        temp = rollout.get("rollout_temperature")
        depth_target = rollout.get("lookahead_steps_target")
        depth_taken = rollout.get("lookahead_steps_taken")
        header = (
            f"\n[Rollout #{idx} (temperature={temp}, depth={depth_taken}/{depth_target}, "
            f"all_success={rollout.get('all_predicted_success')})]"
        )
        sections.append(header)
        for step in rollout.get("steps", []):
            step_offset = step.get("step_offset")
            tool_calls = step.get("tool_calls") or []
            tool_names = [c.get("name", "") for c in tool_calls]
            line = f"  step+{step_offset}: tools={tool_names}"
            err = (step.get("predicted_error_message") or "").strip()
            preview = (step.get("predicted_tool_output_preview") or "").strip()
            if err:
                line += f" | predicted_error={truncate_for_replay(err, observation_cap // 4)}"
            if preview:
                line += f" | predicted_output={truncate_for_replay(preview, observation_cap // 4)}"
            if step.get("predicted_state") is not None:
                line += (
                    " | predicted_state="
                    + json.dumps(step.get("predicted_state"), ensure_ascii=False, default=str)
                )
            elif step.get("raw_world_model_prediction"):
                line += " | raw_world_model_prediction=" + str(step.get("raw_world_model_prediction"))
            decision = step.get("imagined_decision")
            if decision:
                line += f" | imagined_decision={json.dumps(decision, ensure_ascii=False)}"
            if step.get("imagined_repeated_tool_call_loop"):
                line += " | repeated_tool_call_loop"
            if step.get("imagined_parse_error"):
                line += f" | parse_error={step['imagined_parse_error']}"
            sections.append(line)
    return "\n".join(sections)


def build_internal_thinking_messages(
    conversation: list[dict[str, Any]],
    proposed_calls: list[dict[str, Any]],
    feedbacks: list[dict[str, Any]],
    iteration: int,
    max_iterations: int,
    world_model_target: str = WORLD_MODEL_TARGET_STATE,
    revision_rollouts: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    target_is_tool_output = is_tool_output_target(world_model_target)
    observation_cap = _REPLAY_LIMITS["observation_chars"]
    rollouts_summary = _summarize_revision_rollouts(revision_rollouts, observation_cap)
    predicted_failure = any(not feedback.get("predicted_success", True) for feedback in feedbacks)
    predicted_errors = [
        (feedback.get("predicted_error_message") or "").strip()
        for feedback in feedbacks
        if not feedback.get("predicted_success", True)
    ]
    predicted_errors = [msg for msg in predicted_errors if msg]
    predicted_tool_outputs = [
        (feedback.get("predicted_tool_output") or "").strip()
        for feedback in feedbacks
    ]
    predicted_tool_outputs = [text for text in predicted_tool_outputs if text]
    predicted_state_payloads = [
        world_model_state_observation_payload(
            feedback.get("predicted_state"),
            feedback.get("predicted_error_message"),
            feedback.get("raw_prediction"),
        )
        for feedback in feedbacks
        if feedback.get("predicted_state") is not None
    ]
    if target_is_tool_output:
        observation_cap = _REPLAY_LIMITS["observation_chars"]
        directive = (
            "The world model predicted what the tool calls will return as raw tool output, "
            "but did not classify success vs. failure — that judgment is yours. "
            "Inspect the predicted tool output(s) below and decide whether to revise the tool calls. "
            "Modify the tool name, arguments, or pick a different tool if the predicted output "
            "indicates the call is wrong, off-target, or will not surface the information you need. "
            "Return the same tool calls unchanged if the predicted output looks correct and useful.\n"
        )
        if predicted_tool_outputs:
            directive += (
                "Predicted tool output(s) from the world model:\n"
                + "\n---\n".join(
                    truncate_for_replay(text, observation_cap)
                    for text in predicted_tool_outputs
                )
                + "\n"
            )
        else:
            directive += (
                "(The world model produced no tool-output text for these calls; "
                "treat that as a weak signal and decide whether to revise.)\n"
            )
        user_content = (
            f"Conversation so far:\n{render_messages(conversation)}\n\n"
            f"Current proposed tool calls:\n{json.dumps(proposed_calls, indent=2, ensure_ascii=False)}\n\n"
            f"World-model feedback for internal thinking iteration {iteration}/{max_iterations}:\n"
            f"{directive}"
        )
        if rollouts_summary:
            user_content += "\n\nMulti-step lookahead rollouts (use these to inform your decision):\n" + rollouts_summary
        return [
            {
                "role": "system",
                "content": (
                    "You are reviewing planned tool calls before actual MCP execution, "
                    "using the world model's predicted tool output to inform your decision. "
                    "Return JSON only as {\"tool_calls\": [...]} and do not return a final answer."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
    if predicted_failure:
        directive = (
            "The world model predicts these tool calls will FAIL. "
            "You MUST modify the tool calls to address the predicted failure — "
            "do not return the same tool calls unchanged unless you have a concrete reason "
            "to disagree with the world model.\n"
        )
        if predicted_errors:
            directive += (
                "Predicted error message(s): "
                + " | ".join(predicted_errors)
                + "\n"
            )
        if predicted_tool_outputs:
            observation_cap = _REPLAY_LIMITS["observation_chars"]
            directive += (
                "Predicted tool output(s) from the world model:\n"
                + "\n---\n".join(
                    truncate_for_replay(text, observation_cap)
                    for text in predicted_tool_outputs
                )
                + "\n"
            )
        directive += (
            "Use this signal to change the tool name (for ambiguity errors), fix arguments "
            "(for validation errors), or pick a different tool entirely. "
        )
    else:
        directive = (
            "The world model predicts these tool calls will succeed. "
            "Return them unchanged unless you spot a concrete error. "
        )
        if predicted_tool_outputs:
            observation_cap = _REPLAY_LIMITS["observation_chars"]
            directive += (
                "\nPredicted tool output(s) from the world model:\n"
                + "\n---\n".join(
                    truncate_for_replay(text, observation_cap)
                    for text in predicted_tool_outputs
                )
            )
    state_payload_text = ""
    if predicted_state_payloads:
        state_payload_text = (
            "\n\nFull predicted state output(s) from the world model:\n"
            + json.dumps(predicted_state_payloads, indent=2, ensure_ascii=False)
        )
    user_content = (
        f"Conversation so far:\n{render_messages(conversation)}\n\n"
        f"Current proposed tool calls:\n{json.dumps(proposed_calls, indent=2, ensure_ascii=False)}\n\n"
        f"World-model feedback for internal thinking iteration {iteration}/{max_iterations}:\n"
        f"{json.dumps(feedbacks, indent=2, ensure_ascii=False)}"
        f"{state_payload_text}\n\n"
        f"{directive}"
    )
    if rollouts_summary:
        user_content += "\n\nMulti-step lookahead rollouts (use these to inform your decision):\n" + rollouts_summary
    return [
        {
            "role": "system",
            "content": (
                "You are revising planned tool calls before actual MCP execution. "
                "Use the world-model feedback to fix likely failures. "
                "Return JSON only as {\"tool_calls\": [...]} and do not return a final answer."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def summarize_state_for_planning(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(state, dict):
        return None
    context = state_context_from_any(state)
    process = state_process_from_any(state)
    summary = {
        "last_tool_execution_result": context.get("last_tool_execution_result"),
        "last_tool_name": context.get("last_tool_name"),
        "error_message": context.get("error_message"),
        "current_stage": process.get("current_stage"),
        "remaining_stages": process.get("remaining_stages"),
    }
    if is_enterprise_state_payload(state):
        body = state_body_from_any(state)
        delta = enterprise_state_delta_from_any(state)
        summary.update(
            {
                "state_schema": body.get("schema"),
                "mode": body.get("mode"),
                "outcome": delta.get("outcome") or body.get("outcome"),
                "process_state": delta.get("process_state") or body.get("process_state"),
                "last_tool_events": (
                    (delta.get("history_context") or {}).get("last_tool_events")
                    if isinstance(delta.get("history_context"), dict)
                    else (body.get("history_context") or {}).get("last_tool_events")
                ),
            }
        )
    return summary


def build_imagined_trajectory_prompt_message(imagined_steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": (
            "[IMAGINED_TRAJECTORY_FOR_PLANNING_ONLY]\n"
            + json.dumps(imagined_steps, ensure_ascii=False, indent=2)
        ),
    }


def build_imagined_trajectory_selection_judge_prompt() -> str:
    return (
        "You are selecting the best imagined enterprise-agent trajectory before any real tool execution.\n\n"
        "Choose the candidate most likely to succeed in the real environment. Prefer candidates that:\n"
        "- satisfy the user's requirements with the fewest missing dependencies\n"
        "- use tools and arguments coherently\n"
        "- avoid parse errors, empty actions, or repeated tool-call loops\n"
        "- make concrete progress toward a valid final answer\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        '  "selected_index": int,\n'
        '  "scores": [{"index": int, "score": float, "reason": "brief explanation"}],\n'
        '  "comments": "brief explanation"\n'
        "}"
    )


def _generate_with_optional_temperature(
    generator: Any,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
) -> str:
    try:
        return generator.generate_from_messages(messages, temperature=temperature)
    except TypeError:
        return generator.generate_from_messages(messages)


# Fan-out width for backends that parallelize LLM calls with threads (API endpoints and
# vLLM servers, whose continuous batching turns parallel requests into true batches).
# Set from --llm-batch-parallelism in main().
LLM_BATCH_PARALLELISM = 8


# Per-sample temperature ladder for k-sample planning calls (set from
# --sample-temperature-ladder in main()). Off by default: it forfeits the single-request n=k
# path, since one request carries one sampling config.
SAMPLE_TEMPERATURE_LADDER = False
SAMPLE_TEMPERATURE_LADDER_MAX = 1.2


def build_temperature_ladder(
    num_samples: int, base_temperature: float, ladder_max: float | None = None
) -> list[float]:
    """Spread k samples over temperatures instead of drawing them all at one setting.

    k identical requests (or `n=k`) are i.i.d. draws from the same sharply-peaked distribution,
    so several collapse onto the modal plan -- and the collapse that costs is at step 0, the
    only step that gets executed. A ladder keeps ONE near-greedy sample (the model's best single
    guess, so diversity is not bought by giving up the mode) and pushes the rest outward.

    The top of the ladder is capped: past ~1.2 the agent starts emitting malformed actions, and
    in this pipeline a parse failure TRUNCATES the plan at that step, which is worse than a
    duplicate.
    """
    k = max(1, int(num_samples))
    base = max(0.0, float(base_temperature))
    if k == 1 or base <= 0:
        return [base] * k
    ceiling = float(SAMPLE_TEMPERATURE_LADDER_MAX if ladder_max is None else ladder_max)
    top = min(ceiling, base * 1.7)
    bottom = min(base, max(0.05, base * 0.4))          # the exploit slot
    if k == 2:
        return [bottom, top]
    span = max(0.0, top - base * 0.85)
    return [bottom] + [
        round(base * 0.85 + span * index / (k - 2), 4) for index in range(k - 1)
    ]


def _requests_for_temperatures(generator: Any, temperatures: list[float]) -> int:
    """How many backend requests generate_many will issue for these per-item temperatures."""
    if hasattr(generator, "generate_from_messages_batch"):
        return len(set(temperatures))     # one padded batch per distinct temperature
    return len(temperatures)              # parallel (or serial) one-per-item


def sample_many(
    generator: Any,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    num_samples: int,
    temperatures: list[float] | None = None,
) -> tuple[list[str], int]:
    """k independent samples of the SAME prompt. Returns (texts, requests_issued).

    Prefers a backend that can produce k samples in ONE request/forward pass (`n=k` on
    vLLM/OpenAI, `num_return_sequences=k` on local HF): the shared prompt is prefilled once
    and the k sequences decode concurrently, so k candidates cost ~one generation instead of
    k. Backends without that capability fall back to generate_many, which issues k parallel
    requests (or one padded batch) -- same result, k prefills.

    `requests_issued` is reported so callers can log the true LLM call count rather than
    assuming it equals the number of samples.
    """
    k = max(1, int(num_samples))
    if k == 1:
        return [_generate_with_optional_temperature(generator, messages, temperature=temperature)], 1
    if temperatures is None and SAMPLE_TEMPERATURE_LADDER:
        temperatures = build_temperature_ladder(k, temperature)
    if temperatures:
        # A ladder cannot ride the single-request n=k path (one sampling config per request), so
        # it routes through generate_many: one padded batch per distinct temperature on local HF,
        # k parallel requests on API/vLLM backends.
        ladder = list(temperatures)[:k]
        ladder += [ladder[-1]] * (k - len(ladder))
        texts = generate_many(generator, [messages] * k, ladder)
        return texts, _requests_for_temperatures(generator, ladder)
    sampler = getattr(generator, "generate_samples", None)
    if sampler is not None:
        try:
            texts = sampler(messages, temperature=temperature, num_samples=k)
        except NotImplementedError:
            texts = None
        if texts:
            return list(texts)[:k], 1
    return generate_many(generator, [messages] * k, [temperature] * k), k


def generate_many(
    generator: Any,
    messages_list: list[list[dict[str, Any]]],
    temperatures: list[float],
) -> list[str]:
    """Run several independent generations as one batch instead of a serial loop.

    Dispatch, in order of preference:
      1. `generate_from_messages_batch` (HFTextGenerator): requests sharing a temperature
         are grouped into a single padded batched `generate`; a group of one keeps the
         batch-1 path, where speculative decoding applies.
      2. `supports_parallel_requests` backends (OpenAI/Azure/Gemini/src.llm wrappers):
         a thread pool of concurrent requests. vLLM's continuous batching makes these
         genuine server-side batches. Backends whose call mutates shared client state
         expose `clone_for_parallel_requests()` and each worker gets its own clone.
      3. Anything else: the original serial loop.

    A failed parallel request falls back to one serial retry so batching never
    introduces a new failure mode; order of results always matches the input order.
    """
    if len(messages_list) != len(temperatures):
        raise ValueError("messages_list and temperatures must have equal length")
    if not messages_list:
        return []
    if len(messages_list) == 1:
        return [
            _generate_with_optional_temperature(
                generator, messages_list[0], temperature=temperatures[0]
            )
        ]

    if hasattr(generator, "generate_from_messages_batch"):
        results: list[str | None] = [None] * len(messages_list)
        by_temperature: dict[float, list[int]] = {}
        for index, temperature in enumerate(temperatures):
            by_temperature.setdefault(float(temperature), []).append(index)
        for temperature, indices in by_temperature.items():
            outputs = generator.generate_from_messages_batch(
                [messages_list[index] for index in indices], temperature=temperature
            )
            for index, output in zip(indices, outputs):
                results[index] = output
        return [result if result is not None else "" for result in results]

    if getattr(generator, "supports_parallel_requests", False):
        import concurrent.futures
        import threading

        thread_generators = threading.local()

        def call(index: int) -> str:
            worker = generator
            if hasattr(generator, "clone_for_parallel_requests"):
                worker = getattr(thread_generators, "generator", None)
                if worker is None:
                    worker = generator.clone_for_parallel_requests()
                    thread_generators.generator = worker
            return _generate_with_optional_temperature(
                worker, messages_list[index], temperature=temperatures[index]
            )

        max_workers = max(1, min(int(LLM_BATCH_PARALLELISM), len(messages_list)))
        results = [None] * len(messages_list)
        failed: list[int] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(call, index): index for index in range(len(messages_list))}
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:                          # noqa: BLE001
                    failed.append(index)
                    print(f"[generate-many] parallel request {index} failed, retrying serially: {exc}",
                          flush=True)
        for index in failed:
            results[index] = _generate_with_optional_temperature(
                generator, messages_list[index], temperature=temperatures[index]
            )
        return [result if result is not None else "" for result in results]

    return [
        _generate_with_optional_temperature(generator, messages, temperature=temperature)
        for messages, temperature in zip(messages_list, temperatures)
    ]


def score_imagined_trajectory_candidate(candidate_steps: list[dict[str, Any]]) -> float:
    score = 0.0
    if not candidate_steps:
        return score
    for step in candidate_steps:
        if step.get("final_answer"):
            score += 3.0
        if step.get("clarify"):
            score -= 1.0
        if step.get("parse_error"):
            score -= 2.0
        if step.get("error") == "empty_tool_calls":
            score -= 2.0
        if step.get("repeated_tool_call_loop"):
            score -= 1.5
        feedbacks = step.get("predicted_feedback") or []
        if feedbacks:
            score += sum(0.5 if fb.get("predicted_success") else -0.5 for fb in feedbacks)
        if step.get("tool_calls"):
            score += 0.2
    return score


def select_imagined_trajectory_with_llm_judge(
    agent_generator: Any,
    task: TaskTrajectory,
    conversation: list[dict[str, Any]],
    candidate_rollouts: list[dict[str, Any]],
) -> dict[str, Any]:
    if not candidate_rollouts:
        return {
            "selected_index": 0,
            "scores": [],
            "comments": "No candidates provided",
            "fallback_used": True,
        }

    messages = [
        {"role": "system", "content": build_imagined_trajectory_selection_judge_prompt()},
        {
            "role": "user",
            "content": (
                f"User task:\n{task.user_messages[-1] if task.user_messages else ''}\n\n"
                f"Conversation so far:\n{render_messages(conversation)}\n\n"
                "Candidate imagined trajectories:\n"
                + json.dumps(candidate_rollouts, ensure_ascii=False, indent=2)
            ),
        },
    ]
    try:
        raw = _generate_with_optional_temperature(agent_generator, messages, temperature=0.0)
        raw = raw.split("</think>\n", 1)[-1].strip()
        parsed = parse_jsonish(strip_code_fence(raw))
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object from imagined trajectory judge.")
        selected_index = int(parsed.get("selected_index", 0))
        if not 0 <= selected_index < len(candidate_rollouts):
            raise ValueError(f"Selected imagined rollout index out of range: {selected_index}")
        return {
            "selected_index": selected_index,
            "scores": parsed.get("scores") or [],
            "comments": str(parsed.get("comments", "")),
            "raw_response": parsed,
            "fallback_used": False,
        }
    except Exception as exc:
        scored = [
            {
                "index": index,
                "score": score_imagined_trajectory_candidate(rollout.get("imagined_steps") or []),
                "reason": "heuristic fallback",
            }
            for index, rollout in enumerate(candidate_rollouts)
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return {
            "selected_index": scored[0]["index"] if scored else 0,
            "scores": scored,
            "comments": f"LLM judge failed, used heuristic fallback: {exc}",
            "fallback_used": True,
        }


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


def resolve_tokenize_num_proc(requested: int | None, row_count: int) -> int | None:
    """Worker count for the tokenization `map`.

    `None`/0 means "auto": leave a couple of cores for the main process and never spawn more
    workers than there is work for. `map(num_proc=1)` is deliberately returned as None so
    `datasets` stays in-process (spawning one worker only adds pickling cost).
    """
    if requested is not None and requested > 0:
        resolved = requested
    else:
        # 8 rather than "all cores": the batched fast tokenizer already threads its Rust
        # encoder across the box in-process, so extra worker processes only add fork,
        # pickling and Arrow shard-merge cost. Measured on real rows (3.8k chars/row,
        # 128 cores): 270 rows/s per-row -> 870 batched/1 proc -> 1000 at 8 procs ->
        # 660 at 30 procs. Raise --tokenize-num-proc only if profiling says so.
        resolved = max(1, min(8, (os.cpu_count() or 2) - 2))
    resolved = max(1, min(resolved, max(1, row_count // 1000)))
    return None if resolved <= 1 else resolved


def build_prompt_completion_dataset(
    dataset_class: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_seq_length: int,
    min_completion_tokens: int,
    disable_chat_template: bool = False,
    num_proc: int | None = None,
    map_batch_size: int = 500,
    prefilter_chars_per_token: float = 12.0,
) -> tuple[Any, dict[str, int]]:
    dataset = dataset_class.from_list(rows)

    # Resolved once instead of per row (see build_chat_template_renderer).
    render = build_chat_template_renderer(tokenizer, disable_chat_template)
    gemma4_prefix_ids: list[int] = []
    if tokenizer_looks_like_gemma4(tokenizer):
        gemma4_prefix_ids = tokenizer.encode(GEMMA4_EMPTY_THOUGHT_PREFIX, add_special_tokens=False)

    # Rows far longer than --max-seq-length are dropped by the filter below anyway, but they
    # dominate wall-clock on the way there: a single 300k-token trajectory costs ~40x an average
    # row to render and encode. Rejecting them on a raw character count (no template, no
    # tokenizer) keeps the expensive path proportional to the data that survives. The bound is
    # deliberately loose -- 12 chars/token is well above any real ratio for this data, so the
    # cheap check can only drop rows the exact filter would have dropped too.
    max_source_chars = (
        int(max_seq_length * prefilter_chars_per_token) if prefilter_chars_per_token > 0 else 0
    )
    # `full_length` of a prefiltered row is a sentinel above max_seq_length so the existing
    # length filter removes it; no separate keep column is needed.
    oversized_sentinel_length = max_seq_length + 1
    oversized_row = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "prompt_length": 0,
        "completion_length": 0,
        "full_length": oversized_sentinel_length,
    }

    def preprocess_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        prompts = batch["prompt"]
        completions = batch["completion"]
        outcome_labels = batch.get("outcome_label") or [UNKNOWN_OUTCOME_LABEL] * len(prompts)

        prompt_texts: list[str] = []
        full_texts: list[str] = []
        encoded_positions: list[int] = []
        results: list[dict[str, Any]] = [dict(oversized_row) for _ in prompts]

        for position, (prompt_messages, completion_messages) in enumerate(zip(prompts, completions)):
            if max_source_chars:
                source_chars = 0
                for message in prompt_messages:
                    source_chars += len(message.get("content") or "")
                for message in completion_messages:
                    source_chars += len(message.get("content") or "")
                if source_chars > max_source_chars:
                    continue
            prompt_messages = [dict(message) for message in prompt_messages]
            completion_messages = [dict(message) for message in completion_messages]
            prompt_texts.append(render(prompt_messages, add_generation_prompt=True))
            full_texts.append(render(prompt_messages + completion_messages, add_generation_prompt=False))
            encoded_positions.append(position)

        if prompt_texts:
            # One batched call per text set: the fast tokenizer encodes a batch in Rust across
            # its own thread pool, which per-row `tokenizer.encode` calls cannot use at all.
            prompt_id_batch = tokenizer(prompt_texts, add_special_tokens=False)["input_ids"]
            full_id_batch = tokenizer(full_texts, add_special_tokens=False)["input_ids"]
        else:
            prompt_id_batch, full_id_batch = [], []

        for position, prompt_ids, full_ids in zip(encoded_positions, prompt_id_batch, full_id_batch):
            # The templates render the prompt as a literal prefix of the full text, so the
            # token lists normally share that prefix exactly; a C-level slice compare settles
            # it without the per-token Python loop. common_prefix_length() still handles the
            # rare boundary-retokenization case.
            prompt_length = len(prompt_ids)
            if full_ids[:prompt_length] != prompt_ids:
                prompt_length = common_prefix_length(prompt_ids, full_ids)
                prompt_ids = prompt_ids[:prompt_length]
            completion_ids = full_ids[prompt_length:]
            if gemma4_prefix_ids:
                prefix_length = len(gemma4_prefix_ids)
                if prompt_ids[-prefix_length:] != gemma4_prefix_ids:
                    if completion_ids[:prefix_length] == gemma4_prefix_ids:
                        completion_ids = completion_ids[prefix_length:]
                    prompt_ids = prompt_ids + gemma4_prefix_ids
            full_ids = prompt_ids + completion_ids
            prompt_length = len(prompt_ids)
            results[position] = {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": [-100] * prompt_length + completion_ids,
                "prompt_length": prompt_length,
                "completion_length": len(completion_ids),
                "full_length": len(full_ids),
            }

        return {
            "input_ids": [row["input_ids"] for row in results],
            "attention_mask": [row["attention_mask"] for row in results],
            "labels": [row["labels"] for row in results],
            "prompt_length": [row["prompt_length"] for row in results],
            "completion_length": [row["completion_length"] for row in results],
            "full_length": [row["full_length"] for row in results],
            "outcome_label": [int(label) for label in outcome_labels],
        }

    resolved_num_proc = resolve_tokenize_num_proc(num_proc, len(dataset))
    dataset = dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=map_batch_size,
        num_proc=resolved_num_proc,
        remove_columns=dataset.column_names,
        desc="tokenize",
    )
    original_size = len(dataset)
    # Reading the two length columns and `select`ing the survivors is a lazy index remap;
    # `filter` would rewrite every token column to a new Arrow table instead.
    full_lengths = dataset["full_length"]
    completion_lengths = dataset["completion_length"]
    keep_indices = [
        index
        for index, (full_length, completion_length) in enumerate(zip(full_lengths, completion_lengths))
        if full_length <= max_seq_length and completion_length >= min_completion_tokens
    ]
    prefiltered = sum(1 for length in full_lengths if length == oversized_sentinel_length)
    del full_lengths, completion_lengths
    filtered_size = len(keep_indices)
    if filtered_size != original_size:
        dataset = dataset.select(keep_indices)
    del keep_indices
    if filtered_size == 0:
        raise ValueError(
            "No valid SFT examples remain after filtering. Increase --max-seq-length "
            "or reduce the prompt size."
        )

    stats = {
        "original_examples": original_size,
        "kept_examples": filtered_size,
        "dropped_examples": original_size - filtered_size,
        "char_prefiltered_examples": prefiltered,
        "tokenize_num_proc": resolved_num_proc or 1,
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
            none_grad_trainable_tensors += 1
            if len(none_grad_name_sample) < 20:
                none_grad_name_sample.append(name)
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


def tokenized_dataset_fingerprint(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    split: str,
    settings: dict[str, Any],
) -> str:
    """Content fingerprint for the tokenized-dataset cache.

    Hashing all 700k rows would cost minutes, so the row content is sampled at up to 64 evenly
    spaced positions and combined with the row count and every setting that changes the
    tokenized output. Pass --rebuild-tokenized-cache if a change slips past the sample.
    """
    digest = hashlib.sha256()
    payload = {
        "split": split,
        "row_count": len(rows),
        "tokenizer": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "chat_template": hashlib.sha256(
            (getattr(tokenizer, "chat_template", "") or "").encode("utf-8")
        ).hexdigest(),
        "settings": settings,
    }
    digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    if rows:
        stride = max(1, len(rows) // 64)
        for index in range(0, len(rows), stride):
            digest.update(
                json.dumps(rows[index], sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            )
    return digest.hexdigest()[:16]


def wait_for_tokenized_cache(
    dataset_path: Path,
    stats_path: Path,
    *,
    split: str,
    timeout_seconds: float = 14400.0,
    poll_seconds: float = 2.0,
) -> None:
    """Block until rank 0's tokenized-dataset artifacts are readable here.

    Tokenizing a large split legitimately takes an hour or more, so the timeout is generous and
    progress is logged; the alternative (failing fast) turns a slow rank 0 into a crashed job.
    """
    deadline = time.time() + timeout_seconds
    announced = False
    while time.time() < deadline:
        if stats_path.is_file() and (dataset_path / "dataset_info.json").is_file():
            return
        if not announced:
            print(f"[tokenize-cache] rank {distributed_rank()} waiting for {split} cache at "
                  f"{dataset_path} (rank 0 is tokenizing)", flush=True)
            announced = True
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"Timed out after {timeout_seconds:.0f}s waiting for the {split} tokenized cache at "
        f"{dataset_path}. Rank 0 either failed or is still tokenizing; check its log, or pass "
        "--no-tokenized-cache to have every rank tokenize independently."
    )


def build_or_load_prompt_completion_dataset(
    dataset_class: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    split: str,
    max_seq_length: int,
    min_completion_tokens: int,
    disable_chat_template: bool,
    num_proc: int | None,
    map_batch_size: int,
    prefilter_chars_per_token: float,
    cache_dir: Path | None,
    rebuild_cache: bool,
    extra_fingerprint: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, int]]:
    """Tokenize once per host and reuse it, instead of once per rank per run.

    Under torchrun every rank used to run the identical `map` over the identical rows: N times
    the CPU work, N copies of the Arrow table in RAM, and all of it thrown away at exit. Here
    rank 0 builds and saves; the other ranks wait on the barrier and memory-map the result, so
    the cost is paid once and the table is shared by the page cache rather than duplicated.
    """
    settings = {
        "max_seq_length": max_seq_length,
        "min_completion_tokens": min_completion_tokens,
        "disable_chat_template": disable_chat_template,
        "prefilter_chars_per_token": prefilter_chars_per_token,
        **(extra_fingerprint or {}),
    }

    def build() -> tuple[Any, dict[str, int]]:
        started = time.time()
        dataset, stats = build_prompt_completion_dataset(
            dataset_class,
            tokenizer,
            rows,
            max_seq_length=max_seq_length,
            min_completion_tokens=min_completion_tokens,
            disable_chat_template=disable_chat_template,
            num_proc=num_proc,
            map_batch_size=map_batch_size,
            prefilter_chars_per_token=prefilter_chars_per_token,
        )
        stats = dict(stats)
        stats["tokenize_seconds"] = round(time.time() - started, 1)
        return dataset, stats

    if cache_dir is None:
        return build()

    fingerprint = tokenized_dataset_fingerprint(rows, tokenizer, split=split, settings=settings)
    dataset_path = Path(cache_dir) / f"{split}-{fingerprint}"
    stats_path = dataset_path.parent / f"{split}-{fingerprint}.stats.json"

    def load() -> tuple[Any, dict[str, int]] | None:
        if rebuild_cache or not (dataset_path / "dataset_info.json").exists() or not stats_path.exists():
            return None
        try:
            dataset = datasets.load_from_disk(str(dataset_path))
            stats = dict(load_json(stats_path))
        except Exception as exc:                                  # noqa: BLE001
            print(f"[tokenize-cache] ignoring unreadable cache {dataset_path}: {exc}", flush=True)
            return None
        stats["loaded_from_cache"] = True
        print(f"[tokenize-cache] {split}: loaded {len(dataset)} rows from {dataset_path}", flush=True)
        return dataset, stats

    if not distributed_is_enabled():
        cached = load()
        if cached is not None:
            return cached
        dataset, stats = build()
        dataset.save_to_disk(str(dataset_path))
        dump_json(stats_path, stats)
        print(f"[tokenize-cache] {split}: saved {len(dataset)} rows to {dataset_path}", flush=True)
        # Re-open memory-mapped so training reads from the page cache instead of a private copy.
        del dataset
        gc.collect()
        return datasets.load_from_disk(str(dataset_path)), stats

    if is_main_process():
        cached = load()
        if cached is None:
            dataset, stats = build()
            dataset.save_to_disk(str(dataset_path))
            dump_json(stats_path, stats)
            print(f"[tokenize-cache] {split}: saved {len(dataset)} rows to {dataset_path}", flush=True)
            del dataset
            gc.collect()
        else:
            stats = cached[1]
            del cached
        distributed_barrier()
    else:
        # Nothing to tokenize here, so drop the rows before the wait: they are a multi-GB
        # duplicate of what rank 0 is about to write.
        rows.clear()
        gc.collect()
        distributed_barrier()
        # The barrier alone is not enough to order this. It is a NO-OP whenever the process
        # group is not initialized (entry points that reach training without calling
        # setup_distributed()), and even with a real barrier a shared filesystem can lag before
        # rank 0's files are visible here. Waiting for the artifacts is what actually orders it.
        wait_for_tokenized_cache(dataset_path, stats_path, split=split)
        stats = dict(load_json(stats_path))
        stats["loaded_from_cache"] = True
    dataset = datasets.load_from_disk(str(dataset_path))
    print(f"[tokenize-cache] rank {distributed_rank()} {split}: {len(dataset)} rows "
          f"memory-mapped from {dataset_path}", flush=True)
    return dataset, stats

def validate_tokenized_dataset(dataset: Any, vocab_size: int, batch_size: int = 1000) -> dict[str, Any]:
    """Range/supervision check over the tokenized dataset.

    Vectorized on purpose: the row-at-a-time version below (`dataset[index]` + a Python list
    comprehension over every label) costs milliseconds per row, which is another full hour on a
    700k-row split. Reading Arrow batches and reducing with pyarrow/numpy keeps it to seconds.
    """
    import numpy as np
    import pyarrow.compute as pc

    max_input_id = -1
    min_input_id = 10**18
    max_label_id = -1
    min_label_id = 10**18

    row_offset = 0
    for batch in dataset.with_format("arrow").iter(batch_size=batch_size):
        input_column = batch.column("input_ids")
        label_column = batch.column("labels")

        flat_inputs = pc.list_flatten(input_column)
        if len(flat_inputs):
            batch_min_input = pc.min(flat_inputs).as_py()
            batch_max_input = pc.max(flat_inputs).as_py()
            min_input_id = min(min_input_id, batch_min_input)
            max_input_id = max(max_input_id, batch_max_input)
            if batch_min_input < 0 or batch_max_input >= vocab_size:
                index, row_min, row_max = _first_out_of_range_row(input_column, vocab_size)
                raise ValueError(
                    f"Out-of-range input id at dataset row {row_offset + index}: "
                    f"min={row_min}, max={row_max}, vocab_size={vocab_size}"
                )

        label_lengths = np.asarray(
            pc.list_value_length(label_column).to_numpy(zero_copy_only=False), dtype=np.int64
        )
        flat_labels = pc.list_flatten(label_column)
        supervised_mask = pc.not_equal(flat_labels, -100)
        supervised_np = np.asarray(supervised_mask.to_numpy(zero_copy_only=False), dtype=np.int64)
        # Per-row supervised-token counts from a prefix sum; handles zero-length rows, which
        # np.add.reduceat would silently get wrong.
        prefix_counts = np.concatenate(([0], np.cumsum(supervised_np)))
        row_ends = np.cumsum(label_lengths)
        row_starts = row_ends - label_lengths
        supervised_counts = prefix_counts[row_ends] - prefix_counts[row_starts]
        unsupervised_rows = np.nonzero(supervised_counts == 0)[0]
        if len(unsupervised_rows):
            raise ValueError(
                f"Dataset row {row_offset + int(unsupervised_rows[0])} has no supervised completion tokens."
            )

        supervised_labels = pc.filter(flat_labels, supervised_mask)
        if len(supervised_labels):
            batch_min_label = pc.min(supervised_labels).as_py()
            batch_max_label = pc.max(supervised_labels).as_py()
            min_label_id = min(min_label_id, batch_min_label)
            max_label_id = max(max_label_id, batch_max_label)
            if batch_min_label < 0 or batch_max_label >= vocab_size:
                index, row_min, row_max = _first_out_of_range_row(
                    label_column, vocab_size, ignore_value=-100
                )
                raise ValueError(
                    f"Out-of-range label id at dataset row {row_offset + index}: "
                    f"min={row_min}, max={row_max}, vocab_size={vocab_size}"
                )

        row_offset += batch.num_rows

    return {
        "min_input_id": min_input_id,
        "max_input_id": max_input_id,
        "min_label_id": min_label_id,
        "max_label_id": max_label_id,
    }


def _first_out_of_range_row(
    column: Any,
    vocab_size: int,
    ignore_value: int | None = None,
) -> tuple[int, int, int]:
    """Locate the first offending row inside a batch (error path only)."""
    for index, row in enumerate(column.to_pylist()):
        values = [value for value in row if ignore_value is None or value != ignore_value]
        if not values:
            continue
        row_min, row_max = min(values), max(values)
        if row_min < 0 or row_max >= vocab_size:
            return index, row_min, row_max
    return 0, -1, -1


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


class HFTextGenerator:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int,
        trust_remote_code: bool = False,
        dtype: str = "auto",
        disable_chat_template: bool = False,
        attn_implementation: str = "sdpa",
        device_map: str | dict | None = None,
        draft_model_path: str | None = None,
        prompt_lookup_num_tokens: int = 0,
    ) -> None:
        torch, _, AutoModelForCausalLM, AutoTokenizer, _, _, _ = require_training_stack()
        self.torch = torch
        self.disable_chat_template = disable_chat_template
        self.tokenizer = load_tokenizer_with_repair(
            model_path, trust_remote_code=trust_remote_code, auto_tokenizer_class=AutoTokenizer
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        model_cls = resolve_text_generation_model_class(
            model_path,
            trust_remote_code=trust_remote_code,
            causal_lm_class=AutoModelForCausalLM,
        )
        adapter_config_path = Path(model_path) / "adapter_config.json"
        if adapter_config_path.is_file():
            with adapter_config_path.open("r", encoding="utf-8") as handle:
                adapter_config = json.load(handle)
            base_model_path = adapter_config.get("base_model_name_or_path")
            if not base_model_path:
                raise ValueError(f"Missing base_model_name_or_path in {adapter_config_path}")
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "Loading a LoRA adapter for evaluation requires `peft`."
                ) from exc
            ensure_peft_weight_converter_compatibility()
            base_model = model_cls.from_pretrained(
                base_model_path,
                dtype=resolve_torch_dtype(torch, dtype),
                trust_remote_code=trust_remote_code,
                device_map=resolve_inference_device_map(torch, override=device_map),
                attn_implementation=attn_implementation,
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = model_cls.from_pretrained(
                model_path,
                dtype=resolve_torch_dtype(torch, dtype),
                trust_remote_code=trust_remote_code,
                device_map=resolve_inference_device_map(torch, override=device_map),
                attn_implementation=attn_implementation,
            )
        self.input_device = next(self.model.parameters()).device
        self.max_new_tokens = max_new_tokens
        model_config = getattr(self.model, "config", None)
        base_model_config = getattr(getattr(self.model, "base_model", None), "config", None)
        self.disable_generation_cache = "nemotron_h" in {
            getattr(model_config, "model_type", None),
            getattr(base_model_config, "model_type", None),
        }
        if self.disable_generation_cache:
            # Nemotron-H remote code can receive `past_key_values` with
            # `cache_position=None` under Transformers generation, which crashes
            # in `prepare_inputs_for_generation`. Disabling cache is slower but
            # keeps evaluation/replay generation correct.
            self.model.config.use_cache = False
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.use_cache = False

        # Speculative decoding (batch-1 calls only -- Transformers restricts both assisted
        # generation and prompt-lookup to a single sequence; batched calls silently skip it).
        # A draft model takes precedence over prompt lookup; both need the KV cache.
        self.assistant_model = None
        self.prompt_lookup_num_tokens = 0
        if not self.disable_generation_cache:
            if draft_model_path:
                self.assistant_model = AutoModelForCausalLM.from_pretrained(
                    draft_model_path,
                    dtype=resolve_torch_dtype(torch, dtype),
                    trust_remote_code=trust_remote_code,
                    attn_implementation=attn_implementation,
                ).to(self.input_device)
                self.assistant_model.eval()
            elif prompt_lookup_num_tokens and int(prompt_lookup_num_tokens) > 0:
                self.prompt_lookup_num_tokens = int(prompt_lookup_num_tokens)
        elif draft_model_path or prompt_lookup_num_tokens:
            print(
                "[hf-generator] speculative decoding disabled: this model runs without a "
                "generation cache (nemotron_h workaround).",
                flush=True,
            )

    def _generation_kwargs(self, temperature: float, batch_size: int) -> dict[str, Any]:
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.disable_generation_cache:
            generation_kwargs["use_cache"] = False
        if temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.95})
        else:
            generation_kwargs.update({"do_sample": False})
        if batch_size == 1:
            if self.assistant_model is not None:
                generation_kwargs["assistant_model"] = self.assistant_model
            elif self.prompt_lookup_num_tokens > 0:
                generation_kwargs["prompt_lookup_num_tokens"] = self.prompt_lookup_num_tokens
        return generation_kwargs

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}
        generation_kwargs = self._generation_kwargs(temperature, batch_size=1)
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return strip_model_thinking_output(
            raw_text,
            special_tokens=getattr(self.tokenizer, "all_special_tokens", None),
        )

    def generate_batch(self, prompts: list[str], temperature: float = 0.0) -> list[str]:
        """One left-padded batched `generate` for several prompts sharing one temperature.

        With left padding every row's continuation starts at the same column (the padded
        input width), so per-row slicing is uniform. Batch size 1 delegates to `generate`
        so speculative decoding still applies there.
        """
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0], temperature=temperature)]
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        finally:
            self.tokenizer.padding_side = original_padding_side
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}
        generation_kwargs = self._generation_kwargs(temperature, batch_size=len(prompts))
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        input_width = inputs["input_ids"].shape[1]
        special_tokens = getattr(self.tokenizer, "all_special_tokens", None)
        return [
            strip_model_thinking_output(
                self.tokenizer.decode(row[input_width:], skip_special_tokens=False),
                special_tokens=special_tokens,
            )
            for row in output_ids
        ]

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        return apply_chat_template_or_fallback(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
            disable_chat_template=self.disable_chat_template,
        )

    def generate_from_messages(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        return self.generate(self._render_prompt(messages), temperature=temperature)

    def generate_from_messages_batch(
        self, messages_list: list[list[dict[str, str]]], temperature: float = 0.0
    ) -> list[str]:
        return self.generate_batch(
            [self._render_prompt(messages) for messages in messages_list],
            temperature=temperature,
        )

    def generate_samples(
        self, messages: list[dict[str, str]], temperature: float = 0.0, num_samples: int = 1
    ) -> list[str]:
        """`num_samples` independent continuations of ONE prompt in a single forward batch.

        The prompt is encoded once and `num_return_sequences` decodes k sequences together, so
        k diverse candidates cost roughly one prompt's prefill plus the decode of the longest
        candidate -- not k times a full generation. Greedy decoding has only one continuation,
        so temperature<=0 returns the single greedy answer replicated k times rather than
        silently switching to sampling.
        """
        k = max(1, int(num_samples))
        if k == 1 or temperature <= 0:
            return [self.generate_from_messages(messages, temperature=temperature)] * k
        prompt = self._render_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}
        generation_kwargs = self._generation_kwargs(temperature, batch_size=k)
        generation_kwargs["num_return_sequences"] = k
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        input_width = inputs["input_ids"].shape[1]
        special_tokens = getattr(self.tokenizer, "all_special_tokens", None)
        return [
            strip_model_thinking_output(
                self.tokenizer.decode(row[input_width:], skip_special_tokens=False),
                special_tokens=special_tokens,
            )
            for row in output_ids
        ]


def looks_like_jepa_world_model_path(model_path: str | Path | None) -> bool:
    if not model_path:
        return False
    path = Path(model_path)
    return (path / "text_leworldmodel.pt").is_file() and (path / "backbone").is_dir()


class JepaTextWorldModelGenerator:
    """Inference adapter for checkpoints produced by `src/finetuning_jepa.py`.

    The JEPA checkpoint predicts the next raw observation/tool output from
    structured task context, recent action/observation history, and candidate
    tool calls. It exposes the same feedback shape used by the replay code.
    """

    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int,
        trust_remote_code: bool = False,
        dtype: str = "auto",
        max_input_length: int = 2048,
        max_action_length: int = 512,
        imagined_observation_backend: str = "auto",
        arch_defaults: dict[str, Any] | None = None,
    ) -> None:
        from transformers.modeling_outputs import BaseModelOutput

        from src.finetuning_jepa import (
            TextLeWorldModel,
            build_canonical_event_observation_payload,
            decode_canonical_event_logits,
            load_canonical_event_vocab,
            load_jepa_backbone,
            load_text_tokenizer,
            reconstruct_state_from_canonical_event_labels,
            render_raw_replay_history,
        )

        self.model_path = Path(model_path)
        self.max_new_tokens = max_new_tokens
        self.BaseModelOutput = BaseModelOutput

        manifest_path = self.model_path / "jepa_data_manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        # A canonical-event head checkpoint may not carry jepa_data_manifest.json
        # (head training does not always write one). Fall back per-field to the
        # architecture defaults the caller passed (e.g. replay CLI flags); a value
        # present in the manifest always takes precedence.
        resolved: dict[str, Any] = {**(arch_defaults or {}), **manifest}
        if bool(resolved.get("recognition_probe_bypass_projector", False)):
            # Diagnostic checkpoint: its canonical trunk/heads were trained on RAW pooled
            # backbone features (projector bypassed). When hidden_size == latent_dim the trunk
            # shape coincides with a normal one, so it would load cleanly here and then read
            # projected latents at replay time -- silently wrong predictions. Refuse loudly.
            raise SystemExit(
                f"{self.model_path} was trained with --recognition-probe-bypass-projector "
                "(a recognition-probe diagnostic). Its heads read raw pooled features, not "
                "projected latents, and cannot be used for replay."
            )
        canonical_manifest_path = self.model_path / "canonical_event_data_manifest.json"
        canonical_manifest: dict[str, Any] = {}
        if canonical_manifest_path.is_file():
            with canonical_manifest_path.open("r", encoding="utf-8") as handle:
                canonical_manifest = json.load(handle)
        self.backbone_type = str(resolved.get("backbone_type") or "seq2seq")
        self.max_input_length = int(resolved.get("max_input_length") or max_input_length)
        self.max_action_length = int(resolved.get("max_action_length") or max_action_length)
        self.max_observation_length = int(resolved.get("max_observation_length") or 512)
        self.max_goal_length = int(resolved.get("max_goal_length") or self.max_observation_length)
        self.render_raw_replay_history = render_raw_replay_history
        self._reconstruct_state_from_canonical_event_labels = reconstruct_state_from_canonical_event_labels
        self._decode_canonical_event_logits = decode_canonical_event_logits
        self._build_canonical_event_observation_payload = build_canonical_event_observation_payload

        self.tokenizer = load_text_tokenizer(
            self.model_path,
            trust_remote_code=trust_remote_code,
        )
        # Checkpoints trained with --truncate-states-keep-newest tokenized their state texts
        # with LEFT truncation (over-length states keep the newest history, not the oldest).
        # Replay must match, or long states get cut on the opposite side from what the encoder
        # saw in training. Set globally on the tokenizer: it only affects texts that exceed
        # max_length, which at replay are exactly the state texts (contexts/actions are short).
        if bool(resolved.get("truncate_states_keep_newest", False)):
            self.tokenizer.truncation_side = "left"
        # {tool_name: idx} written by build_tool_vocabulary at training time -- the ONLY way
        # to map a family/tool-name string (e.g. from hierarchical_action_sampling._family_key)
        # back to the same integer id the tool_embeddings / action_decoder_tool_embeddings rows
        # were trained against. tool_vocab_size alone (in the manifest) sizes the embedding
        # table but says nothing about which row is which tool. Absent for checkpoints that
        # predate action-head training; callers must treat a missing/empty dict as "no
        # tool-conditioning available" and fall back to index 0 (unknown tool).
        tool_vocab_path = self.model_path / "tool_vocab.json"
        self.tool_vocab: dict[str, int] = {}
        if tool_vocab_path.is_file():
            with tool_vocab_path.open("r", encoding="utf-8") as handle:
                self.tool_vocab = json.load(handle)

        backbone = load_jepa_backbone(
            self.model_path / "backbone",
            backbone_type=self.backbone_type,
            trust_remote_code=trust_remote_code,
            dtype=resolve_torch_dtype(torch, dtype),
        )
        # Rebuild the canonical_event_state/nudge classification heads if the
        # checkpoint was produced by --train-canonical-event-heads-only, so their
        # weights load (and are not flagged as unexpected keys) and the imagined
        # observation can be reconstructed from the head predictions.
        self.canonical_event_vocab = load_canonical_event_vocab(self.model_path)
        canonical_event_vocab_sizes = (
            {field: len(values) for field, values in self.canonical_event_vocab.items()}
            if self.canonical_event_vocab
            else None
        )
        canonical_event_head_hidden_size = int(
            resolved.get("canonical_event_head_hidden_size")
            or canonical_manifest.get("canonical_event_head_hidden_size")
            or 512
        )
        # Optional-module architecture flags (P1 tool_select, P2 action_encoder, Option-C
        # obs_grounding, Fast-LeWM, the action decoder) are recorded in jepa_data_manifest.json
        # at training time (see finetuning_jepa.py's data_manifest / canonical-event-head
        # manifest writers) specifically so replay-time loading can construct a model whose
        # module set matches the checkpoint's state dict BEFORE loading it -- otherwise any of
        # these modules' weights show up as "unexpected keys" below and loading raises.
        self.model = TextLeWorldModel(
            backbone=backbone,
            latent_dim=int(resolved.get("latent_dim") or 0),
            memory_tokens=int(resolved.get("memory_tokens") or 8),
            dropout=0.0,
            predictor_hidden_multiplier=float(resolved.get("predictor_hidden_multiplier") or 4.0),
            goal_conditioning=bool(resolved.get("goal_conditioning", False)),
            latent_type=str(resolved.get("latent_type") or "continuous"),
            latent_categoricals=int(resolved.get("latent_categoricals") or 32),
            latent_classes=int(resolved.get("latent_classes") or 32),
            latent_unimix=float(resolved.get("latent_unimix") or 0.01),
            latent_delta_prediction=bool(resolved.get("latent_delta_prediction", False)),
            # Sets the canonical-event trunk's input width; must match what the checkpoint was
            # trained with or the trunk weights fail to load.
            canonical_event_head_inputs=str(resolved.get("canonical_event_head_inputs") or "all"),
            state_updater=bool(resolved.get("state_updater", False)),
            state_updater_objective=str(resolved.get("state_updater_objective") or "future_event"),
            recurrent_state_init=bool(resolved.get("recurrent_state_init", False)),
            pooling=str(resolved.get("pooling") or "mean"),
            canonical_event_vocab_sizes=canonical_event_vocab_sizes,
            canonical_event_head_hidden_size=canonical_event_head_hidden_size,
            obs_grounding=bool(resolved.get("obs_grounding", False)),
            obs_ground_decoder_dim=int(resolved.get("obs_ground_decoder_dim") or 256),
            obs_ground_decoder_layers=int(resolved.get("obs_ground_decoder_layers") or 4),
            obs_ground_decoder_heads=int(resolved.get("obs_ground_decoder_heads") or 4),
            obs_ground_decoder_memory_tokens=int(resolved.get("obs_ground_decoder_memory_tokens") or 8),
            obs_ground_decoder_max_length=int(resolved.get("obs_ground_decoder_max_length") or 128),
            tool_vocab_size=int(resolved.get("tool_vocab_size") or 0),
            tool_select=bool(resolved.get("tool_select", False)),
            action_encoder=bool(resolved.get("action_encoder", False)),
            action_head_embed_dim=int(resolved.get("action_head_embed_dim") or 256),
            action_decoder=bool(resolved.get("action_decoder", False)),
            action_decoder_max_noise_std=float(resolved.get("action_decoder_max_noise_std") or 0.1),
            action_decoder_dim=int(resolved.get("action_decoder_dim") or 256),
            action_decoder_layers=int(resolved.get("action_decoder_layers") or 4),
            action_decoder_heads=int(resolved.get("action_decoder_heads") or 4),
            action_decoder_memory_tokens=int(resolved.get("action_decoder_memory_tokens") or 8),
            action_decoder_max_length=int(resolved.get("action_decoder_max_length") or 512),
            action_decoder_tool_vocab_size=int(resolved.get("action_decoder_tool_vocab_size") or 0),
            fast_lewm=bool(resolved.get("fast_lewm", False)),
            fast_lewm_dim=int(resolved.get("fast_lewm_dim") or 256),
            fast_lewm_layers=int(resolved.get("fast_lewm_layers") or 3),
            fast_lewm_heads=int(resolved.get("fast_lewm_heads") or 4),
            fast_lewm_max_horizon=int(resolved.get("fast_lewm_max_horizon") or 8),
            terminal_head=bool(resolved.get("terminal_head", False)),
            value_head=bool(resolved.get("value_head", False)),
            # Which predictor module the checkpoint actually contains. `mlp` and `transformer`
            # share no parameter names, so getting this wrong drops the whole predictor as
            # unexpected keys and replays a randomly-initialized one.
            predictor_arch=str(resolved.get("predictor_arch") or "mlp"),
            predictor_transformer_dim=int(resolved.get("predictor_transformer_dim") or 0),
            predictor_transformer_layers=int(resolved.get("predictor_transformer_layers") or 6),
            predictor_transformer_heads=int(resolved.get("predictor_transformer_heads") or 16),
            predictor_transformer_mlp_ratio=float(resolved.get("predictor_transformer_mlp_ratio") or 4.0),
            predictor_history_length=int(resolved.get("predictor_history_length") or 0),
        )
        state_dict_path = self.model_path / "text_leworldmodel.pt"
        try:
            state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(state_dict_path, map_location="cpu")
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        missing_keys = set(getattr(incompatible, "missing_keys", []))
        unexpected_keys = set(getattr(incompatible, "unexpected_keys", []))
        # Optional-module weights may be present in the checkpoint but absent from this
        # replay model (or vice versa) if the manifest predates these flags -- tolerate them
        # in both directions, same prefixes as finetuning_jepa.load_jepa_state_dict_for_training.
        _optional_prefixes = ("obs_ground_", "tool_embeddings", "tool_query", "action_encoder_mlp", "fast_", "action_decoder", "terminal_head", "value_head")
        allowed_missing = {
            key
            for key in missing_keys
            if key.startswith(
                ("backbone.", "success_head.", "canonical_event_heads.", "canonical_event_trunk.")
            )
            or key.startswith(_optional_prefixes)
        }
        disallowed_missing = missing_keys - allowed_missing
        disallowed_unexpected = {key for key in unexpected_keys if not key.startswith(_optional_prefixes)}
        if disallowed_missing or disallowed_unexpected:
            raise RuntimeError(
                "JEPA checkpoint does not match TextLeWorldModel architecture: "
                f"missing={sorted(disallowed_missing)}, unexpected={sorted(disallowed_unexpected)}"
            )
        success_head_weights_present = not any(key.startswith("success_head.") for key in missing_keys)
        loss_coefficients = manifest.get("loss_coefficients") if isinstance(manifest.get("loss_coefficients"), dict) else {}
        try:
            manifest_success_coeff = float(
                manifest.get("success_loss_coeff", loss_coefficients.get("success", 0.0)) or 0.0
            )
        except (TypeError, ValueError):
            manifest_success_coeff = 0.0
        manifest_success_head_trained = bool(manifest.get("train_success_head_only")) or manifest_success_coeff > 0
        self.success_head_available = success_head_weights_present and manifest_success_head_trained

        # Canonical-event heads count as available only when their weights loaded
        # (present vocab + present checkpoint weights) -- never fall back to
        # randomly initialized heads.
        canonical_event_weights_present = canonical_event_vocab_sizes is not None and not any(
            key.startswith("canonical_event_heads.") for key in missing_keys
        )
        self.canonical_event_available = bool(canonical_event_weights_present)
        backend = str(imagined_observation_backend or "auto")
        if backend == "canonical_event" and not self.canonical_event_available:
            raise RuntimeError(
                "--imagined-observation-backend=canonical_event requires a checkpoint with trained "
                "canonical_event heads (canonical_event_vocab.json + canonical_event_heads.* weights); "
                f"none found in {self.model_path}. Train them with --train-canonical-event-heads-only."
            )
        self.canonical_event_state_enabled = self.canonical_event_available and backend in ("auto", "canonical_event")
        self.imagined_observation_backend_resolved = (
            "canonical_event"
            if self.canonical_event_state_enabled
            else ("success" if (backend in ("auto", "success") and self.success_head_available) else "decoder")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _safe_json(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    def _render_history(self, history: list[dict[str, Any]]) -> str:
        return self.render_raw_replay_history(
            history,
            max_chars=self.max_observation_length * 8,
        )

    def _context_text(self, system_prompt: str, user_prompt: str) -> str:
        return f"System prompt:\n{system_prompt}\n\nUser task:\n{user_prompt}"

    def _current_state_text(self, system_prompt: str, user_prompt: str, input_history: list[dict[str, Any]]) -> str:
        context = self._context_text(system_prompt, user_prompt)
        return context + "\n\nHistory before current action:\n" + self._render_history(input_history)

    def _goal_text(self, system_prompt: str, user_prompt: str) -> str:
        goal_text = "Task goal:\n" + self._context_text(system_prompt, user_prompt)
        return goal_text[: self.max_goal_length * 8]

    def _encode(self, text: str, max_length: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _generate_observation(
        self,
        system_prompt: str,
        user_prompt: str,
        action: Any,
        input_history: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> str:
        if self.backbone_type == "encoder":
            raise RuntimeError(
                "Encoder-only JEPA checkpoints are latent-only and cannot decode tool-output predictions; use latent_guided replay."
            )
        context_text = self._context_text(system_prompt, user_prompt)
        current_text = self._current_state_text(system_prompt, user_prompt, input_history)
        action_text = action if isinstance(action, str) else self._safe_json(action)

        current = self._encode(current_text, self.max_input_length)
        context = self._encode(context_text, self.max_input_length)
        action_tokens = self._encode(action_text, self.max_action_length)

        with torch.no_grad():
            z_current = self.model.encode_latent(current["input_ids"], current["attention_mask"])
            z_context = self.model.encode_latent(context["input_ids"], context["attention_mask"])
            z_action = self.model.encode_latent(action_tokens["input_ids"], action_tokens["attention_mask"])
            z_goal = None
            if getattr(self.model, "goal_conditioning", False):
                goal = self._encode(self._goal_text(system_prompt, user_prompt), self.max_goal_length)
                z_goal = self.model.encode_latent(goal["input_ids"], goal["attention_mask"])
            z_pred, _ = self.model.predict_latent(z_current, z_action, z_context, z_goal)
            memory = self.model.memory_projection(z_pred).view(
                -1,
                self.model.memory_tokens,
                self.model.hidden_size,
            )
            backbone_dtype = next(self.model.backbone.parameters()).dtype
            memory = memory.to(dtype=backbone_dtype)
            memory_mask = torch.ones(
                memory.shape[:2],
                dtype=current["attention_mask"].dtype,
                device=memory.device,
            )
            generation_kwargs = {
                "encoder_outputs": self.BaseModelOutput(last_hidden_state=memory),
                "attention_mask": memory_mask,
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.95})
            else:
                generation_kwargs.update({"do_sample": False})
            output_ids = self.model.backbone.generate(**generation_kwargs)
        return strip_model_thinking_output(
            self.tokenizer.decode(output_ids[0], skip_special_tokens=True),
            special_tokens=getattr(self.tokenizer, "all_special_tokens", None),
        ).strip()

    def _parse_prompt_messages(self, messages: list[dict[str, str]]) -> tuple[str, str, str, list[dict[str, Any]]]:
        content = "\n\n".join(str(message.get("content", "")) for message in messages)

        def section(name: str, following: tuple[str, ...]) -> str:
            pattern = re.escape(name) + r":\n"
            match = re.search(pattern, content)
            if not match:
                return ""
            start = match.end()
            end = len(content)
            for next_name in following:
                next_match = re.search(r"\n\n" + re.escape(next_name) + r":\n", content[start:])
                if next_match:
                    end = min(end, start + next_match.start())
            return content[start:end].strip()

        system_prompt = section("System prompt", ("User prompt", "Recent state history", "Action"))
        user_prompt = section("User prompt", ("Recent state history", "Current state", "Action"))
        action_text = section("Action", ("Predict the tool output", "Predict the resulting state", "Predict whether"))
        history_text = section(
            "Recent action/observation history (oldest to newest; input only, not part of the target)",
            ("Action",),
        )
        input_history: list[dict[str, Any]] = []
        if history_text:
            try:
                parsed_history = json.loads(history_text)
                if isinstance(parsed_history, list):
                    input_history = [item for item in parsed_history if isinstance(item, dict)]
            except json.JSONDecodeError:
                input_history = []
        return system_prompt, user_prompt, action_text, input_history

    def generate_from_messages(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        system_prompt, user_prompt, action_text, input_history = self._parse_prompt_messages(messages)
        return self._generate_observation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            action=action_text,
            input_history=input_history,
            temperature=temperature,
        )

    def _encode_latent_text(self, text: str, max_length: int) -> torch.Tensor:
        """Encode one text to a latent, memoized by (max_length, text).

        The context and goal texts are CONSTANT for a whole task and the state text recurs
        between the scorers called at the same step, yet each encode is a full backbone forward
        over up to `max_input_length` tokens -- measured at ~53 ms for a 2.5k-token state on an
        L40S, i.e. the dominant cost of a plan-scoring call. Cached latents are [1, D] tensors,
        so the bounded cache below costs kilobytes.
        """
        cache = getattr(self, "_latent_text_cache", None)
        if cache is None:
            cache = self._latent_text_cache = collections.OrderedDict()
        key = (int(max_length), text)
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            return hit
        encoded = self._encode(text, max_length)
        latent = self.model.encode_latent(encoded["input_ids"], encoded["attention_mask"])
        cache[key] = latent
        while len(cache) > LATENT_TEXT_CACHE_SIZE:
            cache.popitem(last=False)
        return latent

    def _encode_batch(self, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def encode_latents_batched(self, texts: list[str], max_length: int) -> torch.Tensor:
        """Encode many texts to latents in a single backbone forward.

        Padding is masked out by the model's mean-pool, so a batched encode is
        numerically equivalent to encoding each text on its own. Repeated texts
        (identical tool calls recur across plans and rollout steps) are cached so
        they are only encoded once.
        """
        cache = getattr(self, "_action_latent_cache", None)
        if cache is None:
            cache = self._action_latent_cache = {}
        order: dict[str, int] = {}
        misses: list[str] = []
        for text in texts:
            if text not in cache and text not in order:
                order[text] = len(misses)
                misses.append(text)
        if misses:
            encoded = self._encode_batch(misses, max_length)
            latents = self.model.encode_latent(encoded["input_ids"], encoded["attention_mask"])
            for text, idx in order.items():
                cache[text] = latents[idx : idx + 1]
        return torch.cat([cache[text] for text in texts], dim=0)

    def score_action_plans(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        input_history: list[dict[str, Any]],
        action_plans: list[list[Any]],
        goal_text_override: str | None = None,
    ) -> list[dict[str, Any]]:
        if not action_plans:
            return []
        context_text = self._context_text(system_prompt, user_prompt)
        current_text = self._current_state_text(system_prompt, user_prompt, input_history)
        goal_text = goal_text_override or self._goal_text(system_prompt, user_prompt)
        goal_conditioning = getattr(self.model, "goal_conditioning", False)
        num_plans = len(action_plans)
        with torch.no_grad():
            z_context = self._encode_latent_text(context_text, self.max_input_length)
            z_goal = self._encode_latent_text(goal_text, self.max_goal_length)
            z_start = self._encode_latent_text(current_text, self.max_input_length)
            # Roll every plan forward in lockstep so each step encodes all active
            # candidate actions in one batched forward; plans shorter than the
            # current step stay frozen at their last latent.
            z_rollout = z_start.expand(num_plans, -1).contiguous()
            max_len = max(len(plan) for plan in action_plans)
            for step in range(max_len):
                active = [i for i, plan in enumerate(action_plans) if step < len(plan)]
                if not active:
                    break
                action_texts = []
                for i in active:
                    action = action_plans[i][step]
                    action_texts.append(action if isinstance(action, str) else self._safe_json(action))
                z_actions = self.encode_latents_batched(action_texts, self.max_action_length)
                index = torch.as_tensor(active, device=self.device)
                z_sub = z_rollout.index_select(0, index)
                z_ctx = z_context.expand(len(active), -1)
                z_goal_arg = z_goal.expand(len(active), -1) if goal_conditioning else None
                z_new, _ = self.model.predict_latent(z_sub, z_actions, z_ctx, z_goal_arg)
                z_rollout = z_rollout.index_copy(0, index, z_new.to(z_rollout.dtype))
            distances = (z_rollout.float() - z_goal.float()).pow(2).mean(dim=-1)
        scored = [
            {
                "plan_index": plan_index,
                "score": float(distances[plan_index].item()),
                "terminal_goal_mse": float(distances[plan_index].item()),
                "plan": action_plans[plan_index],
            }
            for plan_index in range(num_plans)
        ]
        scored.sort(key=lambda item: item["score"])
        return scored

    def score_action_plans_canonical_event(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        input_history: list[dict[str, Any]],
        action_plans: list[list[Any]],
        goal_text_override: str | None = None,
        score_config: Any = None,
    ) -> list[dict[str, Any]]:
        """Score candidate action plans for beam_plan mode using the canonical-event
        classification heads (higher score = better; opposite of score_action_plans' MSE).

        Rolls every plan forward in lockstep exactly like score_action_plans, but at each step
        also reads the canonical-event heads and scores the per-step field distributions with
        src.canonical_event_scoring. Returns records sorted best-first, each with plan/plan_index,
        score, vetoed, normalized_score, and a reason string.
        """
        from src.canonical_event_scoring import (
            CanonicalEventScoreConfig,
            MISSING_INFO_FIELD,
            logits_to_field_probs_batched,
            rank_trajectories,
        )

        if not action_plans:
            return []
        if not getattr(self, "canonical_event_available", False):
            raise RuntimeError(
                "This JEPA checkpoint has no trained canonical_event heads; beam_plan canonical "
                "scoring requires --train-canonical-event-heads-only weights in the checkpoint."
            )
        config = score_config or CanonicalEventScoreConfig()
        temperature = float(getattr(config, "temperature", 1.0))
        context_text = self._context_text(system_prompt, user_prompt)
        current_text = self._current_state_text(system_prompt, user_prompt, input_history)
        goal_conditioning = getattr(self.model, "goal_conditioning", False)
        num_plans = len(action_plans)
        trajectories: list[list[dict[str, dict[str, float]]]] = [[] for _ in range(num_plans)]
        terminal_prob_steps: list[list[float]] = [[] for _ in range(num_plans)]
        terminal_enabled = bool(getattr(self.model, "terminal_head_enabled", False))
        with torch.inference_mode():
            z_context = self._encode_latent_text(context_text, self.max_input_length)
            # A checkpoint trained with --disable-goal-conditioning never reads z_goal, so
            # encoding the goal text there is a wasted backbone forward per scoring call.
            z_goal = None
            if goal_conditioning:
                goal_text = goal_text_override or self._goal_text(system_prompt, user_prompt)
                z_goal = self._encode_latent_text(goal_text, self.max_goal_length)
            z_start = self._encode_latent_text(current_text, self.max_input_length)
            z_rollout = z_start.expand(num_plans, -1).contiguous()
            max_len = max(len(plan) for plan in action_plans)
            for step in range(max_len):
                active = [i for i, plan in enumerate(action_plans) if step < len(plan)]
                if not active:
                    break
                action_texts = [
                    (action if isinstance(action, str) else self._safe_json(action))
                    for action in (action_plans[i][step] for i in active)
                ]
                z_actions = self.encode_latents_batched(action_texts, self.max_action_length)
                index = torch.as_tensor(active, device=self.device)
                z_sub = z_rollout.index_select(0, index)
                z_ctx = z_context.expand(len(active), -1)
                z_goal_arg = z_goal.expand(len(active), -1) if goal_conditioning else None
                z_new, _, z_state = self.model.predict_latent_with_state(z_sub, z_actions, z_ctx, z_goal_arg)
                logits = self.model.predict_canonical_event_logits(z_sub, z_actions, z_ctx, z_new, z_state)
                probs_rows = logits_to_field_probs_batched(
                    logits, self.canonical_event_vocab, temperature=temperature
                )
                terminal_probs = None
                if terminal_enabled:
                    terminal_logits = self.model.predict_terminal_logit(z_sub, z_actions, z_ctx, z_new)
                    terminal_probs = torch.sigmoid(terminal_logits.detach().float()).cpu().tolist()
                for local_index, plan_index in enumerate(active):
                    trajectories[plan_index].append(probs_rows[local_index])
                    if terminal_probs is not None:
                        terminal_prob_steps[plan_index].append(float(terminal_probs[local_index]))
                z_rollout = z_rollout.index_copy(0, index, z_new.to(z_rollout.dtype))
        _, all_scored = rank_trajectories(trajectories, config, top_k=num_plans)
        for record in all_scored:
            record["plan"] = action_plans[record["index"]]
            record["plan_index"] = record["index"]
            # Decode each step's field distributions into a predicted canonical-event state
            # (argmax per single-label field; >=0.5 for the multi-label field) so callers can
            # show an (action -> predicted state) trajectory. `predicted_state` stays the
            # TERMINAL step for backward compatibility; `per_step_predicted_state` carries one
            # entry per plan step, which multi-step plan consumers (open-loop beam planning)
            # need to annotate every action they inject.
            steps = trajectories[record["index"]]

            def _decode(probs_by_field: dict[str, dict[str, float]]) -> dict[str, Any]:
                decoded: dict[str, Any] = {}
                for field, probs in probs_by_field.items():
                    if not probs:
                        continue
                    if field == MISSING_INFO_FIELD:
                        decoded[field] = [c for c, p in probs.items() if p >= 0.5] or ["none"]
                    else:
                        decoded[field] = max(probs.items(), key=lambda kv: kv[1])[0]
                return decoded

            per_step_states = [_decode(step_probs) for step_probs in steps]
            record["per_step_predicted_state"] = per_step_states
            # Raw {field: {category: prob}} per step. The decoded state above is an argmax and
            # throws away the confidence a gate needs (e.g. P(execution_status=failure) for the
            # critic trigger), so both are returned.
            record["per_step_field_probs"] = steps
            record["per_step_terminal_prob"] = terminal_prob_steps[record["index"]]
            record["terminal_probability"] = (
                terminal_prob_steps[record["index"]][-1] if terminal_prob_steps[record["index"]] else None
            )
            record["predicted_state"] = per_step_states[-1] if per_step_states else {}
        return all_scored

    def predict_action_success_probability(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        action: Any,
        input_history: list[dict[str, Any]],
    ) -> float:
        if not getattr(self, "success_head_available", False):
            raise RuntimeError(
                "This JEPA checkpoint does not contain a trained success_head; "
                "train or fine-tune with --success-loss-coeff > 0 before classifier-guided replay."
            )
        context_text = self._context_text(system_prompt, user_prompt)
        current_text = self._current_state_text(system_prompt, user_prompt, input_history)
        action_text = action if isinstance(action, str) else self._safe_json(action)
        with torch.no_grad():
            z_current = self._encode_latent_text(current_text, self.max_input_length)
            z_context = self._encode_latent_text(context_text, self.max_input_length)
            z_action = self._encode_latent_text(action_text, self.max_action_length)
            z_goal = None
            if getattr(self.model, "goal_conditioning", False):
                z_goal = self._encode_latent_text(self._goal_text(system_prompt, user_prompt), self.max_goal_length)
            z_pred, _ = self.model.predict_latent(z_current, z_action, z_context, z_goal)
            logit = self.model.predict_success_logit(z_current, z_action, z_context, z_pred)
            return float(torch.sigmoid(logit.float()).item())

    def select_best_action_plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        input_history: list[dict[str, Any]],
        candidate_actions: list[Any],
    ) -> dict[str, Any]:
        plans = [[action] for action in candidate_actions]
        scored = self.score_action_plans(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_history=input_history,
            action_plans=plans,
        )
        if not scored:
            raise ValueError("No candidate action plans to score.")
        return {
            "selected_action": scored[0]["plan"][0],
            "selected_score": scored[0]["score"],
            "ranked_plans": scored,
            "objective": "minimize_terminal_latent_goal_mse",
        }

    def predict_canonical_event_labels(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        action: Any,
        input_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the canonical_event_state/nudge classification heads for one action
        and decode the logits into human-readable labels (see
        src.finetuning_jepa.decode_canonical_event_logits)."""
        if not getattr(self, "canonical_event_available", False):
            raise RuntimeError(
                "This JEPA checkpoint does not contain trained canonical_event heads."
            )
        context_text = self._context_text(system_prompt, user_prompt)
        current_text = self._current_state_text(system_prompt, user_prompt, input_history)
        action_text = action if isinstance(action, str) else self._safe_json(action)
        with torch.no_grad():
            z_current = self._encode_latent_text(current_text, self.max_input_length)
            z_context = self._encode_latent_text(context_text, self.max_input_length)
            z_action = self._encode_latent_text(action_text, self.max_action_length)
            z_goal = None
            if getattr(self.model, "goal_conditioning", False):
                z_goal = self._encode_latent_text(self._goal_text(system_prompt, user_prompt), self.max_goal_length)
            z_pred, _, z_state = self.model.predict_latent_with_state(z_current, z_action, z_context, z_goal)
            logits = self.model.predict_canonical_event_logits(z_current, z_action, z_context, z_pred, z_state)
        return self._decode_canonical_event_labels_from_logits(logits)

    def _decode_canonical_event_labels_from_logits(self, logits: dict[str, Any]) -> dict[str, Any]:
        return self._decode_canonical_event_logits(logits, self.canonical_event_vocab)

    def _canonical_event_feedback(
        self,
        *,
        predicted_calls: list[dict[str, Any]],
        action: Any,
        system_prompt: str,
        user_prompt: str,
        input_history: list[dict[str, Any]],
        interaction_index: int,
        world_model_target: str,
    ) -> list[dict[str, Any]]:
        """Reconstruct an imagined feedback/state from the classification heads.

        Produces the same feedback shape as the success-head/decoder paths so the
        imagined-trajectory machinery (imagine_trajectory ->
        build_imagined_trajectory_prompt_message) inserts the reconstructed state
        into the agent prompt unchanged.
        """
        labels = self.predict_canonical_event_labels(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            action=action,
            input_history=input_history,
        )
        tool_name = predicted_calls[0].get("name") if predicted_calls else None
        predicted_state = self._reconstruct_state_from_canonical_event_labels(labels, tool_name=tool_name)
        observation_payload = self._build_canonical_event_observation_payload(labels, tool_name=tool_name)
        outcome = observation_payload["tool_outcome"]
        label = outcome.get("label")
        # Unknown execution_status (label None) is treated as a non-blocking
        # success so imagined planning is not derailed by an abstention.
        predicted_success = True if label is None else label == 1
        predicted_error_message = ""
        if label in (-1, 0):
            context = predicted_state.get("state", {}).get("context", {})
            predicted_error_message = context.get("error_message") or (
                f"JEPA canonical-event head predicts execution_status={labels.get('execution_status')}."
            )
        raw_prediction = json.dumps(observation_payload, ensure_ascii=False, sort_keys=True)
        return [
            {
                "tool_calls": predicted_calls,
                "predicted_success": predicted_success,
                "predicted_success_probability": None,
                "predicted_state": predicted_state,
                "predicted_tool_output": raw_prediction,
                "predicted_error_message": predicted_error_message,
                "predicted_current_stage": None,
                "predicted_remaining_stages": None,
                "predicted_canonical_event_state": labels,
                "raw_prediction": raw_prediction,
                "parse_error": None,
                "world_model_target": world_model_target,
                "world_model_backend": "text_leworldmodel_jepa_canonical_event_heads",
                "interaction_index": interaction_index,
            }
        ]

    def predict_feedback(
        self,
        task: "TaskTrajectory",
        previous_state: dict[str, Any],
        predicted_calls: list[dict[str, Any]],
        interaction_index: int,
        world_model_target: str,
        include_error_message_in_target: bool = False,
        include_stage_in_target: bool = False,
        include_world_model_history: bool = False,
        state_history: list[dict[str, Any]] | None = None,
        input_history: list[dict[str, Any]] | None = None,
        system_prompt_max_chars: int = 0,
        action_max_chars: int = 0,
    ) -> list[dict[str, Any]]:
        del previous_state, include_error_message_in_target, include_stage_in_target, include_world_model_history
        del state_history, system_prompt_max_chars, action_max_chars
        action = {"tool_calls": to_openai_tool_calls(predicted_calls)}
        user_prompt = task.user_messages[-1] if task.user_messages else ""

        if getattr(self, "canonical_event_state_enabled", False):
            return self._canonical_event_feedback(
                predicted_calls=predicted_calls,
                action=action,
                system_prompt=task.system_prompt,
                user_prompt=user_prompt,
                input_history=list(input_history or []),
                interaction_index=interaction_index,
                world_model_target=world_model_target,
            )

        if getattr(self, "success_head_available", False):
            success_probability = self.predict_action_success_probability(
                system_prompt=task.system_prompt,
                user_prompt=user_prompt,
                action=action,
                input_history=list(input_history or []),
            )
            predicted_success = success_probability >= 0.5
            predicted_error_message = "" if predicted_success else "JEPA classifier predicts this tool call is likely to fail."
            predicted_observation = {
                "schema": "ewm_classifier_observation_v1",
                "tool_outcome": {
                    "success": predicted_success,
                    "label": 1 if predicted_success else 0,
                    "success_probability": success_probability,
                    "error_message": predicted_error_message,
                    "summary": (
                        "JEPA classifier predicts this tool call will succeed."
                        if predicted_success
                        else "JEPA classifier predicts this tool call will fail or have no useful effect."
                    ),
                },
            }
            raw_prediction = json.dumps(predicted_observation, ensure_ascii=False, sort_keys=True)
            return [
                {
                    "tool_calls": predicted_calls,
                    "predicted_success": predicted_success,
                    "predicted_success_probability": success_probability,
                    "predicted_state": predicted_observation,
                    "predicted_tool_output": raw_prediction,
                    "predicted_error_message": predicted_error_message,
                    "predicted_current_stage": None,
                    "predicted_remaining_stages": None,
                    "raw_prediction": raw_prediction,
                    "parse_error": None,
                    "world_model_target": world_model_target,
                    "world_model_backend": "text_leworldmodel_jepa_success_classifier",
                    "interaction_index": interaction_index,
                }
            ]

        predicted_tool_output = self._generate_observation(
            system_prompt=task.system_prompt,
            user_prompt=user_prompt,
            action=action,
            input_history=list(input_history or []),
        )
        predicted_success = not tool_output_looks_like_failure(predicted_tool_output)
        return [
            {
                "tool_calls": predicted_calls,
                "predicted_success": predicted_success,
                "predicted_success_probability": None,
                "predicted_state": None,
                "predicted_tool_output": predicted_tool_output,
                "predicted_error_message": predicted_tool_output if not predicted_success else "",
                "predicted_current_stage": None,
                "predicted_remaining_stages": None,
                "raw_prediction": predicted_tool_output,
                "parse_error": None,
                "world_model_target": world_model_target,
                "world_model_backend": "text_leworldmodel_jepa_decoder_fallback",
                "interaction_index": interaction_index,
            }
        ]


class LLMTextGenerator:
    # generate_from_messages mutates self.client (system_prompt state), so parallel
    # requests must go through per-thread clones -- see clone_for_parallel_requests.
    supports_parallel_requests = True

    def __init__(self, method: str, max_new_tokens: int) -> None:
        LLM = require_llm_class()
        self.method = method
        self.max_new_tokens = max_new_tokens
        self.client = LLM(method)
        # Cap the server-side decode budget at what this run actually asks for; src.llm
        # otherwise defaults to 2000 tokens, which at ~20-25 ms/token is a latency cliff.
        self.client.vllm_max_tokens = max(1, int(max_new_tokens))
        self._openai = None
        self._prefix_cache_checked = False

    def clone_for_parallel_requests(self) -> "LLMTextGenerator":
        clone = LLMTextGenerator(self.method, self.max_new_tokens)
        clone._prefix_cache_checked = True   # the parent already reported for this endpoint
        return clone

    def _warn_if_prefix_caching_inactive(self) -> None:
        """One-shot check that a vLLM endpoint is actually reusing shared prompt prefixes.

        Every planning/revision call in this pipeline repeats the agent's system prompt and
        conversation verbatim and only appends a short instruction, so with prefix caching the
        prefill is nearly free and without it every call re-reads the whole prompt. A server
        started without --enable-prefix-caching reports zero prefix-cache queries, which is
        worth saying out loud once rather than paying for it silently.
        """
        if self._prefix_cache_checked or not looks_like_vllm_agent_model(self.method):
            return
        self._prefix_cache_checked = True
        try:
            import urllib.request as _request

            base = self._vllm_chat_base_url().rsplit("/v1", 1)[0]
            with _request.urlopen(f"{base}/metrics", timeout=5) as response:
                body = response.read().decode("utf-8", errors="replace")
            queries = 0.0
            for line in body.splitlines():
                if line.startswith("vllm:prefix_cache_queries_total"):
                    queries = float(line.rsplit(" ", 1)[1])
                    break
            else:
                return   # counter absent: not a vLLM build that reports it, say nothing
            if queries <= 0.0:
                print(
                    f"[vllm] {self.method}: server reports no prefix-cache activity. Planning "
                    "calls re-share the agent's prompt prefix, so restart the server with "
                    "`--enable-prefix-caching --enable-chunked-prefill` to skip the repeated "
                    "prefill (this is the single largest agent-side overhead in world-model "
                    "assisted replay).",
                    flush=True,
                )
        except Exception:                                          # noqa: BLE001
            return   # never let a diagnostic break generation

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.client.system_prompt_enable = False
        self.client.system_prompt = None
        return strip_model_thinking_output(
            self.client(prompt, None, temperature=temperature)
        )

    def generate_from_messages(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        system_parts = [str(message.get("content", "")) for message in messages if message.get("role") == "system"]
        system_prompt = "\n\n".join(part for part in system_parts if part).strip() or None
        non_system_messages = [message for message in messages if message.get("role") != "system"]
        prompt_messages = non_system_messages or messages
        prompt = apply_chat_template_or_fallback(
            None,
            prompt_messages,
            add_generation_prompt=True,
            disable_chat_template=True,
        )
        if self.method == "claude" and system_prompt:
            prompt = f"SYSTEM: {system_prompt}\n\n{prompt}"
        self.client.system_prompt_enable = bool(system_prompt)
        self.client.system_prompt = system_prompt
        output = strip_model_thinking_output(
            self.client(prompt, system_prompt, temperature=temperature)
        )
        self._warn_if_prefix_caching_inactive()
        return output

    def generate_samples(
        self, messages: list[dict[str, Any]], temperature: float = 0.0, num_samples: int = 1
    ) -> list[str] | None:
        """k samples of one prompt in ONE vLLM request (`n=k`).

        The server prefills the shared prompt once and decodes the k sequences concurrently,
        so k candidates cost about one call instead of k. Returns None for non-vLLM methods so
        the caller falls back to parallel requests.
        """
        k = max(1, int(num_samples))
        if not looks_like_vllm_agent_model(self.method):
            return None
        if k == 1:
            return [self.generate_from_messages(messages, temperature=temperature)]
        openai = self._get_openai_client()
        client = openai.OpenAI(
            base_url=self._vllm_chat_base_url(),
            api_key=getattr(self.client, "vllm_api_key", None) or os.environ.get("VLLM_API_KEY") or "not-needed",
        )
        response = client.chat.completions.create(
            model=getattr(self.client, "vllm_model", self.method.split("/", 1)[1]),
            messages=self._to_chat_messages(messages),
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            n=k,
        )
        self._warn_if_prefix_caching_inactive()
        return [
            strip_model_thinking_output(choice.message.content or "")
            for choice in response.choices
        ]

    def _vllm_chat_base_url(self) -> str:
        endpoint = getattr(self.client, "vllm_endpoint", "") or ""
        suffix = "/chat/completions"
        if endpoint.endswith(suffix):
            return endpoint[: -len(suffix)]
        return endpoint

    def _get_openai_client(self) -> Any:
        if self._openai is None:
            try:
                import openai
            except ImportError as exc:
                raise RuntimeError(
                    "The `openai` package is required for vLLM native tool calling."
                ) from exc
            self._openai = openai
        return self._openai

    def _to_chat_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chat_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if role == "assistant":
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    formatted_calls = []
                    for index, tool_call in enumerate(tool_calls):
                        function = tool_call.get("function") if isinstance(tool_call, dict) else None
                        if not isinstance(function, dict):
                            function = tool_call if isinstance(tool_call, dict) else {}
                        arguments = function.get("arguments", {})
                        if not isinstance(arguments, str):
                            arguments = json.dumps(arguments, ensure_ascii=False)
                        formatted_calls.append(
                            {
                                "id": tool_call.get("id", f"call_{len(chat_messages)}_{index}"),
                                "type": "function",
                                "function": {
                                    "name": function.get("name", ""),
                                    "arguments": arguments,
                                },
                            }
                        )
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": content or None,
                            "tool_calls": formatted_calls,
                        }
                    )
                    continue
            elif role == "tool":
                chat_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.get("tool_call_id", ""),
                        "content": content,
                    }
                )
                continue
            if role not in ("system", "user", "assistant"):
                role = "user"
            chat_messages.append({"role": role, "content": content})
        return chat_messages

    def _clean_json_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}, "required": []}

        if "oneOf" in schema:
            for option in schema["oneOf"]:
                if isinstance(option, dict) and option.get("type") == "object":
                    schema = option
                    break
            else:
                return {"type": "object", "properties": {}, "required": []}

        if "allOf" in schema:
            merged_schema = {"type": "object", "properties": {}, "required": []}
            for sub_schema in schema["allOf"]:
                if not isinstance(sub_schema, dict):
                    continue
                if "properties" in sub_schema:
                    merged_schema["properties"].update(sub_schema["properties"])
                if "required" in sub_schema:
                    merged_schema["required"].extend(sub_schema["required"])
            schema = merged_schema

        if "anyOf" in schema:
            for option in schema["anyOf"]:
                if isinstance(option, dict) and option.get("type") == "object":
                    schema = option
                    break
            else:
                return {"type": "object", "properties": {}, "required": []}

        cleaned = dict(schema)
        cleaned.setdefault("type", "object")
        if cleaned["type"] == "object":
            cleaned.setdefault("properties", {})
            cleaned.setdefault("required", [])
        return cleaned

    def _to_openai_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted_tools: list[dict[str, Any]] = []
        for tool in tools:
            input_schema = tool.get("inputSchema") or tool.get("input_schema") or {}
            formatted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": self._clean_json_schema(input_schema),
                    },
                }
            )
        return formatted_tools

    def invoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        # Accepts both `vllm/<name>` and `vllm:<port>/<name>` (see
        # looks_like_vllm_agent_model) -- the port form must not fall through to the
        # NotImplementedError just because of the prefix spelling.
        if not looks_like_vllm_agent_model(self.method):
            raise NotImplementedError(
                f"Native tool calling is not implemented for agent backend {self.method!r}."
            )
        openai = self._get_openai_client()
        client = openai.OpenAI(
            base_url=self._vllm_chat_base_url(),
            api_key=getattr(self.client, "vllm_api_key", None) or os.environ.get("VLLM_API_KEY") or "not-needed",
        )
        request_kwargs: dict[str, Any] = {
            "model": getattr(self.client, "vllm_model", self.method.split("/", 1)[1]),
            "messages": self._to_chat_messages(messages),
            "tools": self._to_openai_tools(tools),
            "tool_choice": "auto",
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
        }
        response = client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        tool_calls = []
        for index, tool_call in enumerate(message.tool_calls or []):
            arguments: Any = tool_call.function.arguments or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = arguments.strip()
            tool_calls.append(
                {
                    "id": tool_call.id or f"call_{index}",
                    "name": tool_call.function.name,
                    "args": arguments,
                    "type": "tool_call",
                }
            )
        return SimpleNamespace(
            content=message.content or "",
            tool_calls=tool_calls,
            usage_metadata={},
            response_metadata={},
        )


def _is_openai_reasoning_model(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith(("o1", "o3", "o4", "gpt-5"))


def _openai_reasoning_effort_for_model(name: str) -> str | None:
    configured = os.environ.get("OPENAI_REASONING_EFFORT")
    if configured:
        return configured
    if name.lower().startswith("gpt-5.1"):
        return "none"
    return None


_OPENAI_AGENT_MODEL_NAME_RE = re.compile(
    r"^(?:gpt-\d|o[134](?:[-_]|$)|chatgpt-)",
    re.IGNORECASE,
)

_GEMINI_AGENT_MODEL_NAME_RE = re.compile(
    r"^gemini-\d",
    re.IGNORECASE,
)


def _looks_like_openai_agent_model(name: str) -> bool:
    """Detect bare OpenAI model names (no `openai/` prefix needed).

    Matches public OpenAI identifiers like `gpt-4o`, `gpt-4o-mini`, `gpt-5`,
    `gpt-5-mini`, `gpt-3.5-turbo`, `o1`, `o1-mini`, `o3`, `o3-mini`, `o4-mini`,
    `chatgpt-4o-latest`. Skips names containing `/` (those are HuggingFace
    repos like `openai-community/gpt2`) and names ending in `.gguf` etc.
    """
    if "/" in name or name.endswith((".gguf", ".bin", ".pt", ".safetensors")):
        return False
    return bool(_OPENAI_AGENT_MODEL_NAME_RE.match(name))


def _looks_like_gemini_agent_model(name: str) -> bool:
    if "/" in name or name.endswith((".gguf", ".bin", ".pt", ".safetensors")):
        return False
    return bool(_GEMINI_AGENT_MODEL_NAME_RE.match(name))


class OpenAIChatGenerator:
    """Direct OpenAI Chat Completions wrapper for use as the agent model.

    Invoked when `--agent-model` is given as `openai/<model>` (e.g.
    `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-5`). Reads credentials
    from the standard `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`) env
    vars via the `openai` SDK. Reasoning models (`gpt-5*`, `o1*`, `o3*`,
    `o4*`) automatically use `max_completion_tokens` and skip `temperature`,
    matching the SDK's expectations for those models.
    """

    # The openai SDK client is thread-safe; parallel requests can share it.
    supports_parallel_requests = True

    def __init__(self, model: str, max_new_tokens: int) -> None:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "The `openai` package is required for `--agent-model openai/...`."
            ) from exc
        self._openai = openai
        self.model = model
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self.client = openai.OpenAI()
        self.is_reasoning_model = _is_openai_reasoning_model(model)

    def _to_chat_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        chat_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if role == "assistant":
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    formatted_calls = []
                    for index, tool_call in enumerate(tool_calls):
                        function = tool_call.get("function") if isinstance(tool_call, dict) else None
                        if not isinstance(function, dict):
                            function = tool_call if isinstance(tool_call, dict) else {}
                        arguments = function.get("arguments", {})
                        if not isinstance(arguments, str):
                            arguments = json.dumps(arguments, ensure_ascii=False)
                        formatted_calls.append(
                            {
                                "id": tool_call.get("id", f"call_{len(chat_messages)}_{index}"),
                                "type": "function",
                                "function": {
                                    "name": function.get("name", ""),
                                    "arguments": arguments,
                                },
                            }
                        )
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": content or None,
                            "tool_calls": formatted_calls,
                        }
                    )
                    continue
            elif role == "tool":
                chat_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.get("tool_call_id", ""),
                        "content": content,
                    }
                )
                continue
            if role not in ("system", "user", "assistant"):
                role = "user"
            chat_messages.append({"role": role, "content": content})
        return chat_messages

    def _clean_json_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}, "required": []}

        if "oneOf" in schema:
            for option in schema["oneOf"]:
                if isinstance(option, dict) and option.get("type") == "object":
                    schema = option
                    break
            else:
                return {"type": "object", "properties": {}, "required": []}

        if "allOf" in schema:
            merged_schema = {"type": "object", "properties": {}, "required": []}
            for sub_schema in schema["allOf"]:
                if not isinstance(sub_schema, dict):
                    continue
                if "properties" in sub_schema:
                    merged_schema["properties"].update(sub_schema["properties"])
                if "required" in sub_schema:
                    merged_schema["required"].extend(sub_schema["required"])
            schema = merged_schema

        if "anyOf" in schema:
            for option in schema["anyOf"]:
                if isinstance(option, dict) and option.get("type") == "object":
                    schema = option
                    break
            else:
                return {"type": "object", "properties": {}, "required": []}

        cleaned = dict(schema)
        cleaned.setdefault("type", "object")
        if cleaned["type"] == "object":
            cleaned.setdefault("properties", {})
            cleaned.setdefault("required", [])
        return cleaned

    def _to_openai_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted_tools: list[dict[str, Any]] = []
        for tool in tools:
            input_schema = tool.get("inputSchema") or tool.get("input_schema") or {}
            formatted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": self._clean_json_schema(input_schema),
                    },
                }
            )
        return formatted_tools

    def invoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_chat_messages(messages),
            "tools": self._to_openai_tools(tools),
            "tool_choice": "auto",
        }
        if self.is_reasoning_model:
            request_kwargs["max_completion_tokens"] = self.max_new_tokens
            reasoning_effort = _openai_reasoning_effort_for_model(self.model)
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = reasoning_effort
        else:
            request_kwargs["max_tokens"] = self.max_new_tokens
            request_kwargs["temperature"] = 0.0

        response = self.client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        tool_calls = []
        for index, tool_call in enumerate(message.tool_calls or []):
            arguments: Any = tool_call.function.arguments or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = arguments.strip()
            tool_calls.append(
                {
                    "id": tool_call.id or f"call_{index}",
                    "name": tool_call.function.name,
                    "args": arguments,
                    "type": "tool_call",
                }
            )
        return SimpleNamespace(
            content=message.content or "",
            tool_calls=tool_calls,
            usage_metadata={},
            response_metadata={},
        )

    def generate_from_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        chat_messages = self._to_chat_messages(messages)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
        }
        if self.is_reasoning_model:
            request_kwargs["max_completion_tokens"] = self.max_new_tokens
            reasoning_effort = _openai_reasoning_effort_for_model(self.model)
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = reasoning_effort
        else:
            request_kwargs["max_tokens"] = self.max_new_tokens
            request_kwargs["temperature"] = temperature

        max_attempts = int(os.environ.get("LLM_RATE_LIMIT_MAX_RETRIES", "30"))
        attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                return strip_model_thinking_output(
                    response.choices[0].message.content or ""
                )
            except self._openai.BadRequestError as exc:
                message = str(exc).lower()
                if "max_tokens" in message and "max_completion_tokens" in message:
                    request_kwargs["max_completion_tokens"] = request_kwargs.pop("max_tokens", self.max_new_tokens)
                    self.is_reasoning_model = True
                    continue
                if "temperature" in message and "unsupported" in message:
                    request_kwargs.pop("temperature", None)
                    continue
                raise
            except self._openai.RateLimitError as exc:
                attempt += 1
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"OpenAI rate limit did not clear after {max_attempts} retries."
                    ) from exc
                wait = min(60, 2 ** min(attempt, 6))
                print(
                    f"[openai-agent] rate-limit, sleeping {wait}s "
                    f"(attempt {attempt}/{max_attempts}): {exc}"
                )
                time.sleep(wait)
            except (
                self._openai.APITimeoutError,
                self._openai.APIConnectionError,
                self._openai.InternalServerError,
            ) as exc:
                attempt += 1
                if attempt >= max_attempts:
                    raise
                wait = min(30, 2 ** min(attempt, 5))
                print(f"[openai-agent] transient error, sleeping {wait}s: {exc}")
                time.sleep(wait)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self.generate_from_messages(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
        )

    def generate_samples(
        self, messages: list[dict[str, str]], temperature: float = 0.0, num_samples: int = 1
    ) -> list[str] | None:
        """k samples of one prompt in ONE request via `n=k` (input tokens billed once).

        Reasoning models reject `n`, so those return None and the caller falls back to
        parallel requests.
        """
        k = max(1, int(num_samples))
        if k == 1:
            return [self.generate_from_messages(messages, temperature=temperature)]
        if self.is_reasoning_model:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._to_chat_messages(messages),
                max_tokens=self.max_new_tokens,
                temperature=temperature,
                n=k,
            )
        except self._openai.BadRequestError:
            return None            # server rejected `n`; caller falls back to parallel calls
        return [
            strip_model_thinking_output(choice.message.content or "")
            for choice in response.choices
        ]


class AzureOpenAIChatGenerator(OpenAIChatGenerator):
    """Direct Azure OpenAI chat-completions wrapper for agent generation."""

    def __init__(self, deployment: str, max_new_tokens: int) -> None:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "The `openai` package is required for `--agent-model azureopenai/...`."
            ) from exc

        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = (
            os.environ.get("AZURE_OPENAI_ENDPOINT")
            or os.environ.get("AZURE_OPENAI_BASE_URL")
        )
        api_version = (
            os.environ.get("AZURE_OPENAI_API_VERSION")
            or os.environ.get("OPENAI_API_VERSION")
        )
        missing = []
        if not api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if missing:
            missing_str = ", ".join(missing)
            raise RuntimeError(
                "Azure OpenAI agent models require environment variables: "
                f"{missing_str}."
            )

        normalized_endpoint = endpoint.rstrip("/")
        if normalized_endpoint.endswith("/openai/v1"):
            client = openai.OpenAI(
                api_key=api_key,
                base_url=normalized_endpoint + "/",
            )
        else:
            if not api_version:
                raise RuntimeError(
                    "Azure OpenAI agent models require `AZURE_OPENAI_API_VERSION` "
                    "when `AZURE_OPENAI_ENDPOINT` is a resource endpoint."
                )
            client = openai.AzureOpenAI(
                azure_endpoint=normalized_endpoint,
                api_key=api_key,
                api_version=api_version,
            )

        self._openai = openai
        self.model = deployment
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self.client = client
        self.is_reasoning_model = _is_openai_reasoning_model(deployment)


class GeminiChatGenerator:
    """Direct Gemini API wrapper for agent generation and native tool calling."""

    # Requests use a fresh urllib request per call; no shared mutable state.
    supports_parallel_requests = True

    def __init__(self, model: str, max_new_tokens: int) -> None:
        self.model = model
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.api_base_url = os.environ.get(
            "GEMINI_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ).rstrip("/")
        if not self.api_key:
            raise RuntimeError(
                "Gemini agent models require `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)."
            )

    def _api_url(self) -> str:
        model_name = self.model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        encoded_model = urllib.parse.quote(model_name, safe="/")
        encoded_key = urllib.parse.quote(self.api_key, safe="")
        return f"{self.api_base_url}/{encoded_model}:generateContent?key={encoded_key}"

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self._api_url(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        max_attempts = int(os.environ.get("LLM_RATE_LIMIT_MAX_RETRIES", "30"))
        attempt = 0
        while True:
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    raw_body = response.read().decode("utf-8")
                return json.loads(raw_body)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                attempt += 1
                if exc.code == 429 and attempt < max_attempts:
                    wait = min(60, 2 ** min(attempt, 6))
                    print(
                        f"[gemini-agent] rate-limit, sleeping {wait}s "
                        f"(attempt {attempt}/{max_attempts}): {body}"
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Gemini API request failed with HTTP {exc.code}: {body}"
                ) from exc
            except urllib.error.URLError as exc:
                attempt += 1
                if attempt >= max_attempts:
                    raise RuntimeError(f"Gemini API request failed: {exc}") from exc
                wait = min(30, 2 ** min(attempt, 5))
                print(f"[gemini-agent] transient error, sleeping {wait}s: {exc}")
                time.sleep(wait)

    def _coerce_message_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    def _extract_system_prompt(self, messages: list[dict[str, Any]]) -> str | None:
        system_parts = [
            self._coerce_message_content(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        ]
        combined = "\n\n".join(part.strip() for part in system_parts if part and part.strip()).strip()
        return combined or None

    def _to_gemini_contents(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                continue

            if role == "assistant":
                parts: list[dict[str, Any]] = []
                content = self._coerce_message_content(message.get("content", "")).strip()
                if content:
                    parts.append({"text": content})
                for index, tool_call in enumerate(message.get("tool_calls") or []):
                    function = tool_call.get("function") if isinstance(tool_call, dict) else None
                    if not isinstance(function, dict):
                        function = tool_call if isinstance(tool_call, dict) else {}
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {"raw": arguments}
                    elif arguments is None:
                        arguments = {}
                    elif not isinstance(arguments, dict):
                        arguments = {"value": arguments}
                    parts.append(
                        {
                            "functionCall": {
                                "id": tool_call.get("id", f"call_{len(contents)}_{index}"),
                                "name": function.get("name", ""),
                                "args": arguments,
                            }
                        }
                    )
                if parts:
                    contents.append({"role": "model", "parts": parts})
                continue

            if role == "tool":
                tool_name = str(message.get("name", "")).strip()
                tool_call_id = str(message.get("tool_call_id", "")).strip()
                raw_content = message.get("content", "")
                if isinstance(raw_content, str):
                    stripped = raw_content.strip()
                    if stripped:
                        try:
                            response_payload: Any = json.loads(stripped)
                        except json.JSONDecodeError:
                            response_payload = {"content": stripped}
                    else:
                        response_payload = {"content": ""}
                else:
                    response_payload = raw_content
                function_response = {
                    "name": tool_name or "tool",
                    "response": {"result": response_payload},
                }
                if tool_call_id:
                    function_response["id"] = tool_call_id
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"functionResponse": function_response}],
                    }
                )
                continue

            content = self._coerce_message_content(message.get("content", "")).strip()
            if content:
                contents.append({"role": "user", "parts": [{"text": content}]})

        return contents

    def _clean_json_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}, "required": []}

        if "oneOf" in schema:
            for option in schema["oneOf"]:
                if isinstance(option, dict) and option.get("type") == "object":
                    schema = option
                    break
            else:
                return {"type": "object", "properties": {}, "required": []}

        if "allOf" in schema:
            merged_schema = {"type": "object", "properties": {}, "required": []}
            for sub_schema in schema["allOf"]:
                if not isinstance(sub_schema, dict):
                    continue
                if "properties" in sub_schema:
                    merged_schema["properties"].update(sub_schema["properties"])
                if "required" in sub_schema:
                    merged_schema["required"].extend(sub_schema["required"])
            schema = merged_schema

        if "anyOf" in schema:
            for option in schema["anyOf"]:
                if isinstance(option, dict) and option.get("type") == "object":
                    schema = option
                    break
            else:
                return {"type": "object", "properties": {}, "required": []}

        cleaned = dict(schema)
        cleaned.setdefault("type", "object")
        if cleaned["type"] == "object":
            cleaned.setdefault("properties", {})
            cleaned.setdefault("required", [])
        return cleaned

    def _to_gemini_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declarations: list[dict[str, Any]] = []
        for tool in tools:
            input_schema = tool.get("inputSchema") or tool.get("input_schema") or {}
            declarations.append(
                {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": self._clean_json_schema(input_schema),
                }
            )
        return [{"function_declarations": declarations}] if declarations else []

    def _base_payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contents": self._to_gemini_contents(messages),
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": self.max_new_tokens,
            },
        }
        system_prompt = self._extract_system_prompt(messages)
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    def _extract_response_text(self, response: dict[str, Any]) -> str:
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return ""
        content = candidates[0].get("content")
        if not isinstance(content, dict):
            return ""
        text_parts: list[str] = []
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        return "\n".join(text_parts).strip()

    def _extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return []
        content = candidates[0].get("content")
        if not isinstance(content, dict):
            return []
        tool_calls: list[dict[str, Any]] = []
        for index, part in enumerate(content.get("parts") or []):
            if not isinstance(part, dict):
                continue
            function_call = part.get("functionCall") or part.get("function_call")
            if not isinstance(function_call, dict):
                continue
            arguments = function_call.get("args", {})
            if arguments is None:
                arguments = {}
            elif not isinstance(arguments, dict):
                arguments = {"value": arguments}
            tool_calls.append(
                {
                    "id": function_call.get("id") or f"call_{index}",
                    "name": function_call.get("name", ""),
                    "args": arguments,
                    "type": "tool_call",
                }
            )
        return tool_calls

    def invoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        payload = self._base_payload(messages, temperature=0.0)
        payload["tools"] = self._to_gemini_tools(tools)
        payload["tool_config"] = {
            "function_calling_config": {
                "mode": "AUTO",
            }
        }
        response = self._request(payload)
        return SimpleNamespace(
            content=self._extract_response_text(response),
            tool_calls=self._extract_tool_calls(response),
            usage_metadata={},
            response_metadata={"gemini_response": response},
        )

    def generate_from_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        payload = self._base_payload(messages, temperature=temperature)
        response = self._request(payload)
        return self._extract_response_text(response)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self.generate_from_messages(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
        )


def build_agent_generator(
    model_path: str,
    max_new_tokens: int,
    trust_remote_code: bool = False,
    dtype: str = "auto",
    disable_chat_template: bool = False,
    attn_implementation: str = "sdpa",
    device_map: str | dict | None = None,
    draft_model_path: str | None = None,
    prompt_lookup_num_tokens: int = 0,
):
    openai_alias = OPENAI_AGENT_MODEL_ALIASES.get(model_path)
    if openai_alias:
        return OpenAIChatGenerator(openai_alias, max_new_tokens=max_new_tokens)
    llm_method = LLM_AGENT_MODEL_ALIASES.get(model_path)
    if llm_method:
        return LLMTextGenerator(llm_method, max_new_tokens=max_new_tokens)
    if model_path.startswith("azureopenai/") or model_path.startswith("azureopenai:"):
        separator = "/" if "/" in model_path else ":"
        azure_deployment = model_path.split(separator, 1)[1].strip()
        if not azure_deployment:
            raise ValueError(
                f"Empty Azure OpenAI deployment name in {model_path!r}. "
                "Use e.g. `azureopenai/my-gpt-5-deployment`."
            )
        return AzureOpenAIChatGenerator(
            azure_deployment,
            max_new_tokens=max_new_tokens,
        )
    if model_path.startswith("gemini/") or model_path.startswith("gemini:"):
        separator = "/" if "/" in model_path else ":"
        gemini_model = model_path.split(separator, 1)[1].strip()
        if not gemini_model:
            raise ValueError(
                f"Empty Gemini model name in {model_path!r}. "
                "Use e.g. `gemini/gemini-2.5-pro`."
            )
        return GeminiChatGenerator(gemini_model, max_new_tokens=max_new_tokens)
    if model_path.startswith("openai/") or model_path.startswith("openai:"):
        separator = "/" if "/" in model_path else ":"
        openai_model = model_path.split(separator, 1)[1].strip()
        if not openai_model:
            raise ValueError(
                f"Empty OpenAI model name in {model_path!r}. "
                "Use e.g. `openai/gpt-4o-mini`."
            )
        return OpenAIChatGenerator(openai_model, max_new_tokens=max_new_tokens)
    if _looks_like_openai_agent_model(model_path):
        return OpenAIChatGenerator(model_path, max_new_tokens=max_new_tokens)
    if _looks_like_gemini_agent_model(model_path):
        return GeminiChatGenerator(model_path, max_new_tokens=max_new_tokens)
    if model_path in API_AGENT_MODEL_METHODS or looks_like_vllm_agent_model(model_path):
        return LLMTextGenerator(model_path, max_new_tokens=max_new_tokens)
    return HFTextGenerator(
        model_path,
        max_new_tokens=max_new_tokens,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        disable_chat_template=disable_chat_template,
        attn_implementation=attn_implementation,
        device_map=device_map,
        draft_model_path=draft_model_path,
        prompt_lookup_num_tokens=prompt_lookup_num_tokens,
    )


def build_tool_lookup(tools: list[Any]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    exact_lookup: dict[str, Any] = {}
    alias_lookup: dict[str, list[Any]] = {}
    for tool in tools:
        exact_lookup[tool.name] = tool
        original_name = getattr(tool, "_original_tool_name", tool.name)
        alias_lookup.setdefault(original_name, []).append(tool)
    return exact_lookup, alias_lookup


def resolve_tool_for_execution(
    requested_name: str,
    exact_lookup: dict[str, Any],
    alias_lookup: dict[str, list[Any]],
) -> tuple[Any | None, str | None]:
    if requested_name in exact_lookup:
        return exact_lookup[requested_name], None
    matches = alias_lookup.get(requested_name, [])
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        resolved_names = ", ".join(tool.name for tool in matches)
        return None, f"Ambiguous tool name `{requested_name}` matched multiple tools: {resolved_names}"
    return None, f"Unknown tool `{requested_name}`"


PER_TOOL_CALL_TIMEOUT_SECONDS = 60.0

# Bound for JepaTextWorldModelGenerator._encode_latent_text's text->latent memo. Entries are
# [1, D] tensors; a task only ever needs a handful (context, goal, the recent state texts).
LATENT_TEXT_CACHE_SIZE = 32


async def execute_actual_tool_calls(
    predicted_calls: list[dict[str, Any]],
    exact_lookup: dict[str, Any],
    alias_lookup: dict[str, list[Any]],
    per_call_timeout_seconds: float = PER_TOOL_CALL_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    execution_results = []
    for predicted_call in predicted_calls:
        emit_progress(
            "TOOL_CALL_START",
            tool=predicted_call.get("name"),
            arguments=predicted_call.get("arguments"),
        )
        tool, resolution_error = resolve_tool_for_execution(
            predicted_call["name"],
            exact_lookup,
            alias_lookup,
        )
        if resolution_error:
            execution_results.append(
                {
                    "requested_name": predicted_call["name"],
                    "resolved_name": None,
                    "arguments": predicted_call["arguments"],
                    "content": f"Error: {resolution_error}",
                    "success": False,
                }
            )
            continue

        try:
            arguments = predicted_call["arguments"]
            if hasattr(tool, "ainvoke"):
                raw_output = await asyncio.wait_for(
                    tool.ainvoke(arguments),
                    timeout=per_call_timeout_seconds,
                )
            else:
                raw_output = tool.invoke(arguments)
            content = stringify_tool_output(raw_output)
            success = not tool_response_is_error(content)
        except asyncio.CancelledError:
            # If our task is genuinely being cancelled from outside (Ctrl+C, run
            # teardown), propagate it. If only the MCP SDK's internal anyio
            # cancel scope fired (server-side timeout, stream close, etc.),
            # treat it as a per-tool failure and let the loop continue.
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            content = (
                "Error: tool call cancelled by MCP cancel scope "
                "(likely server-side timeout or stream close)"
            )
            success = False
        except (asyncio.TimeoutError, TimeoutError):
            content = (
                f"Error: tool call exceeded {per_call_timeout_seconds:.0f}s timeout"
            )
            success = False
        except Exception as exc:
            content = f"Error: {exc}"
            success = False

        execution_results.append(
            {
                "requested_name": predicted_call["name"],
                "resolved_name": tool.name,
                "arguments": predicted_call["arguments"],
                "content": content,
                "success": success,
            }
        )
        emit_progress(
            "TOOL_CALL_RESULT",
            requested_name=predicted_call.get("name"),
            resolved_name=tool.name,
            success=success,
            content=content,
        )
    return execution_results


def predict_world_model_feedback(
    world_model_generator: Any,
    task: TaskTrajectory,
    previous_state: dict[str, Any],
    predicted_calls: list[dict[str, Any]],
    interaction_index: int,
    world_model_target: str,
    include_error_message_in_target: bool = False,
    include_stage_in_target: bool = False,
    include_world_model_history: bool = False,
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> list[dict[str, Any]]:
    if hasattr(world_model_generator, "predict_feedback"):
        feedbacks = world_model_generator.predict_feedback(
            task=task,
            previous_state=previous_state,
            predicted_calls=predicted_calls,
            interaction_index=interaction_index,
            world_model_target=world_model_target,
            include_error_message_in_target=include_error_message_in_target,
            include_stage_in_target=include_stage_in_target,
            include_world_model_history=include_world_model_history,
            state_history=state_history,
            input_history=input_history,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
        )
        last_feedback = feedbacks[-1] if feedbacks else {}
        emit_progress(
            "WORLD_MODEL_RAW",
            interaction_index=interaction_index,
            target=world_model_target,
            tool_calls=preview_tool_calls(predicted_calls),
            raw_prediction=last_feedback.get("raw_prediction"),
            backend=last_feedback.get("world_model_backend"),
        )
        emit_progress(
            "WORLD_MODEL_RESULT",
            interaction_index=interaction_index,
            predicted_success=last_feedback.get("predicted_success"),
            predicted_error_message=last_feedback.get("predicted_error_message"),
            predicted_tool_output=last_feedback.get("predicted_tool_output", ""),
            predicted_current_stage=last_feedback.get("predicted_current_stage"),
            predicted_remaining_stages=last_feedback.get("predicted_remaining_stages"),
            parse_error=last_feedback.get("parse_error"),
            backend=last_feedback.get("world_model_backend"),
        )
        return feedbacks

    prediction = world_model_generator.generate_from_messages(
        build_world_model_prediction_messages(
            task=task,
            previous_state=previous_state,
            predicted_calls=predicted_calls,
            interaction_index=interaction_index,
            world_model_target=world_model_target,
            include_error_message_in_target=include_error_message_in_target,
            include_stage_in_target=include_stage_in_target,
            include_world_model_history=include_world_model_history,
            state_history=state_history,
            input_history=input_history,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
        )
    )
    return finish_world_model_text_feedback(
        prediction,
        predicted_calls=predicted_calls,
        interaction_index=interaction_index,
        world_model_target=world_model_target,
    )


def build_world_model_prediction_messages(
    *,
    task: TaskTrajectory,
    previous_state: dict[str, Any],
    predicted_calls: list[dict[str, Any]],
    interaction_index: int,
    world_model_target: str,
    include_error_message_in_target: bool = False,
    include_stage_in_target: bool = False,
    include_world_model_history: bool = False,
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> list[dict[str, str]]:
    """Chat messages for one text-world-model prediction (shared by the serial path in
    predict_world_model_feedback and the lockstep batched path in
    imagine_trajectories_lockstep)."""
    prediction_example = WorldModelStateExample(
        trajectory_id=str(task.trajectory_index),
        trajectory_index=task.trajectory_index,
        interaction_index=interaction_index,
        system_prompt=task.system_prompt,
        user_prompt=task.user_messages[-1] if task.user_messages else "",
        action={"tool_calls": to_openai_tool_calls(predicted_calls)},
        state_history=list(state_history or []),
        input_history=list(input_history or []),
        previous_state=previous_state,
        state=make_blank_state_like(previous_state),
    )
    return build_state_prediction_chat_messages(
        prediction_example,
        target_mode=world_model_target,
        include_error_message=include_error_message_in_target,
        include_stage=include_stage_in_target,
        include_input_history=include_world_model_history,
        system_prompt_max_chars=system_prompt_max_chars,
        action_max_chars=action_max_chars,
    )


def finish_world_model_text_feedback(
    prediction: str,
    *,
    predicted_calls: list[dict[str, Any]],
    interaction_index: int,
    world_model_target: str,
) -> list[dict[str, Any]]:
    """Parse one raw text-world-model completion into the feedback shape (shared by the
    serial and batched prediction paths)."""
    prediction = prediction.split("</think>\n", 1)[-1].strip()
    emit_progress(
        "WORLD_MODEL_RAW",
        interaction_index=interaction_index,
        target=world_model_target,
        tool_calls=preview_tool_calls(predicted_calls),
        raw_prediction=prediction,
    )
    predicted_state = None
    predicted_tool_output = ""
    predicted_current_stage = None
    predicted_remaining_stages = None
    parse_error = None
    try:
        if is_tool_execution_result_target(world_model_target):
            parsed_payload = parse_binary_world_model_prediction(prediction)
            predicted_result = normalize_tool_execution_result_for_target(
                parsed_payload.get("success"),
                target_mode=world_model_target,
            )
            if predicted_result is None:
                raise ValueError(f"Unable to parse tool-result prediction: {prediction[:200]}")
            predicted_success = predicted_result == 1
            error_message = (parsed_payload.get("error_message") or "").strip()
            predicted_current_stage = parsed_payload.get("current_stage")
            predicted_remaining_stages = parsed_payload.get("remaining_stages")
            predicted_state = make_tool_execution_prediction_state(
                predicted_success=predicted_success,
                predicted_result=predicted_result,
                error_message=error_message,
                current_stage=predicted_current_stage,
                remaining_stages=predicted_remaining_stages,
            )
        elif is_tool_output_target(world_model_target):
            predicted_tool_output = prediction
            looks_like_failure = tool_output_looks_like_failure(prediction)
            predicted_success = not looks_like_failure
            error_message = prediction if looks_like_failure else ""
        else:
            predicted_state = sanitize_state_content(parse_jsonish(prediction))
            context = state_context_from_any(predicted_state)
            predicted_success = normalize_last_tool_execution_result(
                context.get("last_tool_execution_result")
            ) == 1
            error_message = context.get("error_message", "")
            predicted_current_stage = state_current_stage(predicted_state)
            predicted_remaining_stages = state_remaining_stages(predicted_state)
    except Exception as exc:
        predicted_success = False
        error_message = str(exc)
        parse_error = str(exc)

    emit_progress(
        "WORLD_MODEL_RESULT",
        interaction_index=interaction_index,
        predicted_success=predicted_success,
        predicted_error_message=error_message,
        predicted_tool_output=predicted_tool_output,
        predicted_current_stage=predicted_current_stage,
        predicted_remaining_stages=predicted_remaining_stages,
        parse_error=parse_error,
    )

    return [
        {
            "tool_calls": predicted_calls,
            "predicted_success": predicted_success,
            "predicted_state": predicted_state,
            "predicted_tool_output": predicted_tool_output,
            "predicted_error_message": error_message,
            "predicted_current_stage": predicted_current_stage,
            "predicted_remaining_stages": predicted_remaining_stages,
            "raw_prediction": prediction,
            "parse_error": parse_error,
        }
    ]


def imagine_trajectory(
    agent_generator: Any,
    world_model_generator: Any,
    task: TaskTrajectory,
    conversation: list[dict[str, Any]],
    previous_state: dict[str, Any],
    react_system_prompt: str,
    max_imagined_steps: int,
    world_model_target: str,
    include_error_message_in_target: bool = False,
    include_stage_in_target: bool = False,
    include_world_model_history: bool = False,
    start_interaction_index: int = 0,
    rollout_index: int = 0,
    rollout_temperature: float = 0.0,
    observation_source: str = "world_model",
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> list[dict[str, Any]]:
    """Single imagined rollout: thin wrapper over the lockstep driver with one chain."""
    return imagine_trajectories_lockstep(
        agent_generator=agent_generator,
        world_model_generator=world_model_generator,
        task=task,
        conversation=conversation,
        previous_state=previous_state,
        react_system_prompt=react_system_prompt,
        max_imagined_steps=max_imagined_steps,
        world_model_target=world_model_target,
        include_error_message_in_target=include_error_message_in_target,
        include_stage_in_target=include_stage_in_target,
        include_world_model_history=include_world_model_history,
        start_interaction_index=start_interaction_index,
        rollout_temperatures=[rollout_temperature],
        first_rollout_index=rollout_index,
        observation_source=observation_source,
        state_history=state_history,
        input_history=input_history,
        state_history_size=state_history_size,
        system_prompt_max_chars=system_prompt_max_chars,
        action_max_chars=action_max_chars,
    )[0]


def imagine_trajectories_lockstep(
    agent_generator: Any,
    world_model_generator: Any,
    task: TaskTrajectory,
    conversation: list[dict[str, Any]],
    previous_state: dict[str, Any],
    react_system_prompt: str,
    max_imagined_steps: int,
    world_model_target: str,
    rollout_temperatures: list[float],
    include_error_message_in_target: bool = False,
    include_stage_in_target: bool = False,
    include_world_model_history: bool = False,
    start_interaction_index: int = 0,
    first_rollout_index: int = 0,
    observation_source: str = "world_model",
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> list[list[dict[str, Any]]]:
    """Advance N independent imagined rollouts in lockstep, batching every LLM call.

    Serial rollouts cost `N * steps * (think + act + world_model)` sequential generations.
    Here all alive rollouts issue each phase's generations together through generate_many
    (one padded batched `generate` on HF backends; parallel requests on API/vLLM backends),
    so wall-clock is ~`steps * phases` batched calls regardless of N. Per-rollout semantics
    match the previous serial imagine_trajectory exactly: same prompts (in two-call mode),
    same temperatures, same termination rules.

    With IMAGINED_SINGLE_CALL_STEP the two think/act phases collapse into one combined
    generation per step (build_react_step_messages), halving the agent call count again.
    """
    imagined_react_system_prompt = imagined_trajectory_system_prompt(react_system_prompt)
    current_query = task.user_messages[-1] if task.user_messages else ""
    root_state = sanitize_state_content(previous_state)
    rollouts: list[dict[str, Any]] = []
    for offset, temperature in enumerate(rollout_temperatures):
        rollouts.append(
            {
                "rollout_index": first_rollout_index + offset,
                "temperature": float(temperature),
                "conversation": [dict(message) for message in conversation],
                "state": root_state,
                "state_history": append_state_history(
                    list(state_history or []), root_state, max_items=state_history_size
                ),
                "input_history": list(input_history or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:],
                "steps": [],
                "seen_signatures": {},
                "done": False,
            }
        )

    for imagined_index in range(max(0, int(max_imagined_steps))):
        alive = [rollout for rollout in rollouts if not rollout["done"]]
        if not alive:
            break

        # --- Phase A: agent decision (batched across alive rollouts) -----------------
        if IMAGINED_SINGLE_CALL_STEP:
            raw_steps = generate_many(
                agent_generator,
                [
                    build_react_step_messages(
                        rollout["conversation"],
                        current_query=current_query,
                        system_prompt=imagined_react_system_prompt,
                    )
                    for rollout in alive
                ],
                [rollout["temperature"] for rollout in alive],
            )
            raw_steps = [strip_model_thinking_output(raw) for raw in raw_steps]
            raw_actions = raw_steps
            thought_payloads = [parse_thought_payload(raw) for raw in raw_steps]
        else:
            raw_thoughts = generate_many(
                agent_generator,
                [
                    build_react_think_messages(
                        rollout["conversation"],
                        current_query=current_query,
                        system_prompt=imagined_react_system_prompt,
                    )
                    for rollout in alive
                ],
                [rollout["temperature"] for rollout in alive],
            )
            thought_payloads = [
                parse_thought_payload(strip_model_thinking_output(raw)) for raw in raw_thoughts
            ]
            for rollout, thought_payload in zip(alive, thought_payloads):
                rollout["conversation"].append(
                    {"role": "assistant", "content": json.dumps(thought_payload, ensure_ascii=False)}
                )
            raw_actions = [
                strip_model_thinking_output(raw)
                for raw in generate_many(
                    agent_generator,
                    [
                        build_react_action_messages(
                            rollout["conversation"],
                            current_query=current_query,
                            system_prompt=imagined_react_system_prompt,
                        )
                        for rollout in alive
                    ],
                    [rollout["temperature"] for rollout in alive],
                )
            ]

        for rollout, thought_payload in zip(alive, thought_payloads):
            emit_progress(
                "IMAGINED_THOUGHT",
                imagined_step=imagined_index + 1,
                rollout_index=rollout["rollout_index"],
                thought=thought_payload.get("thought"),
            )
            if IMAGINED_SINGLE_CALL_STEP:
                rollout["conversation"].append(
                    {"role": "assistant", "content": json.dumps(thought_payload, ensure_ascii=False)}
                )

        # --- Phase B: interpret decisions; collect world-model requests --------------
        pending_predictions: list[dict[str, Any]] = []
        for rollout, thought_payload, raw_action in zip(alive, thought_payloads, raw_actions):
            try:
                decision = parse_agent_decision(raw_action)
            except Exception as exc:
                emit_progress(
                    "IMAGINED_ACTION_PARSE_ERROR",
                    imagined_step=imagined_index + 1,
                    rollout_index=rollout["rollout_index"],
                    error=str(exc),
                    raw_action=raw_action,
                )
                rollout["steps"].append(
                    {
                        "imagined_step": imagined_index + 1,
                        "thought": thought_payload,
                        "raw_action": raw_action,
                        "parse_error": str(exc),
                    }
                )
                rollout["done"] = True
                continue

            if "final_answer" in decision:
                emit_progress(
                    "IMAGINED_FINAL_ANSWER",
                    imagined_step=imagined_index + 1,
                    rollout_index=rollout["rollout_index"],
                    final_answer=decision["final_answer"],
                )
                rollout["steps"].append(
                    {
                        "imagined_step": imagined_index + 1,
                        "thought": thought_payload,
                        "final_answer": decision["final_answer"],
                        "predicted_state": full_world_model_state_for_agent(rollout["state"]),
                        "predicted_state_summary": summarize_state_for_planning(rollout["state"]),
                    }
                )
                rollout["done"] = True
                continue
            if "clarify" in decision:
                emit_progress(
                    "IMAGINED_CLARIFY",
                    imagined_step=imagined_index + 1,
                    rollout_index=rollout["rollout_index"],
                    clarify=decision["clarify"],
                )
                rollout["steps"].append(
                    {
                        "imagined_step": imagined_index + 1,
                        "thought": thought_payload,
                        "clarify": decision["clarify"],
                        "predicted_state": full_world_model_state_for_agent(rollout["state"]),
                        "predicted_state_summary": summarize_state_for_planning(rollout["state"]),
                    }
                )
                rollout["done"] = True
                continue

            planned_calls = [normalize_tool_call(call) for call in decision.get("tool_calls", [])]
            if not planned_calls:
                emit_progress(
                    "IMAGINED_EMPTY_ACTION",
                    imagined_step=imagined_index + 1,
                    rollout_index=rollout["rollout_index"],
                    raw_action=raw_action,
                )
                rollout["steps"].append(
                    {
                        "imagined_step": imagined_index + 1,
                        "thought": thought_payload,
                        "raw_action": raw_action,
                        "error": "empty_tool_calls",
                    }
                )
                rollout["done"] = True
                continue

            tool_call_signature = json_compact(planned_calls)
            emit_progress(
                "IMAGINED_ACTION",
                imagined_step=imagined_index + 1,
                rollout_index=rollout["rollout_index"],
                tool_calls=preview_tool_calls(planned_calls),
                signature=tool_call_signature,
            )
            repeated_tool_call_count = rollout["seen_signatures"].get(tool_call_signature, 0) + 1
            rollout["seen_signatures"][tool_call_signature] = repeated_tool_call_count
            pending_predictions.append(
                {
                    "rollout": rollout,
                    "thought": thought_payload,
                    "planned_calls": planned_calls,
                    "repeated_tool_call_count": repeated_tool_call_count,
                }
            )

        # --- Phase C: world-model predictions (batched across pending rollouts) ------
        if observation_source == "world_model" and pending_predictions:
            if hasattr(world_model_generator, "predict_feedback"):
                # JEPA-style local generators expose their own prediction interface;
                # keep the existing per-rollout call (already a cheap local forward).
                for pending in pending_predictions:
                    rollout = pending["rollout"]
                    pending["feedbacks"] = predict_world_model_feedback(
                        world_model_generator,
                        task,
                        rollout["state"],
                        pending["planned_calls"],
                        interaction_index=start_interaction_index + imagined_index,
                        world_model_target=world_model_target,
                        include_error_message_in_target=include_error_message_in_target,
                        include_stage_in_target=include_stage_in_target,
                        include_world_model_history=include_world_model_history,
                        state_history=rollout["state_history"],
                        input_history=rollout["input_history"],
                        system_prompt_max_chars=system_prompt_max_chars,
                        action_max_chars=action_max_chars,
                    )
            else:
                predictions = generate_many(
                    world_model_generator,
                    [
                        build_world_model_prediction_messages(
                            task=task,
                            previous_state=pending["rollout"]["state"],
                            predicted_calls=pending["planned_calls"],
                            interaction_index=start_interaction_index + imagined_index,
                            world_model_target=world_model_target,
                            include_error_message_in_target=include_error_message_in_target,
                            include_stage_in_target=include_stage_in_target,
                            include_world_model_history=include_world_model_history,
                            state_history=pending["rollout"]["state_history"],
                            input_history=pending["rollout"]["input_history"],
                            system_prompt_max_chars=system_prompt_max_chars,
                            action_max_chars=action_max_chars,
                        )
                        for pending in pending_predictions
                    ],
                    [0.0] * len(pending_predictions),
                )
                for pending, prediction in zip(pending_predictions, predictions):
                    pending["feedbacks"] = finish_world_model_text_feedback(
                        prediction,
                        predicted_calls=pending["planned_calls"],
                        interaction_index=start_interaction_index + imagined_index,
                        world_model_target=world_model_target,
                    )

        # --- Phase D: fold predictions back into each rollout -------------------------
        for pending in pending_predictions:
            rollout = pending["rollout"]
            planned_calls = pending["planned_calls"]
            repeated_tool_call_count = pending["repeated_tool_call_count"]
            feedbacks: list[dict[str, Any]] = pending.get("feedbacks") or []
            predicted_state = None
            if observation_source == "world_model":
                predicted_state = feedbacks[-1].get("predicted_state") if feedbacks else None
                rollout["input_history"] = append_world_model_input_history(
                    rollout["input_history"],
                    make_imagined_world_model_history_entry(
                        imagined_step=imagined_index + 1,
                        action=planned_calls,
                        state=predicted_state,
                    ),
                )
            rollout["steps"].append(
                {
                    "imagined_step": imagined_index + 1,
                    "thought": pending["thought"],
                    "tool_calls": planned_calls,
                    "repeated_tool_call_loop": repeated_tool_call_count > 1,
                    "repeated_tool_call_count": repeated_tool_call_count,
                    "predicted_feedback": feedbacks,
                    "predicted_state": full_world_model_state_for_agent(predicted_state),
                    "predicted_state_summary": (
                        summarize_state_for_planning(predicted_state) if predicted_state else None
                    ),
                    "raw_world_model_prediction": (
                        feedbacks[-1].get("raw_prediction") if feedbacks else None
                    ),
                    "observation_source": observation_source,
                }
            )
            if repeated_tool_call_count > 1:
                rollout["done"] = True
                continue
            append_imagined_observation_for_planning(
                rollout["conversation"],
                planned_calls,
                feedbacks,
                predicted_state,
                observation_source,
            )
            if observation_source == "world_model" and predicted_state is not None:
                rollout["state"] = predicted_state
                rollout["state_history"] = append_state_history(
                    rollout["state_history"], predicted_state, max_items=state_history_size
                )
                if state_is_finished(predicted_state):
                    rollout["done"] = True

    return [rollout["steps"] for rollout in rollouts]


def append_imagined_observation_for_planning(
    imagined_conversation: list[dict[str, Any]],
    planned_calls: list[dict[str, Any]],
    feedbacks: list[dict[str, Any]],
    predicted_state: Any,
    observation_source: str,
) -> None:
    imagined_conversation.append({"role": "assistant", "tool_calls": to_openai_tool_calls(planned_calls)})
    if observation_source != "world_model":
        imagined_conversation.append(
            {
                "role": "user",
                "content": (
                    "[IMAGINED_LOOKAHEAD_WITHOUT_WORLD_MODEL]\n"
                    "No imagined tool result is available for the previous step. "
                    "Continue planning the next likely step using only the task requirements, "
                    "the prior conversation, and the previously imagined tool calls."
                ),
            }
        )
        return

    last_feedback = feedbacks[-1] if feedbacks else {}
    predicted_tool_output_text = (last_feedback.get("predicted_tool_output") or "").strip()
    if predicted_tool_output_text:
        imagined_conversation.append(
            {
                "role": "tool",
                "name": planned_calls[0].get("name", "") if planned_calls else "",
                "content": "[IMAGINED_TOOL_OUTPUT_FROM_WORLD_MODEL]\n" + predicted_tool_output_text,
            }
        )
        return

    imagined_conversation.append(
        {
            "role": "user",
            "content": (
                "Imagined observation based on world-model prediction:\n"
                + json.dumps(
                    world_model_state_observation_payload(
                        predicted_state,
                        last_feedback.get("predicted_error_message"),
                        last_feedback.get("raw_prediction"),
                    ),
                    ensure_ascii=False,
                )
            ),
        }
    )


def imagined_stage_value(state: Any) -> Any:
    if state is None:
        return None
    return state_current_stage(state)


def score_topk_imagined_step(
    *,
    previous_state: Any,
    predicted_state: Any,
    feedbacks: list[dict[str, Any]],
    repeated_tool_call_count: int,
    final_answer: bool = False,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if final_answer:
        if state_is_finished(previous_state):
            return 1.0, ["final_answer_after_finished_stage=1"]
        return 0.0, ["final_answer_but_stage_remains=0"]

    if repeated_tool_call_count > 1:
        return 0.0, ["same_tool_names_and_arguments=0"]

    score = 0.0
    if feedbacks and all(bool(feedback.get("predicted_success")) for feedback in feedbacks):
        score += 1.0
        reasons.append("success=1")
    else:
        reasons.append("failure=0")

    previous_stage = imagined_stage_value(previous_state)
    next_stage = imagined_stage_value(predicted_state)
    if next_stage is None and feedbacks:
        next_stage = feedbacks[-1].get("predicted_current_stage")
    if next_stage is not None and next_stage != previous_stage:
        score += 1.0
        reasons.append("current_stage_change=1")
    else:
        reasons.append("current_stage_unchanged=0")
    return score, reasons


def make_topk_rollout_record(
    index: int,
    branch: dict[str, Any],
    *,
    observation_source: str,
) -> dict[str, Any]:
    return {
        "rollout_index": index,
        "rollout_temperature": branch.get("rollout_temperature"),
        "observation_source": observation_source,
        "selection_strategy": "topk_search",
        "score": branch.get("score", 0.0),
        "depth": branch.get("depth", 0),
        "terminal": bool(branch.get("terminal")),
        "imagined_steps": branch.get("imagined_steps") or [],
        "terminal_state": full_world_model_state_for_agent(branch.get("state")),
        "terminal_state_summary": summarize_state_for_planning(branch.get("state")),
    }


def imagine_trajectory_topk_search(
    agent_generator: Any,
    world_model_generator: Any,
    task: TaskTrajectory,
    conversation: list[dict[str, Any]],
    previous_state: dict[str, Any],
    react_system_prompt: str,
    max_imagined_steps: int,
    world_model_target: str,
    include_error_message_in_target: bool = False,
    include_stage_in_target: bool = False,
    include_world_model_history: bool = False,
    start_interaction_index: int = 0,
    candidate_action_count: int = 3,
    top_k: int = 3,
    rollout_temperature: float = 0.7,
    observation_source: str = "world_model",
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
    single_call_candidate_proposal: bool = SINGLE_CALL_CANDIDATE_PROPOSAL_DEFAULT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidate_action_count = max(1, int(candidate_action_count))
    top_k = max(1, int(top_k))
    imagined_react_system_prompt = imagined_trajectory_system_prompt(react_system_prompt)
    root_state = sanitize_state_content(previous_state)
    beams: list[dict[str, Any]] = [
        {
            "branch_id": "0",
            "parent_id": None,
            "depth": 0,
            "conversation": [dict(message) for message in conversation],
            "state": root_state,
            "state_history": append_state_history(
                list(state_history or []), root_state, max_items=state_history_size
            ),
            "input_history": list(input_history or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:],
            "imagined_steps": [],
            "seen_signatures": {},
            "score": 0.0,
            "terminal": False,
            "rollout_temperature": 0.0,
        }
    ]
    all_partial_rollouts: list[dict[str, Any]] = []
    branch_counter = 1

    for imagined_index in range(max(0, int(max_imagined_steps))):
        expanded: list[dict[str, Any]] = []
        for branch in beams:
            if branch.get("terminal"):
                expanded.append(branch)
                continue

            branch_conversation = [dict(message) for message in branch["conversation"]]
            raw_thought = _generate_with_optional_temperature(
                agent_generator,
                build_react_think_messages(
                    branch_conversation,
                    current_query=task.user_messages[-1] if task.user_messages else "",
                    system_prompt=imagined_react_system_prompt,
                ),
                temperature=0.0,
            )
            raw_thought = strip_model_thinking_output(raw_thought)
            thought_payload = parse_thought_payload(raw_thought)
            thought_conversation = branch_conversation + [
                {"role": "assistant", "content": json.dumps(thought_payload, ensure_ascii=False)}
            ]

            seen_candidate_signatures: set[str] = set()
            current_query = task.user_messages[-1] if task.user_messages else ""
            if single_call_candidate_proposal and candidate_action_count > 1:
                # One generation yields all K candidates instead of K sequential calls.
                proposal_temperature = rollout_temperature
                raw_candidates = _generate_with_optional_temperature(
                    agent_generator,
                    build_react_candidate_actions_messages(
                        thought_conversation,
                        current_query=current_query,
                        system_prompt=imagined_react_system_prompt,
                        candidate_count=candidate_action_count,
                    ),
                    temperature=proposal_temperature,
                )
                candidate_raw_actions = parse_candidate_actions(
                    strip_model_thinking_output(raw_candidates), candidate_action_count
                )
                if not candidate_raw_actions:
                    # Degrade gracefully to one greedy single-action generation.
                    candidate_raw_actions = [
                        strip_model_thinking_output(
                            _generate_with_optional_temperature(
                                agent_generator,
                                build_react_action_messages(
                                    thought_conversation,
                                    current_query=current_query,
                                    system_prompt=imagined_react_system_prompt,
                                ),
                                temperature=0.0,
                            )
                        )
                    ]
                candidate_temperatures = [proposal_temperature] * len(candidate_raw_actions)
            else:
                candidate_raw_actions = []
                candidate_temperatures = []
                for candidate_index in range(candidate_action_count):
                    temperature = 0.0 if candidate_index == 0 else rollout_temperature
                    raw_action = strip_model_thinking_output(
                        _generate_with_optional_temperature(
                            agent_generator,
                            build_react_action_messages(
                                thought_conversation,
                                current_query=current_query,
                                system_prompt=imagined_react_system_prompt,
                            ),
                            temperature=temperature,
                        )
                    )
                    candidate_raw_actions.append(raw_action)
                    candidate_temperatures.append(temperature)

            for candidate_index, (raw_action, temperature) in enumerate(
                zip(candidate_raw_actions, candidate_temperatures)
            ):
                child_id = f"{branch['branch_id']}.{candidate_index}"
                base_step = {
                    "imagined_step": imagined_index + 1,
                    "topk_branch_id": child_id,
                    "topk_parent_id": branch.get("branch_id"),
                    "topk_candidate_index": candidate_index,
                    "thought": thought_payload,
                    "raw_action": raw_action,
                }

                try:
                    decision = parse_agent_decision(raw_action)
                except Exception as exc:
                    step_score, score_reasons = 0.0, [f"parse_error=0:{exc}"]
                    step = {
                        **base_step,
                        "parse_error": str(exc),
                        "topk_score_delta": step_score,
                        "topk_score_reasons": score_reasons,
                        "topk_total_score": branch["score"] + step_score,
                    }
                    expanded.append(
                        {
                            **branch,
                            "branch_id": child_id,
                            "parent_id": branch.get("branch_id"),
                            "depth": imagined_index + 1,
                            "imagined_steps": branch["imagined_steps"] + [step],
                            "score": branch["score"] + step_score,
                            "terminal": True,
                            "rollout_temperature": temperature,
                        }
                    )
                    continue

                if "final_answer" in decision:
                    step_score, score_reasons = score_topk_imagined_step(
                        previous_state=branch["state"],
                        predicted_state=branch["state"],
                        feedbacks=[],
                        repeated_tool_call_count=0,
                        final_answer=True,
                    )
                    step = {
                        **base_step,
                        "final_answer": decision["final_answer"],
                        "predicted_state": full_world_model_state_for_agent(branch["state"]),
                        "predicted_state_summary": summarize_state_for_planning(branch["state"]),
                        "topk_score_delta": step_score,
                        "topk_score_reasons": score_reasons,
                        "topk_total_score": branch["score"] + step_score,
                    }
                    expanded.append(
                        {
                            **branch,
                            "branch_id": child_id,
                            "parent_id": branch.get("branch_id"),
                            "depth": imagined_index + 1,
                            "imagined_steps": branch["imagined_steps"] + [step],
                            "score": branch["score"] + step_score,
                            "terminal": True,
                            "rollout_temperature": temperature,
                        }
                    )
                    continue

                if "clarify" in decision:
                    step = {
                        **base_step,
                        "clarify": decision["clarify"],
                        "predicted_state": full_world_model_state_for_agent(branch["state"]),
                        "predicted_state_summary": summarize_state_for_planning(branch["state"]),
                        "topk_score_delta": 0.0,
                        "topk_score_reasons": ["clarify=0"],
                        "topk_total_score": branch["score"],
                    }
                    expanded.append(
                        {
                            **branch,
                            "branch_id": child_id,
                            "parent_id": branch.get("branch_id"),
                            "depth": imagined_index + 1,
                            "imagined_steps": branch["imagined_steps"] + [step],
                            "terminal": True,
                            "rollout_temperature": temperature,
                        }
                    )
                    continue

                planned_calls = [normalize_tool_call(call) for call in decision.get("tool_calls", [])]
                if not planned_calls:
                    step = {
                        **base_step,
                        "error": "empty_tool_calls",
                        "topk_score_delta": 0.0,
                        "topk_score_reasons": ["empty_tool_calls=0"],
                        "topk_total_score": branch["score"],
                    }
                    expanded.append(
                        {
                            **branch,
                            "branch_id": child_id,
                            "parent_id": branch.get("branch_id"),
                            "depth": imagined_index + 1,
                            "imagined_steps": branch["imagined_steps"] + [step],
                            "terminal": True,
                            "rollout_temperature": temperature,
                        }
                    )
                    continue

                signature = json_compact(planned_calls)
                local_duplicate_candidate = signature in seen_candidate_signatures
                seen_candidate_signatures.add(signature)
                seen_signatures = dict(branch["seen_signatures"])
                repeated_tool_call_count = seen_signatures.get(signature, 0) + 1
                seen_signatures[signature] = repeated_tool_call_count
                if local_duplicate_candidate:
                    repeated_tool_call_count = max(repeated_tool_call_count, 2)

                feedbacks: list[dict[str, Any]] = []
                predicted_state = branch["state"]
                if observation_source == "world_model":
                    feedbacks = predict_world_model_feedback(
                        world_model_generator,
                        task,
                        branch["state"],
                        planned_calls,
                        interaction_index=start_interaction_index + imagined_index,
                        world_model_target=world_model_target,
                        include_error_message_in_target=include_error_message_in_target,
                        include_stage_in_target=include_stage_in_target,
                        include_world_model_history=include_world_model_history,
                        state_history=branch["state_history"],
                        input_history=branch["input_history"],
                        system_prompt_max_chars=system_prompt_max_chars,
                        action_max_chars=action_max_chars,
                    )
                    predicted_state = feedbacks[-1].get("predicted_state") if feedbacks else branch["state"]

                step_score, score_reasons = score_topk_imagined_step(
                    previous_state=branch["state"],
                    predicted_state=predicted_state,
                    feedbacks=feedbacks,
                    repeated_tool_call_count=repeated_tool_call_count,
                )
                next_score = branch["score"] + step_score
                next_input_history = append_world_model_input_history(
                    branch["input_history"],
                    make_imagined_world_model_history_entry(
                        imagined_step=imagined_index + 1,
                        action=planned_calls,
                        state=predicted_state,
                    ),
                )
                next_conversation = [dict(message) for message in thought_conversation]
                append_imagined_observation_for_planning(
                    next_conversation,
                    planned_calls,
                    feedbacks,
                    predicted_state,
                    observation_source,
                )
                next_state_history = branch["state_history"]
                if predicted_state is not None:
                    next_state_history = append_state_history(
                        next_state_history, predicted_state, max_items=state_history_size
                    )
                terminal = repeated_tool_call_count > 1 or state_is_finished(predicted_state)
                step = {
                    **base_step,
                    "tool_calls": planned_calls,
                    "repeated_tool_call_loop": repeated_tool_call_count > 1,
                    "repeated_tool_call_count": repeated_tool_call_count,
                    "predicted_feedback": feedbacks,
                    "predicted_state": full_world_model_state_for_agent(predicted_state),
                    "predicted_state_summary": (
                        summarize_state_for_planning(predicted_state) if predicted_state else None
                    ),
                    "raw_world_model_prediction": feedbacks[-1].get("raw_prediction") if feedbacks else None,
                    "observation_source": observation_source,
                    "topk_score_delta": step_score,
                    "topk_score_reasons": score_reasons,
                    "topk_total_score": next_score,
                }
                expanded.append(
                    {
                        "branch_id": child_id or str(branch_counter),
                        "parent_id": branch.get("branch_id"),
                        "depth": imagined_index + 1,
                        "conversation": next_conversation,
                        "state": predicted_state,
                        "state_history": next_state_history,
                        "input_history": next_input_history,
                        "imagined_steps": branch["imagined_steps"] + [step],
                        "seen_signatures": seen_signatures,
                        "score": next_score,
                        "terminal": terminal,
                        "rollout_temperature": temperature,
                    }
                )
                branch_counter += 1

        if not expanded:
            break
        expanded.sort(
            key=lambda item: (
                float(item.get("score", 0.0)),
                int(item.get("depth", 0)),
                0 if item.get("terminal") else 1,
            ),
            reverse=True,
        )
        beams = expanded[:top_k]
        all_partial_rollouts.extend(
            make_topk_rollout_record(
                len(all_partial_rollouts) + index,
                branch,
                observation_source=observation_source,
            )
            for index, branch in enumerate(beams)
        )
        if all(branch.get("terminal") for branch in beams):
            break

    final_rollouts = [
        make_topk_rollout_record(index, branch, observation_source=observation_source)
        for index, branch in enumerate(beams)
    ]
    if not final_rollouts:
        selection = {
            "selected_index": 0,
            "scores": [],
            "comments": "topk_search produced no candidates",
            "fallback_used": True,
            "selection_strategy": "topk_search",
            "candidate_action_count": candidate_action_count,
            "top_k": top_k,
        }
        return [], [], selection

    scores = [
        {
            "index": index,
            "score": rollout.get("score", 0.0),
            "reason": " + ".join(
                str(reason)
                for step in rollout.get("imagined_steps", [])
                for reason in step.get("topk_score_reasons", [])
            ),
        }
        for index, rollout in enumerate(final_rollouts)
    ]
    selection = {
        "selected_index": 0,
        "scores": scores,
        "comments": "Selected highest-scoring top-k imagined trajectory",
        "fallback_used": False,
        "selection_strategy": "topk_search",
        "candidate_action_count": candidate_action_count,
        "top_k": top_k,
        "partial_frontier_records": all_partial_rollouts,
    }
    return final_rollouts[0]["imagined_steps"], final_rollouts, selection


def imagine_trajectories_open_loop(
    agent_generator: Any,
    world_model_generator: Any,
    task: TaskTrajectory,
    conversation: list[dict[str, Any]],
    previous_state: dict[str, Any],
    react_system_prompt: str,
    max_imagined_steps: int,
    world_model_target: str,
    num_rollouts: int,
    rollout_temperature: float = 0.7,
    include_error_message_in_target: bool = False,
    include_stage_in_target: bool = False,
    include_world_model_history: bool = False,
    start_interaction_index: int = 0,
    observation_source: str = "world_model",
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> list[list[dict[str, Any]]] | None:
    """Beam_plan-style open-loop rollouts for the text world model.

    ONE agent call proposes all N plans (up to S actions each, symbolic "$stepK.field"
    references for unknown values); the agent never sees a predicted state. The world
    model then rolls each plan's state chain forward -- state k feeds the step-k+1
    prompt within a plan, and step k's predictions are batched ACROSS plans via
    generate_many, mirroring beam_plan's "one batched WM pass per horizon step".

    Cost: 1 agent generation + max_steps batched WM generations (0 WM generations when
    observation_source != "world_model"). Returns one step list per plan in the same
    shape imagine_trajectories_lockstep produces, or None when no plan could be parsed
    (the caller falls back to closed-loop lockstep).
    """
    imagined_react_system_prompt = imagined_trajectory_system_prompt(react_system_prompt)
    current_query = task.user_messages[-1] if task.user_messages else ""
    num_rollouts = max(1, int(num_rollouts))

    # k samples of a ONE-plan prompt, ideally in a single request (n=k / num_return_sequences):
    # the shared prefix is prefilled once and the k plans decode concurrently. One response
    # listing k plans would be the same tokens on a single serial decode stream.
    plan_steps_cap = max(1, int(max_imagined_steps))
    sample_temperature = rollout_temperature if num_rollouts > 1 else 0.0
    raw_samples, agent_requests = sample_many(
        agent_generator,
        build_react_open_loop_plan_messages(
            conversation,
            current_query=current_query,
            system_prompt=imagined_react_system_prompt,
            max_steps=plan_steps_cap,
        ),
        temperature=sample_temperature,
        num_samples=num_rollouts,
    )
    plans: list[dict[str, Any]] = []
    seen_plan_signatures: set[str] = set()
    unparsed = 0
    for raw in raw_samples:
        parsed = parse_open_loop_plans(strip_model_thinking_output(raw), 1, plan_steps_cap)
        if not parsed:
            unparsed += 1
            continue
        plan = parsed[0]
        # Sampling can repeat itself; a duplicate plan would only re-spend world-model compute
        # on an answer already in the candidate set.
        signature = json_compact(plan["steps"])
        if signature in seen_plan_signatures:
            continue
        seen_plan_signatures.add(signature)
        plans.append(plan)
    if not plans:
        emit_progress(
            "IMAGINED_OPEN_LOOP_PARSE_ERROR",
            samples=len(raw_samples),
            agent_requests=agent_requests,
            raw_plans=(raw_samples[0][:500] if raw_samples else ""),
        )
        return None
    emit_progress(
        "IMAGINED_OPEN_LOOP_PLANS",
        plan_count=len(plans),
        samples_requested=num_rollouts,
        agent_requests=agent_requests,
        duplicate_samples=len(raw_samples) - len(plans) - unparsed,
        unparsed_samples=unparsed,
        plan_lengths=[len(plan["steps"]) for plan in plans],
        strategies=[plan["strategy"] for plan in plans],
    )

    root_state = sanitize_state_content(previous_state)
    rollouts: list[dict[str, Any]] = []
    for plan_index, plan in enumerate(plans):
        rollouts.append(
            {
                "rollout_index": plan_index,
                "plan": plan,
                "state": root_state,
                "state_history": append_state_history(
                    list(state_history or []), root_state, max_items=state_history_size
                ),
                "input_history": list(input_history or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:],
                "steps": [],
                "seen_signatures": {},
                "done": False,
            }
        )

    for imagined_index in range(max(1, int(max_imagined_steps))):
        pending_predictions: list[dict[str, Any]] = []
        for rollout in rollouts:
            if rollout["done"] or imagined_index >= len(rollout["plan"]["steps"]):
                rollout["done"] = True
                continue
            decision = rollout["plan"]["steps"][imagined_index]
            # The plan-level strategy doubles as the first step's thought so the judge and
            # the injected planning message keep their usual shape.
            thought_payload = (
                {"thought": rollout["plan"]["strategy"]} if imagined_index == 0 else {"thought": ""}
            )

            if "final_answer" in decision:
                emit_progress(
                    "IMAGINED_FINAL_ANSWER",
                    imagined_step=imagined_index + 1,
                    rollout_index=rollout["rollout_index"],
                    final_answer=decision["final_answer"],
                )
                rollout["steps"].append(
                    {
                        "imagined_step": imagined_index + 1,
                        "thought": thought_payload,
                        "final_answer": decision["final_answer"],
                        "predicted_state": full_world_model_state_for_agent(rollout["state"]),
                        "predicted_state_summary": summarize_state_for_planning(rollout["state"]),
                        "open_loop": True,
                    }
                )
                rollout["done"] = True
                continue

            planned_calls = [normalize_tool_call(call) for call in decision.get("tool_calls", [])]
            if not planned_calls:
                rollout["done"] = True
                continue
            tool_call_signature = json_compact(planned_calls)
            emit_progress(
                "IMAGINED_ACTION",
                imagined_step=imagined_index + 1,
                rollout_index=rollout["rollout_index"],
                tool_calls=preview_tool_calls(planned_calls),
                signature=tool_call_signature,
            )
            repeated_tool_call_count = rollout["seen_signatures"].get(tool_call_signature, 0) + 1
            rollout["seen_signatures"][tool_call_signature] = repeated_tool_call_count
            pending_predictions.append(
                {
                    "rollout": rollout,
                    "thought": thought_payload,
                    "planned_calls": planned_calls,
                    "repeated_tool_call_count": repeated_tool_call_count,
                }
            )

        if not pending_predictions:
            break

        # One batched WM pass for this horizon step across every still-active plan.
        if observation_source == "world_model":
            if hasattr(world_model_generator, "predict_feedback"):
                for pending in pending_predictions:
                    rollout = pending["rollout"]
                    pending["feedbacks"] = predict_world_model_feedback(
                        world_model_generator,
                        task,
                        rollout["state"],
                        pending["planned_calls"],
                        interaction_index=start_interaction_index + imagined_index,
                        world_model_target=world_model_target,
                        include_error_message_in_target=include_error_message_in_target,
                        include_stage_in_target=include_stage_in_target,
                        include_world_model_history=include_world_model_history,
                        state_history=rollout["state_history"],
                        input_history=rollout["input_history"],
                        system_prompt_max_chars=system_prompt_max_chars,
                        action_max_chars=action_max_chars,
                    )
            else:
                predictions = generate_many(
                    world_model_generator,
                    [
                        build_world_model_prediction_messages(
                            task=task,
                            previous_state=pending["rollout"]["state"],
                            predicted_calls=pending["planned_calls"],
                            interaction_index=start_interaction_index + imagined_index,
                            world_model_target=world_model_target,
                            include_error_message_in_target=include_error_message_in_target,
                            include_stage_in_target=include_stage_in_target,
                            include_world_model_history=include_world_model_history,
                            state_history=pending["rollout"]["state_history"],
                            input_history=pending["rollout"]["input_history"],
                            system_prompt_max_chars=system_prompt_max_chars,
                            action_max_chars=action_max_chars,
                        )
                        for pending in pending_predictions
                    ],
                    [0.0] * len(pending_predictions),
                )
                for pending, prediction in zip(pending_predictions, predictions):
                    pending["feedbacks"] = finish_world_model_text_feedback(
                        prediction,
                        predicted_calls=pending["planned_calls"],
                        interaction_index=start_interaction_index + imagined_index,
                        world_model_target=world_model_target,
                    )

        for pending in pending_predictions:
            rollout = pending["rollout"]
            planned_calls = pending["planned_calls"]
            repeated_tool_call_count = pending["repeated_tool_call_count"]
            feedbacks: list[dict[str, Any]] = pending.get("feedbacks") or []
            predicted_state = None
            if observation_source == "world_model":
                predicted_state = feedbacks[-1].get("predicted_state") if feedbacks else None
                rollout["input_history"] = append_world_model_input_history(
                    rollout["input_history"],
                    make_imagined_world_model_history_entry(
                        imagined_step=imagined_index + 1,
                        action=planned_calls,
                        state=predicted_state,
                    ),
                )
            rollout["steps"].append(
                {
                    "imagined_step": imagined_index + 1,
                    "thought": pending["thought"],
                    "tool_calls": planned_calls,
                    "repeated_tool_call_loop": repeated_tool_call_count > 1,
                    "repeated_tool_call_count": repeated_tool_call_count,
                    "predicted_feedback": feedbacks,
                    "predicted_state": full_world_model_state_for_agent(predicted_state),
                    "predicted_state_summary": (
                        summarize_state_for_planning(predicted_state) if predicted_state else None
                    ),
                    "raw_world_model_prediction": (
                        feedbacks[-1].get("raw_prediction") if feedbacks else None
                    ),
                    "observation_source": observation_source,
                    "open_loop": True,
                }
            )
            if repeated_tool_call_count > 1:
                rollout["done"] = True
                continue
            if observation_source == "world_model" and predicted_state is not None:
                rollout["state"] = predicted_state
                rollout["state_history"] = append_state_history(
                    rollout["state_history"], predicted_state, max_items=state_history_size
                )
                if state_is_finished(predicted_state):
                    rollout["done"] = True

    return [rollout["steps"] for rollout in rollouts]

def imagine_trajectory_candidates(
    agent_generator: Any,
    world_model_generator: Any,
    task: TaskTrajectory,
    conversation: list[dict[str, Any]],
    previous_state: dict[str, Any],
    react_system_prompt: str,
    max_imagined_steps: int,
    world_model_target: str,
    include_error_message_in_target: bool = False,
    include_stage_in_target: bool = False,
    include_world_model_history: bool = False,
    start_interaction_index: int = 0,
    num_rollouts: int = 1,
    rollout_temperature: float = 0.7,
    selection_strategy: str = "llm_judge",
    observation_source: str = "world_model",
    candidate_action_count: int = 3,
    top_k: int = 3,
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if selection_strategy == "topk_search":
        return imagine_trajectory_topk_search(
            agent_generator=agent_generator,
            world_model_generator=world_model_generator,
            task=task,
            conversation=conversation,
            previous_state=previous_state,
            react_system_prompt=react_system_prompt,
            max_imagined_steps=max_imagined_steps,
            world_model_target=world_model_target,
            include_error_message_in_target=include_error_message_in_target,
            include_stage_in_target=include_stage_in_target,
            include_world_model_history=include_world_model_history,
            start_interaction_index=start_interaction_index,
            candidate_action_count=candidate_action_count,
            top_k=top_k,
            rollout_temperature=rollout_temperature,
            observation_source=observation_source,
            state_history=state_history,
            input_history=input_history,
            state_history_size=state_history_size,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
        )

    rollout_count = max(num_rollouts, 1)
    rollout_temperature_plan = [
        0.0 if rollout_index == 0 else rollout_temperature
        for rollout_index in range(rollout_count)
    ]
    all_imagined_steps: list[list[dict[str, Any]]] | None = None
    if IMAGINED_ROLLOUT_MODE == "open_loop":
        # One agent call proposes every rollout's full action sequence; the WM fills in
        # the states afterwards (batched across plans per step). Falls through to the
        # closed-loop path below when the plan payload cannot be parsed.
        all_imagined_steps = imagine_trajectories_open_loop(
            agent_generator=agent_generator,
            world_model_generator=world_model_generator,
            task=task,
            conversation=conversation,
            previous_state=previous_state,
            react_system_prompt=react_system_prompt,
            max_imagined_steps=max_imagined_steps,
            world_model_target=world_model_target,
            num_rollouts=rollout_count,
            rollout_temperature=rollout_temperature,
            include_error_message_in_target=include_error_message_in_target,
            include_stage_in_target=include_stage_in_target,
            include_world_model_history=include_world_model_history,
            start_interaction_index=start_interaction_index,
            observation_source=observation_source,
            state_history=state_history,
            input_history=input_history,
            state_history_size=state_history_size,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
        )
        if all_imagined_steps is not None:
            # The agent may return fewer usable plans than requested; pad the bookkeeping.
            rollout_temperature_plan = rollout_temperature_plan[: len(all_imagined_steps)] or [0.0]
    if all_imagined_steps is not None:
        pass
    elif IMAGINED_PARALLEL_ROLLOUTS:
        # All rollouts advance in lockstep; each step's agent and world-model generations
        # are batched across the alive rollouts (see imagine_trajectories_lockstep).
        all_imagined_steps = imagine_trajectories_lockstep(
            agent_generator=agent_generator,
            world_model_generator=world_model_generator,
            task=task,
            conversation=conversation,
            previous_state=previous_state,
            react_system_prompt=react_system_prompt,
            max_imagined_steps=max_imagined_steps,
            world_model_target=world_model_target,
            rollout_temperatures=rollout_temperature_plan,
            include_error_message_in_target=include_error_message_in_target,
            include_stage_in_target=include_stage_in_target,
            include_world_model_history=include_world_model_history,
            start_interaction_index=start_interaction_index,
            observation_source=observation_source,
            state_history=state_history,
            input_history=input_history,
            state_history_size=state_history_size,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
        )
    else:
        all_imagined_steps = [
            imagine_trajectory(
                agent_generator=agent_generator,
                world_model_generator=world_model_generator,
                task=task,
                conversation=conversation,
                previous_state=previous_state,
                react_system_prompt=react_system_prompt,
                max_imagined_steps=max_imagined_steps,
                world_model_target=world_model_target,
                include_error_message_in_target=include_error_message_in_target,
                include_stage_in_target=include_stage_in_target,
                include_world_model_history=include_world_model_history,
                start_interaction_index=start_interaction_index,
                rollout_index=rollout_index,
                rollout_temperature=temperature,
                observation_source=observation_source,
                state_history=state_history,
                input_history=input_history,
                state_history_size=state_history_size,
                system_prompt_max_chars=system_prompt_max_chars,
                action_max_chars=action_max_chars,
            )
            for rollout_index, temperature in enumerate(rollout_temperature_plan)
        ]
    candidate_rollouts = [
        {
            "rollout_index": rollout_index,
            "rollout_temperature": temperature,
            "observation_source": observation_source,
            "imagined_steps": imagined_steps,
        }
        for rollout_index, (temperature, imagined_steps) in enumerate(
            zip(rollout_temperature_plan, all_imagined_steps)
        )
    ]

    if len(candidate_rollouts) == 1 or selection_strategy == "first":
        selection = {
            "selected_index": 0,
            "scores": [],
            "comments": "Selected first imagined trajectory",
            "fallback_used": False,
            "selection_strategy": selection_strategy,
        }
        return (
            candidate_rollouts[0]["imagined_steps"],
            candidate_rollouts,
            selection,
        )

    selection = select_imagined_trajectory_with_llm_judge(
        agent_generator=agent_generator,
        task=task,
        conversation=conversation,
        candidate_rollouts=candidate_rollouts,
    )
    selection["selection_strategy"] = selection_strategy
    selected_index = int(selection.get("selected_index", 0))
    return (
        candidate_rollouts[selected_index]["imagined_steps"],
        candidate_rollouts,
        selection,
    )


def imagine_revision_rollout(
    agent_generator: Any,
    world_model_generator: Any,
    task: TaskTrajectory,
    conversation: list[dict[str, Any]],
    previous_state: dict[str, Any],
    react_system_prompt: str,
    initial_planned_calls: list[dict[str, Any]],
    lookahead_steps: int,
    world_model_target: str,
    include_error_message_in_target: bool,
    include_stage_in_target: bool,
    include_world_model_history: bool,
    start_interaction_index: int,
    rollout_index: int,
    rollout_temperature: float,
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> dict[str, Any]:
    """Roll out `lookahead_steps` ahead, starting from `initial_planned_calls`.

    Step 1 is the agent's already-decided `initial_planned_calls`. Steps 2..K
    are imagined: the agent picks the next tool calls based on the world
    model's prediction of the previous step. The first step's feedbacks are
    returned separately so the caller can keep the existing single-step
    revision telemetry intact.
    """
    imagined_conversation = [dict(message) for message in conversation]
    imagined_react_system_prompt = imagined_trajectory_system_prompt(react_system_prompt)
    imagined_state = sanitize_state_content(previous_state)
    imagined_state_history = append_state_history(
        list(state_history or []), imagined_state, max_items=state_history_size
    )
    imagined_input_history = list(input_history or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:]
    steps: list[dict[str, Any]] = []
    first_step_feedbacks: list[dict[str, Any]] = []
    seen_signatures: dict[str, int] = {}

    current_planned_calls = initial_planned_calls
    for step_offset in range(max(lookahead_steps, 1)):
        feedbacks = predict_world_model_feedback(
            world_model_generator,
            task,
            imagined_state,
            current_planned_calls,
            interaction_index=start_interaction_index + step_offset,
            world_model_target=world_model_target,
            include_error_message_in_target=include_error_message_in_target,
            include_stage_in_target=include_stage_in_target,
            include_world_model_history=include_world_model_history,
            state_history=imagined_state_history,
            input_history=imagined_input_history,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
        )
        if step_offset == 0:
            first_step_feedbacks = feedbacks

        last_feedback = feedbacks[-1] if feedbacks else {}
        predicted_state = last_feedback.get("predicted_state")
        predicted_tool_output = (last_feedback.get("predicted_tool_output") or "").strip()
        predicted_error_message = (last_feedback.get("predicted_error_message") or "").strip()
        imagined_input_history = append_world_model_input_history(
            imagined_input_history,
            make_imagined_world_model_history_entry(
                imagined_step=step_offset + 1,
                action=current_planned_calls,
                state=predicted_state,
            ),
        )

        steps.append(
            {
                "step_offset": step_offset,
                "tool_calls": current_planned_calls,
                "predicted_feedback": feedbacks,
                "predicted_state": full_world_model_state_for_agent(predicted_state),
                "predicted_state_summary": (
                    summarize_state_for_planning(predicted_state) if predicted_state else None
                ),
                "raw_world_model_prediction": last_feedback.get("raw_prediction"),
                "predicted_tool_output_preview": predicted_tool_output[:400] if predicted_tool_output else "",
                "predicted_error_message": predicted_error_message,
            }
        )

        imagined_conversation.append(
            {"role": "assistant", "tool_calls": to_openai_tool_calls(current_planned_calls)}
        )
        if predicted_tool_output:
            imagined_conversation.append(
                {
                    "role": "tool",
                    "name": current_planned_calls[0].get("name", "") if current_planned_calls else "",
                    "content": (
                        "[IMAGINED_TOOL_OUTPUT_FROM_WORLD_MODEL]\n" + predicted_tool_output
                    ),
                }
            )
        else:
            imagined_conversation.append(
                {
                    "role": "user",
                    "content": (
                        "Imagined observation based on world-model prediction:\n"
                        + json.dumps(
                            world_model_state_observation_payload(
                                predicted_state,
                                predicted_error_message or None,
                                last_feedback.get("raw_prediction"),
                            ),
                            ensure_ascii=False,
                        )
                    ),
                }
            )

        if predicted_state is not None:
            imagined_state = predicted_state
            imagined_state_history = append_state_history(
                imagined_state_history, imagined_state, max_items=state_history_size
            )
            if state_is_finished(predicted_state):
                break

        if step_offset + 1 >= max(lookahead_steps, 1):
            break

        # Agent picks the next imagined step based on the imagined conversation.
        try:
            raw_action = agent_generator.generate_from_messages(
                build_react_action_messages(
                    imagined_conversation,
                    current_query=task.user_messages[-1] if task.user_messages else "",
                    system_prompt=imagined_react_system_prompt,
                ),
                temperature=rollout_temperature,
            )
        except TypeError:
            # Generator does not accept temperature kwarg; fall back to default.
            raw_action = agent_generator.generate_from_messages(
                build_react_action_messages(
                    imagined_conversation,
                    current_query=task.user_messages[-1] if task.user_messages else "",
                    system_prompt=imagined_react_system_prompt,
                )
            )
        raw_action = strip_model_thinking_output(raw_action)
        try:
            decision = parse_agent_decision(raw_action)
        except Exception as exc:
            steps.append(
                {
                    "step_offset": step_offset + 1,
                    "imagined_parse_error": str(exc),
                }
            )
            break
        if "final_answer" in decision or "clarify" in decision:
            steps.append(
                {
                    "step_offset": step_offset + 1,
                    "imagined_decision": {
                        k: v for k, v in decision.items() if k in {"final_answer", "clarify"}
                    },
                }
            )
            break
        next_calls = [normalize_tool_call(call) for call in decision.get("tool_calls", [])]
        if not next_calls:
            break
        signature = json_compact(next_calls)
        seen_signatures[signature] = seen_signatures.get(signature, 0) + 1
        if seen_signatures[signature] > 1:
            steps.append(
                {
                    "step_offset": step_offset + 1,
                    "imagined_repeated_tool_call_loop": True,
                    "tool_calls": next_calls,
                }
            )
            break
        current_planned_calls = next_calls

    all_predicted_success = bool(steps) and all(
        all(fb.get("predicted_success", False) for fb in step.get("predicted_feedback", []))
        for step in steps
        if step.get("predicted_feedback")
    )
    any_predicted_failure = any(
        any(not fb.get("predicted_success", True) for fb in step.get("predicted_feedback", []))
        for step in steps
        if step.get("predicted_feedback")
    )

    return {
        "rollout_index": rollout_index,
        "rollout_temperature": rollout_temperature,
        "lookahead_steps_taken": len(steps),
        "lookahead_steps_target": lookahead_steps,
        "steps": steps,
        "first_step_feedbacks": first_step_feedbacks,
        "all_predicted_success": all_predicted_success,
        "any_predicted_failure": any_predicted_failure,
        "terminal_state": full_world_model_state_for_agent(imagined_state),
        "terminal_state_summary": (
            summarize_state_for_planning(imagined_state) if imagined_state else None
        ),
    }


def imagine_revision_rollouts(
    agent_generator: Any,
    world_model_generator: Any,
    task: TaskTrajectory,
    conversation: list[dict[str, Any]],
    previous_state: dict[str, Any],
    react_system_prompt: str,
    initial_planned_calls: list[dict[str, Any]],
    lookahead_steps: int,
    num_rollouts: int,
    world_model_target: str,
    include_error_message_in_target: bool,
    include_stage_in_target: bool,
    include_world_model_history: bool,
    start_interaction_index: int,
    rollout_temperature: float,
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> list[dict[str, Any]]:
    """Run N independent K-step rollouts of the same `initial_planned_calls`.

    The first rollout is deterministic (temperature=0); the remaining rollouts
    use `rollout_temperature` for diversity in the agent's continuation choices.
    """
    rollouts: list[dict[str, Any]] = []
    for index in range(max(num_rollouts, 1)):
        temperature = 0.0 if index == 0 else rollout_temperature
        rollouts.append(
            imagine_revision_rollout(
                agent_generator=agent_generator,
                world_model_generator=world_model_generator,
                task=task,
                conversation=conversation,
                previous_state=previous_state,
                react_system_prompt=react_system_prompt,
                initial_planned_calls=initial_planned_calls,
                lookahead_steps=lookahead_steps,
                world_model_target=world_model_target,
                include_error_message_in_target=include_error_message_in_target,
                include_stage_in_target=include_stage_in_target,
                include_world_model_history=include_world_model_history,
                start_interaction_index=start_interaction_index,
                rollout_index=index,
                rollout_temperature=temperature,
                state_history=state_history,
                input_history=input_history,
                state_history_size=state_history_size,
                system_prompt_max_chars=system_prompt_max_chars,
                action_max_chars=action_max_chars,
            )
        )
    return rollouts


def imagined_step_is_unsatisfactory(imagined_step: dict[str, Any]) -> bool:
    if imagined_step.get("repeated_tool_call_loop"):
        return True
    feedbacks = imagined_step.get("predicted_feedback") or []
    if not feedbacks:
        return True
    final_feedback = feedbacks[-1]
    if final_feedback.get("parse_error"):
        return True
    if (
        final_feedback.get("predicted_state") is None
        and final_feedback.get("predicted_current_stage") is None
        and final_feedback.get("predicted_remaining_stages") is None
        and final_feedback.get("predicted_success") is None
    ):
        return True
    return not final_feedback.get("predicted_success", False)


def update_state_from_actual_execution(
    previous_state: dict[str, Any],
    execution_results: list[dict[str, Any]],
    predicted_feedbacks: list[dict[str, Any]] | None = None,
    trust_predicted_state: bool = True,
) -> dict[str, Any]:
    predicted_state = None
    if predicted_feedbacks:
        predicted_state = predicted_feedbacks[-1].get("predicted_state")

    state_seed = predicted_state if trust_predicted_state and predicted_state is not None else previous_state
    next_state = sanitize_state_content(state_seed if state_seed is not None else make_blank_state())

    final_result = execution_results[-1] if execution_results else None
    if final_result is not None:
        tool_name = final_result.get("resolved_name") or final_result.get("requested_name")
        success = infer_execution_result_success(final_result)
        content = final_result.get("content", "")
        if is_enterprise_state_payload(next_state):
            state_root = next_state.setdefault("state", {})
            target = state_root.get("diff_from_previous_state")
            if not isinstance(target, dict):
                target = state_root
            outcome = target.setdefault("outcome", {})
            if isinstance(outcome, dict):
                outcome["status"] = "success" if success else "failure"
                outcome["summary"] = str(content)[:1000] if content is not None else ""
                outcome.setdefault("failure_category", "none" if success else "tool_error")
                outcome.setdefault("recoverable", not success)
            history = target.setdefault("history_context", {})
            if isinstance(history, dict):
                events = history.setdefault("last_tool_events", [])
                if isinstance(events, list):
                    events.append(
                        {
                            "tool_name": tool_name,
                            "status": "success" if success else "failure",
                            "operation": "execute",
                            "summary": str(content)[:1000] if content is not None else "",
                            "error": "" if success else str(content),
                        }
                    )
        elif is_compact_tool_execution_state(next_state):
            next_state["success"] = success
            next_state["last_tool_execution_result"] = 1 if success else 0
            next_state["last_tool_name"] = tool_name
            next_state["error_message"] = "" if success else str(content)
            next_state.setdefault("current_stage", state_current_stage(previous_state))
            next_state.setdefault("remaining_stages", state_remaining_stages(previous_state) or [])
        else:
            state_root = next_state.setdefault("state", {})
            context = state_root.setdefault("context", {})
            context["last_tool_execution_result"] = 1 if success else 0
            context["last_tool_name"] = tool_name
            context["error_message"] = "" if success else str(content)

    return next_state


def summarize_mode_metrics(
    mode_name: str,
    use_world_model_internal_thinking: bool,
    max_steps: int,
    completed_tasks: int,
    completed_step_counts: list[int],
    completed_tool_call_counts: list[int],
    total_tool_steps_taken: int,
    total_tool_calls_taken: int,
    total_internal_thinking_iterations: int,
    task_records: list[dict[str, Any]],
) -> dict[str, Any]:
    average_steps = sum(completed_step_counts) / len(completed_step_counts) if completed_step_counts else None
    average_tool_calls_per_trajectory = (
        sum(record["tool_calls_taken"] for record in task_records) / len(task_records) if task_records else None
    )
    average_imagined_rollouts = (
        sum(record.get("imagined_rollouts_used", 0) for record in task_records) / len(task_records)
        if task_records
        else None
    )
    average_tool_calls_to_complete = (
        sum(completed_tool_call_counts) / len(completed_tool_call_counts)
        if completed_tool_call_counts
        else None
    )
    average_internal_thinking_iterations = (
        total_internal_thinking_iterations / len(task_records) if task_records else None
    )
    return {
        "mode": mode_name,
        "use_world_model_internal_thinking": use_world_model_internal_thinking,
        "evaluated_tasks": len(task_records),
        "max_steps_per_task": max_steps,
        "total_tool_steps_taken": total_tool_steps_taken,
        "total_tool_calls_taken": total_tool_calls_taken,
        "total_internal_thinking_iterations": total_internal_thinking_iterations,
        "average_internal_thinking_iterations_per_task": average_internal_thinking_iterations,
        "completed_tasks": completed_tasks,
        "completion_rate": completed_tasks / len(task_records) if task_records else 0.0,
        "average_steps_to_complete": average_steps,
        "average_tool_calls_per_trajectory": average_tool_calls_per_trajectory,
        "average_tool_calls_to_complete": average_tool_calls_to_complete,
        "average_imagined_rollouts_per_task": average_imagined_rollouts,
        "task_records": task_records,
    }


async def run_actual_mcp_execution_mode(
    mode_name: str,
    use_world_model_internal_thinking: bool,
    assistance_strategy: str,
    agent_generator: Any,
    world_model_generator: Any,
    tasks: list[TaskTrajectory],
    max_steps: int,
    final_answer_f1_threshold: float,
    world_model_target: str,
    include_error_message_in_target: bool,
    include_stage_in_target: bool,
    include_world_model_history: bool,
    internal_thinking_max_iterations: int,
    imagined_trajectory_max_steps: int,
    imagined_trajectory_rollouts: int,
    imagined_rollout_temperature: float,
    imagined_trajectory_selection_strategy: str,
    imagined_trajectory_observation_source: str,
    react_system_prompt: str,
    exact_lookup: dict[str, Any],
    alias_lookup: dict[str, list[Any]],
    record_replay_trajectories: bool = False,
    imagined_trajectory_candidate_actions: int = 3,
    imagined_trajectory_top_k: int = 3,
    revision_lookahead_steps: int = 1,
    revision_imagined_rollouts: int = 1,
    revision_rollout_temperature: float = 0.7,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> dict[str, Any]:
    state_history_size = max(0, int(state_history_size))
    completed_tasks = 0
    completed_step_counts: list[int] = []
    completed_tool_call_counts: list[int] = []
    total_tool_steps_taken = 0
    total_tool_calls_taken = 0
    total_internal_thinking_iterations = 0
    task_records = []
    imagined_trajectory_records: list[dict[str, Any]] = []
    replay_trajectory_records: list[dict[str, Any]] | None = (
        [] if record_replay_trajectories else None
    )

    DEFAULT_TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
    task_records_jsonl_path = DEFAULT_TRAJECTORIES_DIR / f"{mode_name}_task_records.jsonl"
    imagined_jsonl_path = DEFAULT_TRAJECTORIES_DIR / f"{mode_name}_imagined_rollouts.jsonl"
    replay_jsonl_path = (
        DEFAULT_TRAJECTORIES_DIR / f"{mode_name}_replay_trajectories.jsonl"
        if record_replay_trajectories
        else None
    )
    # Truncate any cache from a previous run so this invocation owns the stream.
    task_records_jsonl_path.write_text("", encoding="utf-8")
    imagined_jsonl_path.write_text("", encoding="utf-8")
    if replay_jsonl_path is not None:
        replay_jsonl_path.write_text("", encoding="utf-8")

    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as flush_exc:
            print(f"[{mode_name}] warning: failed to flush record to {path}: {flush_exc}")

    for i, task in tqdm(enumerate(tasks[20:]), total=len(tasks), desc=mode_name):
        #if not i in [14, 24, 39, 46]: continue
        conversation = [{"role": "system", "content": task.system_prompt}]
        for user_message in task.user_messages:
            conversation.append({"role": "user", "content": user_message})
        task_query = task.user_messages[-1] if task.user_messages else ""
        current_state = sanitize_state_content(task.initial_state or make_blank_state())
        current_state_history = append_state_history(
            [], current_state, max_items=state_history_size
        )
        current_input_history: list[dict[str, Any]] = []
        emit_progress(
            "TASK_START",
            mode=mode_name,
            task_index=task.trajectory_index,
            query=task_query,
        )

        tool_steps_taken = 0
        tool_calls_taken = 0
        task_completed = False
        failure_reason = ""
        final_answer_score = None
        final_answer_evaluation = None
        predicted_final_answer = ""
        internal_iterations_used = 0
        imagined_rollouts_used = 0
        task_imagined_rollouts: list[dict[str, Any]] = []
        wm_predicted_failures = 0
        wm_predicted_successes = 0
        wm_triggered_revisions = 0
        wm_no_op_revisions = 0
        wm_revision_step_details: list[dict[str, Any]] = []

        try:
            while tool_steps_taken < max_steps:
                step_index = tool_steps_taken
                planning_conversation = conversation
                if assistance_strategy == "imagined":
                    imagined_steps, candidate_rollouts, imagined_selection = imagine_trajectory_candidates(
                        agent_generator=agent_generator,
                        world_model_generator=world_model_generator,
                        task=task,
                        conversation=conversation,
                        previous_state=current_state,
                        react_system_prompt=react_system_prompt,
                        max_imagined_steps=imagined_trajectory_max_steps,
                        world_model_target=world_model_target,
                        include_error_message_in_target=include_error_message_in_target,
                        include_stage_in_target=include_stage_in_target,
                        include_world_model_history=include_world_model_history,
                        start_interaction_index=step_index,
                        num_rollouts=imagined_trajectory_rollouts,
                        rollout_temperature=imagined_rollout_temperature,
                        selection_strategy=imagined_trajectory_selection_strategy,
                        observation_source=imagined_trajectory_observation_source,
                        candidate_action_count=imagined_trajectory_candidate_actions,
                        top_k=imagined_trajectory_top_k,
                        state_history=current_state_history,
                        input_history=current_input_history,
                        state_history_size=state_history_size,
                        system_prompt_max_chars=system_prompt_max_chars,
                        action_max_chars=action_max_chars,
                    )
                    if imagined_steps:
                        planning_conversation = conversation + [
                            build_imagined_trajectory_prompt_message(imagined_steps)
                        ]
                        imagined_rollouts_used += 1
                        rollout_record = {
                            "step_index": step_index,
                            "starting_state": summarize_state_for_planning(current_state),
                            "imagined_steps": imagined_steps,
                            "candidate_rollouts": candidate_rollouts,
                            "selection": imagined_selection,
                            "observation_source": imagined_trajectory_observation_source,
                            "replayed_steps": [],
                        }
                        task_imagined_rollouts.append(rollout_record)

                        replayed_imagined_steps = False
                        stop_after_imagined_replay = False
                        for imagined_step in imagined_steps:
                            if tool_steps_taken >= max_steps:
                                break
                            if "final_answer" in imagined_step and not replayed_imagined_steps:
                                predicted_final_answer = imagined_step["final_answer"]
                                final_answer_evaluation = evaluate_final_answer_quality(
                                    task_description=task_query,
                                    final_response=imagined_step["final_answer"],
                                    ground_truth_answer=task.final_answer,
                                    execution_trajectory=conversation
                                    + [
                                        {"role": "assistant", "content": imagined_step["final_answer"]},
                                    ],
                                )
                                final_answer_score = final_answer_evaluation["overall_score"]
                                if final_answer_score >= final_answer_f1_threshold:
                                    task_completed = True
                                    completed_tasks += 1
                                    completed_step_counts.append(tool_steps_taken)
                                    completed_tool_call_counts.append(tool_calls_taken)
                                else:
                                    failure_reason = f"final_answer_below_threshold:score={final_answer_score:.3f}"
                                stop_after_imagined_replay = True
                                break
                            if "clarify" in imagined_step and not replayed_imagined_steps:
                                failure_reason = f"clarification_requested:{imagined_step['clarify']}"
                                stop_after_imagined_replay = True
                                break

                            planned_calls = [normalize_tool_call(call) for call in imagined_step.get("tool_calls", [])]
                            if not planned_calls:
                                break

                            # Option-2 design: drop the imagined thought and re-think against
                            # the real conversation so the recorded thought is grounded in
                            # actual observations rather than the world-model imagination.
                            raw_thought = agent_generator.generate_from_messages(
                                build_react_think_messages(
                                    conversation,
                                    current_query=task.user_messages[-1] if task.user_messages else "",
                                    system_prompt=react_system_prompt,
                                )
                            )
                            raw_thought = strip_model_thinking_output(raw_thought)
                            real_thought_payload = parse_thought_payload(raw_thought)
                            conversation.append(
                                {"role": "assistant", "content": json.dumps(real_thought_payload, ensure_ascii=False)}
                            )

                            predicted_feedbacks = imagined_step.get("predicted_feedback") or []
                            unsatisfactory_prediction = imagined_step_is_unsatisfactory(imagined_step)
                            execution_results = await execute_actual_tool_calls(
                                planned_calls,
                                exact_lookup,
                                alias_lookup,
                            )

                            tool_steps_taken += 1
                            tool_calls_taken += len(planned_calls)
                            total_tool_steps_taken += 1
                            total_tool_calls_taken += len(planned_calls)

                            conversation.append({"role": "assistant", "tool_calls": to_openai_tool_calls(planned_calls)})
                            for result in execution_results:
                                conversation.append(
                                    {
                                        "role": "tool",
                                        "name": result["requested_name"],
                                        "content": result["content"],
                                    }
                                )

                            current_state = update_state_from_actual_execution(
                                previous_state=current_state,
                                execution_results=execution_results,
                                predicted_feedbacks=predicted_feedbacks,
                                trust_predicted_state=not unsatisfactory_prediction,
                            )
                            current_state_history = append_state_history(
                                current_state_history,
                                current_state,
                                max_items=state_history_size,
                            )
                            current_input_history = append_world_model_input_history(
                                current_input_history,
                                make_actual_world_model_history_entry_from_results(
                                    step=step_index + 1,
                                    tool_calls=planned_calls,
                                    execution_results=execution_results,
                                ),
                            )
                            rollout_record["replayed_steps"].append(
                                {
                                    "imagined_step": imagined_step.get("imagined_step"),
                                    "real_thought": real_thought_payload,
                                    "tool_calls": planned_calls,
                                    "unsatisfactory_prediction": unsatisfactory_prediction,
                                    "execution_results": execution_results,
                                    "resulting_state": summarize_state_for_planning(current_state),
                                }
                            )
                            replayed_imagined_steps = True
                            if unsatisfactory_prediction or any(not result["success"] for result in execution_results):
                                break

                        if task_completed or stop_after_imagined_replay:
                            break
                        if replayed_imagined_steps:
                            continue

                raw_thought = agent_generator.generate_from_messages(
                    build_react_think_messages(
                        planning_conversation,
                        current_query=task.user_messages[-1] if task.user_messages else "",
                        system_prompt=react_system_prompt,
                    )
                )
                raw_thought = strip_model_thinking_output(raw_thought)
                thought_payload = parse_thought_payload(raw_thought)
                emit_progress(
                    "AGENT_THOUGHT",
                    mode=mode_name,
                    task_index=task.trajectory_index,
                    step=step_index,
                    thought=thought_payload.get("thought"),
                )
                conversation.append({"role": "assistant", "content": json.dumps(thought_payload, ensure_ascii=False)})

                raw_decision = agent_generator.generate_from_messages(
                    build_react_action_messages(
                        planning_conversation + [
                            {"role": "assistant", "content": json.dumps(thought_payload, ensure_ascii=False)}
                        ],
                        current_query=task.user_messages[-1] if task.user_messages else "",
                        system_prompt=react_system_prompt,
                    )
                )
                raw_decision = strip_model_thinking_output(raw_decision)
                try:
                    decision = parse_agent_decision(raw_decision)
                except Exception as exc:
                    failure_reason = f"unparseable_action:{exc}"
                    emit_progress(
                        "AGENT_ACTION_PARSE_ERROR",
                        mode=mode_name,
                        task_index=task.trajectory_index,
                        step=step_index,
                        error=failure_reason,
                        raw_action=raw_decision,
                    )
                    break

                if "final_answer" in decision:
                    predicted_final_answer = decision["final_answer"]
                    emit_progress(
                        "AGENT_FINAL_ANSWER",
                        mode=mode_name,
                        task_index=task.trajectory_index,
                        step=step_index,
                        final_answer=predicted_final_answer,
                    )
                    final_answer_evaluation = evaluate_final_answer_quality(
                        task_description=task_query,
                        final_response=decision["final_answer"],
                        ground_truth_answer=task.final_answer,
                        execution_trajectory=conversation + [{"role": "assistant", "content": decision["final_answer"]}],
                    )
                    final_answer_score = final_answer_evaluation["overall_score"]
                    if final_answer_score >= final_answer_f1_threshold:
                        task_completed = True
                        completed_tasks += 1
                        completed_step_counts.append(tool_steps_taken)
                        completed_tool_call_counts.append(tool_calls_taken)
                    else:
                        failure_reason = f"final_answer_below_threshold:score={final_answer_score:.3f}"
                    break
                if "clarify" in decision:
                    emit_progress(
                        "AGENT_CLARIFY",
                        mode=mode_name,
                        task_index=task.trajectory_index,
                        step=step_index,
                        clarify=decision["clarify"],
                    )
                    failure_reason = f"clarification_requested:{decision['clarify']}"
                    break

                planned_calls = [normalize_tool_call(call) for call in decision.get("tool_calls", [])]
                if not planned_calls:
                    emit_progress(
                        "AGENT_EMPTY_ACTION",
                        mode=mode_name,
                        task_index=task.trajectory_index,
                        step=step_index,
                        raw_action=raw_decision,
                    )
                    failure_reason = "empty_tool_calls"
                    break
                emit_progress(
                    "AGENT_ACTION",
                    mode=mode_name,
                    task_index=task.trajectory_index,
                    step=step_index,
                    tool_calls=preview_tool_calls(planned_calls),
                )

                internal_feedbacks: list[dict[str, Any]] = []
                revision_rollouts: list[dict[str, Any]] = []
                if assistance_strategy == "revision":
                    use_lookahead_rollouts = (
                        revision_lookahead_steps > 1 or revision_imagined_rollouts > 1
                    )
                    original_planned_calls = [dict(call) for call in planned_calls]
                    iter1_predicted_success: bool | None = None
                    iter1_error_message: str = ""
                    revision_loop_outcome = "no_iterations"
                    for iteration in range(1, internal_thinking_max_iterations + 1):
                        emit_progress(
                            "REVISION_ITERATION_START",
                            mode=mode_name,
                            task_index=task.trajectory_index,
                            step=step_index,
                            iteration=iteration,
                            planned_calls=preview_tool_calls(planned_calls),
                        )
                        if use_lookahead_rollouts:
                            revision_rollouts = imagine_revision_rollouts(
                                agent_generator=agent_generator,
                                world_model_generator=world_model_generator,
                                task=task,
                                conversation=conversation,
                                previous_state=current_state,
                                react_system_prompt=react_system_prompt,
                                initial_planned_calls=planned_calls,
                                lookahead_steps=revision_lookahead_steps,
                                num_rollouts=revision_imagined_rollouts,
                                world_model_target=world_model_target,
                                include_error_message_in_target=include_error_message_in_target,
                                include_stage_in_target=include_stage_in_target,
                                include_world_model_history=include_world_model_history,
                                start_interaction_index=step_index,
                                rollout_temperature=revision_rollout_temperature,
                                state_history=current_state_history,
                                input_history=current_input_history,
                                state_history_size=state_history_size,
                                system_prompt_max_chars=system_prompt_max_chars,
                                action_max_chars=action_max_chars,
                            )
                            # Use the deterministic (rollout #0) first-step
                            # feedback as the canonical iteration-1 signal so the
                            # existing telemetry stays apples-to-apples with
                            # single-step revision runs.
                            feedbacks = (
                                revision_rollouts[0].get("first_step_feedbacks", [])
                                if revision_rollouts
                                else []
                            )
                            if not feedbacks:
                                # Fall back to a single-step prediction if every
                                # rollout failed to produce feedback (defensive).
                                feedbacks = predict_world_model_feedback(
                                    world_model_generator,
                                    task,
                                    current_state,
                                    planned_calls,
                                    interaction_index=step_index,
                                    world_model_target=world_model_target,
                                    include_error_message_in_target=include_error_message_in_target,
                                    include_stage_in_target=include_stage_in_target,
                                    include_world_model_history=include_world_model_history,
                                    state_history=current_state_history,
                                    input_history=current_input_history,
                                    system_prompt_max_chars=system_prompt_max_chars,
                                    action_max_chars=action_max_chars,
                                )
                        else:
                            revision_rollouts = []
                            feedbacks = predict_world_model_feedback(
                                world_model_generator,
                                task,
                                current_state,
                                planned_calls,
                                interaction_index=step_index,
                                world_model_target=world_model_target,
                                include_error_message_in_target=include_error_message_in_target,
                                include_stage_in_target=include_stage_in_target,
                                include_world_model_history=include_world_model_history,
                                state_history=current_state_history,
                                input_history=current_input_history,
                                system_prompt_max_chars=system_prompt_max_chars,
                                action_max_chars=action_max_chars,
                            )
                        internal_feedbacks = feedbacks
                        internal_iterations_used += 1
                        total_internal_thinking_iterations += 1
                        if iteration == 1:
                            iter1_predicted_success = bool(
                                feedbacks
                                and all(fb.get("predicted_success", False) for fb in feedbacks)
                            )
                            iter1_error_message = " | ".join(
                                msg
                                for msg in (
                                    (fb.get("predicted_error_message") or "").strip()
                                    for fb in feedbacks
                                    if not fb.get("predicted_success", True)
                                )
                                if msg
                            )
                        target_is_tool_output = is_tool_output_target(world_model_target)
                        # When multi-step lookahead is in effect, only short-circuit
                        # if every rollout predicted full success across all K steps.
                        # Otherwise fall back to the original single-step check.
                        if use_lookahead_rollouts and revision_rollouts:
                            all_rollouts_success = all(
                                rollout.get("all_predicted_success")
                                for rollout in revision_rollouts
                            )
                        else:
                            all_rollouts_success = all(
                                feedback["predicted_success"] for feedback in feedbacks
                            )
                        if not target_is_tool_output and all_rollouts_success:
                            revision_loop_outcome = "wm_predicted_success"
                            break

                        raw_revision = agent_generator.generate_from_messages(
                            build_internal_thinking_messages(
                                conversation,
                                planned_calls,
                                feedbacks,
                                iteration=iteration,
                                max_iterations=internal_thinking_max_iterations,
                                world_model_target=world_model_target,
                                revision_rollouts=revision_rollouts,
                            )
                        )
                        emit_progress(
                            "REVISION_RAW",
                            mode=mode_name,
                            task_index=task.trajectory_index,
                            step=step_index,
                            iteration=iteration,
                            planned_calls=preview_tool_calls(planned_calls),
                            feedbacks=feedbacks,
                            raw_revision=raw_revision,
                        )
                        try:
                            revised_decision = parse_agent_decision(raw_revision)
                        except Exception as exc:
                            failure_reason = f"internal_thinking_unparseable:{exc}"
                            emit_progress(
                                "REVISION_PARSE_ERROR",
                                mode=mode_name,
                                task_index=task.trajectory_index,
                                step=step_index,
                                iteration=iteration,
                                error=failure_reason,
                            )
                            planned_calls = []
                            revision_loop_outcome = "agent_unparseable_revision"
                            break
                        revised_calls = [normalize_tool_call(call) for call in revised_decision.get("tool_calls", [])]
                        if not revised_calls:
                            emit_progress(
                                "REVISION_EMPTY_ACTION",
                                mode=mode_name,
                                task_index=task.trajectory_index,
                                step=step_index,
                                iteration=iteration,
                            )
                            failure_reason = "internal_thinking_empty_tool_calls"
                            planned_calls = []
                            revision_loop_outcome = "agent_empty_revision"
                            break
                        emit_progress(
                            "REVISION_ACTION",
                            mode=mode_name,
                            task_index=task.trajectory_index,
                            step=step_index,
                            iteration=iteration,
                            revised_calls=preview_tool_calls(revised_calls),
                        )
                        if target_is_tool_output and tool_calls_equal(planned_calls, revised_calls):
                            planned_calls = revised_calls
                            revision_loop_outcome = "agent_kept_calls"
                            break
                        planned_calls = revised_calls
                        revision_loop_outcome = "iterations_exhausted"

                    if iter1_predicted_success is True:
                        wm_predicted_successes += 1
                    elif iter1_predicted_success is False:
                        wm_predicted_failures += 1
                        calls_changed = bool(planned_calls) and not tool_calls_equal(
                            original_planned_calls, planned_calls
                        )
                        if calls_changed:
                            wm_triggered_revisions += 1
                        else:
                            wm_no_op_revisions += 1
                        wm_revision_step_details.append(
                            {
                                "step_index": step_index,
                                "original_calls": original_planned_calls,
                                "executed_calls": list(planned_calls),
                                "iter1_predicted_error_message": iter1_error_message,
                                "calls_changed": calls_changed,
                                "iterations": iteration,
                                "loop_outcome": revision_loop_outcome,
                                "revision_rollouts": revision_rollouts,
                            }
                        )
                        emit_progress(
                            "REVISION_SUMMARY",
                            mode=mode_name,
                            task_index=task.trajectory_index,
                            detail=wm_revision_step_details[-1],
                        )

                    if not planned_calls:
                        break

                execution_results = await execute_actual_tool_calls(
                    planned_calls,
                    exact_lookup,
                    alias_lookup,
                )
                emit_progress(
                    "STEP_EXECUTED",
                    mode=mode_name,
                    task_index=task.trajectory_index,
                    step=step_index,
                    tool_calls=preview_tool_calls(planned_calls),
                    execution_results=execution_results,
                )

                tool_steps_taken += 1
                tool_calls_taken += len(planned_calls)
                total_tool_steps_taken += 1
                total_tool_calls_taken += len(planned_calls)

                conversation.append({"role": "assistant", "tool_calls": to_openai_tool_calls(planned_calls)})
                if assistance_strategy == "revision" and internal_feedbacks:
                    feedback_sections: list[str] = []
                    for feedback in internal_feedbacks:
                        compact_feedback = {
                            key: value
                            for key, value in feedback.items()
                            if key != "predicted_tool_output"
                        }
                        feedback_sections.append(
                            json.dumps(compact_feedback, ensure_ascii=False, indent=2)
                        )
                        predicted_tool_output_text = (
                            feedback.get("predicted_tool_output") or ""
                        ).strip()
                        if predicted_tool_output_text:
                            tool_calls_for_feedback = feedback.get("tool_calls") or []
                            tool_name = (
                                tool_calls_for_feedback[0].get("name", "")
                                if tool_calls_for_feedback
                                else ""
                            )
                            header = "[PREDICTED_TOOL_OUTPUT_FROM_WORLD_MODEL"
                            if tool_name:
                                header += f" name={tool_name}"
                            header += "]"
                            feedback_sections.append(
                                f"{header}\n{predicted_tool_output_text}"
                            )
                    conversation.append(
                        {
                            "role": "system",
                            "content": "[INTERNAL_WORLD_MODEL_THINKING]\n"
                            + "\n\n".join(feedback_sections),
                        }
                    )
                if internal_feedbacks:
                    current_state = update_state_from_actual_execution(
                        previous_state=current_state,
                        execution_results=execution_results,
                        predicted_feedbacks=internal_feedbacks,
                        trust_predicted_state=True,
                    )
                elif assistance_strategy == "revision":
                    executed_feedbacks = predict_world_model_feedback(
                        world_model_generator,
                        task,
                        current_state,
                        planned_calls,
                        interaction_index=step_index,
                        world_model_target=world_model_target,
                        include_error_message_in_target=include_error_message_in_target,
                        include_stage_in_target=include_stage_in_target,
                        include_world_model_history=include_world_model_history,
                        state_history=current_state_history,
                        input_history=current_input_history,
                        system_prompt_max_chars=system_prompt_max_chars,
                        action_max_chars=action_max_chars,
                    )
                    current_state = update_state_from_actual_execution(
                        previous_state=current_state,
                        execution_results=execution_results,
                        predicted_feedbacks=executed_feedbacks,
                        trust_predicted_state=bool(executed_feedbacks),
                    )
                else:
                    current_state = update_state_from_actual_execution(
                        previous_state=current_state,
                        execution_results=execution_results,
                        predicted_feedbacks=None,
                        trust_predicted_state=False,
                    )
                current_state_history = append_state_history(
                    current_state_history, current_state, max_items=state_history_size
                )
                current_input_history = append_world_model_input_history(
                    current_input_history,
                    make_actual_world_model_history_entry_from_results(
                        step=step_index + 1,
                        tool_calls=planned_calls,
                        execution_results=execution_results,
                    ),
                )
                for result in execution_results:
                    conversation.append(
                        {
                            "role": "tool",
                            "name": result["requested_name"],
                            "content": result["content"],
                        }
                    )

            if not task_completed and not failure_reason and tool_steps_taken >= max_steps:
                failure_reason = f"max_steps_exhausted:{max_steps}"

            if not task_completed:
                raw_decision = agent_generator.generate_from_messages(
                    build_react_action_messages(
                        conversation,
                        current_query=task.user_messages[-1] if task.user_messages else "",
                        system_prompt=react_system_prompt,
                    )
                )
                raw_decision = strip_model_thinking_output(raw_decision)
                try:
                    decision = parse_agent_decision(raw_decision)
                except Exception:
                    decision = {}
                if "final_answer" in decision:
                    predicted_final_answer = decision["final_answer"]
                    emit_progress(
                        "AGENT_FINAL_ANSWER_POST_LOOP",
                        mode=mode_name,
                        task_index=task.trajectory_index,
                        final_answer=predicted_final_answer,
                    )
                    final_answer_evaluation = evaluate_final_answer_quality(
                        task_description=task_query,
                        final_response=decision["final_answer"],
                        ground_truth_answer=task.final_answer,
                        execution_trajectory=conversation + [{"role": "assistant", "content": decision["final_answer"]}],
                    )
                    final_answer_score = final_answer_evaluation["overall_score"]
                    if final_answer_score >= final_answer_f1_threshold:
                        task_completed = True
                        completed_tasks += 1
                        completed_step_counts.append(tool_steps_taken)
                        completed_tool_call_counts.append(tool_calls_taken)
                    else:
                        failure_reason = f"final_answer_below_threshold:score={final_answer_score:.3f}"
                elif not failure_reason:
                    failure_reason = "missing_final_answer"

        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            tb_summary = traceback.format_exception_only(type(exc), exc)[-1].strip()
            crash_reason = f"exception:{type(exc).__name__}:{tb_summary}"
            failure_reason = f"{failure_reason};{crash_reason}" if failure_reason else crash_reason
            print(f"[{mode_name}] task {task.trajectory_index} aborted: {crash_reason}")
            traceback.print_exc()

        task_record = {
            "trajectory_index": task.trajectory_index,
            "tool_steps_taken": tool_steps_taken,
            "tool_calls_taken": tool_calls_taken,
            "completed": task_completed,
            "failure_reason": failure_reason,
            "final_answer_score": final_answer_score,
            "final_answer_evaluation": final_answer_evaluation,
            "internal_thinking_iterations": internal_iterations_used,
            "imagined_rollouts_used": imagined_rollouts_used,
            "wm_predicted_failures": wm_predicted_failures,
            "wm_predicted_successes": wm_predicted_successes,
            "wm_triggered_revisions": wm_triggered_revisions,
            "wm_no_op_revisions": wm_no_op_revisions,
            "wm_revision_effectiveness": (
                wm_triggered_revisions / wm_predicted_failures
                if wm_predicted_failures > 0
                else None
            ),
            "wm_revision_step_details": wm_revision_step_details,
            "imagined_rollout_records": task_imagined_rollouts,
            "imagined_trajectory_observation_source": imagined_trajectory_observation_source,
        }
        emit_progress(
            "TASK_END",
            mode=mode_name,
            task_index=task.trajectory_index,
            completed=task_completed,
            failure_reason=failure_reason,
            tool_steps_taken=tool_steps_taken,
            tool_calls_taken=tool_calls_taken,
            final_answer_score=final_answer_score,
        )
        task_records.append(task_record)
        _append_jsonl(task_records_jsonl_path, task_record)

        if replay_trajectory_records is not None:
            recorded_messages = list(conversation)
            if predicted_final_answer:
                recorded_messages.append(
                    {"role": "assistant", "content": predicted_final_answer}
                )
            replay_record = {
                "trajectory_index": task.trajectory_index,
                "task_query": task_query,
                "ground_truth_final_answer": task.final_answer,
                "predicted_final_answer": predicted_final_answer,
                "completed": task_completed,
                "failure_reason": failure_reason,
                "tool_steps_taken": tool_steps_taken,
                "tool_calls_taken": tool_calls_taken,
                "final_answer_score": final_answer_score,
                "messages": recorded_messages,
            }
            replay_trajectory_records.append(replay_record)
            if replay_jsonl_path is not None:
                _append_jsonl(replay_jsonl_path, replay_record)

        if task_imagined_rollouts:
            imagined_record = {
                "trajectory_index": task.trajectory_index,
                "task_query": task.user_messages[-1] if task.user_messages else "",
                "rollouts": task_imagined_rollouts,
            }
            imagined_trajectory_records.append(imagined_record)
            _append_jsonl(imagined_jsonl_path, imagined_record)
    metrics = summarize_mode_metrics(
        mode_name=mode_name,
        use_world_model_internal_thinking=use_world_model_internal_thinking,
        max_steps=max_steps,
        completed_tasks=completed_tasks,
        completed_step_counts=completed_step_counts,
        completed_tool_call_counts=completed_tool_call_counts,
        total_tool_steps_taken=total_tool_steps_taken,
        total_tool_calls_taken=total_tool_calls_taken,
        total_internal_thinking_iterations=total_internal_thinking_iterations,
        task_records=task_records,
    )
    if imagined_trajectory_records:
        imagined_trajectories_path = dump_imagined_trajectories(mode_name, imagined_trajectory_records)
        metrics["imagined_trajectories_path"] = str(imagined_trajectories_path)
    if replay_trajectory_records is not None:
        replay_trajectories_path = dump_replay_trajectories(mode_name, replay_trajectory_records)
        metrics["replay_trajectories_path"] = str(replay_trajectories_path)
        metrics["replay_trajectories_count"] = len(replay_trajectory_records)
    metrics["task_records_jsonl_path"] = str(task_records_jsonl_path)
    metrics["imagined_rollouts_jsonl_path"] = str(imagined_jsonl_path)
    if replay_jsonl_path is not None:
        metrics["replay_trajectories_jsonl_path"] = str(replay_jsonl_path)
    print(metrics)
    return metrics


def train_world_model(
    args: argparse.Namespace,
    train_examples: list[WorldModelStateExample],
    eval_examples: list[WorldModelStateExample],
) -> dict[str, Any]:
    torch, Dataset, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed = require_training_stack()
    set_seed(args.seed)

    tokenizer = load_tokenizer_with_repair(
        args.model, trust_remote_code=args.trust_remote_code, auto_tokenizer_class=AutoTokenizer
    )
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

    # Counts are captured up front because the example lists are DRAINED below.
    train_example_count = len(train_examples)
    eval_example_count = len(eval_examples)

    def drain_to_rows(examples: list[WorldModelStateExample]) -> list[dict[str, Any]]:
        """Build the prompt/completion rows, releasing each example as its row is produced.

        A list comprehension holds the full example list AND the full row list at once; at the
        scale of the large presets (2M examples per split) that is tens of GB of avoidable peak
        on EVERY rank. Draining in place keeps only one of the two at full size. The caller's
        list is emptied deliberately -- main_after_data_prep does not read the examples again.
        """
        rows: list[dict[str, Any]] = []
        for index in range(len(examples)):
            rows.append(
                build_state_prediction_prompt_completion_row(
                    examples[index],
                    target_mode=args.world_model_target,
                    include_error_message=args.include_error_message_in_target,
                    include_stage=args.include_stage_in_target,
                    include_input_history=args.include_world_model_history,
                    system_prompt_max_chars=args.world_model_system_prompt_max_chars,
                    action_max_chars=args.world_model_action_max_chars,
                )
            )
            examples[index] = None  # type: ignore[call-overload]
        examples.clear()
        return rows

    train_rows = drain_to_rows(train_examples)
    # --skip-eval means no eval dataset at all: no rows, no tokenization, no Arrow table.
    eval_rows = [] if args.skip_eval else drain_to_rows(eval_examples)
    eval_examples.clear()
    gc.collect()
    print(f"[train] {len(train_rows)} train rows, {len(eval_rows)} eval rows "
          f"(resident {_resident_gb():.1f} GB)", flush=True)

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

    tokenized_cache_dir = None
    if not args.no_tokenized_cache:
        tokenized_cache_dir = args.tokenized_cache_dir or (args.output_dir / "tokenized_cache")
        tokenized_cache_dir.mkdir(parents=True, exist_ok=True)
    tokenize_kwargs = dict(
        max_seq_length=args.max_seq_length,
        min_completion_tokens=args.min_completion_tokens,
        disable_chat_template=args.disable_chat_template,
        num_proc=args.tokenize_num_proc,
        map_batch_size=args.tokenize_batch_size,
        prefilter_chars_per_token=args.tokenize_prefilter_chars_per_token,
        cache_dir=tokenized_cache_dir,
        rebuild_cache=args.rebuild_tokenized_cache,
        extra_fingerprint={
            "world_model_target": args.world_model_target,
            "include_error_message_in_target": args.include_error_message_in_target,
            "include_stage_in_target": args.include_stage_in_target,
            "include_world_model_history": args.include_world_model_history,
            "world_model_system_prompt_max_chars": args.world_model_system_prompt_max_chars,
            "world_model_action_max_chars": args.world_model_action_max_chars,
            "oversample_minority_outcomes": args.oversample_minority_outcomes,
            "oversample_target_ratio": args.oversample_target_ratio,
            "oversample_max_multiplier": args.oversample_max_multiplier,
            "seed": args.seed,
        },
    )
    train_dataset, train_dataset_stats = build_or_load_prompt_completion_dataset(
        Dataset,
        tokenizer,
        train_rows,
        split="train",
        **tokenize_kwargs,
    )
    # The Arrow table now owns the data; the Python list is a full second copy and .map() below
    # would otherwise run with both resident.
    train_rows = []
    gc.collect()
    eval_dataset = None
    eval_dataset_stats = None
    if eval_rows:
        eval_dataset, eval_dataset_stats = build_or_load_prompt_completion_dataset(
            Dataset,
            tokenizer,
            eval_rows,
            split="eval",
            **tokenize_kwargs,
        )
        eval_rows = []
        gc.collect()
    print(f"[train] tokenized datasets built (resident {_resident_gb():.1f} GB)", flush=True)
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
        eval_strategy="no" if eval_dataset is not None else "no",
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
                "train_examples": train_example_count,
                "eval_examples": eval_example_count,
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
            "train_examples": train_example_count,
            "eval_examples": eval_example_count,
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
        "train_examples": train_example_count,
        "eval_examples": eval_example_count,
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


def evaluate_world_model_predictions(
    generator: HFTextGenerator,
    examples: list[WorldModelStateExample],
    sample_limit: int,
    target_mode: str = "state",
    include_error_message: bool = False,
    include_stage: bool = False,
    include_input_history: bool = False,
) -> dict[str, Any]:
    target_mode = canonicalize_world_model_target(target_mode)
    stage_metrics_enabled = (
        target_mode == WORLD_MODEL_TARGET_STATE
        or (include_stage and target_mode == WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY)
    )
    if not examples:
        run_outcome_error_judge = (
            include_error_message and is_tool_execution_result_target(target_mode)
        )
        empty_metrics = {
            "sampled_examples": 0,
            "exact_match": 0.0,
            "last_tool_execution_result_accuracy": 0.0,
            "binary_classification_accuracy": (
                0.0 if is_tool_execution_result_target(target_mode) else None
            ),
            "outcome_with_error_judge_accuracy": 0.0 if run_outcome_error_judge else None,
            "outcome_with_error_judge_score": 0.0 if run_outcome_error_judge else None,
            "outcome_with_error_judge_evaluated": 0 if run_outcome_error_judge else None,
            "llm_judge_score": (
                0.0
                if target_mode == WORLD_MODEL_TARGET_STATE or is_tool_output_target(target_mode)
                else None
            ),
            "llm_judge_match_rate": (
                0.0
                if target_mode == WORLD_MODEL_TARGET_STATE or is_tool_output_target(target_mode)
                else None
            ),
            "field_level_llm_judge_score": 0.0 if target_mode == WORLD_MODEL_TARGET_STATE else None,
            "field_level_llm_judge_match_rate": 0.0 if target_mode == WORLD_MODEL_TARGET_STATE else None,
            "field_level_llm_judge_scores": {} if target_mode == WORLD_MODEL_TARGET_STATE else None,
            "current_stage_accuracy": 0.0 if stage_metrics_enabled else None,
            "remaining_stages_accuracy": 0.0 if stage_metrics_enabled else None,
        }
        return empty_metrics

    sample_size = len(examples) if sample_limit <= 0 else min(sample_limit, len(examples))
    sampled_examples = examples[:sample_size]
    exact_matches = 0
    parse_successes = 0
    last_tool_execution_result_matches = 0
    llm_judge_score_sum = 0.0
    llm_judge_matches = 0
    field_level_judge_score_sum = 0.0
    field_level_judge_matches = 0
    field_level_judge_count = 0
    field_level_judge_score_sums: dict[str, float] = {}
    field_level_judge_match_sums: dict[str, int] = {}
    field_level_judge_counts: dict[str, int] = {}
    current_stage_matches = 0
    remaining_stages_matches = 0
    outcome_error_judge_score_sum = 0.0
    outcome_error_judge_matches = 0
    outcome_error_judge_evaluated = 0
    records = []
    run_outcome_error_judge = (
        include_error_message and is_tool_execution_result_target(target_mode)
    )

    for example in tqdm(sampled_examples, total=len(sampled_examples)):
        prediction = generator.generate_from_messages(
            build_state_prediction_chat_messages(
                example,
                target_mode=target_mode,
                include_error_message=include_error_message,
                include_stage=include_stage,
                include_input_history=include_input_history,
            )
        )
        prediction = strip_model_thinking_output(prediction)
        gold_last_tool_execution_result = normalize_tool_execution_result_for_target(
            extract_last_tool_execution_result_from_state(example.state),
            target_mode=target_mode,
        )
        if target_mode == WORLD_MODEL_TARGET_STATE:
            gold_target = normalize_state_text(example.state)
        elif is_tool_output_target(target_mode):
            gold_target = example.tool_output or ""
        elif gold_last_tool_execution_result is None:
            gold_target = "None"
        else:
            gold_target = format_tool_execution_result_target(
                gold_last_tool_execution_result,
                example.error_payload,
                include_error_message=include_error_message,
                include_stage=(
                    include_stage
                    and target_mode == WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY
                ),
                current_stage=state_current_stage(example.state),
                remaining_stages=state_remaining_stages(example.state),
            )
        exact_match = False
        parse_success = False
        predicted_last_tool_execution_result = None
        predicted_error_message = ""
        predicted_current_stage = None
        predicted_remaining_stages = None
        if is_tool_output_target(target_mode):
            judge_zero_scores = {key: 0.0 for key in TOOL_OUTPUT_MATCH_JUDGE_KEYS}
        else:
            judge_zero_scores = {key: 0.0 for key in STATE_MATCH_JUDGE_KEYS}
        llm_judge_result = {
            "scores": judge_zero_scores,
            "overall_score": 0.0,
            "match": False,
            "comments": "Prediction could not be judged",
            "raw_response": {},
        }
        outcome_error_judge_result = None
        parse_error = None
        try:
            if is_tool_execution_result_target(target_mode):
                parsed_prediction_payload = parse_binary_world_model_prediction(prediction)
                predicted_last_tool_execution_result = normalize_tool_execution_result_for_target(
                    parsed_prediction_payload.get("success"),
                    target_mode=target_mode,
                )
                predicted_error_message = parsed_prediction_payload.get("error_message", "") or ""
                predicted_current_stage = parsed_prediction_payload.get("current_stage")
                predicted_remaining_stages = parsed_prediction_payload.get("remaining_stages")
                parse_success = predicted_last_tool_execution_result is not None
                exact_match = parse_success and predicted_last_tool_execution_result == gold_last_tool_execution_result
                if include_stage and target_mode == WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY:
                    gold_payload = parse_binary_world_model_prediction(gold_target)
                    exact_match = exact_match and predicted_current_stage == gold_payload.get("current_stage")
                    exact_match = exact_match and predicted_remaining_stages == gold_payload.get("remaining_stages")
                    if include_error_message:
                        exact_match = exact_match and predicted_error_message == gold_payload.get("error_message", "")
            elif is_tool_output_target(target_mode):
                normalized_prediction = (prediction or "").strip()
                parse_success = True
                exact_match = normalized_prediction == (gold_target or "").strip()
                llm_judge_result = evaluate_tool_output_match_quality(
                    system_prompt=example.system_prompt,
                    user_prompt=example.user_prompt,
                    previous_state=example.previous_state,
                    action=example.action,
                    predicted_tool_output=normalized_prediction,
                    target_tool_output=(gold_target or "").strip(),
                )
            else:
                parsed_prediction = sanitize_state_content(parse_jsonish(prediction))
                normalized_prediction = json.dumps(parsed_prediction, ensure_ascii=False, sort_keys=True)
                exact_match = normalized_prediction == gold_target
                parse_success = True
                llm_judge_result = evaluate_state_match_quality(
                    system_prompt=example.system_prompt,
                    user_prompt=example.user_prompt,
                    previous_state=example.previous_state,
                    action=example.action,
                    predicted_state_text=normalized_prediction,
                    target_state_text=gold_target,
                )
                predicted_last_tool_execution_result = normalize_tool_execution_result_for_target(
                    extract_last_tool_execution_result_from_state(parsed_prediction),
                    target_mode=target_mode,
                )
                predicted_current_stage = state_current_stage(parsed_prediction)
                predicted_remaining_stages = state_remaining_stages(parsed_prediction)
        except Exception as exc:
            parse_error = str(exc)
            print(parse_error)

        if run_outcome_error_judge:
            outcome_error_judge_result = evaluate_outcome_error_match_quality(
                system_prompt=example.system_prompt,
                user_prompt=example.user_prompt,
                previous_state=example.previous_state,
                action=example.action,
                predicted_label=predicted_last_tool_execution_result,
                predicted_error_message=predicted_error_message,
                gold_label=gold_last_tool_execution_result,
                gold_error_message=example.error_payload,
            )
            outcome_error_judge_evaluated += 1
            outcome_error_judge_score_sum += outcome_error_judge_result["overall_score"]
            outcome_error_judge_matches += int(outcome_error_judge_result["match"])

        exact_matches += int(exact_match)
        parse_successes += int(parse_success)
        llm_judge_score_sum += llm_judge_result["overall_score"]
        llm_judge_matches += int(llm_judge_result["match"])
        if target_mode == WORLD_MODEL_TARGET_STATE:
            for field_path, field_result in llm_judge_result.get("field_scores", {}).items():
                field_score = clamp_score(field_result.get("score", 0.0))
                field_match = bool(field_result.get("match", field_score >= 0.8))
                field_level_judge_score_sum += field_score
                field_level_judge_matches += int(field_match)
                field_level_judge_count += 1
                field_level_judge_score_sums[field_path] = (
                    field_level_judge_score_sums.get(field_path, 0.0) + field_score
                )
                field_level_judge_match_sums[field_path] = (
                    field_level_judge_match_sums.get(field_path, 0) + int(field_match)
                )
                field_level_judge_counts[field_path] = field_level_judge_counts.get(field_path, 0) + 1
        gold_current_stage = state_current_stage(example.state)
        gold_remaining_stages = state_remaining_stages(example.state)
        last_tool_execution_result_match = (
            predicted_last_tool_execution_result == gold_last_tool_execution_result
        )
        current_stage_match = predicted_current_stage == gold_current_stage
        remaining_stages_match = predicted_remaining_stages == gold_remaining_stages
        last_tool_execution_result_matches += int(last_tool_execution_result_match)
        current_stage_matches += int(current_stage_match)
        remaining_stages_matches += int(remaining_stages_match)

        record = {
            "trajectory_id": example.trajectory_id,
            "trajectory_index": example.trajectory_index,
            "interaction_index": example.interaction_index,
            "parse_success": parse_success,
            "exact_match": exact_match,
            "prediction": prediction,
            "gold": gold_target,
            "llm_judge": llm_judge_result,
            "predicted_last_tool_execution_result": predicted_last_tool_execution_result,
            "gold_last_tool_execution_result": gold_last_tool_execution_result,
            "last_tool_execution_result_match": last_tool_execution_result_match,
            "predicted_current_stage": predicted_current_stage,
            "gold_current_stage": gold_current_stage,
            "current_stage_match": current_stage_match,
            "predicted_remaining_stages": predicted_remaining_stages,
            "gold_remaining_stages": gold_remaining_stages,
            "remaining_stages_match": remaining_stages_match,
        }
        if run_outcome_error_judge:
            record["predicted_error_message"] = predicted_error_message
            record["gold_error_message"] = example.error_payload
            record["outcome_with_error_judge"] = outcome_error_judge_result
        if parse_error:
            record["parse_error"] = parse_error

        print(record)

        records.append(record)

    metrics = {
        "sampled_examples": sample_size,
        "exact_match": exact_matches / sample_size,
        "parse_success_rate": parse_successes / sample_size,
        "last_tool_execution_result_accuracy": last_tool_execution_result_matches / sample_size,
        "binary_classification_accuracy": (
            last_tool_execution_result_matches / sample_size
            if is_tool_execution_result_target(target_mode)
            else None
        ),
        "outcome_with_error_judge_accuracy": (
            outcome_error_judge_matches / outcome_error_judge_evaluated
            if run_outcome_error_judge and outcome_error_judge_evaluated > 0
            else None
        ),
        "outcome_with_error_judge_score": (
            outcome_error_judge_score_sum / outcome_error_judge_evaluated
            if run_outcome_error_judge and outcome_error_judge_evaluated > 0
            else None
        ),
        "outcome_with_error_judge_evaluated": (
            outcome_error_judge_evaluated if run_outcome_error_judge else None
        ),
        "llm_judge_score": (
            llm_judge_score_sum / sample_size
            if target_mode == WORLD_MODEL_TARGET_STATE or is_tool_output_target(target_mode)
            else None
        ),
        "llm_judge_match_rate": (
            llm_judge_matches / sample_size
            if target_mode == WORLD_MODEL_TARGET_STATE or is_tool_output_target(target_mode)
            else None
        ),
        "field_level_llm_judge_score": (
            field_level_judge_score_sum / field_level_judge_count
            if target_mode == WORLD_MODEL_TARGET_STATE and field_level_judge_count > 0
            else None
        ),
        "field_level_llm_judge_match_rate": (
            field_level_judge_matches / field_level_judge_count
            if target_mode == WORLD_MODEL_TARGET_STATE and field_level_judge_count > 0
            else None
        ),
        "field_level_llm_judge_scores": (
            {
                field_path: {
                    "average_score": field_level_judge_score_sums[field_path] / field_level_judge_counts[field_path],
                    "match_rate": field_level_judge_match_sums[field_path] / field_level_judge_counts[field_path],
                    "count": field_level_judge_counts[field_path],
                }
                for field_path in sorted(field_level_judge_counts)
            }
            if target_mode == WORLD_MODEL_TARGET_STATE
            else None
        ),
        "current_stage_accuracy": current_stage_matches / sample_size if stage_metrics_enabled else None,
        "remaining_stages_accuracy": remaining_stages_matches / sample_size if stage_metrics_enabled else None,
        "records": records,
    }
    print(metrics["last_tool_execution_result_accuracy"])
    return metrics


def evaluate_agent_replay_via_enterpriseops_gym(
    agent_generator: Any,
    world_model_generator: Any,
    tasks: list[TaskTrajectory],
    max_steps: int,
    internal_thinking_max_iterations: int,
    imagined_trajectory_max_steps: int,
    imagined_trajectory_rollouts: int,
    imagined_rollout_temperature: float,
    imagined_trajectory_selection_strategy: str,
    imagined_trajectory_observation_source: str,
    final_answer_f1_threshold: float,
    world_model_target: str,
    include_error_message_in_target: bool,
    include_stage_in_target: bool,
    include_world_model_history: bool,
    agent_max_observation_chars: int,
    agent_replay_history_budget_chars: int,
    gym_task_configs_dir: Path,
    gym_repo_path: Path | None,
    imagined_trajectory_candidate_actions: int = 3,
    imagined_trajectory_top_k: int = 3,
    revision_lookahead_steps: int = 1,
    revision_imagined_rollouts: int = 1,
    revision_rollout_temperature: float = 0.7,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
    replay_modes: tuple[str, ...] | None = None,
    latent_plan_samples: int = 10,
    latent_plan_elites: int = 3,
    latent_plan_iters: int = 3,
    latent_plan_horizon: int = 5,
    latent_mpc_execute_steps: int = 1,
    latent_plan_temperature: float = 0.7,
    latent_plan_score_margin: float = 0.0,
    latent_plan_diversity_multiplier: int = 1,
    latent_plan_hard_override: bool = False,
    latent_plan_goal_mode: str = "final",
    gate_flat_score_ratio: float = 1.0,
    imagined_rollout_mode: str = "closed_loop",
    beam_plan_trigger: str = "interval",
    beam_plan_critic_failure_prob: float = 0.3,
    beam_plan_critic_stall_prob: float = 0.7,
    beam_plan_critic_min_score: float | None = None,
    beam_plan_critic_max_quiet_steps: int = 0,
    beam_plan_terminal_advice: bool = False,
    beam_plan_terminal_advice_threshold: float = 0.75,
    hier_cem_anchors: int = 8,
    hier_cem_samples: int = 256,
    hier_cem_elites: int = 16,
    hier_cem_iters: int = 3,
    hier_cem_horizon: int = 5,
    hier_cem_init_std: float = 0.2,
    hier_cem_min_std: float = 0.02,
    hier_cem_smoothing: float = 1.0,
    hier_cem_min_elite_agreement: float = 0.5,
    hier_cem_decode_strategy: str = "nearest_anchor",
    hier_cem_decode_max_new_tokens: int = 96,
) -> dict[str, Any]:
    """Run agent replay against EnterpriseOps-Gym tasks via BenchmarkExecutor.

    For each held-out task with a `gym_task_config_name`, loads the matching
    gym task JSON, instantiates `BenchmarkExecutor` with our
    `WorldModelAssistedOrchestrator`, and runs all three modes (baseline,
    revision, imagined). Verifier scores from the gym replace our internal
    final-answer F1 threshold for completion.
    """
    if gym_repo_path is not None and gym_repo_path.exists():
        gym_path_str = str(gym_repo_path)
        if gym_path_str not in sys.path:
            sys.path.insert(0, gym_path_str)

    try:
        from benchmark.executor import BenchmarkExecutor
        from benchmark.models import BenchmarkConfig, LLMConfig
        from src.enterpriseops_gym.enterpriseops_gym_orchestrator import (
            build_world_model_assisted_orchestrator_class,
        )
    except ImportError as exc:
        return {
            "skipped": True,
            "reason": f"enterpriseops_gym_import_failed:{exc}",
            "evaluated_tasks": 0,
            "gym_repo_path": str(gym_repo_path) if gym_repo_path else None,
        }

    if not gym_task_configs_dir.exists():
        return {
            "skipped": True,
            "reason": f"gym_task_configs_dir_missing:{gym_task_configs_dir}",
            "evaluated_tasks": 0,
        }

    OrchestratorClass = build_world_model_assisted_orchestrator_class()

    matched_tasks: list[tuple[TaskTrajectory, Path]] = []
    unmatched_tasks: list[dict[str, Any]] = []
    for task in tasks:
        if not task.gym_task_config_name:
            unmatched_tasks.append(
                {"trajectory_index": task.trajectory_index, "reason": "no_gym_task_config_name"}
            )
            continue
        config_path = gym_task_configs_dir / task.gym_task_config_name
        if not config_path.exists():
            unmatched_tasks.append(
                {
                    "trajectory_index": task.trajectory_index,
                    "reason": "gym_task_config_not_found",
                    "expected_path": str(config_path),
                }
            )
            continue
        matched_tasks.append((task, config_path))

    if not matched_tasks:
        return {
            "skipped": True,
            "reason": "no_tasks_matched_gym_configs",
            "evaluated_tasks": 0,
            "unmatched_tasks": unmatched_tasks,
        }

    # The orchestrator never calls llm_client.invoke_with_tools, but BenchmarkExecutor
    # still constructs an LLMClient during initialize(). Provide a minimal-but-valid
    # config that initializes a LangChain client without making network calls.
    stub_llm_config = LLMConfig(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_api_key="not-used-by-world-model-assisted-orchestrator",
        temperature=0.0,
        max_tokens=1,
    )

    replay_modes = tuple(replay_modes or ("baseline", "revision", "imagined"))
    print(f"[enterpriseops_gym_replay] modes={list(replay_modes)}", flush=True)
    mode_results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in replay_modes}

    async def _run_one_task(task: TaskTrajectory, config_path: Path) -> None:
        with config_path.open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
        raw_config = {k: v for k, v in raw_config.items() if not k.startswith("_")}
        raw_config = override_enterpriseops_gym_mcp_urls(raw_config)
        bench_config = BenchmarkConfig(**raw_config)

        for mode in replay_modes:
            internal_iters = internal_thinking_max_iterations if mode == "revision" else 0
            imagined_steps = imagined_trajectory_max_steps if mode == "imagined" else 0
            orchestrator_kwargs = {
                "max_iterations": max_steps,
                "agent_generator": agent_generator,
                "world_model_generator": world_model_generator,
                "mode": mode,
                "world_model_target": world_model_target,
                "include_error_message_in_target": include_error_message_in_target,
                "include_stage_in_target": include_stage_in_target,
                "include_world_model_history": include_world_model_history,
                "internal_thinking_max_iterations": internal_iters,
                "imagined_trajectory_max_steps": imagined_steps,
                "imagined_trajectory_rollouts": imagined_trajectory_rollouts if mode == "imagined" else 1,
                "imagined_rollout_temperature": imagined_rollout_temperature,
                "imagined_trajectory_selection_strategy": imagined_trajectory_selection_strategy,
                "imagined_trajectory_observation_source": imagined_trajectory_observation_source,
                "latent_plan_samples": latent_plan_samples,
                "latent_plan_elites": latent_plan_elites,
                "latent_plan_iters": latent_plan_iters,
                "latent_plan_horizon": latent_plan_horizon,
                "latent_mpc_execute_steps": latent_mpc_execute_steps,
                "latent_plan_temperature": latent_plan_temperature,
                "latent_plan_score_margin": latent_plan_score_margin,
                "latent_plan_diversity_multiplier": latent_plan_diversity_multiplier,
                "latent_plan_hard_override": latent_plan_hard_override,
                "latent_plan_goal_mode": latent_plan_goal_mode,
                "gate_flat_score_ratio": gate_flat_score_ratio,
                "imagined_rollout_mode": imagined_rollout_mode,
                "beam_plan_trigger": beam_plan_trigger,
                "beam_plan_critic_failure_prob": beam_plan_critic_failure_prob,
                "beam_plan_critic_stall_prob": beam_plan_critic_stall_prob,
                "beam_plan_critic_min_score": beam_plan_critic_min_score,
                "beam_plan_critic_max_quiet_steps": beam_plan_critic_max_quiet_steps,
                "beam_plan_terminal_advice": beam_plan_terminal_advice,
                "beam_plan_terminal_advice_threshold": beam_plan_terminal_advice_threshold,
                "hier_cem_anchors": hier_cem_anchors,
                "hier_cem_samples": hier_cem_samples,
                "hier_cem_elites": hier_cem_elites,
                "hier_cem_iters": hier_cem_iters,
                "hier_cem_horizon": hier_cem_horizon,
                "hier_cem_init_std": hier_cem_init_std,
                "hier_cem_min_std": hier_cem_min_std,
                "hier_cem_smoothing": hier_cem_smoothing,
                "hier_cem_min_elite_agreement": hier_cem_min_elite_agreement,
                "hier_cem_decode_strategy": hier_cem_decode_strategy,
                "hier_cem_decode_max_new_tokens": hier_cem_decode_max_new_tokens,
                "imagined_trajectory_candidate_actions": imagined_trajectory_candidate_actions,
                "imagined_trajectory_top_k": imagined_trajectory_top_k,
                "final_answer_f1_threshold": final_answer_f1_threshold,
                "agent_max_observation_chars": agent_max_observation_chars,
                "agent_replay_history_budget_chars": agent_replay_history_budget_chars,
                "revision_lookahead_steps": revision_lookahead_steps,
                "revision_imagined_rollouts": revision_imagined_rollouts,
                "revision_rollout_temperature": revision_rollout_temperature,
                "state_history_size": state_history_size,
                "system_prompt_max_chars": system_prompt_max_chars,
                "action_max_chars": action_max_chars,
                "initial_state": task.initial_state,
                "initial_canonical_observation": task.initial_canonical_observation,
            }
            executor = BenchmarkExecutor(
                bench_config,
                llm_config=stub_llm_config,
                orchestrator_class=OrchestratorClass,
                orchestrator_kwargs=orchestrator_kwargs,
                config_path=str(config_path),
            )
            try:
                result = await executor.execute_benchmark()
            except Exception as exc:
                mode_results[mode].append(
                    {
                        "trajectory_index": task.trajectory_index,
                        "gym_task_config_name": task.gym_task_config_name,
                        "error": str(exc),
                    }
                )
                continue
            mode_results[mode].append(
                {
                    "trajectory_index": task.trajectory_index,
                    "gym_task_config_name": task.gym_task_config_name,
                    "result": result,
                }
            )

    async def _run_all() -> None:
        for task, config_path in tqdm(matched_tasks, desc="enterpriseops_gym_replay"):
            await _run_one_task(task, config_path)

    asyncio.run(_run_all())

    def _find_metadata_value(obj: Any, key: str, depth: int = 0) -> Any:
        """First value for `key` anywhere in a nested result dict/list (best-effort:
        the orchestrator's get_result_metadata() lands at an executor-dependent path)."""
        if depth > 8:
            return None
        if isinstance(obj, dict):
            if key in obj and isinstance(obj[key], (int, float)):
                return obj[key]
            for value in obj.values():
                found = _find_metadata_value(value, key, depth + 1)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = _find_metadata_value(value, key, depth + 1)
                if found is not None:
                    return found
        return None

    def _summarize_mode(records: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(records)
        errored = sum(1 for r in records if "error" in r)
        completed = 0
        verifier_scores: list[float] = []
        agent_call_counts: list[int] = []
        for record in records:
            calls = _find_metadata_value(record, "agent_call_count")
            if calls is not None:
                agent_call_counts.append(int(calls))
            result = record.get("result")
            if not result:
                continue
            statistics = result.get("statistics") or {}
            completed += int(statistics.get("successful_runs") or 0)
            verifier_level_pass_rate = statistics.get("verifier_level_pass_rate")
            if isinstance(verifier_level_pass_rate, (int, float)):
                verifier_scores.append(float(verifier_level_pass_rate))
                continue
            for run in result.get("runs", []) or []:
                verification_results = run.get("verification_results") or {}
                if isinstance(verification_results, dict):
                    run_passed = bool(verification_results) and all(
                        bool(v.get("passed", False))
                        for v in verification_results.values()
                        if isinstance(v, dict)
                    )
                    if not statistics:
                        completed += int(run_passed)
                verification_summary = run.get("verification_summary") or {}
                pass_rate = verification_summary.get("pass_rate")
                if isinstance(pass_rate, (int, float)):
                    verifier_scores.append(float(pass_rate))
        return {
            "evaluated_tasks": total,
            "errored_tasks": errored,
            "completed_runs": completed,
            "average_verifier_score": (
                sum(verifier_scores) / len(verifier_scores) if verifier_scores else None
            ),
            "total_agent_calls": sum(agent_call_counts) if agent_call_counts else None,
            "average_agent_calls": (
                sum(agent_call_counts) / len(agent_call_counts) if agent_call_counts else None
            ),
            "task_records": records,
        }

    return {
        "gym_task_configs_dir": str(gym_task_configs_dir),
        "matched_tasks": len(matched_tasks),
        "unmatched_tasks": unmatched_tasks,
        "internal_thinking_max_iterations": internal_thinking_max_iterations,
        "imagined_trajectory_max_steps": imagined_trajectory_max_steps,
        "imagined_trajectory_rollouts": imagined_trajectory_rollouts,
        "imagined_rollout_temperature": imagined_rollout_temperature,
        "imagined_trajectory_selection_strategy": imagined_trajectory_selection_strategy,
        "imagined_trajectory_observation_source": imagined_trajectory_observation_source,
        "imagined_trajectory_candidate_actions": imagined_trajectory_candidate_actions,
        "imagined_trajectory_top_k": imagined_trajectory_top_k,
        "revision_lookahead_steps": revision_lookahead_steps,
        "revision_imagined_rollouts": revision_imagined_rollouts,
        "revision_rollout_temperature": revision_rollout_temperature,
        "world_model_target": world_model_target,
        "include_stage_in_target": include_stage_in_target,
        "include_world_model_history": include_world_model_history,
        "replay_modes": list(replay_modes),
        "latent_plan_samples": latent_plan_samples,
        "latent_plan_elites": latent_plan_elites,
        "latent_plan_iters": latent_plan_iters,
        "latent_plan_horizon": latent_plan_horizon,
        "latent_mpc_execute_steps": latent_mpc_execute_steps,
        "latent_plan_temperature": latent_plan_temperature,
        "latent_plan_score_margin": latent_plan_score_margin,
        "latent_plan_diversity_multiplier": latent_plan_diversity_multiplier,
        "latent_plan_hard_override": latent_plan_hard_override,
        "latent_plan_goal_mode": latent_plan_goal_mode,
        "gate_flat_score_ratio": gate_flat_score_ratio,
        "beam_plan_terminal_advice": beam_plan_terminal_advice,
        "beam_plan_terminal_advice_threshold": beam_plan_terminal_advice_threshold,
        "hier_cem_anchors": hier_cem_anchors,
        "hier_cem_samples": hier_cem_samples,
        "hier_cem_elites": hier_cem_elites,
        "hier_cem_iters": hier_cem_iters,
        "hier_cem_horizon": hier_cem_horizon,
        "hier_cem_init_std": hier_cem_init_std,
        "hier_cem_min_std": hier_cem_min_std,
        "hier_cem_smoothing": hier_cem_smoothing,
        "hier_cem_min_elite_agreement": hier_cem_min_elite_agreement,
        "hier_cem_decode_strategy": hier_cem_decode_strategy,
        "hier_cem_decode_max_new_tokens": hier_cem_decode_max_new_tokens,
        **{mode: _summarize_mode(records) for mode, records in mode_results.items()},
    }


def evaluate_agent_replay(
    agent_generator: Any,
    world_model_generator: Any,
    tasks: list[TaskTrajectory],
    max_tasks: int,
    max_steps: int,
    mcp_config_path: Path | None,
    internal_thinking_max_iterations: int,
    imagined_trajectory_max_steps: int,
    imagined_trajectory_rollouts: int,
    imagined_rollout_temperature: float,
    imagined_trajectory_selection_strategy: str,
    imagined_trajectory_observation_source: str,
    imagined_trajectory_candidate_actions: int = 3,
    imagined_trajectory_top_k: int = 3,
    final_answer_f1_threshold: float = 0.5,
    world_model_target: str = "state",
    include_error_message_in_target: bool = False,
    include_stage_in_target: bool = False,
    include_world_model_history: bool = False,
    agent_max_observation_chars: int = 2000,
    agent_replay_history_budget_chars: int = 60000,
    record_replay_trajectories: bool = False,
    gym_task_configs_dir: Path | None = None,
    gym_repo_path: Path | None = None,
    gym_task_split_manifest: Path | None = None,
    revision_lookahead_steps: int = 1,
    revision_imagined_rollouts: int = 1,
    revision_rollout_temperature: float = 0.7,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
    replay_modes: tuple[str, ...] | None = None,
    latent_plan_samples: int = 10,
    latent_plan_elites: int = 3,
    latent_plan_iters: int = 3,
    latent_plan_horizon: int = 5,
    latent_mpc_execute_steps: int = 1,
    latent_plan_temperature: float = 0.7,
    latent_plan_score_margin: float = 0.0,
    latent_plan_diversity_multiplier: int = 1,
    latent_plan_hard_override: bool = False,
    latent_plan_goal_mode: str = "final",
    gate_flat_score_ratio: float = 1.0,
    imagined_rollout_mode: str = "closed_loop",
    beam_plan_trigger: str = "interval",
    beam_plan_critic_failure_prob: float = 0.3,
    beam_plan_critic_stall_prob: float = 0.7,
    beam_plan_critic_min_score: float | None = None,
    beam_plan_critic_max_quiet_steps: int = 0,
    beam_plan_terminal_advice: bool = False,
    beam_plan_terminal_advice_threshold: float = 0.75,
    hier_cem_anchors: int = 8,
    hier_cem_samples: int = 256,
    hier_cem_elites: int = 16,
    hier_cem_iters: int = 3,
    hier_cem_horizon: int = 5,
    hier_cem_init_std: float = 0.2,
    hier_cem_min_std: float = 0.02,
    hier_cem_smoothing: float = 1.0,
    hier_cem_min_elite_agreement: float = 0.5,
    hier_cem_decode_strategy: str = "nearest_anchor",
    hier_cem_decode_max_new_tokens: int = 96,
) -> dict[str, Any]:
    configure_replay_limits(
        observation_chars=agent_max_observation_chars,
        history_budget_chars=agent_replay_history_budget_chars,
    )
    candidate_tasks = tasks
    gym_task_split_filter = None
    if gym_task_configs_dir is not None and gym_task_split_manifest is not None:
        if not gym_task_split_manifest.exists():
            raise SystemExit(f"EnterpriseOps-Gym task split manifest not found: {gym_task_split_manifest}")
        candidate_tasks, gym_task_split_filter = filter_replay_tasks_by_gym_task_split(
            candidate_tasks,
            gym_task_split_manifest,
        )
    selected_tasks = candidate_tasks[: min(max_tasks, len(candidate_tasks))]
    if gym_task_configs_dir is not None:
        replay_result = evaluate_agent_replay_via_enterpriseops_gym(
            agent_generator=agent_generator,
            world_model_generator=world_model_generator,
            tasks=selected_tasks,
            max_steps=max_steps,
            internal_thinking_max_iterations=internal_thinking_max_iterations,
            imagined_trajectory_max_steps=imagined_trajectory_max_steps,
            imagined_trajectory_rollouts=imagined_trajectory_rollouts,
            imagined_rollout_temperature=imagined_rollout_temperature,
            imagined_trajectory_selection_strategy=imagined_trajectory_selection_strategy,
            imagined_trajectory_observation_source=imagined_trajectory_observation_source,
            imagined_trajectory_candidate_actions=imagined_trajectory_candidate_actions,
            imagined_trajectory_top_k=imagined_trajectory_top_k,
            final_answer_f1_threshold=final_answer_f1_threshold,
            world_model_target=world_model_target,
            include_error_message_in_target=include_error_message_in_target,
            include_stage_in_target=include_stage_in_target,
            include_world_model_history=include_world_model_history,
            agent_max_observation_chars=agent_max_observation_chars,
            agent_replay_history_budget_chars=agent_replay_history_budget_chars,
            gym_task_configs_dir=gym_task_configs_dir,
            gym_repo_path=gym_repo_path,
            revision_lookahead_steps=revision_lookahead_steps,
            revision_imagined_rollouts=revision_imagined_rollouts,
            revision_rollout_temperature=revision_rollout_temperature,
            state_history_size=state_history_size,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
            replay_modes=replay_modes,
            latent_plan_samples=latent_plan_samples,
            latent_plan_elites=latent_plan_elites,
            latent_plan_iters=latent_plan_iters,
            latent_plan_horizon=latent_plan_horizon,
            latent_mpc_execute_steps=latent_mpc_execute_steps,
            latent_plan_temperature=latent_plan_temperature,
            latent_plan_score_margin=latent_plan_score_margin,
            latent_plan_diversity_multiplier=latent_plan_diversity_multiplier,
            latent_plan_hard_override=latent_plan_hard_override,
            latent_plan_goal_mode=latent_plan_goal_mode,
            gate_flat_score_ratio=gate_flat_score_ratio,
            imagined_rollout_mode=imagined_rollout_mode,
            beam_plan_trigger=beam_plan_trigger,
            beam_plan_critic_failure_prob=beam_plan_critic_failure_prob,
            beam_plan_critic_stall_prob=beam_plan_critic_stall_prob,
            beam_plan_critic_min_score=beam_plan_critic_min_score,
            beam_plan_critic_max_quiet_steps=beam_plan_critic_max_quiet_steps,
            beam_plan_terminal_advice=beam_plan_terminal_advice,
            beam_plan_terminal_advice_threshold=beam_plan_terminal_advice_threshold,
            hier_cem_anchors=hier_cem_anchors,
            hier_cem_samples=hier_cem_samples,
            hier_cem_elites=hier_cem_elites,
            hier_cem_iters=hier_cem_iters,
            hier_cem_horizon=hier_cem_horizon,
            hier_cem_init_std=hier_cem_init_std,
            hier_cem_min_std=hier_cem_min_std,
            hier_cem_smoothing=hier_cem_smoothing,
            hier_cem_min_elite_agreement=hier_cem_min_elite_agreement,
            hier_cem_decode_strategy=hier_cem_decode_strategy,
            hier_cem_decode_max_new_tokens=hier_cem_decode_max_new_tokens,
        )
        if gym_task_split_filter is not None:
            replay_result["gym_task_split_filter"] = gym_task_split_filter
            replay_result["selected_tasks_after_split_filter"] = len(selected_tasks)
        return replay_result
    if mcp_config_path is None or not mcp_config_path.exists():
        return {
            "skipped": True,
            "reason": "enterprise_mcp_config_required_for_actual_execution_compare",
            "evaluated_tasks": len(selected_tasks),
        }

    async def _run_compare() -> dict[str, Any]:
        AsyncExitStack, MultiServerMCPClient, load_mcp_tools = require_enterprise_runtime()
        with mcp_config_path.open("r", encoding="utf-8") as handle:
            mcp_config = json.load(handle)
        if not mcp_config.get("mcpServers"):
            raise SystemExit(f"No MCP servers configured in {mcp_config_path}")

        baseline_metrics = None
        assisted_metrics = None
        imagined_metrics = None
        connected_servers: list[str] = []
        skipped_servers: list[dict[str, str]] = []
        cleanup_warning = None

        execution_error: BaseException | None = None
        stack = AsyncExitStack()
        stack_opened = False

        try:
            await stack.__aenter__()
            stack_opened = True
            client = MultiServerMCPClient(mcp_config["mcpServers"])
            tools = []
            for server_name in mcp_config["mcpServers"].keys():
                try:
                    session = await asyncio.wait_for(
                        stack.enter_async_context(client.session(server_name)),
                        timeout=15.0,
                    )
                    server_tools = await asyncio.wait_for(
                        load_mcp_tools(session),
                        timeout=10.0,
                    )
                except Exception as exc:
                    print(f"Error occurred while processing server {server_name}: {exc}")
                    skipped_servers.append({"server_name": server_name, "error": str(exc)})
                    continue

                for tool in server_tools:
                    original_name = tool.name
                    original_desc = tool.description
                    tool.name = f"{server_name}_{original_name}"
                    tool.description = (
                        f"Tool for [{server_name.upper()}] related tasks with functionality: {original_desc}"
                    )
                    tool.__dict__["_server_name"] = server_name
                    tool.__dict__["_original_tool_name"] = original_name
                tools.extend(server_tools)
                connected_servers.append(server_name)

            if not tools:
                return {
                    "skipped": True,
                    "reason": "no_mcp_tools_loaded",
                    "mcp_config_path": str(mcp_config_path),
                    "connected_servers": connected_servers,
                    "skipped_servers": skipped_servers,
                    "evaluated_tasks": len(selected_tasks),
                }

            exact_lookup, alias_lookup = build_tool_lookup(tools)
            react_system_prompt = build_react_system_prompt(build_react_tool_descriptions(tools))
            baseline_metrics = await run_actual_mcp_execution_mode(
                mode_name="baseline_actual_mcp",
                use_world_model_internal_thinking=False,
                assistance_strategy="baseline",
                agent_generator=agent_generator,
                world_model_generator=world_model_generator,
                tasks=selected_tasks,
                max_steps=max_steps,
                final_answer_f1_threshold=final_answer_f1_threshold,
                world_model_target=world_model_target,
                include_error_message_in_target=include_error_message_in_target,
                include_stage_in_target=include_stage_in_target,
                include_world_model_history=include_world_model_history,
                record_replay_trajectories=record_replay_trajectories,
                internal_thinking_max_iterations=0,
                imagined_trajectory_max_steps=0,
                imagined_trajectory_rollouts=1,
                imagined_rollout_temperature=0.0,
                imagined_trajectory_selection_strategy="first",
                imagined_trajectory_observation_source="world_model",
                react_system_prompt=react_system_prompt,
                exact_lookup=exact_lookup,
                alias_lookup=alias_lookup,
            )
            assisted_metrics = await run_actual_mcp_execution_mode(
                mode_name="world_model_assisted_actual_mcp",
                use_world_model_internal_thinking=True,
                assistance_strategy="revision",
                agent_generator=agent_generator,
                world_model_generator=world_model_generator,
                tasks=selected_tasks,
                max_steps=max_steps,
                final_answer_f1_threshold=final_answer_f1_threshold,
                world_model_target=world_model_target,
                include_error_message_in_target=include_error_message_in_target,
                include_stage_in_target=include_stage_in_target,
                include_world_model_history=include_world_model_history,
                record_replay_trajectories=record_replay_trajectories,
                internal_thinking_max_iterations=internal_thinking_max_iterations,
                imagined_trajectory_max_steps=0,
                imagined_trajectory_rollouts=1,
                imagined_rollout_temperature=0.0,
                imagined_trajectory_selection_strategy="first",
                imagined_trajectory_observation_source="world_model",
                react_system_prompt=react_system_prompt,
                exact_lookup=exact_lookup,
                alias_lookup=alias_lookup,
                revision_lookahead_steps=revision_lookahead_steps,
                revision_imagined_rollouts=revision_imagined_rollouts,
                revision_rollout_temperature=revision_rollout_temperature,
            )
            imagined_metrics = await run_actual_mcp_execution_mode(
                mode_name="imagined_trajectory_actual_mcp",
                use_world_model_internal_thinking=False,
                assistance_strategy="imagined",
                agent_generator=agent_generator,
                world_model_generator=world_model_generator,
                tasks=selected_tasks,
                max_steps=max_steps,
                final_answer_f1_threshold=final_answer_f1_threshold,
                world_model_target=world_model_target,
                include_error_message_in_target=include_error_message_in_target,
                include_stage_in_target=include_stage_in_target,
                include_world_model_history=include_world_model_history,
                internal_thinking_max_iterations=0,
                imagined_trajectory_max_steps=imagined_trajectory_max_steps,
                imagined_trajectory_rollouts=imagined_trajectory_rollouts,
                imagined_rollout_temperature=imagined_rollout_temperature,
                imagined_trajectory_selection_strategy=imagined_trajectory_selection_strategy,
                imagined_trajectory_observation_source=imagined_trajectory_observation_source,
                imagined_trajectory_candidate_actions=imagined_trajectory_candidate_actions,
                imagined_trajectory_top_k=imagined_trajectory_top_k,
                react_system_prompt=react_system_prompt,
                exact_lookup=exact_lookup,
                alias_lookup=alias_lookup,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            execution_error = exc
        finally:
            if stack_opened:
                try:
                    await stack.aclose()
                except BaseException as exc:
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    cleanup_warning = (
                        "mcp_session_cleanup_error: "
                        + summarize_exception_group(exc)
                    )

        if execution_error is not None:
            raise RuntimeError(
                "Actual MCP execution comparison failed during execution: "
                + summarize_exception_group(execution_error)
                + (f" Cleanup warning: {cleanup_warning}" if cleanup_warning else "")
            ) from execution_error

        if baseline_metrics is None or assisted_metrics is None or imagined_metrics is None:
            raise RuntimeError(
                "Actual MCP execution comparison did not complete successfully."
                + (f" Cleanup warning: {cleanup_warning}" if cleanup_warning else "")
            )

        baseline_calls = baseline_metrics.get("total_tool_calls_taken")
        assisted_calls = assisted_metrics.get("total_tool_calls_taken")
        imagined_calls = imagined_metrics.get("total_tool_calls_taken")
        baseline_completion = baseline_metrics.get("completion_rate")
        assisted_completion = assisted_metrics.get("completion_rate")
        imagined_completion = imagined_metrics.get("completion_rate")
        baseline_avg_calls = baseline_metrics.get("average_tool_calls_per_trajectory")
        assisted_avg_calls = assisted_metrics.get("average_tool_calls_per_trajectory")
        imagined_avg_calls = imagined_metrics.get("average_tool_calls_per_trajectory")

        comparison = {
            "revision_tool_call_reduction": (
                baseline_calls - assisted_calls
                if baseline_calls is not None and assisted_calls is not None
                else None
            ),
            "revision_average_tool_calls_per_trajectory_delta": (
                baseline_avg_calls - assisted_avg_calls
                if baseline_avg_calls is not None and assisted_avg_calls is not None
                else None
            ),
            "revision_completion_rate_delta": (
                assisted_completion - baseline_completion
                if baseline_completion is not None and assisted_completion is not None
                else None
            ),
            "revision_used_fewer_tool_calls": (
                assisted_calls < baseline_calls
                if baseline_calls is not None and assisted_calls is not None
                else None
            ),
            "imagined_tool_call_reduction": (
                baseline_calls - imagined_calls
                if baseline_calls is not None and imagined_calls is not None
                else None
            ),
            "imagined_average_tool_calls_per_trajectory_delta": (
                baseline_avg_calls - imagined_avg_calls
                if baseline_avg_calls is not None and imagined_avg_calls is not None
                else None
            ),
            "imagined_completion_rate_delta": (
                imagined_completion - baseline_completion
                if baseline_completion is not None and imagined_completion is not None
                else None
            ),
            "imagined_used_fewer_tool_calls": (
                imagined_calls < baseline_calls
                if baseline_calls is not None and imagined_calls is not None
                else None
            ),
        }

        return {
            "mcp_config_path": str(mcp_config_path),
            "connected_servers": connected_servers,
            "skipped_servers": skipped_servers,
            "cleanup_warning": cleanup_warning,
            "internal_thinking_max_iterations": internal_thinking_max_iterations,
            "imagined_trajectory_max_steps": imagined_trajectory_max_steps,
            "imagined_trajectory_rollouts": imagined_trajectory_rollouts,
            "imagined_rollout_temperature": imagined_rollout_temperature,
            "imagined_trajectory_selection_strategy": imagined_trajectory_selection_strategy,
            "imagined_trajectory_observation_source": imagined_trajectory_observation_source,
            "baseline_actual_mcp": baseline_metrics,
            "world_model_assisted_actual_mcp": assisted_metrics,
            "imagined_trajectory_actual_mcp": imagined_metrics,
            "comparison": comparison,
        }

    return asyncio.run(_run_compare())


def write_enterprisearena_tasks(path: Path, tasks: list[TaskTrajectory], max_steps: int) -> Path:
    payload = []
    for task in tasks:
        query = task.user_messages[-1] if task.user_messages else ""
        payload.append(
            {
                "query": query,
                "description": f"heldout_trajectory_{task.trajectory_index}",
                "max_steps": max(1, max_steps),
            }
        )
    dump_json(path, payload)
    return path


def maybe_run_enterprisearena(args: argparse.Namespace, tasks: list[TaskTrajectory]) -> dict[str, Any] | None:
    if not args.prepare_enterprisearena_tasks and not args.run_enterprisearena:
        return None

    tasks_path = args.output_dir / "enterprisearena_test_tasks.json"
    write_enterprisearena_tasks(tasks_path, tasks, max_steps=args.agent_max_steps)

    command = [
        sys.executable,
        str(args.enterprise_runner),
        "--model_path",
        args.agent_model or args.model,
        "--tasks",
        str(tasks_path),
    ]
    if args.enterprise_mcp_config:
        command.extend(["--mcp_config", str(args.enterprise_mcp_config)])
    if args.enterprise_output_trajectories:
        command.extend(["--output_trajectories", str(args.enterprise_output_trajectories)])

    result = {
        "tasks_path": str(tasks_path),
        "runner_path": str(args.enterprise_runner),
        "command": command,
        "ran": False,
    }

    if args.run_enterprisearena:
        if not args.enterprise_runner.exists():
            raise SystemExit(f"EnterpriseArena runner not found: {args.enterprise_runner}")
        subprocess.run(command, check=True)
        result["ran"] = True

    return result


def _per_class_metrics(
    gold: list[Any], predicted: list[Any], *, multi_label: bool
) -> dict[str, Any]:
    """Per-class precision/recall/F1 with macro averages over classes that have gold support.

    Same definitions as evaluate_canonical_event_heads (src/finetuning_jepa.py): zero-support
    vocabulary entries are excluded from the macro average so the number reflects prediction
    quality rather than label-space size, and `classes_missed` names supported classes the
    model never emits.
    """
    def as_set(value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            return {str(item) for item in value}
        return {str(value)}

    support: dict[str, int] = {}
    predicted_counts: dict[str, int] = {}
    true_positive: dict[str, int] = {}
    for gold_value, pred_value in zip(gold, predicted):
        gold_set, pred_set = as_set(gold_value), as_set(pred_value)
        for value in gold_set:
            support[value] = support.get(value, 0) + 1
        for value in pred_set:
            predicted_counts[value] = predicted_counts.get(value, 0) + 1
        for value in gold_set & pred_set:
            true_positive[value] = true_positive.get(value, 0) + 1

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    missed: list[str] = []
    for value in sorted(set(support) | set(predicted_counts)):
        tp = true_positive.get(value, 0)
        pred_count = predicted_counts.get(value, 0)
        gold_count = support.get(value, 0)
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / gold_count if gold_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[value] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": gold_count, "predicted": pred_count,
        }
        if gold_count:
            f1s.append(f1)
            recalls.append(recall)
            precisions.append(precision)
            if pred_count == 0:
                missed.append(value)
    metrics: dict[str, Any] = {
        "macro_recall": (sum(recalls) / len(recalls)) if recalls else 0.0,
        "macro_precision": (sum(precisions) / len(precisions)) if precisions else 0.0,
        "classes_with_support": len(f1s),
        "classes_missed": missed,
        "per_class": per_class,
    }
    if multi_label:
        tp_total = sum(true_positive.values())
        pred_total = sum(predicted_counts.values())
        gold_total = sum(support.values())
        micro_precision = tp_total / pred_total if pred_total else 0.0
        micro_recall = tp_total / gold_total if gold_total else 0.0
        metrics["micro_f1"] = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) else 0.0
        )
        metrics["macro_f1"] = (sum(f1s) / len(f1s)) if f1s else 0.0
    return metrics


def score_canonical_event_predictions(
    predictions: list[dict[str, Any] | None],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-field accuracy + macro-F1 for the reduced LLM canonical-event target."""
    total = len(references)
    unparseable = sum(1 for pred in predictions if pred is None)
    per_field: dict[str, Any] = {}
    for field in CANONICAL_EVENT_LLM_TARGET_FIELDS:
        gold = [ref.get(field) for ref in references]
        got = [None if pred is None else pred.get(field) for pred in predictions]
        if field in NUDGE_MULTI_LABEL_FIELDS:
            jaccard, exact = 0.0, 0
            for g, h in zip(gold, got):
                gset = set(g if isinstance(g, list) else [g])
                hset = set(h if isinstance(h, list) else ([] if h is None else [h]))
                union = gset | hset
                jaccard += len(gset & hset) / len(union) if union else 1.0
                exact += int(gset == hset)
            per_field[field] = {
                "mean_jaccard": jaccard / total if total else 0.0,
                "exact_set_match": exact / total if total else 0.0,
                **_per_class_metrics(gold, got, multi_label=True),
            }
            continue
        labels = sorted({g for g in gold if g is not None})
        correct = sum(int(g == h) for g, h in zip(gold, got))
        f1s = []
        for label in labels:
            tp = sum(int(g == label and h == label) for g, h in zip(gold, got))
            fp = sum(int(g != label and h == label) for g, h in zip(gold, got))
            fn = sum(int(g == label and h != label) for g, h in zip(gold, got))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        # Majority-class rate: the score a constant predictor gets. Accuracy above this is the
        # only part that is evidence of learning, given how skewed several fields are.
        majority = max((gold.count(label) for label in labels), default=0)
        per_field[field] = {
            "accuracy": correct / total if total else 0.0,
            "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "majority_class_rate": majority / total if total else 0.0,
            "num_classes": len(labels),
            # Per-class precision/recall/F1 + macro_recall + classes_missed, matching
            # evaluate_canonical_event_heads in src/finetuning_jepa.py so the LLM and JEPA
            # tables can be read side by side. Accuracy alone hides majority-class drift.
            **_per_class_metrics(gold, got, multi_label=False),
        }
    single = list(CANONICAL_EVENT_LLM_TARGET_FIELDS)
    exact_all = sum(
        int(pred is not None and all(pred.get(f) == ref.get(f) for f in CANONICAL_EVENT_LLM_TARGET_FIELDS))
        for pred, ref in zip(predictions, references)
    )
    return {
        "n": total,
        "unparseable_predictions": unparseable,
        "per_field": per_field,
        "mean_single_field_accuracy": (
            sum(per_field[f]["accuracy"] for f in single) / len(single) if single else 0.0
        ),
        "mean_single_field_macro_f1": (
            sum(per_field[f]["macro_f1"] for f in single) / len(single) if single else 0.0
        ),
        "mean_single_field_macro_recall": (
            sum(per_field[f].get("macro_recall", 0.0) for f in single) / len(single) if single else 0.0
        ),
        "accuracy": {f: per_field[f].get("accuracy", per_field[f].get("exact_set_match", 0.0))
                     for f in CANONICAL_EVENT_LLM_TARGET_FIELDS},
        "macro_f1": {f: per_field[f].get("macro_f1", 0.0) for f in CANONICAL_EVENT_LLM_TARGET_FIELDS},
        "macro_recall": {f: per_field[f].get("macro_recall", 0.0) for f in CANONICAL_EVENT_LLM_TARGET_FIELDS},
        "all_field_exact_match": exact_all / total if total else 0.0,
    }


def main_canonical_event(args: argparse.Namespace) -> None:
    """Phase-2 entry point for the reduced beam-field-plus-terminal JSON target.

    Split out from main() because the source canonical-event labels live in JSONL rows with
    their own schema, not in the {messages: [...]} trajectory JSON the rest of the pipeline
    reads. Continue from a phase-1 checkpoint by passing it as the positional model argument.
    """
    # Only honour --train-data-path/--eval-data-path when the user actually passed them; a
    # --trajectory-dataset preset points at trajectory JSON, which this target cannot read.
    train_paths = (
        list(args.train_data_path)
        if getattr(args, "train_data_path_explicit", False)
        else [DEFAULT_CANONICAL_EVENT_TRAIN_JSONL]
    )
    eval_paths = (
        list(args.eval_data_path)
        if getattr(args, "eval_data_path_explicit", False)
        else [DEFAULT_CANONICAL_EVENT_EVAL_JSONL]
    )
    for path in train_paths + eval_paths:
        if not Path(path).exists():
            raise SystemExit(f"canonical-event label file not found: {path}")

    train_examples = extract_canonical_event_examples(train_paths, state_history_size=args.state_history_size)
    test_examples = extract_canonical_event_examples(eval_paths, state_history_size=args.state_history_size)
    if not train_examples:
        raise SystemExit(f"no usable canonical-event rows in {train_paths}")
    print(f"canonical-event examples: train={len(train_examples)} eval={len(test_examples)}")

    def rows_for(examples):
        return [
            build_state_prediction_prompt_completion_row(
                example,
                target_mode=args.world_model_target,
                include_input_history=args.include_world_model_history,
                system_prompt_max_chars=args.world_model_system_prompt_max_chars,
                action_max_chars=args.world_model_action_max_chars,
            )
            for example in examples
        ]

    train_rows, test_rows = rows_for(train_examples), rows_for(test_examples)
    dump_json(
        args.output_dir / "split_manifest.json",
        {
            "source_data_paths": [str(p) for p in train_paths + eval_paths],
            "seed": args.seed,
            "split_strategy": "use_provided_train_eval_files",
            "world_model_target": args.world_model_target,
            "state_history_size": args.state_history_size,
            "include_world_model_history": args.include_world_model_history,
            "base_model": args.model,
            "canonical_event_fields": list(CANONICAL_EVENT_LLM_TARGET_FIELDS),
            "train_examples": len(train_examples),
            "test_examples": len(test_examples),
        },
    )
    dump_jsonl(args.output_dir / "train_examples.jsonl", (asdict(e) for e in train_examples))
    dump_jsonl(args.output_dir / "test_examples.jsonl", (asdict(e) for e in test_examples))
    dump_jsonl(args.output_dir / "train_prompt_completion.jsonl", train_rows)
    dump_jsonl(args.output_dir / "test_prompt_completion.jsonl", test_rows)

    training_metrics = None
    world_model_path = args.world_model_path or str(args.output_dir)
    if not args.skip_training:
        training_metrics = train_world_model(args, train_examples, test_examples)
        world_model_path = str(args.output_dir)

    evaluation_metrics = None
    if not args.skip_eval and test_examples:
        generator = HFTextGenerator(
            world_model_path,
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            disable_chat_template=args.disable_chat_template,
            attn_implementation=args.attn_implementation,
            device_map=args.inference_device_map,
        )
        # The flag documents "<=0 evaluates every test example", but a negative value used to
        # fall through as a negative slice (dropping rows from the end instead).
        requested = int(args.world_model_eval_samples or 0)
        limit = requested if requested > 0 else len(test_examples)
        subset = test_examples[:limit]
        predictions: list[dict[str, Any] | None] = []
        for index, example in enumerate(subset):
            messages = build_state_prediction_chat_messages(
                example,
                target_mode=args.world_model_target,
                include_input_history=args.include_world_model_history,
                system_prompt_max_chars=args.world_model_system_prompt_max_chars,
                action_max_chars=args.world_model_action_max_chars,
            )
            try:
                text = generator.generate_from_messages(messages)
                parsed = parse_jsonish(strip_code_fence(text))
                predictions.append(parsed if isinstance(parsed, dict) else None)
            except Exception:                                    # noqa: BLE001
                predictions.append(None)
            if (index + 1) % 100 == 0:
                print(f"  eval {index + 1}/{len(subset)}", flush=True)
        evaluation_metrics = score_canonical_event_predictions(
            predictions, [e.state for e in subset]
        )
        dump_json(args.output_dir / "evaluation_metrics_canonical_event.json", evaluation_metrics)
        # Same {"row", "gold", "pred"} shape as the JEPA head eval's
        # --canonical-event-dump-predictions, so one report tool can compare LLM and JEPA
        # checkpoints from stored predictions instead of re-generating for every new metric.
        dump_path = args.world_model_eval_dump_predictions or (
            args.output_dir / "eval_predictions_per_example.jsonl"
        )
        with Path(dump_path).open("w", encoding="utf-8") as handle:
            for row_index, (prediction, example) in enumerate(zip(predictions, subset)):
                handle.write(json.dumps({
                    "row": row_index,
                    "gold": {field: example.state.get(field) for field in CANONICAL_EVENT_LLM_TARGET_FIELDS},
                    "pred": (
                        {} if not isinstance(prediction, dict)
                        else {field: prediction.get(field) for field in CANONICAL_EVENT_LLM_TARGET_FIELDS}
                    ),
                    "parsed": isinstance(prediction, dict),
                }, ensure_ascii=False) + "\n")
        print(f"[canonical-event eval] wrote {len(predictions)} per-example predictions to {dump_path}",
              flush=True)

    summary = {
        "training_metrics": training_metrics,
        "evaluation_metrics": evaluation_metrics,
        "enterprisearena": None,
    }
    dump_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    cleanup_distributed()


def configure_llm_call_acceleration(args: argparse.Namespace) -> None:
    """Apply the imagined-rollout speedup toggles to the module globals they gate."""
    global IMAGINED_SINGLE_CALL_STEP, IMAGINED_PARALLEL_ROLLOUTS, LLM_BATCH_PARALLELISM
    global IMAGINED_ROLLOUT_MODE, SAMPLE_TEMPERATURE_LADDER, SAMPLE_TEMPERATURE_LADDER_MAX
    SAMPLE_TEMPERATURE_LADDER = bool(getattr(args, "sample_temperature_ladder", False))
    SAMPLE_TEMPERATURE_LADDER_MAX = float(getattr(args, "sample_temperature_ladder_max", 1.2) or 1.2)
    IMAGINED_SINGLE_CALL_STEP = bool(args.imagined_single_call_step)
    IMAGINED_PARALLEL_ROLLOUTS = bool(args.imagined_parallel_rollouts)
    IMAGINED_ROLLOUT_MODE = str(args.imagined_rollout_mode)
    LLM_BATCH_PARALLELISM = max(1, int(args.llm_batch_parallelism))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_llm_call_acceleration(args)

    if is_canonical_event_target(args.world_model_target):
        # Create the process group before the canonical-event path runs: rank-0-gated steps
        # inside it (the tokenized-dataset cache) need a REAL barrier, and
        # distributed_barrier() silently does nothing when the group is missing.
        setup_distributed()
        return main_canonical_event(args)

    # Rank 0 alone parses the trajectory JSON and extracts the examples; the other ranks wait
    # and read the compact per-example JSONL rank 0 writes. See setup_distributed().
    distributed = setup_distributed()
    if distributed and not is_main_process():
        distributed_barrier()
        train_examples = [
            WorldModelStateExample(**row)
            for row in load_jsonl_rows(args.output_dir / "train_examples.jsonl")
        ]
        test_examples = [
            WorldModelStateExample(**row)
            for row in load_jsonl_rows(args.output_dir / "test_examples.jsonl")
        ]
        test_tasks = [
            rebuild_task_trajectory(row)
            for row in load_json(args.output_dir / "replay_tasks.json")
        ]
        print(f"[rank {distributed_rank()}] loaded {len(train_examples)} train / "
              f"{len(test_examples)} eval examples prepared by rank 0", flush=True)
        return main_after_data_prep(args, train_examples, test_examples, test_tasks)

    # One file at a time, freed before the next; each distinct path extracted once even when it
    # appears in both --train-data-path and --eval-data-path.
    #
    # --skip-eval skips the eval split ENTIRELY -- not just the post-training evaluation. On the
    # large presets the eval list is the same files as train (ADP has no official split), so it
    # is a duplicate of the training data that costs a second full set of examples, prompt rows
    # and tokenized Arrow table on every rank.
    eval_paths = [] if args.skip_eval else list(args.eval_data_path)
    extracted = extract_split_streaming(list(args.train_data_path) + eval_paths, args)
    train_examples, _train_tasks, train_source_rows, train_trajectory_count = gather_split(
        args.train_data_path, extracted
    )
    if args.skip_eval:
        test_examples, test_tasks, test_source_rows, test_trajectory_count = [], [], [], 0
        print("[data] --skip-eval: eval split not extracted", flush=True)
    else:
        test_examples, test_tasks, test_source_rows, test_trajectory_count = gather_split(
            args.eval_data_path, extracted
        )
    del extracted, _train_tasks
    gc.collect()
    print(f"[data] extraction complete: {len(train_examples)} train / {len(test_examples)} eval "
          f"examples (resident {_resident_gb():.1f} GB)", flush=True)
    train_outcome_summary = summarize_example_outcomes(train_examples)
    test_outcome_summary = summarize_example_outcomes(test_examples)
    # Generators, not lists: these rows are a dumped ARTIFACT only -- train_world_model() builds
    # what it needs from the examples -- so materializing every prompt string alongside the
    # examples was pure peak-memory cost.
    def prompt_completion_rows(examples: list[WorldModelStateExample]) -> Iterable[dict[str, Any]]:
        for example in examples:
            yield build_state_prediction_prompt_completion_row(
                example,
                target_mode=args.world_model_target,
                include_error_message=args.include_error_message_in_target,
                include_stage=args.include_stage_in_target,
                include_input_history=args.include_world_model_history,
                system_prompt_max_chars=args.world_model_system_prompt_max_chars,
                action_max_chars=args.world_model_action_max_chars,
            )

    split_manifest = {
        "source_data_paths": [str(p) for p in args.train_data_path]
        + [str(p) for p in args.eval_data_path],
        "seed": args.seed,
        "split_strategy": "use_provided_train_eval_files",
        "train_ratio": None,
        "world_model_target": args.world_model_target,
        "state_history_size": args.state_history_size,
        "include_error_message_in_target": args.include_error_message_in_target,
        "include_stage_in_target": args.include_stage_in_target,
        "include_world_model_history": args.include_world_model_history,
        "gym_task_split_manifest": str(args.gym_task_split_manifest) if args.gym_task_split_manifest else None,
        "imagined_trajectory_candidate_actions": args.imagined_trajectory_candidate_actions,
        "imagined_trajectory_top_k": args.imagined_trajectory_top_k,
        "source_trajectory_count": train_trajectory_count + test_trajectory_count,
        "source_primary_trajectory_count": train_trajectory_count,
        "source_secondary_trajectory_count": test_trajectory_count,
        "train_trajectory_count": train_trajectory_count,
        "eval_trajectory_count": test_trajectory_count,
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "train_example_outcomes": train_outcome_summary,
        "test_example_outcomes": test_outcome_summary,
        "train_prompt_completion_examples": len(train_examples),
        "test_prompt_completion_examples": len(test_examples),
        "train_split_sources": train_source_rows,
        "test_split_sources": [
            {key: value for key, value in row.items() if key != "source_index"}
            for row in test_source_rows
        ],
    }
    dump_json(args.output_dir / "split_manifest.json", split_manifest)
    dump_jsonl(args.output_dir / "train_examples.jsonl", (asdict(example) for example in train_examples))
    dump_jsonl(args.output_dir / "test_examples.jsonl", (asdict(example) for example in test_examples))
    dump_jsonl(args.output_dir / "train_prompt_completion.jsonl", prompt_completion_rows(train_examples))
    dump_jsonl(args.output_dir / "test_prompt_completion.jsonl", prompt_completion_rows(test_examples))
    dump_json(args.output_dir / "replay_tasks.json", [asdict(task) for task in test_tasks])
    del train_source_rows, test_source_rows
    gc.collect()
    # Releases the other ranks, which now read the files written above.
    distributed_barrier()
    return main_after_data_prep(args, train_examples, test_examples, test_tasks)


def main_after_data_prep(
    args: argparse.Namespace,
    train_examples: list[WorldModelStateExample],
    test_examples: list[WorldModelStateExample],
    test_tasks: list[TaskTrajectory],
) -> None:
    """Everything after data preparation: identical on every rank.

    Split out of main() so non-zero ranks can skip the trajectory parsing entirely and enter
    here with the examples rank 0 extracted.
    """
    training_metrics = None
    world_model_path = args.world_model_path or str(args.output_dir)
    if not args.skip_training:
        training_metrics = train_world_model(args, train_examples, test_examples)
        world_model_path = str(args.output_dir)

    evaluation_metrics = None
    if not args.skip_eval:
        if args.world_model_method:
            world_model_generator = build_agent_generator(
                args.world_model_method,
                max_new_tokens=args.max_new_tokens,
                trust_remote_code=args.trust_remote_code,
                dtype=args.dtype,
                disable_chat_template=args.disable_chat_template,
                attn_implementation=args.attn_implementation,
                device_map=args.inference_device_map,
                draft_model_path=args.world_model_draft_model,
                prompt_lookup_num_tokens=args.prompt_lookup_tokens,
            )
        else:
            if looks_like_jepa_world_model_path(world_model_path):
                world_model_generator = JepaTextWorldModelGenerator(
                    world_model_path,
                    max_new_tokens=args.max_new_tokens,
                    trust_remote_code=args.trust_remote_code,
                    dtype=args.dtype,
                )
            else:
                world_model_generator = HFTextGenerator(
                    world_model_path,
                    max_new_tokens=args.max_new_tokens,
                    trust_remote_code=args.trust_remote_code,
                    dtype=args.dtype,
                    disable_chat_template=args.disable_chat_template,
                    attn_implementation=args.attn_implementation,
                    device_map=args.inference_device_map,
                    draft_model_path=args.world_model_draft_model,
                    prompt_lookup_num_tokens=args.prompt_lookup_tokens,
                )
        agent_generator = build_agent_generator(
            args.agent_model or args.model,
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            disable_chat_template=args.disable_chat_template,
            attn_implementation=args.attn_implementation,
            device_map=args.inference_device_map,
            draft_model_path=args.agent_draft_model,
            prompt_lookup_num_tokens=args.prompt_lookup_tokens,
        )
        effective_world_model_target = (
            WORLD_MODEL_TARGET_TOOL_OUTPUT
            if isinstance(world_model_generator, JepaTextWorldModelGenerator)
            else args.world_model_target
        )
        if effective_world_model_target != args.world_model_target:
            print(
                "[world_model] JEPA checkpoints predict raw tool output; "
                f"using target={effective_world_model_target!r} for evaluation and replay.",
                flush=True,
            )
        # world_model_eval = evaluate_world_model_predictions(
        #     sample_limit=args.world_model_eval_samples,
        #     target_mode=args.world_model_target,
        #     include_error_message=args.include_error_message_in_target,
        #     include_stage=args.include_stage_in_target,
        #     include_input_history=args.include_world_model_history,
        # )
        replay_eval = None
        if args.gym_task_configs is not None:
            replay_eval = evaluate_agent_replay(
                agent_generator,
                world_model_generator,
                test_tasks,
                max_tasks=args.max_agent_tasks,
                max_steps=args.agent_max_steps,
                mcp_config_path=resolve_enterprise_mcp_config(
                    args.enterprise_mcp_config,
                    args.enterprise_runner,
                ),
                internal_thinking_max_iterations=args.internal_thinking_max_iters,
                imagined_trajectory_max_steps=args.imagined_trajectory_max_steps,
                imagined_trajectory_rollouts=args.imagined_trajectory_rollouts,
                imagined_rollout_temperature=args.imagined_rollout_temperature,
                imagined_trajectory_selection_strategy=args.imagined_trajectory_selection_strategy,
                imagined_trajectory_observation_source=args.imagined_trajectory_observation_source,
                imagined_trajectory_candidate_actions=args.imagined_trajectory_candidate_actions,
                imagined_trajectory_top_k=args.imagined_trajectory_top_k,
                final_answer_f1_threshold=args.final_answer_f1_threshold,
                world_model_target=args.world_model_target,
                include_error_message_in_target=args.include_error_message_in_target,
                include_stage_in_target=args.include_stage_in_target,
                include_world_model_history=args.include_world_model_history,
                agent_max_observation_chars=args.agent_max_observation_chars,
                agent_replay_history_budget_chars=args.agent_replay_history_budget_chars,
                record_replay_trajectories=args.record_replay_trajectories,
                gym_task_configs_dir=args.gym_task_configs,
                gym_repo_path=args.gym_repo_path,
                gym_task_split_manifest=args.gym_task_split_manifest,
                revision_lookahead_steps=args.revision_lookahead_steps,
                revision_imagined_rollouts=args.revision_imagined_rollouts,
                revision_rollout_temperature=args.revision_rollout_temperature,
                state_history_size=args.state_history_size,
                system_prompt_max_chars=args.world_model_system_prompt_max_chars,
                action_max_chars=args.world_model_action_max_chars,
                replay_modes=args.replay_modes,
                latent_plan_samples=args.latent_plan_samples,
                latent_plan_elites=args.latent_plan_elites,
                latent_plan_iters=args.latent_plan_iters,
                latent_plan_horizon=args.latent_plan_horizon,
                latent_mpc_execute_steps=args.latent_mpc_execute_steps,
                latent_plan_temperature=args.latent_plan_temperature,
                latent_plan_score_margin=args.latent_plan_score_margin,
                latent_plan_diversity_multiplier=args.latent_plan_diversity_multiplier,
                latent_plan_hard_override=args.latent_plan_hard_override,
                latent_plan_goal_mode=args.latent_plan_goal_mode,
                gate_flat_score_ratio=args.gate_flat_score_ratio,
                imagined_rollout_mode=args.imagined_rollout_mode,
                beam_plan_trigger=args.beam_plan_trigger,
                beam_plan_critic_failure_prob=args.beam_plan_critic_failure_prob,
                beam_plan_critic_stall_prob=args.beam_plan_critic_stall_prob,
                beam_plan_critic_min_score=args.beam_plan_critic_min_score,
                beam_plan_critic_max_quiet_steps=args.beam_plan_critic_max_quiet_steps,
                beam_plan_terminal_advice=args.beam_plan_terminal_advice,
                beam_plan_terminal_advice_threshold=args.beam_plan_terminal_advice_threshold,
                hier_cem_anchors=args.hier_cem_anchors,
                hier_cem_samples=args.hier_cem_samples,
                hier_cem_elites=args.hier_cem_elites,
                hier_cem_iters=args.hier_cem_iters,
                hier_cem_horizon=args.hier_cem_horizon,
                hier_cem_init_std=args.hier_cem_init_std,
                hier_cem_min_std=args.hier_cem_min_std,
                hier_cem_smoothing=args.hier_cem_smoothing,
                hier_cem_min_elite_agreement=args.hier_cem_min_elite_agreement,
                hier_cem_decode_strategy=args.hier_cem_decode_strategy,
                hier_cem_decode_max_new_tokens=args.hier_cem_decode_max_new_tokens,
            )
        evaluation_metrics = {
           # "world_model_state_eval": world_model_eval,
        }
        if replay_eval is not None:
            evaluation_metrics["agent_replay_eval"] = replay_eval
        dump_json(args.output_dir / f"evaluation_metrics_test_{args.agent_model}_replay_none.json", evaluation_metrics)
        if replay_eval is not None:
            dump_agent_replay_strategy_records(args.output_dir, replay_eval)

    summary = {
        "training_metrics": training_metrics,
        "evaluation_metrics": evaluation_metrics,
        "enterprisearena": None,
    }
    dump_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # setup_distributed() created the group, so this function owns tearing it down; without it
    # NCCL warns about leaked resources at exit.
    cleanup_distributed()


if __name__ == "__main__":
    main()
