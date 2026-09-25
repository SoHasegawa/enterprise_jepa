#!/usr/bin/env python3
"""JEPA/LeWorldModel-style text world model for EnterpriseOps-Gym.

The model learns latent dynamics over tool-use trajectories:

    z_t       = encoder(context, observations <= t, actions < t)
    z_{t+1}   = encoder(context, observations <= t+1, actions <= t)
    z_hat     = predictor(z_t, encode(action_t), encode(context))

Loss terms:
- latent prediction loss between z_hat and stop-gradient z_{t+1};
- SIGReg-style Gaussian regularization on encoded latents to avoid collapse;
- observation reconstruction CE, decoding the next observation from z_hat.

This is adapted to text EnterpriseOps-Gym trajectories rather than pixels.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import sys

# Run as `uv run python src/finetuning_jepa.py ...` only `src/` lands on sys.path, so the
# repo-wide `from src.<x> import ...` convention needs the root added explicitly before the
# first such import below.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.finetuning import (
    DEFAULT_ENTERPRISEOPS_GYM_TASK_SPLIT_MANIFEST,
    DEFAULT_STATE_HISTORY_SIZE,
    DEFAULT_TRAJECTORIES_DIR,
    WORLD_MODEL_INPUT_HISTORY_SIZE,
    WorldModelStateExample,
    build_agent_generator,
    dump_json,
    dump_jsonl,
    extract_last_tool_execution_result_from_state,
    extract_state_examples,
    extract_state_tool_output,
    load_json,
    normalize_loaded_trajectories,
    normalize_last_tool_execution_result,
    parse_jsonish,
    resolve_gym_task_config_name,
    state_current_stage,
    state_remaining_stages,
    stringify_tool_output,
    strip_code_fence,
)
from src.observation_grounding import (
    benchmark_key_from_path,
    observation_ground_target_ids,
)


ENTERPRISEOPS_GYM_TRAIN_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "enterpriseops_gym_multi_model_world_model_train_trajectories.json"
ENTERPRISEOPS_GYM_EVAL_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "enterpriseops_gym_multi_model_world_model_test_trajectories.json"
ENTERPRISEOPS_GYM_ENTERPRISE_STATE_TRAIN_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "enterpriseops_gym_multi_model_world_model_enterprise_state_train_trajectories.json"
ENTERPRISEOPS_GYM_ENTERPRISE_STATE_EVAL_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "enterpriseops_gym_multi_model_world_model_enterprise_state_test_trajectories.json"
TERMINALBENCH_2_0_MULTI_MODEL_TRAIN_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "terminalbench_2_0_multi_model_world_model_train_trajectories.json"
TERMINALBENCH_2_0_MULTI_MODEL_EVAL_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "terminalbench_2_0_multi_model_world_model_test_trajectories.json"
CRMARENAPRO_MULTI_MODEL_TRAIN_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "crmarenapro_multi_model_world_model_train_trajectories.json"
CRMARENAPRO_MULTI_MODEL_EVAL_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "crmarenapro_multi_model_world_model_test_trajectories.json"
CRMARENAPRO_BASELINE_CRM_AGENT_TRAIN_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "crmarenapro_baseline_crm_agent_train_results.json"
CRMARENAPRO_BASELINE_CRM_AGENT_EVAL_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "crmarenapro_baseline_crm_agent_test_results.json"
TOUCAN_TRAIN_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "toucan_world_model_train_trajectories.json"
TOUCAN_ALL_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "toucan_world_model_trajectories.json"
TOUCAN_EVAL_DATA_PATH = DEFAULT_TRAJECTORIES_DIR / "toucan_world_model_test_trajectories.json"
# Multi-action extract of the uncurated ADP toucan_1_5m (action_count >= 2; 955,223 of
# 1,520,098 trajectories). Built by src/data_preparation/filter_min_action_trajectories.py.
TOUCAN_1_5M_MULTITURN_DATA_PATH = (
    DEFAULT_TRAJECTORIES_DIR / "toucan_1_5m_multiturn_world_model_trajectories.json"
)

# --- Agent Data Protocol (ADP) benchmarks ------------------------------------
# World-model trajectories generated from the ADP standardized dump by
# src/generation/generate_adp_world_model_trajectories.py. These have roles
# system/user/assistant/action/state, where `state` is a lean
# {"state": {"context": {"last_tool_output": ...}}} message inserted wherever
# the source dataset exposed a real tool/environment observation. Some source
# datasets instead fold that observation into a following `user` turn (their
# harness requires strict user/assistant alternation) -- load_trajectory_file()
# synthesizes an equivalent `state` message for those via
# inject_user_observations_as_state() before checking for signal. Files with
# genuinely zero `last_tool_output` after that (e.g. orca_agentinstruct, which
# has no tool calls at all) are skipped at load time rather than fabricating
# observations (see trajectories_have_last_tool_output() below).
# Each ADP benchmark ships as a single combined file (no official train/test
# split), so its preset reuses the same file for train and eval; pass explicit
# --train-data-path / --eval-data-path for a held-out split.
# The ADP inventory lives in src/adp_datasets.py so the causal-LM trainer
# (src/finetuning.py) selects exactly the same benchmarks -- see that module for the
# per-dataset notes on what is included and why.
from src.adp_datasets import (  # noqa: E402
    ADP_TRAJECTORY_DATASETS,
    TOUCAN_ENTERPRISE_DATA_PATH,
    ADP_TRAJECTORY_PATHS,
    WEB_BROWSING_ADP_DATASETS,
    WEB_BROWSING_TRAJECTORY_PATHS,
)
from src.adp_datasets import adp_trajectory_path as _adp_trajectory_path  # noqa: E402
# NB: a path CONSTRUCTOR, not a lookup into ADP_TRAJECTORY_PATHS -- callers below build paths
# for files that are not in the preset list (e.g. the full `toucan_1_5m`).

# The "core" world-model benchmarks: EnterpriseOps-Gym, CRMArenaPro, and
# TerminalBench. TOUCAN is intentionally excluded from `core`; `all` includes the
# UNCURATED ADP variant (toucan_1_5m_multiturn), and the `toucan` preset now also selects
# that multi-action extract rather than the legacy enterprise-curated toucan_* files
# (still reachable as `toucan_curated`).
CORE_TRAIN_DATA_PATHS = [
    CRMARENAPRO_MULTI_MODEL_TRAIN_DATA_PATH,
    ENTERPRISEOPS_GYM_TRAIN_DATA_PATH,
    TERMINALBENCH_2_0_MULTI_MODEL_TRAIN_DATA_PATH,
]
CORE_EVAL_DATA_PATHS = [
    CRMARENAPRO_MULTI_MODEL_EVAL_DATA_PATH,
    ENTERPRISEOPS_GYM_EVAL_DATA_PATH,
    TERMINALBENCH_2_0_MULTI_MODEL_EVAL_DATA_PATH,
]
# Terminal-Bench-free variants. plan.md's Week-2 ablation asks for a
# "without Terminal-Bench pretraining" model to test whether terminal trajectories help
# general enterprise representation learning, are neutral, or cause negative transfer.
TERMINALBENCH_DATA_PATHS = {
    TERMINALBENCH_2_0_MULTI_MODEL_TRAIN_DATA_PATH,
    TERMINALBENCH_2_0_MULTI_MODEL_EVAL_DATA_PATH,
}
CORE_NO_TB_TRAIN_DATA_PATHS = [p for p in CORE_TRAIN_DATA_PATHS if p not in TERMINALBENCH_DATA_PATHS]
CORE_NO_TB_EVAL_DATA_PATHS = [p for p in CORE_EVAL_DATA_PATHS if p not in TERMINALBENCH_DATA_PATHS]
# Imported rather than rebuilt: a local copy would silently drift from the LLM trainer's view of
# the same corpora.
from src.adp_datasets import (  # noqa: E402
    ADP_ALL_EVAL_DATA_PATHS,
    ADP_ALL_TRAIN_DATA_PATHS,
    ADP_ALL_25K_DATA_PATHS,
    ADP_NO_TERMINAL_EVAL_DATA_PATHS,
    ADP_NO_TERMINAL_TRAIN_DATA_PATHS,
    ADP_NO_TERMINAL_25K_DATA_PATHS,
    WEB_BROWSING_25K_PATHS,
    subsampled_25k_path,
    enterprise_tool_calling_paths,
    TOUCAN_1_5M_MULTITURN_DATA_PATH as _TOUCAN_1_5M_MULTITURN,
)
# ADP subsets that are terminal/shell-centric. Excluded alongside Terminal-Bench itself in the
# *_no_terminalbench presets, so the ablation removes the terminal DOMAIN rather than just one
# benchmark file (nemotron_terminal_corpus and litecoder-terminal-style shell corpora would
# otherwise reintroduce exactly the distribution the ablation is meant to remove).
TERMINAL_DOMAIN_ADP_DATASETS = ("nemotron_terminal_corpus",)
TERMINAL_DOMAIN_ADP_PATHS = {
    ADP_TRAJECTORY_PATHS[name] for name in TERMINAL_DOMAIN_ADP_DATASETS if name in ADP_TRAJECTORY_PATHS
}


TRAJECTORY_DATASET_PRESETS: dict[str, tuple[list[Path], list[Path]]] = {
    "enterpriseops_gym": ([ENTERPRISEOPS_GYM_TRAIN_DATA_PATH], [ENTERPRISEOPS_GYM_EVAL_DATA_PATH]),
    "enterpriseops_gym_enterprise_state": ([ENTERPRISEOPS_GYM_ENTERPRISE_STATE_TRAIN_DATA_PATH], [ENTERPRISEOPS_GYM_ENTERPRISE_STATE_EVAL_DATA_PATH]),
    "terminalbench": ([TERMINALBENCH_2_0_MULTI_MODEL_TRAIN_DATA_PATH], [TERMINALBENCH_2_0_MULTI_MODEL_EVAL_DATA_PATH]),
    "terminalbench_2_0_multi_model": ([TERMINALBENCH_2_0_MULTI_MODEL_TRAIN_DATA_PATH], [TERMINALBENCH_2_0_MULTI_MODEL_EVAL_DATA_PATH]),
    "crmarenapro": ([CRMARENAPRO_MULTI_MODEL_TRAIN_DATA_PATH], [CRMARENAPRO_MULTI_MODEL_EVAL_DATA_PATH]),
    "crmarenapro_multi_model": ([CRMARENAPRO_MULTI_MODEL_TRAIN_DATA_PATH], [CRMARENAPRO_MULTI_MODEL_EVAL_DATA_PATH]),
    "crmarenapro_baseline_crm_agent": ([CRMARENAPRO_BASELINE_CRM_AGENT_TRAIN_DATA_PATH], [CRMARENAPRO_BASELINE_CRM_AGENT_EVAL_DATA_PATH]),
    # `toucan` = the MULTI-ACTION extract (action_count >= 2, 955k of 1.52M) -- single- and
    # zero-action records teach no history-conditioned dynamics. The legacy
    # enterprise-curated files remain available as `toucan_curated`, and the unfiltered
    # 1.52M-record file as `toucan_1_5m`.
    # `toucan` = the enterprise-filtered extract. The uncurated multi-action extract of the
    # full 1.5M snapshot stays reachable as `toucan_1_5m_multiturn`.
    # Loaded from the single combined toucan_enterprise_world_model_trajectories.json, like
    # every other ADP benchmark -- the train/test split files are no longer used by the preset.
    "toucan": ([TOUCAN_ENTERPRISE_DATA_PATH], [TOUCAN_ENTERPRISE_DATA_PATH]),
    "toucan_enterprise": ([TOUCAN_ENTERPRISE_DATA_PATH], [TOUCAN_ENTERPRISE_DATA_PATH]),
    "toucan_curated": ([TOUCAN_TRAIN_DATA_PATH], [TOUCAN_EVAL_DATA_PATH]),
    # EnterpriseOps-Gym + CRMArenaPro + TerminalBench (no TOUCAN).
    "core": (list(CORE_TRAIN_DATA_PATHS), list(CORE_EVAL_DATA_PATHS)),
    # Same, minus Terminal-Bench (plan.md's negative-transfer ablation).
    "core_no_terminalbench": (list(CORE_NO_TB_TRAIN_DATA_PATHS), list(CORE_NO_TB_EVAL_DATA_PATHS)),
    # Every ADP benchmark (state-free).
    "adp_all": (list(ADP_ALL_TRAIN_DATA_PATHS), list(ADP_ALL_EVAL_DATA_PATHS)),
    # Core three + every ADP benchmark. TOUCAN arrives via the ADP list's
    # `toucan_1_5m_multiturn` rather than the legacy curated trajectories/toucan_* files --
    # use `toucan_curated` if you explicitly want the enterprise-curated variant back.
    # Very large: cap with --max-train-examples / --max-eval-examples, or pick a single
    # benchmark preset.
    "all": (
        CORE_TRAIN_DATA_PATHS + ADP_ALL_TRAIN_DATA_PATHS,
        CORE_EVAL_DATA_PATHS + ADP_ALL_EVAL_DATA_PATHS,
    ),
    # `all` minus Terminal-Bench AND the terminal-domain ADP corpora, so the ablation removes
    # the terminal distribution rather than one file (see TERMINAL_DOMAIN_ADP_DATASETS).
    "all_no_terminalbench": (
        CORE_NO_TB_TRAIN_DATA_PATHS + ADP_NO_TERMINAL_TRAIN_DATA_PATHS,
        CORE_NO_TB_EVAL_DATA_PATHS + ADP_NO_TERMINAL_EVAL_DATA_PATHS,
    ),
    # `*_25k`: identical benchmark coverage, with toucan_enterprise / dolci / toolmind replaced
    # by their 25k seeded subsets (11.6 GB -> 1.86 GB). Cuts extraction time, the
    # jepa_train_examples.jsonl cache (140 GB at full size) and per-rank RAM, which is what
    # forces 4-rank runs onto the 2 TB partition.
    "adp_all_25k": (list(ADP_ALL_25K_DATA_PATHS), list(ADP_ALL_25K_DATA_PATHS)),
    "all_25k": (
        CORE_TRAIN_DATA_PATHS + ADP_ALL_25K_DATA_PATHS,
        CORE_EVAL_DATA_PATHS + ADP_ALL_25K_DATA_PATHS,
    ),
    "all_no_terminalbench_25k": (
        CORE_NO_TB_TRAIN_DATA_PATHS + ADP_NO_TERMINAL_25K_DATA_PATHS,
        CORE_NO_TB_EVAL_DATA_PATHS + ADP_NO_TERMINAL_25K_DATA_PATHS,
    ),
    # Downstream-targeted mixtures: the enterprise benchmarks plus tool-calling ADP corpora,
    # with and without the SWE/code arm. Zero-yield and web/terminal corpora are excluded from
    # both (see ENTERPRISE_TOOL_CALLING_ADP_DATASETS). The `_25k` variants swap the three heavy
    # corpora for their subsets, which is what makes running both arms affordable.
    "enterprise_tool_calling": (
        CORE_NO_TB_TRAIN_DATA_PATHS + enterprise_tool_calling_paths(include_swe=False, subsampled=False),
        CORE_NO_TB_EVAL_DATA_PATHS + enterprise_tool_calling_paths(include_swe=False, subsampled=False),
    ),
    "enterprise_tool_calling_plus_swe": (
        CORE_NO_TB_TRAIN_DATA_PATHS + enterprise_tool_calling_paths(include_swe=True, subsampled=False),
        CORE_NO_TB_EVAL_DATA_PATHS + enterprise_tool_calling_paths(include_swe=True, subsampled=False),
    ),
    "enterprise_tool_calling_25k": (
        CORE_NO_TB_TRAIN_DATA_PATHS + enterprise_tool_calling_paths(include_swe=False, subsampled=True),
        CORE_NO_TB_EVAL_DATA_PATHS + enterprise_tool_calling_paths(include_swe=False, subsampled=True),
    ),
    "enterprise_tool_calling_plus_swe_25k": (
        CORE_NO_TB_TRAIN_DATA_PATHS + enterprise_tool_calling_paths(include_swe=True, subsampled=True),
        CORE_NO_TB_EVAL_DATA_PATHS + enterprise_tool_calling_paths(include_swe=True, subsampled=True),
    ),
}
# One selectable preset per ADP benchmark (e.g. --trajectory-dataset swe-smith).
for _adp_name, _adp_path in ADP_TRAJECTORY_PATHS.items():
    TRAJECTORY_DATASET_PRESETS[_adp_name] = ([_adp_path], [_adp_path])
    _sub = subsampled_25k_path(_adp_name)
    if _sub != _adp_path:
        TRAJECTORY_DATASET_PRESETS[f"{_adp_name}_25k"] = ([_sub], [_sub])
# The FULL uncurated TOUCAN (26GB, incl. zero-/single-action records) stays selectable by
# name even though `all`/`adp_all` use the filtered multiturn variant above.
TRAJECTORY_DATASET_PRESETS["toucan_1_5m_multiturn"] = (
    [_TOUCAN_1_5M_MULTITURN], [_TOUCAN_1_5M_MULTITURN]
)
_toucan_full = _adp_trajectory_path("toucan_1_5m")
TRAJECTORY_DATASET_PRESETS["toucan_1_5m"] = ([_toucan_full], [_toucan_full])


# --- canonical_event_state / nudge classification heads ---------------------
# JSONL label files (one row per action, independent of the trajectory JSON
# format) produced outside this repo: each row carries system_prompt,
# task_prompt, action, input_history, plus `canonical_event_state` (a
# descriptive classification of what the action's outcome looked like) and
# `nudge` (an actionable pre-execution guidance signal). See
# --train-canonical-event-heads-only.
# Field names live in src/canonical_event_schema.py so the causal-LM trainer
# (--world-model-target canonical_event_with_nudge) predicts exactly this label space.
from src.canonical_event_schema import (  # noqa: E402
    CANONICAL_EVENT_ALL_FIELDS,
    CANONICAL_EVENT_SINGLE_LABEL_FIELDS,
    CANONICAL_EVENT_STATE_FIELDS,
    NUDGE_MULTI_LABEL_FIELDS,
    NUDGE_SINGLE_LABEL_FIELDS,
)


def resolve_canonical_event_field_sets(heads_mode: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(single_label_fields, multi_label_fields) actually trained, per --canonical-event-heads.

    "beam_plan" keeps only the single-label fields the planner's scorer consumes --
    SCORED_SINGLE_FIELDS in src/canonical_event_scoring.py (imported there, one source of
    truth) -- dropping action_type / object_type (descriptive), risk_signal (~96%% "none", not
    discriminative), information_gain (information_sufficiency classifies more accurately),
    recommended_abstract_action and missing_information_type (weak next-state prediction signal).
    The dropped heads are not constructed at all: the vocab (and hence
    canonical_event_vocab_sizes, the head ModuleDict, the loss, and the eval) is restricted to
    this set, so replay reconstructs the same reduced head set from canonical_event_vocab.json.
    """
    if heads_mode == "beam_plan":
        from src.canonical_event_scoring import SCORED_SINGLE_FIELDS

        single = tuple(field for field in CANONICAL_EVENT_SINGLE_LABEL_FIELDS if field in SCORED_SINGLE_FIELDS)
        return single, tuple()
    return CANONICAL_EVENT_SINGLE_LABEL_FIELDS, NUDGE_MULTI_LABEL_FIELDS

from src.canonical_event_schema import (  # noqa: E402
    DEFAULT_CANONICAL_EVENT_EVAL_JSONL,
    DEFAULT_CANONICAL_EVENT_TRAIN_JSONL,
)

# Maps the classifier's execution_status vocab onto the world-model state's
# ternary context.last_tool_execution_result (1 success / 0 stagnation / -1 failure).
CANONICAL_EVENT_EXECUTION_STATUS_TO_TERNARY: dict[str, int | None] = {
    "success": 1,
    "partial": 0,
    "no_op": 0,
    "failure": -1,
    "unknown": None,
}
CANONICAL_EVENT_CLASSIFIER_STATE_SCHEMA = "ewm_canonical_event_classifier_state_v1"
CANONICAL_EVENT_CLASSIFIER_OBSERVATION_SCHEMA = "ewm_canonical_event_classifier_observation_v1"


def load_canonical_event_vocab(checkpoint_path: str | Path) -> dict[str, list[str]]:
    """Load the per-field label vocabulary saved next to a canonical-event head
    checkpoint (``canonical_event_vocab.json``).

    Returns ``{}`` when the file is absent so callers can treat "no vocab" as
    "this checkpoint has no usable canonical-event heads".
    """
    path = Path(checkpoint_path) / "canonical_event_vocab.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    if not isinstance(vocab, dict):
        return {}
    return {
        str(field): [str(value) for value in values]
        for field, values in vocab.items()
        if isinstance(values, list)
    }


def decode_canonical_event_logits(
    logits: dict[str, torch.Tensor],
    vocab: dict[str, list[str]],
    *,
    multi_label_threshold: float = 0.5,
) -> dict[str, Any]:
    """Turn a single example's raw head logits into human-readable labels.

    Single-label heads return the argmax category string; the multi-label
    ``missing_information_type`` head returns every category whose sigmoid clears
    ``multi_label_threshold``, falling back to the top-1 category so the field is
    never empty. Row 0 is taken (replay predicts one action at a time).
    """
    labels: dict[str, Any] = {}
    for field, field_logits in logits.items():
        values = vocab.get(field) or []
        if not values:
            continue
        row = field_logits[0].detach().float().cpu()
        if field in NUDGE_MULTI_LABEL_FIELDS:
            probabilities = torch.sigmoid(row).tolist()
            picked = [
                values[i]
                for i, probability in enumerate(probabilities)
                if i < len(values) and probability >= multi_label_threshold
            ]
            if not picked:
                picked = [values[int(row.argmax())]]
            labels[field] = picked
        else:
            index = int(row.argmax())
            labels[field] = values[index] if index < len(values) else "unknown"
    return labels


def reconstruct_state_from_canonical_event_labels(
    labels: dict[str, Any],
    *,
    tool_name: str | None = None,
) -> dict[str, Any]:
    """Reconstruct a world-model state JSON from classification-head labels.

    The classifier does not emit a raw tool observation; it predicts a canonical
    description of the action's outcome (``execution_status``, ``object_type``,
    ``risk_signal``, ...) plus a pre-execution ``nudge``. This packs those into
    the standard ``{"state": {"context": ..., ...}}`` envelope the replay code
    understands, mapping ``execution_status`` onto the ternary
    ``context.last_tool_execution_result`` so ``summarize_state_for_planning()``
    and the success/stagnation bookkeeping keep working, while surfacing the full
    ``canonical_event_state`` / ``nudge`` blocks for the agent to read.
    """
    canonical_event_state = {field: labels.get(field) for field in CANONICAL_EVENT_STATE_FIELDS}
    nudge: dict[str, Any] = {field: labels.get(field) for field in NUDGE_SINGLE_LABEL_FIELDS}
    for field in NUDGE_MULTI_LABEL_FIELDS:
        value = labels.get(field)
        nudge[field] = list(value) if isinstance(value, list) else ([] if value is None else [value])

    label = canonical_event_execution_label(labels.get("execution_status"))
    error_signature = labels.get("error_signature")
    error_message = ""
    if label in (-1, 0):
        parts: list[str] = []
        execution_status = labels.get("execution_status")
        if execution_status:
            parts.append(f"execution_status={execution_status}")
        if error_signature and str(error_signature).strip().lower() not in {"none", "unknown"}:
            parts.append(f"error_signature={error_signature}")
        error_message = "; ".join(parts)

    context: dict[str, Any] = {"last_tool_execution_result": label}
    if tool_name:
        context["last_tool_name"] = tool_name
    if error_message:
        context["error_message"] = error_message

    return {
        "schema": CANONICAL_EVENT_CLASSIFIER_STATE_SCHEMA,
        "state": {
            "context": context,
            "canonical_event_state": canonical_event_state,
            "nudge": nudge,
        },
    }


def canonical_event_execution_label(execution_status: Any) -> int | None:
    if execution_status is None:
        return None
    return CANONICAL_EVENT_EXECUTION_STATUS_TO_TERNARY.get(str(execution_status).strip().lower())


def build_canonical_event_observation_payload(
    labels: dict[str, Any],
    *,
    tool_name: str | None = None,
) -> dict[str, Any]:
    """Compact observation the agent sees for an imagined step: the reconstructed
    outcome plus a boolean success read off ``execution_status``.
    """
    del tool_name
    label = canonical_event_execution_label(labels.get("execution_status"))
    return {
        "schema": CANONICAL_EVENT_CLASSIFIER_OBSERVATION_SCHEMA,
        "tool_outcome": {
            "success": (label == 1) if label is not None else None,
            "label": label,
            "execution_status": labels.get("execution_status"),
        },
        "canonical_event_state": {field: labels.get(field) for field in CANONICAL_EVENT_STATE_FIELDS},
        "nudge": {
            **{field: labels.get(field) for field in NUDGE_SINGLE_LABEL_FIELDS},
            **{
                field: (labels.get(field) if isinstance(labels.get(field), list) else [])
                for field in NUDGE_MULTI_LABEL_FIELDS
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a text JEPA world model on normalized tool-use trajectories.")
    parser.add_argument(
        "--model",
        default="google/t5gemma-2-270m-270m",
        help=(
            "Backbone checkpoint. Defaults to Google T5Gemma 2 270M-270M for "
            "seq2seq training. For encoder-only latent planning use, pass an "
            "embedding/encoder model such as Qwen/Qwen3-Embedding-8B with "
            "--backbone-type encoder."
        ),
    )
    parser.add_argument(
        "--backbone-type",
        choices=("seq2seq", "encoder"),
        default="seq2seq",
        help="Backbone architecture: seq2seq supports optional reconstruction; encoder is latent-only.",
    )
    parser.add_argument(
        "--pooling",
        choices=("mean", "last_token"),
        default="mean",
        help=(
            "How to pool the encoder's token states into a latent. `mean` is the "
            "default; use `last_token` for decoder-style embedding backbones such "
            "as Qwen3-Embedding, which are trained for last-token (EOS) pooling."
        ),
    )
    parser.add_argument(
        "--trajectory-dataset",
        choices=sorted(TRAJECTORY_DATASET_PRESETS.keys()),
        default="enterpriseops_gym",
        help=(
            "Trajectory dataset preset for JEPA training. `core` is EnterpriseOps-Gym, "
            "CRMArenaPro, and TerminalBench (no TOUCAN; use the `toucan` preset for that); "
            "`adp_all` is every Agent Data Protocol benchmark; `all` concatenates "
            "core + TOUCAN + all ADP benchmarks. Each ADP benchmark is also selectable by "
            "name (e.g. swe-smith, code_feedback). ADP trajectory files with no "
            "state-message `last_tool_output` signal at all (e.g. orca_agentinstruct) "
            "are skipped at load time rather than training on fabricated "
            "observations. `all`/`adp_all` are very large -- cap with "
            "--max-train-examples/--max-eval-examples. Explicit "
            "--train-data-path / --eval-data-path override the preset."
        ),
    )
    parser.add_argument("--train-data-path", type=Path, nargs="+", default=None, help="One or more train trajectory JSON paths. Defaults to --trajectory-dataset.")
    parser.add_argument("--eval-data-path", type=Path, nargs="+", default=None, help="One or more eval trajectory JSON paths. Defaults to --trajectory-dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data_jepa"))
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=120,
        help=(
            "torch.distributed collective timeout. Only rank 0 loads/extracts trajectories "
            "for --trajectory-dataset presets (other ranks idle at a barrier), so with a large, "
            "uncapped preset like `all` this may need to be raised further; the default 30 "
            "minutes is often too short."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    # 8192 (was 2048): state texts measured at p99 ~5.5k tokens / max ~8k on the canonical
    # JSONL, so 2048 truncated roughly a third of latent-target states badly enough to cut the
    # newest step entirely (see --truncate-states-keep-newest). The value is recorded in the
    # checkpoint manifest, so replay of OLD checkpoints still uses their trained 2048.
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--max-action-length", type=int, default=512)
    parser.add_argument("--max-observation-length", type=int, default=512)
    parser.add_argument("--max-goal-length", type=int, default=512)
    parser.add_argument("--state-history-size", type=int, default=DEFAULT_STATE_HISTORY_SIZE)
    parser.add_argument("--max-train-examples", type=int, default=0, help="0 keeps all examples.")
    parser.add_argument("--max-eval-examples", type=int, default=0, help="0 keeps all examples.")
    parser.add_argument("--latent-dim", type=int, default=0, help="0 uses the backbone hidden size.")
    parser.add_argument(
        "--latent-type",
        choices=("continuous", "categorical"),
        default="continuous",
        help=(
            "Latent representation. `continuous` uses real-valued vectors (MSE/cosine "
            "prediction + SIGReg). `categorical` uses a Dreamer v2/v3-style stack of "
            "discrete categoricals with straight-through sampling and a KL-balanced loss."
        ),
    )
    parser.add_argument(
        "--latent-categoricals",
        type=int,
        default=32,
        help="Number of categorical variables (groups) when --latent-type=categorical. Latent dim becomes categoricals*classes.",
    )
    parser.add_argument(
        "--latent-classes",
        type=int,
        default=32,
        help="Number of classes per categorical when --latent-type=categorical.",
    )
    parser.add_argument(
        "--latent-unimix",
        type=float,
        default=0.01,
        help="Dreamer v3 unimix: fraction of uniform mixed into categorical class probabilities.",
    )
    parser.add_argument(
        "--kl-loss-coeff",
        type=float,
        default=1.0,
        help="Coefficient for the categorical KL latent loss (replaces --latent-loss-coeff when --latent-type=categorical).",
    )
    parser.add_argument(
        "--kl-balance",
        type=float,
        default=0.8,
        help="Dreamer KL balancing: weight on the dynamics term KL(sg(posterior)||prior); 1-value weights the representation term.",
    )
    parser.add_argument(
        "--kl-free-nats",
        type=float,
        default=1.0,
        help="Dreamer v3 free bits: KL terms are clamped to at least this many nats before averaging.",
    )
    parser.add_argument("--predictor-hidden-multiplier", type=float, default=4.0)
    parser.add_argument("--predictor-dropout", type=float, default=0.1)
    parser.add_argument(
        "--predictor-arch", choices=("mlp", "transformer"), default="mlp",
        help=(
            "Predictor F. `mlp` (default) is the concat-MLP over [z_cur, z_act, z_ctx, z_goal]. "
            "`transformer` is the LeWorldModel-style predictor: causal self-attention over N+1 "
            "representation tokens -- one encoding system prompt + task prompt, then the N most "
            "recent tool outputs -- with the action injected at every layer through "
            "zero-initialized AdaLN (no action tokens, no action history), followed by an "
            "encoder-style Linear+LayerNorm projector. The tool-output history comes from the "
            "same per-step log that --recurrent-state-init uses, and is emitted whenever either "
            "option is on."
        ),
    )
    parser.add_argument("--predictor-transformer-layers", type=int, default=6)
    parser.add_argument("--predictor-transformer-heads", type=int, default=16)
    parser.add_argument(
        "--predictor-transformer-dim", type=int, default=0,
        help=(
            "Width of the transformer predictor. 0 uses latent_dim. Must be divisible by "
            "--predictor-transformer-heads. LeWorldModel's ~10M-parameter predictor is 6 layers "
            "x 16 heads at width 384; at a 1024-d latent the same depth is ~75M, so set this "
            "explicitly if parameter count matters."
        ),
    )
    parser.add_argument(
        "--predictor-transformer-mlp-ratio", type=float, default=4.0,
        help="Hidden width of each block's MLP, as a multiple of --predictor-transformer-dim.",
    )
    parser.add_argument(
        "--predictor-history-length", type=int, default=0,
        help=(
            "N: tool-output representations the transformer predictor attends over, INCLUDING "
            "the current one. The input sequence is N+1 tokens -- these plus one token encoding "
            "system prompt + task prompt at position 0. 0 uses the logged history depth + 1 "
            f"(= {WORLD_MODEL_INPUT_HISTORY_SIZE + 1}). Sizes the position embedding, so it is "
            "architecture-affecting."
        ),
    )
    parser.add_argument("--memory-tokens", type=int, default=8, help="Pseudo encoder tokens decoded from predicted latent.")
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=0,
        help="Deprecated and ignored. Evaluation runs once after training completes.",
    )
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help=(
            "Maximum number of periodic checkpoint-<step> directories to keep on disk. When a new "
            "checkpoint is saved, the oldest are deleted so only this many most-recent ones remain "
            "(e.g. 2 keeps checkpoint-1000 and checkpoint-1500 after saving 1500, deleting 500). "
            "The final run directory written at the end of training is separate and never pruned. "
            "Set to 0 to keep all checkpoints."
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help=(
            "Warm-start regular JEPA training from a saved checkpoint directory (e.g. "
            ".../checkpoint-1500 or a finished run dir). Loads the adapter weights "
            "(predictor/projector/heads); the backbone is loaded from the checkpoint's backbone/ "
            "subdir if present, otherwise from --model. This restores model WEIGHTS ONLY, not the "
            "optimizer/scheduler/step (a weights-only warm start; the LR schedule restarts). "
            "Note: periodic checkpoint-<step> dirs are adapter-only (no backbone/), so if the "
            "backbone was not frozen, resume from a full run directory or train with --freeze-backbone."
        ),
    )
    parser.add_argument("--latent-loss-coeff", type=float, default=1.0)
    parser.add_argument(
        "--latent-loss-type",
        choices=("mse_cosine", "smooth_l1", "smooth_l1_cosine"),
        default="smooth_l1_cosine",
        help=(
            "Latent prediction loss form. 'mse_cosine' is the legacy MSE+cosine objective; "
            "'smooth_l1'/'smooth_l1_cosine' replace the MSE term with a Smooth L1 (Huber) loss "
            "(more robust to the outlier spikes seen in JEPA training; cf. NextLat / LSE-MTP). "
            "Default is smooth_l1_cosine (Huber + cosine); pass mse_cosine to reproduce old runs."
        ),
    )
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0, help="Huber transition point (beta) for --latent-loss-type smooth_l1*.")
    parser.add_argument(
        "--latent-delta-prediction",
        action="store_true",
        help=(
            "Delta encoding / Δz supervision (continuous latents only). The predictor's output "
            "is interpreted as the CHANGE Δz rather than the absolute next latent: "
            "z_pred = z_current + Δz (residual parametrization; the predictor's trailing "
            "LayerNorm is dropped so small deltas are representable), and the cosine term of the "
            "latent loss is computed between predicted and true deltas instead of absolute "
            "latents -- consecutive tool-use states share most content, so absolute-space cosine "
            "is dominated by the carried-over state; delta-space cosine supervises the direction "
            "of what actually changed. The point-wise (Smooth L1/MSE) term is unchanged (it is "
            "mathematically identical in either space). Applies to the recursive multi-step "
            "rollout and Fast-LeWM automatically."
        ),
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=1,
        help=(
            "Number of future steps supervised per example. 1 = single next-step prediction "
            "(default; unchanged code path). K>1 recursively rolls the predictor forward feeding "
            "teacher-forced future actions taken from the later steps of the same trajectory, and "
            "supervises each predicted latent against the stop-gradient true encoded next-state "
            "(NextLat/LSE-MTP-style). Multi-step supervision applies only to the point-wise "
            "prediction term (MSE or Smooth L1, no cosine) and to SigReg. Continuous latents only."
        ),
    )
    parser.add_argument(
        "--obs-token-ground-coeff",
        type=float,
        default=0.0,
        help=(
            "Option-C observation-token grounding (ECHO-style). >0 trains a small "
            "autoregressive head on z_pred to predict the first-N tokens of the isolated "
            "environment-output span (per-benchmark split in src/observation_grounding.py), "
            "grounding the world-model latent on unlabeled trajectories. 0 disables it."
        ),
    )
    parser.add_argument(
        "--obs-ground-max-tokens",
        type=int,
        default=None,
        help="Global cap on the number of first-N observation tokens supervised (default: per-benchmark N).",
    )
    parser.add_argument(
        "--obs-ground-decoder-dim", type=int, default=256,
        help="Encoder-only backbones only: internal model dim of the obs-grounding causal Transformer.",
    )
    parser.add_argument(
        "--obs-ground-decoder-layers", type=int, default=4,
        help="Encoder-only backbones only: number of layers in the obs-grounding causal Transformer.",
    )
    parser.add_argument(
        "--obs-ground-decoder-heads", type=int, default=4,
        help="Encoder-only backbones only: attention heads in the obs-grounding causal Transformer.",
    )
    parser.add_argument(
        "--obs-ground-decoder-memory-tokens", type=int, default=8,
        help="Encoder-only backbones only: memory slots z_pred is expanded into for the "
        "obs-grounding Transformer to cross-attend over.",
    )
    # Latent action-head (opt-in; both default 0 = disabled). See the latent tool-action MPC plan.
    parser.add_argument(
        "--tool-select-loss-coeff",
        type=float,
        default=0.0,
        help="P1: weight on the retrieval-style tool-selection head p(tool | z_current, z_goal). 0 disables it.",
    )
    parser.add_argument(
        "--action-encoder-loss-coeff",
        type=float,
        default=0.0,
        help=(
            "P2: weight on the fast structured-action encoder g(tool, args) distilled to match the "
            "backbone-encoded z_action, so search can encode candidate actions without the backbone. 0 disables it."
        ),
    )
    parser.add_argument("--action-head-embed-dim", type=int, default=256, help="Tool/action embedding dim for the action-head.")
    parser.add_argument(
        "--action-decoder-loss-coeff",
        type=float,
        default=0.0,
        help=(
            "Weight on the learnable action decoder D: reconstructs an action's own tokens from a "
            "randomly-noised z_action (reconstruction/cycle-consistency at noise~0, 'prior-sample' "
            "robustness training at noise>0). 0 disables it -- hierarchical_cem_plan then falls back "
            "to nearest-anchor decode, which cannot decode a genuinely interpolated latent."
        ),
    )
    parser.add_argument(
        "--action-decoder-max-noise-std",
        type=float,
        default=0.1,
        help=(
            "Max std of the Gaussian noise added to z_action before decoding (sampled per-example "
            "from Uniform(0, this)), matching the neighborhood a CEM proposal samples around a real "
            "anchor at inference time."
        ),
    )
    parser.add_argument(
        "--action-decoder-dim", type=int, default=256,
        help="Encoder-only backbones only: internal model dim of the action-decoder causal Transformer.",
    )
    parser.add_argument(
        "--action-decoder-layers", type=int, default=4,
        help="Encoder-only backbones only: number of layers in the action-decoder causal Transformer.",
    )
    parser.add_argument(
        "--action-decoder-heads", type=int, default=4,
        help="Encoder-only backbones only: attention heads in the action-decoder causal Transformer.",
    )
    parser.add_argument(
        "--action-decoder-memory-tokens", type=int, default=8,
        help="Encoder-only backbones only: memory slots z_action is expanded into for the "
        "action-decoder Transformer to cross-attend over.",
    )
    parser.add_argument(
        "--action-sigreg-coeff",
        type=float,
        default=0.0,
        help=(
            "Weight on SigReg applied to z_action (maps to the design's L_action_prior): pulls the "
            "action latent geometry toward isotropic Gaussian so a Gaussian CEM proposal distribution "
            "is a reasonable model of it. 0 disables it."
        ),
    )
    parser.add_argument(
        "--canonical-event-recognition-probe",
        action="store_true",
        help=(
            "Recognition upper-bound probe for --train-canonical-event-heads-only: feed the head the "
            "actual observation latent (z_observation) instead of the predicted z_pred, to measure the "
            "recognition ceiling vs prediction. Requires the canonical-event JSONL rows to carry an "
            "observation/tool_output field; startup logs the coverage (0%% => rows have no observation)."
        ),
    )
    parser.add_argument(
        "--recognition-probe-bypass-projector",
        action="store_true",
        help=(
            "Bypass control for the recognition probe: read the classification heads off the RAW "
            "pooled backbone features (last-token hidden state) for ALL inputs -- current state, "
            "action, context AND observation -- skipping the JEPA-trained encoder_projector "
            "entirely. Measures the frozen encoder's own recognition ceiling; compare against the "
            "probe WITH the projector to test whether the projector attenuates outcome-relevant "
            "directions. Diagnostic only: the saved trunk is sized hidden_size*4 and will "
            "(loudly) not load into normally-constructed models. Requires "
            "--canonical-event-recognition-probe and --train-canonical-event-heads-only."
        ),
    )
    parser.add_argument(
        "--fast-lewm",
        action="store_true",
        help=(
            "Use Fast-LeWM (arXiv:2606.26217) action-prefix parallel prediction for multi-step "
            "dynamics instead of recursive one-step rollout: predict every horizon's latent "
            "directly from the anchor + action prefix in one pass (less error accumulation, "
            "faster planning). Requires --prediction-horizon>1; continuous latents only."
        ),
    )
    parser.add_argument("--fast-lewm-dim", type=int, default=256, help="Fast-LeWM action-prefix token / model dim.")
    parser.add_argument("--fast-lewm-layers", type=int, default=3, help="Fast-LeWM causal action-prefix encoder layers.")
    parser.add_argument("--fast-lewm-heads", type=int, default=4, help="Fast-LeWM action-prefix encoder attention heads.")
    parser.add_argument(
        "--action-contrastive-loss-coeff",
        type=float,
        default=0.0,
        help=(
            "Phase-1 action-contrastive transition loss. Requires the predictor to place the "
            "CORRECT action's predicted latent closer to the realized next latent than a "
            "corrupted action's, at the same state. The default softplus objective keeps a "
            "gradient after the margin is satisfied; pass --action-contrastive-loss-type hinge "
            "to reproduce the old max(0, margin + d_pos - d_neg) loss. Plain MSE+SigReg does "
            "not impose this -- MSE is minimized by a persistence map that ignores the action "
            "and persistence is not a global collapse, so SigReg does not penalise it either. "
            "0 disables."
        ),
    )
    parser.add_argument("--action-contrastive-negatives", type=int, default=4,
                        help="K corrupted actions per example. Hard same-tool negatives that change one top-level "
                             "argument field are generated first; remaining slots fall back to same-tool "
                             "other-trajectory calls and then tools unused by this trajectory.")
    parser.add_argument("--action-contrastive-margin", type=float, default=0.7,
                        help="Margin in cosine-distance units for the action-contrastive loss.")
    parser.add_argument("--action-contrastive-loss-type", choices=("softplus", "hinge"), default="softplus",
                        help="Ranking loss for action contrastive training. softplus keeps non-zero gradients "
                             "after the hard margin is satisfied; hinge reproduces the old clamp loss.")
    parser.add_argument("--action-contrastive-temperature", type=float, default=0.1,
                        help="Temperature for --action-contrastive-loss-type softplus.")
    parser.add_argument(
        "--event-target",
        action="store_true",
        help=(
            "Predict the newly-observed environment output instead of the cumulative next "
            "state: the backbone's TARGET input becomes o_{t+1} alone. Consecutive cumulative "
            "states overlap almost entirely, which is what makes a persistence map a near-"
            "optimal MSE solution; the observation is the part that actually changed. The "
            "target excludes the action on purpose -- encoding [a_t; o_{t+1}] would just swap "
            "the persistence shortcut for an action-copy one."
        ),
    )
    parser.add_argument(
        "--canonical-event-dump-predictions",
        type=Path,
        default=None,
        help=(
            "Write per-example predicted/gold labels for every field to this JSONL during the "
            "canonical-event eval. The metrics file keeps only marginal class distributions, "
            "which cannot yield per-class precision/recall/F1 or a confusion matrix -- those "
            "need the joint pairs. Dumping them makes later metrics a file read instead of "
            "another pass over the eval split."
        ),
    )
    parser.add_argument(
        "--canonical-event-head-inputs",
        choices=("all", "ctx_pred", "pred_only", "state", "state_action"),
        default="all",
        help=(
            "Which latents the canonical-event readout sees. `all` = [z_cur, z_act, z_ctx, "
            "z_pred] (historical), which lets the head bypass the predicted future entirely -- "
            "measured, dropping z_pred changes execution_status accuracy by 0.0000 (McNemar "
            "p=0.945) and substituting the TRUE next latent by +0.0012 (p=0.762). `ctx_pred` = "
            "[z_ctx, z_pred] and `pred_only` = [z_pred] remove that skip, so the readout can "
            "only score what the predicted future carries. `state` = the transformer "
            "predictor's belief state h_t (requires --predictor-arch transformer): the event "
            "representation is W_pred h_t, so h_t contains it plus the accumulated history the "
            "projection discards, and there is still no path around the predictor. "
            "`state_action` = [h_t, z_act]: same belief state plus the PROJECTED action latent, "
            "for fields that are properties of the tool call itself (action_type, object_type, "
            "side_effect_type, risk_signal) rather than of the predicted outcome -- h_t is "
            "conditioned on the action but has to spend capacity re-deriving it, so making it "
            "explicit is cheap. Note this reopens a readout path around the predictor for "
            "action-identity information (not for the outcome). Changes the trunk\'s input "
            "width, so it is written to the manifest and inherited on resume."
        ),
    )
    parser.add_argument(
        "--event-state-decomposition",
        action="store_true",
        help=(
            "Two-stage world model: the predictor emits the EVENT e_hat = F(z_t, z_ctx, z_a) "
            "against target sg(E(o_{t+1})), and a separate State Updater composes the next "
            "state z_hat = U(z_t, e_hat). The action is NOT an input to U -- it is already in "
            "e_hat -- so the next state must route through the event, which is what stops U "
            "from rediscovering z_hat = z_t. Implies the event target; the cumulative state "
            "target is kept for U's auxiliary loss. Pair with --canonical-event-head-inputs "
            "ctx_pred (or pred_only): leaving the heads on `all` lets them bypass e_hat via "
            "z_cur/z_act and the decomposition will not show up downstream."
        ),
    )
    parser.add_argument(
        "--state-update-loss-coeff", type=float, default=0.1,
        help="Weight of the State Updater's auxiliary loss d(U(z_t,e_hat), z_{t+1}). Small on "
             "purpose: at parity with the event term U re-learns the persistence copy. Watch "
             "`state_update_vs_persistence` in the logs -- it trending to 0 means exactly that.",
    )
    parser.add_argument(
        "--event-loss-coeff", type=float, default=1.0,
        help="Weight of the primary event-prediction loss d(e_hat, sg(E(o_{t+1}))). Replaces "
             "--latent-loss-coeff when --event-state-decomposition is on (that flag then has no "
             "effect); keep it above --state-update-loss-coeff.",
    )
    parser.add_argument(
        "--trajectory-chunk-size", type=int, default=20000,
        help=(
            "When a trajectory corpus is available as .jsonl, extract it this many trajectories "
            "at a time and free each chunk's raw objects before reading the next, so peak host "
            "RAM does not include the whole file. Only applies to .jsonl inputs (a .json must be "
            "fully materialized by json.load before anything can be extracted) -- convert just "
            "the oversized corpus with src/data_preparation/convert_trajectories_to_jsonl.py and "
            "it is picked up automatically via the sibling path. 0 disables chunking."
        ),
    )
    parser.add_argument(
        "--reuse-extracted-examples",
        action="store_true",
        help=(
            "Reuse <output-dir>/jepa_train_examples.jsonl from a previous run instead of "
            "re-extracting, when the recorded fingerprint (input paths + their size/mtime + the "
            "extraction arguments) matches exactly. Extraction over a large preset can take "
            "hours on rank 0 while the other ranks wait at a barrier, so repeating it is what "
            "pushes a relaunch past --distributed-timeout-minutes. Any fingerprint mismatch "
            "re-extracts and prints which key differed -- a stale cache is never used silently."
        ),
    )
    parser.add_argument(
        "--event-state-recurrent",
        action="store_true",
        help=(
            "Train the two-stage world model as a RECURRENT predictor: "
            "e_hat_{t+1}=F(s_t,a_{t+1},c) against sg(E(o_{t+1})), then s_{t+1}=U(s_t,a_{t+1},e*_{t+1}) "
            "whose only training signal is whether s_{t+1} can still predict the NEXT event "
            "(L_future). Implies --event-state-decomposition and needs --prediction-horizon >= 2 "
            "to have any future step to unroll to. Crucially this DROPS the cumulative "
            "next-state MSE: keeping it just relocates the persistence shortcut from F into U."
        ),
    )
    parser.add_argument(
        "--recurrent-state-init",
        action="store_true",
        help=(
            "Build the state recurrently from s_0 = I(c) instead of encoding the cumulative "
            "history text at every step: s_i = U(s_{i-1}, a_i, E(o_i)) over the logged history. "
            "The system and task prompts are constant within a trajectory, so they are encoded "
            "once into c; history then lives in s_t rather than being re-read as text. Requires "
            "--event-state-recurrent (U must exist and be trained by future-event prediction). "
            "Costs 2 short encodes per history step in place of one long cumulative-text encode."
        ),
    )
    parser.add_argument(
        "--state-updater-objective", choices=("future_event", "next_state"), default="future_event",
        help="How U is trained. `future_event` (default) gives it no direct target and lets "
             "L_future shape it. `next_state` restores the cumulative-next-state MSE -- the naive "
             "baseline, kept so its failure can be measured rather than asserted.",
    )
    parser.add_argument(
        "--future-event-loss-coeff", type=float, default=0.5,
        help="lambda_f on L_future = d(e_hat_{t+2}, sg(E(o_{t+2}))). Keep below "
             "--event-loss-coeff (lambda_e).",
    )
    parser.add_argument(
        "--consistency-loss-coeff", type=float, default=0.1,
        help="lambda_c on L_cons = d(U(s,a,e_hat), U(s,a,e*)) -- the teacher-forcing vs "
             "inference gap. 0 disables.",
    )
    parser.add_argument("--sigreg-coeff", type=float, default=0.05)
    parser.add_argument("--reconstruction-loss-coeff", type=float, default=0.0)
    parser.add_argument("--goal-loss-coeff", type=float, default=0.0, help="Deprecated and ignored; JEPA training no longer uses goal-observation progress loss.")
    parser.add_argument("--success-loss-coeff", type=float, default=0.0)
    parser.add_argument(
        "--terminal-loss-coeff",
        type=float,
        default=0.0,
        help=(
            "Weight on the terminal-step classification head: predicts whether this step is the "
            "LAST step of its trajectory (episode termination/'done', independent of whether the "
            "trajectory succeeded -- see the success/value-target labels for outcome quality). "
            "0 disables the head entirely (no parameters allocated)."
        ),
    )
    parser.add_argument(
        "--terminal-class-balance",
        choices=("none", "inverse", "effective_num"),
        default="none",
        help=(
            "Frequency-aware terminal-head BCE for the terminal/not-terminal imbalance. "
            "'inverse' uses pos_weight = N_nonterminal / N_terminal; 'effective_num' uses "
            "the ratio of effective-number class weights. Applies to TRAIN only; eval "
            "terminal loss/accuracy stay unweighted."
        ),
    )
    parser.add_argument(
        "--terminal-cb-beta",
        type=float,
        default=0.9999,
        help="beta for --terminal-class-balance effective_num.",
    )
    parser.add_argument(
        "--value-loss-coeff",
        type=float,
        default=0.0,
        help=(
            "Weight on the value-head regression loss: predicts the per-step discounted "
            "return-to-go (`value_target`, see src/data_preparation/annotate_step_value_scores.py) "
            "from [z_current, z_action, z_context, z_pred] -- reward-prediction, not "
            "classification. Only trainable via --train-canonical-event-heads-only, since "
            "`value_target` is only annotated on the canonical_event_with_nudge JSONL (it needs "
            "the LLM-labeled canonical fields plus, where available, a real verifier outcome -- "
            "neither exists on the raw world-model trajectory files the main training loop "
            "reads). Rows without a `value_target` (e.g. the plain, non-value-scored JSONL) are "
            "masked out of this loss rather than treated as 0. 0 disables the head entirely."
        ),
    )
    parser.add_argument("--value-loss-beta", type=float, default=1.0, help="Huber transition point (beta) for the value-head Smooth L1 loss.")
    parser.add_argument(
        "--truncate-states-keep-newest",
        action="store_true",
        help=(
            "Tokenize STATE texts (current/next/future-next) with LEFT truncation so over-length "
            "states keep their newest content instead of their oldest. State texts are rendered "
            "oldest-first, so the default right-side truncation cuts the just-appended "
            "action/observation -- measured at ~31%% of latent-target pairs FULLY losing the new "
            "content at max-input-length 2048, making z_next's input identical to z_current's "
            "(identity-map supervision). Context/action/observation texts keep right truncation "
            "(their important content leads). Recorded in the manifest and applied at replay."
        ),
    )
    parser.add_argument(
        "--joint-canonical-event-training",
        action="store_true",
        help=(
            "Train EVERY objective jointly on the labeled canonical_event_with_nudge JSONL: the "
            "latent prediction loss (--latent-loss-type MSE/Smooth L1) + SIGReg + the "
            "classification heads + the value head (--value-loss-coeff) + the terminal head "
            "(--terminal-loss-coeff), all from one batch in one optimizer step. Unlike "
            "--train-canonical-event-heads-only (frozen trunk, heads only) this trains the "
            "encoder/predictor too, since the latent loss is what shapes them.\n"
            "The latent target z_next for step t is the NEXT row's state text within the same "
            "trajectory (row t+1's input_history is row t's history plus row t's own "
            "action/observation, so that text IS the post-action state). Terminal rows have no "
            "successor, so their latent/SIGReg terms are masked out -- the classification, value "
            "and terminal losses still apply to every row."
        ),
    )
    parser.add_argument("--disable-goal-conditioning", action="store_true", default=True, help="Deprecated compatibility flag; goal conditioning is disabled by default.")
    parser.add_argument("--enable-goal-conditioning", action="store_false", dest="disable_goal_conditioning", help="Legacy option to include a goal input in the predictor. Not recommended.")
    parser.add_argument(
        "--canonical-observation-mode",
        choices=("agent", "mixed_api", "heuristic", "raw"),
        default="mixed_api",
        help="Deprecated and ignored for JEPA training; observations are encoded as raw tool outputs.",
    )
    parser.add_argument(
        "--canonicalizer-model",
        default="openai/gpt-4o-mini",
        help="Agent model used when --canonical-observation-mode=agent.",
    )
    parser.add_argument(
        "--canonicalizer-mixed-methods",
        default="gpt5,gemini,claude",
        help=(
            "Comma-separated src.llm.LLM methods randomly sampled when "
            "--canonical-observation-mode=mixed_api. Common values: "
            "gpt5, gpt4, gemini, claude."
        ),
    )
    parser.add_argument("--canonicalizer-max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--canonicalizer-cache-path",
        type=Path,
        default=None,
        help="Deprecated canonicalizer cache path; ignored while raw observations are used.",
    )
    parser.add_argument("--canonicalizer-disable-chat-template", action="store_true")
    parser.add_argument("--canonicalizer-attn-implementation", default="sdpa")
    parser.add_argument("--canonicalizer-device-map", default=None)
    parser.add_argument("--sigreg-projections", type=int, default=64)
    parser.add_argument("--sigreg-eps", type=float, default=1e-6)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--unfreeze-top-backbone-layers",
        type=int,
        default=0,
        help=(
            "With --freeze-backbone, keep the backbone frozen EXCEPT its last N transformer "
            "blocks (plus the final norm), which train alongside the predictor. Top rather than "
            "bottom for two reasons: backprop stops at the shallowest trainable block, so the "
            "top is the cheap direction, and with last_token/mean pooling the latent is read off "
            "the final hidden states, which is what has to adapt. Start around N=4 of 28 for "
            "Qwen3-Embedding-0.6B. 0 (default) keeps the fully-frozen behaviour. NOTE: the JEPA "
            "target is z_next from the SAME encoder (no EMA target), so once the backbone moves "
            "the target moves with it -- watch z_current_std/z_next_std staying near 1.0 and the "
            "downstream head metrics, not just the loss."
        ),
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=None,
        help=(
            "Separate (usually ~10x smaller) learning rate for trainable backbone parameters; "
            "--learning-rate is tuned for the randomly-initialised predictor and will damage a "
            "pretrained encoder. Defaults to --learning-rate when unset. Applies to full "
            "fine-tuning as well as --unfreeze-top-backbone-layers."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable backbone gradient checkpointing. Useful only when the backbone is trainable.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Backbone loading dtype. `auto` keeps the model default.",
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip loading/extracting eval datasets and skip evaluation; train (or skip-train) on the train split only.",
    )
    parser.add_argument(
        "--train-success-head-only",
        action="store_true",
        help="Load a trained JEPA checkpoint, freeze latent dynamics, and train only success_head.*.",
    )
    parser.add_argument(
        "--train-canonical-event-heads-only",
        action="store_true",
        help=(
            "Load a trained JEPA checkpoint, freeze latent dynamics, and train new "
            "canonical_event_state/nudge classification heads (canonical_event_heads.*) "
            "on --canonical-event-train-jsonl/--canonical-event-eval-jsonl. Mutually "
            "exclusive with --train-success-head-only and regular training."
        ),
    )
    parser.add_argument(
        "--canonical-event-train-jsonl",
        type=Path,
        default=DEFAULT_CANONICAL_EVENT_TRAIN_JSONL,
        help="Train JSONL of canonical_event_state/nudge labels (see --train-canonical-event-heads-only).",
    )
    parser.add_argument(
        "--canonical-event-eval-jsonl",
        type=Path,
        default=DEFAULT_CANONICAL_EVENT_EVAL_JSONL,
        help="Eval JSONL counterpart of --canonical-event-train-jsonl.",
    )
    parser.add_argument(
        "--canonical-event-train-sample-percentage",
        type=float,
        default=100.0,
        help=(
            "For --train-canonical-event-heads-only: keep only this percentage (0-100) of "
            "TRAIN examples from EACH benchmark for few-shot learning experiments (e.g. 10 keeps "
            "10%% of CRMArenaPro, 10%% of EnterpriseOps-Gym, and 10%% of Terminal-Bench-2.0). "
            "The label vocabulary/head sizes are built from the full split first, so results are "
            "comparable across percentages; the eval split is never subsampled. Default 100 uses all."
        ),
    )
    parser.add_argument(
        "--canonical-event-train-sample-seed",
        type=int,
        default=1234,
        help=(
            "Dedicated seed for the --canonical-event-train-sample-percentage few-shot subset, "
            "independent of --seed. Fixed by default so the same percentage selects the exact same "
            "training examples across executions (even when --seed changes for training randomness). "
            "Change it to draw a different few-shot subset at the same percentage."
        ),
    )
    parser.add_argument(
        "--canonical-event-head-hidden-size",
        type=int,
        default=512,
        help=(
            "Hidden size of the single shared trunk feeding all canonical_event_state/nudge "
            "heads (each field then gets only a small linear readout off it). Keep this small "
            "-- reusing the predictor's full hidden size per field (~11 fields) would multiply "
            "trainable-parameter/optimizer memory by ~11x for no benefit."
        ),
    )
    parser.add_argument(
        "--canonical-event-class-balance",
        choices=("none", "inverse", "effective_num"),
        default="none",
        help=(
            "Frequency-aware canonical-event head loss for imbalanced label fields. 'inverse' = "
            "inverse-frequency class weights; 'effective_num' = (1-beta)/(1-beta^n) class-balanced "
            "weights (Cui et al.). Applies to TRAIN only; eval loss/accuracy stay unweighted. This "
            "reweights the head loss -- distinct from --outcome-balance-loss (sampler weights)."
        ),
    )
    parser.add_argument("--canonical-event-cb-beta", type=float, default=0.999, help="beta for --canonical-event-class-balance effective_num.")
    parser.add_argument(
        "--canonical-event-focal-gamma",
        type=float,
        default=0.0,
        help="Focal-loss gamma for the single-label canonical-event heads (0 = plain CE). Combines with --canonical-event-class-balance as class-balanced focal loss.",
    )
    parser.add_argument(
        "--canonical-event-heads",
        choices=("all", "beam_plan"),
        default="all",
        help=(
            "Which classification heads to construct and train. `all` (default) trains every "
            "canonical_event_state/nudge field. `beam_plan` trains only the fields the planning "
            "scorer (src/canonical_event_scoring.py) actually consumes: execution_status, "
            "error_signature, progress_signal, side_effect_type, information_sufficiency -- "
            "dropping action_type, object_type, risk_signal, information_gain, "
            "recommended_abstract_action and missing_information_type (unused or unreliable for beam_plan/hier_latent_cem "
            "scoring). Dropped heads are never constructed; the checkpoint's "
            "canonical_event_vocab.json shrinks to match, so replay builds the same reduced set."
        ),
    )
    parser.add_argument(
        "--canonical-event-classification-loss-coeff",
        type=float,
        default=1.0,
        help=(
            "Weight on the canonical_event_state/nudge classification loss within "
            "--train-canonical-event-heads-only. Defaults to 1.0 (today's behavior: train the "
            "classification heads). Set to 0 to train ONLY the value head (--value-loss-coeff) "
            "on the same JSONL -- the classification heads are still constructed (from the "
            "JSONL's label vocabulary) but never called, so no compute is spent on them and "
            "they stay at random initialization."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("train", "replay"),
        default="train",
        help="`train` fits the JEPA world model; `replay` executes EnterpriseOps-Gym replay with a trained JEPA checkpoint.",
    )
    parser.add_argument(
        "--jepa-checkpoint-path",
        type=Path,
        default=None,
        help="Checkpoint directory for --mode=replay. Defaults to --output-dir.",
    )
    parser.add_argument(
        "--agent-model",
        default="gpt5",
        help="Agent model used for EnterpriseOps-Gym replay in --mode=replay.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-agent-tasks", type=int, default=100)
    parser.add_argument("--agent-max-steps", type=int, default=15)
    parser.add_argument("--internal-thinking-max-iters", type=int, default=3)
    parser.add_argument("--imagined-trajectory-max-steps", type=int, default=3)
    parser.add_argument("--imagined-trajectory-rollouts", type=int, default=1)
    parser.add_argument("--imagined-rollout-temperature", type=float, default=0.7)
    parser.add_argument(
        "--imagined-rollout-mode",
        choices=("closed_loop", "open_loop"),
        default="closed_loop",
        help=(
            "closed_loop (default): each imagined step's action is chosen after seeing the "
            "world model's predicted feedback for the previous step (lockstep-batched across "
            "rollouts). open_loop: ONE agent call proposes all rollouts' full action "
            "sequences up front (beam_plan-style skeletons with $stepK.field references); "
            "the JEPA world model then fills in each plan's feedback chain via "
            "predict_feedback. Cheapest agent-LLM budget (1 call per planning cycle); "
            "actions cannot react to predicted failures mid-trajectory. Falls back to "
            "closed_loop for a cycle if the plan payload cannot be parsed. Shares the "
            "implementation in src/finetuning.py."
        ),
    )
    parser.add_argument(
        "--sample-temperature-ladder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Spread the k open-loop planning samples over a temperature ladder instead of one "
            "temperature, to cut duplicate plans. Forfeits the single-request n=k path."
        ),
    )
    parser.add_argument(
        "--sample-temperature-ladder-max",
        type=float,
        default=1.2,
        help="Upper bound of the ladder (above ~1.2 malformed actions truncate plans).",
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
        "--imagined-single-call-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "closed_loop only: generate each imagined step's thought AND action in one LLM "
            "call instead of the two-call think-then-act sequence."
        ),
    )
    parser.add_argument(
        "--imagined-parallel-rollouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "closed_loop only: advance all --imagined-trajectory-rollouts chains in "
            "lockstep, batching each step's agent generations across rollouts."
        ),
    )
    parser.add_argument(
        "--llm-batch-parallelism",
        type=int,
        default=8,
        help=(
            "Max concurrent requests when batching agent LLM calls against API/vLLM "
            "backends (local HF backends batch inside a single generate call instead)."
        ),
    )
    parser.add_argument(
        "--agent-draft-model",
        default=None,
        help=(
            "Optional HF draft model for speculative (assisted) decoding of the agent model "
            "in --mode=replay. Local HF agent backends only; batch-1 generations only."
        ),
    )
    parser.add_argument(
        "--prompt-lookup-tokens",
        type=int,
        default=0,
        help=(
            "Enable prompt-lookup decoding with this candidate length for the local HF agent "
            "backend in --mode=replay. Ignored when --agent-draft-model is set; 0 disables."
        ),
    )
    parser.add_argument(
        "--imagined-trajectory-selection-strategy",
        choices=("first", "llm_judge"),
        default="llm_judge",
    )
    parser.add_argument(
        "--imagined-trajectory-observation-source",
        choices=("world_model", "none"),
        default="world_model",
    )
    parser.add_argument(
        "--imagined-observation-backend",
        choices=("auto", "canonical_event", "success", "decoder"),
        default="auto",
        help=(
            "Which JEPA head reconstructs the imagined observation/state inserted into the "
            "agent prompt during --mode=replay imagined rollouts. `auto` prefers the "
            "canonical_event_state/nudge classification heads when the checkpoint carries them, "
            "then the success head, then the seq2seq decoder. `canonical_event` forces the "
            "classification heads (errors if the checkpoint has none); `success`/`decoder` force "
            "the earlier behaviours. Only affects observation_source=world_model imagined rollouts."
        ),
    )
    parser.add_argument("--revision-lookahead-steps", type=int, default=1)
    parser.add_argument("--revision-imagined-rollouts", type=int, default=1)
    parser.add_argument("--revision-rollout-temperature", type=float, default=0.7)
    parser.add_argument("--latent-plan-samples", type=int, default=10)
    parser.add_argument("--latent-plan-elites", type=int, default=3)
    parser.add_argument("--latent-plan-iters", type=int, default=3)
    parser.add_argument("--latent-plan-horizon", type=int, default=5)
    parser.add_argument("--latent-mpc-execute-steps", type=int, default=1)
    parser.add_argument("--latent-plan-temperature", type=float, default=0.7)
    parser.add_argument(
        "--latent-plan-score-margin",
        type=float,
        default=1e-3,
        help="Keep the baseline action unless the best latent plan improves seed score by this margin.",
    )
    parser.add_argument(
        "--latent-plan-diversity-multiplier",
        type=int,
        default=2,
        help="Over-sample candidate actions by this factor when filling unique latent planning pools.",
    )
    parser.add_argument(
        "--latent-plan-hard-override",
        action="store_true",
        help=(
            "beam_plan: force-execute the world model's chosen action over the agent's own choice "
            "when it beats the seed by the score margin. Off by default -- the beam is ADVISORY: the "
            "agent keeps control and the imagined trajectory is injected only as guidance."
        ),
    )
    parser.add_argument(
        "--latent-plan-goal-mode",
        choices=("final", "next_subgoal"),
        default="next_subgoal",
        help="Use the final task goal or the next stage-derived subgoal for per-action latent planning.",
    )
    parser.add_argument(
        "--gate-flat-score-ratio",
        type=float,
        default=1.0,
        help=(
            "beam_plan: confidence/flatness gate on the imagined trajectory's average normalized "
            "score -- injected/eligible for override only if avg_normalized_score >= ratio / "
            "num_candidates. Lower this to inject/override more often; 0 disables the gate "
            "entirely (subject to --latent-plan-hard-override for whether it can override)."
        ),
    )
    parser.add_argument(
        "--hier-cem-anchors", type=int, default=8,
        help="hier_latent_cem: K diverse anchor actions proposed by the ONE LLM call per planning cycle.",
    )
    parser.add_argument(
        "--hier-cem-samples", type=int, default=256,
        help="hier_latent_cem: N continuous latent-action trajectories sampled per CEM iteration (LLM-free).",
    )
    parser.add_argument(
        "--hier-cem-elites", type=int, default=16,
        help="hier_latent_cem: M elite trajectories kept each CEM iteration to refit (pi, mean, std).",
    )
    parser.add_argument(
        "--hier-cem-iters", type=int, default=3,
        help="hier_latent_cem: CEM refinement iterations per planning cycle.",
    )
    parser.add_argument(
        "--hier-cem-horizon", type=int, default=5,
        help="hier_latent_cem: lookahead steps per latent-action trajectory.",
    )
    parser.add_argument(
        "--hier-cem-init-std", type=float, default=0.2,
        help="hier_latent_cem: initial per-family Gaussian std (fraction of |mean| for singleton-anchor families).",
    )
    parser.add_argument(
        "--hier-cem-min-std", type=float, default=0.02,
        help="hier_latent_cem: std floor preventing premature Gaussian collapse.",
    )
    parser.add_argument(
        "--hier-cem-smoothing", type=float, default=1.0,
        help="hier_latent_cem: CEM update blend with the previous iteration's params (1.0 = classic full replace).",
    )
    parser.add_argument(
        "--hier-cem-min-elite-agreement", type=float, default=0.5,
        help="hier_latent_cem: fraction of final-iteration elites that must share the winning step-0 family "
        "before the plan is trusted enough to inject (CEM-native confidence gate).",
    )
    parser.add_argument(
        "--hier-cem-decode-strategy", choices=("nearest_anchor", "learned_decoder"), default="nearest_anchor",
        help="hier_latent_cem: how to turn the converged latent trajectory into an executable action. "
        "'nearest_anchor' (default, always valid) picks the closest LLM-proposed anchor per family. "
        "'learned_decoder' decodes the actual CEM-sampled latent via TextLeWorldModel.decode_action_latent "
        "(requires a checkpoint trained with --action-decoder-loss-coeff > 0) and falls back to "
        "nearest_anchor if the decode is invalid.",
    )
    parser.add_argument(
        "--hier-cem-decode-max-new-tokens", type=int, default=96,
        help="hier_latent_cem: max tokens generated per learned-decoder decode call.",
    )
    parser.add_argument(
        "--replay-modes",
        default="baseline,imagined",
        help="Comma-separated replay modes to run in --mode=replay. Supported: baseline, imagined, revision, latent_guided, beam_plan, hier_latent_cem.",
    )
    parser.add_argument("--final-answer-f1-threshold", type=float, default=0.35)
    parser.add_argument("--include-world-model-history", action="store_true")
    parser.add_argument("--agent-max-observation-chars", type=int, default=2000)
    parser.add_argument("--agent-replay-history-budget-chars", type=int, default=60000)
    parser.add_argument("--record-replay-trajectories", action="store_true")
    parser.add_argument("--gym-task-configs", type=Path, default=None)
    parser.add_argument(
        "--gym-repo-path",
        type=Path,
        default=Path.home() / "program" / "tools" / "EnterpriseOps-Gym",
    )
    parser.add_argument(
        "--gym-task-split-manifest",
        type=Path,
        default=DEFAULT_ENTERPRISEOPS_GYM_TASK_SPLIT_MANIFEST,
        help=(
            "Train/test JSON manifest for EnterpriseOps-Gym JSONL task files. "
            "When --gym-task-configs is set, replay evaluates only the manifest's "
            "test JSONL tasks, preserving manifest order."
        ),
    )
    parser.add_argument(
        "--no-gym-task-split-manifest",
        action="store_true",
        help="Disable EnterpriseOps-Gym task split filtering during replay evaluation.",
    )
    parser.add_argument(
        "--skip-web-trajectories",
        action="store_true",
        help=(
            "Skip loading the web-browsing ADP trajectory files (agenttuning_mind2web, "
            "go-browse-wa, mind2web, nnetnav-live, nnetnav-wa, synatra), even if they were "
            "selected via --trajectory-dataset (e.g. all/adp_all) or explicit "
            "--train-data-path/--eval-data-path. Use this to cut load time/memory when "
            "these large datasets aren't needed."
        ),
    )
    parser.add_argument("--disable-chat-template", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--inference-device-map", default=None)
    args = parser.parse_args()
    # Resolve the implication HERE, not at the model-construction site: the dataset, the
    # manifest writer and the model all read these flags at different times, and a late
    # mutation would let the manifest disagree with the data the run was actually trained on.
    if getattr(args, "recurrent_state_init", False) and not getattr(args, "event_state_recurrent", False):
        raise SystemExit(
            "--recurrent-state-init requires --event-state-recurrent: s_0 = I(c) is only "
            "meaningful when U exists and is trained by future-event prediction. Without it "
            "there is nothing to accumulate the history into."
        )
    if getattr(args, "event_state_recurrent", False):
        args.event_state_decomposition = True
        if int(getattr(args, "prediction_horizon", 1) or 1) < 2:
            raise SystemExit(
                "--event-state-recurrent needs --prediction-horizon >= 2: with a horizon of 1 "
                "there is no future step, so L_future is empty and the State Updater receives "
                "no gradient at all."
            )
    if getattr(args, "event_state_decomposition", False):
        args.event_target = True
        if args.canonical_event_head_inputs == "all":
            print("[event/state] WARNING: --canonical-event-head-inputs=all leaves the readout "
                  "skip open, so the heads can reach z_cur/z_act and bypass the predicted event "
                  "entirely -- the decomposition will not be visible downstream. Use ctx_pred.",
                  flush=True)
    if args.canonical_event_head_inputs in {"state", "state_action"} and args.canonical_event_recognition_probe:
        # In 'state' mode the heads read h_t; the probe's substitution of z_observation for
        # z_pred never reaches them, so the run would report normal-mode numbers under a probe
        # label. The x_obs-only readout is 'pred_only' + the probe.
        parser.error(
            f"--canonical-event-recognition-probe is meaningless with "
            f"--canonical-event-head-inputs {args.canonical_event_head_inputs}: the heads read "
            "the predictor's hidden state, so substituting the observation latent for z_pred "
            "changes nothing. Use --canonical-event-head-inputs pred_only to read the heads off "
            "z_observation alone."
        )
    if args.canonical_event_head_inputs in {"state", "state_action"} and args.predictor_arch != "transformer":
        # Checked here rather than only at construction so it fails before the backbone loads.
        parser.error(
            f"--canonical-event-head-inputs {args.canonical_event_head_inputs} requires "
            "--predictor-arch transformer: only the transformer predictor has a hidden state "
            "distinct from its output (the MLP equivalent is pred_only)."
        )
    if args.predictor_arch == "transformer" and args.canonical_event_head_inputs == "all":
        print("[predictor] WARNING: --canonical-event-head-inputs=all leaves the readout skip "
              "open ([z_cur, z_act, z_ctx] reach the heads directly). With the transformer "
              "predictor, `state` is the intended readout -- h_t already contains the predicted "
              "event, with no path around the predictor.", flush=True)
    if args.no_gym_task_split_manifest:
        args.gym_task_split_manifest = None
    preset_train_paths, preset_eval_paths = TRAJECTORY_DATASET_PRESETS[args.trajectory_dataset]
    if args.train_data_path is None:
        args.train_data_path = list(preset_train_paths)
    if args.eval_data_path is None:
        args.eval_data_path = list(preset_eval_paths)
    if args.skip_web_trajectories:
        _web = WEB_BROWSING_TRAJECTORY_PATHS | WEB_BROWSING_25K_PATHS
        args.train_data_path = [p for p in args.train_data_path if p not in _web]
        args.eval_data_path = [p for p in args.eval_data_path if p not in _web]
    replay_modes = tuple(split_csv(args.replay_modes))
    allowed_replay_modes = {"baseline", "imagined", "revision", "latent_guided", "beam_plan", "hier_latent_cem"}
    invalid_replay_modes = sorted(set(replay_modes) - allowed_replay_modes)
    if not replay_modes:
        parser.error("--replay-modes must include at least one mode: baseline, imagined, revision, latent_guided, beam_plan, hier_latent_cem")
    if invalid_replay_modes:
        parser.error(
            "--replay-modes contains unsupported mode(s): "
            + ", ".join(invalid_replay_modes)
            + ". Supported: baseline, imagined, revision, latent_guided, beam_plan"
        )
    args.replay_modes = replay_modes
    if args.train_success_head_only and args.success_loss_coeff <= 0:
        args.success_loss_coeff = 1.0
    if args.train_success_head_only and args.train_canonical_event_heads_only:
        parser.error("--train-success-head-only and --train-canonical-event-heads-only are mutually exclusive.")
    if args.joint_canonical_event_training and args.train_success_head_only:
        parser.error("--joint-canonical-event-training and --train-success-head-only are mutually exclusive.")
    if args.joint_canonical_event_training and args.reconstruction_loss_coeff > 0:
        # The JSONL rows carry no observation text, so there is no reconstruction target.
        parser.error("--joint-canonical-event-training does not support --reconstruction-loss-coeff > 0 (the labeled JSONL has no observation text to reconstruct).")
    if args.recognition_probe_bypass_projector and not (
        args.canonical_event_recognition_probe and args.train_canonical_event_heads_only
    ):
        parser.error(
            "--recognition-probe-bypass-projector requires --canonical-event-recognition-probe "
            "and --train-canonical-event-heads-only (it is a control variant of the recognition probe)."
        )
    if args.joint_canonical_event_training and args.latent_type == "categorical":
        # The joint latent term is the point-wise MSE/Smooth L1 objective, which is masked
        # per-row for terminal steps; categorical latents need the KL-balanced prior/posterior
        # loss instead (no masked variant of it exists). Same continuous-only restriction as
        # --prediction-horizon>1 / --fast-lewm / --latent-delta-prediction.
        parser.error("--joint-canonical-event-training supports continuous latents only (--latent-type continuous); categorical latents use the KL loss, which has no masked variant.")
    if not 0.0 < args.canonical_event_train_sample_percentage <= 100.0:
        parser.error("--canonical-event-train-sample-percentage must be in the range (0, 100].")
    return args


def distributed_is_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def distributed_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return distributed_rank() == 0


def setup_distributed(timeout_minutes: int = 120) -> tuple[bool, int, int]:
    if not distributed_is_enabled():
        return False, 0, 0
    local_rank = distributed_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        # Only rank 0 runs the (potentially very long, for uncapped large
        # trajectory presets like --trajectory-dataset all) data loading and
        # extraction step below while other ranks idle at a barrier -- the
        # default 30-minute collective timeout can fire well before rank 0
        # finishes, killing the whole job. Give it a lot of room.
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=datetime.timedelta(minutes=timeout_minutes),
        )
    return True, distributed_rank(), local_rank


def cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


@dataclass
class JepaExample:
    trajectory_id: str
    trajectory_index: int
    interaction_index: int
    context_text: str
    current_state_text: str
    action_text: str
    next_state_text: str
    observation_text: str
    success_label: int | None = None
    terminal_label: int | None = None  # 1 if this is the LAST interaction_index in its trajectory
    benchmark: str = "default"  # source benchmark key for Option-C observation grounding
    tool_name: str | None = None  # primary tool of the action, for the optional action-head
    # --action-contrastive-loss-coeff: corrupted actions for the SAME state. The predictor must
    # place z_pred(s, a+) closer to the true z_next than z_pred(s, a-), which is the requirement
    # plain MSE lacks -- MSE is minimized by a persistence map that ignores the action entirely
    # (measured: R_action = 0.041, persistence beats the trained predictor).
    negative_action_texts: list[str] = field(default_factory=list)
    # --recurrent-state-init: the (action, observation) pairs preceding this step, kept
    # STRUCTURALLY rather than only rendered into current_state_text. The recurrent state is
    # built by unrolling U from s_0 = I(c) over these, so the cumulative history text -- and the
    # long encode of it -- is not needed at all. Populated only when that flag is set, since it
    # duplicates observation text across examples.
    history_action_texts: list[str] = field(default_factory=list)
    history_observation_texts: list[str] = field(default_factory=list)


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def resolve_torch_dtype(dtype_name: str | None) -> torch.dtype | str:
    if dtype_name in {None, "auto"}:
        return "auto"
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _hidden_size_from_config(config: Any) -> int | None:
    for attr in ("d_model", "hidden_size"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        value = _hidden_size_from_config(text_config)
        if value is not None:
            return value
    for attr in ("encoder", "encoder_config", "decoder", "decoder_config"):
        child = getattr(config, attr, None)
        if child is not None:
            value = _hidden_size_from_config(child)
            if value is not None:
                return value
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
        for key in ("d_model", "hidden_size"):
            value = config_dict.get(key)
            if value is not None:
                return int(value)
        for key in ("encoder", "encoder_config", "decoder", "decoder_config", "text_config"):
            child = config_dict.get(key)
            if isinstance(child, dict):
                value = _hidden_size_from_mapping(child)
                if value is not None:
                    return value
    return None


def _hidden_size_from_mapping(config: dict[str, Any]) -> int | None:
    for key in ("d_model", "hidden_size"):
        value = config.get(key)
        if value is not None:
            return int(value)
    for key in ("text_config", "encoder", "encoder_config", "decoder", "decoder_config"):
        child = config.get(key)
        if isinstance(child, dict):
            value = _hidden_size_from_mapping(child)
            if value is not None:
                return value
    return None


def resolve_backbone_hidden_size(backbone: Any) -> int:
    encoder = backbone_encoder(backbone)
    for config in (getattr(encoder, "config", None), getattr(backbone, "config", None)):
        if config is None:
            continue
        hidden_size = _hidden_size_from_config(config)
        if hidden_size is not None:
            return hidden_size
    raise ValueError(
        "Could not infer backbone hidden size. Expected a config field like "
        "d_model, hidden_size, or encoder.text_config.hidden_size."
    )


def load_text_tokenizer(model_name_or_path: str | Path, trust_remote_code: bool = False) -> Any:
    from transformers import AutoProcessor, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    except AttributeError:
        # Checkpoints that saved `extra_special_tokens` as a list rather than a mapping: the
        # same tokenizer loads once the field is overridden. Handled before the multimodal
        # fallback below, which would otherwise report this as "no processor either".
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code, extra_special_tokens={}
        )
        print(f"[tokenizer] {model_name_or_path}: repaired malformed extra_special_tokens "
              "while loading (checkpoint left unchanged).", flush=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def load_jepa_backbone(
    model_name_or_path: str | Path,
    *,
    backbone_type: str,
    trust_remote_code: bool = False,
    dtype: str | torch.dtype = "auto",
) -> Any:
    from transformers import AutoModel, AutoModelForSeq2SeqLM

    model_cls = AutoModelForSeq2SeqLM if backbone_type == "seq2seq" else AutoModel
    return model_cls.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )


def backbone_encoder(backbone: Any) -> Any:
    return backbone.get_encoder() if hasattr(backbone, "get_encoder") else backbone


def backbone_transformer_layers(backbone: Any) -> Any | None:
    """The backbone's ModuleList of transformer blocks, or None if it cannot be found.

    Covers the layouts this repo loads: decoder-style stacks used as encoders
    (`model.layers` on Qwen3/Llama), plain encoders (`encoder.layer` on BERT-likes) and
    encoder-decoder backbones via get_encoder() (`block` on T5/T5Gemma).
    """
    encoder = backbone_encoder(backbone)
    for holder in (encoder, getattr(encoder, "model", None), getattr(encoder, "encoder", None), backbone):
        if holder is None:
            continue
        for attribute in ("layers", "layer", "block", "h"):
            layers = getattr(holder, attribute, None)
            if isinstance(layers, nn.ModuleList) and len(layers):
                return layers
    return None


def freeze_backbone_except_top_layers(backbone: Any, unfreeze_top_layers: int) -> dict[str, Any]:
    """Freeze the backbone except its last N transformer blocks (and the trailing norm).

    Top-N, not bottom-N: with everything below frozen and the input not requiring grad, the
    backward pass stops at the shallowest trainable block, so unfreezing the top costs N blocks
    of backprop while unfreezing the bottom costs the whole stack. The final norm rides along
    with the last block because the pooled latent is read straight through it.

    Returns a summary for the run manifest/log; `unfreeze_top_layers <= 0` is a full freeze.
    """
    for param in backbone.parameters():
        param.requires_grad = False
    summary: dict[str, Any] = {
        "unfreeze_top_backbone_layers": max(0, int(unfreeze_top_layers)),
        "backbone_layers_found": 0,
        "backbone_trainable_layer_indices": [],
        "backbone_trainable_parameters": 0,
        "trailing_norm_trainable": False,
    }
    if unfreeze_top_layers <= 0:
        return summary

    layers = backbone_transformer_layers(backbone)
    if layers is None:
        raise SystemExit(
            "--unfreeze-top-backbone-layers could not locate the backbone's transformer block "
            "list; drop the flag (full freeze) or fine-tune the whole backbone instead."
        )
    summary["backbone_layers_found"] = len(layers)
    if unfreeze_top_layers > len(layers):
        raise SystemExit(
            f"--unfreeze-top-backbone-layers={unfreeze_top_layers} exceeds the backbone's "
            f"{len(layers)} transformer blocks. Use a smaller N, or drop --freeze-backbone to "
            "train all of them."
        )
    trainable_indices = list(range(len(layers) - unfreeze_top_layers, len(layers)))
    for index in trainable_indices:
        for param in layers[index].parameters():
            param.requires_grad = True
    # The trailing norm sits between the last block and the pooled representation; leaving it
    # frozen while its input distribution shifts is the one clearly wrong combination.
    encoder = backbone_encoder(backbone)
    for holder in (encoder, getattr(encoder, "model", None), backbone):
        if holder is None:
            continue
        for attribute in ("norm", "final_layer_norm", "final_norm", "ln_f"):
            module = getattr(holder, attribute, None)
            if isinstance(module, nn.Module):
                for param in module.parameters():
                    param.requires_grad = True
                summary["trailing_norm_trainable"] = True
                break
        if summary["trailing_norm_trainable"]:
            break
    summary["backbone_trainable_layer_indices"] = trainable_indices
    summary["backbone_trainable_parameters"] = sum(
        int(param.numel()) for param in backbone.parameters() if param.requires_grad
    )
    return summary


def backbone_supports_reconstruction(backbone: Any) -> bool:
    return hasattr(backbone, "get_encoder") and callable(getattr(backbone, "generate", None))


def path_list_text(paths: list[Path]) -> str:
    return ",".join(str(path) for path in paths)


# orca_agentinstruct tags its `user` turns with source="user" the same way as
# datasets that genuinely fold tool output into the next user turn, but its
# `user` content after an action is not actually the tool's observation --
# treating it as last_tool_output would train on noise, so this dataset is
# excluded from the injection (and therefore skipped entirely at load time,
# since it carries no real `state` messages either).
ADP_DATASETS_WITHOUT_GENUINE_USER_TOOL_OUTPUT = {"orca_agentinstruct"}


def adp_dataset_name_from_path(path: Path) -> str | None:
    suffix = "_world_model_trajectories.json"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else None


def action_message_has_tool_call(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return message.get("role") == "action" and isinstance(content, dict) and bool(content.get("tool_calls"))


def user_message_text(message: dict[str, Any]) -> str:
    if message.get("role") != "user":
        return ""
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""


def synthesize_state_from_user_content(text: str) -> dict[str, Any]:
    return {"role": "state", "content": {"state": {"context": {"last_tool_output": text}}}}


def inject_user_observations_as_state(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Synthesize a lean `state` message for actions whose observation was folded
    into a following `user` turn instead of a real `state` message.

    Several ADP source datasets (code_feedback, codeactinstruct, agenttuning_*,
    nebius_SWE-agent-trajectories, ...) feed the tool/environment result back as
    an ordinary `user` turn rather than a `state` message, since their harnesses
    require strict user/assistant alternation. Treat that user content as
    `last_tool_output` so these trajectories still produce grounded-observation
    JEPA examples via extract_state_examples() instead of being ignored.

    Only a `user` message immediately following a real tool-call `action` is
    ever treated this way -- a `user` message following an `assistant` turn (a
    message_action / plain-text step, no tool call) is left alone, since that is
    an ordinary conversational turn, not a tool observation.
    """
    messages = trajectory.get("messages", [])
    new_messages: list[dict[str, Any]] = []
    changed = False
    index = 0
    total = len(messages)
    while index < total:
        message = messages[index]
        new_messages.append(message)
        index += 1
        if not action_message_has_tool_call(message):
            continue
        if index < total and messages[index].get("role") == "state":
            continue  # already has a real observation
        observation_parts: list[str] = []
        cursor = index
        while cursor < total and messages[cursor].get("role") == "user":
            text = user_message_text(messages[cursor])
            if text:
                observation_parts.append(text)
            cursor += 1
        if observation_parts:
            new_messages.append(synthesize_state_from_user_content("\n\n".join(observation_parts)))
            changed = True
    if not changed:
        return trajectory
    patched = dict(trajectory)
    patched["messages"] = new_messages
    return patched


def trajectories_have_last_tool_output(trajectories: list[dict[str, Any]]) -> bool:
    """Does any `state` message in these trajectories carry a real tool observation?

    Only `state.context.last_tool_output` is checked -- the same field
    extract_state_examples() reads for its `tool_output` target -- so this is a
    direct probe for "would this file contribute any grounded-observation JEPA
    examples", not a schema check.
    """
    for trajectory in trajectories:
        for message in trajectory.get("messages", []):
            if message.get("role") != "state":
                continue
            if extract_state_tool_output(message.get("content")).strip():
                return True
    return False


def load_trajectory_file(path: Path, *, require_enterpriseops_gym: bool = False) -> list[dict[str, Any]]:
    # Suffix-aware via iter_trajectory_file: .json goes through json.load as before, .jsonl is
    # read line-by-line. Both then normalize + inject identically, so this stays the single
    # definition of "what loading a trajectory file means".
    trajectories = list(iter_trajectory_file(path))
    if require_enterpriseops_gym:
        bad = []
        for index, trajectory in enumerate(trajectories[:25]):
            config_name = trajectory.get("gym_task_config_name") or resolve_gym_task_config_name(trajectory)
            source_path = str(trajectory.get("source_path", trajectory.get("seed_source_path", ""))).lower()
            if not config_name and "enterpriseops_gym" not in source_path:
                bad.append(index)
        if bad:
            raise SystemExit(f"{path} does not look like EnterpriseOps-Gym data; missing gym metadata near rows {bad[:5]}")
    if not trajectories_have_last_tool_output(trajectories):
        print(f"[skip] {path}: no state message carries last_tool_output; ignoring for JEPA training.")
        return []
    return trajectories


def resolve_streamable_path(path: Path) -> Path:
    """Prefer a sibling `.jsonl` when one exists.

    Lets a single oversized corpus be converted to JSONL (see
    src/data_preparation/convert_trajectories_to_jsonl.py) and picked up transparently, with no
    preset edits and no effect on the other files -- which is what "stream the big file only"
    should mean in practice.
    """
    if path.suffix == ".json":
        sibling = path.with_suffix(".jsonl")
        if sibling.is_file():
            return sibling
    return path


def prepare_trajectory(trajectory: dict[str, Any], *, inject: bool) -> dict[str, Any]:
    return inject_user_observations_as_state(trajectory) if inject else trajectory


def iter_trajectory_file(path: Path) -> Any:
    """Yield normalized trajectories one at a time.

    `.jsonl` is read line-by-line, so the caller can extract and discard in chunks and the
    resident set never holds the whole corpus. `.json` still goes through json.load(), which
    must materialize the entire list before anything can be yielded -- that is precisely the
    case worth converting.
    """
    inject = adp_dataset_name_from_path(path) not in ADP_DATASETS_WITHOUT_GENUINE_USER_TOOL_OUTPUT
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for trajectory in normalize_loaded_trajectories([record]):
                    yield prepare_trajectory(trajectory, inject=inject)
        return
    records = load_json(path)
    if not isinstance(records, list):
        raise SystemExit(f"Expected a list of trajectories in {path}")
    for trajectory in normalize_loaded_trajectories(records):
        yield prepare_trajectory(trajectory, inject=inject)


def load_trajectory_paths(paths: list[Path], *, require_enterpriseops_gym: bool = False) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for path in paths:
        trajectories.extend(load_trajectory_file(path, require_enterpriseops_gym=require_enterpriseops_gym))
    return trajectories


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    omitted = len(text) - (2 * keep)
    return text[:keep] + f"\n[truncated {omitted} chars]\n" + text[-keep:]


def render_action(action: Any) -> str:
    if isinstance(action, str):
        return action.strip()
    return safe_json(action)


def action_tool_name(action: Any) -> str | None:
    """Primary tool/function name of an action (first tool_call), for the optional
    latent action-head (P1 tool selection / P2 action encoder). None when the action
    carries no structured tool call."""
    if isinstance(action, dict):
        calls = action.get("tool_calls")
        if isinstance(calls, list) and calls and isinstance(calls[0], dict):
            function = calls[0].get("function")
            if isinstance(function, dict) and function.get("name"):
                return str(function["name"])
        if action.get("name"):
            return str(action["name"])
    return None


def build_tool_vocabulary(examples: list["JepaExample"]) -> dict[str, int]:
    """Deterministic {tool_name: index} over the training actions, index 0 reserved
    for unknown/out-of-vocab tools. Built identically on every rank (examples are
    shared), so no broadcast is needed."""
    names = sorted({ex.tool_name for ex in examples if getattr(ex, "tool_name", None)})
    vocab = {"<unk>": 0}
    for name in names:
        vocab[name] = len(vocab)
    return vocab


def render_history(history: list[dict[str, Any]]) -> str:
    rows = []
    for item in history[-WORLD_MODEL_INPUT_HISTORY_SIZE:]:
        rows.append(
            {
                "step": item.get("step"),
                "action": item.get("action"),
                "observation": item.get("observation"),
            }
        )
    return safe_json(rows)


def build_context_text(system_prompt: str, user_prompt: str) -> str:
    return f"System prompt:\n{system_prompt}\n\nUser task:\n{user_prompt}"


def build_state_text(context_text: str, history: list[dict[str, Any]]) -> str:
    if history:
        return context_text + "\n\nHistory before current action:\n" + render_history(history)
    return context_text + "\n\nHistory before current action:\n[]"


def build_next_state_text(context_text: str, history: list[dict[str, Any]], action_text: str, observation_text: str) -> str:
    current_event = {"step": len(history) + 1, "action": action_text, "observation": observation_text}
    next_history = list(history[-(WORLD_MODEL_INPUT_HISTORY_SIZE - 1):]) + [current_event]
    return context_text + "\n\nHistory after current action:\n" + render_history(next_history)


def render_raw_observation(example: Any, max_chars: int) -> str:
    tool_output = stringify_tool_output(example.tool_output or "").strip()
    if not tool_output and example.error_payload:
        tool_output = stringify_tool_output(example.error_payload).strip()
    if not tool_output:
        tool_output = stringify_tool_output(example.state).strip()
    return truncate_text(tool_output, max_chars)


def extract_raw_success_label(example: Any) -> int | None:
    label = normalize_last_tool_execution_result(extract_last_tool_execution_result_from_state(example.state))
    if label is not None:
        return int(label == 1)
    if example.error_payload:
        return 0
    return None


def render_raw_replay_history(history: list[dict[str, Any]], max_chars: int) -> str:
    rows = []
    for item in history[-WORLD_MODEL_INPUT_HISTORY_SIZE:]:
        observation = item.get("observation")
        if observation is None and "state" in item:
            observation = item.get("state")
        rows.append({
            "step": item.get("step", item.get("imagined step")),
            "action": item.get("action"),
            "observation": truncate_text(stringify_tool_output(observation or ""), max_chars),
        })
    return truncate_text(safe_json(rows), max_chars * WORLD_MODEL_INPUT_HISTORY_SIZE)


def canonical_observation_cache_key(example: Any) -> str:
    payload = {
        "trajectory_id": example.trajectory_id,
        "trajectory_index": example.trajectory_index,
        "interaction_index": example.interaction_index,
        "action": example.action,
        "tool_output": example.tool_output,
        "error_payload": example.error_payload,
        "state": example.state,
    }
    return safe_json(payload)


def coerce_text_list(value: Any, limit: int = 8) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        rows = []
        for item in value[:limit]:
            if item is None:
                continue
            if isinstance(item, str):
                text = item.strip()
            else:
                text = stringify_tool_output(item).strip()
            if text:
                rows.append(text[:240])
        return rows
    return [stringify_tool_output(value).strip()[:240]]


def first_nested_value(value: Any, names: set[str]) -> Any:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in names:
                return child
        for child in value.values():
            found = first_nested_value(child, names)
            if found is not None:
                return found
    if isinstance(value, list):
        for child in value:
            found = first_nested_value(child, names)
            if found is not None:
                return found
    return None


def extract_completed_stages(raw_state: Any) -> list[str]:
    value = first_nested_value(
        raw_state,
        {
            "completed_stages",
            "completed_stage_names",
            "completed_requirements",
            "completed_steps",
            "completed_tasks",
        },
    )
    return coerce_text_list(value)


def canonical_tool_outcome(raw_state: Any, tool_output: str, error_payload: str) -> dict[str, Any]:
    label = extract_last_tool_execution_result_from_state(raw_state)
    label = normalize_last_tool_execution_result(label)
    error_message = (error_payload or "").strip()
    if not error_message and label in {-1, 0}:
        error_message = tool_output.strip()[:500]
    return {
        "success": label == 1 if label is not None else None,
        "label": label,
        "error_message": error_message,
        "summary": truncate_text(tool_output.strip(), 1200),
    }


def heuristic_canonical_observation(
    *,
    system_prompt: str,
    user_prompt: str,
    action: Any,
    raw_state: Any,
    tool_output: str,
    error_payload: str,
) -> dict[str, Any]:
    del system_prompt, user_prompt, action
    return normalize_canonical_observation(
        {
            "schema": "ewm_canonical_observation_v1",
            "tool_outcome": canonical_tool_outcome(raw_state, tool_output, error_payload),
            "stages": {
                "current_stage": state_current_stage(raw_state),
                "remaining_stages": state_remaining_stages(raw_state) or [],
                "completed_stages": extract_completed_stages(raw_state),
            },
            "evidence": [],
        },
        raw_state=raw_state,
        tool_output=tool_output,
        error_payload=error_payload,
    )


def normalize_canonical_observation(
    payload: Any,
    *,
    raw_state: Any,
    tool_output: str,
    error_payload: str,
) -> dict[str, Any]:
    if isinstance(payload, str):
        payload = parse_jsonish(strip_code_fence(payload))
    if not isinstance(payload, dict):
        payload = {}
    fallback = heuristic_canonical_observation(
        system_prompt="",
        user_prompt="",
        action={},
        raw_state=raw_state,
        tool_output=tool_output,
        error_payload=error_payload,
    ) if payload.get("schema") != "ewm_canonical_observation_v1" else None
    tool_outcome = payload.get("tool_outcome") if isinstance(payload.get("tool_outcome"), dict) else {}
    stages = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
    fallback_tool = (fallback or {}).get("tool_outcome", {})
    fallback_stages = (fallback or {}).get("stages", {})
    label = normalize_last_tool_execution_result(tool_outcome.get("label"))
    if label is None:
        label = fallback_tool.get("label")
    success = tool_outcome.get("success")
    if not isinstance(success, bool):
        success = label == 1 if label is not None else fallback_tool.get("success")
    normalized = {
        "schema": "ewm_canonical_observation_v1",
        "tool_outcome": {
            "success": success,
            "label": label,
            "error_message": str(tool_outcome.get("error_message") or fallback_tool.get("error_message") or "")[:800],
            "summary": truncate_text(str(tool_outcome.get("summary") or fallback_tool.get("summary") or tool_output or ""), 1200),
        },
        "stages": {
            "current_stage": stages.get("current_stage") or fallback_stages.get("current_stage") or "unknown",
            "remaining_stages": coerce_text_list(stages.get("remaining_stages") or fallback_stages.get("remaining_stages")),
            "completed_stages": coerce_text_list(stages.get("completed_stages") or fallback_stages.get("completed_stages")),
        },
        "evidence": coerce_text_list(payload.get("evidence"), limit=12),
    }
    return normalized


CANONICAL_OBSERVATION_SCHEMA: dict[str, Any] = {
    "schema": "ewm_canonical_observation_v1",
    "tool_outcome": {
        "success": None,
        "label": None,
        "error_message": "",
        "summary": "",
    },
    "stages": {
        "current_stage": "",
        "remaining_stages": [],
        "completed_stages": [],
    },
    "evidence": [],
}


def extract_canonical_observation_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str):
        payload = parse_jsonish(strip_code_fence(raw))
    else:
        raise ValueError(f"Canonicalizer returned {type(raw).__name__}, expected JSON object.")
    if not isinstance(payload, dict):
        raise ValueError(f"Canonicalizer returned {type(payload).__name__}, expected JSON object.")
    if payload.get("schema") != "ewm_canonical_observation_v1":
        raise ValueError("Canonicalizer response did not include schema ewm_canonical_observation_v1.")
    for key in ("tool_outcome", "stages", "evidence"):
        if key not in payload:
            raise ValueError(f"Canonicalizer response missing required key: {key}.")
    return payload


def build_canonicalizer_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    action: Any,
    raw_state: Any,
    tool_output: str,
    error_payload: str,
) -> list[dict[str, str]]:
    instruction = (
        "You convert an enterprise tool-use step into a compact canonical observation for a latent world model.\n"
        "Output contract: return exactly one JSON object and nothing else. Do not include reasoning, analysis, "
        "markdown, code fences, XML tags, comments, preambles, or explanations.\n"
        "The JSON object must have schema `ewm_canonical_observation_v1` and exactly these top-level keys: "
        "schema, tool_outcome, stages, evidence.\n"
        "`tool_outcome` must contain success, label, error_message, and summary. "
        "`label` must be -1, 0, 1, or null.\n"
        "`stages` must contain current_stage, remaining_stages, and completed_stages.\n"
        "Use the raw tool output and state as evidence, keep summaries concise, and do not invent completed stages."
    )
    content = {
        "system_prompt": system_prompt,
        "user_task": user_prompt,
        "action": action,
        "raw_tool_output": truncate_text(tool_output or "", 5000),
        "error_payload": truncate_text(error_payload or "", 2000),
        "state": raw_state,
        "fallback_stage_signals": {
            "current_stage": state_current_stage(raw_state),
            "remaining_stages": state_remaining_stages(raw_state) or [],
            "completed_stages": extract_completed_stages(raw_state),
            "tool_result_label": extract_last_tool_execution_result_from_state(raw_state),
        },
    }
    return [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": (
                "Canonicalize the following input. Return only the JSON object matching the required schema.\n\n"
                f"{safe_json(content)}"
            ),
        },
    ]


class CanonicalObservationConverter:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mode = args.canonical_observation_mode
        self.cache_path = args.canonicalizer_cache_path or (args.output_dir / "canonical_observation_cache.json")
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path.exists():
            loaded = load_json(self.cache_path)
            if isinstance(loaded, dict):
                self.cache = {str(key): value for key, value in loaded.items() if isinstance(value, dict)}
        self.generator = None
        if self.mode == "agent":
            self.generator = build_agent_generator(
                args.canonicalizer_model,
                max_new_tokens=args.canonicalizer_max_new_tokens,
                trust_remote_code=args.trust_remote_code,
                dtype=args.dtype,
                disable_chat_template=args.canonicalizer_disable_chat_template,
                attn_implementation=args.canonicalizer_attn_implementation,
                device_map=args.canonicalizer_device_map,
            )
        elif self.mode == "mixed_api":
            self.generator = MixedApiCanonicalizer(
                split_csv(args.canonicalizer_mixed_methods),
                seed=args.seed,
            )

    def save(self) -> None:
        if self.mode == "raw":
            return
        dump_json(self.cache_path, self.cache)

    def convert(self, example: Any) -> dict[str, Any]:
        tool_output = stringify_tool_output(example.tool_output or example.error_payload or "").strip()
        if not tool_output:
            tool_output = safe_json(example.state)
        if self.mode == "raw":
            return {
                "schema": "ewm_raw_observation_v1",
                "raw_tool_output": truncate_text(tool_output, self.args.max_observation_length * 8),
            }
        key = canonical_observation_cache_key(example)
        if key in self.cache:
            cached = self.cache[key]
            if "canonical_observation" in cached and isinstance(cached["canonical_observation"], dict):
                return cached["canonical_observation"]
            return cached
        if self.mode == "heuristic":
            canonical = heuristic_canonical_observation(
                system_prompt=example.system_prompt,
                user_prompt=example.user_prompt,
                action=example.action,
                raw_state=example.state,
                tool_output=tool_output,
                error_payload=example.error_payload,
            )
        else:
            provider = None
            try:
                raw = self.generator.generate_from_messages(
                    build_canonicalizer_messages(
                        system_prompt=example.system_prompt,
                        user_prompt=example.user_prompt,
                        action=example.action,
                        raw_state=example.state,
                        tool_output=tool_output,
                        error_payload=example.error_payload,
                    )
                )
                provider = getattr(self.generator, "last_method", None) or self.mode
                raw_payload = extract_canonical_observation_payload(raw)
                canonical = normalize_canonical_observation(
                    raw_payload,
                    raw_state=example.state,
                    tool_output=tool_output,
                    error_payload=example.error_payload,
                )
            except Exception as exc:
                canonical = heuristic_canonical_observation(
                    system_prompt=example.system_prompt,
                    user_prompt=example.user_prompt,
                    action=example.action,
                    raw_state=example.state,
                    tool_output=tool_output,
                    error_payload=example.error_payload,
                )
                provider = "heuristic_fallback"
                cache_error = f"{type(exc).__name__}: {exc}"[:500]
            else:
                cache_error = ""
            self.cache[key] = {
                "canonical_observation": canonical,
                "canonicalizer_mode": self.mode,
                "canonicalizer_provider": provider,
                "canonicalizer_error": cache_error,
            }
            if len(self.cache) % 50 == 0:
                self.save()
            return canonical
        self.cache[key] = canonical
        if len(self.cache) % 50 == 0:
            self.save()
        return canonical


def render_canonical_observation(payload: dict[str, Any], max_chars: int) -> str:
    return truncate_text(safe_json(payload), max_chars)


def build_canonical_goal_observation(user_prompt: str, system_prompt: str = "") -> dict[str, Any]:
    task = truncate_text(user_prompt.strip() or system_prompt.strip() or "complete the requested task", 1200)
    summary = f"Goal satisfied: {task}"
    return normalize_canonical_observation(
        {
            "schema": "ewm_canonical_observation_v1",
            "tool_outcome": {
                "success": True,
                "label": 1,
                "error_message": "",
                "summary": summary,
            },
            "stages": {
                "current_stage": "finished",
                "remaining_stages": [],
                "completed_stages": ["complete_requested_task"],
            },
            "evidence": [task],
        },
        raw_state={},
        tool_output=summary,
        error_payload="",
    )


def render_canonical_goal_observation(user_prompt: str, max_chars: int, system_prompt: str = "") -> str:
    return render_canonical_observation(
        build_canonical_goal_observation(user_prompt=user_prompt, system_prompt=system_prompt),
        max_chars=max_chars,
    )


def canonicalize_replay_observation(observation: Any) -> dict[str, Any]:
    if isinstance(observation, str):
        stripped = observation.strip()
        if stripped:
            try:
                parsed = parse_jsonish(strip_code_fence(stripped))
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("schema") == "ewm_canonical_observation_v1":
                return normalize_canonical_observation(
                    parsed,
                    raw_state={},
                    tool_output=stripped,
                    error_payload=str(parsed.get("tool_outcome", {}).get("error_message") or ""),
                )
    elif isinstance(observation, dict) and observation.get("schema") == "ewm_canonical_observation_v1":
        return normalize_canonical_observation(
            observation,
            raw_state={},
            tool_output=stringify_tool_output(observation),
            error_payload=str(observation.get("tool_outcome", {}).get("error_message") or ""),
        )

    tool_output = stringify_tool_output(observation or "").strip()
    error_payload = ""
    if isinstance(observation, dict):
        for key in ("error_message", "error", "exception"):
            if observation.get(key):
                error_payload = stringify_tool_output(observation.get(key)).strip()
                break
    return heuristic_canonical_observation(
        system_prompt="",
        user_prompt="",
        action={},
        raw_state=observation if isinstance(observation, dict) else {},
        tool_output=tool_output,
        error_payload=error_payload,
    )


def render_canonical_replay_history(history: list[dict[str, Any]], max_chars: int) -> str:
    rows = []
    for item in history[-WORLD_MODEL_INPUT_HISTORY_SIZE:]:
        observation = item.get("observation")
        if observation is None and "state" in item:
            observation = item.get("state")
        rows.append(
            {
                "step": item.get("step", item.get("imagined step")),
                "action": item.get("action"),
                "observation": canonicalize_replay_observation(observation),
            }
        )
    return truncate_text(safe_json(rows), max_chars)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_canonicalizer_method(method: str) -> str:
    aliases = {
        "gpt": "gpt5",
        "openai": "gpt5",
        "google": "gemini",
        "anthropic": "claude",
    }
    lowered = method.strip().lower()
    return aliases.get(lowered, lowered)


class MixedApiCanonicalizer:
    """Randomly route canonicalization calls across src.llm API backends."""

    def __init__(self, methods: list[str], seed: int) -> None:
        from src.llm import LLM

        if not methods:
            raise ValueError("MixedApiCanonicalizer requires at least one method.")
        self.methods = tuple(normalize_canonicalizer_method(method) for method in methods)
        self.rng = random.Random(seed)
        self.backends = {method: LLM(method) for method in self.methods}
        self.last_method: str | None = None

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> tuple[str | None, str]:
        system_parts = [
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system" and str(message.get("content", "")).strip()
        ]
        user_parts = [
            str(message.get("content", ""))
            for message in messages
            if message.get("role") != "system" and str(message.get("content", "")).strip()
        ]
        return "\n\n".join(system_parts).strip() or None, "\n\n".join(user_parts).strip()

    def _method_order(self) -> list[str]:
        methods = list(self.methods)
        first = self.rng.choice(methods)
        methods.remove(first)
        self.rng.shuffle(methods)
        return [first] + methods

    def generate_from_messages(self, messages: list[dict[str, str]], temperature: float = 0.0) -> dict[str, Any]:
        system_prompt, prompt = self._messages_to_prompt(messages)
        errors: list[str] = []
        for method in self._method_order():
            try:
                backend = self.backends[method]
                if method == "gpt5" and system_prompt:
                    backend.system_prompt_enable = True
                    backend.system_prompt = system_prompt
                    output = backend(prompt, None, temperature=temperature)
                elif method == "claude" and system_prompt:
                    prompt_for_method = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{prompt}"
                    output = backend(prompt_for_method, None, temperature=temperature)
                else:
                    output = backend(prompt, system_prompt, temperature=temperature)
                payload = extract_canonical_observation_payload(output)
                self.last_method = method
                return payload
            except Exception as exc:
                errors.append(f"{method}: {type(exc).__name__}: {exc}")
        raise RuntimeError("All mixed API canonicalizer methods failed: " + " | ".join(errors))


def extract_jepa_examples(trajectories: list[dict[str, Any]], args: argparse.Namespace, benchmark: str = "default") -> list[JepaExample]:
    # extract_state_examples only emits for `action`->`state` pairs carrying a
    # real last_tool_output; trajectory loading already drops any file with no
    # such signal at all (see trajectories_have_last_tool_output()), so there is
    # no state-free fallback here -- steps without a grounded observation simply
    # contribute no example.
    state_examples = extract_state_examples(trajectories, state_history_size=args.state_history_size)
    pending_rows: list[dict[str, Any]] = []
    raw_histories: dict[str, list[dict[str, Any]]] = {}

    for example in tqdm(state_examples, desc="raw_observations"):
        trajectory_key = str(example.trajectory_id)
        context_text = build_context_text(example.system_prompt, example.user_prompt)
        action_text = render_action(example.action)
        observation_text = render_raw_observation(example, args.max_observation_length * 8)
        success_label = extract_raw_success_label(example)
        history = raw_histories.setdefault(trajectory_key, [])
        pending_rows.append({
            "trajectory_id": example.trajectory_id,
            "trajectory_index": example.trajectory_index,
            "interaction_index": example.interaction_index,
            "context_text": context_text,
            "current_state_text": build_state_text(context_text, history),
            "action_text": action_text,
            "next_state_text": build_next_state_text(context_text, history, action_text, observation_text),
            "observation_text": observation_text,
            "success_label": success_label,
            "tool_name": action_tool_name(example.action),
            **({"history_action_texts": [h["action"] for h in history],
                "history_observation_texts": [h["observation"] for h in history]}
               if history_frames_enabled(args) else {}),
        })
        history.append({"step": len(history) + 1, "action": action_text, "observation": observation_text})
        raw_histories[trajectory_key] = history[-WORLD_MODEL_INPUT_HISTORY_SIZE:]

    # Terminal label: the LAST interaction_index seen per trajectory_id is the episode's
    # final step. Safe to compute here (rather than needing a separate whole-corpus pass)
    # because a trajectory never spans files (see load_and_extract_jepa_examples_streaming's
    # docstring) -- every one of a trajectory's steps is present in `pending_rows` already.
    max_interaction_index: dict[str, int] = {}
    for row in pending_rows:
        key = str(row["trajectory_id"])
        max_interaction_index[key] = max(max_interaction_index.get(key, -1), row["interaction_index"])
    for row in pending_rows:
        row["terminal_label"] = int(row["interaction_index"] == max_interaction_index[str(row["trajectory_id"])])

    attach_negative_actions(pending_rows, num_negatives=int(getattr(args, "action_contrastive_negatives", 0) or 0))
    return [JepaExample(**item, benchmark=benchmark) for item in pending_rows]


def _parse_action_payload(action_text: str) -> Any | None:
    try:
        return json.loads(action_text)
    except Exception:
        return None


def _first_action_arguments(payload: Any) -> tuple[dict[str, Any] | None, bool]:
    """Return the first call's top-level argument dict and whether it was encoded as a string."""
    function: Any = None
    if isinstance(payload, dict):
        calls = payload.get("tool_calls")
        if isinstance(calls, list) and calls and isinstance(calls[0], dict):
            function = calls[0].get("function")
        elif "arguments" in payload:
            function = payload
    if not isinstance(function, dict):
        return None, False
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        return arguments, False
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except Exception:
            return None, True
        return (parsed, True) if isinstance(parsed, dict) else (None, True)
    return None, False


def _set_first_action_arguments(payload: Any, arguments: dict[str, Any], *, as_string: bool) -> bool:
    function: Any = None
    if isinstance(payload, dict):
        calls = payload.get("tool_calls")
        if isinstance(calls, list) and calls and isinstance(calls[0], dict):
            function = calls[0].get("function")
        elif "arguments" in payload:
            function = payload
    if not isinstance(function, dict):
        return False
    function["arguments"] = safe_json(arguments) if as_string else arguments
    return True


def _one_field_action_negative(action_text: str, key: str, value: Any) -> str | None:
    payload = _parse_action_payload(action_text)
    if payload is None:
        return None
    mutated = copy.deepcopy(payload)
    arguments, as_string = _first_action_arguments(mutated)
    if not isinstance(arguments, dict) or key not in arguments:
        return None
    old_value = arguments.get(key)
    if safe_json(old_value) == safe_json(value):
        return None
    arguments[key] = copy.deepcopy(value)
    if not _set_first_action_arguments(mutated, arguments, as_string=as_string):
        return None
    rendered = render_action(mutated)
    return rendered if rendered and rendered != action_text else None


def attach_negative_actions(rows: list[dict[str, Any]], *, num_negatives: int, seed: int = 20260821) -> None:
    """Populate `negative_action_texts` in place: corrupted actions for the SAME state.

    Negative priority:
      hard-one-field  same tool, same rendered action envelope, exactly one top-level argument
                      replaced by a value observed for that same argument key in another
                      trajectory. These are the ID/enum-sensitive cases the old negatives made
                      too easy by changing many fields at once.
      wrong-arguments same tool name, another trajectory's full call, as a fallback when the
                      action cannot be minimally mutated.
      wrong-tool      a different tool that does NOT appear anywhere in this trajectory.

    The "not in this trajectory" restriction implements the memo's caution that a corrupted
    action must be genuinely wrong: a tool the episode legitimately uses elsewhere may just be a
    valid alternative ordering, so it is excluded rather than mislabelled as a negative. Actions
    are never executed, so these are plausible-but-unintended negatives, not verified-wrong ones.
    """
    if num_negatives <= 0:
        return
    rng = random.Random(seed)
    by_tool: dict[str, list[int]] = {}
    tools_by_trajectory: dict[str, set[str]] = {}
    parsed_args: dict[int, dict[str, Any]] = {}
    arg_values_by_tool: dict[str, dict[str, list[tuple[str, Any, int]]]] = {}
    for index, row in enumerate(rows):
        tool = row.get("tool_name")
        trajectory = str(row["trajectory_id"])
        tools_by_trajectory.setdefault(trajectory, set())
        if not tool:
            continue
        by_tool.setdefault(tool, []).append(index)
        tools_by_trajectory[trajectory].add(tool)
        payload = _parse_action_payload(str(row.get("action_text") or ""))
        arguments, _ = _first_action_arguments(payload)
        if isinstance(arguments, dict):
            parsed_args[index] = arguments
            for key, value in arguments.items():
                arg_values_by_tool.setdefault(tool, {}).setdefault(str(key), []).append((safe_json(value), value, index))

    all_tools = sorted(by_tool)
    for index, row in enumerate(rows):
        tool = row.get("tool_name")
        trajectory = str(row["trajectory_id"])
        action_text = str(row.get("action_text") or "")
        negatives: list[str] = []
        seen = {action_text}

        def add(candidate: str | None) -> None:
            if candidate and candidate not in seen and len(negatives) < num_negatives:
                negatives.append(candidate)
                seen.add(candidate)

        # Hard same-tool negatives: mutate exactly one top-level argument field.
        arguments = parsed_args.get(index)
        if tool and arguments:
            keys = list(arguments)
            rng.shuffle(keys)
            for key in keys:
                if len(negatives) >= num_negatives:
                    break
                current = safe_json(arguments.get(key))
                # Keep this bounded. On large corpora a common (tool, key) can have hundreds
                # of thousands of values; materializing/shuffling that list for every row makes
                # rank 0 spend hours in extraction while other DDP ranks time out at the barrier.
                values = arg_values_by_tool.get(tool, {}).get(str(key), [])
                if not values:
                    continue
                attempts = min(len(values), max(32, 8 * (num_negatives - len(negatives))))
                for _ in range(attempts):
                    serialized, value, source_index = values[rng.randrange(len(values))]
                    if serialized == current or str(rows[source_index]["trajectory_id"]) == trajectory:
                        continue
                    add(_one_field_action_negative(action_text, str(key), value))
                    if len(negatives) >= num_negatives:
                        break

        # Fallback: same tool, other trajectory's full arguments.
        if tool and len(negatives) < num_negatives:
            pool = [i for i in by_tool.get(tool, []) if str(rows[i]["trajectory_id"]) != trajectory]
            for i in rng.sample(pool, min(len(pool), max(1, num_negatives - len(negatives)))):
                add(rows[i]["action_text"])

        # Fallback: tool unused by this trajectory.
        unused = [t for t in all_tools if t not in tools_by_trajectory.get(trajectory, set())]
        rng.shuffle(unused)
        for other in unused:
            if len(negatives) >= num_negatives:
                break
            pool = by_tool.get(other) or []
            if pool:
                add(rows[rng.choice(pool)]["action_text"])
        row["negative_action_texts"] = negatives


def history_frames_enabled(args: argparse.Namespace) -> bool:
    """Does this run need the per-step (action, observation) history on each example?

    Two consumers, same data: --recurrent-state-init unrolls s_0 = I(c) through it, and
    --predictor-arch transformer attends over it as the frame sequence. Either one turns it on;
    nothing else pays the extra per-step tokenization and encoding.
    """
    return bool(getattr(args, "recurrent_state_init", False)) or (
        str(getattr(args, "predictor_arch", "mlp")) == "transformer"
    )


def extracted_examples_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    """Everything that changes what extraction produces, for the --reuse-extracted-examples gate.

    Covers the inputs (paths + their size/mtime, so an edited corpus invalidates the cache) and
    the extraction-time arguments. Tokenizer-time settings (max_input_length,
    truncate_states_keep_newest, event_state_decomposition, ...) are deliberately NOT here:
    they are applied in JepaTextDataset.__getitem__, not in the cached examples.
    """
    def stamp(paths: list[Path]) -> list[list[Any]]:
        out = []
        for path in sorted(paths, key=str):
            try:
                stat = path.stat()
                out.append([str(path), stat.st_size, int(stat.st_mtime)])
            except OSError:
                out.append([str(path), None, None])
        return out

    return {
        "train_data_paths": stamp(list(args.train_data_path)),
        "eval_data_paths": stamp(list(args.eval_data_path)) if not args.skip_eval else [],
        "skip_eval": bool(args.skip_eval),
        "state_history_size": int(args.state_history_size),
        "max_observation_length": int(args.max_observation_length),
        "action_contrastive_negatives": int(getattr(args, "action_contrastive_negatives", 0) or 0)
        if float(getattr(args, "action_contrastive_loss_coeff", 0.0) or 0.0) > 0 else 0,
        "action_contrastive_negative_generation_version": 3,
        "action_contrastive_loss_type": str(getattr(args, "action_contrastive_loss_type", "softplus")),
        "action_contrastive_margin": float(getattr(args, "action_contrastive_margin", 0.7)),
        "action_contrastive_temperature": float(getattr(args, "action_contrastive_temperature", 0.1)),
        "prediction_horizon": int(getattr(args, "prediction_horizon", 1) or 1),
        "max_train_examples": int(args.max_train_examples),
        "max_eval_examples": int(args.max_eval_examples),
        "seed": int(args.seed),
        "skip_web_trajectories": bool(getattr(args, "skip_web_trajectories", False)),
        "trajectory_chunk_size": int(getattr(args, "trajectory_chunk_size", 0) or 0),
        # Extraction-time, not tokenizer-time: history_action_texts/history_observation_texts are
        # written into the cached examples only when this is on, so a cache built without it must
        # not be reused by a run that needs the frame history (it would silently train with an
        # empty history instead of failing).
        "history_frames": history_frames_enabled(args),
    }


def load_and_extract_jepa_examples_streaming(
    paths: list[Path],
    args: argparse.Namespace,
    *,
    require_enterpriseops_gym: bool = False,
) -> list[JepaExample]:
    """Load + extract JEPA examples one trajectory file at a time.

    The multi-benchmark presets (`all`/`adp_all`) total tens of GB of raw JSON,
    which expands several-fold as nested Python objects. Loading every file at
    once (load_trajectory_paths -> extract_jepa_examples) holds all of that in
    host RAM simultaneously and OOM-kills rank 0 on a shared node. Streaming per
    file caps the raw-JSON peak at the single largest file, since each file's
    parsed trajectories are freed before the next is read. Histories are keyed by
    trajectory_id within a file and no trajectory spans files, so per-file
    extraction is equivalent to the previous whole-corpus call.
    """
    import gc

    examples: list[JepaExample] = []
    chunk_size = int(getattr(args, "trajectory_chunk_size", 0) or 0)
    for original_path in paths:
        path = resolve_streamable_path(original_path)
        if path != original_path:
            print(f"[stream] {original_path.name} -> reading {path.name} line-by-line", flush=True)
        benchmark = benchmark_key_from_path(original_path)
        if path.suffix != ".jsonl" or chunk_size <= 0:
            trajectories = load_trajectory_file(path, require_enterpriseops_gym=require_enterpriseops_gym)
            if not trajectories:
                continue
            examples.extend(extract_jepa_examples(trajectories, args, benchmark=benchmark))
            del trajectories
            gc.collect()
            continue

        # Chunked streaming: raw trajectories for one chunk are freed before the next is read,
        # so peak RSS is (chunk of raw trajectories) + (examples accumulated so far) instead of
        # (whole file of raw trajectories) + (that file's examples) simultaneously.
        #
        # Chunking splits extract_jepa_examples' per-trajectory_id history state, which is only
        # equivalent if no trajectory_id spans a chunk boundary. Each JSONL record is one whole
        # trajectory, so that holds unless a file repeats an id; repeats are detected and
        # reported rather than silently changing what gets built.
        flushed_ids: set[str] = set()      # ids already extracted in an earlier chunk
        buffer_ids: set[str] = set()       # ids in the chunk being accumulated
        split_ids: set[str] = set()
        buffer: list[dict[str, Any]] = []
        produced = 0
        any_observation = False

        # extract_jepa_examples numbers trajectories with enumerate() over the list it is
        # given, so a chunk boundary would restart trajectory_index at 0 and (for records with
        # no trajectory_id) restart the id fallback too. Carry a running offset and stamp a
        # deterministic id so chunked output is identical to whole-file output.
        offset = 0

        def flush(buf: list[dict[str, Any]]) -> None:
            nonlocal produced, any_observation, offset
            if not buf:
                return
            if trajectories_have_last_tool_output(buf):
                any_observation = True
            new_examples = extract_jepa_examples(buf, args, benchmark=benchmark)
            if offset:
                for example in new_examples:
                    example.trajectory_index += offset
            examples.extend(new_examples)
            produced += len(new_examples)
            offset += len(buf)

        global_index = 0
        for trajectory in iter_trajectory_file(path):
            # Stable across chunkings: without this, a record lacking trajectory_id would get
            # extract_state_examples' positional fallback, which is chunk-relative.
            if not trajectory.get("trajectory_id"):
                trajectory["trajectory_id"] = global_index
            global_index += 1
            key = str(trajectory.get("trajectory_id", ""))
            if key:
                # O(1) per record: an id is "split" only if an EARLIER chunk already
                # extracted it. Scanning the buffer here would be O(n * chunk_size).
                if key in flushed_ids and key not in buffer_ids:
                    split_ids.add(key)
                buffer_ids.add(key)
            buffer.append(trajectory)
            if len(buffer) >= chunk_size:
                flush(buffer)
                flushed_ids |= buffer_ids
                buffer_ids = set()
                buffer = []
                gc.collect()
        flush(buffer)
        flushed_ids |= buffer_ids
        buffer = []
        gc.collect()

        if not any_observation:
            print(f"[skip] {path}: no state message carries last_tool_output; "
                  f"ignoring for JEPA training.")
            examples = examples[: len(examples) - produced]
        if split_ids:
            print(f"[stream] WARNING: {len(split_ids)} trajectory_id(s) in {path.name} appear in "
                  f"more than one chunk; their per-trajectory history restarts at the boundary. "
                  f"Raise --trajectory-chunk-size or leave the file as .json to avoid this.",
                  flush=True)
    return examples


class JepaTextDataset(Dataset):
    def __init__(self, examples: list[JepaExample], tokenizer: Any, args: argparse.Namespace, tool_vocab: dict[str, int] | None = None) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.args = args
        self.tool_vocab = tool_vocab  # {tool_name: idx} enables the action-head labels
        self.obs_ground = float(getattr(args, "obs_token_ground_coeff", 0.0) or 0.0) > 0.0
        self.action_contrastive = (
            int(getattr(args, "action_contrastive_negatives", 0) or 0)
            if float(getattr(args, "action_contrastive_loss_coeff", 0.0) or 0.0) > 0 else 0
        )
        self.obs_ground_max = getattr(args, "obs_ground_max_tokens", None)
        self.horizon = max(1, int(getattr(args, "prediction_horizon", 1)))
        # For multi-step supervision, precompute up to (horizon-1) CONSECUTIVE successor
        # example indices per example (same trajectory_id, interaction_index + 1, + 2, ...).
        # Their action_text/next_state_text supply the teacher-forced future actions and the
        # true future-state targets for the recursive rollout. Consecutive-only, so a
        # subsampled/missing middle step simply shortens that example's usable horizon.
        self._successors: list[list[int]] | None = self._build_successor_map(examples) if self.horizon > 1 else None

    def _build_successor_map(self, examples: list[JepaExample]) -> list[list[int]]:
        pos = {(str(ex.trajectory_id), int(ex.interaction_index)): idx for idx, ex in enumerate(examples)}
        max_future = self.horizon - 1
        successors: list[list[int]] = []
        for ex in examples:
            trajectory_id = str(ex.trajectory_id)
            step = int(ex.interaction_index) + 1
            future: list[int] = []
            while len(future) < max_future and (trajectory_id, step) in pos:
                future.append(pos[(trajectory_id, step)])
                step += 1
            successors.append(future)
        return successors

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.examples[index]
        keep_newest = bool(getattr(self.args, "truncate_states_keep_newest", False))
        current = tokenize_state_text(
            self.tokenizer, item.current_state_text, self.args.max_input_length, keep_newest=keep_newest
        )
        # --event-target: encode ONLY the newly-observed environment output as the prediction
        # target, instead of the cumulative next-state text. Consecutive cumulative states
        # overlap almost entirely, which is what lets the predictor minimize the loss with a
        # persistence map; the observation is the part that actually changed.
        #
        # The target deliberately excludes the action. E([a_t; o_{t+1}]) would let the predictor
        # lower the loss by copying its own action input -- trading the persistence shortcut for
        # an action-copy shortcut. E(o_{t+1}) admits neither.
        # --event-state-decomposition keeps the cumulative state target (the State Updater is
        # trained against it) and adds a SEPARATE event stream. Plain --event-target instead
        # swaps the single target for the observation -- the step-1 "event predictor only" mode.
        if getattr(self.args, "event_target", False) and not getattr(self.args, "event_state_decomposition", False):
            next_state = self.tokenizer(
                item.observation_text,
                max_length=self.args.max_observation_length,
                truncation=True,
                add_special_tokens=True,
            )
        else:
            next_state = tokenize_state_text(
                self.tokenizer, item.next_state_text, self.args.max_input_length, keep_newest=keep_newest
            )
        context = self.tokenizer(
            item.context_text,
            max_length=self.args.max_input_length,
            truncation=True,
            add_special_tokens=True,
        )
        action = self.tokenizer(
            item.action_text,
            max_length=self.args.max_action_length,
            truncation=True,
            add_special_tokens=True,
        )
        observation = self.tokenizer(
            item.observation_text,
            max_length=self.args.max_observation_length,
            truncation=True,
            add_special_tokens=True,
        )
        if history_frames_enabled(self.args):
            # Encoded per step to unroll s_0 = I(c) -> s_t, and/or to give the transformer
            # predictor its frame sequence. Actions and observations are short
            # (max_action_length / max_observation_length), unlike the cumulative state text
            # this replaces, so the extra encodes are not as costly as the count suggests.
            hist_a = [self.tokenizer(t, max_length=self.args.max_action_length, truncation=True,
                                     add_special_tokens=True)["input_ids"]
                      for t in item.history_action_texts]
            hist_o = [self.tokenizer(t, max_length=self.args.max_observation_length, truncation=True,
                                     add_special_tokens=True)["input_ids"]
                      for t in item.history_observation_texts]
            n = min(len(hist_a), len(hist_o))
        result = {
            **({"history_action_input_ids": hist_a[:n],
                "history_event_input_ids": hist_o[:n]}
               if history_frames_enabled(self.args) else {}),
            **({"event_input_ids": observation["input_ids"],
                "event_attention_mask": observation["attention_mask"]}
               if getattr(self.args, "event_state_decomposition", False) else {}),
            "current_input_ids": current["input_ids"],
            "current_attention_mask": current["attention_mask"],
            "next_input_ids": next_state["input_ids"],
            "next_attention_mask": next_state["attention_mask"],
            "context_input_ids": context["input_ids"],
            "context_attention_mask": context["attention_mask"],
            "action_input_ids": action["input_ids"],
            "action_attention_mask": action["attention_mask"],
            "labels": observation["input_ids"],
            "success_label": -1 if item.success_label is None else int(item.success_label),
            "terminal_label": -1 if item.terminal_label is None else int(item.terminal_label),
            "trajectory_index": item.trajectory_index,
            "interaction_index": item.interaction_index,
        }
        if self.horizon > 1 and self._successors is not None:
            # Teacher-forced future actions + true future-state targets from consecutive
            # successors, for the recursive multi-step rollout in TextLeWorldModel.forward.
            future_action_input_ids: list[list[int]] = []
            future_action_attention_mask: list[list[int]] = []
            future_next_input_ids: list[list[int]] = []
            future_next_attention_mask: list[list[int]] = []
            for successor_index in self._successors[index]:
                successor = self.examples[successor_index]
                future_action = self.tokenizer(
                    successor.action_text, max_length=self.args.max_action_length, truncation=True, add_special_tokens=True
                )
                # Under the recurrent event/state pipeline the horizon targets are the
                # successors' OBSERVATIONS (future events), not their cumulative next states --
                # L_future asks "can s_{t+1} still predict the next event?", which is the only
                # training signal the State Updater gets.
                if getattr(self.args, "event_state_recurrent", False):
                    future_next = self.tokenizer(
                        successor.observation_text,
                        max_length=self.args.max_observation_length,
                        truncation=True,
                        add_special_tokens=True,
                    )
                else:
                    future_next = tokenize_state_text(
                        self.tokenizer, successor.next_state_text, self.args.max_input_length,
                        keep_newest=keep_newest,
                    )
                future_action_input_ids.append(future_action["input_ids"])
                future_action_attention_mask.append(future_action["attention_mask"])
                future_next_input_ids.append(future_next["input_ids"])
                future_next_attention_mask.append(future_next["attention_mask"])
            result["future_action_input_ids"] = future_action_input_ids
            result["future_action_attention_mask"] = future_action_attention_mask
            result["future_next_input_ids"] = future_next_input_ids
            result["future_next_attention_mask"] = future_next_attention_mask
        if self.obs_ground:
            obs_ids = observation_ground_target_ids(
                item.observation_text, item.benchmark, self.tokenizer, max_tokens=self.obs_ground_max
            )
            if obs_ids:
                result["obs_ground_ids"] = obs_ids
        if self.tool_vocab is not None:
            result["tool_label"] = self.tool_vocab.get(item.tool_name, 0)
        if self.action_contrastive and item.negative_action_texts:
            negatives = item.negative_action_texts[: self.action_contrastive]
            encoded = [
                self.tokenizer(text, max_length=self.args.max_action_length, truncation=True,
                               add_special_tokens=True)["input_ids"]
                for text in negatives
            ]
            result["negative_action_input_ids"] = encoded
        return result


def tokenize_state_text(tokenizer: Any, text: str, max_length: int, *, keep_newest: bool) -> dict[str, list[int]]:
    """Tokenize a STATE text, optionally truncating from the LEFT so the newest content survives.

    State texts are rendered oldest-first (system/task context, then history in step order), and
    HF's default right-side truncation keeps the BEGINNING -- so for over-length states the
    newest appended step (the just-executed action and its observation) is exactly what gets cut.
    Measured on the canonical JSONL at max_length=2048: the new content was FULLY truncated away
    for ~31% of latent-target pairs, making z_next's input literally identical to z_current's and
    turning the latent loss into identity-map supervision. `keep_newest` flips truncation to the
    left: the head of the (partially redundant) context/history prefix is dropped instead --
    tolerable because the system/task context is separately encoded as z_context anyway.
    """
    if not keep_newest:
        return tokenizer(text, max_length=max_length, truncation=True, add_special_tokens=True)
    previous_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        return tokenizer(text, max_length=max_length, truncation=True, add_special_tokens=True)
    finally:
        tokenizer.truncation_side = previous_side


def pad_token_rows(rows: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(row) for row in rows)
    return torch.tensor([row + [pad_value] * (max_len - len(row)) for row in rows], dtype=torch.long)


class JepaCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def pad(self, rows: list[list[int]], pad_value: int) -> torch.Tensor:
        return pad_token_rows(rows, pad_value)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id
        batch = {}
        for key in (
            "current_input_ids",
            "next_input_ids",
            "context_input_ids",
            "action_input_ids",
        ):
            batch[key] = self.pad([f[key] for f in features], pad_id)
        for key in (
            "current_attention_mask",
            "next_attention_mask",
            "context_attention_mask",
            "action_attention_mask",
        ):
            batch[key] = self.pad([f[key] for f in features], 0)
        if "event_input_ids" in features[0]:
            batch["event_input_ids"] = self.pad([f["event_input_ids"] for f in features], pad_id)
            batch["event_attention_mask"] = self.pad([f["event_attention_mask"] for f in features], 0)
        if "history_action_input_ids" in features[0]:
            depth = max(len(f["history_action_input_ids"]) for f in features)
            if depth:
                for key in ("history_action_input_ids", "history_event_input_ids"):
                    slots = [
                        self.pad([f[key][i] if i < len(f[key]) else [pad_id] for f in features], pad_id)
                        for i in range(depth)
                    ]
                    width = max(t.shape[1] for t in slots)
                    stacked = torch.stack(
                        [torch.nn.functional.pad(t, (0, width - t.shape[1]), value=pad_id) for t in slots],
                        dim=1,
                    )
                    batch[key] = stacked
                    batch[key.replace("input_ids", "attention_mask")] = (stacked != pad_id).to(torch.long)
                batch["history_step_mask"] = torch.tensor(
                    [[1.0 if i < len(f["history_action_input_ids"]) else 0.0 for i in range(depth)]
                     for f in features], dtype=torch.float)
        labels = self.pad([f["labels"] for f in features], pad_id)
        labels = labels.masked_fill(labels == pad_id, -100)
        batch["labels"] = labels
        success_labels = torch.tensor([float(f.get("success_label", -1)) for f in features], dtype=torch.float)
        batch["success_labels"] = success_labels.clamp_min(0.0)
        batch["success_label_mask"] = (success_labels >= 0).to(torch.float)
        terminal_labels = torch.tensor([float(f.get("terminal_label", -1)) for f in features], dtype=torch.float)
        batch["terminal_labels"] = terminal_labels.clamp_min(0.0)
        batch["terminal_label_mask"] = (terminal_labels >= 0).to(torch.float)
        # Multi-step (--prediction-horizon>1): stack the per-step future action / next-state
        # token rows into [B, K, L] tensors and a [B, K] validity mask. K = the largest number
        # of successors present in this batch; examples with fewer are zero-masked for the
        # missing steps. Skipped entirely when no example carries any successor.
        future_lengths = [len(f.get("future_action_input_ids", [])) for f in features]
        num_future = max(future_lengths, default=0)
        if num_future > 0:
            def stack_future(key: str, pad_value: int) -> torch.Tensor:
                per_step = []
                for step in range(num_future):
                    rows = [f[key][step] if step < len(f.get(key, [])) else [pad_value] for f in features]
                    per_step.append(self.pad(rows, pad_value))  # [B, L_step]
                max_len = max(tensor.shape[1] for tensor in per_step)
                per_step = [
                    torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[1]), value=pad_value)
                    for tensor in per_step
                ]
                return torch.stack(per_step, dim=1)  # [B, K, L]

            batch["future_action_input_ids"] = stack_future("future_action_input_ids", pad_id)
            batch["future_action_attention_mask"] = stack_future("future_action_attention_mask", 0)
            batch["future_next_input_ids"] = stack_future("future_next_input_ids", pad_id)
            batch["future_next_attention_mask"] = stack_future("future_next_attention_mask", 0)
            step_mask = torch.zeros(len(features), num_future, dtype=torch.float)
            for row, length in enumerate(future_lengths):
                step_mask[row, :length] = 1.0
            batch["future_step_mask"] = step_mask
        # Option-C observation-token grounding targets: pad the per-example first-N
        # token ids to [B, T] and emit a [B, T] mask (rows with no target are all-zero).
        obs_lengths = [len(f.get("obs_ground_ids", [])) for f in features]
        if max(obs_lengths, default=0) > 0:
            width = max(obs_lengths)
            ids = torch.zeros(len(features), width, dtype=torch.long)
            mask = torch.zeros(len(features), width, dtype=torch.float)
            for row, f in enumerate(features):
                seq = f.get("obs_ground_ids", [])
                if seq:
                    ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                    mask[row, : len(seq)] = 1.0
            batch["obs_ground_ids"] = ids
            batch["obs_ground_mask"] = mask
        if "tool_label" in features[0]:
            batch["tool_label"] = torch.tensor([f["tool_label"] for f in features], dtype=torch.long)
        # Corrupted actions for the action-contrastive term: [B, K, L] plus a [B, K] validity
        # mask, since an example may yield fewer than K usable negatives.
        neg_counts = [len(f.get("negative_action_input_ids", [])) for f in features]
        num_negatives = max(neg_counts, default=0)
        if num_negatives > 0:
            rows_per_slot = []
            for slot in range(num_negatives):
                rows_per_slot.append(self.pad(
                    [f["negative_action_input_ids"][slot] if slot < len(f.get("negative_action_input_ids", []))
                     else [pad_id] for f in features], pad_id))
            width = max(t.shape[1] for t in rows_per_slot)
            stacked = torch.stack([
                torch.nn.functional.pad(t, (0, width - t.shape[1]), value=pad_id) for t in rows_per_slot
            ], dim=1)                                  # [B, K, L]
            batch["negative_action_input_ids"] = stacked
            batch["negative_action_attention_mask"] = (stacked != pad_id).to(torch.long)
            batch["negative_action_valid"] = torch.tensor(
                [[1.0 if slot < c else 0.0 for slot in range(num_negatives)] for c in neg_counts],
                dtype=torch.float)
        return batch


@dataclass
class CanonicalEventExample:
    trajectory_id: str
    interaction_index: int
    context_text: str
    current_state_text: str
    action_text: str
    single_labels: dict[str, str]
    multi_labels: dict[str, list[str]]
    benchmark: str = ""
    observation_text: str = ""  # tool output of this action, for the recognition probe (often absent)
    value_target: float | None = None  # discounted return-to-go; only present on *_value_scored.jsonl rows
    # --joint-canonical-event-training only: the NEXT row's state text within the same
    # trajectory = the state after THIS row's action, i.e. the latent-prediction target.
    # Empty for a trajectory's terminal row (no successor) -> latent/SIGReg masked out.
    next_state_text: str = ""
    terminal_label: int | None = None  # 1 if this is the last interaction_index in its trajectory
    # Per-step (action, tool output) log preceding this row, for --predictor-arch transformer's
    # frame sequence. Built from the row's `input_history`, which is the same list that
    # current_state_text renders -- so the heads-only phase feeds the predictor the same token
    # sequence the pretraining phase did, instead of the degenerate [context, z_current].
    history_action_texts: list[str] = field(default_factory=list)
    history_observation_texts: list[str] = field(default_factory=list)


def canonical_event_observation_text(record: dict[str, Any]) -> str:
    """Extract the tool output / observation that this action produced, if the row stores it
    (for --canonical-event-recognition-probe). Current label rows usually omit it, so this
    returns "" -> the probe reports 0% coverage until the JSONL is regenerated with it."""
    for key in ("observation", "tool_output", "last_tool_output", "output", "result"):
        value = record.get(key)
        if value:
            return stringify_tool_output(value).strip()
    return ""


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_canonical_event_records(path: Path) -> list[dict[str, Any]]:
    return load_jsonl_rows(path)


def canonical_event_field_value(record: dict[str, Any], field: str) -> Any:
    if field in CANONICAL_EVENT_STATE_FIELDS:
        return (record.get("canonical_event_state") or {}).get(field)
    return (record.get("nudge") or {}).get(field)


def canonical_event_history_action_text(action: Any) -> str:
    """Render a history action the way the row's OWN action is rendered.

    The canonical JSONL stores the row's action as {"tool_calls": [{"function": {name,
    arguments}}]} but flattens history entries to a bare {"name", "arguments"} -- or, for a step
    that issued several calls at once, to a LIST of those. Encoding the two shapes differently
    would make the position-aligned AdaLN conditioning inconsistent between the history positions
    and the candidate position, so both flattened forms are lifted back into the same envelope
    before rendering. Anything already in another shape passes through.
    """
    def is_flat_call(value: Any) -> bool:
        return isinstance(value, dict) and "tool_calls" not in value and ("name" in value or "arguments" in value)

    if is_flat_call(action):
        action = [action]
    if isinstance(action, list) and action and all(is_flat_call(item) for item in action):
        action = {"tool_calls": [
            {"function": {"name": item.get("name"), "arguments": item.get("arguments")}}
            for item in action
        ]}
    return render_action(action)


def canonical_event_history_texts(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(action_texts, observation_texts) from `input_history`, oldest first, trimmed to the same
    WORLD_MODEL_INPUT_HISTORY_SIZE window render_history uses so the two views agree."""
    history = record.get("input_history") or []
    if not isinstance(history, list):
        return [], []
    actions: list[str] = []
    observations: list[str] = []
    for item in history[-WORLD_MODEL_INPUT_HISTORY_SIZE:]:
        if not isinstance(item, dict):
            continue
        actions.append(canonical_event_history_action_text(item.get("action")))
        observations.append(str(item.get("observation") or ""))
    return actions, observations


def build_canonical_event_example(record: dict[str, Any]) -> CanonicalEventExample | None:
    """Turn one canonical_event_with_nudge JSONL row into a training example.

    Returns None for rows missing a required label field -- these are
    classification targets, not something to impute a placeholder for.
    """
    single_labels: dict[str, str] = {}
    for field in CANONICAL_EVENT_SINGLE_LABEL_FIELDS:
        value = canonical_event_field_value(record, field)
        if value is None:
            return None
        single_labels[field] = str(value)
    multi_labels: dict[str, list[str]] = {}
    for field in NUDGE_MULTI_LABEL_FIELDS:
        value = canonical_event_field_value(record, field)
        if not isinstance(value, list) or not value:
            return None
        multi_labels[field] = [str(item) for item in value]
    context_text = build_context_text(str(record.get("system_prompt") or ""), str(record.get("task_prompt") or ""))
    value_target = record.get("value_target")
    history_actions, history_observations = canonical_event_history_texts(record)
    return CanonicalEventExample(
        history_action_texts=history_actions,
        history_observation_texts=history_observations,
        trajectory_id=str(record.get("trajectory_id", "")),
        interaction_index=int(record.get("interaction_index") or 0),
        context_text=context_text,
        current_state_text=build_state_text(context_text, record.get("input_history") or []),
        action_text=render_action(record.get("action")),
        single_labels=single_labels,
        multi_labels=multi_labels,
        benchmark=str(record.get("benchmark") or ""),
        observation_text=canonical_event_observation_text(record),
        # Only present on rows produced by annotate_step_value_scores.py -- the plain
        # canonical_event_with_nudge JSONL has no reward signal, so this stays None there
        # and the value loss is masked out for those examples (see CanonicalEventCollator).
        value_target=float(value_target) if isinstance(value_target, (int, float)) else None,
    )


def link_canonical_event_successors(examples: list[CanonicalEventExample]) -> None:
    """Populate next_state_text / terminal_label in place, per trajectory.

    For --joint-canonical-event-training the latent-prediction target for step t is the state
    AFTER action t. These JSONL rows carry no observation field, but row t+1's input_history is
    row t's history plus row t's own (action, observation) -- verified on the real data: the
    appended item's `step` equals row t's interaction_index + 1 for every consecutive pair --
    so row t+1's already-built current_state_text IS that post-action state.

    Only CONSECUTIVE successors count (interaction_index + 1). A trajectory's terminal row (and
    any row whose successor is missing, e.g. dropped by cleaning) keeps next_state_text="" and
    is masked out of the latent/SIGReg terms rather than being paired with a wrong target.
    """
    by_trajectory: dict[str, dict[int, CanonicalEventExample]] = {}
    for example in examples:
        by_trajectory.setdefault(str(example.trajectory_id), {})[int(example.interaction_index)] = example
    for steps in by_trajectory.values():
        last_index = max(steps)
        for index, example in steps.items():
            successor = steps.get(index + 1)
            example.next_state_text = successor.current_state_text if successor is not None else ""
            example.terminal_label = int(index == last_index)


def build_canonical_event_examples(records: list[dict[str, Any]]) -> list[CanonicalEventExample]:
    examples = []
    for record in records:
        example = build_canonical_event_example(record)
        if example is not None:
            examples.append(example)
    link_canonical_event_successors(examples)
    return examples


def canonical_event_benchmark_counts(examples: list[CanonicalEventExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        key = example.benchmark or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def subsample_canonical_event_examples_by_benchmark(
    examples: list[CanonicalEventExample],
    percentage: float,
    *,
    seed: int,
) -> list[CanonicalEventExample]:
    """Keep ``percentage`` of the examples from each benchmark (few-shot).

    Sampling is stratified per benchmark -- e.g. 10% keeps 10% of CRMArenaPro,
    10% of EnterpriseOps-Gym, and 10% of Terminal-Bench-2.0 rather than 10%
    overall (which the largest benchmark would dominate). Each non-empty
    benchmark retains at least one example so a benchmark is never fully dropped.

    The subset is fully reproducible: for a given ``seed`` and ``percentage`` it
    always selects the exact same examples across executions. It depends only on
    each example's stable identity (trajectory_id, interaction_index) -- not on
    the order rows happen to appear in the JSONL -- because each benchmark group
    is sorted by that identity before the seeded shuffle. ``seed`` is intended to
    be a dedicated few-shot selection seed, independent of the training seed, so
    the same subset can be reused across training runs that vary other
    randomness. Because ``keep_count`` is a prefix of the same shuffled order, a
    larger percentage is a superset of a smaller one at the same seed.
    """
    if percentage >= 100.0:
        return list(examples)
    groups: dict[str, list[CanonicalEventExample]] = {}
    for example in examples:
        groups.setdefault(example.benchmark or "unknown", []).append(example)
    kept: list[CanonicalEventExample] = []
    for offset, benchmark in enumerate(sorted(groups)):
        # Sort by stable identity first so selection is independent of JSONL row
        # order, then a per-benchmark seeded shuffle (independent yet reproducible).
        group = sorted(
            groups[benchmark],
            key=lambda example: (str(example.trajectory_id), int(example.interaction_index)),
        )
        random.Random(seed + offset).shuffle(group)
        keep_count = min(len(group), max(1, int(round(len(group) * percentage / 100.0))))
        kept.extend(group[:keep_count])
    return kept


def build_canonical_event_vocabularies(
    *example_lists: list[CanonicalEventExample],
    fields: tuple[str, ...] | None = None,
) -> dict[str, list[str]]:
    """Build a sorted value vocabulary per field from every example seen.

    Train and eval examples are both passed in so the eval split never hits an
    out-of-vocabulary label -- these are small, closed-ish category sets (see
    CANONICAL_EVENT_ALL_FIELDS), not open-ended text.

    `fields` (--canonical-event-heads beam_plan) restricts the vocabulary to a subset; the
    vocab is the single source of truth for which heads exist downstream (model construction,
    dataset label emission, loss, eval, and replay via canonical_event_vocab.json), so
    restricting it here restricts everything.
    """
    keep = tuple(fields) if fields is not None else CANONICAL_EVENT_ALL_FIELDS
    vocab: dict[str, set[str]] = {field: set() for field in keep}
    for examples in example_lists:
        for example in examples:
            for field, value in example.single_labels.items():
                if field in vocab:
                    vocab[field].add(value)
            for field, values in example.multi_labels.items():
                if field in vocab:
                    vocab[field].update(values)
    return {field: sorted(values) for field, values in vocab.items()}


class CanonicalEventDataset(Dataset):
    def __init__(
        self,
        examples: list[CanonicalEventExample],
        tokenizer: Any,
        args: argparse.Namespace,
        vocab: dict[str, list[str]],
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.args = args
        self.vocab_index = {field: {value: index for index, value in enumerate(values)} for field, values in vocab.items()}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.examples[index]
        keep_newest = bool(getattr(self.args, "truncate_states_keep_newest", False))
        current = tokenize_state_text(
            self.tokenizer, item.current_state_text, self.args.max_input_length, keep_newest=keep_newest
        )
        context = self.tokenizer(item.context_text, max_length=self.args.max_input_length, truncation=True, add_special_tokens=True)
        action = self.tokenizer(item.action_text, max_length=self.args.max_action_length, truncation=True, add_special_tokens=True)
        row: dict[str, Any] = {
            "current_input_ids": current["input_ids"],
            "current_attention_mask": current["attention_mask"],
            "context_input_ids": context["input_ids"],
            "context_attention_mask": context["attention_mask"],
            "action_input_ids": action["input_ids"],
            "action_attention_mask": action["attention_mask"],
        }
        if history_frames_enabled(self.args):
            # Same keys, lengths and ordering as JepaTextDataset.__getitem__, so the transformer
            # predictor sees the token sequence it was pretrained on rather than falling back to
            # the 2-token [context, z_current] degenerate case.
            hist_a = [self.tokenizer(text, max_length=self.args.max_action_length, truncation=True,
                                     add_special_tokens=True)["input_ids"]
                      for text in item.history_action_texts]
            hist_o = [self.tokenizer(text, max_length=self.args.max_observation_length, truncation=True,
                                     add_special_tokens=True)["input_ids"]
                      for text in item.history_observation_texts]
            depth = min(len(hist_a), len(hist_o))
            row["history_action_input_ids"] = hist_a[:depth]
            row["history_event_input_ids"] = hist_o[:depth]
        # vocab_index carries only the fields this run trains (--canonical-event-heads);
        # labels for dropped fields are simply not emitted.
        for field in CANONICAL_EVENT_SINGLE_LABEL_FIELDS:
            if field in self.vocab_index:
                row[f"label_{field}"] = self.vocab_index[field][item.single_labels[field]]
        for field in NUDGE_MULTI_LABEL_FIELDS:
            if field not in self.vocab_index:
                continue
            index_map = self.vocab_index[field]
            multi_hot = [0.0] * len(index_map)
            for value in item.multi_labels[field]:
                multi_hot[index_map[value]] = 1.0
            row[f"label_{field}"] = multi_hot
        if getattr(self.args, "canonical_event_recognition_probe", False) and item.observation_text:
            observation = self.tokenizer(item.observation_text, max_length=self.args.max_input_length, truncation=True, add_special_tokens=True)
            row["canonical_event_observation_input_ids"] = observation["input_ids"]
            row["canonical_event_observation_attention_mask"] = observation["attention_mask"]
        row["value_target"] = item.value_target if item.value_target is not None else 0.0
        row["has_value_target"] = item.value_target is not None
        row["terminal_label"] = -1 if item.terminal_label is None else int(item.terminal_label)
        if getattr(self.args, "joint_canonical_event_training", False):
            # Latent-prediction target: the successor row's state text. Terminal rows have none,
            # so tokenize the current state as a placeholder and mask the loss instead -- keeps
            # every batch tensor the same shape without inventing a bogus target.
            has_next = bool(item.next_state_text)
            next_state = tokenize_state_text(
                self.tokenizer,
                item.next_state_text if has_next else item.current_state_text,
                self.args.max_input_length,
                keep_newest=keep_newest,
            )
            row["next_input_ids"] = next_state["input_ids"]
            row["next_attention_mask"] = next_state["attention_mask"]
            row["has_next_state"] = has_next
        return row


class CanonicalEventCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id
        batch: dict[str, Any] = {}
        for key in ("current_input_ids", "context_input_ids", "action_input_ids"):
            batch[key] = pad_token_rows([f[key] for f in features], pad_id)
        for key in ("current_attention_mask", "context_attention_mask", "action_attention_mask"):
            batch[key] = pad_token_rows([f[key] for f in features], 0)
        # TextLeWorldModel.forward() unconditionally encodes a "next state". With
        # --joint-canonical-event-training the rows carry a REAL next-state text (the successor
        # row's state), which is the latent-prediction target; `latent_target_mask` marks the
        # rows that actually have one. Otherwise there is no such text, so alias the current
        # state -- z_next is simply unused when only compute_canonical_event/compute_value run.
        if "next_input_ids" in features[0]:
            batch["next_input_ids"] = pad_token_rows([f["next_input_ids"] for f in features], pad_id)
            batch["next_attention_mask"] = pad_token_rows([f["next_attention_mask"] for f in features], 0)
            batch["latent_target_mask"] = torch.tensor(
                [float(f.get("has_next_state", False)) for f in features], dtype=torch.float
            )
        else:
            batch["next_input_ids"] = batch["current_input_ids"]
            batch["next_attention_mask"] = batch["current_attention_mask"]
        # Frame history for --predictor-arch transformer. Stacked to [B, depth, L] with a
        # [B, depth] validity mask, byte-for-byte the same layout JepaTextCollator produces --
        # encode_frame_history right-aligns from that mask, so shorter histories are handled
        # there rather than here.
        if "history_action_input_ids" in features[0]:
            depth = max(len(f["history_action_input_ids"]) for f in features)
            if depth:
                for key in ("history_action_input_ids", "history_event_input_ids"):
                    slots = [
                        pad_token_rows(
                            [f[key][i] if i < len(f[key]) else [pad_id] for f in features], pad_id
                        )
                        for i in range(depth)
                    ]
                    width = max(t.shape[1] for t in slots)
                    stacked = torch.stack(
                        [torch.nn.functional.pad(t, (0, width - t.shape[1]), value=pad_id) for t in slots],
                        dim=1,
                    )
                    batch[key] = stacked
                    batch[key.replace("input_ids", "attention_mask")] = (stacked != pad_id).to(torch.long)
                batch["history_step_mask"] = torch.tensor(
                    [[1.0 if i < len(f["history_action_input_ids"]) else 0.0 for i in range(depth)]
                     for f in features], dtype=torch.float)
        terminal_labels = torch.tensor([float(f.get("terminal_label", -1)) for f in features], dtype=torch.float)
        batch["terminal_labels"] = terminal_labels.clamp_min(0.0)
        batch["terminal_label_mask"] = (terminal_labels >= 0).to(torch.float)
        # The dataset emits label_* only for the fields this run trains
        # (--canonical-event-heads); collate whatever is present.
        for field in CANONICAL_EVENT_SINGLE_LABEL_FIELDS:
            key = f"label_{field}"
            if key in features[0]:
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
        for field in NUDGE_MULTI_LABEL_FIELDS:
            key = f"label_{field}"
            if key in features[0]:
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.float)
        # Recognition-probe observation: emitted only when EVERY example in the batch carries
        # it, so the head reads z_observation uniformly (no per-row mixing with z_pred).
        if all("canonical_event_observation_input_ids" in f for f in features):
            batch["canonical_event_observation_input_ids"] = pad_token_rows(
                [f["canonical_event_observation_input_ids"] for f in features], pad_id
            )
            batch["canonical_event_observation_attention_mask"] = pad_token_rows(
                [f["canonical_event_observation_attention_mask"] for f in features], 0
            )
        batch["value_target"] = torch.tensor([f["value_target"] for f in features], dtype=torch.float)
        batch["value_target_mask"] = torch.tensor([float(f["has_value_target"]) for f in features], dtype=torch.float)
        return batch


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """DiT-style AdaLN modulation. scale/shift are zero at init, so this starts as identity."""
    return x * (1.0 + scale) + shift


class AdaLNPredictorBlock(nn.Module):
    """One causal-attention + MLP block with AdaLN-zero conditioning.

    Mirrors ConditionalBlock in the reference implementation
    (github.com/lucas-maes/le-wm, module.py): the conditioning vector produces six modulation
    tensors (shift/scale/gate for the attention sub-layer and for the MLP sub-layer) through a
    SiLU + Linear whose weight AND bias are ZERO-initialized. At step 0 that makes scale=shift=0
    (modulation is the identity) and gate=0 (both residual branches contribute nothing), so the
    block is an exact identity and action conditioning ramps in progressively as training moves
    the modulation weights off zero.
    """

    def __init__(self, dim: int, heads: int, dropout: float, mlp_ratio: float) -> None:
        super().__init__()
        # elementwise_affine=False: AdaLN supplies the scale/shift, so a learned per-channel
        # affine here would fight it (and break the exact-identity-at-init property).
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = max(dim, int(dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.constant_(self.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.adaln_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_modulation(cond).chunk(6, dim=-1)
        )
        normed = _modulate(self.norm1(x), shift_attn, scale_attn)
        attended, _ = self.attention(normed, normed, normed, attn_mask=attention_mask, need_weights=False)
        x = x + gate_attn * attended
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class AdaLNTransformerPredictor(nn.Module):
    """LeWorldModel-style autoregressive predictor (--predictor-arch transformer).

    Follows ARPredictor / ConditionalBlock / Transformer in the reference implementation
    (github.com/lucas-maes/le-wm, module.py): learned positional embeddings added at input
    width, a stack of AdaLN-zero conditional blocks with causal attention, a plain final
    LayerNorm, and an output projector.

    Sequence layout -- N+1 representation tokens, no action tokens:

        tokens : [ E(sys + task prompt) , e_{t-N+1} , ... , e_{t-1} ,      e_t          ]
        AdaLN  : [ NULL                 , u_{t-N+2} , ... , u_t      , candidate u_{t+1} ]

    The action is never a token; it enters only as AdaLN conditioning. Conditioning is
    POSITION-ALIGNED: the token holding event e_i is modulated by the action taken FROM e_i --
    i.e. the one that produced e_{i+1} -- so the pair (e_i, a_i) -> e_{i+1} is preserved at every
    position. The final position carries the candidate action being evaluated. Without this
    alignment a tool output like "Operation completed successfully." is unattributable: the model
    cannot tell whether it came from update_ticket, create_calendar_event or send_message.

    The context token gets a learned NULL conditioning (it is not a tool output, so no action was
    taken "from" it), except when it is also the last position -- a first step with no history --
    where it takes the candidate action, otherwise the action would be invisible.

    Only the LAST position's output is read, so the causal mask serves to keep every position
    honest during training rather than to emit T parallel predictions.

    Two outputs: the hidden state h_t at the last position IS the belief state (the causal
    transformer is the state updater -- there is no separate recurrent U), and the event
    representation is its projection, e_hat_{t+1} = W_pred h_t. Classification heads read h_t.

    Padding: examples carry different history depths. Shorter histories are RIGHT-aligned (pad
    between the context token and the oldest real frame) so the position embedding always
    encodes recency identically and the newest frame is always last. Padded slots are masked out
    of attention, except that every position may always attend to itself -- a fully masked row
    in the pad region would produce NaN through the softmax and poison the whole batch.
    """

    def __init__(
        self,
        latent_dim: int,
        dim: int,
        layers: int,
        heads: int,
        dropout: float,
        cond_inputs: int,
        max_positions: int,
        mlp_ratio: float,
        output_layernorm: bool,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(
                f"--predictor-transformer-dim={dim} must be divisible by "
                f"--predictor-transformer-heads={heads}."
            )
        self.dim = int(dim)
        self.heads = int(heads)
        self.max_positions = int(max_positions)
        # Positional embedding added at INPUT width, before the projection into the transformer
        # (ARPredictor.forward: `x = x + self.pos_embedding[:, :T]`, then Transformer.input_proj).
        self.position_embedding = nn.Parameter(torch.randn(1, self.max_positions, latent_dim) * 0.02)
        self.embedding_dropout = nn.Dropout(dropout)
        self.input_projection = nn.Linear(latent_dim, dim)
        # Conditioning: the action alone (plus the goal latent when the run is goal-conditioned,
        # which has no analogue in the reference implementation). One vector PER POSITION.
        self.cond_dim = latent_dim * cond_inputs
        self.cond_projection = nn.Linear(self.cond_dim, dim)
        # Learned stand-in for "no action was taken from this token", used at the context
        # position. Zeros would also work (AdaLN-zero maps them to the identity at init) but a
        # learned vector lets the model give the context position its own modulation.
        self.null_conditioning = nn.Parameter(torch.zeros(1, 1, self.cond_dim))
        self.blocks = nn.ModuleList(
            AdaLNPredictorBlock(dim, heads, dropout, mlp_ratio) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        # "followed by a projector network with the same implementation as the one used for the
        # encoder" -- encoder_projector is Linear + LayerNorm. The LayerNorm is dropped for
        # categorical latents (the output is prior logits) and in delta mode (it would pin the
        # output norm and make the small deltas of near-identity transitions unrepresentable),
        # matching how the MLP predictor's trailing LayerNorm is gated.
        self.projector = (
            nn.Sequential(nn.Linear(dim, latent_dim), nn.LayerNorm(latent_dim))
            if output_layernorm
            else nn.Linear(dim, latent_dim)
        )

    def _attention_mask(self, valid: torch.Tensor) -> torch.Tensor:
        """[B*heads, T, T] bool mask, True = BLOCKED. Causal, plus padded keys, minus the
        diagonal (always allowed) so no row is fully blocked."""
        length = valid.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=valid.device).tril()
        allowed = causal.unsqueeze(0) & valid.unsqueeze(1)  # [B, T, T]
        allowed = allowed | torch.eye(length, dtype=torch.bool, device=valid.device).unsqueeze(0)
        return (~allowed).repeat_interleave(self.heads, dim=0)

    def forward(
        self,
        tokens: torch.Tensor,
        cond: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens: [B, T, latent_dim] = [context, events...]; cond: [B, T, cond_dim] with one
        action per position; valid: [B, T] bool.

        Returns (event, state): `state` is the last position's hidden state h_t [B, dim] -- the
        belief state -- and `event` is its projection e_hat_{t+1} [B, latent_dim].
        """
        length = tokens.shape[1]
        if length > self.max_positions:
            # Should not happen (the caller trims the frame window), but a longer sequence would
            # index past the position embedding -- keep the newest positions.
            tokens = tokens[:, -self.max_positions:]
            cond, valid = cond[:, -self.max_positions:], valid[:, -self.max_positions:]
            length = self.max_positions
        x = self.input_projection(self.embedding_dropout(tokens + self.position_embedding[:, :length]))
        cond = self.cond_projection(cond)
        attention_mask = self._attention_mask(valid)
        for block in self.blocks:
            x = block(x, cond, attention_mask)
        state = self.final_norm(x[:, -1])
        return self.projector(state), state


class TextLeWorldModel(nn.Module):
    def __init__(
        self,
        backbone: Any,
        latent_dim: int,
        memory_tokens: int,
        dropout: float,
        predictor_hidden_multiplier: float,
        goal_conditioning: bool = True,
        latent_type: str = "continuous",
        latent_categoricals: int = 32,
        latent_classes: int = 32,
        latent_unimix: float = 0.01,
        latent_delta_prediction: bool = False,
        pooling: str = "mean",
        canonical_event_vocab_sizes: dict[str, int] | None = None,
        canonical_event_head_hidden_size: int = 512,
        obs_grounding: bool = False,
        obs_ground_decoder_dim: int = 256,
        obs_ground_decoder_layers: int = 4,
        obs_ground_decoder_heads: int = 4,
        obs_ground_decoder_memory_tokens: int = 8,
        obs_ground_decoder_max_length: int = 128,
        tool_vocab_size: int = 0,
        tool_select: bool = False,
        action_encoder: bool = False,
        action_head_embed_dim: int = 256,
        action_decoder: bool = False,
        action_decoder_max_noise_std: float = 0.1,
        action_decoder_dim: int = 256,
        action_decoder_layers: int = 4,
        action_decoder_heads: int = 4,
        action_decoder_memory_tokens: int = 8,
        action_decoder_max_length: int = 512,
        action_decoder_tool_vocab_size: int = 0,
        fast_lewm: bool = False,
        fast_lewm_dim: int = 256,
        fast_lewm_layers: int = 3,
        fast_lewm_heads: int = 4,
        fast_lewm_max_horizon: int = 8,
        terminal_head: bool = False,
        value_head: bool = False,
        recognition_bypass_projector: bool = False,
        canonical_event_head_inputs: str = "all",
        state_updater: bool = False,
        state_updater_objective: str = "future_event",
        recurrent_state_init: bool = False,
        predictor_arch: str = "mlp",
        predictor_transformer_dim: int = 0,
        predictor_transformer_layers: int = 6,
        predictor_transformer_heads: int = 16,
        predictor_transformer_mlp_ratio: float = 4.0,
        predictor_history_length: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = resolve_backbone_hidden_size(backbone)
        self.pooling = str(pooling)
        self.latent_type = str(latent_type)
        self.latent_categoricals = int(latent_categoricals)
        self.latent_classes = int(latent_classes)
        self.latent_unimix = float(latent_unimix)
        if self.latent_type == "categorical":
            # The latent vector is the flattened straight-through one-hot stack.
            self.latent_dim = self.latent_categoricals * self.latent_classes
        else:
            self.latent_dim = int(latent_dim or self.hidden_size)
        self.memory_tokens = int(memory_tokens)
        self.goal_conditioning = bool(goal_conditioning)
        if self.latent_type == "categorical":
            # Project pooled features to per-group class logits (no LayerNorm on logits).
            self.encoder_projector = nn.Linear(self.hidden_size, self.latent_dim)
        else:
            self.encoder_projector = nn.Sequential(
                nn.Linear(self.hidden_size, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
            )
        # Delta encoding / Δz supervision: the predictor emits the CHANGE Δz and
        # predict_latent returns z_current + Δz. Continuous latents only -- a delta between
        # straight-through one-hot stacks is not a point on the categorical simplex.
        self.latent_delta_prediction = bool(latent_delta_prediction) and self.latent_type != "categorical"
        predictor_hidden = max(self.latent_dim, int(self.latent_dim * predictor_hidden_multiplier))
        predictor_inputs = 4 if self.goal_conditioning else 3
        predictor_layers: list[nn.Module] = [
            nn.Linear(self.latent_dim * predictor_inputs, predictor_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, predictor_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, self.latent_dim),
        ]
        output_layernorm = self.latent_type != "categorical" and not self.latent_delta_prediction
        if output_layernorm:
            # Categorical predictor emits prior logits; continuous predictor emits a normalized
            # vector. In delta mode the output must NOT be LayerNormed: LayerNorm pins the
            # output norm to ~sqrt(latent_dim), which would make small deltas (near-identity
            # transitions -- the common case for consecutive tool-use states) unrepresentable.
            predictor_layers.append(nn.LayerNorm(self.latent_dim))
        # --predictor-arch: `mlp` keeps the concat-MLP above; `transformer` swaps in the
        # LeWorldModel-style causal transformer with AdaLN action conditioning. Only one is
        # built, so a checkpoint's state dict is unambiguous about which it was -- and the
        # manifest records the choice (see jepa_architecture_manifest_fields).
        self.predictor_arch = str(predictor_arch or "mlp")
        if self.predictor_arch not in ("mlp", "transformer"):
            raise ValueError(f"unknown predictor_arch={self.predictor_arch!r}")
        # N = tool-output representations the predictor attends over, INCLUDING the current one.
        # The logged history supplies up to WORLD_MODEL_INPUT_HISTORY_SIZE past steps, and the
        # current frame is one more.
        self.predictor_history_length = int(predictor_history_length or WORLD_MODEL_INPUT_HISTORY_SIZE + 1)
        if self.predictor_arch == "transformer":
            self.predictor = AdaLNTransformerPredictor(
                latent_dim=self.latent_dim,
                dim=int(predictor_transformer_dim or self.latent_dim),
                layers=int(predictor_transformer_layers),
                heads=int(predictor_transformer_heads),
                dropout=dropout,
                # AdaLN conditioning is the action alone (+ the goal latent when the run is
                # goal-conditioned). The context is a TOKEN, not part of the conditioning.
                cond_inputs=2 if self.goal_conditioning else 1,
                # N frames + the context token at position 0.
                max_positions=self.predictor_history_length + 1,
                mlp_ratio=float(predictor_transformer_mlp_ratio),
                output_layernorm=output_layernorm,
            )
        else:
            self.predictor = nn.Sequential(*predictor_layers)
        self.success_head = nn.Sequential(
            nn.Linear(self.latent_dim * 4, predictor_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, 1),
        )
        # Optional terminal-step head: is THIS step the last one in its trajectory (episode
        # 'done'), independent of whether the trajectory succeeded. Same [z_current, z_action,
        # z_context, z_pred] input/architecture as success_head, but a properly optional module
        # (like tool_select/action_encoder/action_decoder) rather than always-allocated, since
        # most training runs won't want it.
        self.terminal_head_enabled = bool(terminal_head)
        if self.terminal_head_enabled:
            self.terminal_head = nn.Sequential(
                nn.Linear(self.latent_dim * 4, predictor_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(predictor_hidden, 1),
            )
        # Optional value head: regresses the per-step discounted return-to-go
        # (`value_target`, see src/data_preparation/annotate_step_value_scores.py) from the
        # same [z_current, z_action, z_context, z_pred] features -- reward prediction rather
        # than classification. Same architecture/input as success_head/terminal_head; the only
        # difference is the loss (Smooth L1 regression, not BCE) and that its raw output IS the
        # prediction (no squeeze-then-sigmoid at call sites).
        self.value_head_enabled = bool(value_head)
        if self.value_head_enabled:
            self.value_head = nn.Sequential(
                nn.Linear(self.latent_dim * 4, predictor_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(predictor_hidden, 1),
            )
        self.memory_projection = nn.Sequential(
            nn.Linear(self.latent_dim, self.memory_tokens * self.hidden_size),
            nn.LayerNorm(self.memory_tokens * self.hidden_size),
        )
        # A single shared trunk projects the same [z_current, z_action, z_context,
        # z_pred] concat as success_head down to a small bottleneck; each
        # canonical_event_state/nudge field then gets only a tiny linear readout
        # off that shared trunk -- see --train-canonical-event-heads-only. Giving
        # every one of the ~11 fields its own full predictor-sized hidden layer
        # (the success_head pattern) would multiply parameter/optimizer memory by
        # ~11x for no benefit, since each head only classifies a handful of classes.
        canonical_event_vocab_sizes = canonical_event_vocab_sizes or {}
        self.canonical_event_head_hidden_size = int(canonical_event_head_hidden_size)
        # Recognition-probe bypass control: the canonical heads read RAW pooled backbone
        # features (hidden_size each) instead of projected latents (latent_dim each), so no
        # JEPA-trained component sits in the readout path. Diagnostic mode -- the resulting
        # checkpoint's trunk shape only matches other bypass-constructed models (a mismatch
        # fails loudly at load, never silently).
        # --event-state-decomposition: the predictor's output is the EVENT e_hat (what this
        # action newly caused); the next STATE is then composed by U(z_t, e_hat). The action is
        # deliberately NOT an input to U -- it is already reflected in e_hat, and routing the
        # state exclusively through the event is what forbids U from rediscovering
        # z_hat_{t+1} = z_t. A dual-head F -> (e_hat, z_hat) would not have that guarantee.
        self.state_updater_enabled = bool(state_updater)
        # U(s_t, a_{t+1}, e_{t+1}) -- the action is an input again, which is safe here ONLY
        # because U has no direct next-state target to game: it is trained solely through
        # L_future (can s_{t+1} still predict the NEXT event?). With a cumulative-next-state
        # MSE target the persistence shortcut simply relocates from F into U.
        self.state_updater_objective = str(state_updater_objective or "future_event")
        # s_0 = I(c). The system prompt + task prompt are constant within a trajectory, so they
        # are encoded once into z_context and turned into the initial state here; every
        # subsequent step's history arrives through U, not by re-encoding the prompt and the
        # whole history as text on every example.
        self.recurrent_state_init = bool(recurrent_state_init)
        if self.recurrent_state_init and not self.state_updater_enabled:
            # s_i = U(s_{i-1}, a_i, E(o_i)) -- the unroll IS the state updater, so without U
            # recurrent_current_state would fail on the first step. The CLI already enforces
            # this (--recurrent-state-init requires --event-state-recurrent); this catches
            # direct construction, which is otherwise an AttributeError deep in the forward.
            raise ValueError("recurrent_state_init=True requires state_updater=True.")
        if self.recurrent_state_init:
            self.state_init = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
            )
        if self.state_updater_enabled:
            hidden = max(self.latent_dim, int(self.latent_dim * predictor_hidden_multiplier))
            self.state_updater = nn.Sequential(
                nn.Linear(self.latent_dim * 3, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
            )
        self.recognition_bypass_projector = bool(recognition_bypass_projector)
        # Which latents the canonical-event readout may see. "all" is the historical behaviour
        # ([z_cur, z_act, z_ctx, z_pred]) and carries a skip connection around the predicted
        # future: measured, [z_cur,z_act,z_ctx] alone scores 0.7764 vs 0.7768 with z_pred added
        # (McNemar p=0.945), and adding the TRUE next latent instead scores 0.7801 (p=0.762).
        # The head simply does not need a future. "ctx_pred" and "pred_only" remove that path so
        # whatever the readout scores is information the predicted future actually carries.
        # "state" reads the transformer predictor's belief state h_t directly. The event
        # representation IS the projection of h_t, so h_t strictly contains it -- and it also
        # carries the accumulated history the projection discards. Its width is the predictor's,
        # not latent_dim.
        self.canonical_event_head_inputs = str(canonical_event_head_inputs or "all")
        _head_input_counts = {"all": 4, "ctx_pred": 2, "pred_only": 1, "state": 1, "state_action": 1}
        if self.canonical_event_head_inputs not in _head_input_counts:
            raise ValueError(f"unknown canonical_event_head_inputs={self.canonical_event_head_inputs!r}")
        if self.canonical_event_head_inputs in {"state", "state_action"}:
            mode_name = self.canonical_event_head_inputs
            if self.predictor_arch != "transformer":
                raise ValueError(
                    f"canonical_event_head_inputs={mode_name!r} requires --predictor-arch "
                    "transformer: the MLP predictor has no hidden state distinct from its output "
                    "(use 'pred_only' for the equivalent readout there)."
                )
            if self.recognition_bypass_projector:
                raise ValueError(
                    f"canonical_event_head_inputs={mode_name!r} is incompatible with "
                    "--recognition-probe-bypass-projector (the bypass reads raw pooled features, "
                    "the state is produced by the trained predictor)."
                )
            # h_t is the predictor's width, NOT latent_dim; 'state_action' appends the projected
            # action latent, so the two widths add rather than multiply.
            canonical_trunk_input_dim = self.predictor.dim
            if self.canonical_event_head_inputs == "state_action":
                canonical_trunk_input_dim += self.latent_dim
        else:
            canonical_trunk_input_dim = (
                (self.hidden_size if self.recognition_bypass_projector else self.latent_dim)
                * _head_input_counts[self.canonical_event_head_inputs]
            )
        if canonical_event_vocab_sizes:
            self.canonical_event_trunk = nn.Sequential(
                nn.Linear(canonical_trunk_input_dim, self.canonical_event_head_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.canonical_event_trunk = nn.Identity()
        self.canonical_event_heads = nn.ModuleDict(
            {
                field: nn.Linear(self.canonical_event_head_hidden_size, num_classes)
                for field, num_classes in canonical_event_vocab_sizes.items()
            }
        )
        self.supports_reconstruction = backbone_supports_reconstruction(backbone)
        # Option-C observation-token grounding decodes the first-N tokens of the isolated
        # environment output from z_pred.
        #
        # Encoder-decoder backbones (T5Gemma, ...) reuse their NATIVE decoder -- a full
        # pretrained Transformer decoder -- via its OWN dedicated memory projection
        # (obs_ground_memory_projection), a small MLP rather than a single Linear for more
        # expressive capacity when expanding one z_pred vector into memory_tokens slots. This
        # is intentionally NOT the same module as the base reconstruction_loss's
        # memory_projection (z_pred -> full next-state text): sharing weights across two
        # different decode objectives would conflate their gradients, the same reasoning that
        # gave the action decoder its own dedicated weights instead of reusing memory_projection.
        #
        # Encoder-only backbones (Qwen3-Embedding, ...) have no such decoder, so this builds a
        # multi-layer CAUSAL TRANSFORMER decoder (not a single-layer GRU) cross-attending to
        # obs_ground_decoder_memory_tokens memory slots expanded from z_pred -- the same
        # architectural upgrade as the action decoder, applied here to observation-grounding.
        self.obs_grounding = bool(obs_grounding)
        if self.obs_grounding:
            if self.supports_reconstruction:
                mem_dim = self.memory_tokens * self.hidden_size
                self.obs_ground_memory_projection = nn.Sequential(
                    nn.Linear(self.latent_dim, mem_dim), nn.GELU(),
                    nn.Linear(mem_dim, mem_dim),
                    nn.LayerNorm(mem_dim),
                )
            else:
                og_dim = int(obs_ground_decoder_dim)
                self.obs_ground_decoder_dim = og_dim
                self.obs_ground_decoder_memory_tokens = int(obs_ground_decoder_memory_tokens)
                self.obs_ground_memory_expand = nn.Sequential(
                    nn.Linear(self.latent_dim, self.obs_ground_decoder_memory_tokens * og_dim), nn.GELU(),
                    nn.Linear(self.obs_ground_decoder_memory_tokens * og_dim, self.obs_ground_decoder_memory_tokens * og_dim),
                )
                self.obs_ground_start = nn.Parameter(torch.zeros(self.hidden_size))
                self.obs_ground_token_in = nn.Linear(self.hidden_size, og_dim)
                self.obs_ground_pos = nn.Embedding(int(obs_ground_decoder_max_length), og_dim)
                og_decoder_layer = nn.TransformerDecoderLayer(
                    d_model=og_dim, nhead=int(obs_ground_decoder_heads), dim_feedforward=og_dim * 4,
                    dropout=dropout, batch_first=True, activation="gelu",
                )
                self.obs_ground_transformer = nn.TransformerDecoder(og_decoder_layer, num_layers=int(obs_ground_decoder_layers))
                self.obs_ground_out = nn.Linear(og_dim, self.hidden_size)
        # Optional latent action-head. P1: retrieval-style tool selection from the state,
        # scoring learned tool embeddings. P2: a fast structured-action encoder distilled to
        # the backbone-encoded z_action, so search can encode candidate actions without the
        # backbone (tool embedding + mean-pooled frozen arg-token embeddings -> latent_dim).
        self.tool_select = bool(tool_select) and tool_vocab_size > 0
        self.action_encoder = bool(action_encoder) and tool_vocab_size > 0
        self.action_head_embed_dim = int(action_head_embed_dim)
        if self.tool_select or self.action_encoder:
            self.tool_embeddings = nn.Embedding(tool_vocab_size, self.action_head_embed_dim)
        if self.tool_select:
            query_in = self.latent_dim * (2 if self.goal_conditioning else 1)
            self.tool_query = nn.Sequential(
                nn.Linear(query_in, self.action_head_embed_dim), nn.GELU(),
                nn.Linear(self.action_head_embed_dim, self.action_head_embed_dim),
            )
        if self.action_encoder:
            self.action_encoder_mlp = nn.Sequential(
                nn.Linear(self.action_head_embed_dim + self.hidden_size, self.latent_dim), nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
        # Learnable action DECODER D: the missing half of the action_encoder (P2) above --
        # reconstructs an action's own tokens from its (randomly noised) z_action. A single
        # combined loss plays both roles the hierarchical-latent-action design calls for:
        # at noise_scale~0 it is reconstruction/cycle-consistency (decode(encode(a)) ~= a);
        # at noise_scale>0 it is "prior-sample" robustness training against exactly the kind
        # of off-manifold point a Gaussian CEM proposal samples near a real anchor. Without
        # this, latent interpolation has no guarantee of landing anywhere decodable -- the
        # hierarchical CEM planner (src/hierarchical_action_sampling.py) can only ever fall
        # back to nearest-anchor lookup.
        #
        # Encoder-decoder backbones (T5Gemma, ...) reuse their NATIVE decoder -- already a
        # full pretrained Transformer decoder, plenty powerful on its own -- via its OWN
        # memory_projection (never shared with obs-grounding/reconstruction's, since z_action
        # and z_pred are different latents and sharing weights would conflate their decode
        # objectives). The projection itself is a small MLP (not a single Linear) for more
        # expressive capacity when expanding one z_action vector into memory_tokens slots.
        #
        # Encoder-only backbones (Qwen3-Embedding, ...) have no such decoder, so this builds
        # its own: z_action is expanded (via an MLP) into action_decoder_memory_tokens memory
        # slots, cross-attended by a multi-layer CAUSAL TRANSFORMER decoder (not a single-layer
        # GRU) -- attention over multiple memory positions instead of one collapsed recurrent
        # hidden state, the same architectural upgrade Fast-LeWM already made for latent
        # prediction (see fast_encoder below), applied here to text decoding.
        self.action_decoder = bool(action_decoder)
        self.action_decoder_max_noise_std = float(action_decoder_max_noise_std)
        self.action_decoder_max_length = int(action_decoder_max_length)
        # Calibration scalar (see calibrate_action_decoder_latent_scale): the empirical std of
        # z_action over real training actions. A checkpoint BUFFER (not a plain attribute) so it
        # is part of state_dict and travels with the weights to replay time -- the inference-time
        # CEM reads it to calibrate its per-family Gaussian init/min std against the exact same
        # number --action-decoder-max-noise-std was a multiple of during training, instead of
        # two independently hand-picked, easily-mismatched constants. Defaults to 1.0 (a no-op
        # multiplier) until calibrate_action_decoder_latent_scale sets it.
        self.register_buffer("action_decoder_latent_scale", torch.tensor(1.0), persistent=True)
        self.action_decoder_tool_vocab_size = int(action_decoder_tool_vocab_size)
        if self.action_decoder:
            if self.supports_reconstruction:
                mem_dim = self.memory_tokens * self.hidden_size
                self.action_decoder_memory_projection = nn.Sequential(
                    nn.Linear(self.latent_dim, mem_dim), nn.GELU(),
                    nn.Linear(mem_dim, mem_dim),
                    nn.LayerNorm(mem_dim),
                )
            else:
                ad_dim = int(action_decoder_dim)
                self.action_decoder_dim = ad_dim
                self.action_decoder_memory_tokens = int(action_decoder_memory_tokens)
                self.action_decoder_memory_expand = nn.Sequential(
                    nn.Linear(self.latent_dim, self.action_decoder_memory_tokens * ad_dim), nn.GELU(),
                    nn.Linear(self.action_decoder_memory_tokens * ad_dim, self.action_decoder_memory_tokens * ad_dim),
                )
                # Family/tool conditioning: an extra memory slot the decoder cross-attends to,
                # alongside the action_decoder_memory_tokens slots expanded from z_action. Index
                # 0 is reserved for "unknown tool" (e.g. a family string from a non-MCP backend
                # -- SQL/shell verb families -- that has no entry in this MCP tool vocabulary) --
                # same convention as build_tool_vocabulary's "<unk>": 0, so no +1 here (mirrors
                # the sibling tool_embeddings table below, which is also sized tool_vocab_size).
                # Without this the decoder has to infer the family from a possibly-noisy/
                # off-manifold z_action alone; with it, the CEM's already-known family (it
                # sampled u FROM that family's Gaussian) is handed to the decoder directly.
                if self.action_decoder_tool_vocab_size > 0:
                    self.action_decoder_tool_embeddings = nn.Embedding(self.action_decoder_tool_vocab_size, ad_dim)
                self.action_decoder_start = nn.Parameter(torch.zeros(self.hidden_size))
                self.action_decoder_token_in = nn.Linear(self.hidden_size, ad_dim)
                self.action_decoder_pos = nn.Embedding(self.action_decoder_max_length, ad_dim)
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=ad_dim, nhead=int(action_decoder_heads), dim_feedforward=ad_dim * 4,
                    dropout=dropout, batch_first=True, activation="gelu",
                )
                self.action_decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=int(action_decoder_layers))
                self.action_decoder_out = nn.Linear(ad_dim, self.hidden_size)
        # Fast-LeWM (arXiv:2606.26217): action-prefix encoder (causal Transformer over a
        # [state_token, a_0..a_{H-1}] sequence) + parallel predictor that maps the anchor
        # latent and each prefix token to the future latent for that horizon, all at once.
        self.fast_lewm = bool(fast_lewm)
        if self.fast_lewm:
            dim = int(fast_lewm_dim)
            self.fast_state_proj = nn.Linear(self.latent_dim * 2, dim)  # [z_current, z_context] -> state token
            self.fast_action_proj = nn.Linear(self.latent_dim, dim)
            self.fast_pos = nn.Embedding(int(fast_lewm_max_horizon) + 2, dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=int(fast_lewm_heads), dim_feedforward=dim * 4,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.fast_encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(fast_lewm_layers))
            predictor_in = self.latent_dim * 2 + dim + (self.latent_dim if self.goal_conditioning else 0)
            self.fast_predictor = nn.Sequential(
                nn.Linear(predictor_in, dim * 4), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dim * 4, self.latent_dim),
            )
        self.deterministic_latent_sampling = False

    def mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def last_token_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Take the last non-padding token's hidden state (Qwen3-Embedding/E5 style).

        Works for both left- and right-padded batches: if every row has a real
        token in the final column the batch is left-padded (take column -1),
        otherwise gather each row's last attended position.
        """
        left_padded = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padded:
            return hidden[:, -1]
        last_index = attention_mask.to(torch.long).sum(dim=1) - 1
        last_index = last_index.clamp_min(0)
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_index, last_index]

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if getattr(self, "pooling", "mean") == "last_token":
            return self.last_token_pool(hidden, attention_mask)
        return self.mean_pool(hidden, attention_mask)

    def backbone_requires_grad(self) -> bool:
        return any(param.requires_grad for param in self.backbone.parameters())

    def _encode_pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder = backbone_encoder(self.backbone)
        with torch.set_grad_enabled(self.backbone_requires_grad()):
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden = outputs[0] if isinstance(outputs, (tuple, list)) and outputs else None
            if hidden is None:
                raise ValueError("Encoder backbone did not return last_hidden_state.")
            pooled = self.pool(hidden, attention_mask)
        projector_dtype = next(self.encoder_projector.parameters()).dtype
        return pooled.to(dtype=projector_dtype)

    def _predictor_conditioning(
        self, z_action: torch.Tensor, z_context: torch.Tensor, z_goal: torch.Tensor | None
    ) -> torch.Tensor:
        """What the predictor is conditioned on besides the frame itself: the action, the task
        context, and the goal when goal-conditioned. Identical for both architectures -- the MLP
        concatenates it with the frame, the transformer feeds it to AdaLN."""
        parts = [z_action, z_context]
        if self.goal_conditioning:
            parts.append(z_goal if z_goal is not None else torch.zeros_like(z_action))
        return torch.cat(parts, dim=-1)

    def _frame_conditioning(self, z_action: torch.Tensor, z_goal: torch.Tensor | None) -> torch.Tensor:
        """The per-position AdaLN vector for one action, matching encode_frame_history's width
        (action, plus the goal latent when goal-conditioned)."""
        if not self.goal_conditioning:
            return z_action
        goal = z_goal if z_goal is not None else torch.zeros_like(z_action)
        return torch.cat([z_action, goal], dim=-1)

    def _extend_frame_history(
        self,
        frame_history: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        frame: torch.Tensor,
        producing_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Push one event representation and the action that PRODUCED it onto the sequence, for
        auto-regressive rollout: X_{t+1} = slide(X_t, e_hat_{t+1}).

        Storing the producing action (not the next one) is what keeps the conditioning shift in
        _predict_latent_transformer correct: position k is modulated by actions[k+1], and the
        newest position by the candidate action. No-op under the MLP predictor.
        """
        if self.predictor_arch != "transformer":
            return None
        frame = frame.unsqueeze(1)
        producing_action = producing_action.unsqueeze(1)
        ones = torch.ones(frame.shape[0], 1, dtype=torch.bool, device=frame.device)
        if frame_history is None:
            return frame, producing_action, ones
        frames, actions, valid = frame_history
        return (
            torch.cat([frames, frame], dim=1),
            torch.cat([actions, producing_action], dim=1),
            torch.cat([valid, ones], dim=1),
        )

    def _expand_frame_history(
        self,
        frame_history: tuple[torch.Tensor, torch.Tensor] | None,
        repeats: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Tile the history to match a [B*repeats] flattened batch (the action-contrastive
        negatives). Without this the negatives would see no history while the positive does, so
        the contrast would confound 'different action' with 'different context'."""
        if frame_history is None:
            return None
        return tuple(tensor.repeat_interleave(repeats, dim=0) for tensor in frame_history)

    def _predict_latent_transformer(
        self,
        z_current: torch.Tensor,
        z_action: torch.Tensor,
        z_context: torch.Tensor,
        z_goal: torch.Tensor | None,
        frame_history: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble [E(context), e_{t-N+1} ... e_t] with position-aligned action conditioning.

        Token 0 is the system-prompt + task-prompt representation and is always valid; the rest
        are the N most recent event (tool-output) representations, newest last. The action never
        becomes a token.

        Conditioning is shifted by one: `frame_history` stores, per frame, the action that
        PRODUCED it, so the action taken FROM frame k is the one stored at k+1. The newest frame
        is conditioned on the candidate action `z_action` being evaluated. The context token gets
        the learned NULL vector, unless it is also the last position (no history at all) -- there
        it takes the candidate action, or the action would never reach the readout.

        `z_current` is used as the sole event token ONLY when no history is available (external
        callers: beam_plan, hier_latent_cem, the analysis scripts). With history present the
        newest logged tool output already is e_t, so the cumulative-state encoding is not a token.

        Returns (event, state).
        """
        batch_size = z_current.shape[0]
        ones = torch.ones(batch_size, 1, dtype=torch.bool, device=z_current.device)
        candidate = [z_action]
        if self.goal_conditioning:
            candidate.append(z_goal if z_goal is not None else torch.zeros_like(z_action))
        candidate = torch.cat(candidate, dim=-1).unsqueeze(1)  # [B, 1, cond_dim]
        if frame_history is None:
            frames, frames_valid = z_current.unsqueeze(1), ones
            frame_cond = candidate
        else:
            frames, producing_actions, frames_valid = frame_history
            frames_valid = frames_valid.to(torch.bool)
            # Shift: position k is modulated by the action stored at k+1 (= taken from frame k);
            # the newest position by the candidate action.
            frame_cond = torch.cat([producing_actions[:, 1:], candidate], dim=1)
        # Trim the FRAME window (not the assembled sequence): the context token is position 0 and
        # must survive, so it is prepended after trimming.
        window = self.predictor.max_positions - 1
        if frames.shape[1] > window:
            frames = frames[:, -window:]
            frame_cond, frames_valid = frame_cond[:, -window:], frames_valid[:, -window:]
        tokens = torch.cat([z_context.unsqueeze(1), frames], dim=1)
        valid = torch.cat([ones, frames_valid], dim=1)
        context_cond = self.predictor.null_conditioning.expand(batch_size, 1, -1)
        if frames.shape[1] == 0:
            context_cond = candidate  # context IS the readout position
        cond = torch.cat([context_cond, frame_cond], dim=1)
        return self.predictor(tokens, cond, valid)

    def encode_frame_history(
        self,
        batch: dict[str, torch.Tensor],
        history_latents: tuple[torch.Tensor | None, torch.Tensor] | None = None,
        z_goal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Encode the logged history into (frames, producing_actions, valid) for the transformer.

        `frames[k]` is the tool-output representation e_k and `producing_actions[k]` is the action
        that produced it -- the caller shifts by one to get "the action taken FROM e_k". Both
        RIGHT-aligned: an example with 2 logged steps out of a batch depth of 5 gets its two
        frames in the last two slots, so the position embedding always means "how many steps back
        from now". Left-aligned padding (how the collator emits it) would make a given position
        mean different things for different examples.

        Returns None when the batch carries no history at all, in which case the sequence is just
        [context, z_current].
        """
        if "history_event_input_ids" not in batch:
            return None
        depth = batch["history_event_input_ids"].shape[1]
        if depth == 0:
            return None
        z_a, z_o = (
            history_latents if history_latents is not None and history_latents[0] is not None
            else self._encode_history_latents(batch)
        )
        valid = batch["history_step_mask"].to(torch.bool)
        # Right-align: slot j of the output reads source slot j - (depth - n_i), which is
        # out of range exactly where the slot should be padding.
        counts = valid.sum(dim=1, keepdim=True)  # [B, 1]
        target = torch.arange(depth, device=z_o.device).unsqueeze(0)  # [1, depth]
        source = target - (depth - counts)  # [B, depth]
        gather_valid = source >= 0
        index = source.clamp_min(0).unsqueeze(-1).expand(-1, -1, z_o.shape[-1])
        keep = gather_valid.unsqueeze(-1)
        frames = torch.gather(z_o, 1, index) * keep.to(z_o.dtype)
        actions = torch.gather(z_a, 1, index) * keep.to(z_a.dtype)
        if self.goal_conditioning:
            goal = z_goal if z_goal is not None else torch.zeros_like(actions[:, 0])
            actions = torch.cat([actions, goal.unsqueeze(1).expand(-1, depth, -1)], dim=-1)
        return frames, actions, gather_valid

    def _encode_history_latents(
        self, batch: dict[str, torch.Tensor], *, encode_actions: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """(actions, observations) history latents, [B, depth, D] each. `encode_actions=False`
        skips the action encode entirely, for callers that need only the tool outputs."""
        o_ids = batch["history_event_input_ids"]
        batch_size, depth = o_ids.shape[0], o_ids.shape[1]
        z_a = None
        if encode_actions:
            a_ids = batch["history_action_input_ids"]
            z_a = self.encode_latent_and_logits(
                a_ids.reshape(batch_size * depth, -1),
                batch["history_action_attention_mask"].reshape(batch_size * depth, -1),
            )[0].reshape(batch_size, depth, -1)
        z_o = self.encode_latent_and_logits(
            o_ids.reshape(batch_size * depth, -1),
            batch["history_event_attention_mask"].reshape(batch_size * depth, -1),
        )[0].reshape(batch_size, depth, -1)
        return z_a, z_o

    def _sample_categorical(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dreamer v2/v3 straight-through one-hot sampling over a stack of categoricals.

        Returns the flattened latent vector (B, categoricals*classes) and the
        per-group logits (B, categoricals, classes) used for the KL objective.
        """
        batch = logits.shape[0]
        logits = logits.view(batch, self.latent_categoricals, self.latent_classes).float()
        probs = torch.softmax(logits, dim=-1)
        if self.latent_unimix > 0:
            uniform = torch.ones_like(probs) / self.latent_classes
            probs = (1.0 - self.latent_unimix) * probs + self.latent_unimix * uniform
            logits = torch.log(probs.clamp_min(1e-8))
        if self.training and not getattr(self, "deterministic_latent_sampling", False):
            index = torch.distributions.Categorical(probs=probs).sample()
        else:
            index = probs.argmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(index, self.latent_classes).to(probs.dtype)
        sample = one_hot + probs - probs.detach()  # straight-through gradient estimator
        return sample.reshape(batch, -1), logits

    def _project_latent(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        projected = self.encoder_projector(pooled)
        if self.latent_type == "categorical":
            return self._sample_categorical(projected)
        return projected, None

    def encode_latent(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        latent, _ = self._project_latent(self._encode_pooled(input_ids, attention_mask))
        return latent

    def encode_latent_and_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._project_latent(self._encode_pooled(input_ids, attention_mask))

    def predict_latent(
        self,
        z_current: torch.Tensor,
        z_action: torch.Tensor,
        z_context: torch.Tensor,
        z_goal: torch.Tensor | None = None,
        *,
        frame_history: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict the next latent. Returns (latent_vector, prior_logits_or_None).

        `frame_history` is (frames [B,T,D], producing_actions [B,T,C], valid [B,T]) -- the event
        representations preceding this transition and the actions that produced them, used only
        by --predictor-arch transformer. Keyword-only and optional so every existing call site
        (rollouts, beam_plan, hier_latent_cem, the analysis scripts) keeps working; without it the
        transformer attends over just [context, z_current], a well-defined degenerate case.

        Use predict_latent_with_state when the belief state h_t is also wanted.
        """
        latent, logits, _ = self.predict_latent_with_state(
            z_current, z_action, z_context, z_goal, frame_history=frame_history
        )
        return latent, logits

    def predict_latent_with_state(
        self,
        z_current: torch.Tensor,
        z_action: torch.Tensor,
        z_context: torch.Tensor,
        z_goal: torch.Tensor | None = None,
        *,
        frame_history: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """(latent, prior_logits_or_None, state).

        `state` is the transformer predictor's last-position hidden state h_t -- the belief state
        that the causal transformer accumulates in place of a separate recurrent updater, and
        what the classification heads read. None under --predictor-arch mlp, which has no
        hidden state distinct from its output.
        """
        state = None
        if self.predictor_arch == "transformer":
            predicted, state = self._predict_latent_transformer(
                z_current, z_action, z_context, z_goal, frame_history
            )
        else:
            predicted = self.predictor(
                torch.cat([z_current, self._predictor_conditioning(z_action, z_context, z_goal)], dim=-1)
            )
        if self.latent_type == "categorical":
            sample, logits = self._sample_categorical(predicted)
            return sample, logits, state
        if self.latent_delta_prediction:
            # Residual parametrization: the predictor output is the change Δz. Applies
            # uniformly at train and inference time (multi-step rollout, beam_plan and
            # hier_latent_cem all route through here), so the trained semantics carry over.
            return z_current + predicted, None, state
        return predicted, None, state

    def recurrent_current_state(
        self,
        batch: dict[str, torch.Tensor],
        z_context: torch.Tensor,
        history_latents: tuple[torch.Tensor | None, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """s_t built recurrently from s_0 = I(c), never from cumulative history TEXT.

            s_0 = I(c);  s_i = U(s_{i-1}, a_i, E(o_i))  for each logged history step

        Steps beyond a shorter example's history are masked out so padded slots leave the state
        untouched, rather than folding a pad-token encode into it.
        """
        state = self.state_init(z_context)
        if "history_action_input_ids" not in batch:
            return state
        step_mask = batch["history_step_mask"].to(state.dtype)
        depth = batch["history_action_input_ids"].shape[1]
        z_a, z_o = history_latents if history_latents is not None else self._encode_history_latents(batch)
        for step in range(depth):
            updated = self.update_state(state, z_a[:, step], z_o[:, step])
            keep = step_mask[:, step].unsqueeze(-1)
            state = keep * updated + (1.0 - keep) * state
        return state

    @torch.no_grad()
    def rollout_events(
        self,
        z_state: torch.Tensor,
        z_actions: torch.Tensor,
        z_context: torch.Tensor,
        z_goal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference-time rollout: no true events available, so feed back the PREDICTED ones.

            e_hat_{k+1} = F(s_k, a_{k+1}, c)
            s_{k+1}     = U(s_k, a_{k+1}, e_hat_{k+1})

        This is the regime L_cons exists to protect -- training advances U with e*, here only
        e_hat exists. `z_actions` is [B, K, D]; returns (events [B, K, D], states [B, K, D]).
        Score a plan by running the canonical-event heads on the returned EVENTS: they are what
        the heads read under --canonical-event-head-inputs ctx_pred/pred_only.
        """
        if not self.state_updater_enabled:
            raise RuntimeError("rollout_events requires a model built with state_updater=True")
        events, states = [], []
        state = z_state
        for step in range(z_actions.shape[1]):
            action = z_actions[:, step]
            event, _ = self.predict_latent(state, action, z_context, z_goal)
            state = self.update_state(state, action, event)
            events.append(event)
            states.append(state)
        return torch.stack(events, dim=1), torch.stack(states, dim=1)

    def update_state(self, z_state: torch.Tensor, z_action: torch.Tensor, z_event: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = U(s_t, a_{t+1}, e_{t+1}). One definition, used by training and rollout."""
        return self.state_updater(torch.cat([z_state, z_action, z_event], dim=-1))

    def _encode_future(self, batch: dict[str, torch.Tensor], latent_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the batched future action / next-state tokens into latents:
        returns (z_future_action [B,K,D], z_future_next [B,K,D])."""
        fa_ids = batch["future_action_input_ids"]
        fa_mask = batch["future_action_attention_mask"]
        fn_ids = batch["future_next_input_ids"]
        fn_mask = batch["future_next_attention_mask"]
        batch_size, horizon_steps = fa_ids.shape[0], fa_ids.shape[1]
        z_future_action = self.encode_latent_and_logits(
            fa_ids.reshape(batch_size * horizon_steps, -1), fa_mask.reshape(batch_size * horizon_steps, -1)
        )[0].reshape(batch_size, horizon_steps, latent_dim)
        z_future_next = self.encode_latent_and_logits(
            fn_ids.reshape(batch_size * horizon_steps, -1), fn_mask.reshape(batch_size * horizon_steps, -1)
        )[0].reshape(batch_size, horizon_steps, latent_dim)
        return z_future_action, z_future_next

    def _fast_lewm_predict(
        self, z_current: torch.Tensor, z_context: torch.Tensor, z_goal: torch.Tensor, action_latents: torch.Tensor
    ) -> torch.Tensor:
        """Fast-LeWM action-prefix parallel prediction. `action_latents` is [B, H, D] =
        (a_0..a_{H-1}). Returns preds [B, H, D] where preds[:, k-1] is the latent reached after
        executing the length-k prefix (so preds[:, 0] is the one-step prediction z_{t+1})."""
        horizon = action_latents.shape[1]
        state_token = self.fast_state_proj(torch.cat([z_current, z_context], dim=-1)).unsqueeze(1)  # [B,1,d]
        action_tokens = self.fast_action_proj(action_latents)  # [B,H,d]
        sequence = torch.cat([state_token, action_tokens], dim=1)  # [B, 1+H, d]
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        sequence = sequence + self.fast_pos(positions).unsqueeze(0)
        length = sequence.shape[1]
        causal_mask = torch.triu(torch.full((length, length), float("-inf"), device=sequence.device), diagonal=1)
        encoded = self.fast_encoder(sequence, mask=causal_mask)  # [B, 1+H, d]
        prefix_tokens = encoded[:, 1:]  # [B, H, d]; position k summarizes a_0..a_{k-1} (+ state)
        anchor = z_current.unsqueeze(1).expand(-1, horizon, -1)
        context = z_context.unsqueeze(1).expand(-1, horizon, -1)
        parts = [anchor, prefix_tokens, context]
        if self.goal_conditioning:
            goal = z_goal if z_goal is not None else torch.zeros_like(z_current)
            parts.append(goal.unsqueeze(1).expand(-1, horizon, -1))
        preds = self.fast_predictor(torch.cat(parts, dim=-1))  # [B, H, D]
        if self.latent_delta_prediction:
            # Residual against the anchor: each horizon's output is the CUMULATIVE change
            # from z_current after executing that action prefix.
            preds = anchor + preds
        return preds

    def predict_success_logit(self, z_current: torch.Tensor, z_action: torch.Tensor, z_context: torch.Tensor, z_pred: torch.Tensor) -> torch.Tensor:
        return self.success_head(torch.cat([z_current, z_action, z_context, z_pred], dim=-1)).squeeze(-1)

    def predict_terminal_logit(self, z_current: torch.Tensor, z_action: torch.Tensor, z_context: torch.Tensor, z_pred: torch.Tensor) -> torch.Tensor:
        return self.terminal_head(torch.cat([z_current, z_action, z_context, z_pred], dim=-1)).squeeze(-1)

    def predict_value(self, z_current: torch.Tensor, z_action: torch.Tensor, z_context: torch.Tensor, z_pred: torch.Tensor) -> torch.Tensor:
        return self.value_head(torch.cat([z_current, z_action, z_context, z_pred], dim=-1)).squeeze(-1)

    def predict_canonical_event_logits(
        self,
        z_current: torch.Tensor,
        z_action: torch.Tensor,
        z_context: torch.Tensor,
        z_pred: torch.Tensor,
        z_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mode = getattr(self, "canonical_event_head_inputs", "all")
        if mode in {"state", "state_action"}:
            if z_state is None:
                raise ValueError(
                    f"canonical_event_head_inputs={mode!r} needs the predictor's hidden state; "
                    "call predict_latent_with_state and pass z_state."
                )
            # h_t is LayerNorm'd by the predictor while z_action is a raw projector output, so
            # the two blocks arrive on different scales; the trunk Linear absorbs that, the same
            # way 'all' already concatenates unnormalized latents.
            features = z_state if mode == "state" else torch.cat([z_state, z_action], dim=-1)
        elif mode == "pred_only":
            features = z_pred
        elif mode == "ctx_pred":
            features = torch.cat([z_context, z_pred], dim=-1)
        else:
            features = torch.cat([z_current, z_action, z_context, z_pred], dim=-1)
        trunk_features = self.canonical_event_trunk(features)
        return {field: head(trunk_features) for field, head in self.canonical_event_heads.items()}

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        compute_reconstruction: bool = False,
        compute_success: bool = False,
        compute_canonical_event: bool = False,
        compute_action_decoder: bool = False,
        compute_terminal: bool = False,
        compute_value: bool = False,
    ) -> dict[str, torch.Tensor]:
        from transformers.modeling_outputs import BaseModelOutput

        # Split pooling from projection for current/context/action: the recognition-probe
        # bypass (recognition_bypass_projector) reads the canonical heads off the RAW pooled
        # backbone features, so the pooled tensors must stay accessible. One backbone pass per
        # text either way; the projection is a single matmul on top.
        pooled_current = self._encode_pooled(batch["current_input_ids"], batch["current_attention_mask"])
        z_current, _ = self._project_latent(pooled_current)
        z_next, z_next_logits = self.encode_latent_and_logits(batch["next_input_ids"], batch["next_attention_mask"])
        pooled_context = self._encode_pooled(batch["context_input_ids"], batch["context_attention_mask"])
        z_context, _ = self._project_latent(pooled_context)
        pooled_action = self._encode_pooled(batch["action_input_ids"], batch["action_attention_mask"])
        z_action, _ = self._project_latent(pooled_action)
        # Encoded ONCE per forward pass and shared: --recurrent-state-init needs the action and
        # observation latents, the transformer predictor needs the observations. With both on,
        # encoding them separately would double the history cost (depth x batch extra encodes).
        history_latents = None
        if (self.recurrent_state_init or self.predictor_arch == "transformer") and (
            "history_event_input_ids" in batch
        ):
            history_latents = self._encode_history_latents(batch)
        if self.recurrent_state_init:
            # Replace the cumulative-history encoding with the recurrent state. z_current above
            # is still computed because other consumers (SigReg, PNI, the persistence
            # diagnostic) reference it, but from here on the state the predictor sees is the
            # one accumulated through U -- which is the point of s_0 = I(c).
            z_current = self.recurrent_current_state(batch, z_context, history_latents)
        if "goal_input_ids" in batch and "goal_attention_mask" in batch:
            z_goal, _ = self.encode_latent_and_logits(batch["goal_input_ids"], batch["goal_attention_mask"])
        else:
            z_goal = torch.zeros_like(z_current)
        # Event target E(o_{t+1}): observation only, never the action.
        z_event = None
        if self.state_updater_enabled and "event_input_ids" in batch:
            z_event, _ = self.encode_latent_and_logits(
                batch["event_input_ids"], batch["event_attention_mask"])
        multi_step_preds = multi_step_targets = multi_step_mask = None
        # Stays None on the Fast-LeWM path: that branch runs its own action-prefix transformer
        # (self.fast_predictor), not the AdaLN predictor, so encoding the history would be waste.
        frame_history = None
        z_predictor_state = None
        if self.fast_lewm and "future_action_input_ids" in batch:
            # Fast-LeWM (action-prefix parallel prediction): predict every horizon's latent
            # directly from the anchor z_current + its action prefix in ONE parallel pass, not
            # by recursive one-step rollout -- so errors don't compound across the horizon and
            # all horizons share one encode+predict pass. z_pred is the k=1 prefix prediction.
            z_future_action, multi_step_targets = self._encode_future(batch, z_current.shape[-1])
            action_latents = torch.cat([z_action.unsqueeze(1), z_future_action], dim=1)  # [B, H, D]
            preds = self._fast_lewm_predict(z_current, z_context, z_goal, action_latents)  # [B, H, D]
            z_pred, z_pred_logits = preds[:, 0], None
            multi_step_preds = preds[:, 1:]
            multi_step_mask = batch["future_step_mask"]
        else:
            # --predictor-arch transformer attends over the logged frame history; the MLP
            # predictor ignores it, so this is None (and unencoded) in the default configuration.
            if self.predictor_arch == "transformer":
                frame_history = self.encode_frame_history(batch, history_latents, z_goal)
            z_pred, z_pred_logits, z_predictor_state = self.predict_latent_with_state(
                z_current, z_action, z_context, z_goal if self.goal_conditioning else None,
                frame_history=frame_history,
            )
            if "future_action_input_ids" in batch:
                # Recursive multi-step rollout (LeWM-style): feed the teacher-forced future
                # actions and keep predicting the next latent from the previous PREDICTED latent.
                z_future_action, multi_step_targets = self._encode_future(batch, z_pred.shape[-1])
                goal = z_goal if self.goal_conditioning else None
                if self.state_updater_enabled and self.state_updater_objective == "future_event":
                    # Recurrent event/state rollout. multi_step_targets here are the successors'
                    # EVENT latents, so what we compare against them is the predicted EVENT at
                    # each step -- NOT the running state. The state is the hidden carrier:
                    #   s_{t+1} = U(s_t, a_{t+1}, e*_{t+1})     (teacher-forced on the TRUE event)
                    #   e_hat_{t+2} = F(s_{t+1}, a_{t+2}, c)
                    # so U's only gradient comes from whether s_{t+1} can still predict the next
                    # event. There is deliberately no target on s_{t+1} itself.
                    z_future_event = multi_step_targets
                    state = self.update_state(z_current, z_action, z_event if z_event is not None else z_pred)
                    predictions = []
                    history = frame_history
                    # Slide the window: the event just predicted becomes the newest token,
                    # tagged with the action that produced it.
                    previous_event, previous_action = z_pred, self._frame_conditioning(z_action, goal)
                    for step in range(z_future_action.shape[1]):
                        history = self._extend_frame_history(history, previous_event, previous_action)
                        step_action = z_future_action[:, step]
                        step_event, _ = self.predict_latent(
                            state, step_action, z_context, goal, frame_history=history)
                        predictions.append(step_event)
                        previous_event = step_event
                        previous_action = self._frame_conditioning(step_action, goal)
                        # Teacher forcing: advance with the TRUE event where we have one, so an
                        # early prediction error does not poison the rest of the horizon.
                        state = self.update_state(
                            state, z_future_action[:, step], z_future_event[:, step])
                    multi_step_preds = torch.stack(predictions, dim=1)  # [B, K, D] EVENTS
                    multi_step_mask = batch["future_step_mask"]
                else:
                    rollout = z_pred
                    predictions = []
                    history = frame_history
                    previous_action = self._frame_conditioning(z_action, goal)
                    for step in range(z_future_action.shape[1]):
                        # X_{t+1} = slide(X_t, e_hat_{t+1}): the prediction just made is the
                        # newest event token, tagged with the action that produced it.
                        history = self._extend_frame_history(history, rollout, previous_action)
                        step_action = z_future_action[:, step]
                        previous_action = self._frame_conditioning(step_action, goal)
                        rollout, _ = self.predict_latent(
                            rollout, step_action, z_context, goal, frame_history=history)
                        predictions.append(rollout)
                    multi_step_preds = torch.stack(predictions, dim=1)  # [B, K, D]
                    multi_step_mask = batch["future_step_mask"]
        # Teacher-forced vs inference-time state, for L_cons: training advances U with the TRUE
        # event, inference only has the predicted one. Penalising the gap between the two keeps
        # rollout from drifting away from the regime U was trained in.
        z_state_pred = z_state_tf = None
        if self.state_updater_enabled and z_event is not None:
            z_state_tf = self.update_state(z_current, z_action, z_event)
            z_state_pred = self.update_state(z_current, z_action, z_pred)

        # Action-contrastive: predict the next latent for each CORRUPTED action at the same
        # state. Reuses the same predictor and the same z_cur/z_context, so the only thing that
        # differs between z_pred_pos and z_pred_neg is the action -- which is exactly the
        # distinction plain MSE never has to make.
        z_pred_negative = None
        if "negative_action_input_ids" in batch:
            neg_ids = batch["negative_action_input_ids"]
            batch_size, num_negatives, length = neg_ids.shape
            z_act_neg, _ = self.encode_latent_and_logits(
                neg_ids.reshape(batch_size * num_negatives, length),
                batch["negative_action_attention_mask"].reshape(batch_size * num_negatives, length),
            )
            cur = z_current.unsqueeze(1).expand(-1, num_negatives, -1).reshape(batch_size * num_negatives, -1)
            ctx = z_context.unsqueeze(1).expand(-1, num_negatives, -1).reshape(batch_size * num_negatives, -1)
            goal = None
            if self.goal_conditioning:
                goal = z_goal.unsqueeze(1).expand(-1, num_negatives, -1).reshape(batch_size * num_negatives, -1)
            z_pred_negative, _ = self.predict_latent(
                cur, z_act_neg, ctx, goal,
                frame_history=self._expand_frame_history(frame_history, num_negatives),
            )
            z_pred_negative = z_pred_negative.reshape(batch_size, num_negatives, -1)
        obs_ground_loss = obs_ground_coverage = None
        if self.obs_grounding and "obs_ground_ids" in batch:
            obs_ground_loss, obs_ground_coverage = self._observation_ground_loss(z_pred, batch)
        tool_select_loss = tool_top1 = tool_top5 = tool_top10 = action_encoder_loss = None
        if (self.tool_select or self.action_encoder) and "tool_label" in batch:
            tool_label = batch["tool_label"]
            if self.tool_select:
                query_input = torch.cat([z_current, z_goal], dim=-1) if self.goal_conditioning else z_current
                tool_logits = self.tool_query(query_input) @ self.tool_embeddings.weight.t()  # [B, num_tools]
                tool_select_loss = torch.nn.functional.cross_entropy(tool_logits.float(), tool_label)
                tool_top1, tool_top5, tool_top10 = topk_recall(tool_logits.detach(), tool_label, (1, 5, 10))
            if self.action_encoder:
                action_encoder_loss = self._action_encoder_loss(z_action, tool_label, batch)
        action_decoder_loss = None
        if self.action_decoder and compute_action_decoder:
            action_decoder_loss = self._action_decoder_loss(z_action, batch)
        success_logits = self.predict_success_logit(z_current, z_action, z_context, z_pred) if compute_success else None
        terminal_logits = (
            self.predict_terminal_logit(z_current, z_action, z_context, z_pred)
            if (self.terminal_head_enabled and compute_terminal)
            else None
        )
        value_pred = (
            self.predict_value(z_current, z_action, z_context, z_pred)
            if (self.value_head_enabled and compute_value)
            else None
        )
        canonical_event_logits = None
        if compute_canonical_event:
            # Recognition-probe: when the batch carries the actual observation, read the head
            # off the encoded observation (z_observation) instead of the predicted latent z_pred.
            # This measures the recognition ceiling (post-hoc, with the outcome visible) vs the
            # default prediction setting (z_pred, before the outcome is known).
            readout_latent = z_pred
            pooled_observation = None
            if "canonical_event_observation_input_ids" in batch:
                pooled_observation = self._encode_pooled(
                    batch["canonical_event_observation_input_ids"], batch["canonical_event_observation_attention_mask"]
                )
                readout_latent, _ = self._project_latent(pooled_observation)
            if self.recognition_bypass_projector:
                # Bypass control: read the heads off RAW pooled backbone features for every
                # input, so no JEPA-trained component (projector included) sits in the readout
                # path -- a trained-MLP probe on the frozen encoder itself. Distinguishes "the
                # projector attenuates outcome directions" from "the pooled representation
                # never had them". Requires probe mode (an observation per row).
                if pooled_observation is None:
                    raise ValueError(
                        "recognition_bypass_projector requires the recognition-probe observation "
                        "batch; run with --canonical-event-recognition-probe on a probe JSONL."
                    )
                canonical_event_logits = self.predict_canonical_event_logits(
                    pooled_current, pooled_action, pooled_context, pooled_observation
                )
            else:
                canonical_event_logits = self.predict_canonical_event_logits(
                    z_current, z_action, z_context, readout_latent, z_predictor_state
                )
        reconstruction_loss = z_pred.new_zeros(())
        logits = None
        if compute_reconstruction:
            if not self.supports_reconstruction:
                raise ValueError("Reconstruction loss requires a seq2seq backbone; use --reconstruction-loss-coeff 0 for encoder-only backbones.")
            memory = self.memory_projection(z_pred).view(-1, self.memory_tokens, self.hidden_size)
            backbone_dtype = next(self.backbone.parameters()).dtype
            memory = memory.to(dtype=backbone_dtype)
            memory_mask = torch.ones(memory.shape[:2], dtype=batch["current_attention_mask"].dtype, device=memory.device)
            decoder_outputs = self.backbone(
                encoder_outputs=BaseModelOutput(last_hidden_state=memory),
                attention_mask=memory_mask,
                labels=batch["labels"],
            )
            reconstruction_loss = decoder_outputs.loss
            logits = decoder_outputs.logits
        return {
            "z_current": z_current,
            "z_next": z_next,
            "z_pred": z_pred,
            "z_goal": z_goal,
            "z_next_logits": z_next_logits,
            "z_pred_logits": z_pred_logits,
            **({"success_logits": success_logits, "success_labels": batch.get("success_labels"), "success_label_mask": batch.get("success_label_mask")} if success_logits is not None else {}),
            **({"terminal_logits": terminal_logits, "terminal_labels": batch.get("terminal_labels"), "terminal_label_mask": batch.get("terminal_label_mask")} if terminal_logits is not None else {}),
            **({"value_pred": value_pred, "value_target": batch.get("value_target"), "value_target_mask": batch.get("value_target_mask")} if value_pred is not None else {}),
            **({"canonical_event_logits": canonical_event_logits} if canonical_event_logits is not None else {}),
            **({"multi_step_preds": multi_step_preds, "multi_step_targets": multi_step_targets, "multi_step_mask": multi_step_mask} if multi_step_preds is not None else {}),
            **({"obs_ground_loss": obs_ground_loss, "obs_ground_coverage": obs_ground_coverage} if obs_ground_loss is not None else {}),
            **({"tool_select_loss": tool_select_loss, "tool_top1": tool_top1, "tool_top5": tool_top5, "tool_top10": tool_top10} if tool_select_loss is not None else {}),
            **({"action_encoder_loss": action_encoder_loss} if action_encoder_loss is not None else {}),
            **({"action_decoder_loss": action_decoder_loss} if action_decoder_loss is not None else {}),
            **({"z_event": z_event, "z_state_pred": z_state_pred, "z_state_tf": z_state_tf}
               if z_event is not None else {}),
            # The transformer predictor's belief state h_t. Not a loss target -- it is trained
            # end-to-end through the event prediction that reads off it -- but exposed for the
            # classification heads and for downstream ranking/probing.
            **({"z_predictor_state": z_predictor_state} if z_predictor_state is not None else {}),
            **({"z_pred_negative": z_pred_negative,
                "negative_action_valid": batch.get("negative_action_valid")} if z_pred_negative is not None else {}),
            "z_action": z_action,
            "reconstruction_loss": reconstruction_loss,
            "logits": logits,
        }

    def _action_encoder_loss(self, z_action: torch.Tensor, tool_label: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """P2: distill a fast structured-action encoder to the backbone-encoded z_action.
        g(tool_embedding, mean-pooled frozen arg-token embeddings) -> latent_dim, trained
        with Smooth L1 against the stop-gradient z_action. Uses only an embedding lookup +
        MLP (no backbone transformer), which is the sampling-acceleration win."""
        embed = self.backbone.get_input_embeddings()
        device_type = "cuda" if z_action.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            emb_weight = embed.weight.detach().float()
            arg_tokens = torch.nn.functional.embedding(batch["action_input_ids"], emb_weight)  # [B, L, E]
            arg_mask = batch["action_attention_mask"].float().unsqueeze(-1)
            arg_pooled = (arg_tokens * arg_mask).sum(dim=1) / arg_mask.sum(dim=1).clamp_min(1.0)  # [B, E]
            tool_emb = self.tool_embeddings(tool_label).float()  # [B, action_head_embed_dim]
            predicted = self.action_encoder_mlp(torch.cat([tool_emb, arg_pooled], dim=-1))  # [B, latent_dim]
            return torch.nn.functional.smooth_l1_loss(predicted, z_action.detach().float())

    def _action_decoder_augment(self, z_action: torch.Tensor, tool_label: torch.Tensor | None) -> torch.Tensor:
        """Two noise regimes, chosen per-example:
        - Same-tool latent mixup: interpolate z_action toward ANOTHER example in this batch
          that shares its tool_label (a real, on-manifold point), alpha ~ Uniform(0.6, 1.0).
          Trains the decoder on the actual geometry between real same-tool actions, which is
          the region a per-family Gaussian CEM proposal samples from -- not an isotropic ball
          that mostly points off-manifold.
        - Isotropic fallback: for examples with no same-tool partner in this batch (tool_label
          absent, or a singleton tool this step), noise_scale ~ Uniform(0, action_decoder_max_noise_std)
          * action_decoder_latent_scale -- the empirical std of real z_action (calibrate_action_decoder_
          latent_scale), so the noise magnitude is calibrated against the same number the
          inference-time CEM's init_std/min_std read, instead of an independent hand-picked constant.
        Either way this covers both training signals the design calls for: near-zero
        perturbation is plain reconstruction/cycle-consistency (decode(encode(a)) ~= a);
        larger perturbation is "prior-sample" robustness training against exactly the kind of
        point a CEM proposal samples near a real anchor -- what makes latent interpolation
        decodable at all, instead of only exact-anchor lookup."""
        batch_size = z_action.shape[0]
        scale = self.action_decoder_max_noise_std * float(self.action_decoder_latent_scale)
        noise_scale = torch.rand(batch_size, 1, device=z_action.device, dtype=z_action.dtype) * scale
        isotropic = z_action + noise_scale * torch.randn_like(z_action)
        if tool_label is None or batch_size < 2:
            return isotropic
        same_tool = tool_label.unsqueeze(0) == tool_label.unsqueeze(1)  # [B, B]
        same_tool.fill_diagonal_(False)
        has_partner = same_tool.any(dim=1)
        if not bool(has_partner.any()):
            return isotropic
        partner_scores = torch.rand(batch_size, batch_size, device=z_action.device)
        partner_scores = partner_scores.masked_fill(~same_tool, float("-inf"))
        partner_idx = torch.where(has_partner, partner_scores.argmax(dim=1), torch.arange(batch_size, device=z_action.device))
        alpha = torch.empty(batch_size, 1, device=z_action.device, dtype=z_action.dtype).uniform_(0.6, 1.0)
        partner_z = z_action.detach()[partner_idx]  # partner is a fixed anchor, no cross-example gradient coupling
        mixed = alpha * z_action + (1.0 - alpha) * partner_z
        return torch.where(has_partner.unsqueeze(-1), mixed, isotropic)

    def _action_decoder_loss(self, z_action: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct the action's OWN tokens (batch["action_input_ids"]) from an augmented
        z_action -- see _action_decoder_augment for the noise/mixup regime."""
        targets = batch["action_input_ids"]
        mask = batch["action_attention_mask"].to(z_action.dtype)
        tool_label = batch.get("tool_label")
        z_noised = self._action_decoder_augment(z_action, tool_label)
        if self.supports_reconstruction:
            return self._action_decoder_loss_seq2seq(z_noised, targets, mask)
        return self._action_decoder_loss_transformer(z_noised, targets, mask, tool_label)

    def _action_decoder_loss_seq2seq(self, z_action: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encoder-decoder path: mirrors _observation_ground_loss_seq2seq but with its OWN
        memory_projection (action_decoder_memory_projection) so gradients for decoding an
        ACTION never share weights with decoding a STATE/observation."""
        from transformers.modeling_outputs import BaseModelOutput

        labels = targets.clone()
        labels[mask == 0] = -100
        memory = self.action_decoder_memory_projection(z_action).view(-1, self.memory_tokens, self.hidden_size)
        backbone_dtype = next(self.backbone.parameters()).dtype
        memory = memory.to(dtype=backbone_dtype)
        memory_mask = torch.ones(memory.shape[:2], dtype=torch.long, device=memory.device)
        decoder_outputs = self.backbone(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=memory_mask,
            labels=labels,
        )
        return decoder_outputs.loss

    def _action_decoder_loss_transformer(
        self, z_action: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, tool_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encoder-only path: a multi-layer CAUSAL TRANSFORMER decoder (action_decoder_transformer)
        cross-attending to action_decoder_memory_tokens memory slots expanded from z_action --
        replaces the earlier single-layer-GRU design with attention over multiple memory
        positions instead of one collapsed recurrent hidden state. Output projection tied to the
        frozen backbone input embeddings, computed in fp32 (same precision convention as the
        seq2seq/obs-grounding paths -- the custom modules stay fp32 regardless of the backbone's
        own dtype, so no autocast surprises in the embedding-lookup + cross-entropy math).
        When action_decoder_tool_embeddings exists, an extra memory slot for tool_label (index 0
        = unknown tool when tool_label is None) hands the decoder the CEM's already-known family
        directly, instead of leaving it to infer from a possibly noised/off-manifold z_action."""
        batch_size, horizon = targets.shape
        embed = self.backbone.get_input_embeddings()
        device_type = "cuda" if z_action.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            emb_weight = embed.weight.detach().float()
            target_emb = torch.nn.functional.embedding(targets, emb_weight)  # [B,T,H]
            start = self.action_decoder_start.float().view(1, 1, -1).expand(batch_size, 1, -1)
            decoder_input_emb = torch.cat([start, target_emb[:, :-1]], dim=1)  # teacher-forced shift
            decoder_input = self.action_decoder_token_in(decoder_input_emb)  # [B,T,ad_dim]
            positions = torch.arange(horizon, device=z_action.device)
            decoder_input = decoder_input + self.action_decoder_pos(positions).unsqueeze(0)
            memory = self.action_decoder_memory_expand(z_action.float()).view(
                batch_size, self.action_decoder_memory_tokens, self.action_decoder_dim
            )
            if self.action_decoder_tool_vocab_size > 0:
                if tool_label is None:
                    tool_idx = torch.zeros(batch_size, dtype=torch.long, device=z_action.device)
                else:
                    tool_idx = tool_label.to(device=z_action.device, dtype=torch.long)
                tool_slot = self.action_decoder_tool_embeddings(tool_idx).float().unsqueeze(1)  # [B,1,ad_dim]
                memory = torch.cat([memory, tool_slot], dim=1)
            causal_mask = torch.triu(torch.full((horizon, horizon), float("-inf"), device=z_action.device), diagonal=1)
            decoded = self.action_decoder_transformer(decoder_input, memory, tgt_mask=causal_mask)  # [B,T,ad_dim]
            projected = self.action_decoder_out(decoded)  # [B,T,H]
            per_token = []
            for t in range(horizon):
                logits_t = projected[:, t].float() @ emb_weight.t()
                per_token.append(torch.nn.functional.cross_entropy(logits_t, targets[:, t], reduction="none"))
            cross_entropy = torch.stack(per_token, dim=1)
        denom = mask.sum().clamp_min(1.0)
        return (cross_entropy * mask).sum() / denom

    @torch.no_grad()
    def decode_action_latent(
        self, z_action: torch.Tensor, tokenizer: Any, max_new_tokens: int = 96, tool_label: torch.Tensor | int | None = None,
    ) -> list[str]:
        """Inference-time greedy decode: turn a (possibly CEM-sampled/interpolated) action
        latent into text, via the SAME action_decoder trained above. Used by
        src.hierarchical_action_sampling as an alternative to nearest-anchor lookup -- the
        caller is responsible for validating/falling back, since a decoder trained on real
        anchors is not guaranteed to produce a syntactically valid action for an arbitrary
        latent, only a *more likely* one than an untrained decode would."""
        if not self.action_decoder:
            raise RuntimeError("This checkpoint has no trained action_decoder.")
        from transformers.modeling_outputs import BaseModelOutput

        self.eval()
        batch_size = z_action.shape[0]
        if self.supports_reconstruction:
            memory = self.action_decoder_memory_projection(z_action).view(-1, self.memory_tokens, self.hidden_size)
            backbone_dtype = next(self.backbone.parameters()).dtype
            memory = memory.to(dtype=backbone_dtype)
            memory_mask = torch.ones(memory.shape[:2], dtype=torch.long, device=memory.device)
            generated = self.backbone.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=memory),
                attention_mask=memory_mask,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
            return [tokenizer.decode(row, skip_special_tokens=True) for row in generated]
        embed = self.backbone.get_input_embeddings()
        emb_weight = embed.weight.detach().float()
        memory = self.action_decoder_memory_expand(z_action.float()).view(
            batch_size, self.action_decoder_memory_tokens, self.action_decoder_dim
        )
        if self.action_decoder_tool_vocab_size > 0:
            # Mirrors _action_decoder_loss_transformer's tool-embedding memory slot -- the
            # caller (hierarchical_action_sampling) already knows the family it sampled u FROM,
            # so hand it to the decoder directly instead of leaving it to infer from z_action.
            if tool_label is None:
                tool_idx = torch.zeros(batch_size, dtype=torch.long, device=z_action.device)
            elif isinstance(tool_label, torch.Tensor):
                tool_idx = tool_label.to(device=z_action.device, dtype=torch.long).view(batch_size)
            else:
                tool_idx = torch.full((batch_size,), int(tool_label), dtype=torch.long, device=z_action.device)
            tool_slot = self.action_decoder_tool_embeddings(tool_idx).float().unsqueeze(1)
            memory = torch.cat([memory, tool_slot], dim=1)
        # No KV cache: the whole prefix-so-far is re-run through the decoder every step. Fine
        # for max_new_tokens ~= 96 (O(L^2) over a short sequence, called at most once per
        # planning cycle) -- add incremental caching later only if this becomes a bottleneck.
        current_embs = self.action_decoder_start.float().view(1, 1, -1).expand(batch_size, 1, -1)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=z_action.device)
        token_rows: list[list[int]] = [[] for _ in range(batch_size)]
        for _ in range(max_new_tokens):
            length = current_embs.shape[1]
            decoder_input = self.action_decoder_token_in(current_embs)
            positions = torch.arange(length, device=z_action.device)
            decoder_input = decoder_input + self.action_decoder_pos(positions).unsqueeze(0)
            causal_mask = torch.triu(torch.full((length, length), float("-inf"), device=z_action.device), diagonal=1)
            decoded = self.action_decoder_transformer(decoder_input, memory, tgt_mask=causal_mask)
            logits = self.action_decoder_out(decoded[:, -1]).float() @ emb_weight.t()
            next_token = logits.argmax(dim=-1)
            for i in range(batch_size):
                if not finished[i].item():
                    token_rows[i].append(int(next_token[i].item()))
            if eos_id is not None:
                finished = finished | (next_token == eos_id)
                if bool(finished.all().item()):
                    break
            next_emb = torch.nn.functional.embedding(next_token, emb_weight).unsqueeze(1)
            current_embs = torch.cat([current_embs, next_emb], dim=1)
        return [tokenizer.decode(row, skip_special_tokens=True) for row in token_rows]

    def _observation_ground_loss(self, z_pred: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the first-N isolated observation tokens from z_pred. Encoder-decoder
        backbones use their native decoder; encoder-only backbones use the causal-Transformer
        head. Returns (masked token-mean CE, coverage = fraction of examples with a target)."""
        targets = batch["obs_ground_ids"]
        mask = batch["obs_ground_mask"].to(z_pred.dtype)
        coverage = (mask.sum(dim=1) > 0).to(z_pred.dtype).mean()
        if self.supports_reconstruction:
            loss = self._observation_ground_loss_seq2seq(z_pred, targets, mask)
        else:
            loss = self._observation_ground_loss_transformer(z_pred, targets, mask)
        return loss, coverage

    def _observation_ground_loss_seq2seq(self, z_pred: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encoder-decoder path: feed z_pred through obs_ground_memory_projection (its OWN
        weights, not shared with reconstruction_loss's memory_projection) as the decoder's
        encoder memory and let the backbone's native decoder predict the (truncated,
        isolated) observation. Padding positions are set to -100 so HF's own token-mean
        cross-entropy ignores them; a fully-masked row contributes nothing."""
        from transformers.modeling_outputs import BaseModelOutput

        labels = targets.clone()
        labels[mask == 0] = -100
        memory = self.obs_ground_memory_projection(z_pred).view(-1, self.memory_tokens, self.hidden_size)
        backbone_dtype = next(self.backbone.parameters()).dtype
        memory = memory.to(dtype=backbone_dtype)
        memory_mask = torch.ones(memory.shape[:2], dtype=torch.long, device=memory.device)
        decoder_outputs = self.backbone(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=memory_mask,
            labels=labels,
        )
        return decoder_outputs.loss

    def _observation_ground_loss_transformer(self, z_pred: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encoder-only path: a multi-layer CAUSAL TRANSFORMER decoder (obs_ground_transformer)
        cross-attending to obs_ground_decoder_memory_tokens memory slots expanded from z_pred --
        same architectural upgrade as the action decoder's _action_decoder_loss_transformer,
        applied here to observation-grounding. Output projection tied to the frozen backbone
        input embeddings, computed in fp32 and chunked over time."""
        batch_size, horizon = targets.shape
        embed = self.backbone.get_input_embeddings()
        device_type = "cuda" if z_pred.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            emb_weight = embed.weight.detach().float()  # [V, E], frozen backbone embeddings
            target_emb = torch.nn.functional.embedding(targets, emb_weight)  # [B, T, E]
            start = self.obs_ground_start.float().view(1, 1, -1).expand(batch_size, 1, -1)
            decoder_input_emb = torch.cat([start, target_emb[:, :-1]], dim=1)  # teacher-forced shift
            decoder_input = self.obs_ground_token_in(decoder_input_emb)  # [B,T,og_dim]
            positions = torch.arange(horizon, device=z_pred.device)
            decoder_input = decoder_input + self.obs_ground_pos(positions).unsqueeze(0)
            memory = self.obs_ground_memory_expand(z_pred.float()).view(
                batch_size, self.obs_ground_decoder_memory_tokens, self.obs_ground_decoder_dim
            )
            causal_mask = torch.triu(torch.full((horizon, horizon), float("-inf"), device=z_pred.device), diagonal=1)
            decoded = self.obs_ground_transformer(decoder_input, memory, tgt_mask=causal_mask)  # [B,T,og_dim]
            projected = self.obs_ground_out(decoded)  # [B, T, E]
            per_token = []
            for t in range(horizon):
                logits_t = projected[:, t].float() @ emb_weight.t()  # [B, V]
                per_token.append(torch.nn.functional.cross_entropy(logits_t, targets[:, t], reduction="none"))
            cross_entropy = torch.stack(per_token, dim=1)  # [B, T]
        denom = mask.sum().clamp_min(1.0)
        return (cross_entropy * mask).sum() / denom


def sigreg_loss(z: torch.Tensor, num_projections: int, eps: float) -> torch.Tensor:
    if z.shape[0] < 2:
        mean_loss = z.mean(dim=0).pow(2).mean()
        var_loss = (z.var(dim=0, unbiased=False) - 1.0).pow(2).mean()
        return mean_loss + var_loss
    z = z.float()
    directions = torch.randn(z.shape[-1], num_projections, device=z.device, dtype=z.dtype)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(eps)
    projected = z @ directions
    sorted_projected = projected.sort(dim=0).values
    gaussian = torch.randn_like(projected).sort(dim=0).values
    sliced_loss = (sorted_projected - gaussian).pow(2).mean()
    mean_loss = z.mean(dim=0).pow(2).mean()
    std_loss = (z.std(dim=0, unbiased=False) - 1.0).pow(2).mean()
    return sliced_loss + mean_loss + std_loss


def latent_prediction_loss(
    z_pred: torch.Tensor,
    z_next: torch.Tensor,
    *,
    loss_type: str = "smooth_l1_cosine",
    beta: float = 1.0,
    delta_anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-step latent prediction loss against the stop-gradient target z_next.

    'mse_cosine' is the legacy MSE + (1 - cosine) objective. 'smooth_l1'/'smooth_l1_cosine'
    replace the MSE term with a Smooth L1 (Huber) loss, which is less sensitive to the
    outlier latents that produced the loss spikes during training (cf. NextLat, LSE-MTP);
    the cosine term is retained unless loss_type == 'smooth_l1'.

    `delta_anchor` (--latent-delta-prediction): when given (z_current), the cosine term is
    computed between the predicted and true DELTAS (z_pred - z_current vs z_next - z_current)
    instead of the absolute latents. Consecutive tool-use states share most of their content,
    so absolute-space cosine is dominated by the carried-over state and is near 1 for any
    reasonable prediction; delta-space cosine supervises the direction of what changed. The
    point-wise term is unaffected: |z_pred - z_next| == |Δz_pred - Δz_true| identically.
    """
    target = z_next.detach()
    if loss_type == "mse_cosine":
        base = (z_pred - target).pow(2).mean()
    else:
        base = torch.nn.functional.smooth_l1_loss(z_pred, target, beta=beta)
    if loss_type == "smooth_l1":
        return base
    if delta_anchor is not None:
        cosine = 1.0 - torch.nn.functional.cosine_similarity(
            z_pred - delta_anchor, (z_next - delta_anchor).detach(), dim=-1
        ).mean()
    else:
        cosine = 1.0 - torch.nn.functional.cosine_similarity(z_pred, target, dim=-1).mean()
    return base + cosine


def masked_latent_prediction_loss(
    z_pred: torch.Tensor,
    z_next: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_type: str = "smooth_l1_cosine",
    beta: float = 1.0,
    delta_anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-example latent loss reduced over only the rows whose mask is 1.

    Same objective as latent_prediction_loss (MSE or Smooth L1, plus the cosine term unless
    loss_type == 'smooth_l1'), but --joint-canonical-event-training has no latent target for a
    trajectory's terminal row: those rows carry a placeholder z_next and mask=0, so they must
    contribute nothing rather than pulling z_pred toward their own current state. Returns a
    zero scalar (still connected to the graph) when no row in the batch has a target.
    """
    target = z_next.detach()
    mask = mask.to(dtype=z_pred.dtype)
    denom = mask.sum().clamp_min(1.0)
    if loss_type == "mse_cosine":
        per_example = (z_pred - target).pow(2).mean(dim=-1)
    else:
        per_example = torch.nn.functional.smooth_l1_loss(z_pred, target, beta=beta, reduction="none").mean(dim=-1)
    total = (per_example * mask).sum() / denom
    if loss_type == "smooth_l1":
        return total
    if delta_anchor is not None:
        cosine = 1.0 - torch.nn.functional.cosine_similarity(
            z_pred - delta_anchor, (z_next - delta_anchor).detach(), dim=-1
        )
    else:
        cosine = 1.0 - torch.nn.functional.cosine_similarity(z_pred, target, dim=-1)
    return total + (cosine * mask).sum() / denom


def action_contrastive_loss(
    z_pred: torch.Tensor,
    z_pred_negative: torch.Tensor,
    z_next: torch.Tensor,
    valid: torch.Tensor | None,
    margin: float,
    loss_type: str = "softplus",
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Ranking loss requiring the CORRECT action's prediction to sit closer to the realized
    next latent than a corrupted action's prediction, at the same state.

    With the default softplus objective:
        L_AC = softplus((margin + d(zhat+, z_next) - d(zhat-_j, z_next)) / temperature)

    Passing loss_type="hinge" reproduces the old clamp objective:
        L_AC = max(0, margin + d(zhat+, z_next) - d(zhat-_j, z_next))

    d is cosine distance (bounded in [0, 2], so `margin` is scale-free and does not have to be
    retuned when latent norms drift). The target is stop-gradient, exactly as in the MSE term.

    This is the requirement plain MSE lacks: MSE is satisfied by a persistence map
    zhat = z_t that is identical for every action, which both leaves the model action-blind and
    still passes SigReg (persistence is not a global collapse -- z_t itself has full variance).

    Returns (loss, diagnostics) where diagnostics carry the mean positive/negative distances,
    their gap, and the fraction of triplets still violating the margin (the "active" rate) --
    an active rate collapsing to 0 early means the margin is too small to be doing work.
    """
    target = z_next.detach()
    d_pos = 1.0 - torch.nn.functional.cosine_similarity(z_pred, target, dim=-1)          # [B]
    d_neg = 1.0 - torch.nn.functional.cosine_similarity(
        z_pred_negative, target.unsqueeze(1).expand_as(z_pred_negative), dim=-1)          # [B, K]
    raw_margin = margin + d_pos.unsqueeze(1) - d_neg                                      # [B, K]
    hinge_violation = torch.clamp(raw_margin, min=0.0)
    if loss_type == "hinge":
        per_pair_loss = hinge_violation
    elif loss_type == "softplus":
        temp = max(float(temperature), 1e-6)
        per_pair_loss = torch.nn.functional.softplus(raw_margin / temp)
    else:
        raise ValueError(f"Unknown action contrastive loss_type: {loss_type}")
    if valid is None:
        valid = torch.ones_like(per_pair_loss)
    valid = valid.to(dtype=per_pair_loss.dtype)
    denom = valid.sum().clamp_min(1.0)
    loss = (per_pair_loss * valid).sum() / denom
    diagnostics = {
        "ac_d_pos": d_pos.mean().detach(),
        "ac_d_neg": ((d_neg * valid).sum() / denom).detach(),
        "ac_gap": (((d_neg - d_pos.unsqueeze(1)) * valid).sum() / denom).detach(),
        "ac_active_rate": (((hinge_violation > 0).to(per_pair_loss.dtype) * valid).sum() / denom).detach(),
    }
    return loss, diagnostics


def persistence_normalized_improvement(z_pred: torch.Tensor, z_current: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
    """PNI = 1 - MSE(zhat, z_next) / MSE(z_t, z_next).

    The memo's headline diagnostic: PNI > 0 means the predictor is genuinely beating a copy of
    the current state. Our measured baseline had persistence WINNING, so this is the number to
    watch when the contrastive term is switched on.
    """
    target = z_next.detach()
    pred_error = (z_pred - target).pow(2).mean()
    copy_error = (z_current.detach() - target).pow(2).mean()
    return 1.0 - pred_error / copy_error.clamp_min(1e-12)


def topk_recall(logits: torch.Tensor, labels: torch.Tensor, ks: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    """Top-k recall (fraction of examples whose true label is in the top-k logits)."""
    maxk = min(max(ks), logits.shape[-1])
    topk = logits.topk(maxk, dim=-1).indices  # [B, maxk]
    correct = topk == labels.unsqueeze(1)
    return tuple(correct[:, :min(k, maxk)].any(dim=1).float().mean() for k in ks)


def pointwise_latent_error(z_pred: torch.Tensor, z_next: torch.Tensor, loss_type: str, beta: float) -> torch.Tensor:
    """Elementwise point-wise error against the stop-gradient target, for multi-step
    supervision. Uses squared error for the 'mse_cosine' family and Smooth L1 otherwise;
    the cosine term is intentionally excluded from multi-step supervision. Caller reduces
    (e.g. mean over the latent dim, then a masked mean over valid horizon steps)."""
    target = z_next.detach()
    if loss_type == "mse_cosine":
        return (z_pred - target).pow(2)
    return torch.nn.functional.smooth_l1_loss(z_pred, target, beta=beta, reduction="none")


def categorical_kl_loss(
    prior_logits: torch.Tensor,
    posterior_logits: torch.Tensor,
    kl_balance: float,
    free_nats: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dreamer v2/v3 KL-balanced loss between the predicted prior and the encoded posterior.

    Logits are shaped (B, categoricals, classes). The KL is summed over the
    categorical groups, clamped by free bits, and balanced between a dynamics
    term (trains the predictor/prior) and a representation term (trains the
    encoder/posterior). Returns (loss, kl_dynamics, kl_representation).
    """
    prior_logits = prior_logits.float()
    posterior_logits = posterior_logits.float()
    prior = torch.distributions.Categorical(logits=prior_logits)
    posterior = torch.distributions.Categorical(logits=posterior_logits)
    prior_sg = torch.distributions.Categorical(logits=prior_logits.detach())
    posterior_sg = torch.distributions.Categorical(logits=posterior_logits.detach())
    kl_dynamics = torch.distributions.kl_divergence(posterior_sg, prior).sum(dim=-1)
    kl_representation = torch.distributions.kl_divergence(posterior, prior_sg).sum(dim=-1)
    if free_nats > 0:
        kl_dynamics = kl_dynamics.clamp_min(free_nats)
        kl_representation = kl_representation.clamp_min(free_nats)
    loss = kl_balance * kl_dynamics + (1.0 - kl_balance) * kl_representation
    return loss.mean(), kl_dynamics.mean(), kl_representation.mean()


def success_classification_loss(outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if "success_logits" not in outputs:
        zero = outputs["z_pred"].new_zeros(())
        return zero, zero
    logits = outputs["success_logits"].float()
    labels = outputs.get("success_labels")
    mask = outputs.get("success_label_mask")
    if labels is None or mask is None:
        zero = logits.new_zeros(())
        return zero, zero
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    denom = mask.sum().clamp_min(1.0)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss = (loss * mask).sum() / denom
    predictions = (torch.sigmoid(logits) >= 0.5).to(labels.dtype)
    accuracy = ((predictions == labels).to(logits.dtype) * mask).sum() / denom
    return loss, accuracy


def terminal_classification_loss(
    outputs: dict[str, torch.Tensor],
    *,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BCE loss for the optional terminal-step head, identical in structure to
    success_classification_loss (same masked-BCE-plus-accuracy shape), predicting a
    different quantity: is this step the LAST one in its trajectory."""
    if "terminal_logits" not in outputs:
        zero = outputs["z_pred"].new_zeros(())
        return zero, zero
    logits = outputs["terminal_logits"].float()
    labels = outputs.get("terminal_labels")
    mask = outputs.get("terminal_label_mask")
    if labels is None or mask is None:
        zero = logits.new_zeros(())
        return zero, zero
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    denom = mask.sum().clamp_min(1.0)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none", pos_weight=pos_weight
    )
    loss = (loss * mask).sum() / denom
    predictions = (torch.sigmoid(logits) >= 0.5).to(labels.dtype)
    accuracy = ((predictions == labels).to(logits.dtype) * mask).sum() / denom
    return loss, accuracy


def value_regression_loss(outputs: dict[str, torch.Tensor], beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked Smooth L1 (Huber) loss for the optional value head, predicting the continuous
    discounted return-to-go rather than a class. Smooth L1 (not MSE) since value_target is
    computed from a terminal reward on the order of +-terminal_reward_scale (default +-5) --
    an occasional large-magnitude target should not dominate the gradient the way a squared
    error would. Returns (loss, mean_absolute_error) -- MAE is the reported metric since
    "accuracy" has no meaning for a regression target."""
    if "value_pred" not in outputs:
        zero = outputs["z_pred"].new_zeros(())
        return zero, zero
    pred = outputs["value_pred"].float()
    target = outputs.get("value_target")
    mask = outputs.get("value_target_mask")
    if target is None or mask is None:
        zero = pred.new_zeros(())
        return zero, zero
    target = target.to(device=pred.device, dtype=pred.dtype)
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    denom = mask.sum().clamp_min(1.0)
    loss = torch.nn.functional.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    loss = (loss * mask).sum() / denom
    mae = ((pred - target).abs() * mask).sum() / denom
    return loss, mae


def class_balanced_weights(counts: list[int], method: str, beta: float) -> list[float]:
    """Per-class loss weights from training counts. 'inverse' = N/(K*n_c);
    'effective_num' = (1-beta)/(1-beta^{n_c}) (Cui et al. class-balanced loss). Weights are
    normalized to mean 1 over present classes so the loss scale is unchanged; absent classes
    get weight 0 (they never appear as a training target)."""
    if method == "inverse":
        total = sum(counts)
        num_classes = max(len([c for c in counts if c > 0]), 1)
        raw = [(total / (num_classes * c)) if c > 0 else 0.0 for c in counts]
    elif method == "effective_num":
        raw = [((1.0 - beta) / (1.0 - beta ** c)) if c > 0 else 0.0 for c in counts]
    else:
        return [1.0] * len(counts)
    present = [w for w in raw if w > 0]
    mean = (sum(present) / len(present)) if present else 1.0
    return [(w / mean) if w > 0 else 0.0 for w in raw]


def canonical_event_class_weights(
    examples: list["CanonicalEventExample"],
    vocab: dict[str, list[str]],
    *,
    method: str,
    beta: float,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Compute per-field single-label class weights and multi-label positive weights from the
    training label frequencies. Returns ({field: weights}, {field: pos_weights})."""
    single_weights: dict[str, list[float]] = {}
    for field in CANONICAL_EVENT_SINGLE_LABEL_FIELDS:
        if field not in vocab:  # dropped by --canonical-event-heads
            continue
        index = {value: i for i, value in enumerate(vocab[field])}
        counts = [0] * len(vocab[field])
        for example in examples:
            value = example.single_labels.get(field)
            if value in index:
                counts[index[value]] += 1
        single_weights[field] = class_balanced_weights(counts, method, beta)
    multi_pos_weights: dict[str, list[float]] = {}
    total = max(len(examples), 1)
    for field in NUDGE_MULTI_LABEL_FIELDS:
        if field not in vocab:
            continue
        index = {value: i for i, value in enumerate(vocab[field])}
        positives = [0] * len(vocab[field])
        for example in examples:
            for value in example.multi_labels.get(field, []):
                if value in index:
                    positives[index[value]] += 1
        # BCE pos_weight_c = negatives/positives, capped to keep it finite/stable.
        multi_pos_weights[field] = [min((total - p) / p, 100.0) if p > 0 else 1.0 for p in positives]
    return single_weights, multi_pos_weights


def terminal_pos_weight_from_examples(
    examples: list["CanonicalEventExample"],
    *,
    method: str,
    beta: float,
) -> tuple[float | None, dict[str, int]]:
    """Return BCE pos_weight for terminal=1 plus terminal/non-terminal counts.

    BCEWithLogitsLoss' ``pos_weight`` scales only positive examples, so class-balanced
    binary training is represented as the ratio w_terminal / w_nonterminal. Eval keeps
    using the unweighted loss, matching the canonical-event class-head convention.
    """
    terminal = sum(1 for example in examples if int(example.terminal_label or 0) == 1)
    nonterminal = sum(1 for example in examples if int(example.terminal_label or 0) == 0)
    counts = {"nonterminal": nonterminal, "terminal": terminal}
    if method == "none" or terminal <= 0 or nonterminal <= 0:
        return None, counts
    if method == "inverse":
        return float(nonterminal) / float(terminal), counts
    weights = class_balanced_weights([nonterminal, terminal], method, beta)
    if weights[0] <= 0:
        return None, counts
    return float(weights[1] / weights[0]), counts


def _focal_cross_entropy(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None, gamma: float) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    cross_entropy = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [B]
    focal = (1.0 - cross_entropy.neg().exp()) ** gamma * cross_entropy      # (1 - p_t)^gamma * CE
    if weight is not None:
        focal = focal * weight[target]
    return focal.mean()


def canonical_event_classification_loss(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    class_weights: dict[str, torch.Tensor] | None = None,
    multi_pos_weights: dict[str, torch.Tensor] | None = None,
    focal_gamma: float = 0.0,
) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
    """Mean per-field loss across every canonical_event_state/nudge head.

    Single-label fields use softmax cross-entropy (optionally class-frequency-weighted and/or
    focal); the multi-label missing_information_type field uses per-class BCE (optionally with a
    positive-class pos_weight). Pass no weights (eval) for the plain, unweighted metric.
    """
    total: torch.Tensor | None = None
    per_field: dict[str, dict[str, torch.Tensor]] = {}
    for field, field_logits in logits.items():
        target = batch[f"label_{field}"].to(field_logits.device)
        if field in NUDGE_MULTI_LABEL_FIELDS:
            pos_weight = None
            if multi_pos_weights and field in multi_pos_weights:
                pos_weight = multi_pos_weights[field].to(field_logits.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                field_logits, target.to(field_logits.dtype), pos_weight=pos_weight
            )
            predictions = (torch.sigmoid(field_logits) >= 0.5).to(target.dtype)
            accuracy = (predictions == target).to(field_logits.dtype).mean()
        else:
            weight = None
            if class_weights and field in class_weights:
                weight = class_weights[field].to(field_logits.device)
            if focal_gamma and focal_gamma > 0:
                loss = _focal_cross_entropy(field_logits, target, weight, focal_gamma)
            else:
                loss = torch.nn.functional.cross_entropy(field_logits, target, weight=weight)
            accuracy = (field_logits.argmax(dim=-1) == target).to(field_logits.dtype).mean()
        per_field[field] = {"loss": loss.detach(), "accuracy": accuracy.detach()}
        total = loss if total is None else total + loss
    total = total / max(len(logits), 1) if total is not None else next(iter(batch.values())).new_zeros(())
    return total, per_field


def compute_jepa_loss(outputs: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # --event-state-decomposition: the predictor's target is the EVENT, so every term that
    # compares z_pred against "the next latent" must compare it against z_event instead. The
    # cumulative next state is handled separately below by the State Updater's auxiliary loss.
    decomposed = outputs.get("z_event") is not None
    # Only the recurrent objective reinterprets the horizon targets as future EVENTS; in
    # `next_state` mode they are still cumulative states and must stay folded into latent_loss.
    recurrent = decomposed and getattr(args, "state_updater_objective", "future_event") == "future_event"
    future_event_loss = None
    if decomposed:
        outputs = {**outputs, "z_next": outputs["z_event"], "_z_state_target": outputs["z_next"]}
    if getattr(args, "latent_type", "continuous") == "categorical":
        latent_loss, kl_dynamics, kl_representation = categorical_kl_loss(outputs["z_pred_logits"], outputs["z_next_logits"], args.kl_balance, args.kl_free_nats)
        sig_loss = outputs["z_pred"].new_zeros(())
        latent_coeff = args.kl_loss_coeff
    else:
        loss_type = getattr(args, "latent_loss_type", "smooth_l1_cosine")
        beta = getattr(args, "smooth_l1_beta", 1.0)
        if outputs.get("multi_step_preds") is not None:
            # Horizon-averaged point-wise (no cosine) latent loss: step 1 (z_pred vs z_next)
            # plus each recursive step (multi_step_preds vs true future states), equal weight
            # per valid step.
            step1 = pointwise_latent_error(outputs["z_pred"], outputs["z_next"], loss_type, beta).mean(dim=-1)  # [B]
            future = pointwise_latent_error(outputs["multi_step_preds"], outputs["multi_step_targets"], loss_type, beta).mean(dim=-1)  # [B, K]
            future_mask = outputs["multi_step_mask"].to(step1.dtype)
            if recurrent:
                # Recurrent event/state: step 1 is L_event and the horizon is L_future, and they
                # carry different weights (lambda_e > lambda_f), so they must not be averaged
                # into one number. L_future is surfaced separately and added below.
                latent_loss = step1.mean()
                future_event_loss = (future * future_mask).sum() / future_mask.sum().clamp_min(1.0)
            else:
                step_losses = torch.cat([step1.unsqueeze(1), future], dim=1)  # [B, 1+K]
                step_mask = torch.cat([torch.ones_like(step1).unsqueeze(1), future_mask], dim=1)
                latent_loss = (step_losses * step_mask).sum() / step_mask.sum().clamp_min(1.0)
            # SigReg over the whole horizon: whiten z_current, z_next and every valid true
            # future state (kept non-detached so gradients reach the projector).
            reps = [outputs["z_current"], outputs["z_next"]]
            targets = outputs["multi_step_targets"]
            valid = outputs["multi_step_mask"].reshape(-1).to(torch.bool)
            future_true = targets.reshape(-1, targets.shape[-1])[valid]
            if future_true.shape[0] > 0:
                reps.append(future_true)
            sig_loss = sigreg_loss(torch.cat(reps, dim=0), args.sigreg_projections, args.sigreg_eps)
        else:
            latent_loss = latent_prediction_loss(
                outputs["z_pred"],
                outputs["z_next"],
                loss_type=loss_type,
                beta=beta,
                delta_anchor=outputs["z_current"] if getattr(args, "latent_delta_prediction", False) else None,
            )
            sig_loss = sigreg_loss(torch.cat([outputs["z_current"], outputs["z_next"]], dim=0), args.sigreg_projections, args.sigreg_eps)
        kl_dynamics = kl_representation = None
        latent_coeff = args.latent_loss_coeff
        if decomposed:
            # Under the decomposition this slot weights the EVENT term, so --event-loss-coeff
            # is what controls it. Without this the flag would be accepted and silently inert.
            latent_coeff = float(getattr(args, "event_loss_coeff", 1.0))
    recon_loss = outputs["reconstruction_loss"]
    success_enabled = success_head_training_enabled(args)
    success_loss, success_accuracy = success_classification_loss(outputs) if success_enabled else (outputs["z_pred"].new_zeros(()), outputs["z_pred"].new_zeros(()))
    terminal_loss_coeff = float(getattr(args, "terminal_loss_coeff", 0.0) or 0.0)
    terminal_enabled = terminal_loss_coeff > 0 and "terminal_logits" in outputs
    terminal_loss, terminal_accuracy = terminal_classification_loss(outputs) if terminal_enabled else (outputs["z_pred"].new_zeros(()), outputs["z_pred"].new_zeros(()))
    obs_ground_coeff = float(getattr(args, "obs_token_ground_coeff", 0.0) or 0.0)
    obs_ground_loss = outputs.get("obs_ground_loss")
    tool_select_coeff = float(getattr(args, "tool_select_loss_coeff", 0.0) or 0.0)
    action_encoder_coeff = float(getattr(args, "action_encoder_loss_coeff", 0.0) or 0.0)
    tool_select_loss = outputs.get("tool_select_loss")
    action_encoder_loss = outputs.get("action_encoder_loss")
    action_decoder_coeff = float(getattr(args, "action_decoder_loss_coeff", 0.0) or 0.0)
    action_decoder_loss = outputs.get("action_decoder_loss")
    # State Updater objective.
    #   future_event (default): NO direct target on s_{t+1}. Its gradient arrives only through
    #     L_future -- the multi-step term above, whose targets are the successors' event latents
    #     and whose predictions were produced from U's output. Constraining what the state can
    #     PREDICT rather than what it should RESEMBLE is the whole point: a cumulative-next-state
    #     MSE just relocates the persistence shortcut from F into U.
    #   next_state: the naive baseline, kept so the failure can be demonstrated rather than
    #     asserted.
    state_update_loss = None
    if decomposed and not recurrent and outputs.get("z_state_pred") is not None:
        state_update_loss = latent_prediction_loss(
            outputs["z_state_pred"], outputs["_z_state_target"],
            loss_type=getattr(args, "latent_loss_type", "smooth_l1_cosine"),
            beta=getattr(args, "smooth_l1_beta", 1.0),
        )

    # L_cons: teacher-forced U(s,a,e*) vs inference-time U(s,a,e_hat). Closes the gap between
    # how U is trained and how it is used at rollout time.
    consistency_loss = None
    if recurrent and outputs.get("z_state_tf") is not None and outputs.get("z_state_pred") is not None:
        consistency_loss = latent_prediction_loss(
            outputs["z_state_pred"], outputs["z_state_tf"],
            loss_type=getattr(args, "latent_loss_type", "smooth_l1_cosine"),
            beta=getattr(args, "smooth_l1_beta", 1.0),
        )
    action_contrastive_coeff = float(getattr(args, "action_contrastive_loss_coeff", 0.0) or 0.0)
    ac_loss, ac_diagnostics = None, {}
    if action_contrastive_coeff > 0 and outputs.get("z_pred_negative") is not None:
        ac_loss, ac_diagnostics = action_contrastive_loss(
            outputs["z_pred"], outputs["z_pred_negative"], outputs["z_next"],
            outputs.get("negative_action_valid"),
            float(getattr(args, "action_contrastive_margin", 0.7)),
            str(getattr(args, "action_contrastive_loss_type", "softplus")),
            float(getattr(args, "action_contrastive_temperature", 0.1)),
        )
    action_sigreg_coeff = float(getattr(args, "action_sigreg_coeff", 0.0) or 0.0)
    action_sigreg_loss = (
        sigreg_loss(outputs["z_action"], args.sigreg_projections, args.sigreg_eps)
        if action_sigreg_coeff > 0 and "z_action" in outputs
        else None
    )
    if getattr(args, "train_success_head_only", False):
        total = args.success_loss_coeff * success_loss
    else:
        total = latent_coeff * latent_loss + args.sigreg_coeff * sig_loss + args.reconstruction_loss_coeff * recon_loss + (args.success_loss_coeff * success_loss if success_enabled else 0.0)
        if obs_ground_coeff > 0 and obs_ground_loss is not None:
            total = total + obs_ground_coeff * obs_ground_loss
        if tool_select_coeff > 0 and tool_select_loss is not None:
            total = total + tool_select_coeff * tool_select_loss
        if action_encoder_coeff > 0 and action_encoder_loss is not None:
            total = total + action_encoder_coeff * action_encoder_loss
        if action_decoder_coeff > 0 and action_decoder_loss is not None:
            total = total + action_decoder_coeff * action_decoder_loss
        if action_sigreg_loss is not None:
            total = total + action_sigreg_coeff * action_sigreg_loss
        if ac_loss is not None:
            total = total + action_contrastive_coeff * ac_loss
        if state_update_loss is not None:
            total = total + float(getattr(args, "state_update_loss_coeff", 0.1)) * state_update_loss
        if consistency_loss is not None:
            total = total + float(getattr(args, "consistency_loss_coeff", 0.1)) * consistency_loss
        if future_event_loss is not None:
            total = total + float(getattr(args, "future_event_loss_coeff", 0.5)) * future_event_loss
        if terminal_enabled:
            total = total + terminal_loss_coeff * terminal_loss
    components = {"latent_loss": latent_loss, "sigreg_loss": sig_loss, "reconstruction_loss": recon_loss}
    if success_enabled:
        components["success_loss"] = success_loss
        components["success_accuracy"] = success_accuracy
    if terminal_enabled:
        components["terminal_loss"] = terminal_loss
        components["terminal_accuracy"] = terminal_accuracy
    if kl_dynamics is not None:
        components["kl_dynamics"] = kl_dynamics
        components["kl_representation"] = kl_representation
    if obs_ground_loss is not None:
        components["obs_ground_loss"] = obs_ground_loss
        components["obs_ground_coverage"] = outputs.get("obs_ground_coverage")
    if tool_select_loss is not None:
        components["tool_select_loss"] = tool_select_loss
        for key in ("tool_top1", "tool_top5", "tool_top10"):
            components[key] = outputs.get(key)
    if action_encoder_loss is not None:
        components["action_encoder_loss"] = action_encoder_loss
    if action_decoder_loss is not None:
        components["action_decoder_loss"] = action_decoder_loss
    if action_sigreg_loss is not None:
        components["action_sigreg_loss"] = action_sigreg_loss
    if ac_loss is not None:
        components["action_contrastive_loss"] = ac_loss
        components.update(ac_diagnostics)
    if consistency_loss is not None:
        components["consistency_loss"] = consistency_loss
    if future_event_loss is not None:
        components["future_event_loss"] = future_event_loss
    if state_update_loss is not None:
        components["state_update_loss"] = state_update_loss
    if decomposed and outputs.get("z_state_pred") is not None:
        # Does U actually use the event, or has it settled on passing s_t through? ~0 means the
        # persistence shortcut reappeared one level down. Logged in BOTH objectives -- it is the
        # failure the future-event objective is meant to prevent, so it matters most there.
        components["state_update_vs_persistence"] = (
            (outputs["z_state_pred"] - outputs["z_current"]).norm(dim=-1).mean().detach()
        )
    # Persistence-normalized improvement: >0 means the predictor beats copying z_t. Always
    # logged (continuous latents only) -- it is the check that the action-contrastive term is
    # fixing the failure it targets, not just lowering its own loss.
    if getattr(args, "latent_type", "continuous") != "categorical":
        components["pni"] = persistence_normalized_improvement(
            outputs["z_pred"], outputs["z_current"], outputs["z_next"]).detach()
    return total, components


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def summarize_examples(examples: list[JepaExample]) -> dict[str, Any]:
    return {
        "examples": len(examples),
        "trajectories": len({example.trajectory_index for example in examples}),
        "avg_observation_chars": (
            sum(len(example.observation_text) for example in examples) / max(len(examples), 1)
        ),
    }


def evaluate(model: TextLeWorldModel, dataloader: DataLoader, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    success_enabled = success_head_training_enabled(args)
    terminal_enabled = float(getattr(args, "terminal_loss_coeff", 0.0) or 0.0) > 0
    totals = {"loss": 0.0, "latent_loss": 0.0, "sigreg_loss": 0.0, "reconstruction_loss": 0.0, "batches": 0}
    if success_enabled:
        totals["success_loss"] = 0.0
        totals["success_accuracy"] = 0.0
    if terminal_enabled:
        totals["terminal_loss"] = 0.0
        totals["terminal_accuracy"] = 0.0
    if args.latent_type == "categorical":
        totals["kl_dynamics"] = 0.0
        totals["kl_representation"] = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch(batch, device)
            outputs = model(
                batch,
                compute_reconstruction=args.reconstruction_loss_coeff > 0,
                compute_success=success_enabled,
                compute_action_decoder=args.action_decoder_loss_coeff > 0,
                compute_terminal=terminal_enabled,
            )
            loss, components = compute_jepa_loss(outputs, args)
            totals["loss"] += float(loss.detach().cpu())
            totals["latent_loss"] += float(components["latent_loss"].detach().cpu())
            totals["sigreg_loss"] += float(components["sigreg_loss"].detach().cpu())
            totals["reconstruction_loss"] += float(components["reconstruction_loss"].detach().cpu())
            if success_enabled:
                totals["success_loss"] += float(components["success_loss"].detach().cpu())
                totals["success_accuracy"] += float(components["success_accuracy"].detach().cpu())
            if terminal_enabled and "terminal_loss" in components:
                totals["terminal_loss"] += float(components["terminal_loss"].detach().cpu())
                totals["terminal_accuracy"] += float(components["terminal_accuracy"].detach().cpu())
            if "kl_dynamics" in components:
                totals["kl_dynamics"] += float(components["kl_dynamics"].detach().cpu())
                totals["kl_representation"] += float(components["kl_representation"].detach().cpu())
            totals["batches"] += 1
    batches = max(totals.pop("batches"), 1)
    return {key: value / batches for key, value in totals.items()}


def jepa_adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in unwrap_model(model).state_dict().items() if not key.startswith("backbone.")}


def jepa_architecture_manifest_fields(args: argparse.Namespace, *, tool_vocab_size: int = 0) -> dict[str, Any]:
    """The subset of jepa_data_manifest.json needed to reconstruct a matching TextLeWorldModel
    downstream -- replay-time loading (finetuning.py's JepaTextWorldModelGenerator) and
    --train-canonical-event-heads-only's base-checkpoint inheritance both read this: core
    architecture dims plus every optional-module flag (obs_grounding/tool_select/
    action_encoder/fast_lewm/action_decoder). Shared by the final end-of-run manifest AND by
    periodic checkpoint saves (see save_checkpoint_and_prune) -- without this, a
    checkpoint-<step> dir from an interrupted run has no manifest at all, so anything built on
    top of it (or replayed from it) silently constructs the model WITHOUT these modules and
    their trained weights get dropped as "unexpected keys" rather than loaded. That is exactly
    what happened to an action_decoder trained for hours before a run stalled and was resumed
    from a periodic checkpoint -- the decoder never even attached, so it always fell back to
    nearest_anchor at replay time with no error anywhere.
    """
    return {
        "backbone_type": args.backbone_type,
        "pooling": args.pooling,
        "latent_type": args.latent_type,
        "latent_dim": args.latent_dim,
        "memory_tokens": args.memory_tokens,
        "predictor_hidden_multiplier": args.predictor_hidden_multiplier,
        "latent_categoricals": args.latent_categoricals,
        "latent_classes": args.latent_classes,
        "latent_unimix": args.latent_unimix,
        # Changes the predictor architecture (no trailing LayerNorm) AND its semantics
        # (output is z_current + Δz) -- replay-time construction must match or the state
        # dict fails to load / silently mispredicts.
        "latent_delta_prediction": bool(getattr(args, "latent_delta_prediction", False)),
        # Data-processing flag, not architecture: state texts were tokenized with LEFT
        # truncation, so replay must tokenize the same way or over-length states are cut on
        # the opposite side from what the encoder saw in training. No state-dict mismatch
        # would flag this -- the manifest is the ONLY carrier.
        "truncate_states_keep_newest": bool(getattr(args, "truncate_states_keep_newest", False)),
        "goal_conditioning": not args.disable_goal_conditioning,
        "max_input_length": args.max_input_length,
        "max_action_length": args.max_action_length,
        "max_observation_length": args.max_observation_length,
        "max_goal_length": args.max_goal_length,
        "obs_grounding": args.obs_token_ground_coeff > 0,
        "obs_ground_decoder_dim": args.obs_ground_decoder_dim,
        "obs_ground_decoder_layers": args.obs_ground_decoder_layers,
        "obs_ground_decoder_heads": args.obs_ground_decoder_heads,
        "obs_ground_decoder_memory_tokens": args.obs_ground_decoder_memory_tokens,
        "obs_ground_decoder_max_length": max(128, (args.obs_ground_max_tokens or 0) + 32),
        "tool_select": args.tool_select_loss_coeff > 0,
        "action_encoder": args.action_encoder_loss_coeff > 0,
        "action_head_embed_dim": args.action_head_embed_dim,
        "tool_vocab_size": tool_vocab_size,
        "fast_lewm": args.fast_lewm,
        "fast_lewm_dim": args.fast_lewm_dim,
        "fast_lewm_layers": args.fast_lewm_layers,
        "fast_lewm_heads": args.fast_lewm_heads,
        "fast_lewm_max_horizon": max(args.prediction_horizon, 8),
        "action_decoder": args.action_decoder_loss_coeff > 0,
        "action_decoder_max_noise_std": args.action_decoder_max_noise_std,
        "action_decoder_dim": args.action_decoder_dim,
        "action_decoder_layers": args.action_decoder_layers,
        "action_decoder_heads": args.action_decoder_heads,
        "action_decoder_memory_tokens": args.action_decoder_memory_tokens,
        "action_decoder_max_length": args.max_action_length,
        # Same tool_vocab (and hence the same string->index mapping saved in tool_vocab.json)
        # as tool_vocab_size above -- kept as its own field since action_decoder can be enabled
        # without tool_select/action_encoder, and replay-time construction reads it as its own
        # named kwarg (action_decoder_tool_vocab_size) on TextLeWorldModel.
        "action_decoder_tool_vocab_size": tool_vocab_size if args.action_decoder_loss_coeff > 0 else 0,
        "terminal_head": args.terminal_loss_coeff > 0,
        "value_head": args.value_loss_coeff > 0,
        # Architecture-affecting (trunk input width) -- MUST round-trip or loading mis-shapes.
        # Base-checkpoint inheritance happens at the construction site, which writes the
        # resolved value back onto args before this runs; `_base_manifest` is not in scope here.
        "canonical_event_head_inputs": str(getattr(args, "canonical_event_head_inputs", "all") or "all"),
        # Data-shaping, not architecture: which text the target encoder consumed.
        "event_target": bool(getattr(args, "event_target", False)),
        # Architecture-affecting: adds the U module, whose weights must have somewhere to load.
        "state_updater": bool(getattr(args, "event_state_decomposition", False)),
        "state_update_loss_coeff": float(getattr(args, "state_update_loss_coeff", 0.1)),
        # Architecture-affecting: U's input width depends on nothing else, but the objective
        # decides whether a next-state target exists at all, so record it for reproducibility.
        "state_updater_objective": str(getattr(args, "state_updater_objective", "future_event")),
        "event_state_recurrent": bool(getattr(args, "event_state_recurrent", False)),
        # Architecture-affecting: adds the I(c) module.
        "recurrent_state_init": bool(getattr(args, "recurrent_state_init", False)),
        # Architecture-affecting: `mlp` and `transformer` are entirely different predictor
        # modules, and the transformer's width/depth/heads/position count all fix tensor shapes.
        "predictor_arch": str(getattr(args, "predictor_arch", "mlp") or "mlp"),
        "predictor_transformer_dim": int(getattr(args, "predictor_transformer_dim", 0) or 0),
        "predictor_transformer_layers": int(getattr(args, "predictor_transformer_layers", 6)),
        "predictor_transformer_heads": int(getattr(args, "predictor_transformer_heads", 16)),
        "predictor_transformer_mlp_ratio": float(getattr(args, "predictor_transformer_mlp_ratio", 4.0)),
        "predictor_history_length": int(getattr(args, "predictor_history_length", 0) or 0),
        # Training-objective provenance (not architecture): records whether this trunk was
        # pretrained action-contrastively, which is the difference the 4-arm ablation turns on.
        "action_contrastive_loss_coeff": float(getattr(args, "action_contrastive_loss_coeff", 0.0) or 0.0),
        "action_contrastive_negatives": int(getattr(args, "action_contrastive_negatives", 0) or 0),
        "action_contrastive_margin": float(getattr(args, "action_contrastive_margin", 0.7)),
        "action_contrastive_loss_type": str(getattr(args, "action_contrastive_loss_type", "softplus")),
        "action_contrastive_temperature": float(getattr(args, "action_contrastive_temperature", 0.1)),
        "action_contrastive_negative_generation_version": 3,
    }


@torch.no_grad()
def calibrate_action_decoder_latent_scale(
    net: Any,
    tokenizer: Any,
    examples: list["JepaExample"],
    device: torch.device,
    max_action_length: int,
    sample_size: int = 512,
    seed: int = 20260101,
) -> float:
    """Empirical std of z_action over a random sample of REAL training actions -- the natural
    'grain size' the backbone's action latent space actually has. --action-decoder-max-noise-std
    is a MULTIPLIER of this (not an absolute number), and the same scalar is saved into
    action_decoder_latent_scale (a checkpoint buffer, so it travels with the weights) for the
    inference-time CEM to calibrate its per-family Gaussian init/min std against -- one shared
    source of truth instead of two independently hand-picked numbers that could easily mismatch.

    Fixed `seed` (not args.seed) so every rank in a torchrun launch draws the identical sample
    from the identical rank-shared example list and computes the identical scalar independently,
    with no extra collective needed to agree on it.
    """
    if not examples:
        return 1.0
    sample = examples if len(examples) <= sample_size else random.Random(seed).sample(examples, sample_size)
    tokens = tokenizer(
        [example.action_text for example in sample],
        return_tensors="pt", padding=True, truncation=True, max_length=max_action_length, add_special_tokens=True,
    )
    was_training = net.training
    net.eval()  # dropout must be off, or the statistic captures dropout noise, not real spread
    try:
        z, _ = net.encode_latent_and_logits(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
    finally:
        net.train(was_training)
    scale = float(z.float().std(dim=0).mean().item())
    return scale if scale > 1e-6 else 1.0


def save_checkpoint_and_prune(
    model: nn.Module,
    tokenizer: Any,
    output_dir: Path,
    global_step: int,
    save_total_limit: int,
    *,
    freeze_backbone: bool,
    base_model: str,
    args: argparse.Namespace | None = None,
    tool_vocab_size: int = 0,
) -> None:
    """Save a periodic checkpoint-<step> dir, then keep only the newest
    `save_total_limit` such dirs (deleting the oldest).

    Rolling by step number, so after saving checkpoint-1500 with a limit of 2 the
    directories are checkpoint-1000 and checkpoint-1500, and checkpoint-500 is
    removed. `save_total_limit <= 0` disables pruning (keep everything).

    jepa_adapter_state_dict drops all backbone.* weights. When the backbone was
    frozen that is safe -- it is byte-identical to `base_model` and can be reloaded
    from there. When the backbone was TRAINED, an adapter-only checkpoint would
    silently lose it, and reloading `base_model` later would pair the trained
    projector/predictor with the wrong feature space. So bundle the trained
    backbone here, and always write jepa_checkpoint_meta.json recording the regime
    so downstream loaders (e.g. --train-canonical-event-heads-only) can tell a
    frozen adapter-only checkpoint from a full-parameter one instead of guessing.

    Also writes jepa_data_manifest.json when `args` is passed (every real call site has it) --
    a checkpoint-<step> dir a run stalled/was interrupted at is otherwise indistinguishable
    from a checkpoint with no optional modules at all, since only the FINAL end-of-run save
    used to write this. Any run built on top of an unfinished one (--jepa-checkpoint-path
    pointing at a checkpoint-<step> dir, or --resume-from-checkpoint) would then silently
    construct the model without action_decoder/obs_grounding/etc. and drop those weights.
    """
    save_dir = output_dir / f"checkpoint-{global_step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(jepa_adapter_state_dict(model), save_dir / "text_leworldmodel.pt")
    tokenizer.save_pretrained(str(save_dir))
    if not freeze_backbone:
        unwrap_model(model).backbone.save_pretrained(str(save_dir / "backbone"))
    dump_json(save_dir / "jepa_checkpoint_meta.json", {
        "freeze_backbone": bool(freeze_backbone),
        "base_model": str(base_model),
        "backbone_included": not freeze_backbone,
        "global_step": int(global_step),
    })
    if args is not None:
        dump_json(save_dir / "jepa_data_manifest.json", jepa_architecture_manifest_fields(args, tool_vocab_size=tool_vocab_size))
    # tool_vocab.json (the {tool_name: idx} string->id mapping the tool_embeddings /
    # action_decoder_tool_embeddings rows were trained against) is dumped once to
    # output_dir before training starts -- copy it into every checkpoint-<step> snapshot too,
    # same reasoning as the manifest fix above: a run built on top of, or replayed from, an
    # interrupted checkpoint-<step> dir must be able to find every companion artifact there,
    # not just in the (possibly nonexistent) final output_dir.
    tool_vocab_path = output_dir / "tool_vocab.json"
    if tool_vocab_path.is_file():
        shutil.copy2(tool_vocab_path, save_dir / "tool_vocab.json")
    prune_old_checkpoints(output_dir, save_total_limit)


def prune_old_checkpoints(output_dir: Path, keep: int) -> None:
    """Delete all but the newest `keep` checkpoint-<step> directories under output_dir."""
    if keep <= 0:
        return
    prefix = "checkpoint-"
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob(f"{prefix}*"):
        if not path.is_dir():
            continue
        suffix = path.name[len(prefix):]
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))
    checkpoints.sort(key=lambda item: item[0])
    for _, path in checkpoints[:-keep]:
        shutil.rmtree(path, ignore_errors=True)


def resume_global_step_from_checkpoint(checkpoint_path: Path) -> int:
    """Recover the optimizer-step count a checkpoint was saved at, so
    --resume-from-checkpoint can continue the deterministic data order and LR
    schedule from that point instead of restarting at step 0. Prefer the recorded
    jepa_checkpoint_meta.json (written by save_checkpoint_and_prune); fall back to
    parsing the checkpoint-<step> directory name. Returns 0 if neither is usable."""
    meta_path = checkpoint_path / "jepa_checkpoint_meta.json"
    if meta_path.is_file():
        try:
            step = json.loads(meta_path.read_text()).get("global_step")
        except (OSError, ValueError):
            step = None
        if isinstance(step, int) and step >= 0:
            return step
    prefix = "checkpoint-"
    name = checkpoint_path.name
    if name.startswith(prefix) and name[len(prefix):].isdigit():
        return int(name[len(prefix):])
    return 0


class ResumableSampler:
    """Wraps a sampler so a one-time prefix of indices can be dropped for the
    resume epoch. This lets --resume-from-checkpoint fast-forward past batches
    that were already trained WITHOUT the DataLoader fetching (and tokenizing)
    them -- JepaTextDataset.__getitem__ tokenizes per sample, so draining skipped
    batches through the loader would re-tokenize hundreds of thousands of samples.
    set_epoch is forwarded so the base sampler's per-epoch shuffle is unchanged;
    the skip applies to exactly one __iter__ and then clears itself."""

    def __init__(self, base: Any) -> None:
        self._base = base
        self._skip = 0

    def set_epoch(self, epoch: int) -> None:
        set_epoch = getattr(self._base, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(epoch)

    def skip_next_epoch(self, num_indices: int) -> None:
        self._skip = max(0, int(num_indices))

    def __iter__(self):
        iterator = iter(self._base)
        for _ in range(self._skip):
            if next(iterator, None) is None:
                break
        self._skip = 0
        return iterator

    def __len__(self) -> int:
        return len(self._base)


def allowed_missing_checkpoint_keys(
    missing_keys: set[str],
    *,
    allow_missing_success_head: bool,
    allow_missing_canonical_event_heads: bool = False,
) -> set[str]:
    allowed = {key for key in missing_keys if key.startswith("backbone.")}
    if allow_missing_success_head:
        allowed.update(key for key in missing_keys if key.startswith("success_head."))
    if allow_missing_canonical_event_heads:
        allowed.update(
            key
            for key in missing_keys
            if key.startswith("canonical_event_heads.") or key.startswith("canonical_event_trunk.")
        )
    return allowed


def load_jepa_state_dict_for_training(
    model: TextLeWorldModel,
    checkpoint_path: Path,
    *,
    allow_missing_success_head: bool,
    allow_missing_canonical_event_heads: bool = False,
) -> None:
    state_dict_path = checkpoint_path / "text_leworldmodel.pt"
    if not state_dict_path.is_file():
        raise SystemExit(f"Missing JEPA state dict: {state_dict_path}")
    try:
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(state_dict_path, map_location="cpu")
    if allow_missing_canonical_event_heads:
        # A head whose class count differs from the checkpoint's (the data-built vocab gained or
        # lost values vs the base run, or bypass mode resized the trunk) would make
        # load_state_dict raise even with strict=False. Since canonical heads are legitimate to
        # (re)initialize in this loading regime, drop the mismatched weights and let that head
        # start fresh -- loudly, so a silent vocab drift is still visible in the logs.
        model_state = model.state_dict()
        shape_skipped = []
        for key in list(state_dict.keys()):
            if (
                key.startswith(("canonical_event_heads.", "canonical_event_trunk."))
                and key in model_state
                and tuple(state_dict[key].shape) != tuple(model_state[key].shape)
            ):
                shape_skipped.append(f"{key}: ckpt{tuple(state_dict[key].shape)} != model{tuple(model_state[key].shape)}")
                del state_dict[key]
        if shape_skipped:
            print(
                "[load_jepa_state_dict] skipped shape-mismatched canonical-head weights "
                "(those heads start freshly initialized): " + "; ".join(shape_skipped),
                flush=True,
            )
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = set(getattr(incompatible, "missing_keys", []))
    unexpected_keys = set(getattr(incompatible, "unexpected_keys", []))
    allowed_missing = allowed_missing_checkpoint_keys(
        missing_keys,
        allow_missing_success_head=allow_missing_success_head,
        allow_missing_canonical_event_heads=allow_missing_canonical_event_heads,
    )
    disallowed_missing = missing_keys - allowed_missing
    # obs_ground_* (Option-C grounding head) exists only in the training model; a
    # checkpoint may carry it while a downstream (e.g. canonical-event head or replay)
    # model omits it, and vice versa -- tolerate it in both directions.
    # Optional heads present only in some models (obs grounding; latent action-head). terminal_head
    # / value_head are newly-trained heads: a base JEPA checkpoint predating them has no such
    # weights, so they MUST be tolerated as missing here or --train-canonical-event-heads-only
    # --value-loss-coeff on any existing checkpoint hard-fails before training starts.
    _optional_prefixes = ("obs_ground_", "tool_embeddings", "tool_query", "action_encoder_mlp", "fast_", "action_decoder", "terminal_head", "value_head")
    allowed_missing |= {key for key in missing_keys if key.startswith(_optional_prefixes)}
    disallowed_missing = missing_keys - allowed_missing
    _unexpected_ok = _optional_prefixes
    if allow_missing_canonical_event_heads:
        # --canonical-event-heads beam_plan builds FEWER heads than an `all` base checkpoint
        # carries; the base's dropped-head weights (and, for a head-less model, the trunk) are
        # intentionally discarded, not an architecture mismatch.
        _unexpected_ok = _optional_prefixes + ("canonical_event_heads.", "canonical_event_trunk.")
    disallowed_unexpected = {key for key in unexpected_keys if not key.startswith(_unexpected_ok)}
    if disallowed_missing or disallowed_unexpected:
        raise SystemExit(f"Checkpoint does not match TextLeWorldModel architecture: missing={sorted(disallowed_missing)}, unexpected={sorted(disallowed_unexpected)}")


def freeze_for_success_head_training(model: TextLeWorldModel) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("success_head.")
    model.deterministic_latent_sampling = True


def freeze_for_canonical_event_head_training(
    model: TextLeWorldModel, *, train_value_head: bool = False, train_terminal_head: bool = False
) -> None:
    """Unfreeze only the heads this run actually trains, on top of a frozen JEPA trunk.

    `train_value_head` (--value-loss-coeff > 0) / `train_terminal_head`
    (--terminal-loss-coeff > 0) additionally unfreeze value_head.* / terminal_head.* -- without
    them that head stays frozen and its loss backpropagates into nothing, silently training a
    randomly-initialized head while the loss curve looks plausible (canonical_event_heads.*
    would still be trainable, so even the "no trainable parameters" guard would not fire).
    """
    trainable_prefixes = ["canonical_event_heads.", "canonical_event_trunk."]
    if train_value_head:
        trainable_prefixes.append("value_head.")
    if train_terminal_head:
        trainable_prefixes.append("terminal_head.")
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(tuple(trainable_prefixes))
    model.deterministic_latent_sampling = True


def success_head_training_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "train_success_head_only", False)) or float(getattr(args, "success_loss_coeff", 0.0) or 0.0) > 0.0


def freeze_success_head(model: TextLeWorldModel) -> None:
    for param in model.success_head.parameters():
        param.requires_grad = False


def run_jepa_replay(args: argparse.Namespace) -> dict[str, Any]:
    from src.finetuning import (
        JepaTextWorldModelGenerator,
        WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
        configure_llm_call_acceleration,
        dump_agent_replay_strategy_records,
        evaluate_agent_replay,
        extract_replay_tasks_from_state_trajectories,
    )

    # The imagined-rollout machinery lives in src.finetuning and is gated by its module
    # globals; apply this run's flags there (rollout mode, single-call step, lockstep
    # batching, API parallelism).
    configure_llm_call_acceleration(args)

    checkpoint_path = args.jepa_checkpoint_path or args.output_dir
    if not (checkpoint_path / "text_leworldmodel.pt").is_file() or not (checkpoint_path / "backbone").is_dir():
        raise SystemExit(
            f"{checkpoint_path} is not a JEPA checkpoint directory; expected text_leworldmodel.pt and backbone/."
        )

    eval_trajectories = load_trajectory_paths(args.eval_data_path, require_enterpriseops_gym=True)
    replay_tasks = extract_replay_tasks_from_state_trajectories(eval_trajectories)
    if not replay_tasks:
        raise SystemExit(f"No replay tasks extracted from {path_list_text(args.eval_data_path)}.")

    world_model_generator = JepaTextWorldModelGenerator(
        checkpoint_path,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        imagined_observation_backend=args.imagined_observation_backend,
        # A canonical-event head checkpoint may lack jepa_data_manifest.json (the
        # head-training run does not always write one), so pass the replay CLI
        # architecture flags as fallbacks the generator uses when the manifest is
        # missing a field. A present manifest always wins per-field.
        arch_defaults={
            "backbone_type": args.backbone_type,
            "pooling": args.pooling,
            "latent_type": args.latent_type,
            "latent_dim": args.latent_dim,
            "memory_tokens": args.memory_tokens,
            "predictor_hidden_multiplier": args.predictor_hidden_multiplier,
            "latent_categoricals": args.latent_categoricals,
            "latent_classes": args.latent_classes,
            "latent_unimix": args.latent_unimix,
            "goal_conditioning": not args.disable_goal_conditioning,
            "max_input_length": args.max_input_length,
            "max_action_length": args.max_action_length,
            "max_observation_length": args.max_observation_length,
            "max_goal_length": args.max_goal_length,
            "canonical_event_head_hidden_size": args.canonical_event_head_hidden_size,
        },
    )
    agent_generator = build_agent_generator(
        args.agent_model,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        disable_chat_template=args.disable_chat_template,
        attn_implementation=args.attn_implementation,
        device_map=args.inference_device_map,
        draft_model_path=args.agent_draft_model,
        prompt_lookup_num_tokens=args.prompt_lookup_tokens,
    )

    replay_eval = evaluate_agent_replay(
        agent_generator,
        world_model_generator,
        replay_tasks,
        max_tasks=args.max_agent_tasks,
        max_steps=args.agent_max_steps,
        mcp_config_path=None,
        internal_thinking_max_iterations=args.internal_thinking_max_iters,
        imagined_trajectory_max_steps=args.imagined_trajectory_max_steps,
        imagined_trajectory_rollouts=args.imagined_trajectory_rollouts,
        imagined_rollout_temperature=args.imagined_rollout_temperature,
        imagined_trajectory_selection_strategy=args.imagined_trajectory_selection_strategy,
        imagined_trajectory_observation_source=args.imagined_trajectory_observation_source,
        final_answer_f1_threshold=args.final_answer_f1_threshold,
        world_model_target=WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
        include_error_message_in_target=False,
        include_stage_in_target=False,
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
        system_prompt_max_chars=0,
        action_max_chars=0,
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
    metrics = {
        "mode": "replay",
        "jepa_checkpoint_path": str(checkpoint_path),
        "agent_model": args.agent_model,
        "world_model_target": WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
        "goal_guidance": "jepa_latent_success_classifier_imagined_feedback",
        "imagined_observation_backend": args.imagined_observation_backend,
        "imagined_observation_backend_resolved": getattr(
            world_model_generator, "imagined_observation_backend_resolved", args.imagined_observation_backend
        ),
        "replay_modes": list(args.replay_modes),
        "imagined_rollout_mode": args.imagined_rollout_mode,
        "beam_plan_trigger": args.beam_plan_trigger,
        "sample_temperature_ladder": bool(args.sample_temperature_ladder),
        "sample_temperature_ladder_max": args.sample_temperature_ladder_max,
        "beam_plan_critic_failure_prob": args.beam_plan_critic_failure_prob,
        "beam_plan_critic_stall_prob": args.beam_plan_critic_stall_prob,
        "beam_plan_critic_min_score": args.beam_plan_critic_min_score,
        "beam_plan_critic_max_quiet_steps": args.beam_plan_critic_max_quiet_steps,
        "beam_plan_terminal_advice": bool(args.beam_plan_terminal_advice),
        "beam_plan_terminal_advice_threshold": args.beam_plan_terminal_advice_threshold,
        "imagined_single_call_step": bool(args.imagined_single_call_step),
        "imagined_parallel_rollouts": bool(args.imagined_parallel_rollouts),
        "llm_batch_parallelism": args.llm_batch_parallelism,
        "agent_draft_model": args.agent_draft_model,
        "prompt_lookup_tokens": args.prompt_lookup_tokens,
        "latent_plan_samples": args.latent_plan_samples,
        "latent_plan_elites": args.latent_plan_elites,
        "latent_plan_iters": args.latent_plan_iters,
        "latent_plan_horizon": args.latent_plan_horizon,
        "latent_mpc_execute_steps": args.latent_mpc_execute_steps,
        "latent_plan_temperature": args.latent_plan_temperature,
        "latent_plan_score_margin": args.latent_plan_score_margin,
        "latent_plan_diversity_multiplier": args.latent_plan_diversity_multiplier,
        "latent_plan_hard_override": args.latent_plan_hard_override,
        "latent_plan_goal_mode": args.latent_plan_goal_mode,
        "gate_flat_score_ratio": args.gate_flat_score_ratio,
        "hier_cem_anchors": args.hier_cem_anchors,
        "hier_cem_samples": args.hier_cem_samples,
        "hier_cem_elites": args.hier_cem_elites,
        "hier_cem_iters": args.hier_cem_iters,
        "hier_cem_horizon": args.hier_cem_horizon,
        "hier_cem_init_std": args.hier_cem_init_std,
        "hier_cem_min_std": args.hier_cem_min_std,
        "hier_cem_smoothing": args.hier_cem_smoothing,
        "hier_cem_min_elite_agreement": args.hier_cem_min_elite_agreement,
        "hier_cem_decode_strategy": args.hier_cem_decode_strategy,
        "hier_cem_decode_max_new_tokens": args.hier_cem_decode_max_new_tokens,
        "trajectory_dataset": args.trajectory_dataset,
        "eval_data_paths": [str(path) for path in args.eval_data_path],
        "gym_task_configs": str(args.gym_task_configs) if args.gym_task_configs else None,
        "gym_task_split_manifest": str(args.gym_task_split_manifest) if args.gym_task_split_manifest else None,
        "agent_replay_eval": replay_eval,
    }
    dump_json(args.output_dir / f"jepa_replay_metrics_{args.agent_model}.json", metrics)
    dump_agent_replay_strategy_records(args.output_dir, replay_eval)
    return metrics


def evaluate_canonical_event_heads(
    model: nn.Module,
    eval_loader: DataLoader,
    vocab: dict[str, list[str]],
    device: torch.device,
    *,
    compute_classification: bool = True,
    compute_value: bool = False,
    compute_terminal: bool = False,
    compute_latent: bool = False,
    latent_loss_type: str = "smooth_l1_cosine",
    smooth_l1_beta: float = 1.0,
    latent_delta_prediction: bool = False,
    dump_predictions_path: Path | None = None,
) -> dict[str, Any]:
    """Per-head classification metrics on the canonical-event eval split.

    For every canonical_event_state/nudge head this returns:

    - ``accuracy``: exact per-example accuracy accumulated from summed correct
      counts rather than a mean of per-batch means, so an uneven final batch no
      longer skews the number. Single-label fields use ``argmax == label``;
      the multi-label field uses exact multi-hot match.
    - ``predicted_distribution`` / ``gold_distribution``: how often each category
      is predicted versus how often it appears in the gold labels (both as counts
      and fractions, keyed by the human-readable vocab value). This surfaces a
      head that has collapsed onto a single majority class -- which can post a
      deceptively high accuracy while never predicting the minority categories.
    - ``per_class``: precision / recall / f1 / support / predicted count per category,
      and ``macro_f1`` / ``macro_precision`` / ``macro_recall`` averaged over the classes
      that actually occur in the gold labels (plus support-weighted variants). Accuracy
      alone hides majority-class drift -- a head can gain accuracy while its minority
      recall falls, which is exactly the regime that matters for downstream gating on
      P(failure)/P(deleted)/P(no-progress). ``classes_missed`` lists categories with gold
      support that the head never predicts.

    The multi-label field additionally reports ``label_accuracy`` (the
    element-wise accuracy the training loop logs) so the two are comparable.

    ``dump_predictions_path`` writes one JSONL row per eval example with its predicted and
    gold label for every field, in dataset order (the eval loader is unshuffled). The stored
    metrics only keep MARGINAL distributions, from which per-class precision/recall/F1 cannot
    be recovered -- those need the joint (gold, predicted) pairs -- so dumping them here makes
    any later metric, confusion matrix or calibration check a file read instead of another
    forward pass over the split.
    """
    model.eval()
    # Derive the evaluated field set from the vocab, which mirrors the heads the model was
    # actually built with (--canonical-event-heads may have dropped some).
    single_fields = [field for field in CANONICAL_EVENT_SINGLE_LABEL_FIELDS if field in vocab]
    multi_fields = [field for field in NUDGE_MULTI_LABEL_FIELDS if field in vocab]
    all_fields = single_fields + multi_fields

    total_examples = 0
    total_loss = 0.0
    batches = 0
    field_loss_sum = {field: 0.0 for field in all_fields}
    correct = {field: 0 for field in all_fields}
    label_correct = {field: 0 for field in multi_fields}  # element-wise, multi-label only
    label_total = {field: 0 for field in multi_fields}
    pred_counts = {field: [0] * len(vocab[field]) for field in all_fields}
    gold_counts = {field: [0] * len(vocab[field]) for field in all_fields}
    # True positives per class: with pred_counts/gold_counts these give per-class precision
    # (tp/predicted) and recall (tp/support) without storing a confusion matrix.
    true_positive_counts = {field: [0] * len(vocab[field]) for field in all_fields}
    value_loss_sum = 0.0
    value_mae_sum = 0.0
    value_batches = 0
    terminal_loss_sum = terminal_acc_sum = 0.0
    terminal_batches = 0
    terminal_total = 0
    terminal_pred_counts = [0, 0]  # nonterminal, terminal
    terminal_gold_counts = [0, 0]
    terminal_true_positive_counts = [0, 0]
    latent_loss_sum = sigreg_loss_sum = 0.0
    latent_batches = 0
    dumped_rows: list[dict[str, Any]] | None = [] if dump_predictions_path is not None else None

    with torch.no_grad():
        for batch in tqdm(
            eval_loader,
            total=len(eval_loader),
            desc="canonical_event_head_eval",
            disable=not is_main_process(),
        ):
            batch = move_batch(batch, device)
            outputs = model(
                batch,
                compute_canonical_event=compute_classification,
                compute_value=compute_value,
                compute_terminal=compute_terminal,
            )
            batches += 1
            total_examples += int(batch["current_input_ids"].shape[0])
            if compute_classification:
                logits = outputs["canonical_event_logits"]
                loss, per_field = canonical_event_classification_loss(logits, batch)
                total_loss += float(loss.detach().cpu())
                for field, values in per_field.items():
                    field_loss_sum[field] += float(values["loss"].detach().cpu())
                batch_rows: list[dict[str, Any]] | None = None
                if dumped_rows is not None:
                    batch_rows = [
                        {"row": len(dumped_rows) + offset, "pred": {}, "gold": {}}
                        for offset in range(int(batch["current_input_ids"].shape[0]))
                    ]
                for field in single_fields:
                    field_logits = logits[field]
                    target = batch[f"label_{field}"].to(field_logits.device)
                    preds = field_logits.argmax(dim=-1)
                    hits = preds == target
                    correct[field] += int(hits.sum().cpu())
                    for value in preds.cpu().tolist():
                        pred_counts[field][value] += 1
                    for value in target.cpu().tolist():
                        gold_counts[field][value] += 1
                    matched = target[hits]
                    if matched.numel():
                        for index, count in enumerate(
                            torch.bincount(matched, minlength=len(vocab[field])).cpu().tolist()
                        ):
                            true_positive_counts[field][index] += int(count)
                    if batch_rows is not None:
                        values = vocab[field]
                        for offset, (pred_index, gold_index) in enumerate(
                            zip(preds.cpu().tolist(), target.cpu().tolist())
                        ):
                            batch_rows[offset]["pred"][field] = values[pred_index]
                            batch_rows[offset]["gold"][field] = values[gold_index]
                for field in multi_fields:
                    field_logits = logits[field]
                    target = batch[f"label_{field}"].to(field_logits.device)
                    preds = (torch.sigmoid(field_logits) >= 0.5).to(target.dtype)
                    correct[field] += int((preds == target).all(dim=-1).sum().cpu())
                    label_correct[field] += int((preds == target).sum().cpu())
                    label_total[field] += int(target.numel())
                    for idx, count in enumerate(preds.sum(dim=0).cpu().tolist()):
                        pred_counts[field][idx] += int(count)
                    for idx, count in enumerate(target.sum(dim=0).cpu().tolist()):
                        gold_counts[field][idx] += int(count)
                    both = ((preds > 0) & (target > 0)).sum(dim=0).cpu().tolist()
                    for idx, count in enumerate(both):
                        true_positive_counts[field][idx] += int(count)
                    if batch_rows is not None:
                        values = vocab[field]
                        pred_rows = preds.cpu().tolist()
                        gold_rows = target.cpu().tolist()
                        for offset in range(len(batch_rows)):
                            batch_rows[offset]["pred"][field] = [
                                values[i] for i, flag in enumerate(pred_rows[offset]) if flag
                            ]
                            batch_rows[offset]["gold"][field] = [
                                values[i] for i, flag in enumerate(gold_rows[offset]) if flag
                            ]
                if dumped_rows is not None and batch_rows is not None:
                    dumped_rows.extend(batch_rows)
            if compute_value:
                value_loss, value_mae = value_regression_loss(outputs)
                value_loss_sum += float(value_loss.detach().cpu())
                value_mae_sum += float(value_mae.detach().cpu())
                value_batches += 1
            if compute_terminal and "terminal_logits" in outputs:
                t_loss, t_acc = terminal_classification_loss(outputs)
                terminal_loss_sum += float(t_loss.detach().cpu())
                terminal_acc_sum += float(t_acc.detach().cpu())
                terminal_batches += 1
                labels = outputs.get("terminal_labels")
                mask = outputs.get("terminal_label_mask")
                if labels is not None and mask is not None:
                    logits = outputs["terminal_logits"].detach().float()
                    labels = labels.to(device=logits.device)
                    mask_bool = mask.to(device=logits.device).to(torch.bool)
                    if bool(mask_bool.any()):
                        preds = (torch.sigmoid(logits) >= 0.5).to(torch.long)[mask_bool]
                        gold = labels.to(torch.long)[mask_bool]
                        terminal_total += int(gold.numel())
                        for value in preds.cpu().tolist():
                            terminal_pred_counts[int(value)] += 1
                        for value in gold.cpu().tolist():
                            terminal_gold_counts[int(value)] += 1
                        hits = preds == gold
                        matched = gold[hits]
                        if matched.numel():
                            for index, count in enumerate(torch.bincount(matched, minlength=2).cpu().tolist()):
                                terminal_true_positive_counts[index] += int(count)
            if compute_latent and "latent_target_mask" in batch:
                mask = batch["latent_target_mask"].to(dtype=outputs["z_pred"].dtype)
                l_loss = masked_latent_prediction_loss(
                    outputs["z_pred"], outputs["z_next"], mask,
                    loss_type=latent_loss_type, beta=smooth_l1_beta,
                    delta_anchor=outputs["z_current"] if latent_delta_prediction else None,
                )
                reps = [outputs["z_current"]]
                valid = mask.to(torch.bool)
                if bool(valid.any()):
                    reps.append(outputs["z_next"][valid])
                s_loss = sigreg_loss(torch.cat(reps, dim=0), 64, 1e-6)
                latent_loss_sum += float(l_loss.detach().cpu())
                sigreg_loss_sum += float(s_loss.detach().cpu())
                latent_batches += 1

    def distribution(counts: list[int], values: list[str], denom: int) -> dict[str, dict[str, float]]:
        return {
            values[i]: {"count": counts[i], "fraction": (counts[i] / denom if denom else 0.0)}
            for i in range(len(values))
        }

    if dumped_rows is not None and dump_predictions_path is not None:
        dump_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_predictions_path.open("w", encoding="utf-8") as handle:
            for row in dumped_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[canonical_event_head_eval] wrote {len(dumped_rows)} per-example predictions to "
              f"{dump_predictions_path}", flush=True)

    def class_metrics(field: str) -> dict[str, Any]:
        """Per-class precision/recall/F1 plus macro and support-weighted averages.

        The macro averages cover only categories with gold support: a vocab entry that never
        occurs in the eval split would otherwise contribute a hard 0 and make the number a
        function of vocabulary size rather than of head quality. `classes_missed` names the
        supported categories the head never predicts -- the collapse signal that accuracy hides.
        """
        values = vocab[field]
        per_class: dict[str, dict[str, float]] = {}
        f1s: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        weighted_f1 = 0.0
        support_total = 0
        missed: list[str] = []
        for index, value in enumerate(values):
            tp = true_positive_counts[field][index]
            predicted = pred_counts[field][index]
            support = gold_counts[field][index]
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            per_class[value] = {
                "precision": precision, "recall": recall, "f1": f1,
                "support": support, "predicted": predicted,
            }
            if support:
                f1s.append(f1)
                precisions.append(precision)
                recalls.append(recall)
                weighted_f1 += f1 * support
                support_total += support
                if predicted == 0:
                    missed.append(value)
        return {
            "per_class": per_class,
            "macro_f1": (sum(f1s) / len(f1s)) if f1s else 0.0,
            "macro_precision": (sum(precisions) / len(precisions)) if precisions else 0.0,
            "macro_recall": (sum(recalls) / len(recalls)) if recalls else 0.0,
            "weighted_f1": (weighted_f1 / support_total) if support_total else 0.0,
            "classes_with_support": len(f1s),
            "classes_missed": missed,
        }

    def binary_terminal_metrics() -> dict[str, Any]:
        values = ["nonterminal", "terminal"]
        per_class: dict[str, dict[str, float]] = {}
        f1s: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        weighted_f1 = 0.0
        support_total = 0
        missed: list[str] = []
        for index, value in enumerate(values):
            tp = terminal_true_positive_counts[index]
            predicted = terminal_pred_counts[index]
            support = terminal_gold_counts[index]
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            per_class[value] = {
                "precision": precision, "recall": recall, "f1": f1,
                "support": support, "predicted": predicted,
            }
            if support:
                f1s.append(f1)
                precisions.append(precision)
                recalls.append(recall)
                weighted_f1 += f1 * support
                support_total += support
                if predicted == 0:
                    missed.append(value)
        return {
            "per_class": per_class,
            "macro_f1": (sum(f1s) / len(f1s)) if f1s else 0.0,
            "macro_precision": (sum(precisions) / len(precisions)) if precisions else 0.0,
            "macro_recall": (sum(recalls) / len(recalls)) if recalls else 0.0,
            "weighted_f1": (weighted_f1 / support_total) if support_total else 0.0,
            "classes_with_support": len(f1s),
            "classes_missed": missed,
        }

    denom = max(total_examples, 1)
    per_head: dict[str, Any] = {}
    if compute_classification:
        for field in all_fields:
            entry: dict[str, Any] = {
                "accuracy": correct[field] / denom,
                "loss": field_loss_sum[field] / max(batches, 1),
                "num_classes": len(vocab[field]),
                "predicted_distribution": distribution(pred_counts[field], vocab[field], total_examples),
                "gold_distribution": distribution(gold_counts[field], vocab[field], total_examples),
                **class_metrics(field),
            }
            if field in multi_fields:
                entry["accuracy_is_exact_match"] = True
                entry["label_accuracy"] = (label_correct[field] / label_total[field]) if label_total[field] else 0.0
                # Element-wise micro-F1 over all (example, class) slots -- the multi-label
                # counterpart to accuracy, unaffected by the exact-match requirement.
                tp_total = sum(true_positive_counts[field])
                predicted_total = sum(pred_counts[field])
                support_sum = sum(gold_counts[field])
                micro_precision = tp_total / predicted_total if predicted_total else 0.0
                micro_recall = tp_total / support_sum if support_sum else 0.0
                entry["micro_f1"] = (
                    2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                    if (micro_precision + micro_recall) else 0.0
                )
            per_head[field] = entry
    terminal_entry: dict[str, Any] | None = None
    terminal_accuracy_value = (
        (sum(terminal_true_positive_counts) / terminal_total)
        if terminal_total else (terminal_acc_sum / max(terminal_batches, 1))
    )
    if terminal_batches:
        terminal_entry = {
            "accuracy": terminal_accuracy_value,
            "loss": terminal_loss_sum / max(terminal_batches, 1),
            "num_classes": 2,
            "predicted_distribution": distribution(terminal_pred_counts, ["nonterminal", "terminal"], terminal_total),
            "gold_distribution": distribution(terminal_gold_counts, ["nonterminal", "terminal"], terminal_total),
            **binary_terminal_metrics(),
        }
        per_head["terminal"] = terminal_entry
    metric_fields = all_fields + (["terminal"] if terminal_entry is not None else [])

    return {
        "eval_examples": total_examples,
        **({
            "loss": total_loss / max(batches, 1),
            "accuracy": {field: per_head[field]["accuracy"] for field in metric_fields},
            # Compact cross-run comparison maps: accuracy can rise while macro_f1/macro_recall
            # fall (drift toward the majority class), so both are surfaced at the top level.
            "macro_f1": {field: per_head[field]["macro_f1"] for field in metric_fields},
            "macro_recall": {field: per_head[field]["macro_recall"] for field in metric_fields},
            "per_head": per_head,
        } if compute_classification else {}),
        **({
            "value_loss": value_loss_sum / max(value_batches, 1),
            "value_mae": value_mae_sum / max(value_batches, 1),
        } if compute_value else {}),
        **({
            "terminal_loss": terminal_loss_sum / max(terminal_batches, 1),
            "terminal_accuracy": terminal_accuracy_value,
            "terminal_macro_f1": terminal_entry["macro_f1"] if terminal_entry else 0.0,
            "terminal_macro_recall": terminal_entry["macro_recall"] if terminal_entry else 0.0,
            "terminal_macro_precision": terminal_entry["macro_precision"] if terminal_entry else 0.0,
            "terminal_per_head": terminal_entry or {},
        } if terminal_batches else {}),
        **({
            "latent_loss": latent_loss_sum / max(latent_batches, 1),
            "sigreg_loss": sigreg_loss_sum / max(latent_batches, 1),
        } if latent_batches else {}),
    }


def evaluate_canonical_event_heads_by_benchmark(
    model: nn.Module,
    eval_examples: list[CanonicalEventExample],
    tokenizer: Any,
    args: argparse.Namespace,
    vocab: dict[str, list[str]],
    device: torch.device,
) -> dict[str, Any]:
    """Per-benchmark, per-field accuracy and macro-F1 on the canonical-event eval split.

    Groups the eval examples by their ``benchmark`` field and runs the same
    per-head evaluation on each group, so accuracy can be read per field per
    benchmark (e.g. CRMArenaPro vs EnterpriseOps-Gym vs Terminal-Bench-2.0).
    Each entry keeps the compact accuracy / macro_f1 / macro_recall maps (plus loss and
    count); the per-class breakdown and category distributions stay in the overall
    per_head block.
    """
    groups: dict[str, list[CanonicalEventExample]] = {}
    for example in eval_examples:
        groups.setdefault(example.benchmark or "unknown", []).append(example)
    collator = CanonicalEventCollator(tokenizer)
    per_benchmark: dict[str, Any] = {}
    for benchmark in sorted(groups):
        loader = DataLoader(
            CanonicalEventDataset(groups[benchmark], tokenizer, args, vocab),
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        metrics = evaluate_canonical_event_heads(model, loader, vocab, device)
        per_benchmark[benchmark] = {
            "eval_examples": metrics["eval_examples"],
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_recall": metrics["macro_recall"],
        }
    return per_benchmark


def print_canonical_event_eval_summary(eval_metrics: dict[str, Any]) -> None:
    """Human-readable console dump of per-head accuracy and category spread."""
    for key, label in (("value_loss", "value"), ("terminal_loss", "terminal"), ("latent_loss", "latent")):
        if key not in eval_metrics:
            continue
        extra = {
            "value": f"value_mae={eval_metrics.get('value_mae', 0.0):.4f}",
            "terminal": (
                f"terminal_accuracy={eval_metrics.get('terminal_accuracy', 0.0):.4f} "
                f"terminal_macro_f1={eval_metrics.get('terminal_macro_f1', 0.0):.4f} "
                f"terminal_macro_recall={eval_metrics.get('terminal_macro_recall', 0.0):.4f}"
            ),
            "latent": f"sigreg_loss={eval_metrics.get('sigreg_loss', 0.0):.4f}",
        }[label]
        print(
            f"[canonical_event_head_eval] {key}={eval_metrics[key]:.4f} {extra} "
            f"examples={eval_metrics.get('eval_examples')}",
            flush=True,
        )
    per_head = eval_metrics.get("per_head") or {}
    if not per_head:
        return
    print(
        f"[canonical_event_head_eval] loss={eval_metrics.get('loss'):.4f} "
        f"examples={eval_metrics.get('eval_examples')}",
        flush=True,
    )
    for field, entry in per_head.items():
        acc_label = "exact-match acc" if entry.get("accuracy_is_exact_match") else "accuracy"
        line = (
            f"  {field:32s} {acc_label}={entry['accuracy']:.4f} "
            f"macro_f1={entry.get('macro_f1', 0.0):.4f} macro_recall={entry.get('macro_recall', 0.0):.4f} "
            f"(classes={entry.get('classes_with_support', entry['num_classes'])}"
            f"/{entry['num_classes']} with support)"
        )
        if "label_accuracy" in entry:
            line += f" label_acc={entry['label_accuracy']:.4f}"
        if "micro_f1" in entry:
            line += f" micro_f1={entry['micro_f1']:.4f}"
        print(line, flush=True)
        missed = entry.get("classes_missed") or []
        if missed:
            print(f"      NEVER PREDICTED (has gold support): {', '.join(missed)}", flush=True)
        # Per-class precision/recall next to the predicted-vs-gold share: a class whose share
        # is held up by a low-precision flood reads very differently from a well-fit one.
        gold = entry.get("gold_distribution", {})
        pred = entry.get("predicted_distribution", {})
        per_class = entry.get("per_class", {})
        top = sorted(gold.items(), key=lambda kv: kv[1]["count"], reverse=True)[:5]
        for value, stats in top:
            klass = per_class.get(value, {})
            print(
                f"      {value:28s} gold={stats['fraction']:.3f} pred={pred.get(value, {}).get('fraction', 0.0):.3f}"
                f"  P={klass.get('precision', 0.0):.3f} R={klass.get('recall', 0.0):.3f}"
                f" F1={klass.get('f1', 0.0):.3f} (n={klass.get('support', 0)})",
                flush=True,
            )
    per_benchmark = eval_metrics.get("per_benchmark") or {}
    for benchmark, entry in per_benchmark.items():
        print(
            f"  [benchmark: {benchmark}] examples={entry.get('eval_examples')} loss={entry.get('loss'):.4f}",
            flush=True,
        )
        macro_f1_map = entry.get("macro_f1") or {}
        macro_recall_map = entry.get("macro_recall") or {}
        for field, accuracy in (entry.get("accuracy") or {}).items():
            print(
                f"      {field:32s} accuracy={accuracy:.4f} "
                f"macro_f1={macro_f1_map.get(field, 0.0):.4f} "
                f"macro_recall={macro_recall_map.get(field, 0.0):.4f}",
                flush=True,
            )


def run_canonical_event_head_training(args: argparse.Namespace, distributed: bool, local_rank: int) -> dict[str, Any]:
    """--train-canonical-event-heads-only: fit new classification heads on top of
    a frozen, already-trained JEPA checkpoint using the canonical_event_state /
    nudge JSONL label files -- independent of the trajectory-based JEPA data
    pipeline (extract_jepa_examples()/JepaTextDataset), since these labels come
    with their own context/action/history fields per row.

    Mirrors main()'s generic training loop's distributed setup (per-rank device,
    DistributedSampler, DDP with find_unused_parameters=True since only
    canonical_event_heads.* ever receives a gradient here) -- a single backbone
    the size of Qwen3-Embedding-8B replicated onto every rank's device is
    already ~16GB in bf16, so getting per-rank placement wrong reliably OOMs
    under torchrun --nproc_per_node>1.
    """
    from transformers import get_cosine_schedule_with_warmup

    if (
        args.canonical_event_classification_loss_coeff <= 0
        and args.value_loss_coeff <= 0
        and args.terminal_loss_coeff <= 0
        and not args.joint_canonical_event_training
    ):
        raise SystemExit(
            "--train-canonical-event-heads-only has nothing to train: "
            "--canonical-event-classification-loss-coeff, --value-loss-coeff and "
            "--terminal-loss-coeff are all 0 (and --joint-canonical-event-training is off, so "
            "there is no latent loss either)."
        )
    init_checkpoint = args.jepa_checkpoint_path or args.output_dir
    if not (init_checkpoint / "text_leworldmodel.pt").is_file():
        raise SystemExit(
            f"{init_checkpoint} is not a JEPA checkpoint directory; expected text_leworldmodel.pt. "
            "--train-canonical-event-heads-only requires an existing --jepa-checkpoint-path."
        )
    # Choose the backbone. jepa_adapter_state_dict drops backbone.* weights, so a
    # periodic checkpoint-<step> dir is adapter-only. The projector/predictor in
    # text_leworldmodel.pt were fit against a SPECIFIC backbone; loading a different
    # one silently mismatches the feature space and the heads train on garbage.
    #   * backbone/ present            -> use it (self-contained checkpoint).
    #   * meta says frozen backbone    -> identical to base_model, safe to reload.
    #   * meta says full-parameter     -> trained backbone was never saved here and
    #                                     cannot be recovered; refuse rather than
    #                                     silently substitute the base backbone.
    #   * no meta (pre-fix checkpoint) -> can't prove the regime; fall back to base
    #                                     if the head run froze the backbone, but WARN
    #                                     loudly that this is only correct for a frozen trunk.
    # NB: the head run's own --freeze-backbone says nothing about how the TRUNK was
    # trained -- do not use it to infer that base_model matches the trunk's backbone.
    meta_path = init_checkpoint / "jepa_checkpoint_meta.json"
    meta = None
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            meta = None
    if (init_checkpoint / "backbone").is_dir():
        backbone_source = init_checkpoint / "backbone"
    elif meta is not None and not meta.get("freeze_backbone", False):
        raise SystemExit(
            f"{init_checkpoint} is an adapter-only checkpoint from a FULL-PARAMETER JEPA run "
            "(jepa_checkpoint_meta.json: freeze_backbone=false) with no bundled backbone/. Its trained "
            "backbone was never saved and cannot be recovered; loading the base --model would silently "
            "mismatch the projector/predictor. Point --jepa-checkpoint-path at a completed run directory "
            "that contains backbone/, or re-run the trunk (periodic checkpoints of full-parameter runs now "
            "bundle backbone/)."
        )
    elif args.freeze_backbone:
        backbone_source = (meta or {}).get("base_model") or args.model
        if is_main_process():
            warn = "" if meta is not None else (
                " WARNING: no jepa_checkpoint_meta.json -- this is only correct if the trunk was trained "
                "with --freeze-backbone; a full-parameter trunk would silently mismatch the backbone."
            )
            print(
                f"[canonical_event_head_train] {init_checkpoint} has no backbone/ (adapter-only checkpoint); "
                f"loading frozen backbone from base --model {backbone_source}.{warn}",
                flush=True,
            )
    else:
        raise SystemExit(
            f"{init_checkpoint} has no backbone/ and --freeze-backbone was not set; the mid-training backbone "
            "cannot be recovered from an adapter-only checkpoint. Re-run with --freeze-backbone (if the original "
            "JEPA run froze the backbone) or point --jepa-checkpoint-path at a full run directory."
        )

    tokenizer = load_text_tokenizer(init_checkpoint, trust_remote_code=args.trust_remote_code)
    backbone = load_jepa_backbone(
        backbone_source, backbone_type=args.backbone_type, trust_remote_code=args.trust_remote_code, dtype=resolve_torch_dtype(args.dtype)
    )
    if args.gradient_checkpointing:
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
        else:
            raise SystemExit(f"Backbone {args.model} does not support gradient checkpointing.")
        if hasattr(backbone, "config") and hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False
    if args.freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    train_examples = build_canonical_event_examples(load_canonical_event_records(args.canonical_event_train_jsonl))
    eval_examples = (
        []
        if args.skip_eval
        else build_canonical_event_examples(load_canonical_event_records(args.canonical_event_eval_jsonl))
    )
    if not train_examples:
        raise SystemExit(f"No usable canonical-event examples in {args.canonical_event_train_jsonl}.")
    if args.value_loss_coeff > 0 and is_main_process():
        with_value = sum(1 for ex in train_examples if ex.value_target is not None)
        coverage = with_value / max(len(train_examples), 1)
        print(
            f"[canonical_event_head_train] value head: {with_value}/{len(train_examples)} "
            f"train examples carry a value_target ({coverage:.1%} coverage)."
            + ("" if with_value else " WARNING: 0% -- pass the *_value_scored.jsonl produced by "
               "src/data_preparation/annotate_step_value_scores.py, not the plain "
               "canonical_event_with_nudge JSONL, or the value loss trains on nothing."),
            flush=True,
        )
    if args.canonical_event_recognition_probe and is_main_process():
        with_obs = sum(1 for ex in train_examples if ex.observation_text)
        coverage = with_obs / max(len(train_examples), 1)
        print(
            f"[canonical_event_head_train] recognition probe: {with_obs}/{len(train_examples)} "
            f"train examples carry an observation ({coverage:.1%} coverage)."
            + ("" if with_obs else " WARNING: 0% -- rows have no observation field; the probe is inert. "
               "Regenerate the JSONL with a tool_output/observation field to use it."),
            flush=True,
        )
    trained_single_fields, trained_multi_fields = resolve_canonical_event_field_sets(
        getattr(args, "canonical_event_heads", "all")
    )
    vocab = build_canonical_event_vocabularies(
        train_examples, eval_examples, fields=trained_single_fields + trained_multi_fields
    )
    # Inherit the BASE checkpoint's per-field value vocabulary whenever this run's data uses a
    # subset of it. The vocab defines BOTH each head's output size and the value->index mapping;
    # rebuilding it from a filtered JSONL (e.g. the recognition-probe files, which drop terminal
    # rows) can silently lose rare values -- 'rollback' occurs only on terminal steps -- which
    # (1) shrinks the head and crashes the warm-start load with a size mismatch, and (2) even
    # when sizes happen to match, permutes the index of every value sorted after the missing one,
    # so warm-started head weights would score the wrong classes. A field whose data contains
    # values the base has never seen keeps the data-built vocab; that head is then freshly
    # initialized (see the shape-mismatch tolerance in load_jepa_state_dict_for_training).
    _base_vocab_path = init_checkpoint / "canonical_event_vocab.json"
    if _base_vocab_path.is_file():
        _base_vocab = json.loads(_base_vocab_path.read_text())
        _inherited_fields, _kept_fields = [], []
        for _field, _data_values in vocab.items():
            _base_values = _base_vocab.get(_field)
            if isinstance(_base_values, list) and set(_data_values) <= set(_base_values):
                vocab[_field] = list(_base_values)
                _inherited_fields.append(_field)
            else:
                _kept_fields.append(_field)
        if is_main_process() and _inherited_fields:
            print(
                f"[canonical_event_head_train] inherited value vocab from {_base_vocab_path} "
                f"for {len(_inherited_fields)}/{len(vocab)} fields"
                + (f"; data-built vocab kept for {_kept_fields} (those heads start fresh)" if _kept_fields else ""),
                flush=True,
            )
    vocab_sizes = {field: len(values) for field, values in vocab.items()}
    if is_main_process() and getattr(args, "canonical_event_heads", "all") != "all":
        dropped = [f for f in CANONICAL_EVENT_ALL_FIELDS if f not in vocab]
        print(
            f"[canonical_event_head_train] --canonical-event-heads={args.canonical_event_heads}: "
            f"training {len(vocab)} heads {sorted(vocab)}; dropped {dropped}",
            flush=True,
        )

    # Few-shot: keep only a percentage of TRAIN examples per benchmark. Done after
    # the vocabulary is built from the full split above, so head sizes stay
    # identical across percentages and few-shot runs are directly comparable.
    train_benchmark_counts_full = canonical_event_benchmark_counts(train_examples)
    if args.canonical_event_train_sample_percentage < 100.0:
        train_examples = subsample_canonical_event_examples_by_benchmark(
            train_examples,
            args.canonical_event_train_sample_percentage,
            seed=args.canonical_event_train_sample_seed,
        )
    train_benchmark_counts_kept = canonical_event_benchmark_counts(train_examples)
    if is_main_process() and args.canonical_event_train_sample_percentage < 100.0:
        print(
            "[canonical_event_head_train] few-shot "
            f"{args.canonical_event_train_sample_percentage}% per benchmark "
            f"(sample-seed={args.canonical_event_train_sample_seed}): kept "
            f"{sum(train_benchmark_counts_kept.values())}/{sum(train_benchmark_counts_full.values())} train examples "
            f"{train_benchmark_counts_kept}",
            flush=True,
        )

    if args.max_train_examples > 0:
        random.Random(args.seed).shuffle(train_examples)
        train_examples = train_examples[: args.max_train_examples]
    if args.max_eval_examples > 0:
        random.Random(args.seed + 1).shuffle(eval_examples)
        eval_examples = eval_examples[: args.max_eval_examples]

    # Read the BASE checkpoint's optional-module flags so this model is built with the SAME
    # module set before load_jepa_state_dict_for_training below -- otherwise a module the base
    # checkpoint trained (obs_grounding/tool_select/action_encoder/fast_lewm/action_decoder)
    # has nowhere to load into here, and its weights are silently dropped from the
    # canonical-event-head checkpoint this run produces (the _optional_prefixes tolerance in
    # load_jepa_state_dict_for_training only means the mismatch doesn't crash -- it does NOT
    # mean the weights survive).
    _base_manifest_path = init_checkpoint / "jepa_data_manifest.json"
    _base_manifest = json.loads(_base_manifest_path.read_text()) if _base_manifest_path.is_file() else {}
    _optional_head_keys = (
        "obs_grounding", "obs_ground_decoder_dim", "obs_ground_decoder_layers", "obs_ground_decoder_heads",
        "obs_ground_decoder_memory_tokens", "obs_ground_decoder_max_length",
        "tool_select", "action_encoder", "action_head_embed_dim", "tool_vocab_size",
        "fast_lewm", "fast_lewm_dim", "fast_lewm_layers", "fast_lewm_heads", "fast_lewm_max_horizon",
        "action_decoder", "action_decoder_max_noise_std", "action_decoder_dim", "action_decoder_layers",
        "action_decoder_heads", "action_decoder_memory_tokens", "action_decoder_max_length",
        "action_decoder_tool_vocab_size",
        # Not an optional MODULE, but it changes the predictor's architecture (LayerNorm
        # dropped) and semantics (residual Δz) -- must survive into the head checkpoint's
        # manifest or downstream replay constructs a mismatched predictor.
        "latent_delta_prediction",
        "terminal_head",
        "value_head",
        # Data-processing flag: the trunk's latents were learned on left-truncated state texts,
        # so head training/replay on top must tokenize identically.
        "truncate_states_keep_newest",
        # Predictor architecture. `mlp` and `transformer` share NO parameter names, and the
        # transformer's width/depth/heads/position count fix every tensor shape -- omitting
        # these makes a head checkpoint's manifest claim an MLP predictor, so anything
        # rebuilding from it drops the whole trained predictor as unexpected keys.
        "predictor_arch", "predictor_transformer_dim", "predictor_transformer_layers",
        "predictor_transformer_heads", "predictor_transformer_mlp_ratio",
        "predictor_history_length",
        # Fixes the canonical-event trunk's input width (and, for 'state', which tensor the
        # heads read at all).
        "canonical_event_head_inputs",
        # Optional modules on the event/state decomposition path.
        "state_updater", "state_updater_objective", "recurrent_state_init",
        # Data-shaping: which text the target encoder consumed.
        "event_target",
    )
    _inherited_flags = {key: _base_manifest[key] for key in _optional_head_keys if key in _base_manifest}
    # truncate_states_keep_newest steers the DATASET (tokenization side), not model
    # construction, so inheriting it into the manifest alone is not enough -- this run's own
    # tokenization must match the trunk's training-time tokenization too. OR semantics: honor
    # the base checkpoint's regime, or the user's explicit flag for a joint run that trains
    # the trunk further under the new regime.
    if bool(_base_manifest.get("truncate_states_keep_newest", False)):
        if is_main_process() and not args.truncate_states_keep_newest:
            print(
                "[canonical_event_head_train] base checkpoint was trained with "
                "--truncate-states-keep-newest; enabling it for this run's tokenization to match.",
                flush=True,
            )
        args.truncate_states_keep_newest = True
    # tool_vocab.json (the {tool_name: idx} string->id mapping) is not part of the manifest --
    # inheriting action_decoder_tool_vocab_size/tool_vocab_size above sizes the embedding
    # tables correctly, but without this file downstream replay has no way to map a family/
    # tool-name string back to the row those tables were trained against, and silently
    # degrades to "unknown tool" (index 0) for everything. This head-training stage doesn't
    # rebuild tool_vocab from its own examples (the action-head modules are frozen, not
    # retrained), so carry the base checkpoint's copy forward verbatim.
    _base_tool_vocab_path = init_checkpoint / "tool_vocab.json"
    if is_main_process() and _base_tool_vocab_path.is_file():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_base_tool_vocab_path, args.output_dir / "tool_vocab.json")

    def _canonical_event_head_manifest() -> dict[str, Any]:
        # Shared by periodic (mid-training) and final checkpoint saves, so a checkpoint-<step>
        # dir this run stalls/gets interrupted at is just as safe to build on top of (or
        # replay from) as a normally-completed run -- see jepa_architecture_manifest_fields'
        # docstring for the incident this class of fix addresses.
        return {
            "backbone_type": args.backbone_type,
            "pooling": args.pooling,
            "latent_type": args.latent_type,
            "latent_dim": args.latent_dim,
            "memory_tokens": args.memory_tokens,
            "predictor_hidden_multiplier": args.predictor_hidden_multiplier,
            "latent_categoricals": args.latent_categoricals,
            "latent_classes": args.latent_classes,
            "latent_unimix": args.latent_unimix,
            "goal_conditioning": not args.disable_goal_conditioning,
            "max_input_length": args.max_input_length,
            "max_action_length": args.max_action_length,
            "max_observation_length": args.max_observation_length,
            "max_goal_length": args.max_goal_length,
            "canonical_event_head_hidden_size": args.canonical_event_head_hidden_size,
            "canonical_event_vocab_sizes": vocab_sizes,
            **_inherited_flags,
            # _inherited_flags only carries a key when the BASE checkpoint's manifest already
            # had it -- a value/terminal head trained for the FIRST time in this
            # --train-canonical-event-heads-only run (base checkpoint predates it) would
            # otherwise be silently absent from this run's own manifest, even though the model
            # was correctly constructed with the head attached (see the TextLeWorldModel(...)
            # call below) -- exactly the "trained for hours, manifest doesn't say so, replay
            # drops the weights" incident jepa_architecture_manifest_fields' docstring warns
            # about. Recompute explicitly from this run's own args as the override.
            # MUST mirror the TextLeWorldModel(...) construction below exactly -- a manifest that
            # disagrees with the constructed module set is the "trained for hours, replay drops
            # the weights" failure jepa_architecture_manifest_fields' docstring describes.
            "terminal_head": bool(_base_manifest.get("terminal_head", False)) or args.terminal_loss_coeff > 0,
            "value_head": bool(_base_manifest.get("value_head", False)) or args.value_loss_coeff > 0,
            # args was synced with the base manifest above, so this reflects the tokenization
            # regime actually used by this run (inherited or newly enabled).
            "truncate_states_keep_newest": bool(args.truncate_states_keep_newest),
            # Diagnostic marker: a bypass-trained trunk reads raw pooled features, and when
            # hidden_size == latent_dim its shape coincides with a normal trunk -- replay must
            # refuse it explicitly or it would load cleanly and silently mispredict.
            "recognition_probe_bypass_projector": bool(getattr(args, "recognition_probe_bypass_projector", False)),
        }

    # Resolve base-checkpoint inheritance ONTO args before constructing, so the model and every
    # later jepa_architecture_manifest_fields(args) call agree on the trunk's input width.
    # Inherit ONLY when the base checkpoint actually carries trunk weights whose input width
    # would otherwise mismatch. A predictor-only checkpoint (phase-1 pretraining) has no trunk,
    # so the trunk is built fresh here and this run is free to choose its own readout -- which
    # is what makes a `pred_only` recognition probe on a `state`-trained trunk possible.
    _base_has_trunk = False
    _base_state_path = init_checkpoint / "text_leworldmodel.pt"
    if _base_state_path.is_file():
        try:
            _base_keys = torch.load(_base_state_path, map_location="meta", weights_only=True).keys()
        except Exception:  # noqa: BLE001 - fall back to the conservative "inherit" path
            _base_keys = ()
        _base_has_trunk = any(str(k).startswith("canonical_event_trunk.") for k in _base_keys)
    _inherited_head_inputs = _base_manifest.get("canonical_event_head_inputs") if _base_has_trunk else None
    if not _base_has_trunk and _base_manifest.get("canonical_event_head_inputs") and is_main_process():
        print(f"[canonical-event heads] base checkpoint has no canonical_event_trunk weights; "
              f"using --canonical-event-head-inputs="
              f"{getattr(args, 'canonical_event_head_inputs', 'all')!r} rather than the recorded "
              f"{_base_manifest['canonical_event_head_inputs']!r} (nothing to shape-match).",
              flush=True)
    if _inherited_head_inputs and _inherited_head_inputs != getattr(args, "canonical_event_head_inputs", "all"):
        print(f"[canonical-event heads] inheriting canonical_event_head_inputs="
              f"{_inherited_head_inputs!r} from the base checkpoint "
              f"(overrides --canonical-event-head-inputs="
              f"{getattr(args, 'canonical_event_head_inputs', 'all')!r}; the trunk shape is fixed by it).",
              flush=True)
    if _inherited_head_inputs:
        args.canonical_event_head_inputs = str(_inherited_head_inputs)

    model = TextLeWorldModel(
        backbone=backbone,
        latent_dim=args.latent_dim,
        memory_tokens=args.memory_tokens,
        dropout=args.predictor_dropout,
        predictor_hidden_multiplier=args.predictor_hidden_multiplier,
        goal_conditioning=not args.disable_goal_conditioning,
        latent_type=args.latent_type,
        latent_categoricals=args.latent_categoricals,
        latent_classes=args.latent_classes,
        latent_unimix=args.latent_unimix,
        latent_delta_prediction=bool(_base_manifest.get("latent_delta_prediction", args.latent_delta_prediction)),
        pooling=args.pooling,
        canonical_event_vocab_sizes=vocab_sizes,
        canonical_event_head_hidden_size=args.canonical_event_head_hidden_size,
        obs_grounding=bool(_base_manifest.get("obs_grounding", False)),
        obs_ground_decoder_dim=int(_base_manifest.get("obs_ground_decoder_dim") or args.obs_ground_decoder_dim),
        obs_ground_decoder_layers=int(_base_manifest.get("obs_ground_decoder_layers") or args.obs_ground_decoder_layers),
        obs_ground_decoder_heads=int(_base_manifest.get("obs_ground_decoder_heads") or args.obs_ground_decoder_heads),
        obs_ground_decoder_memory_tokens=int(_base_manifest.get("obs_ground_decoder_memory_tokens") or args.obs_ground_decoder_memory_tokens),
        obs_ground_decoder_max_length=int(_base_manifest.get("obs_ground_decoder_max_length") or max(128, (args.obs_ground_max_tokens or 0) + 32)),
        tool_vocab_size=int(_base_manifest.get("tool_vocab_size") or 0),
        tool_select=bool(_base_manifest.get("tool_select", False)),
        action_encoder=bool(_base_manifest.get("action_encoder", False)),
        action_head_embed_dim=int(_base_manifest.get("action_head_embed_dim") or args.action_head_embed_dim),
        action_decoder=bool(_base_manifest.get("action_decoder", False)),
        action_decoder_max_noise_std=float(_base_manifest.get("action_decoder_max_noise_std") or args.action_decoder_max_noise_std),
        action_decoder_dim=int(_base_manifest.get("action_decoder_dim") or args.action_decoder_dim),
        action_decoder_layers=int(_base_manifest.get("action_decoder_layers") or args.action_decoder_layers),
        action_decoder_heads=int(_base_manifest.get("action_decoder_heads") or args.action_decoder_heads),
        action_decoder_memory_tokens=int(_base_manifest.get("action_decoder_memory_tokens") or args.action_decoder_memory_tokens),
        action_decoder_max_length=int(_base_manifest.get("action_decoder_max_length") or args.max_action_length),
        action_decoder_tool_vocab_size=int(_base_manifest.get("action_decoder_tool_vocab_size") or 0),
        fast_lewm=bool(_base_manifest.get("fast_lewm", False)),
        fast_lewm_dim=int(_base_manifest.get("fast_lewm_dim") or args.fast_lewm_dim),
        fast_lewm_layers=int(_base_manifest.get("fast_lewm_layers") or args.fast_lewm_layers),
        fast_lewm_heads=int(_base_manifest.get("fast_lewm_heads") or args.fast_lewm_heads),
        fast_lewm_max_horizon=int(_base_manifest.get("fast_lewm_max_horizon") or max(args.prediction_horizon, 8)),
        # Attach a head when EITHER the base checkpoint carries weights for it (so those
        # pretrained weights load instead of being dropped as unexpected keys) OR this run
        # trains it. `.get(key, fallback)` alone is wrong: the fallback only applies when the
        # key is ABSENT, and every checkpoint written by the current code records
        # "terminal_head"/"value_head" explicitly -- so a base manifest saying `false` would
        # veto a head this run was asked to train, silently skipping that loss.
        terminal_head=bool(_base_manifest.get("terminal_head", False)) or args.terminal_loss_coeff > 0,
        value_head=bool(_base_manifest.get("value_head", False)) or args.value_loss_coeff > 0,
        recognition_bypass_projector=bool(getattr(args, "recognition_probe_bypass_projector", False)),
        # Inherit from the base checkpoint when continuing from one: the trunk's input width
        # depends on this, so a mismatch would fail at load rather than silently mis-wire.
        canonical_event_head_inputs=args.canonical_event_head_inputs,
        state_updater=bool(
            _base_manifest.get("state_updater", getattr(args, "event_state_decomposition", False))
        ),
        state_updater_objective=str(
            _base_manifest.get("state_updater_objective")
            or getattr(args, "state_updater_objective", "future_event")
        ),
        recurrent_state_init=bool(
            _base_manifest.get("recurrent_state_init", getattr(args, "recurrent_state_init", False))
        ),
        # Inherited like every other shape-fixing field: heads training on top of a transformer-
        # predictor trunk must rebuild the same predictor or its weights fail to load.
        predictor_arch=str(_base_manifest.get("predictor_arch") or args.predictor_arch),
        predictor_transformer_dim=int(
            _base_manifest.get("predictor_transformer_dim", args.predictor_transformer_dim) or 0
        ),
        predictor_transformer_layers=int(
            _base_manifest.get("predictor_transformer_layers") or args.predictor_transformer_layers
        ),
        predictor_transformer_heads=int(
            _base_manifest.get("predictor_transformer_heads") or args.predictor_transformer_heads
        ),
        predictor_transformer_mlp_ratio=float(
            _base_manifest.get("predictor_transformer_mlp_ratio") or args.predictor_transformer_mlp_ratio
        ),
        predictor_history_length=int(
            _base_manifest.get("predictor_history_length", args.predictor_history_length) or 0
        ),
    )
    load_jepa_state_dict_for_training(
        model, init_checkpoint, allow_missing_success_head=True, allow_missing_canonical_event_heads=True
    )
    backbone_partially_unfrozen = False
    if args.joint_canonical_event_training:
        # Joint mode trains the TRUNK too -- the latent/SIGReg terms exist precisely to shape
        # encoder_projector/predictor, so freezing them (as heads-only does) would make those
        # losses backpropagate into nothing. Only the backbone is optionally frozen, exactly as
        # the main training loop treats --freeze-backbone -- including --unfreeze-top-backbone-
        # layers, so a partial-backbone ablation can run through this entry point too.
        for param in model.parameters():
            param.requires_grad = True
        backbone_freeze_summary: dict[str, Any] = {"unfreeze_top_backbone_layers": 0}
        if args.freeze_backbone:
            backbone_freeze_summary = freeze_backbone_except_top_layers(
                model.backbone, int(getattr(args, "unfreeze_top_backbone_layers", 0) or 0)
            )
        backbone_partially_unfrozen = bool(backbone_freeze_summary.get("unfreeze_top_backbone_layers"))
        if not success_head_training_enabled(args):
            freeze_success_head(model)
        if is_main_process():
            if not args.freeze_backbone:
                backbone_state = "trainable"
            elif backbone_partially_unfrozen:
                backbone_state = (
                    f"top {backbone_freeze_summary['unfreeze_top_backbone_layers']} of "
                    f"{backbone_freeze_summary['backbone_layers_found']} layers trainable"
                )
            else:
                backbone_state = "frozen"
            print(
                f"[canonical_event_head_train] joint mode: training trunk + heads (backbone {backbone_state}).",
                flush=True,
            )
    else:
        freeze_for_canonical_event_head_training(
            model,
            train_value_head=args.value_loss_coeff > 0,
            train_terminal_head=args.terminal_loss_coeff > 0,
        )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Frequency-aware loss weights (train only), computed from the full train label counts.
    canonical_event_train_class_weights: dict[str, torch.Tensor] | None = None
    canonical_event_train_pos_weights: dict[str, torch.Tensor] | None = None
    terminal_train_pos_weight: torch.Tensor | None = None
    terminal_pos_weight_value, terminal_counts = terminal_pos_weight_from_examples(
        train_examples, method=args.terminal_class_balance, beta=args.terminal_cb_beta
    )
    if terminal_pos_weight_value is not None:
        terminal_train_pos_weight = torch.tensor(terminal_pos_weight_value, dtype=torch.float, device=device)
    if args.canonical_event_class_balance != "none" or args.canonical_event_focal_gamma > 0:
        single_weights, multi_weights = canonical_event_class_weights(
            train_examples, vocab, method=args.canonical_event_class_balance, beta=args.canonical_event_cb_beta
        )
        canonical_event_train_class_weights = {
            field: torch.tensor(weights, dtype=torch.float, device=device) for field, weights in single_weights.items()
        }
        canonical_event_train_pos_weights = {
            field: torch.tensor(weights, dtype=torch.float, device=device) for field, weights in multi_weights.items()
        }
        if is_main_process():
            print(
                f"[canonical_event_head_train] frequency-aware loss: class_balance={args.canonical_event_class_balance} "
                f"beta={args.canonical_event_cb_beta} focal_gamma={args.canonical_event_focal_gamma} (train-only; eval unweighted).",
                flush=True,
            )
    if is_main_process() and args.terminal_class_balance != "none":
        print(
            f"[canonical_event_head_train] terminal balance: method={args.terminal_class_balance} "
            f"beta={args.terminal_cb_beta} counts={terminal_counts} "
            f"pos_weight={terminal_pos_weight_value if terminal_pos_weight_value is not None else 'disabled'} "
            "(train-only; eval unweighted).",
            flush=True,
        )
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )

    collator = CanonicalEventCollator(tokenizer)
    train_dataset = CanonicalEventDataset(train_examples, tokenizer, args, vocab)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collator,
    )
    eval_loader = (
        DataLoader(
            CanonicalEventDataset(eval_examples, tokenizer, args, vocab),
            batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=collator,
        )
        if eval_examples
        else None
    )

    data_manifest = {
        "method": "text_leworldmodel_canonical_event_heads",
        "checkpoint_format": "adapter_state_dict_without_backbone",
        "initialized_from_jepa_checkpoint": str(init_checkpoint),
        "canonical_event_train_jsonl": str(args.canonical_event_train_jsonl),
        "canonical_event_eval_jsonl": str(args.canonical_event_eval_jsonl),
        "canonical_event_heads": getattr(args, "canonical_event_heads", "all"),
        "recognition_probe": bool(args.canonical_event_recognition_probe),
        "recognition_probe_bypass_projector": bool(getattr(args, "recognition_probe_bypass_projector", False)),
        "fields": {"single_label": list(trained_single_fields), "multi_label": list(trained_multi_fields)},
        "vocab_sizes": vocab_sizes,
        "canonical_event_head_hidden_size": args.canonical_event_head_hidden_size,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "canonical_event_train_sample_percentage": args.canonical_event_train_sample_percentage,
        "canonical_event_train_sample_seed": args.canonical_event_train_sample_seed,
        "train_examples_per_benchmark_full": train_benchmark_counts_full,
        "train_examples_per_benchmark_kept": train_benchmark_counts_kept,
        "classification_loss_coeff": args.canonical_event_classification_loss_coeff,
        "value_loss_coeff": args.value_loss_coeff,
        "value_loss_beta": args.value_loss_beta,
        "train_examples_with_value_target": sum(1 for ex in train_examples if ex.value_target is not None),
        "joint_canonical_event_training": bool(args.joint_canonical_event_training),
        **({
            "terminal_loss_coeff": args.terminal_loss_coeff,
            "terminal_class_balance": args.terminal_class_balance,
            "terminal_cb_beta": args.terminal_cb_beta,
            "terminal_counts": terminal_counts,
            "terminal_pos_weight": terminal_pos_weight_value,
            "latent_loss_coeff": args.latent_loss_coeff,
            "latent_loss_type": args.latent_loss_type,
            "sigreg_coeff": args.sigreg_coeff,
            "freeze_backbone": bool(args.freeze_backbone),
            # Rows lacking a consecutive successor get the latent/SIGReg terms masked out.
            "train_examples_with_latent_target": sum(1 for ex in train_examples if ex.next_state_text),
        } if args.joint_canonical_event_training else {}),
        "distributed": {"enabled": distributed, "world_size": int(os.environ.get("WORLD_SIZE", "1"))},
    }
    if is_main_process():
        dump_json(args.output_dir / "canonical_event_data_manifest.json", data_manifest)
        dump_json(args.output_dir / "canonical_event_vocab.json", vocab)
    if distributed:
        torch.distributed.barrier()

    global_step = 0
    log_history: list[dict[str, Any]] = []

    classification_coeff = float(getattr(args, "canonical_event_classification_loss_coeff", 1.0) or 0.0)
    classification_enabled = classification_coeff > 0
    value_coeff = float(getattr(args, "value_loss_coeff", 0.0) or 0.0)
    # unwrap_model: under torchrun the model is already DDP-wrapped here, and DDP forwards
    # only parameters/buffers/submodules -- plain Python attributes like value_head_enabled /
    # terminal_head_enabled raise AttributeError on the wrapper.
    value_enabled = value_coeff > 0 and unwrap_model(model).value_head_enabled
    joint = bool(getattr(args, "joint_canonical_event_training", False))
    terminal_coeff = float(getattr(args, "terminal_loss_coeff", 0.0) or 0.0)
    # NOT gated on `joint`: the terminal label is derived from the JSONL's own
    # (trajectory_id, interaction_index) grouping (see link_canonical_event_successors), so it is
    # available whether or not the latent terms are. Gating it on joint silently allocated an
    # untrained head, froze it, saved its random weights, and still recorded terminal_head=true
    # in the manifest.
    terminal_enabled = terminal_coeff > 0 and unwrap_model(model).terminal_head_enabled

    def forward_pass(active_model: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Skip computing whichever half is disabled -- e.g. --canonical-event-classification-loss-coeff 0
        # (value-only training) never runs the canonical_event_trunk/heads forward at all.
        return active_model(
            batch,
            compute_canonical_event=classification_enabled,
            compute_value=value_enabled,
            compute_terminal=terminal_enabled,
        )

    def joint_latent_terms(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """(latent_loss, sigreg_loss) for --joint-canonical-event-training.

        Both are masked to the rows that actually have a successor state (see
        link_canonical_event_successors): a terminal row's z_next is a placeholder copy of its
        own current state, so including it would teach the predictor the identity map and would
        double-count that row's latent in the SIGReg whitening statistics.
        """
        mask = batch["latent_target_mask"].to(dtype=outputs["z_pred"].dtype)
        latent = masked_latent_prediction_loss(
            outputs["z_pred"], outputs["z_next"], mask,
            loss_type=getattr(args, "latent_loss_type", "smooth_l1_cosine"),
            beta=getattr(args, "smooth_l1_beta", 1.0),
            delta_anchor=outputs["z_current"] if getattr(args, "latent_delta_prediction", False) else None,
        )
        reps = [outputs["z_current"]]
        valid = mask.to(torch.bool)
        if bool(valid.any()):
            reps.append(outputs["z_next"][valid])
        sig = sigreg_loss(torch.cat(reps, dim=0), args.sigreg_projections, args.sigreg_eps)
        return latent, sig

    # --skip-training still runs the per-head evaluation below against whatever
    # canonical_event_heads.* weights the checkpoint already carries, so a
    # trained checkpoint's heads can be scored without refitting them.
    if args.skip_training:
        if is_main_process():
            print("[canonical_event_head_train] --skip-training: evaluating loaded heads only.", flush=True)
    else:
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        if not trainable_params:
            raise SystemExit("No trainable parameters remain after freezing for canonical-event head training.")
        # Any trainable backbone parameters (full joint fine-tuning, or --unfreeze-top-backbone-
        # layers) get their own param group so --backbone-learning-rate can be smaller than
        # --learning-rate, which is tuned for the randomly-initialised predictor/heads and would
        # damage a pretrained encoder at the same rate -- same reasoning as main()'s Stage-1 loop.
        backbone_param_ids = {id(param) for param in unwrap_model(model).backbone.parameters() if param.requires_grad}
        if backbone_param_ids:
            other_params = [p for p in trainable_params if id(p) not in backbone_param_ids]
            backbone_params = [p for p in trainable_params if id(p) in backbone_param_ids]
            backbone_lr = float(args.backbone_learning_rate if args.backbone_learning_rate is not None else args.learning_rate)
            optimizer = torch.optim.AdamW(
                [{"params": other_params, "lr": args.learning_rate}, {"params": backbone_params, "lr": backbone_lr}],
                lr=args.learning_rate, weight_decay=args.weight_decay,
            )
            if is_main_process():
                print(
                    f"[canonical_event_head_train] optimizer groups: heads/trunk lr={args.learning_rate:g} "
                    f"({sum(p.numel() for p in other_params) / 1e6:.1f}M params), backbone "
                    f"lr={backbone_lr:g} ({sum(p.numel() for p in backbone_params) / 1e6:.1f}M params)",
                    flush=True,
                )
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        total_update_steps = max(1, math.ceil(len(train_loader) * args.num_train_epochs / max(1, args.gradient_accumulation_steps)))
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=min(100, max(1, total_update_steps // 20)), num_training_steps=total_update_steps)
        scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and torch.cuda.is_available())
        amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
        use_amp = (args.fp16 or args.bf16) and torch.cuda.is_available()

        optimizer.zero_grad(set_to_none=True)
        target_steps = int(math.ceil(len(train_loader) * args.num_train_epochs))
        progress = tqdm(total=target_steps, desc="canonical_event_head_train", disable=not is_main_process())
        for epoch in range(math.ceil(args.num_train_epochs)):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            for step, batch in enumerate(train_loader):
                batch = move_batch(batch, device)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    outputs = forward_pass(model, batch)
                    per_field: dict[str, dict[str, torch.Tensor]] = {}
                    loss = outputs["z_pred"].new_zeros(())
                    if classification_enabled:
                        classification_loss, per_field = canonical_event_classification_loss(
                            outputs["canonical_event_logits"], batch,
                            class_weights=canonical_event_train_class_weights,
                            multi_pos_weights=canonical_event_train_pos_weights,
                            focal_gamma=args.canonical_event_focal_gamma,
                        )
                        loss = loss + classification_coeff * classification_loss
                    value_loss = value_mae = None
                    if value_enabled:
                        value_loss, value_mae = value_regression_loss(outputs, beta=args.value_loss_beta)
                        loss = loss + value_coeff * value_loss
                    latent_loss = sig_loss = None
                    terminal_loss = terminal_accuracy = None
                    if joint:
                        latent_loss, sig_loss = joint_latent_terms(outputs, batch)
                        loss = loss + args.latent_loss_coeff * latent_loss + args.sigreg_coeff * sig_loss
                    if terminal_enabled:
                        terminal_loss, terminal_accuracy = terminal_classification_loss(
                            outputs, pos_weight=terminal_train_pos_weight
                        )
                        loss = loss + terminal_coeff * terminal_loss
                    scaled_loss = loss / max(1, args.gradient_accumulation_steps)
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                should_step = (step + 1) % args.gradient_accumulation_steps == 0
                if should_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if scaler.is_enabled():
                        scaler.step(optimizer); scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    metrics = {
                        "step": global_step, "epoch": epoch, "loss": float(loss.detach().cpu()),
                        "lr": scheduler.get_last_lr()[0],
                        **{f"{field}_accuracy": float(values["accuracy"].cpu()) for field, values in per_field.items()},
                        **{f"{field}_loss": float(values["loss"].cpu()) for field, values in per_field.items()},
                        **({"value_loss": float(value_loss.detach().cpu()), "value_mae": float(value_mae.detach().cpu())} if value_loss is not None else {}),
                        **({"latent_loss": float(latent_loss.detach().cpu()), "sigreg_loss": float(sig_loss.detach().cpu())} if latent_loss is not None else {}),
                        **({"terminal_loss": float(terminal_loss.detach().cpu()), "terminal_accuracy": float(terminal_accuracy.detach().cpu())} if terminal_loss is not None else {}),
                    }
                    if is_main_process() and (global_step % args.logging_steps == 0 or global_step == 1):
                        print("[canonical_event_head_train] " + json.dumps(metrics, ensure_ascii=False), flush=True)
                    if is_main_process() and args.save_steps > 0 and global_step % args.save_steps == 0:
                        save_checkpoint_and_prune(
                            model, tokenizer, args.output_dir, global_step, args.save_total_limit,
                            # heads-only always leaves the backbone frozen, but joint mode may be
                            # TRAINING it (fully, or its top --unfreeze-top-backbone-layers blocks)
                            # -- passing freeze_backbone=True there would drop the trained backbone
                            # from the checkpoint and silently pair the trained trunk with the
                            # wrong feature space on reload.
                            freeze_backbone=(
                                True if not args.joint_canonical_event_training
                                else (args.freeze_backbone and not backbone_partially_unfrozen)
                            ),
                            base_model=str(args.model),
                        )
                        dump_json(
                            args.output_dir / f"checkpoint-{global_step}" / "jepa_data_manifest.json",
                            _canonical_event_head_manifest(),
                        )
                    if is_main_process():
                        log_history.append(metrics)
                progress.update(1)
                if progress.n >= target_steps:
                    break
            if progress.n >= target_steps:
                break
        progress.close()

        if distributed:
            torch.distributed.barrier()

    raw_model = unwrap_model(model)
    eval_metrics: dict[str, Any] = {}
    if is_main_process() and eval_loader is not None:
        eval_metrics = evaluate_canonical_event_heads(
            raw_model, eval_loader, vocab, device,
            compute_classification=classification_enabled, compute_value=value_enabled,
            compute_terminal=terminal_enabled, compute_latent=joint,
            latent_loss_type=getattr(args, "latent_loss_type", "smooth_l1_cosine"),
            smooth_l1_beta=getattr(args, "smooth_l1_beta", 1.0),
            latent_delta_prediction=getattr(args, "latent_delta_prediction", False),
            dump_predictions_path=getattr(args, "canonical_event_dump_predictions", None),
        )
        if classification_enabled:
            eval_metrics["per_benchmark"] = evaluate_canonical_event_heads_by_benchmark(
                raw_model, eval_examples, tokenizer, args, vocab, device
            )
        print_canonical_event_eval_summary(eval_metrics)

    final_metrics = {
        "skipped_training": bool(args.skip_training),
        "global_step": global_step,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "eval_metrics": eval_metrics,
        "vocab_sizes": vocab_sizes,
        "trainable_parameter_count": sum(param.numel() for param in raw_model.parameters() if param.requires_grad),
        "terminal_class_balance": args.terminal_class_balance,
        "terminal_cb_beta": args.terminal_cb_beta,
        "terminal_counts": terminal_counts,
        "terminal_pos_weight": terminal_pos_weight_value,
        "distributed": {"enabled": distributed, "world_size": int(os.environ.get("WORLD_SIZE", "1"))},
        "log_history_tail": log_history[-20:],
    }
    if is_main_process():
        # Eval-only (--skip-training) leaves the checkpoint weights untouched, so
        # skip the expensive backbone/adapter re-save; still record the metrics.
        if not args.skip_training:
            torch.save(jepa_adapter_state_dict(raw_model), args.output_dir / "text_leworldmodel.pt")
            backbone.save_pretrained(str(args.output_dir / "backbone"))
            tokenizer.save_pretrained(str(args.output_dir))
        dump_json(args.output_dir / "canonical_event_training_metrics.json", final_metrics)
        # Write a self-contained architecture manifest so JepaTextWorldModelGenerator can load
        # this head checkpoint standalone in --mode=replay (the classifier heads reconstruct
        # the imagined observation/state from an action). Same helper the periodic saves above
        # use, so the final checkpoint and every checkpoint-<step> snapshot are equally safe to
        # build on top of or replay from.
        dump_json(args.output_dir / "jepa_data_manifest.json", _canonical_event_head_manifest())
    return final_metrics


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank = setup_distributed(timeout_minutes=args.distributed_timeout_minutes)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "replay":
        metrics = run_jepa_replay(args)
        summary = {"training_metrics": None, "replay_metrics": metrics}
        if is_main_process():
            dump_json(args.output_dir / "run_summary.json", summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        cleanup_distributed()
        return

    if args.value_loss_coeff > 0 and not (args.train_canonical_event_heads_only or args.joint_canonical_event_training):
        raise SystemExit(
            "--value-loss-coeff requires --train-canonical-event-heads-only or "
            "--joint-canonical-event-training: value_target only exists on the "
            "canonical_event_with_nudge JSONL (see "
            "src/data_preparation/annotate_step_value_scores.py), not on the raw world-model "
            "trajectory files the main training loop reads."
        )

    # Joint training reads the same labeled JSONL as the heads-only path, so it runs through the
    # same entry point -- it just additionally trains the trunk with the latent/SIGReg terms.
    if args.train_canonical_event_heads_only or args.joint_canonical_event_training:
        metrics = run_canonical_event_head_training(args, distributed, local_rank)
        if is_main_process():
            dump_json(args.output_dir / "run_summary.json", {"canonical_event_head_training_metrics": metrics})
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
        cleanup_distributed()
        return

    from transformers import get_cosine_schedule_with_warmup

    if args.backbone_type == "encoder" and args.reconstruction_loss_coeff > 0:
        raise SystemExit("--backbone-type encoder is latent-only; set --reconstruction-loss-coeff 0.")
    if args.prediction_horizon > 1 and args.latent_type == "categorical":
        raise SystemExit("--prediction-horizon>1 (multi-step latent supervision) supports continuous latents only.")
    if args.latent_delta_prediction and args.latent_type == "categorical":
        raise SystemExit("--latent-delta-prediction supports continuous latents only (a delta between straight-through one-hot stacks is not on the categorical simplex).")
    if args.fast_lewm:
        if args.latent_type == "categorical":
            raise SystemExit("--fast-lewm (action-prefix prediction) supports continuous latents only.")
        if args.prediction_horizon <= 1:
            raise SystemExit("--fast-lewm requires --prediction-horizon>1 (it predicts multiple action-prefix horizons).")
        if args.predictor_arch == "transformer":
            # Fast-LeWM routes every prediction (including the one-step z_pred) through its own
            # action-prefix transformer, so the AdaLN predictor would be built, saved, and never
            # called. Fail rather than train an unused ~18-120M-parameter module.
            raise SystemExit(
                "--fast-lewm and --predictor-arch transformer are mutually exclusive: Fast-LeWM "
                "predicts through its own action-prefix transformer, leaving the AdaLN predictor "
                "unused. Pick one."
            )
    if args.predictor_arch == "transformer" and args.event_state_decomposition:
        # The causal transformer IS the state updater: h_t accumulates the history and is
        # trained end-to-end by the event prediction read off it. A separate U would be
        # redundant, and worse, untrained -- with a frame history present the predictor's tokens
        # are the logged events, so U's output (passed as z_current) is not even a token.
        raise SystemExit(
            "--predictor-arch transformer is incompatible with --event-state-decomposition / "
            "--event-state-recurrent: the transformer's hidden state h_t already is the belief "
            "state, so the separate State Updater U has no role and would receive no gradient. "
            "Use --event-target to keep the next-observation target without the U module."
        )

    init_checkpoint = args.jepa_checkpoint_path or args.output_dir
    tokenizer_source = init_checkpoint if args.train_success_head_only and init_checkpoint.is_dir() else args.model
    backbone_source = init_checkpoint / "backbone" if args.train_success_head_only and (init_checkpoint / "backbone").is_dir() else args.model
    # --resume-from-checkpoint warm-starts regular training: prefer that checkpoint's
    # own backbone/tokenizer when it bundles them (a full run dir), else fall back to
    # --model (periodic checkpoint-<step> dirs are adapter-only, no backbone/).
    if args.resume_from_checkpoint is not None and not args.train_success_head_only:
        if (args.resume_from_checkpoint / "backbone").is_dir():
            backbone_source = args.resume_from_checkpoint / "backbone"
            tokenizer_source = args.resume_from_checkpoint
    tokenizer = load_text_tokenizer(tokenizer_source, trust_remote_code=args.trust_remote_code)
    backbone = load_jepa_backbone(backbone_source, backbone_type=args.backbone_type, trust_remote_code=args.trust_remote_code, dtype=resolve_torch_dtype(args.dtype))
    if args.gradient_checkpointing:
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
        else:
            raise SystemExit(f"Backbone {args.model} does not support gradient checkpointing.")
        if hasattr(backbone, "config") and hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False
    backbone_freeze_summary: dict[str, Any] = {"unfreeze_top_backbone_layers": 0}
    backbone_partially_unfrozen = False
    if args.freeze_backbone:
        backbone_freeze_summary = freeze_backbone_except_top_layers(
            backbone, int(getattr(args, "unfreeze_top_backbone_layers", 0) or 0)
        )
        if is_main_process() and backbone_freeze_summary["unfreeze_top_backbone_layers"]:
            print(
                "[jepa_train] backbone partially unfrozen: last "
                f"{backbone_freeze_summary['unfreeze_top_backbone_layers']} of "
                f"{backbone_freeze_summary['backbone_layers_found']} blocks "
                f"(indices {backbone_freeze_summary['backbone_trainable_layer_indices']}), "
                f"trailing_norm={backbone_freeze_summary['trailing_norm_trainable']}, "
                f"{backbone_freeze_summary['backbone_trainable_parameters'] / 1e6:.1f}M trainable "
                "backbone parameters. The JEPA target comes from this same encoder, so watch "
                "z_current_std/z_next_std (collapse) alongside the loss.",
                flush=True,
            )
    backbone_partially_unfrozen = bool(backbone_freeze_summary.get("unfreeze_top_backbone_layers"))

    # Only rank 0 pays for parsing the raw trajectory JSON -- with
    # --trajectory-dataset all/adp_all this is tens of GB of nested Python
    # objects, and doing that identically on every torchrun process can exhaust
    # host RAM; a rank getting OOM-killed then hangs every surviving rank on the
    # next NCCL collective. Other ranks instead read back the already-extracted
    # (far smaller, flat) examples rank 0 just wrote out below.
    train_examples_path = args.output_dir / "jepa_train_examples.jsonl"
    eval_examples_path = args.output_dir / "jepa_eval_examples.jsonl"
    fingerprint_path = args.output_dir / "jepa_examples_fingerprint.json"
    if is_main_process():
        # Extraction over a large preset can run for hours; re-running it on every relaunch is
        # what pushes rank 0 past the collective timeout while the other ranks sit at the
        # barrier. With --reuse-extracted-examples, reuse a previous run's output when the
        # fingerprint (input paths + size/mtime + extraction args) matches exactly; any
        # mismatch re-extracts and says which key differed, so a stale cache cannot be used
        # silently.
        reused = False
        if getattr(args, "reuse_extracted_examples", False):
            want = extracted_examples_fingerprint(args)
            if fingerprint_path.is_file() and train_examples_path.is_file():
                have = json.loads(fingerprint_path.read_text())
                differing = [k for k in want if have.get(k) != want[k]]
                if differing:
                    print(f"[reuse] fingerprint mismatch on {differing}; re-extracting.", flush=True)
                else:
                    print(f"[reuse] reusing extracted examples from {train_examples_path}", flush=True)
                    train_examples = [JepaExample(**row) for row in load_jsonl_rows(train_examples_path)]
                    eval_examples = ([JepaExample(**row) for row in load_jsonl_rows(eval_examples_path)]
                                     if eval_examples_path.is_file() else [])
                    reused = True
            else:
                print("[reuse] no cached examples found; extracting.", flush=True)

        if not reused:
            # Stream per-file so rank 0's raw-JSON peak is one file, not the whole
            # (tens of GB) corpus -- the previous load-all-then-extract path OOM-killed
            # rank 0 on large presets like --trajectory-dataset all.
            train_examples = load_and_extract_jepa_examples_streaming(args.train_data_path, args)
            eval_examples = []
            if not args.skip_eval:
                eval_examples = load_and_extract_jepa_examples_streaming(args.eval_data_path, args)
            # extract_jepa_examples appends state-free (ADP) examples after the
            # state-bearing ones, so a plain prefix cut would drop them. Shuffle with the
            # run seed before capping to keep a representative mix across all sources.
            # Only on a fresh extraction -- a reused cache was already capped by the run that
            # wrote it, and the cap args are part of the fingerprint.
            if args.max_train_examples > 0:
                random.Random(args.seed).shuffle(train_examples)
                train_examples = train_examples[: args.max_train_examples]
            if args.max_eval_examples > 0:
                random.Random(args.seed + 1).shuffle(eval_examples)
                eval_examples = eval_examples[: args.max_eval_examples]

        data_manifest = {
            "method": "text_leworldmodel_jepa_tool_use",
            "checkpoint_format": "adapter_state_dict_without_backbone",
            "trajectory_dataset": args.trajectory_dataset,
            "train_data_paths": [str(path) for path in args.train_data_path],
            "eval_data_paths": [str(path) for path in args.eval_data_path],
            "backbone": str(backbone_source),
            "train": summarize_examples(train_examples),
            "eval": summarize_examples(eval_examples),
            "resolved_hidden_size": resolve_backbone_hidden_size(backbone),
            "goal_representation": None,
            "inference_goal_representation": None,
            "history_observation_representation": "raw_observation",
            "goal_loss_coeff": 0.0,
            "goal_loss_enabled": False,
            "success_loss_coeff": args.success_loss_coeff,
            "dtype": args.dtype,
            "train_success_head_only": args.train_success_head_only,
            "initialized_from_jepa_checkpoint": str(args.jepa_checkpoint_path or args.output_dir) if args.train_success_head_only else None,
            "loss_coefficients": {"latent": args.latent_loss_coeff, "sigreg": args.sigreg_coeff, "reconstruction": args.reconstruction_loss_coeff, "kl": args.kl_loss_coeff, "success": args.success_loss_coeff},
            "kl_balance": args.kl_balance,
            "kl_free_nats": args.kl_free_nats,
            "distributed": {"enabled": distributed, "world_size": int(os.environ.get("WORLD_SIZE", "1"))},
            # Core architecture dims + every optional-module flag (obs_grounding/tool_select/
            # action_encoder/fast_lewm/action_decoder): JepaTextWorldModelGenerator (finetuning.py)
            # and --train-canonical-event-heads-only's base-checkpoint inheritance both read
            # these at load time to construct a matching TextLeWorldModel BEFORE loading the
            # state dict -- without them, any of these modules' weights show up as "unexpected
            # keys" and get silently dropped rather than loaded.
            **jepa_architecture_manifest_fields(
                args,
                tool_vocab_size=(
                    len(build_tool_vocabulary(train_examples))
                    if (args.tool_select_loss_coeff > 0 or args.action_encoder_loss_coeff > 0)
                    else 0
                ),
            ),
        }
        dump_json(args.output_dir / "jepa_data_manifest.json", data_manifest)
        if not reused:
            dump_jsonl(train_examples_path, (asdict(example) for example in train_examples))
            dump_jsonl(eval_examples_path, (asdict(example) for example in eval_examples))
            dump_json(fingerprint_path, extracted_examples_fingerprint(args))
    if distributed:
        torch.distributed.barrier()
        if not is_main_process():
            train_examples = [JepaExample(**row) for row in load_jsonl_rows(train_examples_path)]
            eval_examples = [JepaExample(**row) for row in load_jsonl_rows(eval_examples_path)]

    # Optional latent action-head (P1/P2) AND the action decoder's family conditioning all need
    # the same deterministic tool vocabulary from the (rank-shared) train examples -- build it
    # whenever any of the three is enabled, not just P1/P2, so tool_label is available for the
    # decoder's same-tool mixup pairing and tool-embedding conditioning too.
    action_head_enabled = (
        args.tool_select_loss_coeff > 0 or args.action_encoder_loss_coeff > 0 or args.action_decoder_loss_coeff > 0
    )
    tool_vocab = build_tool_vocabulary(train_examples) if action_head_enabled else None
    if action_head_enabled and is_main_process():
        dump_json(args.output_dir / "tool_vocab.json", tool_vocab)
        print(f"[jepa_train] action-head tool vocabulary size = {len(tool_vocab)}", flush=True)
    model = TextLeWorldModel(backbone=backbone, latent_dim=args.latent_dim, memory_tokens=args.memory_tokens, dropout=args.predictor_dropout, predictor_hidden_multiplier=args.predictor_hidden_multiplier, goal_conditioning=not args.disable_goal_conditioning, latent_type=args.latent_type, latent_categoricals=args.latent_categoricals, latent_classes=args.latent_classes, latent_unimix=args.latent_unimix, latent_delta_prediction=args.latent_delta_prediction, pooling=args.pooling, obs_grounding=args.obs_token_ground_coeff > 0, obs_ground_decoder_dim=args.obs_ground_decoder_dim, obs_ground_decoder_layers=args.obs_ground_decoder_layers, obs_ground_decoder_heads=args.obs_ground_decoder_heads, obs_ground_decoder_memory_tokens=args.obs_ground_decoder_memory_tokens, obs_ground_decoder_max_length=max(128, (args.obs_ground_max_tokens or 0) + 32), tool_vocab_size=len(tool_vocab) if tool_vocab else 0, tool_select=args.tool_select_loss_coeff > 0, action_encoder=args.action_encoder_loss_coeff > 0, action_head_embed_dim=args.action_head_embed_dim, action_decoder=args.action_decoder_loss_coeff > 0, action_decoder_max_noise_std=args.action_decoder_max_noise_std, action_decoder_dim=args.action_decoder_dim, action_decoder_layers=args.action_decoder_layers, action_decoder_heads=args.action_decoder_heads, action_decoder_memory_tokens=args.action_decoder_memory_tokens, action_decoder_max_length=args.max_action_length, action_decoder_tool_vocab_size=len(tool_vocab) if tool_vocab else 0, fast_lewm=args.fast_lewm, fast_lewm_dim=args.fast_lewm_dim, fast_lewm_layers=args.fast_lewm_layers, fast_lewm_heads=args.fast_lewm_heads, fast_lewm_max_horizon=max(args.prediction_horizon, 8), terminal_head=args.terminal_loss_coeff > 0, value_head=args.value_loss_coeff > 0,
                             # Event/state decomposition. These MUST be passed: without them the
                             # constructor defaults win, U and I(c) are never built, z_event stays
                             # None, and --event-loss-coeff / --future-event-loss-coeff /
                             # --consistency-loss-coeff are all silently inert while
                             # jepa_architecture_manifest_fields still records them as enabled.
                             canonical_event_head_inputs=args.canonical_event_head_inputs,
                             state_updater=bool(getattr(args, "event_state_decomposition", False)),
                             state_updater_objective=str(getattr(args, "state_updater_objective", "future_event")),
                             recurrent_state_init=bool(getattr(args, "recurrent_state_init", False)),
                             predictor_arch=args.predictor_arch,
                             predictor_transformer_dim=args.predictor_transformer_dim,
                             predictor_transformer_layers=args.predictor_transformer_layers,
                             predictor_transformer_heads=args.predictor_transformer_heads,
                             predictor_transformer_mlp_ratio=args.predictor_transformer_mlp_ratio,
                             predictor_history_length=args.predictor_history_length)
    if args.train_success_head_only:
        load_jepa_state_dict_for_training(model, init_checkpoint, allow_missing_success_head=True)
        freeze_for_success_head_training(model)
    else:
        if args.resume_from_checkpoint is not None:
            if is_main_process():
                print(f"[jepa_train] resuming (weights-only warm start) from {args.resume_from_checkpoint}", flush=True)
            load_jepa_state_dict_for_training(
                model,
                args.resume_from_checkpoint,
                allow_missing_success_head=True,
                allow_missing_canonical_event_heads=True,
            )
        if not success_head_training_enabled(args):
            freeze_success_head(model)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.action_decoder_loss_coeff > 0:
        # Calibrate the noise/mixup scale (see _action_decoder_augment) against the empirical
        # spread of REAL z_action, instead of leaving --action-decoder-max-noise-std as an
        # absolute number the inference-time CEM's init_std/min_std has to independently guess
        # at. Computed once, before training perturbs the encoder, and stored as a checkpoint
        # buffer (action_decoder_latent_scale) so it travels to replay time automatically.
        latent_scale = calibrate_action_decoder_latent_scale(model, tokenizer, train_examples, device, args.max_action_length)
        model.action_decoder_latent_scale.fill_(latent_scale)
        if is_main_process():
            print(f"[jepa_train] action_decoder_latent_scale calibrated to {latent_scale:.4f}", flush=True)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None, output_device=local_rank if torch.cuda.is_available() else None, find_unused_parameters=True)

    train_dataset = JepaTextDataset(train_examples, tokenizer, args, tool_vocab=tool_vocab)
    eval_dataset = JepaTextDataset(eval_examples, tokenizer, args, tool_vocab=tool_vocab)
    # Explicit seed so the per-epoch shuffle (DistributedSampler shuffles by
    # seed + epoch) is reproducible across runs -- required for --resume-from-checkpoint
    # to continue the exact same data order after skipping already-consumed batches.
    # Wrapped so the resume epoch can drop its already-trained prefix at the sampler
    # level (see ResumableSampler); no-op when not resuming.
    train_sampler = ResumableSampler(DistributedSampler(train_dataset, shuffle=True, seed=args.seed)) if distributed else None
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=train_sampler is None, sampler=train_sampler, collate_fn=JepaCollator(tokenizer))
    eval_loader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=JepaCollator(tokenizer))

    if args.skip_training:
        if is_main_process():
            dump_json(args.output_dir / "jepa_training_metrics.json", {"skipped": True})
        cleanup_distributed()
        return

    # Backbone parameters go in their own group so --backbone-learning-rate can be smaller than
    # the predictor's: --learning-rate is set for a randomly-initialised predictor and applying
    # it to a pretrained encoder damages the representation the pooled latent depends on. The
    # cosine schedule scales every group by the same factor, so the ratio holds throughout.
    backbone_param_ids = {
        id(param) for param in unwrap_model(model).backbone.parameters() if param.requires_grad
    }
    backbone_params = [param for param in model.parameters() if param.requires_grad and id(param) in backbone_param_ids]
    other_params = [param for param in model.parameters() if param.requires_grad and id(param) not in backbone_param_ids]
    if not backbone_params and not other_params:
        raise SystemExit("No trainable parameters remain after applying training/freezing options.")
    backbone_lr = float(
        args.backbone_learning_rate if args.backbone_learning_rate is not None else args.learning_rate
    )
    param_groups = [{"params": other_params, "lr": args.learning_rate}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
        if is_main_process():
            print(f"[jepa_train] optimizer groups: predictor/heads lr={args.learning_rate:g} "
                  f"({sum(p.numel() for p in other_params) / 1e6:.1f}M params), backbone "
                  f"lr={backbone_lr:g} ({sum(p.numel() for p in backbone_params) / 1e6:.1f}M params)",
                  flush=True)
    elif args.backbone_learning_rate is not None and is_main_process():
        print("[jepa_train] --backbone-learning-rate ignored: no trainable backbone parameters "
              "(fully frozen backbone).", flush=True)
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_update_steps = max(1, math.ceil(len(train_loader) * args.num_train_epochs / max(1, args.gradient_accumulation_steps)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=min(100, max(1, total_update_steps // 20)), num_training_steps=total_update_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and torch.cuda.is_available())
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    use_amp = (args.fp16 or args.bf16) and torch.cuda.is_available()
    success_enabled = success_head_training_enabled(args)
    terminal_enabled = float(getattr(args, "terminal_loss_coeff", 0.0) or 0.0) > 0

    optimizer.zero_grad(set_to_none=True)
    log_history: list[dict[str, Any]] = []
    batches_per_epoch = max(1, len(train_loader))
    target_steps = int(math.ceil(batches_per_epoch * args.num_train_epochs))
    # Resume continues the deterministic data order rather than restarting at
    # epoch 0 / step 0. global_step counts optimizer steps, so the number of
    # batches already consumed is global_step * gradient_accumulation_steps. The
    # sampler shuffles by (seed + epoch), so with the same data + seed each epoch's
    # order reproduces exactly and we can fast-forward past the consumed batches.
    global_step = resume_global_step_from_checkpoint(args.resume_from_checkpoint) if args.resume_from_checkpoint is not None else 0
    start_batch = min(global_step * max(1, args.gradient_accumulation_steps), target_steps)
    start_epoch = start_batch // batches_per_epoch
    skip_in_start_epoch = start_batch % batches_per_epoch
    # Re-align the cosine LR schedule to the resumed step. (Optimizer moment
    # buffers are NOT restored -- resume is a weights-only warm start -- but a
    # continuous LR schedule matters far more than the momentum estimates.)
    for _ in range(min(global_step, total_update_steps)):
        scheduler.step()
    if is_main_process() and global_step:
        print(
            f"[jepa_train] resuming at global_step={global_step}: skipping to epoch {start_epoch}, "
            f"batch {skip_in_start_epoch}/{batches_per_epoch} ({start_batch}/{target_steps} batches).",
            flush=True,
        )
    progress = tqdm(total=target_steps, initial=start_batch, desc="jepa_train", disable=not is_main_process())
    for epoch in range(math.ceil(args.num_train_epochs)):
        if epoch < start_epoch:
            continue  # fully completed in a previous run
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        skip = skip_in_start_epoch if epoch == start_epoch else 0
        # Prefer dropping the already-trained prefix at the sampler level so those
        # batches are never fetched/tokenized. With no sampler (non-distributed
        # shuffle) there is nothing to wrap, so drain them from the loader instead.
        loader_skip = 0
        if skip:
            if train_sampler is not None:
                train_sampler.skip_next_epoch(skip * args.per_device_train_batch_size)
            else:
                loader_skip = skip
        model.train()
        for step, batch in enumerate(train_loader):
            if step < loader_skip:
                continue  # fast-forward (same order, not retrained); non-distributed fallback only
            batch = move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                outputs = model(
                    batch,
                    compute_reconstruction=args.reconstruction_loss_coeff > 0,
                    compute_success=success_enabled,
                    compute_action_decoder=args.action_decoder_loss_coeff > 0,
                    compute_terminal=terminal_enabled,
                )
                loss, components = compute_jepa_loss(outputs, args)
                latent_loss = components["latent_loss"]
                sig_loss = components["sigreg_loss"]
                recon_loss = components["reconstruction_loss"]
                success_loss = components.get("success_loss")
                success_accuracy = components.get("success_accuracy")
                scaled_loss = loss / max(1, args.gradient_accumulation_steps)
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            should_step = (step + 1) % args.gradient_accumulation_steps == 0
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                metrics = {
                    "step": global_step, "epoch": epoch, "loss": float(loss.detach().cpu()),
                    "latent_loss": float(latent_loss.detach().cpu()),
                    "sigreg_loss": float(sig_loss.detach().cpu()),
                    "reconstruction_loss": float(recon_loss.detach().cpu()),
                    **({"success_loss": float(success_loss.detach().cpu()), "success_accuracy": float(success_accuracy.detach().cpu())} if success_loss is not None and success_accuracy is not None else {}),
                    **({"terminal_loss": float(components["terminal_loss"].detach().cpu()), "terminal_accuracy": float(components["terminal_accuracy"].detach().cpu())} if components.get("terminal_loss") is not None else {}),
                    **({"kl_dynamics": float(components["kl_dynamics"].detach().cpu()), "kl_representation": float(components["kl_representation"].detach().cpu())} if "kl_dynamics" in components else {}),
                    **({"obs_ground_loss": float(components["obs_ground_loss"].detach().cpu()), "obs_ground_coverage": float(components["obs_ground_coverage"].detach().cpu())} if components.get("obs_ground_loss") is not None else {}),
                    **({"tool_select_loss": float(components["tool_select_loss"].detach().cpu()), "tool_top1": float(components["tool_top1"].detach().cpu()), "tool_top5": float(components["tool_top5"].detach().cpu()), "tool_top10": float(components["tool_top10"].detach().cpu())} if components.get("tool_select_loss") is not None else {}),
                    **({"action_encoder_loss": float(components["action_encoder_loss"].detach().cpu())} if components.get("action_encoder_loss") is not None else {}),
                    **({"action_decoder_loss": float(components["action_decoder_loss"].detach().cpu())} if components.get("action_decoder_loss") is not None else {}),
                    **({"action_sigreg_loss": float(components["action_sigreg_loss"].detach().cpu())} if components.get("action_sigreg_loss") is not None else {}),
                    **({k: float(components[k].detach().cpu()) for k in
                        ("action_contrastive_loss", "ac_d_pos", "ac_d_neg", "ac_gap", "ac_active_rate")
                        } if components.get("action_contrastive_loss") is not None else {}),
                    **({k: float(components[k].detach().cpu()) for k in
                        ("future_event_loss", "consistency_loss", "state_update_loss", "state_update_vs_persistence")
                        if components.get(k) is not None}),
                    **({"pni": float(components["pni"].detach().cpu())} if components.get("pni") is not None else {}),
                    "lr": scheduler.get_last_lr()[0],
                    "z_current_mean": float(outputs["z_current"].detach().mean().cpu()),
                    "z_current_std": float(outputs["z_current"].detach().std(unbiased=False).cpu()),
                    "z_next_mean": float(outputs["z_next"].detach().mean().cpu()),
                    "z_next_std": float(outputs["z_next"].detach().std(unbiased=False).cpu()),
                }
                if is_main_process() and (global_step % args.logging_steps == 0 or global_step == 1):
                    print("[jepa_train] " + json.dumps(metrics, ensure_ascii=False), flush=True)
                if is_main_process() and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # A PARTIALLY unfrozen backbone still has modified weights, so the snapshot
                    # must bundle backbone/ -- passing freeze_backbone=True here would silently
                    # drop the trained top blocks and a resume would load pristine pretrained
                    # weights instead. (The end-of-run save always writes backbone/.)
                    save_checkpoint_and_prune(
                        model, tokenizer, args.output_dir, global_step, args.save_total_limit,
                        freeze_backbone=(args.freeze_backbone and not backbone_partially_unfrozen),
                        base_model=str(args.model),
                        args=args, tool_vocab_size=len(tool_vocab) if tool_vocab else 0,
                    )
                if is_main_process():
                    log_history.append(metrics)
            progress.update(1)
            if progress.n >= target_steps:
                break
        if progress.n >= target_steps:
            break
    progress.close()

    if distributed:
        torch.distributed.barrier()
    if is_main_process():
        raw_model = unwrap_model(model)
        eval_metrics = evaluate(raw_model, eval_loader, args, device) if eval_examples else {}
        final_metrics = {
            "global_step": global_step, "train_examples": len(train_examples), "eval_examples": len(eval_examples),
            "eval_metrics": eval_metrics, "train_success_head_only": args.train_success_head_only,
            "trainable_parameter_count": sum(param.numel() for param in raw_model.parameters() if param.requires_grad),
            "backbone_freeze": backbone_freeze_summary,
            "backbone_learning_rate": backbone_lr,
            "distributed": {"enabled": distributed, "world_size": int(os.environ.get("WORLD_SIZE", "1"))},
            "log_history_tail": log_history[-20:],
        }
        torch.save(jepa_adapter_state_dict(raw_model), args.output_dir / "text_leworldmodel.pt")
        backbone.save_pretrained(str(args.output_dir / "backbone"))
        tokenizer.save_pretrained(str(args.output_dir))
        dump_json(args.output_dir / "jepa_training_metrics.json", final_metrics)
    cleanup_distributed()


if __name__ == "__main__":
    main()
