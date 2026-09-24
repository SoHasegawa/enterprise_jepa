#!/usr/bin/env python3
# Provenance: vendored from the EWM repo, branch `jepa` (commit 7b62196, 2026-07-17).
# Two deliberate deviations from that snapshot:
#   1. `build_agent_generator` is imported from `src.evaluation` instead of
#      `src.finetuning`; it moved there in the later `feat/llama-factory` snapshot
#      that the rest of `training/src` comes from.
#   2. See docs/training.md: this snapshot predates the heads used by the
#      checkpoint the paper reports (terminal / value / obs-grounding / fast-LeWM /
#      action-decoder). The authoritative definition of the full net is
#      `src/ejepa_wm/backends/_ewm_jepa.py` (`TextLeWorldModel`).
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
import datetime
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
from src.evaluation import build_agent_generator


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
ADP_TRAJECTORY_DATASETS = (
    "agenttuning_alfworld",
    "agenttuning_db",
    "agenttuning_kg",
    "agenttuning_mind2web",
    "agenttuning_os",
    "agenttuning_webshop",
    "code_feedback",
    "codeactinstruct",
    "coderforge_preview",
    "go-browse-wa",
    "mind2web",
    "mini-coder",
    "nebius_SWE-agent-trajectories",
    "nemotron_terminal_corpus",
    "nnetnav-live",
    "nnetnav-wa",
    "openhands",
    "orca_agentinstruct",
    "swe-gym_openhands_sampled_trajectories",
    "swe-play-trajectories",
    "swe-smith",
    "synatra",
)


def _adp_trajectory_path(name: str) -> Path:
    return DEFAULT_TRAJECTORIES_DIR / f"{name}_world_model_trajectories.json"


ADP_TRAJECTORY_PATHS: dict[str, Path] = {
    name: _adp_trajectory_path(name) for name in ADP_TRAJECTORY_DATASETS
}

# Web-browsing ADP benchmarks. These trajectory files are large and expensive
# to load; --skip-web-trajectories drops them from any preset/explicit path
# list to reduce load on the server.
WEB_BROWSING_ADP_DATASETS = (
    "agenttuning_mind2web",
    "go-browse-wa",
    "mind2web",
    "nnetnav-live",
    "nnetnav-wa",
    "synatra",
    "mini-coder",
    "coderforge_preview",
    "nemotron_terminal_corpus"
)
WEB_BROWSING_TRAJECTORY_PATHS = {ADP_TRAJECTORY_PATHS[name] for name in WEB_BROWSING_ADP_DATASETS}

# The "core" world-model benchmarks: EnterpriseOps-Gym, CRMArenaPro, and
# TerminalBench. TOUCAN is intentionally excluded from `core` (select it via the
# `toucan` preset, or use `all` which still includes it).
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
ADP_ALL_DATA_PATHS = list(ADP_TRAJECTORY_PATHS.values())

TRAJECTORY_DATASET_PRESETS: dict[str, tuple[list[Path], list[Path]]] = {
    "enterpriseops_gym": ([ENTERPRISEOPS_GYM_TRAIN_DATA_PATH], [ENTERPRISEOPS_GYM_EVAL_DATA_PATH]),
    "enterpriseops_gym_enterprise_state": ([ENTERPRISEOPS_GYM_ENTERPRISE_STATE_TRAIN_DATA_PATH], [ENTERPRISEOPS_GYM_ENTERPRISE_STATE_EVAL_DATA_PATH]),
    "terminalbench": ([TERMINALBENCH_2_0_MULTI_MODEL_TRAIN_DATA_PATH], [TERMINALBENCH_2_0_MULTI_MODEL_EVAL_DATA_PATH]),
    "terminalbench_2_0_multi_model": ([TERMINALBENCH_2_0_MULTI_MODEL_TRAIN_DATA_PATH], [TERMINALBENCH_2_0_MULTI_MODEL_EVAL_DATA_PATH]),
    "crmarenapro": ([CRMARENAPRO_MULTI_MODEL_TRAIN_DATA_PATH], [CRMARENAPRO_MULTI_MODEL_EVAL_DATA_PATH]),
    "crmarenapro_multi_model": ([CRMARENAPRO_MULTI_MODEL_TRAIN_DATA_PATH], [CRMARENAPRO_MULTI_MODEL_EVAL_DATA_PATH]),
    "crmarenapro_baseline_crm_agent": ([CRMARENAPRO_BASELINE_CRM_AGENT_TRAIN_DATA_PATH], [CRMARENAPRO_BASELINE_CRM_AGENT_EVAL_DATA_PATH]),
    "toucan": ([TOUCAN_TRAIN_DATA_PATH], [TOUCAN_EVAL_DATA_PATH]),
    # EnterpriseOps-Gym + CRMArenaPro + TerminalBench (no TOUCAN).
    "core": (list(CORE_TRAIN_DATA_PATHS), list(CORE_EVAL_DATA_PATHS)),
    # Every ADP benchmark (state-free).
    "adp_all": (list(ADP_ALL_DATA_PATHS), list(ADP_ALL_DATA_PATHS)),
    # Core three + TOUCAN + every ADP benchmark. Very large: cap with
    # --max-train-examples / --max-eval-examples, or pick a single benchmark preset.
    "all": (
        CORE_TRAIN_DATA_PATHS + [TOUCAN_ALL_DATA_PATH] + ADP_ALL_DATA_PATHS,
        CORE_EVAL_DATA_PATHS + [TOUCAN_EVAL_DATA_PATH] + ADP_ALL_DATA_PATHS,
    ),
}
# One selectable preset per ADP benchmark (e.g. --trajectory-dataset swe-smith).
for _adp_name, _adp_path in ADP_TRAJECTORY_PATHS.items():
    TRAJECTORY_DATASET_PRESETS[_adp_name] = ([_adp_path], [_adp_path])


# --- canonical_event_state / nudge classification heads ---------------------
# JSONL label files (one row per action, independent of the trajectory JSON
# format) produced outside this repo: each row carries system_prompt,
# task_prompt, action, input_history, plus `canonical_event_state` (a
# descriptive classification of what the action's outcome looked like) and
# `nudge` (an actionable pre-execution guidance signal). See
# --train-canonical-event-heads-only.
CANONICAL_EVENT_STATE_FIELDS = (
    "action_type",
    "error_signature",
    "execution_status",
    "object_type",
    "progress_signal",
    "risk_signal",
    "side_effect_type",
)
NUDGE_SINGLE_LABEL_FIELDS = (
    "information_gain",
    "information_sufficiency",
    "recommended_abstract_action",
)
NUDGE_MULTI_LABEL_FIELDS = ("missing_information_type",)
CANONICAL_EVENT_SINGLE_LABEL_FIELDS = CANONICAL_EVENT_STATE_FIELDS + NUDGE_SINGLE_LABEL_FIELDS
CANONICAL_EVENT_ALL_FIELDS = CANONICAL_EVENT_SINGLE_LABEL_FIELDS + NUDGE_MULTI_LABEL_FIELDS

DEFAULT_CANONICAL_EVENT_TRAIN_JSONL = (
    DEFAULT_TRAJECTORIES_DIR
    / "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro_train_examples.jsonl"
)
DEFAULT_CANONICAL_EVENT_EVAL_JSONL = (
    DEFAULT_TRAJECTORIES_DIR
    / "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro_eval_examples.jsonl"
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
    parser.add_argument("--max-input-length", type=int, default=2048)
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
    parser.add_argument("--memory-tokens", type=int, default=8, help="Pseudo encoder tokens decoded from predicted latent.")
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
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
    parser.add_argument("--sigreg-coeff", type=float, default=0.05)
    parser.add_argument("--reconstruction-loss-coeff", type=float, default=0.0)
    parser.add_argument("--goal-loss-coeff", type=float, default=0.0, help="Deprecated and ignored; JEPA training no longer uses goal-observation progress loss.")
    parser.add_argument("--success-loss-coeff", type=float, default=0.0)
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
        "--latent-plan-goal-mode",
        choices=("final", "next_subgoal"),
        default="next_subgoal",
        help="Use the final task goal or the next stage-derived subgoal for per-action latent planning.",
    )
    parser.add_argument(
        "--replay-modes",
        default="baseline,imagined",
        help="Comma-separated replay modes to run in --mode=replay. Supported: baseline, imagined, revision, latent_guided.",
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
    if args.no_gym_task_split_manifest:
        args.gym_task_split_manifest = None
    preset_train_paths, preset_eval_paths = TRAJECTORY_DATASET_PRESETS[args.trajectory_dataset]
    if args.train_data_path is None:
        args.train_data_path = list(preset_train_paths)
    if args.eval_data_path is None:
        args.eval_data_path = list(preset_eval_paths)
    if args.skip_web_trajectories:
        args.train_data_path = [p for p in args.train_data_path if p not in WEB_BROWSING_TRAJECTORY_PATHS]
        args.eval_data_path = [p for p in args.eval_data_path if p not in WEB_BROWSING_TRAJECTORY_PATHS]
    replay_modes = tuple(split_csv(args.replay_modes))
    allowed_replay_modes = {"baseline", "imagined", "revision", "latent_guided"}
    invalid_replay_modes = sorted(set(replay_modes) - allowed_replay_modes)
    if not replay_modes:
        parser.error("--replay-modes must include at least one mode: baseline, imagined, revision, latent_guided")
    if invalid_replay_modes:
        parser.error(
            "--replay-modes contains unsupported mode(s): "
            + ", ".join(invalid_replay_modes)
            + ". Supported: baseline, imagined, revision, latent_guided"
        )
    args.replay_modes = replay_modes
    if args.train_success_head_only and args.success_loss_coeff <= 0:
        args.success_loss_coeff = 1.0
    if args.train_success_head_only and args.train_canonical_event_heads_only:
        parser.error("--train-success-head-only and --train-canonical-event-heads-only are mutually exclusive.")
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
    records = load_json(path)
    if not isinstance(records, list):
        raise SystemExit(f"Expected a list of trajectories in {path}")
    trajectories = normalize_loaded_trajectories(records)
    if adp_dataset_name_from_path(path) not in ADP_DATASETS_WITHOUT_GENUINE_USER_TOOL_OUTPUT:
        trajectories = [inject_user_observations_as_state(trajectory) for trajectory in trajectories]
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


def extract_jepa_examples(trajectories: list[dict[str, Any]], args: argparse.Namespace) -> list[JepaExample]:
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
        })
        history.append({"step": len(history) + 1, "action": action_text, "observation": observation_text})
        raw_histories[trajectory_key] = history[-WORLD_MODEL_INPUT_HISTORY_SIZE:]

    return [JepaExample(**item) for item in pending_rows]


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
    for path in paths:
        trajectories = load_trajectory_file(path, require_enterpriseops_gym=require_enterpriseops_gym)
        if not trajectories:
            continue
        examples.extend(extract_jepa_examples(trajectories, args))
        del trajectories
        gc.collect()
    return examples


class JepaTextDataset(Dataset):
    def __init__(self, examples: list[JepaExample], tokenizer: Any, args: argparse.Namespace) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.args = args
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
        current = self.tokenizer(
            item.current_state_text,
            max_length=self.args.max_input_length,
            truncation=True,
            add_special_tokens=True,
        )
        next_state = self.tokenizer(
            item.next_state_text,
            max_length=self.args.max_input_length,
            truncation=True,
            add_special_tokens=True,
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
        result = {
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
                future_next = self.tokenizer(
                    successor.next_state_text, max_length=self.args.max_input_length, truncation=True, add_special_tokens=True
                )
                future_action_input_ids.append(future_action["input_ids"])
                future_action_attention_mask.append(future_action["attention_mask"])
                future_next_input_ids.append(future_next["input_ids"])
                future_next_attention_mask.append(future_next["attention_mask"])
            result["future_action_input_ids"] = future_action_input_ids
            result["future_action_attention_mask"] = future_action_attention_mask
            result["future_next_input_ids"] = future_next_input_ids
            result["future_next_attention_mask"] = future_next_attention_mask
        return result


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
        labels = self.pad([f["labels"] for f in features], pad_id)
        labels = labels.masked_fill(labels == pad_id, -100)
        batch["labels"] = labels
        success_labels = torch.tensor([float(f.get("success_label", -1)) for f in features], dtype=torch.float)
        batch["success_labels"] = success_labels.clamp_min(0.0)
        batch["success_label_mask"] = (success_labels >= 0).to(torch.float)
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
    return CanonicalEventExample(
        trajectory_id=str(record.get("trajectory_id", "")),
        interaction_index=int(record.get("interaction_index") or 0),
        context_text=context_text,
        current_state_text=build_state_text(context_text, record.get("input_history") or []),
        action_text=render_action(record.get("action")),
        single_labels=single_labels,
        multi_labels=multi_labels,
        benchmark=str(record.get("benchmark") or ""),
    )


def build_canonical_event_examples(records: list[dict[str, Any]]) -> list[CanonicalEventExample]:
    examples = []
    for record in records:
        example = build_canonical_event_example(record)
        if example is not None:
            examples.append(example)
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
) -> dict[str, list[str]]:
    """Build a sorted value vocabulary per field from every example seen.

    Train and eval examples are both passed in so the eval split never hits an
    out-of-vocabulary label -- these are small, closed-ish category sets (see
    CANONICAL_EVENT_ALL_FIELDS), not open-ended text.
    """
    vocab: dict[str, set[str]] = {field: set() for field in CANONICAL_EVENT_ALL_FIELDS}
    for examples in example_lists:
        for example in examples:
            for field, value in example.single_labels.items():
                vocab[field].add(value)
            for field, values in example.multi_labels.items():
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
        current = self.tokenizer(item.current_state_text, max_length=self.args.max_input_length, truncation=True, add_special_tokens=True)
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
        for field in CANONICAL_EVENT_SINGLE_LABEL_FIELDS:
            row[f"label_{field}"] = self.vocab_index[field][item.single_labels[field]]
        for field in NUDGE_MULTI_LABEL_FIELDS:
            index_map = self.vocab_index[field]
            multi_hot = [0.0] * len(index_map)
            for value in item.multi_labels[field]:
                multi_hot[index_map[value]] = 1.0
            row[f"label_{field}"] = multi_hot
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
        # TextLeWorldModel.forward() unconditionally encodes a "next state";
        # canonical-event examples have no such text, so alias current state --
        # z_next is simply unused when compute_canonical_event=True.
        batch["next_input_ids"] = batch["current_input_ids"]
        batch["next_attention_mask"] = batch["current_attention_mask"]
        for field in CANONICAL_EVENT_SINGLE_LABEL_FIELDS:
            key = f"label_{field}"
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
        for field in NUDGE_MULTI_LABEL_FIELDS:
            key = f"label_{field}"
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.float)
        return batch


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
        pooling: str = "mean",
        canonical_event_vocab_sizes: dict[str, int] | None = None,
        canonical_event_head_hidden_size: int = 512,
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
        if self.latent_type != "categorical":
            # Categorical predictor emits prior logits; continuous predictor emits a normalized vector.
            predictor_layers.append(nn.LayerNorm(self.latent_dim))
        self.predictor = nn.Sequential(*predictor_layers)
        self.success_head = nn.Sequential(
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
        if canonical_event_vocab_sizes:
            self.canonical_event_trunk = nn.Sequential(
                nn.Linear(self.latent_dim * 4, self.canonical_event_head_hidden_size),
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict the next latent. Returns (latent_vector, prior_logits_or_None)."""
        predictor_inputs = [z_current, z_action, z_context]
        if self.goal_conditioning:
            predictor_inputs.append(z_goal if z_goal is not None else torch.zeros_like(z_current))
        predicted = self.predictor(torch.cat(predictor_inputs, dim=-1))
        if self.latent_type == "categorical":
            return self._sample_categorical(predicted)
        return predicted, None

    def predict_success_logit(self, z_current: torch.Tensor, z_action: torch.Tensor, z_context: torch.Tensor, z_pred: torch.Tensor) -> torch.Tensor:
        return self.success_head(torch.cat([z_current, z_action, z_context, z_pred], dim=-1)).squeeze(-1)

    def predict_canonical_event_logits(
        self, z_current: torch.Tensor, z_action: torch.Tensor, z_context: torch.Tensor, z_pred: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        features = torch.cat([z_current, z_action, z_context, z_pred], dim=-1)
        trunk_features = self.canonical_event_trunk(features)
        return {field: head(trunk_features) for field, head in self.canonical_event_heads.items()}

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        compute_reconstruction: bool = False,
        compute_success: bool = False,
        compute_canonical_event: bool = False,
    ) -> dict[str, torch.Tensor]:
        from transformers.modeling_outputs import BaseModelOutput

        z_current, _ = self.encode_latent_and_logits(batch["current_input_ids"], batch["current_attention_mask"])
        z_next, z_next_logits = self.encode_latent_and_logits(batch["next_input_ids"], batch["next_attention_mask"])
        z_context, _ = self.encode_latent_and_logits(batch["context_input_ids"], batch["context_attention_mask"])
        z_action, _ = self.encode_latent_and_logits(batch["action_input_ids"], batch["action_attention_mask"])
        if "goal_input_ids" in batch and "goal_attention_mask" in batch:
            z_goal, _ = self.encode_latent_and_logits(batch["goal_input_ids"], batch["goal_attention_mask"])
        else:
            z_goal = torch.zeros_like(z_current)
        z_pred, z_pred_logits = self.predict_latent(
            z_current, z_action, z_context, z_goal if self.goal_conditioning else None
        )
        multi_step_preds = multi_step_targets = multi_step_mask = None
        if "future_action_input_ids" in batch:
            # Recursive multi-step rollout: feed the teacher-forced future actions and keep
            # predicting the next latent from the previous PREDICTED latent. Targets are the
            # true encoded future next-states (used stop-grad in the prediction loss, and with
            # gradient in SigReg -- see compute_jepa_loss).
            future_action_ids = batch["future_action_input_ids"]
            future_action_mask = batch["future_action_attention_mask"]
            future_next_ids = batch["future_next_input_ids"]
            future_next_mask = batch["future_next_attention_mask"]
            batch_size, horizon_steps = future_action_ids.shape[0], future_action_ids.shape[1]
            latent_dim = z_pred.shape[-1]
            z_future_action = self.encode_latent_and_logits(
                future_action_ids.reshape(batch_size * horizon_steps, -1),
                future_action_mask.reshape(batch_size * horizon_steps, -1),
            )[0].reshape(batch_size, horizon_steps, latent_dim)
            multi_step_targets = self.encode_latent_and_logits(
                future_next_ids.reshape(batch_size * horizon_steps, -1),
                future_next_mask.reshape(batch_size * horizon_steps, -1),
            )[0].reshape(batch_size, horizon_steps, latent_dim)
            goal = z_goal if self.goal_conditioning else None
            rollout = z_pred
            predictions = []
            for step in range(horizon_steps):
                rollout, _ = self.predict_latent(rollout, z_future_action[:, step], z_context, goal)
                predictions.append(rollout)
            multi_step_preds = torch.stack(predictions, dim=1)  # [B, K, D]
            multi_step_mask = batch["future_step_mask"]
        success_logits = self.predict_success_logit(z_current, z_action, z_context, z_pred) if compute_success else None
        canonical_event_logits = (
            self.predict_canonical_event_logits(z_current, z_action, z_context, z_pred) if compute_canonical_event else None
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
            **({"canonical_event_logits": canonical_event_logits} if canonical_event_logits is not None else {}),
            **({"multi_step_preds": multi_step_preds, "multi_step_targets": multi_step_targets, "multi_step_mask": multi_step_mask} if multi_step_preds is not None else {}),
            "reconstruction_loss": reconstruction_loss,
            "logits": logits,
        }


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
) -> torch.Tensor:
    """Single-step latent prediction loss against the stop-gradient target z_next.

    'mse_cosine' is the legacy MSE + (1 - cosine) objective. 'smooth_l1'/'smooth_l1_cosine'
    replace the MSE term with a Smooth L1 (Huber) loss, which is less sensitive to the
    outlier latents that produced the loss spikes during training (cf. NextLat, LSE-MTP);
    the cosine term is retained unless loss_type == 'smooth_l1'.
    """
    target = z_next.detach()
    if loss_type == "mse_cosine":
        base = (z_pred - target).pow(2).mean()
    else:
        base = torch.nn.functional.smooth_l1_loss(z_pred, target, beta=beta)
    if loss_type == "smooth_l1":
        return base
    cosine = 1.0 - torch.nn.functional.cosine_similarity(z_pred, target, dim=-1).mean()
    return base + cosine


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


def canonical_event_classification_loss(
    logits: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
    """Mean per-field loss across every canonical_event_state/nudge head.

    Single-label fields use softmax cross-entropy; the multi-label
    missing_information_type field uses per-class binary cross-entropy. Each
    field is weighted equally regardless of its class count.
    """
    total: torch.Tensor | None = None
    per_field: dict[str, dict[str, torch.Tensor]] = {}
    for field, field_logits in logits.items():
        target = batch[f"label_{field}"].to(field_logits.device)
        if field in NUDGE_MULTI_LABEL_FIELDS:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(field_logits, target.to(field_logits.dtype))
            predictions = (torch.sigmoid(field_logits) >= 0.5).to(target.dtype)
            accuracy = (predictions == target).to(field_logits.dtype).mean()
        else:
            loss = torch.nn.functional.cross_entropy(field_logits, target)
            accuracy = (field_logits.argmax(dim=-1) == target).to(field_logits.dtype).mean()
        per_field[field] = {"loss": loss.detach(), "accuracy": accuracy.detach()}
        total = loss if total is None else total + loss
    total = total / max(len(logits), 1) if total is not None else next(iter(batch.values())).new_zeros(())
    return total, per_field


def compute_jepa_loss(outputs: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
            step_losses = torch.cat([step1.unsqueeze(1), future], dim=1)  # [B, 1+K]
            step_mask = torch.cat(
                [torch.ones_like(step1).unsqueeze(1), outputs["multi_step_mask"].to(step1.dtype)], dim=1
            )
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
            latent_loss = latent_prediction_loss(outputs["z_pred"], outputs["z_next"], loss_type=loss_type, beta=beta)
            sig_loss = sigreg_loss(torch.cat([outputs["z_current"], outputs["z_next"]], dim=0), args.sigreg_projections, args.sigreg_eps)
        kl_dynamics = kl_representation = None
        latent_coeff = args.latent_loss_coeff
    recon_loss = outputs["reconstruction_loss"]
    success_enabled = success_head_training_enabled(args)
    success_loss, success_accuracy = success_classification_loss(outputs) if success_enabled else (outputs["z_pred"].new_zeros(()), outputs["z_pred"].new_zeros(()))
    if getattr(args, "train_success_head_only", False):
        total = args.success_loss_coeff * success_loss
    else:
        total = latent_coeff * latent_loss + args.sigreg_coeff * sig_loss + args.reconstruction_loss_coeff * recon_loss + (args.success_loss_coeff * success_loss if success_enabled else 0.0)
    components = {"latent_loss": latent_loss, "sigreg_loss": sig_loss, "reconstruction_loss": recon_loss}
    if success_enabled:
        components["success_loss"] = success_loss
        components["success_accuracy"] = success_accuracy
    if kl_dynamics is not None:
        components["kl_dynamics"] = kl_dynamics
        components["kl_representation"] = kl_representation
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
    totals = {"loss": 0.0, "latent_loss": 0.0, "sigreg_loss": 0.0, "reconstruction_loss": 0.0, "batches": 0}
    if success_enabled:
        totals["success_loss"] = 0.0
        totals["success_accuracy"] = 0.0
    if args.latent_type == "categorical":
        totals["kl_dynamics"] = 0.0
        totals["kl_representation"] = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch(batch, device)
            outputs = model(batch, compute_reconstruction=args.reconstruction_loss_coeff > 0, compute_success=success_enabled)
            loss, components = compute_jepa_loss(outputs, args)
            totals["loss"] += float(loss.detach().cpu())
            totals["latent_loss"] += float(components["latent_loss"].detach().cpu())
            totals["sigreg_loss"] += float(components["sigreg_loss"].detach().cpu())
            totals["reconstruction_loss"] += float(components["reconstruction_loss"].detach().cpu())
            if success_enabled:
                totals["success_loss"] += float(components["success_loss"].detach().cpu())
                totals["success_accuracy"] += float(components["success_accuracy"].detach().cpu())
            if "kl_dynamics" in components:
                totals["kl_dynamics"] += float(components["kl_dynamics"].detach().cpu())
                totals["kl_representation"] += float(components["kl_representation"].detach().cpu())
            totals["batches"] += 1
    batches = max(totals.pop("batches"), 1)
    return {key: value / batches for key, value in totals.items()}


def jepa_adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in unwrap_model(model).state_dict().items() if not key.startswith("backbone.")}


def save_checkpoint_and_prune(
    model: nn.Module,
    tokenizer: Any,
    output_dir: Path,
    global_step: int,
    save_total_limit: int,
    *,
    freeze_backbone: bool,
    base_model: str,
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
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = set(getattr(incompatible, "missing_keys", []))
    unexpected_keys = set(getattr(incompatible, "unexpected_keys", []))
    allowed_missing = allowed_missing_checkpoint_keys(
        missing_keys,
        allow_missing_success_head=allow_missing_success_head,
        allow_missing_canonical_event_heads=allow_missing_canonical_event_heads,
    )
    disallowed_missing = missing_keys - allowed_missing
    if disallowed_missing or unexpected_keys:
        raise SystemExit(f"Checkpoint does not match TextLeWorldModel architecture: missing={sorted(disallowed_missing)}, unexpected={sorted(unexpected_keys)}")


def freeze_for_success_head_training(model: TextLeWorldModel) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("success_head.")
    model.deterministic_latent_sampling = True


def freeze_for_canonical_event_head_training(model: TextLeWorldModel) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("canonical_event_heads.") or name.startswith("canonical_event_trunk.")
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
        dump_agent_replay_strategy_records,
        evaluate_agent_replay,
        extract_replay_tasks_from_state_trajectories,
    )

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
        latent_plan_goal_mode=args.latent_plan_goal_mode,
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
        "latent_plan_samples": args.latent_plan_samples,
        "latent_plan_elites": args.latent_plan_elites,
        "latent_plan_iters": args.latent_plan_iters,
        "latent_plan_horizon": args.latent_plan_horizon,
        "latent_mpc_execute_steps": args.latent_mpc_execute_steps,
        "latent_plan_temperature": args.latent_plan_temperature,
        "latent_plan_score_margin": args.latent_plan_score_margin,
        "latent_plan_diversity_multiplier": args.latent_plan_diversity_multiplier,
        "latent_plan_goal_mode": args.latent_plan_goal_mode,
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

    The multi-label field additionally reports ``label_accuracy`` (the
    element-wise accuracy the training loop logs) so the two are comparable.
    """
    model.eval()
    single_fields = list(CANONICAL_EVENT_SINGLE_LABEL_FIELDS)
    multi_fields = list(NUDGE_MULTI_LABEL_FIELDS)
    all_fields = single_fields + multi_fields

    total_examples = 0
    total_loss = 0.0
    batches = 0
    correct = {field: 0 for field in all_fields}
    label_correct = {field: 0 for field in multi_fields}  # element-wise, multi-label only
    label_total = {field: 0 for field in multi_fields}
    pred_counts = {field: [0] * len(vocab[field]) for field in all_fields}
    gold_counts = {field: [0] * len(vocab[field]) for field in all_fields}

    with torch.no_grad():
        for batch in tqdm(
            eval_loader,
            total=len(eval_loader),
            desc="canonical_event_head_eval",
            disable=not is_main_process(),
        ):
            batch = move_batch(batch, device)
            logits = model(batch, compute_canonical_event=True)["canonical_event_logits"]
            loss, _ = canonical_event_classification_loss(logits, batch)
            total_loss += float(loss.detach().cpu())
            batches += 1
            total_examples += int(batch["current_input_ids"].shape[0])
            for field in single_fields:
                field_logits = logits[field]
                target = batch[f"label_{field}"].to(field_logits.device)
                preds = field_logits.argmax(dim=-1)
                correct[field] += int((preds == target).sum().cpu())
                for value in preds.cpu().tolist():
                    pred_counts[field][value] += 1
                for value in target.cpu().tolist():
                    gold_counts[field][value] += 1
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

    def distribution(counts: list[int], values: list[str], denom: int) -> dict[str, dict[str, float]]:
        return {
            values[i]: {"count": counts[i], "fraction": (counts[i] / denom if denom else 0.0)}
            for i in range(len(values))
        }

    denom = max(total_examples, 1)
    per_head: dict[str, Any] = {}
    for field in all_fields:
        entry: dict[str, Any] = {
            "accuracy": correct[field] / denom,
            "num_classes": len(vocab[field]),
            "predicted_distribution": distribution(pred_counts[field], vocab[field], total_examples),
            "gold_distribution": distribution(gold_counts[field], vocab[field], total_examples),
        }
        if field in multi_fields:
            entry["accuracy_is_exact_match"] = True
            entry["label_accuracy"] = (label_correct[field] / label_total[field]) if label_total[field] else 0.0
        per_head[field] = entry

    return {
        "loss": total_loss / max(batches, 1),
        "eval_examples": total_examples,
        "accuracy": {field: per_head[field]["accuracy"] for field in all_fields},
        "per_head": per_head,
    }


def evaluate_canonical_event_heads_by_benchmark(
    model: nn.Module,
    eval_examples: list[CanonicalEventExample],
    tokenizer: Any,
    args: argparse.Namespace,
    vocab: dict[str, list[str]],
    device: torch.device,
) -> dict[str, Any]:
    """Per-benchmark, per-field accuracy on the canonical-event eval split.

    Groups the eval examples by their ``benchmark`` field and runs the same
    per-head evaluation on each group, so accuracy can be read per field per
    benchmark (e.g. CRMArenaPro vs EnterpriseOps-Gym vs Terminal-Bench-2.0).
    Each entry keeps only the compact accuracy map (plus loss and count); the
    full category distributions stay in the overall per_head block.
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
        }
    return per_benchmark


def print_canonical_event_eval_summary(eval_metrics: dict[str, Any]) -> None:
    """Human-readable console dump of per-head accuracy and category spread."""
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
        line = f"  {field:32s} {acc_label}={entry['accuracy']:.4f} (classes={entry['num_classes']})"
        if "label_accuracy" in entry:
            line += f" label_acc={entry['label_accuracy']:.4f}"
        print(line, flush=True)
        # Show the most-predicted categories vs. gold to spot collapse at a glance.
        pred = entry.get("predicted_distribution", {})
        top = sorted(pred.items(), key=lambda kv: kv[1]["count"], reverse=True)[:5]
        gold = entry.get("gold_distribution", {})
        for value, stats in top:
            gold_frac = gold.get(value, {}).get("fraction", 0.0)
            print(
                f"      {value:28s} pred={stats['fraction']:.3f} ({stats['count']})  gold={gold_frac:.3f}",
                flush=True,
            )
    per_benchmark = eval_metrics.get("per_benchmark") or {}
    for benchmark, entry in per_benchmark.items():
        print(
            f"  [benchmark: {benchmark}] examples={entry.get('eval_examples')} loss={entry.get('loss'):.4f}",
            flush=True,
        )
        for field, accuracy in (entry.get("accuracy") or {}).items():
            print(f"      {field:32s} accuracy={accuracy:.4f}", flush=True)


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
    vocab = build_canonical_event_vocabularies(train_examples, eval_examples)
    vocab_sizes = {field: len(values) for field, values in vocab.items()}

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
        pooling=args.pooling,
        canonical_event_vocab_sizes=vocab_sizes,
        canonical_event_head_hidden_size=args.canonical_event_head_hidden_size,
    )
    load_jepa_state_dict_for_training(
        model, init_checkpoint, allow_missing_success_head=True, allow_missing_canonical_event_heads=True
    )
    freeze_for_canonical_event_head_training(model)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
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
        "fields": {"single_label": list(CANONICAL_EVENT_SINGLE_LABEL_FIELDS), "multi_label": list(NUDGE_MULTI_LABEL_FIELDS)},
        "vocab_sizes": vocab_sizes,
        "canonical_event_head_hidden_size": args.canonical_event_head_hidden_size,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "canonical_event_train_sample_percentage": args.canonical_event_train_sample_percentage,
        "canonical_event_train_sample_seed": args.canonical_event_train_sample_seed,
        "train_examples_per_benchmark_full": train_benchmark_counts_full,
        "train_examples_per_benchmark_kept": train_benchmark_counts_kept,
        "distributed": {"enabled": distributed, "world_size": int(os.environ.get("WORLD_SIZE", "1"))},
    }
    if is_main_process():
        dump_json(args.output_dir / "canonical_event_data_manifest.json", data_manifest)
        dump_json(args.output_dir / "canonical_event_vocab.json", vocab)
    if distributed:
        torch.distributed.barrier()

    global_step = 0
    log_history: list[dict[str, Any]] = []

    def forward_logits(active_model: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return active_model(batch, compute_canonical_event=True)["canonical_event_logits"]

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
                    logits = forward_logits(model, batch)
                    loss, per_field = canonical_event_classification_loss(logits, batch)
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
                    }
                    if is_main_process() and (global_step % args.logging_steps == 0 or global_step == 1):
                        print("[canonical_event_head_train] " + json.dumps(metrics, ensure_ascii=False), flush=True)
                    if is_main_process() and args.save_steps > 0 and global_step % args.save_steps == 0:
                        save_checkpoint_and_prune(
                            model, tokenizer, args.output_dir, global_step, args.save_total_limit,
                            freeze_backbone=True, base_model=str(args.model),
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
        eval_metrics = evaluate_canonical_event_heads(raw_model, eval_loader, vocab, device)
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
        # Write a self-contained architecture manifest so JepaTextWorldModelGenerator
        # can load this head checkpoint standalone in --mode=replay (the classifier
        # heads reconstruct the imagined observation/state from an action).
        dump_json(
            args.output_dir / "jepa_data_manifest.json",
            {
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
            },
        )
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

    if args.train_canonical_event_heads_only:
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
    if args.freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    # Only rank 0 pays for parsing the raw trajectory JSON -- with
    # --trajectory-dataset all/adp_all this is tens of GB of nested Python
    # objects, and doing that identically on every torchrun process can exhaust
    # host RAM; a rank getting OOM-killed then hangs every surviving rank on the
    # next NCCL collective. Other ranks instead read back the already-extracted
    # (far smaller, flat) examples rank 0 just wrote out below.
    train_examples_path = args.output_dir / "jepa_train_examples.jsonl"
    eval_examples_path = args.output_dir / "jepa_eval_examples.jsonl"
    if is_main_process():
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
            "backbone_type": args.backbone_type,
            "train": summarize_examples(train_examples),
            "eval": summarize_examples(eval_examples),
            "latent_dim": args.latent_dim,
            "pooling": args.pooling,
            "latent_type": args.latent_type,
            "latent_categoricals": args.latent_categoricals,
            "latent_classes": args.latent_classes,
            "latent_unimix": args.latent_unimix,
            "resolved_hidden_size": resolve_backbone_hidden_size(backbone),
            "memory_tokens": args.memory_tokens,
            "predictor_hidden_multiplier": args.predictor_hidden_multiplier,
            "goal_conditioning": not args.disable_goal_conditioning,
            "goal_representation": None,
            "inference_goal_representation": None,
            "history_observation_representation": "raw_observation",
            "goal_loss_coeff": 0.0,
            "goal_loss_enabled": False,
            "success_loss_coeff": args.success_loss_coeff,
            "max_input_length": args.max_input_length,
            "max_action_length": args.max_action_length,
            "max_observation_length": args.max_observation_length,
            "max_goal_length": args.max_goal_length,
            "dtype": args.dtype,
            "train_success_head_only": args.train_success_head_only,
            "initialized_from_jepa_checkpoint": str(args.jepa_checkpoint_path or args.output_dir) if args.train_success_head_only else None,
            "loss_coefficients": {"latent": args.latent_loss_coeff, "sigreg": args.sigreg_coeff, "reconstruction": args.reconstruction_loss_coeff, "kl": args.kl_loss_coeff, "success": args.success_loss_coeff},
            "kl_balance": args.kl_balance,
            "kl_free_nats": args.kl_free_nats,
            "distributed": {"enabled": distributed, "world_size": int(os.environ.get("WORLD_SIZE", "1"))},
        }
        dump_json(args.output_dir / "jepa_data_manifest.json", data_manifest)
        dump_jsonl(train_examples_path, (asdict(example) for example in train_examples))
        dump_jsonl(eval_examples_path, (asdict(example) for example in eval_examples))
    if distributed:
        torch.distributed.barrier()
        if not is_main_process():
            train_examples = [JepaExample(**row) for row in load_jsonl_rows(train_examples_path)]
            eval_examples = [JepaExample(**row) for row in load_jsonl_rows(eval_examples_path)]

    model = TextLeWorldModel(backbone=backbone, latent_dim=args.latent_dim, memory_tokens=args.memory_tokens, dropout=args.predictor_dropout, predictor_hidden_multiplier=args.predictor_hidden_multiplier, goal_conditioning=not args.disable_goal_conditioning, latent_type=args.latent_type, latent_categoricals=args.latent_categoricals, latent_classes=args.latent_classes, latent_unimix=args.latent_unimix, pooling=args.pooling)
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
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None, output_device=local_rank if torch.cuda.is_available() else None, find_unused_parameters=True)

    train_dataset = JepaTextDataset(train_examples, tokenizer, args)
    eval_dataset = JepaTextDataset(eval_examples, tokenizer, args)
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

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise SystemExit("No trainable parameters remain after applying training/freezing options.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_update_steps = max(1, math.ceil(len(train_loader) * args.num_train_epochs / max(1, args.gradient_accumulation_steps)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=min(100, max(1, total_update_steps // 20)), num_training_steps=total_update_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and torch.cuda.is_available())
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    use_amp = (args.fp16 or args.bf16) and torch.cuda.is_available()
    success_enabled = success_head_training_enabled(args)

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
                outputs = model(batch, compute_reconstruction=args.reconstruction_loss_coeff > 0, compute_success=success_enabled)
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
                    **({"kl_dynamics": float(components["kl_dynamics"].detach().cpu()), "kl_representation": float(components["kl_representation"].detach().cpu())} if "kl_dynamics" in components else {}),
                    "lr": scheduler.get_last_lr()[0],
                    "z_current_mean": float(outputs["z_current"].detach().mean().cpu()),
                    "z_current_std": float(outputs["z_current"].detach().std(unbiased=False).cpu()),
                    "z_next_mean": float(outputs["z_next"].detach().mean().cpu()),
                    "z_next_std": float(outputs["z_next"].detach().std(unbiased=False).cpu()),
                }
                if is_main_process() and (global_step % args.logging_steps == 0 or global_step == 1):
                    print("[jepa_train] " + json.dumps(metrics, ensure_ascii=False), flush=True)
                if is_main_process() and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint_and_prune(
                        model, tokenizer, args.output_dir, global_step, args.save_total_limit,
                        freeze_backbone=args.freeze_backbone, base_model=str(args.model),
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
