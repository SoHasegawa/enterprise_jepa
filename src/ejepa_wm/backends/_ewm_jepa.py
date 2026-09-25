"""JEPA world-model inference for the ``ewm_imagined`` replay loop.

This is a deliberate, inference-only **port** of the text-JEPA world model from
the EWM repo (``ewm/src/finetuning_jepa.py`` — the ``TextLeWorldModel`` net and
its checkpoint loaders — plus ``JepaTextWorldModelGenerator`` from
``ewm/src/finetuning.py``). Like the sibling ``_ewm_*`` modules it stands on its
own: it has **no imports** from an external EWM checkout and does not need one on
``sys.path``. It depends only on ``torch`` / ``transformers`` (imported at module
top, so the module is itself imported lazily — only when a JEPA generator is
actually requested — mirroring ``_ewm_k_controller``).

Why this exists: the imagined-trajectory replay loop in :mod:`_ewm_runtime`
(``imagine_trajectory`` / ``imagine_trajectory_topk_search`` →
``predict_wm_feedback``) is world-model-agnostic. In the text-LLM path the world
model is :class:`_ewm_runtime.EwmGenerator` (a vLLM chat client whose text output
is parsed). Here the world model is a **JEPA** net that predicts the outcome of a
planned action *in latent space* and reconstructs an imagined observation/state
from its heads. :class:`JepaEwmGenerator` exposes the same
``predict_feedback(...)`` seam the rollout loop can call, so swapping in JEPA is a
generator swap — the agent stays a text LLM; only the world model changes.

Three imagined-observation backends, resolved from the loaded checkpoint and the
``imagined_observation_backend`` axis (``auto`` prefers them in this order):

* ``canonical_event`` — classification heads predict a canonical outcome
  (``execution_status`` / ``risk_signal`` / ``nudge`` / ...); reconstructed into
  a ``{"state": {"context": ...}}`` envelope the replay code understands.
* ``success`` — a success head emits ``P(action succeeds)``; turned into an
  ``ewm_classifier_observation_v1`` payload.
* ``decoder`` — a seq2seq backbone decodes the predicted latent back to raw
  tool-output text (``text_leworldmodel_jepa_decoder_fallback``); failure is
  inferred heuristically.
"""
from __future__ import annotations

import collections
import contextlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn

from ejepa_wm.backends import _ewm_finetuning as ft
from ejepa_wm.backends import _ewm_runtime as ewm

logger = logging.getLogger(__name__)

# Cap on how many past action/observation entries the JEPA model conditions on;
# matches ``finetuning.WORLD_MODEL_INPUT_HISTORY_SIZE``.
WORLD_MODEL_INPUT_HISTORY_SIZE = 8

# Bound for JepaEwmGenerator._encode_latent_text's text->latent memo. Entries are [1, D]
# tensors; a task only ever needs a handful (context, goal, the recent state texts).
LATENT_TEXT_CACHE_SIZE = 32

# Opt-in inference acceleration (``WM_JEPA_COMPILE``). The 0.6B text backbone is
# launch-overhead bound at the ~300-token inputs a scoring call sees (two passes take
# ~42 ms of a ~59 ms call), so the win is CUDA-graph replay, which needs recurring
# shapes: token lengths are bucket-padded to ``WM_JEPA_PAD_MULTIPLE`` and batches to a
# power of two. Padding is masked out by the pooling, so latents are unchanged.
#   WM_JEPA_COMPILE=1|graphs   torch.compile(mode="reduce-overhead") backbone+predictor+heads
#   WM_JEPA_COMPILE=default    torch.compile(mode="default"), kernel fusion only, no graphs
#   unset / 0                  eager (the default)
FAST_INFERENCE_ENV = "WM_JEPA_COMPILE"
FAST_INFERENCE_PAD_MULTIPLE_ENV = "WM_JEPA_PAD_MULTIPLE"

# Prediction-ablation controls for the beam planner ("is the planner doing the
# work?"). The planner, its utilities and vetoes run unchanged; only the per-step
# canonical-event distributions (and terminal probabilities) it consumes change:
#   WM_JEPA_PREDICTION_CONTROL=shuffled  the model's own rows, randomly permuted
#                                        across (plan, step) within the call --
#                                        same marginals, wrong assignment
#   WM_JEPA_PREDICTION_CONTROL=uniform   1/K over each field's classes (0.5 for
#                                        the multi-label field and terminal)
#   WM_JEPA_PREDICTION_CONTROL=prior     training-set class priors from the JSON
#                                        at WM_JEPA_PRIOR_PATH, no input conditioning
#   WM_JEPA_PREDICTION_CONTROL=no_state  no predicted state reaches the planner at all:
#                                        every candidate gets an identical constant row
#                                        and terminal probability zero, so scores tie and
#                                        the stable sort in rank_trajectories falls
#                                        through to the policy's own first candidate.
#                                        Candidate generation, the reflection and critic
#                                        instructions, the refinement rounds and the
#                                        policy-call budget are untouched, which isolates
#                                        the scaffolding from the learned predictions.
#                                        Distinct from `uniform`, which also supplies a
#                                        constant row but keeps terminal advice live at
#                                        probability 0.5.
#   unset / none                         real predictions (the default)
PREDICTION_CONTROL_ENV = "WM_JEPA_PREDICTION_CONTROL"
PREDICTION_PRIOR_PATH_ENV = "WM_JEPA_PRIOR_PATH"
PREDICTION_CONTROL_SEED_ENV = "WM_JEPA_CONTROL_SEED"
PREDICTION_CONTROLS = ("shuffled", "uniform", "prior", "no_state")


def resolve_prediction_control(value: Optional[str] = None) -> Optional[str]:
    raw = (os.environ.get(PREDICTION_CONTROL_ENV, "") if value is None else value).strip().lower()
    if raw in ("", "0", "none", "off", "real"):
        return None
    if raw in PREDICTION_CONTROLS:
        return raw
    raise ValueError(f"{PREDICTION_CONTROL_ENV}={raw!r}; expected one of {PREDICTION_CONTROLS}")


def load_class_priors(path: Optional[str] = None) -> Dict[str, Any]:
    """``{"fields": {field: {class: prob}}, "terminal_probability": p}`` as written by
    the prior-extraction step (see ``results/analysis/canonical_event_class_priors.json``)."""
    target = path if path is not None else os.environ.get(PREDICTION_PRIOR_PATH_ENV, "")
    if not target:
        raise ValueError(f"{PREDICTION_CONTROL_ENV}=prior requires {PREDICTION_PRIOR_PATH_ENV}")
    with open(target, encoding="utf-8") as handle:
        data = json.load(handle)
    if "fields" not in data:
        raise ValueError(f"{target}: missing 'fields'")
    return data


def apply_prediction_control(
    trajectories: List[List[Dict[str, Dict[str, float]]]],
    terminal_steps: List[List[float]],
    mode: Optional[str],
    *,
    vocab: Dict[str, List[str]],
    priors: Optional[Dict[str, Any]] = None,
    rng: Optional[Any] = None,
    multi_label_fields: tuple = ("missing_information_type",),
) -> tuple:
    """Return control versions of the per-step prediction rows (see the module note)."""
    if mode is None:
        return trajectories, terminal_steps
    if mode == "shuffled":
        import random

        rng = rng or random.Random(0)
        slots = [(p, t) for p, traj in enumerate(trajectories) for t in range(len(traj))]
        rows = [trajectories[p][t] for p, t in slots]
        terms = [
            terminal_steps[p][t] if p < len(terminal_steps) and t < len(terminal_steps[p]) else None
            for p, t in slots
        ]
        order = list(range(len(slots)))
        rng.shuffle(order)
        new_traj = [list(traj) for traj in trajectories]
        new_term = [list(ts) for ts in terminal_steps]
        for (p, t), src in zip(slots, order):
            new_traj[p][t] = rows[src]
            if terms[src] is not None and p < len(new_term) and t < len(new_term[p]):
                new_term[p][t] = terms[src]
        return new_traj, new_term
    if mode in ("uniform", "no_state"):
        row = {
            field: {c: (0.5 if field in multi_label_fields else 1.0 / len(classes)) for c in classes}
            for field, classes in vocab.items()
            if classes
        }
        # no_state withholds terminal advice too: a terminal probability is a prediction.
        term_p = 0.0 if mode == "no_state" else 0.5
    elif mode == "prior":
        if priors is None:
            raise ValueError("prior control needs class priors")
        fields = priors["fields"]
        row = {
            field: {c: float(fields.get(field, {}).get(c, 1.0 / len(classes))) for c in classes}
            for field, classes in vocab.items()
            if classes
        }
        term_p = float(priors.get("terminal_probability", 0.5))
    else:
        raise ValueError(f"unknown prediction control {mode!r}")
    new_traj = [[dict(row) for _ in traj] for traj in trajectories]
    new_term = [[term_p for _ in ts] for ts in terminal_steps]
    return new_traj, new_term


def resolve_fast_inference_mode(value: Optional[str] = None) -> Optional[str]:
    raw = (os.environ.get(FAST_INFERENCE_ENV, "") if value is None else value).strip().lower()
    if raw in ("", "0", "false", "off", "no", "eager"):
        return None
    if raw in ("1", "true", "on", "yes", "graphs", "reduce-overhead"):
        return "reduce-overhead"
    if raw in ("default", "max-autotune", "max-autotune-no-cudagraphs"):
        return raw
    raise ValueError(f"{FAST_INFERENCE_ENV}={raw!r} is not a recognised torch.compile mode")


def pad_to_multiple(
    encoded: Dict[str, "torch.Tensor"], multiple: int, max_length: int, pad_id: int
) -> Dict[str, "torch.Tensor"]:
    """Right-pad ``input_ids``/``attention_mask`` up to the next multiple of ``multiple``
    tokens (never beyond ``max_length``); padded positions carry attention_mask 0."""
    ids = encoded.get("input_ids")
    if ids is None or multiple <= 1:
        return encoded
    target = min(-(-ids.shape[-1] // multiple) * multiple, int(max_length))
    extra = target - ids.shape[-1]
    if extra <= 0:
        return encoded
    out = dict(encoded)
    out["input_ids"] = torch.nn.functional.pad(ids, (0, extra), value=pad_id)
    if "attention_mask" in out:
        out["attention_mask"] = torch.nn.functional.pad(out["attention_mask"], (0, extra), value=0)
    return out


class _CompiledChild(nn.Module):
    """Holds a ``torch.compile``d child module and clones its tensor outputs.

    Under ``reduce-overhead`` the CUDA-graph output buffers are overwritten by the
    next replay, while the scorer keeps earlier outputs (cached latents) alive, so
    outputs are cloned (a small device copy). Attribute access falls through to the
    wrapped module (``config``, ``generate`` ...), and ``get_encoder`` returns the
    wrapper itself so decoder-only backbones stay on the compiled path.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return _clone_tensors(self.inner(*args, **kwargs))

    def get_encoder(self) -> Any:
        return self

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


def _clone_tensors(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(_clone_tensors(o) for o in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone_tensors(o) for o in obj)
    if isinstance(obj, dict):  # includes HF ModelOutput
        for key in list(obj.keys()):
            obj[key] = _clone_tensors(obj[key])
        return obj
    return obj

# --- canonical-event vocab layout (ported from finetuning_jepa.py) -----------
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

# Maps the classifier's execution_status vocab onto the world-model state's
# ternary context.last_tool_execution_result (1 success / 0 stagnation / -1 failure).
CANONICAL_EVENT_EXECUTION_STATUS_TO_TERNARY: Dict[str, Optional[int]] = {
    "success": 1,
    "partial": 0,
    "no_op": 0,
    "failure": -1,
    "unknown": None,
}
CANONICAL_EVENT_CLASSIFIER_STATE_SCHEMA = "ewm_canonical_event_classifier_state_v1"
CANONICAL_EVENT_CLASSIFIER_OBSERVATION_SCHEMA = "ewm_canonical_event_classifier_observation_v1"


# ---------------------------------------------------------------------------
# Small text helpers (ported)
# ---------------------------------------------------------------------------


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _truncate_text(text: str, char_limit: int) -> str:
    if char_limit <= 0 or len(text) <= char_limit:
        return text
    return text[:char_limit] + "... [truncated]"


def _extend_batch_frame_history(
    frame_history: Optional[tuple],
    num_total: int,
    active_index: "torch.Tensor",
    new_frame_active: "torch.Tensor",
    new_action_active: "torch.Tensor",
    device: Any,
) -> tuple:
    """Append one autoregressive time-step to a (num_total-sized) batched ``frame_history``,
    filling only ``active_index`` with the just-predicted (event, producing-action) pair and
    marking the rest as invalid padding -- so every candidate's history tensor stays rectangular
    across a multi-step rollout where candidates (shorter plans) finish at different depths.
    Generalizes ``finetuning_jepa.TextLeWorldModel._extend_frame_history`` (single, always-active
    step) to ``score_action_plans_canonical_event``'s/``hier_latent_cem``'s batched-candidate case.
    """
    latent_dim = new_frame_active.shape[-1]
    cond_dim = new_action_active.shape[-1]
    new_frame_full = torch.zeros(num_total, latent_dim, device=device, dtype=new_frame_active.dtype)
    new_action_full = torch.zeros(num_total, cond_dim, device=device, dtype=new_action_active.dtype)
    new_valid_full = torch.zeros(num_total, dtype=torch.bool, device=device)
    new_frame_full[active_index] = new_frame_active
    new_action_full[active_index] = new_action_active
    new_valid_full[active_index] = True
    frame, action, valid_col = (
        new_frame_full.unsqueeze(1),
        new_action_full.unsqueeze(1),
        new_valid_full.unsqueeze(1),
    )
    if frame_history is None:
        return frame, action, valid_col
    frames, actions, valid = frame_history
    return (
        torch.cat([frames, frame], dim=1),
        torch.cat([actions, action], dim=1),
        torch.cat([valid, valid_col], dim=1),
    )


def strip_model_thinking_output(text: str, *, special_tokens: Optional[Iterable[str]] = None) -> str:
    """Remove model reasoning traces while preserving the final answer.

    Fuller variant of :func:`_ewm_runtime.strip_model_thinking_output` that also
    strips residual special tokens (needed because the JEPA decoder decodes with
    Gemma thought channels and can leave special tokens behind).
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    had_model_turn_prefix = cleaned.startswith("<|turn>model\n") or cleaned.startswith("<turn|>model\n")
    cleaned = re.sub(r"<\|channel>thought\s*.*?<channel\|>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if special_tokens:
        for token in sorted({t for t in special_tokens if t}, key=len, reverse=True):
            cleaned = cleaned.replace(token, "")
    if had_model_turn_prefix and cleaned.startswith("model\n"):
        cleaned = cleaned[len("model\n") :]
    return cleaned.strip()


def render_raw_replay_history(history: List[Dict[str, Any]], max_chars: int) -> str:
    """Render recent action/observation history into the model-input text shape
    the JEPA world model was trained on (ported from finetuning_jepa.py).

    Reads ``step``/``imagined step`` and ``action``, and takes ``observation`` (or
    ``state`` when there is no explicit observation), so it renders both real
    replay entries and imagined ``{imagined step, action, state}`` entries.
    """
    rows = []
    for item in history[-WORLD_MODEL_INPUT_HISTORY_SIZE:]:
        if not isinstance(item, dict):
            continue
        observation = item.get("observation")
        if observation is None and "state" in item:
            observation = item.get("state")
        rows.append(
            {
                "step": item.get("step", item.get("imagined step")),
                "action": item.get("action"),
                "observation": _truncate_text(ewm.stringify_tool_output(observation or ""), max_chars),
            }
        )
    return _truncate_text(_safe_json(rows), max_chars * WORLD_MODEL_INPUT_HISTORY_SIZE)


# ---------------------------------------------------------------------------
# Canonical-event decoding / state reconstruction (ported from finetuning_jepa.py)
# ---------------------------------------------------------------------------


def load_canonical_event_vocab(checkpoint_path: str | Path) -> Dict[str, List[str]]:
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
    logits: Dict[str, "torch.Tensor"],
    vocab: Dict[str, List[str]],
    *,
    multi_label_threshold: float = 0.5,
) -> Dict[str, Any]:
    labels: Dict[str, Any] = {}
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


def canonical_event_execution_label(execution_status: Any) -> Optional[int]:
    if execution_status is None:
        return None
    return CANONICAL_EVENT_EXECUTION_STATUS_TO_TERNARY.get(str(execution_status).strip().lower())


def reconstruct_state_from_canonical_event_labels(
    labels: Dict[str, Any],
    *,
    tool_name: Optional[str] = None,
) -> Dict[str, Any]:
    canonical_event_state = {field: labels.get(field) for field in CANONICAL_EVENT_STATE_FIELDS}
    nudge: Dict[str, Any] = {field: labels.get(field) for field in NUDGE_SINGLE_LABEL_FIELDS}
    for field in NUDGE_MULTI_LABEL_FIELDS:
        value = labels.get(field)
        nudge[field] = list(value) if isinstance(value, list) else ([] if value is None else [value])

    label = canonical_event_execution_label(labels.get("execution_status"))
    error_signature = labels.get("error_signature")
    error_message = ""
    if label in (-1, 0):
        parts: List[str] = []
        execution_status = labels.get("execution_status")
        if execution_status:
            parts.append(f"execution_status={execution_status}")
        if error_signature and str(error_signature).strip().lower() not in {"none", "unknown"}:
            parts.append(f"error_signature={error_signature}")
        error_message = "; ".join(parts)

    context: Dict[str, Any] = {"last_tool_execution_result": label}
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


def build_canonical_event_observation_payload(
    labels: Dict[str, Any],
    *,
    tool_name: Optional[str] = None,
) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Backbone / dtype loaders (ported from finetuning_jepa.py / finetuning.py)
# ---------------------------------------------------------------------------


def resolve_torch_dtype(dtype_name: Optional[str]) -> Any:
    if not dtype_name or dtype_name == "auto":
        return "auto"
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def backbone_encoder(backbone: Any) -> Any:
    return backbone.get_encoder() if hasattr(backbone, "get_encoder") else backbone


def backbone_supports_reconstruction(backbone: Any) -> bool:
    return hasattr(backbone, "get_encoder") and callable(getattr(backbone, "generate", None))


def _hidden_size_from_mapping(config: Dict[str, Any]) -> Optional[int]:
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


def _hidden_size_from_config(config: Any) -> Optional[int]:
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
    dtype: str | Any = "auto",
) -> Any:
    from transformers import AutoModel, AutoModelForSeq2SeqLM

    model_cls = AutoModelForSeq2SeqLM if backbone_type == "seq2seq" else AutoModel
    return model_cls.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# --predictor-arch transformer (AdaLN causal-attention predictor), ported verbatim.
# Only the ``frame_history=None`` (degenerate ``[context, z_current]``) path is exercised by
# this port's callers today -- see :meth:`TextLeWorldModel.predict_latent`'s docstring. The
# keyword-only, optional ``frame_history`` param is kept so the interface can grow into real
# multi-step history later without another signature change, exactly as upstream intends.
# ---------------------------------------------------------------------------


def _modulate(x: "torch.Tensor", shift: "torch.Tensor", scale: "torch.Tensor") -> "torch.Tensor":
    """DiT-style AdaLN modulation. scale/shift are zero at init, so this starts as identity."""
    return x * (1.0 + scale) + shift


class AdaLNPredictorBlock(nn.Module):
    """One causal-attention + MLP block with AdaLN-zero conditioning.

    The conditioning vector produces six modulation tensors (shift/scale/gate for the attention
    sub-layer and for the MLP sub-layer) through a SiLU + Linear whose weight AND bias are
    ZERO-initialized. At step 0 that makes scale=shift=0 (modulation is the identity) and gate=0
    (both residual branches contribute nothing), so the block is an exact identity and action
    conditioning ramps in progressively as training moves the modulation weights off zero.
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

    def forward(self, x: "torch.Tensor", cond: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_modulation(cond).chunk(6, dim=-1)
        )
        normed = _modulate(self.norm1(x), shift_attn, scale_attn)
        attended, _ = self.attention(normed, normed, normed, attn_mask=attention_mask, need_weights=False)
        x = x + gate_attn * attended
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class AdaLNTransformerPredictor(nn.Module):
    """LeWorldModel-style autoregressive predictor (``--predictor-arch transformer``).

    Learned positional embeddings added at input width, a stack of AdaLN-zero conditional
    blocks with causal attention, a plain final LayerNorm, and an output projector.

    Sequence layout -- N+1 representation tokens, no action tokens:

        tokens : [ E(sys + task prompt) , e_{t-N+1} , ... , e_{t-1} ,      e_t          ]
        AdaLN  : [ NULL                 , u_{t-N+2} , ... , u_t      , candidate u_{t+1} ]

    The action is never a token; it enters only as AdaLN conditioning, position-aligned: the
    token holding event e_i is modulated by the action taken FROM e_i (the one that produced
    e_{i+1}). The context token gets a learned NULL conditioning, except when it is also the
    last position (a first step with no history), where it takes the candidate action.

    Only the LAST position's output is read. Two outputs: the hidden state h_t at the last
    position IS the belief state, and the event representation is its projection,
    e_hat_{t+1} = W_pred h_t.
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
                f"predictor_transformer_dim={dim} must be divisible by predictor_transformer_heads={heads}."
            )
        self.dim = int(dim)
        self.heads = int(heads)
        self.max_positions = int(max_positions)
        self.position_embedding = nn.Parameter(torch.randn(1, self.max_positions, latent_dim) * 0.02)
        self.embedding_dropout = nn.Dropout(dropout)
        self.input_projection = nn.Linear(latent_dim, dim)
        self.cond_dim = latent_dim * cond_inputs
        self.cond_projection = nn.Linear(self.cond_dim, dim)
        self.null_conditioning = nn.Parameter(torch.zeros(1, 1, self.cond_dim))
        self.blocks = nn.ModuleList(
            AdaLNPredictorBlock(dim, heads, dropout, mlp_ratio) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        self.projector = (
            nn.Sequential(nn.Linear(dim, latent_dim), nn.LayerNorm(latent_dim))
            if output_layernorm
            else nn.Linear(dim, latent_dim)
        )

    def _attention_mask(self, valid: "torch.Tensor") -> "torch.Tensor":
        """[B*heads, T, T] bool mask, True = BLOCKED. Causal, plus padded keys, minus the
        diagonal (always allowed) so no row is fully blocked."""
        length = valid.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=valid.device).tril()
        allowed = causal.unsqueeze(0) & valid.unsqueeze(1)  # [B, T, T]
        allowed = allowed | torch.eye(length, dtype=torch.bool, device=valid.device).unsqueeze(0)
        return (~allowed).repeat_interleave(self.heads, dim=0)

    def forward(
        self, tokens: "torch.Tensor", cond: "torch.Tensor", valid: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """tokens: [B, T, latent_dim] = [context, events...]; cond: [B, T, cond_dim] with one
        action per position; valid: [B, T] bool. Returns (event, state)."""
        length = tokens.shape[1]
        if length > self.max_positions:
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


# ---------------------------------------------------------------------------
# The JEPA net (inference subset, ported verbatim from finetuning_jepa.py)
# ---------------------------------------------------------------------------


class TextLeWorldModel(nn.Module):
    """Text-JEPA world model: encode text→latent, predict the next latent, and
    read an imagined observation/state off the reconstruction/classification heads.

    Only the inference surface is ported (``encode_latent``, ``predict_latent``,
    ``predict_success_logit``, ``predict_terminal_logit``,
    ``predict_canonical_event_logits``, ``memory_projection``). ``__init__`` still constructs
    every submodule so a trained ``text_leworldmodel.pt`` state dict loads without unexpected keys.
    """

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
        canonical_event_vocab_sizes: Optional[Dict[str, int]] = None,
        canonical_event_head_hidden_size: int = 512,
        canonical_event_head_inputs: str = "all",
        predictor_arch: str = "mlp",
        predictor_transformer_dim: int = 0,
        predictor_transformer_layers: int = 6,
        predictor_transformer_heads: int = 16,
        predictor_transformer_mlp_ratio: float = 4.0,
        predictor_history_length: int = 0,
        terminal_head: bool = False,
        value_head: bool = False,
        action_decoder: bool = False,
        action_decoder_max_noise_std: float = 0.1,
        action_decoder_dim: int = 256,
        action_decoder_layers: int = 4,
        action_decoder_heads: int = 4,
        action_decoder_memory_tokens: int = 8,
        action_decoder_max_length: int = 512,
        obs_grounding: bool = False,
        obs_ground_decoder_dim: int = 256,
        obs_ground_decoder_layers: int = 4,
        obs_ground_decoder_heads: int = 4,
        obs_ground_decoder_memory_tokens: int = 8,
        obs_ground_decoder_max_length: int = 128,
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
            self.latent_dim = self.latent_categoricals * self.latent_classes
        else:
            self.latent_dim = int(latent_dim or self.hidden_size)
        self.memory_tokens = int(memory_tokens)
        self.goal_conditioning = bool(goal_conditioning)
        if self.latent_type == "categorical":
            self.encoder_projector = nn.Linear(self.hidden_size, self.latent_dim)
        else:
            self.encoder_projector = nn.Sequential(
                nn.Linear(self.hidden_size, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
            )
        # Delta encoding / Δz supervision: the predictor emits the CHANGE Δz and predict_latent
        # returns z_current + Δz. Continuous latents only -- a delta between straight-through
        # one-hot stacks is not a point on the categorical simplex.
        self.latent_delta_prediction = bool(latent_delta_prediction) and self.latent_type != "categorical"
        predictor_hidden = max(self.latent_dim, int(self.latent_dim * predictor_hidden_multiplier))
        predictor_inputs = 4 if self.goal_conditioning else 3
        predictor_layers: List[nn.Module] = [
            nn.Linear(self.latent_dim * predictor_inputs, predictor_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, predictor_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, self.latent_dim),
        ]
        # Categorical predictor emits prior logits; continuous predictor emits a normalized
        # vector. In delta mode the output must NOT be LayerNormed: LayerNorm pins the output
        # norm to ~sqrt(latent_dim), which would make small deltas (near-identity transitions --
        # the common case for consecutive tool-use states) unrepresentable.
        output_layernorm = self.latent_type != "categorical" and not self.latent_delta_prediction
        if output_layernorm:
            predictor_layers.append(nn.LayerNorm(self.latent_dim))
        # --predictor-arch: `mlp` keeps the concat-MLP above; `transformer` swaps in the
        # LeWorldModel-style causal transformer with AdaLN action conditioning. Only one is
        # built, so a checkpoint's state dict is unambiguous about which it was.
        self.predictor_arch = str(predictor_arch or "mlp")
        if self.predictor_arch not in ("mlp", "transformer"):
            raise ValueError(f"unknown predictor_arch={self.predictor_arch!r}")
        # N = tool-output representations the predictor attends over, including the current one.
        # Sets max_positions so a checkpoint's position-embedding shape matches on load; the real
        # multi-step frame_history threaded by beam_plan/hier_latent_cem is windowed to this size
        # (see predict_latent's docstring and _predict_latent_transformer's window-trim).
        self.predictor_history_length = int(predictor_history_length or WORLD_MODEL_INPUT_HISTORY_SIZE + 1)
        if self.predictor_arch == "transformer":
            self.predictor = AdaLNTransformerPredictor(
                latent_dim=self.latent_dim,
                dim=int(predictor_transformer_dim or self.latent_dim),
                layers=int(predictor_transformer_layers),
                heads=int(predictor_transformer_heads),
                dropout=dropout,
                cond_inputs=2 if self.goal_conditioning else 1,
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
        # Optional terminal-step predictor: P(done after this action). Beam planning uses
        # it only as an advisory harness, never as a hard stop.
        self.terminal_head_enabled = bool(terminal_head)
        if self.terminal_head_enabled:
            self.terminal_head = nn.Sequential(
                nn.Linear(self.latent_dim * 4, predictor_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(predictor_hidden, 1),
            )
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
        canonical_event_vocab_sizes = canonical_event_vocab_sizes or {}
        self.canonical_event_head_hidden_size = int(canonical_event_head_hidden_size)
        # Which latents the canonical-event readout sees (ported from finetuning_jepa.py):
        # "all" (default/historical) = [z_current, z_action, z_context, z_pred]; "ctx_pred" =
        # [z_context, z_pred]; "pred_only" = [z_pred]; "state" reads the transformer predictor's
        # belief state h_t directly (predict_latent_with_state's ``state`` return) -- only valid
        # with predictor_arch="transformer", and sized by the predictor's own width, not latent_dim;
        # "state_action" = [h_t, z_action] -- same belief state plus the (already-projected)
        # action latent, for fields that are properties of the tool call itself (action_type,
        # object_type, side_effect_type, risk_signal) rather than of the predicted outcome. h_t
        # is conditioned on the action but has to spend capacity re-deriving it, so making it
        # explicit is cheap -- at the cost of reopening a readout path around the predictor for
        # action-identity information (not for the outcome).
        self.canonical_event_head_inputs = str(canonical_event_head_inputs or "all")
        _head_input_counts = {"all": 4, "ctx_pred": 2, "pred_only": 1, "state": 1, "state_action": 1}
        if self.canonical_event_head_inputs not in _head_input_counts:
            raise ValueError(f"unknown canonical_event_head_inputs={self.canonical_event_head_inputs!r}")
        if self.canonical_event_head_inputs in {"state", "state_action"} and self.predictor_arch != "transformer":
            raise ValueError(
                f"canonical_event_head_inputs={self.canonical_event_head_inputs!r} requires "
                "predictor_arch='transformer': the MLP predictor has no hidden state distinct "
                "from its output (use 'pred_only' for the equivalent readout there)."
            )
        if self.canonical_event_head_inputs in {"state", "state_action"}:
            # h_t is the predictor's width, NOT latent_dim; "state_action" appends the (already
            # latent_dim-wide) action latent, so the two widths add rather than multiply.
            canonical_trunk_input_dim = self.predictor.dim
            if self.canonical_event_head_inputs == "state_action":
                canonical_trunk_input_dim += self.latent_dim
        else:
            canonical_trunk_input_dim = self.latent_dim * _head_input_counts[self.canonical_event_head_inputs]
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
        # Learnable action DECODER: reconstructs an action's own tokens from its z_action, used
        # by the hierarchical latent-action CEM (_ewm_hier_cem.py) to decode a CEM-sampled/
        # interpolated latent that doesn't correspond exactly to any LLM-proposed anchor.
        # Encoder-decoder backbones (T5Gemma, ...) reuse their NATIVE decoder via their own
        # memory_projection (never shared with the observation decoder's -- z_action and z_pred
        # are different latents). Encoder-only backbones have no native decoder, so this builds
        # a small causal Transformer decoder cross-attending over memory_tokens expanded from
        # z_action, with its output projection tied to the frozen input embeddings.
        self.action_decoder = bool(action_decoder)
        self.action_decoder_max_noise_std = float(action_decoder_max_noise_std)
        self.action_decoder_max_length = int(action_decoder_max_length)
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
                self.action_decoder_start = nn.Parameter(torch.zeros(self.hidden_size))
                self.action_decoder_token_in = nn.Linear(self.hidden_size, ad_dim)
                self.action_decoder_pos = nn.Embedding(self.action_decoder_max_length, ad_dim)
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=ad_dim, nhead=int(action_decoder_heads), dim_feedforward=ad_dim * 4,
                    dropout=dropout, batch_first=True, activation="gelu",
                )
                self.action_decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=int(action_decoder_layers))
                self.action_decoder_out = nn.Linear(ad_dim, self.hidden_size)
        # Observation-grounding DECODER: reconstructs the predicted resulting OBSERVATION
        # (tool-output text) from z_pred -- the counterpart of action_decoder (which decodes
        # z_action), used to show a beam_plan/hier_latent_cem trajectory's imagined tool output,
        # not just its canonical-event labels. Same two branches as action_decoder: seq2seq
        # backbones reuse their native decoder via their OWN obs_ground_memory_projection (never
        # shared with action_decoder's or the general memory_projection's -- z_action/z_pred/the
        # general reconstruction target are different latents/targets); encoder-only backbones
        # get their own causal Transformer decoder cross-attending over memory slots expanded
        # from z_pred, output-tied to the frozen backbone embeddings, mirroring action_decoder's
        # encoder-only branch exactly (obs_ground_* instead of action_decoder_*).
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
                self.obs_ground_decoder_max_length = int(obs_ground_decoder_max_length)
                self.obs_ground_memory_expand = nn.Sequential(
                    nn.Linear(self.latent_dim, self.obs_ground_decoder_memory_tokens * og_dim), nn.GELU(),
                    nn.Linear(self.obs_ground_decoder_memory_tokens * og_dim, self.obs_ground_decoder_memory_tokens * og_dim),
                )
                self.obs_ground_start = nn.Parameter(torch.zeros(self.hidden_size))
                self.obs_ground_token_in = nn.Linear(self.hidden_size, og_dim)
                self.obs_ground_pos = nn.Embedding(self.obs_ground_decoder_max_length, og_dim)
                og_decoder_layer = nn.TransformerDecoderLayer(
                    d_model=og_dim, nhead=int(obs_ground_decoder_heads), dim_feedforward=og_dim * 4,
                    dropout=dropout, batch_first=True, activation="gelu",
                )
                self.obs_ground_transformer = nn.TransformerDecoder(og_decoder_layer, num_layers=int(obs_ground_decoder_layers))
                self.obs_ground_out = nn.Linear(og_dim, self.hidden_size)
        self.deterministic_latent_sampling = False

    def mean_pool(self, hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def last_token_pool(self, hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        left_padded = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padded:
            return hidden[:, -1]
        last_index = attention_mask.to(torch.long).sum(dim=1) - 1
        last_index = last_index.clamp_min(0)
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_index, last_index]

    def pool(self, hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        if getattr(self, "pooling", "mean") == "last_token":
            return self.last_token_pool(hidden, attention_mask)
        return self.mean_pool(hidden, attention_mask)

    def backbone_requires_grad(self) -> bool:
        return any(param.requires_grad for param in self.backbone.parameters())

    def _encode_pooled(self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
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

    def _sample_categorical(self, logits: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
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

    def _project_latent(self, pooled: "torch.Tensor") -> tuple["torch.Tensor", Optional["torch.Tensor"]]:
        projected = self.encoder_projector(pooled)
        if self.latent_type == "categorical":
            return self._sample_categorical(projected)
        return projected, None

    def encode_latent(self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        latent, _ = self._project_latent(self._encode_pooled(input_ids, attention_mask))
        return latent

    def encode_latent_and_logits(
        self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor"
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """Same as :meth:`encode_latent` but also returns the (categorical-latent-only) prior
        logits. Used by ``_ewm_hier_cem`` to encode LLM-proposed anchor actions to latents."""
        return self._project_latent(self._encode_pooled(input_ids, attention_mask))

    def _predictor_conditioning(
        self, z_action: "torch.Tensor", z_context: "torch.Tensor", z_goal: Optional["torch.Tensor"]
    ) -> "torch.Tensor":
        """What the MLP predictor concatenates with the frame: action, task context, and the
        goal when goal-conditioned (the transformer feeds the same triple to AdaLN instead)."""
        parts = [z_action, z_context]
        if self.goal_conditioning:
            parts.append(z_goal if z_goal is not None else torch.zeros_like(z_action))
        return torch.cat(parts, dim=-1)

    def _frame_conditioning(
        self, z_action: "torch.Tensor", z_goal: Optional["torch.Tensor"]
    ) -> "torch.Tensor":
        """Per-position AdaLN vector for one historical transition: the action alone, plus goal
        when goal-conditioned -- NOT ``z_context``, which is its own token rather than per-
        position conditioning. Matches the width ``frame_history``'s ``producing_actions`` must
        have (see ``encode_frame_history`` in ``finetuning_jepa.py``)."""
        if not self.goal_conditioning:
            return z_action
        goal = z_goal if z_goal is not None else torch.zeros_like(z_action)
        return torch.cat([z_action, goal], dim=-1)

    def _predict_latent_transformer(
        self,
        z_current: "torch.Tensor",
        z_action: "torch.Tensor",
        z_context: "torch.Tensor",
        z_goal: Optional["torch.Tensor"],
        frame_history: Optional[tuple] = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Assemble [E(context), e_{t-N+1} ... e_t] with position-aligned action conditioning
        and run the AdaLN transformer. Without ``frame_history`` the sequence degenerates to
        exactly [context, z_current], both well-defined positions -- see the class/method
        docstrings this mirrors in ``finetuning_jepa.py``."""
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
            frame_cond = torch.cat([producing_actions[:, 1:], candidate], dim=1)
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

    def predict_latent(
        self,
        z_current: "torch.Tensor",
        z_action: "torch.Tensor",
        z_context: "torch.Tensor",
        z_goal: Optional["torch.Tensor"] = None,
        *,
        frame_history: Optional[tuple] = None,
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """Predict the next latent. Returns (latent_vector, prior_logits_or_None).

        ``frame_history`` is ``(frames [B,T,D], producing_actions [B,T,C], valid [B,T])``, used
        only by ``predictor_arch="transformer"``: ``frames[k]`` is the tool-output representation
        at step k -- its LAST entry stands in for "now" (the same role ``z_current`` plays in the
        degenerate case below) -- and ``producing_actions[k]`` is the action that produced it.
        ``z_current`` is otherwise unused when ``frame_history`` is given (only its shape/device).
        Keyword-only and optional; without it the transformer attends over just
        ``[context, z_current]``, a well-defined degenerate case (see
        :meth:`_predict_latent_transformer`). ``beam_plan``/``hier_latent_cem`` build this from
        the real logged history and autoregressively extend it through their lookahead horizon
        (see :func:`_extend_batch_frame_history`).
        """
        latent, logits, _ = self.predict_latent_with_state(
            z_current, z_action, z_context, z_goal, frame_history=frame_history
        )
        return latent, logits

    def predict_latent_with_state(
        self,
        z_current: "torch.Tensor",
        z_action: "torch.Tensor",
        z_context: "torch.Tensor",
        z_goal: Optional["torch.Tensor"] = None,
        *,
        frame_history: Optional[tuple] = None,
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """(latent, prior_logits_or_None, state). ``state`` is the transformer predictor's
        last-position hidden state h_t; ``None`` under ``predictor_arch="mlp"``."""
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
            # Residual parametrization: the predictor output is the change Δz. Applies uniformly
            # at inference (single-step rollout, beam_plan and hier_latent_cem all route through
            # predict_latent), so the trained semantics carry over.
            return z_current + predicted, None, state
        return predicted, None, state

    def predict_success_logit(
        self,
        z_current: "torch.Tensor",
        z_action: "torch.Tensor",
        z_context: "torch.Tensor",
        z_pred: "torch.Tensor",
    ) -> "torch.Tensor":
        return self.success_head(torch.cat([z_current, z_action, z_context, z_pred], dim=-1)).squeeze(-1)

    def predict_terminal_logit(
        self,
        z_current: "torch.Tensor",
        z_action: "torch.Tensor",
        z_context: "torch.Tensor",
        z_pred: "torch.Tensor",
    ) -> "torch.Tensor":
        return self.terminal_head(torch.cat([z_current, z_action, z_context, z_pred], dim=-1)).squeeze(-1)

    def predict_value(
        self,
        z_current: "torch.Tensor",
        z_action: "torch.Tensor",
        z_context: "torch.Tensor",
        z_pred: "torch.Tensor",
    ) -> "torch.Tensor":
        return self.value_head(torch.cat([z_current, z_action, z_context, z_pred], dim=-1)).squeeze(-1)

    def predict_canonical_event_logits(
        self,
        z_current: "torch.Tensor",
        z_action: "torch.Tensor",
        z_context: "torch.Tensor",
        z_pred: "torch.Tensor",
        z_state: Optional["torch.Tensor"] = None,
    ) -> Dict[str, "torch.Tensor"]:
        mode = getattr(self, "canonical_event_head_inputs", "all")
        if mode in ("state", "state_action"):
            if z_state is None:
                raise ValueError(
                    f"canonical_event_head_inputs={mode!r} needs the predictor's hidden state; "
                    "call predict_latent_with_state and pass its state as z_state."
                )
            # h_t is LayerNorm'd by the predictor while z_action is a raw projector output, so
            # the two blocks arrive on different scales; the trunk Linear absorbs that, the same
            # way "all" already concatenates unnormalized latents.
            features = z_state if mode == "state" else torch.cat([z_state, z_action], dim=-1)
        elif mode == "pred_only":
            features = z_pred
        elif mode == "ctx_pred":
            features = torch.cat([z_context, z_pred], dim=-1)
        else:
            features = torch.cat([z_current, z_action, z_context, z_pred], dim=-1)
        trunk_features = self.canonical_event_trunk(features)
        return {field: head(trunk_features) for field, head in self.canonical_event_heads.items()}

    def decode_action_latent(self, z_action: "torch.Tensor", tokenizer: Any, max_new_tokens: int = 96) -> List[str]:
        """Inference-time greedy decode: turn a (possibly CEM-sampled/interpolated) action
        latent into text, via the trained ``action_decoder``. Used by ``_ewm_hier_cem`` as an
        alternative to nearest-anchor lookup -- the caller is responsible for validating/falling
        back, since a decoder trained on real anchors is not guaranteed to produce a syntactically
        valid action for an arbitrary latent, only a *more likely* one than an untrained decode."""
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
        # No KV cache: the whole prefix-so-far is re-run through the decoder every step. Fine
        # for max_new_tokens ~= 96 (O(L^2) over a short sequence, called at most once per
        # planning cycle) -- add incremental caching later only if this becomes a bottleneck.
        current_embs = self.action_decoder_start.float().view(1, 1, -1).expand(batch_size, 1, -1)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=z_action.device)
        token_rows: List[List[int]] = [[] for _ in range(batch_size)]
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

    def decode_observation_latent(self, z_pred: "torch.Tensor", tokenizer: Any, max_new_tokens: int = 96) -> List[str]:
        """Inference-time greedy decode: turn a predicted-resulting-state latent (z_pred) into
        predicted tool-output text, via the trained ``obs_grounding`` decoder. Structurally
        identical to :meth:`decode_action_latent` (same two branches, same no-KV-cache greedy
        loop) but targets ``obs_ground_*`` instead of ``action_decoder_*`` -- this is new
        inference code: the upstream EWM repo trains ``obs_ground_transformer`` (a loss-only
        teacher-forced module) but has no inference-time greedy-decode method for it yet."""
        if not self.obs_grounding:
            raise RuntimeError("This checkpoint has no trained obs_grounding decoder.")
        from transformers.modeling_outputs import BaseModelOutput

        self.eval()
        batch_size = z_pred.shape[0]
        if self.supports_reconstruction:
            memory = self.obs_ground_memory_projection(z_pred).view(-1, self.memory_tokens, self.hidden_size)
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
        memory = self.obs_ground_memory_expand(z_pred.float()).view(
            batch_size, self.obs_ground_decoder_memory_tokens, self.obs_ground_decoder_dim
        )
        current_embs = self.obs_ground_start.float().view(1, 1, -1).expand(batch_size, 1, -1)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=z_pred.device)
        token_rows: List[List[int]] = [[] for _ in range(batch_size)]
        for _ in range(max_new_tokens):
            length = current_embs.shape[1]
            decoder_input = self.obs_ground_token_in(current_embs)
            positions = torch.arange(length, device=z_pred.device)
            decoder_input = decoder_input + self.obs_ground_pos(positions).unsqueeze(0)
            causal_mask = torch.triu(torch.full((length, length), float("-inf"), device=z_pred.device), diagonal=1)
            decoded = self.obs_ground_transformer(decoder_input, memory, tgt_mask=causal_mask)
            logits = self.obs_ground_out(decoded[:, -1]).float() @ emb_weight.t()
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


# ---------------------------------------------------------------------------
# The JEPA world-model generator (ported from finetuning.JepaTextWorldModelGenerator)
# ---------------------------------------------------------------------------


def _load_state_dict(path: "str | Path") -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _infer_transformer_decoder_arch(state_dict: Dict[str, Any], prefix: str) -> Optional[Dict[str, int]]:
    """Infer ``{dim, layers, max_length, memory_tokens}`` for an encoder-only causal-Transformer
    decoder (``action_decoder_*`` / ``obs_ground_*``) directly from its saved tensor shapes, so
    merging in decoder weights from a checkpoint with no architecture manifest (this port's
    ``merge_checkpoint``, see :class:`JepaEwmGenerator`) doesn't require guessing training
    hyperparameters. Returns ``None`` if ``prefix`` isn't present in ``state_dict`` (seq2seq
    branch, or the checkpoint has no such decoder at all)."""
    token_in_key = f"{prefix}token_in.weight"
    if token_in_key not in state_dict:
        return None
    dim = int(state_dict[token_in_key].shape[0])
    arch: Dict[str, int] = {"dim": dim}
    pos_key = f"{prefix}pos.weight"
    if pos_key in state_dict:
        arch["max_length"] = int(state_dict[pos_key].shape[0])
    expand_key = f"{prefix}memory_expand.2.weight"
    if expand_key in state_dict and dim:
        arch["memory_tokens"] = int(state_dict[expand_key].shape[0] // dim)
    layer_pattern = re.compile(rf"^{re.escape(prefix)}transformer\.layers\.(\d+)\.")
    layer_indices = {int(m.group(1)) for k in state_dict if (m := layer_pattern.match(k))}
    if layer_indices:
        arch["layers"] = max(layer_indices) + 1
    return arch


class JepaEwmGenerator:
    """Inference adapter for a ``finetuning_jepa.py`` checkpoint, plugged into the
    imagined-trajectory rollout loop as the world model.

    Exposes :meth:`predict_feedback` — the polymorphic seam
    :func:`_ewm_runtime.predict_wm_feedback` prefers when present — so the loop
    conditions on the JEPA heads directly (no text parsing). Also exposes
    :meth:`generate_from_messages` for the decoder path / debugging.

    The checkpoint directory must contain ``text_leworldmodel.pt`` and
    ``backbone/`` (validated by the caller); optionally
    ``jepa_data_manifest.json`` (architecture), ``canonical_event_vocab.json`` and
    ``canonical_event_data_manifest.json`` (classification heads).
    """

    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int = 512,
        trust_remote_code: bool = False,
        dtype: str = "auto",
        max_input_length: int = 2048,
        max_action_length: int = 512,
        imagined_observation_backend: str = "auto",
        arch_defaults: Optional[Dict[str, Any]] = None,
        merge_checkpoint: Optional["str | Path"] = None,
        merge_decode_max_new_tokens: int = 96,
    ) -> None:
        from transformers.modeling_outputs import BaseModelOutput

        self.model_path = Path(model_path)
        self.max_new_tokens = int(max_new_tokens)
        self.BaseModelOutput = BaseModelOutput

        manifest_path = self.model_path / "jepa_data_manifest.json"
        manifest: Dict[str, Any] = {}
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        # A canonical-event head checkpoint may not carry jepa_data_manifest.json.
        # Fall back per-field to the architecture defaults the caller passed; a
        # value present in the manifest always wins.
        resolved: Dict[str, Any] = {**(arch_defaults or {}), **manifest}
        canonical_manifest_path = self.model_path / "canonical_event_data_manifest.json"
        canonical_manifest: Dict[str, Any] = {}
        if canonical_manifest_path.is_file():
            with canonical_manifest_path.open("r", encoding="utf-8") as handle:
                canonical_manifest = json.load(handle)
        self.backbone_type = str(resolved.get("backbone_type") or "seq2seq")
        self.max_input_length = int(resolved.get("max_input_length") or max_input_length)
        self.max_action_length = int(resolved.get("max_action_length") or max_action_length)
        self.max_observation_length = int(resolved.get("max_observation_length") or 512)
        self.max_goal_length = int(resolved.get("max_goal_length") or self.max_observation_length)

        self.tokenizer = load_text_tokenizer(self.model_path, trust_remote_code=trust_remote_code)
        backbone = load_jepa_backbone(
            self.model_path / "backbone",
            backbone_type=self.backbone_type,
            trust_remote_code=trust_remote_code,
            dtype=resolve_torch_dtype(dtype),
        )
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

        # Optional: merge in action_decoder_*/obs_ground_* weights from a SEPARATE checkpoint.
        # Exists for checkpoints whose canonical-event-head training run silently dropped these
        # optional modules -- e.g. because the base checkpoint it was initialized from had no
        # jepa_data_manifest.json to preserve their architecture flags across the head-only
        # training step (finetuning_jepa.py's build_canonical_event_head_model reads the base's
        # manifest specifically to avoid this; a manifest-less base means it can't). Since
        # head-only training updates ONLY canonical_event_heads.*/canonical_event_trunk.*
        # (backbone/predictor/etc are frozen), the merge checkpoint's other weights are expected
        # to be numerically identical to this checkpoint's -- we only ever borrow its
        # action_decoder_*/obs_ground_* weights, never its backbone/predictor/heads.
        self._merge_state_dict: Optional[Dict[str, Any]] = None
        merge_action_decoder_arch: Optional[Dict[str, int]] = None
        merge_obs_ground_arch: Optional[Dict[str, int]] = None
        merge_has_action_decoder = False
        merge_has_obs_grounding = False
        if merge_checkpoint:
            merge_state_dict_path = Path(merge_checkpoint) / "text_leworldmodel.pt"
            if not merge_state_dict_path.is_file():
                raise ValueError(f"merge_checkpoint has no text_leworldmodel.pt: {merge_checkpoint}")
            self._merge_state_dict = _load_state_dict(merge_state_dict_path)
            merge_has_action_decoder = any(k.startswith("action_decoder_") for k in self._merge_state_dict)
            merge_has_obs_grounding = any(k.startswith("obs_ground_") for k in self._merge_state_dict)
            if not (merge_has_action_decoder or merge_has_obs_grounding):
                raise ValueError(
                    f"merge_checkpoint {merge_checkpoint} has no action_decoder_*/obs_ground_* "
                    "weights to merge -- nothing to do."
                )
            merge_action_decoder_arch = _infer_transformer_decoder_arch(self._merge_state_dict, "action_decoder_")
            merge_obs_ground_arch = _infer_transformer_decoder_arch(self._merge_state_dict, "obs_ground_")
            logger.info(
                "JepaEwmGenerator: merging action_decoder=%s (arch=%s) obs_grounding=%s (arch=%s) from %s",
                merge_has_action_decoder, merge_action_decoder_arch,
                merge_has_obs_grounding, merge_obs_ground_arch, merge_checkpoint,
            )

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
            pooling=str(resolved.get("pooling") or "mean"),
            canonical_event_vocab_sizes=canonical_event_vocab_sizes,
            canonical_event_head_hidden_size=canonical_event_head_hidden_size,
            canonical_event_head_inputs=str(
                resolved.get("canonical_event_head_inputs")
                or canonical_manifest.get("canonical_event_head_inputs")
                or "all"
            ),
            predictor_arch=str(resolved.get("predictor_arch") or "mlp"),
            predictor_transformer_dim=int(resolved.get("predictor_transformer_dim") or 0),
            predictor_transformer_layers=int(resolved.get("predictor_transformer_layers") or 6),
            predictor_transformer_heads=int(resolved.get("predictor_transformer_heads") or 16),
            predictor_transformer_mlp_ratio=float(resolved.get("predictor_transformer_mlp_ratio") or 4.0),
            predictor_history_length=int(resolved.get("predictor_history_length") or 0),
            terminal_head=bool(resolved.get("terminal_head", False) or canonical_manifest.get("terminal_head", False)),
            value_head=bool(resolved.get("value_head", False) or canonical_manifest.get("value_head", False)),
            action_decoder=bool(resolved.get("action_decoder", False)) or merge_has_action_decoder,
            action_decoder_max_noise_std=float(resolved.get("action_decoder_max_noise_std") or 0.1),
            action_decoder_dim=int((merge_action_decoder_arch or {}).get("dim") or resolved.get("action_decoder_dim") or 256),
            action_decoder_layers=int((merge_action_decoder_arch or {}).get("layers") or resolved.get("action_decoder_layers") or 4),
            action_decoder_heads=int(resolved.get("action_decoder_heads") or 4),
            action_decoder_memory_tokens=int(
                (merge_action_decoder_arch or {}).get("memory_tokens") or resolved.get("action_decoder_memory_tokens") or 8
            ),
            action_decoder_max_length=int(
                (merge_action_decoder_arch or {}).get("max_length") or resolved.get("action_decoder_max_length") or 512
            ),
            obs_grounding=bool(resolved.get("obs_grounding", False)) or merge_has_obs_grounding,
            obs_ground_decoder_dim=int((merge_obs_ground_arch or {}).get("dim") or resolved.get("obs_ground_decoder_dim") or 256),
            obs_ground_decoder_layers=int((merge_obs_ground_arch or {}).get("layers") or resolved.get("obs_ground_decoder_layers") or 4),
            obs_ground_decoder_heads=int(resolved.get("obs_ground_decoder_heads") or 4),
            obs_ground_decoder_memory_tokens=int(
                (merge_obs_ground_arch or {}).get("memory_tokens") or resolved.get("obs_ground_decoder_memory_tokens") or 8
            ),
            obs_ground_decoder_max_length=int(
                (merge_obs_ground_arch or {}).get("max_length") or resolved.get("obs_ground_decoder_max_length") or 128
            ),
        )
        self.merge_decode_max_new_tokens = int(merge_decode_max_new_tokens)
        state_dict_path = self.model_path / "text_leworldmodel.pt"
        try:
            state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(state_dict_path, map_location="cpu")
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        missing_keys = set(getattr(incompatible, "missing_keys", []))
        unexpected_keys = set(getattr(incompatible, "unexpected_keys", []))
        # Optional-module weights (obs_ground_/tool_embeddings/tool_query/action_encoder_mlp/
        # fast_/action_decoder) may be present in the checkpoint but absent from this replay
        # model (or vice versa) if the manifest predates a flag, or this port doesn't build a
        # module (tool_select/action_encoder/fast_lewm aren't ported -- unused by any ejepa_wm
        # strategy) -- tolerate them in BOTH directions, same prefixes as finetuning_jepa.py.
        _optional_prefixes = (
            "obs_ground_", "tool_embeddings", "tool_query", "action_encoder_mlp",
            "fast_", "action_decoder", "terminal_head", "value_head",
        )
        allowed_missing = {
            key
            for key in missing_keys
            if key.startswith(("backbone.", "success_head.", "canonical_event_heads.", "canonical_event_trunk."))
            or key.startswith(_optional_prefixes)
        }
        disallowed_missing = missing_keys - allowed_missing
        disallowed_unexpected = {key for key in unexpected_keys if not key.startswith(_optional_prefixes)}
        if disallowed_missing or disallowed_unexpected:
            raise RuntimeError(
                "JEPA checkpoint does not match TextLeWorldModel architecture: "
                f"missing={sorted(disallowed_missing)}, unexpected={sorted(disallowed_unexpected)}"
            )

        if self._merge_state_dict is not None:
            merge_filtered = {
                key: value for key, value in self._merge_state_dict.items()
                if key.startswith(("action_decoder_", "obs_ground_"))
            }
            merge_incompatible = self.model.load_state_dict(merge_filtered, strict=False)
            merge_unexpected = set(getattr(merge_incompatible, "unexpected_keys", []))
            merge_missing_decoder = {
                key for key in getattr(merge_incompatible, "missing_keys", [])
                if key.startswith(("action_decoder_", "obs_ground_"))
            }
            if merge_unexpected or merge_missing_decoder:
                raise RuntimeError(
                    "merge_checkpoint action_decoder_*/obs_ground_* weights do not match this "
                    f"TextLeWorldModel's architecture: missing={sorted(merge_missing_decoder)}, "
                    f"unexpected={sorted(merge_unexpected)}"
                )
            self._merge_state_dict = None  # free the ~300MB extra dict once merged in

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
        terminal_head_weights_present = getattr(self.model, "terminal_head_enabled", False) and not any(
            key.startswith("terminal_head.") for key in missing_keys
        )
        self.terminal_head_available = bool(terminal_head_weights_present)

        canonical_event_weights_present = canonical_event_vocab_sizes is not None and not any(
            key.startswith("canonical_event_heads.") for key in missing_keys
        )
        self.canonical_event_available = bool(canonical_event_weights_present)
        backend = str(imagined_observation_backend or "auto")
        if backend == "canonical_event" and not self.canonical_event_available:
            raise RuntimeError(
                "imagined_observation_backend=canonical_event requires a checkpoint with trained "
                "canonical_event heads (canonical_event_vocab.json + canonical_event_heads.* weights); "
                f"none found in {self.model_path}."
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
        self.fast_inference_mode: Optional[str] = None
        self.pad_multiple = 1
        self.prediction_control = resolve_prediction_control()
        self.class_priors = load_class_priors() if self.prediction_control == "prior" else None
        if self.prediction_control:
            import random

            self._control_rng = random.Random(int(os.environ.get(PREDICTION_CONTROL_SEED_ENV, "0")))
            logger.warning(
                "JepaEwmGenerator: PREDICTION CONTROL %s active -- beam scores do not use real predictions",
                self.prediction_control,
            )
        if self.device.type == "cuda":
            mode = resolve_fast_inference_mode()
            if mode:
                self.enable_fast_inference(mode)
        logger.info(
            "JepaEwmGenerator: backend=%s backbone_type=%s device=%s model_path=%s",
            self.imagined_observation_backend_resolved,
            self.backbone_type,
            self.device,
            self.model_path,
        )

    def enable_fast_inference(
        self, mode: str = "reduce-overhead", pad_multiple: Optional[int] = None
    ) -> List[str]:
        """``torch.compile`` the model's parameterised children in place and turn on
        shape bucketing (see ``FAST_INFERENCE_ENV``). Returns the compiled child names.
        The first call per distinct (batch, length) bucket compiles/captures (seconds);
        use with ``WM_SHARE_MODEL_WEIGHTS=1`` so parallel tasks share one warmed model."""
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        with contextlib.suppress(AttributeError, TypeError):
            torch._inductor.config.triton.cudagraph_trees_generation_cloning = "user_visible"
        compiled: List[str] = []
        for name, child in list(self.model.named_children()):
            if isinstance(child, _CompiledChild) or not any(True for _ in child.parameters()):
                continue
            is_backbone = name == "backbone"
            self.model.add_module(
                name,
                _CompiledChild(torch.compile(child, mode=mode, dynamic=False if is_backbone else None)),
            )
            compiled.append(name)
        self.fast_inference_mode = mode
        self.pad_multiple = int(
            pad_multiple
            if pad_multiple is not None
            else os.environ.get(FAST_INFERENCE_PAD_MULTIPLE_ENV, 64)
        )
        logger.info(
            "JepaEwmGenerator: fast inference mode=%s pad_multiple=%d compiled=%s",
            mode,
            self.pad_multiple,
            compiled,
        )
        return compiled

    def _pad_id(self) -> int:
        return int(getattr(self.tokenizer, "pad_token_id", None) or 0)

    # -- text assembly ------------------------------------------------------

    def _render_history(self, history: List[Dict[str, Any]]) -> str:
        return render_raw_replay_history(history, max_chars=self.max_observation_length * 8)

    def _context_text(self, system_prompt: str, user_prompt: str) -> str:
        return f"System prompt:\n{system_prompt}\n\nUser task:\n{user_prompt}"

    def _current_state_text(self, system_prompt: str, user_prompt: str, input_history: List[Dict[str, Any]]) -> str:
        context = self._context_text(system_prompt, user_prompt)
        return context + "\n\nHistory before current action:\n" + self._render_history(input_history)

    def _goal_text(self, system_prompt: str, user_prompt: str) -> str:
        goal_text = "Task goal:\n" + self._context_text(system_prompt, user_prompt)
        return goal_text[: self.max_goal_length * 8]

    def _encode(self, text: str, max_length: int) -> Dict[str, "torch.Tensor"]:
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        moved = {key: value.to(self.device) for key, value in encoded.items()}
        return pad_to_multiple(moved, self.pad_multiple, max_length, self._pad_id())

    def _encode_latent_text(self, text: str, max_length: int) -> "torch.Tensor":
        """Encode one text to a latent, memoized by ``(max_length, text)``.

        The context and goal texts are CONSTANT for a whole task and the state text recurs
        between the scorers called at the same step, yet each encode is a full backbone forward
        over up to ``max_length`` tokens -- the dominant cost of a plan-scoring call. Cached
        latents are ``[1, D]`` tensors, so the bounded cache below costs kilobytes.
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

    def _encode_batch(self, texts: List[str], max_length: int) -> Dict[str, "torch.Tensor"]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        moved = {key: value.to(self.device) for key, value in encoded.items()}
        return pad_to_multiple(moved, self.pad_multiple, max_length, self._pad_id())

    def encode_latents_batched(self, texts: List[str], max_length: int) -> "torch.Tensor":
        """Encode many texts to latents in a single backbone forward (ported from
        ``finetuning.JepaTextWorldModelGenerator``).

        Padding is masked out by the model's pooling, so a batched encode is numerically
        equivalent to encoding each text on its own. Repeated texts (identical tool calls
        recur across plans and rollout steps) are cached so they are only encoded once.
        """
        cache = getattr(self, "_action_latent_cache", None)
        if cache is None:
            cache = self._action_latent_cache = {}
        order: Dict[str, int] = {}
        misses: List[str] = []
        for text in texts:
            if text not in cache and text not in order:
                order[text] = len(misses)
                misses.append(text)
        if misses:
            encoded = self._encode_batch(misses, max_length)
            ids, mask = encoded["input_ids"], encoded["attention_mask"]
            if self.pad_multiple > 1 and ids.shape[0] > 1:
                # bucket the batch dimension to a power of two so compiled shapes recur
                target = 1 << (ids.shape[0] - 1).bit_length()
                if target > ids.shape[0]:
                    reps = target - ids.shape[0]
                    ids = torch.cat([ids, ids[-1:].expand(reps, -1)], dim=0)
                    mask = torch.cat([mask, mask[-1:].expand(reps, -1)], dim=0)
            latents = self.model.encode_latent(ids, mask)[: len(misses)]
            for text, idx in order.items():
                cache[text] = latents[idx : idx + 1]
        return torch.cat([cache[text] for text in texts], dim=0)

    def _encode_frame_history(
        self, input_history: List[Dict[str, Any]], *, z_goal: Optional["torch.Tensor"] = None
    ) -> Optional[tuple]:
        """Build ``(frames [1,T,D], producing_actions [1,T,C], valid [1,T])`` for
        ``predictor_arch="transformer"`` from the same ``{step, action, observation}`` history
        already threaded everywhere in this module (the source list behind
        :func:`render_raw_replay_history`). ``frames[k]`` is the encoded OBSERVATION text at
        step k; ``producing_actions[k]`` is the encoded ACTION text that produced it (+ goal,
        when goal-conditioned) -- exactly ``encode_frame_history``'s training-time
        representation in ``finetuning_jepa.py``.

        Returns ``None`` for ``predictor_arch="mlp"`` or when there is no usable history, in
        which case the model's own ``[context, z_current]`` degenerate case applies. Windows to
        the same trailing ``WORLD_MODEL_INPUT_HISTORY_SIZE`` steps used for the text rendering;
        the model's own window-trim (``predictor.max_positions - 1``) handles any further
        truncation to the newest steps.
        """
        if getattr(self.model, "predictor_arch", "mlp") != "transformer":
            return None
        entries = [
            item
            for item in (input_history or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:]
            if isinstance(item, dict) and item.get("action") is not None
        ]
        if not entries:
            return None
        action_texts = [
            (item["action"] if isinstance(item["action"], str) else _safe_json(item["action"]))
            for item in entries
        ]
        observation_texts = [
            _truncate_text(
                ewm.stringify_tool_output(item.get("observation") or item.get("state") or ""),
                self.max_observation_length * 8,
            )
            for item in entries
        ]
        z_frames = self.encode_latents_batched(observation_texts, self.max_observation_length).unsqueeze(0)
        z_actions_raw = self.encode_latents_batched(action_texts, self.max_action_length)
        z_actions = self._frame_conditioning_batch(z_actions_raw, z_goal).unsqueeze(0)
        valid = torch.ones(1, len(entries), dtype=torch.bool, device=self.device)
        return z_frames, z_actions, valid

    def _frame_conditioning_batch(
        self, z_actions: "torch.Tensor", z_goal: Optional["torch.Tensor"]
    ) -> "torch.Tensor":
        if not getattr(self.model, "goal_conditioning", False):
            return z_actions
        goal = z_goal if z_goal is not None else torch.zeros_like(z_actions[:1])
        return torch.cat([z_actions, goal.expand(z_actions.shape[0], -1)], dim=-1)

    def _predict_latents(
        self, system_prompt: str, user_prompt: str, action_text: str, input_history: List[Dict[str, Any]]
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
        """Returns (z_current, z_context, z_action, z_pred, z_state). ``z_state`` is the
        transformer predictor's hidden state h_t (``None`` under ``predictor_arch="mlp"``),
        needed by ``predict_canonical_event_logits`` when the checkpoint was trained with
        ``canonical_event_head_inputs="state"``."""
        context_text = self._context_text(system_prompt, user_prompt)
        current_text = self._current_state_text(system_prompt, user_prompt, input_history)
        z_current = self._encode_latent_text(current_text, self.max_input_length)
        z_context = self._encode_latent_text(context_text, self.max_input_length)
        z_action = self._encode_latent_text(action_text, self.max_action_length)
        z_goal = None
        if getattr(self.model, "goal_conditioning", False):
            z_goal = self._encode_latent_text(self._goal_text(system_prompt, user_prompt), self.max_goal_length)
        frame_history = self._encode_frame_history(input_history, z_goal=z_goal)
        z_pred, _, z_state = self.model.predict_latent_with_state(
            z_current, z_action, z_context, z_goal, frame_history=frame_history
        )
        return z_current, z_context, z_action, z_pred, z_state

    # -- the three imagined-observation backends ----------------------------

    def _generate_observation(
        self,
        system_prompt: str,
        user_prompt: str,
        action: Any,
        input_history: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> str:
        if self.backbone_type == "encoder":
            raise RuntimeError(
                "Encoder-only JEPA checkpoints are latent-only and cannot decode tool-output predictions."
            )
        action_text = action if isinstance(action, str) else _safe_json(action)
        with torch.no_grad():
            _, _, _, z_pred, _ = self._predict_latents(system_prompt, user_prompt, action_text, input_history)
            memory = self.model.memory_projection(z_pred).view(
                -1, self.model.memory_tokens, self.model.hidden_size
            )
            backbone_dtype = next(self.model.backbone.parameters()).dtype
            memory = memory.to(dtype=backbone_dtype)
            memory_mask = torch.ones(memory.shape[:2], dtype=torch.long, device=memory.device)
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

    def decode_plan_observations(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        input_history: List[Dict[str, Any]],
        plan: List[Any],
        goal_text_override: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Decode the predicted tool-output text for each step of ONE plan, via the trained
        ``obs_grounding`` decoder. Called post-hoc on a single ALREADY-CHOSEN trajectory (e.g.
        beam_plan's winning, confidence-gated plan) -- decoding is a generative greedy loop, far
        too expensive to run for every scored candidate at every horizon step the way
        :meth:`score_action_plans_canonical_event` does.

        Rolls the plan forward exactly like that method's single-plan case (same context/goal/
        current-state encoding), but decodes the observation from z_pred at each step instead of
        reading the canonical-event heads. Requires a checkpoint with a trained ``obs_grounding``
        decoder (native seq2seq reconstruction, or the encoder-only ``obs_ground_transformer`` --
        see ``merge_checkpoint`` on :meth:`__init__` for merging one in from a separate checkpoint).
        """
        if not getattr(self.model, "obs_grounding", False):
            raise RuntimeError(
                "This JEPA checkpoint has no trained obs_grounding decoder; decode_plan_observations "
                "requires one (train with --obs-token-ground-coeff > 0, or merge one in via "
                "merge_checkpoint)."
            )
        if not plan:
            return []
        max_new_tokens = int(max_new_tokens) if max_new_tokens is not None else self.merge_decode_max_new_tokens
        context_text = self._context_text(system_prompt, user_prompt)
        current_text = self._current_state_text(system_prompt, user_prompt, input_history)
        goal_text = goal_text_override or self._goal_text(system_prompt, user_prompt)
        goal_conditioning = getattr(self.model, "goal_conditioning", False)
        texts: List[str] = []
        with torch.no_grad():
            z_context = self._encode_latent_text(context_text, self.max_input_length)
            z_goal = self._encode_latent_text(goal_text, self.max_goal_length) if goal_conditioning else None
            z_current = self._encode_latent_text(current_text, self.max_input_length)
            frame_history = self._encode_frame_history(input_history, z_goal=z_goal)
            for action in plan:
                action_text = action if isinstance(action, str) else _safe_json(action)
                z_action = self._encode_latent_text(action_text, self.max_action_length)
                z_pred, _ = self.model.predict_latent(
                    z_current, z_action, z_context, z_goal, frame_history=frame_history
                )
                decoded = self.model.decode_observation_latent(z_pred, self.tokenizer, max_new_tokens=max_new_tokens)
                text = decoded[0] if decoded else ""
                texts.append(
                    strip_model_thinking_output(
                        text, special_tokens=getattr(self.tokenizer, "all_special_tokens", None)
                    ).strip()
                )
                if frame_history is not None or getattr(self.model, "predictor_arch", "mlp") == "transformer":
                    producing_action = self._frame_conditioning_batch(z_action, z_goal)
                    frame_history = _extend_batch_frame_history(
                        frame_history, 1, torch.arange(1, device=self.device), z_pred, producing_action, self.device
                    )
                z_current = z_pred
        return texts

    def predict_action_success_probability(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        action: Any,
        input_history: List[Dict[str, Any]],
    ) -> float:
        if not getattr(self, "success_head_available", False):
            raise RuntimeError("This JEPA checkpoint does not contain a trained success_head.")
        action_text = action if isinstance(action, str) else _safe_json(action)
        with torch.no_grad():
            z_current, z_context, z_action, z_pred, _ = self._predict_latents(
                system_prompt, user_prompt, action_text, input_history
            )
            logit = self.model.predict_success_logit(z_current, z_action, z_context, z_pred)
            return float(torch.sigmoid(logit.float()).item())

    def predict_canonical_event_labels(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        action: Any,
        input_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not getattr(self, "canonical_event_available", False):
            raise RuntimeError("This JEPA checkpoint does not contain trained canonical_event heads.")
        action_text = action if isinstance(action, str) else _safe_json(action)
        with torch.no_grad():
            z_current, z_context, z_action, z_pred, z_state = self._predict_latents(
                system_prompt, user_prompt, action_text, input_history
            )
            logits = self.model.predict_canonical_event_logits(z_current, z_action, z_context, z_pred, z_state)
        return decode_canonical_event_logits(logits, self.canonical_event_vocab)

    def score_action_plans_canonical_event(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        input_history: List[Dict[str, Any]],
        action_plans: List[List[Any]],
        goal_text_override: Optional[str] = None,
        score_config: Any = None,
    ) -> List[Dict[str, Any]]:
        """Score candidate action plans for ``beam_plan`` using the canonical-event
        classification heads (higher score = better). Port of
        ``finetuning.JepaTextWorldModelGenerator.score_action_plans_canonical_event``.

        Rolls every plan forward in lockstep (each step encodes all active candidate actions
        in one batched forward), reads the canonical-event heads per step and scores the
        per-step field distributions with :mod:`_ewm_canonical_event_scoring`. Returns records
        sorted best-first, each with ``plan``/``plan_index``, ``score``, ``vetoed``,
        ``normalized_score`` and a ``reason`` string.
        """
        from ejepa_wm.backends._ewm_canonical_event_scoring import (
            CanonicalEventScoreConfig,
            logits_to_field_probs_batched,
        )

        if not action_plans:
            return []
        if not getattr(self, "canonical_event_available", False):
            raise RuntimeError(
                "This JEPA checkpoint has no trained canonical_event heads; beam_plan canonical "
                "scoring requires a checkpoint with canonical_event_vocab.json + canonical_event_heads.* weights."
            )
        config = score_config or CanonicalEventScoreConfig()
        temperature = float(getattr(config, "temperature", 1.0))
        context_text = self._context_text(system_prompt, user_prompt)
        current_text = self._current_state_text(system_prompt, user_prompt, input_history)
        goal_conditioning = getattr(self.model, "goal_conditioning", False)
        num_plans = len(action_plans)
        trajectories: List[List[Dict[str, Dict[str, float]]]] = [[] for _ in range(num_plans)]
        terminal_prob_steps: List[List[float]] = [[] for _ in range(num_plans)]
        terminal_enabled = bool(getattr(self, "terminal_head_available", False))
        with torch.inference_mode():
            z_context = self._encode_latent_text(context_text, self.max_input_length)
            # A checkpoint trained with goal_conditioning=False never reads z_goal, so encoding
            # the goal text there is a wasted backbone forward per scoring call.
            z_goal = None
            if goal_conditioning:
                goal_text = goal_text_override or self._goal_text(system_prompt, user_prompt)
                z_goal = self._encode_latent_text(goal_text, self.max_goal_length)
            z_start = self._encode_latent_text(current_text, self.max_input_length)
            z_rollout = z_start.expand(num_plans, -1).contiguous()
            frame_history0 = self._encode_frame_history(input_history, z_goal=z_goal)
            frame_history = (
                tuple(t.expand(num_plans, *t.shape[1:]).contiguous() for t in frame_history0)
                if frame_history0 is not None
                else None
            )
            max_len = max(len(plan) for plan in action_plans)
            for step in range(max_len):
                active = [i for i, plan in enumerate(action_plans) if step < len(plan)]
                if not active:
                    break
                action_texts = [
                    (action if isinstance(action, str) else _safe_json(action))
                    for action in (action_plans[i][step] for i in active)
                ]
                z_actions = self.encode_latents_batched(action_texts, self.max_action_length)
                index = torch.as_tensor(active, device=self.device)
                z_sub = z_rollout.index_select(0, index)
                z_ctx = z_context.expand(len(active), -1)
                z_goal_arg = z_goal.expand(len(active), -1) if goal_conditioning else None
                frame_history_sub = (
                    tuple(t.index_select(0, index) for t in frame_history) if frame_history is not None else None
                )
                z_new, _, z_state = self.model.predict_latent_with_state(
                    z_sub, z_actions, z_ctx, z_goal_arg, frame_history=frame_history_sub
                )
                logits = self.model.predict_canonical_event_logits(z_sub, z_actions, z_ctx, z_new, z_state)
                # One host transfer per field instead of one per (plan, field, class) scalar.
                probs_rows = logits_to_field_probs_batched(logits, self.canonical_event_vocab, temperature=temperature)
                terminal_probs = None
                if terminal_enabled:
                    terminal_logits = self.model.predict_terminal_logit(z_sub, z_actions, z_ctx, z_new)
                    terminal_probs = torch.sigmoid(terminal_logits.detach().float()).cpu().tolist()
                for local_index, plan_index in enumerate(active):
                    trajectories[plan_index].append(probs_rows[local_index])
                    if terminal_probs is not None:
                        terminal_prob_steps[plan_index].append(float(terminal_probs[local_index]))
                z_rollout = z_rollout.index_copy(0, index, z_new.to(z_rollout.dtype))
                if frame_history is not None or getattr(self.model, "predictor_arch", "mlp") == "transformer":
                    producing_action = self._frame_conditioning_batch(z_actions, z_goal_arg)
                    frame_history = _extend_batch_frame_history(
                        frame_history, num_plans, index, z_new, producing_action, self.device
                    )
        from ejepa_wm.backends._ewm_canonical_event_scoring import finalize_scored_plans

        control = getattr(self, "prediction_control", None)
        if control:
            trajectories, terminal_prob_steps = apply_prediction_control(
                trajectories,
                terminal_prob_steps,
                control,
                vocab=self.canonical_event_vocab,
                priors=getattr(self, "class_priors", None),
                rng=getattr(self, "_control_rng", None),
            )
        records = finalize_scored_plans(
            trajectories,
            action_plans,
            config,
            task_text=user_prompt,
            input_history=input_history,
        )
        for record in records:
            index = int(record.get("index", -1))
            probs = terminal_prob_steps[index] if 0 <= index < len(terminal_prob_steps) else []
            record["per_step_terminal_prob"] = probs
            record["terminal_probability"] = probs[-1] if probs else None
            if control:
                record["prediction_control"] = control
        return records

    def _canonical_event_feedback(
        self,
        *,
        predicted_calls: List[Dict[str, Any]],
        action: Any,
        system_prompt: str,
        user_prompt: str,
        input_history: List[Dict[str, Any]],
        interaction_index: int,
        world_model_target: str,
    ) -> List[Dict[str, Any]]:
        labels = self.predict_canonical_event_labels(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            action=action,
            input_history=input_history,
        )
        tool_name = predicted_calls[0].get("name") if predicted_calls else None
        predicted_state = reconstruct_state_from_canonical_event_labels(labels, tool_name=tool_name)
        observation_payload = build_canonical_event_observation_payload(labels, tool_name=tool_name)
        outcome = observation_payload["tool_outcome"]
        label = outcome.get("label")
        # Unknown execution_status (label None) is treated as a non-blocking success
        # so imagined planning is not derailed by an abstention.
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

    # -- the polymorphic seam predict_wm_feedback prefers -------------------

    @staticmethod
    def _input_history_from_state_history(
        state_history: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Shape the rollout loop's state_history into the ``{step, state}`` entries
        :func:`render_raw_replay_history` expects.

        The text-LLM rollout in :mod:`_ewm_runtime` carries a *state* history
        (from ``append_state_history``), not the source's
        ``{imagined step, action, state}`` entries. We render each state as the
        step's observation; per-step action text is not threaded through the
        shared loop (a deliberate simplification — the current action is still
        supplied separately).
        """
        entries: List[Dict[str, Any]] = []
        for index, state in enumerate(state_history or []):
            entries.append({"step": index, "state": state})
        return entries

    def predict_feedback(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        planned_calls: List[Dict[str, Any]],
        previous_state: Optional[Dict[str, Any]] = None,
        state_history: Optional[List[Dict[str, Any]]] = None,
        wm_state: Optional[str] = None,
        interaction_index: int = 0,
        world_model_target: str = ft.WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
    ) -> List[Dict[str, Any]]:
        """Predict the outcome of ``planned_calls`` with the JEPA world model.

        Returns a one-element ``list[dict]`` with the same keys the text-LLM path
        (:func:`_ewm_runtime.predict_wm_feedback`) returns, so the imagined-
        trajectory loop consumes it unchanged. ``wm_state``/``previous_state`` are
        accepted for interface parity — the JEPA backend is fixed at load time.
        """
        del previous_state, wm_state
        action = {"tool_calls": ewm.to_openai_tool_calls(planned_calls)}
        input_history = self._input_history_from_state_history(state_history)

        if getattr(self, "canonical_event_state_enabled", False):
            return self._canonical_event_feedback(
                predicted_calls=planned_calls,
                action=action,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                input_history=input_history,
                interaction_index=interaction_index,
                world_model_target=world_model_target,
            )

        if getattr(self, "success_head_available", False):
            success_probability = self.predict_action_success_probability(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                action=action,
                input_history=input_history,
            )
            predicted_success = success_probability >= 0.5
            predicted_error_message = (
                "" if predicted_success else "JEPA classifier predicts this tool call is likely to fail."
            )
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
                    "tool_calls": planned_calls,
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
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            action=action,
            input_history=input_history,
        )
        predicted_success = not ft.tool_output_looks_like_failure(predicted_tool_output)
        return [
            {
                "tool_calls": planned_calls,
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

    def generate_from_messages(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Decoder-path convenience: parse an EWM state-prediction prompt back into
        (system, user, action, history) and decode a raw predicted observation.

        Provided for interface parity with :class:`_ewm_runtime.EwmGenerator`; the
        imagined loop uses :meth:`predict_feedback` instead.
        """
        system_prompt, user_prompt, action_text, input_history = self._parse_prompt_messages(messages)
        return self._generate_observation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            action=action_text,
            input_history=input_history,
            temperature=temperature,
        )

    def _parse_prompt_messages(
        self, messages: List[Dict[str, str]]
    ) -> tuple[str, str, str, List[Dict[str, Any]]]:
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
        input_history: List[Dict[str, Any]] = []
        if history_text:
            try:
                parsed_history = json.loads(history_text)
                if isinstance(parsed_history, list):
                    input_history = [item for item in parsed_history if isinstance(item, dict)]
            except json.JSONDecodeError:
                input_history = []
        return system_prompt, user_prompt, action_text, input_history


__all__ = ["JepaEwmGenerator", "TextLeWorldModel"]
