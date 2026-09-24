"""K-controller training (Stage I `label` + Stage II `sft`) for ewm-enterprisearena.

Adapted port of the K-controller trainer from
``ewm-state-design-and-model-training-experiments``, plugged into
ewm-enterprisearena's data + world-model APIs:

* **Stage I (``label``)**: for each expert ``(state, action)`` step from ewm's
  trajectory JSONs, score K=0..kmax by ``logprob(action | state, foresight_K)
  - lambda_k * K`` under a small policy LM, where ``foresight_K`` is the
  K-step world-model imagination produced by the ewm WM generator. The
  argmax K is written to a labelled JSONL.
* **Stage II (``sft``)**: SFT a small policy LM with a ``<CTRL>``-positioned
  K-head on the labelled JSONL. Joint loss = action LM loss + ``beta_k`` *
  ``CE(k_logits, k_label)``.

Stage III (online A2C ``rl_k``) is **not** ported here -- it needs a tight
gym-interactive loop with the action policy inside the gradient step, which
is incompatible with the EnterpriseOps-Gym + GPT-5.1 action policy setup.

Usage::

    python -m src.itp.training.train_adaptive_k label \\
        --train-trajectories trajectories/enterpriseops_gym_multi_model_world_model_train_trajectories.json \\
        --world-model-target tool_execution_result_binary \\
        --include-error-message-in-target \\
        --policy-model-path Qwen/Qwen2.5-1.5B-Instruct \\
        --wm-model-path /data/ewm/gymops_world_model \\
        --out-labeled-jsonl sessions/k_controller_binary/labeled.jsonl \\
        --kmax 3 --lambda-k 0.2

    python -m src.itp.training.train_adaptive_k sft \\
        --train-jsonl sessions/k_controller_binary/labeled.jsonl \\
        --policy-model-path Qwen/Qwen2.5-1.5B-Instruct \\
        --out-dir sessions/k_controller_binary/policy_sft_khead \\
        --kmax 3 --epochs 3 --bf16
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# NOTE: `get_scheduler` is intentionally NOT imported at module level. It
# lives in `transformers.optimization`, which transitively pulls in
# `transformers.integrations` -> `Trainer` / `TrainingArguments` /
# `modeling_utils` -> `loss_utils` -> torchvision. In environments where any
# link in that chain is broken (typical with mismatched torch/torchvision/
# transformers ABIs) the import fails the moment this module is imported,
# even from the inference-only runtime KController. By importing it lazily
# inside Stage II's `stage_sft`, the runtime K-controller path can load the
# trained checkpoint without touching the training-only optimization stack.

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<CTRL>", "<STATE>", "</STATE>", "<FORESIGHT>", "</FORESIGHT>"]


# =============================================================================
# Prompt builders -- shared by labelling, SFT, and runtime K-controller.
# =============================================================================


def build_prompt_for_action(state_text: str, k: int, obs_k: str) -> str:
    """Action-scoring prompt used at Stage I (label) and Stage II (SFT).

    The action that follows ``Action:`` is the expert tool-call JSON.
    """
    return (
        "You are an enterprise-operations agent.\n"
        "Given the current state, output ONLY the next action (a JSON object "
        "with `tool_calls`).\n"
        "<STATE>\n"
        f"{state_text}\n"
        "</STATE>\n"
        "<CTRL>\n"
        "<FORESIGHT>\n"
        f"K={k}\n"
        f"Obs@K: {obs_k}\n"
        "</FORESIGHT>\n"
        "Action:"
    )


def build_prompt_for_controller(state_text: str) -> str:
    """K-controller scoring prompt used at Stage II (SFT) and runtime."""
    return (
        "You are an enterprise-operations agent.\n"
        "Decide how many steps to look ahead with a world model.\n"
        "<STATE>\n"
        f"{state_text}\n"
        "</STATE>\n"
        "<CTRL>"
    )


def build_state_text_from_example(
    example: Any,
    *,
    include_input_history: bool,
    system_prompt_max_chars: int,
) -> str:
    """Render the state-side of ewm's WM prompt body (without the action).

    Reuses :func:`src.finetuning.build_state_context_input_text` so the K-
    controller sees the **same** state representation used to train the world
    model -- byte-identical at training and inference time.
    """
    from src.finetuning import (
        WORLD_MODEL_INPUT_HISTORY_SIZE,
        build_state_context_input_text,
        normalize_world_model_input_history_text,
    )

    if system_prompt_max_chars and system_prompt_max_chars > 0:
        system_prompt_text = (example.system_prompt or "")[:system_prompt_max_chars]
    else:
        system_prompt_text = example.system_prompt or ""

    state_context_text = build_state_context_input_text(example)
    if include_input_history and getattr(example, "input_history", None):
        truncated = list(example.input_history)[-WORLD_MODEL_INPUT_HISTORY_SIZE:]
        state_context_text += (
            "\n\nRecent action/observation history (oldest to newest; input only):\n"
            + normalize_world_model_input_history_text(truncated)
        )

    return (
        f"System prompt:\n{system_prompt_text}\n\n"
        f"User prompt:\n{example.user_prompt}\n\n"
        f"{state_context_text}"
    )


def action_text_from_example(example: Any) -> str:
    """Normalise the expert action into a JSON-stringified ``tool_calls`` body."""
    action = example.action
    if isinstance(action, str):
        return action.strip()
    if isinstance(action, dict):
        return json.dumps(action, ensure_ascii=False)
    return json.dumps({"tool_calls": action or []}, ensure_ascii=False)


def summarize_predicted_state(predicted_state: Any, max_chars: int = 400) -> str:
    """Short text snippet for the ``Obs@K`` field of the action prompt."""
    if predicted_state is None:
        return "(no foresight)"
    if isinstance(predicted_state, str):
        text = predicted_state
    else:
        text = json.dumps(predicted_state, ensure_ascii=False, default=str)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


# =============================================================================
# Tokenization / batching helpers.
# =============================================================================


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _truncate_left_1d(
    ids: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor | None,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if ids.numel() <= max_len:
        return ids, mask, labels
    ids = ids[-max_len:]
    mask = mask[-max_len:]
    if labels is not None:
        labels = labels[-max_len:]
    return ids, mask, labels


def _pad_1d_batch(
    seqs: list[torch.Tensor],
    pad_value: int,
    padding_side: Literal["left", "right"],
    target_len: int | None = None,
) -> torch.Tensor:
    assert len(seqs) > 0
    max_len = max(int(s.numel()) for s in seqs) if target_len is None else int(target_len)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        L = int(s.numel())
        if padding_side == "left":
            out[i, max_len - L :] = s
        else:
            out[i, :L] = s
    return out


def _ensure_pad(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has no pad_token_id and no eos_token_id; set pad/eos first."
            )
        tokenizer.pad_token = tokenizer.eos_token


def make_lm_batch(
    tokenizer: AutoTokenizer,
    prompts: list[str],
    actions: list[str],
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    assert len(prompts) == len(actions)
    _ensure_pad(tokenizer)
    pad_id = tokenizer.pad_token_id
    padding_side: Literal["left", "right"] = getattr(tokenizer, "padding_side", "right")
    eos_str = tokenizer.eos_token if tokenizer.eos_token is not None else ""

    ids_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for p, a in zip(prompts, actions):
        p_ids = tokenizer(p, add_special_tokens=False).input_ids
        a_ids = tokenizer(a + eos_str, add_special_tokens=False).input_ids
        input_ids = torch.tensor(p_ids + a_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = torch.tensor([-100] * len(p_ids) + a_ids, dtype=torch.long)

        input_ids, attention_mask, labels = _truncate_left_1d(
            input_ids, attention_mask, labels, max_seq_len
        )
        ids_list.append(input_ids)
        mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": _pad_1d_batch(ids_list, pad_value=pad_id, padding_side=padding_side),
        "attention_mask": _pad_1d_batch(mask_list, pad_value=0, padding_side=padding_side),
        "labels": _pad_1d_batch(labels_list, pad_value=-100, padding_side=padding_side),
    }


def make_controller_batch(
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    _ensure_pad(tokenizer)
    pad_id = tokenizer.pad_token_id
    padding_side: Literal["left", "right"] = getattr(tokenizer, "padding_side", "right")

    ids_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    for p in prompts:
        p_ids = tokenizer(p, add_special_tokens=False).input_ids
        input_ids = torch.tensor(p_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        input_ids, attention_mask, _ = _truncate_left_1d(
            input_ids, attention_mask, None, max_seq_len
        )
        ids_list.append(input_ids)
        mask_list.append(attention_mask)

    return {
        "input_ids": _pad_1d_batch(ids_list, pad_value=pad_id, padding_side=padding_side),
        "attention_mask": _pad_1d_batch(mask_list, pad_value=0, padding_side=padding_side),
    }


def find_ctrl_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ctrl_token_id: int,
) -> torch.Tensor:
    B, L = input_ids.shape
    ctrl_pos = torch.full((B,), L - 1, dtype=torch.long, device=input_ids.device)
    for i in range(B):
        idx = (input_ids[i] == ctrl_token_id).nonzero(as_tuple=False)
        if idx.numel() > 0:
            ctrl_pos[i] = idx[-1].item()
        else:
            nonpad = (attention_mask[i] == 1).nonzero(as_tuple=False)
            if nonpad.numel() > 0:
                ctrl_pos[i] = nonpad[-1].item()
    return ctrl_pos


def add_special_tokens_and_resize(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    special_tokens: list[str],
) -> int:
    old_vocab = model.get_input_embeddings().weight.size(0)
    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}
    )
    _ensure_pad(tokenizer)
    if n_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            mean_vec = emb[:old_vocab].mean(dim=0, keepdim=True)
            emb[old_vocab:] = mean_vec
    else:
        if model.get_input_embeddings().weight.size(0) != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
    return n_added


# =============================================================================
# PolicyWithHeads -- base LM + K-head + V-head.
# =============================================================================


class PolicyWithHeads(nn.Module):
    """Base causal LM augmented with a K-head and a value head at ``<CTRL>``."""

    def __init__(self, lm: AutoModelForCausalLM, hidden_size: int, kmax: int):
        super().__init__()
        self.lm = lm
        self.kmax = kmax
        self.k_head = nn.Linear(hidden_size, kmax + 1)
        self.v_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ctrl_pos: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hasattr(self.lm, "model") and hasattr(self.lm, "lm_head"):
            base_out = self.lm.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden = base_out.last_hidden_state
            logits = self.lm.lm_head(hidden)
        else:
            out = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = out.logits
            hidden = out.hidden_states[-1]

        if ctrl_pos.device != hidden.device:
            ctrl_pos = ctrl_pos.to(hidden.device)
        h_ctrl = hidden[torch.arange(hidden.size(0), device=hidden.device), ctrl_pos]

        head_dtype = self.k_head.weight.dtype
        if h_ctrl.dtype != head_dtype:
            h_ctrl_heads = h_ctrl.to(dtype=head_dtype)
        else:
            h_ctrl_heads = h_ctrl

        k_logits = self.k_head(h_ctrl_heads)
        values = self.v_head(h_ctrl_heads).squeeze(-1)

        result = {"logits": logits, "k_logits": k_logits, "values": values}

        if labels is not None:
            if labels.device != logits.device:
                labels = labels.to(logits.device)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
            result["lm_loss"] = lm_loss
        return result


@torch.inference_mode()
def compute_action_logprobs_batch(
    model: PolicyWithHeads,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    actions: list[str],
    max_seq_len: int,
    ctrl_token_id: int,
    device: torch.device,
    normalize: Literal["sum", "mean"] = "sum",
) -> torch.Tensor:
    batch = make_lm_batch(tokenizer, prompts, actions, max_seq_len)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    ctrl_pos = find_ctrl_positions(input_ids, attention_mask, ctrl_token_id)

    use_amp = device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=use_amp):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ctrl_pos=ctrl_pos,
            labels=None,
        )
        logits = out["logits"]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size(0), shift_labels.size(1))

    valid = (shift_labels != -100).float()
    sum_nll = (token_nll * valid).sum(dim=1)
    if normalize == "mean":
        denom = valid.sum(dim=1).clamp(min=1.0)
        return -(sum_nll / denom)
    return -sum_nll


# =============================================================================
# Save / load policy-with-heads checkpoint.
# =============================================================================


def save_policy_with_heads(
    out_dir: str,
    tokenizer: AutoTokenizer,
    base_lm: AutoModelForCausalLM,
    policy: PolicyWithHeads,
    kmax: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    base_lm.save_pretrained(out_dir)
    heads = {
        "kmax": int(kmax),
        "k_head": policy.k_head.state_dict(),
        "v_head": policy.v_head.state_dict(),
    }
    torch.save(heads, os.path.join(out_dir, "adaptive_heads.pt"))


def load_policy_with_heads(
    model_dir: str,
    device: torch.device,
    kmax: int,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str | None = None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM, PolicyWithHeads, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    lm_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if device_map:
        lm_kwargs["device_map"] = device_map
        base_lm = AutoModelForCausalLM.from_pretrained(model_dir, **lm_kwargs)
    else:
        base_lm = AutoModelForCausalLM.from_pretrained(model_dir, **lm_kwargs).to(device)

    add_special_tokens_and_resize(tokenizer, base_lm, SPECIAL_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    hidden_size = getattr(base_lm.config, "hidden_size", None) or getattr(
        base_lm.config, "n_embd", None
    )
    if hidden_size is None:
        raise ValueError("Cannot infer hidden size from model config.")

    heads_path = os.path.join(model_dir, "adaptive_heads.pt")
    if not os.path.exists(heads_path):
        policy = PolicyWithHeads(base_lm, hidden_size=hidden_size, kmax=kmax)
        if device_map:
            head_dev = getattr(getattr(base_lm, "lm_head", None), "weight", None)
            head_device = head_dev.device if head_dev is not None else next(base_lm.parameters()).device
            policy.k_head.to(device=head_device, dtype=torch_dtype)
            policy.v_head.to(device=head_device, dtype=torch_dtype)
        else:
            policy = policy.to(device=device, dtype=torch_dtype)
        return tokenizer, base_lm, policy, kmax

    heads = torch.load(heads_path, map_location="cpu")
    saved_kmax = int(heads["kmax"])
    policy = PolicyWithHeads(base_lm, hidden_size=hidden_size, kmax=saved_kmax)
    policy.k_head.load_state_dict(heads["k_head"])
    policy.v_head.load_state_dict(heads["v_head"])
    if device_map:
        head_dev = getattr(getattr(base_lm, "lm_head", None), "weight", None)
        head_device = head_dev.device if head_dev is not None else next(base_lm.parameters()).device
        policy.k_head.to(device=head_device, dtype=torch_dtype)
        policy.v_head.to(device=head_device, dtype=torch_dtype)
    else:
        policy = policy.to(device=device, dtype=torch_dtype)
    return tokenizer, base_lm, policy, saved_kmax


# =============================================================================
# Stage I (label) -- ewm-flavoured.
# =============================================================================


def _load_ewm_examples_grouped(
    train_paths: list[Path],
    state_history_size: int,
) -> dict[str, list]:
    """Load ewm trajectories and group ``WorldModelStateExample``s by episode."""
    from src.evaluation import _load_eval_trajectories
    from src.finetuning import extract_state_examples

    trajectories = _load_eval_trajectories(train_paths)
    examples = extract_state_examples(trajectories, state_history_size=state_history_size)

    grouped: dict[str, list] = {}
    for ex in examples:
        episode_id = f"{ex.trajectory_id}"
        grouped.setdefault(episode_id, []).append(ex)
    # Sort each episode by interaction_index to preserve trajectory order.
    for episode_id in grouped:
        grouped[episode_id].sort(key=lambda e: e.interaction_index)
    return grouped


def _build_world_model_generator_for_labeller(args: argparse.Namespace) -> Any:
    """Reuse evaluation.py's WM factory so vllm and HF paths both work."""
    from src.evaluation import HFTextGenerator, build_agent_generator

    if args.wm_method:
        return build_agent_generator(
            args.wm_method,
            max_new_tokens=args.wm_max_new_tokens,
            trust_remote_code=True,
            dtype=args.wm_dtype,
            disable_chat_template=False,
            attn_implementation=None,
            device_map=None,
            vllm_server_port=args.vllm_server_port,
        )
    return HFTextGenerator(
        args.wm_model_path,
        max_new_tokens=args.wm_max_new_tokens,
        trust_remote_code=True,
        dtype=args.wm_dtype,
        disable_chat_template=False,
        attn_implementation=None,
        device_map=None,
    )


def stage_label(args: argparse.Namespace) -> None:
    """Stage I: write a labelled JSONL with per-step pseudo-K labels."""
    from src.evaluation import predict_world_model_feedback
    from src.finetuning import (
        TaskTrajectory,
        canonicalize_world_model_target,
        make_blank_state_like,
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    target = canonicalize_world_model_target(args.world_model_target)

    # 1) Load expert episodes from ewm trajectory JSON(s).
    train_paths = [Path(p) for p in args.train_trajectories]
    episodes = _load_ewm_examples_grouped(train_paths, args.state_history_size)
    if not episodes:
        raise SystemExit(
            f"No expert examples extracted from {train_paths}; check the inputs."
        )

    # 2) Construct policy LM with K/V heads for action-logprob scoring.
    tok_p = AutoTokenizer.from_pretrained(args.policy_model_path, trust_remote_code=True)
    lm_p = AutoModelForCausalLM.from_pretrained(
        args.policy_model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)
    add_special_tokens_and_resize(tok_p, lm_p, SPECIAL_TOKENS)
    tok_p.padding_side = args.padding_side
    hidden_size = getattr(lm_p.config, "hidden_size", None) or getattr(
        lm_p.config, "n_embd", None
    )
    if hidden_size is None:
        raise ValueError("Cannot infer hidden size from policy model config.")
    policy = PolicyWithHeads(lm_p, hidden_size=hidden_size, kmax=args.kmax).to(device)
    policy.eval()
    ctrl_token_id = tok_p.convert_tokens_to_ids("<CTRL>")
    if ctrl_token_id is None or ctrl_token_id < 0:
        raise ValueError("Cannot find <CTRL> token id after adding special tokens.")

    # 3) Build the ewm world model generator (HF or vllm-served).
    world_model_generator = _build_world_model_generator_for_labeller(args)

    # 4) Resolve K candidates.
    if args.k_candidates:
        k_candidates = [int(x) for x in args.k_candidates.split(",") if x.strip()]
        k_candidates = sorted({k for k in k_candidates if 0 <= k <= args.kmax})
        if 0 not in k_candidates:
            k_candidates = [0] + k_candidates
    else:
        k_candidates = list(range(args.kmax + 1))

    out_path = Path(args.out_labeled_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w", encoding="utf-8")
    total_written = 0

    for episode_id, steps in tqdm(episodes.items(), desc="K-label episodes"):
        T = len(steps)
        if T == 0:
            continue

        # 4a) Per-step single-step WM predictions on the expert action.
        # We use ewm's predict_world_model_feedback which already handles the
        # full prompt/parse stack and supports both binary/state/tool_output
        # targets uniformly.
        pred_states: list[Any] = []
        pred_state_texts: list[str] = []
        for ex in steps:
            try:
                feedbacks = predict_world_model_feedback(
                    world_model_generator,
                    TaskTrajectory(
                        trajectory_index=ex.trajectory_index,
                        system_prompt=ex.system_prompt,
                        user_messages=[ex.user_prompt],
                        steps=[],
                        final_answer="",
                        initial_state=make_blank_state_like(ex.previous_state),
                    ),
                    previous_state=ex.previous_state,
                    predicted_calls=_extract_predicted_calls(ex),
                    interaction_index=ex.interaction_index,
                    world_model_target=target,
                    include_error_message_in_target=args.include_error_message_in_target,
                    include_stage_in_target=args.include_stage_in_target,
                    include_world_model_history=args.include_input_history,
                    state_history=ex.state_history,
                    input_history=ex.input_history,
                    system_prompt_max_chars=args.system_prompt_max_chars,
                    action_max_chars=args.action_max_chars,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "WM prediction failed for episode %s step %d: %s",
                    episode_id,
                    ex.interaction_index,
                    exc,
                )
                feedbacks = []
            if feedbacks:
                pred_states.append(feedbacks[-1].get("predicted_state"))
                pred_state_texts.append(feedbacks[-1].get("raw_prediction", "") or "")
            else:
                pred_states.append(None)
                pred_state_texts.append("")

        # 4b) Build obs_lists[t][k]: the textual observation at step t+k-1.
        obs_lists: list[list[str]] = []
        for t in range(T):
            obs0 = summarize_predicted_state(steps[t].previous_state)
            obs_list = [obs0]
            for k in range(1, args.kmax + 1):
                idx = t + k - 1
                if idx < T and pred_states[idx] is not None:
                    obs_list.append(summarize_predicted_state(pred_states[idx]))
                else:
                    obs_list.append(obs_list[-1])
            obs_lists.append(obs_list)

        # 4c) Score K candidates via policy LM logprob(action | prompt_K).
        scores = torch.full(
            (T, args.kmax + 1), -1e30, dtype=torch.float32, device="cpu"
        )
        prompts_buf: list[str] = []
        actions_buf: list[str] = []
        meta_buf: list[tuple[int, int]] = []

        def flush() -> None:
            nonlocal prompts_buf, actions_buf, meta_buf
            if not prompts_buf:
                return
            logp = compute_action_logprobs_batch(
                model=policy,
                tokenizer=tok_p,
                prompts=prompts_buf,
                actions=actions_buf,
                max_seq_len=args.max_seq_len,
                ctrl_token_id=ctrl_token_id,
                device=device,
                normalize=args.logprob_norm,
            ).detach().cpu()
            for i, (t2, k2) in enumerate(meta_buf):
                scores[t2, k2] = logp[i].float()
            prompts_buf, actions_buf, meta_buf = [], [], []

        for t in range(T):
            state_text = build_state_text_from_example(
                steps[t],
                include_input_history=args.include_input_history,
                system_prompt_max_chars=args.system_prompt_max_chars,
            )
            action_text = action_text_from_example(steps[t])
            for k in k_candidates:
                prompt_text = build_prompt_for_action(state_text, k, obs_lists[t][k])
                prompts_buf.append(prompt_text)
                actions_buf.append(action_text)
                meta_buf.append((t, k))
                if len(prompts_buf) >= args.score_batch_size:
                    flush()
        flush()

        # 4d) Pick the best K under logprob - lambda_k * K penalty.
        for t in range(T):
            best_k = 0
            best_val = -1e30
            for k in k_candidates:
                val = float(scores[t, k].item()) - args.lambda_k * float(k)
                if val > best_val:
                    best_val = val
                    best_k = k
            ex = steps[t]
            state_text = build_state_text_from_example(
                ex,
                include_input_history=args.include_input_history,
                system_prompt_max_chars=args.system_prompt_max_chars,
            )
            obj = {
                "episode_id": episode_id,
                "step": int(ex.interaction_index),
                "state": state_text,
                "action": action_text_from_example(ex),
                "next_state_raw": pred_state_texts[t],
                "k_label": int(best_k),
                "obs_list": obs_lists[t],
                "world_model_target": target,
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total_written += 1

    fout.close()
    print(f"[label] wrote {total_written} samples -> {out_path}")


def _extract_predicted_calls(example: Any) -> list[dict[str, Any]]:
    """Mirror evaluation.py's `to_openai_tool_calls` input shape."""
    from src.finetuning import normalize_tool_call

    action = example.action
    if isinstance(action, dict) and "tool_calls" in action:
        raw_calls = action.get("tool_calls") or []
    elif isinstance(action, list):
        raw_calls = action
    elif isinstance(action, str):
        try:
            parsed = json.loads(action)
        except Exception:  # noqa: BLE001
            return []
        if isinstance(parsed, dict) and "tool_calls" in parsed:
            raw_calls = parsed.get("tool_calls") or []
        elif isinstance(parsed, list):
            raw_calls = parsed
        else:
            return []
    else:
        return []
    return [normalize_tool_call(call) for call in raw_calls]


# =============================================================================
# Stage II (sft) -- joint LM + K-head training on labelled JSONL.
# =============================================================================


class LabeledKDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.samples: list[dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


@dataclass
class CollateConfig:
    max_seq_len: int
    kmax: int


def collate_sft(
    batch: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    cfg: CollateConfig,
) -> dict[str, torch.Tensor]:
    actions = [b["action"] for b in batch]
    k_labels = torch.tensor([int(b["k_label"]) for b in batch], dtype=torch.long)
    prompts: list[str] = []
    for b in batch:
        k = int(b["k_label"])
        obs_list = b.get("obs_list") or []
        obs_k = obs_list[k] if 0 <= k < len(obs_list) else ""
        prompts.append(build_prompt_for_action(b["state"], k, obs_k))
    lm_batch = make_lm_batch(tokenizer, prompts, actions, cfg.max_seq_len)
    return {
        "input_ids": lm_batch["input_ids"],
        "attention_mask": lm_batch["attention_mask"],
        "labels": lm_batch["labels"],
        "k_labels": k_labels,
    }


def stage_sft(args: argparse.Namespace) -> None:
    """Stage II: SFT a policy LM + K-head on the labelled JSONL."""
    # Lazy-import here so the runtime KController never has to walk the
    # transformers optimization / Trainer import chain. See the note at the
    # top of this file.
    from transformers import get_scheduler

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and args.bf16:
        param_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif device.type == "cuda" and args.fp16:
        param_dtype = torch.float16
        autocast_dtype = torch.float16
    else:
        param_dtype = torch.float32
        autocast_dtype = torch.float16
    amp_enabled = device.type == "cuda" and (args.fp16 or args.bf16)

    tokenizer, base_lm, policy, loaded_kmax = load_policy_with_heads(
        args.policy_model_path,
        device=device,
        kmax=args.kmax,
        torch_dtype=param_dtype,
        device_map=args.device_map or None,
    )
    if loaded_kmax != args.kmax:
        raise SystemExit(
            f"[sft] loaded heads kmax={loaded_kmax} != args kmax={args.kmax}. "
            "Keep kmax consistent across stages."
        )
    tokenizer.padding_side = args.padding_side
    ctrl_token_id = tokenizer.convert_tokens_to_ids("<CTRL>")
    if ctrl_token_id is None or ctrl_token_id < 0:
        raise ValueError("Cannot find <CTRL> token id after loading policy.")

    if args.gradient_checkpointing and hasattr(base_lm, "gradient_checkpointing_enable"):
        base_lm.gradient_checkpointing_enable()

    ds = LabeledKDataset(args.train_jsonl)
    cfg = CollateConfig(max_seq_len=args.max_seq_len, kmax=args.kmax)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_sft(b, tokenizer, cfg),
    )

    for p in base_lm.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.wd)

    updates_per_epoch = max(1, math.ceil(len(dl) / max(int(args.grad_accum), 1)))
    total_train_steps = max(1, int(args.epochs) * updates_per_epoch)
    warmup_steps = int(float(args.warmup_ratio) * total_train_steps)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    policy.train()
    global_step = 0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"SFT epoch {epoch + 1}/{args.epochs}")
        opt.zero_grad(set_to_none=True)
        accum_in_epoch = 0

        for batch in pbar:
            input_device = (
                base_lm.get_input_embeddings().weight.device if args.device_map else device
            )
            input_ids = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            labels = batch["labels"].to(input_device)
            k_labels = batch["k_labels"].to(input_device)
            ctrl_pos = find_ctrl_positions(input_ids, attention_mask, ctrl_token_id)

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=autocast_dtype):
                out = policy(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ctrl_pos=ctrl_pos,
                    labels=labels,
                )
                lm_loss = out["lm_loss"]
                k_logits = out["k_logits"]
                if k_labels.device != k_logits.device:
                    k_labels = k_labels.to(k_logits.device)
                k_loss = F.cross_entropy(k_logits, k_labels, reduction="mean")
                loss = lm_loss + args.beta_k * k_loss

            if not torch.isfinite(loss):
                pbar.set_postfix({"loss": "nan/inf"})
                raise SystemExit(
                    "[sft] Non-finite loss; aborting to avoid corrupting weights."
                )

            pbar.set_postfix(
                {
                    "loss": f"{float(loss.detach().cpu()):.4f}",
                    "lm": f"{float(lm_loss.detach().cpu()):.4f}",
                    "k": f"{float(k_loss.detach().cpu()):.4f}",
                }
            )

            scaled_loss = loss / max(int(args.grad_accum), 1)
            scaled_loss.backward()
            accum_in_epoch += 1
            global_step += 1

            if accum_in_epoch % args.grad_accum == 0:
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save_policy_with_heads(
                    os.path.join(args.out_dir, f"ckpt_step_{global_step}"),
                    tokenizer,
                    base_lm,
                    policy,
                    kmax=args.kmax,
                )

        if accum_in_epoch % args.grad_accum != 0:
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)

        save_policy_with_heads(
            os.path.join(args.out_dir, f"ckpt_epoch_{epoch + 1}"),
            tokenizer,
            base_lm,
            policy,
            kmax=args.kmax,
        )

    save_policy_with_heads(args.out_dir, tokenizer, base_lm, policy, kmax=args.kmax)
    print(f"[sft] saved policy + K-head -> {args.out_dir}")
    # Silence unused-var warnings for scaler in case we wire AMP later.
    del scaler


# =============================================================================
# CLI.
# =============================================================================


def _add_common_padding_and_norm(p: argparse.ArgumentParser) -> None:
    p.add_argument("--padding-side", type=str, default="left", choices=["left", "right"])
    p.add_argument(
        "--logprob-norm",
        type=str,
        default="sum",
        choices=["sum", "mean"],
        help="How to aggregate action-token logprob for K labelling.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="K-controller training (Stages I + II) for `react_wm_rl_k` in ewm-enterprisearena."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- Stage I: label --------------------------------------------------
    p = sub.add_parser("label", help="Stage I: write per-step pseudo-K labels.")
    p.add_argument(
        "--train-trajectories",
        nargs="+",
        required=True,
        help="ewm trajectory JSON file(s) (e.g. "
        "trajectories/enterpriseops_gym_multi_model_world_model_train_trajectories.json).",
    )
    p.add_argument(
        "--world-model-target",
        default="tool_execution_result_binary",
        help="Which ewm WM target the labeller uses for the foresight rollouts.",
    )
    p.add_argument("--include-error-message-in-target", action="store_true")
    p.add_argument("--include-stage-in-target", action="store_true")
    p.add_argument("--include-input-history", action="store_true")
    p.add_argument("--state-history-size", type=int, default=3)
    p.add_argument("--system-prompt-max-chars", type=int, default=0)
    p.add_argument("--action-max-chars", type=int, default=0)
    p.add_argument("--policy-model-path", type=str, required=True)
    p.add_argument(
        "--wm-model-path",
        type=str,
        default=None,
        help="HF checkpoint dir for the world model (used when --wm-method is unset).",
    )
    p.add_argument(
        "--wm-method",
        type=str,
        default=None,
        help="Hosted backend for the world model, e.g. vllm/gymops_world_model.",
    )
    p.add_argument("--wm-dtype", type=str, default="auto")
    p.add_argument(
        "--vllm-server-port",
        type=int,
        default=None,
        help="vLLM server port for the WM if --wm-method=vllm/...",
    )
    p.add_argument("--wm-max-new-tokens", type=int, default=192)
    p.add_argument("--out-labeled-jsonl", type=str, required=True)
    p.add_argument("--kmax", type=int, default=3)
    p.add_argument("--lambda-k", type=float, default=0.2)
    p.add_argument("--k-candidates", type=str, default="")
    p.add_argument("--score-batch-size", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    _add_common_padding_and_norm(p)

    # ---- Stage II: sft ---------------------------------------------------
    p = sub.add_parser("sft", help="Stage II: SFT policy LM + K-head on labelled JSONL.")
    p.add_argument("--train-jsonl", type=str, required=True)
    p.add_argument("--policy-model-path", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--kmax", type=int, default=3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--beta-k", type=float, default=0.5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"],
    )
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--device-map", type=str, default="")
    p.add_argument("--save-every-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    _add_common_padding_and_norm(p)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "label":
        if not args.wm_model_path and not args.wm_method:
            parser.error("--wm-model-path or --wm-method is required for stage `label`.")
        stage_label(args)
    elif args.cmd == "sft":
        stage_sft(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
