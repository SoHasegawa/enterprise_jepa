"""Vendored runtime K-controller for the ``react_wm_rl_k`` dynamic-K mode.

Self-contained port of ``ewm/src/itp/k_controller.py`` plus the inference-only
subset of ``ewm/src/itp/training/train_adaptive_k.py`` it depends on
(``PolicyWithHeads``, ``load_policy_with_heads``, the ``<CTRL>`` batching
helpers). Vendored here so the purple executor imports nothing cross-directory.

``torch`` / ``transformers`` are imported at module top, so ``wm_react.py``
imports this module **lazily** — only when ``K_CONTROLLER=react_wm_rl_k`` — and
the other strategies never require those heavy deps. The checkpoint is the
small policy-LM + K-head produced by EWM's ``train_adaptive_k`` Stage I+II.

Usage (from wm_react.py):

    from k_controller import KController
    ctrl = KController(model_path=os.getenv("K_CONTROLLER_MODEL_PATH"),
                       kmax=wm_imagined_max_steps, device_str="auto", dtype_str="auto")
    k = ctrl.decide_k(state_text).k   # 0..kmax
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<CTRL>", "<STATE>", "</STATE>", "<FORESIGHT>", "</FORESIGHT>"]


def build_prompt_for_controller(state_text: str) -> str:
    """K-controller scoring prompt (must match train_adaptive_k Stage II / runtime)."""
    return (
        "You are an enterprise-operations agent.\n"
        "Decide how many steps to look ahead with a world model.\n"
        "<STATE>\n"
        f"{state_text}\n"
        "</STATE>\n"
        "<CTRL>"
    )


def _ensure_pad(tokenizer: "AutoTokenizer") -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token_id; set pad/eos first.")
        tokenizer.pad_token = tokenizer.eos_token


def _truncate_left_1d(ids: "torch.Tensor", mask: "torch.Tensor", max_len: int):
    if ids.numel() <= max_len:
        return ids, mask
    return ids[-max_len:], mask[-max_len:]


def _pad_1d_batch(
    seqs: list["torch.Tensor"],
    pad_value: int,
    padding_side: "Literal['left', 'right']",
) -> "torch.Tensor":
    assert len(seqs) > 0
    max_len = max(int(s.numel()) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        length = int(s.numel())
        if padding_side == "left":
            out[i, max_len - length:] = s
        else:
            out[i, :length] = s
    return out


def make_controller_batch(tokenizer: "AutoTokenizer", prompts: list[str], max_seq_len: int) -> dict:
    _ensure_pad(tokenizer)
    pad_id = tokenizer.pad_token_id
    padding_side = getattr(tokenizer, "padding_side", "right")
    ids_list, mask_list = [], []
    for p in prompts:
        p_ids = tokenizer(p, add_special_tokens=False).input_ids
        input_ids = torch.tensor(p_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        input_ids, attention_mask = _truncate_left_1d(input_ids, attention_mask, max_seq_len)
        ids_list.append(input_ids)
        mask_list.append(attention_mask)
    return {
        "input_ids": _pad_1d_batch(ids_list, pad_value=pad_id, padding_side=padding_side),
        "attention_mask": _pad_1d_batch(mask_list, pad_value=0, padding_side=padding_side),
    }


def find_ctrl_positions(
    input_ids: "torch.Tensor", attention_mask: "torch.Tensor", ctrl_token_id: int
) -> "torch.Tensor":
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
    tokenizer: "AutoTokenizer", model: "AutoModelForCausalLM", special_tokens: list[str]
) -> int:
    old_vocab = model.get_input_embeddings().weight.size(0)
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    _ensure_pad(tokenizer)
    if n_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            emb[old_vocab:] = emb[:old_vocab].mean(dim=0, keepdim=True)
    elif model.get_input_embeddings().weight.size(0) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return n_added


class PolicyWithHeads(nn.Module):
    """Base causal LM augmented with a K-head and a value head at ``<CTRL>``."""

    def __init__(self, lm: "AutoModelForCausalLM", hidden_size: int, kmax: int):
        super().__init__()
        self.lm = lm
        self.kmax = kmax
        self.k_head = nn.Linear(hidden_size, kmax + 1)
        self.v_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, ctrl_pos, labels=None):
        if hasattr(self.lm, "model") and hasattr(self.lm, "lm_head"):
            base_out = self.lm.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True
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
        h_ctrl_heads = h_ctrl.to(dtype=head_dtype) if h_ctrl.dtype != head_dtype else h_ctrl
        return {
            "logits": logits,
            "k_logits": self.k_head(h_ctrl_heads),
            "values": self.v_head(h_ctrl_heads).squeeze(-1),
        }


def load_policy_with_heads(
    model_dir: str,
    device: "torch.device",
    kmax: int,
    torch_dtype: "torch.dtype" = torch.float16,
):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    base_lm = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    ).to(device)
    add_special_tokens_and_resize(tokenizer, base_lm, SPECIAL_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    hidden_size = getattr(base_lm.config, "hidden_size", None) or getattr(base_lm.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Cannot infer hidden size from model config.")
    heads_path = os.path.join(model_dir, "adaptive_heads.pt")
    if not os.path.exists(heads_path):
        raise FileNotFoundError(
            f"K-controller checkpoint at {model_dir} has no adaptive_heads.pt — it was not "
            "trained with stage I+II of train_adaptive_k."
        )
    heads = torch.load(heads_path, map_location="cpu")
    saved_kmax = int(heads["kmax"])
    policy = PolicyWithHeads(base_lm, hidden_size=hidden_size, kmax=saved_kmax)
    policy.k_head.load_state_dict(heads["k_head"])
    policy.v_head.load_state_dict(heads["v_head"])
    policy = policy.to(device=device, dtype=torch_dtype)
    return tokenizer, base_lm, policy, saved_kmax


@dataclass
class KDecision:
    k: int
    logits: Optional[list] = None
    raw: str = ""


class KController:
    """Wraps a K-controller Stage II checkpoint for runtime K decisions.

    Lazily constructs the model on the first ``decide_k`` call.
    """

    def __init__(
        self,
        model_path: str,
        kmax: int,
        device_str: str = "auto",
        dtype_str: str = "auto",
        max_seq_len: int = 2048,
        padding_side: str = "left",
    ) -> None:
        self.model_path = str(model_path)
        self.kmax = int(kmax)
        self.device_str = device_str
        self.dtype_str = dtype_str
        self.max_seq_len = int(max_seq_len)
        self.padding_side = str(padding_side)
        self._tokenizer = None
        self._policy = None
        self._ctrl_token_id: Optional[int] = None
        self._device = None

    @staticmethod
    def _select_device(preferred: str) -> "torch.device":
        if preferred and preferred not in {"auto", ""}:
            return torch.device(preferred)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _select_dtype(device: "torch.device", requested: str) -> "torch.dtype":
        r = (requested or "auto").lower()
        if r == "fp32" or device.type == "cpu":
            return torch.float32
        if r == "bf16":
            return torch.bfloat16
        if r == "fp16":
            return torch.float16
        return torch.bfloat16 if device.type == "cuda" else torch.float32

    def _ensure_loaded(self) -> None:
        if self._policy is not None:
            return
        device = self._select_device(self.device_str)
        dtype = self._select_dtype(device, self.dtype_str)
        tokenizer, _base_lm, policy, loaded_kmax = load_policy_with_heads(
            self.model_path, device=device, kmax=self.kmax, torch_dtype=dtype
        )
        if loaded_kmax != self.kmax:
            logger.warning(
                "K-controller checkpoint kmax=%d != requested kmax=%d; using checkpoint kmax.",
                loaded_kmax, self.kmax,
            )
            self.kmax = int(loaded_kmax)
        tokenizer.padding_side = self.padding_side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        ctrl_id = tokenizer.convert_tokens_to_ids("<CTRL>")
        if ctrl_id is None or ctrl_id < 0:
            raise RuntimeError(
                f"K-controller checkpoint at {self.model_path} has no <CTRL> token."
            )
        policy.eval()
        self._tokenizer = tokenizer
        self._policy = policy
        self._ctrl_token_id = int(ctrl_id)
        self._device = device

    def decide_k(self, state_text: str) -> KDecision:
        self._ensure_loaded()
        prompt = build_prompt_for_controller(state_text)
        batch = make_controller_batch(self._tokenizer, [prompt], self.max_seq_len)
        input_ids = batch["input_ids"].to(self._device)
        attention_mask = batch["attention_mask"].to(self._device)
        ctrl_pos = find_ctrl_positions(input_ids, attention_mask, self._ctrl_token_id)
        with torch.no_grad():
            out = self._policy(input_ids=input_ids, attention_mask=attention_mask, ctrl_pos=ctrl_pos, labels=None)
        k_logits = out["k_logits"].float().detach().cpu()[0]
        k = int(torch.argmax(k_logits).item())
        k = max(0, min(self.kmax, k))
        return KDecision(k=k, logits=k_logits.tolist(), raw=f"argmax={k}")

    def __call__(self, state_text: str) -> int:
        return self.decide_k(state_text).k


__all__ = ["KController", "KDecision"]
