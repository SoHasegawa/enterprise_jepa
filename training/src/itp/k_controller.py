"""Runtime K-controller for ``react_wm_rl_k`` (decoupled K head).

This is the deployment hook for the K-controller checkpoint produced by
:mod:`src.itp.training.train_adaptive_k`. The K-controller is the **small
policy LM + K-head** trained on offline-RL pseudo-K labels; at deployment
time it predicts how many steps to look ahead with the world model BEFORE
the real action policy (e.g. GPT-5.1) picks tool calls.

Decoupling rationale: the action policy at inference may be an external API
(GPT-5.1) that cannot host the K-head. Splitting the responsibility lets a
small local policy LM with a K-head learn from labelled K signals while the
API model keeps choosing the actual tool calls.

Usage example::

    from src.itp.k_controller import KController
    k = KController(
        model_path="/path/to/k_controller/policy_sft_khead",
        kmax=3,
        device_str="cuda",
        dtype_str="bf16",
    )
    k_for_this_turn = k.decide_k(state_text)
    foresight = build_world_model_foresight(..., k_steps=k_for_this_turn)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KDecision:
    k: int
    logits: Optional[list[float]] = None  # length kmax+1, softmax-able floats
    raw: str = ""


class KController:
    """Wraps a K-controller Stage II checkpoint for runtime K decisions.

    Lazy-imports ``torch`` and ``transformers`` so this module can be imported
    on machines without a CUDA build (e.g. for unit tests or for runs that
    never use the ``react_wm_rl_k`` mode). Only constructs the underlying
    model when :meth:`decide_k` is first called.
    """

    def __init__(
        self,
        model_path: str,
        kmax: int,
        device_str: str = "auto",
        dtype_str: str = "auto",
        do_sample: bool = False,
        temperature: float = 1.0,
        max_seq_len: int = 2048,
        padding_side: str = "left",
    ) -> None:
        self.model_path = str(model_path)
        self.kmax = int(kmax)
        self.device_str = device_str
        self.dtype_str = dtype_str
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.max_seq_len = int(max_seq_len)
        self.padding_side = str(padding_side)

        self._tokenizer = None
        self._policy = None
        self._ctrl_token_id: Optional[int] = None
        self._device = None
        self._torch = None
        self._helpers: dict = {}

    def _ensure_loaded(self) -> None:
        if self._policy is not None:
            return
        import torch

        from src.itp.training.train_adaptive_k import (
            build_prompt_for_controller,
            find_ctrl_positions,
            load_policy_with_heads,
            make_controller_batch,
        )

        self._torch = torch
        self._helpers = {
            "build_prompt_for_controller": build_prompt_for_controller,
            "make_controller_batch": make_controller_batch,
            "find_ctrl_positions": find_ctrl_positions,
        }

        device = self._select_device(self.device_str)
        dtype = self._select_dtype(device, self.dtype_str)

        tokenizer, _base_lm, policy, loaded_kmax = load_policy_with_heads(
            self.model_path,
            device=device,
            kmax=self.kmax,
            torch_dtype=dtype,
        )
        if loaded_kmax != self.kmax:
            logger.warning(
                "K-controller checkpoint kmax=%d != requested kmax=%d; "
                "using checkpoint kmax to avoid silent truncation.",
                loaded_kmax,
                self.kmax,
            )
            self.kmax = int(loaded_kmax)
        tokenizer.padding_side = self.padding_side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        ctrl_id = tokenizer.convert_tokens_to_ids("<CTRL>")
        if ctrl_id is None or ctrl_id < 0:
            raise RuntimeError(
                f"K-controller checkpoint at {self.model_path} has no <CTRL> "
                "special token -- it was not trained with stage I+II of "
                "src.itp.training.train_adaptive_k."
            )

        policy.eval()
        self._tokenizer = tokenizer
        self._policy = policy
        self._ctrl_token_id = int(ctrl_id)
        self._device = device

    @staticmethod
    def _select_device(preferred: str):
        import torch

        if preferred and preferred not in {"auto", ""}:
            return torch.device(preferred)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _select_dtype(device, requested: str):
        import torch

        r = (requested or "auto").lower()
        if r == "fp32" or device.type == "cpu":
            return torch.float32
        if r == "bf16":
            return torch.bfloat16
        if r == "fp16":
            return torch.float16
        if device.type == "cuda":
            return torch.bfloat16
        if device.type == "mps":
            return torch.float16
        return torch.float32

    def decide_k(self, state_text: str) -> KDecision:
        """Return the K-head's chosen K for the given state text.

        ``state_text`` must be the textual rendering produced by the same
        prompt format used during stage II SFT -- i.e. the user-side body of
        :func:`src.finetuning.build_state_prediction_chat_messages` for the
        configured ``--world-model-target``. The runtime orchestrator is
        responsible for passing the right text in.
        """
        self._ensure_loaded()
        torch = self._torch
        ctrl_id = self._ctrl_token_id

        build_prompt_for_controller = self._helpers["build_prompt_for_controller"]
        make_controller_batch = self._helpers["make_controller_batch"]
        find_ctrl_positions = self._helpers["find_ctrl_positions"]

        prompt = build_prompt_for_controller(state_text)
        batch = make_controller_batch(self._tokenizer, [prompt], self.max_seq_len)
        input_ids = batch["input_ids"].to(self._device)
        attention_mask = batch["attention_mask"].to(self._device)
        ctrl_pos = find_ctrl_positions(input_ids, attention_mask, ctrl_id)

        with torch.no_grad():
            out = self._policy(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ctrl_pos=ctrl_pos,
                labels=None,
            )
        k_logits = out["k_logits"].float().detach().cpu()[0]

        if self.do_sample and self.temperature > 0:
            probs = torch.softmax(k_logits / self.temperature, dim=-1)
            k = int(torch.multinomial(probs, num_samples=1).item())
        else:
            k = int(torch.argmax(k_logits).item())

        k = max(0, min(self.kmax, k))
        return KDecision(k=k, logits=k_logits.tolist(), raw=f"argmax={k}")

    def __call__(self, state_text: str) -> int:
        return self.decide_k(state_text).k


__all__ = ["KController", "KDecision"]
