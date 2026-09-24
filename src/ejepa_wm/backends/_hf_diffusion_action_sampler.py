"""Local Hugging Face DiffusionGemma backend for beam-plan action sampling.

Heavy dependencies are imported lazily so ordinary ``ejepa_wm`` users do not need
PyTorch or Transformers. Unlike an OpenAI-compatible endpoint, this backend
duplicates the shared prompt in a tensor batch and produces every candidate in
one local ``generate`` call.
"""

from __future__ import annotations

import threading
from typing import Any


class HuggingFaceDiffusionActionSampler:
    """Generate independent DiffusionGemma plans in one local HF batch."""

    supports_parallel_requests = False

    def __init__(
        self,
        model: str,
        *,
        max_new_tokens: int = 256,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_denoising_steps: int | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        if "nvfp4" in model.lower():
            raise ValueError(
                "The NVIDIA DiffusionGemma NVFP4 checkpoint is packaged for the "
                "vLLM/ModelOpt runtime, not Hugging Face Transformers. Use "
                "google/diffusiongemma-26B-A4B-it with the huggingface backend."
            )
        try:
            import torch
            from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "The Hugging Face DiffusionGemma sampler requires torch, accelerate, "
                "and a Transformers release containing "
                "DiffusionGemmaForBlockDiffusion (transformers>=5.11)."
            ) from exc

        resolved_dtype: Any = dtype
        if dtype != "auto":
            resolved_dtype = getattr(torch, dtype, None)
            if resolved_dtype is None:
                raise ValueError(f"Unsupported Hugging Face action-sampler dtype: {dtype}")

        load_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "dtype": resolved_dtype,
            "trust_remote_code": trust_remote_code,
        }
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(model, **load_kwargs)
        self.model.eval()
        self.model_name = model
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.max_denoising_steps = (
            max(1, int(max_denoising_steps)) if max_denoising_steps is not None else None
        )
        self._torch = torch
        self._generate_lock = threading.Lock()

    def _model_device(self) -> Any:
        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        return next(self.model.parameters()).device

    @staticmethod
    def _repeat_batch(inputs: Any, num_samples: int) -> Any:
        for key, value in list(inputs.items()):
            if hasattr(value, "shape") and value.shape and value.shape[0] == 1:
                repeats = (num_samples,) + (1,) * (value.ndim - 1)
                inputs[key] = value.repeat(*repeats)
        return inputs

    def generate_samples(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.8,
        num_samples: int = 1,
        response_format: dict[str, Any] | None = None,
    ) -> list[str]:
        """Return ``num_samples`` completions from one batched diffusion pass."""
        del response_format  # JSON-only output is imposed by the shared prompt.
        count = max(1, int(num_samples))
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = self._repeat_batch(inputs, count).to(self._model_device())
        input_length = inputs["input_ids"].shape[-1]
        generate_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        # DiffusionGemma uses its masking-time range instead of autoregressive
        # temperature. Preserve the default lower bound while allowing the existing
        # imagined-temperature knob to control exploration.
        if temperature > 0:
            t_max = min(1.0, max(0.01, float(temperature)))
            generate_kwargs.update(t_min=min(0.4, t_max), t_max=t_max)
        if self.max_denoising_steps is not None:
            generate_kwargs["max_denoising_steps"] = self.max_denoising_steps

        # Transformers generation mutates generation state, so guard against an
        # accidental concurrent call. Candidate diversity itself remains batched.
        with self._generate_lock, self._torch.inference_mode():
            outputs = self.model.generate(**inputs, **generate_kwargs)
        sequences = getattr(outputs, "sequences", outputs)
        generated = sequences[:, input_length:]
        return list(self.processor.batch_decode(generated, skip_special_tokens=True))

    def generate_from_messages(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.8,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        return self.generate_samples(
            messages,
            temperature=temperature,
            num_samples=1,
            response_format=response_format,
        )[0]
