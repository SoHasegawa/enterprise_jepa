from __future__ import annotations

import contextlib
import sys
import types

import pytest

from ejepa_wm.backends._hf_diffusion_action_sampler import (
    HuggingFaceDiffusionActionSampler,
)


class _FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    def repeat(self, *repeats):
        return _FakeTensor(
            tuple(size * factor for size, factor in zip(self.shape, repeats, strict=True))
        )

    def __getitem__(self, key):
        row_key, column_key = key
        rows = self.shape[0] if isinstance(row_key, slice) else 1
        columns = self.shape[1] - (column_key.start or 0)
        return _FakeTensor((rows, columns))


class _FakeBatch(dict):
    def to(self, device):
        self.device = device
        return self


def test_huggingface_sampler_batches_candidates_in_one_generate(monkeypatch):
    model_calls = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            assert model == "google/diffusiongemma-26B-A4B-it"
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["role"] == "user"
            return _FakeBatch(input_ids=_FakeTensor((1, 4)), attention_mask=_FakeTensor((1, 4)))

        def batch_decode(self, generated, skip_special_tokens):
            assert generated.shape == (3, 2)
            return [f"plan-{index}" for index in range(generated.shape[0])]

    class FakeModel:
        device = "cuda:0"

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            assert kwargs["device_map"] == "auto"
            assert kwargs["dtype"] == "bf16"
            return cls()

        def eval(self):
            return self

        def generate(self, **kwargs):
            model_calls.append(kwargs)
            assert kwargs["input_ids"].shape == (3, 4)
            return types.SimpleNamespace(sequences=_FakeTensor((3, 6)))

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bf16"
    fake_torch.inference_mode = contextlib.nullcontext
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeProcessor
    fake_transformers.DiffusionGemmaForBlockDiffusion = FakeModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    sampler = HuggingFaceDiffusionActionSampler(
        "google/diffusiongemma-26B-A4B-it",
        max_new_tokens=256,
        max_denoising_steps=24,
    )
    samples = sampler.generate_samples(
        [{"role": "user", "content": "plan"}], temperature=0.7, num_samples=3
    )

    assert samples == ["plan-0", "plan-1", "plan-2"]
    assert len(model_calls) == 1
    assert model_calls[0]["max_new_tokens"] == 256
    assert model_calls[0]["max_denoising_steps"] == 24
    assert model_calls[0]["t_min"] == pytest.approx(0.4)
    assert model_calls[0]["t_max"] == pytest.approx(0.7)


def test_huggingface_sampler_rejects_vllm_only_nvfp4_checkpoint():
    with pytest.raises(ValueError, match="vLLM/ModelOpt"):
        HuggingFaceDiffusionActionSampler("nvidia/diffusiongemma-26B-A4B-it-NVFP4")
