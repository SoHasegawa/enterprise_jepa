"""CPU-only checks for the opt-in JEPA fast-inference helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from ejepa_wm.backends import _ewm_jepa as jepa


def test_resolve_fast_inference_mode_values():
    assert jepa.resolve_fast_inference_mode("") is None
    assert jepa.resolve_fast_inference_mode("0") is None
    assert jepa.resolve_fast_inference_mode("1") == "reduce-overhead"
    assert jepa.resolve_fast_inference_mode("graphs") == "reduce-overhead"
    assert jepa.resolve_fast_inference_mode("default") == "default"
    with pytest.raises(ValueError):
        jepa.resolve_fast_inference_mode("turbo")


def test_resolve_fast_inference_mode_reads_env(monkeypatch):
    monkeypatch.delenv(jepa.FAST_INFERENCE_ENV, raising=False)
    assert jepa.resolve_fast_inference_mode() is None
    monkeypatch.setenv(jepa.FAST_INFERENCE_ENV, "1")
    assert jepa.resolve_fast_inference_mode() == "reduce-overhead"


def test_pad_to_multiple_pads_ids_and_mask_up_to_bucket():
    enc = {
        "input_ids": torch.ones(2, 301, dtype=torch.long),
        "attention_mask": torch.ones(2, 301, dtype=torch.long),
    }
    out = jepa.pad_to_multiple(enc, 64, max_length=2048, pad_id=7)
    assert out["input_ids"].shape == (2, 320)
    assert out["attention_mask"].shape == (2, 320)
    assert out["input_ids"][:, 301:].eq(7).all()
    assert out["attention_mask"][:, 301:].eq(0).all()
    assert out["attention_mask"][:, :301].eq(1).all()


def test_pad_to_multiple_respects_max_length_and_noop_cases():
    enc = {
        "input_ids": torch.ones(1, 250, dtype=torch.long),
        "attention_mask": torch.ones(1, 250, dtype=torch.long),
    }
    assert jepa.pad_to_multiple(enc, 64, max_length=256, pad_id=0)["input_ids"].shape == (1, 256)
    assert jepa.pad_to_multiple(enc, 1, max_length=256, pad_id=0) is enc
    exact = {"input_ids": torch.ones(1, 64, dtype=torch.long)}
    assert jepa.pad_to_multiple(exact, 64, max_length=256, pad_id=0) is exact


def test_compiled_child_wrapper_clones_and_delegates():
    inner = torch.nn.Linear(3, 2)
    inner.some_attr = "cfg"
    wrapped = jepa._CompiledChild(inner)
    x = torch.zeros(1, 3)
    out = wrapped(x)
    assert torch.equal(out, inner(x))
    assert wrapped.get_encoder() is wrapped
    assert wrapped.some_attr == "cfg"
    assert sum(p.numel() for p in wrapped.parameters()) == 8
