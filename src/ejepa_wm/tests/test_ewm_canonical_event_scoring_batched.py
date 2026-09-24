"""Tests for ``logits_to_field_probs_batched`` -- the batched-host-transfer form of
``logits_to_field_probs`` used by all of beam_plan's, hier_latent_cem's, and the critic
trigger's rollout scoring (via ``score_action_plans_canonical_event``). The whole point of the
batched form is that it must be numerically IDENTICAL to calling the per-row form once per row
(only the host-transfer pattern changes, not the math) -- verified here per-row, and this
mirrors the live 0.000e+00-score-difference check done against a real checkpoint.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ejepa_wm.backends._ewm_canonical_event_scoring import (
    MISSING_INFO_FIELD,
    logits_to_field_probs,
    logits_to_field_probs_batched,
)

VOCAB = {
    "execution_status": ["success", "failure", "unknown"],
    "progress_signal": ["positive", "neutral", "negative"],
    MISSING_INFO_FIELD: ["object_id", "date", "none"],
}


def _per_row_reference(field_logits_batched: dict, vocab, temperature: float) -> list[dict]:
    """The old per-row path: slice each field's [B, C] tensor into B single-row calls."""
    batch_size = next(iter(field_logits_batched.values())).shape[0]
    results = []
    for row in range(batch_size):
        per_field = {name: logits[row] for name, logits in field_logits_batched.items()}
        results.append(logits_to_field_probs(per_field, vocab, temperature=temperature))
    return results


def test_batched_matches_per_row_reference_bit_for_bit():
    torch.manual_seed(0)
    batch = 7
    field_logits = {
        "execution_status": torch.randn(batch, 3),
        "progress_signal": torch.randn(batch, 3),
        MISSING_INFO_FIELD: torch.randn(batch, 3),
    }
    expected = _per_row_reference(field_logits, VOCAB, temperature=1.0)
    actual = logits_to_field_probs_batched(field_logits, VOCAB, temperature=1.0)
    assert len(actual) == batch
    for row in range(batch):
        for field in VOCAB:
            for category in VOCAB[field]:
                assert actual[row][field][category] == pytest.approx(expected[row][field][category], abs=0.0)


def test_batched_matches_per_row_reference_with_temperature_scaling():
    torch.manual_seed(1)
    batch = 4
    field_logits = {
        "execution_status": torch.randn(batch, 3) * 3.0,
        "progress_signal": torch.randn(batch, 3) * 3.0,
    }
    vocab = {k: v for k, v in VOCAB.items() if k != MISSING_INFO_FIELD}
    expected = _per_row_reference(field_logits, vocab, temperature=0.5)
    actual = logits_to_field_probs_batched(field_logits, vocab, temperature=0.5)
    for row in range(batch):
        for field in vocab:
            for category in vocab[field]:
                assert actual[row][field][category] == pytest.approx(expected[row][field][category], abs=0.0)


def test_batched_uses_sigmoid_for_multi_label_field():
    field_logits = {MISSING_INFO_FIELD: torch.tensor([[2.0, -2.0, 0.0]])}
    out = logits_to_field_probs_batched(field_logits, VOCAB, temperature=1.0)
    probs = out[0][MISSING_INFO_FIELD]
    # sigmoid, not softmax -- entries need not sum to 1 and each is independently in (0, 1)
    assert probs["object_id"] == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())
    assert probs["date"] == pytest.approx(torch.sigmoid(torch.tensor(-2.0)).item())
    assert sum(probs.values()) != pytest.approx(1.0)


def test_batched_skips_fields_not_in_vocab():
    field_logits = {"unknown_field": torch.randn(3, 5), "execution_status": torch.randn(3, 3)}
    out = logits_to_field_probs_batched(field_logits, VOCAB, temperature=1.0)
    assert len(out) == 3
    for row in out:
        assert "unknown_field" not in row
        assert "execution_status" in row


def test_batched_handles_single_row_1d_logits():
    # logits_to_field_probs_batched must also accept unbatched [C] tensors (dim() == 1) the same
    # way single-plan callers (e.g. the critic trigger's one-step probe) produce them.
    field_logits = {"execution_status": torch.tensor([1.0, 0.0, -1.0])}
    out = logits_to_field_probs_batched(field_logits, VOCAB, temperature=1.0)
    assert len(out) == 1
    expected = torch.softmax(torch.tensor([1.0, 0.0, -1.0]), dim=-1)
    assert out[0]["execution_status"]["success"] == pytest.approx(expected[0].item())
