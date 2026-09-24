"""Tests for the ``predictor_arch="transformer"`` (AdaLN causal-attention) predictor in
``_ewm_jepa.TextLeWorldModel`` -- ported from EWM's ``finetuning_jepa.py`` alongside the
existing ``mlp`` predictor. Guarded by ``importorskip('torch')`` since ``_ewm_jepa`` imports
torch at module top; live end-to-end verification against a real trained checkpoint
(``checkpoints/data_jepa_pred_transformer2``) was done separately outside this suite.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ejepa_wm.backends._ewm_jepa import (
    AdaLNPredictorBlock,
    AdaLNTransformerPredictor,
    TextLeWorldModel,
    _extend_batch_frame_history,
    _modulate,
)


class _FakeBackbone(torch.nn.Module):
    """Minimal stand-in with the config surface resolve_backbone_hidden_size needs."""

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden_size})()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids=None, attention_mask=None):
        return self.linear(torch.zeros(input_ids.shape[0], input_ids.shape[1], 16))


def test_modulate_is_identity_at_zero():
    x = torch.randn(2, 3, 4)
    zero = torch.zeros_like(x)
    assert torch.allclose(_modulate(x, zero, zero), x)


def test_adaln_block_is_identity_at_init():
    # adaln_modulation's final Linear is zero-initialized -> gate=0, scale=shift=0 -> exact
    # identity regardless of the attention/mlp weights, per the class's own design contract.
    # The batched [B*heads, T, T] mask is built by AdaLNTransformerPredictor._attention_mask
    # (the block itself just consumes a precomputed mask), so reuse that helper here too.
    heads = 2
    predictor = AdaLNTransformerPredictor(
        latent_dim=8, dim=8, layers=1, heads=heads, dropout=0.0,
        cond_inputs=1, max_positions=4, mlp_ratio=2.0, output_layernorm=True,
    )
    block = AdaLNPredictorBlock(dim=8, heads=heads, dropout=0.0, mlp_ratio=2.0)
    block.eval()
    x = torch.randn(2, 3, 8)
    cond = torch.randn(2, 3, 8)
    valid = torch.ones(2, 3, dtype=torch.bool)
    attn_mask = predictor._attention_mask(valid)
    out = block(x, cond, attn_mask)
    assert torch.allclose(out, x, atol=1e-5)


def test_transformer_predictor_dim_must_divide_by_heads():
    with pytest.raises(ValueError, match="must be divisible by"):
        AdaLNTransformerPredictor(
            latent_dim=8, dim=10, layers=1, heads=3, dropout=0.0,
            cond_inputs=1, max_positions=4, mlp_ratio=2.0, output_layernorm=True,
        )


def test_transformer_predictor_forward_shape():
    pred = AdaLNTransformerPredictor(
        latent_dim=8, dim=8, layers=2, heads=2, dropout=0.0,
        cond_inputs=1, max_positions=4, mlp_ratio=2.0, output_layernorm=True,
    )
    pred.eval()
    tokens = torch.randn(2, 3, 8)   # [context, e_{t-1}, e_t]
    cond = torch.randn(2, 3, 8)
    valid = torch.ones(2, 3, dtype=torch.bool)
    event, state = pred(tokens, cond, valid)
    assert event.shape == (2, 8)
    assert state.shape == (2, 8)
    assert torch.isfinite(event).all() and torch.isfinite(state).all()


def _build_model(predictor_arch: str, **kwargs) -> TextLeWorldModel:
    return TextLeWorldModel(
        backbone=_FakeBackbone(hidden_size=16),
        latent_dim=16,
        memory_tokens=2,
        dropout=0.0,
        predictor_hidden_multiplier=2.0,
        goal_conditioning=False,
        predictor_arch=predictor_arch,
        predictor_transformer_dim=16,
        predictor_transformer_layers=1,
        predictor_transformer_heads=2,
        predictor_history_length=3,
        **kwargs,
    )


def test_model_builds_mlp_predictor_by_default():
    model = _build_model("mlp")
    assert model.predictor_arch == "mlp"
    assert isinstance(model.predictor, torch.nn.Sequential)


def test_model_builds_transformer_predictor():
    model = _build_model("transformer")
    assert model.predictor_arch == "transformer"
    assert isinstance(model.predictor, AdaLNTransformerPredictor)
    assert model.predictor.max_positions == 4  # predictor_history_length(3) + 1


def test_model_rejects_unknown_predictor_arch():
    with pytest.raises(ValueError, match="unknown predictor_arch"):
        _build_model("rnn")


def test_predict_latent_transformer_matches_with_state_variant():
    model = _build_model("transformer")
    model.eval()
    z_current = torch.randn(2, 16)
    z_action = torch.randn(2, 16)
    z_context = torch.randn(2, 16)
    with torch.no_grad():
        latent, logits = model.predict_latent(z_current, z_action, z_context)
        latent2, logits2, state = model.predict_latent_with_state(z_current, z_action, z_context)
    assert torch.allclose(latent, latent2)
    assert logits is None and logits2 is None
    assert state is not None and state.shape == (2, 16)


def test_predict_latent_mlp_has_no_state():
    model = _build_model("mlp")
    model.eval()
    z_current = torch.randn(2, 16)
    z_action = torch.randn(2, 16)
    z_context = torch.randn(2, 16)
    with torch.no_grad():
        _, _, state = model.predict_latent_with_state(z_current, z_action, z_context)
    assert state is None


def test_latent_delta_prediction_adds_to_current():
    delta_model = _build_model("mlp", latent_delta_prediction=True)
    plain_model = _build_model("mlp", latent_delta_prediction=False)
    delta_model.eval()
    plain_model.eval()
    # Copy weights so the two models compute the identical raw predictor output.
    plain_model.load_state_dict(delta_model.state_dict(), strict=False)
    z_current = torch.randn(2, 16)
    z_action = torch.randn(2, 16)
    z_context = torch.randn(2, 16)
    with torch.no_grad():
        raw = delta_model.predictor(
            torch.cat([z_current, delta_model._predictor_conditioning(z_action, z_context, None)], dim=-1)
        )
        delta_latent, _ = delta_model.predict_latent(z_current, z_action, z_context)
    assert torch.allclose(delta_latent, z_current + raw, atol=1e-5)


def test_transformer_predictor_degenerate_frame_history_is_two_positions():
    """Without frame_history the sequence is exactly [context, z_current] -- the documented
    degenerate case every existing ejepa_wm caller (beam_plan, hier_latent_cem) relies on."""
    model = _build_model("transformer")
    model.eval()
    calls = {}
    original_forward = model.predictor.forward

    def spy_forward(tokens, cond, valid):
        calls["tokens_shape"] = tuple(tokens.shape)
        calls["valid"] = valid.clone()
        return original_forward(tokens, cond, valid)

    model.predictor.forward = spy_forward
    z_current = torch.randn(1, 16)
    z_action = torch.randn(1, 16)
    z_context = torch.randn(1, 16)
    with torch.no_grad():
        model.predict_latent(z_current, z_action, z_context)
    assert calls["tokens_shape"] == (1, 2, 16)  # [context, z_current]
    assert bool(calls["valid"].all())


def test_extend_batch_frame_history_from_none():
    device = torch.device("cpu")
    active = torch.tensor([0, 1, 2])
    frame, action, valid = _extend_batch_frame_history(
        None, 3, active, torch.randn(3, 8), torch.randn(3, 8), device
    )
    assert frame.shape == (3, 1, 8)
    assert action.shape == (3, 1, 8)
    assert valid.shape == (3, 1)
    assert bool(valid.all())


def test_extend_batch_frame_history_partial_active_pads_and_masks_rest():
    device = torch.device("cpu")
    frame_history = (torch.randn(4, 1, 8), torch.randn(4, 1, 8), torch.ones(4, 1, dtype=torch.bool))
    active = torch.tensor([0, 2])  # plans 1 and 3 already finished this rollout
    new_frame_active = torch.randn(2, 8)
    new_action_active = torch.randn(2, 8)
    frames, actions, valid = _extend_batch_frame_history(
        frame_history, 4, active, new_frame_active, new_action_active, device
    )
    assert frames.shape == (4, 2, 8)
    assert bool(valid[:, 0].all())  # first (seed) position untouched, still valid for all 4
    assert valid[0, 1] and valid[2, 1]
    assert not valid[1, 1] and not valid[3, 1]  # inactive candidates: padded + masked invalid
    assert torch.allclose(frames[0, 1], new_frame_active[0])
    assert torch.allclose(frames[2, 1], new_frame_active[1])
    assert torch.allclose(frames[1, 1], torch.zeros(8))
    assert torch.allclose(frames[3, 1], torch.zeros(8))


def test_predict_latent_transformer_consumes_real_frame_history():
    """With a real (non-degenerate) frame_history, the transformer attends over
    [context, frame_0, frame_1] -- frame_history's LAST position stands in for "now" (e_t),
    exactly like z_current does in the degenerate case; it is not separately re-appended."""
    model = _build_model("transformer")  # predictor_history_length=3 -> max_positions=4
    model.eval()
    calls = {}
    original_forward = model.predictor.forward

    def spy_forward(tokens, cond, valid):
        calls["tokens_shape"] = tuple(tokens.shape)
        calls["valid"] = valid.clone()
        return original_forward(tokens, cond, valid)

    model.predictor.forward = spy_forward
    z_current = torch.randn(1, 16)
    z_action = torch.randn(1, 16)
    z_context = torch.randn(1, 16)
    frame_history = (torch.randn(1, 2, 16), torch.randn(1, 2, 16), torch.ones(1, 2, dtype=torch.bool))
    with torch.no_grad():
        model.predict_latent(z_current, z_action, z_context, frame_history=frame_history)
    assert calls["tokens_shape"] == (1, 3, 16)  # [context, frame_0, frame_1==e_t]
    assert bool(calls["valid"].all())


def test_predict_latent_transformer_window_trims_oldest_frame_history():
    """max_positions=4 -> window=3 frame slots; a longer history must be trimmed to the newest
    ones rather than erroring or silently exceeding the predictor's learned position range."""
    model = _build_model("transformer")
    model.eval()
    calls = {}
    original_forward = model.predictor.forward

    def spy_forward(tokens, cond, valid):
        calls["tokens_shape"] = tuple(tokens.shape)
        return original_forward(tokens, cond, valid)

    model.predictor.forward = spy_forward
    z_current = torch.randn(1, 16)
    z_action = torch.randn(1, 16)
    z_context = torch.randn(1, 16)
    # 5 historical frames -- more than the window of 3 -- must be trimmed, not overflow max_positions.
    frame_history = (torch.randn(1, 5, 16), torch.randn(1, 5, 16), torch.ones(1, 5, dtype=torch.bool))
    with torch.no_grad():
        model.predict_latent(z_current, z_action, z_context, frame_history=frame_history)
    assert calls["tokens_shape"] == (1, 4, 16)  # [context] + newest 3 frames, within max_positions


def _build_model_with_heads(predictor_arch: str, canonical_event_head_inputs: str = "all") -> TextLeWorldModel:
    return _build_model(
        predictor_arch,
        canonical_event_vocab_sizes={"execution_status": 3},
        canonical_event_head_inputs=canonical_event_head_inputs,
    )


def test_canonical_event_head_inputs_all_sizes_trunk_by_four_latents():
    model = _build_model_with_heads("mlp", "all")
    assert model.canonical_event_trunk[0].in_features == 16 * 4


def test_canonical_event_head_inputs_pred_only_sizes_trunk_by_one_latent():
    model = _build_model_with_heads("mlp", "pred_only")
    assert model.canonical_event_trunk[0].in_features == 16


def test_canonical_event_head_inputs_ctx_pred_sizes_trunk_by_two_latents():
    model = _build_model_with_heads("mlp", "ctx_pred")
    assert model.canonical_event_trunk[0].in_features == 16 * 2


def test_canonical_event_head_inputs_state_sizes_trunk_by_predictor_dim():
    """The state readout is sized by the transformer predictor's own width (predictor.dim),
    not latent_dim -- this is exactly what breaks with a size mismatch if a caller builds the
    trunk with latent_dim*4 for a checkpoint trained with canonical_event_head_inputs='state'."""
    model = _build_model_with_heads("transformer", "state")
    assert model.canonical_event_trunk[0].in_features == model.predictor.dim


def test_canonical_event_head_inputs_state_requires_transformer_predictor():
    with pytest.raises(ValueError, match="requires predictor_arch='transformer'"):
        _build_model_with_heads("mlp", "state")


def test_canonical_event_head_inputs_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown canonical_event_head_inputs"):
        _build_model_with_heads("mlp", "bogus")


def test_predict_canonical_event_logits_state_mode_ignores_z_pred():
    """mode="state" reads z_state only -- changing z_pred (with z_state held fixed) must not
    move the logits at all, proving the branch doesn't silently fall back to the z_pred path."""
    model = _build_model_with_heads("transformer", "state")
    model.eval()
    z_current = torch.randn(1, 16)
    z_action = torch.randn(1, 16)
    z_context = torch.randn(1, 16)
    z_state = torch.randn(1, model.predictor.dim)
    z_pred_a = torch.randn(1, 16)
    z_pred_b = torch.randn(1, 16)
    with torch.no_grad():
        logits_a = model.predict_canonical_event_logits(z_current, z_action, z_context, z_pred_a, z_state)
        logits_b = model.predict_canonical_event_logits(z_current, z_action, z_context, z_pred_b, z_state)
    assert torch.allclose(logits_a["execution_status"], logits_b["execution_status"])


def test_predict_canonical_event_logits_state_mode_requires_z_state():
    model = _build_model_with_heads("transformer", "state")
    model.eval()
    z = torch.randn(1, 16)
    with pytest.raises(ValueError, match="needs the predictor's hidden state"):
        model.predict_canonical_event_logits(z, z, z, z, None)


def test_canonical_event_head_inputs_state_action_sizes_trunk_by_predictor_dim_plus_latent_dim():
    """state_action = [h_t, z_action] -- the two widths ADD (predictor.dim + latent_dim), unlike
    "all"'s multiplicative latent_dim*4."""
    model = _build_model_with_heads("transformer", "state_action")
    assert model.canonical_event_trunk[0].in_features == model.predictor.dim + model.latent_dim


def test_canonical_event_head_inputs_state_action_requires_transformer_predictor():
    with pytest.raises(ValueError, match="requires predictor_arch='transformer'"):
        _build_model_with_heads("mlp", "state_action")


def test_predict_canonical_event_logits_state_action_mode_uses_z_action_not_z_pred_or_z_current():
    """state_action reads [z_state, z_action] -- changing z_action must move the logits (unlike
    plain "state", which ignores it entirely), but z_pred/z_current must still be ignored."""
    model = _build_model_with_heads("transformer", "state_action")
    model.eval()
    z_state = torch.randn(1, model.predictor.dim)
    z_action_a = torch.randn(1, 16)
    z_action_b = torch.randn(1, 16)
    z_current_a = torch.randn(1, 16)
    z_current_b = torch.randn(1, 16)
    z_pred_a = torch.randn(1, 16)
    z_pred_b = torch.randn(1, 16)
    with torch.no_grad():
        base = model.predict_canonical_event_logits(z_current_a, z_action_a, z_current_a, z_pred_a, z_state)
        same_action = model.predict_canonical_event_logits(z_current_b, z_action_a, z_current_b, z_pred_b, z_state)
        diff_action = model.predict_canonical_event_logits(z_current_a, z_action_b, z_current_a, z_pred_a, z_state)
    # same z_action, different z_current/z_pred -> identical logits (those are still ignored)
    assert torch.allclose(base["execution_status"], same_action["execution_status"])
    # different z_action, same z_state -> logits must move
    assert not torch.allclose(base["execution_status"], diff_action["execution_status"])


def test_predict_canonical_event_logits_state_action_mode_requires_z_state():
    model = _build_model_with_heads("transformer", "state_action")
    model.eval()
    z = torch.randn(1, 16)
    with pytest.raises(ValueError, match="needs the predictor's hidden state"):
        model.predict_canonical_event_logits(z, z, z, z, None)


def test_predict_canonical_event_logits_state_action_gradient_reaches_action_input_and_predictor():
    """Confirms both halves of [h_t, z_action] are live in the backward graph: a loss on the
    logits must produce a gradient on the z_action INPUT itself (proving the concat branch
    doesn't silently detach it) and on the predictor's parameters (which produced h_t)."""
    model = _build_model_with_heads("transformer", "state_action")
    model.train()
    z_current = torch.randn(1, 16)
    z_action = torch.randn(1, 16, requires_grad=True)
    z_context = torch.randn(1, 16)
    _, _, z_state = model.predict_latent_with_state(z_current, z_action, z_context)
    logits = model.predict_canonical_event_logits(z_current, z_action, z_context, None, z_state)
    loss = logits["execution_status"].sum()
    loss.backward()
    predictor_grad = next(model.predictor.parameters()).grad
    assert z_action.grad is not None and torch.any(z_action.grad != 0)
    assert predictor_grad is not None and torch.any(predictor_grad != 0)
