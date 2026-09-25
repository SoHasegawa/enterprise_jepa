#!/usr/bin/env python3
"""Is the trained AdaLN transformer predictor action-conditioned, or action-blind?

week1_action_sensitivity.py answers this for the concat-MLP predictor by rebuilding it from
`predictor.0.weight` and feeding it concatenated latents. That cannot read a
--predictor-arch transformer checkpoint at all (no such key), and more importantly the question
is different: the action never enters as an input vector, only as AdaLN modulation. If the
modulation weights stayed near their zero init, or the blocks learned to ignore the conditioning
signal, the predictor is action-blind and every candidate action yields the same predicted event.

What is measured, on real eval batches:

  Perturbation deltas -- how far the prediction (and the belief state, and the head logits)
  move when only the action changes:
    zero_action      the candidate action latent is zeroed
    shuffled_action  candidate actions permuted across the batch (same states, wrong actions)
    history_dropped  the frame history removed, keeping context + candidate action

  R_action -- the quantity week1_action_sensitivity reports, on an M x K (state, action) grid:
    R_action = E_s[Var_a[zhat(s,a)]] / E_a[Var_s[zhat(s,a)]]
  >> 1 means the prediction is driven more by which action is taken than by which state it is
  taken from; ~0 means action-blind.

  Head-level sensitivity -- the fraction of rows whose argmax canonical-event label FLIPS when
  the action is shuffled. This is the decision-relevant version: a predictor can move in latent
  space yet still produce identical downstream labels.

  AdaLN gate magnitude per block -- did the zero-initialized modulation actually open?

Usage:
  uv run python src/analysis/transformer_predictor_action_sensitivity.py \
      --checkpoint checkpoints/data_jepa_heads_ensemble \
      --jsonl trajectories/..._eval_examples_cleaned_ensemble_value_scored.jsonl \
      --limit 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import src.finetuning_jepa as fj


def build_model(checkpoint: Path, device: torch.device, dtype: torch.dtype) -> tuple[Any, Any, dict]:
    manifest = json.loads((checkpoint / "jepa_data_manifest.json").read_text())
    if manifest.get("predictor_arch") != "transformer":
        raise SystemExit(
            f"{checkpoint} manifest says predictor_arch="
            f"{manifest.get('predictor_arch')!r}; use week1_action_sensitivity.py for the MLP predictor."
        )
    backbone_dir = checkpoint / "backbone"
    source = str(backbone_dir if backbone_dir.is_dir() else manifest.get("base_model") or "Qwen/Qwen3-Embedding-0.6B")
    # Tokenizer files are saved at the checkpoint ROOT; only the backbone weights live in backbone/.
    tokenizer_source = checkpoint if (checkpoint / "tokenizer.json").is_file() else source
    tokenizer = fj.load_text_tokenizer(tokenizer_source, trust_remote_code=True)
    backbone = fj.load_jepa_backbone(source, backbone_type=manifest.get("backbone_type", "encoder"),
                                     dtype=dtype, trust_remote_code=True)
    vocab_sizes = manifest.get("canonical_event_vocab_sizes") or {}
    model = fj.TextLeWorldModel(
        backbone=backbone,
        latent_dim=int(manifest.get("latent_dim") or 0),
        memory_tokens=int(manifest.get("memory_tokens") or 8),
        dropout=0.0,
        predictor_hidden_multiplier=float(manifest.get("predictor_hidden_multiplier") or 4.0),
        goal_conditioning=bool(manifest.get("goal_conditioning", False)),
        latent_type=str(manifest.get("latent_type") or "continuous"),
        pooling=str(manifest.get("pooling") or "mean"),
        canonical_event_vocab_sizes=vocab_sizes,
        canonical_event_head_hidden_size=int(manifest.get("canonical_event_head_hidden_size") or 512),
        canonical_event_head_inputs=str(manifest.get("canonical_event_head_inputs") or "all"),
        predictor_arch="transformer",
        predictor_transformer_dim=int(manifest.get("predictor_transformer_dim") or 0),
        predictor_transformer_layers=int(manifest.get("predictor_transformer_layers") or 6),
        predictor_transformer_heads=int(manifest.get("predictor_transformer_heads") or 16),
        predictor_transformer_mlp_ratio=float(manifest.get("predictor_transformer_mlp_ratio") or 4.0),
        predictor_history_length=int(manifest.get("predictor_history_length") or 0),
    )
    state = torch.load(checkpoint / "text_leworldmodel.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("backbone.")]
    unexpected = [k for k in unexpected if not k.startswith("backbone.")]
    if missing or unexpected:
        print(f"[load] missing={missing[:6]} unexpected={unexpected[:6]}", flush=True)
    model.eval().to(device=device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer, manifest


def build_batch(jsonl: Path, tokenizer: Any, manifest: dict, limit: int) -> dict[str, torch.Tensor]:
    rows = []
    with jsonl.open() as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    examples = [e for e in (fj.build_canonical_event_example(r) for r in rows) if e is not None]
    vocab = fj.build_canonical_event_vocabularies(examples)
    args = argparse.Namespace(
        max_input_length=int(manifest.get("max_input_length") or 8192),
        max_action_length=int(manifest.get("max_action_length") or 512),
        max_observation_length=int(manifest.get("max_observation_length") or 512),
        truncate_states_keep_newest=bool(manifest.get("truncate_states_keep_newest", False)),
        canonical_event_recognition_probe=False,
        joint_canonical_event_training=False,
        predictor_arch="transformer",
        recurrent_state_init=False,
    )
    dataset = fj.CanonicalEventDataset(examples, tokenizer, args, vocab)
    collator = fj.CanonicalEventCollator(tokenizer)
    return collator([dataset[i] for i in range(len(examples))]), examples


@torch.no_grad()
def _encode_chunked(model, ids, mask, device, chunk: int):
    """State texts run to max_input_length (8192), so the whole split will not fit on one GPU
    in a single forward -- encode in slices and keep only the pooled latents."""
    out = []
    for start in range(0, ids.shape[0], chunk):
        sl = slice(start, start + chunk)
        z, _ = model.encode_latent_and_logits(ids[sl].to(device), mask[sl].to(device))
        out.append(z)
    return torch.cat(out)


@torch.no_grad()
def encode(model, batch, device, chunk: int = 8):
    """Encode once; the perturbations reuse these latents so nothing is re-tokenized."""
    z_current = _encode_chunked(model, batch["current_input_ids"], batch["current_attention_mask"], device, chunk)
    z_context = _encode_chunked(model, batch["context_input_ids"], batch["context_attention_mask"], device, chunk)
    z_action = _encode_chunked(model, batch["action_input_ids"], batch["action_attention_mask"], device, chunk)
    history = None
    if "history_event_input_ids" in batch:
        # The two streams are padded to their OWN widths (max_observation_length vs
        # max_action_length), so each must be flattened with its own sequence length.
        def flat_encode(key: str) -> torch.Tensor:
            ids = batch[key]
            b, depth, length = ids.shape
            return _encode_chunked(
                model, ids.reshape(b * depth, length),
                batch[key.replace("input_ids", "attention_mask")].reshape(b * depth, length),
                device, chunk,
            ).reshape(b, depth, -1)

        z_o = flat_encode("history_event_input_ids")
        z_a = flat_encode("history_action_input_ids")
        history = model.encode_frame_history(
            {**{k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}}, (z_a, z_o))
    return z_current, z_context, z_action, history


@torch.no_grad()
def run(model, z_current, z_context, z_action, history):
    event, state = model._predict_latent_transformer(z_current, z_action, z_context, None, history)
    logits = None
    if getattr(model, "canonical_event_heads", None) and len(model.canonical_event_heads):
        logits = model.predict_canonical_event_logits(z_current, z_action, z_context, event, state)
    return event, state, logits


def rel_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean L2 difference, normalized by the mean norm of the reference -- scale-free."""
    return float((a - b).norm(dim=-1).mean() / a.norm(dim=-1).mean().clamp_min(1e-9))


def label_flip_rate(base: dict, other: dict) -> dict[str, float]:
    return {
        field: float((base[field].argmax(-1) != other[field].argmax(-1)).float().mean())
        for field in base
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--encode-chunk", type=int, default=8)
    parser.add_argument("--grid-states", type=int, default=64)
    parser.add_argument("--grid-actions", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    model, tokenizer, manifest = build_model(args.checkpoint, device, dtype)
    batch, examples = build_batch(args.jsonl, tokenizer, manifest, args.limit)
    print(f"[setup] {len(examples)} rows | head_inputs={manifest.get('canonical_event_head_inputs')!r} "
          f"| predictor dim={model.predictor.dim} layers={len(model.predictor.blocks)} "
          f"| history slots={batch.get('history_event_input_ids', torch.zeros(1,0)).shape[1]}", flush=True)

    z_current, z_context, z_action, history = encode(model, batch, device, args.encode_chunk)
    base_event, base_state, base_logits = run(model, z_current, z_context, z_action, history)

    report: dict[str, Any] = {"rows": len(examples), "checkpoint": str(args.checkpoint)}

    print("\n=== perturbation deltas (relative L2; 0 = the change had NO effect) ===")
    print(f"{'condition':18s} {'event':>10s} {'state h_t':>10s}")
    perms = torch.randperm(z_action.shape[0], generator=torch.Generator().manual_seed(0)).to(device)
    conditions = {
        "zero_action": (torch.zeros_like(z_action), history),
        "shuffled_action": (z_action[perms], history),
        "history_dropped": (z_action, None),
    }
    report["perturbations"] = {}
    for name, (act, hist) in conditions.items():
        event, state, logits = run(model, z_current, z_context, act, hist)
        entry = {"event": rel_delta(base_event, event), "state": rel_delta(base_state, state)}
        if base_logits is not None:
            entry["label_flip_rate"] = label_flip_rate(base_logits, logits)
            entry["mean_label_flip_rate"] = sum(entry["label_flip_rate"].values()) / len(entry["label_flip_rate"])
        report["perturbations"][name] = entry
        print(f"{name:18s} {entry['event']:10.5f} {entry['state']:10.5f}")

    if base_logits is not None:
        print("\n=== head label-flip rate when the action is SHUFFLED (0 = heads ignore the action) ===")
        flips = report["perturbations"]["shuffled_action"]["label_flip_rate"]
        for field, rate in sorted(flips.items(), key=lambda kv: -kv[1]):
            print(f"  {field:32s} {rate:6.1%}")
        print(f"  {'MEAN':32s} {report['perturbations']['shuffled_action']['mean_label_flip_rate']:6.1%}")

    print("\n=== R_action on an M x K (state, action) grid ===")
    g = torch.Generator().manual_seed(0)
    m = min(args.grid_states, z_current.shape[0])
    k = min(args.grid_actions, z_action.shape[0])
    s_idx = torch.randperm(z_current.shape[0], generator=g)[:m].to(device)
    a_idx = torch.randperm(z_action.shape[0], generator=g)[:k].to(device)
    grid = torch.empty(m, k, base_event.shape[-1], device=device, dtype=base_event.dtype)
    for j in range(k):
        act = z_action[a_idx[j]].unsqueeze(0).expand(m, -1)
        hist = None
        if history is not None:
            hist = tuple(t[s_idx] for t in history)
        grid[:, j] = model._predict_latent_transformer(
            z_current[s_idx], act, z_context[s_idx], None, hist)[0]
    var_over_actions = grid.var(dim=1, unbiased=False).mean()
    var_over_states = grid.var(dim=0, unbiased=False).mean()
    r_action = float(var_over_actions / var_over_states.clamp_min(1e-12))
    cos = torch.nn.functional.normalize(grid, dim=-1)
    pair = 1.0 - torch.einsum("mkd,mjd->mkj", cos, cos)
    off = ~torch.eye(k, dtype=torch.bool, device=device)
    report["r_action"] = r_action
    report["mean_cosine_distance_between_actions"] = float(pair[:, off].mean())
    print(f"  grid {m} states x {k} actions")
    print(f"  E_s[Var_a] = {float(var_over_actions):.6f}   E_a[Var_s] = {float(var_over_states):.6f}")
    print(f"  R_action = {r_action:.4f}   (>>1: action-driven, ~0: action-blind)")
    print(f"  mean cosine distance between predictions for different actions at the same state: "
          f"{report['mean_cosine_distance_between_actions']:.5f}")

    print("\n=== AdaLN modulation: did the zero-init gates open? ===")
    report["adaln_gate_rms"] = []
    for i, block in enumerate(model.predictor.blocks):
        w = block.adaln_modulation[-1].weight
        b = block.adaln_modulation[-1].bias
        dim = model.predictor.dim
        gate_msa = b[2 * dim:3 * dim]
        gate_mlp = b[5 * dim:6 * dim]
        entry = {"block": i, "weight_rms": float(w.float().pow(2).mean().sqrt()),
                 "gate_msa_bias_rms": float(gate_msa.float().pow(2).mean().sqrt()),
                 "gate_mlp_bias_rms": float(gate_mlp.float().pow(2).mean().sqrt())}
        report["adaln_gate_rms"].append(entry)
        print(f"  block {i}: modulation W rms={entry['weight_rms']:.5f}  "
              f"gate_msa bias rms={entry['gate_msa_bias_rms']:.5f}  "
              f"gate_mlp bias rms={entry['gate_mlp_bias_rms']:.5f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))
        print(f"\nreport -> {args.output}")


if __name__ == "__main__":
    main()
