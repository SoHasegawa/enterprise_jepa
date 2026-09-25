#!/usr/bin/env python3
"""Week 1.4 -- is the predictor action-conditioned, or has it learned an action-ignoring shortcut?

Hypothesis under test: most transitions are ordinary successes, so the latent MSE can be
minimized by predicting a persistence/majority-like next latent; the heads then ride the
majority class and the scorer assigns optimistic states to wrong actions.

Predictor conditions (state metrics for each):
  normal            z_pred = P(z_cur, z_act, z_ctx)              current model
  zero_action       z_act := 0                                    is the action needed at all?
  shuffled_action   z_act permuted across rows                    does state-action correspondence matter?
  persistence       z_pred := z_cur                               does JEPA beat copying the state?
  context_only      z_cur := 0, z_act := 0                        does context alone suffice?
  state_only        z_act := 0 (alias of zero_action, kept for the table's symmetry)

Action-conditioning strength, on an M x K grid of (state, action) pairs:
  R_action = E_s[Var_a[zhat(s,a)]] / E_a[Var_s[zhat(s,a)]]
  plus the mean pairwise cosine DISTANCE between predicted latents for different actions at
  the same state (the interpretable version of the same quantity).

Loads only the projector+predictor from the checkpoint (no backbone) and reuses the cached
latents from week1_ceiling_ladder.py --stage encode, so this runs in seconds on CPU.

Usage:
  uv run python src/analysis/week1_action_sensitivity.py \
      --workdir checkpoints/week1_ceiling_ladder \
      --checkpoint checkpoints/data_jepa_all_adp_06b_decoder_msp
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
from torch import nn

import src.finetuning_jepa as fj
from src.analysis.week1_ceiling_ladder import (
    MULTI_FIELDS, OUTCOME_FIELDS, SINGLE_FIELDS, Heads, score_multi, score_single, summarize, train_and_eval,
)

DEFAULT_WORKDIR = Path("checkpoints/week1_ceiling_ladder")
DEFAULT_CHECKPOINT = Path("checkpoints/data_jepa_all_adp_06b_decoder_msp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid-states", type=int, default=256, help="M in the R_action grid")
    p.add_argument("--grid-actions", type=int, default=32, help="K in the R_action grid")
    return p.parse_args()


def load_predictor(checkpoint: Path, device: torch.device) -> nn.Module:
    """Rebuild ONLY the predictor MLP from the checkpoint state dict -- no backbone load."""
    manifest = json.loads((checkpoint / "jepa_data_manifest.json").read_text())
    state = torch.load(checkpoint / "text_leworldmodel.pt", map_location="cpu", weights_only=True)
    weights = {k[len("predictor."):]: v for k, v in state.items() if k.startswith("predictor.")}
    if not weights:
        raise SystemExit(f"{checkpoint} has no predictor.* weights")
    latent_dim = weights["0.weight"].shape[1] // (4 if manifest.get("goal_conditioning") else 3)
    hidden = weights["0.weight"].shape[0]
    layers: list[nn.Module] = [
        nn.Linear(weights["0.weight"].shape[1], hidden), nn.GELU(), nn.Dropout(0.0),
        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.0),
        nn.Linear(hidden, latent_dim),
    ]
    if any(k.startswith("7.") for k in weights):   # trailing LayerNorm (absolute-mode predictor)
        layers.append(nn.LayerNorm(latent_dim))
    predictor = nn.Sequential(*layers)
    predictor.load_state_dict(weights)
    predictor.eval().to(device)
    for param in predictor.parameters():
        param.requires_grad = False
    return predictor


@torch.no_grad()
def predict(predictor, z_cur, z_act, z_ctx, chunk: int = 4096) -> torch.Tensor:
    """Cached latents live on CPU; move each chunk to the predictor's device and bring the
    result back, so the caller keeps working with CPU tensors regardless of --device."""
    device = next(predictor.parameters()).device
    out = []
    for start in range(0, z_cur.shape[0], chunk):
        sl = slice(start, start + chunk)
        batch = torch.cat([z_cur[sl], z_act[sl], z_ctx[sl]], dim=-1).to(device)
        out.append(predictor(batch).cpu())
    return torch.cat(out)


def action_conditioning_metrics(predictor, z_cur, z_act, z_ctx, m: int, k: int, seed: int) -> dict[str, float]:
    """R_action and cosine spread on an M x K (state, action) grid."""
    g = torch.Generator().manual_seed(seed)
    states = torch.randperm(z_cur.shape[0], generator=g)[:m]
    actions = torch.randperm(z_act.shape[0], generator=g)[:k]
    grid = torch.empty(m, k, z_cur.shape[-1])
    for i, s in enumerate(states):
        cur = z_cur[s].unsqueeze(0).expand(k, -1)
        ctx = z_ctx[s].unsqueeze(0).expand(k, -1)
        grid[i] = predict(predictor, cur, z_act[actions], ctx)
    var_over_actions = grid.var(dim=1, unbiased=False).mean()       # E_s Var_a
    var_over_states = grid.var(dim=0, unbiased=False).mean()        # E_a Var_s
    normalized = torch.nn.functional.normalize(grid, dim=-1)
    cos = torch.einsum("mkd,mld->mkl", normalized, normalized)
    off_diagonal = ~torch.eye(k, dtype=torch.bool).unsqueeze(0).expand(m, -1, -1)
    return {
        "R_action": float(var_over_actions / var_over_states.clamp_min(1e-12)),
        "E_s_Var_a": float(var_over_actions),
        "E_a_Var_s": float(var_over_states),
        "mean_pairwise_cosine_distance_same_state": float(1.0 - cos[off_diagonal].mean()),
        "grid": [m, k],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    vocab = json.loads((args.workdir / "vocab.json").read_text())
    tr = torch.load(args.workdir / "ladder_train.pt", weights_only=False)
    ev = torch.load(args.workdir / "ladder_eval.pt", weights_only=False)
    ltr, lev = tr["labels"], ev["labels"]
    predictor = load_predictor(args.checkpoint, device)

    g = torch.Generator().manual_seed(args.seed)
    conditions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split, d in (("tr", tr), ("ev", ev)):
        z_cur, z_act, z_ctx = d["z"]["cur"], d["z"]["act"], d["z"]["ctx"]
        zero = torch.zeros_like(z_act)
        perm = torch.randperm(z_act.shape[0], generator=g)
        conditions.setdefault("normal", {})[split] = predict(predictor, z_cur, z_act, z_ctx)
        conditions.setdefault("zero_action", {})[split] = predict(predictor, z_cur, zero, z_ctx)
        conditions.setdefault("shuffled_action", {})[split] = predict(predictor, z_cur, z_act[perm], z_ctx)
        conditions.setdefault("persistence", {})[split] = z_cur.clone()
        conditions.setdefault("context_only", {})[split] = predict(
            predictor, torch.zeros_like(z_cur), zero, z_ctx)

    print("=== action-conditioning strength (predictor only) ===")
    metrics = action_conditioning_metrics(predictor, ev["z"]["cur"], ev["z"]["act"], ev["z"]["ctx"],
                                          args.grid_states, args.grid_actions, args.seed)
    print(f"  R_action = E_s Var_a / E_a Var_s = {metrics['R_action']:.4f}"
          f"   (E_s Var_a={metrics['E_s_Var_a']:.5f}, E_a Var_s={metrics['E_a_Var_s']:.5f})")
    print(f"  mean pairwise cosine distance between z_pred for different actions at the SAME state: "
          f"{metrics['mean_pairwise_cosine_distance_same_state']:.5f}")
    print("  (R_action << 1 or ~0 cosine distance => state identity dominates; the predictor ignores the action)\n")

    # how far does each ablated prediction move from the normal one?
    drift = {name: float((tensors["ev"] - conditions["normal"]["ev"]).norm(dim=-1).mean())
             for name, tensors in conditions.items()}
    baseline_norm = float(conditions["normal"]["ev"].norm(dim=-1).mean())
    print("=== L2 drift of z_pred vs the normal predictor (eval) ===")
    for name, value in drift.items():
        print(f"  {name:18s} {value:9.4f}   ({value / max(baseline_norm, 1e-9):6.2%} of ||z_pred||)")
    print()

    print("=== state metrics: heads trained on [z_cur, z_act, z_ctx, z_pred(condition)] ===")
    results: dict[str, Any] = {"action_conditioning": metrics, "z_pred_drift_vs_normal": drift, "conditions": {}}
    for name, tensors in conditions.items():
        x_tr = torch.cat([tr["z"]["cur"], tr["z"]["act"], tr["z"]["ctx"], tensors["tr"]], dim=-1)
        x_ev = torch.cat([ev["z"]["cur"], ev["z"]["act"], ev["z"]["ctx"], tensors["ev"]], dim=-1)
        per_field, _ = train_and_eval(lambda d=x_tr.shape[-1]: Heads(d, vocab),
                                      (x_tr,), (x_ev,), ltr, lev, vocab, args, device)
        results["conditions"][name] = {"per_field": per_field, **summarize(per_field)}
        s = results["conditions"][name]
        print(f"  [{name:18s}] outcome4_acc={s['outcome4_accuracy']:.4f} "
              f"outcome4_F1={s['outcome4_macro_f1']:.4f} macroF1={s['macro_f1']:.4f}")

    out = args.workdir / "action_sensitivity_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nresults -> {out}")
    print("\nReading: if normal ~= zero_action ~= shuffled_action, the predictor learned an\n"
          "action-ignoring shortcut and further decoder/multi-step losses will not fix ranking.")


if __name__ == "__main__":
    main()
