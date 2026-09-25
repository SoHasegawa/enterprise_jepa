#!/usr/bin/env python3
"""Is the cumulative-state target actually insensitive to the new event -- and if so, where?

Text overlap between the current-state and next-state strings does NOT imply their latents are
close: with last-token pooling a causal backbone attends over everything, so appending
"permission denied" instead of "success" can move the pooled vector a lot. This script measures
the latents rather than inferring from the text.

Stage 0 -- TRUNCATION AUDIT (run first; everything else is uninterpretable if this fails).
  With right-side truncation, the newly appended action+observation live at the END of the
  target string and are the first thing dropped when the text exceeds max_input_length. That
  alone would make z_{t+1} ~ z_t by construction, with no representational explanation needed.
  Reports truncation_side, the length distribution, the fraction of targets that overflow, and
  -- decisively -- whether the observation's own tokens survive in the truncated target.

Stage 1 -- DISTANCES on eval data:
    D_ct = E[1 - cos(z_t, z_{t+1})]        current vs target
    D_pt = E[1 - cos(z_hat, z_{t+1})]      predictor vs target
    D_pc = E[1 - cos(z_hat, z_t)]          predictor vs current
    PNI  = 1 - MSE(z_hat, z_{t+1}) / MSE(z_t, z_{t+1})
    persistence proximity rate = Pr[d(z_hat, z_t) < d(z_hat, z_{t+1})]

Stage 2 -- BACKBONE vs PROJECTOR. The same distances before and after the projector:
    1 - cos(b_t, b_{t+1})   pooled backbone output
    1 - cos(z_t, z_{t+1})   projected latent
  Large at the backbone but small after projection => the projector (or the Phase-1 objective
  that shaped it) is discarding the event. Already small at the backbone => cumulative-text
  embedding itself dilutes it, and no projector change will fix it.

Stage 3 -- SUFFIX SENSITIVITY, controlled. Same prefix with deliberately contrasting suffixes,
  to see what the encoder actually responds to:
    same prefix, same action, different OBSERVATION  (success vs permission denied)
    same prefix, different ACTION, same observation
    different PREFIX, same action+observation
  If the observation contrast moves the latent far less than the prefix contrast, the encoder is
  dominated by context/history regardless of pooling.

Usage:
  uv run python src/analysis/week1_persistence_diagnostics.py --device cuda:0
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

DEFAULT_CHECKPOINT = Path("checkpoints/data_jepa_all_adp_06b_decoder_msp")
DEFAULT_EVAL = REPO_ROOT / "trajectories" / "enterpriseops_gym_world_model_test_trajectories.json"
DEFAULT_OUT = REPO_ROOT / "data" / "week1" / "persistence_diagnostics.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-examples", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def cos_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=-1)


@torch.no_grad()
def encode_both(model, tokenizer, texts: list[str], max_len: int, device, batch_size: int):
    """Return (pooled backbone output, projected latent) for each text."""
    pooled_all, latent_all = [], []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_len, add_special_tokens=True)
        pooled = model._encode_pooled(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        latent, _ = model._project_latent(pooled)
        pooled_all.append(pooled.float().cpu())
        latent_all.append(latent.float().cpu())
    return torch.cat(pooled_all), torch.cat(latent_all)


def truncation_audit(tokenizer, examples, max_input_length: int) -> dict[str, Any]:
    lengths_cur, lengths_next, obs_survives, overflow = [], [], [], 0
    for ex in examples:
        ids_cur = tokenizer(ex.current_state_text, add_special_tokens=True)["input_ids"]
        ids_next_full = tokenizer(ex.next_state_text, add_special_tokens=True)["input_ids"]
        lengths_cur.append(len(ids_cur))
        lengths_next.append(len(ids_next_full))
        if len(ids_next_full) > max_input_length:
            overflow += 1
        # Does the observation actually survive truncation? Compare the observation's own token
        # ids against the tail of the TRUNCATED target.
        kept = tokenizer(ex.next_state_text, truncation=True, max_length=max_input_length,
                         add_special_tokens=True)["input_ids"]
        obs_ids = tokenizer(ex.observation_text, add_special_tokens=False)["input_ids"][:24]
        obs_survives.append(bool(obs_ids) and _contains(kept, obs_ids[: min(8, len(obs_ids))]))
    lengths_next.sort()
    n = len(lengths_next)
    return {
        "truncation_side": getattr(tokenizer, "truncation_side", "?"),
        "max_input_length": max_input_length,
        "n": n,
        "target_tokens_median": lengths_next[n // 2] if n else 0,
        "target_tokens_p90": lengths_next[int(n * 0.9)] if n else 0,
        "target_tokens_max": lengths_next[-1] if n else 0,
        "fraction_target_overflows": overflow / n if n else 0.0,
        "fraction_observation_survives_truncation": sum(obs_survives) / n if n else 0.0,
    }


def _contains(haystack: list[int], needle: list[int]) -> bool:
    if not needle:
        return False
    first = needle[0]
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i] == first and haystack[i : i + len(needle)] == needle:
            return True
    return False


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer

    manifest = json.loads((args.checkpoint / "jepa_data_manifest.json").read_text())
    max_input = int(manifest.get("max_input_length") or 2048)
    max_action = int(manifest.get("max_action_length") or 512)
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))
    backbone = fj.load_jepa_backbone(args.checkpoint / "backbone", backbone_type=manifest["backbone_type"],
                                     trust_remote_code=True, dtype=torch.bfloat16)
    model = fj.TextLeWorldModel(
        backbone=backbone, latent_dim=int(manifest.get("latent_dim") or 0),
        memory_tokens=int(manifest.get("memory_tokens") or 8), dropout=0.0,
        predictor_hidden_multiplier=float(manifest.get("predictor_hidden_multiplier") or 4.0),
        goal_conditioning=bool(manifest.get("goal_conditioning", False)),
        latent_type=str(manifest.get("latent_type") or "continuous"),
        pooling=str(manifest.get("pooling") or "mean"),
    )
    fj.load_jepa_state_dict_for_training(model, args.checkpoint, allow_missing_success_head=True,
                                         allow_missing_canonical_event_heads=True)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device).eval()
    print(f"checkpoint={args.checkpoint.name}  pooling={manifest.get('pooling')}  "
          f"max_input_length={max_input}  latent_dim={manifest.get('latent_dim')}")

    extract_args = argparse.Namespace(
        state_history_size=8, max_observation_length=512, max_action_length=max_action,
        max_input_length=max_input, action_contrastive_negatives=0, action_contrastive_loss_coeff=0.0,
        obs_token_ground_coeff=0.0, prediction_horizon=1, truncate_states_keep_newest=False,
        obs_ground_max_tokens=None,
    )
    records = json.loads(args.eval_data.read_text())
    examples = fj.extract_jepa_examples(records, extract_args, benchmark="gym")
    examples = [e for e in examples if e.next_state_text and e.observation_text][: args.max_examples]
    print(f"examples: {len(examples)}")

    report: dict[str, Any] = {"checkpoint": str(args.checkpoint), "pooling": manifest.get("pooling"),
                              "n": len(examples)}

    # ---------------- Stage 0: truncation ----------------
    audit = truncation_audit(tokenizer, examples, max_input)
    report["truncation"] = audit
    print("\n=== Stage 0: truncation audit ===")
    for k, v in audit.items():
        print(f"  {k:42s} {v}")
    if audit["fraction_observation_survives_truncation"] < 0.99:
        print("  !! the new observation is being truncated out of some targets -- that alone")
        print("     makes z_next ~ z_current by construction.")

    # ---------------- Stages 1-2: distances, backbone vs projector ----------------
    b_cur, z_cur = encode_both(model, tokenizer, [e.current_state_text for e in examples], max_input, device, args.batch_size)
    b_next, z_next = encode_both(model, tokenizer, [e.next_state_text for e in examples], max_input, device, args.batch_size)
    _, z_ctx = encode_both(model, tokenizer, [e.context_text for e in examples], max_input, device, args.batch_size)
    _, z_act = encode_both(model, tokenizer, [e.action_text for e in examples], max_action, device, args.batch_size)
    b_obs, z_obs = encode_both(model, tokenizer, [e.observation_text for e in examples], max_input, device, args.batch_size)
    with torch.no_grad():
        z_pred, _ = model.predict_latent(z_cur.to(device), z_act.to(device), z_ctx.to(device))
    z_pred = z_pred.float().cpu()

    d_ct, d_pt, d_pc = cos_dist(z_cur, z_next), cos_dist(z_pred, z_next), cos_dist(z_pred, z_cur)
    mse_pred = (z_pred - z_next).pow(2).mean().item()
    mse_copy = (z_cur - z_next).pow(2).mean().item()
    pni = 1.0 - mse_pred / max(mse_copy, 1e-12)
    proximity = (cos_dist(z_pred, z_cur) < cos_dist(z_pred, z_next)).float().mean().item()
    report["distances"] = {
        "D_ct_current_vs_target": d_ct.mean().item(),
        "D_pt_predictor_vs_target": d_pt.mean().item(),
        "D_pc_predictor_vs_current": d_pc.mean().item(),
        "MSE_predictor_vs_target": mse_pred, "MSE_current_vs_target": mse_copy, "PNI": pni,
        "persistence_proximity_rate": proximity,
    }
    print("\n=== Stage 1: distances (cosine unless noted) ===")
    print(f"  D_ct  current   vs target    = {d_ct.mean():.5f}")
    print(f"  D_pt  predictor vs target    = {d_pt.mean():.5f}")
    print(f"  D_pc  predictor vs current   = {d_pc.mean():.5f}")
    print(f"  MSE(pred,target)={mse_pred:.5f}   MSE(current,target)={mse_copy:.5f}   PNI={pni:+.4f}")
    print(f"  persistence proximity rate Pr[d(pred,cur) < d(pred,target)] = {proximity:.4f}")

    bb = cos_dist(b_cur, b_next).mean().item()
    pj = d_ct.mean().item()
    report["backbone_vs_projector"] = {
        "backbone_1_minus_cos_cur_next": bb, "projected_1_minus_cos_cur_next": pj,
        "ratio_projected_over_backbone": pj / bb if bb else float("nan"),
        "backbone_1_minus_cos_next_obs": cos_dist(b_next, b_obs).mean().item(),
    }
    print("\n=== Stage 2: backbone vs projector (current -> target separation) ===")
    print(f"  backbone  1-cos(b_t, b_t+1) = {bb:.5f}")
    print(f"  projected 1-cos(z_t, z_t+1) = {pj:.5f}   (ratio {pj / bb if bb else float('nan'):.3f})")
    print("  ratio << 1 => the projector shrinks the event; ~1 => it is already gone at the backbone")

    # ---------------- Stage 3: controlled suffix sensitivity ----------------
    base = examples[0]
    prefix, action = base.current_state_text, base.action_text
    other = next((e for e in examples if e.context_text != base.context_text), examples[-1])

    def target_of(pfx: str, act: str, obs: str) -> str:
        return pfx + "\n\nHistory after current action:\n" + fj.render_history(
            [{"step": 1, "action": act, "observation": obs}])

    OK = "The operation completed successfully."
    NO = "Permission denied. The operation was not executed."
    variants = {
        "same_prefix_same_action_obs_success": target_of(prefix, action, OK),
        "same_prefix_same_action_obs_denied": target_of(prefix, action, NO),
        "same_prefix_diff_action_obs_success": target_of(prefix, other.action_text, OK),
        "diff_prefix_same_action_obs_success": target_of(other.current_state_text, action, OK),
    }
    keys = list(variants)
    _, zv = encode_both(model, tokenizer, [variants[k] for k in keys], max_input, device, 4)
    idx = {k: i for i, k in enumerate(keys)}
    contrasts = {
        "observation success vs denied (same prefix, same action)":
            cos_dist(zv[idx["same_prefix_same_action_obs_success"]], zv[idx["same_prefix_same_action_obs_denied"]]).item(),
        "action differs (same prefix, same observation)":
            cos_dist(zv[idx["same_prefix_same_action_obs_success"]], zv[idx["same_prefix_diff_action_obs_success"]]).item(),
        "prefix differs (same action, same observation)":
            cos_dist(zv[idx["same_prefix_same_action_obs_success"]], zv[idx["diff_prefix_same_action_obs_success"]]).item(),
    }
    report["suffix_sensitivity"] = contrasts
    print("\n=== Stage 3: what does the encoder respond to? (1 - cos) ===")
    for k, v in contrasts.items():
        print(f"  {k:56s} {v:.5f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
