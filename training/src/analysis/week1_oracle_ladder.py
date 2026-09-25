#!/usr/bin/env python3
"""Week 1 -- the oracle ranking ladder on EXECUTED counterfactuals.

The revised plan's ladder is  gold labels -> true next latent -> predicted next latent ->
deployed score.  Rungs above "deployed" need an outcome for the counterfactual action, which
synthetic distractors never have; week1_executed_counterfactuals.py supplies real ones (two
different actions taken from the same state by two models, each with its real observation).

Rungs, all scored through the SAME deployed head+scorer stack so only the future-latent input
differs:

  no_future     heads see (z_cur, z_act, z_ctx, 0)        -- the skip-connection-only baseline:
                                                             what the scorer knows WITHOUT any
                                                             future at all
  persistence   heads see (..., z_cur)                    -- the action-blind solution MSE
                                                             training is minimized by
  predicted     heads see (..., F(z_cur,z_act,z_ctx))     -- the DEPLOYED path
  oracle_next   heads see (..., E(true next state))       -- a perfect predictor: the real
                                                             observation that action produced

Reading it:
  * oracle_next ~= predicted  => a better predictor cannot help; the ceiling is the
    scorer/ontology, not the transition model.
  * predicted ~= no_future    => the predicted future contributes nothing over the skip.
  * oracle_next >> predicted  => foresight IS useful and the predictor is what to fix. This is
    the submission gate for continuing the world-model story.

Metric is pairwise preference accuracy against the executed outcome, reported separately by
label strength (tool_success is decisive; verifier_rate attributes a whole-run outcome to one
step and is weaker), with a paired bootstrap CI and a chance line of 0.5.

Usage:
  uv run python src/analysis/week1_oracle_ladder.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import src.finetuning_jepa as fj
from src.canonical_event_scoring import CanonicalEventScoreConfig, logits_to_field_probs, score_step, step_veto

DEFAULT_PAIRS = REPO_ROOT / "data" / "week1" / "executed_counterfactuals.jsonl"
DEFAULT_CHECKPOINT = Path("checkpoints/jepa_msp_decoder_cls_head")
DEFAULT_OUT = REPO_ROOT / "data" / "week1" / "oracle_ladder_results.json"
RUNGS = ("no_future", "persistence", "predicted", "oracle_next")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-pairs", type=int, default=0, help="0 = all decisive pairs")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def as_tool_call(tool_name: str, arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"_raw": arguments}
    return {"tool_calls": [{"type": "function", "function": {"name": tool_name, "arguments": arguments}}]}


def bootstrap_ci(values: list[float], n: int, seed: int) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        means.append(sum(values[rng.randrange(len(values))] for _ in range(len(values))) / len(values))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


@torch.no_grad()
def encode(model, tokenizer, texts: list[str], max_len: int, device) -> torch.Tensor:
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_len, add_special_tokens=True)
    z, _ = model.encode_latent_and_logits(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return z


@torch.no_grad()
def score_with_future(model, vocab, cfg, z_cur, z_act, z_ctx, z_future) -> tuple[float, bool]:
    logits = model.predict_canonical_event_logits(z_cur, z_act, z_ctx, z_future)
    probs = logits_to_field_probs({f: logits[f][0] for f in logits}, vocab)
    return score_step(probs, cfg)[0], bool(step_veto(probs, cfg))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = [json.loads(l) for l in args.pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
    decisive = [r for r in rows if r.get("preferred") in ("a", "b")]
    if args.max_pairs:
        decisive = decisive[: args.max_pairs]
    print(f"pairs: {len(rows)} total, {len(decisive)} with a decisive executed preference")

    from transformers import AutoTokenizer
    manifest = json.loads((args.checkpoint / "jepa_data_manifest.json").read_text())
    max_input = int(manifest.get("max_input_length") or 2048)
    max_action = int(manifest.get("max_action_length") or 512)
    vocab = fj.load_canonical_event_vocab(args.checkpoint)
    if not vocab:
        raise SystemExit(f"{args.checkpoint} has no canonical_event_vocab.json (need a heads checkpoint)")
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
        # Without these the canonical-event heads are never constructed and every
        # predict_canonical_event_logits call fails.
        canonical_event_vocab_sizes={f: len(v) for f, v in vocab.items()},
        canonical_event_head_hidden_size=int(manifest.get("canonical_event_head_hidden_size") or 512),
        canonical_event_head_inputs=str(manifest.get("canonical_event_head_inputs") or "all"),
    )
    fj.load_jepa_state_dict_for_training(model, args.checkpoint, allow_missing_success_head=True,
                                         allow_missing_canonical_event_heads=False)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device).eval()
    print("loaded checkpoint + canonical-event heads")
    cfg = CanonicalEventScoreConfig()

    correct: dict[str, list[float]] = {r: [] for r in RUNGS}
    by_strength: dict[str, dict[str, list[float]]] = defaultdict(lambda: {r: [] for r in RUNGS})
    margins: dict[str, list[float]] = {r: [] for r in RUNGS}
    skipped = Counter()

    for index, row in enumerate(decisive):
        context_text = fj.build_context_text(row["system_prompt"], row["user_prompt"])
        history = [
            {"step": i + 1,
             "action": fj.render_action(as_tool_call(s["tool_name"], s["arguments"])),
             "observation": s["observation"]}
            for i, s in enumerate(row.get("prefix") or [])
        ]
        current_state_text = fj.build_state_text(context_text, history)
        z = encode(model, tokenizer, [current_state_text, context_text], max_input, device)
        z_cur, z_ctx = z[0:1], z[1:2]

        scores: dict[str, dict[str, float]] = {r: {} for r in RUNGS}
        ok = True
        for branch in ("a", "b"):
            data = row[f"branch_{branch}"]
            action_text = fj.render_action(as_tool_call(data["tool_name"], data["arguments"]))
            z_act = encode(model, tokenizer, [action_text], max_action, device)
            next_state_text = fj.build_next_state_text(
                context_text, history, action_text, data.get("observation") or "")
            z_true = encode(model, tokenizer, [next_state_text], max_input, device)
            z_pred, _ = model.predict_latent(z_cur, z_act, z_ctx)
            futures = {
                "no_future": torch.zeros_like(z_pred),
                "persistence": z_cur,
                "predicted": z_pred,
                "oracle_next": z_true,
            }
            for rung, z_future in futures.items():
                # Deliberately NOT wrapped in try/except: a scoring failure here means the
                # model/heads are misconfigured, and silently counting it as "skipped" produces
                # an empty result table that looks like a finding.
                scores[rung][branch] = score_with_future(
                    model, vocab, cfg, z_cur, z_act, z_ctx, z_future)[0]
            if not ok:
                break
        if not ok:
            skipped["scoring_error"] += 1
            continue

        preferred, other = row["preferred"], ("b" if row["preferred"] == "a" else "a")
        strength = row.get("preference_strength", "none")
        for rung in RUNGS:
            hit = 1.0 if scores[rung][preferred] > scores[rung][other] else 0.0
            correct[rung].append(hit)
            by_strength[strength][rung].append(hit)
            margins[rung].append(scores[rung][preferred] - scores[rung][other])
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(decisive)}", flush=True)

    n = len(correct["predicted"])
    print(f"\n=== pairwise preference accuracy on EXECUTED counterfactuals (n={n}, chance=0.500) ===")
    report: dict[str, Any] = {"n": n, "chance": 0.5, "checkpoint": str(args.checkpoint), "rungs": {}}
    for rung in RUNGS:
        acc = sum(correct[rung]) / n if n else float("nan")
        lo, hi = bootstrap_ci(correct[rung], args.bootstrap, args.seed)
        mean_margin = sum(margins[rung]) / n if n else float("nan")
        report["rungs"][rung] = {"accuracy": acc, "ci95": [lo, hi], "mean_score_margin": mean_margin}
        print(f"  {rung:12s} acc={acc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  mean margin={mean_margin:+.4f}")

    print("\n=== by preference-label strength ===")
    for strength, buckets in sorted(by_strength.items()):
        m = len(buckets["predicted"])
        line = "  ".join(f"{r}={sum(buckets[r]) / m:.3f}" for r in RUNGS) if m else "n/a"
        print(f"  {strength:14s} (n={m:3d})  {line}")
        report.setdefault("by_strength", {})[strength] = {
            "n": m, **{r: (sum(buckets[r]) / m if m else None) for r in RUNGS}
        }

    # The decisive comparison: does knowing the REAL outcome beat the predicted one?
    if n:
        gain = (sum(correct["oracle_next"]) - sum(correct["predicted"])) / n
        flips = sum(1 for a, b in zip(correct["oracle_next"], correct["predicted"]) if a > b)
        breaks = sum(1 for a, b in zip(correct["oracle_next"], correct["predicted"]) if a < b)
        report["oracle_minus_predicted"] = {"delta_accuracy": gain, "oracle_fixes": flips,
                                            "oracle_breaks": breaks}
        print(f"\noracle_next - predicted: {gain:+.4f}  (oracle fixes {flips}, breaks {breaks})")
        print("  ~0 => a perfect predictor would not improve ranking; the ceiling is the "
              "scorer/ontology, not the transition model.")

    if skipped:
        print(f"\nskipped: {dict(skipped)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
