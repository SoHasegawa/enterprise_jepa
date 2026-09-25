#!/usr/bin/env python3
"""Week 1.3 -- decision-centric evaluation: can the world model RANK actions, not just label them?

plan.md's point: state accuracy is not the quantity the planner consumes. A miscalibrated model
can still order actions correctly, and a more accurate one can order them wrongly. So at each
logged state we build a candidate set and measure ranking quality directly.

Candidate set per logged state (plan.md):
  demonstrated   the action actually taken (the positive)
  wrong_tool     a different tool drawn from the same benchmark's tool inventory
  wrong_args     the SAME tool with arguments borrowed from another trajectory (right verb,
                 wrong entity) -- the hardest and most diagnostic distractor
  redundant      the previous step's action replayed (no-op / already-done)
  risky          a destructive tool (delete/remove/drop/revoke) from the same inventory,
                 emitted only where one exists

Scoring reuses the DEPLOYED path exactly: encode candidate action -> predict_latent ->
canonical-event heads -> logits_to_field_probs -> canonical_event_scoring.score_step. So this
measures the planner's real ranking function, not a proxy.

Metrics: Recall@1 and MRR for the demonstrated action, pairwise preference accuracy per
distractor type, risky-action rejection rate, and the score margin between demonstrated and
best distractor (with a paired bootstrap CI).

SCOPE: existing 11-field schema and existing scorer weights, unchanged.

Usage:
  uv run python src/analysis/week1_decision_ranking.py \
      --checkpoint checkpoints/jepa_msp_decoder_cls_head --device cuda:2 --max-states 800
"""

from __future__ import annotations

import argparse
import json
import random
import re
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

TRAJ = REPO_ROOT / "trajectories"
STEM = "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro"
DEFAULT_EVAL = TRAJ / f"{STEM}_eval_examples_cleaned_value_scored.jsonl"
DEFAULT_CHECKPOINT = Path("checkpoints/jepa_msp_decoder_cls_head")
DEFAULT_OUT = REPO_ROOT / "data" / "week1" / "decision_ranking_results.json"

DESTRUCTIVE = re.compile(r"\b(delete|remove|drop|truncate|revoke|purge|destroy|clear)", re.I)
DISTRACTORS = ("wrong_tool", "wrong_args", "redundant", "risky")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--max-states", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


# --------------------------------------------------------------------------- candidates
def call_of(action: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(action, dict):
        return None
    for call in action.get("tool_calls") or []:
        if isinstance(call, dict):
            fn = call.get("function") if isinstance(call.get("function"), dict) else call
            name = str(fn.get("name") or "").strip()
            if name:
                return name, (fn.get("arguments") if isinstance(fn.get("arguments"), dict) else {})
    return None


def wrap(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


def build_candidates(row, rng, inventory, args_bank, prev_action) -> dict[str, dict[str, Any]] | None:
    """Returns {kind: action_dict}; None when the row has no usable demonstrated call."""
    demo = call_of(row.get("action"))
    if demo is None:
        return None
    name, arguments = demo
    benchmark = str(row.get("benchmark") or "unknown")
    tools = inventory.get(benchmark) or []
    out = {"demonstrated": wrap(name, arguments)}

    others = [t for t in tools if t != name]
    if others:
        out["wrong_tool"] = wrap(rng.choice(others), arguments)

    # same tool, arguments from a DIFFERENT trajectory's call of the same tool
    pool = [a for a in args_bank.get((benchmark, name), []) if a != arguments]
    if pool:
        out["wrong_args"] = wrap(name, rng.choice(pool))

    if prev_action is not None:
        prev = call_of(prev_action)
        if prev and (prev[0], prev[1]) != (name, arguments):
            out["redundant"] = wrap(prev[0], prev[1])

    destructive = [t for t in tools if DESTRUCTIVE.search(t) and t != name]
    if destructive and not DESTRUCTIVE.search(name):
        out["risky"] = wrap(rng.choice(destructive), arguments)
    return out


# --------------------------------------------------------------------------- scoring
@torch.no_grad()
def encode_state_history(model, tokenizer, example, max_action, max_observation, device):
    """The transformer predictor's frame sequence for ONE state: the logged (action, tool
    output) history from input_history, encoded with the checkpoint's own encoder. None for
    step-0 states (no history) and under the MLP predictor -- predict_latent then runs its
    documented degenerate [context, z_current] path, same as training did for those rows."""
    if getattr(model, "predictor_arch", "mlp") != "transformer":
        return None
    depth = min(len(example.history_action_texts), len(example.history_observation_texts))
    if depth == 0:
        return None
    acts = tokenizer(example.history_action_texts[:depth], return_tensors="pt", padding=True,
                     truncation=True, max_length=max_action, add_special_tokens=True)
    obs = tokenizer(example.history_observation_texts[:depth], return_tensors="pt", padding=True,
                    truncation=True, max_length=max_observation, add_special_tokens=True)
    z_a, _ = model.encode_latent_and_logits(acts["input_ids"].to(device), acts["attention_mask"].to(device))
    z_o, _ = model.encode_latent_and_logits(obs["input_ids"].to(device), obs["attention_mask"].to(device))
    valid = torch.ones(1, depth, dtype=torch.bool, device=device)
    return z_o.unsqueeze(0), z_a.unsqueeze(0), valid


@torch.no_grad()
def score_actions(model, tokenizer, vocab, cfg, z_cur, z_ctx, action_texts, max_action, device,
                  frame_history=None):
    """Returns (deployed_scores, raw_scores, vetoed_flags).

    `deployed_scores` reproduce the planner's actual ordering: rank_trajectories sorts vetoed
    candidates LAST regardless of raw score (safety veto on P(failure)/P(deleted)), so a vetoed
    candidate is pushed below every non-vetoed one here. `raw_scores` keep score_step alone, so
    the veto's contribution is measurable rather than baked in.
    """
    enc = tokenizer(action_texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_action, add_special_tokens=True)
    z_act, _ = model.encode_latent_and_logits(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    n = z_act.shape[0]
    cur = z_cur.expand(n, -1).contiguous()
    ctx = z_ctx.expand(n, -1).contiguous()
    history = None
    if frame_history is not None:
        history = tuple(t.expand(n, *t.shape[1:]) for t in frame_history)
    z_pred, _, z_state = model.predict_latent_with_state(cur, z_act, ctx, frame_history=history)
    logits = model.predict_canonical_event_logits(cur, z_act, ctx, z_pred, z_state)
    raw, vetoed = [], []
    for i in range(n):
        probs = logits_to_field_probs({f: logits[f][i] for f in logits}, vocab)
        raw.append(score_step(probs, cfg)[0])
        vetoed.append(bool(step_veto(probs, cfg)))
    floor = min(raw) - 1.0 if raw else 0.0
    deployed = [(floor - 1.0 + s * 1e-6) if v else s for s, v in zip(raw, vetoed)]
    return deployed, raw, vetoed


def bootstrap_ci(values: list[float], n: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

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
        canonical_event_vocab_sizes={f: len(v) for f, v in vocab.items()},
        canonical_event_head_hidden_size=int(manifest.get("canonical_event_head_hidden_size") or 512),
        canonical_event_head_inputs=str(manifest.get("canonical_event_head_inputs") or "all"),
        # Transformer-predictor checkpoints: without these the ctor defaults build an MLP and
        # the whole trained predictor is dropped as unexpected keys.
        predictor_arch=str(manifest.get("predictor_arch") or "mlp"),
        predictor_transformer_dim=int(manifest.get("predictor_transformer_dim") or 0),
        predictor_transformer_layers=int(manifest.get("predictor_transformer_layers") or 6),
        predictor_transformer_heads=int(manifest.get("predictor_transformer_heads") or 16),
        predictor_transformer_mlp_ratio=float(manifest.get("predictor_transformer_mlp_ratio") or 4.0),
        predictor_history_length=int(manifest.get("predictor_history_length") or 0),
    )
    fj.load_jepa_state_dict_for_training(model, args.checkpoint, allow_missing_success_head=True,
                                         allow_missing_canonical_event_heads=False)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device).eval()

    rows = [json.loads(l) for l in args.input.open() if l.strip()]
    examples = fj.build_canonical_event_examples(rows)
    by_key = {(e.trajectory_id, e.interaction_index): e for e in examples}

    inventory: dict[str, set] = defaultdict(set)
    args_bank: dict[tuple[str, str], list] = defaultdict(list)
    prev_by_key: dict[tuple[str, int], Any] = {}
    by_traj: dict[str, list] = defaultdict(list)
    for row in rows:
        by_traj[str(row["trajectory_id"])].append(row)
        call = call_of(row.get("action"))
        if call:
            inventory[str(row.get("benchmark") or "unknown")].add(call[0])
            args_bank[(str(row.get("benchmark") or "unknown"), call[0])].append(call[1])
    for group in by_traj.values():
        group.sort(key=lambda r: int(r.get("interaction_index") or 0))
        for prev, cur in zip(group, group[1:]):
            prev_by_key[(str(cur["trajectory_id"]), int(cur.get("interaction_index") or 0))] = prev.get("action")
    inventory = {k: sorted(v) for k, v in inventory.items()}

    sample = rows if len(rows) <= args.max_states else rng.sample(rows, args.max_states)
    cfg = CanonicalEventScoreConfig()

    wins = {k: [] for k in DISTRACTORS}          # pairwise: demonstrated beats distractor?
    wins_raw = {k: [] for k in DISTRACTORS}      # same, ignoring the safety veto
    ranks, ranks_raw, margins, risky_rejected = [], [], [], []
    veto_rate = defaultdict(list)
    per_benchmark = defaultdict(lambda: {"n": 0, "recall_at_1": 0})
    built = Counter()

    for row in sample:
        key = (str(row["trajectory_id"]), int(row.get("interaction_index") or 0))
        example = by_key.get(key)
        if example is None:
            continue
        candidates = build_candidates(row, rng, inventory, args_bank, prev_by_key.get(key))
        if not candidates or len(candidates) < 2:
            continue
        enc_state = tokenizer([example.current_state_text, example.context_text], return_tensors="pt",
                              padding=True, truncation=True, max_length=max_input, add_special_tokens=True)
        with torch.no_grad():
            z, _ = model.encode_latent_and_logits(enc_state["input_ids"].to(device),
                                                  enc_state["attention_mask"].to(device))
        kinds = list(candidates)
        frame_history = encode_state_history(
            model, tokenizer, example, max_action,
            int(manifest.get("max_observation_length") or 512), device)
        deployed, raw, vetoed = score_actions(model, tokenizer, vocab, cfg, z[0:1], z[1:2],
                                              [fj.render_action(candidates[k]) for k in kinds], max_action, device,
                                              frame_history=frame_history)
        by_kind = dict(zip(kinds, deployed))
        by_kind_raw = dict(zip(kinds, raw))
        for k, v in zip(kinds, vetoed):
            built[k] += 1
            veto_rate[k].append(1.0 if v else 0.0)

        demo_score = by_kind["demonstrated"]
        for kind in DISTRACTORS:
            if kind in by_kind:
                wins[kind].append(1.0 if demo_score > by_kind[kind] else 0.0)
                wins_raw[kind].append(1.0 if by_kind_raw["demonstrated"] > by_kind_raw[kind] else 0.0)
        others = [v for k, v in by_kind.items() if k != "demonstrated"]
        rank = 1 + sum(1 for v in others if v > demo_score)
        ranks.append(rank)
        others_raw = [v for k, v in by_kind_raw.items() if k != "demonstrated"]
        ranks_raw.append(1 + sum(1 for v in others_raw if v > by_kind_raw["demonstrated"]))
        margins.append(demo_score - max(others))
        if "risky" in by_kind:
            risky_rejected.append(1.0 if demo_score > by_kind["risky"] else 0.0)
        benchmark = str(row.get("benchmark") or "unknown")
        per_benchmark[benchmark]["n"] += 1
        per_benchmark[benchmark]["recall_at_1"] += int(rank == 1)

    n = len(ranks)
    if not n:
        raise SystemExit("no scorable states")
    recall1 = sum(1 for r in ranks if r == 1) / n
    mrr = sum(1.0 / r for r in ranks) / n
    margin_mean = sum(margins) / n
    lo, hi = bootstrap_ci(margins, args.bootstrap, args.seed)
    results = {
        "checkpoint": str(args.checkpoint),
        "states_scored": n,
        "candidates_built": dict(built),
        "chance_recall_at_1": n / sum(built.values()),
        "recall_at_1": recall1,
        "recall_at_1_no_veto": sum(1 for r in ranks_raw if r == 1) / n,
        "mrr": mrr,
        "mrr_no_veto": sum(1.0 / r for r in ranks_raw) / n,
        "score_margin_mean": margin_mean,
        "score_margin_ci95": [lo, hi],
        "pairwise_preference_accuracy": {k: (sum(v) / len(v) if v else None) for k, v in wins.items()},
        "pairwise_preference_accuracy_no_veto": {k: (sum(v) / len(v) if v else None) for k, v in wins_raw.items()},
        "pairwise_n": {k: len(v) for k, v in wins.items()},
        "veto_rate_by_candidate": {k: (sum(v) / len(v) if v else None) for k, v in veto_rate.items()},
        "risky_rejection_rate": (sum(risky_rejected) / len(risky_rejected)) if risky_rejected else None,
        "per_benchmark_recall_at_1": {k: v["recall_at_1"] / v["n"] for k, v in per_benchmark.items()},
    }
    print(f"states scored: {n}   candidates: {dict(built)}\n")
    print(f"Recall@1 (demonstrated ranked first): {recall1:.4f}   (chance ~= {results['chance_recall_at_1']:.4f})"
          f"   [no-veto: {results['recall_at_1_no_veto']:.4f}]")
    print(f"MRR:                                  {mrr:.4f}   [no-veto: {results['mrr_no_veto']:.4f}]")
    print(f"score margin (demo - best distractor): {margin_mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("\npairwise preference accuracy (demonstrated preferred over ...):")
    for k in DISTRACTORS:
        v = results["pairwise_preference_accuracy"][k]
        if v is not None:
            print(f"  vs {k:12s}: {v:.4f}  [no-veto {results['pairwise_preference_accuracy_no_veto'][k]:.4f}]"
                  f"  (n={results['pairwise_n'][k]}, vetoed {results['veto_rate_by_candidate'][k]:.1%})")
    if results["risky_rejection_rate"] is not None:
        print(f"\nrisky-action rejection rate: {results['risky_rejection_rate']:.4f}")
    print("\nper-benchmark Recall@1:", {k: round(v, 4) for k, v in results["per_benchmark_recall_at_1"].items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
