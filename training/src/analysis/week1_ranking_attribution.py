#!/usr/bin/env python3
"""Week 1.5 -- which fields drive the ranking failure, and is the utility function or the
representation at fault?

Runs one scoring pass over the candidate sets from week1_decision_ranking (demonstrated /
wrong_tool / wrong_args / redundant / risky), caching every candidate's 11 probability vectors,
then evaluates many SCORERS over the identical cached probabilities -- so scorer variants cost
nothing extra and are perfectly paired.

Scorers compared:
  handcrafted_all      current DEFAULT_WEIGHTS/UTILITIES (the deployed scorer)
  transition_only      execution_status + error_signature + side_effect_type
  nudge_only           information_sufficiency + recommended_abstract_action + missing_info
  single:<field>       each scored field alone
  loo:<field>          leave-one-field-out
  oracle_labels        gold labels pushed through the SAME handcrafted utility (one-hot probs)
  learned_linear       logistic regression on the 11 probability vectors, fit to prefer the
                       demonstrated action (Bradley-Terry style, fit on a train split of states,
                       reported on a disjoint eval split)

Statistics:
  * within-state PERMUTATION NULL for every metric: candidate scores are shuffled within each
    state, so Recall@1 / MRR / margin get an empirical null distribution. This is the correct
    reference -- margin = demo - max(K distractors) is negative in expectation even for a random
    scorer, so raw margin sign is NOT evidence of inversion.
  * breakdowns by candidate type and by DEMONSTRATED action_type (read/search/validate vs
    create/update/delete), testing whether information-gathering demonstrations are penalised
    against premature mutations.

Usage:
  uv run python src/analysis/week1_ranking_attribution.py \
      --checkpoint checkpoints/jepa_msp_decoder_cls_head --device cuda:2 --max-states 1200
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
import src.analysis.week1_decision_ranking as dr
from src.canonical_event_scoring import (
    DEFAULT_UTILITIES, DEFAULT_WEIGHTS, MISSING_INFO_BENIGN, MISSING_INFO_FIELD,
    SCORED_SINGLE_FIELDS, CanonicalEventScoreConfig, logits_to_field_probs,
)

TRAJ = REPO_ROOT / "trajectories"
STEM = "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro"
DEFAULT_EVAL = TRAJ / f"{STEM}_eval_examples_cleaned_value_scored.jsonl"
DEFAULT_CHECKPOINT = Path("checkpoints/jepa_msp_decoder_cls_head")
DEFAULT_OUT = REPO_ROOT / "data" / "week1" / "ranking_attribution_results.json"

TRANSITION = ("execution_status", "error_signature", "side_effect_type")
NUDGE = ("information_sufficiency", "recommended_abstract_action", MISSING_INFO_FIELD)
GATHER = {"read", "search", "validate", "test"}
MUTATE = {"create", "update", "delete", "communicate", "run"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--max-states", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--permutations", type=int, default=2000)
    return p.parse_args()


# --------------------------------------------------------------------------- scorers
def field_utility(field: str, probs: dict[str, float], utilities) -> float:
    if field == MISSING_INFO_FIELD:
        nonbenign = [p for c, p in probs.items() if c not in MISSING_INFO_BENIGN]
        return -(sum(nonbenign) / len(nonbenign)) if nonbenign else 0.0
    utility = utilities.get(field) or {}
    return sum(p * utility.get(c, 0.0) for c, p in probs.items())


def score_with(fields, per_field_probs, weights, utilities) -> float:
    return sum(weights.get(f, 0.0) * field_utility(f, per_field_probs[f], utilities)
               for f in fields if f in per_field_probs)


ALL_SCORED = tuple(SCORED_SINGLE_FIELDS) + (MISSING_INFO_FIELD,)


def build_scorers() -> dict[str, tuple]:
    scorers: dict[str, tuple] = {
        "handcrafted_all": ALL_SCORED,
        "transition_only": TRANSITION,
        "nudge_only": NUDGE,
    }
    for f in ALL_SCORED:
        scorers[f"single:{f}"] = (f,)
        scorers[f"loo:{f}"] = tuple(x for x in ALL_SCORED if x != f)
    return scorers


# --------------------------------------------------------------------------- metrics
def rank_metrics(states: list[dict[str, float]]) -> dict[str, float]:
    """states: list of {kind: score}; the positive is always 'demonstrated'."""
    ranks, margins = [], []
    for by_kind in states:
        demo = by_kind["demonstrated"]
        others = [v for k, v in by_kind.items() if k != "demonstrated"]
        if not others:
            continue
        ranks.append(1 + sum(1 for v in others if v > demo))
        margins.append(demo - max(others))
    n = max(len(ranks), 1)
    return {
        "recall_at_1": sum(1 for r in ranks if r == 1) / n,
        "mrr": sum(1.0 / r for r in ranks) / n,
        "margin": sum(margins) / n,
        "n": len(ranks),
    }


def permutation_null(states: list[dict[str, float]], trials: int, seed: int) -> dict[str, Any]:
    """Shuffle scores WITHIN each state -- the correct null for max-of-K comparisons."""
    rng = random.Random(seed)
    draws = {"recall_at_1": [], "mrr": [], "margin": []}
    for _ in range(trials):
        shuffled = []
        for by_kind in states:
            kinds = list(by_kind)
            values = [by_kind[k] for k in kinds]
            rng.shuffle(values)
            shuffled.append(dict(zip(kinds, values)))
        m = rank_metrics(shuffled)
        for key in draws:
            draws[key].append(m[key])
    out: dict[str, Any] = {}
    for key, values in draws.items():
        values.sort()
        out[key] = {"mean": sum(values) / len(values),
                    "p2.5": values[int(0.025 * len(values))],
                    "p97.5": values[int(0.975 * len(values))]}
    return out


def p_value(observed: float, null_draws: list[float]) -> float:
    """One-sided (observed greater) empirical p with add-one smoothing."""
    at_least = sum(1 for v in null_draws if v >= observed)
    return (at_least + 1) / (len(null_draws) + 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    from transformers import AutoTokenizer
    manifest = json.loads((args.checkpoint / "jepa_data_manifest.json").read_text())
    max_input = int(manifest.get("max_input_length") or 2048)
    max_action = int(manifest.get("max_action_length") or 512)
    vocab = fj.load_canonical_event_vocab(args.checkpoint)
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
    )
    fj.load_jepa_state_dict_for_training(model, args.checkpoint, allow_missing_success_head=True,
                                         allow_missing_canonical_event_heads=False)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device).eval()

    rows = [json.loads(l) for l in args.input.open() if l.strip()]
    examples = fj.build_canonical_event_examples(rows)
    by_key = {(e.trajectory_id, e.interaction_index): e for e in examples}
    inventory, args_bank, prev_by_key = defaultdict(set), defaultdict(list), {}
    by_traj = defaultdict(list)
    for row in rows:
        by_traj[str(row["trajectory_id"])].append(row)
        call = dr.call_of(row.get("action"))
        if call:
            inventory[str(row.get("benchmark") or "unknown")].add(call[0])
            args_bank[(str(row.get("benchmark") or "unknown"), call[0])].append(call[1])
    for group in by_traj.values():
        group.sort(key=lambda r: int(r.get("interaction_index") or 0))
        for prev, cur in zip(group, group[1:]):
            prev_by_key[(str(cur["trajectory_id"]), int(cur.get("interaction_index") or 0))] = prev.get("action")
    inventory = {k: sorted(v) for k, v in inventory.items()}

    sample = rows if len(rows) <= args.max_states else rng.sample(rows, args.max_states)

    # ---- one scoring pass: cache probability vectors for every candidate ----
    cached: list[dict[str, Any]] = []
    for row in sample:
        key = (str(row["trajectory_id"]), int(row.get("interaction_index") or 0))
        example = by_key.get(key)
        if example is None:
            continue
        candidates = dr.build_candidates(row, rng, inventory, args_bank, prev_by_key.get(key))
        if not candidates or len(candidates) < 2:
            continue
        kinds = list(candidates)
        enc_state = tokenizer([example.current_state_text, example.context_text], return_tensors="pt",
                              padding=True, truncation=True, max_length=max_input, add_special_tokens=True)
        enc_act = tokenizer([fj.render_action(candidates[k]) for k in kinds], return_tensors="pt",
                            padding=True, truncation=True, max_length=max_action, add_special_tokens=True)
        with torch.no_grad():
            z, _ = model.encode_latent_and_logits(enc_state["input_ids"].to(device), enc_state["attention_mask"].to(device))
            z_act, _ = model.encode_latent_and_logits(enc_act["input_ids"].to(device), enc_act["attention_mask"].to(device))
            n = z_act.shape[0]
            cur, ctx = z[0:1].expand(n, -1).contiguous(), z[1:2].expand(n, -1).contiguous()
            z_pred, _ = model.predict_latent(cur, z_act, ctx)
            logits = model.predict_canonical_event_logits(cur, z_act, ctx, z_pred)
        cached.append({
            "benchmark": str(row.get("benchmark") or "unknown"),
            "demo_action_type": (row.get("canonical_event_with_nudge") or {}).get("canonical_event_state", {}).get("action_type"),
            "gold": (row.get("canonical_event_with_nudge") or {}),
            "probs": {k: logits_to_field_probs({f: logits[f][i] for f in logits}, vocab) for i, k in enumerate(kinds)},
        })
    print(f"scored {len(cached)} states\n")

    # ---- evaluate every scorer over the SAME cached probabilities ----
    scorers = build_scorers()
    results: dict[str, Any] = {"states": len(cached), "scorers": {}}
    for name, fields in scorers.items():
        states = [{k: score_with(fields, p, DEFAULT_WEIGHTS, DEFAULT_UTILITIES) for k, p in c["probs"].items()}
                  for c in cached]
        results["scorers"][name] = rank_metrics(states)

    # oracle labels through the same handcrafted utility (one-hot probabilities)
    def one_hot(gold: dict[str, Any]) -> dict[str, dict[str, float]]:
        state, nudge = gold.get("canonical_event_state") or {}, gold.get("nudge") or {}
        probs = {f: {str(state.get(f) or nudge.get(f)): 1.0} for f in SCORED_SINGLE_FIELDS}
        missing = nudge.get(MISSING_INFO_FIELD) or ["none"]
        probs[MISSING_INFO_FIELD] = {str(v): 1.0 for v in missing}
        return probs
    # the oracle only knows the DEMONSTRATED action's true outcome; distractors keep predictions,
    # which is the fairest available approximation without executing them (see the oracle ladder).
    oracle_states = []
    for c in cached:
        entry = {k: score_with(ALL_SCORED, p, DEFAULT_WEIGHTS, DEFAULT_UTILITIES) for k, p in c["probs"].items()}
        entry["demonstrated"] = score_with(ALL_SCORED, one_hot(c["gold"]), DEFAULT_WEIGHTS, DEFAULT_UTILITIES)
        oracle_states.append(entry)
    results["scorers"]["oracle_demo_labels"] = rank_metrics(oracle_states)

    # ---- learned linear scorer (train/eval split over states) ----
    def featurize(p) -> list[float]:
        feats = []
        for f in ALL_SCORED:
            for c in sorted(vocab[f]):
                feats.append(p[f].get(c, 0.0))
        return feats
    split = int(0.6 * len(cached))
    train_c, eval_c = cached[:split], cached[split:]
    X, Y = [], []
    for c in train_c:                      # pairwise: demonstrated (+1) vs each distractor (-1)
        demo = featurize(c["probs"]["demonstrated"])
        for k, p in c["probs"].items():
            if k != "demonstrated":
                diff = [a - b for a, b in zip(demo, featurize(p))]
                X.append(diff); Y.append(1.0)
                X.append([-d for d in diff]); Y.append(0.0)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    w = torch.zeros(Xt.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05, weight_decay=1e-4)
    for _ in range(400):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w, Yt)
        opt.zero_grad(); loss.backward(); opt.step()
    learned_states = [{k: float(torch.tensor(featurize(p)) @ w.detach()) for k, p in c["probs"].items()}
                      for c in eval_c]
    results["scorers"]["learned_linear(eval split)"] = rank_metrics(learned_states)
    hand_eval = [{k: score_with(ALL_SCORED, p, DEFAULT_WEIGHTS, DEFAULT_UTILITIES) for k, p in c["probs"].items()}
                 for c in eval_c]
    results["scorers"]["handcrafted(eval split)"] = rank_metrics(hand_eval)

    # ---- permutation null on the deployed scorer ----
    deployed = [{k: score_with(ALL_SCORED, p, DEFAULT_WEIGHTS, DEFAULT_UTILITIES) for k, p in c["probs"].items()}
                for c in cached]
    null = permutation_null(deployed, args.permutations, args.seed)
    observed = rank_metrics(deployed)
    rng2 = random.Random(args.seed)
    draws = {"recall_at_1": [], "mrr": [], "margin": []}
    for _ in range(args.permutations):
        shuffled = []
        for by_kind in deployed:
            kinds = list(by_kind); values = [by_kind[k] for k in kinds]
            rng2.shuffle(values); shuffled.append(dict(zip(kinds, values)))
        m = rank_metrics(shuffled)
        for key in draws:
            draws[key].append(m[key])
    results["permutation_null"] = {
        "null": null,
        "observed": observed,
        "p_values": {k: p_value(observed[k], draws[k]) for k in draws},
    }

    # ---- breakdown by demonstrated action_type (gather vs mutate) ----
    groups = {"gather(read/search/validate/test)": GATHER, "mutate(create/update/delete/run/comm)": MUTATE}
    results["by_demo_action_type"] = {}
    for label, members in groups.items():
        subset = [d for d, c in zip(deployed, cached) if c["demo_action_type"] in members]
        if subset:
            results["by_demo_action_type"][label] = rank_metrics(subset)

    # ---------------------------------------------------------------- report
    print("=== permutation null (scores shuffled WITHIN each state) ===")
    for key in ("recall_at_1", "mrr", "margin"):
        o, nl = observed[key], null[key]
        print(f"  {key:12s} observed={o:+.4f}   null mean={nl['mean']:+.4f} "
              f"[{nl['p2.5']:+.4f},{nl['p97.5']:+.4f}]   p={results['permutation_null']['p_values'][key]:.3f}")
    print("  (a negative margin is EXPECTED under the null: demo - max(K distractors))\n")

    print("=== scorer variants (Recall@1 / MRR) ===")
    order = sorted(results["scorers"], key=lambda k: -results["scorers"][k]["recall_at_1"])
    for name in order:
        m = results["scorers"][name]
        print(f"  {name:34s} R@1={m['recall_at_1']:.4f}  MRR={m['mrr']:.4f}  n={m['n']}")

    print("\n=== by demonstrated action type ===")
    for label, m in results["by_demo_action_type"].items():
        print(f"  {label:38s} R@1={m['recall_at_1']:.4f}  MRR={m['mrr']:.4f}  n={m['n']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
