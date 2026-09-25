"""Scoring for beam-search / lookahead planning over imagined canonical-event steps.

Every n execution steps the agent proposes m candidate trajectories of n imagined
steps; the world model predicts, per step, each canonical-event/nudge field's
probability distribution (softmax for single-label heads, per-class sigmoid for the
multi-label head). This module turns those distributions into a per-step score, a
per-trajectory score (discounted sum), a safety veto, and a top-k ranking with
human-readable reasons to hand back to the agent.

Design (see discussion):
  * Score each field by its EXPECTED UTILITY under the head distribution
    (uses the probabilities, not argmax): U_f = sum_c p_f(c) * u_f(c).
  * step_score = sum_f w_f * U_f.
  * trajectory_score = sum_k gamma^{k-1} * step_score_k  (discount later, less
    reliable, horizons).
  * A hard SAFETY VETO prunes trajectories whose any step is a likely failure or an
    irreversible deletion -- catastrophic tails should not be averaged away by a sum.

Scored fields: execution_status, progress_signal, information_sufficiency,
error_signature, side_effect_type.
Ignored: action_type / object_type (descriptive), risk_signal (mostly 'none', not
discriminative), information_gain (information_sufficiency classifies more accurately),
recommended_abstract_action and missing_information_type (weak next-state prediction signal).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCORED_SINGLE_FIELDS = (
    "execution_status",
    "progress_signal",
    "information_sufficiency",
    "error_signature",
    "side_effect_type",
)
MISSING_INFO_FIELD = "missing_information_type"
MISSING_INFO_BENIGN = frozenset({"none", "unknown"})

# Per-category utility in [-1, 1], grounded in the real vocab (canonical_event_vocab.json).
DEFAULT_UTILITIES: dict[str, dict[str, float]] = {
    "execution_status": {"success": 1.0, "partial": 0.0, "no_op": 0.0, "failure": -1.0, "unknown": 0.0},
    "progress_signal": {"positive": 1.0, "neutral": 0.0, "negative": -1.0, "unknown": 0.0},
    "information_sufficiency": {"sufficient": 1.0, "insufficient": 0.0},
    # Errors: any concrete error is bad; none/unknown neutral. Low weight (correlated with execution_status).
    "error_signature": {
        "none": 0.0, "unknown": 0.0, "dependency_missing": -1.0, "invalid_argument": -1.0,
        "not_found": -1.0, "parse_error": -1.0, "permission_denied": -1.0, "policy_risk": -1.0,
        "runtime_error": -1.0, "test_failed": -1.0, "timeout": -1.0,
    },
    # Side effects: mutations the agent intends (created/modified/...) are neutral; irreversible /
    # externally-visible / failed ones are penalised. `deleted` is the irreversible-destructive signal.
    "side_effect_type": {
        "deleted": -1.0, "failed_validation": -0.5, "sent": -0.3, "created": 0.0, "modified": 0.0,
        "executed": 0.0, "installed": 0.0, "retrieved": 0.0, "validated": 0.0, "none": 0.0, "unknown": 0.0,
    },
}
DEFAULT_WEIGHTS: dict[str, float] = {
    "execution_status": 1.0,
    "progress_signal": 0.8,
    "information_sufficiency": 0.5,
    "side_effect_type": 0.6,
    "error_signature": 0.4,
}


@dataclass
class CanonicalEventScoreConfig:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    utilities: dict[str, dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_UTILITIES.items()}
    )
    gamma: float = 0.9              # horizon discount (later steps less reliable); 1.0 = plain sum
    veto_failure_prob: float = 0.6  # prune if any step P(execution_status=failure) exceeds this
    veto_deleted_prob: float = 0.5  # prune if any step P(side_effect_type=deleted) exceeds this
    top_k: int = 3


def _expected_utility(probs: dict[str, float], utility: dict[str, float]) -> float:
    return sum(prob * utility.get(category, 0.0) for category, prob in probs.items())


def score_step(field_probs: dict[str, dict[str, float]], config: CanonicalEventScoreConfig) -> tuple[float, dict[str, float]]:
    """Per-step score = sum_f w_f * expected_utility_f. Returns (score, per-field contributions)."""
    total = 0.0
    contributions: dict[str, float] = {}
    for name in SCORED_SINGLE_FIELDS:
        probs = field_probs.get(name)
        utility = config.utilities.get(name)
        if not probs or not utility:
            continue
        contribution = config.weights.get(name, 0.0) * _expected_utility(probs, utility)
        total += contribution
        contributions[name] = contribution
    return total, contributions


def step_veto(field_probs: dict[str, dict[str, float]], config: CanonicalEventScoreConfig) -> list[str]:
    """Catastrophic-tail reasons to veto (empty list = no veto)."""
    reasons: list[str] = []
    failure = field_probs.get("execution_status", {}).get("failure", 0.0)
    if failure > config.veto_failure_prob:
        reasons.append(f"P(failure)={failure:.2f}")
    deleted = field_probs.get("side_effect_type", {}).get("deleted", 0.0)
    if deleted > config.veto_deleted_prob:
        reasons.append(f"P(deleted)={deleted:.2f}")
    return reasons


def score_trajectory(
    steps: list[dict[str, dict[str, float]]], config: CanonicalEventScoreConfig
) -> dict[str, Any]:
    """Discounted sum of per-step scores + safety veto over the horizon."""
    per_step: list[dict[str, Any]] = []
    total = 0.0
    vetoed = False
    veto_reasons: list[dict[str, Any]] = []
    aggregate: dict[str, float] = {}
    for index, field_probs in enumerate(steps):
        step_value, contributions = score_step(field_probs, config)
        discount = config.gamma ** index
        total += discount * step_value
        for name, value in contributions.items():
            aggregate[name] = aggregate.get(name, 0.0) + discount * value
        reasons = step_veto(field_probs, config)
        if reasons:
            vetoed = True
            veto_reasons.append({"step": index + 1, "reasons": reasons})
        per_step.append({"step": index + 1, "score": step_value, "contributions": contributions})
    return {
        "score": total,
        "vetoed": vetoed,
        "veto_reasons": veto_reasons,
        "per_step": per_step,
        "field_contributions": aggregate,
    }


def explain_trajectory(scored: dict[str, Any], top_fields: int = 3) -> str:
    """Short reason string: dominant +/- field contributions (and any veto)."""
    if scored.get("vetoed"):
        flat = "; ".join(f"step {r['step']}: {', '.join(r['reasons'])}" for r in scored["veto_reasons"])
        return f"VETOED ({flat})"
    contributions = sorted(scored["field_contributions"].items(), key=lambda kv: kv[1], reverse=True)
    positives = [f"{name}:+{value:.2f}" for name, value in contributions if value > 1e-6][:top_fields]
    negatives = [f"{name}:{value:.2f}" for name, value in reversed(contributions) if value < -1e-6][:top_fields]
    parts = []
    if positives:
        parts.append("+ " + ", ".join(positives))
    if negatives:
        parts.append("- " + ", ".join(negatives))
    return " | ".join(parts) if parts else "neutral"


def rank_trajectories(
    trajectories: list[list[dict[str, dict[str, float]]]],
    config: CanonicalEventScoreConfig | None = None,
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score and rank candidate trajectories. Returns (top_k, all_scored). Vetoed trajectories
    sort last regardless of raw score. Each record carries index, score, vetoed, reason, and
    normalized_score (softmax over non-vetoed candidates) for handing back to the agent."""
    config = config or CanonicalEventScoreConfig()
    top_k = top_k or config.top_k
    scored: list[dict[str, Any]] = []
    for index, trajectory in enumerate(trajectories):
        record = score_trajectory(trajectory, config)
        record["index"] = index
        record["reason"] = explain_trajectory(record)
        scored.append(record)
    _add_normalized_scores(scored)
    # Non-vetoed first, then by score descending.
    scored.sort(key=lambda record: (record["vetoed"], -record["score"]))
    return scored[:top_k], scored


def _add_normalized_scores(scored: list[dict[str, Any]]) -> None:
    import math

    live = [record for record in scored if not record["vetoed"]]
    if not live:
        for record in scored:
            record["normalized_score"] = 0.0
        return
    highest = max(record["score"] for record in live)
    exps = {id(record): math.exp(record["score"] - highest) for record in live}
    denom = sum(exps.values()) or 1.0
    for record in scored:
        record["normalized_score"] = (exps[id(record)] / denom) if not record["vetoed"] else 0.0


def logits_to_field_probs_batched(
    field_logits: dict[str, Any],
    vocab: dict[str, list[str]],
    *,
    temperature: float = 1.0,
    multi_label_fields: tuple[str, ...] = (MISSING_INFO_FIELD,),
) -> list[dict[str, dict[str, float]]]:
    """Batched form of logits_to_field_probs: takes [B, C] logits per field and returns one
    {field: {category: prob}} dict per row.

    Why this exists: the per-row version reads probabilities out of CUDA tensors one scalar at
    a time (`float(per_class[i])`), and every one of those is a device synchronisation. At
    B plans x 11 fields x ~10 classes that is hundreds of syncs per rollout step -- measured
    at 9.3 ms per step for B=7 versus 2.8 ms when the transfer happens once per field. The
    softmax/sigmoid still runs on the original device, so the numbers are the same; only the
    host transfer is hoisted out of the inner loop.
    """
    import torch

    per_field_rows: dict[str, list[list[float]]] = {}
    batch_size = 0
    for name, logits in field_logits.items():
        if vocab.get(name) is None:
            continue
        values = logits.detach().float()
        if values.dim() == 1:
            values = values.unsqueeze(0)
        values = values / max(temperature, 1e-6)
        activated = torch.sigmoid(values) if name in multi_label_fields else torch.softmax(values, dim=-1)
        per_field_rows[name] = activated.cpu().tolist()   # one host transfer per field
        batch_size = max(batch_size, len(per_field_rows[name]))

    results: list[dict[str, dict[str, float]]] = []
    for row in range(batch_size):
        probs: dict[str, dict[str, float]] = {}
        for name, rows in per_field_rows.items():
            if row >= len(rows):
                continue
            categories = vocab[name]
            values = rows[row]
            probs[name] = {
                categories[i]: float(values[i]) for i in range(min(len(categories), len(values)))
            }
        results.append(probs)
    return results


def logits_to_field_probs(
    field_logits: dict[str, Any],
    vocab: dict[str, list[str]],
    *,
    temperature: float = 1.0,
    multi_label_fields: tuple[str, ...] = (MISSING_INFO_FIELD,),
) -> dict[str, dict[str, float]]:
    """Convert one step's raw head logits (torch tensors, shape [num_classes]) into
    {field: {category: prob}} using softmax (single-label) or sigmoid (multi-label),
    with optional temperature scaling for calibration."""
    import torch

    probs: dict[str, dict[str, float]] = {}
    for name, logits in field_logits.items():
        categories = vocab.get(name)
        if categories is None:
            continue
        values = logits.detach().float().reshape(-1) / max(temperature, 1e-6)
        if name in multi_label_fields:
            per_class = torch.sigmoid(values)
        else:
            per_class = torch.softmax(values, dim=-1)
        probs[name] = {categories[i]: float(per_class[i]) for i in range(min(len(categories), per_class.shape[0]))}
    return probs
