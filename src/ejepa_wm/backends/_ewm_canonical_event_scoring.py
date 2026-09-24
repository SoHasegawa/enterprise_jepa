"""Scoring for beam-search / lookahead planning over imagined canonical-event steps.

Inference-only **port** of the EWM repo's ``src/canonical_event_scoring.py`` (kept
byte-for-byte in behaviour so the mcp_react ``beam_plan`` strategy scores plans
exactly like the standalone ``--replay-modes beam_plan`` harness). Like the sibling
``_ewm_*`` modules it is self-contained — no imports from an external EWM checkout —
and imports ``torch`` lazily inside :func:`logits_to_field_probs` so importing this
module never pulls torch in on its own.

Every ``n`` execution steps the agent proposes ``m`` candidate trajectories of ``n``
imagined steps; the world model predicts, per step, each canonical-event/nudge field's
probability distribution (softmax for single-label heads, per-class sigmoid for the
multi-label head). This module turns those distributions into a per-step score, a
per-trajectory score (plain sum), a safety veto, and a top-k ranking with
human-readable reasons to hand back to the agent.

Design:
  * Score each field by its EXPECTED UTILITY under the head distribution
    (uses the probabilities, not argmax): U_f = sum_c p_f(c) * u_f(c).
  * step_score = sum_f w_f * U_f.
  * trajectory_score = sum_k step_score_k. Later steps receive equal weight so
    useful state-changing actions at the end of a short plan remain competitive.
  * A hard SAFETY VETO prunes trajectories whose any step is a likely failure or an
    irreversible deletion -- catastrophic tails should not be averaged away by a sum.

Scored fields: execution_status, progress_signal, information_sufficiency,
error_signature, side_effect_type.
Ignored: action_type / object_type (descriptive), risk_signal (mostly 'none', not
discriminative), information_gain (information_sufficiency classifies more accurately),
recommended_abstract_action and missing_information_type (weak next-state prediction signal).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ejepa_wm.backends._canonical_event_state import infer_action_type

SCORED_SINGLE_FIELDS = (
    "execution_status",
    "progress_signal",
    "information_sufficiency",
    "error_signature",
    "side_effect_type",
)
MISSING_INFO_FIELD = "missing_information_type"

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
    veto_failure_prob: float = 0.6  # prune if any step P(execution_status=failure) exceeds this
    veto_deleted_prob: float = 0.5  # prune if any step P(side_effect_type=deleted) exceeds this
    temperature: float = 1.0        # head-logit temperature for scoring calibration
    top_k: int = 3
    read_saturation_threshold: int = 2
    read_after_saturation_penalty: float = 1.0
    read_after_progress_scale: float = 0.5
    first_step_score_weight: float = 0.0
    required_action_coverage_bonus: float = 0.0
    first_required_action_bonus: float = 0.0
    # Keep destructive-side-effect vetoes intact until candidate tool calls are validated
    # against the benchmark's tool schema. This may be enabled explicitly for controlled
    # experiments once malformed destructive calls cannot receive the exemption.
    exempt_task_required_side_effects: bool = False


_DESTRUCTIVE_TASK_STEMS = (
    "archiv", "cancel", "delet", "drop", "remov", "revok", "terminat", "truncat",
)
_NEGATED_OPERATION = re.compile(
    r"(?:do\s+not|don't|must\s+not|never|avoid|without)\s+(?:\w+\s+){0,2}$"
)

_REQUIRED_ACTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "create": (r"\b(?:add|create|log|onboard|open|register|schedule)\b",),
    "update": (
        r"\b(?:assign|change|configure|extend|modify|move|publish|rename|set|star|update|upgrade)\b",
    ),
    "delete": (r"\b(?:archive|cancel|delete|remove|revoke|terminate)\b",),
    "run": (r"\b(?:build|deploy|execute|install|restart|run|start|stop)\b",),
    "test": (r"\btest\b",),
    "communicate": (r"\b(?:email|message|notify|reply|send)\b",),
    "validate": (r"\b(?:check|confirm|validate|verify)\b",),
}


def _tool_calls(action: Any) -> list[dict[str, Any]]:
    if isinstance(action, list):
        return [call for item in action for call in _tool_calls(item)]
    if not isinstance(action, dict):
        return []
    if isinstance(action.get("tool_calls"), list):
        return [call for item in action["tool_calls"] for call in _tool_calls(item)]
    function = action.get("function")
    if isinstance(function, dict) and function.get("name"):
        return [{"name": function["name"], "arguments": function.get("arguments", {})}]
    name = action.get("name") or action.get("tool") or action.get("tool_name")
    if not name:
        return []
    return [{"name": name, "arguments": action.get("arguments", action.get("args", {}))}]


def _inferred_action_types(action: Any) -> list[str]:
    return [
        infer_action_type(str(call["name"]), call.get("arguments", {}))
        for call in _tool_calls(action)
    ]


def _action_signature(action: Any) -> str:
    return json.dumps(_tool_calls(action), ensure_ascii=False, sort_keys=True, default=str)


def _observation_succeeded(observation: Any) -> bool:
    if observation is None:
        return False
    if isinstance(observation, dict) and (
        observation.get("isError") is True or observation.get("success") is False
    ):
        return False
    text = observation if isinstance(observation, str) else json.dumps(observation, default=str)
    lowered = text.lower()
    return not any(
        marker in lowered
        for marker in ("permission denied", "unauthorized", "tool error", "runtime error")
    )


def _successful_recent_reads(input_history: list[dict[str, Any]] | None) -> set[str]:
    signatures: set[str] = set()
    for entry in reversed(input_history or []):
        if not isinstance(entry, dict):
            break
        action_types = _inferred_action_types(entry.get("action"))
        if not action_types or any(kind not in {"read", "search"} for kind in action_types):
            break
        if not _observation_succeeded(entry.get("observation")):
            break
        signatures.add(_action_signature(entry["action"]))
    return signatures


def _successful_progress_actions(input_history: list[dict[str, Any]] | None) -> int:
    count = 0
    for entry in input_history or []:
        if not isinstance(entry, dict) or not _observation_succeeded(entry.get("observation")):
            continue
        action_types = _inferred_action_types(entry.get("action"))
        if action_types and any(kind not in {"read", "search", "unknown"} for kind in action_types):
            count += 1
    return count


def _task_requires_destructive_action(task_text: str) -> bool:
    lowered = task_text.lower()

    for match in re.finditer(r"\b[a-z]+\b", lowered):
        word = match.group(0)
        if any(word.startswith(stem) for stem in _DESTRUCTIVE_TASK_STEMS):
            prefix = lowered[max(0, match.start() - 32) : match.start()]
            if not _NEGATED_OPERATION.search(prefix):
                return True
    return False


def _task_required_action_types(task_text: str) -> set[str]:
    """Infer explicitly requested operation classes without benchmark/tool-name rules."""
    lowered = str(task_text or "").lower()
    required: set[str] = set()
    for action_type, patterns in _REQUIRED_ACTION_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, lowered):
                prefix = lowered[max(0, match.start() - 32) : match.start()]
                if not _NEGATED_OPERATION.search(prefix):
                    required.add(action_type)
                    break
            if action_type in required:
                break
    return required


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
    """Plain sum of per-step scores + safety veto over the horizon."""
    per_step: list[dict[str, Any]] = []
    total = 0.0
    vetoed = False
    veto_reasons: list[dict[str, Any]] = []
    aggregate: dict[str, float] = {}
    for index, field_probs in enumerate(steps):
        step_value, contributions = score_step(field_probs, config)
        total += step_value
        for name, value in contributions.items():
            aggregate[name] = aggregate.get(name, 0.0) + value
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

def _apply_action_aware_adjustment(
    record: dict[str, Any],
    trajectory: list[dict[str, dict[str, float]]],
    action_plan: list[Any],
    config: CanonicalEventScoreConfig,
    task_text: str,
    input_history: list[dict[str, Any]] | None,
) -> None:
    reads = _successful_recent_reads(input_history)
    progress_actions = _successful_progress_actions(input_history)
    threshold = max(0, int(config.read_saturation_threshold))
    penalty = max(0.0, float(config.read_after_saturation_penalty))
    task_requires_delete = _task_requires_destructive_action(task_text)
    required_action_types = _task_required_action_types(task_text)
    adjustments: list[dict[str, Any]] = []

    first_step_weight = max(0.0, float(config.first_step_score_weight))
    if record.get("per_step") and first_step_weight:
        first_score = float(record["per_step"][0].get("score", 0.0))
        delta = first_step_weight * first_score
        record["score"] += delta
        record["per_step"][0]["score"] += delta
        record["per_step"][0]["contributions"]["first_step_priority"] = delta
        record["field_contributions"]["first_step_priority"] = delta
        adjustments.append(
            {
                "step": 1,
                "kind": "first_step_priority",
                "first_step_score": first_score,
                "weight": first_step_weight,
                "score_delta": delta,
            }
        )

    for index, (field_probs, action) in enumerate(zip(trajectory, action_plan, strict=False)):
        action_probs = field_probs.get("action_type", {})
        predicted_type = (
            max(action_probs.items(), key=lambda item: item[1])[0] if action_probs else "unknown"
        )
        read_prob = sum(float(action_probs.get(kind, 0.0)) for kind in ("read", "search"))
        if not action_probs:
            inferred = _inferred_action_types(action)
            read_prob = float(bool(inferred) and all(x in {"read", "search"} for x in inferred))

        sufficient_prob = float(
            field_probs.get("information_sufficiency", {}).get("sufficient", 0.0)
        )
        read_saturated = bool(threshold and len(reads) >= threshold)
        task_progressed = progress_actions > 0
        pressure = max(
            1.0 if read_saturated else 0.0,
            sufficient_prob,
            float(config.read_after_progress_scale) if task_progressed else 0.0,
        )
        if read_prob and penalty and pressure:
            delta = -(penalty * read_prob * pressure)
            record["score"] += delta
            record["per_step"][index]["score"] += delta
            record["per_step"][index]["contributions"]["read_after_saturation"] = delta
            record["field_contributions"]["read_after_saturation"] = (
                record["field_contributions"].get("read_after_saturation", 0.0)
                + delta
            )
            adjustments.append(
                {
                    "step": index + 1,
                    "kind": "read_after_saturation",
                    "read_probability": read_prob,
                    "distinct_prior_reads": len(reads),
                    "information_sufficient_probability": sufficient_prob,
                    "prior_progress_actions": progress_actions,
                    "penalty_pressure": pressure,
                    "score_delta": delta,
                }
            )

        if (
            config.exempt_task_required_side_effects
            and predicted_type == "delete"
            and task_requires_delete
        ):
            for veto in record["veto_reasons"]:
                if veto.get("step") == index + 1:
                    veto["reasons"] = [
                        reason
                        for reason in veto.get("reasons", [])
                        if not reason.startswith("P(deleted)=")
                    ]
            side_effect = float(
                record["per_step"][index]["contributions"].get("side_effect_type", 0.0)
            )
            neutralization = max(0.0, -side_effect)
            record["score"] += neutralization
            record["per_step"][index]["score"] += neutralization
            record["field_contributions"]["task_required_side_effect"] = (
                record["field_contributions"].get("task_required_side_effect", 0.0)
                + neutralization
            )
            adjustments.append(
                {
                    "step": index + 1,
                    "kind": "task_required_side_effect",
                    "action_type_probability": float(action_probs.get("delete", 0.0)),
                    "score_delta": neutralization,
                }
            )

        success_prob = float(field_probs.get("execution_status", {}).get("success", 0.0))
        if predicted_type in {"read", "search"} and success_prob >= 0.5:
            reads.add(_action_signature(action))
        elif predicted_type not in {"read", "search", "unknown"}:
            reads.clear()
            if success_prob >= 0.5:
                progress_actions += 1

    inferred_plan_types: list[str] = []
    for field_probs, action in zip(trajectory, action_plan, strict=False):
        action_probs = field_probs.get("action_type", {})
        if action_probs:
            predicted_type = max(action_probs.items(), key=lambda item: item[1])[0]
            if predicted_type != "unknown":
                inferred_plan_types.append(predicted_type)
                continue
        inferred_plan_types.extend(
            kind for kind in _inferred_action_types(action) if kind != "unknown"
        )
    covered_required_types = required_action_types.intersection(inferred_plan_types)
    coverage_bonus = max(0.0, float(config.required_action_coverage_bonus))
    if required_action_types and covered_required_types and coverage_bonus:
        coverage = len(covered_required_types) / len(required_action_types)
        delta = coverage_bonus * coverage
        record["score"] += delta
        record["field_contributions"]["required_action_coverage"] = delta
        adjustments.append(
            {
                "kind": "required_action_coverage",
                "required_action_types": sorted(required_action_types),
                "covered_action_types": sorted(covered_required_types),
                "coverage": coverage,
                "score_delta": delta,
            }
        )

    first_required_bonus = max(0.0, float(config.first_required_action_bonus))
    first_action_types: set[str] = set()
    if trajectory and trajectory[0].get("action_type"):
        first_probs = trajectory[0]["action_type"]
        first_action_types.add(max(first_probs.items(), key=lambda item: item[1])[0])
    elif action_plan:
        first_action_types.update(_inferred_action_types(action_plan[0]))
    first_required_types = required_action_types.intersection(first_action_types)
    if first_required_types and first_required_bonus:
        delta = first_required_bonus
        record["score"] += delta
        record["per_step"][0]["score"] += delta
        record["per_step"][0]["contributions"]["first_required_action"] = delta
        record["field_contributions"]["first_required_action"] = delta
        adjustments.append(
            {
                "step": 1,
                "kind": "first_required_action",
                "action_types": sorted(first_required_types),
                "score_delta": delta,
            }
        )

    record["veto_reasons"] = [veto for veto in record["veto_reasons"] if veto.get("reasons")]
    record["vetoed"] = bool(record["veto_reasons"])
    record["action_aware_adjustments"] = adjustments



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


def finalize_scored_plans(
    trajectories: list[list[dict[str, dict[str, float]]]],
    action_plans: list[list[Any]],
    config: CanonicalEventScoreConfig | None = None,
    *,
    task_text: str = "",
    input_history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Shared tail of every ``score_action_plans_canonical_event`` implementation
    (JEPA-latent or LLM-text): rank the already-computed per-step field-probability
    trajectories, then decode each step's argmax state and attach the raw per-step
    probabilities so callers (beam_plan's margin/critic logic) can read e.g.
    ``P(execution_status=failure)`` without re-deriving it from an argmax.

    Backend-agnostic by construction: everything here operates on plain
    ``{field: {category: prob}}`` dicts, never on how those probabilities were produced
    (classifier-head softmax, or an LLM's forced-choice next-token softmax).
    """
    if not action_plans:
        return []
    config = config or CanonicalEventScoreConfig()
    num_plans = len(action_plans)
    _, all_scored = rank_trajectories(trajectories, config, top_k=num_plans)
    for raw_rank, record in enumerate(all_scored, start=1):
        record["raw_score"] = float(record["score"])
        record["raw_normalized_score"] = float(record.get("normalized_score", 0.0))
        record["raw_rank"] = raw_rank
        record["raw_vetoed"] = bool(record["vetoed"])
        record["raw_veto_reasons"] = [
            {"step": veto.get("step"), "reasons": list(veto.get("reasons", []))}
            for veto in record.get("veto_reasons", [])
        ]
        record["plan"] = action_plans[record["index"]]
        record["plan_index"] = record["index"]
        steps = trajectories[record["index"]]

        def _decode(probs_by_field: dict[str, dict[str, float]]) -> dict[str, Any]:
            decoded: dict[str, Any] = {}
            for field_name, probs in probs_by_field.items():
                if not probs:
                    continue
                if field_name == MISSING_INFO_FIELD:
                    decoded[field_name] = [c for c, p in probs.items() if p >= 0.5] or ["none"]
                else:
                    decoded[field_name] = max(probs.items(), key=lambda kv: kv[1])[0]
            return decoded

        per_step_states = [_decode(step_probs) for step_probs in steps]
        record["per_step_predicted_state"] = per_step_states
        record["per_step_field_probs"] = steps
        record["predicted_state"] = per_step_states[-1] if per_step_states else {}
        _apply_action_aware_adjustment(
            record,
            steps,
            record["plan"],
            config,
            task_text,
            input_history,
        )
        record["reason"] = explain_trajectory(record)
    _add_normalized_scores(all_scored)
    all_scored.sort(key=lambda record: (record["vetoed"], -record["score"]))
    for adjusted_rank, record in enumerate(all_scored, start=1):
        record["adjusted_score"] = float(record["score"])
        record["adjusted_normalized_score"] = float(record.get("normalized_score", 0.0))
        record["adjusted_rank"] = adjusted_rank
        record["adjusted_vetoed"] = bool(record["vetoed"])
        record["action_aware_score_delta"] = record["adjusted_score"] - record["raw_score"]
    return all_scored


def logits_to_field_probs_batched(
    field_logits: dict[str, Any],
    vocab: dict[str, list[str]],
    *,
    temperature: float = 1.0,
    multi_label_fields: tuple[str, ...] = (MISSING_INFO_FIELD,),
) -> list[dict[str, dict[str, float]]]:
    """Batched form of :func:`logits_to_field_probs`: takes ``[B, C]`` logits per field and
    returns one ``{field: {category: prob}}`` dict per row.

    Why this exists: the per-row version reads probabilities out of CUDA tensors one scalar at
    a time (``float(per_class[i])``), and every one of those is a device synchronization. At
    B plans x 11 fields x ~10 classes that is hundreds of syncs per rollout step -- measured at
    9.3ms per step for B=7 vs. 2.8ms when the transfer happens once per field. The softmax/
    sigmoid still runs on the original device, so the numbers are bit-identical; only the host
    transfer is hoisted out of the inner loop.
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
        per_field_rows[name] = activated.cpu().tolist()  # one host transfer per field
        batch_size = max(batch_size, len(per_field_rows[name]))

    results: list[dict[str, dict[str, float]]] = []
    for row in range(batch_size):
        probs: dict[str, dict[str, float]] = {}
        for name, rows in per_field_rows.items():
            if row >= len(rows):
                continue
            categories = vocab[name]
            values = rows[row]
            probs[name] = {categories[i]: float(values[i]) for i in range(min(len(categories), len(values)))}
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


__all__ = [
    "CanonicalEventScoreConfig",
    "explain_trajectory",
    "finalize_scored_plans",
    "logits_to_field_probs",
    "logits_to_field_probs_batched",
    "rank_trajectories",
    "score_step",
    "score_trajectory",
    "step_veto",
]
