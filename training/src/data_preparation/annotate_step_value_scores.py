#!/usr/bin/env python3
"""Annotate per-step value-head training targets for canonical_event_with_nudge examples.

Motivation: the canonical_event_state/nudge classification heads already predict *what
happened* at a step (a categorical schema). A value head is meant to predict a scalar
*how good was it* from the same latent vectors -- but nothing in the data currently
supplies that scalar. This script derives it from two ingredients the user specified:

  1. The annotated canonical_event_state/nudge fields already on each record (a per-step
     shaping signal -- was THIS action locally good: did it succeed, make progress, avoid
     risk, leave the agent well-informed).
  2. Whether the trajectory as a whole succeeded (a sparse, trajectory-level outcome that
     the per-step fields cannot see -- a step can look locally fine yet be on a trajectory
     that never accomplishes the task).

Design (see the accompanying write-up in the conversation this script was requested from):

  * Immediate per-step reward. Reuses `src.canonical_event_scoring.score_step` UNCHANGED --
    the same weights/utilities that already score imagined rollouts for beam_plan and
    hier_latent_cem (src/hierarchical_action_sampling.py). Hard labels are lifted into the
    same {field: {category: probability}} shape that function expects by putting all mass
    on the annotated category (a one-hot "probability"), so scoring an annotated example and
    scoring a head's soft prediction are the IDENTICAL computation. This means a value head
    trained on these labels predicts the same quantity the planner already computes online --
    not a second, inconsistent notion of "good."
  * Terminal outcome reward. Added ONLY to the trajectory's last step, from a real verifier
    outcome when available (see below), else a same-file heuristic. Scaled by
    `--terminal-reward-scale` and by the fraction of individual verifiers passed (graded,
    not a step function), so "2/3 checks passed" outranks "0/3 passed" even though both are
    "not fully successful".
  * Per-step regression target = discounted return-to-go: value_target[t] = sum_{k=t}^{T}
    gamma^(k-t) * reward[k], computed with a standard backward pass. `gamma` defaults to
    0.9, the same discount already used by CanonicalEventScoreConfig for trajectory-level
    scoring -- so a value head trained on this target and the planner's discounting share
    one horizon-decay assumption instead of two independently-chosen ones.

Trajectory outcome sources (in priority order, recorded per-trajectory as `success_source`):
  1. "verifier" -- EnterpriseOps-Gym task_key (parsed out of `trajectory_id`) is looked up
     under `--oracle-results-root` (default: the oracle react-agent run results this repo's
     CLAUDE.md documents at /data/user/enterprisegym/results/react/{model}/{domain}/
     oracle/run_1/results_oracle__{domain}__{task_key}.json) for a REAL, verifier-checked
     `overall_success` / `verification_summary.pass_rate`. Multiple models ran the same
     task_key; if more than one match is found (and they disagree), the mean pass_rate is
     used and `success_source` is suffixed "_disputed" so disagreement is visible rather
     than silently averaged away. This ONLY resolves EnterpriseOps-Gym trajectories --
     Terminal-Bench-2.0/CRMArenaPro trajectory_ids don't carry a matching task_key.
  2. "heuristic_last_step" -- fallback when no verifier result is found: the trajectory's
     OWN final step's execution_status/progress_signal/error_signature. This is markedly
     weaker evidence (it reflects whether the last ACTION succeeded, not whether the task
     goal was achieved) and is flagged as such in the output so it can be filtered out or
     down-weighted later if the distinction matters for training.

Output: one JSONL per input, `<stem>_value_scored.jsonl`, each record augmented with
`step_score` (this step's immediate reward alone), `value_target` (the discounted
return-to-go -- the actual regression label), `trajectory_success`, `trajectory_pass_rate`,
and `success_source`. All original fields are preserved unchanged.

Usage:

    uv run python src/data_preparation/annotate_step_value_scores.py
    # or explicit paths / oracle root:
    uv run python src/data_preparation/annotate_step_value_scores.py \
        --train-path trajectories/..._train_examples_cleaned.jsonl \
        --eval-path trajectories/..._eval_examples_cleaned.jsonl \
        --oracle-results-root /data/user/enterprisegym/results/react
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.canonical_event_scoring import CanonicalEventScoreConfig, MISSING_INFO_FIELD, SCORED_SINGLE_FIELDS, score_step

TRAJECTORIES_DIR = REPO_ROOT / "trajectories"
DEFAULT_TRAIN = TRAJECTORIES_DIR / (
    "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro_train_examples_cleaned.jsonl"
)
DEFAULT_EVAL = TRAJECTORIES_DIR / (
    "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro_eval_examples_cleaned.jsonl"
)
DEFAULT_ORACLE_RESULTS_ROOT = Path("/data/user/enterprisegym/results/react")

# e.g. "task_20251117_165528_648_bca89e7d_3e81ece9" embedded at the tail of trajectory_id.
TASK_KEY_PATTERN = re.compile(r"task_\d{8}_\d{6}_\d{3}_[0-9a-f]{8}_[0-9a-f]{8}")

# Heuristic fallback: what counts as a "good" final step when no verifier result exists.
HEURISTIC_GOOD_EXECUTION_STATUS = {"success", "no_op"}
HEURISTIC_BAD_ERROR_SIGNATURES = {"none", "unknown"}  # anything else = a concrete error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL)
    parser.add_argument(
        "--oracle-results-root",
        type=Path,
        default=DEFAULT_ORACLE_RESULTS_ROOT,
        help="Root of EnterpriseOps-Gym oracle react-agent run results (see module docstring). "
        "Pass a nonexistent path to force every trajectory onto the heuristic fallback.",
    )
    parser.add_argument("--gamma", type=float, default=None, help="Discount for return-to-go. Defaults to CanonicalEventScoreConfig.gamma (0.9).")
    parser.add_argument(
        "--terminal-reward-scale",
        type=float,
        default=5.0,
        help="Magnitude of the terminal outcome reward (mapped to [-scale, +scale] by pass_rate), "
        "added to the trajectory's LAST step only. Per-step canonical scores are bounded by "
        "roughly the sum of CanonicalEventScoreConfig.weights (~4.2 by default), so this should "
        "comfortably exceed that to make task outcome the dominant signal at the terminal step.",
    )
    parser.add_argument("--output-suffix", default="_value_scored")
    return parser.parse_args()


def load_lines(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def hard_field_probs(event: dict[str, Any], nudge: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Lift hard annotated labels into the {field: {category: probability}} shape
    `score_step` expects, by putting all probability mass on the annotated category."""
    probs: dict[str, dict[str, float]] = {}
    for field in SCORED_SINGLE_FIELDS:
        # information_sufficiency/recommended_abstract_action live under nudge, the rest under event.
        value = nudge.get(field) if field in ("information_sufficiency", "recommended_abstract_action") else event.get(field)
        if value is not None:
            probs[field] = {str(value): 1.0}
    missing = nudge.get(MISSING_INFO_FIELD)
    if isinstance(missing, list) and missing:
        probs[MISSING_INFO_FIELD] = {str(category): 1.0 for category in missing}
    return probs


def label_of(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    label = record.get("canonical_event_with_nudge") or {}
    event = label.get("canonical_event_state") or record.get("canonical_event_state") or {}
    nudge = label.get("nudge") or record.get("nudge") or {}
    return (event if isinstance(event, dict) else {}), (nudge if isinstance(nudge, dict) else {})


def extract_task_key(trajectory_id: str) -> str | None:
    matches = TASK_KEY_PATTERN.findall(trajectory_id or "")
    return matches[-1] if matches else None


class OracleResultsIndex:
    """Lazily globs and caches EnterpriseOps-Gym oracle results by task_key, so the
    (possibly large) results tree is scanned once per distinct task_key rather than once
    per trajectory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, list[Path]] = {}
        self._available = root.is_dir()

    def find(self, task_key: str) -> list[Path]:
        if not self._available:
            return []
        if task_key not in self._cache:
            self._cache[task_key] = sorted(self.root.glob(f"*/*/oracle/run_1/*{task_key}.json"))
        return self._cache[task_key]

    def outcome(self, task_key: str) -> dict[str, Any] | None:
        """Returns {"pass_rate": float, "overall_success": bool, "num_matches": int,
        "disputed": bool} aggregated (mean) over every model's run(s) for this task_key,
        or None if nothing matched."""
        paths = self.find(task_key)
        if not paths:
            return None
        pass_rates: list[float] = []
        successes: list[bool] = []
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for run in payload.get("runs") or []:
                summary = run.get("verification_summary") or {}
                if "pass_rate" in summary:
                    pass_rates.append(float(summary["pass_rate"]))
                if "overall_success" in run:
                    successes.append(bool(run["overall_success"]))
        if not pass_rates and not successes:
            return None
        pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else (sum(successes) / len(successes))
        overall_success = (sum(successes) / len(successes) >= 0.5) if successes else (pass_rate >= 1.0)
        disputed = (len(set(pass_rates)) > 1) if len(pass_rates) > 1 else (len(set(successes)) > 1)
        return {
            "pass_rate": pass_rate,
            "overall_success": overall_success,
            "num_matches": len(paths),
            "disputed": disputed,
        }


def heuristic_outcome(final_event: dict[str, Any]) -> dict[str, Any]:
    """Weak fallback when no verifier result exists: was the trajectory's own last action
    clean (succeeded, no concrete error)? This says nothing about whether the task's actual
    goal was reached -- only that the trajectory didn't end on a visible failure."""
    execution_status = str(final_event.get("execution_status", "unknown"))
    error_signature = str(final_event.get("error_signature", "unknown"))
    good = execution_status in HEURISTIC_GOOD_EXECUTION_STATUS and error_signature in HEURISTIC_BAD_ERROR_SIGNATURES
    return {"pass_rate": 1.0 if good else 0.0, "overall_success": good, "num_matches": 0, "disputed": False}


def annotate_split(
    records: list[dict[str, Any]],
    *,
    oracle_index: OracleResultsIndex,
    score_config: CanonicalEventScoreConfig,
    gamma: float,
    terminal_reward_scale: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_trajectory: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_trajectory[record.get("trajectory_id")].append(index)

    stats = {"trajectories": len(by_trajectory), "success_source_counts": defaultdict(int), "disputed_trajectories": 0}
    annotated = list(records)  # shallow copy of the list; we replace entries below

    for trajectory_id, indices in by_trajectory.items():
        indices = sorted(indices, key=lambda i: records[i].get("interaction_index", 0))
        step_events = [label_of(records[i]) for i in indices]

        immediate_rewards = [score_step(hard_field_probs(event, nudge), score_config)[0] for event, nudge in step_events]
        contributions = [score_step(hard_field_probs(event, nudge), score_config)[1] for event, nudge in step_events]

        task_key = extract_task_key(trajectory_id)
        outcome = oracle_index.outcome(task_key) if task_key else None
        success_source = "verifier"
        if outcome is None:
            outcome = heuristic_outcome(step_events[-1][0])
            success_source = "heuristic_last_step"
        elif outcome["disputed"]:
            success_source = "verifier_disputed"
            stats["disputed_trajectories"] += 1
        stats["success_source_counts"][success_source] += 1

        terminal_reward = terminal_reward_scale * (2.0 * outcome["pass_rate"] - 1.0)
        rewards = list(immediate_rewards)
        rewards[-1] += terminal_reward

        # Backward pass: value_target[t] = reward[t] + gamma * value_target[t+1].
        value_targets = [0.0] * len(rewards)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = rewards[t] + gamma * running
            value_targets[t] = running

        for position, record_index in enumerate(indices):
            record = dict(records[record_index])
            record["step_score"] = immediate_rewards[position]
            record["step_score_contributions"] = contributions[position]
            record["value_target"] = value_targets[position]
            record["trajectory_success"] = outcome["overall_success"]
            record["trajectory_pass_rate"] = outcome["pass_rate"]
            record["success_source"] = success_source
            annotated[record_index] = record

    stats["success_source_counts"] = dict(stats["success_source_counts"])
    return annotated, stats


def output_path(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix + path.suffix)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    score_config = CanonicalEventScoreConfig()
    gamma = args.gamma if args.gamma is not None else score_config.gamma
    oracle_index = OracleResultsIndex(args.oracle_results_root)
    if not oracle_index._available:
        print(f"[warn] --oracle-results-root {args.oracle_results_root} not found; every trajectory falls back to the heuristic outcome.")

    manifest: dict[str, Any] = {"gamma": gamma, "terminal_reward_scale": args.terminal_reward_scale, "oracle_results_root": str(args.oracle_results_root)}
    for split, path in (("train", args.train_path), ("eval", args.eval_path)):
        records = load_lines(path)
        annotated, stats = annotate_split(
            records,
            oracle_index=oracle_index,
            score_config=score_config,
            gamma=gamma,
            terminal_reward_scale=args.terminal_reward_scale,
        )
        out_path = output_path(path, args.output_suffix)
        write_jsonl(out_path, annotated)
        value_targets = [record["value_target"] for record in annotated]
        stats["value_target_min"] = min(value_targets)
        stats["value_target_max"] = max(value_targets)
        stats["value_target_mean"] = sum(value_targets) / len(value_targets)
        manifest[split] = {"input": str(path), "output": str(out_path), "records": len(records), **stats}
        print(
            f"[{split}] {len(records)} records over {stats['trajectories']} trajectories -> {out_path}\n"
            f"  success_source: {stats['success_source_counts']} (disputed: {stats['disputed_trajectories']})\n"
            f"  value_target: min={stats['value_target_min']:.2f} mean={stats['value_target_mean']:.2f} max={stats['value_target_max']:.2f}"
        )

    manifest_path = TRAJECTORIES_DIR / "canonical_event_with_nudge_step_value_scores_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
