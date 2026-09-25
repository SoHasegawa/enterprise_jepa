#!/usr/bin/env python3
"""Week 1 -- build the EXECUTED-counterfactual evaluation set.

Why this exists: week1_decision_ranking.py ranks the demonstrated action against SYNTHETIC
distractors (wrong_tool / wrong_args / redundant). Those are assumed-wrong; none was ever run,
so a distractor may in fact be a perfectly good alternative and "the model ranked it first" is
not necessarily an error. Every oracle rung above the deployed scorer -- "true next latent",
"gold labels" -- also needs an outcome for the counterfactual action, which synthetic
distractors by construction do not have.

The source of real counterfactuals here is CROSS-MODEL DIVERGENCE on identical tasks: 396
EnterpriseOps-Gym oracle tasks were run by both gpt-5 and qwen3. The gym resets its databases
per task, and the environment is deterministic given the executed action sequence, so:

    same task  +  identical executed action prefix  =>  identical environment state

At the first step where the two runs' actions differ, we therefore have two DIFFERENT actions
taken from the SAME state, each with its real observation, its real per-call success flag, and
the real task-level verifier outcome of the branch it belongs to. That is an executed
counterfactual, obtained with no live gym and no re-execution.

Preference label (which action a planner should have ranked higher), in priority order:
  1. tool_success   -- one call succeeded and the other failed
  2. verifier_rate  -- both succeeded/failed alike, but the branches' task-level verifier pass
                       rates differ (weaker: attributes a whole-run outcome to one step)
  3. none           -- undecidable; kept in the set but excluded from ranking metrics

Pairs are emitted with `preference_strength` so downstream metrics can report rung-1-only and
rung-1+2 numbers separately rather than silently mixing a strong and a weak label.

Usage:
  uv run python src/analysis/week1_executed_counterfactuals.py
  uv run python src/analysis/week1_executed_counterfactuals.py --max-prefix-divergences 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GYM_RESULTS = Path("/data/user/enterprisegym/results/react")
DEFAULT_MODELS = ("gpt-5", "qwen3")
DEFAULT_OUT = REPO_ROOT / "data" / "week1" / "executed_counterfactuals.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=GYM_RESULTS)
    p.add_argument("--models", nargs=2, default=list(DEFAULT_MODELS))
    p.add_argument("--split", default="teams/oracle/run_1",
                   help="Sub-path under <results-root>/<model>/ holding the per-task JSON files.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-prefix-divergences", type=int, default=1,
                   help="How many divergence points to emit per task. 1 = only the first "
                        "(strictly state-matched). >1 emits later divergences too, which are "
                        "matched only up to the shared prefix and are flagged accordingly.")
    p.add_argument("--max-observation-chars", type=int, default=4000)
    return p.parse_args()


def task_id_from_filename(name: str) -> str:
    return re.sub(r"^results_oracle__", "", name).rsplit(".json", 1)[0]


def normalize_arguments(arguments: Any) -> str:
    """Canonical form for prefix comparison: key order must not create a false divergence."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments.strip()
    if isinstance(arguments, dict):
        return json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    return json.dumps(arguments, ensure_ascii=False)


# The gym's MCP envelope reports `success: true` for any call that completed the round trip,
# so it is a TRANSPORT flag, not a semantic one -- it was True for every observation sampled.
# Real failures appear in the observation payload as `API Error [tool]: ...`, which is also how
# this repo renders gym failures into state text (see finetuning_echo.py). Classify on that,
# plus the repo's existing explicit-failure markers, so the counterfactual labels agree with how
# the training data was labeled in the first place.
GYM_API_ERROR = re.compile(r"\bAPI\s+Error\b", re.I)
EXPLICIT_FAILURE_MARKERS = (
    "an error occurred when calling tool",
    "mcperror",
    "traceback (most recent call last)",
    "exception:",
    "error:",
    "failed:",
)


def observation_is_failure(text: str) -> bool:
    if not text:
        return False
    if GYM_API_ERROR.search(text):
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in EXPLICIT_FAILURE_MARKERS)


def observation_text(result: Any, limit: int) -> str:
    """Flatten the gym's nested MCP result envelope to the text the agent would have seen."""
    if isinstance(result, dict):
        inner = result.get("result")
        if isinstance(inner, dict) and isinstance(inner.get("content"), list):
            parts = [
                c.get("text", "")
                for c in inner["content"]
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            if parts:
                return "\n".join(parts)[:limit]
    return json.dumps(result, ensure_ascii=False)[:limit]


def extract_steps(run: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    """One entry per executed tool call: name, canonical args, observation, success."""
    steps = []
    for entry in run.get("tool_results") or []:
        if not isinstance(entry, dict):
            continue
        result = entry.get("result") or {}
        observation = observation_text(result, limit)
        transport_ok = bool(result.get("success")) if isinstance(result, dict) else None
        steps.append({
            "tool_name": entry.get("tool_name", ""),
            "arguments": entry.get("arguments"),
            "arguments_key": normalize_arguments(entry.get("arguments")),
            "observation": observation,
            "transport_success": transport_ok,
            # Semantic outcome: the call round-tripped but the API may still have rejected it.
            "tool_success": (False if observation_is_failure(observation)
                             else (transport_ok if transport_ok is not None else None)),
        })
    return steps


def verifier_rate(run: dict[str, Any], stats: dict[str, Any]) -> float | None:
    summary = run.get("verification_summary")
    if isinstance(summary, dict):
        total = summary.get("total") or summary.get("total_verifiers")
        passed = summary.get("passed") or summary.get("passed_verifiers")
        if isinstance(total, (int, float)) and total:
            return float(passed or 0) / float(total)
    results = run.get("verification_results")
    if isinstance(results, list) and results:
        passed = sum(1 for r in results if isinstance(r, dict) and r.get("passed"))
        return passed / len(results)
    rate = stats.get("verifier_level_pass_rate")
    return float(rate) if isinstance(rate, (int, float)) else None


def load_task(path: Path, limit: int) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    runs = payload.get("runs") or []
    if not runs:
        return None
    run = runs[0]
    config = payload.get("benchmark_config") or {}
    flow = run.get("conversation_flow") or []
    system_prompt = next((e.get("content", "") for e in flow if e.get("type") == "system_message"), "")
    user_prompt = next((e.get("content", "") for e in flow if e.get("type") == "user_message"),
                       config.get("user_prompt", ""))
    return {
        "model": config.get("model", ""),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "steps": extract_steps(run, limit),
        "overall_success": bool(run.get("overall_success")),
        "verifier_rate": verifier_rate(run, payload.get("statistics") or {}),
    }


def decide_preference(a: dict[str, Any], b: dict[str, Any],
                      ta: dict[str, Any], tb: dict[str, Any]) -> tuple[str | None, str]:
    """(preferred branch, strength). Strong when the two calls differ in execution success."""
    if a["tool_success"] is not None and b["tool_success"] is not None and a["tool_success"] != b["tool_success"]:
        return ("a" if a["tool_success"] else "b"), "tool_success"
    ra, rb = ta.get("verifier_rate"), tb.get("verifier_rate")
    if isinstance(ra, float) and isinstance(rb, float) and abs(ra - rb) > 1e-9:
        return ("a" if ra > rb else "b"), "verifier_rate"
    return None, "none"


def main() -> None:
    args = parse_args()
    model_a, model_b = args.models
    dir_a = args.results_root / model_a / args.split
    dir_b = args.results_root / model_b / args.split
    for d in (dir_a, dir_b):
        if not d.exists():
            raise SystemExit(f"results dir not found: {d}")

    ids_a = {task_id_from_filename(p.name): p for p in dir_a.glob("*.json")}
    ids_b = {task_id_from_filename(p.name): p for p in dir_b.glob("*.json")}
    shared = sorted(set(ids_a) & set(ids_b))
    print(f"{model_a}={len(ids_a)} tasks  {model_b}={len(ids_b)} tasks  shared={len(shared)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    strengths = Counter()
    prefix_lengths = Counter()
    rows: list[dict[str, Any]] = []

    for task_id in shared:
        ta = load_task(ids_a[task_id], args.max_observation_chars)
        tb = load_task(ids_b[task_id], args.max_observation_chars)
        if ta is None or tb is None:
            stats["unloadable"] += 1
            continue
        sa, sb = ta["steps"], tb["steps"]
        if not sa or not sb:
            stats["no_tool_calls"] += 1
            continue

        emitted = 0
        cursor = 0
        while cursor < min(len(sa), len(sb)) and emitted < args.max_prefix_divergences:
            same = (sa[cursor]["tool_name"] == sb[cursor]["tool_name"]
                    and sa[cursor]["arguments_key"] == sb[cursor]["arguments_key"])
            if same:
                cursor += 1
                continue
            preferred, strength = decide_preference(sa[cursor], sb[cursor], ta, tb)
            rows.append({
                "task_id": task_id,
                "prefix_len": cursor,
                # Only the FIRST divergence is strictly state-matched; later ones share just
                # the common prefix, because the branches already differ upstream.
                "state_matched": emitted == 0,
                "system_prompt": ta["system_prompt"],
                "user_prompt": ta["user_prompt"],
                "prefix": [
                    {"tool_name": s["tool_name"], "arguments": s["arguments"], "observation": s["observation"]}
                    for s in sa[:cursor]
                ],
                "branch_a": {
                    "model": ta["model"], "tool_name": sa[cursor]["tool_name"],
                    "arguments": sa[cursor]["arguments"], "observation": sa[cursor]["observation"],
                    "tool_success": sa[cursor]["tool_success"],
                    "task_verifier_rate": ta["verifier_rate"], "task_success": ta["overall_success"],
                },
                "branch_b": {
                    "model": tb["model"], "tool_name": sb[cursor]["tool_name"],
                    "arguments": sb[cursor]["arguments"], "observation": sb[cursor]["observation"],
                    "tool_success": sb[cursor]["tool_success"],
                    "task_verifier_rate": tb["verifier_rate"], "task_success": tb["overall_success"],
                },
                "preferred": preferred,
                "preference_strength": strength,
                "same_tool_different_args": sa[cursor]["tool_name"] == sb[cursor]["tool_name"],
            })
            strengths[strength] += 1
            prefix_lengths[cursor] += 1
            stats["pairs"] += 1
            emitted += 1
            cursor += 1
        if emitted == 0:
            stats["no_divergence"] += 1

    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    decisive = strengths["tool_success"] + strengths["verifier_rate"]
    print(f"\nexecuted-counterfactual pairs: {stats['pairs']}")
    print(f"  tasks with no divergence : {stats['no_divergence']}")
    print(f"  tasks with no tool calls : {stats['no_tool_calls']}")
    print(f"\npreference label strength:")
    for k in ("tool_success", "verifier_rate", "none"):
        print(f"  {k:14s} {strengths[k]:5d}")
    print(f"  decisive (usable for ranking metrics): {decisive}")
    same_tool = sum(1 for r in rows if r["same_tool_different_args"])
    print(f"\nsame tool / different arguments: {same_tool} "
          f"({same_tool / len(rows):.1%} of pairs)" if rows else "")
    print(f"state-matched (first divergence): {sum(1 for r in rows if r['state_matched'])}")
    print(f"divergence step distribution (top): "
          f"{dict(sorted(prefix_lengths.items())[:8])}")
    print(f"\nwrote {len(rows)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
