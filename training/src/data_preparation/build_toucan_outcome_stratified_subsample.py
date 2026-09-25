#!/usr/bin/env python3
"""Outcome-stratified TOUCAN-enterprise subsample with reasoning pseudo-tool steps collapsed.

Two problems with subsampling TOUCAN uniformly (what `toucan_enterprise_25k` does):

1. **36% of its tool steps are reasoning pseudo-tools** (`sequentialthinking`, `think`,
   `lotuswisdom`, ...). Those are MCP "thinking" servers, not enterprise operations: they
   always "succeed", carry no observation worth predicting, and still consume a training
   example each. They are collapsed here rather than deleted -- the action is relabelled to
   `assistant` and its state is merged forward into the previous tool action's state, the same
   treatment src/data_preparation/world_model_trajectory_cleanup.py gives pure-thought steps.
   The step stays in the conversation as context but stops producing a training example
   (extract_state_examples only emits for `action`->`state` pairs carrying tool calls).

2. **Uniform sampling reproduces TOUCAN's success skew** (~4% explicit failure after
   collapsing, versus ~15% in EnterpriseOps-Gym, the benchmark the classification heads are
   evaluated on). The world model's outcome signal and the canonical-event `execution_status` /
   `progress_signal` heads are limited by exactly those minority steps, so a bigger uniform
   sample buys tokens without buying signal. Selection here takes every failure-bearing
   trajectory first, then dilutes with the rest until the step-level failure share hits
   `--target-failure-share`.

Whole trajectories are selected, never individual steps, so no successor chain is broken.

The failure share and the subsample size trade off against each other -- the corpus contains a
fixed number of failure steps, so a higher share means a smaller sample. `--report-only` prints
that frontier without writing anything.

    uv run python src/data_preparation/build_toucan_outcome_stratified_subsample.py --report-only
    uv run python src/data_preparation/build_toucan_outcome_stratified_subsample.py \
        --target-failure-share 0.15 --output trajectories/toucan_enterprise_failure15_world_model_trajectories.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_preparation.world_model_trajectory_cleanup import (  # noqa: E402
    merge_followup_state_into_tool_state,
)

DEFAULT_SOURCE = ROOT / "trajectories" / "toucan_enterprise_world_model_trajectories.jsonl"
# Substrings identifying MCP reasoning servers rather than enterprise tools. Matched on the
# lowercased tool name; a step is collapsed only when EVERY call in it matches, so a step that
# mixes thinking with a real tool call is kept.
REASONING_TOOL_MARKERS = (
    "sequentialthinking", "sequential-thinking", "sequential_thinking",
    "think", "lotuswisdom", "lotus-wisdom", "reasoning", "wisdom",
)


def is_reasoning_tool(name: str) -> bool:
    return any(marker in name.lower() for marker in REASONING_TOOL_MARKERS)


def step_tool_names(content: Any) -> list[str]:
    if not isinstance(content, dict):
        return []
    names: list[str] = []
    for call in content.get("tool_calls") or []:
        function = (call.get("function") if isinstance(call, dict) else None) or call or {}
        name = function.get("name") if isinstance(function, dict) else None
        if name:
            names.append(str(name))
    return names


def collapse_reasoning_steps(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Relabel reasoning-only tool actions to `assistant` and merge their state forward.

    Returns (messages, collapsed_count). Mirrors cleanup_world_model_messages: the state that
    followed a collapsed action is folded into the most recent REAL tool action's state, so
    stage/temporal progress recorded during the reasoning step is not lost.
    """
    output: list[dict[str, Any]] = []
    last_tool_state_index: int | None = None
    collapsed = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        content = message.get("content")
        is_tool_action = message.get("role") == "action" and isinstance(content, dict) and content.get("tool_calls")
        if is_tool_action:
            names = step_tool_names(content)
            if names and all(is_reasoning_tool(name) for name in names):
                follower = messages[index + 1] if index + 1 < len(messages) else None
                has_state = bool(follower and follower.get("role") == "state")
                # Keep the reasoning as context, but as `assistant` so it yields no example.
                output.append({"role": "assistant", "content": json.dumps(
                    {"thought": f"used reasoning tool {', '.join(names)}"}, ensure_ascii=False)})
                if has_state and last_tool_state_index is not None:
                    output[last_tool_state_index] = merge_followup_state_into_tool_state(
                        output[last_tool_state_index], follower
                    )
                collapsed += 1
                index += 2 if has_state else 1
                continue
            output.append(message)
            follower = messages[index + 1] if index + 1 < len(messages) else None
            if follower and follower.get("role") == "state":
                output.append(follower)
                last_tool_state_index = len(output) - 1
                index += 2
            else:
                index += 1
            continue
        output.append(message)
        index += 1
    return output, collapsed


def trajectory_outcomes(messages: list[dict[str, Any]]) -> Counter:
    """Step-level outcome counts over REAL tool actions (post-collapse messages)."""
    counts: Counter = Counter()
    for index, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") != "action" or not isinstance(content, dict) or not content.get("tool_calls"):
            continue
        follower = messages[index + 1] if index + 1 < len(messages) else None
        if not (follower and follower.get("role") == "state" and isinstance(follower.get("content"), dict)):
            counts[None] += 1
            continue
        state = follower["content"].get("state") or follower["content"]
        counts[(state.get("context") or {}).get("last_tool_execution_result")] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL (default: derived from --target-failure-share).")
    parser.add_argument("--target-failure-share", type=float, default=0.15,
                        help="Desired step-level share of explicit failures (-1). "
                             "EnterpriseOps-Gym, the eval benchmark, sits at 0.15.")
    parser.add_argument("--max-trajectories", type=int, default=0,
                        help="Hard cap on selected trajectories (0 = only the share constrains it).")
    parser.add_argument("--keep-reasoning-steps", action="store_true",
                        help="Do not collapse reasoning pseudo-tool steps (for an A/B).")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--report-only", action="store_true",
                        help="Scan and print the achievable size/failure-share frontier, write nothing.")
    parser.add_argument("--scan-limit", type=int, default=0,
                        help="Only scan the first N lines (debug).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise SystemExit(f"missing source: {args.source}")

    # Pass 1: byte offsets + per-trajectory outcome counts. Only metadata is held in memory
    # (~131k rows), never the 8 GB of trajectories.
    records: list[dict[str, Any]] = []
    totals: Counter = Counter()
    collapsed_total = 0
    steps_before = 0
    print(f"[scan] {args.source}", flush=True)
    with args.source.open("rb") as handle:
        offset = handle.tell()
        line = handle.readline()
        scanned = 0
        while line:
            if len(line) > 60:
                try:
                    trajectory = json.loads(line)
                except json.JSONDecodeError:
                    trajectory = None
                if trajectory is not None:
                    messages = trajectory.get("messages") or []
                    steps_before += sum(
                        1 for m in messages
                        if m.get("role") == "action" and isinstance(m.get("content"), dict)
                        and m["content"].get("tool_calls")
                    )
                    if args.keep_reasoning_steps:
                        kept_messages, collapsed = messages, 0
                    else:
                        kept_messages, collapsed = collapse_reasoning_steps(messages)
                    collapsed_total += collapsed
                    counts = trajectory_outcomes(kept_messages)
                    steps = sum(counts.values())
                    if steps:
                        records.append({
                            "offset": offset, "length": len(line), "steps": steps,
                            "fail": counts.get(-1, 0), "stag": counts.get(0, 0),
                            "succ": counts.get(1, 0),
                        })
                        totals.update(counts)
                    scanned += 1
                    if scanned % 20000 == 0:
                        print(f"  scanned {scanned} trajectories", flush=True)
            if args.scan_limit and scanned >= args.scan_limit:
                break
            offset = handle.tell()
            line = handle.readline()

    steps_total = sum(record["steps"] for record in records)
    fail_total = sum(record["fail"] for record in records)
    with_fail = [record for record in records if record["fail"]]
    without_fail = [record for record in records if not record["fail"]]
    print(f"\n[corpus] {len(records)} usable trajectories, {steps_total} steps after "
          f"{'keeping' if args.keep_reasoning_steps else 'collapsing'} reasoning steps "
          f"({steps_before} before, {collapsed_total} collapsed)")
    denominator = max(1, steps_total)
    print(f"[corpus] outcome mix: " + "  ".join(
        f"{key}={100 * value / denominator:.2f}%" for key, value in sorted(totals.items(), key=lambda kv: str(kv[0]))))
    print(f"[corpus] failure-bearing trajectories: {len(with_fail)} ({100 * len(with_fail) / max(1, len(records)):.1f}%), "
          f"{fail_total} failure steps")

    other_steps_in_fail = sum(record["steps"] - record["fail"] for record in with_fail)
    mean_steps_other = (sum(record["steps"] for record in without_fail) / len(without_fail)) if without_fail else 0.0
    print("\n[frontier] whole-trajectory selection, all failure-bearing trajectories taken first:")
    for share in (0.20, 0.15, 0.10, 0.05, 0.03):
        needed_total = fail_total / share
        extra_steps = needed_total - fail_total - other_steps_in_fail
        extra_traj = max(0.0, extra_steps / mean_steps_other) if mean_steps_other else 0.0
        reachable = extra_steps >= 0
        print(f"   failure {share:.0%}: {len(with_fail) + extra_traj:>9,.0f} trajectories, "
              f"{max(needed_total, fail_total + other_steps_in_fail):>9,.0f} steps"
              + ("" if reachable else "   <- share too high: taking ONLY failure trajectories gives "
                                      f"{100 * fail_total / max(1, fail_total + other_steps_in_fail):.1f}%"))
    if args.report_only:
        return

    # Selection: every failure-bearing trajectory, then dilute to the target share.
    rng = random.Random(args.seed)
    rng.shuffle(without_fail)
    selected = list(with_fail)
    steps_selected = sum(record["steps"] for record in selected)
    fail_selected = sum(record["fail"] for record in selected)
    for record in without_fail:
        if fail_selected / max(1, steps_selected) <= args.target_failure_share:
            break
        if args.max_trajectories and len(selected) >= args.max_trajectories:
            break
        selected.append(record)
        steps_selected += record["steps"]
    if args.max_trajectories and len(selected) > args.max_trajectories:
        rng.shuffle(selected)
        selected = selected[: args.max_trajectories]

    share = args.target_failure_share
    destination = args.output or (
        args.source.with_name(f"toucan_enterprise_failure{int(round(share * 100))}_world_model_trajectories.jsonl")
    )
    if destination.exists() and not args.overwrite:
        raise SystemExit(f"{destination} exists; pass --overwrite")

    rng.shuffle(selected)
    written = Counter()
    written_steps = 0
    with args.source.open("rb") as reader, destination.open("w", encoding="utf-8") as writer:
        for record in selected:
            reader.seek(record["offset"])
            trajectory = json.loads(reader.read(record["length"]))
            messages = trajectory.get("messages") or []
            if not args.keep_reasoning_steps:
                messages, _ = collapse_reasoning_steps(messages)
            trajectory["messages"] = messages
            written.update(trajectory_outcomes(messages))
            written_steps += sum(trajectory_outcomes(messages).values())
            writer.write(json.dumps(trajectory, ensure_ascii=False) + "\n")

    denominator = max(1, sum(written.values()))
    manifest = {
        "source": str(args.source),
        "output": str(destination),
        "target_failure_share": share,
        "collapse_reasoning_steps": not args.keep_reasoning_steps,
        "seed": args.seed,
        "trajectories": len(selected),
        "steps": sum(written.values()),
        "outcome_mix": {str(key): value for key, value in written.items()},
        "outcome_share": {str(key): value / denominator for key, value in written.items()},
        "corpus_trajectories_scanned": len(records),
        "corpus_failure_steps": fail_total,
        "reasoning_steps_collapsed_in_corpus": collapsed_total,
    }
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[write] {destination.name}: {len(selected)} trajectories, {sum(written.values())} steps")
    print("[write] outcome mix: " + "  ".join(
        f"{key}={100 * value / denominator:.2f}%" for key, value in sorted(written.items(), key=lambda kv: str(kv[0]))))
    print(f"[write] manifest: {manifest_path.name}")


if __name__ == "__main__":
    main()
