#!/usr/bin/env python3
"""Count action-state pairs across world-model trajectory files.

For each (action, state) pair extracted by `extract_state_examples`:
  - `action` is the pipe-joined tool names emitted by the action's `tool_calls`.
  - `state` is the tool-execution-result label, comma-separated with the error
    message when the label is `0` (stagnation) or `-1` (failure). Format
    matches `format_tool_execution_result_target(..., include_error_message=True)`,
    so success rows look like `1` and failure rows look like `-1,API Error: ...`.

The output JSON is written to `trajectories/enterpriseops_gym_world_model_action_state_pair_counts.json`
by default and contains both fine-grained action-state counts and an
action-label rollup.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finetuning import (  # noqa: E402
    extract_state_examples,
    format_tool_execution_result_target,
    normalize_last_tool_execution_result,
)


DEFAULT_INPUTS = [
    ROOT / "trajectories" / "enterpriseops_gym_multi_model_world_model_train_trajectories.json",
    ROOT / "trajectories" / "enterpriseops_gym_multi_model_world_model_test_trajectories.json",
]
DEFAULT_OUTPUT = ROOT / "trajectories" / "enterpriseops_gym_world_model_action_state_pair_counts.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help=(
            "Trajectory JSON path. Repeatable; defaults to "
            "EnterpriseOps-Gym multi-model train/test trajectories."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the counter JSON.",
    )
    return parser.parse_args()


def _canonical_arguments(arguments: object) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(arguments)


def action_signature(action: object) -> str:
    if not isinstance(action, dict):
        return str(action)[:120]
    tool_calls = action.get("tool_calls") or []
    parts: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else None
        if function:
            name = function.get("name")
            arguments = function.get("arguments", {})
        else:
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
        if not name:
            continue
        parts.append(f"{name}({_canonical_arguments(arguments)})")
    if not parts:
        return "<no_named_tool_calls>"
    return " | ".join(parts)


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in (args.input or DEFAULT_INPUTS)]

    pair_counter: Counter[tuple[str, str]] = Counter()
    label_counter: Counter[tuple[str, int]] = Counter()
    trajectory_total = 0
    pair_total = 0

    for path in input_paths:
        with path.open("r", encoding="utf-8") as handle:
            trajectories = json.load(handle)
        trajectory_total += len(trajectories)
        examples = extract_state_examples(trajectories)
        for example in examples:
            action_key = action_signature(example.action)
            label_raw = (
                example.state.get("state", {}).get("context", {}).get("last_tool_execution_result")
            )
            label = normalize_last_tool_execution_result(label_raw)
            if label is None:
                label_int = -99
                state_key = "unknown"
            else:
                label_int = int(label)
                state_key = format_tool_execution_result_target(
                    label_int,
                    example.error_payload or "",
                    include_error_message=True,
                )
            pair_counter[(action_key, state_key)] += 1
            label_counter[(action_key, label_int)] += 1
            pair_total += 1

    pair_rows = sorted(
        pair_counter.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    )
    label_rows = sorted(
        label_counter.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    )

    payload = {
        "source_files": [str(path) for path in input_paths],
        "trajectory_count": trajectory_total,
        "pair_count": pair_total,
        "unique_action_state_pairs": len(pair_counter),
        "unique_actions": len({action for action, _ in pair_counter.keys()}),
        "label_distribution": {
            str(label): sum(count for (_, lbl), count in label_counter.items() if lbl == label)
            for label in sorted({lbl for _, lbl in label_counter.keys()}, reverse=True)
        },
        "action_state_counts": [
            {"action": action, "state": state, "count": count}
            for (action, state), count in pair_rows
        ],
        "action_label_counts": [
            {"action": action, "label": label, "count": count}
            for (action, label), count in label_rows
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(
        f"Read {trajectory_total} trajectories from {len(input_paths)} files, "
        f"counted {pair_total} action-state pairs, "
        f"{len(pair_counter)} unique pairs over {payload['unique_actions']} unique actions."
    )
    print(f"Wrote counter to {args.output}")


if __name__ == "__main__":
    main()
