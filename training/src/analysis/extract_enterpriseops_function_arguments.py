#!/usr/bin/env python3
"""Extract EnterpriseOps-Gym tool function names and observed argument names."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = ROOT / "trajectories" / "enterpriseops_gym_multi_model_world_model_trajectories.json"
DEFAULT_OUTPUT_PATH = ROOT / "trajectories" / "enterpriseops_gym_multi_model_function_arguments.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a JSON list of function names and observed argument names from EnterpriseOps-Gym trajectories."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--include-counts",
        action="store_true",
        help="Include observed function and argument counts in the output.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def iter_message_tool_calls(message: dict[str, Any]):
    if message.get("role") != "action":
        return
    content = message.get("content")
    if not isinstance(content, dict):
        return
    for tool_call in content.get("tool_calls") or []:
        if isinstance(tool_call, dict):
            yield tool_call


def iter_tool_calls(trajectories: list[dict[str, Any]]):
    for trajectory in trajectories:
        for message in trajectory.get("messages") or []:
            yield from iter_message_tool_calls(message)


def argument_names(arguments: Any) -> list[str]:
    if isinstance(arguments, dict):
        return sorted(str(key) for key in arguments)
    return []


def extract_function_arguments(trajectories: list[dict[str, Any]], *, include_counts: bool) -> list[dict[str, Any]]:
    function_counts: Counter[str] = Counter()
    argument_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for tool_call in iter_tool_calls(trajectories):
        function = tool_call.get("function") or {}
        if not isinstance(function, dict):
            continue
        function_name = function.get("name")
        if not function_name:
            continue
        function_name = str(function_name)
        function_counts[function_name] += 1
        for argument_name in argument_names(function.get("arguments")):
            argument_counts[function_name][argument_name] += 1

    records = []
    for function_name in sorted(function_counts):
        record: dict[str, Any] = {
            "function_name": function_name,
            "arguments": sorted(argument_counts[function_name]),
        }
        if include_counts:
            record["function_count"] = function_counts[function_name]
            record["argument_counts"] = dict(sorted(argument_counts[function_name].items()))
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    trajectories = load_json(args.input_path)
    if not isinstance(trajectories, list):
        raise ValueError(f"Expected a JSON list in {args.input_path}")
    records = extract_function_arguments(trajectories, include_counts=args.include_counts)
    dump_json(args.output_path, records)
    print(f"Wrote {len(records)} function argument records to {args.output_path}")


if __name__ == "__main__":
    main()
