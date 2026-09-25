#!/usr/bin/env python3
"""Convert Purple TerminalBench JSONL logs to standard message trajectories."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path(
    "/data/Trajectory/"
    "bm-Terminal-Bench-2_0_ex-llm_shell_tg-all_ts-all_cf-77cb399f613c_"
    "us-user_rn-20260519T220446Z-77cb399f613c/trajectories"
)
DEFAULT_OUTPUT_PATH = Path("trajectories/terminalbench_2_0_llm_shell_trajectories.json")
DEFAULT_TRAIN_OUTPUT_PATH = Path("trajectories/terminalbench_2_0_llm_shell_train_trajectories.json")
DEFAULT_TEST_OUTPUT_PATH = Path("trajectories/terminalbench_2_0_llm_shell_test_trajectories.json")
DEFAULT_SPLIT_MANIFEST_PATH = Path("trajectories/terminalbench_2_0_llm_shell_trajectory_split_manifest.json")

SYSTEM_PROMPT = """You are a terminal task-solving agent.
Use shell commands to inspect and modify the environment until the user's task is complete.
Each action is a shell execution request and each state is the resulting terminal output."""


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
    return records


def dump_records(records: list[dict[str, Any]], output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if output_format == "jsonl":
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        else:
            json.dump(records, handle, indent=2, ensure_ascii=False)
            handle.write("\n")


def dump_json(record: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def extract_task_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        if record.get("event_type") == "ShellProtocolTask":
            payload = record.get("payload")
            if isinstance(payload, dict):
                return payload
    return {}


def extract_final_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        if record.get("event_type") == "ShellProtocolFinal":
            payload = record.get("payload")
            if isinstance(payload, dict):
                return payload
    return {}


def extract_latest_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    latest_steps = -1
    for record in records:
        if record.get("event_type") != "PurpleInternalTrajectoryMetadata":
            continue
        info = (record.get("payload") or {}).get("info")
        if not isinstance(info, dict):
            continue
        steps = info.get("steps")
        if isinstance(steps, int) and steps >= latest_steps:
            latest = info
            latest_steps = steps
        elif latest_steps < 0:
            latest = info
    return latest


def parse_json_content(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_final_internal_messages(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the final accumulated assistant/tool transcript.

    Purple emits a full internal trajectory snapshot after each shell step, then
    flattens those snapshots into repeated PurpleInternalMessage records whose
    sequence numbers restart at zero. Keeping the last payload seen for each
    sequence reconstructs the final snapshot without duplicates.
    """

    by_sequence: dict[int, dict[str, Any]] = {}
    unsequenced: list[dict[str, Any]] = []
    for record in records:
        if record.get("event_type") != "PurpleInternalMessage":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        sequence = record.get("sequence")
        if isinstance(sequence, int):
            by_sequence[sequence] = payload
        else:
            unsequenced.append(payload)

    if by_sequence:
        return [by_sequence[index] for index in sorted(by_sequence)]
    return unsequenced


def exec_request_to_action(request: dict[str, Any]) -> dict[str, Any]:
    command = request.get("command")
    timeout = request.get("timeout")
    arguments = {"command": command}
    if timeout is not None:
        arguments["timeout"] = timeout
    return {
        "role": "action",
        "content": {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "arguments": arguments,
                    },
                }
            ]
        },
    }


def result_output(result: dict[str, Any]) -> str:
    stdout = result.get("stdout") or ""
    stderr = result.get("stderr") or ""
    if stdout and stderr:
        return f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
    if stdout:
        return str(stdout)
    if stderr:
        return f"STDERR:\n{stderr}"
    return ""


def exec_result_to_state(result: dict[str, Any]) -> dict[str, Any]:
    exit_code = result.get("exit_code")
    return {
        "role": "state",
        "content": {
            "last_tool_execution_result": 1 if exit_code == 0 else 0,
            "last_tool_name": "exec",
            "last_tool_exit_code": exit_code,
            "last_tool_output": result_output(result),
        },
    }


def convert_internal_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        parsed = parse_json_content(message.get("content"))
        if not parsed:
            continue

        kind = parsed.get("kind")
        if role == "assistant" and kind == "exec_request":
            converted.append(exec_request_to_action(parsed))
        elif role == "tool" and kind == "exec_result":
            converted.append(exec_result_to_state(parsed))
    return converted


def convert_file(path: Path) -> dict[str, Any]:
    records = load_jsonl_records(path)
    task_payload = extract_task_payload(records)
    final_payload = extract_final_payload(records)
    metadata = extract_latest_metadata(records)
    internal_messages = extract_final_internal_messages(records)

    instruction = task_payload.get("instruction")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if isinstance(instruction, str):
        messages.append({"role": "user", "content": instruction})

    messages.extend(convert_internal_messages(internal_messages))

    final_output = final_payload.get("output")
    if isinstance(final_output, str):
        messages.append({"role": "assistant", "content": final_output})

    return {
        "trajectory_id": f"terminalbench-2-0-llm-shell-{path.stem}",
        "source": "terminalbench_2_0_llm_shell_jsonl",
        "benchmark": "TerminalBench-2.0",
        "task_id": path.stem,
        "protocol": task_payload.get("protocol"),
        "executor": "llm_shell",
        "model": metadata.get("deployment"),
        "provider": metadata.get("provider"),
        "steps": metadata.get("steps"),
        "task": task_payload,
        "final": final_payload,
        "messages": messages,
    }


def split_trajectories(
    trajectories: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    test_task_count: int | None,
    input_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError(f"--train-ratio must be between 0 and 1, got {train_ratio}")

    task_ids = sorted(str(trajectory["task_id"]) for trajectory in trajectories)
    if test_task_count is None:
        test_task_count = len(task_ids) - int(round(len(task_ids) * train_ratio))
    if test_task_count < 0:
        raise ValueError(f"--test-task-count must be non-negative, got {test_task_count}")
    if test_task_count > len(task_ids):
        raise ValueError(
            f"Requested {test_task_count} test tasks, but only {len(task_ids)} "
            "TerminalBench trajectories are available."
        )

    shuffled_task_ids = list(task_ids)
    random.Random(seed).shuffle(shuffled_task_ids)
    test_task_id_set = set(shuffled_task_ids[:test_task_count])

    train_payload: list[dict[str, Any]] = []
    test_payload: list[dict[str, Any]] = []
    for trajectory in trajectories:
        split = "test" if str(trajectory["task_id"]) in test_task_id_set else "train"
        trajectory["split"] = split
        if split == "train":
            train_payload.append(trajectory)
        else:
            test_payload.append(trajectory)

    train_task_ids = [str(trajectory["task_id"]) for trajectory in train_payload]
    test_task_ids = [str(trajectory["task_id"]) for trajectory in test_payload]
    manifest = {
        "data_path": str(input_dir.resolve()),
        "seed": seed,
        "train_ratio": len(train_payload) / len(trajectories) if trajectories else 1.0,
        "split_unit": "task_id",
        "trajectory_count": len(trajectories),
        "task_count": len(task_ids),
        "test_task_count": test_task_count,
        "train_trajectories": len(train_payload),
        "test_trajectories": len(test_payload),
        "task_ids": task_ids,
        "train_task_ids": train_task_ids,
        "test_task_ids": test_task_ids,
    }
    return train_payload, test_payload, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert TerminalBench-2.0 Purple JSONL trajectories into the repo's "
            "standard message trajectory schema."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--train-output-path", type=Path, default=DEFAULT_TRAIN_OUTPUT_PATH)
    parser.add_argument("--test-output-path", type=Path, default=DEFAULT_TEST_OUTPUT_PATH)
    parser.add_argument("--generated-split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_PATH)
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl"],
        default="json",
        help="Write a JSON list or JSONL records.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument(
        "--test-task-count",
        type=int,
        default=None,
        help="Exact number of trajectories to place in the test split. Defaults from --train-ratio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    source_files = sorted(path for path in args.input_dir.glob("*.jsonl") if path.is_file())
    if not source_files:
        raise SystemExit(f"No .jsonl files found in {args.input_dir}")

    converted = [convert_file(path) for path in source_files]
    train_payload, test_payload, split_manifest = split_trajectories(
        converted,
        seed=args.seed,
        train_ratio=args.train_ratio,
        test_task_count=args.test_task_count,
        input_dir=args.input_dir,
    )
    dump_records(converted, args.output_path, args.output_format)
    dump_records(train_payload, args.train_output_path, args.output_format)
    dump_records(test_payload, args.test_output_path, args.output_format)
    dump_json(
        {
            **split_manifest,
            "output_path": str(args.output_path),
            "train_output_path": str(args.train_output_path),
            "test_output_path": str(args.test_output_path),
        },
        args.generated_split_manifest,
    )
    print(f"Wrote {len(converted)} trajectories to {args.output_path}")
    print(f"Wrote {len(train_payload)} train trajectories to {args.train_output_path}")
    print(f"Wrote {len(test_payload)} test trajectories to {args.test_output_path}")
    print(f"Wrote split manifest to {args.generated_split_manifest}")


if __name__ == "__main__":
    main()
