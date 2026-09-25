#!/usr/bin/env python3
"""Convert Purple CRMArenaPro JSONL trajectories to CRMArena result JSON.

The output shape matches CRMArena result files such as:

[
  {
    "task_id": 200,
    "task_type": "activity_priority",
    "gt_answer": [...],
    "reward": 0,
    "agent_info": {...},
    "traj": [{"role": "system", "content": "..."}, ...]
  }
]

Ground truth and reward are not present in the Purple JSONL files. Pass
--reference-results to copy those fields from an existing CRMArena-style result
file for overlapping task ids; otherwise they are emitted as null.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path(
    "/data/Trajectory/"
    "bm-crmarenapro_ex-baseline_crm_agent_tg-sample_ts-0_2134_cf-"
    "f9299b609828_us-user_rn-20260519T050759Z-f9299b609828/"
    "trajectories"
)
DEFAULT_OUTPUT = Path("trajectories/crmarenapro_baseline_crm_agent_results.json")
DEFAULT_TRAIN_OUTPUT = Path("trajectories/crmarenapro_baseline_crm_agent_train_results.json")
DEFAULT_TEST_OUTPUT = Path("trajectories/crmarenapro_baseline_crm_agent_test_results.json")
DEFAULT_SPLIT_MANIFEST = Path("trajectories/crmarenapro_baseline_crm_agent_split_manifest.json")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
    return records


def dump_results(records: list[dict[str, Any]], output_path: Path, indent: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        if indent and indent > 0:
            json.dump(records, f, indent=indent, ensure_ascii=False)
            f.write("\n")
        else:
            json.dump(records, f, separators=(",", ":"), ensure_ascii=False)


def dump_json(record: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
        f.write("\n")


def maybe_int(value: Any) -> Any:
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def numeric_path_key(path: Path) -> tuple[int, Any]:
    if path.stem.isdigit():
        return (0, int(path.stem))
    return (1, path.stem)


def message_text(message: dict[str, Any]) -> str | None:
    parts = message.get("parts") or []
    texts = [
        part.get("text")
        for part in parts
        if isinstance(part, dict) and part.get("kind") == "text"
    ]
    if not texts:
        return None
    return "\n".join(text for text in texts if text is not None)


def extract_task_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        if record.get("event_type") != "Message":
            continue
        payload = record.get("payload") or {}
        text = message_text(payload)
        if not text:
            continue
        try:
            task_payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(task_payload, dict):
            return task_payload
    return {}


def extract_answer_artifact(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        artifact = (record.get("event") or {}).get("artifact") or {}
        if artifact.get("name") != "Answer":
            continue
        for part in artifact.get("parts") or []:
            if isinstance(part, dict) and part.get("kind") == "data":
                data = part.get("data")
                if isinstance(data, dict):
                    return data
    return {}


def extract_internal_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        if record.get("event_type") != "PurpleInternalTrajectoryMetadata":
            continue
        payload = record.get("payload") or {}
        info = payload.get("info")
        if isinstance(info, dict):
            return info
    return {}


def internal_messages(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = []
    for record in records:
        if record.get("event_type") != "PurpleInternalMessage":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        message = dict(payload)
        message["_sequence"] = record.get("sequence")
        messages.append(message)
    return messages


def compact_role_content(message: dict[str, Any]) -> dict[str, str] | None:
    role = message.get("role")
    content = message.get("content")
    if role not in {"system", "user", "assistant"}:
        return None
    if not isinstance(content, str):
        return None
    return {"role": role, "content": content.strip()}


def reconstruct_traj(records: list[dict[str, Any]], answer_data: dict[str, Any]) -> list[dict[str, str]]:
    """Reconstruct CRMArena's role/content message list.

    Purple records contain full prompt histories for every LLM request. The last
    request history is closest to CRMArena's agent.get_messages() output, so use
    that history and append the final assistant response emitted after it.
    """

    messages = internal_messages(records)
    request_indices = [
        message["request_index"]
        for message in messages
        if isinstance(message.get("request_index"), int)
        and message.get("role") in {"system", "user", "assistant"}
    ]
    if request_indices:
        final_request_index = max(request_indices)
        prompt_messages = [
            message
            for message in messages
            if message.get("request_index") == final_request_index
            and compact_role_content(message) is not None
        ]
        traj = [compact_role_content(message) for message in prompt_messages]
        traj = [message for message in traj if message is not None]

        last_prompt_sequence = max(
            (
                message.get("_sequence")
                for message in prompt_messages
                if isinstance(message.get("_sequence"), int)
            ),
            default=None,
        )
        for message in messages:
            if message.get("role") != "assistant" or message.get("request_index") is not None:
                continue
            sequence = message.get("_sequence")
            if last_prompt_sequence is not None and isinstance(sequence, int):
                if sequence <= last_prompt_sequence:
                    continue
            compacted = compact_role_content(message)
            if compacted is not None:
                traj.append(compacted)
                break
        if traj:
            return traj

    traj = []
    seen: set[tuple[str, str]] = set()
    for message in messages:
        compacted = compact_role_content(message)
        if compacted is None:
            continue
        key = (compacted["role"], compacted["content"])
        if key in seen:
            continue
        seen.add(key)
        traj.append(compacted)

    full_response = answer_data.get("full_response")
    if isinstance(full_response, str) and (
        not traj or traj[-1] != {"role": "assistant", "content": full_response.strip()}
    ):
        traj.append({"role": "assistant", "content": full_response.strip()})
    return traj


def observation_sizes(records: list[dict[str, Any]]) -> list[int]:
    sizes = []
    for message in internal_messages(records):
        if message.get("role") != "tool":
            continue
        result = message.get("result")
        if not isinstance(result, dict):
            continue
        if isinstance(result.get("count"), int):
            sizes.append(result["count"])
        elif isinstance(result.get("data"), list):
            sizes.append(len(result["data"]))
    return sizes


def load_reference_results(paths: list[Path]) -> dict[Any, dict[str, Any]]:
    references: dict[Any, dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON array of CRMArena results")
        for item in data:
            if not isinstance(item, dict) or "task_id" not in item:
                continue
            references[item["task_id"]] = item
            references[str(item["task_id"])] = item
    return references


def parsed_answer(answer: Any) -> list[Any] | None:
    if answer is None:
        return None
    if isinstance(answer, list):
        return answer
    return [answer]


def convert_file(path: Path, references: dict[Any, dict[str, Any]]) -> dict[str, Any]:
    records = load_jsonl(path)
    task_payload = extract_task_payload(records)
    answer_data = extract_answer_artifact(records)
    internal_metadata = extract_internal_metadata(records)
    metrics = answer_data.get("metrics") or internal_metadata.get("metrics") or {}

    task_id = maybe_int(answer_data.get("task_id") or task_payload.get("task_id") or path.stem)
    task_type = (
        answer_data.get("category")
        or task_payload.get("task_category")
        or internal_metadata.get("category")
    )
    reference = references.get(task_id) or references.get(str(task_id)) or {}
    answer = answer_data.get("answer")

    agent_info = {
        "observation_sizes": observation_sizes(records),
        "end_reason": {
            "source": "agent",
            "message": "Answer artifact" if answer_data else "Converted trajectory",
            "content": answer_data.get("full_response") or answer,
            "parsed_answer": parsed_answer(answer),
        },
        "usage": {
            "cost": [],
            "completion_tokens": [],
            "prompt_tokens": [],
            "total_tokens": [],
        },
        "total_cost": None,
        "num_turns": [metrics.get("turns")] if isinstance(metrics, dict) else [],
        "source_info": {
            "provider": internal_metadata.get("provider"),
            "model": internal_metadata.get("model"),
            "executor": "baseline_crm_agent",
            "metrics": metrics,
        },
    }

    return {
        "task_id": task_id,
        "task_type": task_type,
        "gt_answer": reference.get("gt_answer"),
        "reward": reference.get("reward"),
        "agent_info": agent_info,
        "traj": reconstruct_traj(records, answer_data),
    }


def split_results(
    records: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    test_task_count: int | None,
    input_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError(f"--train-ratio must be between 0 and 1, got {train_ratio}")

    task_ids = sorted(str(record["task_id"]) for record in records)
    if test_task_count is None:
        test_task_count = len(task_ids) - int(round(len(task_ids) * train_ratio))
    if test_task_count < 0:
        raise ValueError(f"--test-task-count must be non-negative, got {test_task_count}")
    if test_task_count > len(task_ids):
        raise ValueError(
            f"Requested {test_task_count} test tasks, but only {len(task_ids)} "
            "CRMArenaPro trajectories are available."
        )

    shuffled_task_ids = list(task_ids)
    random.Random(seed).shuffle(shuffled_task_ids)
    test_task_id_set = set(shuffled_task_ids[:test_task_count])

    train_payload = [record for record in records if str(record["task_id"]) not in test_task_id_set]
    test_payload = [record for record in records if str(record["task_id"]) in test_task_id_set]
    train_task_ids = [record["task_id"] for record in train_payload]
    test_task_ids = [record["task_id"] for record in test_payload]
    manifest = {
        "data_path": str(input_dir.resolve()),
        "seed": seed,
        "train_ratio": len(train_payload) / len(records) if records else 1.0,
        "split_unit": "task_id",
        "trajectory_count": len(records),
        "task_count": len(task_ids),
        "test_task_count": test_task_count,
        "train_trajectories": len(train_payload),
        "test_trajectories": len(test_payload),
        "task_ids": [record["task_id"] for record in records],
        "train_task_ids": train_task_ids,
        "test_task_ids": test_task_ids,
    }
    return train_payload, test_payload, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Purple CRMArenaPro JSONL trajectories to CRMArena result JSON."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory of per-task .jsonl files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CRMArena-style JSON file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT,
        help=f"Output train split CRMArena-style JSON file. Default: {DEFAULT_TRAIN_OUTPUT}",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=DEFAULT_TEST_OUTPUT,
        help=f"Output test split CRMArena-style JSON file. Default: {DEFAULT_TEST_OUTPUT}",
    )
    parser.add_argument(
        "--generated-split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
        help=f"Output split manifest JSON file. Default: {DEFAULT_SPLIT_MANIFEST}",
    )
    parser.add_argument(
        "--reference-results",
        type=Path,
        action="append",
        default=[],
        help="Optional CRMArena-style result JSON to copy gt_answer/reward by task_id. May be repeated.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use 0 for compact output.",
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

    references = load_reference_results(args.reference_results)
    paths = sorted(args.input_dir.glob("*.jsonl"), key=numeric_path_key)
    if not paths:
        raise SystemExit(f"No .jsonl files found in {args.input_dir}")

    converted = [convert_file(path, references) for path in paths]
    train_payload, test_payload, split_manifest = split_results(
        converted,
        seed=args.seed,
        train_ratio=args.train_ratio,
        test_task_count=args.test_task_count,
        input_dir=args.input_dir,
    )
    dump_results(converted, args.output, args.indent)
    dump_results(train_payload, args.train_output, args.indent)
    dump_results(test_payload, args.test_output, args.indent)
    dump_json(
        {
            **split_manifest,
            "output": str(args.output),
            "train_output": str(args.train_output),
            "test_output": str(args.test_output),
        },
        args.generated_split_manifest,
    )

    print(f"Wrote {len(converted)} trajectories to {args.output}")
    print(f"Wrote {len(train_payload)} train trajectories to {args.train_output}")
    print(f"Wrote {len(test_payload)} test trajectories to {args.test_output}")
    print(f"Wrote split manifest to {args.generated_split_manifest}")


if __name__ == "__main__":
    main()
