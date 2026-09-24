import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preparation.canonical_event_state import (  # noqa: E402
    canonical_event_from_action_state,
    event_state_value_counts,
)
from src.finetuning import (  # noqa: E402
    TRAJECTORY_DATASET_PRESETS,
    extract_state_examples,
    load_trajectory_records,
    normalize_loaded_trajectories,
)

DEFAULT_OUTPUT_DIR = ROOT / "trajectories"
DEFAULT_TRAIN_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "canonical_event_state_train_examples.jsonl"
DEFAULT_EVAL_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "canonical_event_state_eval_examples.jsonl"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "canonical_event_state_manifest.json"
DEFAULT_TRAJECTORY_DATASET = "enterpriseops_gym_terminalbench_2_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate categorical canonical-event EWM train/eval examples. "
            "Targets intentionally contain no free-text summary or evidence fields."
        )
    )
    parser.add_argument(
        "--trajectory-dataset",
        choices=sorted(TRAJECTORY_DATASET_PRESETS),
        default=DEFAULT_TRAJECTORY_DATASET,
    )
    parser.add_argument("--train-data-path", type=Path, nargs="+", default=None)
    parser.add_argument("--eval-data-path", type=Path, nargs="+", default=None)
    parser.add_argument("--train-output-path", type=Path, default=DEFAULT_TRAIN_OUTPUT_PATH)
    parser.add_argument("--eval-output-path", type=Path, default=DEFAULT_EVAL_OUTPUT_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--state-history-size", type=int, default=3)
    parser.add_argument(
        "--include-uncertainty",
        action="store_true",
        help="Include optional uncertainty_level and uncertainty_reason fields in canonical targets.",
    )
    return parser.parse_args()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_trajectories(paths: list[Path]) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for path in paths:
        trajectories.extend(normalize_loaded_trajectories(load_trajectory_records(path)))
    return trajectories


def benchmark_name_from_trajectory_id(trajectory_id: str) -> str:
    if trajectory_id.startswith("terminalbench"):
        return "Terminal-Bench-2.0"
    if trajectory_id.startswith("enterpriseops"):
        return "EnterpriseOps-Gym"
    return "unknown"


def build_rows(
    trajectories: list[dict[str, Any]],
    *,
    split: str,
    state_history_size: int,
    include_uncertainty: bool,
) -> list[dict[str, Any]]:
    examples = extract_state_examples(trajectories, state_history_size=state_history_size)
    rows = []
    for example in examples:
        event_state = canonical_event_from_action_state(
            example.action,
            example.state,
            include_uncertainty=include_uncertainty,
        )
        rows.append(
            {
                "split": split,
                "trajectory_id": example.trajectory_id,
                "trajectory_index": example.trajectory_index,
                "interaction_index": example.interaction_index,
                "benchmark": benchmark_name_from_trajectory_id(example.trajectory_id),
                "input_format": "system_task_history_action_v1",
                "target_format": "canonical_event_state_v1_categorical",
                "system_prompt": example.system_prompt,
                "task_prompt": example.user_prompt,
                "state_history": example.state_history,
                "input_history": example.input_history,
                "previous_state": example.previous_state,
                "action": example.action,
                "canonical_event_state": event_state,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    preset_train, preset_eval = TRAJECTORY_DATASET_PRESETS[args.trajectory_dataset]
    train_paths = args.train_data_path or list(preset_train)
    eval_paths = args.eval_data_path or list(preset_eval)

    train_trajectories = load_trajectories(train_paths)
    eval_trajectories = load_trajectories(eval_paths)
    train_rows = build_rows(
        train_trajectories,
        split="train",
        state_history_size=args.state_history_size,
        include_uncertainty=args.include_uncertainty,
    )
    eval_rows = build_rows(
        eval_trajectories,
        split="eval",
        state_history_size=args.state_history_size,
        include_uncertainty=args.include_uncertainty,
    )

    dump_jsonl(args.train_output_path, train_rows)
    dump_jsonl(args.eval_output_path, eval_rows)
    manifest = {
        "trajectory_dataset": args.trajectory_dataset,
        "source_train_paths": [str(path) for path in train_paths],
        "source_eval_paths": [str(path) for path in eval_paths],
        "train_trajectory_count": len(train_trajectories),
        "eval_trajectory_count": len(eval_trajectories),
        "train_example_count": len(train_rows),
        "eval_example_count": len(eval_rows),
        "target_format": "canonical_event_state_v1_categorical",
        "free_text_target_fields": [],
        "include_uncertainty": args.include_uncertainty,
        "optional_target_fields": ["uncertainty_level", "uncertainty_reason"],
        "train_value_counts": event_state_value_counts(train_rows),
        "eval_value_counts": event_state_value_counts(eval_rows),
    }
    dump_json(args.manifest_path, manifest)

    print(f"Wrote {len(train_rows)} train examples to {args.train_output_path}")
    print(f"Wrote {len(eval_rows)} eval examples to {args.eval_output_path}")
    print(f"Wrote manifest to {args.manifest_path}")


if __name__ == "__main__":
    main()
