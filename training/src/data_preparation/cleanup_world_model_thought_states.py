#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preparation.world_model_trajectory_cleanup import cleanup_world_model_trajectory

DEFAULT_PATHS = [
    ROOT / "trajectories" / "world_model_train_trajectories.json",
    ROOT / "trajectories" / "world_model_test_trajectories.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse assistant-thought state pairs in world-model trajectories.",
    )
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_PATHS)
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def summarize(trajectories: list[dict]) -> tuple[int, int, int]:
    action_count = 0
    assistant_count = 0
    state_count = 0
    for trajectory in trajectories:
        for message in trajectory.get("messages", []):
            if message.get("role") == "action":
                action_count += 1
            elif message.get("role") == "assistant":
                assistant_count += 1
            elif message.get("role") == "state":
                state_count += 1
    return action_count, assistant_count, state_count


def main() -> None:
    args = parse_args()
    for path in args.paths:
        trajectories = load_json(path)
        cleaned = [cleanup_world_model_trajectory(trajectory) for trajectory in trajectories]
        dump_json(path, cleaned)
        action_count, assistant_count, state_count = summarize(cleaned)
        print(
            f"{path}: trajectories={len(cleaned)} actions={action_count} assistants={assistant_count} states={state_count}"
        )


if __name__ == "__main__":
    main()
