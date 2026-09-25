import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PATH = ROOT / "trajectories" / "world_model_train_trajectories.json"
DEFAULT_TEST_PATH = ROOT / "trajectories" / "world_model_test_trajectories.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert `state.context.last_tool_execution_result` in reconstructed "
            "world-model trajectories from binary labels to {-1, 0, 1}."
        )
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        type=Path,
        default=[DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH],
        help="Trajectory JSON files to rewrite in place.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def try_parse_json(text):
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def is_tool_action(action_content) -> bool:
    return isinstance(action_content, dict) and bool(action_content.get("tool_calls"))


def detect_explicit_error_text(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "api error",
        "mcperror",
        "error code",
        "traceback",
        "input validation error",
        "bad request",
        "please fix your mistakes",
        "rate limit",
        "maximum context length",
        "internal server error",
        "service unavailable",
        "tool returned no output",
    )
    if lowered.startswith("error:"):
        return True
    return any(marker in lowered for marker in markers)


def context_has_explicit_tool_error(context: dict) -> bool:
    candidates = [
        context.get("error_message"),
        context.get("last_tool_output"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if not isinstance(candidate, str):
            candidate = json.dumps(candidate, ensure_ascii=False)
        parsed = try_parse_json(candidate)
        if isinstance(parsed, dict):
            if any(key in parsed for key in ("error", "errors")):
                return True
            status_code = parsed.get("status") or parsed.get("status_code")
            if isinstance(status_code, int) and status_code >= 400:
                return True
            if parsed.get("isError") is True:
                return True
        if detect_explicit_error_text(candidate):
            return True
    return False


def current_stage_of(state_message: dict) -> str:
    return (
        state_message.get("content", {})
        .get("state", {})
        .get("process", {})
        .get("current_stage", "")
        or ""
    )


def convert_trajectory_labels(trajectory: dict) -> dict[str, int]:
    counts = {-1: 0, 0: 0, 1: 0}
    previous_state_message = None
    messages = trajectory.get("messages", [])

    for cursor in range(len(messages) - 1):
        message = messages[cursor]
        next_message = messages[cursor + 1]
        if message.get("role") != "action" or next_message.get("role") != "state":
            continue

        current_state_root = next_message.setdefault("content", {}).setdefault("state", {})
        current_context = current_state_root.setdefault("context", {})
        previous_stage = current_stage_of(previous_state_message) if previous_state_message else ""
        current_stage = current_state_root.setdefault("process", {}).get("current_stage", "") or ""

        if is_tool_action(message.get("content")) and context_has_explicit_tool_error(current_context):
            converted = -1
        elif current_stage != previous_stage:
            converted = 1
        else:
            converted = 0

        current_context["last_tool_execution_result"] = converted
        counts[converted] += 1
        previous_state_message = next_message

    return counts


def convert_file(path: Path):
    trajectories = load_json(path)
    aggregate = {-1: 0, 0: 0, 1: 0}
    for trajectory in trajectories:
        counts = convert_trajectory_labels(trajectory)
        for key, value in counts.items():
            aggregate[key] += value
    dump_json(path, trajectories)
    return aggregate, len(trajectories)


def main():
    args = parse_args()
    for path in args.paths:
        counts, trajectory_count = convert_file(path)
        print(
            f"{path}: trajectories={trajectory_count} "
            f"labels[-1]={counts[-1]} labels[0]={counts[0]} labels[1]={counts[1]}"
        )


if __name__ == "__main__":
    main()
