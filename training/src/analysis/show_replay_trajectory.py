"""Show baseline and imagined replay trajectories for one trajectory index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY_PATH = (
    ROOT
    / "sessions"
    / "gymops_tool_output_qwen36_27b"
    / "evaluation_metrics_test_openai"
    / "gpt-4o-mini_replay_main.json"
)
DEFAULT_MODES = ("baseline", "imagined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-path",
        type=Path,
        default=DEFAULT_REPLAY_PATH,
        help="Replay JSON path.",
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        help="Trajectory index to show. Use --list-indices to inspect available indices.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=["baseline", "revision", "imagined"],
        default=[],
        help="Mode to show. May be repeated. Defaults to baseline and imagined.",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="Zero-based run index within the selected task record.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Maximum characters to print for long content fields.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Do not truncate long content fields.",
    )
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Include system messages in the printed conversation.",
    )
    parser.add_argument(
        "--show-rollouts",
        action="store_true",
        help="For imagined mode, show imagined rollout records.",
    )
    parser.add_argument(
        "--no-inline-imagined",
        action="store_true",
        help=(
            "Do not interleave selected imagined steps before each real imagined-mode "
            "agent action."
        ),
    )
    parser.add_argument(
        "--list-indices",
        action="store_true",
        help="List available trajectory indices and exit.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object at {path}, got {type(payload).__name__}.")
    return payload


def get_agent_replay_eval(payload: dict[str, Any]) -> dict[str, Any]:
    agent_replay_eval = payload.get("agent_replay_eval")
    if not isinstance(agent_replay_eval, dict):
        raise SystemExit("Missing `agent_replay_eval` object in replay JSON.")
    return agent_replay_eval


def get_mode_records(agent_replay_eval: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    mode_payload = agent_replay_eval.get(mode)
    if not isinstance(mode_payload, dict):
        raise SystemExit(f"Missing replay mode `{mode}`.")
    records = mode_payload.get("task_records")
    if not isinstance(records, list):
        raise SystemExit(f"Replay mode `{mode}` has no `task_records` list.")
    return records


def record_by_trajectory_index(
    records: list[dict[str, Any]],
    trajectory_index: int,
) -> dict[str, Any] | None:
    for record in records:
        if record.get("trajectory_index") == trajectory_index:
            return record
    return None


def truncate_text(value: Any, *, max_chars: int, full: bool) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    if full or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}... [truncated {omitted} chars]"


def format_tool_calls(tool_calls: Any) -> str:
    if not tool_calls:
        return ""
    formatted = []
    for index, call in enumerate(tool_calls, start=1):
        if not isinstance(call, dict):
            formatted.append(f"{index}. {call}")
            continue
        name = call.get("name") or call.get("tool_name") or "<unknown_tool>"
        args = call.get("args", call.get("arguments", {}))
        formatted.append(
            f"{index}. {name} args={json.dumps(args, ensure_ascii=False, sort_keys=True)}"
        )
    return "\n".join(formatted)


def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def print_stats(record: dict[str, Any]) -> None:
    result = record.get("result") or {}
    stats = result.get("statistics") or {}
    print(f"trajectory_index: {record.get('trajectory_index')}")
    print(f"gym_task_config_name: {record.get('gym_task_config_name')}")
    if stats:
        print(
            "stats: "
            f"success={stats.get('successful_runs')}/{stats.get('total_runs')}, "
            f"verifier_rate={stats.get('verifier_level_pass_rate')}, "
            f"verifiers={stats.get('total_verifiers_passed')}/"
            f"{stats.get('total_verifiers_checked')}, "
            f"time_ms={stats.get('mean_execution_time_ms')}"
        )
        verifier_stats = stats.get("individual_verifier_stats") or {}
        if verifier_stats:
            print("individual_verifiers:")
            for name, payload in verifier_stats.items():
                if isinstance(payload, dict):
                    print(
                        f"  - {name}: {payload.get('passed')}/"
                        f"{payload.get('total')} ({payload.get('pass_rate')})"
                    )


def print_message(
    message: dict[str, Any],
    *,
    index: int,
    max_chars: int,
    full: bool,
) -> None:
    message_type = message.get("type", "<unknown>")
    print()
    print(f"[{index:02d}] {message_type}")

    if message_type == "tool_result":
        print(f"tool_name: {message.get('tool_name')}")
        if message.get("gym_server"):
            print(f"gym_server: {message.get('gym_server')}")
        print(truncate_text(message.get("result"), max_chars=max_chars, full=full))
        return

    content = message.get("content")
    if content:
        print(truncate_text(content, max_chars=max_chars, full=full))

    tool_calls = format_tool_calls(message.get("tool_calls"))
    if tool_calls:
        print("tool_calls:")
        print(tool_calls)


def group_imagined_rollouts_by_step(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for record in run.get("imagined_rollout_records") or []:
        step_index = record.get("step_index")
        if isinstance(step_index, int):
            grouped[step_index] = record
    return grouped


def print_indented_block(prefix: str, value: Any, *, max_chars: int, full: bool) -> None:
    text = truncate_text(value, max_chars=max_chars, full=full)
    print(prefix + text.replace("\n", "\n" + " " * len(prefix)))


def print_imagined_step(
    step: dict[str, Any],
    *,
    max_chars: int,
    full: bool,
    indent: str = "  ",
) -> None:
    print(f"{indent}imagined_step={step.get('imagined_step')}")
    thought = step.get("thought")
    if thought:
        print_indented_block(
            f"{indent}thought: ",
            thought,
            max_chars=max_chars,
            full=full,
        )

    tool_calls = format_tool_calls(step.get("tool_calls"))
    if tool_calls:
        print(f"{indent}tool_calls:")
        for line in tool_calls.splitlines():
            print(f"{indent}  {line}")

    if step.get("final_answer"):
        print_indented_block(
            f"{indent}final_answer: ",
            step.get("final_answer"),
            max_chars=max_chars,
            full=full,
        )

    feedbacks = step.get("predicted_feedback") or []
    for feedback_index, feedback in enumerate(feedbacks, start=1):
        print(
            f"{indent}predicted_feedback[{feedback_index}]: "
            f"success={feedback.get('predicted_success')} "
            f"parse_error={feedback.get('parse_error')}"
        )
        predicted = feedback.get("predicted_tool_output")
        if predicted:
            print_indented_block(
                f"{indent}  predicted_tool_output: ",
                predicted,
                max_chars=max_chars,
                full=full,
            )
        error = feedback.get("predicted_error_message")
        if error:
            print_indented_block(
                f"{indent}  predicted_error_message: ",
                error,
                max_chars=max_chars,
                full=full,
            )


def print_inline_imagined_record(
    record: dict[str, Any],
    *,
    max_chars: int,
    full: bool,
) -> None:
    print()
    print("  " + "~" * 96)
    print(
        "  world_model_imagined_before_real_step: "
        f"step_index={record.get('step_index')} "
        f"observation_source={record.get('observation_source')}"
    )
    selection = record.get("selection")
    if selection:
        selected_index = selection.get("selected_index")
        strategy = selection.get("selection_strategy")
        fallback = selection.get("fallback_used")
        comments = selection.get("comments")
        print(
            "  selection: "
            f"selected_index={selected_index}, strategy={strategy}, fallback={fallback}"
        )
        if comments:
            print_indented_block(
                "  selection_comments: ",
                comments,
                max_chars=max_chars,
                full=full,
            )

    imagined_steps = record.get("imagined_steps") or []
    if not imagined_steps:
        print("  no selected imagined steps")
    for step in imagined_steps:
        print_imagined_step(step, max_chars=max_chars, full=full, indent="  ")
    print("  " + "~" * 96)


def print_run(
    run: dict[str, Any],
    *,
    include_system: bool,
    max_chars: int,
    full: bool,
    inline_imagined: bool,
) -> None:
    print(
        "run: "
        f"number={run.get('run_number')}, "
        f"world_model_mode={run.get('world_model_mode')}, "
        f"target={run.get('world_model_target')}, "
        f"overall_success={run.get('overall_success')}, "
        f"steps_taken={run.get('steps_taken')}, "
        f"execution_time_ms={run.get('execution_time_ms')}"
    )
    print()
    print("final_model_response:")
    print(truncate_text(run.get("model_response", ""), max_chars=max_chars, full=full))

    print()
    print("-" * 100)
    print("conversation_flow")
    print("-" * 100)
    imagined_by_step = group_imagined_rollouts_by_step(run) if inline_imagined else {}
    ai_step_index = 0
    for index, message in enumerate(run.get("conversation_flow") or []):
        if not include_system and message.get("type") == "system_message":
            continue
        if message.get("type") == "ai_message":
            imagined_record = imagined_by_step.get(ai_step_index)
            if imagined_record:
                print_inline_imagined_record(
                    imagined_record,
                    max_chars=max_chars,
                    full=full,
                )
        print_message(message, index=index, max_chars=max_chars, full=full)
        if message.get("type") == "ai_message":
            ai_step_index += 1


def print_imagined_rollouts(
    run: dict[str, Any],
    *,
    max_chars: int,
    full: bool,
) -> None:
    rollout_records = run.get("imagined_rollout_records") or []
    print()
    print("-" * 100)
    print(f"imagined_rollout_records ({len(rollout_records)})")
    print("-" * 100)
    for record in rollout_records:
        print()
        print(
            f"step_index={record.get('step_index')} "
            f"observation_source={record.get('observation_source')}"
        )
        selection = record.get("selection")
        if selection:
            print(f"selection: {json.dumps(selection, ensure_ascii=False, sort_keys=True)}")

        for step in record.get("imagined_steps") or []:
            print_imagined_step(step, max_chars=max_chars, full=full)


def list_indices(agent_replay_eval: dict[str, Any], modes: list[str]) -> None:
    for mode in modes:
        records = get_mode_records(agent_replay_eval, mode)
        print_header(f"{mode} trajectory indices")
        for record in records:
            result = record.get("result") or {}
            stats = result.get("statistics") or {}
            print(
                f"{record.get('trajectory_index')}: "
                f"{record.get('gym_task_config_name')} "
                f"verifier_rate={stats.get('verifier_level_pass_rate')}"
            )


def main() -> None:
    args = parse_args()
    payload = load_payload(args.replay_path)
    agent_replay_eval = get_agent_replay_eval(payload)
    modes = args.mode or list(DEFAULT_MODES)

    if args.list_indices:
        list_indices(agent_replay_eval, modes)
        return

    if args.trajectory_index is None:
        raise SystemExit("Pass --trajectory-index or use --list-indices.")

    for mode in modes:
        records = get_mode_records(agent_replay_eval, mode)
        record = record_by_trajectory_index(records, args.trajectory_index)
        if record is None:
            available = sorted(
                item.get("trajectory_index")
                for item in records
                if item.get("trajectory_index") is not None
            )
            raise SystemExit(
                f"No `{mode}` record for trajectory_index={args.trajectory_index}. "
                f"Available range: {available[:5]} ... {available[-5:]}"
            )

        print_header(f"{mode} trajectory")
        print_stats(record)

        runs = (record.get("result") or {}).get("runs") or []
        if not runs:
            print("No runs found.")
            continue
        if args.run_index < 0 or args.run_index >= len(runs):
            raise SystemExit(
                f"Invalid --run-index {args.run_index} for `{mode}`; "
                f"available runs: 0..{len(runs) - 1}"
            )

        run = runs[args.run_index]
        print_run(
            run,
            include_system=args.include_system,
            max_chars=args.max_chars,
            full=args.full,
            inline_imagined=(mode == "imagined" and not args.no_inline_imagined),
        )
        if args.show_rollouts:
            print_imagined_rollouts(run, max_chars=args.max_chars, full=args.full)


if __name__ == "__main__":
    main()
