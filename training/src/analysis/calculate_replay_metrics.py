"""Calculate replay success, verifier, tool-call, and step-count metrics."""

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
    / "gpt-4o-mini_replay.json"
)

# Meta-modes mirror `--agent-replay-mode` aliases in src/evaluation.py: passing
# one to `--mode` expands to its constituent replay modes (intersected with the
# modes actually present in the file).
META_MODES: dict[str, list[str]] = {
    "all": ["baseline", "revision", "imagined"],
    "all_with_itp": [
        "baseline",
        "revision",
        "imagined",
        "react_wm",
        "react_wm_decide_k",
        "react_wm_rl_k",
    ],
    "itp_only": ["react_wm", "react_wm_decide_k", "react_wm_rl_k"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "replay_path",
        nargs="?",
        type=Path,
        default=DEFAULT_REPLAY_PATH,
        help="Path to an agent replay JSON file.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help=(
            "Only calculate metrics for the specified mode(s), e.g. baseline. "
            "Meta-modes (all, all_with_itp, itp_only) expand to their "
            "constituent replay modes present in the file."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text output.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a JSON object at {path}, got {type(payload).__name__}.")
    return payload


def iter_mode_payloads(agent_replay_eval: dict[str, Any]) -> dict[str, dict[str, Any]]:
    modes: dict[str, dict[str, Any]] = {}
    for key, value in agent_replay_eval.items():
        if isinstance(value, dict) and isinstance(value.get("task_records"), list):
            modes[key] = value
    return modes


def count_tools_for_run(run: dict[str, Any]) -> int | None:
    tools_used = run.get("tools_used")
    if isinstance(tools_used, list):
        return len(tools_used)

    tool_results = run.get("tool_results")
    if isinstance(tool_results, list):
        return len(tool_results)

    return None


def count_steps_for_run(run: dict[str, Any]) -> int | None:
    steps_taken = run.get("steps_taken")
    if isinstance(steps_taken, int) and not isinstance(steps_taken, bool):
        return steps_taken
    if isinstance(steps_taken, float) and not isinstance(steps_taken, bool):
        return int(steps_taken)
    return None


def count_tools_from_statistics(statistics: dict[str, Any]) -> int:
    tool_usage = statistics.get("tool_usage") or {}
    if not isinstance(tool_usage, dict):
        return 0
    total = 0
    for value in tool_usage.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += int(value)
    return total


def add_run_counts(
    run: dict[str, Any],
    summary: dict[str, int],
) -> bool:
    tools_called = count_tools_for_run(run)
    if tools_called is not None:
        summary["total_tools_called"] += tools_called
        summary["tool_count_runs"] += 1

    steps_taken = count_steps_for_run(run)
    if steps_taken is not None:
        summary["total_steps_taken"] += steps_taken
        summary["step_count_runs"] += 1

    return tools_called is not None


def add_statistics_counts(statistics: dict[str, Any], summary: dict[str, int]) -> None:
    summary["total_runs"] += int(statistics.get("total_runs") or 0)
    summary["successful_runs"] += int(statistics.get("successful_runs") or 0)
    summary["total_verifiers_checked"] += int(statistics.get("total_verifiers_checked") or 0)
    summary["total_verifiers_passed"] += int(statistics.get("total_verifiers_passed") or 0)


def add_task_counts(task_record: dict[str, Any], summary: dict[str, int]) -> None:
    result = task_record.get("result") or {}
    statistics = result.get("statistics") or {}
    add_statistics_counts(statistics, summary)

    task_tool_count_runs = 0
    runs = result.get("runs") or []
    if isinstance(runs, list):
        for run in runs:
            if isinstance(run, dict) and add_run_counts(run, summary):
                task_tool_count_runs += 1

    if task_tool_count_runs:
        return

    statistics_tools = count_tools_from_statistics(statistics)
    if statistics_tools:
        summary["total_tools_called"] += statistics_tools
        summary["tool_count_runs"] += int(statistics.get("total_runs") or 1)


def summarize_mode(mode_name: str, mode_payload: dict[str, Any]) -> dict[str, Any]:
    task_records = mode_payload.get("task_records") or []
    counts = {
        "total_runs": 0,
        "successful_runs": 0,
        "total_verifiers_checked": 0,
        "total_verifiers_passed": 0,
        "total_tools_called": 0,
        "tool_count_runs": 0,
        "total_steps_taken": 0,
        "step_count_runs": 0,
    }

    for task_record in task_records:
        add_task_counts(task_record, counts)

    success_rate = counts["successful_runs"] / counts["total_runs"] if counts["total_runs"] else 0.0
    verifier_rate = (
        counts["total_verifiers_passed"] / counts["total_verifiers_checked"]
        if counts["total_verifiers_checked"]
        else 0.0
    )
    average_tools_called = (
        counts["total_tools_called"] / counts["tool_count_runs"]
        if counts["tool_count_runs"]
        else 0.0
    )
    average_steps_taken = (
        counts["total_steps_taken"] / counts["step_count_runs"]
        if counts["step_count_runs"]
        else 0.0
    )

    return {
        "mode": mode_name,
        "evaluated_tasks": int(mode_payload.get("evaluated_tasks") or len(task_records)),
        "errored_tasks": int(mode_payload.get("errored_tasks") or 0),
        "task_records": len(task_records),
        "total_runs": counts["total_runs"],
        "successful_runs": counts["successful_runs"],
        "success_rate": success_rate,
        "total_verifiers_checked": counts["total_verifiers_checked"],
        "total_verifiers_passed": counts["total_verifiers_passed"],
        "verifier_rate": verifier_rate,
        "tool_count_runs": counts["tool_count_runs"],
        "total_tools_called": counts["total_tools_called"],
        "average_tools_called": average_tools_called,
        "step_count_runs": counts["step_count_runs"],
        "total_steps_taken": counts["total_steps_taken"],
        "average_steps_taken": average_steps_taken,
    }


def format_text(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"[{summary['mode']}]",
            f"evaluated_tasks: {summary['evaluated_tasks']}",
            f"errored_tasks: {summary['errored_tasks']}",
            f"task_records: {summary['task_records']}",
            f"successful_runs: {summary['successful_runs']}/{summary['total_runs']}",
            f"success_rate: {summary['success_rate']:.4%}",
            (
                "verifiers_passed: "
                f"{summary['total_verifiers_passed']}/{summary['total_verifiers_checked']}"
            ),
            f"verifier_rate: {summary['verifier_rate']:.4%}",
            (
                "tools_called: "
                f"{summary['total_tools_called']} total / "
                f"{summary['average_tools_called']:.4f} avg per counted run "
                f"({summary['tool_count_runs']} runs)"
            ),
            (
                "steps_taken: "
                f"{summary['total_steps_taken']} total / "
                f"{summary['average_steps_taken']:.4f} avg per counted run "
                f"({summary['step_count_runs']} runs)"
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    payload = load_payload(args.replay_path)
    agent_replay_eval = payload.get("agent_replay_eval")
    if not isinstance(agent_replay_eval, dict):
        raise SystemExit("Missing `agent_replay_eval` object in replay file.")

    all_modes = iter_mode_payloads(agent_replay_eval)
    if not all_modes:
        raise SystemExit("No replay modes with `task_records` found in the replay file.")

    if args.mode:
        selected_modes = []
        missing_modes = []
        for requested in args.mode:
            if requested in META_MODES:
                present = [m for m in META_MODES[requested] if m in all_modes]
                if not present:
                    missing_modes.append(requested)
                selected_modes.extend(present)
            elif requested in all_modes:
                selected_modes.append(requested)
            else:
                missing_modes.append(requested)
        # De-duplicate while preserving order (meta-modes may overlap).
        selected_modes = list(dict.fromkeys(selected_modes))
    else:
        selected_modes = list(all_modes)
        missing_modes = []

    if missing_modes:
        available = ", ".join(sorted(all_modes))
        meta = ", ".join(sorted(META_MODES))
        missing = ", ".join(missing_modes)
        raise SystemExit(
            f"Unknown or empty mode(s): {missing}. "
            f"Available modes: {available}. Meta-modes: {meta}."
        )

    summaries = [summarize_mode(mode, all_modes[mode]) for mode in selected_modes]

    if args.json:
        print(json.dumps(summaries, indent=2))
        return

    print(f"replay_path: {args.replay_path}")
    for summary in summaries:
        print()
        print(format_text(summary))


if __name__ == "__main__":
    main()
