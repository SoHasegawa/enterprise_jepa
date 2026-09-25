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
        help="Only calculate metrics for the specified mode(s), e.g. baseline.",
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


def summarize_mode(mode_name: str, mode_payload: dict[str, Any]) -> dict[str, Any]:
    task_records = mode_payload.get("task_records") or []
    total_runs = 0
    successful_runs = 0
    total_verifiers_checked = 0
    total_verifiers_passed = 0
    total_tools_called = 0
    tool_count_runs = 0
    total_steps_taken = 0
    step_count_runs = 0
    latent_plan_record_count = 0
    latent_override_applied = 0
    latent_margin_blocked = 0
    latent_top_score_spread_total = 0.0
    latent_top_score_spread_count = 0
    latent_pool_count = 0
    latent_pool_unique_tool_total = 0
    latent_pool_duplicate_total = 0
    latent_pool_requested_total = 0
    latent_pool_sampled_total = 0
    latent_pool_accepted_different_tool_total = 0
    latent_pool_accepted_same_tool_args_total = 0
    latent_pool_accepted_other_total = 0
    latent_pool_accepted_bucket_count = 0
    latent_next_subgoal_records = 0
    latent_active_subgoal_records = 0
    latent_missing_subgoal_skips = 0

    for task_record in task_records:
        result = task_record.get("result") or {}
        statistics = result.get("statistics") or {}
        total_runs += int(statistics.get("total_runs") or 0)
        successful_runs += int(statistics.get("successful_runs") or 0)
        total_verifiers_checked += int(statistics.get("total_verifiers_checked") or 0)
        total_verifiers_passed += int(statistics.get("total_verifiers_passed") or 0)

        runs = result.get("runs") or []
        task_tool_count_runs = 0
        if isinstance(runs, list):
            for run in runs:
                if not isinstance(run, dict):
                    continue
                tools_called = count_tools_for_run(run)
                if tools_called is not None:
                    total_tools_called += tools_called
                    tool_count_runs += 1
                    task_tool_count_runs += 1
                steps_taken = count_steps_for_run(run)
                if steps_taken is not None:
                    total_steps_taken += steps_taken
                    step_count_runs += 1
                latent_records = run.get("latent_plan_records") or []
                if isinstance(latent_records, list):
                    for latent_record in latent_records:
                        if not isinstance(latent_record, dict):
                            continue
                        latent_plan_record_count += 1
                        if latent_record.get("override_applied") is True:
                            latent_override_applied += 1
                        if latent_record.get("goal_mode") == "next_subgoal":
                            latent_next_subgoal_records += 1
                        if isinstance(latent_record.get("active_subgoal"), dict):
                            latent_active_subgoal_records += 1
                        if latent_record.get("override_reason") == "score_margin_not_met":
                            latent_margin_blocked += 1
                        if latent_record.get("override_reason") == "missing_active_subgoal":
                            latent_missing_subgoal_skips += 1
                        spread = latent_record.get("top_score_spread")
                        if isinstance(spread, (int, float)) and not isinstance(spread, bool):
                            latent_top_score_spread_total += float(spread)
                            latent_top_score_spread_count += 1
                        pools = latent_record.get("candidate_pool_diagnostics") or []
                        if isinstance(pools, list):
                            for pool in pools:
                                if not isinstance(pool, dict):
                                    continue
                                latent_pool_count += 1
                                unique_tools = pool.get("unique_tool_count")
                                if isinstance(unique_tools, (int, float)) and not isinstance(unique_tools, bool):
                                    latent_pool_unique_tool_total += int(unique_tools)
                                duplicates = pool.get("duplicate_candidates")
                                if isinstance(duplicates, (int, float)) and not isinstance(duplicates, bool):
                                    latent_pool_duplicate_total += int(duplicates)
                                requested = pool.get("requested_candidates")
                                if isinstance(requested, (int, float)) and not isinstance(requested, bool):
                                    latent_pool_requested_total += int(requested)
                                sampled = pool.get("sampled_candidates")
                                if isinstance(sampled, (int, float)) and not isinstance(sampled, bool):
                                    latent_pool_sampled_total += int(sampled)
                                accepted_different = pool.get("accepted_different_tool_name")
                                accepted_same = pool.get("accepted_same_tool_name_different_args")
                                accepted_other = pool.get("accepted_other")
                                if (
                                    isinstance(accepted_different, (int, float))
                                    and not isinstance(accepted_different, bool)
                                    and isinstance(accepted_same, (int, float))
                                    and not isinstance(accepted_same, bool)
                                    and isinstance(accepted_other, (int, float))
                                    and not isinstance(accepted_other, bool)
                                ):
                                    latent_pool_accepted_bucket_count += 1
                                    latent_pool_accepted_different_tool_total += int(accepted_different)
                                    latent_pool_accepted_same_tool_args_total += int(accepted_same)
                                    latent_pool_accepted_other_total += int(accepted_other)

        if not task_tool_count_runs:
            statistics_tools = count_tools_from_statistics(statistics)
            if statistics_tools:
                total_tools_called += statistics_tools
                tool_count_runs += int(statistics.get("total_runs") or 1)

    success_rate = successful_runs / total_runs if total_runs else 0.0
    verifier_rate = (
        total_verifiers_passed / total_verifiers_checked
        if total_verifiers_checked
        else 0.0
    )
    average_tools_called = total_tools_called / tool_count_runs if tool_count_runs else 0.0
    average_steps_taken = total_steps_taken / step_count_runs if step_count_runs else 0.0
    average_latent_top_score_spread = (
        latent_top_score_spread_total / latent_top_score_spread_count
        if latent_top_score_spread_count
        else 0.0
    )
    average_latent_pool_unique_tool_count = (
        latent_pool_unique_tool_total / latent_pool_count if latent_pool_count else 0.0
    )

    return {
        "mode": mode_name,
        "evaluated_tasks": int(mode_payload.get("evaluated_tasks") or len(task_records)),
        "errored_tasks": int(mode_payload.get("errored_tasks") or 0),
        "task_records": len(task_records),
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "success_rate": success_rate,
        "total_verifiers_checked": total_verifiers_checked,
        "total_verifiers_passed": total_verifiers_passed,
        "verifier_rate": verifier_rate,
        "tool_count_runs": tool_count_runs,
        "total_tools_called": total_tools_called,
        "average_tools_called": average_tools_called,
        "step_count_runs": step_count_runs,
        "total_steps_taken": total_steps_taken,
        "average_steps_taken": average_steps_taken,
        "latent_plan_records": latent_plan_record_count,
        "latent_override_applied": latent_override_applied,
        "latent_margin_blocked": latent_margin_blocked,
        "latent_top_score_spread_count": latent_top_score_spread_count,
        "average_latent_top_score_spread": average_latent_top_score_spread,
        "latent_pool_count": latent_pool_count,
        "average_latent_pool_unique_tool_count": average_latent_pool_unique_tool_count,
        "latent_pool_duplicate_candidates": latent_pool_duplicate_total,
        "latent_pool_requested_candidates": latent_pool_requested_total,
        "latent_pool_sampled_candidates": latent_pool_sampled_total,
        "latent_pool_accepted_bucket_count": latent_pool_accepted_bucket_count,
        "latent_pool_accepted_different_tool_name": latent_pool_accepted_different_tool_total,
        "latent_pool_accepted_same_tool_name_different_args": latent_pool_accepted_same_tool_args_total,
        "latent_pool_accepted_other": latent_pool_accepted_other_total,
        "latent_next_subgoal_records": latent_next_subgoal_records,
        "latent_active_subgoal_records": latent_active_subgoal_records,
        "latent_missing_subgoal_skips": latent_missing_subgoal_skips,
    }


def format_text(summary: dict[str, Any]) -> str:
    lines = [
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
    if summary.get("latent_plan_records"):
        if not summary.get("latent_top_score_spread_count") and not summary.get("latent_pool_count"):
            if summary.get("latent_missing_subgoal_skips"):
                lines.append(
                    "latent_plans: "
                    f"{summary['latent_plan_records']} records / "
                    f"{summary['latent_override_applied']} overrides / "
                    f"{summary['latent_margin_blocked']} margin-blocked / "
                    f"{summary['latent_active_subgoal_records']} active subgoals / "
                    f"{summary['latent_missing_subgoal_skips']} missing-subgoal skips "
                    "(latent scoring skipped)"
                )
            else:
                lines.append(
                    f"latent_plans: {summary['latent_plan_records']} records "
                    "(diagnostics unavailable in this replay file)"
                )
        else:
            lines.extend(
                [
                    (
                        "latent_plans: "
                        f"{summary['latent_plan_records']} records / "
                        f"{summary['latent_override_applied']} overrides / "
                        f"{summary['latent_margin_blocked']} margin-blocked / "
                        f"{summary['latent_active_subgoal_records']} active subgoals / "
                        f"{summary['latent_missing_subgoal_skips']} missing-subgoal skips"
                    ),
                    (
                        "latent_score_spread: "
                        f"{summary['average_latent_top_score_spread']:.6f} avg top-plan spread "
                        f"({summary['latent_top_score_spread_count']} records)"
                    ),
                    (
                        "latent_pool_diversity: "
                        f"{summary['average_latent_pool_unique_tool_count']:.4f} avg unique tools per pool / "
                        f"{summary['latent_pool_duplicate_candidates']} duplicate candidates / "
                        f"{summary['latent_pool_sampled_candidates']} sampled "
                        f"from {summary['latent_pool_requested_candidates']} requested"
                        + (
                            f" / accepted buckets: "
                            f"{summary['latent_pool_accepted_different_tool_name']} different-tool, "
                            f"{summary['latent_pool_accepted_same_tool_name_different_args']} same-tool-diff-args, "
                            f"{summary['latent_pool_accepted_other']} other"
                            if summary.get("latent_pool_accepted_bucket_count")
                            else " / accepted buckets unavailable"
                        )
                    ),
                ]
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload = load_payload(args.replay_path)
    agent_replay_eval = payload.get("agent_replay_eval")
    if not isinstance(agent_replay_eval, dict):
        raise SystemExit("Missing `agent_replay_eval` object in replay file.")

    all_modes = iter_mode_payloads(agent_replay_eval)
    if not all_modes:
        raise SystemExit("No replay modes with `task_records` found in the replay file.")

    selected_modes = args.mode or list(all_modes)
    missing_modes = [mode for mode in selected_modes if mode not in all_modes]
    if missing_modes:
        available = ", ".join(sorted(all_modes))
        missing = ", ".join(missing_modes)
        raise SystemExit(f"Unknown mode(s): {missing}. Available modes: {available}.")

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
