from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_bench_repeated.py"
SPEC = importlib.util.spec_from_file_location("run_bench_repeated", SCRIPT_PATH)
assert SPEC is not None
repeat = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = repeat
SPEC.loader.exec_module(repeat)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_trajectory(path: Path, payload: dict[str, Any]) -> None:
    event = {
        "direction": "purple_internal",
        "event_type": "PurpleInternalRecord",
        "payload": payload,
    }
    path.write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_events(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )


def test_summarize_result_dir_extracts_agentic_and_wm_metrics(tmp_path: Path) -> None:
    result_dir = tmp_path / "run"
    trajectories = result_dir / "trajectories"
    trajectories.mkdir(parents=True)
    task1_path = trajectories / "task1.jsonl"
    task2_path = trajectories / "task2.jsonl"
    _write_trajectory(
        task1_path,
        {
            "task_id": "task1",
            "execution_time_ms": 1500,
            "tool_results": [
                {"tool_name": "lookup", "result": {"success": True, "error": None}},
                {"tool_name": "bad", "result": {"success": False, "error": "failed"}},
            ],
            "wm_react": {
                "steps": [
                    {
                        "strategy": "beam_plan",
                        "event": "GYM_BEAM_PLAN",
                        "replanned": True,
                        "override_applied": True,
                        "injected": True,
                        "imagined_plan_len": 2,
                        "beam_llm_calls": 7,
                        "beam_plan_terminal_advice_count": 1,
                        "critic": {"fires": True, "world_model_calls": 1},
                    },
                    {
                        "strategy": "beam_plan",
                        "event": "GYM_BEAM_PLAN_COOLDOWN",
                        "replanned": False,
                        "override_applied": False,
                        "critic": {"fires": False, "world_model_calls": 1},
                    },
                ]
            },
        },
    )
    _write_trajectory(
        task2_path,
        {
            "task_id": "task2",
            "execution_time_ms": 500,
            "tool_results": [
                {"tool_name": "ok", "result": {"result": {"isError": False}, "success": True}}
            ],
            "wm_react": {"steps": []},
        },
    )
    _write_json(
        result_dir / "detail.json",
        {
            "benchmark_name": "EnterpriseOps-Gym",
            "benchmark_version": "0.1.0",
            "executor_name": "mcp_react",
            "target": "sample",
            "status": "completed",
            "total_tasks": 2,
            "total_score": 1.0,
            "score_rate": 0.5,
            "duration_seconds": 20.0,
            "avg_verifier_pass_rate": 0.75,
            "details": [
                {"task_id": "task1", "score": 0.0, "trajectory_file_path": str(task1_path)},
                {"task_id": "task2", "score": 1.0, "trajectory_file_path": str(task2_path)},
            ],
        },
    )

    summary = repeat.summarize_result_dir(result_dir)

    assert summary["score_rate"] == 0.5
    assert summary["benchmark_metrics"]["avg_verifier_pass_rate"] == 0.75
    metrics = summary["agentic_task_metrics"]
    assert metrics["avg_task_execution_time_seconds"] == 1.0
    assert metrics["avg_tool_calls"] == 1.5
    assert metrics["avg_failed_tool_calls"] == 0.5
    assert metrics["avg_unnecessary_tool_calls"] == 0.5
    assert metrics["avg_wm_beam_planning_count"] == 0.5
    assert metrics["avg_wm_action_change_count"] == 0.5
    assert metrics["total_wm_beam_planning_count"] == 1
    assert metrics["total_wm_action_change_count"] == 1


def test_effective_command_auto_adds_capture_trajectory() -> None:
    command = ["ejepa", "bench", "run", "EnterpriseOps-Gym", "--executor", "mcp_react"]

    assert repeat.effective_command(command, auto_capture_trajectory=True) == [
        *command,
        "--config",
        "capture_trajectory=true",
    ]
    assert repeat.effective_command(
        [*command, "--config", "capture_trajectory=true"],
        auto_capture_trajectory=True,
    ) == [*command, "--config", "capture_trajectory=true"]


def test_action_feedback_call_metrics_are_summarized() -> None:
    payload = {
        "wm_react": {
            "steps": [
                {
                    "detail": {
                        "strategy": "revision",
                        "event": "action_feedback",
                        "world_model_calls": 1,
                        "judge_calls": 1,
                        "model_calls": 2,
                    }
                }
            ]
        }
    }

    metrics = repeat.wm_metrics_from_payload(payload)

    assert metrics["wm_world_model_call_count"] == 1
    assert metrics["wm_judge_call_count"] == 1
    assert metrics["wm_model_call_count"] == 2
    assert metrics["wm_beam_refinement_round_count"] == 0
    assert metrics["wm_beam_refinement_score_pass_count"] == 0
    assert metrics["wm_revision_count"] == 1
    assert metrics["wm_reference_count"] == 0


def test_beam_refinement_metrics_are_summarized() -> None:
    payload = {
        "wm_react": {
            "steps": [
                {
                    "detail": {
                        "strategy": "beam_plan",
                        "event": "GYM_BEAM_PLAN",
                        "replanned": True,
                        "beam_refinement_rounds_completed": 3,
                        "beam_refinement_score_passes": 3,
                    }
                }
            ]
        }
    }

    metrics = repeat.wm_metrics_from_payload(payload)

    assert metrics["wm_beam_refinement_round_count"] == 3
    assert metrics["wm_beam_refinement_score_pass_count"] == 3


def test_only_steps_that_spent_a_planning_cycle_count_as_replans() -> None:
    # Following an already-cached plan, or having nothing to plan for, spends no planning cycle.
    # Counting those made the critic trigger (rarely follows, often re-plans) look cheaper than
    # the interval trigger (often follows, re-plans on a fixed cadence) -- the reverse of reality.
    payload = {
        "wm_react": {
            "steps": [
                {"detail": {"event": "GYM_BEAM_PLAN", "replanned": True}},
                {"detail": {"event": "GYM_BEAM_PLAN_FOLLOW", "replanned": False}},
                {"detail": {"event": "GYM_BEAM_PLAN_FOLLOW", "replanned": False}},
                {"detail": {"event": "GYM_BEAM_PLAN_COOLDOWN", "replanned": False}},
                {"detail": {"event": "GYM_BEAM_PLAN_NO_SEED"}},
                {"detail": {"event": "GYM_BEAM_PLAN_ERROR"}},
            ]
        }
    }

    metrics = repeat.wm_metrics_from_payload(payload)

    assert metrics["wm_beam_plan_steps"] == 6
    # the completed re-plan plus the attempt that errored out mid-cycle
    assert metrics["wm_beam_planning_count"] == 2


def test_critic_fire_rate_is_pooled_over_checks() -> None:
    per_task = [
        {"wm_critic_check_count": 10, "wm_critic_fire_count": 2},
        {"wm_critic_check_count": 30, "wm_critic_fire_count": 3},
    ]

    metrics = repeat.agentic_task_metrics(per_task)

    assert metrics["wm_critic_fire_rate"] == 0.125  # 5/40, not the mean of 0.2 and 0.1
    assert metrics["avg_wm_critic_check_count"] == 20.0


def test_summarize_shell_protocol_counts_each_executed_command(tmp_path: Path) -> None:
    result_dir = tmp_path / "run"
    trajectories = result_dir / "trajectories"
    trajectories.mkdir(parents=True)
    task_path = trajectories / "shell-task.jsonl"
    _write_events(
        task_path,
        [
            {
                "event_type": "ShellProtocolExecRequest",
                "payload": {"kind": "exec_request", "command": "ls"},
                "command": "ls",
            },
            {
                "event_type": "ShellProtocolExecResult",
                "payload": {"kind": "exec_result", "exit_code": 0},
                "exit_code": 0,
            },
            {
                "event_type": "ShellProtocolExecRequest",
                "payload": {"kind": "exec_request", "command": "false"},
                "command": "false",
            },
            {
                "event_type": "ShellProtocolExecResult",
                "payload": {"kind": "exec_result", "exit_code": 1},
                "exit_code": 1,
            },
        ],
    )
    _write_json(
        result_dir / "detail.json",
        {
            "benchmark_name": "Terminal-Bench-2.0",
            "details": [
                {
                    "task_id": "shell-task",
                    "score": 0.0,
                    "trajectory_file_path": str(task_path),
                }
            ],
        },
    )

    metrics = repeat.summarize_result_dir(result_dir)["agentic_task_metrics"]

    assert metrics["avg_tool_calls"] == 2.0
    assert metrics["avg_failed_tool_calls"] == 1.0
    assert metrics["avg_unnecessary_tool_calls"] == 1.0


def test_aggregate_runs_keeps_benchmark_specific_numeric_metrics() -> None:
    runs = [
        {
            "returncode": 0,
            "wrapper_elapsed_seconds": 10.0,
            "result_summary": {
                "status": "completed",
                "score_rate": 0.2,
                "duration_seconds": 8.0,
                "benchmark_metrics": {"avg_verifier_pass_rate": 0.4},
                "agentic_task_metrics": {"avg_tool_calls": 2.0},
            },
        },
        {
            "returncode": 0,
            "wrapper_elapsed_seconds": 20.0,
            "result_summary": {
                "status": "completed",
                "score_rate": 0.6,
                "duration_seconds": 18.0,
                "benchmark_metrics": {"avg_verifier_pass_rate": 0.8},
                "agentic_task_metrics": {"avg_tool_calls": 4.0},
            },
        },
    ]

    aggregate = repeat.aggregate_runs(runs)

    assert aggregate["score_rate"]["mean"] == 0.4
    assert aggregate["benchmark_metrics"]["avg_verifier_pass_rate"]["mean"] == pytest.approx(0.6)
    assert aggregate["agentic_task_metrics"]["avg_tool_calls"]["mean"] == 3.0


def test_summarize_result_dir_handles_crmarenapro_schema_and_a2a_artifacts(tmp_path: Path) -> None:
    result_dir = (
        tmp_path / "bm-crmarenapro_ex-mcp_react_tg-world_model_test_ts-all_cf-test_us-user_rn-run"
    )
    trajectories = result_dir / "trajectories"
    trajectories.mkdir(parents=True)
    task_path = trajectories / "0.jsonl"
    empty_task_path = trajectories / "1.jsonl"
    _write_trajectory(empty_task_path, {"task_id": "1", "wm_react": {"steps": []}})

    answer_data = {
        "task_id": 0,
        "metrics": {"tool_calls": 2, "failed_queries": 1},
        "wm_strategy": "beam_plan",
        "wm_steps": 3,
    }
    truncated_internal = (
        '{"source": "purple_executor", "payload": {"task_id": "0", "wm_react": {"steps": ['
        '{"event": "GYM_BEAM_PLAN", "replanned": true, "override_applied": true, '
        '"injected": true, "imagined_plan_len": 2, "beam_llm_calls": 7, '
        '"beam_plan_terminal_advice_count": 1, "critic": {"fires": true, "world_model_calls": 1}}'
    )
    events = [
        {
            "event_type": "TaskArtifactUpdateEvent",
            "task": {"artifacts": [{"name": "Answer", "data": answer_data}]},
        },
        {
            "event_type": "TaskStatusUpdateEvent",
            "task": {"artifacts": [{"name": "Answer", "data": answer_data}]},
        },
        {
            "event_type": "TaskArtifactUpdateEvent",
            "task": {
                "artifacts": [
                    {"name": "internal_trajectory", "parts": [{"text": truncated_internal}]}
                ]
            },
        },
    ]
    task_path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8"
    )

    _write_json(
        result_dir / "detail.json",
        {
            "version": "0.1.0",
            "summary": {"pass_rate": 0.5, "total_tasks": 2, "total_passed": 1, "avg_score": 60.0},
            "timing": {"total_seconds": 100.0, "avg_seconds_per_task": 50.0},
            "dimension_averages": {"functional": 0.75},
            "results": [
                {
                    "task_idx": 0,
                    "crm_reward": 1.0,
                    "success": True,
                    "timing": {"purple_agent_seconds": 12.0},
                    "metrics": {"tool_calls": 2, "failed_queries": 1},
                    "trajectory_file_path": str(task_path),
                },
                {
                    "task_idx": 1,
                    "crm_reward": 0.0,
                    "success": False,
                    "timing": {"purple_agent_seconds": 8.0},
                    "metrics": {"queries": 1, "failed_queries": 0},
                    "trajectory_file_path": str(empty_task_path),
                },
            ],
        },
    )

    summary = repeat.summarize_result_dir(result_dir)

    assert summary["benchmark_name"] == "crmarenapro"
    assert summary["executor_name"] == "mcp_react"
    assert summary["target"] == "world_model_test"
    assert summary["status"] == "completed"
    assert summary["total_tasks"] == 2
    assert summary["total_score"] == 1
    assert summary["score_rate"] == 0.5
    assert summary["duration_seconds"] == 100.0
    assert summary["benchmark_metrics"]["dimension_functional"] == 0.75
    assert summary["benchmark_metrics"]["summary_avg_score"] == 60.0

    metrics = summary["agentic_task_metrics"]
    assert metrics["avg_task_execution_time_seconds"] == 10.0
    assert metrics["avg_tool_calls"] == 1.5
    assert metrics["avg_failed_tool_calls"] == 0.5
    assert metrics["avg_wm_beam_planning_count"] == 0.5
    assert metrics["avg_wm_action_change_count"] == 0.5
    assert metrics["total_tool_calls"] == 3
    assert metrics["total_unnecessary_tool_calls"] == 1
    assert metrics["total_wm_beam_planning_count"] == 1
    assert metrics["total_wm_action_change_count"] == 1
