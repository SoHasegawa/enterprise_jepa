from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "summarize_wm_harness_summaries.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("summarize_wm_harness_summaries", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_file_skips_rows_without_rate_or_latency(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "ejepa-wm-harnesses-demo-20260908T000000Z.json"
    path.write_text(
        json.dumps(
            {
                "created_at_utc": "2026-09-08T00:00:00Z",
                "base_command": [
                    "ejepa",
                    "bench",
                    "run",
                    "DemoBench",
                    "--executor",
                    "mcp_react",
                    "--config",
                    "target=lite",
                ],
                "runs": [
                    {
                        "harness": "itp_i",
                        "returncode": 0,
                        "result_dir": "/tmp/result-a",
                        "result_summary": {
                            "benchmark_name": "DemoBench",
                            "executor_name": "mcp_react",
                            "target": "lite",
                            "status": "completed",
                            "score_rate": 0.42,
                            "total_tasks": 10,
                            "agentic_task_metrics": {
                                "avg_task_execution_time_seconds": 3.5,
                            },
                        },
                    },
                    {
                        "harness": "beam_interval",
                        "returncode": None,
                        "result_dir": None,
                    },
                ],
                "comparison": [
                    {
                        "harness": "itp_i",
                        "score_rate": 0.42,
                        "avg_task_execution_time_seconds": 3.5,
                    },
                    {
                        "harness": "beam_interval",
                        "score_rate": None,
                        "avg_task_execution_time_seconds": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    rows, skipped = module.summarize_file(path)

    assert skipped == 1
    assert len(rows) == 1
    assert rows[0]["harness"] == "itp_i"
    assert rows[0]["success_rate_pct"] == 42.0
    assert rows[0]["latency_per_task_seconds"] == 3.5


def test_summarize_file_derives_latency_from_duration_and_task_count(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "ejepa-wm-harnesses-demo-20260908T000000Z.json"
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "harness": "revision",
                        "result_summary": {
                            "score_rate": 75.0,
                            "duration_seconds": 50.0,
                            "total_tasks": 5,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows, skipped = module.summarize_file(path)

    assert skipped == 0
    assert rows[0]["success_rate_pct"] == 75.0
    assert rows[0]["latency_per_task_seconds"] == 10.0


def test_summarize_file_uses_task_pass_rate_for_workspace_bench(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "ejepa-wm-harnesses-workspace-bench-20260909T000000Z.json"
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "harness": "itp_i",
                        "result_summary": {
                            "benchmark_name": "Workspace-Bench",
                            "score_rate": 0.75,
                            "total_tasks": 4,
                            "agentic_task_metrics": {
                                "avg_task_execution_time_seconds": 2.0,
                            },
                            "per_task": [
                                {"task_id": "1", "score": 1.0, "success": None},
                                {"task_id": "2", "score": 0.5, "success": None},
                                {"task_id": "3", "score": 0.0, "success": None},
                                {"task_id": "4", "score": 1.0, "success": None},
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows, skipped = module.summarize_file(path)

    assert skipped == 0
    assert rows[0]["succeeded_tasks"] == 2
    assert rows[0]["success_rate_pct"] == 50.0


def test_summarize_file_skips_workspace_bench_rows_without_per_task(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "ejepa-wm-harnesses-workspace-bench-20260909T000000Z.json"
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "harness": "itp_i",
                        "result_summary": {
                            "benchmark_name": "Workspace-Bench",
                            "score_rate": 0.75,
                            "total_tasks": 4,
                            "agentic_task_metrics": {
                                "avg_task_execution_time_seconds": 2.0,
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows, skipped = module.summarize_file(path)

    assert rows == []
    assert skipped == 1


def test_summarize_file_uses_pass_rate_for_automationbench(tmp_path: Path) -> None:
    """AutomationBench score_rate is partial credit; the success rate is pass_rate.

    A task satisfying 4 of 5 assertions contributes 0.8 to score_rate but is NOT a
    success, so reporting score_rate overstates the success rate substantially.
    """
    module = _load_script()
    path = tmp_path / "ejepa-wm-harnesses-automationbench-20260916T000000Z.json"
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "harness": "itp_i",
                        "result_summary": {
                            "benchmark_name": "AutomationBench",
                            "score_rate": 0.609,  # partial credit -- must NOT be used
                            "total_tasks": 600,
                            "agentic_task_metrics": {
                                "avg_task_execution_time_seconds": 52.0,
                            },
                            "benchmark_metrics": {
                                "score_rate": 0.609,
                                "pass_rate": 0.175,
                                "total_passed": 105,
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows, skipped = module.summarize_file(path)

    assert skipped == 0
    assert rows[0]["success_metric"] == "pass_rate"
    assert rows[0]["succeeded_tasks"] == 105
    assert rows[0]["success_rate_pct"] == 17.5
    assert rows[0]["total_tasks"] == 600


def test_summarize_file_uses_original_accuracy_for_crmarenapro(tmp_path: Path) -> None:
    """crmarenapro grades one run twice; report upstream CRMArena-Pro's own grader."""
    module = _load_script()
    path = tmp_path / "ejepa-wm-harnesses-crmarenapro-20260916T000000Z.json"
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "harness": "beam_critic",
                        "result_summary": {
                            "benchmark_name": "crmarenapro",
                            "score_rate": 0.369,  # this repo's own grader -- not used
                            "total_tasks": 428,
                            "agentic_task_metrics": {
                                "avg_task_execution_time_seconds": 100.0,
                            },
                            "benchmark_metrics": {
                                "score_rate": 0.369,
                                "summary_pass_rate": 0.369,
                                "original_scores_accuracy": 0.313,
                                "original_scores_accuracy_percent": 31.31,
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows, skipped = module.summarize_file(path)

    assert skipped == 0
    assert rows[0]["success_metric"] == "original_accuracy"
    assert rows[0]["success_rate_pct"] == 31.3
    # No pass count is published for the upstream grader, so it is derived.
    assert rows[0]["succeeded_tasks"] == 134


def test_other_benchmarks_keep_using_score_rate(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "ejepa-wm-harnesses-workbench-20260916T000000Z.json"
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "harness": "itp_i",
                        "result_summary": {
                            "benchmark_name": "WorkBench",
                            "score_rate": 0.59,
                            "total_tasks": 100,
                            "agentic_task_metrics": {
                                "avg_task_execution_time_seconds": 3.0,
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows, skipped = module.summarize_file(path)

    assert skipped == 0
    assert rows[0]["success_metric"] == "score_rate"
    assert rows[0]["success_rate_pct"] == 59.0
