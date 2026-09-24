from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "summarize_bench_repeat_averages.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_bench_repeat_averages", SCRIPT_PATH)
assert SPEC is not None
averages = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = averages
SPEC.loader.exec_module(averages)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summarize_file_reads_explicit_aggregate_means(tmp_path: Path) -> None:
    summary_path = tmp_path / "ejepa-bench-repeat-sample.json"
    _write_json(
        summary_path,
        {
            "aggregate": {
                "runs_with_result": 3,
                "score_rate": {"mean": 0.5},
                "agentic_task_metrics": {
                    "avg_task_execution_time_seconds": {"mean": 12.5},
                    "avg_tool_calls": {"mean": 4.25},
                    "avg_unnecessary_tool_calls": {"mean": 0.75},
                },
            },
            "runs": [
                {
                    "result_summary": {
                        "benchmark_name": "EnterpriseOps-Gym",
                        "executor_name": "mcp_react",
                        "target": "sample",
                    }
                }
            ],
        },
    )

    row = averages.summarize_file(summary_path)

    assert row == {
        "file": str(summary_path),
        "benchmark": "EnterpriseOps-Gym",
        "executor": "mcp_react",
        "target": "sample",
        "runs_with_result": 3,
        "success_metric": "score_rate",
        "success_rate": 0.5,
        "score_rate": 0.5,
        "inference_time_per_task_seconds": 12.5,
        "avg_tool_calls": 4.25,
        "avg_unnecessary_tool_calls": 0.75,
    }


def test_summarize_file_falls_back_to_run_summaries_and_duration_per_task(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "ejepa-bench-repeat-fallback.json"
    _write_json(
        summary_path,
        {
            "command": [
                "ejepa",
                "bench",
                "run",
                "WorkBench",
                "--executor",
                "mcp_react",
                "--config",
                "target=all",
            ],
            "runs": [
                {
                    "result_summary": {
                        "score_rate": 0.25,
                        "duration_seconds": 10.0,
                        "total_tasks": 5,
                        "agentic_task_metrics": {
                            "avg_tool_calls": 2.0,
                            "avg_unnecessary_tool_calls": 0.0,
                        },
                    }
                },
                {
                    "result_summary": {
                        "score_rate": 0.75,
                        "duration_seconds": 30.0,
                        "total_tasks": 10,
                        "agentic_task_metrics": {
                            "avg_tool_calls": 4.0,
                            "avg_unnecessary_tool_calls": 1.0,
                        },
                    }
                },
            ],
        },
    )

    row = averages.summarize_file(summary_path)

    assert row["benchmark"] == "WorkBench"
    assert row["executor"] == "mcp_react"
    assert row["target"] == "all"
    assert row["runs_with_result"] == 2
    assert row["success_rate"] == 0.5
    assert row["inference_time_per_task_seconds"] == pytest.approx(2.5)
    assert row["avg_tool_calls"] == 3.0
    assert row["avg_unnecessary_tool_calls"] == 0.5


def test_main_prints_csv_for_default_glob(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "out" / "ejepa_bench_repeat_summaries"
    output_dir.mkdir(parents=True)
    _write_json(
        output_dir / "ejepa-bench-repeat-one.json",
        {
            "aggregate": {
                "runs_with_result": 1,
                "score_rate": {"mean": 100.0},
                "agentic_task_metrics": {
                    "avg_task_execution_time_seconds": {"mean": 1.25},
                    "avg_tool_calls": {"mean": 2.0},
                    "avg_unnecessary_tool_calls": {"mean": 0.0},
                },
            }
        },
    )

    exit_code = averages.main(
        [
            "--input-glob",
            str(output_dir / "ejepa-bench-repeat-*.json"),
            "--format",
            "csv",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert (
        "success_metric,success_rate,score_rate,inference_time_per_task_seconds,avg_tool_calls"
        in output
    )
    assert "ejepa-bench-repeat-one.json,,," in output
    assert ",1,score_rate,1,1,1.25,2,0\n" in output


def test_summarize_file_uses_pass_rate_for_automationbench(tmp_path: Path) -> None:
    """AutomationBench score_rate is partial credit; success_rate must be pass_rate."""
    summary_path = tmp_path / "ejepa-bench-repeat-automationbench.json"
    _write_json(
        summary_path,
        {
            "runs": [
                {
                    "result_summary": {
                        "benchmark_name": "AutomationBench",
                        "score_rate": 0.603,  # partial credit -- must NOT be used
                        "benchmark_metrics": {
                            "score_rate": 0.603,
                            "pass_rate": 0.208,
                            "total_passed": 125,
                        },
                    }
                },
                {
                    "result_summary": {
                        "benchmark_name": "AutomationBench",
                        "score_rate": 0.611,
                        "benchmark_metrics": {"score_rate": 0.611, "pass_rate": 0.212},
                    }
                },
            ]
        },
    )

    row = averages.summarize_file(summary_path)

    assert row["success_metric"] == "pass_rate"
    assert row["success_rate"] == pytest.approx((0.208 + 0.212) / 2)


def test_summarize_file_uses_original_accuracy_for_crmarenapro(tmp_path: Path) -> None:
    """crmarenapro grades one run twice; report upstream CRMArena-Pro's own grader."""
    summary_path = tmp_path / "ejepa-bench-repeat-crmarenapro.json"
    _write_json(
        summary_path,
        {
            "runs": [
                {
                    "result_summary": {
                        "benchmark_name": "crmarenapro",
                        "score_rate": 0.369,  # this repo's grader -- must NOT be used
                        "benchmark_metrics": {
                            "score_rate": 0.369,
                            "summary_pass_rate": 0.369,
                            "original_scores_accuracy": 0.313,
                        },
                    }
                }
            ]
        },
    )

    row = averages.summarize_file(summary_path)

    assert row["success_metric"] == "original_accuracy"
    assert row["success_rate"] == pytest.approx(0.313)


def test_summarize_file_derives_task_pass_rate_for_workspace_bench(tmp_path: Path) -> None:
    """Workspace-Bench score_rate is a mean rubric rate, not the task pass rate."""
    summary_path = tmp_path / "ejepa-bench-repeat-workspace-bench.json"
    _write_json(
        summary_path,
        {
            "runs": [
                {
                    "result_summary": {
                        "benchmark_name": "Workspace-Bench",
                        "score_rate": 0.75,  # mean rubric rate -- must NOT be used
                        "per_task": [
                            {"task_id": "1", "score": 1.0, "success": None},
                            {"task_id": "2", "score": 0.5, "success": None},
                            {"task_id": "3", "score": 0.0, "success": None},
                            {"task_id": "4", "score": 1.0, "success": None},
                        ],
                    }
                }
            ]
        },
    )

    row = averages.summarize_file(summary_path)

    assert row["success_metric"] == "task_pass_rate"
    assert row["success_rate"] == pytest.approx(0.5)  # 2 of 4 tasks fully passed


def test_summarize_file_keeps_score_rate_for_other_benchmarks(tmp_path: Path) -> None:
    summary_path = tmp_path / "ejepa-bench-repeat-workbench.json"
    _write_json(
        summary_path,
        {
            "runs": [
                {
                    "result_summary": {
                        "benchmark_name": "WorkBench",
                        "score_rate": 0.59,
                        "per_task": [{"task_id": "1", "score": 0.0, "success": None}],
                    }
                }
            ]
        },
    )

    row = averages.summarize_file(summary_path)

    assert row["success_metric"] == "score_rate"
    assert row["success_rate"] == pytest.approx(0.59)


def test_override_falls_back_with_an_honest_label(tmp_path: Path) -> None:
    """skip_original=true crmarenapro runs have no upstream accuracy to report."""
    summary_path = tmp_path / "ejepa-bench-repeat-crmarenapro-skip.json"
    _write_json(
        summary_path,
        {
            "runs": [
                {
                    "result_summary": {
                        "benchmark_name": "crmarenapro",
                        "score_rate": 0.369,
                        "benchmark_metrics": {"score_rate": 0.369},
                    }
                }
            ]
        },
    )

    row = averages.summarize_file(summary_path)

    assert row["success_rate"] == pytest.approx(0.369)
    assert row["success_metric"] == "score_rate (no original_accuracy)"


def _automationbench_payload() -> dict:
    """One run with two domains, a duplicate row, and an unscorable task."""
    return {
        "runs": [
            {
                "result_summary": {
                    "benchmark_name": "AutomationBench",
                    "score_rate": 0.5,
                    "benchmark_metrics": {"pass_rate": 0.25, "score_rate": 0.5},
                    "per_task": [
                        {
                            "task_id": "sales.a",
                            "score": 1.0,
                            "reason": "4/4 scored assertions satisfied",
                        },
                        {
                            "task_id": "sales.b",
                            "score": 0.5,
                            "reason": "2/4 scored assertions satisfied",
                        },
                        # duplicate of sales.b -- cumulative re-emission, must not double count
                        {
                            "task_id": "sales.b",
                            "score": 0.5,
                            "reason": "2/4 scored assertions satisfied",
                        },
                        {
                            "task_id": "sales.c",
                            "score": 0.0,
                            "reason": "0/0 scored assertions satisfied",
                        },
                        {
                            "task_id": "marketing.x",
                            "score": 0.0,
                            "reason": "0/5 scored assertions satisfied",
                        },
                        {
                            "task_id": "finance.y",
                            "score": 0.0,
                            "reason": "0/3 scored assertions satisfied",
                        },
                    ],
                }
            }
        ]
    }


def test_exclude_domains_adds_filtered_columns(tmp_path: Path) -> None:
    summary_path = tmp_path / "ejepa-bench-repeat-ab.json"
    _write_json(summary_path, _automationbench_payload())

    row = averages.summarize_file(summary_path, ["marketing", "finance"])

    # sales.a (1.0) and sales.b (0.5) survive; sales.c has 0 scorable assertions.
    assert row["filtered_tasks"] == 2
    assert row["filtered_success_rate"] == pytest.approx(0.75)
    assert row["filtered_pass_rate"] == pytest.approx(0.5)
    # the unfiltered headline is untouched
    assert row["success_metric"] == "pass_rate"
    assert row["success_rate"] == pytest.approx(0.25)


def test_no_exclusion_leaves_output_shape_unchanged(tmp_path: Path) -> None:
    summary_path = tmp_path / "ejepa-bench-repeat-ab.json"
    _write_json(summary_path, _automationbench_payload())

    row = averages.summarize_file(summary_path)

    assert not any(field in row for field in averages.FILTERED_FIELDS)
    assert averages.active_fields([row]) == averages.FIELDS


def test_exclusion_is_ignored_without_domain_prefixes(tmp_path: Path) -> None:
    """WorkBench-style ids have no 'domain.' prefix, so nothing should be dropped."""
    summary_path = tmp_path / "ejepa-bench-repeat-wb.json"
    _write_json(
        summary_path,
        {
            "runs": [
                {
                    "result_summary": {
                        "benchmark_name": "WorkBench",
                        "score_rate": 0.5,
                        "per_task": [
                            {"task_id": "email_0019", "score": 1.0},
                            {"task_id": "email_0020", "score": 0.0},
                        ],
                    }
                }
            ]
        },
    )

    row = averages.summarize_file(summary_path, ["marketing"])

    assert row["filtered_tasks"] == 2
    assert row["filtered_success_rate"] == pytest.approx(0.5)
