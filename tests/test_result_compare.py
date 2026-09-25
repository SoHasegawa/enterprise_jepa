from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from common.result_store import build_execution_identity
from ejepa_cli import cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def plain_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress Rich ANSI control sequences."""
    monkeypatch.setattr(
        cli,
        "console",
        Console(force_terminal=False, color_system=None, width=220),
    )


def _write_manifest(
    result_root: Path,
    *,
    run_id: str,
    task_results: list[dict[str, Any]],
    benchmark_name: str = "TDI-AgenticSearch",
    benchmark_version: str | None = "1.0.0",
    green_agent_version: str | None = "0.1.0",
    purple_agent_version: str | None = "0.1.0",
    executor_version: str | None = "0.1.0",
    executor_name: str = "doc_search",
    target: str = "hope",
    user_name: str = "tester",
) -> Path:
    """Write the minimal manifest.json and detail.json that compare needs."""
    config_hash = run_id.split("-")[-1]
    result_paths = build_execution_identity(
        benchmark_name=benchmark_name,
        executor_name=executor_name,
        request_config={"target": target, "task_ids": []},
        participants={"agent": "http://127.0.0.1:9019/"},
        result_root=result_root,
        run_id=run_id,
        user_name=user_name,
        config_hash=config_hash,
    )
    result_dir = result_paths.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    total_score = sum(float(task_result.get("score", 0.0)) for task_result in task_results)
    total_tasks = len(task_results)
    score_rate = total_score / total_tasks if total_tasks else 0.0
    task_ids = [str(task_result["task_id"]) for task_result in task_results]
    payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "user_name": user_name,
        "status": "completed",
        "benchmark_name": benchmark_name,
        "benchmark_version": benchmark_version,
        "green_agent_version": green_agent_version,
        "purple_agent_version": purple_agent_version,
        "executor_version": executor_version,
        "executor_name": executor_name,
        "target": target,
        "task_ids": task_ids,
        "task_selection_label": "all",
        "config_hash": config_hash,
        "created_at_utc": "2026-04-20T00:00:00Z",
        "completed_at_utc": "2026-04-20T00:01:00Z",
        "duration_seconds": 60.0,
        "result_dir": str(result_dir),
        "detail_file_path": str(result_dir / "detail.json"),
        "benchmark_dir": str(result_dir / "benchmark"),
        "assets_root": str(result_dir / "assets"),
        "participants": {
            "agent": "http://127.0.0.1:9019/",
        },
        "request_config": {
            "target": target,
            "task_ids": task_ids,
        },
        "score_summary": f"Total Score: {total_score}, Score Rate: {score_rate:.2%}",
        "eval_result": {
            "target": target,
            "total_tasks": total_tasks,
            "total_score": total_score,
            "score_rate": score_rate,
            "task_results": task_results,
        },
        "fatal_error": None,
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (result_dir / "detail.json").write_text("{}", encoding="utf-8")
    return result_dir


def test_result_store_builds_user_scoped_directory_name(tmp_path: Path) -> None:
    """The run directory name includes the user name."""
    paths = build_execution_identity(
        benchmark_name="TDI-AgenticSearch",
        executor_name="doc_search",
        request_config={"target": "hope", "task_ids": []},
        participants={"agent": "http://127.0.0.1:9019/"},
        result_root=tmp_path / "experiments",
        run_id="20260420T000016Z-aaaaaaaaaaaa",
        user_name="alice",
        config_hash="aaaaaaaaaaaa",
    )

    assert paths.user_name == "alice"
    assert "_us-alice_" in paths.result_dir.name


def test_result_list_shows_user_name_column(tmp_path: Path) -> None:
    """`ejepa result list` shows the user who ran it."""
    result_root = tmp_path / "experiments"
    _write_manifest(
        result_root,
        run_id="20260420T000017Z-bbbbbbbbbbbb",
        user_name="alice",
        task_results=[
            {"task_id": "TASK_001", "score": 1.0, "eval_func": "exact_match"},
        ],
    )

    result = runner.invoke(
        cli.app,
        [
            "--result-root",
            str(result_root),
            "result",
            "list",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "User" in result.output
    assert "alice" in result.output


def test_result_show_displays_component_versions(tmp_path: Path) -> None:
    """`ejepa result show` displays green / purple / executor versions."""
    result_root = tmp_path / "experiments"
    run_id = "20260420T000018Z-ccccbbbbbbbb"
    _write_manifest(
        result_root,
        run_id=run_id,
        user_name="alice",
        green_agent_version="0.2.0",
        purple_agent_version="0.3.0",
        executor_version="1.4.5",
        task_results=[
            {"task_id": "TASK_001", "score": 1.0, "eval_func": "exact_match"},
        ],
    )

    result = runner.invoke(
        cli.app,
        [
            "--result-root",
            str(result_root),
            "result",
            "show",
            run_id,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Green Ver   : 0.2.0" in result.output
    assert "Purple Ver  : 0.3.0" in result.output
    assert "Exec Ver    : 1.4.5" in result.output


def test_result_compare_outputs_pass_fail_matrix_and_pass_at_k(tmp_path: Path) -> None:
    """The PASS/FAIL matrix over several runs and PASS@k appear in the JSON."""
    result_root = tmp_path / "experiments"
    run_id_1 = "20260420T000001Z-aaaaaaaaaaaa"
    run_id_2 = "20260420T000002Z-bbbbbbbbbbbb"
    _write_manifest(
        result_root,
        run_id=run_id_1,
        task_results=[
            {"task_id": "TASK_001", "score": 1.0, "eval_func": "exact_match"},
            {"task_id": "TASK_002", "score": 0.0, "eval_func": "exact_match"},
        ],
    )
    _write_manifest(
        result_root,
        run_id=run_id_2,
        task_results=[
            {"task_id": "TASK_001", "score": 0.0, "eval_func": "exact_match"},
        ],
    )

    result = runner.invoke(
        cli.app,
        [
            "--result-root",
            str(result_root),
            "result",
            "compare",
            run_id_1,
            run_id_2,
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["benchmark_name"] == "TDI-AgenticSearch"
    assert payload["target"] == "hope"
    assert payload["has_partial_coverage"] is True

    tasks_by_id = {task["task_id"]: task for task in payload["tasks"]}
    assert [entry["status"] for entry in tasks_by_id["TASK_001"]["runs"]] == [
        "PASS",
        "FAIL",
    ]
    assert [entry["status"] for entry in tasks_by_id["TASK_002"]["runs"]] == [
        "FAIL",
        "MISSING",
    ]

    pass_at_k = {entry["k"]: entry for entry in payload["pass_at_k"]}
    assert pass_at_k[1]["score"] == pytest.approx(0.25)
    assert pass_at_k[1]["eligible_tasks"] == 2
    assert pass_at_k[2]["score"] == pytest.approx(1.0)
    assert pass_at_k[2]["eligible_tasks"] == 1


def test_result_compare_honors_pass_threshold(tmp_path: Path) -> None:
    """`--pass-threshold` switches the PASS/FAIL decision."""
    result_root = tmp_path / "experiments"
    run_id_1 = "20260420T000003Z-cccccccccccc"
    run_id_2 = "20260420T000004Z-dddddddddddd"
    _write_manifest(
        result_root,
        run_id=run_id_1,
        task_results=[
            {"task_id": "TASK_001", "score": 0.7, "eval_func": "partial_match"},
        ],
    )
    _write_manifest(
        result_root,
        run_id=run_id_2,
        task_results=[
            {"task_id": "TASK_001", "score": 0.4, "eval_func": "partial_match"},
        ],
    )

    result = runner.invoke(
        cli.app,
        [
            "--result-root",
            str(result_root),
            "result",
            "compare",
            run_id_1,
            run_id_2,
            "--pass-threshold",
            "0.5",
            "--k",
            "1",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    task_record = payload["tasks"][0]
    assert [entry["status"] for entry in task_record["runs"]] == ["PASS", "FAIL"]
    assert payload["pass_at_k"][0]["score"] == pytest.approx(0.5)


def test_result_compare_rejects_mixed_targets(tmp_path: Path) -> None:
    """Comparing runs with different targets is rejected."""
    result_root = tmp_path / "experiments"
    run_id_1 = "20260420T000005Z-eeeeeeeeeeee"
    run_id_2 = "20260420T000006Z-ffffffffffff"
    _write_manifest(
        result_root,
        run_id=run_id_1,
        task_results=[
            {"task_id": "TASK_001", "score": 1.0, "eval_func": "exact_match"},
        ],
        target="hope",
    )
    _write_manifest(
        result_root,
        run_id=run_id_2,
        task_results=[
            {"task_id": "TASK_001", "score": 1.0, "eval_func": "exact_match"},
        ],
        target="micjet",
    )

    result = runner.invoke(
        cli.app,
        [
            "--result-root",
            str(result_root),
            "result",
            "compare",
            run_id_1,
            run_id_2,
            "--format",
            "json",
        ],
    )

    assert result.exit_code != 0
    assert "same target" in result.output


def test_result_compare_can_select_runs_by_query(tmp_path: Path) -> None:
    """A query alone can select the set of runs to compare."""
    result_root = tmp_path / "experiments"
    run_id_1 = "20260420T000010Z-444444444444"
    run_id_2 = "20260420T000011Z-555555555555"
    run_id_3 = "20260420T000012Z-666666666666"
    _write_manifest(
        result_root,
        run_id=run_id_1,
        executor_name="doc_search",
        task_results=[
            {"task_id": "TASK_001", "score": 1.0, "eval_func": "exact_match"},
        ],
    )
    _write_manifest(
        result_root,
        run_id=run_id_2,
        executor_name="doc_search",
        task_results=[
            {"task_id": "TASK_001", "score": 0.0, "eval_func": "exact_match"},
        ],
    )
    _write_manifest(
        result_root,
        run_id=run_id_3,
        executor_name="mcp_react",
        task_results=[
            {"task_id": "TASK_001", "score": 1.0, "eval_func": "exact_match"},
        ],
    )

    result = runner.invoke(
        cli.app,
        [
            "--result-root",
            str(result_root),
            "result",
            "compare",
            "--query",
            "doc_search",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_count"] == 2
    assert {run["run_id"] for run in payload["runs"]} == {run_id_1, run_id_2}


def test_result_pass_at_k_outputs_summary_json(tmp_path: Path) -> None:
    """The dedicated command returns only the PASS@k summary as JSON."""
    result_root = tmp_path / "experiments"
    run_id_1 = "20260420T000007Z-111111111111"
    run_id_2 = "20260420T000008Z-222222222222"
    _write_manifest(
        result_root,
        run_id=run_id_1,
        task_results=[
            {"task_id": "TASK_001", "score": 1.0, "eval_func": "exact_match"},
            {"task_id": "TASK_002", "score": 0.0, "eval_func": "exact_match"},
        ],
    )
    _write_manifest(
        result_root,
        run_id=run_id_2,
        task_results=[
            {"task_id": "TASK_001", "score": 1.0, "eval_func": "exact_match"},
            {"task_id": "TASK_002", "score": 1.0, "eval_func": "exact_match"},
        ],
    )

    result = runner.invoke(
        cli.app,
        [
            "--result-root",
            str(result_root),
            "result",
            "pass-at-k",
            run_id_1,
            run_id_2,
            "--k",
            "1",
            "--k",
            "2",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_count"] == 2
    assert "tasks" not in payload
    pass_at_k = {entry["k"]: entry for entry in payload["pass_at_k"]}
    assert pass_at_k[1]["score"] == pytest.approx(0.75)
    assert pass_at_k[2]["score"] == pytest.approx(1.0)


def test_result_pass_at_k_can_select_runs_by_filters_without_duplicates(
    tmp_path: Path,
) -> None:
    """Mixing filters and explicit identifiers does not duplicate runs."""
    result_root = tmp_path / "experiments"
    run_id_1 = "20260420T000013Z-777777777777"
    run_id_2 = "20260420T000014Z-888888888888"
    run_id_3 = "20260420T000015Z-999999999999"
    _write_manifest(
        result_root,
        run_id=run_id_1,
        executor_name="doc_search",
        target="hope",
        task_results=[
            {"task_id": "TASK_001", "score": 1.0, "eval_func": "exact_match"},
            {"task_id": "TASK_002", "score": 0.0, "eval_func": "exact_match"},
        ],
    )
    _write_manifest(
        result_root,
        run_id=run_id_2,
        executor_name="doc_search",
        target="hope",
        task_results=[
            {"task_id": "TASK_001", "score": 1.0, "eval_func": "exact_match"},
            {"task_id": "TASK_002", "score": 1.0, "eval_func": "exact_match"},
        ],
    )
    _write_manifest(
        result_root,
        run_id=run_id_3,
        executor_name="doc_search",
        target="micjet",
        task_results=[
            {"task_id": "TASK_001", "score": 1.0, "eval_func": "exact_match"},
            {"task_id": "TASK_002", "score": 1.0, "eval_func": "exact_match"},
        ],
    )

    result = runner.invoke(
        cli.app,
        [
            "--result-root",
            str(result_root),
            "result",
            "pass-at-k",
            run_id_1,
            "--executor",
            "doc_search",
            "--target",
            "hope",
            "--k",
            "1",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_count"] == 2
    assert {run["run_id"] for run in payload["runs"]} == {run_id_1, run_id_2}
    assert payload["pass_at_k"][0]["score"] == pytest.approx(0.75)


def test_result_pass_at_k_allows_single_run(tmp_path: Path) -> None:
    """The dedicated command computes PASS@1 even from a single run."""
    result_root = tmp_path / "experiments"
    run_id = "20260420T000009Z-333333333333"
    _write_manifest(
        result_root,
        run_id=run_id,
        task_results=[
            {"task_id": "TASK_001", "score": 1.0, "eval_func": "exact_match"},
            {"task_id": "TASK_002", "score": 0.0, "eval_func": "exact_match"},
        ],
    )

    result = runner.invoke(
        cli.app,
        [
            "--result-root",
            str(result_root),
            "result",
            "pass-at-k",
            run_id,
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_count"] == 1
    assert len(payload["pass_at_k"]) == 1
    assert payload["pass_at_k"][0]["k"] == 1
    assert payload["pass_at_k"][0]["score"] == pytest.approx(0.5)
