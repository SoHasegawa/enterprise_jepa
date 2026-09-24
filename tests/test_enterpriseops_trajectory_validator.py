from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_validator_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "EnterpriseOps-Gym"
        / "validate_trajectories.py"
    )
    spec = importlib.util.spec_from_file_location("enterpriseops_trajectory_validator", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, lines: list[dict | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in lines:
            if isinstance(item, str):
                handle.write(item)
            else:
                handle.write(json.dumps(item))
            handle.write("\n")


def _valid_events() -> list[dict]:
    return [
        {
            "direction": "outbound",
            "event_type": "Message",
            "payload": {"kind": "message"},
        },
        {
            "direction": "inbound",
            "event_type": "TaskStatusUpdateEvent",
            "event": {"status": {"state": "working"}},
        },
        {
            "direction": "inbound",
            "event_type": "TaskStatusUpdateEvent",
            "task": {"status": {"state": "completed"}},
        },
    ]


def _write_detail(
    detail_path: Path,
    *,
    trajectory_root: Path,
    trajectory_path: Path | None,
    expected_event_count: int | None,
    task_id: str = "task-1",
) -> None:
    payload = {
        "benchmark_name": "EnterpriseOps-Gym",
        "run_id": "run-1",
        "trajectory_capture": {
            "enabled": True,
            "directory": str(trajectory_root),
        },
        "details": [
            {
                "task_id": task_id,
                "trajectory_file_path": str(trajectory_path) if trajectory_path else None,
                "trajectory_event_count": expected_event_count,
            }
        ],
    }
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_index_accepts_valid_referenced_trajectory(tmp_path: Path) -> None:
    module = _load_validator_module()
    search_root = tmp_path / ".cache"
    trajectory_path = search_root / "run" / "experiments" / "trajectories" / "task-1.jsonl"
    _write_jsonl(trajectory_path, _valid_events())
    _write_detail(
        search_root / "run" / "experiments" / "detail.json",
        trajectory_root=trajectory_path.parent,
        trajectory_path=trajectory_path,
        expected_event_count=3,
    )

    summary, issues = module.build_index(
        search_root=search_root,
        index_out=search_root / "index.jsonl",
    )

    assert summary.detail_file_count == 1
    assert summary.expected_trajectory_count == 1
    assert summary.indexed_trajectory_count == 1
    assert summary.valid_trajectory_count == 1
    assert summary.invalid_trajectory_count == 0
    assert summary.issue_count == 0
    assert issues == []
    assert summary.performance.run_count == 1
    assert summary.performance.completed_run_count == 0
    assert summary.performance.task_score_stats.count == 0

    index_records = [json.loads(line) for line in summary.index_path.read_text(encoding="utf-8").splitlines()]
    assert len(index_records) == 1
    assert index_records[0]["path"] == str(trajectory_path.resolve())
    assert index_records[0]["valid"] is True
    assert index_records[0]["orphan"] is False


def test_build_index_reports_missing_invalid_and_orphan_trajectories(tmp_path: Path) -> None:
    module = _load_validator_module()
    search_root = tmp_path / ".cache"
    trajectory_root = search_root / "run" / "experiments" / "trajectories"
    missing_path = trajectory_root / "missing.jsonl"
    invalid_path = trajectory_root / "task-1.jsonl"
    orphan_path = search_root / "run" / "experiments" / "extra" / "trajectories" / "orphan.jsonl"

    _write_jsonl(invalid_path, ['{"direction":"outbound"}', '{"direction":'])
    _write_jsonl(orphan_path, _valid_events())
    _write_detail(
        search_root / "run" / "experiments" / "detail.json",
        trajectory_root=trajectory_root,
        trajectory_path=missing_path,
        expected_event_count=3,
    )

    summary, issues = module.build_index(
        search_root=search_root,
        index_out=search_root / "index.jsonl",
    )

    assert summary.expected_trajectory_count == 1
    assert summary.indexed_trajectory_count == 2
    assert summary.valid_trajectory_count == 1
    assert summary.invalid_trajectory_count == 1
    assert summary.missing_expected_count == 1
    assert summary.orphan_trajectory_count == 2
    assert summary.issue_count >= 4
    assert summary.performance.run_count == 1
    assert any("missing trajectory file" in issue for issue in issues)
    assert any("invalid trajectory file" in issue for issue in issues)
    assert any("orphan trajectory file" in issue for issue in issues)


def test_build_index_computes_performance_stats(tmp_path: Path) -> None:
    module = _load_validator_module()
    search_root = tmp_path / ".cache"
    trajectory_root = search_root / "run" / "experiments" / "trajectories"
    trajectory_path_1 = trajectory_root / "task-1.jsonl"
    trajectory_path_2 = trajectory_root / "task-2.jsonl"
    _write_jsonl(trajectory_path_1, _valid_events())
    _write_jsonl(trajectory_path_2, _valid_events())

    payload = {
        "benchmark_name": "EnterpriseOps-Gym",
        "run_id": "run-1",
        "status": "completed",
        "target": "opsgym_train",
        "duration_seconds": 12.0,
        "total_tasks": 2,
        "total_score": 1.5,
        "score_rate": 0.75,
        "avg_verifier_pass_rate": 0.625,
        "trajectory_capture": {"enabled": True, "directory": str(trajectory_root)},
        "details": [
            {
                "task_id": "task-1",
                "trajectory_file_path": str(trajectory_path_1),
                "trajectory_event_count": 3,
                "score": 1.0,
                "verifier_pass_rate": 0.75,
            },
            {
                "task_id": "task-2",
                "trajectory_file_path": str(trajectory_path_2),
                "trajectory_event_count": 3,
                "score": 0.5,
                "verifier_pass_rate": 0.5,
            },
        ],
    }
    detail_path = search_root / "run" / "experiments" / "detail.json"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(json.dumps(payload), encoding="utf-8")

    summary, _ = module.build_index(
        search_root=search_root,
        index_out=search_root / "index.jsonl",
    )

    assert summary.performance.run_count == 1
    assert summary.performance.completed_run_count == 1
    assert summary.performance.total_score_stats.mean == 1.5
    assert summary.performance.score_rate_stats.mean == 0.75
    assert summary.performance.task_score_stats.count == 2
    assert summary.performance.task_score_stats.mean == 0.75
    assert summary.performance.task_score_stats.stdev == 0.25
    assert summary.performance.task_verifier_pass_rate_stats.mean == 0.625
    assert summary.performance.task_verifier_pass_rate_stats.stdev == 0.125
    assert summary.performance.split_groups["train"].run_count == 1
    assert summary.performance.split_groups["train"].task_score_stats.mean == 0.75
