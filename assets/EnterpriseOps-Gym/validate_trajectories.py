#!/usr/bin/env python3
"""Validate EnterpriseOps-Gym trajectory JSONL files and write an index."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TERMINAL_STATES = {"completed", "failed", "canceled", "cancelled"}


@dataclass(frozen=True)
class ExpectedTrajectory:
    run_id: str
    detail_path: Path
    trajectory_root: Path | None
    task_id: str | None
    trajectory_path: Path | None
    expected_event_count: int | None


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    detail_path: Path
    target: str | None
    status: str | None
    duration_seconds: float | None
    total_tasks: int | None
    total_score: float | None
    score_rate: float | None
    avg_verifier_pass_rate: float | None
    task_scores: list[float]
    task_verifier_pass_rates: list[float]
    task_trajectory_event_counts: list[float]


@dataclass
class TrajectoryValidation:
    path: Path
    valid: bool
    line_count: int
    outbound_count: int
    inbound_count: int
    terminal_states: list[str]
    errors: list[str]


@dataclass
class TrajectoryCounts:
    line_count: int = 0
    outbound_count: int = 0
    inbound_count: int = 0
    terminal_states: list[str] | None = None


@dataclass
class NumericStats:
    count: int
    mean: float | None
    stdev: float | None
    minimum: float | None
    maximum: float | None
    total: float | None


@dataclass
class RunPerformance:
    run_id: str
    split: str | None
    target: str | None
    status: str | None
    duration_seconds: float | None
    total_tasks: int | None
    total_score: float | None
    score_rate: float | None
    avg_verifier_pass_rate: float | None
    task_score_stats: NumericStats
    task_verifier_pass_rate_stats: NumericStats
    task_trajectory_event_count_stats: NumericStats


@dataclass
class PerformanceSummary:
    run_count: int
    completed_run_count: int
    duration_seconds_stats: NumericStats
    total_tasks_stats: NumericStats
    total_score_stats: NumericStats
    score_rate_stats: NumericStats
    avg_verifier_pass_rate_stats: NumericStats
    task_score_stats: NumericStats
    task_verifier_pass_rate_stats: NumericStats
    task_trajectory_event_count_stats: NumericStats
    runs: list[RunPerformance]
    split_groups: dict[str, PerformanceGroup]


@dataclass
class PerformanceGroup:
    split: str
    run_count: int
    completed_run_count: int
    duration_seconds_stats: NumericStats
    total_tasks_stats: NumericStats
    total_score_stats: NumericStats
    score_rate_stats: NumericStats
    avg_verifier_pass_rate_stats: NumericStats
    task_score_stats: NumericStats
    task_verifier_pass_rate_stats: NumericStats
    task_trajectory_event_count_stats: NumericStats


@dataclass
class ValidationSummary:
    search_root: Path
    detail_file_count: int
    expected_trajectory_count: int
    indexed_trajectory_count: int
    valid_trajectory_count: int
    invalid_trajectory_count: int
    missing_expected_count: int
    orphan_trajectory_count: int
    event_count_mismatch_count: int
    misplaced_expected_count: int
    missing_path_count: int
    issue_count: int
    index_path: Path
    performance: PerformanceSummary


@dataclass
class ReconcileStats:
    references_by_path: dict[Path, list[ExpectedTrajectory]]
    issues: list[str]
    event_count_mismatch_count: int = 0
    misplaced_expected_count: int = 0
    missing_expected_count: int = 0
    missing_path_count: int = 0


def _extract_status_state(record: dict[str, Any]) -> str | None:
    for key in ("task", "event"):
        candidate = record.get(key)
        if not isinstance(candidate, dict):
            continue
        status = candidate.get("status")
        if not isinstance(status, dict):
            continue
        state = status.get("state")
        if isinstance(state, str) and state.strip():
            return state.strip().lower()
    return None


def _json_dump_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    handle.write("\n")


def _parse_trajectory_line(text: str, *, line_number: int, errors: list[str]) -> dict[str, Any] | None:
    if not text:
        errors.append(f"line {line_number}: empty line")
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
        return None
    if not isinstance(payload, dict):
        errors.append(f"line {line_number}: top-level JSON value must be an object")
        return None
    return payload


def _update_trajectory_counts(payload: dict[str, Any], counts: TrajectoryCounts) -> None:
    if payload.get("direction") == "outbound":
        counts.outbound_count += 1
    elif payload.get("direction") == "inbound":
        counts.inbound_count += 1

    state = _extract_status_state(payload)
    if state in TERMINAL_STATES:
        terminal_states = counts.terminal_states if counts.terminal_states is not None else []
        terminal_states.append(state)
        counts.terminal_states = terminal_states


def _finalize_trajectory_errors(counts: TrajectoryCounts, errors: list[str]) -> None:
    if counts.line_count == 0:
        errors.append("file is empty")
    if counts.outbound_count == 0:
        errors.append("missing outbound events")
    if counts.inbound_count == 0:
        errors.append("missing inbound events")
    if not counts.terminal_states:
        errors.append("missing terminal task status")


def validate_trajectory_file(path: Path) -> TrajectoryValidation:
    errors: list[str] = []
    counts = TrajectoryCounts(terminal_states=[])

    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                counts.line_count += 1
                payload = _parse_trajectory_line(raw_line.strip(), line_number=line_number, errors=errors)
                if payload is not None:
                    _update_trajectory_counts(payload, counts)
    except OSError as exc:
        errors.append(f"failed to read file: {exc}")

    _finalize_trajectory_errors(counts, errors)

    return TrajectoryValidation(
        path=path,
        valid=not errors,
        line_count=counts.line_count,
        outbound_count=counts.outbound_count,
        inbound_count=counts.inbound_count,
        terminal_states=counts.terminal_states or [],
        errors=errors,
    )


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _compute_numeric_stats(values: list[float]) -> NumericStats:
    if not values:
        return NumericStats(
            count=0,
            mean=None,
            stdev=None,
            minimum=None,
            maximum=None,
            total=None,
        )
    return NumericStats(
        count=len(values),
        mean=statistics.fmean(values),
        stdev=statistics.pstdev(values),
        minimum=min(values),
        maximum=max(values),
        total=sum(values),
    )


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def infer_split(target: str | None) -> str | None:
    if not target:
        return None
    lowered = target.lower()
    if "train" in lowered:
        return "train"
    if "valid" in lowered or "dev" in lowered:
        return "valid"
    if "test" in lowered or "smoke" in lowered:
        return "test"
    return None


def _load_enterpriseops_detail(detail_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(detail_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("benchmark_name") != "EnterpriseOps-Gym":
        return None
    return payload


def _trajectory_root_from_detail(payload: dict[str, Any]) -> Path | None:
    trajectory_capture = payload.get("trajectory_capture")
    if not isinstance(trajectory_capture, dict):
        return None
    directory = trajectory_capture.get("directory")
    if isinstance(directory, str) and directory.strip():
        return Path(directory)
    return None


def _metric_values(items: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in items:
        value = _coerce_float(item.get(key))
        if value is not None:
            values.append(value)
    return values


def _detail_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in payload.get("details") or [] if isinstance(item, dict)]


def _expected_trajectory_from_item(
    *,
    item: dict[str, Any],
    run_id: str,
    detail_path: Path,
    trajectory_root: Path | None,
) -> ExpectedTrajectory:
    path_text = item.get("trajectory_file_path")
    trajectory_path = Path(path_text) if isinstance(path_text, str) and path_text.strip() else None
    return ExpectedTrajectory(
        run_id=run_id,
        detail_path=detail_path,
        trajectory_root=trajectory_root,
        task_id=item.get("task_id") if isinstance(item.get("task_id"), str) else None,
        trajectory_path=trajectory_path,
        expected_event_count=_coerce_int(item.get("trajectory_event_count")),
    )


def _run_record_from_detail(
    *,
    payload: dict[str, Any],
    detail_path: Path,
    run_id: str,
    details: list[dict[str, Any]],
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        detail_path=detail_path,
        target=payload.get("target") if isinstance(payload.get("target"), str) else None,
        status=payload.get("status") if isinstance(payload.get("status"), str) else None,
        duration_seconds=_coerce_float(payload.get("duration_seconds")),
        total_tasks=_coerce_int(payload.get("total_tasks")),
        total_score=_coerce_float(payload.get("total_score")),
        score_rate=_coerce_float(payload.get("score_rate")),
        avg_verifier_pass_rate=_coerce_float(payload.get("avg_verifier_pass_rate")),
        task_scores=_metric_values(details, "score"),
        task_verifier_pass_rates=_metric_values(details, "verifier_pass_rate"),
        task_trajectory_event_counts=_metric_values(details, "trajectory_event_count"),
    )


def load_run_records(search_root: Path) -> tuple[int, list[RunRecord], list[ExpectedTrajectory]]:
    detail_file_count = 0
    runs: list[RunRecord] = []
    expected: list[ExpectedTrajectory] = []
    for detail_path in sorted(search_root.glob("**/detail.json")):
        payload = _load_enterpriseops_detail(detail_path)
        if payload is None:
            continue
        detail_file_count += 1
        run_id = str(payload.get("run_id") or "")
        trajectory_root = _trajectory_root_from_detail(payload)
        details = _detail_items(payload)
        expected.extend(
            _expected_trajectory_from_item(
                item=item,
                run_id=run_id,
                detail_path=detail_path,
                trajectory_root=trajectory_root,
            )
            for item in details
        )
        runs.append(_run_record_from_detail(payload=payload, detail_path=detail_path, run_id=run_id, details=details))
    return detail_file_count, runs, expected


def _present_run_values(runs: list[RunRecord], attr: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        value = getattr(run, attr)
        if value is not None:
            values.append(float(value))
    return values


def _combined_task_values(runs: list[RunRecord], attr: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        values.extend(getattr(run, attr))
    return values


def _build_performance_group(split: str, scoped_runs: list[RunRecord]) -> PerformanceGroup:
    completed_run_count = sum(1 for run in scoped_runs if (run.status or "").lower() == "completed")
    return PerformanceGroup(
        split=split,
        run_count=len(scoped_runs),
        completed_run_count=completed_run_count,
        duration_seconds_stats=_compute_numeric_stats(_present_run_values(scoped_runs, "duration_seconds")),
        total_tasks_stats=_compute_numeric_stats(_present_run_values(scoped_runs, "total_tasks")),
        total_score_stats=_compute_numeric_stats(_present_run_values(scoped_runs, "total_score")),
        score_rate_stats=_compute_numeric_stats(_present_run_values(scoped_runs, "score_rate")),
        avg_verifier_pass_rate_stats=_compute_numeric_stats(
            _present_run_values(scoped_runs, "avg_verifier_pass_rate")
        ),
        task_score_stats=_compute_numeric_stats(_combined_task_values(scoped_runs, "task_scores")),
        task_verifier_pass_rate_stats=_compute_numeric_stats(
            _combined_task_values(scoped_runs, "task_verifier_pass_rates")
        ),
        task_trajectory_event_count_stats=_compute_numeric_stats(
            _combined_task_values(scoped_runs, "task_trajectory_event_counts")
        ),
    )


def _run_performance(run: RunRecord) -> RunPerformance:
    split = infer_split(run.target)
    return RunPerformance(
        run_id=run.run_id,
        split=split,
        target=run.target,
        status=run.status,
        duration_seconds=run.duration_seconds,
        total_tasks=run.total_tasks,
        total_score=run.total_score,
        score_rate=run.score_rate,
        avg_verifier_pass_rate=run.avg_verifier_pass_rate,
        task_score_stats=_compute_numeric_stats(run.task_scores),
        task_verifier_pass_rate_stats=_compute_numeric_stats(run.task_verifier_pass_rates),
        task_trajectory_event_count_stats=_compute_numeric_stats(run.task_trajectory_event_counts),
    )


def _split_run_buckets(runs: list[RunRecord]) -> dict[str, list[RunRecord]]:
    split_buckets: dict[str, list[RunRecord]] = defaultdict(list)
    for run in runs:
        split = infer_split(run.target)
        if split is not None:
            split_buckets[split].append(run)
    return split_buckets


def build_performance_summary(runs: list[RunRecord]) -> PerformanceSummary:
    run_performances = [_run_performance(run) for run in runs]
    split_buckets = _split_run_buckets(runs)
    overall = _build_performance_group("all", runs)
    split_groups = {
        split: _build_performance_group(split, split_buckets.get(split, [])) for split in ("train", "test", "valid")
    }
    for split, scoped_runs in sorted(split_buckets.items()):
        if split not in split_groups:
            split_groups[split] = _build_performance_group(split, scoped_runs)

    return PerformanceSummary(
        run_count=overall.run_count,
        completed_run_count=overall.completed_run_count,
        duration_seconds_stats=overall.duration_seconds_stats,
        total_tasks_stats=overall.total_tasks_stats,
        total_score_stats=overall.total_score_stats,
        score_rate_stats=overall.score_rate_stats,
        avg_verifier_pass_rate_stats=overall.avg_verifier_pass_rate_stats,
        task_score_stats=overall.task_score_stats,
        task_verifier_pass_rate_stats=overall.task_verifier_pass_rate_stats,
        task_trajectory_event_count_stats=overall.task_trajectory_event_count_stats,
        runs=run_performances,
        split_groups=split_groups,
    )


def discover_actual_trajectory_paths(expected: list[ExpectedTrajectory]) -> list[Path]:
    run_roots = sorted({item.detail_path.parent.resolve(strict=False) for item in expected})
    actual_paths: set[Path] = set()
    for run_root in run_roots:
        for path in run_root.glob("**/trajectories/*.jsonl"):
            actual_paths.add(path.resolve())
    for item in expected:
        if item.trajectory_path is not None and item.trajectory_path.exists():
            actual_paths.add(item.trajectory_path.resolve())
    return sorted(actual_paths)


def _record_expected_path_issue(
    *,
    item: ExpectedTrajectory,
    validation: TrajectoryValidation | None,
    expected_path: Path,
    stats: ReconcileStats,
) -> None:
    if item.trajectory_root is not None:
        trajectory_root = item.trajectory_root.resolve(strict=False)
        if trajectory_root not in expected_path.parents:
            stats.misplaced_expected_count += 1
            stats.issues.append(
                "trajectory path outside declared trajectory root: "
                f"{expected_path} (detail={item.detail_path})"
            )

    if validation is None:
        stats.missing_expected_count += 1
        stats.issues.append(
            f"missing trajectory file: {expected_path} (detail={item.detail_path}, task_id={item.task_id or '<unknown>'})"
        )
        return

    if item.expected_event_count is not None and validation.line_count != item.expected_event_count:
        stats.event_count_mismatch_count += 1
        stats.issues.append(
            "trajectory event count mismatch: "
            f"{expected_path} expected={item.expected_event_count} actual={validation.line_count}"
        )


def _reconcile_expected_trajectories(
    expected: list[ExpectedTrajectory],
    validations: dict[Path, TrajectoryValidation],
) -> ReconcileStats:
    stats = ReconcileStats(references_by_path=defaultdict(list), issues=[])
    for item in expected:
        if item.trajectory_path is None:
            stats.missing_path_count += 1
            stats.issues.append(
                f"missing trajectory_file_path in {item.detail_path} task_id={item.task_id or '<unknown>'}"
            )
            continue

        expected_path = item.trajectory_path.resolve(strict=False)
        stats.references_by_path[expected_path].append(item)
        _record_expected_path_issue(
            item=item,
            validation=validations.get(expected_path),
            expected_path=expected_path,
            stats=stats,
        )
    return stats


def _trajectory_index_record(
    *,
    path: Path,
    search_root: Path,
    validation: TrajectoryValidation,
    references: list[ExpectedTrajectory],
) -> dict[str, Any]:
    resolved_search_root = search_root.resolve()
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(resolved_search_root))
        if resolved_search_root in path.parents
        else str(path),
        "valid": validation.valid,
        "line_count": validation.line_count,
        "outbound_count": validation.outbound_count,
        "inbound_count": validation.inbound_count,
        "terminal_states": validation.terminal_states,
        "errors": validation.errors,
        "detail_reference_count": len(references),
        "run_ids": sorted({item.run_id for item in references if item.run_id}),
        "task_ids": sorted({item.task_id for item in references if item.task_id}),
        "expected_event_counts": sorted(
            {item.expected_event_count for item in references if item.expected_event_count is not None}
        ),
        "orphan": not references,
    }


def _write_trajectory_index(
    *,
    search_root: Path,
    index_out: Path,
    actual_paths: list[Path],
    validations: dict[Path, TrajectoryValidation],
    references_by_path: dict[Path, list[ExpectedTrajectory]],
) -> None:
    index_out.parent.mkdir(parents=True, exist_ok=True)
    with index_out.open("w", encoding="utf-8") as handle:
        for path in actual_paths:
            record = _trajectory_index_record(
                path=path,
                search_root=search_root,
                validation=validations[path],
                references=references_by_path.get(path, []),
            )
            _json_dump_line(handle, record)


def _invalid_trajectory_issues(validations: dict[Path, TrajectoryValidation]) -> list[str]:
    issues: list[str] = []
    for validation in validations.values():
        if validation.valid:
            continue
        for error in validation.errors:
            issues.append(f"invalid trajectory file: {validation.path} ({error})")
    return issues


def build_index(
    *,
    search_root: Path,
    index_out: Path,
) -> tuple[ValidationSummary, list[str]]:
    detail_file_count, runs, expected = load_run_records(search_root)
    actual_paths = discover_actual_trajectory_paths(expected)
    actual_set = set(actual_paths)
    validations = {path: validate_trajectory_file(path) for path in actual_paths}
    reconcile = _reconcile_expected_trajectories(expected, validations)

    orphan_paths = sorted(actual_set - set(reconcile.references_by_path))
    for orphan in orphan_paths:
        reconcile.issues.append(f"orphan trajectory file not referenced by detail.json: {orphan}")

    _write_trajectory_index(
        search_root=search_root,
        index_out=index_out,
        actual_paths=actual_paths,
        validations=validations,
        references_by_path=reconcile.references_by_path,
    )

    invalid_trajectory_count = sum(1 for item in validations.values() if not item.valid)
    issues = [*reconcile.issues, *_invalid_trajectory_issues(validations)]
    summary = ValidationSummary(
        search_root=search_root,
        detail_file_count=detail_file_count,
        expected_trajectory_count=len(expected),
        indexed_trajectory_count=len(actual_paths),
        valid_trajectory_count=len(actual_paths) - invalid_trajectory_count,
        invalid_trajectory_count=invalid_trajectory_count,
        missing_expected_count=reconcile.missing_expected_count,
        orphan_trajectory_count=len(orphan_paths),
        event_count_mismatch_count=reconcile.event_count_mismatch_count,
        misplaced_expected_count=reconcile.misplaced_expected_count,
        missing_path_count=reconcile.missing_path_count,
        issue_count=len(issues),
        index_path=index_out,
        performance=build_performance_summary(runs),
    )

    return summary, issues


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate EnterpriseOps-Gym trajectory JSONL files and write an index."
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path(".cache"),
        help="Directory to scan for EnterpriseOps detail.json files and trajectories (default: .cache).",
    )
    parser.add_argument(
        "--index-out",
        type=Path,
        default=Path(".cache/enterpriseops-trajectories/index.jsonl"),
        help="Output JSONL index path (default: .cache/enterpriseops-trajectories/index.jsonl).",
    )
    parser.add_argument(
        "--show-issues",
        type=int,
        default=20,
        help="Maximum number of issues to print to stdout (default: 20).",
    )
    parser.add_argument(
        "--allow-issues",
        action="store_true",
        help="Exit with status 0 even when validation finds issues.",
    )
    return parser.parse_args(argv)


def print_summary(summary: ValidationSummary, issues: list[str], *, show_issues: int) -> None:
    status = "OK" if summary.issue_count == 0 else "FAILED"
    print(f"Validation status: {status}")
    print(f"Search root: {summary.search_root.resolve()}")
    print(f"EnterpriseOps detail.json files: {summary.detail_file_count}")
    print(f"Expected trajectories from detail.json: {summary.expected_trajectory_count}")
    print(f"Indexed trajectory JSONL files: {summary.indexed_trajectory_count}")
    print(f"Valid trajectory files: {summary.valid_trajectory_count}")
    print(f"Invalid trajectory files: {summary.invalid_trajectory_count}")
    print(f"Missing expected trajectory files: {summary.missing_expected_count}")
    print(f"Orphan trajectory files: {summary.orphan_trajectory_count}")
    print(f"Trajectory event count mismatches: {summary.event_count_mismatch_count}")
    print(f"Missing trajectory_file_path entries: {summary.missing_path_count}")
    print(f"Paths outside declared trajectory root: {summary.misplaced_expected_count}")
    print(f"Index written: {summary.index_path.resolve()}")
    print("Performance across all runs:")
    print(
        "  Runs="
        f"{summary.performance.run_count} completed={summary.performance.completed_run_count}"
    )
    print(
        "  Run duration seconds: "
        f"mean={_format_metric(summary.performance.duration_seconds_stats.mean)} "
        f"std={_format_metric(summary.performance.duration_seconds_stats.stdev)} "
        f"min={_format_metric(summary.performance.duration_seconds_stats.minimum)} "
        f"max={_format_metric(summary.performance.duration_seconds_stats.maximum)}"
    )
    print(
        "  Run total tasks: "
        f"mean={_format_metric(summary.performance.total_tasks_stats.mean)} "
        f"std={_format_metric(summary.performance.total_tasks_stats.stdev)} "
        f"min={_format_metric(summary.performance.total_tasks_stats.minimum)} "
        f"max={_format_metric(summary.performance.total_tasks_stats.maximum)}"
    )
    print(
        "  Run total score: "
        f"mean={_format_metric(summary.performance.total_score_stats.mean)} "
        f"std={_format_metric(summary.performance.total_score_stats.stdev)} "
        f"min={_format_metric(summary.performance.total_score_stats.minimum)} "
        f"max={_format_metric(summary.performance.total_score_stats.maximum)}"
    )
    print(
        "  Run score rate: "
        f"mean={_format_metric(summary.performance.score_rate_stats.mean)} "
        f"std={_format_metric(summary.performance.score_rate_stats.stdev)} "
        f"min={_format_metric(summary.performance.score_rate_stats.minimum)} "
        f"max={_format_metric(summary.performance.score_rate_stats.maximum)}"
    )
    print(
        "  Run avg verifier pass rate: "
        f"mean={_format_metric(summary.performance.avg_verifier_pass_rate_stats.mean)} "
        f"std={_format_metric(summary.performance.avg_verifier_pass_rate_stats.stdev)} "
        f"min={_format_metric(summary.performance.avg_verifier_pass_rate_stats.minimum)} "
        f"max={_format_metric(summary.performance.avg_verifier_pass_rate_stats.maximum)}"
    )
    print(
        "  Task score: "
        f"mean={_format_metric(summary.performance.task_score_stats.mean)} "
        f"std={_format_metric(summary.performance.task_score_stats.stdev)} "
        f"min={_format_metric(summary.performance.task_score_stats.minimum)} "
        f"max={_format_metric(summary.performance.task_score_stats.maximum)}"
    )
    print(
        "  Task verifier pass rate: "
        f"mean={_format_metric(summary.performance.task_verifier_pass_rate_stats.mean)} "
        f"std={_format_metric(summary.performance.task_verifier_pass_rate_stats.stdev)} "
        f"min={_format_metric(summary.performance.task_verifier_pass_rate_stats.minimum)} "
        f"max={_format_metric(summary.performance.task_verifier_pass_rate_stats.maximum)}"
    )
    print(
        "  Task trajectory event count: "
        f"mean={_format_metric(summary.performance.task_trajectory_event_count_stats.mean)} "
        f"std={_format_metric(summary.performance.task_trajectory_event_count_stats.stdev)} "
        f"min={_format_metric(summary.performance.task_trajectory_event_count_stats.minimum)} "
        f"max={_format_metric(summary.performance.task_trajectory_event_count_stats.maximum)}"
    )
    print("Performance by split:")
    for split in ("train", "test", "valid"):
        group = summary.performance.split_groups[split]
        print(f"  {split}: runs={group.run_count} completed={group.completed_run_count}")
        print(
            "    "
            f"run_total_score mean={_format_metric(group.total_score_stats.mean)} "
            f"std={_format_metric(group.total_score_stats.stdev)}; "
            f"run_score_rate mean={_format_metric(group.score_rate_stats.mean)} "
            f"std={_format_metric(group.score_rate_stats.stdev)}; "
            f"run_avg_verifier mean={_format_metric(group.avg_verifier_pass_rate_stats.mean)} "
            f"std={_format_metric(group.avg_verifier_pass_rate_stats.stdev)}"
        )
        print(
            "    "
            f"task_score mean={_format_metric(group.task_score_stats.mean)} "
            f"std={_format_metric(group.task_score_stats.stdev)}; "
            f"task_verifier mean={_format_metric(group.task_verifier_pass_rate_stats.mean)} "
            f"std={_format_metric(group.task_verifier_pass_rate_stats.stdev)}; "
            f"task_events mean={_format_metric(group.task_trajectory_event_count_stats.mean)} "
            f"std={_format_metric(group.task_trajectory_event_count_stats.stdev)}"
        )
    print("Performance per run:")
    for run in summary.performance.runs:
        print(
            "  "
            f"{run.run_id or '<unknown>'} "
            f"split={run.split or 'n/a'} "
            f"target={run.target or 'n/a'} status={run.status or 'n/a'} "
            f"tasks={run.total_tasks if run.total_tasks is not None else 'n/a'} "
            f"total_score={_format_metric(run.total_score)} "
            f"score_rate={_format_metric(run.score_rate)} "
            f"avg_verifier={_format_metric(run.avg_verifier_pass_rate)} "
            f"duration_s={_format_metric(run.duration_seconds)}"
        )
        print(
            "    "
            f"task_score mean={_format_metric(run.task_score_stats.mean)} "
            f"std={_format_metric(run.task_score_stats.stdev)}; "
            f"task_verifier mean={_format_metric(run.task_verifier_pass_rate_stats.mean)} "
            f"std={_format_metric(run.task_verifier_pass_rate_stats.stdev)}; "
            f"task_events mean={_format_metric(run.task_trajectory_event_count_stats.mean)} "
            f"std={_format_metric(run.task_trajectory_event_count_stats.stdev)}"
        )
    if issues:
        print("Sample issues:")
        for issue in issues[: max(show_issues, 0)]:
            print(f"  - {issue}")
        remaining = len(issues) - max(show_issues, 0)
        if remaining > 0:
            print(f"  - ... {remaining} more issue(s) not shown")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    summary, issues = build_index(
        search_root=args.search_root,
        index_out=args.index_out,
    )
    print_summary(summary, issues, show_issues=args.show_issues)
    if summary.issue_count > 0 and not args.allow_issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
