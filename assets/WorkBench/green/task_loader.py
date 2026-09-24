"""WorkBench task loader (bundled sample tasks + upstream tasks-and-outcomes CSVs).

Upstream repository: https://github.com/olly-styles/WorkBench
Paper: https://arxiv.org/abs/2405.00823

Unlike some other wrapped benchmarks, WorkBench's dataset is committed inside
its own repository (no separate download step): each domain has a CSV under
``data/processed/tasks_and_outcomes/<domain>_tasks_and_outcomes.csv`` with
columns ``task`` (text), ``outcome`` (a stringified Python list of
ground-truth action-strings), and ``domains``. WorkBench does not assign task
IDs itself (it always evaluates a whole CSV), so this loader synthesizes
stable IDs as ``f"{target}_{index:04d}"`` over the CSV's row order.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import tomllib
from pathlib import Path
from typing import Any

_QUICK_TARGET = "longest_100_balanced"

_SAMPLE_TARGET = "sample"
_ALL_TARGET = "all"
_DATASET_TARGETS = (
    "email",
    "calendar",
    "analytics",
    "project_management",
    "customer_relationship_manager",
    "multi_domain",
)


def _candidate_repo_paths() -> list[Path]:
    """Return candidate paths where the upstream WorkBench repository may live."""
    candidates: list[Path] = []
    explicit = os.getenv("WORKBENCH_REPO_PATH")
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())

    symlink = Path(__file__).resolve().parents[1] / "WorkBench"
    if symlink.exists():
        candidates.append(symlink.resolve())

    benchmark_home = os.getenv("BENCHMARK_HOME")
    if benchmark_home:
        candidates.append((Path(benchmark_home) / "repos" / "WorkBench").expanduser().resolve())

    return candidates


def resolve_repo_path() -> Path:
    """Return the upstream WorkBench repository root (first candidate that exists)."""
    for candidate in _candidate_repo_paths():
        if (candidate / "src" / "evals" / "agent.py").exists():
            return candidate
    raise RuntimeError(
        "WorkBench repository not found. Set WORKBENCH_REPO_PATH, place a symlink at "
        "assets/WorkBench/WorkBench, or clone https://github.com/olly-styles/WorkBench "
        "into ${BENCHMARK_HOME}/repos/."
    )


def _normalize_sample_entry(raw: dict[str, Any]) -> dict[str, Any]:
    task_id = raw.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("Each bundled sample task must have a non-empty string 'id'")
    task_text = raw.get("task")
    if not isinstance(task_text, str) or not task_text.strip():
        raise ValueError(f"Task {task_id} must define non-empty 'task' text")
    outcome = raw.get("outcome")
    if not isinstance(outcome, list):
        raise ValueError(f"Task {task_id} must define 'outcome' as a list of action strings")
    domains = raw.get("domains")
    if not isinstance(domains, list):
        raise ValueError(f"Task {task_id} must define 'domains' as a list")
    return {
        "id": task_id.strip(),
        "task": task_text.strip(),
        "outcome": [str(a) for a in outcome],
        "domains": [str(d) for d in domains],
        "target": str(raw.get("target", _SAMPLE_TARGET)),
    }


def _normalize_dataset_row(target: str, index: int, row: dict[str, str]) -> dict[str, Any]:
    task_text = (row.get("task") or "").strip()
    if not task_text:
        raise ValueError(f"{target} row {index} is missing 'task' text")
    outcome = ast.literal_eval(row.get("outcome") or "[]")
    domains = ast.literal_eval(row.get("domains") or "[]")
    if not isinstance(outcome, list):
        raise ValueError(f"{target} row {index} 'outcome' did not parse to a list")
    if not isinstance(domains, list):
        raise ValueError(f"{target} row {index} 'domains' did not parse to a list")
    return {
        "id": f"{target}_{index:04d}",
        "task": task_text,
        "outcome": [str(a) for a in outcome],
        "domains": [str(d) for d in domains],
        "target": target,
    }


class TaskLoader:
    """Loads the bundled sample task or an upstream WorkBench tasks-and-outcomes CSV."""

    def __init__(
        self,
        tasks_dir: Path | None = None,
        task_ids_path: Path | None = None,
    ) -> None:
        base = Path(__file__).resolve().parent / "tasks"
        self._tasks_dir = tasks_dir or base
        self._task_ids_path = task_ids_path or (base / "task_ids.toml")

    def _load_sample_task_ids(self, target: str) -> list[str]:
        if not self._task_ids_path.exists():
            raise FileNotFoundError(f"task_ids.toml not found: {self._task_ids_path}")
        with self._task_ids_path.open("rb") as fh:
            raw = tomllib.load(fh)
        ids = raw.get(target)
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"No task IDs defined for target '{target}' in {self._task_ids_path}")
        return [str(i) for i in ids]

    def _load_sample_catalog(self) -> dict[str, dict[str, Any]]:
        catalog: dict[str, dict[str, Any]] = {}
        for path in sorted(self._tasks_dir.glob("Tasks_*.json")):
            raw_list = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw_list, list):
                raise ValueError(f"Task file must contain a JSON array: {path}")
            for entry in raw_list:
                normalized = _normalize_sample_entry(entry)
                if normalized["id"] in catalog:
                    raise ValueError(f"Duplicate task ID: {normalized['id']}")
                catalog[normalized["id"]] = normalized
        return catalog

    def _load_dataset_target(self, target: str) -> list[dict[str, Any]]:
        if target not in _DATASET_TARGETS:
            raise ValueError(
                f"Unknown target '{target}'. Expected 'sample' or one of: "
                + ", ".join(_DATASET_TARGETS)
            )
        repo_path = resolve_repo_path()
        csv_path = (
            repo_path
            / "data"
            / "processed"
            / "tasks_and_outcomes"
            / f"{target}_tasks_and_outcomes.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(f"Tasks file not found: {csv_path}")
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [_normalize_dataset_row(target, index, row) for index, row in enumerate(reader)]

    def _load_all_datasets(self) -> list[dict[str, Any]]:
        # Task IDs are already prefixed per-domain (f"{target}_{index:04d}"),
        # so concatenating every real domain's tasks can't collide.
        tasks: list[dict[str, Any]] = []
        for target in _DATASET_TARGETS:
            tasks.extend(self._load_dataset_target(target))
        return tasks

    def load_tasks(
        self,
        target: str,
        requested_task_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if target == _SAMPLE_TARGET:
            task_ids = requested_task_ids or self._load_sample_task_ids(target)
            catalog = self._load_sample_catalog()
            missing = [tid for tid in task_ids if tid not in catalog]
            if missing:
                raise KeyError(f"Task IDs not found in catalog: {', '.join(missing)}")
            return [catalog[tid] for tid in task_ids]

        if target == _QUICK_TARGET:
            tasks = self._load_all_datasets()
            requested_task_ids = requested_task_ids or self._load_sample_task_ids(target)
        else:
            tasks = (
                self._load_all_datasets()
                if target == _ALL_TARGET
                else self._load_dataset_target(target)
            )

        if requested_task_ids:
            catalog = {task["id"]: task for task in tasks}
            missing = [tid for tid in requested_task_ids if tid not in catalog]
            if missing:
                raise KeyError(f"Task IDs not found in target '{target}': {', '.join(missing)}")
            return [catalog[tid] for tid in requested_task_ids]
        return tasks
