"""Terminal-Bench 2.0 task selection helpers."""

from __future__ import annotations

import os
from pathlib import Path
import tomllib


class TaskLoader:
    """Resolve task names from ejepa config (target / task_ids) and the task repo."""

    def __init__(
        self,
        *,
        benchmark_dir: Path | None = None,
        task_ids_path: Path | None = None,
    ) -> None:
        self._benchmark_dir = benchmark_dir or Path(__file__).resolve().parents[1]
        self._task_ids_path = task_ids_path or (
            Path(__file__).resolve().parent / "tasks" / "task_ids.toml"
        )

    def task_repo_dir(self) -> Path:
        """Return the local terminal-bench-2 checkout used by Harbor."""
        explicit = (
            os.getenv("TERMINAL_BENCH_TASK_REPO")
            or os.getenv("TERMINAL_BENCH_2_TASK_REPO")
        )
        if explicit:
            return Path(explicit).expanduser().resolve()

        benchmark_home = os.getenv("BENCHMARK_HOME")
        if benchmark_home:
            candidate = (
                Path(benchmark_home).expanduser()
                / "repos"
                / "terminal-bench-2"
            )
            if candidate.exists():
                return candidate.resolve()

        bundled = self._benchmark_dir / "tasks" / "terminal-bench-2"
        if bundled.exists():
            return bundled.resolve()

        raise FileNotFoundError(
            "terminal-bench-2 task repo not found. Clone "
            "https://github.com/laude-institute/terminal-bench-2 into "
            f"{bundled} or set TERMINAL_BENCH_TASK_REPO."
        )

    def list_all_task_names(self) -> list[str]:
        repo = self.task_repo_dir()
        return sorted(
            entry.name
            for entry in repo.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )

    def load_task_ids(self, target: str) -> list[str]:
        if target == "all":
            return self.list_all_task_names()
        if not self._task_ids_path.exists():
            raise FileNotFoundError(f"task_ids.toml not found: {self._task_ids_path}")
        with self._task_ids_path.open("rb") as handle:
            raw = tomllib.load(handle)
        ids = raw.get(target)
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"No task IDs defined for target '{target}'")
        return [str(task_id) for task_id in ids]

    def resolve_task_names(
        self,
        *,
        target: str,
        requested_ids: list[str] | None = None,
    ) -> list[str]:
        if requested_ids:
            return [str(task_id) for task_id in requested_ids]
        return self.load_task_ids(target)
