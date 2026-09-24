from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "assets" / "Terminal-Bench-2.0"
GREEN_DIR = BENCHMARK_DIR / "green"
TASK_LOADER_PATH = GREEN_DIR / "task_loader.py"


def _load_task_loader_module():
    if str(GREEN_DIR) not in sys.path:
        sys.path.insert(0, str(GREEN_DIR))
    spec = importlib.util.spec_from_file_location(
        "terminal_bench_task_loader",
        TASK_LOADER_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["terminal_bench_task_loader"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def task_loader(monkeypatch: pytest.MonkeyPatch):
    bundled = BENCHMARK_DIR / "tasks" / "terminal-bench-2"
    monkeypatch.setenv("TERMINAL_BENCH_TASK_REPO", str(bundled.resolve()))
    if not Path(os.environ["TERMINAL_BENCH_TASK_REPO"]).exists():
        pytest.skip("terminal-bench-2 reference repo not available")

    module = _load_task_loader_module()
    return module.TaskLoader(benchmark_dir=BENCHMARK_DIR)


def test_sample_target_includes_fix_git(task_loader) -> None:
    names = task_loader.resolve_task_names(target="sample", requested_ids=None)
    assert names == ["fix-git"]


def test_requested_task_ids_override_target(task_loader) -> None:
    names = task_loader.resolve_task_names(
        target="sample",
        requested_ids=["sanitize-git-repo"],
    )
    assert names == ["sanitize-git-repo"]


def test_world_model_test_target_resolves_from_task_ids_toml() -> None:
    # load_task_ids reads tasks/task_ids.toml directly (no task repo needed), so this
    # runs without the terminal-bench-2 checkout the `task_loader` fixture requires.
    module = _load_task_loader_module()
    loader = module.TaskLoader(benchmark_dir=BENCHMARK_DIR)
    names = loader.load_task_ids("world_model_test")
    assert len(names) == len(set(names)) == 29  # held-out WM test split
    assert "adaptive-rejection-sampler" in names
    assert all(isinstance(name, str) and name for name in names)
