"""Task loading for AutomationBench.

Two task sources, mirroring the other wrapped benchmarks in this repo:

* ``sample`` -- the bundled ``tasks/Tasks_SAMPLE.json``. Self-contained, needs no
  upstream checkout, and is what the offline dry-run test exercises.
* a domain target (``simple``, ``sales``, ..., ``all_domains``) -- loaded from the
  upstream repository through its own ``automationbench.domains.get_combined_dataset``,
  so task text, initial state, per-task tool allow-lists and assertions are always
  upstream's, never a copy that can drift.

Upstream task rows carry ``prompt`` (the trigger message), ``info.initial_state``
(a ``WorldState`` dict), ``info.assertions`` (ground truth -- Green keeps these and
never forwards them to Purple), ``info.zapier_tools`` (per-task allow-list for the
``limited_zapier`` toolset) and ``info.task_name``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_TASKS_DIR = Path(__file__).resolve().parent / "tasks"
_SAMPLE_PATH = _TASKS_DIR / "Tasks_SAMPLE.json"
_SAMPLE_TARGET = "sample"
_ALL_DOMAINS_TARGET = "all_domains"
_ALL_WITH_SIMPLE_TARGET = "all_with_simple"
# Upstream's PUBLIC_DOMAINS / DEFAULT_DOMAINS: the six business domains, 100 tasks
# each. This is what upstream's own "all domains" evaluation covers (600 tasks).
_PUBLIC_DOMAINS = (
    "sales",
    "marketing",
    "operations",
    "support",
    "finance",
    "hr",
)
# `simple` is upstream's separate 200-task foundational/diagnostic domain; it is
# registered only when importable and is deliberately NOT part of PUBLIC_DOMAINS,
# so it is selectable on its own but excluded from `all_domains`.
_EXTRA_DOMAINS = ("simple",)
_DOMAIN_TARGETS = (*_PUBLIC_DOMAINS, *_EXTRA_DOMAINS)
_BENCHMARK_DIR = Path(__file__).resolve().parents[1]


def _candidate_repo_paths() -> list[Path]:
    """Where the upstream AutomationBench checkout may live."""
    candidates: list[Path] = []
    explicit = os.getenv("AUTOMATIONBENCH_REPO_PATH")
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    link = _BENCHMARK_DIR / "AutomationBench"
    if link.exists():
        candidates.append(link.resolve())
    benchmark_home = os.getenv("BENCHMARK_HOME")
    if benchmark_home:
        candidates.append(
            (Path(benchmark_home) / "repos" / "AutomationBench").expanduser().resolve()
        )
    candidates.append((_BENCHMARK_DIR.parents[1] / "tools" / "AutomationBench").resolve())
    return candidates


def resolve_repo_path() -> Path:
    """Return the upstream repository root (first candidate that looks right)."""
    for candidate in _candidate_repo_paths():
        if (candidate / "automationbench" / "domains").is_dir():
            return candidate
    raise RuntimeError(
        "AutomationBench repository not found. Clone "
        "https://github.com/zapier/AutomationBench and set AUTOMATIONBENCH_REPO_PATH, "
        "or use --config target=sample for the bundled smoke tasks."
    )


def ensure_upstream_importable() -> Path:
    """Put the upstream repo on ``sys.path`` so ``automationbench`` imports resolve."""
    import sys

    repo = resolve_repo_path()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def _normalize_sample_entry(index: int, raw: dict[str, Any]) -> dict[str, Any]:
    info = dict(raw.get("info") or {})
    task_id = str(raw.get("id") or info.get("task_name") or f"sample_{index:04d}")
    return {
        "id": task_id,
        "name": str(info.get("task_name") or task_id),
        "domain": str(info.get("domain") or "sample"),
        "prompt": str(raw.get("prompt") or ""),
        "initial_state": dict(info.get("initial_state") or {}),
        "assertions": list(info.get("assertions") or []),
        "zapier_tools": list(info.get("zapier_tools") or []),
        "source": "bundled_sample",
    }


def _row_info(row: dict[str, Any]) -> dict[str, Any]:
    """Upstream stores ``info`` either as a dict or as a JSON string."""
    info = row.get("info")
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            info = {}
    return dict(info or {})


def _row_prompt(row: dict[str, Any]) -> str:
    """Flatten upstream's ``prompt`` (a chat message list) into trigger text."""
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        chunks: list[str] = []
        for message in prompt:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    chunks.append(content)
        return "\n\n".join(chunks)
    return ""


def _normalize_dataset_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    info = _row_info(row)
    name = str(info.get("task_name") or row.get("task") or f"task_{index:04d}")
    return {
        "id": name,
        "name": name,
        "domain": str(info.get("domain") or name.split(".", 1)[0]),
        "prompt": _row_prompt(row),
        "initial_state": dict(info.get("initial_state") or {}),
        "assertions": list(info.get("assertions") or []),
        "zapier_tools": list(info.get("zapier_tools") or []),
        "source": "upstream_dataset",
    }


class TaskLoader:
    """Load AutomationBench tasks for a target, optionally filtered by task id."""

    def load_tasks(
        self, target: str, requested_task_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        if target == _SAMPLE_TARGET:
            tasks = self._load_sample()
        elif target == _ALL_DOMAINS_TARGET:
            tasks = self._load_domains(list(_PUBLIC_DOMAINS))
        elif target == _ALL_WITH_SIMPLE_TARGET:
            tasks = self._load_domains([*_PUBLIC_DOMAINS, *_EXTRA_DOMAINS])
        elif target in _DOMAIN_TARGETS:
            tasks = self._load_domains([target])
        else:
            valid = [
                _SAMPLE_TARGET,
                *_DOMAIN_TARGETS,
                _ALL_DOMAINS_TARGET,
                _ALL_WITH_SIMPLE_TARGET,
            ]
            raise ValueError(f"Unknown target {target!r}. Valid targets: {valid}")

        if requested_task_ids:
            wanted = {str(task_id) for task_id in requested_task_ids}
            tasks = [task for task in tasks if task["id"] in wanted]
            missing = wanted - {task["id"] for task in tasks}
            if missing:
                raise ValueError(f"Unknown task ids for target {target!r}: {sorted(missing)}")
        if not tasks:
            raise ValueError(f"No AutomationBench tasks loaded for target {target!r}")
        return tasks

    def _load_sample(self) -> list[dict[str, Any]]:
        raw = json.loads(_SAMPLE_PATH.read_text(encoding="utf-8"))
        entries = raw.get("tasks") if isinstance(raw, dict) else raw
        return [
            _normalize_sample_entry(index, entry)
            for index, entry in enumerate(entries or [])
            if isinstance(entry, dict)
        ]

    def _load_domains(self, domains: list[str]) -> list[dict[str, Any]]:
        ensure_upstream_importable()
        from automationbench.domains import get_combined_dataset

        dataset = get_combined_dataset(domains)
        return [_normalize_dataset_row(index, dict(row)) for index, row in enumerate(dataset)]
