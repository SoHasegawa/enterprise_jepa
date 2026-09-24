"""Contract test for the CRMArenaPro `world_model_test` target split.

The green agent (`assets/crmarenapro/green/agent.py`, `_resolve_target_task_ids`)
resolves a named `--config target` to a task-id list in `green/tasks/task_ids.toml`.
That module can't be imported in the shared test venv (its `crm.*` deps live only in
the CRMArenaPro green venv), so this validates the split's data contract and mirrors
the resolver's target rules directly against the file it reads.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_IDS_PATH = REPO_ROOT / "assets" / "crmarenapro" / "green" / "tasks" / "task_ids.toml"


def _resolve(target: str | None) -> list[str] | None:
    """Faithful copy of agent._resolve_target_task_ids's rules against the real file."""
    name = (target or "").strip()
    if not name or name in {"sample", "default"}:
        return None
    if not TASK_IDS_PATH.exists():
        return None
    with TASK_IDS_PATH.open("rb") as handle:
        raw = tomllib.load(handle)
    ids = raw.get(name)
    if not isinstance(ids, list) or not ids:
        return None
    return [str(task_id) for task_id in ids]


def test_world_model_test_split_is_well_formed() -> None:
    with TASK_IDS_PATH.open("rb") as handle:
        raw = tomllib.load(handle)
    ids = raw["world_model_test"]
    assert len(ids) == len(set(ids)) == 428  # held-out B2B WM test split
    # task_id == CRM dataset idx (numeric strings), fetched via get_task_by_idx.
    assert all(str(i).isdigit() for i in ids)


def test_world_model_longest_split_is_a_fixed_subset() -> None:
    with TASK_IDS_PATH.open("rb") as handle:
        raw = tomllib.load(handle)
    ids = raw["world_model_test_longest_100"]
    assert len(ids) == len(set(ids)) == 100
    assert set(ids) < set(raw["world_model_test"])


def test_resolver_rules() -> None:
    # Named split resolves to its ids; default/sample/unknown fall through to None
    # (so existing task_limit sampling is preserved).
    assert _resolve("world_model_test") is not None
    assert len(_resolve("world_model_test")) == 428
    assert _resolve("sample") is None
    assert _resolve("default") is None
    assert _resolve("") is None
    assert len(_resolve("world_model_test_longest_100")) == 100
    assert _resolve(None) is None
    assert _resolve("does_not_exist") is None
