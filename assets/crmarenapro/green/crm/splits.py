"""Fixed train/valid/test splits for crmarenapro.

Mirrors EnterpriseOps-Gym's opsgym_80_test design:
- ``crmarena_100_test``: fixed held-out ids committed in tasks/task_ids.toml
  (identical to the seed-42 ``task_limit=100`` sample used by the baseline).
- ``crmarena_train`` / ``crmarena_valid``: derived deterministically from the
  remaining corpus — exclude every test id, order by sha256 of the string id,
  first 10% (rounded, min 1) -> valid, rest -> train.

Both derived splits therefore can never overlap the test set (contamination
guard is still asserted explicitly in :func:`resolve_split_task_ids`).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

TASK_IDS_TOML = Path(__file__).resolve().parents[1] / "tasks" / "task_ids.toml"
TEST_TARGET = "crmarena_100_test"
# world_model_test: the 428 held-out B2B tasks used as the EWM (world-model) EVAL set,
# disjoint from the EWM training trajectories. A fixed split (like TEST_TARGET), listed
# explicitly in tasks/task_ids.toml rather than derived.
WM_TEST_TARGET = "world_model_test"
WM_LONGEST_TARGET = "world_model_test_longest_100"
SPLIT_TARGETS = (
    TEST_TARGET,
    "crmarena_train",
    "crmarena_valid",
    WM_TEST_TARGET,
    WM_LONGEST_TARGET,
)
VALID_FRACTION = 0.1


def is_split_target(target: str | None) -> bool:
    return target in SPLIT_TARGETS


def _load_ids(key: str) -> list[int]:
    with TASK_IDS_TOML.open("rb") as fh:
        data = tomllib.load(fh)
    if key not in data:
        raise ValueError(f"split {key!r} is not defined in {TASK_IDS_TOML}")
    ids = data[key]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ids in {TASK_IDS_TOML}:{key}")
    return list(ids)


def _load_test_ids() -> list[int]:
    return _load_ids(TEST_TARGET)


def _hash_order(ids: list[int]) -> list[int]:
    return sorted(ids, key=lambda i: hashlib.sha256(str(i).encode()).hexdigest())


def resolve_split_task_ids(target: str, corpus_ids: list[int]) -> list[int]:
    """Return the task ids belonging to ``target`` given the full corpus."""
    if target not in SPLIT_TARGETS:
        raise ValueError(f"unknown split target: {target!r} (expected one of {SPLIT_TARGETS})")

    corpus = set(corpus_ids)

    # world_model_test is an independent fixed split (the EWM held-out eval ids).
    if target in {WM_TEST_TARGET, WM_LONGEST_TARGET}:
        wm_ids = _load_ids(target)
        missing = [i for i in wm_ids if i not in corpus]
        if missing:
            raise ValueError(
                f"{len(missing)} {WM_TEST_TARGET} ids missing from corpus (e.g. {missing[:3]})"
            )
        return wm_ids

    test_ids = _load_test_ids()
    missing = [i for i in test_ids if i not in corpus]
    if missing:
        raise ValueError(f"{len(missing)} test ids missing from corpus (e.g. {missing[:3]})")

    if target == TEST_TARGET:
        return test_ids

    remaining = [i for i in corpus_ids if i not in set(test_ids)]
    ordered = _hash_order(remaining)
    valid_count = max(1, round(len(ordered) * VALID_FRACTION)) if ordered else 0
    valid_ids = set(ordered[:valid_count])

    if target == "crmarena_valid":
        selected = [i for i in corpus_ids if i in valid_ids]
    else:  # crmarena_train
        selected = [i for i in corpus_ids if i not in valid_ids and i not in set(test_ids)]

    # Contamination guard: derived splits must never touch the test set.
    overlap = set(selected) & set(test_ids)
    if overlap:
        raise AssertionError(f"split {target} contaminated by test ids: {sorted(overlap)[:5]}")

    logger.info(
        "Resolved split %s: %d tasks (corpus=%d, test=%d, valid=%d)",
        target,
        len(selected),
        len(corpus_ids),
        len(test_ids),
        valid_count,
    )
    return selected
