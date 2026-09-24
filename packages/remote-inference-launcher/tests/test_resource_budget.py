from __future__ import annotations

import pytest

from remote_inference_launcher.config_types import ResourceBudget
from remote_inference_launcher.resource_budget import (
    ResourceBudgetManager,
    acquire_current_resource_budget,
    resource_budget_scope,
)


def test_budget_rejects_single_request_that_can_never_fit() -> None:
    manager = ResourceBudgetManager(ResourceBudget(max_total_requested_gpus=1))

    with pytest.raises(ValueError, match="max_total_requested_gpus=1"):
        manager.acquire(requested_gpus=2)


def test_budget_allows_waiting_request_that_can_fit_later() -> None:
    manager = ResourceBudgetManager(ResourceBudget(max_total_requested_gpus=2))

    token = manager.acquire(requested_gpus=2)
    assert manager.acquire(requested_gpus=1, block=False) is None

    token.release()
    waiting_token = manager.acquire(requested_gpus=1, block=False)

    assert waiting_token is not None
    waiting_token.release()


def test_empty_nested_budget_scope_preserves_outer_budget() -> None:
    manager = ResourceBudgetManager(ResourceBudget(max_active_candidate_attempts=1))

    with resource_budget_scope(manager), resource_budget_scope(None):
        token = acquire_current_resource_budget(candidate_attempts=1)
        try:
            assert manager.snapshot().active_candidate_attempts == 1
        finally:
            token.release()

    assert manager.snapshot().active_candidate_attempts == 0


def test_explicit_nested_budget_scope_enforces_outer_and_inner_budgets() -> None:
    outer = ResourceBudgetManager(ResourceBudget(max_active_candidate_attempts=1))
    inner = ResourceBudgetManager(ResourceBudget(max_active_candidate_attempts=1))

    with resource_budget_scope(outer), resource_budget_scope(inner):
        token = acquire_current_resource_budget(candidate_attempts=1)
        try:
            assert outer.snapshot().active_candidate_attempts == 1
            assert inner.snapshot().active_candidate_attempts == 1
        finally:
            token.release()

    assert outer.snapshot().active_candidate_attempts == 0
    assert inner.snapshot().active_candidate_attempts == 0
