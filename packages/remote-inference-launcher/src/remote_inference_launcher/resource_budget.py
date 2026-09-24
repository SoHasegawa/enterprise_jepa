"""In-process aggregate resource budget coordination."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from remote_inference_launcher.config_types import ResourceBudget


@dataclass(frozen=True)
class ResourceBudgetSnapshot:
    """Current and peak observed resource usage."""

    concurrent_logical_launches: int
    active_candidate_attempts: int
    submitted_slurm_jobs: int
    total_requested_gpus: int
    peak_concurrent_logical_launches: int
    peak_active_candidate_attempts: int
    peak_submitted_slurm_jobs: int
    peak_total_requested_gpus: int

    def to_dict(self) -> dict[str, int]:
        return {
            "concurrent_logical_launches": self.concurrent_logical_launches,
            "active_candidate_attempts": self.active_candidate_attempts,
            "submitted_slurm_jobs": self.submitted_slurm_jobs,
            "total_requested_gpus": self.total_requested_gpus,
            "peak_concurrent_logical_launches": self.peak_concurrent_logical_launches,
            "peak_active_candidate_attempts": self.peak_active_candidate_attempts,
            "peak_submitted_slurm_jobs": self.peak_submitted_slurm_jobs,
            "peak_total_requested_gpus": self.peak_total_requested_gpus,
        }


class ResourceBudgetToken:
    """A held budget reservation that can be released exactly once."""

    def __init__(
        self,
        manager: ResourceBudgetManager | None,
        *,
        logical_launches: int = 0,
        candidate_attempts: int = 0,
        submitted_slurm_jobs: int = 0,
        requested_gpus: int = 0,
    ) -> None:
        self._manager = manager
        self._logical_launches = logical_launches
        self._candidate_attempts = candidate_attempts
        self._submitted_slurm_jobs = submitted_slurm_jobs
        self._requested_gpus = requested_gpus
        self._released = False

    def release(self) -> None:
        """Release this reservation."""

        if self._released:
            return
        self._released = True
        if self._manager is not None:
            self._manager.release(
                logical_launches=self._logical_launches,
                candidate_attempts=self._candidate_attempts,
                submitted_slurm_jobs=self._submitted_slurm_jobs,
                requested_gpus=self._requested_gpus,
            )

    def __enter__(self) -> ResourceBudgetToken:
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


class CompositeResourceBudgetToken:
    """A held reservation across one or more budget managers."""

    def __init__(self, tokens: list[ResourceBudgetToken]) -> None:
        self._tokens = tokens
        self._released = False

    def release(self) -> None:
        """Release all reservations in reverse acquisition order."""

        if self._released:
            return
        self._released = True
        for token in reversed(self._tokens):
            token.release()

    def __enter__(self) -> CompositeResourceBudgetToken:
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


ResourceBudgetReservation = ResourceBudgetToken | CompositeResourceBudgetToken


class ResourceBudgetManager:
    """Coordinate active endpoint attempts against an optional hard budget."""

    def __init__(self, budget: ResourceBudget | None) -> None:
        self.budget = budget or ResourceBudget()
        self._condition = threading.Condition()
        self._concurrent_logical_launches = 0
        self._active_candidate_attempts = 0
        self._submitted_slurm_jobs = 0
        self._total_requested_gpus = 0
        self._peak_concurrent_logical_launches = 0
        self._peak_active_candidate_attempts = 0
        self._peak_submitted_slurm_jobs = 0
        self._peak_total_requested_gpus = 0

    def acquire(
        self,
        *,
        logical_launches: int = 0,
        candidate_attempts: int = 0,
        submitted_slurm_jobs: int = 0,
        requested_gpus: int = 0,
        block: bool = True,
    ) -> ResourceBudgetToken | None:
        """Acquire budget, optionally waiting locally until capacity is available."""

        self._validate_non_negative(
            logical_launches=logical_launches,
            candidate_attempts=candidate_attempts,
            submitted_slurm_jobs=submitted_slurm_jobs,
            requested_gpus=requested_gpus,
        )
        self._validate_request_fits_caps(
            logical_launches=logical_launches,
            candidate_attempts=candidate_attempts,
            submitted_slurm_jobs=submitted_slurm_jobs,
            requested_gpus=requested_gpus,
        )
        with self._condition:
            while not self._fits(
                logical_launches=logical_launches,
                candidate_attempts=candidate_attempts,
                submitted_slurm_jobs=submitted_slurm_jobs,
                requested_gpus=requested_gpus,
            ):
                if not block:
                    return None
                self._condition.wait(timeout=1.0)
            self._concurrent_logical_launches += logical_launches
            self._active_candidate_attempts += candidate_attempts
            self._submitted_slurm_jobs += submitted_slurm_jobs
            self._total_requested_gpus += requested_gpus
            self._record_peaks()
        return ResourceBudgetToken(
            self,
            logical_launches=logical_launches,
            candidate_attempts=candidate_attempts,
            submitted_slurm_jobs=submitted_slurm_jobs,
            requested_gpus=requested_gpus,
        )

    def release(
        self,
        *,
        logical_launches: int = 0,
        candidate_attempts: int = 0,
        submitted_slurm_jobs: int = 0,
        requested_gpus: int = 0,
    ) -> None:
        """Release previously acquired budget."""

        with self._condition:
            self._concurrent_logical_launches = max(
                0, self._concurrent_logical_launches - logical_launches
            )
            self._active_candidate_attempts = max(
                0, self._active_candidate_attempts - candidate_attempts
            )
            self._submitted_slurm_jobs = max(0, self._submitted_slurm_jobs - submitted_slurm_jobs)
            self._total_requested_gpus = max(0, self._total_requested_gpus - requested_gpus)
            self._condition.notify_all()

    def snapshot(self) -> ResourceBudgetSnapshot:
        """Return current and peak observed usage."""

        with self._condition:
            return ResourceBudgetSnapshot(
                concurrent_logical_launches=self._concurrent_logical_launches,
                active_candidate_attempts=self._active_candidate_attempts,
                submitted_slurm_jobs=self._submitted_slurm_jobs,
                total_requested_gpus=self._total_requested_gpus,
                peak_concurrent_logical_launches=self._peak_concurrent_logical_launches,
                peak_active_candidate_attempts=self._peak_active_candidate_attempts,
                peak_submitted_slurm_jobs=self._peak_submitted_slurm_jobs,
                peak_total_requested_gpus=self._peak_total_requested_gpus,
            )

    def _fits(
        self,
        *,
        logical_launches: int,
        candidate_attempts: int,
        submitted_slurm_jobs: int,
        requested_gpus: int,
    ) -> bool:
        checks = (
            (
                self.budget.max_concurrent_logical_launches,
                self._concurrent_logical_launches + logical_launches,
            ),
            (
                self.budget.max_active_candidate_attempts,
                self._active_candidate_attempts + candidate_attempts,
            ),
            (
                self.budget.max_submitted_slurm_jobs,
                self._submitted_slurm_jobs + submitted_slurm_jobs,
            ),
            (
                self.budget.max_total_requested_gpus,
                self._total_requested_gpus + requested_gpus,
            ),
        )
        return all(cap is None or value <= cap for cap, value in checks)

    def _record_peaks(self) -> None:
        self._peak_concurrent_logical_launches = max(
            self._peak_concurrent_logical_launches,
            self._concurrent_logical_launches,
        )
        self._peak_active_candidate_attempts = max(
            self._peak_active_candidate_attempts,
            self._active_candidate_attempts,
        )
        self._peak_submitted_slurm_jobs = max(
            self._peak_submitted_slurm_jobs,
            self._submitted_slurm_jobs,
        )
        self._peak_total_requested_gpus = max(
            self._peak_total_requested_gpus,
            self._total_requested_gpus,
        )

    @staticmethod
    def _validate_non_negative(**values: int) -> None:
        for name, value in values.items():
            if value < 0:
                raise ValueError(f"Resource budget acquisition {name} must be non-negative.")

    def _validate_request_fits_caps(
        self,
        *,
        logical_launches: int,
        candidate_attempts: int,
        submitted_slurm_jobs: int,
        requested_gpus: int,
    ) -> None:
        checks = (
            (
                "max_concurrent_logical_launches",
                self.budget.max_concurrent_logical_launches,
                logical_launches,
            ),
            (
                "max_active_candidate_attempts",
                self.budget.max_active_candidate_attempts,
                candidate_attempts,
            ),
            (
                "max_submitted_slurm_jobs",
                self.budget.max_submitted_slurm_jobs,
                submitted_slurm_jobs,
            ),
            ("max_total_requested_gpus", self.budget.max_total_requested_gpus, requested_gpus),
        )
        for field_name, cap, requested in checks:
            if cap is not None and requested > cap:
                raise ValueError(f"Resource budget request {requested} exceeds {field_name}={cap}.")


_CURRENT_BUDGET_MANAGERS: ContextVar[tuple[ResourceBudgetManager, ...]] = ContextVar(
    "remote_inference_launcher_resource_budget_managers",
    default=(),
)


@contextmanager
def resource_budget_scope(manager: ResourceBudgetManager | None) -> Iterator[None]:
    """Make a resource budget manager visible to nested launcher work."""

    if manager is None:
        yield
        return
    managers = _CURRENT_BUDGET_MANAGERS.get()
    token = _CURRENT_BUDGET_MANAGERS.set((*managers, manager))
    try:
        yield
    finally:
        _CURRENT_BUDGET_MANAGERS.reset(token)


def current_resource_budget_managers() -> tuple[ResourceBudgetManager, ...]:
    """Return all active budget managers from outermost to innermost."""

    return _CURRENT_BUDGET_MANAGERS.get()


def acquire_current_resource_budget(
    *,
    logical_launches: int = 0,
    candidate_attempts: int = 0,
    submitted_slurm_jobs: int = 0,
    requested_gpus: int = 0,
) -> ResourceBudgetReservation:
    """Acquire from the current budget manager, or return a no-op token."""

    managers = current_resource_budget_managers()
    if not managers:
        return ResourceBudgetToken(None)
    tokens: list[ResourceBudgetToken] = []
    try:
        for manager in managers:
            acquired = manager.acquire(
                logical_launches=logical_launches,
                candidate_attempts=candidate_attempts,
                submitted_slurm_jobs=submitted_slurm_jobs,
                requested_gpus=requested_gpus,
            )
            if acquired is not None:
                tokens.append(acquired)
    except BaseException:
        for token in reversed(tokens):
            token.release()
        raise
    return CompositeResourceBudgetToken(tokens)
