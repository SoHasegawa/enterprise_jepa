"""Shared config dataclasses for managed endpoint policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Optional startup diagnostics controls."""

    collect_vllm_capacity: bool = True
    classify_startup_failures: bool = True


@dataclass(frozen=True)
class QueuePolicy:
    """Polling and fallback policy for queued resource attempts."""

    max_pending_seconds: int = 0
    fallback_on_pending: bool = False
    poll_interval_seconds: int = 60
    status_interval_seconds: int = 300


@dataclass(frozen=True)
class CandidateRaceConfig:
    """Candidate racing policy for one logical endpoint."""

    enabled: bool = False
    max_active_candidates: int = 1
    winner_condition: str = "endpoint_ready"
    cancel_losers: bool = True
    launch_stagger_seconds: float = 0.0


@dataclass(frozen=True)
class ResourceBudget:
    """Aggregate launch and owned-resource budget across multi-endpoint launchers."""

    max_concurrent_logical_launches: int | None = None
    max_active_candidate_attempts: int | None = None
    max_submitted_slurm_jobs: int | None = None
    max_total_requested_gpus: int | None = None
