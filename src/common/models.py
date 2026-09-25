from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class EvalRequest(BaseModel):
    """Evaluation request handed to the Green agent."""

    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class EvalResult(BaseModel):
    """Aggregate result over a whole benchmark."""

    target: str
    total_tasks: int
    total_score: float
    score_rate: float
    task_results: list[dict[str, Any]]


class RuntimeFeedbackRequest(BaseModel):
    """Request asking Green for the online reward of one Purple generation."""

    kind: Literal["runtime_feedback"] = "runtime_feedback"
    target: str
    task_id: str
    generation: int
    answer: str


class RuntimeFeedbackResponse(BaseModel):
    """Per-generation scalar reward returned by Green."""

    kind: Literal["runtime_feedback_response"] = "runtime_feedback_response"
    target: str
    task_id: str
    generation: int
    score: float
    reward: float
    all_passed: bool
    precision: float
    recall: float
    matched_count: int
    detected_count: int
    labeled_count: int
    matching_mode: str
    reason: str
    eval_result: dict[str, Any]
    benchmark_detail: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRunPaths(BaseModel):
    """Result destination derived from the run configuration."""

    result_root: Path
    result_dir: Path
    manifest_path: Path
    detail_path: Path
    run_id: str
    user_name: str
    config_hash: str
    task_selection_label: str
    created_at_utc: datetime


class BenchmarkRunManifest(BaseModel):
    """Metadata describing a single benchmark run."""

    schema_version: str = "1.0"
    run_id: str
    user_name: str | None = None
    status: str
    benchmark_name: str
    benchmark_version: str | None = None
    green_agent_version: str | None = None
    purple_agent_version: str | None = None
    executor_version: str | None = None
    executor_name: str
    target: str
    task_ids: list[str]
    task_selection_label: str
    config_hash: str
    created_at_utc: datetime
    completed_at_utc: datetime
    duration_seconds: float
    result_dir: Path
    detail_file_path: Path
    benchmark_dir: Path | None = None
    assets_root: Path | None = None
    participants: dict[str, str]
    request_config: dict[str, Any]
    score_summary: str
    eval_result: EvalResult
    fatal_error: str | None = None


class BenchmarkRunDetail(BaseModel):
    """Shared schema for detail.json.

    Every benchmark's detail.json must satisfy this model. Benchmark-specific fields
    (trajectory_capture, self_evolution, ...) are accepted as extra fields through
    extra='allow'.

    Required fields:
        schema_version : format version; currently "1.0".
        status         : "completed" or "failed".
        benchmark_name : must match the name in benchmark.toml.
        executor_name  : the executor that was used.
        total_tasks    : number of tasks evaluated.
        total_score    : sum of the scores.
        score_rate     : total_score / total_tasks（0.0〜1.0）。
        details        : per-task detail list; each entry may be benchmark-specific.

    Backwards compatibility:
        Because of extra='allow', additional fields emitted by existing benchmarks
        are accepted as-is rather than raising ValidationError, so no benchmark
        needs changing.
    """

    model_config = {"extra": "allow"}

    schema_version: str = "1.0"
    status: str
    benchmark_name: str
    executor_name: str
    total_tasks: int
    total_score: float
    score_rate: float
    fatal_error: str | None = None
    details: list[dict[str, Any]]
