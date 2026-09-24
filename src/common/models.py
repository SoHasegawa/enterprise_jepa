from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class EvalRequest(BaseModel):
    """Green に渡す評価要求を表す。"""

    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class EvalResult(BaseModel):
    """ベンチマーク全体の集計結果を表す。"""

    target: str
    total_tasks: int
    total_score: float
    score_rate: float
    task_results: list[dict[str, Any]]


class RuntimeFeedbackRequest(BaseModel):
    """Purple generation の online reward を Green へ問い合わせる要求。"""

    kind: Literal["runtime_feedback"] = "runtime_feedback"
    target: str
    task_id: str
    generation: int
    answer: str


class RuntimeFeedbackResponse(BaseModel):
    """Green が返す generation 単位の scalar reward。"""

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
    """実行構成から導出した結果保存先を表す。"""

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
    """1 回のベンチマーク実行結果を表すメタデータ。"""

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
    """detail.json の共通スキーマ。

    すべてのベンチマークが出力する detail.json はこのモデルを満たす必要がある。
    ベンチマーク固有のフィールド（trajectory_capture、self_evolution など）は
    extra='allow' により追加フィールドとして許容される。

    必須フィールド:
        schema_version : フォーマットバージョン。現在は "1.0"。
        status         : "completed" または "failed"。
        benchmark_name : benchmark.toml の name と一致すること。
        executor_name  : 使用した executor 名。
        total_tasks    : 評価したタスク数。
        total_score    : スコアの合計値。
        score_rate     : total_score / total_tasks（0.0〜1.0）。
        details        : タスクごとの詳細リスト。各要素の構造はベンチマーク固有で可。

    後方互換性:
        extra='allow' のため、既存ベンチマークが出力する追加フィールドは
        ValidationError を起こさずそのまま受け入れられる。
        既存ベンチマーク側の修正は不要。
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
