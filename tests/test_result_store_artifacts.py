from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from common.models import BenchmarkRunManifest, EvalResult
from common.result_store import (
    build_execution_identity,
    build_manifest_search_blob,
    iter_result_manifest_paths,
    load_result_manifest,
    resolve_result_root,
    resolve_user_name,
    write_json_file,
    write_result_artifacts,
)


def _manifest(paths, *, status: str = "completed") -> BenchmarkRunManifest:
    completed_at = datetime(2026, 4, 20, 0, 1, tzinfo=UTC)
    return BenchmarkRunManifest(
        run_id=paths.run_id,
        user_name=paths.user_name,
        status=status,
        benchmark_name="ExampleBench",
        benchmark_version="1.2.3",
        green_agent_version="0.2.0",
        purple_agent_version="0.3.0",
        executor_version="0.4.0",
        executor_name="local",
        target="hope",
        task_ids=["TASK_001"],
        task_selection_label=paths.task_selection_label,
        config_hash=paths.config_hash,
        created_at_utc=paths.created_at_utc,
        completed_at_utc=completed_at,
        duration_seconds=60.0,
        result_dir=paths.result_dir,
        detail_file_path=paths.detail_path,
        benchmark_dir=Path("assets/ExampleBench"),
        assets_root=Path("assets"),
        participants={"agent": "http://127.0.0.1:9019"},
        request_config={"target": "hope", "task_ids": ["TASK_001"]},
        score_summary="Total Score: 1.0, Score Rate: 100.00%",
        eval_result=EvalResult(
            target="hope",
            total_tasks=1,
            total_score=1.0,
            score_rate=1.0,
            task_results=[{"task_id": "TASK_001", "score": 1.0}],
        ),
        fatal_error=None,
    )


def test_resolve_result_root_prefers_explicit_then_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_root = tmp_path / "env-results"
    explicit_root = tmp_path / "explicit-results"
    monkeypatch.setenv("BENCHMARK_RESULT_ROOT", str(env_root))

    assert resolve_result_root(explicit_root) == explicit_root.resolve()
    assert resolve_result_root() == env_root.resolve()


def test_resolve_result_root_uses_default_when_env_is_absent(monkeypatch) -> None:
    monkeypatch.delenv("BENCHMARK_RESULT_ROOT", raising=False)

    assert resolve_result_root() == resolve_result_root(None)


def test_resolve_user_name_uses_fallback_when_candidates_are_empty(monkeypatch) -> None:
    monkeypatch.delenv("BENCHMARK_USER_NAME", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("USERNAME", raising=False)
    monkeypatch.setattr("common.result_store.getpass.getuser", lambda: "")

    assert resolve_user_name("") == "unknown"


def test_build_execution_identity_is_stable_for_participant_order_and_url_slashes(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)
    common_kwargs = {
        "benchmark_name": "ExampleBench",
        "executor_name": "local",
        "request_config": {"target": "hope", "task_ids": ["TASK_001", "TASK_002"]},
        "result_root": tmp_path / "experiments",
        "user_name": "tester",
        "created_at_utc": created_at,
    }

    first = build_execution_identity(
        **common_kwargs,
        participants={
            "judge": "http://127.0.0.1:9020/",
            "agent": "http://127.0.0.1:9019/",
        },
    )
    second = build_execution_identity(
        **common_kwargs,
        participants={
            "agent": "http://127.0.0.1:9019",
            "judge": "http://127.0.0.1:9020",
        },
    )

    assert first.config_hash == second.config_hash
    assert first.run_id == second.run_id
    assert first.task_selection_label == "TASK_001__TASK_002"
    assert "_bm-" not in first.result_dir.name
    assert first.result_dir.name.startswith("bm-ExampleBench_ex-local_tg-hope_")


def test_build_execution_identity_handles_empty_and_many_task_labels(tmp_path: Path) -> None:
    created_at = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)

    all_tasks = build_execution_identity(
        benchmark_name="★/with spaces",  # non-ASCII is dropped, spaces become underscores
        executor_name="",
        request_config={"target": "", "task_ids": []},
        participants={},
        result_root=tmp_path / "experiments",
        run_id="run",
        user_name="",
        config_hash="hash",
        created_at_utc=created_at,
    )
    many_tasks = build_execution_identity(
        benchmark_name="ExampleBench",
        executor_name="local",
        request_config={"target": "hope", "task_ids": ["A", "B", "C", "D"]},
        participants={},
        result_root=tmp_path / "experiments",
        run_id="run",
        user_name="tester",
        config_hash="hash",
        created_at_utc=created_at,
    )

    assert all_tasks.task_selection_label == "all"
    assert "bm-with_spaces_ex-executor_tg-default_ts-all_" in all_tasks.result_dir.name
    assert many_tasks.task_selection_label == "A+3"
    assert "_ts-A_3_" in many_tasks.result_dir.name


def test_write_iter_load_and_search_result_artifacts(tmp_path: Path) -> None:
    paths = build_execution_identity(
        benchmark_name="ExampleBench",
        executor_name="local",
        request_config={"target": "hope", "task_ids": ["TASK_001"]},
        participants={"agent": "http://127.0.0.1:9019/"},
        result_root=tmp_path / "experiments",
        run_id="20260420T000000Z-abc123abc123",
        user_name="tester",
        config_hash="abc123abc123",
        created_at_utc=datetime(2026, 4, 20, 0, 0, tzinfo=UTC),
    )
    manifest = _manifest(paths)

    detail_path, manifest_path = write_result_artifacts(
        detail_payload={"status": "completed", "details": [{"task_id": "TASK_001"}]},
        manifest=manifest,
        paths=paths,
    )

    assert detail_path == paths.detail_path
    assert manifest_path == paths.manifest_path
    assert list(iter_result_manifest_paths(paths.result_root)) == [manifest_path]
    loaded = load_result_manifest(manifest_path)
    assert loaded.run_id == "20260420T000000Z-abc123abc123"
    assert loaded.eval_result.task_results == [{"task_id": "TASK_001", "score": 1.0}]
    blob = build_manifest_search_blob(loaded)
    assert "examplebench" in blob
    assert "task_001" in blob


def test_write_json_file_serializes_paths_and_datetimes(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "payload.json"
    when = datetime(2026, 4, 20, 0, 0, tzinfo=UTC)

    write_json_file(output, {"path": tmp_path / "artifact", "when": when})

    payload = output.read_text(encoding="utf-8")
    assert str(tmp_path / "artifact") in payload
    assert "2026-04-20T00:00:00+00:00" in payload


def test_iter_result_manifest_paths_returns_empty_for_missing_root(tmp_path: Path) -> None:
    assert list(iter_result_manifest_paths(tmp_path / "missing")) == []
