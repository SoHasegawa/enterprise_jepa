from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from common.experiments_publisher import (
    _build_metadata_yaml,
    _build_resolved_by_repo,
    _build_results_json,
    _run_dir_name,
    _slug,
    publish_to_experiments,
)
from common.models import BenchmarkRunManifest, EvalResult
from common.storage import (
    BENCHMARK_HOME_ENV,
    default_result_root,
    default_vllm_dtype,
    default_vllm_model_dir,
    default_vllm_openai_sif_path,
    default_vllm_reasoning_parser,
    default_vllm_tool_call_parser,
    default_vllm_trust_remote_code,
    resolve_shared_storage_root,
    vllm_model_slug,
)
from common.uvicorn_utils import reserve_tcp_listener, run_uvicorn_with_socket, write_port_file


def _manifest(tmp_path: Path) -> BenchmarkRunManifest:
    created = datetime(2026, 6, 21, 1, 2, 3, tzinfo=UTC)
    result_dir = tmp_path / "result"
    return BenchmarkRunManifest(
        run_id="run-1",
        user_name="tester",
        status="completed",
        benchmark_name="SWE-Bench",
        benchmark_version="1.0.0",
        green_agent_version="0.2.0",
        purple_agent_version="0.3.0",
        executor_version="0.4.0",
        executor_name="Sample Executor!",
        target="verified",
        task_ids=["org__repo-1", "org__repo-2", "plain-3"],
        task_selection_label="sample",
        config_hash="abc123",
        created_at_utc=created,
        completed_at_utc=created,
        duration_seconds=1.5,
        result_dir=result_dir,
        detail_file_path=result_dir / "detail.json",
        participants={"agent": "http://127.0.0.1:8080/"},
        request_config={"model": "Vendor/Model 1"},
        score_summary="Total Score: 1/3",
        eval_result=EvalResult(
            target="verified",
            total_tasks=3,
            total_score=1.0,
            score_rate=1.0 / 3.0,
            task_results=[
                {"task_id": "org__repo-1", "score": 1.0},
                {"task_id": "org__repo-2", "score": 0.0},
                {"task_id": "plain-3", "score": 0.0, "reason": "no patch"},
            ],
        ),
    )


def test_experiments_publisher_builds_results_and_metadata(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    results = _build_results_json(manifest, {"task_results": None})

    assert results["resolved"] == ["org__repo-1"]
    assert results["unresolved"] == ["org__repo-2"]
    assert results["no_generation"] == ["plain-3"]
    assert _build_resolved_by_repo(results) == {
        "org/repo": {"resolved": 1, "total": 2},
        "plain": {"resolved": 0, "total": 1},
    }
    assert _slug("  a/b  c!!  ") == "a_b_c"
    assert _run_dir_name(manifest) == "20260621_Sample_Executor_Vendor_Model_1"

    metadata = _build_metadata_yaml(
        manifest,
        submitted_by="qa",
        org="ExampleOrg",
        extra_info={"name": "Custom Name", "attempts": 2},
    )
    assert 'name: "Custom Name"' in metadata
    assert 'attempts: "2"' in metadata


def test_publish_to_experiments_writes_expected_files(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    run_dir = publish_to_experiments(
        manifest=manifest,
        detail={},
        experiments_root=tmp_path / "experiments",
        submitted_by="qa",
        agent_readme="# Custom\n",
        run_dir_name="manual-run",
    )

    assert run_dir == tmp_path / "experiments" / "evaluation" / "SWE-Bench" / "manual-run"
    assert json.loads((run_dir / "results" / "results.json").read_text())["total_instances"] == 3
    assert (run_dir / "README.md").read_text(encoding="utf-8") == "# Custom\n"


def test_storage_defaults_and_model_specific_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(BENCHMARK_HOME_ENV, str(tmp_path / "shared"))
    assert resolve_shared_storage_root() == (tmp_path / "shared").resolve()
    assert default_result_root() == (tmp_path / "shared" / "experiments").resolve()
    assert vllm_model_slug(" Vendor/Model 1!! ") == "vendor-model-1"
    assert default_vllm_reasoning_parser("Qwen/Qwen3.5-27B") == "qwen3"
    assert default_vllm_tool_call_parser("google/gemma-4-31B") == "gemma4"
    assert default_vllm_dtype("google/gemma-4-31B") == "bfloat16"
    assert default_vllm_trust_remote_code("google/gemma-4-31B") is True
    assert default_vllm_model_dir("vendor/model", tmp_path) == tmp_path / "models" / "vendor/model"
    assert default_vllm_openai_sif_path("new/model", tmp_path).name == "vllm-openai-new-model.sif"


def test_uvicorn_helpers_write_port_and_use_reserved_socket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    listener, port = reserve_tcp_listener("127.0.0.1", 0)
    try:
        assert port > 0
        port_file = tmp_path / "nested" / "port.txt"
        write_port_file(port_file, port)
        assert port_file.read_text(encoding="utf-8") == f"{port}\n"

        calls = []

        class _FakeServer:
            def __init__(self, config) -> None:
                calls.append(("config", config.host, config.port))

            def run(self, *, sockets) -> None:
                calls.append(("run", sockets))

        monkeypatch.setattr("common.uvicorn_utils.uvicorn.Server", _FakeServer)
        run_uvicorn_with_socket(object(), host="127.0.0.1", port=port, listener=listener)
        assert calls[0] == ("config", "127.0.0.1", port)
        assert calls[1] == ("run", [listener])
    finally:
        listener.close()
