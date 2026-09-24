from __future__ import annotations

import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from common.inference_runtime import (
    attach_reproducibility_metadata,
    managed_inference,
    prepare_inference_runtime_config,
    write_inference_env_file,
)
from ejepa_cli import cli

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = REPO_ROOT / "assets"


@dataclass(frozen=True)
class FakeSession:
    name: str
    api_base: str
    served_model_name: str
    model: str = ""
    api_key: str = ""
    local_port: int | None = None
    logs: str = ""
    run_id: str = "20260603-153012-a3f91c2b"
    summary_path: str = "/tmp/inference-summary.json"


@dataclass
class FakeLauncher:
    session: FakeSession
    stopped: bool = False

    def start(self) -> FakeSession:
        return self.session

    def stop(self) -> None:
        self.stopped = True


class _FakeInferenceContext:
    def __init__(
        self,
        lifecycle: dict[str, bool],
        *,
        served_model_name: str = "served/model",
        recommended_max_parallel: int | None = None,
        cleanup_error: Exception | None = None,
    ) -> None:
        self._lifecycle = lifecycle
        self._served_model_name = served_model_name
        self._recommended_max_parallel = recommended_max_parallel
        self._cleanup_error = cleanup_error
        self.sessions = {
            "default": SimpleNamespace(
                api_base="http://127.0.0.1:8123/v1",
                served_model_name=served_model_name,
                metadata={},
            ),
        }

    def __enter__(self) -> _FakeInferenceContext:
        self._lifecycle["entered"] = True
        return self

    def __exit__(self, *_args: object) -> None:
        self._lifecycle["exited"] = True
        if self._cleanup_error is not None:
            raise self._cleanup_error

    def apply_to(self, env: dict[str, str]) -> None:
        env.update(
            {
                "OPENAI_BASE_URL": "http://127.0.0.1:8123/v1",
                "OPENAI_MODEL_NAME": self._served_model_name,
                "INFERENCE_DEFAULT_BASE_URL": "http://127.0.0.1:8123/v1",
                "INFERENCE_DEFAULT_MODEL": self._served_model_name,
            }
        )

    def payload(self) -> dict[str, object]:
        return _valid_inference_handoff(
            served_model_name=self._served_model_name,
            summary_path="/tmp/inference-launch-summary.json",
        )

    def recommended_max_parallel(self) -> int | None:
        return self._recommended_max_parallel


def test_managed_inference_runtime_sets_only_generic_env(monkeypatch, tmp_path: Path) -> None:
    import remote_inference_launcher.inference_config as inference_config

    launcher = FakeLauncher(
        FakeSession(
            name="actor",
            api_base="http://127.0.0.1:8123/v1",
            served_model_name="actor-model",
            api_key="secret",
            local_port=8123,
            logs="/tmp/logs",
        )
    )
    monkeypatch.setattr(inference_config, "load_inference_launcher", lambda _path: launcher)
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        "kind: existing_endpoint\n"
        "api_base: http://127.0.0.1:8123/v1\n"
        "served_model_name: actor-model\n",
        encoding="utf-8",
    )

    env: dict[str, str] = {}
    with managed_inference(config_path) as runtime:
        runtime.apply_to(env)
        assert runtime.sessions["actor"] is launcher.session
        handoff = runtime.payload()

    assert launcher.stopped
    assert env["INFERENCE_ACTOR_BASE_URL"] == "http://127.0.0.1:8123/v1"
    assert env["INFERENCE_ACTOR_MODEL"] == "actor-model"
    assert env["INFERENCE_ACTOR_API_KEY"] == "secret"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8123/v1"
    assert env["OPENAI_MODEL_NAME"] == "actor-model"
    assert env["OPENAI_API_KEY"] == "secret"
    assert "SIMPLETASKS_LLM_BASE_URL" not in env
    assert "MMMU_LLM_BASE_URL" not in env
    assert "NO_PROXY" in env
    assert handoff["schema_version"] == "ril-handoff/v1"
    assert handoff["identity_key"] == "run:20260603-153012-a3f91c2b"
    assert handoff["endpoints"]["actor"]["summary_path"] == "/tmp/inference-summary.json"


def test_absolute_cli_path_resolves_relative_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "sessions" / "wm"
    checkpoint.mkdir(parents=True)

    assert cli._absolute_cli_path("sessions/wm") == str(checkpoint.resolve())
    assert cli._absolute_cli_path("  ") is None
    assert cli._absolute_cli_path(None) is None


def test_managed_inference_runtime_stops_launcher_on_start_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import remote_inference_launcher.inference_config as inference_config

    class FailingLauncher:
        stopped = False

        def start(self) -> object:
            raise RuntimeError("boom")

        def stop(self) -> None:
            self.stopped = True

    launcher = FailingLauncher()
    monkeypatch.setattr(inference_config, "load_inference_launcher", lambda _path: launcher)
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        "kind: existing_endpoint\napi_base: http://127.0.0.1:8123/v1\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="boom"), managed_inference(config_path):
        pass

    assert launcher.stopped


def test_managed_inference_runtime_reports_cleanup_failure_after_start_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import remote_inference_launcher.inference_config as inference_config

    class FailingLauncher:
        def start(self) -> object:
            raise RuntimeError("startup failed")

        def stop(self) -> None:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(
        inference_config, "load_inference_launcher", lambda _path: FailingLauncher()
    )
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        "kind: existing_endpoint\napi_base: http://127.0.0.1:8123/v1\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="cleanup also failed"), managed_inference(config_path):
        pass


def test_write_inference_env_file_quotes_values(tmp_path: Path) -> None:
    env_file = tmp_path / "nested" / "inference.env"

    write_inference_env_file(
        env_file,
        {
            "OPENAI_BASE_URL": "http://127.0.0.1:8123/v1",
            "OPENAI_MODEL_NAME": "model'with-quote",
        },
    )

    content = env_file.read_text(encoding="utf-8")
    assert "export OPENAI_BASE_URL='http://127.0.0.1:8123/v1'" in content
    assert """export OPENAI_MODEL_NAME='model'"'"'with-quote'""" in content


def test_cli_translates_sigterm_to_keyboard_interrupt() -> None:
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"), cli._translate_termination_signals():
        os.kill(os.getpid(), signal.SIGTERM)


def test_benchmark_run_sigterm_exits_managed_inference_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import common.inference_runtime as inference_runtime

    lifecycle = {"entered": False, "exited": False}

    def fake_start_agent_process(*_args: object, **_kwargs: object) -> object:
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM handler did not interrupt benchmark run")

    monkeypatch.setattr(
        inference_runtime,
        "managed_inference",
        lambda _path, **_kwargs: _FakeInferenceContext(lifecycle, served_model_name="model"),
    )
    monkeypatch.setattr(cli, "_resolve_entrypoint_python", lambda **_: sys.executable)
    monkeypatch.setattr(cli, "_start_agent_process", fake_start_agent_process)
    monkeypatch.setattr(cli, "_terminate_processes", lambda _processes: None)

    config_path = tmp_path / "inference.yaml"
    config_path.write_text("kind: existing_endpoint\napi_base: http://127.0.0.1:8123/v1\n")

    caught: BaseException | None = None
    try:
        cli.main(
            [
                "--assets-root",
                str(ASSETS_ROOT),
                "--result-root",
                str(tmp_path / "results"),
                "bench",
                "run",
                "AutomationBench",
                "--task-id",
                "sample_sf_contact_phone_update",
                "--inference-config",
                str(config_path),
            ]
        )
    except BaseException as error:
        caught = error

    assert lifecycle == {"entered": True, "exited": True}
    assert caught is not None
    assert getattr(caught, "exit_code", getattr(caught, "code", None)) == 130


def test_benchmark_run_applies_generic_inference_env(monkeypatch, tmp_path: Path) -> None:
    import common.inference_runtime as inference_runtime

    captured_envs: list[dict[str, str]] = []
    captured_request_configs: list[dict[str, Any]] = []
    resolved_ports = iter([39001, 39002])
    lifecycle = {"entered": False, "exited": False}

    class DummyProcess:
        pid = 12345

        def poll(self) -> None:
            return None

    def fake_start_agent_process(
        command: list[str],
        *,
        python_executable: str,
        workdir: Path,
        env: dict[str, str],
        show_logs: bool,
    ) -> DummyProcess:
        del command, python_executable, workdir, show_logs
        captured_envs.append(env.copy())
        return DummyProcess()

    def fake_resolve_actual_agent_port(**_: Any) -> int:
        return next(resolved_ports)

    async def fake_wait_for_agents(*_: Any, **__: Any) -> bool:
        return True

    async def fake_run_client(eval_request: Any, *_args: Any, **_kwargs: Any) -> dict[str, str]:
        captured_request_configs.append(dict(eval_request.config))
        return {"status": "completed"}

    monkeypatch.setattr(
        inference_runtime,
        "managed_inference",
        lambda _path, **_kwargs: _FakeInferenceContext(
            lifecycle,
            served_model_name="actor-model",
            recommended_max_parallel=2,
        ),
    )
    monkeypatch.setattr(cli, "_resolve_entrypoint_python", lambda **_: __import__("sys").executable)
    monkeypatch.setattr(cli, "_start_agent_process", fake_start_agent_process)
    monkeypatch.setattr(cli, "_resolve_actual_agent_port", fake_resolve_actual_agent_port)
    monkeypatch.setattr(cli, "_wait_for_agents", fake_wait_for_agents)
    monkeypatch.setattr(cli, "_run_client", fake_run_client)
    monkeypatch.setattr(cli, "_terminate_processes", lambda _processes: None)

    config_path = tmp_path / "inference.yaml"
    config_path.write_text("kind: existing_endpoint\napi_base: http://127.0.0.1:8123/v1\n")

    result = runner.invoke(
        cli.app,
        [
            "--assets-root",
            str(ASSETS_ROOT),
            "--result-root",
            str(tmp_path / "results"),
            "bench",
            "run",
            "AutomationBench",
            "--task-id",
            "sample_sf_contact_phone_update",
            "--inference-config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert lifecycle == {"entered": True, "exited": True}
    assert captured_envs
    assert captured_envs[0]["OPENAI_BASE_URL"] == "http://127.0.0.1:8123/v1"
    assert captured_envs[0]["OPENAI_MODEL_NAME"] == "actor-model"
    assert captured_envs[0]["INFERENCE_DEFAULT_MODEL"] == "actor-model"
    assert captured_envs[0]["BENCHMARK_INFERENCE_LAUNCH_SUMMARY"].endswith(
        "inference-launch-summary.json"
    )
    assert "max_parallel" not in captured_request_configs[0]
    assert captured_request_configs[0]["reproducibility"]["launcher_recommended_max_parallel"] == 2
    assert (
        captured_request_configs[0]["reproducibility"]["inference_identity_key"]
        == "run:20260603-153012-a3f91c2b"
    )
    assert (
        captured_request_configs[0]["reproducibility"]["inference_served_model_name"]
        == "actor-model"
    )
    assert "SIMPLETASKS_LLM_BASE_URL" not in captured_envs[0]


def test_benchmark_run_fails_when_managed_inference_cleanup_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import common.inference_runtime as inference_runtime

    resolved_ports = iter([39001, 39002])
    lifecycle = {"entered": False, "exited": False}

    class DummyProcess:
        pid = 12345

        def poll(self) -> None:
            return None

    def fake_start_agent_process(
        command: list[str],
        *,
        python_executable: str,
        workdir: Path,
        env: dict[str, str],
        show_logs: bool,
    ) -> DummyProcess:
        del command, python_executable, workdir, env, show_logs
        return DummyProcess()

    async def fake_wait_for_agents(*_: Any, **__: Any) -> bool:
        return True

    async def fake_run_client(*_: Any, **__: Any) -> dict[str, str]:
        return {"status": "completed"}

    monkeypatch.setattr(
        inference_runtime,
        "managed_inference",
        lambda _path, **_kwargs: _FakeInferenceContext(
            lifecycle,
            served_model_name="model",
            cleanup_error=RuntimeError("cleanup failed"),
        ),
    )
    monkeypatch.setattr(cli, "_resolve_entrypoint_python", lambda **_: sys.executable)
    monkeypatch.setattr(cli, "_start_agent_process", fake_start_agent_process)
    monkeypatch.setattr(cli, "_resolve_actual_agent_port", lambda **_: next(resolved_ports))
    monkeypatch.setattr(cli, "_wait_for_agents", fake_wait_for_agents)
    monkeypatch.setattr(cli, "_run_client", fake_run_client)
    monkeypatch.setattr(cli, "_terminate_processes", lambda _processes: None)

    config_path = tmp_path / "inference.yaml"
    config_path.write_text("kind: existing_endpoint\napi_base: http://127.0.0.1:8123/v1\n")

    result = runner.invoke(
        cli.app,
        [
            "--assets-root",
            str(ASSETS_ROOT),
            "--result-root",
            str(tmp_path / "results"),
            "bench",
            "run",
            "AutomationBench",
            "--task-id",
            "sample_sf_contact_phone_update",
            "--inference-config",
            str(config_path),
        ],
    )

    assert result.exit_code == 1
    assert lifecycle == {"entered": True, "exited": True}
    assert "Failed to stop inference runtime" in result.output


def test_prepare_inference_runtime_config_requires_handoff_payload() -> None:
    class RuntimeWithoutPayload:
        def apply_to(self, _env: dict[str, str]) -> None:
            return None

        def recommended_max_parallel(self) -> None:
            return None

    with pytest.raises(RuntimeError, match="handoff payload"):
        prepare_inference_runtime_config(
            runtime_config={"config": {}},
            env={},
            result_paths=_result_paths(),
            inference_summary_path=None,
            inference_runtime=RuntimeWithoutPayload(),
        )


def test_reproducibility_metadata_uses_handoff_identity_not_endpoint_url() -> None:
    request_a = {
        "vllm_model_id": "request/model",
        "model": "fallback/model",
        "llm_model": "legacy/model",
    }
    request_b = dict(request_a)
    handoff_a = _valid_inference_handoff(api_base="http://127.0.0.1:8123/v1")
    handoff_b = _valid_inference_handoff(api_base="http://127.0.0.1:29000/v1")

    attach_reproducibility_metadata(
        request_a,
        result_paths=_result_paths(run_id="result-a"),
        inference_summary_path=None,
        inference_handoff=handoff_a,
    )
    attach_reproducibility_metadata(
        request_b,
        result_paths=_result_paths(run_id="result-b"),
        inference_summary_path=None,
        inference_handoff=handoff_b,
    )

    metadata_a = request_a["reproducibility"]
    metadata_b = request_b["reproducibility"]
    assert metadata_a["inference_identity_key"] == metadata_b["inference_identity_key"]
    assert (
        metadata_a["inference_semantic_config_hash"] == metadata_b["inference_semantic_config_hash"]
    )
    assert metadata_a["inference_served_model_name"] == "served/model"
    assert "model_id" not in metadata_a
    assert (
        metadata_a["inference_sessions"]["endpoints"]["default"]["api_base"]
        == "http://127.0.0.1:8123/v1"
    )
    assert (
        metadata_b["inference_sessions"]["endpoints"]["default"]["api_base"]
        == "http://127.0.0.1:29000/v1"
    )


def _valid_inference_handoff(
    *,
    api_base: str = "http://127.0.0.1:8123/v1",
    served_model_name: str = "served/model",
    summary_path: str = "/tmp/inference-launch-summary.json",
) -> dict[str, object]:
    return {
        "schema_version": "ril-handoff/v1",
        "run_id": "20260603-153012-a3f91c2b",
        "semantic_config_hash": "sha256:semantic",
        "identity_key": "run:20260603-153012-a3f91c2b",
        "cleanup_owner": "runtime",
        "endpoints": {
            "default": {
                "api_base": api_base,
                "model": served_model_name,
                "served_model_name": served_model_name,
                "summary_path": summary_path,
            }
        },
    }


def _result_paths(*, run_id: str = "result-run") -> SimpleNamespace:
    return SimpleNamespace(
        run_id=run_id,
        result_dir=Path("/tmp/results") / run_id,
        config_hash="sha256:result",
    )


def test_slurm_passthrough_forwards_inference_config(tmp_path: Path) -> None:
    config_path = tmp_path / "inference.yaml"

    args = cli._build_slurm_passthrough_args(
        runtime_config={
            "green_agent": {"host": "127.0.0.1", "port": 0},
            "participants": [{"role": "agent", "host": "127.0.0.1", "port": 0}],
            "config": {
                "target": "sample",
                "task_ids": ["task-001"],
            },
        },
        workdir=tmp_path,
        show_logs=True,
        serve_only=False,
        inference_config=config_path,
    )

    assert "--inference-config" in args
    assert args[args.index("--inference-config") + 1] == str(config_path)
