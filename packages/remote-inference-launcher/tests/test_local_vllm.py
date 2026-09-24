from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from remote_inference_launcher.local_vllm import (
    LocalVllmConfig,
    LocalVllmLauncher,
    _resolved_out_dir,
    _resolved_runtime_tmp_root,
)
from remote_inference_launcher.readiness import ReadinessResult


def test_progress_summary_includes_local_cleanup_command(tmp_path: Path) -> None:
    launcher = LocalVllmLauncher(LocalVllmConfig(model="vendor/test-model"))
    recording = _RecordingSummary(tmp_path / "summary.json")
    launcher._summary = recording
    launcher._process = SimpleNamespace(pid=12345)
    launcher._write_summary("STARTING")

    assert recording.payloads[-1]["lifecycle_state"] == "STARTING"
    assert recording.payloads[-1]["pid"] == 12345
    assert recording.payloads[-1]["cleanup_command"] == "kill -TERM -12345"


def test_start_derives_api_base_from_reserved_port(monkeypatch, tmp_path: Path) -> None:
    observed_api_bases: list[str] = []

    class FakeReservation:
        port = 8124

        def close(self) -> None:
            return None

    class FakeProcess:
        pid = 12345

        def poll(self) -> int | None:
            return None

    def wait_for_readiness(api_base: str, **_kwargs: object) -> ReadinessResult:
        observed_api_bases.append(api_base)
        return ReadinessResult(
            models_endpoint_ok=True,
            smoke_test_ok=True,
            smoke_test_kind="chat_completion",
        )

    monkeypatch.setattr(
        "remote_inference_launcher.local_vllm._require_vllm_available",
        lambda _: None,
    )
    monkeypatch.setattr(
        "remote_inference_launcher.local_vllm.reserve_local_port",
        lambda _host, _port: FakeReservation(),
    )
    monkeypatch.setattr(
        "remote_inference_launcher.local_vllm.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        "remote_inference_launcher.local_vllm.wait_for_openai_readiness",
        wait_for_readiness,
    )
    monkeypatch.setattr(
        "remote_inference_launcher.local_vllm._terminate_process_group",
        lambda _process: None,
    )
    launcher = LocalVllmLauncher(
        LocalVllmConfig(
            model="vendor/test-model",
            host="0.0.0.0",
            out_dir=str(tmp_path / "local"),
            python_bin="/venv/bin/python",
        )
    )

    session = launcher.start()
    launcher.stop()

    assert observed_api_bases == ["http://127.0.0.1:8124/v1"]
    assert session.api_base == "http://127.0.0.1:8124/v1"
    assert session.local_port == 8124


def test_generated_local_paths_use_run_id() -> None:
    config = LocalVllmConfig(name="actor", model="vendor/test-model", port=8124)
    run_id = "20260603-153012-a3f91c2b"

    out_dir = _resolved_out_dir(config, run_id=run_id)
    runtime_tmp = _resolved_runtime_tmp_root(config, out_dir, run_id=run_id)

    assert str(out_dir).endswith(
        "/.remote-inference-launcher/local-vllm/vendor-test-model/actor-" + run_id
    )
    assert runtime_tmp.name == "ril-vllm-vendor-test-model-8124-" + run_id


def test_failure_summary_before_log_dir_is_initialized(monkeypatch, tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"

    def fail_import(_python_bin: str) -> None:
        raise RuntimeError("vLLM is not installed")

    monkeypatch.setattr(
        "remote_inference_launcher.local_vllm._require_vllm_available",
        fail_import,
    )
    launcher = LocalVllmLauncher(
        LocalVllmConfig(
            model="vendor/test-model",
            python_bin="/missing/python",
            launch_summary_path=str(summary_path),
        )
    )

    with pytest.raises(RuntimeError, match="vLLM is not installed"):
        launcher.start()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["lifecycle_state"] == "FAILED"
    assert payload["local_log_dir"] is None
    assert payload["local_api_base"] is None


class _RecordingSummary:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.payloads: list[dict[str, object]] = []
        self._reserved = False

    def reserve(self) -> Path:
        if not self._reserved:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text('{"lifecycle_state": "RESERVED"}\n', encoding="utf-8")
            self._reserved = True
        return self.path

    def write(self, payload: dict[str, object]) -> None:
        self.reserve()
        self.payloads.append(dict(payload))
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
