from __future__ import annotations

import json
from pathlib import Path

import pytest

from remote_inference_launcher.ssh_vllm import (
    SshVllmConfig,
    SshVllmLauncher,
    render_ssh_vllm_script,
)


def base_config(**overrides: object) -> SshVllmConfig:
    values = {
        "ssh_target": "gpu-host",
        "model": "vendor/test-model",
        "served_model_name": "served",
        "local_port": 8123,
    }
    values.update(overrides)
    return SshVllmConfig(**values)


def test_start_returns_session_and_stop_kills_remote_process_group() -> None:
    launcher = _ReadySshLauncher(base_config(api_key="secret"))

    session = launcher.start()
    launcher.stop()

    assert session.api_base == "http://127.0.0.1:8123/v1"
    assert session.served_model_name == "served"
    assert session.api_key == "secret"
    assert session.pid == 2468
    assert session.remote_port == 18817
    assert session.logs.startswith(
        "/remote/home/tmp/remote-inference-launcher/ssh-vllm/vendor-test-model/default-"
    )
    assert launcher.copied_remote_path.endswith("/launch_vllm.sh")
    assert launcher.tunnel_port == 8123
    assert launcher.ready_api_base == "http://127.0.0.1:8123/v1"
    assert launcher.stopped_pids == ["2468"]


def test_start_cleans_up_remote_process_when_readiness_fails() -> None:
    launcher = _FailingReadySshLauncher(base_config())

    with pytest.raises(TimeoutError, match="not ready"):
        launcher.start()

    assert launcher.stopped_pids == ["2468"]
    assert launcher._remote_pid == ""


def test_start_cleans_up_when_failure_summary_log_tail_fails() -> None:
    launcher = _FailingLogTailSshLauncher(base_config())

    with pytest.raises(TimeoutError, match="not ready"):
        launcher.start()

    assert launcher.stopped_pids == ["2468"]
    assert launcher._remote_pid == ""


def test_start_fails_fast_when_remote_state_has_invalid_port() -> None:
    launcher = _InvalidRemoteStateSshLauncher(base_config())

    with pytest.raises(RuntimeError, match="valid remote_port"):
        launcher.start()

    assert launcher.stopped_pids == ["2468"]


def test_start_fails_fast_when_process_exits_after_writing_remote_state() -> None:
    launcher = _ExitedAfterRemoteStateSshLauncher(base_config())

    with pytest.raises(RuntimeError, match="exited after writing remote port state"):
        launcher.start()

    assert launcher.stopped_pids == ["2468"]


def test_stop_respects_keep_remote_process() -> None:
    launcher = _ReadySshLauncher(base_config(keep_remote_process=True))

    launcher.start()
    launcher.stop()

    assert launcher.stopped_pids == []
    assert launcher._remote_pid == "2468"


def test_generated_local_port_retries_tunnel_startup() -> None:
    launcher = _RetryTunnelSshLauncher(base_config(local_port=None))

    session = launcher.start()
    launcher.stop()

    assert len(launcher.tunnel_ports) == 2
    assert session.local_port == launcher.tunnel_ports[-1]
    assert session.api_base == f"http://127.0.0.1:{session.local_port}/v1"
    assert launcher.ready_api_base == session.api_base


def test_start_updates_summary_with_remote_process_and_tunnel(tmp_path: Path) -> None:
    launcher = _ReadySshLauncher(base_config())
    recording = _RecordingSummary(tmp_path / "summary.json")
    launcher._summary = recording

    launcher.start()
    launcher.stop()

    assert [payload["lifecycle_state"] for payload in recording.payloads] == [
        "REMOTE_PROCESS_STARTED",
        "REMOTE_PORT_READY",
        "TUNNEL_READY",
        "READY",
        "RELEASED",
    ]
    assert recording.payloads[0]["pid"] == 2468
    assert recording.payloads[0]["cleanup_command"] == "ssh gpu-host 'kill -TERM -2468'"
    assert recording.payloads[1]["remote_port"] == 18817
    assert recording.payloads[1]["remote_state_path"].endswith("remote-inference-state.json")


def test_failed_start_summary_remains_failed_after_cleanup(tmp_path: Path) -> None:
    launcher = _FailingReadySshLauncher(base_config())
    recording = _RecordingSummary(tmp_path / "summary.json")
    launcher._summary = recording

    with pytest.raises(TimeoutError, match="not ready"):
        launcher.start()

    assert recording.payloads[-1]["lifecycle_state"] == "FAILED"
    assert recording.payloads[-1]["pid"] == 2468


def test_explicit_local_port_does_not_retry_tunnel_startup() -> None:
    launcher = _AlwaysFailTunnelSshLauncher(base_config(local_port=8123))

    with pytest.raises(RuntimeError, match="bind failed"):
        launcher.start()

    assert launcher.tunnel_ports == [8123]
    assert launcher.stopped_pids == ["2468"]


def test_rocm_ssh_script_moves_rocr_visible_devices_to_hip_visible_devices() -> None:
    script = render_ssh_vllm_script(
        base_config(target_device="rocm", remote_port=18817),
        out_dir="/remote/out",
    )

    assert 'export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$ROCR_VISIBLE_DEVICES}"' in script
    assert "unset ROCR_VISIBLE_DEVICES" in script


class _ReadySshLauncher(SshVllmLauncher):
    def __init__(self, config: SshVllmConfig) -> None:
        super().__init__(config)
        self.copied_remote_path = ""
        self.tunnel_port = 0
        self.ready_api_base = ""
        self.stopped_pids: list[str] = []

    def _ssh_text(self, command: str, *, command_is_retry_safe: bool = True) -> str:
        del command_is_retry_safe
        if command == "cd && pwd":
            return "/remote/home\n"
        if command.startswith("mkdir -p "):
            return ""
        if command.startswith("rm -f ") and command.endswith("/remote-inference-state.json"):
            return ""
        if command.startswith("chmod +x "):
            return ""
        if "setsid bash" in command:
            assert "&& { setsid bash" in command
            assert "& echo $! ; }" in command
            return "2468\n"
        if command.startswith("kill -TERM -2468"):
            self.stopped_pids.append("2468")
            return ""
        raise AssertionError(f"unexpected SSH command: {command}")

    def _copy_to_remote(self, local_path: Path, remote_path: str) -> None:
        assert local_path.is_file()
        self.copied_remote_path = remote_path

    def _start_tunnel(self, local_port: int) -> None:
        self.tunnel_port = local_port

    def _remote_process_is_alive(self, pid: str) -> bool:
        return pid == "2468"

    def _read_remote_state(self) -> dict[str, object]:
        return {"remote_port": 18817}

    def _wait_for_ready(self, api_base: str) -> None:
        self.ready_api_base = api_base

    def _remote_log_tail(self, *, lines: int = 80) -> str:
        del lines
        return ""


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
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class _FailingReadySshLauncher(_ReadySshLauncher):
    def _wait_for_ready(self, api_base: str) -> None:
        self.ready_api_base = api_base
        raise TimeoutError("not ready")


class _FailingLogTailSshLauncher(_FailingReadySshLauncher):
    def _remote_log_tail(self, *, lines: int = 80) -> str:
        del lines
        raise RuntimeError("tail failed")


class _InvalidRemoteStateSshLauncher(_ReadySshLauncher):
    def _read_remote_state(self) -> dict[str, object]:
        return {"remote_port": "18817"}


class _ExitedAfterRemoteStateSshLauncher(_ReadySshLauncher):
    def _remote_process_is_alive(self, pid: str) -> bool:
        del pid
        return False


class _RetryTunnelSshLauncher(_ReadySshLauncher):
    def __init__(self, config: SshVllmConfig) -> None:
        super().__init__(config)
        self.tunnel_ports: list[int] = []

    def _start_tunnel(self, local_port: int) -> None:
        self.tunnel_ports.append(local_port)
        if len(self.tunnel_ports) == 1:
            raise RuntimeError("bind failed")
        self.tunnel_port = local_port


class _AlwaysFailTunnelSshLauncher(_RetryTunnelSshLauncher):
    def _start_tunnel(self, local_port: int) -> None:
        self.tunnel_ports.append(local_port)
        raise RuntimeError("bind failed")
