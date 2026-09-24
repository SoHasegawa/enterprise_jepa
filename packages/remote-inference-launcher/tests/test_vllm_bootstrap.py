from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from remote_inference_launcher.slurm_vllm_bootstrap import SlurmVllmBootstrapConfig
from remote_inference_launcher.vllm_bootstrap import (
    LocalVllmBootstrapConfig,
    LocalVllmBootstrapper,
    SshVllmBootstrapConfig,
    SshVllmBootstrapper,
    load_local_vllm_bootstrap_config,
    load_vllm_bootstrap_config,
    render_vllm_bootstrap_script,
    validate_local_vllm_bootstrap_config,
    with_local_vllm_bootstrap_defaults,
)


def test_render_local_bootstrap_script_installs_vllm_with_uv() -> None:
    script = render_vllm_bootstrap_script(
        LocalVllmBootstrapConfig(
            environment_name="qwen-vllm",
            venv_path="~/.qwen-vllm",
            vllm_package="vllm==0.20.1",
            backend="rocm",
            python="3.12",
            install_uv_if_missing=True,
            setup_cmd="module load rocm",
        ),
        out_dir="/tmp/bootstrap",
        helper_path="/tmp/bootstrap/vllm_bootstrap_manifest.py",
        success_path="/tmp/bootstrap/bootstrap_success.json",
    )

    assert "ENVIRONMENT_NAME=qwen-vllm" in script
    assert "PYTHON_VERSION=3.12" in script
    assert "module load rocm" in script
    assert "uv/install.sh | sh" in script
    assert '"${UV_BIN}" venv "${VENV_PATH}" --python "${PYTHON_VERSION}"' in script
    assert "RAY_PACKAGE=ray" in script
    assert '"${UV_BIN}" pip install --python "${VENV_PATH}/bin/python" "${RAY_PACKAGE}"' in script
    assert (
        "--index https://wheels.vllm.ai/rocm/ "
        "--default-index https://pypi.org/simple "
        "--index-strategy unsafe-best-match --upgrade" in script
    )
    assert '"${VENV_PATH}/bin/python" "${HELPER_PATH}"' in script


def test_render_local_bootstrap_script_can_skip_ray_install() -> None:
    script = render_vllm_bootstrap_script(
        LocalVllmBootstrapConfig(
            environment_name="qwen-vllm",
            venv_path="~/.qwen-vllm",
            vllm_package="vllm==0.20.1",
            ray_package="",
        ),
        out_dir="/tmp/bootstrap",
        helper_path="/tmp/bootstrap/vllm_bootstrap_manifest.py",
        success_path="/tmp/bootstrap/bootstrap_success.json",
    )

    assert "RAY_PACKAGE=''" in script
    assert 'if [ -n "${RAY_PACKAGE}" ]; then' in script


def test_load_generic_bootstrap_config_dispatches_by_kind(tmp_path: Path) -> None:
    local_path = tmp_path / "local.yaml"
    local_path.write_text(
        "\n".join(
            [
                "kind: local_vllm_bootstrap",
                "environment_name: qwen-vllm",
                "venv_path: ~/.qwen-vllm",
                "vllm_package: vllm==0.20.1",
            ]
        ),
        encoding="utf-8",
    )
    slurm_path = tmp_path / "slurm.yaml"
    slurm_path.write_text(
        "\n".join(
            [
                "kind: slurm_vllm_bootstrap",
                "ssh_target: gpu-cluster",
                "environment_name: qwen-vllm",
                "vllm_package: vllm==0.20.1",
                "partition: batch",
            ]
        ),
        encoding="utf-8",
    )

    assert isinstance(load_vllm_bootstrap_config(local_path), LocalVllmBootstrapConfig)
    assert isinstance(load_vllm_bootstrap_config(slurm_path), SlurmVllmBootstrapConfig)


def test_load_local_bootstrap_config_accepts_optional_kind(tmp_path: Path) -> None:
    path = tmp_path / "local.yaml"
    path.write_text(
        "\n".join(
            [
                "kind: local_vllm_bootstrap",
                "environment_name: qwen-vllm",
                "vllm_package: vllm==0.20.1",
                "force: true",
                "ray_package: ''",
            ]
        ),
        encoding="utf-8",
    )

    config = load_local_vllm_bootstrap_config(path)

    assert config.environment_name == "qwen-vllm"
    assert config.force
    assert config.ray_package == ""


@pytest.mark.parametrize("venv_path", ["~", "~/..", "/tmp/../env", "relative/env"])
def test_local_bootstrap_rejects_unsafe_venv_path(venv_path: str) -> None:
    with pytest.raises(ValueError, match="venv_path"):
        validate_local_vllm_bootstrap_config(
            with_local_vllm_bootstrap_defaults(
                LocalVllmBootstrapConfig(
                    environment_name="qwen-vllm",
                    venv_path=venv_path,
                    vllm_package="vllm==0.20.1",
                )
            )
        )


def test_local_bootstrap_rejects_ray_package_with_whitespace() -> None:
    with pytest.raises(ValueError, match="ray_package"):
        validate_local_vllm_bootstrap_config(
            with_local_vllm_bootstrap_defaults(
                LocalVllmBootstrapConfig(
                    environment_name="qwen-vllm",
                    vllm_package="vllm==0.20.1",
                    ray_package="ray == 2.55.1",
                )
            )
        )


def test_local_bootstrap_runs_script_and_reads_manifest(monkeypatch, tmp_path: Path) -> None:
    venv_path = tmp_path / "qwen-vllm"
    out_dir = tmp_path / "out"

    def fake_run(command, **_kwargs):
        assert command == ["bash", str(out_dir / "bootstrap_vllm_env.sh")]
        _write_success_manifest(
            out_dir / "bootstrap_success.json",
            environment_name="qwen-vllm",
            venv_path=str(venv_path),
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("remote_inference_launcher.vllm_bootstrap.subprocess.run", fake_run)

    result = LocalVllmBootstrapper(
        LocalVllmBootstrapConfig(
            environment_name="qwen-vllm",
            venv_path=str(venv_path),
            vllm_package="vllm==0.20.1",
            out_dir=str(out_dir),
        )
    ).run()

    assert result.kind == "local_vllm_bootstrap"
    assert result.manifest["python_bin"] == f"{venv_path}/bin/python"
    assert (out_dir / "bootstrap_vllm_env.sh").is_file()
    assert (out_dir / "vllm_bootstrap_manifest.py").is_file()


def test_ssh_bootstrap_copies_script_runs_remote_and_reads_manifest() -> None:
    bootstrapper = _ReadySshBootstrapper(
        SshVllmBootstrapConfig(
            ssh_target="gpu-host",
            environment_name="qwen-vllm",
            vllm_package="vllm==0.20.1",
            venv_path="~/.qwen-vllm",
            out_dir="~/bootstrap-out",
        )
    )

    result = bootstrapper.run()

    assert result.kind == "ssh_vllm_bootstrap"
    assert result.target == "gpu-host"
    assert result.out_dir == "/home/test/bootstrap-out"
    assert result.manifest["python_bin"] == "/home/test/.qwen-vllm/bin/python"
    assert bootstrapper.copied_paths == [
        "/home/test/bootstrap-out/vllm_bootstrap_manifest.py",
        "/home/test/bootstrap-out/bootstrap_vllm_env.sh",
    ]
    assert "bash /home/test/bootstrap-out/bootstrap_vllm_env.sh" in bootstrapper.commands


class _ReadySshBootstrapper(SshVllmBootstrapper):
    def __init__(self, config: SshVllmBootstrapConfig) -> None:
        super().__init__(config)
        self.commands: list[str] = []
        self.copied_paths: list[str] = []

    def _ssh_text(self, command: str, *, command_is_retry_safe: bool = True) -> str:
        del command_is_retry_safe
        self.commands.append(command)
        if command == "cd && pwd":
            return "/home/test\n"
        if command.startswith(("mkdir -p ", "rm -f ")):
            return ""
        if command.startswith("chmod +x "):
            return ""
        if command == "bash /home/test/bootstrap-out/bootstrap_vllm_env.sh":
            return "ok\n"
        if command == "cat /home/test/bootstrap-out/bootstrap_success.json":
            return json.dumps(
                _manifest_payload(
                    environment_name="qwen-vllm",
                    venv_path="/home/test/.qwen-vllm",
                )
            )
        raise AssertionError(f"unexpected SSH command: {command}")

    def _copy_to_remote(self, local_path: Path, remote_path: str) -> None:
        assert local_path.is_file()
        self.copied_paths.append(remote_path)


def _write_success_manifest(
    path: Path,
    *,
    environment_name: str,
    venv_path: str,
) -> None:
    path.write_text(
        json.dumps(_manifest_payload(environment_name=environment_name, venv_path=venv_path)),
        encoding="utf-8",
    )


def _manifest_payload(*, environment_name: str, venv_path: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "environment_name": environment_name,
        "venv_path": venv_path,
        "python_bin": f"{venv_path}/bin/python",
        "requested_vllm_package": "vllm==0.20.1",
        "installed_vllm_version": "0.20.1",
        "requested_ray_package": "ray",
        "installed_ray_version": "2.55.1",
        "backend": "auto",
        "uv_version": "uv 0.9.0",
        "install_uv_if_missing": False,
        "reused_existing_env": False,
        "force": False,
        "manifest_path": f"{venv_path}/remote-inference-launcher-bootstrap-manifest.json",
        "diagnostics": {},
    }
