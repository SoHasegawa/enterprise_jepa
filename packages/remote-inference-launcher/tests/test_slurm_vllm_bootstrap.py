from __future__ import annotations

import json
import runpy
import subprocess
from importlib import resources

import pytest

from remote_inference_launcher.cli import build_parser, slurm_vllm_bootstrap_config_from_args
from remote_inference_launcher.slurm_vllm_bootstrap import (
    BOOTSTRAP_HELPER_NAME,
    SlurmVllmBootstrapConfig,
    SlurmVllmBootstrapper,
    bootstrap_cluster_profile,
    load_slurm_vllm_bootstrap_config,
    render_bootstrap_sbatch_script,
    validate_slurm_vllm_bootstrap_config,
    with_bootstrap_defaults,
)

TEST_VLLM_PACKAGE = "vllm==0.19.1"


def base_config(**overrides: object) -> SlurmVllmBootstrapConfig:
    values = {
        "ssh_target": "cluster",
        "environment_name": "qwen35-vllm",
        "vllm_package": TEST_VLLM_PACKAGE,
        "partition": "batch",
    }
    values.update(overrides)
    return SlurmVllmBootstrapConfig(**values)


def test_bootstrap_requires_environment_name() -> None:
    with pytest.raises(ValueError, match="environment_name"):
        with_bootstrap_defaults(SlurmVllmBootstrapConfig(ssh_target="cluster"))


def test_bootstrap_defaults_derive_names_from_environment_name() -> None:
    config = with_bootstrap_defaults(base_config())
    assert config.venv_path == "~/.qwen35-vllm"
    assert config.job_name.startswith("ril-vllm-bootstrap_qwen35-vllm_")
    assert config.backend == "auto"


def test_bootstrap_defaults_use_known_gpu_cluster_profile() -> None:
    config = with_bootstrap_defaults(SlurmVllmBootstrapConfig(ssh_target="gpu-cluster"))

    assert config.environment_name == "ejepa-vllm-rocm"
    assert config.venv_path == "~/.ejepa-vllm-rocm"
    assert config.vllm_package == "vllm==0.22.0+rocm722"
    assert config.backend == "rocm"
    assert config.partition == "batch-1gpu-short"
    assert config.ray_package == ""


def test_bootstrap_defaults_use_known_slurm_login_profile() -> None:
    config = with_bootstrap_defaults(SlurmVllmBootstrapConfig(ssh_target="slurm-login02"))

    assert config.environment_name == "ejepa-vllm-env"
    assert config.venv_path == "~/.ejepa-vllm-env"
    assert config.vllm_package == TEST_VLLM_PACKAGE
    assert config.backend == "auto"
    assert config.partition == "batch-1gpu-exclusive"
    assert config.ray_package == ""


def test_bootstrap_known_profile_preserves_explicit_overrides() -> None:
    config = with_bootstrap_defaults(
        SlurmVllmBootstrapConfig(
            ssh_target="gpu-cluster",
            environment_name="custom",
            vllm_package="vllm==0.20.1",
            ray_package="ray==2.55.1",
            backend="auto",
            partition="custom-partition",
        )
    )

    assert config.environment_name == "custom"
    assert config.vllm_package == "vllm==0.20.1"
    assert config.ray_package == "ray==2.55.1"
    assert config.backend == "auto"
    assert config.partition == "custom-partition"


@pytest.mark.parametrize(
    "target",
    ["gpu-cluster", "gpu-cluster02", "user@gpu-cluster", "ssh://user@slurm-login:22"],
)
def test_bootstrap_cluster_profile_matches_alias_variants(target: str) -> None:
    assert bootstrap_cluster_profile(target) is not None


def test_bootstrap_normalizes_trailing_slash_venv_path() -> None:
    config = with_bootstrap_defaults(base_config(venv_path="~/.qwen35-vllm/"))
    assert config.venv_path == "~/.qwen35-vllm"


def test_bootstrap_rejects_invalid_environment_name() -> None:
    with pytest.raises(ValueError, match="environment_name"):
        validate_slurm_vllm_bootstrap_config(
            with_bootstrap_defaults(base_config(environment_name="bad name"))
        )


def test_bootstrap_rejects_path_like_environment_name_before_deriving_venv_path() -> None:
    with pytest.raises(ValueError, match="environment_name"):
        with_bootstrap_defaults(base_config(environment_name="."))


@pytest.mark.parametrize(
    "venv_path",
    [
        "~",
        "~/",
        "~/..",
        "~/.env/../other",
        "/",
        "/scratch/../env",
        "relative/env",
        "~other/env",
    ],
)
def test_bootstrap_rejects_unsafe_remote_venv_paths(venv_path: str) -> None:
    with pytest.raises(ValueError, match="venv_path"):
        validate_slurm_vllm_bootstrap_config(
            with_bootstrap_defaults(base_config(venv_path=venv_path))
        )


def test_bootstrap_rejects_non_exact_vllm_package() -> None:
    with pytest.raises(ValueError, match="vllm_package"):
        validate_slurm_vllm_bootstrap_config(
            with_bootstrap_defaults(base_config(vllm_package="vllm>=0.19"))
        )


def test_bootstrap_rejects_ray_package_with_whitespace() -> None:
    with pytest.raises(ValueError, match="ray_package"):
        validate_slurm_vllm_bootstrap_config(
            with_bootstrap_defaults(base_config(ray_package="ray == 2.55.1"))
        )


def test_bootstrap_rejects_invalid_verbosity() -> None:
    with pytest.raises(ValueError, match="verbosity"):
        validate_slurm_vllm_bootstrap_config(
            with_bootstrap_defaults(base_config(verbosity="debug"))
        )


def test_render_rocm_bootstrap_installs_uv_when_flag_is_set() -> None:
    config = base_config(
        backend="rocm",
        install_uv_if_missing=True,
        vllm_package="vllm==0.20.1",
    )
    script = render_bootstrap_sbatch_script(config, out_dir="/remote/out")
    assert "#SBATCH --gres=gpu:1" in script
    assert "VLLM_PACKAGE=vllm==0.20.1" in script
    assert "RAY_PACKAGE=ray" in script
    assert "ENVIRONMENT_NAME=qwen35-vllm" in script
    assert 'elif [ "${1#\\~/}" != "$1" ]; then' in script
    assert "uv/install.sh | sh" in script
    assert '"${UV_BIN}" venv "${VENV_PATH}" --python 3.12 --seed --managed-python' in script
    assert '"${VLLM_PACKAGE}"' in script
    assert '"${UV_BIN}" pip install --python "${VENV_PATH}/bin/python" "${RAY_PACKAGE}"' in script
    assert (
        "--index https://wheels.vllm.ai/rocm/ "
        "--default-index https://pypi.org/simple "
        "--index-strategy unsafe-best-match --upgrade" in script
    )
    assert f"HELPER_PATH=/remote/out/{BOOTSTRAP_HELPER_NAME}" in script
    assert (
        '"${VENV_PATH}/bin/python" "${HELPER_PATH}" "${MANIFEST_PATH}" "${SUCCESS_PATH}"' in script
    )


def test_render_auto_backend_requires_uv_and_uses_torch_backend_auto() -> None:
    script = render_bootstrap_sbatch_script(base_config(), out_dir="/remote/out")
    assert "uv is not installed on the allocated node." in script
    assert "Rerun with --install-uv-if-missing" in script
    assert '"${VLLM_PACKAGE}" --torch-backend=auto' in script


def test_bootstrap_manifest_version_check_accepts_exact_local_version() -> None:
    namespace = _bootstrap_manifest_namespace()
    version_matches = namespace["vllm_version_matches"]

    assert version_matches("0.22.0+rocm722", "0.22.0+rocm722")
    assert version_matches("0.22.0+rocm722", "0.22.0")
    assert not version_matches("0.22.0+cu130", "0.22.0+rocm722")


def test_render_bootstrap_can_skip_ray_install() -> None:
    script = render_bootstrap_sbatch_script(base_config(ray_package=""), out_dir="/remote/out")

    assert "RAY_PACKAGE=''" in script
    assert 'if [ -n "${RAY_PACKAGE}" ]; then' in script


def test_render_existing_env_is_reused_and_writes_success_manifest() -> None:
    script = render_bootstrap_sbatch_script(base_config(), out_dir="/remote/out")
    assert 'REUSED_EXISTING_ENV="0"' in script
    assert 'echo "Remote bootstrap environment already exists: ${VENV_PATH}"' in script
    assert 'REUSED_EXISTING_ENV="1"' in script
    assert 'if [ "${REUSED_EXISTING_ENV}" = "0" ]; then' in script
    assert 'export RIL_BOOTSTRAP_REUSED_EXISTING_ENV="${REUSED_EXISTING_ENV}"' in script


def test_render_force_bootstrap_refuses_to_delete_non_venv_path() -> None:
    script = render_bootstrap_sbatch_script(base_config(force=True), out_dir="/remote/out")

    assert 'if [ -e "${VENV_PATH}" ] || [ -L "${VENV_PATH}" ]; then' in script
    assert 'if [ ! -f "${VENV_PATH}/pyvenv.cfg" ]; then' in script
    assert "Refusing to remove remote bootstrap path because it is not a Python virtual" in script
    assert 'rm -rf -- "${VENV_PATH}"' in script


def test_load_bootstrap_yaml_config(tmp_path) -> None:
    config_path = tmp_path / "bootstrap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "ssh_target: cluster",
                "environment_name: qwen35-vllm-cu129",
                "partition: batch",
                "walltime: '1:00:00'",
                "num_gpus: 1",
                "memory: 32GB",
                "cpus_per_task: 4",
                "verbosity: quiet",
                "vllm_package: vllm==0.19.1",
                "ray_package: ray==2.55.1",
                "install_uv_if_missing: true",
            ]
        ),
        encoding="utf-8",
    )
    config = load_slurm_vllm_bootstrap_config(config_path)
    assert config.environment_name == "qwen35-vllm-cu129"
    assert config.vllm_package == TEST_VLLM_PACKAGE
    assert config.ray_package == "ray==2.55.1"
    assert config.verbosity == "quiet"
    assert config.install_uv_if_missing


@pytest.mark.parametrize("payload", ["false\n", "0\n", "[]\n", "''\n", "\n"])
def test_load_bootstrap_yaml_rejects_non_mapping_document(tmp_path, payload: str) -> None:
    config_path = tmp_path / "bootstrap.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_slurm_vllm_bootstrap_config(config_path)


def test_cli_flags_override_bootstrap_yaml(tmp_path) -> None:
    config_path = tmp_path / "bootstrap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "ssh_target: cluster",
                "environment_name: from-yaml",
                "partition: batch",
                "vllm_package: vllm==0.19.1",
            ]
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "slurm-vllm-bootstrap",
            "--config",
            str(config_path),
            "--environment-name",
            "from-cli",
            "--vllm-package",
            "vllm==0.20.1",
            "--install-uv-if-missing",
        ]
    )
    config = slurm_vllm_bootstrap_config_from_args(args)
    assert config.environment_name == "from-cli"
    assert config.vllm_package == "vllm==0.20.1"
    assert config.install_uv_if_missing


def test_load_bootstrap_yaml_rejects_invalid_verbosity(tmp_path) -> None:
    config_path = tmp_path / "bootstrap.yaml"
    config_path.write_text("verbosity: debug\n", encoding="utf-8")
    with pytest.raises(ValueError, match="verbosity"):
        load_slurm_vllm_bootstrap_config(config_path)


def test_bootstrap_job_info_query_failures_raise_clear_error() -> None:
    bootstrapper = _BootstrapSqueueFailure(base_config())

    with pytest.raises(RuntimeError, match="squeue unavailable"):
        bootstrapper._get_remote_job_info("12345")


def test_bootstrap_manifest_probe_failures_raise_clear_error() -> None:
    bootstrapper = _BootstrapManifestProbeFailure(base_config())

    with pytest.raises(RuntimeError, match="ssh unavailable"):
        bootstrapper._read_success_manifest(
            "/remote/bootstrap_success.json",
            expected_job_id="12345",
            expected_venv_path="/remote/home/.qwen35-vllm",
        )


def test_bootstrap_cancel_failure_is_reported_when_job_is_still_present() -> None:
    bootstrapper = _BootstrapCancelFailure(base_config())

    with pytest.raises(RuntimeError, match="scancel failed"):
        bootstrapper._cancel_remote_job("12345")


def test_bootstrap_cancel_allows_race_when_job_is_already_absent() -> None:
    bootstrapper = _BootstrapCancelAlreadyGone(base_config())

    bootstrapper._cancel_remote_job("12345")


def test_bootstrap_cancel_retries_when_job_is_still_present() -> None:
    bootstrapper = _BootstrapCancelTransientThenSuccess(base_config())

    bootstrapper._cancel_remote_job("12345")

    assert bootstrapper.scancel_attempts == 2


def test_bootstrap_run_reports_cancel_failure_after_primary_failure() -> None:
    bootstrapper = _BootstrapRunCancelFailure(base_config())

    with pytest.raises(RuntimeError, match=r"scancel failed.*wait failed"):
        bootstrapper.run()


def test_bootstrap_run_removes_stale_success_manifest_before_submission() -> None:
    bootstrapper = _BootstrapRemovesStaleManifest(base_config(out_dir="/remote/out"))

    result = bootstrapper.run()

    assert result.job_id == "12345"
    assert bootstrapper.commands == [
        "mkdir",
        "remove-success",
        "copy-helper",
        "copy-script",
        "submit",
    ]


def test_bootstrap_quiet_suppresses_progress(capsys) -> None:
    bootstrapper = _BootstrapRemovesStaleManifest(
        base_config(out_dir="/remote/out", verbosity="quiet")
    )

    bootstrapper.run()

    captured = capsys.readouterr()
    assert captured.out == ""


def test_bootstrap_verbose_prints_remote_command_diagnostics(monkeypatch, capsys) -> None:
    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(
        "remote_inference_launcher.slurm_vllm_bootstrap.subprocess.run",
        fake_run,
    )
    bootstrapper = SlurmVllmBootstrapper(base_config(verbosity="verbose"))

    assert bootstrapper._ssh_text("hostname") == "ok\n"

    captured = capsys.readouterr()
    assert "Remote command on cluster: hostname" in captured.out


def test_bootstrap_ssh_text_retries_transient_transport_failure(monkeypatch) -> None:
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                command,
                255,
                stdout="",
                stderr="kex_exchange_identification: read: Connection reset by peer",
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(
        "remote_inference_launcher.slurm_vllm_bootstrap.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "remote_inference_launcher.slurm_vllm_bootstrap.time.sleep",
        lambda _seconds: None,
    )
    bootstrapper = SlurmVllmBootstrapper(base_config())

    assert bootstrapper._ssh_text("hostname") == "ok\n"
    assert len(calls) == 2


def test_bootstrap_submit_remote_job_does_not_retry_transient_transport_failure(
    monkeypatch,
) -> None:
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command[-1])
        if "squeue" in command[-1]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            command,
            255,
            stdout="",
            stderr="kex_exchange_identification: read: Connection reset by peer",
        )

    monkeypatch.setattr(
        "remote_inference_launcher.slurm_vllm_bootstrap.subprocess.run",
        fake_run,
    )
    bootstrapper = SlurmVllmBootstrapper(base_config())

    with pytest.raises(RuntimeError, match="refusing to replay sbatch"):
        bootstrapper._submit_remote_job("/remote/out/bootstrap_vllm.sbatch")

    assert sum("sbatch" in command for command in calls) == 1


def test_bootstrap_submit_remote_job_recovers_job_id_after_transient_transport_failure(
    monkeypatch,
) -> None:
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command[-1])
        if "squeue" in command[-1]:
            return subprocess.CompletedProcess(command, 0, stdout="12345\n", stderr="")
        return subprocess.CompletedProcess(
            command,
            255,
            stdout="",
            stderr="kex_exchange_identification: read: Connection reset by peer",
        )

    monkeypatch.setattr(
        "remote_inference_launcher.slurm_vllm_bootstrap.subprocess.run",
        fake_run,
    )
    bootstrapper = SlurmVllmBootstrapper(base_config())

    assert bootstrapper._submit_remote_job("/remote/out/bootstrap_vllm.sbatch") == "12345"
    assert sum("sbatch" in command for command in calls) == 1
    assert sum("squeue" in command for command in calls) == 1


def test_bootstrap_run_cancels_job_when_success_manifest_belongs_to_previous_job() -> None:
    bootstrapper = _BootstrapWrongJobManifest(base_config(out_dir="/remote/out"))

    with pytest.raises(RuntimeError, match="slurm_job_id"):
        bootstrapper.run()

    assert bootstrapper.cancelled_jobs == ["12345"]


@pytest.mark.parametrize("venv_path", ["/remote/home", "/remote"])
def test_bootstrap_run_rejects_venv_path_that_targets_remote_home_or_parent(
    venv_path: str,
) -> None:
    bootstrapper = _BootstrapRemoteHomeOnly(base_config(out_dir="/remote/out", venv_path=venv_path))

    with pytest.raises(ValueError, match="remote home"):
        bootstrapper.run()

    assert bootstrapper._temp_dir is None


class _BootstrapSqueueFailure(SlurmVllmBootstrapper):
    def _ssh_text(self, command: str) -> str:
        assert command == "squeue -j 12345 -h -o '%T|%N|%R'"
        raise RuntimeError("squeue unavailable")


class _BootstrapManifestProbeFailure(SlurmVllmBootstrapper):
    def _ssh_text(self, command: str) -> str:
        assert command.startswith("if [ -s /remote/bootstrap_success.json ]")
        raise RuntimeError("ssh unavailable")


class _BootstrapCancelFailure(SlurmVllmBootstrapper):
    def _ssh_text(self, command: str) -> str:
        if command == "scancel 12345":
            raise RuntimeError("scancel failed")
        if command == "squeue -j 12345 -h -o '%T|%N|%R'":
            return "RUNNING|node001|None\n"
        raise AssertionError(f"unexpected SSH command: {command}")


class _BootstrapCancelAlreadyGone(SlurmVllmBootstrapper):
    def _ssh_text(self, command: str) -> str:
        if command == "scancel 12345":
            raise RuntimeError("invalid job id")
        if command == "squeue -j 12345 -h -o '%T|%N|%R'":
            return ""
        raise AssertionError(f"unexpected SSH command: {command}")


class _BootstrapCancelTransientThenSuccess(SlurmVllmBootstrapper):
    def __init__(self, config: SlurmVllmBootstrapConfig) -> None:
        super().__init__(config)
        self.scancel_attempts = 0

    def _ssh_text(self, command: str) -> str:
        if command == "scancel 12345":
            self.scancel_attempts += 1
            if self.scancel_attempts == 1:
                raise RuntimeError("scancel failed")
            return ""
        if command == "squeue -j 12345 -h -o '%T|%N|%R'":
            return "RUNNING|node001|None\n"
        raise AssertionError(f"unexpected SSH command: {command}")


class _BootstrapRunCancelFailure(_BootstrapCancelFailure):
    def _ssh_text(self, command: str) -> str:
        if command == "cd && pwd":
            return "/remote/home\n"
        if command.startswith("mkdir -p"):
            return ""
        if command.startswith("if [ -e ") and command.endswith("/bootstrap_success.json; fi"):
            return ""
        return super()._ssh_text(command)

    def _copy_to_remote(self, local_path, remote_path: str) -> None:
        assert local_path.is_file()
        assert remote_path.endswith(("bootstrap_vllm.sbatch", "vllm_bootstrap_manifest.py"))

    def _submit_remote_job(self, remote_script: str) -> str:
        assert remote_script.endswith("/bootstrap_vllm.sbatch")
        return "12345"

    def _wait_for_success(
        self,
        job_id: str,
        success_path: str,
        out_dir: str,
        *,
        expected_venv_path: str,
    ):
        assert job_id == "12345"
        assert success_path.endswith("/bootstrap_success.json")
        assert out_dir.startswith("/remote/home/")
        assert expected_venv_path == "/remote/home/.qwen35-vllm"
        raise TimeoutError("wait failed")


class _BootstrapRemovesStaleManifest(SlurmVllmBootstrapper):
    def __init__(self, config: SlurmVllmBootstrapConfig) -> None:
        super().__init__(config)
        self.commands: list[str] = []

    def _ssh_text(self, command: str) -> str:
        if command == "cd && pwd":
            return "/remote/home\n"
        if command == "mkdir -p /remote/out":
            self.commands.append("mkdir")
            return ""
        if (
            command == "if [ -e /remote/out/bootstrap_success.json ] "
            "|| [ -L /remote/out/bootstrap_success.json ]; then "
            "rm -- /remote/out/bootstrap_success.json; fi"
        ):
            self.commands.append("remove-success")
            return ""
        raise AssertionError(f"unexpected SSH command: {command}")

    def _copy_to_remote(self, local_path, remote_path: str) -> None:
        assert local_path.is_file()
        if remote_path.endswith("vllm_bootstrap_manifest.py"):
            self.commands.append("copy-helper")
            return
        if remote_path.endswith("bootstrap_vllm.sbatch"):
            self.commands.append("copy-script")
            return
        raise AssertionError(f"unexpected remote path: {remote_path}")

    def _submit_remote_job(self, remote_script: str) -> str:
        assert remote_script == "/remote/out/bootstrap_vllm.sbatch"
        assert self.commands == ["mkdir", "remove-success", "copy-helper", "copy-script"]
        self.commands.append("submit")
        return "12345"

    def _wait_for_success(
        self,
        job_id: str,
        success_path: str,
        out_dir: str,
        *,
        expected_venv_path: str,
    ):
        assert job_id == "12345"
        assert success_path == "/remote/out/bootstrap_success.json"
        assert out_dir == "/remote/out"
        assert expected_venv_path == "/remote/home/.qwen35-vllm"
        return _success_manifest()


class _BootstrapWrongJobManifest(_BootstrapRemovesStaleManifest):
    def __init__(self, config: SlurmVllmBootstrapConfig) -> None:
        super().__init__(config)
        self.cancelled_jobs: list[str] = []

    def _ssh_text(self, command: str) -> str:
        if (
            command == "if [ -s /remote/out/bootstrap_success.json ]; then "
            "printf '%s\\n' present; else printf '%s\\n' absent; fi"
        ):
            return "present\n"
        if command == "cat /remote/out/bootstrap_success.json":
            return json.dumps(_success_manifest(slurm_job_id="old-job"))
        if command == "scancel 12345":
            self.cancelled_jobs.append("12345")
            return ""
        return super()._ssh_text(command)

    def _wait_for_success(
        self,
        job_id: str,
        success_path: str,
        out_dir: str,
        *,
        expected_venv_path: str,
    ):
        return SlurmVllmBootstrapper._wait_for_success(
            self,
            job_id,
            success_path,
            out_dir,
            expected_venv_path=expected_venv_path,
        )


class _BootstrapRemoteHomeOnly(SlurmVllmBootstrapper):
    def _ssh_text(self, command: str) -> str:
        if command == "cd && pwd":
            return "/remote/home\n"
        raise AssertionError(f"unexpected SSH command: {command}")


def _success_manifest(**overrides: object) -> dict[str, object]:
    manifest = {
        "schema_version": 1,
        "slurm_job_id": "12345",
        "environment_name": "qwen35-vllm",
        "venv_path": "/remote/home/.qwen35-vllm",
        "python_bin": "/remote/home/.qwen35-vllm/bin/python",
        "requested_vllm_package": TEST_VLLM_PACKAGE,
        "requested_ray_package": "ray",
        "backend": "auto",
        "manifest_path": (
            "/remote/home/.qwen35-vllm/remote-inference-launcher-bootstrap-manifest.json"
        ),
    }
    manifest.update(overrides)
    return manifest


def _bootstrap_manifest_namespace() -> dict[str, object]:
    template = resources.files("remote_inference_launcher.templates").joinpath(
        "vllm_bootstrap_manifest.py"
    )
    with resources.as_file(template) as path:
        return runpy.run_path(str(path))
