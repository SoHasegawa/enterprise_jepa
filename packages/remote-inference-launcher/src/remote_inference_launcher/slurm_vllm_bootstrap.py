"""Slurm-backed remote vLLM environment bootstrap."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any, cast, get_type_hints

from remote_inference_launcher.process import exception_summary, process_error
from remote_inference_launcher.slurm import (
    new_slurm_submission_comment,
    parse_squeue_job_info,
    scancel_job_command,
    squeue_job_ids_by_comment_command,
    squeue_job_info_command,
)
from remote_inference_launcher.ssh import (
    SSH_TRANSPORT_FAILURE_ATTEMPTS,
    NonRetryableTransientSshError,
    is_transient_ssh_failure,
    ssh_options,
)
from remote_inference_launcher.verbosity import (
    progress_enabled,
    validate_verbosity,
    verbose_enabled,
)
from remote_inference_launcher.yaml_config import load_yaml_mapping

DEFAULT_BACKEND = "auto"
DEFAULT_RAY_PACKAGE = "ray"
DEFAULT_NUM_GPUS = 1
DEFAULT_CPUS_PER_TASK = 4
DEFAULT_MEMORY = "32GB"
DEFAULT_WALLTIME = "1:00:00"
DEFAULT_CHECK_INTERVAL_SECONDS = 10
DEFAULT_TIMEOUT_SECONDS = 7200
DEFAULT_JOB_NAME_PREFIX = "ril-vllm-bootstrap"
DEFAULT_REMOTE_OUT_DIR_ROOT = "tmp/remote-inference-launcher/bootstrap-vllm"
REMOTE_CANCEL_ATTEMPTS = 3
SUCCESS_MANIFEST_NAME = "bootstrap_success.json"
BOOTSTRAP_HELPER_NAME = "vllm_bootstrap_manifest.py"
ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class BootstrapClusterProfile:
    """Known-good defaults for shared Slurm clusters."""

    environment_name: str
    vllm_package: str
    backend: str
    partition: str
    single_node_ray_package: str


BOOTSTRAP_CLUSTER_PROFILES: Mapping[str, BootstrapClusterProfile] = {
    "gpu-cluster": BootstrapClusterProfile(
        environment_name="ejepa-vllm-rocm",
        vllm_package="vllm==0.22.0+rocm722",
        backend="rocm",
        partition="batch-1gpu-short",
        single_node_ray_package="",
    ),
    "slurm-login": BootstrapClusterProfile(
        environment_name="ejepa-vllm-env",
        vllm_package="vllm==0.19.1",
        backend="auto",
        partition="batch-1gpu-exclusive",
        single_node_ray_package="",
    ),
}


@dataclass(frozen=True)
class SlurmVllmBootstrapConfig:
    """Configuration for provisioning one remote vLLM environment through Slurm."""

    ssh_target: str = ""
    verbosity: str = "progress"
    environment_name: str = ""
    venv_path: str = ""
    vllm_package: str = ""
    ray_package: str = DEFAULT_RAY_PACKAGE
    backend: str = ""
    install_uv_if_missing: bool = False
    force: bool = False
    sbatch_cmd: str = "sbatch"
    out_dir: str = ""
    remote_out_dir_root: str = ""
    job_name: str = ""
    job_name_prefix: str = DEFAULT_JOB_NAME_PREFIX
    partition: str = ""
    walltime: str = DEFAULT_WALLTIME
    num_gpus: int = DEFAULT_NUM_GPUS
    memory: str = DEFAULT_MEMORY
    cpus_per_task: int = DEFAULT_CPUS_PER_TASK
    nodes: int = 1
    exclude: str = ""
    nodelist: str = ""
    check_interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class SlurmVllmBootstrapResult:
    """Completed remote bootstrap details."""

    job_id: str
    out_dir: str
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class SlurmBootstrapJobInfo:
    """Current Slurm scheduler state for the submitted bootstrap job."""

    state: str
    node: str = ""
    reason: str = ""


class SlurmVllmBootstrapper:
    """Submit and monitor a Slurm job that provisions a remote vLLM environment."""

    def __init__(self, config: SlurmVllmBootstrapConfig) -> None:
        self.config = validate_slurm_vllm_bootstrap_config(with_bootstrap_defaults(config))
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def run(self) -> SlurmVllmBootstrapResult:
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix="remote-inference-launcher-bootstrap-vllm."
        )
        job_id = ""
        try:
            remote_home = self._ssh_text("cd && pwd").strip()
            out_dir = self.config.out_dir or _default_remote_out_dir(remote_home, self.config)
            out_dir = _expand_remote_home(out_dir, remote_home)
            expected_venv_path = _expand_remote_home(self.config.venv_path, remote_home)
            _require_expanded_remote_venv_path_safe(expected_venv_path, remote_home)
            remote_script = f"{out_dir.rstrip('/')}/bootstrap_vllm.sbatch"
            remote_helper = f"{out_dir.rstrip('/')}/{BOOTSTRAP_HELPER_NAME}"
            success_path = f"{out_dir.rstrip('/')}/{SUCCESS_MANIFEST_NAME}"
            local_script = Path(self._temp_dir.name) / "bootstrap_vllm.sbatch"
            local_helper = Path(self._temp_dir.name) / BOOTSTRAP_HELPER_NAME
            local_script.write_text(
                render_bootstrap_sbatch_script(
                    self.config,
                    out_dir=out_dir,
                    helper_path=remote_helper,
                ),
                encoding="utf-8",
            )
            local_helper.write_text(_bootstrap_helper_text(), encoding="utf-8")

            self._ssh_text(f"mkdir -p {shlex.quote(out_dir)}")
            self._remove_remote_file_if_exists(success_path)
            self._copy_to_remote(local_helper, remote_helper)
            self._copy_to_remote(local_script, remote_script)
            job_id = self._submit_remote_job(remote_script)
            self._log_progress(f"Submitted remote vLLM bootstrap job {job_id}.")
            self._log_progress(f"Remote bootstrap logs: {out_dir}")
            manifest = self._wait_for_success(
                job_id,
                success_path,
                out_dir,
                expected_venv_path=expected_venv_path,
            )
            return SlurmVllmBootstrapResult(job_id=job_id, out_dir=out_dir, manifest=manifest)
        except BaseException as error:
            if job_id:
                try:
                    self._cancel_remote_job(job_id)
                except RuntimeError as cancel_error:
                    raise RuntimeError(
                        "Remote vLLM bootstrap failed and failed to cancel Slurm job "
                        f"{job_id}: {cancel_error}. Original failure: "
                        f"{exception_summary(error)}"
                    ) from error
            raise
        finally:
            self._cleanup_temp_dir()

    def _wait_for_success(
        self,
        job_id: str,
        success_path: str,
        out_dir: str,
        *,
        expected_venv_path: str,
    ) -> Mapping[str, Any]:
        started_at = time.monotonic()
        not_found_once = False
        while True:
            manifest = self._read_success_manifest(
                success_path,
                expected_job_id=job_id,
                expected_venv_path=expected_venv_path,
            )
            if manifest is not None:
                self._log_progress(f"Remote vLLM bootstrap completed: job_id={job_id}")
                return manifest

            info = self._get_remote_job_info(job_id)
            if info.state != "NOT_FOUND":
                not_found_once = False
            if info.state in _FAILED_TERMINAL_STATES:
                raise RuntimeError(
                    "Remote vLLM bootstrap job failed: "
                    f"job_id={job_id} state={info.state} reason={info.reason} "
                    f"remote_logs={out_dir}\n{self._tail_remote_logs(out_dir)}"
                )
            if info.state == "NOT_FOUND":
                if not not_found_once:
                    not_found_once = True
                    time.sleep(self.config.check_interval_seconds)
                    continue
                raise RuntimeError(
                    "Remote vLLM bootstrap job left the scheduler without writing "
                    f"{SUCCESS_MANIFEST_NAME}: job_id={job_id} remote_logs={out_dir}\n"
                    f"{self._tail_remote_logs(out_dir)}"
                )
            if _timed_out(started_at, self.config.timeout_seconds):
                raise TimeoutError(
                    f"Timed out waiting for remote vLLM bootstrap job {job_id}; "
                    f"remote_logs={out_dir}"
                )
            self._log_progress(
                f"Waiting for remote vLLM bootstrap job {job_id} "
                f"(state={info.state}, reason={info.reason})..."
            )
            time.sleep(self.config.check_interval_seconds)

    def _read_success_manifest(
        self,
        success_path: str,
        *,
        expected_job_id: str,
        expected_venv_path: str,
    ) -> Mapping[str, Any] | None:
        status = self._ssh_text(
            f"if [ -s {shlex.quote(success_path)} ]; then "
            "printf '%s\\n' present; else printf '%s\\n' absent; fi"
        ).strip()
        if status == "absent":
            return None
        if status != "present":
            raise RuntimeError(
                f"Unexpected remote bootstrap manifest status for {success_path}: {status!r}"
            )
        raw = self._ssh_text(f"cat {shlex.quote(success_path)}")
        manifest = json.loads(raw)
        if not isinstance(manifest, dict):
            raise RuntimeError(f"Remote bootstrap manifest is not a JSON object: {success_path}")
        _validate_success_manifest(
            manifest,
            self.config,
            success_path,
            expected_job_id=expected_job_id,
            expected_venv_path=expected_venv_path,
        )
        return manifest

    def _submit_remote_job(self, remote_script: str) -> str:
        submission_comment = new_slurm_submission_comment()
        command = " ".join(
            [
                shlex.quote(self.config.sbatch_cmd),
                "--parsable",
                f"--comment={shlex.quote(submission_comment)}",
                shlex.quote(remote_script),
            ]
        )
        try:
            output = self._ssh_text(command, command_is_retry_safe=False).strip().splitlines()
        except NonRetryableTransientSshError as error:
            try:
                recovered_job_id = self._recover_submitted_job_id(submission_comment)
            except RuntimeError as recovery_error:
                raise RuntimeError(
                    f"{error}. Failed to check for an accepted Slurm job with submission "
                    f"comment {submission_comment!r}: {recovery_error}"
                ) from error
            if recovered_job_id:
                return recovered_job_id
            raise RuntimeError(
                f"{error}. No active Slurm job with submission comment "
                f"{submission_comment!r} was found; refusing to replay sbatch."
            ) from error
        job_id = output[-1].split(";", maxsplit=1)[0] if output else ""
        if not job_id:
            raise RuntimeError("Remote Slurm bootstrap submission did not return a job id.")
        return job_id

    def _recover_submitted_job_id(self, submission_comment: str) -> str:
        output = self._ssh_text(squeue_job_ids_by_comment_command(submission_comment))
        job_ids = tuple(line.strip() for line in output.splitlines() if line.strip())
        if len(job_ids) > 1:
            raise RuntimeError(
                "Found multiple active Slurm jobs with submission comment "
                f"{submission_comment!r}: {', '.join(job_ids)}."
            )
        return job_ids[0] if job_ids else ""

    def _get_remote_job_info(self, job_id: str) -> SlurmBootstrapJobInfo:
        parsed = parse_squeue_job_info(self._ssh_text(squeue_job_info_command(job_id)))
        if parsed is None:
            return SlurmBootstrapJobInfo(state="NOT_FOUND")

        state, node, reason = parsed
        return SlurmBootstrapJobInfo(state=state, node=node, reason=reason)

    def _cancel_remote_job(self, job_id: str) -> None:
        cancel_error: RuntimeError | None = None
        info = SlurmBootstrapJobInfo(state="UNKNOWN")
        for attempt in range(REMOTE_CANCEL_ATTEMPTS):
            try:
                self._ssh_text(scancel_job_command(job_id))
                return
            except RuntimeError as error:
                cancel_error = error
            try:
                info = self._get_remote_job_info(job_id)
            except RuntimeError as query_error:
                if attempt < REMOTE_CANCEL_ATTEMPTS - 1:
                    continue
                raise RuntimeError(
                    f"Failed to cancel remote Slurm job {job_id}: {cancel_error}. "
                    "Also failed to verify whether the job is still present: "
                    f"{query_error}"
                ) from query_error
            if info.state == "NOT_FOUND":
                return
        raise RuntimeError(
            f"Failed to cancel remote Slurm job {job_id}: {cancel_error}. "
            f"Job is still present with state={info.state} reason={info.reason}."
        ) from cancel_error

    def _remove_remote_file_if_exists(self, remote_path: str) -> None:
        quoted_path = shlex.quote(remote_path)
        self._ssh_text(
            f"if [ -e {quoted_path} ] || [ -L {quoted_path} ]; then rm -- {quoted_path}; fi"
        )

    def _tail_remote_logs(self, out_dir: str) -> str:
        quoted = shlex.quote(out_dir.rstrip("/"))
        command = (
            f"for file in {quoted}/std.log {quoted}/err.log {quoted}/bootstrap.log; do "
            'if [ -f "${file}" ]; then '
            'echo "== ${file} =="; tail -n 80 "${file}"; '
            "fi; "
            "done"
        )
        try:
            return self._ssh_text(command).strip()
        except RuntimeError as error:
            return f"Could not read remote bootstrap logs: {error}"

    def _copy_to_remote(self, local_path: Path, remote_path: str) -> None:
        self._log_verbose(f"Copying {local_path} to {self.config.ssh_target}:{remote_path}")
        command = [
            "ssh",
            *ssh_options(),
            "-T",
            self.config.ssh_target,
            f"cat > {shlex.quote(remote_path)}",
        ]
        for attempt in range(1, SSH_TRANSPORT_FAILURE_ATTEMPTS + 1):
            with local_path.open("rb") as source:
                completed = subprocess.run(
                    command,
                    stdin=source,
                    capture_output=True,
                    check=False,
                )
            if not _should_retry_ssh(completed, attempt):
                break
            self._sleep_before_ssh_retry(attempt, label="remote bootstrap file copy")
        if completed.returncode:
            raise RuntimeError(process_error("remote bootstrap file copy", completed))

    def _ssh_text(self, command: str, *, command_is_retry_safe: bool = True) -> str:
        self._log_verbose(f"Remote command on {self.config.ssh_target}: {command}")
        attempts = SSH_TRANSPORT_FAILURE_ATTEMPTS if command_is_retry_safe else 1
        for attempt in range(1, attempts + 1):
            completed = self._ssh_completed(command)
            if not _should_retry_ssh(completed, attempt, max_attempts=attempts):
                break
            self._sleep_before_ssh_retry(attempt, label=command)
        if completed.returncode:
            error_message = _ssh_process_error(
                command,
                completed,
                command_is_retry_safe=command_is_retry_safe,
            )
            if not command_is_retry_safe and is_transient_ssh_failure(completed):
                raise NonRetryableTransientSshError(error_message)
            raise RuntimeError(error_message)
        return completed.stdout

    def _ssh_completed(self, command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "ssh",
                *ssh_options(),
                "-T",
                self.config.ssh_target,
                f"bash -lc {shlex.quote(command)}",
            ],
            capture_output=True,
            check=False,
            text=True,
        )

    def _cleanup_temp_dir(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def _sleep_before_ssh_retry(self, attempt: int, *, label: str) -> None:
        delay_seconds = _ssh_retry_delay_seconds(attempt)
        self._log_verbose(
            f"Retrying transient SSH failure in {delay_seconds}s "
            f"(attempt {attempt + 1}/{SSH_TRANSPORT_FAILURE_ATTEMPTS}): {label}"
        )
        time.sleep(delay_seconds)

    def _log_progress(self, message: str) -> None:
        if progress_enabled(self.config.verbosity):
            print(message)

    def _log_verbose(self, message: str) -> None:
        if verbose_enabled(self.config.verbosity):
            print(message)


def load_slurm_vllm_bootstrap_config(path: str | Path) -> SlurmVllmBootstrapConfig:
    """Load a strict Slurm vLLM bootstrap config from YAML."""

    config_path = Path(path)
    raw_config = load_yaml_mapping(config_path, kind="Slurm vLLM bootstrap config")
    return slurm_vllm_bootstrap_config_from_mapping(raw_config, source=str(config_path))


def slurm_vllm_bootstrap_config_from_mapping(
    values: Mapping[str, Any],
    *,
    source: str = "Slurm vLLM bootstrap config",
) -> SlurmVllmBootstrapConfig:
    field_names = {field.name for field in fields(SlurmVllmBootstrapConfig)}
    unknown = sorted(set(values) - field_names)
    if unknown:
        raise ValueError(
            f"{source} contains unsupported Slurm vLLM bootstrap fields: {', '.join(unknown)}."
        )

    hints = get_type_hints(SlurmVllmBootstrapConfig)
    converted = {
        field.name: _coerce_config_value(
            values[field.name],
            annotation=hints[field.name],
            field_name=field.name,
            source=source,
        )
        for field in fields(SlurmVllmBootstrapConfig)
        if field.name in values
    }
    return SlurmVllmBootstrapConfig(**converted)


def with_bootstrap_defaults(config: SlurmVllmBootstrapConfig) -> SlurmVllmBootstrapConfig:
    """Fill defaults derived from explicit bootstrap fields."""

    profile = bootstrap_cluster_profile(config.ssh_target)
    environment_name = config.environment_name or (
        profile.environment_name if profile is not None else ""
    )
    if not environment_name:
        raise ValueError("Slurm vLLM bootstrap config is missing: environment_name.")
    _require_valid_environment_name(environment_name)
    vllm_package = config.vllm_package or (profile.vllm_package if profile is not None else "")
    backend = config.backend or (profile.backend if profile is not None else DEFAULT_BACKEND)
    partition = config.partition or (profile.partition if profile is not None else "")
    ray_package = config.ray_package
    if profile is not None and config.nodes == 1 and config.ray_package == DEFAULT_RAY_PACKAGE:
        ray_package = profile.single_node_ray_package
    job_name_prefix = config.job_name_prefix or DEFAULT_JOB_NAME_PREFIX
    venv_path = _normalize_remote_venv_path(config.venv_path or f"~/.{environment_name}")
    job_name = config.job_name or _default_job_name(environment_name, job_name_prefix)
    return cast(
        SlurmVllmBootstrapConfig,
        replace(
            config,
            environment_name=environment_name,
            venv_path=venv_path,
            vllm_package=vllm_package,
            ray_package=ray_package,
            backend=backend,
            partition=partition,
            job_name=job_name,
            job_name_prefix=job_name_prefix,
        ),
    )


def bootstrap_cluster_profile(ssh_target: str) -> BootstrapClusterProfile | None:
    """Return shared-cluster bootstrap defaults for a target alias or hostname."""

    target = ssh_target.strip().lower()
    if not target:
        return None
    host = target.rsplit("@", maxsplit=1)[-1]
    if host.startswith("ssh://"):
        host = host.removeprefix("ssh://").rsplit("@", maxsplit=1)[-1]
    host = host.split(":", maxsplit=1)[0].split(".", maxsplit=1)[0]
    for cluster_name, profile in BOOTSTRAP_CLUSTER_PROFILES.items():
        if host == cluster_name or host.startswith(cluster_name):
            return profile
    return None


def validate_slurm_vllm_bootstrap_config(
    config: SlurmVllmBootstrapConfig,
) -> SlurmVllmBootstrapConfig:
    """Reject bootstrap configs that cannot produce a valid remote environment."""

    validate_verbosity(config.verbosity, label="Slurm vLLM bootstrap")
    for field_name in (
        "ssh_target",
        "environment_name",
        "venv_path",
        "vllm_package",
        "backend",
        "sbatch_cmd",
        "job_name",
        "job_name_prefix",
        "partition",
        "walltime",
        "memory",
    ):
        _require_non_empty_string(config, field_name)
    _require_valid_environment_name(config.environment_name)
    _require_safe_remote_venv_path(config.venv_path)
    _require_exact_vllm_package(config.vllm_package)
    _require_safe_package_spec(config.ray_package, field_name="ray_package")
    if config.backend not in {"auto", "rocm"}:
        raise ValueError("Slurm vLLM bootstrap backend must be 'auto' or 'rocm'.")
    for field_name in (
        "num_gpus",
        "cpus_per_task",
        "nodes",
        "check_interval_seconds",
        "timeout_seconds",
    ):
        _require_positive_int(config, field_name)
    return config


def render_bootstrap_sbatch_script(
    config: SlurmVllmBootstrapConfig,
    *,
    out_dir: str,
    helper_path: str | None = None,
) -> str:
    """Render the Slurm script used to provision the remote vLLM environment."""

    config = validate_slurm_vllm_bootstrap_config(with_bootstrap_defaults(config))
    resolved_helper_path = helper_path or f"{out_dir.rstrip('/')}/{BOOTSTRAP_HELPER_NAME}"
    template = _AtTemplate(_bootstrap_template_text())
    return template.substitute(
        sbatch_directives=_sbatch_directives(config, out_dir=out_dir),
        out_dir=shlex.quote(out_dir),
        environment_name=shlex.quote(config.environment_name),
        venv_path=shlex.quote(config.venv_path),
        vllm_package=shlex.quote(config.vllm_package),
        ray_package=shlex.quote(config.ray_package),
        backend=shlex.quote(config.backend),
        install_uv_if_missing=_shell_bool(config.install_uv_if_missing),
        force_bootstrap=_shell_bool(config.force),
        helper_path=shlex.quote(resolved_helper_path),
        success_manifest_name=SUCCESS_MANIFEST_NAME,
    )


def _validate_success_manifest(
    manifest: Mapping[str, Any],
    config: SlurmVllmBootstrapConfig,
    success_path: str,
    *,
    expected_job_id: str,
    expected_venv_path: str,
) -> None:
    expected_manifest_path = (
        f"{expected_venv_path}/remote-inference-launcher-bootstrap-manifest.json"
    )
    expected = {
        "schema_version": 1,
        "slurm_job_id": expected_job_id,
        "environment_name": config.environment_name,
        "venv_path": expected_venv_path,
        "python_bin": f"{expected_venv_path}/bin/python",
        "requested_vllm_package": config.vllm_package,
        "requested_ray_package": config.ray_package,
        "backend": config.backend,
        "manifest_path": expected_manifest_path,
    }
    for field_name, expected_value in expected.items():
        if manifest.get(field_name) != expected_value:
            raise RuntimeError(
                "Remote bootstrap manifest does not match requested config: "
                f"{field_name}={manifest.get(field_name)!r}, expected {expected_value!r}; "
                f"manifest={success_path}"
            )


def _sbatch_directives(config: SlurmVllmBootstrapConfig, *, out_dir: str) -> str:
    lines = [
        f"#SBATCH --job-name={config.job_name}",
        f"#SBATCH --partition={config.partition}",
        f"#SBATCH --nodes={config.nodes}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --mem={config.memory}",
        f"#SBATCH --cpus-per-task={config.cpus_per_task}",
        f"#SBATCH --gres=gpu:{config.num_gpus}",
        f"#SBATCH --time={config.walltime}",
        f"#SBATCH --output={out_dir.rstrip('/')}/std.log",
        f"#SBATCH --error={out_dir.rstrip('/')}/err.log",
    ]
    if config.exclude:
        lines.append(f"#SBATCH --exclude={config.exclude}")
    if config.nodelist:
        lines.append(f"#SBATCH --nodelist={config.nodelist}")
    return "\n".join(lines)


def _coerce_config_value(
    value: Any,
    *,
    annotation: object,
    field_name: str,
    source: str,
) -> object:
    if annotation is str:
        if field_name == "ray_package":
            if not isinstance(value, str):
                raise ValueError(f"{source} field {field_name!r} must be a string.")
            return value
        converted = _string_value(value, field_name=field_name, source=source)
        if field_name == "verbosity":
            validate_verbosity(converted, label=source)
        return converted
    if annotation is int:
        return _positive_int_value(value, field_name=field_name, source=source)
    if annotation is bool:
        return _bool_value(value, field_name=field_name, source=source)
    raise TypeError(
        f"Unsupported Slurm vLLM bootstrap config field annotation for {field_name}: {annotation}"
    )


def _string_value(value: Any, *, field_name: str, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} field {field_name!r} must be a non-empty string.")
    return value


def _positive_int_value(value: Any, *, field_name: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{source} field {field_name!r} must be a positive integer.")
    return value


def _bool_value(value: Any, *, field_name: str, source: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{source} field {field_name!r} must be a boolean.")
    return value


def _require_non_empty_string(config: SlurmVllmBootstrapConfig, field_name: str) -> None:
    value = getattr(config, field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Slurm vLLM bootstrap {field_name} must be a non-empty string.")


def _require_positive_int(config: SlurmVllmBootstrapConfig, field_name: str) -> None:
    value = getattr(config, field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Slurm vLLM bootstrap {field_name} must be positive.")


def _require_valid_environment_name(environment_name: str) -> None:
    if not ENVIRONMENT_NAME_PATTERN.fullmatch(environment_name):
        raise ValueError(
            "Slurm vLLM bootstrap environment_name must start with a letter or number "
            "and contain only letters, numbers, '.', '_', or '-'."
        )


def _require_safe_remote_venv_path(venv_path: str) -> None:
    if not venv_path.startswith(("/", "~/")):
        raise ValueError("Slurm vLLM bootstrap venv_path must be absolute or start with '~/'.")
    if venv_path in {"/", "~", "~/"}:
        raise ValueError("Slurm vLLM bootstrap venv_path must name a virtual environment.")

    path_without_anchor = venv_path[2:] if venv_path.startswith("~/") else venv_path.lstrip("/")
    components = [component for component in path_without_anchor.split("/") if component]
    if not components or any(component in {".", ".."} for component in components):
        raise ValueError(
            "Slurm vLLM bootstrap venv_path must not contain '.' or '..' path components."
        )


def _require_exact_vllm_package(vllm_package: str) -> None:
    if not vllm_package.startswith("vllm==") or vllm_package == "vllm==":
        raise ValueError("Slurm vLLM bootstrap vllm_package must be an exact 'vllm==VERSION'.")


def _require_safe_package_spec(package: str, *, field_name: str) -> None:
    if any(character.isspace() for character in package):
        raise ValueError(f"Slurm vLLM bootstrap {field_name} must not contain whitespace.")


def _normalize_remote_venv_path(venv_path: str) -> str:
    if venv_path in {"/", "~", "~/"}:
        return venv_path
    return venv_path.rstrip("/")


def _default_remote_out_dir(remote_home: str, config: SlurmVllmBootstrapConfig) -> str:
    root = config.remote_out_dir_root or f"{remote_home.rstrip('/')}/{DEFAULT_REMOTE_OUT_DIR_ROOT}"
    return f"{root.rstrip('/')}/{config.job_name}"


def _default_job_name(environment_name: str, prefix: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    user = os.getenv("USER") or "unknown"
    return f"{prefix}_{environment_name}_{stamp}_{user}"


def _expand_remote_home(path: str, remote_home: str) -> str:
    if path == "~":
        return remote_home
    if path.startswith("~/"):
        return f"{remote_home.rstrip('/')}/{path[2:]}"
    return path


def _require_expanded_remote_venv_path_safe(venv_path: str, remote_home: str) -> None:
    venv_components = _absolute_remote_path_components(venv_path, label="venv_path")
    home_components = _absolute_remote_path_components(remote_home, label="remote home")
    if _is_same_or_parent_path(venv_components, home_components):
        raise ValueError(
            "Slurm vLLM bootstrap venv_path must not be the remote home directory "
            "or one of its parents."
        )


def _absolute_remote_path_components(path: str, *, label: str) -> tuple[str, ...]:
    if not path.startswith("/"):
        raise ValueError(f"Slurm vLLM bootstrap {label} must be absolute after expansion.")
    components = tuple(component for component in path.split("/") if component)
    if any(component in {".", ".."} for component in components):
        raise ValueError(f"Slurm vLLM bootstrap {label} must not contain '.' or '..'.")
    return components


def _is_same_or_parent_path(candidate: tuple[str, ...], child: tuple[str, ...]) -> bool:
    return len(candidate) <= len(child) and child[: len(candidate)] == candidate


def _shell_bool(value: bool) -> str:
    return "1" if value else "0"


def _timed_out(started_at: float, timeout_seconds: int) -> bool:
    return timeout_seconds > 0 and time.monotonic() - started_at >= timeout_seconds


def _should_retry_ssh(
    completed: subprocess.CompletedProcess,
    attempt: int,
    *,
    max_attempts: int = SSH_TRANSPORT_FAILURE_ATTEMPTS,
) -> bool:
    return attempt < max_attempts and is_transient_ssh_failure(completed)


def _ssh_process_error(
    command: str,
    completed: subprocess.CompletedProcess,
    *,
    command_is_retry_safe: bool,
) -> str:
    label = command
    if not command_is_retry_safe and is_transient_ssh_failure(completed):
        label = f"{command} (not retried because this remote command is not safe to replay)"
    return process_error(label, completed)


def _ssh_retry_delay_seconds(attempt: int) -> int:
    return min(2 * attempt, 10)


def _bootstrap_template_text() -> str:
    return (
        resources.files("remote_inference_launcher.templates")
        .joinpath("bootstrap_vllm.sbatch")
        .read_text(encoding="utf-8")
    )


def _bootstrap_helper_text() -> str:
    return (
        resources.files("remote_inference_launcher.templates")
        .joinpath(BOOTSTRAP_HELPER_NAME)
        .read_text(encoding="utf-8")
    )


class _AtTemplate(Template):
    delimiter = "@"


_FAILED_TERMINAL_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "NODE_FAIL",
    "DEADLINE",
}
