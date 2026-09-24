"""Local and SSH vLLM Python environment bootstrap."""

from __future__ import annotations

import json
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

from remote_inference_launcher.process import process_error
from remote_inference_launcher.slurm_vllm_bootstrap import (
    SlurmVllmBootstrapConfig,
    SlurmVllmBootstrapper,
    slurm_vllm_bootstrap_config_from_mapping,
)
from remote_inference_launcher.ssh import (
    SSH_TRANSPORT_FAILURE_ATTEMPTS,
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
DEFAULT_PYTHON_VERSION = "3.12"
DEFAULT_RAY_PACKAGE = "ray"
DEFAULT_LOCAL_OUT_DIR_ROOT = ".remote-inference-launcher/bootstrap-vllm"
DEFAULT_REMOTE_OUT_DIR_ROOT = "tmp/remote-inference-launcher/bootstrap-vllm"
BOOTSTRAP_SCRIPT_NAME = "bootstrap_vllm_env.sh"
BOOTSTRAP_HELPER_NAME = "vllm_bootstrap_manifest.py"
SUCCESS_MANIFEST_NAME = "bootstrap_success.json"
LOCAL_VLLM_BOOTSTRAP_LABEL = "Local vLLM bootstrap"
SSH_VLLM_BOOTSTRAP_LABEL = "SSH vLLM bootstrap"
ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class LocalVllmBootstrapConfig:
    """Configuration for provisioning a local vLLM Python environment."""

    verbosity: str = "progress"
    environment_name: str = ""
    venv_path: str = ""
    vllm_package: str = ""
    ray_package: str = DEFAULT_RAY_PACKAGE
    backend: str = DEFAULT_BACKEND
    python: str = DEFAULT_PYTHON_VERSION
    install_uv_if_missing: bool = False
    force: bool = False
    setup_cmd: str = ""
    out_dir: str = ""
    out_dir_root: str = ""


@dataclass(frozen=True)
class SshVllmBootstrapConfig:
    """Configuration for provisioning a vLLM Python environment over SSH."""

    ssh_target: str = ""
    verbosity: str = "progress"
    environment_name: str = ""
    venv_path: str = ""
    vllm_package: str = ""
    ray_package: str = DEFAULT_RAY_PACKAGE
    backend: str = DEFAULT_BACKEND
    python: str = DEFAULT_PYTHON_VERSION
    install_uv_if_missing: bool = False
    force: bool = False
    setup_cmd: str = ""
    out_dir: str = ""
    remote_out_dir_root: str = ""


@dataclass(frozen=True)
class VllmBootstrapResult:
    """Completed local or SSH bootstrap details."""

    kind: str
    target: str
    out_dir: str
    manifest: Mapping[str, Any]


class LocalVllmBootstrapper:
    """Create or verify a local uv environment with vLLM installed."""

    def __init__(self, config: LocalVllmBootstrapConfig) -> None:
        self.config = validate_local_vllm_bootstrap_config(
            with_local_vllm_bootstrap_defaults(config)
        )

    def run(self) -> VllmBootstrapResult:
        """Run the local bootstrap script and return its success manifest."""

        home = Path.home().resolve()
        venv_path = _expand_local_path(self.config.venv_path).resolve()
        _require_expanded_local_venv_path_safe(venv_path, home)
        out_dir = _resolved_local_out_dir(self.config)
        out_dir.mkdir(parents=True, exist_ok=True)
        helper_path = out_dir / BOOTSTRAP_HELPER_NAME
        script_path = out_dir / BOOTSTRAP_SCRIPT_NAME
        success_path = out_dir / SUCCESS_MANIFEST_NAME
        helper_path.write_text(_bootstrap_helper_text(), encoding="utf-8")
        script_path.write_text(
            render_vllm_bootstrap_script(
                self.config,
                out_dir=str(out_dir),
                helper_path=str(helper_path),
                success_path=str(success_path),
            ),
            encoding="utf-8",
        )
        success_path.unlink(missing_ok=True)
        completed = subprocess.run(
            ["bash", str(script_path)],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                process_error("local vLLM bootstrap", completed)
                + f"\nLocal bootstrap logs: {out_dir / 'bootstrap.log'}"
            )
        manifest = _read_json_file(success_path)
        _validate_bootstrap_manifest(
            manifest,
            environment_name=self.config.environment_name,
            venv_path=str(venv_path),
            vllm_package=self.config.vllm_package,
            ray_package=self.config.ray_package,
            backend=self.config.backend,
            success_path=str(success_path),
        )
        return VllmBootstrapResult(
            kind="local_vllm_bootstrap",
            target="local",
            out_dir=str(out_dir),
            manifest=manifest,
        )


class SshVllmBootstrapper:
    """Create or verify a remote uv environment with vLLM installed over SSH."""

    def __init__(self, config: SshVllmBootstrapConfig) -> None:
        self.config = validate_ssh_vllm_bootstrap_config(with_ssh_vllm_bootstrap_defaults(config))
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def run(self) -> VllmBootstrapResult:
        """Run the remote bootstrap script and return its success manifest."""

        self._temp_dir = tempfile.TemporaryDirectory(
            prefix="remote-inference-launcher-ssh-bootstrap-vllm."
        )
        try:
            remote_home = self._ssh_text("cd && pwd").strip()
            venv_path = _expand_remote_home(self.config.venv_path, remote_home)
            _require_expanded_remote_venv_path_safe(venv_path, remote_home)
            out_dir = _expand_remote_home(
                self.config.out_dir or _default_remote_out_dir(remote_home, self.config),
                remote_home,
            )
            remote_helper = f"{out_dir.rstrip('/')}/{BOOTSTRAP_HELPER_NAME}"
            remote_script = f"{out_dir.rstrip('/')}/{BOOTSTRAP_SCRIPT_NAME}"
            success_path = f"{out_dir.rstrip('/')}/{SUCCESS_MANIFEST_NAME}"
            local_helper = Path(self._temp_dir.name) / BOOTSTRAP_HELPER_NAME
            local_script = Path(self._temp_dir.name) / BOOTSTRAP_SCRIPT_NAME
            local_helper.write_text(_bootstrap_helper_text(), encoding="utf-8")
            local_script.write_text(
                render_vllm_bootstrap_script(
                    self.config,
                    out_dir=out_dir,
                    helper_path=remote_helper,
                    success_path=success_path,
                ),
                encoding="utf-8",
            )
            self._ssh_text(f"mkdir -p {shlex.quote(out_dir)}")
            self._ssh_text(f"rm -f {shlex.quote(success_path)}")
            self._copy_to_remote(local_helper, remote_helper)
            self._copy_to_remote(local_script, remote_script)
            self._ssh_text(f"chmod +x {shlex.quote(remote_script)}")
            self._log_progress(
                f"Bootstrapping remote vLLM environment on {self.config.ssh_target}."
            )
            self._log_progress(f"Remote bootstrap logs: {out_dir}")
            self._ssh_text(f"bash {shlex.quote(remote_script)}", command_is_retry_safe=False)
            manifest = self._read_remote_manifest(success_path)
            _validate_bootstrap_manifest(
                manifest,
                environment_name=self.config.environment_name,
                venv_path=venv_path,
                vllm_package=self.config.vllm_package,
                ray_package=self.config.ray_package,
                backend=self.config.backend,
                success_path=success_path,
            )
            return VllmBootstrapResult(
                kind="ssh_vllm_bootstrap",
                target=self.config.ssh_target,
                out_dir=out_dir,
                manifest=manifest,
            )
        finally:
            if self._temp_dir is not None:
                self._temp_dir.cleanup()
                self._temp_dir = None

    def _read_remote_manifest(self, success_path: str) -> Mapping[str, Any]:
        raw_manifest = self._ssh_text(f"cat {shlex.quote(success_path)}")
        try:
            manifest = json.loads(raw_manifest)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Remote bootstrap manifest is not valid JSON: {success_path}"
            ) from error
        if not isinstance(manifest, dict):
            raise RuntimeError(f"Remote bootstrap manifest is not a JSON object: {success_path}")
        return manifest

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
                completed = subprocess.run(command, stdin=source, capture_output=True, check=False)
            if not _should_retry_ssh(completed, attempt):
                break
            self._sleep_before_ssh_retry(attempt, label="remote bootstrap copy")
        if completed.returncode:
            raise RuntimeError(process_error("remote bootstrap copy", completed))

    def _ssh_text(self, command: str, *, command_is_retry_safe: bool = True) -> str:
        self._log_verbose(f"Remote command on {self.config.ssh_target}: {command}")
        ssh_command = [
            "ssh",
            *ssh_options(),
            "-T",
            self.config.ssh_target,
            f"bash -lc {shlex.quote(command)}",
        ]
        attempts = SSH_TRANSPORT_FAILURE_ATTEMPTS if command_is_retry_safe else 1
        for attempt in range(1, attempts + 1):
            completed = subprocess.run(ssh_command, capture_output=True, check=False, text=True)
            if not _should_retry_ssh(completed, attempt, max_attempts=attempts):
                break
            self._sleep_before_ssh_retry(attempt, label=command)
        if completed.returncode:
            raise RuntimeError(process_error(command, completed))
        return completed.stdout

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


def load_vllm_bootstrap_config(path: str | Path) -> object:
    """Load a strict vLLM bootstrap config from YAML."""

    raw_config = load_yaml_mapping(path, kind="vLLM bootstrap config")
    return vllm_bootstrap_config_from_mapping(raw_config, source=str(Path(path)))


def vllm_bootstrap_config_from_mapping(
    values: Mapping[str, Any],
    *,
    source: str = "vLLM bootstrap config",
) -> object:
    """Build one bootstrap config object from a strict mapping."""

    kind = values.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"{source} field 'kind' must be a non-empty string.")
    normalized_kind = kind.strip()
    config_values = {key: value for key, value in values.items() if key != "kind"}
    if normalized_kind == "local_vllm_bootstrap":
        return local_vllm_bootstrap_config_from_mapping(config_values, source=source)
    if normalized_kind == "ssh_vllm_bootstrap":
        return ssh_vllm_bootstrap_config_from_mapping(config_values, source=source)
    if normalized_kind == "slurm_vllm_bootstrap":
        return slurm_vllm_bootstrap_config_from_mapping(config_values, source=source)
    raise ValueError(
        f"{source} field 'kind' must be one of: "
        "local_vllm_bootstrap, ssh_vllm_bootstrap, slurm_vllm_bootstrap."
    )


def local_vllm_bootstrap_config_from_mapping(
    values: Mapping[str, Any],
    *,
    source: str = "local vLLM bootstrap config",
) -> LocalVllmBootstrapConfig:
    """Build a local bootstrap config from a strict mapping."""

    return _dataclass_config_from_mapping(LocalVllmBootstrapConfig, values, source=source)


def ssh_vllm_bootstrap_config_from_mapping(
    values: Mapping[str, Any],
    *,
    source: str = "SSH vLLM bootstrap config",
) -> SshVllmBootstrapConfig:
    """Build an SSH bootstrap config from a strict mapping."""

    return _dataclass_config_from_mapping(SshVllmBootstrapConfig, values, source=source)


def load_vllm_bootstrapper(path: str | Path):
    """Load a bootstrap config and return the corresponding bootstrapper."""

    return bootstrapper_from_config(load_vllm_bootstrap_config(path))


def bootstrapper_from_config(config: object):
    """Build a bootstrapper from an already parsed config object."""

    if isinstance(config, LocalVllmBootstrapConfig):
        return LocalVllmBootstrapper(config)
    if isinstance(config, SshVllmBootstrapConfig):
        return SshVllmBootstrapper(config)
    if isinstance(config, SlurmVllmBootstrapConfig):
        return SlurmVllmBootstrapper(config)
    if hasattr(config, "run"):
        return config
    raise TypeError(f"Unsupported vLLM bootstrap config: {type(config).__name__}")


def load_local_vllm_bootstrap_config(path: str | Path) -> LocalVllmBootstrapConfig:
    """Load a local bootstrap config from YAML without requiring a `kind` field."""

    raw_config = load_yaml_mapping(path, kind="local vLLM bootstrap config")
    raw_config = {key: value for key, value in raw_config.items() if key != "kind"}
    return local_vllm_bootstrap_config_from_mapping(raw_config, source=str(Path(path)))


def load_ssh_vllm_bootstrap_config(path: str | Path) -> SshVllmBootstrapConfig:
    """Load an SSH bootstrap config from YAML without requiring a `kind` field."""

    raw_config = load_yaml_mapping(path, kind="SSH vLLM bootstrap config")
    raw_config = {key: value for key, value in raw_config.items() if key != "kind"}
    return ssh_vllm_bootstrap_config_from_mapping(raw_config, source=str(Path(path)))


def with_local_vllm_bootstrap_defaults(
    config: LocalVllmBootstrapConfig,
) -> LocalVllmBootstrapConfig:
    """Fill defaults derived from local bootstrap fields."""

    if not config.environment_name:
        raise ValueError("Local vLLM bootstrap config is missing: environment_name.")
    _require_valid_environment_name(config.environment_name, label=LOCAL_VLLM_BOOTSTRAP_LABEL)
    venv_path = _normalize_venv_path(config.venv_path or f"~/.{config.environment_name}")
    return cast(LocalVllmBootstrapConfig, replace(config, venv_path=venv_path))


def with_ssh_vllm_bootstrap_defaults(config: SshVllmBootstrapConfig) -> SshVllmBootstrapConfig:
    """Fill defaults derived from SSH bootstrap fields."""

    if not config.environment_name:
        raise ValueError("SSH vLLM bootstrap config is missing: environment_name.")
    _require_valid_environment_name(config.environment_name, label=SSH_VLLM_BOOTSTRAP_LABEL)
    venv_path = _normalize_venv_path(config.venv_path or f"~/.{config.environment_name}")
    return cast(SshVllmBootstrapConfig, replace(config, venv_path=venv_path))


def validate_local_vllm_bootstrap_config(
    config: LocalVllmBootstrapConfig,
) -> LocalVllmBootstrapConfig:
    """Reject local bootstrap configs that cannot produce a valid environment."""

    validate_verbosity(config.verbosity, label=LOCAL_VLLM_BOOTSTRAP_LABEL)
    for field_name in ("environment_name", "venv_path", "vllm_package", "backend", "python"):
        _require_non_empty_string(config, field_name, label=LOCAL_VLLM_BOOTSTRAP_LABEL)
    _validate_shared_bootstrap_fields(
        environment_name=config.environment_name,
        venv_path=config.venv_path,
        vllm_package=config.vllm_package,
        ray_package=config.ray_package,
        backend=config.backend,
        python=config.python,
        label=LOCAL_VLLM_BOOTSTRAP_LABEL,
    )
    return config


def validate_ssh_vllm_bootstrap_config(
    config: SshVllmBootstrapConfig,
) -> SshVllmBootstrapConfig:
    """Reject SSH bootstrap configs that cannot produce a valid environment."""

    validate_verbosity(config.verbosity, label=SSH_VLLM_BOOTSTRAP_LABEL)
    for field_name in (
        "ssh_target",
        "environment_name",
        "venv_path",
        "vllm_package",
        "backend",
        "python",
    ):
        _require_non_empty_string(config, field_name, label=SSH_VLLM_BOOTSTRAP_LABEL)
    _validate_shared_bootstrap_fields(
        environment_name=config.environment_name,
        venv_path=config.venv_path,
        vllm_package=config.vllm_package,
        ray_package=config.ray_package,
        backend=config.backend,
        python=config.python,
        label=SSH_VLLM_BOOTSTRAP_LABEL,
    )
    return config


def render_vllm_bootstrap_script(
    config: LocalVllmBootstrapConfig | SshVllmBootstrapConfig,
    *,
    out_dir: str,
    helper_path: str,
    success_path: str,
) -> str:
    """Render the local/SSH bootstrap shell script."""

    if isinstance(config, LocalVllmBootstrapConfig):
        config = validate_local_vllm_bootstrap_config(with_local_vllm_bootstrap_defaults(config))
    else:
        config = validate_ssh_vllm_bootstrap_config(with_ssh_vllm_bootstrap_defaults(config))
    setup_cmd_block = f"\n{config.setup_cmd}\n" if config.setup_cmd else ""
    template = _AtTemplate(_bootstrap_script_text())
    return template.substitute(
        out_dir=shlex.quote(out_dir),
        environment_name=shlex.quote(config.environment_name),
        venv_path=shlex.quote(config.venv_path),
        vllm_package=shlex.quote(config.vllm_package),
        ray_package=shlex.quote(config.ray_package),
        backend=shlex.quote(config.backend),
        python_version=shlex.quote(config.python),
        install_uv_if_missing=_shell_bool(config.install_uv_if_missing),
        force_bootstrap=_shell_bool(config.force),
        helper_path=shlex.quote(helper_path),
        success_path=shlex.quote(success_path),
        setup_cmd_block=setup_cmd_block,
    )


def _dataclass_config_from_mapping(
    config_class: type[LocalVllmBootstrapConfig] | type[SshVllmBootstrapConfig],
    values: Mapping[str, Any],
    *,
    source: str,
) -> object:
    field_names = {field.name for field in fields(config_class)}
    unknown = sorted(set(values) - field_names)
    if unknown:
        raise ValueError(f"{source} contains unsupported fields: {', '.join(unknown)}.")
    hints = get_type_hints(config_class)
    converted = {
        field.name: _coerce_config_value(
            values[field.name],
            annotation=hints[field.name],
            field_name=field.name,
            source=source,
        )
        for field in fields(config_class)
        if field.name in values
    }
    return config_class(**converted)


def _coerce_string_config_value(value: Any, *, field_name: str, source: str) -> str:
    if field_name == "ray_package":
        if not isinstance(value, str):
            raise ValueError(f"{source} field {field_name!r} must be a string.")
        return value
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} field {field_name!r} must be a non-empty string.")
    return value


def _coerce_int_config_value(value: Any, *, field_name: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{source} field {field_name!r} must be a positive integer.")
    return value


def _coerce_bool_config_value(value: Any, *, field_name: str, source: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{source} field {field_name!r} must be a boolean.")
    return value


def _coerce_config_value(
    value: Any,
    *,
    annotation: object,
    field_name: str,
    source: str,
) -> object:
    if annotation is str:
        return _coerce_string_config_value(value, field_name=field_name, source=source)
    if annotation is int:
        return _coerce_int_config_value(value, field_name=field_name, source=source)
    if annotation is bool:
        return _coerce_bool_config_value(value, field_name=field_name, source=source)
    raise TypeError(f"Unsupported bootstrap config field annotation for {field_name}: {annotation}")


def _validate_shared_bootstrap_fields(
    *,
    environment_name: str,
    venv_path: str,
    vllm_package: str,
    ray_package: str,
    backend: str,
    python: str,
    label: str,
) -> None:
    _require_valid_environment_name(environment_name, label=label)
    _require_safe_venv_path(venv_path, label=label)
    _require_exact_vllm_package(vllm_package, label=label)
    _require_safe_package_spec(ray_package, field_name="ray_package", label=label)
    if backend not in {"auto", "rocm"}:
        raise ValueError(f"{label} backend must be 'auto' or 'rocm'.")
    if any(character.isspace() for character in python):
        raise ValueError(f"{label} python must not contain whitespace.")


def _validate_bootstrap_manifest(
    manifest: Mapping[str, Any],
    *,
    environment_name: str,
    venv_path: str,
    vllm_package: str,
    ray_package: str,
    backend: str,
    success_path: str,
) -> None:
    expected_manifest_path = f"{venv_path}/remote-inference-launcher-bootstrap-manifest.json"
    expected = {
        "schema_version": 1,
        "environment_name": environment_name,
        "venv_path": venv_path,
        "python_bin": f"{venv_path}/bin/python",
        "requested_vllm_package": vllm_package,
        "requested_ray_package": ray_package,
        "backend": backend,
        "manifest_path": expected_manifest_path,
    }
    for field_name, expected_value in expected.items():
        if manifest.get(field_name) != expected_value:
            raise RuntimeError(
                "vLLM bootstrap manifest does not match requested config: "
                f"{field_name}={manifest.get(field_name)!r}, expected {expected_value!r}; "
                f"manifest={success_path}"
            )


def _read_json_file(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Bootstrap manifest is not readable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Bootstrap manifest is not a JSON object: {path}")
    return payload


def _resolved_local_out_dir(config: LocalVllmBootstrapConfig) -> Path:
    if config.out_dir:
        return _expand_local_path(config.out_dir).resolve()
    root = (
        _expand_local_path(config.out_dir_root).resolve()
        if config.out_dir_root
        else Path.home().resolve() / DEFAULT_LOCAL_OUT_DIR_ROOT
    )
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return root / f"{_safe_label(config.environment_name)}-{stamp}"


def _default_remote_out_dir(remote_home: str, config: SshVllmBootstrapConfig) -> str:
    root = config.remote_out_dir_root or f"{remote_home.rstrip('/')}/{DEFAULT_REMOTE_OUT_DIR_ROOT}"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{root.rstrip('/')}/{_safe_label(config.environment_name)}-{stamp}"


def _expand_local_path(path: str) -> Path:
    return Path(path).expanduser()


def _expand_remote_home(path: str, remote_home: str) -> str:
    if path == "~":
        return remote_home
    if path.startswith("~/"):
        return f"{remote_home.rstrip('/')}/{path[2:]}"
    return path


def _require_expanded_local_venv_path_safe(venv_path: Path, home: Path) -> None:
    if not venv_path.is_absolute():
        raise ValueError("Local vLLM bootstrap venv_path must be absolute after expansion.")
    try:
        if _is_same_or_parent_path(venv_path.parts, home.parts):
            raise ValueError(
                "Local vLLM bootstrap venv_path must not be the local home directory "
                "or one of its parents."
            )
    except RuntimeError:
        return


def _require_expanded_remote_venv_path_safe(venv_path: str, remote_home: str) -> None:
    venv_components = _absolute_remote_path_components(venv_path, label="venv_path")
    home_components = _absolute_remote_path_components(remote_home, label="remote home")
    if _is_same_or_parent_path(venv_components, home_components):
        raise ValueError(
            "SSH vLLM bootstrap venv_path must not be the remote home directory "
            "or one of its parents."
        )


def _absolute_remote_path_components(path: str, *, label: str) -> tuple[str, ...]:
    if not path.startswith("/"):
        raise ValueError(f"SSH vLLM bootstrap {label} must be absolute after expansion.")
    components = tuple(component for component in path.split("/") if component)
    if any(component in {".", ".."} for component in components):
        raise ValueError(f"SSH vLLM bootstrap {label} must not contain '.' or '..'.")
    return components


def _is_same_or_parent_path(candidate: tuple[str, ...], child: tuple[str, ...]) -> bool:
    return len(candidate) <= len(child) and child[: len(candidate)] == candidate


def _require_non_empty_string(config: object, field_name: str, *, label: str) -> None:
    value = getattr(config, field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} {field_name} must be a non-empty string.")


def _require_valid_environment_name(environment_name: str, *, label: str) -> None:
    if not ENVIRONMENT_NAME_PATTERN.fullmatch(environment_name):
        raise ValueError(
            f"{label} environment_name must start with a letter or number "
            "and contain only letters, numbers, '.', '_', or '-'."
        )


def _require_safe_venv_path(venv_path: str, *, label: str) -> None:
    if not venv_path.startswith(("/", "~/")):
        raise ValueError(f"{label} venv_path must be absolute or start with '~/'.")
    if venv_path in {"/", "~", "~/"}:
        raise ValueError(f"{label} venv_path must name a virtual environment.")
    path_without_anchor = venv_path[2:] if venv_path.startswith("~/") else venv_path.lstrip("/")
    components = [component for component in path_without_anchor.split("/") if component]
    if not components or any(component in {".", ".."} for component in components):
        raise ValueError(f"{label} venv_path must not contain '.' or '..' path components.")


def _require_exact_vllm_package(vllm_package: str, *, label: str) -> None:
    if not vllm_package.startswith("vllm==") or vllm_package == "vllm==":
        raise ValueError(f"{label} vllm_package must be an exact 'vllm==VERSION'.")


def _require_safe_package_spec(package: str, *, field_name: str, label: str) -> None:
    if any(character.isspace() for character in package):
        raise ValueError(f"{label} {field_name} must not contain whitespace.")


def _normalize_venv_path(venv_path: str) -> str:
    if venv_path in {"/", "~", "~/"}:
        return venv_path
    return venv_path.rstrip("/")


def _safe_label(value: str) -> str:
    safe = "".join(
        char if char.isascii() and (char.isalnum() or char in {"-", "_"}) else "_" for char in value
    )
    return safe.strip("_") or "vllm"


def _shell_bool(value: bool) -> str:
    return "1" if value else "0"


def _should_retry_ssh(
    completed: subprocess.CompletedProcess,
    attempt: int,
    *,
    max_attempts: int = SSH_TRANSPORT_FAILURE_ATTEMPTS,
) -> bool:
    return attempt < max_attempts and is_transient_ssh_failure(completed)


def _ssh_retry_delay_seconds(attempt: int) -> int:
    return min(2 * attempt, 10)


def _bootstrap_script_text() -> str:
    return (
        resources.files("remote_inference_launcher.templates")
        .joinpath(BOOTSTRAP_SCRIPT_NAME)
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
