"""Local vLLM process launcher."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import BinaryIO

from remote_inference_launcher.config_types import DiagnosticsConfig
from remote_inference_launcher.core import InferenceSession
from remote_inference_launcher.diagnostics import (
    CapacityInfo,
    classify_failure,
    parse_vllm_capacity,
    sanitized_excerpt,
)
from remote_inference_launcher.plan_paths import model_label, safe_endpoint_label
from remote_inference_launcher.ports import reserve_local_port
from remote_inference_launcher.readiness import (
    ReadinessConfig,
    ReadinessResult,
    validate_readiness_config,
    wait_for_openai_readiness,
)
from remote_inference_launcher.resource_budget import (
    ResourceBudgetReservation,
    acquire_current_resource_budget,
)
from remote_inference_launcher.summaries import SummaryWriter, endpoint_summary, new_run_id

DEFAULT_READY_TIMEOUT_SECONDS = 7200
DEFAULT_CHECK_INTERVAL_SECONDS = 10
DEFAULT_OUT_DIR_ROOT = ".remote-inference-launcher/local-vllm"


@dataclass(frozen=True)
class LocalVllmConfig:
    """Configuration for one local vLLM server session."""

    name: str = "default"
    model: str = ""
    served_model_name: str = ""
    host: str = "0.0.0.0"
    port: int | None = None
    api_key: str = ""
    python_bin: str = "python"
    out_dir: str = ""
    runtime_tmp_root: str = ""
    hf_home: str = ""
    target_device: str = ""
    rocm_eager_fallback: bool = True
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    data_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    extra_args: tuple[str, ...] = ()
    check_interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS
    ready_timeout_seconds: int = DEFAULT_READY_TIMEOUT_SECONDS
    keep_server: bool = False
    readiness: ReadinessConfig = field(default_factory=ReadinessConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    launch_summary_path: str = ""
    overwrite_launch_summary: bool = False


class LocalVllmLauncher:
    """Start local vLLM and wait for OpenAI-compatible API readiness."""

    owns_resources = True

    def __init__(self, config: LocalVllmConfig, *, run_id: str | None = None) -> None:
        self.config = validate_local_vllm_config(with_local_vllm_defaults(config))
        self._process: subprocess.Popen[bytes] | None = None
        self._log_file: BinaryIO | None = None
        self._out_dir: Path | None = None
        self._log_path: Path | None = None
        self._runtime_tmp_root: Path | None = None
        self._run_id = run_id or new_run_id()
        self._summary = SummaryWriter(
            self.config.launch_summary_path,
            run_id=self._run_id,
            endpoint_name=self.config.name,
            backend_kind="local_vllm",
            overwrite=self.config.overwrite_launch_summary,
        )
        self._last_summary: dict[str, object] | None = None
        self._budget_token: ResourceBudgetReservation | None = None

    def start(self) -> InferenceSession:
        """Start the local server and return a ready inference session."""

        self._summary.reserve()
        self._budget_token = acquire_current_resource_budget(
            candidate_attempts=1,
            requested_gpus=_requested_gpus(self.config),
        )
        self._out_dir = None
        self._log_path = None
        try:
            _require_vllm_available(self.config.python_bin)
            reservation = reserve_local_port(self.config.host, self.config.port)
            try:
                self.config = with_local_vllm_defaults(self.config, port=reservation.port)
                self._out_dir = _resolved_out_dir(self.config, run_id=self._run_id)
                self._out_dir.mkdir(parents=True, exist_ok=False)
                self._runtime_tmp_root = _resolved_runtime_tmp_root(
                    self.config,
                    self._out_dir,
                    run_id=self._run_id,
                )
                self._runtime_tmp_root.mkdir(parents=True, exist_ok=True)

                log_path = self._out_dir / "vllm.log"
                self._log_path = log_path
                command = vllm_server_command(self.config)
                env = local_vllm_env(self.config, runtime_tmp_root=self._runtime_tmp_root)
                self._log_file = log_path.open("ab")
                reservation.close()
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                self._write_summary("STARTING")
            finally:
                reservation.close()
            readiness = self._wait_for_ready(log_path)
            api_base = _api_base_for_config(self.config)
            capacity = _parse_capacity_if_enabled(
                self.config,
                _tail_log(log_path, lines=400),
                max_model_len=self.config.max_model_len,
                max_num_seqs=self.config.max_num_seqs,
            )
            session = InferenceSession(
                name=self.config.name,
                api_base=api_base,
                served_model_name=self.config.served_model_name,
                model=self.config.model,
                api_key=self.config.api_key,
                local_port=self.config.port,
                logs=str(self._out_dir),
                pid=self._process.pid if self._process is not None else None,
                backend_kind="local_vllm",
                run_id=self._run_id,
                summary_path=str(self._summary.path),
                cleanup_command=f"kill -TERM -{self._process.pid}" if self._process else "",
                metadata={
                    "readiness": readiness.to_summary(),
                    "recommended_benchmark_max_parallel": (
                        capacity.recommended_benchmark_max_parallel
                        if capacity is not None
                        else None
                    ),
                    "launch_summary_path": str(self._summary.path),
                },
            )
            self._write_summary("READY", readiness=readiness, capacity=capacity, session=session)
            return session
        except BaseException as error:
            self._write_failure_summary(error)
            self.stop()
            raise

    def stop(self) -> None:
        """Stop the local process unless configured to keep it."""

        if self._process is not None and not self.config.keep_server:
            _terminate_process_group(self._process)
            self._process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if (
            self._runtime_tmp_root is not None
            and self._runtime_tmp_root.exists()
            and not self.config.keep_server
        ):
            shutil.rmtree(self._runtime_tmp_root)
            self._runtime_tmp_root = None
        if (
            self._last_summary is not None
            and self._last_summary.get("lifecycle_state") != "FAILED"
            and not self.config.keep_server
        ):
            released = dict(self._last_summary)
            released["lifecycle_state"] = "RELEASED"
            self._summary.write(released)
        if self._budget_token is not None:
            self._budget_token.release()
            self._budget_token = None

    def reserve_summary(self):
        """Reserve the launch summary path before starting the local process."""

        return self._summary.reserve()

    @contextmanager
    def running(self) -> Iterator[InferenceSession]:
        """Context manager that stops the launcher on exit."""

        session = self.start()
        try:
            yield session
        finally:
            self.stop()

    def is_running(self) -> bool:
        """Return whether the local vLLM process is still alive."""

        return self._process is not None and self._process.poll() is None

    def _wait_for_ready(self, log_path: Path) -> ReadinessResult:
        def ensure_process_alive() -> None:
            if self._process is None:
                raise RuntimeError("Local vLLM process was not started.")
            returncode = self._process.poll()
            if returncode is not None:
                raise RuntimeError(
                    "Local vLLM process exited before API readiness: "
                    f"pid={self._process.pid} returncode={returncode} "
                    f"log_path={log_path}\n{_tail_log(log_path)}"
                )

        result = wait_for_openai_readiness(
            _api_base_for_config(self.config),
            model=self.config.served_model_name,
            api_key=self.config.api_key,
            config=ReadinessConfig(
                smoke_test=self.config.readiness.smoke_test,
                prompt=self.config.readiness.prompt,
                max_tokens=self.config.readiness.max_tokens,
                timeout_seconds=self.config.readiness.timeout_seconds,
                retry_interval_seconds=self.config.check_interval_seconds,
            ),
            timeout_seconds=self.config.ready_timeout_seconds,
            should_continue=ensure_process_alive,
        )
        if not result.ok:
            raise TimeoutError(
                "Timed out waiting for local vLLM API readiness: "
                f"api_base={_api_base_for_config(self.config)} log_path={log_path}\n"
                f"{_tail_log(log_path)}"
            )
        return result

    def _write_summary(
        self,
        lifecycle_state: str,
        *,
        readiness: ReadinessResult | None = None,
        capacity=None,
        session: InferenceSession | None = None,
        failure_code: str | None = None,
        failure_message: str | None = None,
    ) -> None:
        pid = session.pid if session is not None else None
        if pid is None and self._process is not None:
            pid = self._process.pid
        payload = endpoint_summary(
            endpoint_name=self.config.name,
            backend_kind="local_vllm",
            lifecycle_state=lifecycle_state,
            run_id=self._run_id,
            api_base=session.api_base if session is not None else _api_base_for_config(self.config),
            served_model_name=self.config.served_model_name,
            model=self.config.model,
            api_key_set=bool(self.config.api_key),
            local_port=self.config.port,
            pid=pid,
            cleanup_command=(
                session.cleanup_command
                if session is not None
                else f"kill -TERM -{pid}"
                if pid is not None
                else ""
            ),
            local_log_dir=str(self._out_dir) if self._out_dir is not None else None,
            readiness=readiness,
            capacity=capacity,
            failure_code=failure_code,
            failure_message=failure_message,
        )
        self._last_summary = payload
        self._summary.write(payload)

    def _write_failure_summary(self, error: BaseException) -> None:
        message = str(error)
        log_text = _tail_log(self._log_path) if self._log_path is not None else ""
        failure_code = _classify_failure_if_enabled(
            self.config,
            f"{message}\n{log_text}",
            fallback="local_process_exited",
        )
        self._write_summary(
            "FAILED",
            capacity=_parse_capacity_if_enabled(
                self.config,
                log_text,
                max_model_len=self.config.max_model_len,
                max_num_seqs=self.config.max_num_seqs,
            ),
            failure_code=failure_code,
            failure_message=sanitized_excerpt(message),
        )


def with_local_vllm_defaults(
    config: LocalVllmConfig,
    *,
    port: int | None = None,
) -> LocalVllmConfig:
    """Fill derived local vLLM defaults."""

    resolved_port = port if port is not None else config.port
    served_model_name = config.served_model_name or config.model
    return replace(
        config,
        served_model_name=served_model_name,
        port=resolved_port,
    )


def validate_local_vllm_config(config: LocalVllmConfig) -> LocalVllmConfig:
    """Reject configs that cannot produce a valid local server session."""

    for field_name in ("name", "model", "served_model_name", "host", "python_bin"):
        _require_non_empty_string(config, field_name)
    if any(character.isspace() for character in config.host):
        raise ValueError("Local vLLM host must not contain whitespace.")
    if config.port is not None:
        _require_port(config.port, "port")
    for field_name in ("check_interval_seconds", "ready_timeout_seconds"):
        _require_positive_int(config, field_name)
    for field_name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
    ):
        _require_optional_positive_int(config, field_name)
    if config.gpu_memory_utilization is not None and (
        isinstance(config.gpu_memory_utilization, bool)
        or config.gpu_memory_utilization <= 0
        or config.gpu_memory_utilization > 1
    ):
        raise ValueError("Local vLLM gpu_memory_utilization must be in (0, 1].")
    try:
        validate_readiness_config(config.readiness)
    except ValueError as error:
        raise ValueError(f"Local vLLM {error}") from error
    _require_string_sequence(config.extra_args, "extra_args")
    return config


def vllm_server_command(config: LocalVllmConfig) -> list[str]:
    """Return the vLLM serve command."""

    command = [
        config.python_bin,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        config.model,
        "--trust-remote-code",
        "--served-model-name",
        config.served_model_name,
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--host",
        config.host,
    ]
    if config.port is not None:
        command.extend(["--port", str(config.port)])
    if config.api_key:
        command.extend(["--api-key", config.api_key])
    command.extend(_vllm_serve_options(config))
    command.extend(effective_vllm_extra_args(config))
    return command


def effective_vllm_extra_args(config: LocalVllmConfig) -> tuple[str, ...]:
    """Return extra args after applying target-specific safety additions."""

    args = list(config.extra_args)
    if config.target_device == "rocm" and config.rocm_eager_fallback:
        if "--enforce-eager" not in args:
            args.append("--enforce-eager")
        if not _has_any_option(args, {"--compilation-config", "-cc"}):
            args.extend(["--compilation-config", '{"mode":0,"backend":"eager"}'])
    return tuple(args)


def local_vllm_env(
    config: LocalVllmConfig,
    *,
    runtime_tmp_root: Path,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an isolated environment for the vLLM process."""

    env = dict(os.environ if base_env is None else base_env)
    tmp_dir = runtime_tmp_root / "ipc"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    torchinductor_cache = runtime_tmp_root / "torchinductor"
    triton_cache = runtime_tmp_root / "triton"
    torchinductor_cache.mkdir(parents=True, exist_ok=True)
    triton_cache.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(torchinductor_cache)
    env["TRITON_CACHE_DIR"] = str(triton_cache)
    if config.hf_home:
        hf_home = Path(config.hf_home).expanduser()
        env["HF_HOME"] = str(hf_home)
        env["HF_HUB_CACHE"] = str(hf_home / "hub")
        env["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    _prepend_python_package_library_paths(env, config.python_bin)
    if config.target_device:
        env["VLLM_TARGET_DEVICE"] = config.target_device
    if config.target_device == "rocm":
        env.setdefault("ROCM_PATH", "/opt/rocm")
        env.setdefault("ROCM_HOME", env["ROCM_PATH"])
    return env


def _vllm_serve_options(config: LocalVllmConfig) -> list[str]:
    options: list[str] = []
    for name, value in (
        ("--tensor-parallel-size", config.tensor_parallel_size),
        ("--pipeline-parallel-size", config.pipeline_parallel_size),
        ("--data-parallel-size", config.data_parallel_size),
        ("--gpu-memory-utilization", config.gpu_memory_utilization),
        ("--max-model-len", config.max_model_len),
        ("--max-num-seqs", config.max_num_seqs),
        ("--max-num-batched-tokens", config.max_num_batched_tokens),
    ):
        if value is not None:
            options.extend([name, str(value)])
    return options


def _require_vllm_available(python_bin: str) -> None:
    try:
        completed = subprocess.run(
            [
                python_bin,
                "-c",
                "import importlib.util; raise SystemExit(importlib.util.find_spec('vllm') is None)",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(f"Local vLLM python is not executable: {python_bin!r}.") from error
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        suffix = f"\n{detail}" if detail else ""
        raise RuntimeError(f"vLLM is not installed for local server python {python_bin!r}.{suffix}")


def _resolved_out_dir(config: LocalVllmConfig, *, run_id: str) -> Path:
    if config.out_dir:
        return Path(config.out_dir).expanduser().resolve()
    return (
        Path.home()
        / DEFAULT_OUT_DIR_ROOT
        / model_label(config.model)
        / f"{safe_endpoint_label(config.name)}-{run_id}"
    )


def _resolved_runtime_tmp_root(
    config: LocalVllmConfig,
    out_dir: Path,
    *,
    run_id: str,
) -> Path:
    if config.runtime_tmp_root:
        return Path(config.runtime_tmp_root).expanduser().resolve()
    safe_model = model_label(config.model)
    port = config.port if config.port is not None else "auto"
    return Path(tempfile.gettempdir()) / f"ril-vllm-{safe_model}-{port}-{run_id}"


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError):
            process.kill()
        process.wait(timeout=30)


def _tail_log(path: Path, *, lines: int = 80) -> str:
    if not path.exists():
        return "Local vLLM log file does not exist yet."
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _parse_capacity_if_enabled(
    config: LocalVllmConfig,
    log_text: str,
    *,
    max_model_len: int | None,
    max_num_seqs: int | None,
) -> CapacityInfo | None:
    if not config.diagnostics.collect_vllm_capacity:
        return None
    return parse_vllm_capacity(
        log_text,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )


def _classify_failure_if_enabled(
    config: LocalVllmConfig,
    text: str,
    *,
    fallback: str,
) -> str:
    if not config.diagnostics.classify_startup_failures:
        return fallback
    return classify_failure(text, fallback=fallback)


def _requested_gpus(config: LocalVllmConfig) -> int:
    if config.target_device == "cpu":
        return 0
    return config.tensor_parallel_size or 1


def _api_host_for_bind_host(bind_host: str) -> str:
    if bind_host in {"*", "0.0.0.0"}:
        return "127.0.0.1"
    return bind_host


def _api_base_for_config(config: LocalVllmConfig) -> str:
    if config.port is None:
        return ""
    return f"http://{_api_host_for_bind_host(config.host)}:{config.port}/v1"


def _timed_out(started_at: float, timeout_seconds: int) -> bool:
    return timeout_seconds > 0 and time.monotonic() - started_at >= timeout_seconds


def _has_any_option(args: Sequence[str], option_names: set[str]) -> bool:
    for arg in args:
        if arg in option_names:
            return True
        if any(arg.startswith(f"{name}=") for name in option_names):
            return True
    return False


def _prepend_python_package_library_paths(env: dict[str, str], python_bin: str) -> None:
    library_paths = _python_package_library_paths(python_bin)
    if not library_paths:
        return
    existing = [path for path in env.get("LD_LIBRARY_PATH", "").split(":") if path]
    env["LD_LIBRARY_PATH"] = ":".join([*library_paths, *existing])


def _python_package_library_paths(python_bin: str) -> list[str]:
    python_path = Path(python_bin).expanduser()
    if not python_path.is_absolute():
        python_path = Path.cwd() / python_path
    venv_path = python_path.parent.parent
    site_packages_roots = sorted((venv_path / "lib").glob("python*/site-packages"))
    paths: list[str] = []
    for site_packages in site_packages_roots:
        for library_path in sorted((site_packages / "nvidia").glob("*/lib")):
            if library_path.is_dir():
                paths.append(str(library_path))
        torch_library_path = site_packages / "torch" / "lib"
        if torch_library_path.is_dir():
            paths.append(str(torch_library_path))
    return paths


def _require_non_empty_string(config: LocalVllmConfig, field_name: str) -> None:
    value = getattr(config, field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Local vLLM {field_name} must be a non-empty string.")


def _require_port(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError(f"Local vLLM {field_name} must be between 1 and 65535.")


def _require_positive_int(config: LocalVllmConfig, field_name: str) -> None:
    value = getattr(config, field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Local vLLM {field_name} must be positive.")


def _require_optional_positive_int(config: LocalVllmConfig, field_name: str) -> None:
    value = getattr(config, field_name)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Local vLLM {field_name} must be positive.")


def _require_string_sequence(values: Sequence[str], field_name: str) -> None:
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"Local vLLM {field_name} must contain only non-empty strings.")
