"""SSH-backed vLLM launcher for hosts without Slurm."""

from __future__ import annotations

import json
import math
import shlex
import subprocess
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from string import Template

from remote_inference_launcher.config_types import DiagnosticsConfig
from remote_inference_launcher.core import InferenceSession
from remote_inference_launcher.diagnostics import (
    CapacityInfo,
    classify_failure,
    parse_vllm_capacity,
    sanitized_excerpt,
)
from remote_inference_launcher.plan_paths import join_remote_path, model_label, safe_endpoint_label
from remote_inference_launcher.ports import PortReservation, reserve_local_port
from remote_inference_launcher.process import exception_summary, process_error
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
from remote_inference_launcher.ssh import (
    SSH_TRANSPORT_FAILURE_ATTEMPTS,
    is_transient_ssh_failure,
    ssh_options,
)
from remote_inference_launcher.summaries import SummaryWriter, endpoint_summary, new_run_id
from remote_inference_launcher.verbosity import (
    progress_enabled,
    validate_verbosity,
    verbose_enabled,
)

DEFAULT_REMOTE_OUT_DIR_ROOT = "tmp/remote-inference-launcher/ssh-vllm"
REMOTE_STATE_FILENAME = "remote-inference-state.json"
TUNNEL_START_ATTEMPTS = 3


@dataclass(frozen=True)
class SshVllmConfig:
    """Configuration for one SSH-backed vLLM server session."""

    name: str = "default"
    ssh_target: str = ""
    verbosity: str = "progress"
    model: str = ""
    served_model_name: str = ""
    remote_port: int | None = None
    local_port: int | None = None
    local_bind_host: str = "127.0.0.1"
    setup_cmd: str = ""
    python_bin: str = "python"
    out_dir: str = ""
    remote_out_dir_root: str = ""
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
    api_key: str = ""
    extra_args: tuple[str, ...] = ()
    check_interval_seconds: int = 10
    ready_timeout_seconds: int = 7200
    keep_remote_process: bool = False
    readiness: ReadinessConfig = field(default_factory=ReadinessConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    launch_summary_path: str = ""
    overwrite_launch_summary: bool = False


class SshVllmLauncher:
    """Start vLLM on a remote SSH host, tunnel it locally, and wait for readiness."""

    owns_resources = True

    def __init__(self, config: SshVllmConfig, *, run_id: str | None = None) -> None:
        self.config = validate_ssh_vllm_config(with_ssh_vllm_defaults(config))
        self._remote_home = ""
        self._remote_out_dir = ""
        self._remote_state_path = ""
        self._remote_pid = ""
        self._selected_local_port: int | None = None
        self._tunnel: subprocess.Popen[bytes] | None = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._run_id = run_id or new_run_id()
        self._summary = SummaryWriter(
            self.config.launch_summary_path,
            run_id=self._run_id,
            endpoint_name=self.config.name,
            backend_kind="ssh_vllm",
            overwrite=self.config.overwrite_launch_summary,
        )
        self._last_summary: dict[str, object] | None = None
        self._budget_token: ResourceBudgetReservation | None = None

    def start(self) -> InferenceSession:
        """Start the remote server and return a ready inference session."""

        self._summary.reserve()
        self._budget_token = acquire_current_resource_budget(
            candidate_attempts=1,
            requested_gpus=_requested_gpus(self.config),
        )
        self._temp_dir = tempfile.TemporaryDirectory(prefix="remote-inference-launcher-ssh-vllm.")
        try:
            local_reservation = reserve_local_port(
                self.config.local_bind_host,
                self.config.local_port,
            )
            self._remote_home = self._ssh_text("cd && pwd").strip()
            self._remote_out_dir = _expand_remote_home(
                self.config.out_dir
                or _default_remote_out_dir(
                    self._remote_home,
                    self.config,
                    run_id=self._run_id,
                ),
                self._remote_home,
            )
            self._remote_state_path = f"{self._remote_out_dir.rstrip('/')}/{REMOTE_STATE_FILENAME}"
            remote_script = f"{self._remote_out_dir.rstrip('/')}/launch_vllm.sh"
            local_script = Path(self._temp_dir.name) / "launch_vllm.sh"
            local_script.write_text(
                render_ssh_vllm_script(
                    self.config,
                    out_dir=self._remote_out_dir,
                    state_path=self._remote_state_path,
                ),
                encoding="utf-8",
            )
            self._ssh_text(f"mkdir -p {shlex.quote(self._remote_out_dir)}")
            self._ssh_text(f"rm -f {shlex.quote(self._remote_state_path)}")
            self._copy_to_remote(local_script, remote_script)
            self._ssh_text(f"chmod +x {shlex.quote(remote_script)}")
            self._remote_pid = self._start_remote_process(remote_script)
            self._write_summary("REMOTE_PROCESS_STARTED")
            self.config = replace(self.config, remote_port=self._wait_for_remote_state_port())
            self._write_summary("REMOTE_PORT_READY")
            selected_local_port = self._start_tunnel_with_reservation(local_reservation)
            api_base = _api_base_for_bind_host(self.config.local_bind_host, selected_local_port)
            self._write_summary("TUNNEL_READY")
            self._log_progress(
                f"Started remote vLLM process {self._remote_pid} on {self.config.ssh_target}."
            )
            self._log_progress(f"Local API base: {api_base}")
            self._log_progress(f"Remote logs: {self._remote_out_dir}")
            readiness = self._wait_for_ready(api_base) or ReadinessResult(
                models_endpoint_ok=True,
                smoke_test_ok=True,
                smoke_test_kind=self.config.readiness.smoke_test,
            )
            capacity = _parse_capacity_if_enabled(
                self.config,
                self._safe_remote_log_tail(lines=400),
                max_model_len=self.config.max_model_len,
                max_num_seqs=self.config.max_num_seqs,
            )
            session = InferenceSession(
                name=self.config.name,
                api_base=api_base,
                served_model_name=self.config.served_model_name,
                model=self.config.model,
                api_key=self.config.api_key,
                local_port=selected_local_port,
                remote_port=self.config.remote_port,
                logs=self._remote_out_dir,
                pid=int(self._remote_pid),
                node=self.config.ssh_target,
                cleanup_command=_remote_kill_command(self.config.ssh_target, self._remote_pid),
                backend_kind="ssh_vllm",
                run_id=self._run_id,
                summary_path=str(self._summary.path),
                metadata={
                    "remote_state_path": self._remote_state_path,
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
            summary_error: Exception | None = None
            try:
                self._write_failure_summary(error)
            except Exception as failure_summary_error:
                summary_error = failure_summary_error
            try:
                self.stop()
            except Exception as stop_error:
                summary_suffix = (
                    f" Failure summary also failed: {summary_error}."
                    if summary_error is not None
                    else ""
                )
                raise RuntimeError(
                    "SSH vLLM launcher failed and failed to stop cleanly: "
                    f"{stop_error}. Original failure: {exception_summary(error)}."
                    f"{summary_suffix}"
                ) from error
            raise

    def stop(self) -> None:
        """Stop the SSH tunnel and remote process unless configured otherwise."""

        if self._tunnel is not None:
            self._tunnel.terminate()
            try:
                self._tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._tunnel.kill()
                self._tunnel.wait(timeout=10)
            self._tunnel = None
        if self._remote_pid and not self.config.keep_remote_process:
            self._stop_remote_process(self._remote_pid)
            self._remote_pid = ""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
        if (
            self._last_summary is not None
            and self._last_summary.get("lifecycle_state") != "FAILED"
            and not self.config.keep_remote_process
        ):
            released = dict(self._last_summary)
            released["lifecycle_state"] = "RELEASED"
            self._summary.write(released)
        if self._budget_token is not None:
            self._budget_token.release()
            self._budget_token = None

    def reserve_summary(self):
        """Reserve the launch summary path before starting remote work."""

        return self._summary.reserve()

    @contextmanager
    def running(self) -> Iterator[InferenceSession]:
        """Context manager that stops the launcher on exit."""

        session = self.start()
        try:
            yield session
        finally:
            self.stop()

    def _start_remote_process(self, remote_script: str) -> str:
        launch_command = " ".join(
            [
                "{",
                "setsid",
                "bash",
                shlex.quote(remote_script),
                ">/dev/null",
                "2>&1",
                "</dev/null",
                "&",
                "echo",
                "$!",
                ";",
                "}",
            ]
        )
        output = self._ssh_text(
            " ".join(
                [
                    "cd",
                    shlex.quote(self._remote_out_dir),
                    "&&",
                    launch_command,
                ]
            ),
            command_is_retry_safe=False,
        ).strip()
        pid = output.splitlines()[-1].strip() if output else ""
        if not pid.isdigit():
            raise RuntimeError(f"Remote SSH vLLM process did not return a pid: {output!r}")
        return pid

    def _wait_for_remote_state_port(self) -> int:
        started_at = time.monotonic()
        while True:
            state = self._read_remote_state()
            remote_port = state.get("remote_port")
            if state:
                if type(remote_port) is int and 1 <= remote_port <= 65535:
                    if not self._remote_process_is_alive(self._remote_pid):
                        raise RuntimeError(
                            "Remote SSH vLLM process exited after writing remote port state: "
                            f"pid={self._remote_pid} remote_port={remote_port} "
                            f"remote_state_path={self._remote_state_path} "
                            f"remote_logs={self._remote_out_dir}\n{self._remote_log_tail()}"
                        )
                    return remote_port
                raise RuntimeError(
                    "Remote SSH vLLM state file did not contain a valid remote_port: "
                    f"remote_state_path={self._remote_state_path} state={state!r}"
                )
            if not self._remote_process_is_alive(self._remote_pid):
                raise RuntimeError(
                    "Remote SSH vLLM process exited before writing remote port state: "
                    f"pid={self._remote_pid} remote_state_path={self._remote_state_path} "
                    f"remote_logs={self._remote_out_dir}\n{self._remote_log_tail()}"
                )
            if _timed_out(started_at, self.config.ready_timeout_seconds):
                raise TimeoutError(
                    "Timed out waiting for remote SSH vLLM port state: "
                    f"pid={self._remote_pid} remote_state_path={self._remote_state_path} "
                    f"remote_logs={self._remote_out_dir}\n{self._remote_log_tail()}"
                )
            time.sleep(self.config.check_interval_seconds)

    def _read_remote_state(self) -> dict[str, object]:
        command = f"cat {shlex.quote(self._remote_state_path)} 2>/dev/null || true"
        output = self._ssh_text(command)
        if not output.strip():
            return {}
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Remote state file is not valid JSON: {self._remote_state_path}: {error}"
            ) from error
        if not isinstance(parsed, dict):
            raise RuntimeError(
                f"Remote state file must contain a JSON object: {self._remote_state_path}"
            )
        return parsed

    def _start_tunnel_with_reservation(self, reservation: PortReservation) -> int:
        attempts = TUNNEL_START_ATTEMPTS if self.config.local_port is None else 1
        current = reservation
        try:
            for attempt in range(1, attempts + 1):
                try:
                    current.close()
                    self._start_tunnel(current.port)
                    self._selected_local_port = current.port
                    return current.port
                except RuntimeError:
                    if attempt >= attempts:
                        raise
                    current = reserve_local_port(self.config.local_bind_host, None)
        finally:
            current.close()
        raise RuntimeError("SSH tunnel did not start.")

    def _start_tunnel(self, local_port: int) -> None:
        command = [
            "ssh",
            *ssh_options(),
            "-o",
            "ExitOnForwardFailure=yes",
            "-N",
            "-L",
            f"{self.config.local_bind_host}:{local_port}:127.0.0.1:{self.config.remote_port}",
            self.config.ssh_target,
        ]
        self._log_verbose(
            "Starting SSH tunnel: "
            f"{self.config.local_bind_host}:{local_port} -> "
            f"127.0.0.1:{self.config.remote_port} via {self.config.ssh_target}"
        )
        self._tunnel = subprocess.Popen(command, stdin=subprocess.DEVNULL)
        time.sleep(1)
        if self._tunnel.poll() is not None:
            raise RuntimeError("SSH tunnel exited before vLLM readiness check.")

    def _wait_for_ready(self, api_base: str) -> ReadinessResult:
        def ensure_remote_alive() -> None:
            if not self._remote_process_is_alive(self._remote_pid):
                raise RuntimeError(
                    "Remote SSH vLLM process exited before API readiness: "
                    f"pid={self._remote_pid} remote_logs={self._remote_out_dir}\n"
                    f"{self._remote_log_tail()}"
                )
            if self._tunnel is None or self._tunnel.poll() is not None:
                raise RuntimeError(
                    "SSH tunnel stopped while waiting for vLLM readiness: "
                    f"pid={self._remote_pid} remote_logs={self._remote_out_dir}"
                )

        result = wait_for_openai_readiness(
            api_base,
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
            should_continue=ensure_remote_alive,
        )
        if not result.ok:
            raise TimeoutError(
                "Timed out waiting for remote SSH vLLM API readiness: "
                f"pid={self._remote_pid} api_base={api_base} "
                f"remote_logs={self._remote_out_dir}\n{self._remote_log_tail()}"
            )
        self._log_progress(f"Remote SSH vLLM is ready at {api_base}.")
        return result

    def _remote_process_is_alive(self, pid: str) -> bool:
        completed = self._ssh_completed(f"kill -0 {shlex.quote(pid)}")
        return completed.returncode == 0

    def _remote_log_tail(self, *, lines: int = 80) -> str:
        completed = self._ssh_completed(
            f"tail -n {lines} {shlex.quote(self._remote_out_dir.rstrip('/') + '/vllm.log')} "
            "2>/dev/null"
        )
        return completed.stdout.strip() or "(remote vLLM log is empty or missing)"

    def _safe_remote_log_tail(self, *, lines: int = 80) -> str:
        try:
            return self._remote_log_tail(lines=lines)
        except Exception:
            return ""

    def _stop_remote_process(self, pid: str) -> None:
        command = (
            f"kill -TERM -{shlex.quote(pid)} 2>/dev/null || "
            f"kill -TERM {shlex.quote(pid)} 2>/dev/null || true; "
            "sleep 2; "
            f"kill -KILL -{shlex.quote(pid)} 2>/dev/null || "
            f"kill -KILL {shlex.quote(pid)} 2>/dev/null || true"
        )
        self._ssh_text(command)

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
            time.sleep(_ssh_retry_delay_seconds(attempt))
        if completed.returncode:
            raise RuntimeError(process_error("remote copy", completed))

    def _ssh_text(self, command: str, *, command_is_retry_safe: bool = True) -> str:
        completed = self._ssh_completed(command, command_is_retry_safe=command_is_retry_safe)
        if completed.returncode:
            raise RuntimeError(process_error(command, completed))
        return completed.stdout

    def _ssh_completed(
        self,
        command: str,
        *,
        command_is_retry_safe: bool = True,
    ) -> subprocess.CompletedProcess[str]:
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
            time.sleep(_ssh_retry_delay_seconds(attempt))
        return completed

    def _log_progress(self, message: str) -> None:
        if progress_enabled(self.config.verbosity):
            print(message)

    def _log_verbose(self, message: str) -> None:
        if verbose_enabled(self.config.verbosity):
            print(message)

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
        if pid is None and self._remote_pid:
            pid = int(self._remote_pid)
        payload = endpoint_summary(
            endpoint_name=self.config.name,
            backend_kind="ssh_vllm",
            lifecycle_state=lifecycle_state,
            run_id=self._run_id,
            api_base=session.api_base if session is not None else "",
            served_model_name=self.config.served_model_name,
            model=self.config.model,
            api_key_set=bool(self.config.api_key),
            local_port=session.local_port if session is not None else self._selected_local_port,
            remote_port=self.config.remote_port,
            pid=pid,
            node=self.config.ssh_target,
            cleanup_command=(
                session.cleanup_command
                if session is not None
                else _remote_kill_command(self.config.ssh_target, self._remote_pid)
                if self._remote_pid
                else ""
            ),
            remote_log_dir=self._remote_out_dir or None,
            remote_state_path=self._remote_state_path or None,
            readiness=readiness,
            capacity=capacity,
            failure_code=failure_code,
            failure_message=failure_message,
        )
        self._last_summary = payload
        self._summary.write(payload)

    def _write_failure_summary(self, error: BaseException) -> None:
        message = str(error)
        log_text = self._safe_remote_log_tail() if self._remote_out_dir else ""
        failure_code = _classify_failure_if_enabled(
            self.config,
            f"{message}\n{log_text}",
            fallback="remote_process_exited",
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


def with_ssh_vllm_defaults(config: SshVllmConfig) -> SshVllmConfig:
    """Fill derived defaults for SSH vLLM configs."""

    if not config.ssh_target:
        raise ValueError("SSH vLLM config is missing: ssh_target.")
    if not config.model:
        raise ValueError("SSH vLLM config is missing: model.")
    served_model_name = config.served_model_name or config.model
    return SshVllmConfig(**{**config.__dict__, "served_model_name": served_model_name})


def validate_ssh_vllm_config(config: SshVllmConfig) -> SshVllmConfig:
    """Reject configs that cannot produce a remote SSH vLLM server."""

    validate_verbosity(config.verbosity, label="SSH vLLM")
    for field_name in ("name", "ssh_target", "model", "served_model_name", "python_bin"):
        _require_non_empty_string(config, field_name)
    _require_supported_bind_host(config.local_bind_host)
    if config.remote_port is not None:
        _require_port(config.remote_port, "remote_port")
    if config.local_port is not None:
        _require_port(config.local_port, "local_port")
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
    if config.gpu_memory_utilization is not None and _invalid_gpu_memory_utilization(
        config.gpu_memory_utilization
    ):
        raise ValueError("SSH vLLM gpu_memory_utilization must be in (0, 1].")
    try:
        validate_readiness_config(config.readiness)
    except ValueError as error:
        raise ValueError(f"SSH vLLM {error}") from error
    for item in config.extra_args:
        if not isinstance(item, str) or not item:
            raise ValueError("SSH vLLM extra_args must contain only non-empty strings.")
    return config


def render_ssh_vllm_script(
    config: SshVllmConfig,
    *,
    out_dir: str,
    state_path: str | None = None,
) -> str:
    """Render the remote wrapper script used to start vLLM over SSH."""

    config = validate_ssh_vllm_config(with_ssh_vllm_defaults(config))
    setup_cmd_block = f"\n{config.setup_cmd}\n" if config.setup_cmd else ""
    api_key_line = f"  --api-key {shlex.quote(config.api_key)} \\\n" if config.api_key else ""
    template = _AtTemplate(_template_text("launch_vllm_ssh.sh"))
    return template.substitute(
        out_dir=shlex.quote(out_dir),
        model=shlex.quote(config.model),
        served_model_name=shlex.quote(config.served_model_name),
        remote_port="" if config.remote_port is None else str(config.remote_port),
        state_path=shlex.quote(state_path or f"{out_dir.rstrip('/')}/{REMOTE_STATE_FILENAME}"),
        python_bin=shlex.quote(config.python_bin),
        target_device=shlex.quote(config.target_device),
        runtime_tmp_root=shlex.quote(config.runtime_tmp_root),
        hf_home=shlex.quote(config.hf_home),
        extra_args=_bash_array(effective_vllm_extra_args(config)),
        vllm_options=_bash_array(_vllm_serve_options(config)),
        setup_cmd_block=setup_cmd_block,
        api_key_line=api_key_line,
    )


def effective_vllm_extra_args(config: SshVllmConfig) -> tuple[str, ...]:
    """Return extra args after applying target-specific safety additions."""

    args = list(config.extra_args)
    if config.target_device == "rocm" and config.rocm_eager_fallback:
        if "--enforce-eager" not in args:
            args.append("--enforce-eager")
        if not _has_any_option(args, {"--compilation-config", "-cc"}):
            args.extend(["--compilation-config", '{"mode":0,"backend":"eager"}'])
    return tuple(args)


def _vllm_serve_options(config: SshVllmConfig) -> tuple[str, ...]:
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
    return tuple(options)


def _default_remote_out_dir(
    remote_home: str,
    config: SshVllmConfig,
    *,
    run_id: str,
) -> str:
    root = config.remote_out_dir_root or f"{remote_home.rstrip('/')}/{DEFAULT_REMOTE_OUT_DIR_ROOT}"
    return join_remote_path(
        root,
        model_label(config.model),
        f"{safe_endpoint_label(config.name)}-{run_id}",
    )


def _api_base_for_bind_host(local_bind_host: str, local_port: int) -> str:
    return f"http://{_api_host_for_bind_host(local_bind_host)}:{local_port}/v1"


def _api_host_for_bind_host(local_bind_host: str) -> str:
    if local_bind_host in {"*", "0.0.0.0"}:
        return "127.0.0.1"
    return local_bind_host


def _expand_remote_home(path: str, remote_home: str) -> str:
    if path == "~":
        return remote_home
    if path.startswith("~/"):
        return f"{remote_home.rstrip('/')}/{path[2:]}"
    return path


def _remote_kill_command(ssh_target: str, pid: str) -> str:
    return f"ssh {shlex.quote(ssh_target)} 'kill -TERM -{shlex.quote(pid)}'"


def _bash_array(values: Sequence[str]) -> str:
    return "(" + " ".join(shlex.quote(value) for value in values) + ")"


def _has_any_option(args: Sequence[str], option_names: set[str]) -> bool:
    for arg in args:
        if arg in option_names:
            return True
        if any(arg.startswith(f"{name}=") for name in option_names):
            return True
    return False


def _parse_capacity_if_enabled(
    config: SshVllmConfig,
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
    config: SshVllmConfig,
    text: str,
    *,
    fallback: str,
) -> str:
    if not config.diagnostics.classify_startup_failures:
        return fallback
    return classify_failure(text, fallback=fallback)


def _requested_gpus(config: SshVllmConfig) -> int:
    if config.target_device == "cpu":
        return 0
    return config.tensor_parallel_size or 1


def _template_text(filename: str) -> str:
    return (
        resources.files("remote_inference_launcher.templates")
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )


class _AtTemplate(Template):
    delimiter = "@"


def _timed_out(started_at: float, timeout_seconds: int) -> bool:
    return timeout_seconds > 0 and time.monotonic() - started_at >= timeout_seconds


def _should_retry_ssh(
    completed: subprocess.CompletedProcess,
    attempt: int,
    *,
    max_attempts: int = SSH_TRANSPORT_FAILURE_ATTEMPTS,
) -> bool:
    return attempt < max_attempts and is_transient_ssh_failure(completed)


def _ssh_retry_delay_seconds(attempt: int) -> int:
    return min(2 * attempt, 10)


def _require_non_empty_string(config: SshVllmConfig, field_name: str) -> None:
    value = getattr(config, field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SSH vLLM {field_name} must be a non-empty string.")


def _require_supported_bind_host(local_bind_host: str) -> None:
    if any(character.isspace() for character in local_bind_host):
        raise ValueError("SSH vLLM local_bind_host must not contain whitespace.")
    if ":" in local_bind_host:
        raise ValueError("SSH vLLM local_bind_host must be an IPv4 address, hostname, or '*'.")


def _require_port(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError(f"SSH vLLM {field_name} must be between 1 and 65535.")


def _require_positive_int(config: SshVllmConfig, field_name: str) -> None:
    value = getattr(config, field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"SSH vLLM {field_name} must be positive.")


def _require_optional_positive_int(config: SshVllmConfig, field_name: str) -> None:
    value = getattr(config, field_name)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"SSH vLLM {field_name} must be positive.")


def _invalid_gpu_memory_utilization(value: object) -> bool:
    return (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or value <= 0
        or value > 1
    )
