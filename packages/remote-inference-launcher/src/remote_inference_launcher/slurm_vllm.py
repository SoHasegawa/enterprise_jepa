"""Slurm-backed vLLM launcher."""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from remote_inference_launcher.core import InferenceSession
from remote_inference_launcher.diagnostics import sanitized_excerpt
from remote_inference_launcher.ports import reserve_local_port
from remote_inference_launcher.process import exception_summary
from remote_inference_launcher.readiness import (
    ReadinessResult,
)
from remote_inference_launcher.resource_budget import (
    ResourceBudgetReservation,
    acquire_current_resource_budget,
)
from remote_inference_launcher.slurm_vllm_config import (
    DEFAULT_REMOTE_OUT_DIR_ROOT,
    REMOTE_STATE_FILENAME,
    SlurmJobInfo,
    SlurmPendingTimeoutError,
    SlurmVllmConfig,
    _api_base_for_bind_host,
    _default_remote_out_dir,
    _expand_remote_home,
    effective_vllm_extra_args,
    render_slurm_vllm_sbatch,
    validate_slurm_vllm_config,
    with_slurm_vllm_defaults,
)
from remote_inference_launcher.slurm_vllm_remote_ops import (
    SlurmVllmRemoteOpsMixin,
    _pending_timeout_seconds,
)
from remote_inference_launcher.slurm_vllm_resource_preferences import (
    SlurmVllmResourcePreferencesMixin,
    _failure_code_for_exception,
    _parse_capacity_if_enabled,
    _remote_scancel_command,
    _resource_attempt_summary,
    _resource_preference_candidates,
    slurm_resource_preference_candidates,
)
from remote_inference_launcher.summaries import SummaryWriter, endpoint_summary, new_run_id
from remote_inference_launcher.verbosity import (
    progress_enabled,
    verbose_enabled,
)

__all__ = [
    "DEFAULT_REMOTE_OUT_DIR_ROOT",
    "REMOTE_STATE_FILENAME",
    "SlurmJobInfo",
    "SlurmPendingTimeoutError",
    "SlurmVllmConfig",
    "SlurmVllmLauncher",
    "_pending_timeout_seconds",
    "_resource_attempt_summary",
    "_resource_preference_candidates",
    "effective_vllm_extra_args",
    "render_slurm_vllm_sbatch",
    "slurm_resource_preference_candidates",
    "validate_slurm_vllm_config",
    "with_slurm_vllm_defaults",
]


class SlurmVllmLauncher(SlurmVllmResourcePreferencesMixin, SlurmVllmRemoteOpsMixin):
    """Submit a Slurm vLLM job, open an SSH tunnel, and wait for readiness."""

    owns_resources = True

    def __init__(
        self,
        config: SlurmVllmConfig,
        *,
        run_id: str | None = None,
        _candidate_mode: bool = False,
    ) -> None:
        self._input_config = config
        self._run_id = run_id or new_run_id()
        self._generated = {
            "job_name": not bool(config.job_name),
            "out_dir": not bool(config.out_dir),
            "local_port": config.local_port is None,
            "remote_port": config.remote_port is None,
        }
        self._candidate_mode = _candidate_mode
        self.config = validate_slurm_vllm_config(
            with_slurm_vllm_defaults(config, run_id=self._run_id)
        )
        self._remote_home = ""
        self._remote_out_dir = ""
        self._remote_state_path = ""
        self._job_id = ""
        self._remote_node = ""
        self._selected_remote_port: int | None = None
        self._selected_local_port: int | None = None
        self._latest_slurm_state = ""
        self._latest_slurm_reason = ""
        self._latest_slurm_diagnostics = ""
        self._pending_duration_seconds: float | None = None
        self._tunnel: subprocess.Popen[bytes] | None = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._delegate_launcher: SlurmVllmLauncher | None = None
        self._candidate_launchers: dict[str, SlurmVllmLauncher] = {}
        self._winner_candidate_name = ""
        self._summary = SummaryWriter(
            self.config.launch_summary_path,
            run_id=self._run_id,
            endpoint_name=self.config.name,
            backend_kind="slurm_vllm",
            overwrite=self.config.overwrite_launch_summary,
        )
        self._last_summary: dict[str, object] | None = None
        self._budget_token: ResourceBudgetReservation | None = None

    def start(self) -> InferenceSession:
        """Start the remote vLLM server and return details once it is ready."""

        if self.config.resource_preferences and not self._candidate_mode:
            return self._start_resource_preferences()
        self._summary.reserve()
        self._budget_token = acquire_current_resource_budget(
            candidate_attempts=1,
            submitted_slurm_jobs=1,
            requested_gpus=self.config.num_gpus or 0,
        )
        self._temp_dir = tempfile.TemporaryDirectory(prefix="remote-inference-launcher-slurm-vllm.")
        try:
            self._prepare_remote_paths()
            self._job_id = self._submit_launch_script()
            self._write_summary("SUBMITTED")
            self._log_progress(f"Submitted remote vLLM job {self._job_id}.")
            self._log_progress(f"Remote logs: {self._remote_out_dir}")
            self._remote_node = self._wait_for_running_node()
            self._write_summary("ALLOCATED")
            self._selected_remote_port = self._wait_for_remote_state_port()
            self._write_summary("REMOTE_PORT_READY")
            reservation = reserve_local_port(self.config.local_bind_host, self.config.local_port)
            selected_local_port = self._start_tunnel_with_reservation(
                reservation,
                self._selected_remote_port,
            )
            self._write_summary("TUNNEL_READY")
            api_base = _api_base_for_bind_host(self.config.local_bind_host, selected_local_port)
            self._log_progress(f"Local API base: {api_base}")
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
                remote_port=self._selected_remote_port,
                logs=self._remote_out_dir,
                job_id=self._job_id,
                node=self._remote_node,
                cleanup_command=_remote_scancel_command(self.config.ssh_target, self._job_id),
                backend_kind="slurm_vllm",
                run_id=self._run_id,
                summary_path=str(self._summary.path),
                metadata={
                    "generated": dict(self._generated),
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
            self._stop_after_start_failure(error)
            raise

    def stop(self) -> None:
        """Stop the SSH tunnel and cancel the Slurm job unless configured otherwise."""

        stop_error: RuntimeError | None = None
        if self._candidate_launchers:
            stop_error = self._stop_candidate_launchers()
        else:
            self._stop_tunnel()
            stop_error = self._cancel_active_job()
            self._cleanup_temp_dir()
            self._write_released_summary()
            self._release_budget_token()
        if stop_error is not None:
            raise stop_error

    def _stop_candidate_launchers(self) -> RuntimeError | None:
        stop_error: RuntimeError | None = None
        for name, launcher in reversed(list(self._candidate_launchers.items())):
            try:
                launcher.stop()
            except RuntimeError as error:
                stop_error = error
            if name == self._winner_candidate_name:
                self._delegate_launcher = None
        self._candidate_launchers.clear()
        self._winner_candidate_name = ""
        self._job_id = ""
        self._write_released_summary(respect_keep_remote_job=False)
        return stop_error

    def _stop_tunnel(self) -> None:
        if self._tunnel is not None:
            self._tunnel.terminate()
            try:
                self._tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._tunnel.kill()
                self._tunnel.wait(timeout=10)
            self._tunnel = None

    def _cancel_active_job(self) -> RuntimeError | None:
        if self._job_id and not self.config.keep_remote_job:
            try:
                self._cancel_remote_job(self._job_id)
            except RuntimeError as error:
                return error
            else:
                self._job_id = ""
        return None

    def _cleanup_temp_dir(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def _write_released_summary(self, *, respect_keep_remote_job: bool = True) -> None:
        if (
            self._last_summary is not None
            and self._last_summary.get("lifecycle_state") != "FAILED"
            and (not respect_keep_remote_job or not self.config.keep_remote_job)
        ):
            released = dict(self._last_summary)
            released["lifecycle_state"] = "RELEASED"
            self._summary.write(released)

    def _release_budget_token(self) -> None:
        if self._budget_token is not None:
            self._budget_token.release()
            self._budget_token = None

    def reserve_summary(self):
        """Reserve the launch summary path before submitting remote work."""

        return self._summary.reserve()

    @contextmanager
    def running(self) -> Iterator[InferenceSession]:
        """Context manager that stops the launcher on exit."""

        session = self.start()
        try:
            yield session
        finally:
            self.stop()

    def job_info(self) -> SlurmJobInfo:
        """Return the current scheduler state for the managed job."""

        if not self._job_id:
            return SlurmJobInfo(state="NOT_SUBMITTED")
        return self._get_remote_job_info(self._job_id)

    def _prepare_remote_paths(self) -> None:
        self._remote_home = self._ssh_text("cd && pwd").strip()
        self._write_summary(
            "SUBMITTING",
            extra={
                "registry_event": "remote_home_resolved",
                "registry_event_payload": {"remote_home_resolved": True},
            },
        )
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
        self._write_summary(
            "SUBMITTING",
            extra={
                "registry_event": "remote_output_directory_created",
                "registry_event_payload": {
                    "remote_log_dir": self._remote_out_dir,
                    "remote_state_path": self._remote_state_path,
                },
            },
        )

    def _submit_launch_script(self) -> str:
        if self._temp_dir is None:
            raise RuntimeError("Temporary launch directory is not initialized.")
        remote_script = f"{self._remote_out_dir.rstrip('/')}/launch_vllm.sbatch"
        local_script = Path(self._temp_dir.name) / "launch_vllm.sbatch"
        local_script.write_text(
            render_slurm_vllm_sbatch(
                self.config,
                out_dir=self._remote_out_dir,
                state_path=self._remote_state_path,
            ),
            encoding="utf-8",
        )
        self._ssh_text(f"mkdir -p {shlex.quote(self._remote_out_dir)}")
        self._copy_to_remote(local_script, remote_script)
        self._write_summary(
            "SUBMITTING",
            extra={
                "registry_event": "slurm_script_uploaded",
                "registry_event_payload": {"remote_script": remote_script},
            },
        )
        return self._submit_remote_job(remote_script)

    def _stop_after_start_failure(self, error: BaseException) -> None:
        try:
            self._write_failure_summary(error)
            self.stop()
        except Exception as stop_error:
            raise RuntimeError(
                "Slurm vLLM launcher failed and failed to stop cleanly: "
                f"{stop_error}. Original failure: {exception_summary(error)}"
            ) from error

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
        extra: Mapping[str, object] | None = None,
    ) -> None:
        extra_payload: dict[str, object] = {
            "generated": dict(self._generated),
            "queue_policy": self.config.queue_policy.__dict__,
            "candidate_race": self.config.candidate_race.__dict__,
        }
        if extra:
            extra_payload.update(dict(extra))
        payload = endpoint_summary(
            endpoint_name=self.config.name,
            backend_kind="slurm_vllm",
            lifecycle_state=lifecycle_state,
            run_id=self._run_id,
            api_base=session.api_base if session is not None else "",
            served_model_name=self.config.served_model_name,
            model=self.config.model,
            api_key_set=bool(self.config.api_key),
            local_port=session.local_port if session is not None else self._selected_local_port,
            remote_port=session.remote_port if session is not None else self._selected_remote_port,
            job_id=self._job_id,
            job_name=self.config.job_name,
            node=self._remote_node,
            cleanup_command=(
                session.cleanup_command
                if session is not None
                else _remote_scancel_command(self.config.ssh_target, self._job_id)
                if self._job_id
                else ""
            ),
            remote_log_dir=self._remote_out_dir or None,
            remote_state_path=self._remote_state_path or None,
            readiness=readiness,
            capacity=capacity,
            failure_code=failure_code,
            failure_message=failure_message,
            resource_attempts=[
                {
                    "candidate_index": 0,
                    "candidate_name": self.config.name,
                    "backend_kind": "slurm_vllm",
                    "ssh_target": self.config.ssh_target,
                    "partition": self.config.partition,
                    "job_id": self._job_id or None,
                    "final_state": lifecycle_state,
                    "failure_code": failure_code,
                    "winner": lifecycle_state == "READY",
                    "pending_duration_seconds": self._pending_duration_seconds,
                    "latest_slurm_state": self._latest_slurm_state or None,
                    "latest_slurm_reason": self._latest_slurm_reason or None,
                    "latest_slurm_diagnostics": self._latest_slurm_diagnostics or None,
                    "cleanup_command": _remote_scancel_command(
                        self.config.ssh_target,
                        self._job_id,
                    )
                    if self._job_id
                    else None,
                }
            ],
            extra=extra_payload,
        )
        self._last_summary = payload
        self._summary.write(payload)

    def _write_failure_summary(self, error: BaseException) -> None:
        message = str(error)
        log_text = ""
        try:
            log_text = self._remote_log_tail()
        except Exception:
            log_text = ""
        failure_code = _failure_code_for_exception(
            self.config,
            error,
            text=f"{message}\n{log_text}",
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

    def _mark_last_summary_failed(self, message: str) -> None:
        payload = dict(self._last_summary or {})
        if not payload:
            return
        payload["lifecycle_state"] = "FAILED"
        diagnostics = dict(payload.get("diagnostics") or {})
        diagnostics["failure_code"] = "cleanup_failed"
        diagnostics["failure_message"] = sanitized_excerpt(message)
        payload["diagnostics"] = diagnostics
        self._last_summary = payload
        self._summary.write(payload)
