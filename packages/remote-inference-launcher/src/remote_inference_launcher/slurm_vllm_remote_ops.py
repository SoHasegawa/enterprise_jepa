"""Remote SSH and Slurm operations for Slurm vLLM launches."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path

from remote_inference_launcher.diagnostics import sanitized_excerpt
from remote_inference_launcher.ports import PortReservation, reserve_local_port
from remote_inference_launcher.process import process_error
from remote_inference_launcher.readiness import (
    ReadinessConfig,
    ReadinessResult,
    wait_for_openai_readiness,
)
from remote_inference_launcher.slurm import (
    first_scontrol_hostname,
    new_slurm_submission_comment,
    parse_squeue_job_info,
    scancel_job_command,
    scontrol_hostnames_command,
    scontrol_show_job_command,
    squeue_job_ids_by_comment_command,
    squeue_job_info_command,
    squeue_start_command,
)
from remote_inference_launcher.slurm_vllm_config import (
    SlurmJobInfo,
    SlurmPendingTimeoutError,
    SlurmVllmConfig,
)
from remote_inference_launcher.ssh import (
    SSH_TRANSPORT_FAILURE_ATTEMPTS,
    NonRetryableTransientSshError,
    is_transient_ssh_failure,
    ssh_options,
)

REMOTE_CANCEL_ATTEMPTS = 3
TUNNEL_START_ATTEMPTS = 3


class SlurmVllmRemoteOpsMixin:
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
            raise RuntimeError("Remote Slurm submission did not return a job id.")
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

    def _wait_for_running_node(self) -> str:
        started_at = time.monotonic()
        last_status_at = 0.0
        last_state = ""
        last_reason = ""
        while True:
            info = self._get_remote_job_info(self._job_id)
            self._latest_slurm_state = info.state
            self._latest_slurm_reason = info.reason
            if info.state == "RUNNING" and info.node and info.node != "(null)":
                self._pending_duration_seconds = time.monotonic() - started_at
                return info.node
            if info.state in _TERMINAL_STATES:
                self._pending_duration_seconds = time.monotonic() - started_at
                raise RuntimeError(
                    "Remote vLLM job stopped before readiness: "
                    f"job_id={self._job_id} state={info.state} reason={info.reason} "
                    f"remote_logs={self._remote_out_dir}"
                )
            if _timed_out(started_at, _pending_timeout_seconds(self.config)):
                self._pending_duration_seconds = time.monotonic() - started_at
                raise SlurmPendingTimeoutError(
                    f"Timed out waiting for remote vLLM job {self._job_id}; "
                    f"remote_logs={self._remote_out_dir}"
                )
            now = time.monotonic()
            state_changed = (info.state, info.reason) != (last_state, last_reason)
            heartbeat_due = now - last_status_at >= self.config.queue_policy.status_interval_seconds
            if state_changed or heartbeat_due:
                if info.state == "PENDING":
                    self._latest_slurm_diagnostics = self._collect_slurm_pending_diagnostics()
                    self._write_summary("PENDING")
                self._log_progress(
                    f"Waiting for remote vLLM job {self._job_id} "
                    f"(state={info.state}, reason={info.reason})..."
                )
                last_state = info.state
                last_reason = info.reason
                last_status_at = now
            time.sleep(_queue_poll_interval(self.config))

    def _wait_for_remote_state_port(self) -> int:
        started_at = time.monotonic()
        while True:
            state = self._read_remote_state()
            remote_port = state.get("remote_port")
            if isinstance(remote_port, int) and 1 <= remote_port <= 65535:
                return remote_port
            info = self._get_remote_job_info(self._job_id)
            if info.state in _TERMINAL_STATES:
                raise RuntimeError(
                    "Remote vLLM job stopped before writing remote port state: "
                    f"job_id={self._job_id} state={info.state} reason={info.reason} "
                    f"remote_state_path={self._remote_state_path} "
                    f"remote_logs={self._remote_out_dir}"
                )
            if _timed_out(started_at, self.config.ready_timeout_seconds):
                raise TimeoutError(
                    "Timed out waiting for remote vLLM port state: "
                    f"job_id={self._job_id} remote_state_path={self._remote_state_path} "
                    f"remote_logs={self._remote_out_dir}"
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

    def _start_tunnel_with_reservation(
        self,
        reservation: PortReservation,
        remote_port: int,
    ) -> int:
        attempts = TUNNEL_START_ATTEMPTS if self._generated["local_port"] else 1
        current = reservation
        try:
            for attempt in range(1, attempts + 1):
                try:
                    self._start_tunnel(current.port, remote_port, reservation=current)
                    self._selected_local_port = current.port
                    return current.port
                except RuntimeError:
                    if attempt >= attempts:
                        self._refresh_job_info_after_tunnel_failure()
                        raise
                    current.close()
                    current = reserve_local_port(self.config.local_bind_host, None)
        finally:
            current.close()
        raise RuntimeError("SSH tunnel did not start.")

    def _refresh_job_info_after_tunnel_failure(self) -> None:
        """Best-effort scheduler refresh before cleanup after final tunnel failure."""

        if not self._job_id:
            return
        try:
            info = self._get_remote_job_info(self._job_id)
        except RuntimeError as error:
            self._latest_slurm_diagnostics = sanitized_excerpt(
                f"Failed to refresh Slurm job state after tunnel failure: {error}",
                max_chars=2000,
            )
            return
        self._latest_slurm_state = info.state
        self._latest_slurm_reason = info.reason

    def _start_tunnel(
        self,
        local_port: int,
        remote_port: int,
        *,
        reservation: PortReservation | None = None,
    ) -> None:
        command = [
            "ssh",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ExitOnForwardFailure=yes",
            "-N",
            "-L",
            f"{self.config.local_bind_host}:{local_port}:{self._remote_node}:{remote_port}",
            self.config.ssh_target,
        ]
        self._log_verbose(
            "Starting SSH tunnel: "
            f"{self.config.local_bind_host}:{local_port} -> "
            f"{self._remote_node}:{remote_port} "
            f"via {self.config.ssh_target}"
        )
        if reservation is not None:
            reservation.close()
        self._tunnel = subprocess.Popen(command, stdin=subprocess.DEVNULL)
        time.sleep(1)
        if self._tunnel.poll() is not None:
            raise RuntimeError("SSH tunnel exited before vLLM readiness check.")

    def _wait_for_ready(self, api_base: str) -> ReadinessResult:
        def ensure_slurm_and_tunnel_alive() -> None:
            info = self._get_remote_job_info(self._job_id)
            if info.state not in {"RUNNING", "CONFIGURING", "PENDING"}:
                raise RuntimeError(
                    "Remote vLLM job stopped while waiting for API readiness: "
                    f"job_id={self._job_id} state={info.state} reason={info.reason} "
                    f"remote_logs={self._remote_out_dir}"
                )
            if self._tunnel is None or self._tunnel.poll() is not None:
                raise RuntimeError(
                    "SSH tunnel stopped while waiting for vLLM readiness: "
                    f"job_id={self._job_id} remote_logs={self._remote_out_dir}"
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
            should_continue=ensure_slurm_and_tunnel_alive,
        )
        if not result.ok:
            raise TimeoutError(
                "Timed out waiting for remote vLLM API readiness: "
                f"job_id={self._job_id} api_base={api_base} "
                f"remote_logs={self._remote_out_dir}"
            )
        self._log_progress(f"Remote vLLM is ready at {api_base}.")
        return result

    def _get_remote_job_info(self, job_id: str) -> SlurmJobInfo:
        parsed = parse_squeue_job_info(self._ssh_text(squeue_job_info_command(job_id)))
        if parsed is None:
            return SlurmJobInfo(state="NOT_FOUND")

        state, node, reason = parsed
        if node and node != "(null)" and "[" in node:
            node = first_scontrol_hostname(
                self._ssh_text(scontrol_hostnames_command(node)),
                node_expression=node,
            )
        return SlurmJobInfo(state=state, node=node, reason=reason)

    def _collect_slurm_pending_diagnostics(self) -> str:
        if not self._job_id:
            return ""
        excerpts: list[str] = []
        for label, command in (
            ("scontrol_show_job", scontrol_show_job_command(self._job_id)),
            ("squeue_start", squeue_start_command(self._job_id)),
        ):
            try:
                output = self._ssh_text(f"{command} 2>/dev/null || true").strip()
            except Exception as error:
                output = f"unavailable: {error}"
            if output:
                excerpts.append(f"{label}:\n{output}")
        return sanitized_excerpt("\n\n".join(excerpts), max_chars=2000)

    def _cancel_remote_job(self, job_id: str) -> None:
        cancel_error: RuntimeError | None = None
        info = SlurmJobInfo(state="UNKNOWN")
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

    def _remote_log_tail(self, *, lines: int = 80) -> str:
        if not self._remote_out_dir:
            return ""
        output = self._ssh_text(
            f"tail -n {lines} {shlex.quote(self._remote_out_dir.rstrip('/') + '/vllm.log')} "
            "2>/dev/null || true"
        )
        return output.strip()

    def _safe_remote_log_tail(self, *, lines: int = 80) -> str:
        try:
            return self._remote_log_tail(lines=lines)
        except Exception:
            return ""

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
            self._sleep_before_ssh_retry(attempt, label="remote copy")
        if completed.returncode:
            raise RuntimeError(process_error("remote copy", completed))

    def _ssh_text(self, command: str, *, command_is_retry_safe: bool = True) -> str:
        completed = self._ssh_completed(command, command_is_retry_safe=command_is_retry_safe)
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
            completed = subprocess.run(
                ssh_command,
                capture_output=True,
                check=False,
                text=True,
            )
            if not _should_retry_ssh(completed, attempt, max_attempts=attempts):
                break
            self._sleep_before_ssh_retry(attempt, label=command)
        return completed

    def _sleep_before_ssh_retry(self, attempt: int, *, label: str) -> None:
        delay_seconds = _ssh_retry_delay_seconds(attempt)
        self._log_verbose(
            f"Retrying transient SSH failure in {delay_seconds}s "
            f"(attempt {attempt + 1}/{SSH_TRANSPORT_FAILURE_ATTEMPTS}): {label}"
        )
        time.sleep(delay_seconds)


def _timed_out(started_at: float, timeout_seconds: int) -> bool:
    return timeout_seconds > 0 and time.monotonic() - started_at >= timeout_seconds


def _queue_poll_interval(config: SlurmVllmConfig) -> int:
    return max(config.queue_policy.poll_interval_seconds, 1)


def _pending_timeout_seconds(config: SlurmVllmConfig) -> int:
    return config.queue_policy.max_pending_seconds or config.queue_timeout_seconds


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


_TERMINAL_STATES = {
    "NOT_FOUND",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "NODE_FAIL",
    "DEADLINE",
}
