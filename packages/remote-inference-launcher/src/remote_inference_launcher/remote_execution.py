"""Shared SSH command execution helpers."""

from __future__ import annotations

import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from random import uniform

from remote_inference_launcher.ssh import ssh_options

CommandRunner = Callable[[list[str], int], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class RemoteCommandPolicy:
    """Retry policy for remote commands."""

    attempts: int = 3
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 4.0
    jitter_seconds: float = 0.25
    per_attempt_timeout_seconds: int = 20


@dataclass(frozen=True)
class RemoteCommand:
    """One logical command in a remote batch."""

    name: str
    command: str


@dataclass(frozen=True)
class RemoteResult:
    """One logical remote command result."""

    name: str
    returncode: int
    stdout: str
    stderr: str
    attempts: int
    duration_seconds: float
    command_summary: str

    def to_completed_process(self) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [self.name],
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


@dataclass(frozen=True)
class _RemoteBatchResult:
    returncode: int
    stdout: str
    stderr: str


_REMOTE_BATCH_BEGIN = "__RIL_PREFLIGHT_CHECK_BEGIN__"
_REMOTE_BATCH_STDERR = "__RIL_PREFLIGHT_STDERR__"
_REMOTE_BATCH_STDOUT = "__RIL_PREFLIGHT_STDOUT__"
_REMOTE_BATCH_END = "__RIL_PREFLIGHT_CHECK_END__"


def run_ssh_batch(
    *,
    ssh_target: str,
    commands: tuple[RemoteCommand, ...],
    command_runner: CommandRunner,
    policy: RemoteCommandPolicy,
) -> tuple[RemoteResult, ...]:
    """Run logical commands in one SSH session and return per-command results."""

    if not commands:
        return ()
    start = time.monotonic()
    ssh_command = [
        "ssh",
        *ssh_options(),
        "-T",
        ssh_target,
        f"bash -lc {shlex.quote(_remote_batch_script(commands))}",
    ]
    attempts = max(policy.attempts, 1)
    last_completed: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, attempts + 1):
        completed = command_runner(ssh_command, _remote_batch_timeout(policy, len(commands)))
        last_completed = completed
        parsed_results = _parse_remote_batch_results(completed.stdout, expected_count=len(commands))
        if parsed_results:
            results = _remote_results_from_batch_results(
                ssh_target=ssh_target,
                commands=commands,
                parsed_results=parsed_results,
                attempts=attempt,
                duration_seconds=time.monotonic() - start,
            )
        else:
            results = _remote_results_from_batch_completion(
                ssh_target=ssh_target,
                commands=commands,
                completed=completed,
                attempts=attempt,
                duration_seconds=time.monotonic() - start,
            )
        if not _has_transient_transport_failure(results) or attempt == attempts:
            return results
        _sleep_before_retry(policy, attempt)
    if last_completed is None:
        raise RuntimeError("SSH remote batch retry loop did not run.")
    return _remote_results_from_batch_completion(
        ssh_target=ssh_target,
        commands=commands,
        completed=last_completed,
        attempts=attempts,
        duration_seconds=time.monotonic() - start,
    )


def run_ssh_command(
    *,
    ssh_target: str,
    name: str,
    command: str,
    command_runner: CommandRunner,
    policy: RemoteCommandPolicy,
    retry_safe: bool = True,
) -> RemoteResult:
    """Run one remote SSH command with shared retry and redaction behavior."""

    start = time.monotonic()
    ssh_command = [
        "ssh",
        *ssh_options(),
        "-T",
        ssh_target,
        f"bash -lc {shlex.quote(command)}",
    ]
    attempts = max(policy.attempts, 1) if retry_safe else 1
    last_completed: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, attempts + 1):
        completed = command_runner(ssh_command, policy.per_attempt_timeout_seconds)
        last_completed = completed
        if not retry_safe or not transient_transport_code(completed) or attempt == attempts:
            return RemoteResult(
                name=name,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                attempts=attempt,
                duration_seconds=time.monotonic() - start,
                command_summary=command_summary(ssh_command, ssh_target=ssh_target),
            )
        _sleep_before_retry(policy, attempt)
    if last_completed is None:
        raise RuntimeError("SSH remote command retry loop did not run.")
    return RemoteResult(
        name=name,
        returncode=last_completed.returncode,
        stdout=last_completed.stdout,
        stderr=last_completed.stderr,
        attempts=attempts,
        duration_seconds=time.monotonic() - start,
        command_summary=command_summary(ssh_command, ssh_target=ssh_target),
    )


def transient_transport_code(completed: subprocess.CompletedProcess[str]) -> str:
    """Return a stable SSH transport failure code when a result is retryable."""

    detail = f"{completed.stderr}\n{completed.stdout}".lower()
    if completed.returncode == 124 or "timed out" in detail or "timeout" in detail:
        return "remote_command_timeout" if completed.returncode == 124 else "ssh_timeout"
    if completed.returncode != 255:
        return ""
    if "connection reset" in detail:
        return "ssh_connection_reset"
    if "connection closed" in detail:
        return "ssh_connection_closed"
    if "broken pipe" in detail:
        return "ssh_broken_pipe"
    if "kex_exchange_identification" in detail or "kex" in detail:
        return "ssh_kex_failed"
    if "control socket" in detail or "mux_client" in detail:
        return "ssh_control_socket_failed"
    if "timed out" in detail or "timeout" in detail:
        return "ssh_timeout"
    return ""


def command_summary(command: object, *, ssh_target: str = "") -> str:
    if not isinstance(command, list):
        return redact(str(command), sensitive_values=(ssh_target,))
    parts = [str(part) for part in command]
    if parts and parts[0] == "ssh":
        redacted_parts = ["<ssh-target>" if part == ssh_target else part for part in parts]
        return redact(" ".join(redacted_parts), sensitive_values=(ssh_target,))
    return redact(" ".join(parts), sensitive_values=(ssh_target,))


def redact(value: str, *, sensitive_values: tuple[str, ...] = ()) -> str:
    redacted = value
    for marker in ("Bearer ", "OPENAI_API_KEY=", "api_key=", "API_KEY="):
        if marker in redacted:
            before, _sep, _after = redacted.partition(marker)
            redacted = before + marker + "<redacted>"
    for sensitive_value in sensitive_values:
        if sensitive_value:
            redacted = redacted.replace(sensitive_value, "<redacted>")
    return redacted


def _remote_batch_script(commands: tuple[RemoteCommand, ...]) -> str:
    lines = [
        "set +e",
        'tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ril-preflight.XXXXXX") || exit 125',
        'cleanup() { rm -rf "$tmpdir"; }',
        "trap cleanup EXIT HUP INT TERM",
        "run_ril_check() {",
        "  idx=$1",
        "  command=$2",
        '  stdout_file="$tmpdir/stdout-$idx"',
        '  stderr_file="$tmpdir/stderr-$idx"',
        '  bash -lc "$command" >"$stdout_file" 2>"$stderr_file"',
        "  rc=$?",
        f"  printf '%s\\n' {shlex.quote(_REMOTE_BATCH_BEGIN)}",
        '  printf "index=%s\\n" "$idx"',
        '  printf "returncode=%s\\n" "$rc"',
        f"  printf '%s\\n' {shlex.quote(_REMOTE_BATCH_STDERR)}",
        '  sed -n "1,80p" "$stderr_file"',
        f"  printf '%s\\n' {shlex.quote(_REMOTE_BATCH_STDOUT)}",
        '  sed -n "1,80p" "$stdout_file"',
        f"  printf '%s\\n' {shlex.quote(_REMOTE_BATCH_END)}",
        "}",
    ]
    for index, command in enumerate(commands):
        lines.append(f"run_ril_check {index} {shlex.quote(command.command)}")
    lines.append("exit 0")
    return "\n".join(lines)


def _remote_batch_timeout(policy: RemoteCommandPolicy, check_count: int) -> int:
    return max(
        policy.per_attempt_timeout_seconds,
        policy.per_attempt_timeout_seconds * max(check_count, 1),
    )


def _parse_remote_batch_results(
    output: str,
    *,
    expected_count: int,
) -> dict[int, _RemoteBatchResult]:
    results: dict[int, _RemoteBatchResult] = {}
    for section in _remote_batch_sections(output):
        parsed = _parse_remote_batch_section(section, expected_count=expected_count)
        if parsed is None:
            continue
        index, result = parsed
        results[index] = result
    return results


def _remote_batch_sections(output: str) -> tuple[tuple[str, ...], ...]:
    sections: list[tuple[str, ...]] = []
    current: list[str] = []
    in_section = False
    for line in output.splitlines():
        if line == _REMOTE_BATCH_BEGIN:
            current = []
            in_section = True
            continue
        if not in_section:
            continue
        if line == _REMOTE_BATCH_END:
            sections.append(tuple(current))
            current = []
            in_section = False
            continue
        current.append(line)
    return tuple(sections)


def _parse_remote_batch_section(
    lines: tuple[str, ...],
    *,
    expected_count: int,
) -> tuple[int, _RemoteBatchResult] | None:
    if len(lines) < 4:
        return None
    index = _parse_header_int(lines[0], "index=")
    returncode = _parse_header_int(lines[1], "returncode=")
    if index is None or returncode is None or index < 0 or index >= expected_count:
        return None
    try:
        stderr_marker_index = lines.index(_REMOTE_BATCH_STDERR, 2)
        stdout_marker_index = lines.index(_REMOTE_BATCH_STDOUT, stderr_marker_index + 1)
    except ValueError:
        return None
    return index, _RemoteBatchResult(
        returncode=returncode,
        stdout="\n".join(lines[stdout_marker_index + 1 :]),
        stderr="\n".join(lines[stderr_marker_index + 1 : stdout_marker_index]),
    )


def _parse_header_int(line: str, prefix: str) -> int | None:
    if not line.startswith(prefix):
        return None
    return _parse_int(line.removeprefix(prefix))


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _remote_results_from_batch_results(
    *,
    ssh_target: str,
    commands: tuple[RemoteCommand, ...],
    parsed_results: dict[int, _RemoteBatchResult],
    attempts: int,
    duration_seconds: float,
) -> tuple[RemoteResult, ...]:
    results: list[RemoteResult] = []
    summary = _remote_batch_command_summary(ssh_target, len(commands))
    for index, command in enumerate(commands):
        result = parsed_results.get(index)
        if result is None:
            results.append(
                RemoteResult(
                    name=command.name,
                    returncode=1,
                    stdout="",
                    stderr="Remote batch did not report this command result.",
                    attempts=attempts,
                    duration_seconds=duration_seconds,
                    command_summary=summary,
                )
            )
            continue
        results.append(
            RemoteResult(
                name=command.name,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                attempts=attempts,
                duration_seconds=duration_seconds,
                command_summary=summary,
            )
        )
    return tuple(results)


def _remote_results_from_batch_completion(
    *,
    ssh_target: str,
    commands: tuple[RemoteCommand, ...],
    completed: subprocess.CompletedProcess[str],
    attempts: int,
    duration_seconds: float,
) -> tuple[RemoteResult, ...]:
    summary = _remote_batch_command_summary(ssh_target, len(commands))
    return tuple(
        RemoteResult(
            name=command.name,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            attempts=attempts,
            duration_seconds=duration_seconds,
            command_summary=summary,
        )
        for command in commands
    )


def _remote_batch_command_summary(ssh_target: str, command_count: int) -> str:
    return command_summary(
        [
            "ssh",
            *ssh_options(),
            "-T",
            ssh_target,
            f"bash -lc <remote batch: {command_count} commands>",
        ],
        ssh_target=ssh_target,
    )


def _has_transient_transport_failure(results: tuple[RemoteResult, ...]) -> bool:
    return any(transient_transport_code(result.to_completed_process()) for result in results)


def _sleep_before_retry(policy: RemoteCommandPolicy, completed_attempts: int) -> None:
    backoff = min(
        policy.max_backoff_seconds,
        policy.initial_backoff_seconds * (2 ** max(completed_attempts - 1, 0)),
    )
    if policy.jitter_seconds > 0:
        backoff += uniform(0, policy.jitter_seconds)
    if backoff > 0:
        time.sleep(backoff)
