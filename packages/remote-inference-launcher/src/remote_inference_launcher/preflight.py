"""Advisory resource-target preflight checks."""

from __future__ import annotations

import shlex
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from remote_inference_launcher.endpoint_race import EndpointRaceConfig
from remote_inference_launcher.existing_endpoint import ExistingEndpointConfig
from remote_inference_launcher.fleet import FleetConfig
from remote_inference_launcher.local_vllm import LocalVllmConfig
from remote_inference_launcher.remote_execution import (
    CommandRunner,
    RemoteCommand,
    RemoteCommandPolicy,
    run_ssh_batch,
    transient_transport_code,
)
from remote_inference_launcher.remote_execution import (
    command_summary as _command_summary,
)
from remote_inference_launcher.remote_execution import (
    redact as _redact,
)
from remote_inference_launcher.slurm_vllm import (
    SlurmVllmConfig,
    slurm_resource_preference_candidates,
)
from remote_inference_launcher.ssh_vllm import SshVllmConfig

PreflightOutcome = Literal[
    "ok",
    "skipped",
    "durable_failure",
    "transient_failure",
    "unknown",
]


@dataclass(frozen=True)
class PreflightCheck:
    """One preflight check result."""

    source: str
    endpoint_name: str
    name: str
    outcome: PreflightOutcome
    code: str
    attempts: int
    duration_seconds: float
    detail: str = ""
    command_summary: str = ""
    layer: str = ""

    @property
    def status(self) -> str:
        """Compatibility status for internal callers that have not switched yet."""

        if self.outcome in {"ok", "skipped"}:
            return self.outcome
        return "failed"

    @property
    def ok(self) -> bool:
        return self.outcome in {"ok", "skipped"}

    @property
    def resolved_layer(self) -> str:
        return self.layer or _preflight_check_layer(self.name, self.code)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "endpoint_name": self.endpoint_name,
            "name": self.name,
            "layer": self.resolved_layer,
            "outcome": self.outcome,
            "code": self.code,
            "attempts": self.attempts,
            "duration_seconds": self.duration_seconds,
            "detail": self.detail,
            "command_summary": self.command_summary,
        }


_SLURM_SCHEDULER_CHECKS = {
    "slurm-sbatch",
    "slurm-squeue",
    "slurm-scontrol",
    "slurm-sinfo",
    "slurm-sbatch-test-only",
    "slurm-squeue-start",
    "slurm-partition",
    "slurm-partition-state",
    "slurm-gres-gpu",
    "slurm-partition-walltime",
}


def _preflight_check_layer(name: str, code: str) -> str:
    if name in _SLURM_SCHEDULER_CHECKS:
        return "slurm-scheduler"
    if name == "vllm-import" or code == "requires_compute_allocation":
        return "compute-runtime"
    if name == "existing-endpoint":
        return "local"
    return "remote-control"


@dataclass(frozen=True)
class _RemoteCheckSpec:
    name: str
    command: str
    durable_code: str = ""


@dataclass(frozen=True)
class _RemotePreflightUnit:
    config: SshVllmConfig | SlurmVllmConfig
    source: str
    endpoint_name: str
    slurm: bool


def run_preflight_checks(
    config: object,
    *,
    source: str,
    timeout_seconds: int = 20,
    command_runner: CommandRunner | None = None,
    remote_command_policy: RemoteCommandPolicy | None = None,
) -> tuple[PreflightCheck, ...]:
    """Run cheap non-allocating checks for an inference config."""

    return run_preflight_checks_for_configs(
        ((source, config),),
        timeout_seconds=timeout_seconds,
        command_runner=command_runner,
        remote_command_policy=remote_command_policy,
    )


def run_preflight_checks_for_configs(
    config_sources: Iterable[tuple[str, object]],
    *,
    timeout_seconds: int = 20,
    command_runner: CommandRunner | None = None,
    remote_command_policy: RemoteCommandPolicy | None = None,
) -> tuple[PreflightCheck, ...]:
    """Run cheap checks for multiple configs, batching SSH work by target."""

    runner = command_runner or _run_command
    policy = remote_command_policy or RemoteCommandPolicy(
        per_attempt_timeout_seconds=timeout_seconds
    )
    checks: list[PreflightCheck] = []
    remote_units: list[_RemotePreflightUnit] = []
    for source, config in config_sources:
        _append_checks(
            checks,
            remote_units,
            config,
            source=source,
            endpoint_name=str(getattr(config, "name", "default") or "default"),
            timeout_seconds=timeout_seconds,
            command_runner=runner,
        )
    checks.extend(
        _grouped_remote_vllm_preflight_checks(
            remote_units,
            command_runner=runner,
            remote_command_policy=policy,
        )
    )
    return tuple(checks)


def _append_checks(
    checks: list[PreflightCheck],
    remote_units: list[_RemotePreflightUnit],
    config: object,
    *,
    source: str,
    endpoint_name: str,
    timeout_seconds: int,
    command_runner: CommandRunner,
) -> None:
    if isinstance(config, FleetConfig):
        for name, endpoint in sorted(config.endpoints.items()):
            _append_checks(
                checks,
                remote_units,
                endpoint,
                source=f"{source} endpoint {name!r}",
                endpoint_name=str(getattr(endpoint, "name", name) or name),
                timeout_seconds=timeout_seconds,
                command_runner=command_runner,
            )
        return
    if isinstance(config, EndpointRaceConfig):
        for name, candidate in sorted(config.candidates.items()):
            _append_checks(
                checks,
                remote_units,
                candidate,
                source=f"{source} candidate {name!r}",
                endpoint_name=str(getattr(candidate, "name", name) or name),
                timeout_seconds=timeout_seconds,
                command_runner=command_runner,
            )
        return
    if isinstance(config, SlurmVllmConfig):
        for candidate_source, candidate in _slurm_preflight_candidates(config, source=source):
            remote_units.append(
                _RemotePreflightUnit(
                    config=candidate,
                    source=candidate_source,
                    endpoint_name=candidate.name,
                    slurm=True,
                )
            )
        return
    if isinstance(config, SshVllmConfig):
        remote_units.append(
            _RemotePreflightUnit(
                config=config,
                source=source,
                endpoint_name=endpoint_name,
                slurm=False,
            )
        )
        return
    if isinstance(config, LocalVllmConfig):
        checks.extend(
            _local_vllm_preflight_checks(
                config,
                source=source,
                endpoint_name=endpoint_name,
                timeout_seconds=timeout_seconds,
                command_runner=command_runner,
            )
        )
        return
    if isinstance(config, ExistingEndpointConfig):
        checks.append(
            PreflightCheck(
                source=source,
                endpoint_name=endpoint_name,
                name="existing-endpoint",
                outcome="skipped",
                code="not_applicable",
                attempts=0,
                duration_seconds=0.0,
                detail="No resource-target preflight checks apply to existing endpoints.",
                layer="local",
            )
        )


def _local_vllm_preflight_checks(
    config: LocalVllmConfig,
    *,
    source: str,
    endpoint_name: str,
    timeout_seconds: int,
    command_runner: CommandRunner,
) -> tuple[PreflightCheck, ...]:
    checks = [
        _shell_check(
            source=source,
            endpoint_name=endpoint_name,
            name="python-version",
            command=_python_command(config.python_bin, "--version"),
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
            durable_code="remote_python_missing",
            layer="local",
        ),
        _shell_check(
            source=source,
            endpoint_name=endpoint_name,
            name="vllm-import",
            command=_python_command(config.python_bin, "-c", "import vllm"),
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
            durable_code="vllm_import_failed",
            layer="compute-runtime",
        ),
    ]
    checks.extend(
        _local_path_checks(
            config,
            source=source,
            endpoint_name=endpoint_name,
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
        )
    )
    return tuple(checks)


def _grouped_remote_vllm_preflight_checks(
    remote_units: list[_RemotePreflightUnit],
    *,
    command_runner: CommandRunner,
    remote_command_policy: RemoteCommandPolicy,
) -> tuple[PreflightCheck, ...]:
    groups: dict[str, list[tuple[_RemotePreflightUnit, _RemoteCheckSpec]]] = {}
    skipped_checks: list[PreflightCheck] = []
    for unit in remote_units:
        remote_specs, skipped = _remote_vllm_preflight_specs(unit)
        skipped_checks.extend(skipped)
        groups.setdefault(unit.config.ssh_target, []).extend((unit, spec) for spec in remote_specs)
    checks: list[PreflightCheck] = []
    for ssh_target, items in groups.items():
        checks.extend(
            _ssh_batch_checks(
                ssh_target=ssh_target,
                items=tuple(items),
                command_runner=command_runner,
                policy=remote_command_policy,
            )
        )
    checks.extend(skipped_checks)
    return tuple(checks)


def _remote_vllm_preflight_specs(
    unit: _RemotePreflightUnit,
) -> tuple[tuple[_RemoteCheckSpec, ...], tuple[PreflightCheck, ...]]:
    config = unit.config
    remote_specs: list[_RemoteCheckSpec] = [
        _RemoteCheckSpec(
            name="ssh-reachable",
            command="true",
        )
    ]
    skipped_checks: list[PreflightCheck] = []
    if unit.slurm and isinstance(config, SlurmVllmConfig):
        for command_name, command in (
            ("slurm-sbatch", _command_binary(config.sbatch_cmd)),
            ("slurm-squeue", "squeue"),
            ("slurm-scontrol", "scontrol"),
            ("slurm-sinfo", "sinfo"),
        ):
            remote_specs.append(
                _RemoteCheckSpec(
                    name=command_name,
                    command=f"command -v {shlex.quote(command)}",
                    durable_code="slurm_command_missing",
                )
            )
        remote_specs.append(
            _RemoteCheckSpec(
                name="slurm-sbatch-test-only",
                command=_slurm_sbatch_test_only_command(config),
            )
        )
        remote_specs.append(
            _RemoteCheckSpec(
                name="slurm-squeue-start",
                command='squeue --start -h -u "$USER" >/dev/null',
            )
        )
        if config.partition:
            remote_specs.append(
                _RemoteCheckSpec(
                    name="slurm-partition",
                    command=(f"sinfo -h -p {shlex.quote(config.partition)} -o '%P %t' >/dev/null"),
                    durable_code="slurm_partition_missing",
                )
            )
            remote_specs.append(
                _RemoteCheckSpec(
                    name="slurm-partition-state",
                    command=_slurm_partition_state_command(config),
                    durable_code="slurm_partition_unavailable",
                )
            )
            if config.num_gpus:
                remote_specs.append(
                    _RemoteCheckSpec(
                        name="slurm-gres-gpu",
                        command=_slurm_gres_gpu_command(config),
                        durable_code="slurm_gres_insufficient",
                    )
                )
            remote_specs.append(
                _RemoteCheckSpec(
                    name="slurm-partition-walltime",
                    command=_slurm_partition_walltime_command(config),
                    durable_code="slurm_partition_walltime_exceeded",
                )
            )
    remote_specs.append(
        _RemoteCheckSpec(
            name="python-version",
            command=_with_setup(config.setup_cmd, _python_command(config.python_bin, "--version")),
            durable_code="remote_python_missing",
        )
    )
    if config.setup_cmd:
        remote_specs.append(
            _RemoteCheckSpec(
                name="setup-command",
                command=config.setup_cmd,
            )
        )
    if unit.slurm:
        skipped_checks.append(
            PreflightCheck(
                source=unit.source,
                endpoint_name=unit.endpoint_name,
                name="vllm-import",
                outcome="skipped",
                code="requires_compute_allocation",
                attempts=0,
                duration_seconds=0.0,
                detail=(
                    "Skipped login-node vLLM import for Slurm serving. "
                    "Runtime validation must run inside a Slurm allocation."
                ),
                layer="compute-runtime",
            )
        )
    else:
        remote_specs.append(
            _RemoteCheckSpec(
                name="vllm-import",
                command=_with_setup(
                    config.setup_cmd,
                    _python_command(config.python_bin, "-c", "import vllm"),
                ),
                durable_code="vllm_import_failed",
            )
        )
    remote_specs.extend(_remote_path_check_specs(config))
    return tuple(remote_specs), tuple(skipped_checks)


def _remote_path_check_specs(
    config: SshVllmConfig | SlurmVllmConfig,
) -> tuple[_RemoteCheckSpec, ...]:
    specs: list[_RemoteCheckSpec] = []
    for label, path, writable in _path_targets(config):
        specs.append(
            _RemoteCheckSpec(
                name=f"path-{label}",
                command=_path_check_command(path, writable=writable),
                durable_code="remote_path_unwritable" if writable else "remote_path_missing",
            )
        )
    return tuple(specs)


def _local_path_checks(
    config: LocalVllmConfig,
    *,
    source: str,
    endpoint_name: str,
    timeout_seconds: int,
    command_runner: CommandRunner,
) -> tuple[PreflightCheck, ...]:
    checks: list[PreflightCheck] = []
    for label, path, writable in _path_targets(config):
        checks.append(
            _shell_check(
                source=source,
                endpoint_name=endpoint_name,
                name=f"path-{label}",
                command=_path_check_command(path, writable=writable),
                timeout_seconds=timeout_seconds,
                command_runner=command_runner,
                durable_code="remote_path_unwritable" if writable else "remote_path_missing",
                layer="local",
            )
        )
    return tuple(checks)


def _path_targets(
    config: LocalVllmConfig | SshVllmConfig | SlurmVllmConfig,
) -> tuple[tuple[str, str, bool], ...]:
    targets: list[tuple[str, str, bool]] = []
    if _looks_like_path(config.model):
        targets.append(("model", config.model, False))
    if config.hf_home:
        targets.append(("hf-home", config.hf_home, True))
    if config.runtime_tmp_root:
        targets.append(("runtime-tmp-root", config.runtime_tmp_root, True))
    out_dir = getattr(config, "out_dir", "")
    remote_out_dir_root = getattr(config, "remote_out_dir_root", "")
    if out_dir:
        targets.append(("out-dir", out_dir, True))
    elif remote_out_dir_root:
        targets.append(("remote-out-dir-root", remote_out_dir_root, True))
    return tuple(targets)


def _slurm_sbatch_test_only_command(config: SlurmVllmConfig) -> str:
    options = [
        shlex.quote(_command_binary(config.sbatch_cmd)),
        "--test-only",
        f"--partition={shlex.quote(config.partition)}",
        f"--nodes={config.nodes}",
        f"--ntasks={config.nodes}",
        "--ntasks-per-node=1",
        f"--mem={shlex.quote(config.memory)}",
        f"--cpus-per-task={config.cpus_per_task}",
        f"--gres=gpu:{config.num_gpus}",
        f"--time={shlex.quote(config.walltime)}",
    ]
    if config.exclude:
        options.append(f"--exclude={shlex.quote(config.exclude)}")
    if config.nodelist:
        options.append(f"--nodelist={shlex.quote(config.nodelist)}")
    options.append("--wrap=:")
    return " ".join(options)


def _slurm_partition_walltime_command(config: SlurmVllmConfig) -> str:
    partition = shlex.quote(config.partition)
    requested = shlex.quote(config.walltime)
    return " ".join(
        [
            f"max_time=$(sinfo -h -p {partition} -o '%l' | awk 'NF {{ print $1; exit }}');",
            'if [ -z "$max_time" ]; then',
            'echo "partition max walltime unavailable" >&2;',
            "exit 1;",
            "fi;",
            "to_seconds() {",
            "value=$1;",
            'case "$value" in',
            "infinite|INFINITE|unlimited|UNLIMITED) echo -1; return 0 ;;",
            "esac;",
            "days=0;",
            "clock=$value;",
            'case "$clock" in',
            "*-*) days=${clock%%-*}; clock=${clock#*-} ;;",
            "esac;",
            "old_ifs=$IFS;",
            "IFS=:;",
            "set -- $clock;",
            "IFS=$old_ifs;",
            "case $# in",
            "3) hours=$1; minutes=$2; seconds=$3 ;;",
            "2) hours=0; minutes=$1; seconds=$2 ;;",
            "1) hours=0; minutes=$1; seconds=0 ;;",
            "*) return 1 ;;",
            "esac;",
            'case "$days$hours$minutes$seconds" in',
            "*[!0-9]*) return 1 ;;",
            "esac;",
            "echo $((days * 86400 + hours * 3600 + minutes * 60 + seconds));",
            "};",
            f"requested_seconds=$(to_seconds {requested}) || exit 1;",
            'max_seconds=$(to_seconds "$max_time") || exit 1;',
            'if [ "$max_seconds" -lt 0 ] ||',
            '[ "$requested_seconds" -le "$max_seconds" ]; then',
            "exit 0;",
            "fi;",
            (f'echo requested walltime {requested} exceeds partition max "$max_time" >&2;'),
            "exit 1",
        ]
    )


def _slurm_partition_state_command(config: SlurmVllmConfig) -> str:
    partition = shlex.quote(config.partition)
    partition_label = shlex.quote(config.partition)
    return " ".join(
        [
            f"states=$(sinfo -h -p {partition} -o '%t' | awk 'NF {{ print $1 }}');",
            'if [ -z "$states" ]; then',
            f"echo partition {partition_label} state unavailable >&2;",
            "exit 1;",
            "fi;",
            "schedulable=0;",
            "bad_states=;",
            "for raw_state in $states; do",
            "state=${raw_state%\\*};",
            'case "$state" in',
            "idle|alloc|mix|comp|resv|plnd) schedulable=1 ;;",
            '*) bad_states="$bad_states $raw_state" ;;',
            "esac;",
            "done;",
            'if [ "$schedulable" -eq 1 ]; then',
            "exit 0;",
            "fi;",
            f'echo partition {partition_label} has no schedulable node states: "$states" >&2;',
            "exit 1",
        ]
    )


def _slurm_gres_gpu_command(config: SlurmVllmConfig) -> str:
    partition = shlex.quote(config.partition)
    partition_label = shlex.quote(config.partition)
    requested_gpus = int(config.num_gpus or 0)
    return " ".join(
        [
            f"gres_values=$(sinfo -h -p {partition} -o '%G' | awk 'NF {{ print $0 }}');",
            'if [ -z "$gres_values" ]; then',
            "exit 0;",
            "fi;",
            "max_gpus=0;",
            "saw_gpu=0;",
            "while IFS= read -r item; do",
            'case "$item" in',
            '""|"(null)"|none|N/A) continue ;;',
            "esac;",
            'case "$item" in',
            "gpu|gpu:*) saw_gpu=1 ;;",
            "*) continue ;;",
            "esac;",
            "count=${item##*:};",
            "count=${count%%(*};",
            "count=${count%%[*};",
            'case "$count" in',
            '""|*[!0-9]*) count=1 ;;',
            "esac;",
            'if [ "$count" -gt "$max_gpus" ]; then',
            "max_gpus=$count;",
            "fi;",
            'done < <(printf "%s\\n" "$gres_values" | tr "," "\\n");',
            'if [ "$saw_gpu" -eq 0 ]; then',
            "exit 0;",
            "fi;",
            f'if [ "$max_gpus" -ge {requested_gpus} ]; then',
            "exit 0;",
            "fi;",
            (
                f"echo partition {partition_label} exposes at most "
                f'"$max_gpus" GPUs via GRES but {requested_gpus} were requested >&2;'
            ),
            "exit 1",
        ]
    )


def _slurm_preflight_candidates(
    config: SlurmVllmConfig,
    *,
    source: str,
) -> tuple[tuple[str, SlurmVllmConfig], ...]:
    if not config.resource_preferences:
        return ((source, config),)
    return tuple(
        (f"{source} resource {name!r}", candidate)
        for name, candidate in slurm_resource_preference_candidates(config)
    )


def _shell_check(
    *,
    source: str,
    endpoint_name: str,
    name: str,
    command: str,
    timeout_seconds: int,
    command_runner: CommandRunner,
    durable_code: str = "",
    layer: str = "",
) -> PreflightCheck:
    start = time.monotonic()
    completed = command_runner(
        ["bash", "-lc", command],
        timeout_seconds,
    )
    return _completed_check(
        source=source,
        endpoint_name=endpoint_name,
        name=name,
        completed=completed,
        attempts=1,
        duration_seconds=time.monotonic() - start,
        command_summary=_command_summary(completed.args),
        durable_code=durable_code,
        layer=layer,
    )


def _ssh_batch_checks(
    *,
    ssh_target: str,
    items: tuple[tuple[_RemotePreflightUnit, _RemoteCheckSpec], ...],
    command_runner: CommandRunner,
    policy: RemoteCommandPolicy,
) -> tuple[PreflightCheck, ...]:
    return tuple(
        _completed_check(
            source=unit.source,
            endpoint_name=unit.endpoint_name,
            name=spec.name,
            completed=result.to_completed_process(),
            attempts=result.attempts,
            duration_seconds=result.duration_seconds,
            command_summary=result.command_summary,
            durable_code=spec.durable_code,
            ssh_target=ssh_target,
            layer=_preflight_check_layer(spec.name, spec.durable_code),
        )
        for (unit, spec), result in zip(
            items,
            run_ssh_batch(
                ssh_target=ssh_target,
                commands=tuple(
                    RemoteCommand(name=spec.name, command=spec.command) for _unit, spec in items
                ),
                command_runner=command_runner,
                policy=policy,
            ),
            strict=True,
        )
    )


def _completed_check(
    *,
    source: str,
    endpoint_name: str,
    name: str,
    completed: subprocess.CompletedProcess[str],
    attempts: int,
    duration_seconds: float,
    command_summary: str,
    durable_code: str = "",
    ssh_target: str = "",
    layer: str = "",
) -> PreflightCheck:
    outcome, code = _classify_completed_check(completed, durable_code=durable_code)
    if outcome == "ok":
        return PreflightCheck(
            source=source,
            endpoint_name=endpoint_name,
            name=name,
            outcome="ok",
            code="ok",
            attempts=attempts,
            duration_seconds=duration_seconds,
            command_summary=command_summary,
            layer=layer,
        )
    detail = _check_excerpt(completed)
    return PreflightCheck(
        source=source,
        endpoint_name=endpoint_name,
        name=name,
        outcome=outcome,
        code=code,
        attempts=attempts,
        duration_seconds=duration_seconds,
        detail=_redact(detail, sensitive_values=(ssh_target,)),
        command_summary=command_summary,
        layer=layer,
    )


def _classify_completed_check(
    completed: subprocess.CompletedProcess[str],
    *,
    durable_code: str,
) -> tuple[PreflightOutcome, str]:
    if completed.returncode == 0:
        return "ok", "ok"
    transient_code = transient_transport_code(completed)
    if transient_code:
        return "transient_failure", transient_code
    if durable_code:
        return "durable_failure", durable_code
    return "unknown", "unknown"


def _run_command(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return subprocess.CompletedProcess(
            command,
            returncode=124,
            stdout=error.stdout or "",
            stderr=f"Timed out after {timeout_seconds}s.",
        )
    except OSError as error:
        return subprocess.CompletedProcess(
            command,
            returncode=127,
            stdout="",
            stderr=str(error),
        )


def _python_command(python_bin: str, *args: str) -> str:
    return " ".join([shlex.quote(python_bin), *(shlex.quote(arg) for arg in args)])


def _with_setup(setup_cmd: str, command: str) -> str:
    if not setup_cmd:
        return command
    return f"{setup_cmd}\n{command}"


def _command_binary(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        return command
    return parts[0] if parts else command


def _path_check_command(path: str, *, writable: bool) -> str:
    quoted = shlex.quote(path)
    prefix = f'p={quoted}; case "$p" in \'~\') p="$HOME" ;; \'~/\'*) p="$HOME/${{p#~/}}" ;; esac; '
    if writable:
        return (
            prefix + 'if [ -e "$p" ]; then [ -d "$p" ] && [ -w "$p" ]; '
            'else parent=$(dirname "$p"); [ -d "$parent" ] && [ -w "$parent" ]; fi'
        )
    return prefix + 'test -e "$p" && test -r "$p"'


def _looks_like_path(value: str) -> bool:
    if not value:
        return False
    if value.startswith(("/", "~/", "./", "../")):
        return True
    path = PurePosixPath(value)
    return len(path.parts) > 1 and value.count("/") != 1


def _check_excerpt(completed: subprocess.CompletedProcess[str]) -> str:
    text = "\n".join(part for part in (completed.stderr, completed.stdout) if part).strip()
    if not text:
        text = f"Command exited with status {completed.returncode}."
    return _redact(text[:500])
