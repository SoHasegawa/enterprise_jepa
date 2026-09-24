"""Allocation-backed runtime preflight checks."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import Any

from remote_inference_launcher.endpoint_race import EndpointRaceConfig
from remote_inference_launcher.fleet import FleetConfig
from remote_inference_launcher.plan_paths import join_remote_path, model_label, safe_endpoint_label
from remote_inference_launcher.plans import EffectiveEndpointPlan, EffectiveLaunchPlan
from remote_inference_launcher.preflight import PreflightCheck
from remote_inference_launcher.remote_execution import (
    CommandRunner,
    RemoteCommandPolicy,
    RemoteResult,
    redact,
    run_ssh_command,
    transient_transport_code,
)
from remote_inference_launcher.slurm import (
    new_slurm_submission_comment,
    parse_squeue_job_info,
    scancel_job_command,
    squeue_job_ids_by_comment_command,
    squeue_job_info_command,
)
from remote_inference_launcher.slurm_vllm import (
    DEFAULT_REMOTE_OUT_DIR_ROOT,
    SlurmVllmConfig,
    slurm_resource_preference_candidates,
    with_slurm_vllm_defaults,
)

RUNTIME_PREFLIGHT_SCHEMA_VERSION = "ril-slurm-runtime-preflight/v1"
RUNTIME_PREFLIGHT_MANIFEST = "runtime_preflight_manifest.json"
RUNTIME_PREFLIGHT_SCRIPT = "runtime_preflight.sbatch"
DEFAULT_RUNTIME_PREFLIGHT_TIMEOUT_SECONDS = 900
DEFAULT_RUNTIME_PREFLIGHT_POLL_INTERVAL_SECONDS = 10.0

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
_TERMINAL_STATES = _FAILED_TERMINAL_STATES | {"COMPLETED", "NOT_FOUND"}


@dataclass(frozen=True)
class RuntimePreflightPolicy:
    """Runtime preflight polling and SSH policy."""

    timeout_seconds: int = DEFAULT_RUNTIME_PREFLIGHT_TIMEOUT_SECONDS
    poll_interval_seconds: float = DEFAULT_RUNTIME_PREFLIGHT_POLL_INTERVAL_SECONDS
    ssh_policy: RemoteCommandPolicy = field(default_factory=RemoteCommandPolicy)


@dataclass(frozen=True)
class _RuntimeCheckSpec:
    name: str
    failure_code: str
    command: str


@dataclass(frozen=True)
class _SlurmRuntimeContext:
    config: SlurmVllmConfig
    source: str
    endpoint_plan: EffectiveEndpointPlan | None
    run_id: str
    remote_home: str
    out_dir: str
    script_path: str
    manifest_path: str


def run_runtime_preflight_checks(
    config: object,
    *,
    source: str,
    effective_plan: EffectiveLaunchPlan | None = None,
    command_runner: CommandRunner | None = None,
    policy: RuntimePreflightPolicy | None = None,
) -> tuple[PreflightCheck, ...]:
    """Run opt-in compute-node runtime preflight checks for Slurm endpoints."""

    runner = command_runner or _run_command
    resolved_policy = policy or RuntimePreflightPolicy()
    checks: list[PreflightCheck] = []
    _append_runtime_preflight_checks(
        checks,
        config,
        source=source,
        effective_plan=effective_plan,
        command_runner=runner,
        policy=resolved_policy,
    )
    return tuple(checks)


def _append_runtime_preflight_checks(
    checks: list[PreflightCheck],
    config: object,
    *,
    source: str,
    effective_plan: EffectiveLaunchPlan | None,
    command_runner: CommandRunner,
    policy: RuntimePreflightPolicy,
) -> None:
    if isinstance(config, FleetConfig):
        for name, endpoint in sorted(config.endpoints.items()):
            _append_runtime_preflight_checks(
                checks,
                endpoint,
                source=f"{source} endpoint {name!r}",
                effective_plan=effective_plan,
                command_runner=command_runner,
                policy=policy,
            )
        return
    if isinstance(config, EndpointRaceConfig):
        for name, candidate in sorted(config.candidates.items()):
            _append_runtime_preflight_checks(
                checks,
                candidate,
                source=f"{source} candidate {name!r}",
                effective_plan=effective_plan,
                command_runner=command_runner,
                policy=policy,
            )
        return
    if not isinstance(config, SlurmVllmConfig):
        return
    for candidate_source, candidate in _slurm_runtime_candidates(config, source=source):
        endpoint_plan = _matching_endpoint_plan(effective_plan, source=candidate_source)
        checks.extend(
            _run_slurm_runtime_preflight(
                candidate,
                source=candidate_source,
                endpoint_plan=endpoint_plan,
                run_id=effective_plan.run_id if effective_plan is not None else "",
                command_runner=command_runner,
                policy=policy,
            )
        )


def _run_slurm_runtime_preflight(
    config: SlurmVllmConfig,
    *,
    source: str,
    endpoint_plan: EffectiveEndpointPlan | None,
    run_id: str,
    command_runner: CommandRunner,
    policy: RuntimePreflightPolicy,
) -> tuple[PreflightCheck, ...]:
    resolved = _runtime_config(config, endpoint_plan=endpoint_plan, run_id=run_id)
    remote_home = run_ssh_command(
        ssh_target=resolved.ssh_target,
        name="runtime-preflight-remote-home",
        command="cd && pwd",
        command_runner=command_runner,
        policy=policy.ssh_policy,
    )
    if remote_home.returncode:
        return (
            _remote_failure_check(
                source=source,
                endpoint_name=resolved.name,
                name="runtime-preflight-remote-home",
                result=remote_home,
                durable_code="runtime_preflight_remote_home_failed",
                ssh_target=resolved.ssh_target,
            ),
        )
    context = _runtime_context(
        resolved,
        source=source,
        endpoint_plan=endpoint_plan,
        run_id=run_id,
        remote_home=remote_home.stdout.strip(),
    )
    setup = run_ssh_command(
        ssh_target=resolved.ssh_target,
        name="runtime-preflight-setup",
        command=_write_remote_script_command(
            render_slurm_runtime_preflight_sbatch(context.config, context.manifest_path),
            script_path=context.script_path,
            out_dir=context.out_dir,
        ),
        command_runner=command_runner,
        policy=policy.ssh_policy,
    )
    if setup.returncode:
        return (
            _remote_failure_check(
                source=source,
                endpoint_name=resolved.name,
                name="runtime-preflight-setup",
                result=setup,
                durable_code="runtime_preflight_setup_failed",
                ssh_target=resolved.ssh_target,
                command_summary="ssh <ssh-target> bash -lc <runtime preflight script upload>",
            ),
        )
    submitted = _submit_runtime_preflight(context, command_runner=command_runner, policy=policy)
    if isinstance(submitted, PreflightCheck):
        return (submitted,)
    job_id = submitted
    return _wait_for_runtime_manifest(
        context,
        job_id=job_id,
        command_runner=command_runner,
        policy=policy,
    )


def render_slurm_runtime_preflight_sbatch(
    config: SlurmVllmConfig,
    manifest_path: str,
) -> str:
    """Render a Slurm job that writes runtime validation JSON and exits."""

    checks = _runtime_check_specs(config)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            *_sbatch_directives(config, out_dir=_parent_dir(manifest_path)),
            "set +e",
            "set +u",
            "set +o pipefail",
            f"manifest_path={shlex.quote(manifest_path)}",
            'check_dir="$(dirname "$manifest_path")/runtime-checks"',
            'mkdir -p "$check_dir"',
            _runtime_check_function(),
            *(
                f"run_runtime_check {index} {shlex.quote(check.name)} {shlex.quote(check.command)}"
                for index, check in enumerate(checks)
            ),
            _runtime_manifest_writer(checks),
        ]
    )


def _runtime_config(
    config: SlurmVllmConfig,
    *,
    endpoint_plan: EffectiveEndpointPlan | None,
    run_id: str,
) -> SlurmVllmConfig:
    resolved = with_slurm_vllm_defaults(config, run_id=run_id or None)
    if endpoint_plan is not None and endpoint_plan.job_name:
        resolved = replace(resolved, job_name=endpoint_plan.job_name)
    return resolved


def _runtime_context(
    config: SlurmVllmConfig,
    *,
    source: str,
    endpoint_plan: EffectiveEndpointPlan | None,
    run_id: str,
    remote_home: str,
) -> _SlurmRuntimeContext:
    runtime_id = run_id or uuid.uuid4().hex[:16]
    base_out_dir = endpoint_plan.out_dir if endpoint_plan is not None else config.out_dir
    if not base_out_dir:
        base_out_dir = join_remote_path(
            config.remote_out_dir_root or f"~/{DEFAULT_REMOTE_OUT_DIR_ROOT}",
            model_label(config.model),
            f"{safe_endpoint_label(config.name)}-{runtime_id}",
        )
    out_dir = join_remote_path(_expand_remote_home(base_out_dir, remote_home), "runtime-preflight")
    return _SlurmRuntimeContext(
        config=config,
        source=source,
        endpoint_plan=endpoint_plan,
        run_id=runtime_id,
        remote_home=remote_home,
        out_dir=out_dir,
        script_path=join_remote_path(out_dir, RUNTIME_PREFLIGHT_SCRIPT),
        manifest_path=join_remote_path(out_dir, RUNTIME_PREFLIGHT_MANIFEST),
    )


def _submit_runtime_preflight(
    context: _SlurmRuntimeContext,
    *,
    command_runner: CommandRunner,
    policy: RuntimePreflightPolicy,
) -> str | PreflightCheck:
    submission_comment = new_slurm_submission_comment()
    submit = run_ssh_command(
        ssh_target=context.config.ssh_target,
        name="runtime-preflight-submit",
        command=" ".join(
            [
                shlex.quote(context.config.sbatch_cmd),
                "--parsable",
                f"--comment={shlex.quote(submission_comment)}",
                shlex.quote(context.script_path),
            ]
        ),
        command_runner=command_runner,
        policy=policy.ssh_policy,
        retry_safe=False,
    )
    if submit.returncode:
        recovered_job_id = ""
        if transient_transport_code(submit.to_completed_process()):
            recovered_job_id = _recover_submitted_job_id(
                context,
                submission_comment,
                command_runner=command_runner,
                policy=policy,
            )
        if recovered_job_id:
            return recovered_job_id
        return _remote_failure_check(
            source=context.source,
            endpoint_name=context.config.name,
            name="runtime-preflight-submit",
            result=submit,
            durable_code="runtime_preflight_submission_failed",
            ssh_target=context.config.ssh_target,
        )
    job_id = _parse_sbatch_job_id(submit.stdout)
    if not job_id:
        return _manual_check(
            source=context.source,
            endpoint_name=context.config.name,
            name="runtime-preflight-submit",
            outcome="durable_failure",
            code="runtime_preflight_submission_failed",
            attempts=submit.attempts,
            duration_seconds=submit.duration_seconds,
            detail="Remote Slurm submission did not return a job id.",
            command_summary=submit.command_summary,
        )
    return job_id


def _recover_submitted_job_id(
    context: _SlurmRuntimeContext,
    submission_comment: str,
    *,
    command_runner: CommandRunner,
    policy: RuntimePreflightPolicy,
) -> str:
    query = run_ssh_command(
        ssh_target=context.config.ssh_target,
        name="runtime-preflight-submit-recovery",
        command=squeue_job_ids_by_comment_command(submission_comment),
        command_runner=command_runner,
        policy=policy.ssh_policy,
    )
    if query.returncode:
        return ""
    job_ids = tuple(line.strip() for line in query.stdout.splitlines() if line.strip())
    return job_ids[0] if len(job_ids) == 1 else ""


def _wait_for_runtime_manifest(
    context: _SlurmRuntimeContext,
    *,
    job_id: str,
    command_runner: CommandRunner,
    policy: RuntimePreflightPolicy,
) -> tuple[PreflightCheck, ...]:
    started_at = time.monotonic()
    not_found_once = False
    while True:
        manifest_result = _read_runtime_manifest(
            context,
            command_runner=command_runner,
            policy=policy,
        )
        if isinstance(manifest_result, PreflightCheck):
            return (manifest_result,)
        if manifest_result is not None:
            return _checks_from_manifest(
                context,
                manifest_result,
                job_id=job_id,
                duration_seconds=time.monotonic() - started_at,
            )
        info_result = _runtime_job_info(
            context,
            job_id=job_id,
            command_runner=command_runner,
            policy=policy,
        )
        if isinstance(info_result, PreflightCheck):
            return (info_result,)
        state, _node, reason = info_result
        if state != "NOT_FOUND":
            not_found_once = False
        if state in _FAILED_TERMINAL_STATES or (state == "NOT_FOUND" and not_found_once):
            return (
                _manual_check(
                    source=context.source,
                    endpoint_name=context.config.name,
                    name="runtime-preflight-job",
                    outcome="durable_failure",
                    code="runtime_preflight_failed",
                    attempts=1,
                    duration_seconds=time.monotonic() - started_at,
                    detail=(
                        f"Slurm runtime preflight job {job_id} ended with "
                        f"state={state} reason={reason} remote_logs={context.out_dir}"
                    ),
                    command_summary=f"slurm runtime preflight job {job_id}",
                ),
            )
        if state == "NOT_FOUND":
            not_found_once = True
        if time.monotonic() - started_at >= policy.timeout_seconds:
            _cancel_runtime_preflight(context, job_id, command_runner=command_runner, policy=policy)
            return (
                _manual_check(
                    source=context.source,
                    endpoint_name=context.config.name,
                    name="runtime-preflight-job",
                    outcome="transient_failure",
                    code="runtime_preflight_timeout",
                    attempts=1,
                    duration_seconds=time.monotonic() - started_at,
                    detail=(
                        f"Timed out waiting for Slurm runtime preflight job {job_id}; "
                        f"remote_logs={context.out_dir}"
                    ),
                    command_summary=f"slurm runtime preflight job {job_id}",
                ),
            )
        if policy.poll_interval_seconds > 0:
            time.sleep(policy.poll_interval_seconds)


def _read_runtime_manifest(
    context: _SlurmRuntimeContext,
    *,
    command_runner: CommandRunner,
    policy: RuntimePreflightPolicy,
) -> Mapping[str, Any] | PreflightCheck | None:
    result = run_ssh_command(
        ssh_target=context.config.ssh_target,
        name="runtime-preflight-manifest",
        command=f"if [ -s {shlex.quote(context.manifest_path)} ]; then cat "
        f"{shlex.quote(context.manifest_path)}; fi",
        command_runner=command_runner,
        policy=policy.ssh_policy,
    )
    if result.returncode:
        return _remote_failure_check(
            source=context.source,
            endpoint_name=context.config.name,
            name="runtime-preflight-manifest",
            result=result,
            durable_code="runtime_preflight_manifest_read_failed",
            ssh_target=context.config.ssh_target,
        )
    if not result.stdout.strip():
        return None
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return _manual_check(
            source=context.source,
            endpoint_name=context.config.name,
            name="runtime-preflight-manifest",
            outcome="durable_failure",
            code="runtime_preflight_manifest_invalid",
            attempts=result.attempts,
            duration_seconds=result.duration_seconds,
            detail=f"Runtime preflight manifest is not valid JSON: {error}",
            command_summary=result.command_summary,
        )
    if not isinstance(parsed, Mapping):
        return _manual_check(
            source=context.source,
            endpoint_name=context.config.name,
            name="runtime-preflight-manifest",
            outcome="durable_failure",
            code="runtime_preflight_manifest_invalid",
            attempts=result.attempts,
            duration_seconds=result.duration_seconds,
            detail="Runtime preflight manifest must be a JSON object.",
            command_summary=result.command_summary,
        )
    return parsed


def _runtime_job_info(
    context: _SlurmRuntimeContext,
    *,
    job_id: str,
    command_runner: CommandRunner,
    policy: RuntimePreflightPolicy,
) -> tuple[str, str, str] | PreflightCheck:
    result = run_ssh_command(
        ssh_target=context.config.ssh_target,
        name="runtime-preflight-squeue",
        command=squeue_job_info_command(job_id),
        command_runner=command_runner,
        policy=policy.ssh_policy,
    )
    if result.returncode:
        return _remote_failure_check(
            source=context.source,
            endpoint_name=context.config.name,
            name="runtime-preflight-squeue",
            result=result,
            durable_code="runtime_preflight_squeue_failed",
            ssh_target=context.config.ssh_target,
        )
    parsed = parse_squeue_job_info(result.stdout)
    return parsed if parsed is not None else ("NOT_FOUND", "", "")


def _checks_from_manifest(
    context: _SlurmRuntimeContext,
    manifest: Mapping[str, Any],
    *,
    job_id: str,
    duration_seconds: float,
) -> tuple[PreflightCheck, ...]:
    checks = [
        _manual_check(
            source=context.source,
            endpoint_name=context.config.name,
            name="runtime-preflight-job",
            outcome="ok",
            code="ok",
            attempts=1,
            duration_seconds=duration_seconds,
            detail=(
                f"job_id={job_id} node={manifest.get('node') or ''} remote_logs={context.out_dir}"
            ),
            command_summary=f"slurm runtime preflight job {job_id}",
        )
    ]
    for item in _manifest_checks(manifest):
        returncode = int(item.get("returncode", 1))
        checks.append(
            _manual_check(
                source=context.source,
                endpoint_name=context.config.name,
                name=str(item.get("name") or "runtime-check"),
                outcome="ok" if returncode == 0 else "durable_failure",
                code="ok" if returncode == 0 else str(item.get("code") or "runtime_check_failed"),
                attempts=1,
                duration_seconds=duration_seconds,
                detail=_manifest_check_detail(item),
                command_summary=f"slurm runtime preflight job {job_id}",
            )
        )
    return tuple(checks)


def _manifest_checks(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    checks = manifest.get("checks")
    if not isinstance(checks, list):
        return ()
    return tuple(item for item in checks if isinstance(item, Mapping))


def _manifest_check_detail(item: Mapping[str, Any]) -> str:
    parts = [
        str(value) for value in (item.get("stderr_excerpt"), item.get("stdout_excerpt")) if value
    ]
    return redact("\n".join(parts)[:500])


def _cancel_runtime_preflight(
    context: _SlurmRuntimeContext,
    job_id: str,
    *,
    command_runner: CommandRunner,
    policy: RuntimePreflightPolicy,
) -> None:
    run_ssh_command(
        ssh_target=context.config.ssh_target,
        name="runtime-preflight-cancel",
        command=scancel_job_command(job_id),
        command_runner=command_runner,
        policy=policy.ssh_policy,
    )


def _runtime_check_specs(config: SlurmVllmConfig) -> tuple[_RuntimeCheckSpec, ...]:
    specs: list[_RuntimeCheckSpec] = []
    if config.setup_cmd:
        specs.append(
            _RuntimeCheckSpec(
                name="runtime-setup-command",
                failure_code="runtime_setup_failed",
                command=config.setup_cmd,
            )
        )
    specs.append(
        _RuntimeCheckSpec(
            name="runtime-python-version",
            failure_code="runtime_python_failed",
            command=_with_setup(config.setup_cmd, _python_command(config.python_bin, "--version")),
        )
    )
    specs.append(
        _RuntimeCheckSpec(
            name="runtime-vllm-import",
            failure_code="runtime_vllm_import_failed",
            command=_with_setup(
                config.setup_cmd, _python_command(config.python_bin, "-c", "import vllm")
            ),
        )
    )
    if _target_device_requires_torch_device(config.target_device):
        specs.append(
            _RuntimeCheckSpec(
                name="runtime-torch-device",
                failure_code="runtime_torch_device_failed",
                command=_with_setup(
                    config.setup_cmd,
                    _python_command(
                        config.python_bin,
                        "-c",
                        (
                            "import torch; "
                            "ok=torch.cuda.is_available(); "
                            "print(f'torch_cuda_available={ok}'); "
                            "raise SystemExit(0 if ok else 1)"
                        ),
                    ),
                ),
            )
        )
    if config.distributed_backend == "ray":
        specs.append(
            _RuntimeCheckSpec(
                name="runtime-ray-import",
                failure_code="runtime_ray_import_failed",
                command=_with_setup(
                    config.setup_cmd, _python_command(config.python_bin, "-c", "import ray")
                ),
            )
        )
    if _looks_like_path(config.model):
        specs.append(
            _RuntimeCheckSpec(
                name="runtime-model-path",
                failure_code="runtime_model_path_missing",
                command=_path_check_command(config.model),
            )
        )
    return tuple(specs)


def _runtime_check_function() -> str:
    return "\n".join(
        [
            "run_runtime_check() {",
            "  idx=$1",
            "  name=$2",
            "  command=$3",
            '  prefix="$check_dir/${idx}-${name}"',
            '  bash -lc "$command" >"${prefix}.stdout" 2>"${prefix}.stderr"',
            "  rc=$?",
            '  printf "%s\\n" "$rc" >"${prefix}.rc"',
            "}",
        ]
    )


def _runtime_manifest_writer(checks: tuple[_RuntimeCheckSpec, ...]) -> str:
    specs_payload = json.dumps(
        [
            {"index": index, "name": check.name, "code": check.failure_code}
            for index, check in enumerate(checks)
        ],
        sort_keys=True,
    )
    return "\n".join(
        [
            'python3 - "$manifest_path" "$check_dir" <<\'PY\'',
            "import json",
            "import os",
            "import pathlib",
            "import sys",
            f"specs = {specs_payload}",
            "manifest_path = pathlib.Path(sys.argv[1])",
            "check_dir = pathlib.Path(sys.argv[2])",
            "checks = []",
            "for spec in specs:",
            "    prefix = check_dir / f\"{spec['index']}-{spec['name']}\"",
            "    rc_text = (prefix.with_suffix('.rc')).read_text(encoding='utf-8').strip()",
            "    rc = int(rc_text) if rc_text else 1",
            "    stdout = (prefix.with_suffix('.stdout')).read_text(encoding='utf-8', errors='replace')",
            "    stderr = (prefix.with_suffix('.stderr')).read_text(encoding='utf-8', errors='replace')",
            "    checks.append({",
            "        'name': spec['name'],",
            "        'returncode': rc,",
            "        'code': 'ok' if rc == 0 else spec['code'],",
            "        'stdout_excerpt': stdout[:1000],",
            "        'stderr_excerpt': stderr[:1000],",
            "    })",
            "manifest = {",
            f"    'schema_version': {RUNTIME_PREFLIGHT_SCHEMA_VERSION!r},",
            "    'status': 'ok' if all(item['returncode'] == 0 for item in checks) else 'failed',",
            "    'job_id': os.environ.get('SLURM_JOB_ID', ''),",
            "    'node': os.environ.get('SLURMD_NODENAME') or os.environ.get('HOSTNAME', ''),",
            "    'checks': checks,",
            "}",
            "manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')",
            "PY",
        ]
    )


def _sbatch_directives(config: SlurmVllmConfig, *, out_dir: str) -> tuple[str, ...]:
    job_name = safe_endpoint_label(f"{config.job_name}-runtime-preflight")[:64]
    lines = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={config.partition}",
        f"#SBATCH --nodes={config.nodes}",
        f"#SBATCH --ntasks={config.nodes}",
        f"#SBATCH --mem={config.memory}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task={config.cpus_per_task}",
        f"#SBATCH --gres=gpu:{config.num_gpus}",
        f"#SBATCH --time={config.walltime}",
        f"#SBATCH --output={out_dir.rstrip('/')}/runtime-preflight-%j.out",
        f"#SBATCH --error={out_dir.rstrip('/')}/runtime-preflight-%j.err",
    ]
    if config.exclude:
        lines.append(f"#SBATCH --exclude={config.exclude}")
    if config.nodelist:
        lines.append(f"#SBATCH --nodelist={config.nodelist}")
    return tuple(lines)


def _write_remote_script_command(script: str, *, script_path: str, out_dir: str) -> str:
    delimiter = "__RIL_RUNTIME_PREFLIGHT_SCRIPT__"
    while delimiter in script:
        delimiter += "_"
    return "\n".join(
        [
            f"mkdir -p {shlex.quote(out_dir)}",
            f"cat > {shlex.quote(script_path)} <<'{delimiter}'",
            script,
            delimiter,
            f"chmod 700 {shlex.quote(script_path)}",
        ]
    )


def _remote_failure_check(
    *,
    source: str,
    endpoint_name: str,
    name: str,
    result: RemoteResult,
    durable_code: str,
    ssh_target: str,
    command_summary: str | None = None,
) -> PreflightCheck:
    completed = result.to_completed_process()
    transient_code = transient_transport_code(completed)
    outcome = "transient_failure" if transient_code else "durable_failure"
    return _manual_check(
        source=source,
        endpoint_name=endpoint_name,
        name=name,
        outcome=outcome,
        code=transient_code or durable_code,
        attempts=result.attempts,
        duration_seconds=result.duration_seconds,
        detail=redact(_result_excerpt(result), sensitive_values=(ssh_target,)),
        command_summary=command_summary or result.command_summary,
    )


def _manual_check(
    *,
    source: str,
    endpoint_name: str,
    name: str,
    outcome: str,
    code: str,
    attempts: int,
    duration_seconds: float,
    detail: str = "",
    command_summary: str = "",
) -> PreflightCheck:
    return PreflightCheck(
        source=source,
        endpoint_name=endpoint_name,
        name=name,
        outcome=outcome,  # type: ignore[arg-type]
        code=code,
        attempts=attempts,
        duration_seconds=duration_seconds,
        detail=detail,
        command_summary=command_summary,
        layer="compute-runtime",
    )


def _result_excerpt(result: RemoteResult) -> str:
    text = "\n".join(part for part in (result.stderr, result.stdout) if part).strip()
    return text[:500] if text else f"Command exited with status {result.returncode}."


def _slurm_runtime_candidates(
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


def _matching_endpoint_plan(
    effective_plan: EffectiveLaunchPlan | None,
    *,
    source: str,
) -> EffectiveEndpointPlan | None:
    if effective_plan is None:
        return None
    for endpoint in effective_plan.endpoints:
        if endpoint.raw_config_source == source and endpoint.kind == "slurm_vllm":
            return endpoint
    return None


def _parse_sbatch_job_id(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1].split(";", maxsplit=1)[0] if lines else ""


def _python_command(python_bin: str, *args: str) -> str:
    return " ".join([shlex.quote(python_bin), *(shlex.quote(arg) for arg in args)])


def _with_setup(setup_cmd: str, command: str) -> str:
    if not setup_cmd:
        return command
    return f"{setup_cmd}\n{command}"


def _path_check_command(path: str) -> str:
    quoted = shlex.quote(path)
    prefix = f'p={quoted}; case "$p" in \'~\') p="$HOME" ;; \'~/\'*) p="$HOME/${{p#~/}}" ;; esac; '
    return prefix + 'test -e "$p" && test -r "$p"'


def _target_device_requires_torch_device(target_device: str) -> bool:
    return target_device.lower() in {"cuda", "gpu", "rocm"}


def _looks_like_path(value: str) -> bool:
    if not value:
        return False
    if value.startswith(("/", "~/", "./", "../")):
        return True
    path = PurePosixPath(value)
    return len(path.parts) > 1 and value.count("/") != 1


def _expand_remote_home(path: str, remote_home: str) -> str:
    if path == "~":
        return remote_home
    if path.startswith("~/"):
        return f"{remote_home.rstrip('/')}/{path[2:]}"
    return path


def _parent_dir(path: str) -> str:
    return path.rsplit("/", maxsplit=1)[0] if "/" in path else "."


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
