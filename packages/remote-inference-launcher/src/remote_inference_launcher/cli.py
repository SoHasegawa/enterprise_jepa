"""Command line interface for remote inference launcher."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

from remote_inference_launcher.lease_cli import add_lease_subcommands
from remote_inference_launcher.pool_cli import add_pool_subcommands
from remote_inference_launcher.registry_cli import add_registry_subcommands
from remote_inference_launcher.session_env import (
    bootstrap_result_payload,
    env_lines,
    generic_inference_env,
    json_dumps,
    normalize_sessions,
    sessions_payload,
    write_env_file,
)
from remote_inference_launcher.slurm_vllm import (
    SlurmVllmConfig,
    SlurmVllmLauncher,
)
from remote_inference_launcher.slurm_vllm_bootstrap import (
    SlurmVllmBootstrapConfig,
    SlurmVllmBootstrapper,
    SlurmVllmBootstrapResult,
    load_slurm_vllm_bootstrap_config,
)
from remote_inference_launcher.ssh_setup import (
    REMOTE_HOST_PRESETS,
    RemoteHost,
    prompt_missing_settings,
    render_setup,
    settings_from_preset,
    settings_from_values,
)
from remote_inference_launcher.termination import (
    shield_termination_signals,
    translate_termination_signals,
)
from remote_inference_launcher.verbosity import VERBOSITY_LEVELS
from remote_inference_launcher.vllm_bootstrap import (
    LocalVllmBootstrapConfig,
    LocalVllmBootstrapper,
    SshVllmBootstrapConfig,
    SshVllmBootstrapper,
    VllmBootstrapResult,
    load_local_vllm_bootstrap_config,
    load_ssh_vllm_bootstrap_config,
)

OUTPUT_FORMAT_HELP = "Output format."

ConfigClass = (
    type[SlurmVllmConfig]
    | type[SlurmVllmBootstrapConfig]
    | type[LocalVllmBootstrapConfig]
    | type[SshVllmBootstrapConfig]
)
ConfigObject = (
    SlurmVllmConfig | SlurmVllmBootstrapConfig | LocalVllmBootstrapConfig | SshVllmBootstrapConfig
)


def main(argv: list[str] | None = None) -> int:
    """Run the package CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    with translate_termination_signals():
        return args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(prog="remote-inference-launcher")
    subparsers = parser.add_subparsers(dest="command")
    check = subparsers.add_parser(
        "check",
        help="validate an already-running OpenAI-compatible endpoint",
    )
    check.add_argument("--base-url", required=True, help="Endpoint base URL ending in /v1.")
    check.add_argument("--api-key", default="", help="Optional endpoint API key.")
    check.add_argument(
        "--format",
        choices=("env", "json"),
        default="env",
        help=OUTPUT_FORMAT_HELP,
    )
    check.set_defaults(handler=_run_check)

    validate = subparsers.add_parser(
        "validate",
        help="validate inference configs and report launch collisions before submission",
    )
    validate.add_argument("configs", nargs="+", help="Inference launcher YAML config path.")
    validate.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    validate.add_argument(
        "--preflight",
        action="store_true",
        help="Run advisory resource-target checks such as SSH, Slurm, Python, and vLLM import.",
    )
    validate.add_argument(
        "--strict-preflight",
        action="store_true",
        help="Treat failed preflight checks as validation errors.",
    )
    validate.add_argument(
        "--fail-on-unknown-preflight",
        action="store_true",
        help="Treat unknown strict preflight failures as validation errors.",
    )
    validate.add_argument(
        "--runtime-preflight",
        action="store_true",
        help=(
            "Submit short Slurm allocation-backed runtime checks for Python, vLLM, "
            "device, and model visibility."
        ),
    )
    validate.set_defaults(handler=_run_validate)

    add_registry_subcommands(subparsers)
    add_lease_subcommands(subparsers)
    add_pool_subcommands(subparsers)

    start = subparsers.add_parser(
        "start",
        help="start an inference endpoint from a strict YAML config",
    )
    start.add_argument("--config", required=True, help="Inference launcher YAML config path.")
    start.add_argument("--env-file", help="Write shell exports after readiness.")
    start.add_argument("--launch-summary", help="Write launch summary JSON to this path.")
    start.add_argument(
        "--overwrite-launch-summary",
        action="store_true",
        help="Overwrite an existing explicit launch summary path.",
    )
    start.add_argument(
        "--format",
        choices=("env", "json", "jsonl"),
        default="env",
        help=OUTPUT_FORMAT_HELP,
    )
    start.add_argument(
        "--exit-after-ready",
        action="store_true",
        help="Exit after readiness and cleanup owned resources.",
    )
    start.add_argument(
        "--detach",
        action="store_true",
        help="Run the launcher under a detached controller process.",
    )
    start.add_argument(
        "--wait-ready",
        action="store_true",
        help="With --detach, wait until the registry reaches READY or FAILED.",
    )
    start.add_argument(
        "--ready-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for --detach --wait-ready.",
    )
    start.set_defaults(handler=_run_start)

    generic_bootstrap = subparsers.add_parser(
        "bootstrap",
        help="bootstrap a vLLM Python environment from a strict YAML config",
    )
    generic_bootstrap.add_argument(
        "--config",
        required=True,
        help="vLLM bootstrap YAML config path.",
    )
    generic_bootstrap.add_argument(
        "--format",
        choices=("env", "json"),
        default="env",
        help=OUTPUT_FORMAT_HELP,
    )
    generic_bootstrap.set_defaults(handler=_run_bootstrap)

    slurm_vllm = subparsers.add_parser(
        "slurm-vllm",
        help="launch a vLLM OpenAI-compatible endpoint through Slurm",
    )
    slurm_vllm.add_argument("--config", help="YAML Slurm vLLM config path.")
    _add_slurm_vllm_config_flags(slurm_vllm)
    _add_verbosity_alias_flags(slurm_vllm)
    slurm_vllm.add_argument(
        "--hold",
        action="store_true",
        help="Keep the launched endpoint alive until interrupted.",
    )
    slurm_vllm.set_defaults(handler=_run_slurm_vllm)

    bootstrap = subparsers.add_parser(
        "slurm-vllm-bootstrap",
        help="bootstrap a remote vLLM Python environment through Slurm",
    )
    bootstrap.add_argument("--config", help="YAML Slurm vLLM bootstrap config path.")
    _add_dataclass_config_flags(bootstrap, SlurmVllmBootstrapConfig)
    _add_verbosity_alias_flags(bootstrap)
    bootstrap.set_defaults(handler=_run_slurm_vllm_bootstrap)

    local_bootstrap = subparsers.add_parser(
        "local-vllm-bootstrap",
        help="bootstrap a local vLLM Python environment",
    )
    local_bootstrap.add_argument("--config", help="YAML local vLLM bootstrap config path.")
    _add_dataclass_config_flags(local_bootstrap, LocalVllmBootstrapConfig)
    _add_verbosity_alias_flags(local_bootstrap)
    local_bootstrap.set_defaults(handler=_run_local_vllm_bootstrap)

    ssh_bootstrap = subparsers.add_parser(
        "ssh-vllm-bootstrap",
        help="bootstrap a vLLM Python environment on a remote SSH host without Slurm",
    )
    ssh_bootstrap.add_argument("--config", help="YAML SSH vLLM bootstrap config path.")
    _add_dataclass_config_flags(ssh_bootstrap, SshVllmBootstrapConfig)
    _add_verbosity_alias_flags(ssh_bootstrap)
    ssh_bootstrap.set_defaults(handler=_run_ssh_vllm_bootstrap)

    ssh_setup = subparsers.add_parser(
        "ssh-setup",
        help="print SSH config and ssh-agent setup snippets",
    )
    ssh_setup.add_argument(
        "--preset",
        action="append",
        choices=sorted(REMOTE_HOST_PRESETS),
        help="Known internal team host preset. Can be passed multiple times.",
    )
    ssh_setup.add_argument("--alias", help="SSH host alias for a generic host.")
    ssh_setup.add_argument("--hostname", help="Hostname or address for a generic host.")
    ssh_setup.add_argument("--port", type=int, help="SSH port for a generic host.")
    ssh_setup.add_argument("--user", help="SSH username.")
    ssh_setup.add_argument("--identity-file", help="Private key path.")
    ssh_setup.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for missing values.",
    )
    ssh_setup.set_defaults(handler=_run_ssh_setup)
    return parser


def slurm_vllm_config_from_args(args: argparse.Namespace) -> SlurmVllmConfig:
    """Build `SlurmVllmConfig` from CLI args using default < YAML < CLI precedence."""

    if args.config:
        from remote_inference_launcher.inference_config import load_inference_config

        loaded = load_inference_config(args.config)
        if not isinstance(loaded, SlurmVllmConfig):
            raise ValueError(
                "slurm-vllm --config requires an inference config with kind: slurm_vllm."
            )
        config = loaded
    else:
        config = SlurmVllmConfig()
    overrides = _config_overrides_from_args(args)
    if overrides:
        config = replace(config, **overrides)
    return _apply_cli_verbosity_aliases(config, args)


def slurm_vllm_bootstrap_config_from_args(args: argparse.Namespace) -> SlurmVllmBootstrapConfig:
    """Build bootstrap config from CLI args using default < YAML < CLI precedence."""

    config = (
        load_slurm_vllm_bootstrap_config(args.config) if args.config else SlurmVllmBootstrapConfig()
    )
    overrides = _config_overrides_from_args(args, SlurmVllmBootstrapConfig)
    if overrides:
        config = replace(config, **overrides)
    return _apply_cli_verbosity_aliases(config, args)


def local_vllm_bootstrap_config_from_args(args: argparse.Namespace) -> LocalVllmBootstrapConfig:
    """Build local bootstrap config from CLI args using default < YAML < CLI precedence."""

    config = (
        load_local_vllm_bootstrap_config(args.config) if args.config else LocalVllmBootstrapConfig()
    )
    overrides = _config_overrides_from_args(args, LocalVllmBootstrapConfig)
    if overrides:
        config = replace(config, **overrides)
    return _apply_cli_verbosity_aliases(config, args)


def ssh_vllm_bootstrap_config_from_args(args: argparse.Namespace) -> SshVllmBootstrapConfig:
    """Build SSH bootstrap config from CLI args using default < YAML < CLI precedence."""

    config = (
        load_ssh_vllm_bootstrap_config(args.config) if args.config else SshVllmBootstrapConfig()
    )
    overrides = _config_overrides_from_args(args, SshVllmBootstrapConfig)
    if overrides:
        config = replace(config, **overrides)
    return _apply_cli_verbosity_aliases(config, args)


def _run_check(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.existing_endpoint import (
            ExistingEndpointConfig,
            ExistingEndpointLauncher,
        )

        session = ExistingEndpointLauncher(
            ExistingEndpointConfig(api_base=args.base_url, api_key=args.api_key)
        ).start()
        sessions = {"default": session}
        env = generic_inference_env(sessions)
        _print_sessions_result(sessions, env, output_format=args.format)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_validate(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.validation import (
            format_validation_report,
            validate_config_paths,
            validation_report_json,
        )

        report = validate_config_paths(
            args.configs,
            preflight=args.preflight,
            strict_preflight=args.strict_preflight,
            fail_on_unknown_preflight=args.fail_on_unknown_preflight,
            runtime_preflight=args.runtime_preflight,
        )
        if args.format == "json":
            print(validation_report_json(report))
        else:
            print(format_validation_report(report))
        if report.result == "ok":
            return 0
        if report.result == "inconclusive":
            return 2
        return 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run_start(args: argparse.Namespace) -> int:
    if args.detach:
        return _run_start_detached(args)
    launcher = None
    status = 0
    registry_dir = None
    try:
        plan = _prepare_start_registry(args.config)
        registry_dir = plan.registry_dir
        launcher = _load_start_launcher(args, plan=plan)
        if args.format == "jsonl" and hasattr(launcher, "set_event_callback"):
            launcher.set_event_callback(_print_fleet_event_jsonl)
        _mark_registry_submitting(registry_dir)
        sessions = normalize_sessions(launcher.start())
        _record_registry_ready(registry_dir, sessions)
        env = generic_inference_env(sessions)
        if args.env_file:
            write_env_file(args.env_file, _non_secret_env(env))
        _print_sessions_result(sessions, env, output_format=args.format)
        if _should_hold_start_command(launcher, exit_after_ready=args.exit_after_ready):
            _hold_launcher(launcher)
    except KeyboardInterrupt:
        status = 130
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        _record_registry_failure(registry_dir, error)
        print(f"error: {error}", file=sys.stderr)
        status = 2
    finally:
        if launcher is not None:
            with shield_termination_signals():
                try:
                    launcher.stop()
                except (OSError, RuntimeError) as error:
                    print(f"error: failed to stop inference launcher: {error}", file=sys.stderr)
                    if status == 0:
                        status = 2
    return status


def _run_start_detached(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.controller import (
            spawn_detached_controller,
            wait_for_controller_start,
            wait_for_ready_state,
        )
        from remote_inference_launcher.registry import env_text_from_state, read_registry

        plan = _prepare_start_registry(args.config, controller_policy="detached")
        launch_summary = args.launch_summary or _single_endpoint_summary_path(plan)
        controller = spawn_detached_controller(
            registry_dir=plan.registry_dir,
            config_path=args.config,
            launch_summary=launch_summary,
            overwrite_launch_summary=args.overwrite_launch_summary,
        )
        state = wait_for_controller_start(plan.registry_dir)
        if state.get("lifecycle_state") == "FAILED":
            print(f"error: detached controller failed; see {controller.log_path}", file=sys.stderr)
            return 2
        if args.wait_ready:
            ready_state = wait_for_ready_state(
                plan.registry_dir,
                timeout_seconds=float(args.ready_timeout),
            )
            if ready_state.get("lifecycle_state") not in {"READY", "LEASED"}:
                print(
                    f"error: detached run did not become ready; see {controller.log_path}",
                    file=sys.stderr,
                )
                return 2
            registry = read_registry(plan.registry_dir)
            if args.env_file:
                _write_env_text_file(args.env_file, env_text_from_state(registry["state"]))
            _print_detached_ready(registry, output_format=args.format)
            return 0
        _print_detached_started(plan, controller.log_path, output_format=args.format)
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _prepare_start_registry(
    config_path: str,
    *,
    controller_policy: str = "foreground",
    ownership_policy: str = "owned",
) -> object:
    from remote_inference_launcher.plans import build_effective_launch_plan_from_path
    from remote_inference_launcher.registry import create_run_registry

    plan = build_effective_launch_plan_from_path(
        config_path,
        controller_policy=controller_policy,
        ownership_policy=ownership_policy,
    )
    create_run_registry(plan)
    return plan


def _print_detached_started(plan: object, log_path: Path, *, output_format: str) -> None:
    payload = {
        "run_id": str(getattr(plan, "run_id", "")),
        "registry": str(getattr(plan, "registry_dir", "")),
        "status_command": f"remote-inference-launcher status {getattr(plan, 'run_id', '')}",
        "logs_command": f"remote-inference-launcher logs {getattr(plan, 'run_id', '')}",
        "controller_log": str(log_path),
    }
    if output_format in {"json", "jsonl"}:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    for name, value in payload.items():
        print(f"{name}={value}")


def _print_detached_ready(registry: object, *, output_format: str) -> None:
    from remote_inference_launcher.registry import env_text_from_state

    if output_format == "json":
        print(json_dumps(registry))
        return
    if output_format == "jsonl":
        print(json.dumps({"event": "start_ready", **dict(registry)}, default=str, sort_keys=True))
        return
    print(env_text_from_state(dict(registry)["state"]), end="")


def _write_env_text_file(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _load_start_launcher(args: argparse.Namespace, *, plan: object | None = None) -> object:
    from remote_inference_launcher.inference_config import load_inference_launcher

    launch_summary = args.launch_summary or _single_endpoint_summary_path(plan)
    if launch_summary or args.overwrite_launch_summary:
        return load_inference_launcher(
            args.config,
            launch_summary_path=launch_summary or None,
            overwrite_launch_summary=args.overwrite_launch_summary,
            run_id=str(getattr(plan, "run_id", "") or "") or None,
            endpoint_plan=_single_endpoint_plan(plan),
        )
    return load_inference_launcher(
        args.config,
        run_id=str(getattr(plan, "run_id", "") or "") or None,
        endpoint_plan=_single_endpoint_plan(plan),
    )


def _single_endpoint_summary_path(plan: object | None) -> str:
    endpoint = _single_endpoint_plan(plan)
    if endpoint is None:
        return ""
    return str(getattr(endpoint, "summary_path", "") or "")


def _single_endpoint_plan(plan: object | None) -> object | None:
    endpoints = getattr(plan, "endpoints", ())
    if not isinstance(endpoints, tuple) or len(endpoints) != 1:
        return None
    return endpoints[0]


def _mark_registry_submitting(registry_dir: object) -> None:
    from remote_inference_launcher.registry import mark_submitting

    mark_submitting(registry_dir)


def _record_registry_ready(registry_dir: object, sessions: dict[str, object]) -> None:
    from remote_inference_launcher.registry import record_ready_sessions

    record_ready_sessions(registry_dir, sessions)


def _record_registry_failure(registry_dir: object | None, error: BaseException) -> None:
    if registry_dir is None:
        return
    from remote_inference_launcher.registry import record_failure

    try:
        record_failure(registry_dir, error)
    except (OSError, RuntimeError, ValueError) as registry_error:
        print(f"error: failed to update run registry: {registry_error}", file=sys.stderr)


def _non_secret_env(env: dict[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in env.items()
        if name != "OPENAI_API_KEY" and not name.endswith("_API_KEY")
    }


def _run_bootstrap(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.vllm_bootstrap import load_vllm_bootstrapper

        result = load_vllm_bootstrapper(args.config).run()
        payload = bootstrap_result_payload(result)
        if args.format == "json":
            print(json_dumps(payload))
        else:
            _print_bootstrap_env(payload)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_slurm_vllm(args: argparse.Namespace) -> int:
    launcher: SlurmVllmLauncher | None = None
    status = 0
    try:
        config = slurm_vllm_config_from_args(args)
        launcher = SlurmVllmLauncher(config)
        session = launcher.start()
        _print_session(session)
        if args.hold:
            _hold_session(launcher)
    except KeyboardInterrupt:
        status = 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        status = 2
    finally:
        if launcher is not None:
            with shield_termination_signals():
                try:
                    launcher.stop()
                except (OSError, RuntimeError) as error:
                    print(f"error: failed to stop Slurm vLLM launcher: {error}", file=sys.stderr)
                    if status == 0:
                        status = 2
    return status


def _run_slurm_vllm_bootstrap(args: argparse.Namespace) -> int:
    try:
        config = slurm_vllm_bootstrap_config_from_args(args)
        result = SlurmVllmBootstrapper(config).run()
        _print_bootstrap_result(result)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_local_vllm_bootstrap(args: argparse.Namespace) -> int:
    try:
        config = local_vllm_bootstrap_config_from_args(args)
        result = LocalVllmBootstrapper(config).run()
        _print_vllm_bootstrap_result(result)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_ssh_vllm_bootstrap(args: argparse.Namespace) -> int:
    try:
        config = ssh_vllm_bootstrap_config_from_args(args)
        result = SshVllmBootstrapper(config).run()
        _print_vllm_bootstrap_result(result)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _hold_session(launcher: SlurmVllmLauncher) -> None:
    while True:
        job_info = launcher.job_info()
        if job_info.state not in {"RUNNING", "COMPLETING"}:
            raise RuntimeError(
                f"Remote vLLM job exited: state={job_info.state} reason={job_info.reason}"
            )
        time.sleep(launcher.config.check_interval_seconds)


def _hold_launcher(launcher: object) -> None:
    while True:
        if hasattr(launcher, "job_info"):
            job_info = launcher.job_info()
            if job_info.state not in {"RUNNING", "COMPLETING"}:
                raise RuntimeError(
                    f"Remote vLLM job exited: state={job_info.state} reason={job_info.reason}"
                )
        if hasattr(launcher, "is_running") and not launcher.is_running():
            raise RuntimeError("Inference server exited while start was keeping it alive.")
        interval = int(getattr(getattr(launcher, "config", None), "check_interval_seconds", 1))
        time.sleep(max(interval, 1))


def _should_hold_start_command(launcher: object, *, exit_after_ready: bool) -> bool:
    if exit_after_ready:
        return False
    return bool(getattr(launcher, "owns_resources", True))


def _print_sessions_result(
    sessions: dict[str, object],
    env: dict[str, str],
    *,
    output_format: str,
) -> None:
    if output_format == "jsonl":
        payload = sessions_payload(sessions)
        payload["env"] = dict(sorted(env.items()))
        print(
            json.dumps(
                {"event": "start_complete", **payload},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )
        return
    if output_format == "json":
        payload = sessions_payload(sessions)
        payload["env"] = dict(sorted(env.items()))
        print(json_dumps(payload))
        return
    for line in env_lines(env):
        print(line)


def _print_fleet_event_jsonl(event: object) -> None:
    payload = event.to_dict() if hasattr(event, "to_dict") else {"event": str(event)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str), flush=True)


def _print_bootstrap_env(payload: dict[str, object]) -> None:
    values = {
        "BOOTSTRAP_KIND": str(payload.get("kind") or ""),
        "BOOTSTRAP_TARGET": str(payload.get("target") or ""),
        "BOOTSTRAP_JOB_ID": str(payload.get("job_id") or ""),
        "BOOTSTRAP_LOGS": str(payload.get("out_dir") or ""),
        "BOOTSTRAP_VENV_PATH": str(payload.get("venv_path") or ""),
        "BOOTSTRAP_PYTHON_BIN": str(payload.get("python_bin") or ""),
        "BOOTSTRAP_MANIFEST_PATH": str(payload.get("manifest_path") or ""),
    }
    for line in env_lines({name: value for name, value in values.items() if value}):
        print(line)


def _print_session(session: object) -> None:
    print(f"OPENAI_BASE_URL={session.api_base}")
    print(f"SLURM_JOB_ID={session.job_id}")
    print(f"REMOTE_LOGS={session.logs}")
    summary_path = str(getattr(session, "summary_path", "") or "")
    if summary_path:
        print(f"INFERENCE_LAUNCH_SUMMARY_PATH={summary_path}")


def _print_bootstrap_result(result: SlurmVllmBootstrapResult) -> None:
    print(f"SLURM_JOB_ID={result.job_id}")
    print(f"REMOTE_LOGS={result.out_dir}")
    print(f"REMOTE_ENVIRONMENT={result.manifest['venv_path']}")
    print(f"REMOTE_PYTHON={result.manifest['python_bin']}")
    print(f"REMOTE_BOOTSTRAP_MANIFEST={result.manifest['manifest_path']}")


def _print_vllm_bootstrap_result(result: VllmBootstrapResult) -> None:
    prefix = "REMOTE" if result.kind == "ssh_vllm_bootstrap" else "LOCAL"
    print(f"{prefix}_LOGS={result.out_dir}")
    print(f"{prefix}_ENVIRONMENT={result.manifest['venv_path']}")
    print(f"{prefix}_PYTHON={result.manifest['python_bin']}")
    print(f"{prefix}_BOOTSTRAP_MANIFEST={result.manifest['manifest_path']}")


def _run_ssh_setup(args: argparse.Namespace) -> int:
    try:
        print(render_setup(_ssh_setup_settings_from_args(args)))
        return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _ssh_setup_settings_from_args(args: argparse.Namespace):
    presets = tuple(args.preset or ())
    if presets:
        return tuple(_ssh_setup_setting_from_preset_arg(args, preset) for preset in presets)

    return _ssh_setup_settings_from_explicit_args(args)


def _ssh_setup_setting_from_preset_arg(args: argparse.Namespace, preset: str):
    host = REMOTE_HOST_PRESETS[preset]
    if args.interactive:
        return prompt_missing_settings(
            host,
            user=args.user or "",
            identity_file=args.identity_file or "",
        )
    _require_ssh_setup_values(
        (args.user, "--user"),
        (args.identity_file, "--identity-file"),
    )
    return settings_from_preset(
        preset,
        user=args.user,
        identity_file=args.identity_file,
    )


def _ssh_setup_settings_from_explicit_args(args: argparse.Namespace):
    if args.interactive:
        return (_interactive_ssh_setup_setting_from_args(args),)

    _require_ssh_setup_values(
        (args.alias, "--alias"),
        (args.hostname, "--hostname"),
        (args.port, "--port"),
        (args.user, "--user"),
        (args.identity_file, "--identity-file"),
    )
    return (
        settings_from_values(
            alias=args.alias,
            hostname=args.hostname,
            port=args.port,
            user=args.user,
            identity_file=args.identity_file,
        ),
    )


def _interactive_ssh_setup_setting_from_args(args: argparse.Namespace):
    alias = args.alias or _prompt_cli_value("SSH host alias: ")
    hostname = args.hostname or _prompt_cli_value(f"Hostname for {alias}: ")
    port = args.port if args.port is not None else int(_prompt_cli_value(f"SSH port for {alias}: "))
    return prompt_missing_settings(
        RemoteHost(alias=alias, hostname=hostname, port=port),
        user=args.user or "",
        identity_file=args.identity_file or "",
    )


def _require_ssh_setup_values(*required_values: tuple[object, str]) -> None:
    for value, flag in required_values:
        _require_ssh_setup_value(value, flag)


def _require_ssh_setup_value(value: object, flag: str) -> None:
    if value in {"", None}:
        raise ValueError(f"{flag} is required unless --interactive supplies it.")


def _prompt_cli_value(prompt: str) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    raw_value = sys.stdin.readline()
    if raw_value == "":
        raise ValueError(f"missing input for {prompt.rstrip(': ')}.")
    value = raw_value.strip()
    if not value:
        raise ValueError(f"{prompt.rstrip(': ')} must not be empty.")
    return value


def _add_slurm_vllm_config_flags(parser: argparse.ArgumentParser) -> None:
    _add_dataclass_config_flags(parser, SlurmVllmConfig)
    parser.add_argument(
        "--launch-summary",
        dest="launch_summary_path",
        default=argparse.SUPPRESS,
        help="Write launch summary JSON to this path.",
    )


def _add_verbosity_alias_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages. Final key/value output and errors still print.",
    )
    group.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages plus remote command diagnostics.",
    )


def _add_dataclass_config_flags(
    parser: argparse.ArgumentParser,
    config_class: ConfigClass,
) -> None:
    hints = get_type_hints(config_class)
    for field in fields(config_class):
        annotation = hints[field.name]
        if field.name == "extra_args":
            parser.add_argument(
                "--extra-arg",
                dest=field.name,
                action="append",
                default=argparse.SUPPRESS,
                help="Append one passthrough argument to the vLLM serve command.",
            )
        elif annotation is bool:
            parser.add_argument(
                _flag_name(field.name),
                dest=field.name,
                action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
            )
        elif annotation is int or _is_optional_int(annotation):
            parser.add_argument(
                _flag_name(field.name),
                dest=field.name,
                type=int,
                default=argparse.SUPPRESS,
            )
        elif annotation is float or _is_optional_float(annotation):
            parser.add_argument(
                _flag_name(field.name),
                dest=field.name,
                type=float,
                default=argparse.SUPPRESS,
            )
        elif annotation is str:
            kwargs: dict[str, object] = {
                "dest": field.name,
                "default": argparse.SUPPRESS,
            }
            if field.name == "verbosity":
                kwargs["choices"] = VERBOSITY_LEVELS
            parser.add_argument(_flag_name(field.name), **kwargs)
        elif _is_structured_config_annotation(annotation):
            continue
        else:
            raise TypeError(f"Unsupported config field annotation: {annotation}")


def _config_overrides_from_args(
    args: argparse.Namespace,
    config_class: ConfigClass = SlurmVllmConfig,
) -> dict[str, object]:
    field_names = {field.name for field in fields(config_class)}
    values: dict[str, object] = {}
    for name in field_names:
        if hasattr(args, name):
            value = getattr(args, name)
            values[name] = tuple(value) if name == "extra_args" else value
    return values


def _apply_cli_verbosity_aliases(
    config: ConfigObject,
    args: argparse.Namespace,
) -> ConfigObject:
    quiet = getattr(args, "quiet", False)
    verbose = getattr(args, "verbose", False)
    if not quiet and not verbose:
        return config
    if hasattr(args, "verbosity"):
        raise ValueError("Use either --verbosity or --quiet/--verbose, not both.")
    return replace(config, verbosity="quiet" if quiet else "verbose")


def _flag_name(field_name: str) -> str:
    return "--" + field_name.replace("_", "-")


def _is_optional_int(annotation: object) -> bool:
    return set(get_args(annotation)) == {int, type(None)} and get_origin(annotation) is not None


def _is_optional_float(annotation: object) -> bool:
    return set(get_args(annotation)) == {float, type(None)} and get_origin(annotation) is not None


def _is_structured_config_annotation(annotation: object) -> bool:
    origin = get_origin(annotation)
    if origin is tuple and get_args(annotation):
        return True
    return bool(hasattr(annotation, "__dataclass_fields__"))
