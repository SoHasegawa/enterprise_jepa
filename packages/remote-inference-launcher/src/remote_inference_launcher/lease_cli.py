"""CLI wiring for reusable endpoint leases."""

from __future__ import annotations

import argparse
import sys

from remote_inference_launcher.session_env import json_dumps, normalize_sessions


def add_lease_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """Add lease commands to the top-level parser."""

    lease = subparsers.add_parser("lease", help="manage reusable endpoint leases")
    lease_subparsers = lease.add_subparsers(dest="lease_command")

    start = lease_subparsers.add_parser("start", help="start a reusable endpoint lease")
    start.add_argument("--config", required=True, help="Inference launcher YAML config path.")
    start.add_argument("--ttl", required=True, help="Lease attachment TTL, such as 12h.")
    start.add_argument(
        "--ttl-start",
        choices=("created_at", "ready_at"),
        default="created_at",
        help="Start TTL at lease creation or when the endpoint becomes ready.",
    )
    start.add_argument("--cleanup-on-expiry", action="store_true")
    start.add_argument("--detach", action="store_true")
    start.add_argument("--wait-ready", action="store_true")
    start.add_argument(
        "--ready-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for a detached lease to become ready when --wait-ready is set.",
    )
    start.add_argument("--format", choices=("text", "json"), default="text")
    start.set_defaults(handler=_run_lease_start)

    status = lease_subparsers.add_parser("status", help="show lease status")
    status.add_argument("lease_id")
    status.add_argument("--format", choices=("text", "json"), default="text")
    status.set_defaults(handler=_run_lease_status)

    env = lease_subparsers.add_parser("env", help="print non-secret lease env exports")
    env.add_argument("lease_id")
    env.add_argument("--endpoint", help="Endpoint name to export.")
    env.set_defaults(handler=_run_lease_env)

    attach = lease_subparsers.add_parser("attach", help="attach to an active lease")
    attach.add_argument("lease_id")
    attach.add_argument("--expect-config", help="Config whose semantic hash must match.")
    attach.add_argument("--allow-expired", action="store_true")
    attach.add_argument("--format", choices=("json",), default="json")
    attach.set_defaults(handler=_run_lease_attach)

    recover = lease_subparsers.add_parser(
        "recover",
        help="recover a lease whose parent process timed out before readiness",
    )
    recover.add_argument("lease_id")
    recover.add_argument("--expect-config", help="Config whose semantic hash must match.")
    recover.add_argument("--expected-model", default="")
    recover.add_argument(
        "--health-depth",
        choices=("cached", "local", "models", "full"),
        default="models",
    )
    recover.add_argument("--format", choices=("text", "json"), default="text")
    recover.set_defaults(handler=_run_lease_recover)

    stop = lease_subparsers.add_parser("stop", help="stop a lease")
    stop.add_argument("lease_id")
    stop.add_argument("--force", action="store_true")
    stop.add_argument(
        "--dry-run", action="store_true", help="Print planned cleanup without running it."
    )
    stop.set_defaults(handler=_run_lease_stop)


def _run_lease_start(args: argparse.Namespace) -> int:
    if args.detach:
        return _run_lease_start_detached(args)
    launcher = None
    lease_record: dict[str, object] | None = None
    try:
        from remote_inference_launcher.inference_config import load_inference_launcher
        from remote_inference_launcher.leases import (
            activate_lease,
            create_pending_lease,
        )
        from remote_inference_launcher.plans import build_effective_launch_plan_from_path
        from remote_inference_launcher.registry import (
            create_run_registry,
            mark_submitting,
            record_ready_sessions,
        )

        plan = build_effective_launch_plan_from_path(args.config, ownership_policy="lease")
        create_run_registry(plan)
        lease_record = create_pending_lease(
            plan,
            ttl=args.ttl,
            ttl_start=args.ttl_start,
            cleanup_on_expiry=args.cleanup_on_expiry,
        )
        launcher = load_inference_launcher(
            args.config,
            launch_summary_path=_single_endpoint_summary_path(plan) or None,
            run_id=plan.run_id,
            endpoint_plan=_single_endpoint_plan(plan),
        )
        mark_submitting(plan.registry_dir)
        sessions = normalize_sessions(launcher.start())
        record_ready_sessions(plan.registry_dir, sessions)
        lease_record = activate_lease(str(lease_record["lease_id"]))
        _print_lease_start_result(lease_record, output_format=args.format)
        _hold_lease_launcher_if_needed(launcher)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        _record_lease_start_failure(lease_record, error)
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        _stop_launcher(launcher)


def _run_lease_start_detached(args: argparse.Namespace) -> int:
    lease_record: dict[str, object] | None = None
    try:
        from remote_inference_launcher.controller import (
            spawn_detached_controller,
            wait_for_controller_start,
            wait_for_ready_state,
        )
        from remote_inference_launcher.leases import activate_lease, create_pending_lease
        from remote_inference_launcher.plans import build_effective_launch_plan_from_path
        from remote_inference_launcher.registry import create_run_registry

        plan = build_effective_launch_plan_from_path(
            args.config,
            controller_policy="detached",
            ownership_policy="lease",
        )
        create_run_registry(plan)
        lease_record = create_pending_lease(
            plan,
            ttl=args.ttl,
            ttl_start=args.ttl_start,
            cleanup_on_expiry=args.cleanup_on_expiry,
        )
        controller = spawn_detached_controller(
            registry_dir=plan.registry_dir,
            config_path=args.config,
            launch_summary=_single_endpoint_summary_path(plan) or None,
        )
        state = wait_for_controller_start(plan.registry_dir)
        if state.get("lifecycle_state") == "FAILED":
            print(
                f"error: detached lease controller failed; see {controller.log_path}",
                file=sys.stderr,
            )
            return 2
        if args.wait_ready:
            try:
                ready_state = wait_for_ready_state(
                    plan.registry_dir,
                    timeout_seconds=args.ready_timeout,
                )
            except TimeoutError as error:
                from remote_inference_launcher.leases import record_lease_ready_timeout

                lease_record = record_lease_ready_timeout(lease_record, error)
                print(
                    "error: detached lease did not become ready within "
                    f"{args.ready_timeout:g}s; run lease recover {lease_record['lease_id']} "
                    f"after the endpoint becomes healthy; see {controller.log_path}",
                    file=sys.stderr,
                )
                return 2
            if ready_state.get("lifecycle_state") not in {"READY", "LEASED"}:
                print(
                    f"error: detached lease did not become ready; see {controller.log_path}",
                    file=sys.stderr,
                )
                return 2
            lease_record = activate_lease(str(lease_record["lease_id"]))
        _print_lease_start_result(lease_record, output_format=args.format)
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        _record_lease_start_failure(lease_record, error)
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_lease_status(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.leases import format_lease_status, read_lease

        record = read_lease(args.lease_id)
        print(json_dumps(record) if args.format == "json" else format_lease_status(record))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_lease_env(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.leases import lease_env_text, read_lease

        record = read_lease(args.lease_id)
        print(lease_env_text(record, endpoint_name=args.endpoint or ""), end="")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_lease_attach(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.leases import attach_lease

        handoff = attach_lease(
            args.lease_id,
            expect_config=args.expect_config,
            allow_expired=args.allow_expired,
        )
        print(json_dumps(handoff))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_lease_recover(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.leases import format_lease_status, recover_lease

        payload = recover_lease(
            args.lease_id,
            expect_config=args.expect_config,
            expected_model=args.expected_model,
            health_depth=args.health_depth,
        )
        if args.format == "json":
            print(json_dumps(payload))
        else:
            print(format_lease_status(payload["lease"]))
            print(f"handoff_schema={payload['handoff']['schema_version']}")
            print(f"identity_key={payload['handoff']['identity_key']}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_lease_stop(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.leases import format_lease_stop_plan, stop_lease

        if args.dry_run:
            print(format_lease_stop_plan(args.lease_id))
            return 0
        return stop_lease(args.lease_id, force=args.force)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _print_lease_start_result(record: dict[str, object], *, output_format: str) -> None:
    if output_format == "json":
        print(json_dumps(record))
        return
    from remote_inference_launcher.leases import format_lease_status

    print(format_lease_status(record))
    print(f"env_path={record['registry_dir']}/env.sh")
    endpoints = record.get("endpoints", {})
    if isinstance(endpoints, dict):
        first = next(iter(endpoints.values()), {})
        if isinstance(first, dict):
            print(f"summary_path={first.get('summary_path', '')}")


def _single_endpoint_summary_path(plan: object) -> str:
    endpoint = _single_endpoint_plan(plan)
    if endpoint is None:
        return ""
    return str(getattr(endpoint, "summary_path", "") or "")


def _single_endpoint_plan(plan: object) -> object | None:
    endpoints = getattr(plan, "endpoints", ())
    if not isinstance(endpoints, tuple) or len(endpoints) != 1:
        return None
    return endpoints[0]


def _record_lease_start_failure(
    lease_record: dict[str, object] | None,
    error: BaseException,
) -> None:
    if not lease_record:
        return
    from remote_inference_launcher.leases import write_lease
    from remote_inference_launcher.registry import record_failure

    failed = {**lease_record, "status": "failed"}
    write_lease(failed, lease_root=".remote-inference-launcher/leases")
    record_failure(str(lease_record["registry_dir"]), error)


def _hold_lease_launcher_if_needed(launcher: object) -> None:
    from remote_inference_launcher.cli import _hold_launcher, _should_hold_start_command

    if _should_hold_start_command(launcher, exit_after_ready=False):
        _hold_launcher(launcher)


def _stop_launcher(launcher: object | None) -> None:
    if launcher is None:
        return
    from remote_inference_launcher.termination import shield_termination_signals

    with shield_termination_signals():
        try:
            launcher.stop()
        except (OSError, RuntimeError) as error:
            print(f"error: failed to stop inference launcher: {error}", file=sys.stderr)
