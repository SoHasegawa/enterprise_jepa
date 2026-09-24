"""CLI wiring for run registry commands."""

from __future__ import annotations

import argparse
import sys

from remote_inference_launcher.session_env import json_dumps


def add_registry_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """Add run registry commands to the top-level parser."""

    status = subparsers.add_parser(
        "status",
        help="show a recorded inference launcher run registry",
    )
    status.add_argument("run", help="Run ID, registry directory, or summary path.")
    status.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    status.set_defaults(handler=_run_status)

    env = subparsers.add_parser(
        "env",
        help="print non-secret shell exports from a run registry",
    )
    env.add_argument("run", help="Run ID, registry directory, or summary path.")
    env.add_argument("--endpoint", help="Endpoint name to export.")
    env.set_defaults(handler=_run_env)

    cleanup = subparsers.add_parser(
        "cleanup-command",
        help="print recorded cleanup commands for a run registry",
    )
    cleanup.add_argument("run", help="Run ID, registry directory, or summary path.")
    cleanup.set_defaults(handler=_run_cleanup_command)

    stop = subparsers.add_parser(
        "stop",
        help="run recorded cleanup commands for a run registry",
    )
    stop.add_argument("run", help="Run ID, registry directory, or summary path.")
    stop.add_argument(
        "--force",
        action="store_true",
        help="Run cleanup even for inconsistent state.",
    )
    stop.set_defaults(handler=_run_stop)

    logs = subparsers.add_parser(
        "logs",
        help="show controller and endpoint log paths for a run registry",
    )
    logs.add_argument("run", help="Run ID, registry directory, or summary path.")
    logs.add_argument("--tail", type=int, default=80, help="Controller log lines to print.")
    logs.add_argument(
        "--paths-only",
        action="store_true",
        help="Only print known log paths.",
    )
    logs.set_defaults(handler=_run_logs)


def _run_status(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.registry import format_status, read_registry

        registry = read_registry(args.run)
        if args.format == "json":
            print(json_dumps(registry))
        else:
            print(format_status(registry))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_env(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.registry import env_text_from_state, read_registry

        registry = read_registry(args.run)
        print(
            env_text_from_state(
                registry["state"],
                endpoint_name=args.endpoint or "",
            ),
            end="",
        )
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_cleanup_command(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.registry import (
            cleanup_commands_from_state,
            read_registry,
            resolve_registry_path,
        )

        registry = read_registry(args.run)
        registry_dir = resolve_registry_path(args.run)
        for command in cleanup_commands_from_state(registry["state"], registry_dir=registry_dir):
            print(command)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_stop(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.registry import resolve_registry_path, run_stop

        registry_dir = resolve_registry_path(args.run)
        return run_stop(registry_dir, force=args.force)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_logs(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.registry import format_logs, read_registry

        registry = read_registry(args.run)
        print(format_logs(registry, tail=args.tail, paths_only=args.paths_only), end="")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
