"""CLI wiring for endpoint pool operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from remote_inference_launcher.session_env import json_dumps


def add_pool_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """Add pool commands to the top-level parser."""

    pool = subparsers.add_parser("pool", help="manage reusable endpoint pools")
    pool_subparsers = pool.add_subparsers(dest="pool_command")

    status = pool_subparsers.add_parser("status", help="show endpoint pool status")
    _add_pool_roots(status)
    status.add_argument(
        "--health-depth",
        choices=("cached", "scheduler", "light", "models", "full"),
        default="cached",
    )
    status.add_argument("--leases-only", action="store_true")
    status.add_argument("--format", choices=("text", "json"), default="text")
    status.set_defaults(handler=_run_pool_status)

    acquire = pool_subparsers.add_parser("acquire", help="claim a healthy free endpoint")
    _add_pool_roots(acquire)
    acquire.add_argument("--model", default="")
    acquire.add_argument("--min-context", type=int)
    acquire.add_argument("--min-remaining", default="")
    acquire.add_argument("--owner", required=True)
    acquire.add_argument("--shard", default="")
    acquire.add_argument("--assignment-ttl", default="")
    acquire.add_argument(
        "--health-depth",
        choices=("cached", "local", "models", "full"),
        default="full",
    )
    acquire.add_argument("--format", choices=("json",), default="json")
    acquire.set_defaults(handler=_run_pool_acquire)

    acquire_batch = pool_subparsers.add_parser(
        "acquire-batch",
        help="claim a batch of healthy free endpoints for deterministic shards",
    )
    _add_pool_roots(acquire_batch)
    acquire_batch.add_argument("--model", default="")
    acquire_batch.add_argument("--min-context", type=int)
    acquire_batch.add_argument("--min-remaining", default="")
    acquire_batch.add_argument("--owner", required=True)
    acquire_batch.add_argument("--count", type=int, required=True)
    acquire_batch.add_argument("--shard-file", default="")
    acquire_batch.add_argument("--assignment-ttl", default="")
    acquire_batch.add_argument("--partial", action="store_true")
    acquire_batch.add_argument(
        "--health-depth",
        choices=("cached", "local", "models", "full"),
        default="full",
    )
    acquire_batch.add_argument("--format", choices=("json",), default="json")
    acquire_batch.set_defaults(handler=_run_pool_acquire_batch)

    release = pool_subparsers.add_parser("release", help="release an endpoint assignment")
    _add_pool_roots(release)
    release.add_argument("assignment_id", nargs="?")
    release.add_argument("--endpoint-id", default="")
    release.add_argument("--owner", default="")
    release.add_argument("--format", choices=("text", "json"), default="text")
    release.set_defaults(handler=_run_pool_release)

    release_batch = pool_subparsers.add_parser(
        "release-batch", help="release endpoint assignments by batch"
    )
    _add_pool_roots(release_batch)
    release_batch.add_argument("batch_id", nargs="?")
    release_batch.add_argument("--assignment", action="append", default=[])
    release_batch.add_argument("--owner", default="")
    release_batch.add_argument("--format", choices=("text", "json"), default="text")
    release_batch.set_defaults(handler=_run_pool_release_batch)

    health = pool_subparsers.add_parser("health", help="check one endpoint or assignment")
    _add_pool_roots(health)
    health.add_argument("--assignment", default="")
    health.add_argument("--endpoint-id", default="")
    health.add_argument(
        "--depth",
        choices=("cached", "local", "models", "full"),
        default="cached",
    )
    health.add_argument("--format", choices=("json",), default="json")
    health.set_defaults(handler=_run_pool_health)

    manifest = pool_subparsers.add_parser("manifest", help="emit endpoint pool manifest")
    _add_pool_roots(manifest)
    manifest.add_argument("--assignment", default="")
    manifest.add_argument("--endpoint-id", default="")
    manifest.add_argument("--available", action="store_true")
    manifest.add_argument("--format", choices=("json",), default="json")
    manifest.set_defaults(handler=_run_pool_manifest)

    reconnect = pool_subparsers.add_parser("reconnect", help="rebuild a leased Slurm tunnel")
    _add_pool_roots(reconnect)
    reconnect.add_argument("--assignment", default="")
    reconnect.add_argument("--endpoint-id", default="")
    reconnect.add_argument("--local-port", type=int)
    reconnect.add_argument("--format", choices=("json",), default="json")
    reconnect.set_defaults(handler=_run_pool_reconnect)

    recommend = pool_subparsers.add_parser(
        "recommend-partitions",
        help="recommend compatible Slurm partitions from cheap scheduler inventory",
    )
    recommend.add_argument("--config", action="append", required=True)
    recommend.add_argument("--format", choices=("text", "json"), default="text")
    recommend.set_defaults(handler=_run_pool_recommend_partitions)


def _add_pool_roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lease-root", default=".remote-inference-launcher/leases")
    parser.add_argument("--registry-root", default=".remote-inference-launcher/runs")
    parser.add_argument("--pool-root", default=".remote-inference-launcher/pool")


def _run_pool_status(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.pool import build_pool_status

        payload = build_pool_status(
            lease_root=args.lease_root,
            registry_root=args.registry_root,
            pool_root=args.pool_root,
            health_depth=args.health_depth,
            include_registry_endpoints=not args.leases_only,
        )
        print(json_dumps(payload) if args.format == "json" else _format_pool_status(payload))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_pool_acquire(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.pool import acquire_endpoint

        payload = acquire_endpoint(
            model=args.model,
            min_context=args.min_context,
            min_remaining=args.min_remaining,
            owner=args.owner,
            shard_id=args.shard,
            assignment_ttl=args.assignment_ttl,
            health_depth=args.health_depth,
            lease_root=args.lease_root,
            registry_root=args.registry_root,
            pool_root=args.pool_root,
        )
        print(json_dumps(payload))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_pool_acquire_batch(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.pool import acquire_batch

        payload = acquire_batch(
            model=args.model,
            min_context=args.min_context,
            min_remaining=args.min_remaining,
            owner=args.owner,
            count=args.count,
            shard_ids=_load_shard_ids(args.shard_file),
            assignment_ttl=args.assignment_ttl,
            health_depth=args.health_depth,
            partial=args.partial,
            lease_root=args.lease_root,
            registry_root=args.registry_root,
            pool_root=args.pool_root,
        )
        print(json_dumps(payload))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_pool_release(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.pool import release_assignment

        payload = release_assignment(
            assignment_id=args.assignment_id or "",
            endpoint_id=args.endpoint_id,
            owner=args.owner,
            pool_root=args.pool_root,
        )
        print(
            json_dumps(payload) if args.format == "json" else f"released={payload['assignment_id']}"
        )
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_pool_release_batch(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.pool import release_batch

        payload = release_batch(
            batch_id=args.batch_id or "",
            assignment_ids=args.assignment,
            owner=args.owner,
            pool_root=args.pool_root,
        )
        if args.format == "json":
            print(json_dumps(payload))
        else:
            print(
                f"batch_id={payload.get('batch_id', '')} "
                f"released_count={payload.get('released_count', 0)}"
            )
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_pool_health(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.pool import pool_health_signal

        payload = pool_health_signal(
            endpoint_id=args.endpoint_id,
            assignment_id=args.assignment,
            depth=args.depth,
            lease_root=args.lease_root,
            registry_root=args.registry_root,
            pool_root=args.pool_root,
        )
        print(json_dumps(payload))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_pool_manifest(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.pool import pool_manifest

        payload = pool_manifest(
            endpoint_id=args.endpoint_id,
            assignment_id=args.assignment,
            available=args.available,
            lease_root=args.lease_root,
            registry_root=args.registry_root,
            pool_root=args.pool_root,
        )
        print(json_dumps(payload))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_pool_reconnect(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.pool import reconnect_endpoint

        payload = reconnect_endpoint(
            endpoint_id=args.endpoint_id,
            assignment_id=args.assignment,
            preferred_local_port=args.local_port,
            lease_root=args.lease_root,
            registry_root=args.registry_root,
            pool_root=args.pool_root,
        )
        print(json_dumps(payload))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_pool_recommend_partitions(args: argparse.Namespace) -> int:
    try:
        from remote_inference_launcher.scheduler_status import partition_recommendations

        requests = _partition_requests(args.config)
        payload = partition_recommendations(tuple(requests))
        print(json_dumps(payload) if args.format == "json" else _format_recommendations(payload))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _partition_requests(paths: list[str]) -> list[dict[str, object]]:
    from remote_inference_launcher.plans import build_effective_launch_plan_from_path

    requests: list[dict[str, object]] = []
    for path in paths:
        plan = build_effective_launch_plan_from_path(Path(path))
        for endpoint in plan.endpoints:
            if endpoint.backend_kind != "slurm_vllm":
                continue
            slurm = dict(endpoint.slurm)
            requests.append(
                {
                    "source": endpoint.raw_config_source,
                    "endpoint_name": endpoint.name,
                    "ssh_target": endpoint.ssh_target,
                    "partition": slurm.get("partition", ""),
                    "walltime": slurm.get("walltime", ""),
                    "num_gpus": slurm.get("num_gpus"),
                    "nodes": slurm.get("nodes"),
                }
            )
    return requests


def _load_shard_ids(path: str) -> list[str]:
    if not path:
        return []
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict):
        for key in ("shards", "shard_ids", "tasks"):
            value = payload.get(key)
            if isinstance(value, list):
                return [str(_shard_id(item)) for item in value]
    raise ValueError(f"Unsupported shard file format: {path}")


def _shard_id(item: object) -> object:
    if isinstance(item, dict):
        for key in ("shard_id", "id", "name"):
            if item.get(key) not in {None, ""}:
                return item[key]
    return item


def _format_pool_status(payload: dict[str, object]) -> str:
    summary = payload.get("summary", {})
    lines = [f"generated_at={payload.get('generated_at', '')}"]
    if isinstance(summary, dict):
        lines.extend(f"{name}={summary[name]}" for name in sorted(summary))
    endpoints = payload.get("endpoints", ())
    if isinstance(endpoints, list):
        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                continue
            health = endpoint.get("health", {})
            health_status = health.get("status", "") if isinstance(health, dict) else ""
            reason = health.get("reason_code", "") if isinstance(health, dict) else ""
            capabilities = endpoint.get("capabilities", {})
            capability_model = (
                capabilities.get("served_model_name", "") if isinstance(capabilities, dict) else ""
            )
            acquirable = endpoint.get("acquirable", "")
            not_acquirable_reason = endpoint.get("not_acquirable_reason", "")
            lines.append(
                "endpoint="
                f"{endpoint.get('endpoint_id', '')} "
                f"state={endpoint.get('lifecycle_state', '')} "
                f"lease={endpoint.get('lease_status', '')} "
                f"assignment={endpoint.get('assignment_state', '')} "
                f"health={health_status} "
                f"reason={reason} "
                f"acquirable={acquirable} "
                f"not_acquirable_reason={not_acquirable_reason} "
                f"model={endpoint.get('served_model_name', '') or capability_model}"
            )
    return "\n".join(lines)


def _format_recommendations(payload: dict[str, object]) -> str:
    lines: list[str] = []
    targets = payload.get("targets", {})
    if not isinstance(targets, dict):
        return ""
    for target, target_payload in sorted(targets.items()):
        lines.append(f"target={target}")
        if not isinstance(target_payload, dict):
            continue
        for item in target_payload.get("partitions", []):
            if not isinstance(item, dict):
                continue
            request = item.get("request", {})
            endpoint = request.get("endpoint_name", "") if isinstance(request, dict) else ""
            compatible = item.get("compatible", [])
            labels = [
                str(candidate.get("partition", ""))
                for candidate in compatible
                if isinstance(candidate, dict)
            ]
            lines.append(f"endpoint={endpoint} compatible={','.join(labels)}")
    return "\n".join(lines)
