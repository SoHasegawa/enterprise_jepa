"""Durable local SSH tunnel records and reconnect helpers."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from remote_inference_launcher.ports import reserve_local_port
from remote_inference_launcher.registry import utc_now, write_json_atomic
from remote_inference_launcher.ssh import ssh_options

TUNNEL_SCHEMA_VERSION = "ril-tunnel/v1"
DEFAULT_POOL_ROOT = ".remote-inference-launcher/pool"


def reconnect_slurm_tunnel(
    endpoint: Mapping[str, object],
    *,
    pool_root: str | Path = DEFAULT_POOL_ROOT,
    preferred_local_port: int | None = None,
) -> dict[str, object]:
    """Start a new local SSH tunnel for a Slurm endpoint without remote teardown."""

    if str(endpoint.get("backend_kind", "") or "") != "slurm_vllm":
        raise RuntimeError("Tunnel reconnect currently supports only slurm_vllm endpoints.")
    endpoint_id = str(endpoint.get("endpoint_id", "") or "")
    ssh_target = str(endpoint.get("ssh_target", "") or "")
    node = str(endpoint.get("node", "") or "")
    remote_port = _required_int(endpoint.get("remote_port"), "remote_port")
    bind_host = str(endpoint.get("local_bind_host", "") or "127.0.0.1")
    if not endpoint_id:
        raise RuntimeError("Tunnel reconnect requires endpoint_id.")
    if not ssh_target:
        raise RuntimeError("Tunnel reconnect requires ssh_target.")
    if not node:
        raise RuntimeError("Tunnel reconnect requires Slurm node.")
    old_record = read_tunnel_record(endpoint_id, pool_root=pool_root)
    if old_record:
        stop_owned_tunnel(old_record)
    selected_port = _select_local_port(
        endpoint,
        bind_host=bind_host,
        preferred_local_port=preferred_local_port,
    )
    process = _start_ssh_tunnel(
        ssh_target=ssh_target,
        bind_host=bind_host,
        local_port=selected_port,
        node=node,
        remote_port=remote_port,
    )
    api_base = _api_base(bind_host, selected_port)
    record = {
        "schema_version": TUNNEL_SCHEMA_VERSION,
        "endpoint_id": endpoint_id,
        "ssh_target": ssh_target,
        "local_bind_host": bind_host,
        "local_port": selected_port,
        "remote_node": node,
        "remote_port": remote_port,
        "pid": process.pid,
        "started_at": utc_now(),
        "api_base": api_base,
        "command_hash": _command_hash(
            ssh_target=ssh_target,
            bind_host=bind_host,
            local_port=selected_port,
            node=node,
            remote_port=remote_port,
        ),
    }
    write_tunnel_record(record, pool_root=pool_root)
    return record


def read_tunnel_record(
    endpoint_id: str,
    *,
    pool_root: str | Path = DEFAULT_POOL_ROOT,
) -> dict[str, object]:
    """Read a tunnel record if one exists."""

    path = tunnel_record_path(endpoint_id, pool_root=pool_root)
    if not path.exists():
        return {}
    from remote_inference_launcher.registry import read_json

    payload = read_json(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def write_tunnel_record(
    record: Mapping[str, object],
    *,
    pool_root: str | Path = DEFAULT_POOL_ROOT,
) -> None:
    """Write a durable tunnel record."""

    endpoint_id = str(record.get("endpoint_id", "") or "")
    if not endpoint_id:
        raise ValueError("Tunnel record requires endpoint_id.")
    write_json_atomic(tunnel_record_path(endpoint_id, pool_root=pool_root), record)


def tunnel_record_path(endpoint_id: str, *, pool_root: str | Path = DEFAULT_POOL_ROOT) -> Path:
    """Return the tunnel record path for an endpoint id."""

    digest = hashlib.sha256(endpoint_id.encode("utf-8")).hexdigest()[:16]
    return Path(pool_root).expanduser() / "tunnels" / f"{digest}.json"


def stop_owned_tunnel(record: Mapping[str, object]) -> None:
    """Terminate a recorded local tunnel process when it is still alive."""

    pid = _optional_int(record.get("pid"))
    if pid is None or pid == os.getpid() or not _pid_alive(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    _wait_for_exit(pid, timeout_seconds=5.0)


def _select_local_port(
    endpoint: Mapping[str, object],
    *,
    bind_host: str,
    preferred_local_port: int | None,
) -> int:
    desired = preferred_local_port or _optional_int(endpoint.get("local_port"))
    strategy = str(endpoint.get("local_port_strategy", "") or "")
    try:
        reservation = reserve_local_port(bind_host, desired)
    except OSError:
        if strategy == "explicit" or preferred_local_port is not None:
            raise
        reservation = reserve_local_port(bind_host, None)
    try:
        return reservation.port
    finally:
        reservation.close()


def _start_ssh_tunnel(
    *,
    ssh_target: str,
    bind_host: str,
    local_port: int,
    node: str,
    remote_port: int,
) -> subprocess.Popen[bytes]:
    command = [
        "ssh",
        *ssh_options(),
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ExitOnForwardFailure=yes",
        "-N",
        "-L",
        f"{bind_host}:{local_port}:{node}:{remote_port}",
        ssh_target,
    ]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    time.sleep(1)
    if process.poll() is not None:
        raise RuntimeError("SSH tunnel exited before reconnect health check.")
    return process


def _api_base(bind_host: str, local_port: int) -> str:
    host = "127.0.0.1" if bind_host in {"", "*", "0.0.0.0"} else bind_host
    return f"http://{host}:{local_port}/v1"


def _command_hash(
    *,
    ssh_target: str,
    bind_host: str,
    local_port: int,
    node: str,
    remote_port: int,
) -> str:
    text = f"{ssh_target}|{bind_host}|{local_port}|{node}|{remote_port}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _required_int(value: object, label: str) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise RuntimeError(f"Tunnel reconnect requires {label}.")
    return parsed


def _optional_int(value: object) -> int | None:
    if value in {"", None}:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_exit(pid: int, *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    return not _pid_alive(pid)
