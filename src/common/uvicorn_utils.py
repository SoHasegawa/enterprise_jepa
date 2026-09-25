from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import uvicorn


def reserve_tcp_listener(host: str, port: int) -> tuple[socket.socket, int]:
    """Reserve a listening socket for the given host/port up front."""
    last_error: OSError | None = None
    for family, socktype, proto, _, sockaddr in socket.getaddrinfo(
        host,
        port,
        type=socket.SOCK_STREAM,
        flags=socket.AI_PASSIVE,
    ):
        if family not in {socket.AF_INET, socket.AF_INET6}:
            continue

        sock = socket.socket(family, socktype, proto)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6 and hasattr(socket, "IPV6_V6ONLY"):
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind(sockaddr)
            sock.listen(socket.SOMAXCONN)
            return sock, int(sock.getsockname()[1])
        except OSError as exc:
            sock.close()
            last_error = exc

    if last_error is not None:
        raise last_error
    raise OSError(f"Failed to reserve TCP listener for host={host!r} port={port!r}")


def write_port_file(port_file: Path | None, port: int) -> None:
    """Write the port actually bound to a file."""
    if port_file is None:
        return
    port_file.parent.mkdir(parents=True, exist_ok=True)
    port_file.write_text(f"{port}\n", encoding="utf-8")


def run_uvicorn_with_socket(
    app: Any,
    *,
    host: str,
    port: int,
    listener: socket.socket,
) -> None:
    """Start uvicorn on the pre-reserved socket."""
    config = uvicorn.Config(app, host=host, port=port)
    uvicorn.Server(config).run(sockets=[listener])
