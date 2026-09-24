"""Local port utilities."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from types import TracebackType


def find_available_port() -> int:
    """Ask the OS for an available local TCP port."""

    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


@dataclass
class PortReservation:
    """A local TCP port reservation held by an open socket."""

    host: str
    port: int
    socket: socket.socket

    def close(self) -> None:
        """Release the reservation."""

        self.socket.close()

    def __enter__(self) -> PortReservation:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()


def reserve_local_port(host: str = "127.0.0.1", port: int | None = None) -> PortReservation:
    """Bind a local port and keep it reserved until the caller closes it."""

    bind_host = "" if host in {"*", "0.0.0.0"} else host
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((bind_host, 0 if port is None else port))
    except OSError:
        sock.close()
        raise
    return PortReservation(host=host, port=int(sock.getsockname()[1]), socket=sock)
