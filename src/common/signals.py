"""Process-signal helpers shared by the CLI."""

from __future__ import annotations

import signal
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def translate_termination_signals() -> Iterator[None]:
    """Convert SIGINT/SIGTERM into KeyboardInterrupt for managed cleanup paths."""

    previous_handlers: dict[int, signal.Handlers] = {}

    def handle_signal(signum: int, _frame: object) -> None:
        signame = signal.Signals(signum).name
        raise KeyboardInterrupt(f"received {signame}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)
    except ValueError:
        yield
        return

    try:
        yield
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)
