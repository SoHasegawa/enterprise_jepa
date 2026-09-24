"""Managed detached controller for inference launcher runs."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread

from remote_inference_launcher.registry import (
    append_event,
    record_failure,
    record_ready_sessions,
    update_controller_heartbeat,
    update_controller_pid,
    update_state,
)
from remote_inference_launcher.session_env import normalize_sessions

HEARTBEAT_INTERVAL_SECONDS = 10.0


@dataclass(frozen=True)
class DetachedController:
    """Details for a spawned detached controller process."""

    pid: int
    log_path: Path


def spawn_detached_controller(
    *,
    registry_dir: str | Path,
    config_path: str | Path,
    launch_summary: str | None = None,
    overwrite_launch_summary: bool = False,
) -> DetachedController:
    """Start a controller process in a new session."""

    registry_path = Path(registry_dir)
    log_path = registry_path / "controller.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "remote_inference_launcher.controller",
        "--registry-dir",
        str(registry_path),
        "--config",
        str(config_path),
    ]
    if launch_summary:
        command.extend(["--launch-summary", launch_summary])
    if overwrite_launch_summary:
        command.append("--overwrite-launch-summary")
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    update_controller_pid(registry_path, process.pid)
    return DetachedController(pid=process.pid, log_path=log_path)


def wait_for_controller_start(
    registry_dir: str | Path,
    *,
    timeout_seconds: float = 10.0,
    poll_seconds: float = 0.1,
) -> dict[str, object]:
    """Wait until the controller records startup progress or failure."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = _read_state(registry_dir)
        lifecycle = str(state.get("lifecycle_state", "") or "")
        controller = state.get("controller", {})
        heartbeat_at = controller.get("heartbeat_at") if isinstance(controller, dict) else None
        if lifecycle not in {"", "PLANNED"} or heartbeat_at:
            return state
        time.sleep(poll_seconds)
    raise TimeoutError(f"Controller did not start within {timeout_seconds:g} seconds.")


def wait_for_ready_state(
    registry_dir: str | Path,
    *,
    timeout_seconds: float,
    poll_seconds: float = 0.5,
) -> dict[str, object]:
    """Wait until a detached run reaches ready or terminal failure."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = _read_state(registry_dir)
        lifecycle = str(state.get("lifecycle_state", "") or "")
        if lifecycle in {"READY", "LEASED", "FAILED", "ORPHANED", "STOPPED"}:
            return state
        time.sleep(poll_seconds)
    raise TimeoutError(f"Run did not become ready within {timeout_seconds:g} seconds.")


def run_controller(
    *,
    registry_dir: str | Path,
    config_path: str | Path,
    launch_summary: str | None = None,
    overwrite_launch_summary: bool = False,
) -> int:
    """Run the detached controller loop in the current process."""

    from remote_inference_launcher.inference_config import load_inference_launcher

    registry_path = Path(registry_dir)
    stop_requested = False
    launcher = None

    def request_stop(signum: int, _frame: object) -> None:
        del signum, _frame
        nonlocal stop_requested
        stop_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    update_controller_pid(registry_path, os.getpid())
    append_event(
        registry_path,
        event="controller_started",
        level="info",
        payload={"pid": os.getpid()},
    )
    try:
        with _controller_heartbeat(registry_path):
            update_state(registry_path, lifecycle_state="SUBMITTING")
            state = _read_state(registry_path)
            launcher = load_inference_launcher(
                config_path,
                launch_summary_path=launch_summary,
                overwrite_launch_summary=overwrite_launch_summary,
                run_id=str(state.get("run_id", "") or "") or None,
                endpoint_plan=_single_endpoint_plan_payload(registry_path),
            )
            append_event(
                registry_path,
                event="launcher_loaded",
                level="info",
                payload={"launch_summary": bool(launch_summary)},
            )
            sessions = normalize_sessions(launcher.start())
            record_ready_sessions(registry_path, sessions)
            _hold_controller(registry_path, lambda: stop_requested)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        record_failure(registry_path, error)
        return 2
    finally:
        if launcher is not None:
            try:
                launcher.stop()
            except (OSError, RuntimeError) as error:
                record_failure(registry_path, error)
        if stop_requested:
            update_state(registry_path, lifecycle_state="STOPPED")


def build_parser() -> argparse.ArgumentParser:
    """Build the internal controller parser."""

    parser = argparse.ArgumentParser(prog="python -m remote_inference_launcher.controller")
    parser.add_argument("--registry-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--launch-summary")
    parser.add_argument("--overwrite-launch-summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the internal detached controller command."""

    args = build_parser().parse_args(argv)
    return run_controller(
        registry_dir=args.registry_dir,
        config_path=args.config,
        launch_summary=args.launch_summary,
        overwrite_launch_summary=args.overwrite_launch_summary,
    )


def _hold_controller(registry_dir: str | Path, stop_requested: object) -> None:
    while not stop_requested():
        update_controller_heartbeat(registry_dir, os.getpid())
        time.sleep(HEARTBEAT_INTERVAL_SECONDS)


@contextmanager
def _controller_heartbeat(registry_dir: str | Path):
    stop = Event()

    def heartbeat_loop() -> None:
        while not stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            update_controller_heartbeat(registry_dir, os.getpid())

    update_controller_heartbeat(registry_dir, os.getpid())
    thread = Thread(
        target=heartbeat_loop,
        name="remote-inference-controller-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _read_state(registry_dir: str | Path) -> dict[str, object]:
    from remote_inference_launcher.registry import read_json

    return read_json(Path(registry_dir) / "state.json")


def _single_endpoint_plan_payload(registry_dir: str | Path) -> object | None:
    from remote_inference_launcher.registry import read_json

    plan = read_json(Path(registry_dir) / "plan.json")
    endpoints = plan.get("endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != 1:
        return None
    endpoint = endpoints[0]
    return endpoint if isinstance(endpoint, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())
