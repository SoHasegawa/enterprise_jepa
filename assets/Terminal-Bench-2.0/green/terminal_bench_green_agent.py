# ruff: noqa: E402

"""A2A server entrypoint for the Terminal-Bench 2.0 green agent."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from harbor.environments.base import BaseEnvironment

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.logging_utils import configure_logging, get_logger
from common.uvicorn_utils import reserve_tcp_listener, run_uvicorn_with_socket, write_port_file
from agent import _resolve_green_agent_version
from executor import Executor

configure_logging()
LOGGER = get_logger(__name__)

exec_sessions: dict[str, BaseEnvironment] = {}


def build_agent_card(host: str, port: int, card_url: str | None) -> AgentCard:
    skill = AgentSkill(
        id="terminal_bench_2_green",
        name="terminal_bench_2_green",
        description=(
            "Orchestrates Terminal-Bench 2.0 tasks with Harbor Docker environments "
            "and the terminal-bench-shell-v1 purple protocol."
        ),
        tags=["terminal-bench", "benchmark", "green", "docker"],
        examples=['{"target":"sample","task_ids":["fix-git"]}'],
    )
    return AgentCard(
        name="terminal_bench_2_green",
        description="Green agent for the Terminal-Bench 2.0 benchmark.",
        url=card_url or f"http://{host}:{port}/",
        version=_resolve_green_agent_version(),
        default_input_modes=["text", "text/plain", "application/json"],
        default_output_modes=["text", "text/plain", "application/json"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Terminal-Bench 2.0 Green Agent.")
    parser.add_argument("--version", action="version", version=_resolve_green_agent_version())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file")
    parser.add_argument("--card-url")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    port_file = Path(args.port_file).expanduser().resolve() if args.port_file else None
    listener, actual_port = reserve_tcp_listener(args.host, args.port)
    write_port_file(port_file, actual_port)

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(exec_sessions),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=build_agent_card(args.host, actual_port, args.card_url),
        http_handler=request_handler,
    )
    LOGGER.info(
        "Starting Terminal-Bench 2.0 green agent version=%s on %s:%s",
        _resolve_green_agent_version(),
        args.host,
        actual_port,
    )
    run_uvicorn_with_socket(
        server.build(),
        host=args.host,
        port=actual_port,
        listener=listener,
    )


if __name__ == "__main__":
    main()
