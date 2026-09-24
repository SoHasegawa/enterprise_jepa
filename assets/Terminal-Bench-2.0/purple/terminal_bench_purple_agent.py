# ruff: noqa: E402

"""Purple agent wrapper for Terminal-Bench 2.0."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.logging_utils import configure_logging, get_logger
from common.purple_router import PurpleExecutorRegistry, RoutedPurpleExecutor
from common.uvicorn_utils import reserve_tcp_listener, run_uvicorn_with_socket, write_port_file
from common.versioning import load_component_version

configure_logging()
LOGGER = get_logger(__name__)

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent


def _resolve_purple_agent_version() -> str:
    version = load_component_version(
        Path(__file__).resolve().parent,
        env_name="BENCHMARK_PURPLE_VERSION",
    )
    if version is None:
        raise RuntimeError("Terminal-Bench-2.0 purple agent version is not configured")
    return version


def default_executor_name() -> str:
    return (
        os.getenv("TERMINAL_BENCH_EXECUTOR")
        or os.getenv("BENCHMARK_EXECUTOR")
        or "llm_shell"
    )


def resolve_executor_version(executor_name: str) -> str | None:
    return load_component_version(
        _BENCHMARK_DIR / "purple-executors" / executor_name,
        env_name="BENCHMARK_EXECUTOR_VERSION",
    )


def build_agent_card(
    host: str, port: int, card_url: str | None, executor_name: str
) -> AgentCard:
    skill = AgentSkill(
        id=f"terminal_bench_{executor_name}",
        name=f"terminal_bench_{executor_name}",
        description=(
            f"Terminal-Bench 2.0 purple wrapper for executor '{executor_name}' "
            "(terminal-bench-shell-v1 protocol)."
        ),
        tags=["terminal-bench", "benchmark", "purple", executor_name],
        examples=[
            '{"kind":"task","protocol":"terminal-bench-shell-v1","instruction":"..."}'
        ],
    )
    return AgentCard(
        name="terminal_bench_purple",
        description="Purple agent wrapper for the Terminal-Bench 2.0 benchmark.",
        url=card_url or f"http://{host}:{port}/",
        version=_resolve_purple_agent_version(),
        default_input_modes=["text", "text/plain", "application/json"],
        default_output_modes=["text", "text/plain", "application/json"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Terminal-Bench 2.0 Purple Agent."
    )
    parser.add_argument("--version", action="version", version=_resolve_purple_agent_version())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file")
    parser.add_argument("--card-url")
    parser.add_argument("--executor", default=default_executor_name())
    args = parser.parse_args()

    port_file = Path(args.port_file).expanduser().resolve() if args.port_file else None
    listener, actual_port = reserve_tcp_listener(args.host, args.port)
    write_port_file(port_file, actual_port)

    registry = PurpleExecutorRegistry(
        benchmark_dir=_BENCHMARK_DIR,
        module_prefix="terminal_bench",
    )
    request_handler = DefaultRequestHandler(
        agent_executor=RoutedPurpleExecutor(
            registry=registry, default_executor=args.executor
        ),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=build_agent_card(args.host, actual_port, args.card_url, args.executor),
        http_handler=request_handler,
    )
    LOGGER.info(
        "Starting Terminal-Bench purple agent executor=%s version=%s",
        args.executor,
        _resolve_purple_agent_version(),
    )
    run_uvicorn_with_socket(
        server.build(),
        host=args.host,
        port=actual_port,
        listener=listener,
    )


if __name__ == "__main__":
    main()
