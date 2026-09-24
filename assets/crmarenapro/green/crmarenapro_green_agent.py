# ruff: noqa: E402

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
SRC_DIR_STR = str(SRC_DIR)
if SRC_DIR_STR not in sys.path:
    sys.path.insert(0, SRC_DIR_STR)

from common.logging_utils import configure_logging, get_logger
from common.uvicorn_utils import reserve_tcp_listener, run_uvicorn_with_socket, write_port_file
from common.versioning import load_component_version
from executor import Executor

configure_logging()
LOGGER = get_logger(__name__)


def _resolve_green_agent_version() -> str:
    version = load_component_version(
        Path(__file__).resolve().parent,
        env_name="BENCHMARK_GREEN_VERSION",
    )
    if version is None:
        raise RuntimeError("crmarenapro green agent version is not configured")
    return version


def create_agent_card(host: str, port: int, card_url: str | None) -> AgentCard:
    skills = [
        AgentSkill(
            id="crm-database",
            name="CRM Database Operations",
            description="Evaluate CRM database querying, schema understanding, and information extraction.",
            tags=["database", "sql", "crm", "salesforce"],
            examples=["Find all leads with status 'Qualified'"],
        ),
        AgentSkill(
            id="crm-reasoning",
            name="CRM Multi-hop Reasoning",
            description="Evaluate reasoning across multiple CRM records.",
            tags=["reasoning", "multi-hop", "analysis"],
            examples=["Which region has the highest conversion rate?"],
        ),
        AgentSkill(
            id="schema-drift-adaptation",
            name="Schema Drift Adaptation",
            description="Test robustness to renamed or modified database columns.",
            tags=["robustness", "schema", "adaptation", "adversarial"],
            examples=["Adapt to column name changes without explicit notification"],
        ),
    ]
    return AgentCard(
        name="crmarenapro_green",
        description="Green agent for CRMArenaPro CRM agent evaluation with entropic robustness scoring.",
        url=card_url or f"http://{host}:{port}/",
        version=_resolve_green_agent_version(),
        default_input_modes=["text", "text/plain", "application/json"],
        default_output_modes=["text", "text/plain", "application/json"],
        capabilities=AgentCapabilities(streaming=True),
        skills=skills,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CRMArenaPro Green Agent.")
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
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=create_agent_card(args.host, actual_port, args.card_url),
        http_handler=request_handler,
    )

    LOGGER.info("Starting crmarenapro green agent version=%s", _resolve_green_agent_version())
    run_uvicorn_with_socket(
        server.build(),
        host=args.host,
        port=actual_port,
        listener=listener,
    )


if __name__ == "__main__":
    main()
