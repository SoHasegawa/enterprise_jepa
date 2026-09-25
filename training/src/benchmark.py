import argparse
import json
from pathlib import Path


STATE_KEYS = [
    "identify",
    "artifacts",
    "process",
    "relational",
    "context",
    "temporal",
]


PRIMITIVE_DIR = Path("primitive_libraries")
TEMPLATE_DIR = Path("scenario_templates")
MCP_SERVER_DIR = Path("mcp_servers")

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_keys(payload: dict, required_keys: list[str], label: str):
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"{label} missing keys: {missing}")


def discover_json_files(directory: Path):
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def validate_state_keys(payload: dict, label: str):
    missing = [key for key in STATE_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{label} missing state aspects: {missing}")


def load_primitive_libraries(directory: Path):
    libraries_by_aspect = {aspect: {} for aspect in STATE_KEYS}
    primitive_index = {aspect: {} for aspect in STATE_KEYS}

    for path in discover_json_files(directory):
        payload = load_json(path)
        require_keys(
            payload,
            ["library_id", "aspect", "version", "description", "primitives"],
            str(path),
        )

        aspect = payload["aspect"]
        if aspect not in STATE_KEYS:
            raise ValueError(f"{path} has unknown aspect: {aspect}")
        if not isinstance(payload["primitives"], list) or not payload["primitives"]:
            raise ValueError(f"{path} must define at least one primitive")

        libraries_by_aspect[aspect][payload["library_id"]] = path

        for primitive in payload["primitives"]:
            require_keys(
                primitive,
                ["primitive_id", "name", "summary", "state_fragment"],
                f"{path} primitive",
            )
            if primitive["primitive_id"] in primitive_index[aspect]:
                raise ValueError(
                    f"duplicate primitive_id for aspect {aspect}: {primitive['primitive_id']}"
                )
            if not isinstance(primitive["state_fragment"], dict):
                raise ValueError(f"{path} primitive state_fragment must be an object")
            primitive_index[aspect][primitive["primitive_id"]] = path

    return libraries_by_aspect, primitive_index


def load_mcp_servers(directory: Path):
    servers = {}

    for path in discover_json_files(directory):
        payload = load_json(path)
        require_keys(
            payload,
            ["server_id", "display_name", "transport", "description", "tools"],
            str(path),
        )
        server_id = payload["server_id"]
        if server_id in servers:
            raise ValueError(f"duplicate MCP server_id: {server_id}")
        if not isinstance(payload["tools"], list) or not payload["tools"]:
            raise ValueError(f"{path} must define at least one tool")

        tool_names = set()
        for tool in payload["tools"]:
            require_keys(
                tool,
                ["tool_name", "description", "input_schema"],
                f"{path} tool",
            )
            if tool["tool_name"] in tool_names:
                raise ValueError(f"{path} contains duplicate tool_name: {tool['tool_name']}")
            if not isinstance(tool["input_schema"], dict):
                raise ValueError(f"{path} tool input_schema must be an object")
            tool_names.add(tool["tool_name"])

        servers[server_id] = path

    return servers


def validate_agents(agents: list[dict], primitive_index: dict[str, dict[str, Path]], label: str):
    if not isinstance(agents, list) or len(agents) < 2:
        raise ValueError(f"{label} must define at least two agents")

    seen_agents = set()
    for agent in agents:
        require_keys(
            agent,
            ["agent_id", "display_name", "identify_primitive", "responsibilities"],
            label,
        )
        agent_id = agent["agent_id"]
        if agent_id in seen_agents:
            raise ValueError(f"{label} contains duplicate agent_id: {agent_id}")
        if agent["identify_primitive"] not in primitive_index["identify"]:
            raise ValueError(
                f"{label} references unknown identify primitive: {agent['identify_primitive']}"
            )
        seen_agents.add(agent_id)

    return seen_agents


def validate_steps(
    steps: list[dict],
    agent_ids: set[str],
    label: str,
):
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError(f"{label} must define at least two steps")

    for expected_step_id, step in enumerate(steps, start=1):
        require_keys(
            step,
            [
                "step_id",
                "title",
                "objective",
                "owner_agent",
                "turns",
                "state_delta",
                "success_checks",
            ],
            label,
        )
        if step["step_id"] != expected_step_id:
            raise ValueError(f"{label} step_id must be sequential starting at 1")
        if step["owner_agent"] not in agent_ids:
            raise ValueError(f"{label} step owner is unknown: {step['owner_agent']}")
        if not isinstance(step["turns"], list) or len(step["turns"]) < 2:
            raise ValueError(f"{label} step {expected_step_id} must define at least two turns")
        for turn in step["turns"]:
            require_keys(turn, ["speaker_agent", "audience", "message_goal"], label)
            if turn["speaker_agent"] not in agent_ids:
                raise ValueError(
                    f"{label} step {expected_step_id} has unknown speaker_agent: {turn['speaker_agent']}"
                )
        if not isinstance(step["state_delta"], dict):
            raise ValueError(f"{label} step {expected_step_id} state_delta must be an object")
        if not isinstance(step["success_checks"], list) or not step["success_checks"]:
            raise ValueError(f"{label} step {expected_step_id} must define success checks")


def validate_template(
    payload: dict,
    path: Path,
    libraries_by_aspect: dict[str, dict[str, Path]],
    primitive_index: dict[str, dict[str, Path]],
    mcp_servers: dict[str, Path],
):
    require_keys(
        payload,
        [
            "template_id",
            "title",
            "summary",
            "perspectives",
            "mcp_servers",
            "primitive_refs",
            "agents",
            "initial_state",
            "steps",
            "evaluation",
        ],
        str(path),
    )

    perspectives = payload["perspectives"]
    require_keys(perspectives, ["multi_agent", "multi_step", "multi_turn"], f"{path} perspectives")
    for key in ["multi_agent", "multi_step", "multi_turn"]:
        if perspectives[key] is not True:
            raise ValueError(f"{path} requires {key}=true for benchmark templates")

    if not isinstance(payload["mcp_servers"], list) or not payload["mcp_servers"]:
        raise ValueError(f"{path} must reference at least one MCP server")
    for server_id in payload["mcp_servers"]:
        if server_id not in mcp_servers:
            raise ValueError(f"{path} references unknown MCP server: {server_id}")

    primitive_refs = payload["primitive_refs"]
    validate_state_keys(primitive_refs, f"{path} primitive_refs")
    for aspect in STATE_KEYS:
        refs = primitive_refs[aspect]
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"{path} primitive_refs[{aspect}] must contain at least one primitive")
        for primitive_id in refs:
            if primitive_id not in primitive_index[aspect]:
                raise ValueError(f"{path} references unknown {aspect} primitive: {primitive_id}")
        if not libraries_by_aspect[aspect]:
            raise ValueError(f"{path} cannot resolve any library for aspect {aspect}")

    validate_state_keys(payload["initial_state"], f"{path} initial_state")
    agent_ids = validate_agents(payload["agents"], primitive_index, f"{path} agents")
    validate_steps(payload["steps"], agent_ids, f"{path} steps")

    evaluation = payload["evaluation"]
    require_keys(
        evaluation,
        ["pass_conditions", "failure_conditions", "state_coverage"],
        f"{path} evaluation",
    )
    if evaluation["state_coverage"] != STATE_KEYS:
        raise ValueError(f"{path} evaluation.state_coverage must list all six aspects in order")


def command_inventory(_args):
    libraries_by_aspect, _primitive_index = load_primitive_libraries(PRIMITIVE_DIR)
    template_paths = discover_json_files(TEMPLATE_DIR)
    mcp_servers = load_mcp_servers(MCP_SERVER_DIR)

    print("primitive_libraries:")
    for aspect in STATE_KEYS:
        print(f"  {aspect}: {len(libraries_by_aspect[aspect])}")
        for library_id, path in sorted(libraries_by_aspect[aspect].items()):
            print(f"    {library_id} -> {path}")

    print("scenario_templates:")
    for path in template_paths:
        print(f"  {path}")

    print("mcp_servers:")
    for server_id, path in sorted(mcp_servers.items()):
        print(f"  {server_id} -> {path}")


def command_validate(_args):
    libraries_by_aspect, primitive_index = load_primitive_libraries(PRIMITIVE_DIR)
    template_paths = discover_json_files(TEMPLATE_DIR)
    mcp_servers = load_mcp_servers(MCP_SERVER_DIR)

    if not template_paths:
        raise ValueError("no scenario templates found")
    if not mcp_servers:
        raise ValueError("no MCP servers found")

    for path in template_paths:
        payload = load_json(path)
        validate_template(payload, path, libraries_by_aspect, primitive_index, mcp_servers)

    library_count = sum(len(entries) for entries in libraries_by_aspect.values())
    print(
        f"Validated {library_count} primitive libraries, {len(mcp_servers)} MCP servers, and {len(template_paths)} scenario templates"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Enterprise benchmark scaffold utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory", help="show primitive libraries and templates")
    inventory_parser.set_defaults(func=command_inventory)

    validate_parser = subparsers.add_parser("validate", help="validate primitive libraries and templates")
    validate_parser.set_defaults(func=command_validate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
