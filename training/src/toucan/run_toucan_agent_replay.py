"""Generate and evaluate TOUCAN agent trajectories with world-model assistance.

This script is the TOUCAN counterpart to the `evaluate_agent_replay` /
`evaluate_agent_replay_via_enterpriseops_gym` paths in `src/finetuning.py`.

For each held-out TOUCAN trajectory in the test split, it:

1. Resolves the requested MCP servers via `ToucanMcpRegistry`, renders Smithery
   URLs using a key/profile from the configured pool, opens them through
   `langchain_mcp_adapters.MultiServerMCPClient` (`streamable_http` transport),
   and loads tools as LangChain `BaseTool`s.
2. Runs three replay modes — `baseline` (agent + actual MCP), `revision`
   (world-model interposed before each tool call), and `imagined`
   (world-model rolls out a planning trajectory first) — by reusing the loop
   in `run_actual_mcp_execution_mode`. Each call passes a single task plus
   placeholder padding so the existing `tasks[20:]` slice yields exactly one
   real iteration.
3. Optionally runs TOUCAN's `step4.1_response_quality_check.py` /
   `step4.3_process_completion.py` pipeline (in-process port in
   `src.toucan.toucan_response_quality_judge`) on the captured trajectories to score
   completeness / conciseness / tool-call accuracy with an LLM judge.

Outputs land under `--output-dir` (defaults to `./data/toucan_replay`):
  - `run_summary.json` — aggregate metrics + judge aggregates per mode
  - `<mode>_task_records.jsonl` — per-task replay metrics
  - `<mode>_replay_trajectories.jsonl` — recorded messages per task
  - `<mode>_judge_records.jsonl` (when judge is enabled) — judged records.

Example usage:

    uv run python src/toucan/run_toucan_agent_replay.py \
        --world-model-path checkpoints/world_model \
        --agent-model openai/gpt-4o-mini \
        --judge-method gpt5-mini \
        --enable-judge \
        --max-tasks 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from contextlib import AsyncExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finetuning import (  # noqa: E402
    DEFAULT_TRAJECTORIES_DIR,
    HFTextGenerator,
    TaskTrajectory,
    build_agent_generator,
    build_react_system_prompt,
    build_react_tool_descriptions,
    build_tool_lookup,
    canonicalize_world_model_target,
    configure_replay_limits,
    extract_replay_tasks_from_state_trajectories,
    run_actual_mcp_execution_mode,
)
from src.toucan.toucan_mcp_registry import (  # noqa: E402
    ToucanMcpRegistry,
    ToucanMcpServer,
    load_smithery_api_pool,
    render_smithery_url,
    safe_server_slug,
)
from src.toucan.toucan_response_quality_judge import (  # noqa: E402
    aggregate_mode_scores,
    evaluate_trajectory,
    load_prompt_template,
)


DEFAULT_TRAJECTORIES_PATH = (
    ROOT / "trajectories" / "toucan_world_model_test_trajectories.json"
)
DEFAULT_SMITHERY_POOL = (
    Path.home() / "program" / "tools" / "Toucan" / "datagen" / "smithery_api_pool.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "toucan_replay"


# `run_actual_mcp_execution_mode` iterates `tasks[20:]`. Padding each call with
# 20 placeholder tasks lets us hand it exactly one real task per invocation
# without modifying the function (the placeholders are never iterated).
_DEBUG_SLICE_PAD = 20


def _placeholder_task() -> TaskTrajectory:
    return TaskTrajectory(
        trajectory_index=-1,
        system_prompt="",
        user_messages=[""],
        steps=[],
        final_answer="",
    )


MODES: list[tuple[str, str, str]] = [
    # (mode key, assistance_strategy, mode_name suffix)
    ("baseline", "baseline", "baseline_actual_mcp"),
    ("revision", "revision", "world_model_assisted_actual_mcp"),
    ("imagined", "imagined", "imagined_trajectory_actual_mcp"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--trajectories-path",
        type=Path,
        default=DEFAULT_TRAJECTORIES_PATH,
        help="Held-out TOUCAN test trajectories (lean world-model format).",
    )
    parser.add_argument(
        "--smithery-api-pool",
        type=Path,
        default=DEFAULT_SMITHERY_POOL,
        help="Path to TOUCAN's `smithery_api_pool.json`.",
    )
    parser.add_argument(
        "--toucan-mcp-registry",
        type=Path,
        default=Path.home() / "program" / "tools" / "Toucan" / "mcp_servers",
        help="Directory of TOUCAN `*_labeled.json` MCP server metadata.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--world-model-path",
        required=True,
        help="HF checkpoint dir for the world model (state predictor).",
    )
    parser.add_argument(
        "--agent-model",
        default="openai/gpt-4o-mini",
        help=(
            "Model used to generate ReAct decisions. Same syntax as "
            "`finetuning.py --agent-model` (HF path, `openai/...`, "
            "`azureopenai/...`, `gemini/...`, bare OpenAI/Gemini model name, "
            "`gpt-5.1` via src.llm, legacy API method, etc.)."
        ),
    )
    parser.add_argument(
        "--world-model-target",
        default="state",
        choices=("state", "tool_execution_result_ternary", "tool_execution_result_success_failure"),
    )
    parser.add_argument("--include-error-message-in-target", action="store_true")
    parser.add_argument("--include-stage-in-target", action="store_true")
    parser.add_argument("--include-world-model-history", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--internal-thinking-max-iters", type=int, default=2)
    parser.add_argument("--imagined-trajectory-max-steps", type=int, default=3)
    parser.add_argument("--imagined-trajectory-selection-strategy", choices=("first", "llm_judge", "topk_search"), default="llm_judge")
    parser.add_argument("--imagined-trajectory-rollouts", type=int, default=1)
    parser.add_argument("--imagined-rollout-temperature", type=float, default=0.7)
    parser.add_argument("--imagined-trajectory-candidate-actions", type=int, default=3)
    parser.add_argument("--imagined-trajectory-top-k", type=int, default=3)
    parser.add_argument("--final-answer-f1-threshold", type=float, default=0.5)
    parser.add_argument("--agent-max-new-tokens", type=int, default=2048)
    parser.add_argument("--world-model-max-new-tokens", type=int, default=1024)
    parser.add_argument("--agent-max-observation-chars", type=int, default=2000)
    parser.add_argument("--agent-replay-history-budget-chars", type=int, default=60000)
    parser.add_argument("--mcp-session-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--smithery-server-config",
        type=str,
        default="",
        help=(
            "Optional JSON string passed as a per-server `config` (replaces "
            "`{config_b64}` in URL templates). Defaults to TOUCAN's "
            "`{\"debug\": false}`."
        ),
    )
    parser.add_argument(
        "--enable-judge",
        action="store_true",
        help="Run TOUCAN's response-quality-check on each generated trajectory.",
    )
    parser.add_argument(
        "--judge-method",
        default="gpt5-mini",
        help="LLM method (see src/llm.py) used for the response-quality judge.",
    )
    parser.add_argument(
        "--judge-prompt-template",
        type=Path,
        default=None,
        help=(
            "Override path to TOUCAN's `response_quality_check.md`. Defaults "
            "to `~/program/tools/Toucan/datagen/prompts/response_quality_check.md`."
        ),
    )
    parser.add_argument(
        "--skip-modes",
        nargs="*",
        default=[],
        choices=[mode[0] for mode in MODES],
        help="Mode keys to skip (useful for partial reruns).",
    )
    return parser.parse_args()


def _load_trajectories(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise SystemExit(f"Expected a JSON list at {path}, got {type(payload).__name__}.")
    return payload


def _select_tasks_with_servers(
    trajectories: list[dict[str, Any]],
    registry: ToucanMcpRegistry,
    max_tasks: int,
) -> tuple[list[tuple[dict[str, Any], TaskTrajectory, list[ToucanMcpServer]]], list[dict[str, Any]]]:
    """Pair each held-out trajectory with its resolvable MCP servers.

    Trajectories whose requested servers are not present in the registry are
    skipped (and listed in the `unresolved` return value) so the caller can
    surface those gaps in the run summary.
    """
    tasks_obj = extract_replay_tasks_from_state_trajectories(trajectories)
    selected: list[tuple[dict[str, Any], TaskTrajectory, list[ToucanMcpServer]]] = []
    unresolved: list[dict[str, Any]] = []
    for trajectory_dict, task in zip(trajectories, tasks_obj):
        if max_tasks > 0 and len(selected) >= max_tasks:
            break
        requested = trajectory_dict.get("requested_mcp_servers") or []
        matched = trajectory_dict.get("matched_mcp_servers") or []
        names = list(matched) or list(requested)
        if not names:
            unresolved.append(
                {
                    "trajectory_id": trajectory_dict.get("trajectory_id"),
                    "reason": "no_mcp_servers_recorded_in_trajectory",
                }
            )
            continue
        servers, missing = registry.lookup_many(names)
        if not servers:
            unresolved.append(
                {
                    "trajectory_id": trajectory_dict.get("trajectory_id"),
                    "reason": "no_servers_resolved_against_registry",
                    "requested": names,
                    "missing": missing,
                }
            )
            continue
        selected.append((trajectory_dict, task, servers))
    return selected, unresolved


def _build_mcp_connections(
    servers: list[ToucanMcpServer],
    *,
    api_key: str,
    profile: str,
    server_config: dict[str, Any] | None,
    sse_read_timeout_seconds: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, ToucanMcpServer]]:
    """Map TOUCAN servers to a `MultiServerMCPClient` connections dict.

    Returns `(connections, slug_to_server)` so callers can later reverse-look
    a slug back to its source registry entry (used for tool prefixing /
    diagnostics).
    """
    connections: dict[str, dict[str, Any]] = {}
    slug_to_server: dict[str, ToucanMcpServer] = {}
    for server in servers:
        url = render_smithery_url(
            server.url_template,
            api_key=api_key,
            profile=profile,
            server_config=server_config,
        )
        slug = safe_server_slug(server.server_name)
        if slug in connections:
            # Two TOUCAN names slugify to the same key — fall back to qualified.
            slug = safe_server_slug(server.qualified_name) or slug
        connections[slug] = {
            "transport": "streamable_http",
            "url": url,
            "sse_read_timeout": sse_read_timeout_seconds,
        }
        slug_to_server[slug] = server
    return connections, slug_to_server


async def _load_tools_for_task(
    connections: dict[str, dict[str, Any]],
    *,
    session_timeout_seconds: float,
) -> tuple[list[Any], list[str], list[dict[str, Any]], AsyncExitStack]:
    """Open `MultiServerMCPClient` sessions and load all tools.

    Returns `(tools, connected_slugs, skipped_servers, exit_stack)`. The caller
    is responsible for closing the returned exit stack.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    stack = AsyncExitStack()
    await stack.__aenter__()
    client = MultiServerMCPClient(connections)

    tools: list[Any] = []
    connected: list[str] = []
    skipped: list[dict[str, Any]] = []

    for server_name in connections.keys():
        try:
            session = await asyncio.wait_for(
                stack.enter_async_context(client.session(server_name)),
                timeout=session_timeout_seconds,
            )
            server_tools = await asyncio.wait_for(
                load_mcp_tools(session),
                timeout=session_timeout_seconds,
            )
        except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
            skipped.append({"server_name": server_name, "error": str(exc)})
            continue
        for tool in server_tools:
            original_name = tool.name
            original_desc = tool.description
            tool.name = f"{server_name}_{original_name}"
            tool.description = (
                f"Tool for [{server_name.upper()}] related tasks with functionality: {original_desc}"
            )
            tool.__dict__["_server_name"] = server_name
            tool.__dict__["_original_tool_name"] = original_name
        tools.extend(server_tools)
        connected.append(server_name)

    return tools, connected, skipped, stack


async def _run_one_task_one_mode(
    *,
    mode_key: str,
    base_mode_name: str,
    assistance_strategy: str,
    agent_generator: Any,
    world_model_generator: Any,
    task: TaskTrajectory,
    react_system_prompt: str,
    exact_lookup: dict[str, Any],
    alias_lookup: dict[str, list[Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Drive `run_actual_mcp_execution_mode` for a single task.

    The function pads `tasks` with 20 placeholder entries so the slice
    `tasks[20:]` inside the helper yields exactly one real task. The recorded
    JSONL paths use a per-task suffix so concurrent calls don't clobber each
    other. The captured replay record is loaded back and returned alongside
    the task metric.
    """
    mode_name = f"{base_mode_name}__toucan_traj_{task.trajectory_index}"
    padded_tasks = [_placeholder_task()] * _DEBUG_SLICE_PAD + [task]
    internal_iters = (
        args.internal_thinking_max_iters if assistance_strategy == "revision" else 0
    )
    imagined_steps = (
        args.imagined_trajectory_max_steps if assistance_strategy == "imagined" else 0
    )
    metrics = await run_actual_mcp_execution_mode(
        mode_name=mode_name,
        use_world_model_internal_thinking=(assistance_strategy == "revision"),
        assistance_strategy=assistance_strategy,
        agent_generator=agent_generator,
        world_model_generator=world_model_generator,
        tasks=padded_tasks,
        max_steps=args.max_steps,
        final_answer_f1_threshold=args.final_answer_f1_threshold,
        world_model_target=canonicalize_world_model_target(args.world_model_target),
        include_error_message_in_target=args.include_error_message_in_target,
        include_stage_in_target=args.include_stage_in_target,
        include_world_model_history=args.include_world_model_history,
        internal_thinking_max_iterations=internal_iters,
        imagined_trajectory_max_steps=imagined_steps,
        imagined_trajectory_rollouts=args.imagined_trajectory_rollouts,
        imagined_rollout_temperature=args.imagined_rollout_temperature,
        imagined_trajectory_selection_strategy=args.imagined_trajectory_selection_strategy,
        imagined_trajectory_observation_source="world_model",
        imagined_trajectory_candidate_actions=args.imagined_trajectory_candidate_actions,
        imagined_trajectory_top_k=args.imagined_trajectory_top_k,
        react_system_prompt=react_system_prompt,
        exact_lookup=exact_lookup,
        alias_lookup=alias_lookup,
        record_replay_trajectories=True,
    )
    task_records = metrics.get("task_records") or []
    task_record = dict(task_records[0]) if task_records else {}
    task_record["trajectory_index"] = task.trajectory_index
    task_record["mode"] = mode_key

    replay_jsonl_path = (
        DEFAULT_TRAJECTORIES_DIR / f"{mode_name}_replay_trajectories.jsonl"
    )
    replay_record: dict[str, Any] | None = None
    if replay_jsonl_path.exists():
        try:
            with replay_jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    replay_record = json.loads(line)
                    # The padded placeholder tasks all have trajectory_index=-1
                    # and never reach the recording branch (they're filtered by
                    # `tasks[20:]`), so the first non-empty line is our task.
                    break
        except (OSError, json.JSONDecodeError) as exc:
            replay_record = {"error": f"failed_to_read_replay_jsonl:{exc}"}

    return {"metrics": metrics, "task_record": task_record, "replay_record": replay_record}


async def _run_for_one_task(
    *,
    trajectory_dict: dict[str, Any],
    task: TaskTrajectory,
    servers: list[ToucanMcpServer],
    api_pool_entry: dict[str, str],
    agent_generator: Any,
    world_model_generator: Any,
    server_config: dict[str, Any] | None,
    args: argparse.Namespace,
    selected_modes: list[tuple[str, str, str]],
) -> dict[str, Any]:
    connections, slug_to_server = _build_mcp_connections(
        servers,
        api_key=api_pool_entry["api_key"],
        profile=api_pool_entry["profile"],
        server_config=server_config,
        sse_read_timeout_seconds=args.mcp_session_timeout_seconds,
    )

    tools, connected, skipped_servers, stack = await _load_tools_for_task(
        connections, session_timeout_seconds=args.mcp_session_timeout_seconds
    )

    per_task_outcome: dict[str, Any] = {
        "trajectory_index": task.trajectory_index,
        "trajectory_id": trajectory_dict.get("trajectory_id"),
        "requested_servers": [server.server_name for server in servers],
        "connected_servers": connected,
        "skipped_servers": skipped_servers,
        "modes": {},
    }

    if not tools:
        try:
            await stack.aclose()
        except BaseException:  # noqa: BLE001
            pass
        per_task_outcome["error"] = "no_mcp_tools_loaded_for_task"
        return per_task_outcome

    try:
        exact_lookup, alias_lookup = build_tool_lookup(tools)
        react_system_prompt = build_react_system_prompt(
            build_react_tool_descriptions(tools)
        )

        for mode_key, assistance_strategy, base_mode_name in selected_modes:
            try:
                outcome = await _run_one_task_one_mode(
                    mode_key=mode_key,
                    base_mode_name=base_mode_name,
                    assistance_strategy=assistance_strategy,
                    agent_generator=agent_generator,
                    world_model_generator=world_model_generator,
                    task=task,
                    react_system_prompt=react_system_prompt,
                    exact_lookup=exact_lookup,
                    alias_lookup=alias_lookup,
                    args=args,
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # noqa: BLE001
                tb = traceback.format_exception_only(type(exc), exc)[-1].strip()
                per_task_outcome["modes"][mode_key] = {
                    "error": f"mode_execution_failed:{tb}"
                }
                continue
            per_task_outcome["modes"][mode_key] = {
                "task_record": outcome["task_record"],
                "replay_record": outcome["replay_record"],
            }
    finally:
        try:
            await stack.aclose()
        except BaseException as exc:  # noqa: BLE001
            per_task_outcome.setdefault(
                "cleanup_warning",
                f"mcp_session_cleanup_error:{type(exc).__name__}",
            )

    return per_task_outcome


def _judge_mode_records(
    judge_template: str,
    judge_llm: Any,
    mode_outcomes: list[dict[str, Any]],
    trajectory_lookup: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    judged: list[dict[str, Any]] = []
    for outcome in mode_outcomes:
        if "task_record" not in outcome:
            continue
        replay = outcome.get("replay_record") or {}
        messages = replay.get("messages") or []
        trajectory_index = outcome.get("task_record", {}).get("trajectory_index", -1)
        traj_dict = trajectory_lookup.get(trajectory_index, {})
        question = replay.get("task_query") or ""
        target_tools = traj_dict.get("target_tools") or []
        try:
            assessment = evaluate_trajectory(
                judge_llm=judge_llm,
                template=judge_template,
                question=question,
                target_tools=target_tools,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001
            assessment = {"error": f"judge_call_failed:{exc}"}
        judged.append(
            {
                "trajectory_index": trajectory_index,
                "task_record": outcome.get("task_record"),
                "replay_record": replay,
                "response_quality_assessment": assessment,
            }
        )
    return judged


def _summarize_mode_records(
    mode_key: str,
    mode_outcomes: list[dict[str, Any]],
    judge_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    completed = 0
    final_scores: list[float] = []
    tool_calls: list[int] = []
    tool_steps: list[int] = []
    for outcome in mode_outcomes:
        record = outcome.get("task_record") or {}
        if record.get("completed"):
            completed += 1
        score = record.get("final_answer_score")
        if isinstance(score, (int, float)):
            final_scores.append(float(score))
        if isinstance(record.get("tool_calls_taken"), int):
            tool_calls.append(record["tool_calls_taken"])
        if isinstance(record.get("tool_steps_taken"), int):
            tool_steps.append(record["tool_steps_taken"])

    summary: dict[str, Any] = {
        "mode": mode_key,
        "evaluated_tasks": len(mode_outcomes),
        "completed_tasks": completed,
        "completion_rate": completed / len(mode_outcomes) if mode_outcomes else None,
        "average_final_answer_score": (
            sum(final_scores) / len(final_scores) if final_scores else None
        ),
        "average_tool_calls": (
            sum(tool_calls) / len(tool_calls) if tool_calls else None
        ),
        "average_tool_steps": (
            sum(tool_steps) / len(tool_steps) if tool_steps else None
        ),
    }
    if judge_records is not None:
        summary["judge"] = aggregate_mode_scores(judge_records)
    return summary


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TOUCAN test trajectories from {args.trajectories_path}")
    trajectories = _load_trajectories(args.trajectories_path)

    print(f"Loading TOUCAN MCP registry from {args.toucan_mcp_registry}")
    registry = ToucanMcpRegistry.load(args.toucan_mcp_registry)

    print(f"Loading Smithery API pool from {args.smithery_api_pool}")
    api_pool = load_smithery_api_pool(args.smithery_api_pool)
    if not api_pool:
        raise SystemExit(
            f"Smithery API pool at {args.smithery_api_pool} contains no usable entries. "
            "Populate at least one {'api_key': ..., 'profile': ...} record."
        )

    server_config: dict[str, Any] | None = None
    if args.smithery_server_config:
        try:
            server_config = json.loads(args.smithery_server_config)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--smithery-server-config is not valid JSON: {exc}") from exc

    selected, unresolved = _select_tasks_with_servers(
        trajectories, registry, args.max_tasks
    )
    print(
        f"Selected {len(selected)} tasks with resolvable MCP servers; "
        f"{len(unresolved)} unresolved."
    )
    if not selected:
        raise SystemExit(
            "No held-out TOUCAN trajectories matched the registry. "
            "Check --toucan-mcp-registry and the trajectory file's "
            "`requested_mcp_servers` / `matched_mcp_servers` fields."
        )

    configure_replay_limits(
        observation_chars=args.agent_max_observation_chars,
        history_budget_chars=args.agent_replay_history_budget_chars,
    )

    print(f"Loading world model from {args.world_model_path}")
    world_model_generator = HFTextGenerator(
        args.world_model_path,
        max_new_tokens=args.world_model_max_new_tokens,
    )
    print(f"Building agent generator for {args.agent_model}")
    agent_generator = build_agent_generator(
        args.agent_model, max_new_tokens=args.agent_max_new_tokens
    )

    selected_modes = [m for m in MODES if m[0] not in set(args.skip_modes)]
    if not selected_modes:
        raise SystemExit("All modes were skipped via --skip-modes; nothing to run.")

    trajectory_lookup: dict[int, dict[str, Any]] = {}
    for index, trajectory in enumerate(trajectories):
        trajectory_lookup[index] = trajectory

    started_at = time.time()
    per_task_results: list[dict[str, Any]] = []

    async def _run_all() -> None:
        for round_index, (trajectory_dict, task, servers) in enumerate(
            tqdm(selected, desc="toucan_replay")
        ):
            api_entry = api_pool[round_index % len(api_pool)]
            outcome = await _run_for_one_task(
                trajectory_dict=trajectory_dict,
                task=task,
                servers=servers,
                api_pool_entry=api_entry,
                agent_generator=agent_generator,
                world_model_generator=world_model_generator,
                server_config=server_config,
                args=args,
                selected_modes=selected_modes,
            )
            per_task_results.append(outcome)

    asyncio.run(_run_all())

    mode_outcomes: dict[str, list[dict[str, Any]]] = {
        mode_key: [] for mode_key, _, _ in selected_modes
    }
    for task_outcome in per_task_results:
        for mode_key in mode_outcomes:
            payload = task_outcome.get("modes", {}).get(mode_key)
            if not payload:
                continue
            payload_with_meta = dict(payload)
            payload_with_meta["trajectory_index"] = task_outcome["trajectory_index"]
            payload_with_meta["trajectory_id"] = task_outcome.get("trajectory_id")
            mode_outcomes[mode_key].append(payload_with_meta)

    for mode_key, outcomes in mode_outcomes.items():
        _write_jsonl(
            args.output_dir / f"{mode_key}_task_records.jsonl",
            [
                {
                    "trajectory_index": outcome["trajectory_index"],
                    "trajectory_id": outcome.get("trajectory_id"),
                    "task_record": outcome.get("task_record"),
                }
                for outcome in outcomes
            ],
        )
        _write_jsonl(
            args.output_dir / f"{mode_key}_replay_trajectories.jsonl",
            [
                {
                    "trajectory_index": outcome["trajectory_index"],
                    "trajectory_id": outcome.get("trajectory_id"),
                    "replay_record": outcome.get("replay_record"),
                }
                for outcome in outcomes
                if outcome.get("replay_record") is not None
            ],
        )

    judge_records_per_mode: dict[str, list[dict[str, Any]]] | None = None
    if args.enable_judge:
        from src.llm import LLM  # noqa: PLC0415 — defer heavy import

        template_path = args.judge_prompt_template
        judge_template = load_prompt_template(template_path) if template_path else load_prompt_template()
        judge_llm = LLM(args.judge_method)
        judge_records_per_mode = {}
        for mode_key, outcomes in mode_outcomes.items():
            judged = _judge_mode_records(
                judge_template=judge_template,
                judge_llm=judge_llm,
                mode_outcomes=outcomes,
                trajectory_lookup=trajectory_lookup,
            )
            judge_records_per_mode[mode_key] = judged
            _write_jsonl(args.output_dir / f"{mode_key}_judge_records.jsonl", judged)

    summary: dict[str, Any] = {
        "trajectories_path": str(args.trajectories_path),
        "registry_path": str(args.toucan_mcp_registry),
        "smithery_api_pool_path": str(args.smithery_api_pool),
        "world_model_path": str(args.world_model_path),
        "agent_model": args.agent_model,
        "world_model_target": canonicalize_world_model_target(args.world_model_target),
        "max_tasks": args.max_tasks,
        "max_steps": args.max_steps,
        "internal_thinking_max_iters": args.internal_thinking_max_iters,
        "imagined_trajectory_max_steps": args.imagined_trajectory_max_steps,
        "selected_tasks": len(selected),
        "unresolved_tasks": unresolved,
        "duration_seconds": time.time() - started_at,
        "modes": {
            mode_key: _summarize_mode_records(
                mode_key,
                mode_outcomes[mode_key],
                judge_records_per_mode.get(mode_key) if judge_records_per_mode else None,
            )
            for mode_key, _, _ in selected_modes
        },
    }
    if args.enable_judge:
        summary["judge_method"] = args.judge_method

    summary_path = args.output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    print(f"Wrote run summary to {summary_path}")
    for mode_key in mode_outcomes:
        mode_summary = summary["modes"][mode_key]
        print(
            f"  [{mode_key}] evaluated={mode_summary['evaluated_tasks']} "
            f"completed={mode_summary['completed_tasks']} "
            f"avg_score={mode_summary['average_final_answer_score']}"
        )


if __name__ == "__main__":
    main()
