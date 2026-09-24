import argparse
import json
import re
import sys
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.generate_enterpriseops_gym_stateful_samples import extract_action_batches  # noqa: E402
from src.generation.generate_enterpriseops_gym_world_model_trajectories import (  # noqa: E402
    build_planned_state_message,
    build_stage_labels,
    dump_records,
    extract_system_and_user_messages,
    summarize_tool_batch,
)
from src.generation.generate_world_model_trajectories import (  # noqa: E402
    make_default_tool_context,
)
from src.data_preparation.world_model_trajectory_cleanup import cleanup_world_model_trajectory  # noqa: E402


TASK_ID_RE = re.compile(r"(task_\d{8}_\d{6}_\d{3}_[0-9a-f]+_[0-9a-f]+)")
DEFAULT_SOURCE_DIR = Path(
    "/data/Trajectory/"
    "bm-EnterpriseOps-Gym_ex-mcp_react_tg-hf_dataset_ts-all_cf-cfff05db2ef8_us-user_rn-20260513T044856Z-cfff05db2ef8/trajectories"
)
DEFAULT_SEEDS_PATH = ROOT / "trajectories" / "imported_benchmark_seeds.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "trajectories" / "enterpriseops_gym_world_model_mcp_react_trajectories.json"
DEFAULT_STATE_CONVERTER = "tool_context"


class StateConverter(ABC):
    """Build trajectory `state` messages from reconstructed action batches.

    Different downstream experiments may want different state definitions even
    when they share the same action sequence. Subclasses should encapsulate
    that policy so the rest of the converter only needs a single interface.
    """

    name = "base"

    @abstractmethod
    def build_messages(
        self,
        *,
        system_prompt: str,
        action_batches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return alternating `action` / `state` messages for the trajectory."""


class PlannedStateConverter(StateConverter):
    """Default state converter.

    Mirrors the current world-model trajectory format by summarizing tool
    batches and inferring process state via `build_planned_state_message`.
    """

    name = "planned"

    def build_messages(
        self,
        *,
        system_prompt: str,
        action_batches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        stage_labels = build_stage_labels(action_batches)
        previous_process_state = None
        latest_tool_context = make_default_tool_context()
        total_steps = len(action_batches)
        messages: list[dict[str, Any]] = []

        for step_index, batch in enumerate(action_batches, start=1):
            action_content = batch["action_content"]
            if batch["tool_results"]:
                latest_tool_context = summarize_tool_batch(batch["tool_results"])

            messages.append({"role": "action", "content": action_content})
            state_message, previous_process_state = build_planned_state_message(
                system_prompt=system_prompt,
                action_content=action_content,
                latest_tool_context=latest_tool_context,
                previous_process_state=previous_process_state,
                stage_labels=stage_labels,
                step_index=step_index,
                total_steps=total_steps,
            )
            messages.append(state_message)

        return messages


class ToolContextStateConverter(StateConverter):
    """Minimal state converter.

    Emits only the tool outcome summary needed for tool-output-centric world
    models: binary execution result, tool name, and summarized tool output.
    """

    name = "tool_context"

    def build_messages(
        self,
        *,
        system_prompt: str,
        action_batches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del system_prompt
        messages: list[dict[str, Any]] = []

        for batch in action_batches:
            action_content = batch["action_content"]
            tool_context = summarize_tool_batch(batch["tool_results"])
            messages.append({"role": "action", "content": action_content})
            messages.append(
                {
                    "role": "state",
                    "content": {
                        "last_tool_execution_result": tool_context.get(
                            "last_tool_execution_result"
                        ),
                        "last_tool_name": tool_context.get("last_tool_name"),
                        "last_tool_output": tool_context.get("last_tool_output"),
                    },
                }
            )

        return messages


STATE_CONVERTER_REGISTRY: dict[str, type[StateConverter]] = {
    ToolContextStateConverter.name: ToolContextStateConverter,
    PlannedStateConverter.name: PlannedStateConverter,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert EnterpriseOps-Gym JSONL task event logs into the same world-model "
            "trajectory schema used by trajectories/enterpriseops_gym_world_model_*_trajectories.json."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--seeds-path", type=Path, default=DEFAULT_SEEDS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl"],
        default="json",
        help="Serialize trajectories as a JSON list or JSONL.",
    )
    parser.add_argument(
        "--source-variant",
        default=None,
        help="Optional explicit source_variant string. Defaults to a slug of the dataset directory name.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any source file is not reconstructable.",
    )
    parser.add_argument(
        "--state-converter",
        choices=sorted(STATE_CONVERTER_REGISTRY),
        default=DEFAULT_STATE_CONVERTER,
        help=(
            "State-construction strategy used to infer `state` messages from "
            "the reconstructed action trajectory."
        ),
    )
    return parser.parse_args()


def slugify_path_part(part: str) -> str:
    slug = "".join(character if character.isalnum() else "_" for character in str(part))
    slug = slug.strip("_")
    return slug or "unknown"


def default_source_variant(source_dir: Path) -> str:
    dataset_root = source_dir.parent if source_dir.name == "trajectories" else source_dir
    return slugify_path_part(dataset_root.name)


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_task_id_from_text(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = TASK_ID_RE.search(text)
    if match:
        return match.group(1)
    return None


def load_seed_lookup(seeds_path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    task_lookup: dict[str, dict[str, Any]] = {}
    max_task_index = -1
    source_entries: list[tuple[str, str, dict[str, Any]]] = []

    with seeds_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("source_dataset") != "EnterpriseOps-Gym":
                continue
            source_path = (record.get("source_record") or {}).get("source_path")
            if not source_path:
                continue
            task_id = extract_task_id_from_text(source_path)
            if not task_id:
                continue
            source_entries.append((str(source_path), task_id, record))

    task_index_by_task_id = {
        task_id: index
        for index, (_, task_id, _) in enumerate(sorted(source_entries, key=lambda item: item[0]))
    }

    for source_path, task_id, record in source_entries:
        task_index = task_index_by_task_id[task_id]
        max_task_index = max(max_task_index, task_index)
        task_lookup[task_id] = {
            "seed_id": record.get("seed_id"),
            "coordination_pattern": record.get("coordination_pattern"),
            "domain": (record.get("environment") or {}).get("domain"),
            "seed_source_path": source_path,
            "task_index": task_index,
        }

    return task_lookup, max_task_index


def request_metadata_from_text(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        task_id = extract_task_id_from_text(text)
        return {"task_id": task_id} if task_id else None
    return parsed if isinstance(parsed, dict) else None


def extract_request_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        if event.get("event_type") != "Message":
            continue
        payload = event.get("payload") or {}
        for part in payload.get("parts") or []:
            metadata = request_metadata_from_text(part.get("text"))
            if metadata is not None:
                return metadata
    return {}


def extract_purple_internal_record(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in events:
        if event.get("event_type") == "PurpleInternalRecord":
            payload = event.get("payload")
            if isinstance(payload, dict):
                return payload
    return None


def build_synthetic_seed_metadata(task_id: str, domain: str | None, task_index: int) -> dict[str, Any]:
    normalized_domain = domain or "unknown"
    return {
        "seed_id": f"import.enterpriseops_gym_jsonl.{normalized_domain}.{task_id}",
        "coordination_pattern": "synthetic_single_agent_replay",
        "domain": normalized_domain,
        "seed_source_path": None,
        "task_index": task_index,
    }


def normalize_seed_metadata(
    *,
    seed_lookup: dict[str, dict[str, Any]],
    synthetic_task_indices: dict[str, int],
    next_synthetic_task_index: list[int],
    task_id: str,
    domain: str | None,
) -> dict[str, Any]:
    existing = seed_lookup.get(task_id)
    if existing is not None:
        normalized = dict(existing)
        if domain and not normalized.get("domain"):
            normalized["domain"] = domain
        return normalized

    if task_id not in synthetic_task_indices:
        synthetic_task_indices[task_id] = next_synthetic_task_index[0]
        next_synthetic_task_index[0] += 1
    task_index = synthetic_task_indices[task_id]
    return build_synthetic_seed_metadata(task_id, domain, task_index)


def build_state_converter(name: str) -> StateConverter:
    converter_cls = STATE_CONVERTER_REGISTRY.get(name)
    if converter_cls is None:
        available = ", ".join(sorted(STATE_CONVERTER_REGISTRY))
        raise ValueError(f"Unknown state converter `{name}`. Available: {available}")
    return converter_cls()


def reconstruct_trajectory(
    *,
    source_path: Path,
    source_variant: str,
    seed_metadata: dict[str, Any],
    conversation_flow: list[dict[str, Any]],
    state_converter: StateConverter,
) -> dict[str, Any]:
    system_prompt, user_prompt = extract_system_and_user_messages(conversation_flow)
    action_batches = extract_action_batches(conversation_flow)
    if not action_batches:
        raise ValueError("no reconstructable ai_message batches")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    messages.extend(
        state_converter.build_messages(
            system_prompt=system_prompt,
            action_batches=action_batches,
        )
    )

    trajectory = {
        "trajectory_id": f"enterpriseops-gym-world-model-{source_variant}-{source_path.stem}",
        "source": "enterprise_ops_gym_mcp_react_jsonl",
        "domain": seed_metadata["domain"],
        "seed_id": seed_metadata["seed_id"],
        "task_index": seed_metadata["task_index"],
        "task_stem": source_path.stem,
        "coordination_pattern": seed_metadata["coordination_pattern"],
        "seed_source_path": seed_metadata.get("seed_source_path") or str(source_path),
        "source_path": str(source_path),
        "source_variant": source_variant,
        "messages": messages,
    }
    return cleanup_world_model_trajectory(trajectory)


def skipped_record(
    source_path: Path,
    task_id: str,
    seed_metadata: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "source_path": str(source_path),
        "task_id": task_id,
        "seed_id": seed_metadata["seed_id"],
        "reason": reason,
    }


def convert_source_file(
    *,
    source_path: Path,
    source_variant: str,
    seed_lookup: dict[str, dict[str, Any]],
    synthetic_task_indices: dict[str, int],
    next_synthetic_task_index: list[int],
    state_converter: StateConverter,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
    events = load_jsonl_records(source_path)
    request_metadata = extract_request_metadata(events)
    task_id = request_metadata.get("task_id") or source_path.stem
    domain = request_metadata.get("domain")
    seed_metadata = normalize_seed_metadata(
        seed_lookup=seed_lookup,
        synthetic_task_indices=synthetic_task_indices,
        next_synthetic_task_index=next_synthetic_task_index,
        task_id=task_id,
        domain=domain,
    )

    purple_record = extract_purple_internal_record(events)
    if purple_record is None:
        return (
            None,
            skipped_record(source_path, task_id, seed_metadata, "missing PurpleInternalRecord"),
            task_id in seed_lookup,
        )

    conversation_flow = purple_record.get("conversation_flow") or []
    if not conversation_flow:
        return (
            None,
            skipped_record(source_path, task_id, seed_metadata, "missing conversation_flow"),
            task_id in seed_lookup,
        )

    if not seed_metadata.get("domain"):
        seed_metadata["domain"] = domain or "unknown"

    try:
        trajectory = reconstruct_trajectory(
            source_path=source_path,
            source_variant=source_variant,
            seed_metadata=seed_metadata,
            conversation_flow=conversation_flow,
            state_converter=state_converter,
        )
    except Exception as exc:
        return (
            None,
            skipped_record(source_path, task_id, seed_metadata, str(exc)),
            task_id in seed_lookup,
        )

    return trajectory, None, task_id in seed_lookup


def source_jsonl_files(source_dir: Path) -> list[Path]:
    source_files = sorted(path for path in source_dir.glob("*.jsonl") if path.is_file())
    if not source_files:
        raise ValueError(f"No JSONL trajectory files found under {source_dir}")
    return source_files


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    source_variant = args.source_variant or default_source_variant(source_dir)
    state_converter = build_state_converter(args.state_converter)

    seed_lookup, max_seed_task_index = load_seed_lookup(args.seeds_path)
    synthetic_task_indices: dict[str, int] = {}
    next_synthetic_task_index = [max_seed_task_index + 1]

    trajectories: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    matched_seed_records = 0
    synthetic_seed_records = 0

    for source_path in source_jsonl_files(source_dir):
        trajectory, skipped, matched_seed = convert_source_file(
            source_path=source_path,
            source_variant=source_variant,
            seed_lookup=seed_lookup,
            synthetic_task_indices=synthetic_task_indices,
            next_synthetic_task_index=next_synthetic_task_index,
            state_converter=state_converter,
        )
        if matched_seed:
            matched_seed_records += 1
        else:
            synthetic_seed_records += 1
        if skipped is not None:
            skipped_records.append(skipped)
        if trajectory is not None:
            trajectories.append(trajectory)

    if args.strict and skipped_records:
        preview = "\n".join(item["source_path"] for item in skipped_records[:10])
        raise ValueError(f"Found non-reconstructable JSONL trajectories:\n{preview}")

    if not trajectories:
        raise ValueError(f"No trajectories were reconstructed from {source_dir}")

    dump_records(args.output_path, trajectories, args.output_format)

    domain_counts = Counter(item["domain"] for item in trajectories)

    print(f"Wrote {len(trajectories)} trajectories to {args.output_path}")
    print(f"State converter: {state_converter.name}")
    print(f"Matched seed metadata for {matched_seed_records} trajectories")
    print(f"Synthesized seed metadata for {synthetic_seed_records} trajectories")
    print(f"Skipped {len(skipped_records)} source files")
    if domain_counts:
        print(f"Domain counts: {dict(sorted(domain_counts.items()))}")
    if skipped_records:
        print("Sample skipped records:")
        for item in skipped_records[:5]:
            print(f"  {item['source_path']}: {item['reason']}")


if __name__ == "__main__":
    main()
