#!/usr/bin/env python3
"""Convert canonical_event_with_nudge data into LlamaFactory (alpaca) datasets.

Two input kinds are supported, both emitted in the identical alpaca format:

1. Materialized example JSONL (default inputs) -- produced by
   ``label_canonical_events_with_llm.py`` for the ``canonical_event_with_nudge``
   target. Each record stores the full model input (system prompt, task prompt,
   action, history) plus the LLM gold ``canonical_event_with_nudge`` label, which
   becomes the training target verbatim.

2. Raw world-model trajectory JSON (``--train-trajectories`` / ``--test-trajectories``)
   -- e.g. ``enterpriseops_gym_multi_model_world_model_{train,test}_trajectories.json``.
   These have no LLM labels, so we reuse ``src/finetuning.py``'s
   ``extract_state_examples`` to turn each (action -> resulting state) pair into an
   example and ``build_state_prediction_completion_messages`` to derive the
   ``canonical_event_with_nudge`` target heuristically (via
   ``canonical_event_with_nudge_from_action_state``), exactly as finetuning does.

Either way the prompt matches *exactly* the EWM input built by ``src/finetuning.py``
for the ``canonical_event_with_nudge`` world-model target: we reuse the real prompt
builder (``build_state_prediction_chat_messages``), the real target derivation
(``build_state_prediction_completion_messages``), and the real serializer
(``canonical_event_with_nudge_json``) rather than re-implementing them here.

The EWM prompt is a two-message chat (a fixed "transition critic" system message
plus a user message that concatenates the trajectory system prompt, task prompt,
previous canonical state, recent action/observation history, and current action).
We emit it in LlamaFactory's ``alpaca`` format:

  - instruction : the transition-critic system instruction (the fixed task the
                  model must perform).
  - input       : the *entire* EWM user message produced by
                  ``build_state_prediction_chat_messages`` -> trajectory system
                  prompt + task prompt + previous canonical state + recent
                  action/observation history (default size
                  ``WORLD_MODEL_INPUT_HISTORY_SIZE``) + current action + predict
                  directive.
  - output      : the gold canonical_event_with_nudge JSON

So the two chat messages map straight across: the EWM system message becomes the
alpaca ``instruction`` and the EWM user message becomes the alpaca ``input``,
matching what ``src/finetuning.py`` feeds the model.

Records are grouped by benchmark and written to a separate file per benchmark and
per input split, then registered in a ``dataset_info.json`` so LlamaFactory can
consume them directly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Make ``import src.*`` work when this file is run directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preparation.canonical_event_state import canonical_event_with_nudge_json
from src.finetuning import (
    DEFAULT_STATE_HISTORY_SIZE,
    WORLD_MODEL_INPUT_HISTORY_SIZE,
    WORLD_MODEL_TARGET_CANONICAL_EVENT_WITH_NUDGE,
    WorldModelStateExample,
    build_state_prediction_chat_messages,
    build_state_prediction_completion_messages,
    extract_state_examples,
    load_trajectory_records,
    normalize_loaded_trajectories,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TRAJECTORIES_DIR = REPO_ROOT / "trajectories"
DEFAULT_TRAIN = (
    TRAJECTORIES_DIR
    / "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro_train_examples.jsonl"
)
DEFAULT_EVAL = (
    TRAJECTORIES_DIR
    / "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro_eval_examples.jsonl"
)
DEFAULT_TRAIN_TRAJECTORIES = (
    TRAJECTORIES_DIR / "enterpriseops_gym_multi_model_world_model_train_trajectories.json"
)
DEFAULT_TEST_TRAJECTORIES = (
    TRAJECTORIES_DIR / "enterpriseops_gym_multi_model_world_model_test_trajectories.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "llama_factory" / "canonical_event_with_nudge"
DATASET_PREFIX = "canonical_event_with_nudge"
# Distinct prefix for the raw-trajectory (heuristically-labeled) datasets so they
# never overwrite the LLM-labeled materialized datasets that share a benchmark.
TRAJECTORY_DATASET_PREFIX = "canonical_event_with_nudge_multi_model"
# Raw world-model trajectory files under trajectories/ are EnterpriseOps-Gym runs.
DEFAULT_TRAJECTORY_BENCHMARK = "EnterpriseOps-Gym"


def benchmark_slug(benchmark: str) -> str:
    """Filesystem/dataset-name-safe slug for a benchmark label."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", (benchmark or "unknown").lower()).strip("_")
    return slug or "unknown"


# ---------------------------------------------------------------------------
# Record -> example / sample
# ---------------------------------------------------------------------------
def example_from_record(record: dict[str, Any], index: int) -> WorldModelStateExample:
    """Reconstruct the EWM example, mirroring evaluation._load_canonical_eval_examples."""
    return WorldModelStateExample(
        trajectory_id=str(record.get("trajectory_id", index)),
        trajectory_index=int(record.get("trajectory_index", index)),
        interaction_index=int(record.get("interaction_index", 0)),
        system_prompt=record.get("system_prompt", "") or "",
        user_prompt=record.get("task_prompt", "") or "",
        action=record.get("action"),
        state_history=list(record.get("state_history") or []),
        input_history=list(record.get("input_history") or []),
        previous_state=record.get("previous_state"),
        state={},
    )


def gold_label(record: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the canonical_event_with_nudge gold label from a record."""
    label = record.get("canonical_event_with_nudge")
    if isinstance(label, dict):
        return label
    event = record.get("canonical_event_state")
    nudge = record.get("nudge")
    if isinstance(event, dict) and isinstance(nudge, dict):
        return {"canonical_event_state": event, "nudge": nudge}
    return None


def build_alpaca_sample(
    example: WorldModelStateExample,
    output: str,
    history_size: int,
    system_prompt_max_chars: int,
    action_max_chars: int,
) -> dict[str, str]:
    """Map one EWM example + target into an alpaca sample.

    The EWM system message becomes the alpaca `instruction`; the EWM user message
    (system prompt + task prompt + previous state + history + action + directive)
    becomes the alpaca `input`.
    """
    # Truncate the recorded history to the requested size before the EWM builder
    # embeds it (finetuning.py's normalize_world_model_input_history_text keeps the
    # last WORLD_MODEL_INPUT_HISTORY_SIZE items; slicing here honors --history-size).
    example.input_history = list(example.input_history or [])[-history_size:] if history_size > 0 else []
    messages = build_state_prediction_chat_messages(
        example,
        target_mode=WORLD_MODEL_TARGET_CANONICAL_EVENT_WITH_NUDGE,
        include_input_history=True,
        system_prompt_max_chars=system_prompt_max_chars,
        action_max_chars=action_max_chars,
    )
    system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
    return {
        "instruction": system_content,
        "input": user_content,
        "output": output,
    }


def sample_from_record(
    record: dict[str, Any],
    index: int,
    history_size: int,
    system_prompt_max_chars: int,
    action_max_chars: int,
) -> dict[str, str] | None:
    """Build one alpaca sample from a materialized (LLM-labeled) record.

    Returns None if the record has no usable gold label.
    """
    label = gold_label(record)
    if label is None:
        return None
    try:
        output = canonical_event_with_nudge_json(label)
    except ValueError:
        return None
    return build_alpaca_sample(
        example_from_record(record, index),
        output,
        history_size=history_size,
        system_prompt_max_chars=system_prompt_max_chars,
        action_max_chars=action_max_chars,
    )


def sample_from_example(
    example: WorldModelStateExample,
    history_size: int,
    system_prompt_max_chars: int,
    action_max_chars: int,
) -> dict[str, str] | None:
    """Build one alpaca sample from a raw-trajectory example.

    The target is derived from (action -> resulting state) exactly as
    finetuning.py does for the canonical_event_with_nudge target. Returns None if
    the target cannot be derived/validated.
    """
    try:
        completion = build_state_prediction_completion_messages(
            example, target_mode=WORLD_MODEL_TARGET_CANONICAL_EVENT_WITH_NUDGE
        )
    except (ValueError, KeyError, TypeError):
        return None
    output = completion[0]["content"]
    return build_alpaca_sample(
        example,
        output,
        history_size=history_size,
        system_prompt_max_chars=system_prompt_max_chars,
        action_max_chars=action_max_chars,
    )


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def iter_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Malformed JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"Expected a JSON object at {path}:{line_number}")
            yield record


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def register_dataset(dataset_info_path: Path, name: str, file_name: str) -> None:
    """Add/update an alpaca-format entry (with a system column) in dataset_info.json."""
    if dataset_info_path.exists():
        with dataset_info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
    else:
        info = {}
    info[name] = {
        "file_name": file_name,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }
    write_json(dataset_info_path, info)


# ---------------------------------------------------------------------------
# Producers: input file -> {benchmark_slug: [alpaca sample, ...]}
# ---------------------------------------------------------------------------
def produce_from_materialized(
    input_path: Path,
    history_size: int,
    system_prompt_max_chars: int,
    action_max_chars: int,
) -> tuple[dict[str, list[dict[str, str]]], int]:
    """Build samples from a materialized (LLM-labeled) example JSONL."""
    per_benchmark: dict[str, list[dict[str, str]]] = {}
    skipped = 0
    for index, record in enumerate(iter_records(input_path)):
        sample = sample_from_record(
            record,
            index,
            history_size=history_size,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
        )
        if sample is None:
            skipped += 1
            continue
        slug = benchmark_slug(record.get("benchmark", "unknown"))
        per_benchmark.setdefault(slug, []).append(sample)
    return per_benchmark, skipped


def produce_from_trajectories(
    input_path: Path,
    benchmark: str,
    history_size: int,
    system_prompt_max_chars: int,
    action_max_chars: int,
    state_history_size: int,
) -> tuple[dict[str, list[dict[str, str]]], int]:
    """Build samples from a raw world-model trajectory JSON file.

    Uses finetuning's extraction + heuristic canonical target derivation.
    """
    trajectories = normalize_loaded_trajectories(load_trajectory_records(input_path))
    examples = extract_state_examples(trajectories, state_history_size=state_history_size)
    per_benchmark: dict[str, list[dict[str, str]]] = {}
    skipped = 0
    slug = benchmark_slug(benchmark)
    for example in examples:
        sample = sample_from_example(
            example,
            history_size=history_size,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
        )
        if sample is None:
            skipped += 1
            continue
        per_benchmark.setdefault(slug, []).append(sample)
    return per_benchmark, skipped


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def write_datasets(
    per_benchmark: dict[str, list[dict[str, str]]],
    split_name: str,
    dataset_prefix: str,
    source_name: str,
    skipped: int,
    out_dir: Path,
    dataset_info_path: Path,
    register: bool,
) -> None:
    print(f"[{split_name}] {source_name}")
    if not per_benchmark:
        print(f"[{split_name}] no samples produced")
        return
    if skipped:
        print(f"[{split_name}] skipped {skipped} records without a usable target")
    total = sum(len(v) for v in per_benchmark.values())
    with_history = sum(
        1
        for samples in per_benchmark.values()
        for s in samples
        if "Recent action/observation history" in s["input"]
    )
    print(f"[{split_name}] {with_history}/{total} samples carry recent history in the input")
    for slug in sorted(per_benchmark):
        samples = per_benchmark[slug]
        dataset_name = f"{dataset_prefix}_{slug}_{split_name}"
        file_name = f"{dataset_name}.json"
        out_path = out_dir / file_name
        write_json(out_path, samples)
        lengths = [
            len(s["instruction"]) + len(s["input"]) + len(s["output"])
            for s in samples
        ]
        approx_tokens_max = int(max(lengths) / 4) if lengths else 0
        print(
            f"  {slug:20s} -> {len(samples):6d} samples  "
            f"(max ~{approx_tokens_max} tokens)  {out_path}"
        )
        if register:
            register_dataset(dataset_info_path, dataset_name, file_name)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN,
                   help="Materialized canonical_event_with_nudge train JSONL.")
    p.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL,
                   help="Materialized canonical_event_with_nudge eval JSONL.")
    p.add_argument("--train-trajectories", type=Path, default=DEFAULT_TRAIN_TRAJECTORIES,
                   help="Raw world-model train trajectory JSON (heuristic labels).")
    p.add_argument("--test-trajectories", type=Path, default=DEFAULT_TEST_TRAJECTORIES,
                   help="Raw world-model test trajectory JSON (heuristic labels).")
    p.add_argument("--trajectory-dataset-prefix", type=str, default=TRAJECTORY_DATASET_PREFIX,
                   help="Dataset/file prefix for raw-trajectory datasets.")
    p.add_argument("--trajectory-benchmark", type=str, default=DEFAULT_TRAJECTORY_BENCHMARK,
                   help="Benchmark label for raw-trajectory datasets (drives the file slug).")
    p.add_argument("--skip-materialized", action="store_true",
                   help="Skip the materialized JSONL inputs.")
    p.add_argument("--skip-trajectories", action="store_true",
                   help="Skip the raw trajectory JSON inputs.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Directory to write LlamaFactory dataset files into.")
    p.add_argument("--dataset-info", type=Path, default=None,
                   help="Path to dataset_info.json (default: <out-dir>/dataset_info.json).")
    p.add_argument("--history-size", type=int, default=WORLD_MODEL_INPUT_HISTORY_SIZE,
                   help="Number of most-recent action/observation history items to embed "
                        "in the input block (0 disables history). "
                        f"Default: {WORLD_MODEL_INPUT_HISTORY_SIZE} (WORLD_MODEL_INPUT_HISTORY_SIZE).")
    p.add_argument("--state-history-size", type=int, default=DEFAULT_STATE_HISTORY_SIZE,
                   help="State-history depth passed to extract_state_examples for raw "
                        f"trajectory inputs. Default: {DEFAULT_STATE_HISTORY_SIZE}.")
    p.add_argument("--world-model-system-prompt-max-chars", type=int, default=0,
                   help="Truncate the trajectory system prompt to this many chars (0 = no limit).")
    p.add_argument("--world-model-action-max-chars", type=int, default=0,
                   help="Truncate the action text to this many chars (0 = no limit).")
    p.add_argument("--no-register", action="store_true",
                   help="Skip writing/updating dataset_info.json.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_info_path = args.dataset_info or (args.out_dir / "dataset_info.json")
    register = not args.no_register

    if not args.skip_materialized:
        for split_name, path in [("train", args.train_path), ("eval", args.eval_path)]:
            if path is None:
                continue
            if not Path(path).exists():
                print(f"[{split_name}] materialized input not found, skipping: {path}")
                continue
            per_benchmark, skipped = produce_from_materialized(
                Path(path),
                history_size=args.history_size,
                system_prompt_max_chars=args.world_model_system_prompt_max_chars,
                action_max_chars=args.world_model_action_max_chars,
            )
            write_datasets(
                per_benchmark,
                split_name=split_name,
                dataset_prefix=DATASET_PREFIX,
                source_name=Path(path).name,
                skipped=skipped,
                out_dir=args.out_dir,
                dataset_info_path=dataset_info_path,
                register=register,
            )

    if not args.skip_trajectories:
        for split_name, path in [("train", args.train_trajectories), ("test", args.test_trajectories)]:
            if path is None:
                continue
            if not Path(path).exists():
                print(f"[{split_name}] trajectory input not found, skipping: {path}")
                continue
            per_benchmark, skipped = produce_from_trajectories(
                Path(path),
                benchmark=args.trajectory_benchmark,
                history_size=args.history_size,
                system_prompt_max_chars=args.world_model_system_prompt_max_chars,
                action_max_chars=args.world_model_action_max_chars,
                state_history_size=args.state_history_size,
            )
            write_datasets(
                per_benchmark,
                split_name=split_name,
                dataset_prefix=args.trajectory_dataset_prefix,
                source_name=Path(path).name,
                skipped=skipped,
                out_dir=args.out_dir,
                dataset_info_path=dataset_info_path,
                register=register,
            )

    if register:
        print(f"Registered datasets in {dataset_info_path}")


if __name__ == "__main__":
    main()
