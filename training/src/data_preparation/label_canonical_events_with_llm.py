#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preparation.canonical_event_state import (  # noqa: E402
    CANONICAL_EVENT_WITH_NUDGE_SCHEMA_VERSION,
    EVENT_STATE_SCHEMA_VERSION,
    INFORMATION_GAIN_VALUES,
    INFORMATION_SUFFICIENCY_VALUES,
    MISSING_INFORMATION_TYPE_VALUES,
    NUDGE_SCHEMA_VERSION,
    RECOMMENDED_ABSTRACT_ACTION_VALUES,
    REQUIRED_CATEGORICAL_FIELDS,
    canonical_event_from_action_state,
    canonical_event_with_nudge_from_action_state,
    validate_canonical_event_with_nudge,
    validate_event_state,
)
from src.data_preparation.generate_canonical_event_state_examples import (  # noqa: E402
    dump_json,
    dump_jsonl,
    load_trajectories,
)
from src.finetuning import (  # noqa: E402
    TRAJECTORY_DATASET_PRESETS,
    WorldModelStateExample,
    extract_state_examples,
)

DEFAULT_OUTPUT_DIR = ROOT / "trajectories"
DEFAULT_TRAJECTORY_DATASET = "enterpriseops_gym_terminalbench_2_0_crmarenapro"
DEFAULT_MODEL = "gpt-5.1"
TARGET_CANONICAL_EVENT_STATE = "canonical_event_state"
TARGET_CANONICAL_EVENT_WITH_NUDGE = "canonical_event_with_nudge"
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label canonical event and optional nudge targets with a strong LLM. "
            "The script validates every model label against the categorical schema."
        )
    )
    parser.add_argument(
        "--trajectory-dataset",
        choices=sorted(TRAJECTORY_DATASET_PRESETS),
        default=DEFAULT_TRAJECTORY_DATASET,
    )
    parser.add_argument("--train-data-path", type=Path, nargs="+", default=None)
    parser.add_argument("--eval-data-path", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--target-format",
        choices=[TARGET_CANONICAL_EVENT_STATE, TARGET_CANONICAL_EVENT_WITH_NUDGE],
        default=TARGET_CANONICAL_EVENT_WITH_NUDGE,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None)
    parser.add_argument(
        "--response-format",
        choices=["json_schema", "json_object", "none"],
        default="json_schema",
        help="Use strict JSON schema by default; choose json_object/none for less capable OpenAI-compatible endpoints.",
    )
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--state-history-size", type=int, default=3)
    parser.add_argument("--max-input-history-items", type=int, default=8)
    parser.add_argument("--max-field-chars", type=int, default=4000)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--train-output-path", type=Path, default=None)
    parser.add_argument("--eval-output-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip examples already present in the output JSONL files and append newly labeled rows.",
    )
    parser.add_argument(
        "--fallback-to-heuristic",
        action="store_true",
        help="Use local heuristic labels only when the LLM response fails after all retries.",
    )
    parser.add_argument(
        "--include-raw-response",
        action="store_true",
        help="Store raw LLM response text for auditing. Disabled by default to keep artifacts clean.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and print the first payload without calling the LLM.",
    )
    return parser.parse_args()


def default_output_paths(target_format: str, trajectory_dataset: str) -> tuple[Path, Path, Path]:
    suffix = "canonical_event_with_nudge_llm" if target_format == TARGET_CANONICAL_EVENT_WITH_NUDGE else "canonical_event_state_llm"
    dataset_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", trajectory_dataset).strip("_")
    return (
        DEFAULT_OUTPUT_DIR / f"{suffix}_{dataset_suffix}_train_examples.jsonl",
        DEFAULT_OUTPUT_DIR / f"{suffix}_{dataset_suffix}_eval_examples.jsonl",
        DEFAULT_OUTPUT_DIR / f"{suffix}_{dataset_suffix}_manifest.json",
    )


def benchmark_name_from_trajectory_id(trajectory_id: str) -> str:
    normalized = trajectory_id.lower()
    if normalized.startswith("terminalbench") or "terminal-bench" in normalized or "terminalbench" in normalized:
        return "Terminal-Bench-2.0"
    if normalized.startswith("crmarenapro") or "crmarenapro" in normalized or "crmarena-pro" in normalized:
        return "CRMArenaPro"
    if normalized.startswith("enterpriseops") or "enterpriseops" in normalized:
        return "EnterpriseOps-Gym"
    return "unknown"


def benchmark_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("benchmark", "unknown") for row in rows).items()))


def example_key(example: WorldModelStateExample, split: str) -> tuple[str, str, int]:
    return (split, example.trajectory_id, example.interaction_index)


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("split", "")),
        str(row.get("trajectory_id", "")),
        int(row.get("interaction_index", -1)),
    )


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}: expected object")
            rows.append(payload)
    return rows


def dedupe_rows_by_key(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, int]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def truncate_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "...<truncated>"
    if isinstance(value, list):
        return [truncate_value(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {key: truncate_value(item, max_chars) for key, item in value.items()}
    return value


def compact_example_payload(example: WorldModelStateExample, *, max_input_history_items: int, max_field_chars: int) -> dict[str, Any]:
    return truncate_value(
        {
            "trajectory_id": example.trajectory_id,
            "benchmark": benchmark_name_from_trajectory_id(example.trajectory_id),
            "trajectory_index": example.trajectory_index,
            "interaction_index": example.interaction_index,
            "system_prompt": example.system_prompt,
            "task_prompt": example.user_prompt,
            "recent_state_history": [],
            "recent_input_history": example.input_history[-max_input_history_items:],
            "previous_state": None,
            "candidate_action": example.action,
            "observed_resulting_state": example.state,
            "observed_tool_output": example.tool_output,
            "observed_error_payload": example.error_payload,
        },
        max_field_chars,
    )


def values_text(values: set[str]) -> str:
    return ", ".join(sorted(values))


def build_event_schema_instruction() -> str:
    lines = [
        "canonical_event_state fields and allowed values:",
    ]
    for field, values in REQUIRED_CATEGORICAL_FIELDS.items():
        lines.append(f"- {field}: {values_text(values)}")
    return "\n".join(lines)


def build_nudge_schema_instruction() -> str:
    return "\n".join(
        [
            "nudge fields and allowed values:",
            f"- information_sufficiency: {values_text(INFORMATION_SUFFICIENCY_VALUES)}",
            f"- information_gain: {values_text(INFORMATION_GAIN_VALUES)}",
            f"- recommended_abstract_action: {values_text(RECOMMENDED_ABSTRACT_ACTION_VALUES)}",
            "- missing_information_type: non-empty list using only "
            f"{values_text(MISSING_INFORMATION_TYPE_VALUES)}; use ['none'] only by itself.",
        ]
    )


def build_labeling_messages(example_payload: dict[str, Any], target_format: str) -> list[dict[str, str]]:
    if target_format == TARGET_CANONICAL_EVENT_WITH_NUDGE:
        output_contract = "\n".join(
            [
                "Return exactly one JSON object with top-level fields: canonical_event_state, nudge.",
                "Do not include schema_version anywhere in the output.",
                build_event_schema_instruction(),
                build_nudge_schema_instruction(),
            ]
        )
    else:
        output_contract = "\n".join(
            [
                "Return exactly one JSON object containing only the canonical event state fields.",
                "Do not include schema_version anywhere in the output.",
                build_event_schema_instruction(),
            ]
        )

    system_prompt = (
        "You are an expert trajectory labeler for enterprise agent world-model training. "
        "Your job is to label the observed effect of exactly one candidate action using only the provided categorical schema. "
        "Use the observed resulting state/tool output as evidence for the label. "
        "Do not solve the task, do not invent benchmark-specific fields, and do not write explanations. "
        "Return JSON only."
    )
    user_prompt = (
        f"{output_contract}\n\n"
        "Labeling guidance:\n"
        "- execution_status is whether the concrete action executed operationally, not whether the whole task is solved.\n"
        "- progress_signal is task-progress relevance after seeing the observation.\n"
        "- risk_signal marks confidentiality, policy, destructive, or irreversible risk visible from the action/observation.\n"
        "- For nudge labels, recommend the best next abstract behavior for recovery or progress, not a summary of how the trajectory ended.\n"
        "- Use observed_resulting_state, observed_tool_output, and observed_error_payload as evidence for both canonical_event_state and nudge labels.\n"
        "- Use recommended_abstract_action=finalize only when the observation gives explicit task-completion evidence, such as relational.task_completion.success=true, final verifier success, all required verifiers passing, task_success=true, or an equivalent benchmark completion signal.\n"
        "- Treat task-completion signals as nudge-only evidence; do not convert them into action-level execution_status unless the candidate action itself is a final evaluator/task wrapper.\n"
        "- EnterpriseOps-Gym task completion may be shown by successful completion of multiple verifiers; TerminalBench by final tests/reward/verifier success; CRMArenaPro by task_success=true or equivalent final evaluator success.\n"
        "- If the action failed, errored, had negative progress, or left stated problems unresolved, do not recommend finalize/proceed; choose inspect, search, retrieve, validate, clarify, avoid, or rollback as appropriate.\n"
        "- Do not use finalize merely because the trajectory stopped, a final response was emitted, or current_stage says completed while errors or unresolved problems remain.\n"
        "- EnterpriseOps-Gym actions are typed enterprise tool calls; tool return success/failure is usually the best execution evidence.\n"
        "- Terminal-Bench actions are shell/file/test actions; a command can execute successfully while validation still fails.\n"
        "- CRMArenaPro actions are CRM SQL/describe/respond actions; successful SQL execution is operational success even when rows are empty or the final evaluator later fails.\n"
        "- For CRMArenaPro, use SQL errors, schema drift, empty results, CRM object access, final response, and visible policy/confidentiality cues to label effect, missing information, and risk.\n"
        "- Do not turn trajectory-level task_success/task_score into action-level execution_status unless the candidate action is a final evaluator or task wrapper action.\n"
        "- Prefer unknown only when the provided evidence is genuinely insufficient.\n\n"
        "Example to label:\n"
        f"{json.dumps(example_payload, ensure_ascii=False, sort_keys=True)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(text)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM label response must be a JSON object")
    return payload


def validate_label_payload(payload: dict[str, Any], target_format: str) -> dict[str, Any]:
    if target_format == TARGET_CANONICAL_EVENT_WITH_NUDGE:
        return validate_canonical_event_with_nudge(payload)
    return validate_event_state(payload)


def heuristic_label(example: WorldModelStateExample, target_format: str) -> dict[str, Any]:
    if target_format == TARGET_CANONICAL_EVENT_WITH_NUDGE:
        return canonical_event_with_nudge_from_action_state(example.action, example.state)
    return canonical_event_from_action_state(example.action, example.state)


def uses_repo_gemini_client(model: str) -> bool:
    return model == "gemini"


def create_repo_gemini_client() -> Any:
    from src.llm import LLM

    return LLM("gemini")


def create_openai_client(args: argparse.Namespace) -> Any:
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("The `openai` package is required for LLM labeling.") from exc
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    kwargs = {"api_key": api_key}
    if args.api_base:
        kwargs["base_url"] = args.api_base
    return openai.OpenAI(**kwargs)


def json_schema_for_target(target_format: str) -> dict[str, Any]:
    canonical_event_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(REQUIRED_CATEGORICAL_FIELDS.keys()),
        "properties": {
            field: {"type": "string", "enum": sorted(values)}
            for field, values in REQUIRED_CATEGORICAL_FIELDS.items()
        },
    }
    if target_format == TARGET_CANONICAL_EVENT_STATE:
        schema = canonical_event_schema
    else:
        nudge_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "information_sufficiency",
                "information_gain",
                "recommended_abstract_action",
                "missing_information_type",
            ],
            "properties": {
                "information_sufficiency": {"type": "string", "enum": sorted(INFORMATION_SUFFICIENCY_VALUES)},
                "information_gain": {"type": "string", "enum": sorted(INFORMATION_GAIN_VALUES)},
                "recommended_abstract_action": {"type": "string", "enum": sorted(RECOMMENDED_ABSTRACT_ACTION_VALUES)},
                "missing_information_type": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "enum": sorted(MISSING_INFORMATION_TYPE_VALUES)},
                },
            },
        }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["canonical_event_state", "nudge"],
            "properties": {
                "canonical_event_state": canonical_event_schema,
                "nudge": nudge_schema,
            },
        }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": target_format,
            "strict": True,
            "schema": schema,
        },
    }


def call_repo_gemini(client: Any, messages: list[dict[str, str]]) -> str:
    system_prompt = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    user_prompt = "\n\n".join(message["content"] for message in messages if message["role"] != "system")
    return client._generate_gemini(
        user_prompt,
        system_prompt=system_prompt or None,
        endpoint=client.gemini_endpoint,
    )


def call_openai_chat(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_completion_tokens: int,
    reasoning_effort: str | None,
    response_format: str,
    target_format: str,
) -> str:
    if uses_repo_gemini_client(model):
        return call_repo_gemini(client, messages)

    request_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    if response_format == "json_schema":
        request_kwargs["response_format"] = json_schema_for_target(target_format)
    elif response_format == "json_object":
        request_kwargs["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        request_kwargs["reasoning_effort"] = reasoning_effort
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        request_kwargs.pop("temperature", None)
    response = client.chat.completions.create(**request_kwargs)
    return response.choices[0].message.content or ""


def label_one(
    example: WorldModelStateExample,
    *,
    split: str,
    args: argparse.Namespace,
    client_factory: Callable[[], Any],
) -> dict[str, Any]:
    example_payload = compact_example_payload(
        example,
        max_input_history_items=args.max_input_history_items,
        max_field_chars=args.max_field_chars,
    )
    messages = build_labeling_messages(example_payload, args.target_format)
    raw_response = ""
    last_error = ""
    for attempt in range(args.max_retries + 1):
        try:
            raw_response = call_openai_chat(
                client_factory(),
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_completion_tokens=args.max_completion_tokens,
                reasoning_effort=args.reasoning_effort,
                response_format=args.response_format,
                target_format=args.target_format,
            )
            label = validate_label_payload(parse_json_response(raw_response), args.target_format)
            label_source = "llm"
            break
        except Exception as exc:  # noqa: BLE001 - labeling scripts should retry SDK and validation failures.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= args.max_retries:
                if not args.fallback_to_heuristic:
                    raise RuntimeError(
                        f"Failed to label {example.trajectory_id}:{example.interaction_index}: {last_error}"
                    ) from exc
                label = heuristic_label(example, args.target_format)
                label_source = "heuristic_fallback"
                break
            delay = args.retry_base_delay * (2 ** attempt) + random.random()
            time.sleep(delay)

    row = {
        "split": split,
        "trajectory_id": example.trajectory_id,
        "trajectory_index": example.trajectory_index,
        "interaction_index": example.interaction_index,
        "benchmark": benchmark_name_from_trajectory_id(example.trajectory_id),
        "input_format": "system_task_history_action_v1",
        "target_format": args.target_format,
        "label_source": label_source,
        "label_model": args.model if label_source == "llm" else None,
        "system_prompt": example.system_prompt,
        "task_prompt": example.user_prompt,
        "state_history": [],
        "input_history": example.input_history,
        "previous_state": None,
        "action": example.action,
    }
    if args.target_format == TARGET_CANONICAL_EVENT_WITH_NUDGE:
        row["canonical_event_with_nudge"] = label
        row["canonical_event_state"] = label["canonical_event_state"]
        row["nudge"] = label["nudge"]
    else:
        row["canonical_event_state"] = label
    if args.include_raw_response:
        row["raw_llm_response"] = raw_response
    if label_source == "heuristic_fallback":
        row["llm_label_error"] = last_error
    return row


def make_client_factory(args: argparse.Namespace) -> Callable[[], Any]:
    client_holder: dict[str, Any] = {}

    def get_client() -> Any:
        if "client" not in client_holder:
            if uses_repo_gemini_client(args.model):
                client_holder["client"] = create_repo_gemini_client()
            else:
                client_holder["client"] = create_openai_client(args)
        return client_holder["client"]

    return get_client


def limited_examples(examples: list[WorldModelStateExample], limit: int | None) -> list[WorldModelStateExample]:
    return examples if limit is None else examples[:limit]


def build_rows(
    examples: list[WorldModelStateExample],
    *,
    split: str,
    args: argparse.Namespace,
    append_output_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    client_factory = make_client_factory(args)
    if args.workers <= 1:
        for index, example in enumerate(examples, start=1):
            row = label_one(example, split=split, args=args, client_factory=client_factory)
            rows.append(row)
            if append_output_path is not None:
                write_jsonl_rows(append_output_path, [row], append=True)
            if index % 25 == 0 or index == len(examples):
                print(f"[{split}] labeled {index}/{len(examples)}")
        return rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_index = {
            executor.submit(label_one, example, split=split, args=args, client_factory=make_client_factory(args)): index
            for index, example in enumerate(examples)
        }
        completed = 0
        ordered: dict[int, dict[str, Any]] = {}
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            row = future.result()
            ordered[index] = row
            if append_output_path is not None:
                write_jsonl_rows(append_output_path, [row], append=True)
            completed += 1
            if completed % 25 == 0 or completed == len(examples):
                print(f"[{split}] labeled {completed}/{len(examples)}")
        rows = [ordered[index] for index in sorted(ordered)]
    return rows


def target_value_counts(rows: list[dict[str, Any]], target_format: str) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = {}
    for row in rows:
        event = row["canonical_event_state"]
        for field in REQUIRED_CATEGORICAL_FIELDS:
            counters.setdefault(f"canonical_event_state.{field}", Counter())[event[field]] += 1
        if target_format == TARGET_CANONICAL_EVENT_WITH_NUDGE:
            nudge = row["nudge"]
            for field in ("information_sufficiency", "information_gain", "recommended_abstract_action"):
                counters.setdefault(f"nudge.{field}", Counter())[nudge[field]] += 1
            for value in nudge["missing_information_type"]:
                counters.setdefault("nudge.missing_information_type", Counter())[value] += 1
    return {field: dict(sorted(counter.items())) for field, counter in sorted(counters.items())}


def main() -> None:
    args = parse_args()
    default_train_path, default_eval_path, default_manifest_path = default_output_paths(
        args.target_format,
        args.trajectory_dataset,
    )
    args.train_output_path = args.train_output_path or default_train_path
    args.eval_output_path = args.eval_output_path or default_eval_path
    args.manifest_path = args.manifest_path or default_manifest_path

    preset_train, preset_eval = TRAJECTORY_DATASET_PRESETS[args.trajectory_dataset]
    train_paths = args.train_data_path or list(preset_train)
    eval_paths = args.eval_data_path or list(preset_eval)

    train_trajectories = load_trajectories(train_paths)
    eval_trajectories = load_trajectories(eval_paths)
    train_examples = limited_examples(
        extract_state_examples(train_trajectories, state_history_size=args.state_history_size),
        args.max_train_examples,
    )
    eval_examples = limited_examples(
        extract_state_examples(eval_trajectories, state_history_size=args.state_history_size),
        args.max_eval_examples,
    )

    print(f"Prepared {len(train_examples)} train examples and {len(eval_examples)} eval examples")
    if args.dry_run:
        sample = train_examples[0] if train_examples else eval_examples[0] if eval_examples else None
        if sample is None:
            print("No examples available for dry run.")
            return
        payload = compact_example_payload(
            sample,
            max_input_history_items=args.max_input_history_items,
            max_field_chars=args.max_field_chars,
        )
        print(json.dumps(build_labeling_messages(payload, args.target_format), indent=2, ensure_ascii=False))
        print(json.dumps(json_schema_for_target(args.target_format), indent=2, ensure_ascii=False))
        return

    existing_train_rows = dedupe_rows_by_key(load_jsonl_rows(args.train_output_path)) if args.resume else []
    existing_eval_rows = dedupe_rows_by_key(load_jsonl_rows(args.eval_output_path)) if args.resume else []
    existing_train_keys = {row_key(row) for row in existing_train_rows}
    existing_eval_keys = {row_key(row) for row in existing_eval_rows}
    pending_train_examples = [
        example for example in train_examples if example_key(example, "train") not in existing_train_keys
    ]
    pending_eval_examples = [
        example for example in eval_examples if example_key(example, "eval") not in existing_eval_keys
    ]

    if args.resume:
        print(
            f"Resume enabled: loaded {len(existing_train_rows)} train rows and {len(existing_eval_rows)} eval rows; "
            f"labeling {len(pending_train_examples)} train and {len(pending_eval_examples)} eval pending examples"
        )

    new_train_rows = build_rows(
        pending_train_examples,
        split="train",
        args=args,
        append_output_path=args.train_output_path if args.resume else None,
    )
    new_eval_rows = build_rows(
        pending_eval_examples,
        split="eval",
        args=args,
        append_output_path=args.eval_output_path if args.resume else None,
    )
    train_rows = dedupe_rows_by_key(existing_train_rows + new_train_rows)
    eval_rows = dedupe_rows_by_key(existing_eval_rows + new_eval_rows)

    if not args.resume:
        dump_jsonl(args.train_output_path, train_rows)
        dump_jsonl(args.eval_output_path, eval_rows)

    manifest = {
        "trajectory_dataset": args.trajectory_dataset,
        "source_train_paths": [str(path) for path in train_paths],
        "source_eval_paths": [str(path) for path in eval_paths],
        "train_trajectory_count": len(train_trajectories),
        "eval_trajectory_count": len(eval_trajectories),
        "train_example_count": len(train_rows),
        "eval_example_count": len(eval_rows),
        "new_train_example_count": len(new_train_rows),
        "new_eval_example_count": len(new_eval_rows),
        "pending_train_example_count_after_run": max(0, len(train_examples) - len(train_rows)),
        "pending_eval_example_count_after_run": max(0, len(eval_examples) - len(eval_rows)),
        "resume": args.resume,
        "target_format": args.target_format,
        "label_model": args.model,
        "label_provider": "openai_chat_completions",
        "fallback_to_heuristic": args.fallback_to_heuristic,
        "include_raw_response": args.include_raw_response,
        "free_text_target_fields": [],
        "train_label_sources": dict(Counter(row["label_source"] for row in train_rows)),
        "eval_label_sources": dict(Counter(row["label_source"] for row in eval_rows)),
        "train_benchmark_counts": benchmark_counts(train_rows),
        "eval_benchmark_counts": benchmark_counts(eval_rows),
        "train_value_counts": target_value_counts(train_rows, args.target_format),
        "eval_value_counts": target_value_counts(eval_rows, args.target_format),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"api_base"}
        },
    }
    dump_json(args.manifest_path, manifest)
    verb = "Updated" if args.resume else "Wrote"
    print(f"{verb} {len(train_rows)} train examples at {args.train_output_path} ({len(new_train_rows)} new)")
    print(f"{verb} {len(eval_rows)} eval examples at {args.eval_output_path} ({len(new_eval_rows)} new)")
    print(f"Wrote manifest to {args.manifest_path}")


if __name__ == "__main__":
    main()
