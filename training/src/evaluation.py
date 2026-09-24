#!/usr/bin/env python3
"""Evaluate EnterpriseOps-Gym world models.

This module owns next-state prediction evaluation and EnterpriseOps-Gym agent
replay. Shared trajectory parsing and prompt construction live in
`src.finetuning` so training and evaluation use the same example format.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.calculate_replay_metrics import summarize_mode as summarize_replay_mode_metrics
from src.data_preparation.canonical_event_state import canonical_field_matches
from src.finetuning import (
    DEFAULT_ENTERPRISEOPS_GYM_REPO_PATH,
    DEFAULT_ENTERPRISEOPS_GYM_TASK_CONFIGS_DIR,
    DEFAULT_ENTERPRISEOPS_GYM_TASK_SPLIT_MANIFEST,
    DEFAULT_STATE_HISTORY_SIZE,
    DEFAULT_TRAJECTORIES_DIR,
    LEGACY_WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
    OUTCOME_ERROR_MATCH_JUDGE_KEYS,
    STATE_MATCH_JUDGE_KEYS,
    TASK_COMPLETION_JUDGE_KEYS,
    TOOL_OUTPUT_JUDGE_CHAR_LIMIT,
    TOOL_OUTPUT_MATCH_JUDGE_KEYS,
    TRAJECTORY_DATASET_PRESETS,
    WORLD_MODEL_INPUT_HISTORY_SIZE,
    WORLD_MODEL_TARGET_CANONICAL_EVENT_STATE,
    WORLD_MODEL_TARGET_CANONICAL_EVENT_WITH_NUDGE,
    WORLD_MODEL_TARGET_STATE,
    WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
    WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY,
    WORLD_MODEL_TARGET_TOOL_OUTPUT,
    WorldModelStateExample,
    canonical_event_from_action_state,
    canonical_event_json,
    canonical_event_with_nudge_from_action_state,
    canonical_event_with_nudge_json,
    _REPLAY_LIMITS,
    _is_agent_decision_shaped,
    append_state_history,
    append_world_model_input_history,
    apply_chat_template_or_fallback,
    build_state_prediction_chat_messages,
    canonicalize_world_model_target,
    dump_json,
    dump_jsonl,
    emit_progress,
    ensure_peft_weight_converter_compatibility,
    enterprise_state_delta_from_any,
    extract_last_tool_execution_result_from_state,
    extract_replay_tasks_from_state_trajectories,
    extract_state_examples,
    filter_replay_tasks_by_gym_task_split,
    format_tool_execution_result_target,
    infer_execution_result_success,
    is_canonical_event_state_target,
    is_canonical_event_with_nudge_target,
    is_compact_tool_execution_state,
    is_enterprise_state_payload,
    is_tool_execution_result_target,
    is_tool_output_target,
    iter_balanced_json_objects,
    json_compact,
    load_json,
    make_blank_state,
    make_blank_state_like,
    make_imagined_world_model_history_entry,
    make_tool_execution_prediction_state,
    normalize_last_tool_execution_result,
    normalize_loaded_trajectories,
    normalize_state_text,
    normalize_tool_call,
    normalize_tool_execution_result_for_target,
    override_enterpriseops_gym_mcp_urls,
    parse_binary_world_model_prediction,
    parse_jsonish,
    parse_thought_payload,
    preview_tool_calls,
    render_messages,
    require_training_stack,
    resolve_inference_device_map,
    resolve_text_generation_model_class,
    resolve_torch_dtype,
    sanitize_state_content,
    state_body_from_any,
    state_context_from_any,
    state_current_stage,
    state_is_finished,
    state_process_from_any,
    state_remaining_stages,
    stringify_tool_output,
    strip_action_wrappers,
    strip_code_fence,
    strip_model_thinking_output,
    to_openai_tool_calls,
    tool_output_looks_like_failure,
    tqdm,
    truncate_for_replay,
)

API_AGENT_MODEL_METHODS = {"gemini", "claude", "vllm/nemotron3-nano-4B-BF16", "vllm/qwen3-8b", "vllm/gymops_world_model"}
OPENAI_AGENT_MODEL_ALIASES = {
    "gpt-5": "gpt-5",
    "openai/gpt5": "gpt-5",
    "openai:gpt5": "gpt-5",
    "openai/gpt-5": "gpt-5",
    "openai:gpt-5": "gpt-5",
    "gpt5.1": "gpt-5.1",
    "gpt-5.1": "gpt-5.1",
    "openai/gpt5.1": "gpt-5.1",
    "openai:gpt5.1": "gpt-5.1",
    "openai/gpt-5.1": "gpt-5.1",
    "openai:gpt-5.1": "gpt-5.1",
}
LLM_AGENT_MODEL_ALIASES: dict[str, str] = {
    "gpt5": "gpt5",
}
def dump_imagined_trajectories(mode_name: str, payload: list[dict[str, Any]]) -> Path:
    path = DEFAULT_TRAJECTORIES_DIR / f"{mode_name}_trajectories.json"
    dump_json(path, payload)
    return path


def dump_replay_trajectories(mode_name: str, payload: list[dict[str, Any]]) -> Path:
    """Persist the actual replay conversations (per task) for one execution mode."""
    path = DEFAULT_TRAJECTORIES_DIR / f"{mode_name}_replay_trajectories.json"
    dump_json(path, payload)
    return path


def _extract_tool_call_sequence_from_conversation_flow(
    conversation_flow: list[dict[str, Any]] | None,
) -> list[list[str]]:
    sequence: list[list[str]] = []
    for message in conversation_flow or []:
        if not isinstance(message, dict):
            continue
        if message.get("type") != "ai_message":
            continue
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            continue
        names = [call.get("name", "") for call in tool_calls if isinstance(call, dict)]
        if names:
            sequence.append(names)
    return sequence


def build_agent_replay_strategy_records(
    replay_eval: dict[str, Any],
    strategy: str,
) -> list[dict[str, Any]]:
    mode_payload = replay_eval.get(strategy) or {}
    task_records = mode_payload.get("task_records") or []
    records: list[dict[str, Any]] = []

    for task_record in task_records:
        if "result" in task_record:
            result = task_record.get("result") or {}
            benchmark_config = result.get("benchmark_config") or {}
            runs = result.get("runs") or []
            statistics = result.get("statistics") or {}
            base = {
                "strategy": strategy,
                "trajectory_index": task_record.get("trajectory_index"),
                "gym_task_config_name": task_record.get("gym_task_config_name"),
                "task_query": benchmark_config.get("user_prompt"),
                "statistics": statistics,
            }
            for run_index, run in enumerate(runs, start=1):
                record = dict(base)
                record.update(
                    {
                        "run_index": run_index,
                        "completed": run.get("overall_success"),
                        "verification_summary": run.get("verification_summary"),
                        "steps_taken": run.get("steps_taken"),
                        "internal_thinking_iterations": run.get("internal_thinking_iterations"),
                        "wm_revisions_applied": run.get("wm_revisions_applied"),
                        "wm_imagined_rollouts": run.get("wm_imagined_rollouts"),
                        "wm_revision_step_details": run.get("wm_revision_step_details") or [],
                        "imagined_rollout_records": run.get("imagined_rollout_records") or [],
                        "model_response": run.get("model_response"),
                        "tool_call_sequence": _extract_tool_call_sequence_from_conversation_flow(
                            run.get("conversation_flow")
                        ),
                    }
                )
                records.append(record)
            continue

        records.append(
            {
                "strategy": strategy,
                "trajectory_index": task_record.get("trajectory_index"),
                "completed": task_record.get("completed"),
                "failure_reason": task_record.get("failure_reason"),
                "final_answer_score": task_record.get("final_answer_score"),
                "final_answer_evaluation": task_record.get("final_answer_evaluation"),
                "tool_steps_taken": task_record.get("tool_steps_taken"),
                "tool_calls_taken": task_record.get("tool_calls_taken"),
                "internal_thinking_iterations": task_record.get("internal_thinking_iterations"),
                "imagined_rollouts_used": task_record.get("imagined_rollouts_used"),
                "wm_predicted_failures": task_record.get("wm_predicted_failures"),
                "wm_predicted_successes": task_record.get("wm_predicted_successes"),
                "wm_triggered_revisions": task_record.get("wm_triggered_revisions"),
                "wm_no_op_revisions": task_record.get("wm_no_op_revisions"),
                "wm_revision_effectiveness": task_record.get("wm_revision_effectiveness"),
                "wm_revision_step_details": task_record.get("wm_revision_step_details") or [],
                "imagined_rollout_records": task_record.get("imagined_rollout_records") or [],
            }
        )
    return records


def dump_agent_replay_strategy_records(output_dir: Path, replay_eval: dict[str, Any]) -> None:
    for strategy in ("revision", "imagined"):
        records = build_agent_replay_strategy_records(replay_eval, strategy)
        dump_json(output_dir / f"agent_replay_{strategy}_records.json", records)
        dump_jsonl(output_dir / f"agent_replay_{strategy}_records.jsonl", records)


def calculate_agent_replay_metrics(replay_eval: dict[str, Any]) -> dict[str, Any]:
    metrics_by_mode: dict[str, Any] = {}
    for mode in ("baseline", "revision", "imagined"):
        mode_payload = replay_eval.get(mode)
        if not isinstance(mode_payload, dict):
            continue
        if not isinstance(mode_payload.get("task_records"), list):
            continue
        metrics_by_mode[mode] = summarize_replay_mode_metrics(mode, mode_payload)
    return {
        "modes": metrics_by_mode,
        "summaries": list(metrics_by_mode.values()),
    }


def dump_agent_replay_metrics(output_dir: Path, replay_eval: dict[str, Any]) -> dict[str, Any]:
    replay_metrics = calculate_agent_replay_metrics(replay_eval)
    dump_json(output_dir / "agent_replay_metrics.json", replay_metrics)
    return replay_metrics


def build_task_completion_judge_prompt() -> str:
    return (
        "You are evaluating how well an AI agent completed the assigned task. "
        "Focus on the final outcome and how well it addresses the original requirements.\n\n"
        "**TASK COMPLETION EVALUATION CRITERIA:**\n"
        "- requirement_coverage: Did the agent address all aspects of the task? (0.0-1.0)\n"
        "- accuracy: Is the information and analysis factually correct? (0.0-1.0)\n"
        "- completeness: Is the response thorough and comprehensive? (0.0-1.0)\n"
        "- usefulness: Is the final result practically valuable to the user? (0.0-1.0)\n\n"
        "Compare the final response against:\n"
        "- Original task requirements\n"
        "- Expected deliverables\n"
        "- Quality of information provided\n"
        "- Practical utility for the user\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        '  "requirement_coverage": float,\n'
        '  "accuracy": float,\n'
        '  "completeness": float,\n'
        '  "usefulness": float,\n'
        '  "comments": "Brief explanation"\n'
        "}"
    )


def build_outcome_error_match_judge_prompt() -> str:
    return (
        "You are evaluating whether a predicted tool-execution outcome matches the gold outcome.\n\n"
        "Each outcome has a label and (when the label denotes failure or stagnation) an error message "
        "or API response that explains what went wrong. The label is `1` for success, `0` for "
        "stagnation, or `-1` for explicit tool failure. When the label is `1` (success), the error "
        "message is empty and only the label needs to match.\n\n"
        "EVALUATION CRITERIA:\n"
        "- label_alignment: Does the predicted label match the gold label? (0.0-1.0)\n"
        "- error_alignment: Does the predicted error message describe the same problem as the gold "
        "error message? Ignore harmless wording, formatting, identifier, and timestamp differences. "
        "(0.0-1.0)\n"
        "- semantic_consistency: Overall, does the prediction convey the same outcome and the same "
        "underlying reason as the gold? (0.0-1.0)\n\n"
        "Set match=true ONLY when the label is correct AND, if the label is `0` or `-1`, the error "
        "message captures the same problem semantically. If the gold label is `1`, judge match purely "
        "on label correctness.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        '  "label_alignment": float,\n'
        '  "error_alignment": float,\n'
        '  "semantic_consistency": float,\n'
        '  "match": boolean,\n'
        '  "comments": "Brief explanation"\n'
        "}"
    )


def build_tool_output_match_judge_prompt() -> str:
    return (
        "You are evaluating whether a predicted tool-execution output (a free-form natural-language "
        "or JSON-stringified payload returned by an enterprise tool call) matches the gold output.\n\n"
        "Judge semantic equivalence rather than strict string equality. Ignore harmless wording, "
        "formatting, key ordering, whitespace, identifier, and timestamp differences. Numeric values "
        "and named entities that drive downstream behavior must still agree.\n\n"
        "EVALUATION CRITERIA:\n"
        "- semantic_equivalence: Do the two outputs convey the same information? (0.0-1.0)\n"
        "- factual_consistency: Are the load-bearing facts (IDs, counts, statuses, names, key fields) "
        "consistent between prediction and gold? (0.0-1.0)\n"
        "- intent_alignment: Does the predicted output answer the same question / serve the same "
        "purpose for the action that produced it? (0.0-1.0)\n"
        "- outcome_alignment: Do the two outputs imply the same success/failure outcome of the tool "
        "call (e.g., both report success, or both surface the same kind of error)? (0.0-1.0)\n\n"
        "Set match=true only when the predicted output is a strong semantic match overall and "
        "preserves the load-bearing facts a downstream agent would rely on.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        '  "semantic_equivalence": float,\n'
        '  "factual_consistency": float,\n'
        '  "intent_alignment": float,\n'
        '  "outcome_alignment": float,\n'
        '  "match": boolean,\n'
        '  "comments": "Brief explanation"\n'
        "}"
    )


def build_state_match_judge_prompt() -> str:
    return (
        "You are evaluating whether a predicted enterprise task state matches the target state.\n\n"
        "Score at FIELD LEVEL, not as one whole-schema score. The user message provides a "
        "`Fields to score` object whose keys are target field paths. Return exactly one score for "
        "each listed field path. Judge semantic equivalence rather than strict string equality; "
        "ignore harmless wording, formatting, key ordering, and minor paraphrases. Numeric values, "
        "IDs, named entities, stage status, success/failure labels, and unresolved "
        "requirements must preserve the same operational meaning.\n\n"
        "For each field:\n"
        "- score: 1.0 means the predicted field is semantically correct for that target field.\n"
        "- score: 0.5 means partially correct or too vague to safely drive downstream behavior.\n"
        "- score: 0.0 means missing, contradictory, or operationally wrong.\n"
        "- match: true only when score is at least 0.8.\n\n"
        "overall_score must be the arithmetic mean of the per-field scores you returned. "
        "overall_match must be true only when every listed field has match=true.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        '  "field_scores": {\n'
        '    "state.aspect.field": {"score": float, "match": boolean, "comments": "Brief explanation"}\n'
        "  },\n"
        '  "overall_score": float,\n'
        '  "overall_match": boolean,\n'
        '  "comments": "Brief explanation"\n'
        "}"
    )


@lru_cache(maxsize=1)
def get_task_completion_judge_client() -> Any:
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("The `openai` package is required for LLM-judge final-answer evaluation.") from exc

    return openai.OpenAI()


def evaluate_final_answer_quality(
    task_description: str,
    final_response: str,
    ground_truth_answer: str,
    execution_trajectory: list[dict[str, Any]] | None = None,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    if not final_response.strip():
        zero_scores = {key: 0.0 for key in TASK_COMPLETION_JUDGE_KEYS}
        return {
            "scores": zero_scores,
            "overall_score": 0.0,
            "comments": "No final answer provided",
            "raw_response": {},
        }

    user_parts = [
        "Task description:\n" + (task_description or ""),
        "Agent's final response:\n" + final_response,
    ]
    if execution_trajectory:
        user_parts.append(
            "Full execution trajectory:\n"
            + json.dumps(execution_trajectory, ensure_ascii=False, indent=2)
        )
    if ground_truth_answer:
        user_parts.append("Expected answer:\n" + ground_truth_answer)

    messages = [
        {"role": "system", "content": build_task_completion_judge_prompt()},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]

    try:
        client = get_task_completion_judge_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        raw_response = parse_jsonish(strip_code_fence(content))
        if not isinstance(raw_response, dict):
            raise ValueError("Expected a JSON object from task completion judge.")
        scores = {
            key: float(raw_response.get(key, 0.0))
            for key in TASK_COMPLETION_JUDGE_KEYS
        }
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        return {
            "scores": scores,
            "overall_score": overall_score,
            "comments": str(raw_response.get("comments", "")),
            "raw_response": raw_response,
        }
    except Exception as exc:
        zero_scores = {key: 0.0 for key in TASK_COMPLETION_JUDGE_KEYS}
        return {
            "scores": zero_scores,
            "overall_score": 0.0,
            "comments": f"Error: {exc}",
            "raw_response": {},
        }


def evaluate_outcome_error_match_quality(
    system_prompt: str,
    user_prompt: str,
    previous_state: dict[str, Any],
    action: Any,
    predicted_label: int | None,
    predicted_error_message: str,
    gold_label: int | None,
    gold_error_message: str,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    zero_scores = {key: 0.0 for key in OUTCOME_ERROR_MATCH_JUDGE_KEYS}
    if predicted_label is None:
        return {
            "scores": zero_scores,
            "overall_score": 0.0,
            "match": False,
            "comments": "No predicted label parsed",
            "raw_response": {},
        }

    action_text = (
        action
        if isinstance(action, str)
        else json.dumps(action, ensure_ascii=False, indent=2)
    )
    messages = [
        {"role": "system", "content": build_outcome_error_match_judge_prompt()},
        {
            "role": "user",
            "content": (
                "System prompt:\n"
                + (system_prompt or "")
                + "\n\nUser prompt:\n"
                + (user_prompt or "")
                + "\n\nPrevious state:\n"
                + normalize_state_text(previous_state)
                + "\n\nAction:\n"
                + action_text
                + f"\n\nPredicted label: {predicted_label}"
                + f"\nPredicted error/response: {predicted_error_message or '(empty)'}"
                + f"\n\nGold label: {gold_label}"
                + f"\nGold error/response: {gold_error_message or '(empty)'}"
            ),
        },
    ]

    try:
        client = get_task_completion_judge_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        raw_response = parse_jsonish(strip_code_fence(content))
        if not isinstance(raw_response, dict):
            raise ValueError("Expected a JSON object from outcome error match judge.")
        scores = {
            key: float(raw_response.get(key, 0.0))
            for key in OUTCOME_ERROR_MATCH_JUDGE_KEYS
        }
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        return {
            "scores": scores,
            "overall_score": overall_score,
            "match": bool(raw_response.get("match", False)),
            "comments": str(raw_response.get("comments", "")),
            "raw_response": raw_response,
        }
    except Exception as exc:
        return {
            "scores": zero_scores,
            "overall_score": 0.0,
            "match": False,
            "comments": f"Error: {exc}",
            "raw_response": {},
        }


def flatten_state_fields(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        fields: dict[str, Any] = {}
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            fields.update(flatten_state_fields(value[key], child_prefix))
        return fields
    return {prefix: value} if prefix else {"$": value}


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def normalize_state_field_judge_response(
    raw_response: dict[str, Any],
    target_fields: dict[str, Any],
) -> dict[str, Any]:
    raw_field_scores = raw_response.get("field_scores", {})
    if not isinstance(raw_field_scores, dict):
        raw_field_scores = {}

    field_scores: dict[str, dict[str, Any]] = {}
    for field_path in target_fields:
        raw_entry = raw_field_scores.get(field_path, {})
        if isinstance(raw_entry, dict):
            score = clamp_score(raw_entry.get("score", 0.0))
            comments = str(raw_entry.get("comments", ""))
            match = bool(raw_entry.get("match", score >= 0.8))
        else:
            score = clamp_score(raw_entry)
            comments = ""
            match = score >= 0.8
        field_scores[field_path] = {
            "score": score,
            "match": match,
            "comments": comments,
        }

    if field_scores:
        overall_score = sum(item["score"] for item in field_scores.values()) / len(field_scores)
        overall_match = all(item["match"] for item in field_scores.values())
    else:
        overall_score = 0.0
        overall_match = False

    return {
        "scores": {"field_accuracy": overall_score},
        "field_scores": field_scores,
        "overall_score": overall_score,
        "match": bool(raw_response.get("overall_match", overall_match)) and overall_match,
        "comments": str(raw_response.get("comments", "")),
        "raw_response": raw_response,
    }


def evaluate_state_match_quality(
    system_prompt: str,
    user_prompt: str,
    previous_state: dict[str, Any],
    action: Any,
    predicted_state_text: str,
    target_state_text: str,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    try:
        target_state = sanitize_state_content(parse_jsonish(target_state_text))
        target_fields = flatten_state_fields(target_state)
    except Exception:
        target_fields = {}

    if not predicted_state_text.strip():
        raw_response = {
            "field_scores": {
                field_path: {
                    "score": 0.0,
                    "match": False,
                    "comments": "No predicted state provided",
                }
                for field_path in target_fields
            },
            "overall_match": False,
            "comments": "No predicted state provided",
        }
        return normalize_state_field_judge_response(raw_response, target_fields)

    try:
        predicted_state = sanitize_state_content(parse_jsonish(predicted_state_text))
        predicted_fields = flatten_state_fields(predicted_state)
    except Exception:
        predicted_fields = {}

    action_text = (
        action
        if isinstance(action, str)
        else json.dumps(action, ensure_ascii=False, indent=2)
    )
    messages = [
        {"role": "system", "content": build_state_match_judge_prompt()},
        {
            "role": "user",
            "content": (
                "System prompt:\n"
                + (system_prompt or "")
                + "\n\nUser prompt:\n"
                + (user_prompt or "")
                + "\n\nPrevious state:\n"
                + normalize_state_text(previous_state)
                + "\n\nAction:\n"
                + action_text
                + "\n\nFields to score:\n"
                + json.dumps(target_fields, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n\nPredicted field values:\n"
                + json.dumps(predicted_fields, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n\nTarget field values:\n"
                + json.dumps(target_fields, ensure_ascii=False, indent=2, sort_keys=True)
            ),
        },
    ]

    try:
        client = get_task_completion_judge_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        raw_response = parse_jsonish(strip_code_fence(content))
        if not isinstance(raw_response, dict):
            raise ValueError("Expected a JSON object from state field match judge.")
        return normalize_state_field_judge_response(raw_response, target_fields)
    except Exception as exc:
        raw_response = {
            "field_scores": {
                field_path: {
                    "score": 0.0,
                    "match": False,
                    "comments": f"Error: {exc}",
                }
                for field_path in target_fields
            },
            "overall_match": False,
            "comments": f"Error: {exc}",
        }
        return normalize_state_field_judge_response(raw_response, target_fields)

def evaluate_tool_output_match_quality(
    system_prompt: str,
    user_prompt: str,
    previous_state: dict[str, Any],
    action: Any,
    predicted_tool_output: str,
    target_tool_output: str,
    model: str = "gpt-4o",
    char_limit: int = TOOL_OUTPUT_JUDGE_CHAR_LIMIT,
) -> dict[str, Any]:
    zero_scores = {key: 0.0 for key in TOOL_OUTPUT_MATCH_JUDGE_KEYS}
    if not predicted_tool_output.strip():
        return {
            "scores": zero_scores,
            "overall_score": 0.0,
            "match": False,
            "comments": "No predicted tool output provided",
            "raw_response": {},
        }

    action_text = (
        action
        if isinstance(action, str)
        else json.dumps(action, ensure_ascii=False, indent=2)
    )
    messages = [
        {"role": "system", "content": build_tool_output_match_judge_prompt()},
        {
            "role": "user",
            "content": (
                "System prompt:\n"
                + (system_prompt or "")
                + "\n\nUser prompt:\n"
                + (user_prompt or "")
                + "\n\nPrevious state:\n"
                + truncate_for_replay(normalize_state_text(previous_state), char_limit)
                + "\n\nAction:\n"
                + truncate_for_replay(action_text, char_limit)
                + "\n\nPredicted tool output:\n"
                + truncate_for_replay(predicted_tool_output, char_limit)
                + "\n\nGold tool output:\n"
                + truncate_for_replay(target_tool_output, char_limit)
            ),
        },
    ]

    try:
        client = get_task_completion_judge_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        raw_response = parse_jsonish(strip_code_fence(content))
        if not isinstance(raw_response, dict):
            raise ValueError("Expected a JSON object from tool output match judge.")
        scores = {
            key: float(raw_response.get(key, 0.0))
            for key in TOOL_OUTPUT_MATCH_JUDGE_KEYS
        }
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        return {
            "scores": scores,
            "overall_score": overall_score,
            "match": bool(raw_response.get("match", False)),
            "comments": str(raw_response.get("comments", "")),
            "raw_response": raw_response,
        }
    except Exception as exc:
        return {
            "scores": zero_scores,
            "overall_score": 0.0,
            "match": False,
            "comments": f"Error: {exc}",
            "raw_response": {},
        }


def missing_argument_paths(predicted: Any, gold: Any, prefix: str = "") -> list[str]:
    if isinstance(gold, dict):
        if not isinstance(predicted, dict):
            return [prefix.rstrip(".") or "<root>"]
        missing: list[str] = []
        for key, gold_value in gold.items():
            child_prefix = f"{prefix}{key}"
            if key not in predicted:
                missing.append(child_prefix)
            else:
                missing.extend(missing_argument_paths(predicted[key], gold_value, f"{child_prefix}."))
        return missing
    if isinstance(gold, list):
        if not isinstance(predicted, list) or len(predicted) < len(gold):
            return [prefix.rstrip(".") or "<root>"]
        missing: list[str] = []
        for index, gold_value in enumerate(gold):
            missing.extend(missing_argument_paths(predicted[index], gold_value, f"{prefix}{index}."))
        return missing
    if predicted != gold:
        return [prefix.rstrip(".") or "<root>"]
    return []


def tool_calls_match(predicted_calls: list[dict[str, Any]], gold_calls: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    if len(predicted_calls) != len(gold_calls):
        return False, [f"count:{len(predicted_calls)}!={len(gold_calls)}"]
    failures: list[str] = []
    for index, gold_call in enumerate(gold_calls):
        pred_call = predicted_calls[index]
        if pred_call.get("name") != gold_call.get("name"):
            failures.append(f"name[{index}]")
            continue
        failures.extend(missing_argument_paths(pred_call.get("arguments"), gold_call.get("arguments"), prefix=f"args[{index}]."))
    return not failures, failures


def normalize_agent_action_name(action: str) -> str:
    return re.sub(r"[\s_-]+", " ", action).strip().lower()


def parse_malformed_final_answer_decision(cleaned: str) -> dict[str, Any] | None:
    action_match = re.search(
        r"[\"']action[\"']\s*:\s*[\"']([^\"']+)[\"']",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not action_match:
        return None
    if normalize_agent_action_name(action_match.group(1)) not in {"final answer", "final", "answer"}:
        return None

    input_match = re.search(
        r"[\"']action_input[\"']\s*:",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not input_match:
        return {"final_answer": ""}

    cursor = input_match.end()
    while cursor < len(cleaned) and cleaned[cursor].isspace():
        cursor += 1
    if cursor >= len(cleaned):
        return {"final_answer": ""}

    quote = cleaned[cursor] if cleaned[cursor] in {'\"', "'"} else ""
    if quote:
        tail = cleaned[cursor + 1 :].strip()
        closing = re.match(rf"(?s)(.*){re.escape(quote)}\s*}}\s*$", tail)
        if closing:
            answer = closing.group(1)
        else:
            answer = tail
            if answer.endswith("}"):
                answer = answer[:-1].rstrip()
            if answer.endswith(quote):
                answer = answer[:-1]
        return {"final_answer": answer}

    tail = cleaned[cursor:].strip()
    if tail.endswith("}"):
        tail = tail[:-1].rstrip()
    try:
        parsed_tail = parse_jsonish(tail)
    except Exception:
        parsed_tail = tail
    if isinstance(parsed_tail, dict):
        parsed_tail = json.dumps(parsed_tail, ensure_ascii=False)
    return {"final_answer": str(parsed_tail)}


def infer_tool_call_from_argument_object(payload: dict[str, Any]) -> dict[str, Any] | None:
    keys = set(payload.keys())
    if {"calendarId", "start", "end", "summary"}.issubset(keys):
        if "eventId" in keys:
            return {"tool_calls": [normalize_tool_call({"name": "patch_event", "arguments": payload})]}
        return {"tool_calls": [normalize_tool_call({"name": "create_event", "arguments": payload})]}
    return None


def parse_agent_decision(raw_text: str) -> dict[str, Any]:
    cleaned = strip_action_wrappers(raw_text)

    parsed: Any = None
    try:
        parsed = parse_jsonish(cleaned)
    except ValueError:
        parsed = None

    if not _is_agent_decision_shaped(parsed):
        first_parseable: Any = None
        for candidate in iter_balanced_json_objects(cleaned):
            if _is_agent_decision_shaped(candidate):
                parsed = candidate
                break
            if first_parseable is None:
                first_parseable = candidate
        else:
            if not _is_agent_decision_shaped(parsed) and first_parseable is not None:
                parsed = first_parseable

    if parsed is None:
        recovered = parse_malformed_final_answer_decision(cleaned)
        if recovered is not None:
            return recovered
        raise ValueError(f"Unable to parse JSON from model output: {raw_text[:200]}")

    if isinstance(parsed, list):
        return {"tool_calls": [normalize_tool_call(item) for item in parsed]}
    if isinstance(parsed, dict):
        if "action" in parsed:
            action = str(parsed.get("action", "")).strip()
            normalized_action = normalize_agent_action_name(action)
            action_input = parsed.get("action_input", {})
            if normalized_action in {"final answer", "final", "answer"}:
                if isinstance(action_input, dict):
                    action_input = json.dumps(action_input, ensure_ascii=False)
                return {"final_answer": str(action_input)}
            if normalized_action == "clarify":
                question = action_input.get("question", "") if isinstance(action_input, dict) else str(action_input)
                return {"clarify": question}
            return {"tool_calls": [normalize_tool_call({"name": action, "arguments": action_input})]}
        if "tool_calls" in parsed:
            tool_calls = parsed["tool_calls"] or []
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]
            return {"tool_calls": [normalize_tool_call(item) for item in tool_calls]}
        if "final_answer" in parsed:
            return {"final_answer": str(parsed["final_answer"])}
        if "name" in parsed or "function" in parsed:
            return {"tool_calls": [normalize_tool_call(parsed)]}
        inferred = infer_tool_call_from_argument_object(parsed)
        if inferred is not None:
            return inferred
    raise ValueError(f"Unsupported next-action payload: {raw_text[:200]}")


def require_llm_class():
    try:
        from src.llm import LLM
    except ImportError:
        try:
            from src.llm import LLM
        except ImportError as exc:
            print(exc)
            raise SystemExit(
                "API-backed agent models require src/llm.py and its dependencies to be importable."
            ) from exc
    return LLM


def require_enterprise_runtime():
    try:
        from contextlib import AsyncExitStack
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError as exc:
        raise SystemExit(
            "Actual MCP execution evaluation requires `langchain-mcp-adapters` and its runtime dependencies."
        ) from exc
    return AsyncExitStack, MultiServerMCPClient, load_mcp_tools


def resolve_enterprise_mcp_config(
    explicit_path: Path | None,
    enterprise_runner: Path,
) -> Path | None:
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None
    default_path = enterprise_runner.parent / "mcp_config_http.json"
    return default_path if default_path.exists() else None


def full_world_model_state_for_agent(predicted_state: Any) -> Any:
    if predicted_state is None:
        return None
    try:
        return sanitize_state_content(predicted_state)
    except Exception:
        return predicted_state


def world_model_state_observation_payload(
    predicted_state: Any,
    predicted_error_message: str | None = None,
    raw_prediction: str | None = None,
) -> dict[str, Any]:
    return {
        "predicted_state": full_world_model_state_for_agent(predicted_state),
        "predicted_state_summary": (
            summarize_state_for_planning(predicted_state)
            if isinstance(predicted_state, dict)
            else None
        ),
        "predicted_error_message": predicted_error_message or None,
        "raw_world_model_prediction": raw_prediction or None,
    }


def _summarize_revision_rollouts(
    rollouts: list[dict[str, Any]] | None,
    observation_cap: int,
) -> str:
    """Render a compact human-readable summary of multi-step / multi-rollout lookaheads."""
    if not rollouts:
        return ""
    sections: list[str] = []
    rollout_count = len(rollouts)
    success_count = sum(1 for r in rollouts if r.get("all_predicted_success"))
    failure_count = sum(1 for r in rollouts if r.get("any_predicted_failure"))
    sections.append(
        f"Imagined rollouts: {rollout_count} total — "
        f"{success_count} predicted fully successful, "
        f"{failure_count} predicted at least one failure."
    )
    for rollout in rollouts:
        idx = rollout.get("rollout_index")
        temp = rollout.get("rollout_temperature")
        depth_target = rollout.get("lookahead_steps_target")
        depth_taken = rollout.get("lookahead_steps_taken")
        header = (
            f"\n[Rollout #{idx} (temperature={temp}, depth={depth_taken}/{depth_target}, "
            f"all_success={rollout.get('all_predicted_success')})]"
        )
        sections.append(header)
        for step in rollout.get("steps", []):
            step_offset = step.get("step_offset")
            tool_calls = step.get("tool_calls") or []
            tool_names = [c.get("name", "") for c in tool_calls]
            line = f"  step+{step_offset}: tools={tool_names}"
            err = (step.get("predicted_error_message") or "").strip()
            preview = (step.get("predicted_tool_output_preview") or "").strip()
            if err:
                line += f" | predicted_error={truncate_for_replay(err, observation_cap // 4)}"
            if preview:
                line += f" | predicted_output={truncate_for_replay(preview, observation_cap // 4)}"
            if step.get("predicted_state") is not None:
                line += (
                    " | predicted_state="
                    + json.dumps(step.get("predicted_state"), ensure_ascii=False, default=str)
                )
            elif step.get("raw_world_model_prediction"):
                line += " | raw_world_model_prediction=" + str(step.get("raw_world_model_prediction"))
            decision = step.get("imagined_decision")
            if decision:
                line += f" | imagined_decision={json.dumps(decision, ensure_ascii=False)}"
            if step.get("imagined_repeated_tool_call_loop"):
                line += " | repeated_tool_call_loop"
            if step.get("imagined_parse_error"):
                line += f" | parse_error={step['imagined_parse_error']}"
            sections.append(line)
    return "\n".join(sections)


def build_internal_thinking_messages(
    conversation: list[dict[str, Any]],
    proposed_calls: list[dict[str, Any]],
    feedbacks: list[dict[str, Any]],
    iteration: int,
    max_iterations: int,
    world_model_target: str = WORLD_MODEL_TARGET_STATE,
    revision_rollouts: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    target_is_tool_output = is_tool_output_target(world_model_target)
    observation_cap = _REPLAY_LIMITS["observation_chars"]
    rollouts_summary = _summarize_revision_rollouts(revision_rollouts, observation_cap)
    predicted_failure = any(not feedback.get("predicted_success", True) for feedback in feedbacks)
    predicted_errors = [
        (feedback.get("predicted_error_message") or "").strip()
        for feedback in feedbacks
        if not feedback.get("predicted_success", True)
    ]
    predicted_errors = [msg for msg in predicted_errors if msg]
    predicted_tool_outputs = [
        (feedback.get("predicted_tool_output") or "").strip()
        for feedback in feedbacks
    ]
    predicted_tool_outputs = [text for text in predicted_tool_outputs if text]
    predicted_state_payloads = [
        world_model_state_observation_payload(
            feedback.get("predicted_state"),
            feedback.get("predicted_error_message"),
            feedback.get("raw_prediction"),
        )
        for feedback in feedbacks
        if feedback.get("predicted_state") is not None
    ]
    if target_is_tool_output:
        observation_cap = _REPLAY_LIMITS["observation_chars"]
        directive = (
            "The world model predicted what the tool calls will return as raw tool output, "
            "but did not classify success vs. failure — that judgment is yours. "
            "Inspect the predicted tool output(s) below and decide whether to revise the tool calls. "
            "Modify the tool name, arguments, or pick a different tool if the predicted output "
            "indicates the call is wrong, off-target, or will not surface the information you need. "
            "Return the same tool calls unchanged if the predicted output looks correct and useful.\n"
        )
        if predicted_tool_outputs:
            directive += (
                "Predicted tool output(s) from the world model:\n"
                + "\n---\n".join(
                    truncate_for_replay(text, observation_cap)
                    for text in predicted_tool_outputs
                )
                + "\n"
            )
        else:
            directive += (
                "(The world model produced no tool-output text for these calls; "
                "treat that as a weak signal and decide whether to revise.)\n"
            )
        user_content = (
            f"Conversation so far:\n{render_messages(conversation)}\n\n"
            f"Current proposed tool calls:\n{json.dumps(proposed_calls, indent=2, ensure_ascii=False)}\n\n"
            f"World-model feedback for internal thinking iteration {iteration}/{max_iterations}:\n"
            f"{directive}"
        )
        if rollouts_summary:
            user_content += "\n\nMulti-step lookahead rollouts (use these to inform your decision):\n" + rollouts_summary
        return [
            {
                "role": "system",
                "content": (
                    "You are reviewing planned tool calls before actual MCP execution, "
                    "using the world model's predicted tool output to inform your decision. "
                    "Return JSON only as {\"tool_calls\": [...]} and do not return a final answer."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
    if predicted_failure:
        directive = (
            "The world model predicts these tool calls will FAIL. "
            "You MUST modify the tool calls to address the predicted failure — "
            "do not return the same tool calls unchanged unless you have a concrete reason "
            "to disagree with the world model.\n"
        )
        if predicted_errors:
            directive += (
                "Predicted error message(s): "
                + " | ".join(predicted_errors)
                + "\n"
            )
        if predicted_tool_outputs:
            observation_cap = _REPLAY_LIMITS["observation_chars"]
            directive += (
                "Predicted tool output(s) from the world model:\n"
                + "\n---\n".join(
                    truncate_for_replay(text, observation_cap)
                    for text in predicted_tool_outputs
                )
                + "\n"
            )
        directive += (
            "Use this signal to change the tool name (for ambiguity errors), fix arguments "
            "(for validation errors), or pick a different tool entirely. "
        )
    else:
        directive = (
            "The world model predicts these tool calls will succeed. "
            "Return them unchanged unless you spot a concrete error. "
        )
        if predicted_tool_outputs:
            observation_cap = _REPLAY_LIMITS["observation_chars"]
            directive += (
                "\nPredicted tool output(s) from the world model:\n"
                + "\n---\n".join(
                    truncate_for_replay(text, observation_cap)
                    for text in predicted_tool_outputs
                )
            )
    state_payload_text = ""
    if predicted_state_payloads:
        state_payload_text = (
            "\n\nFull predicted state output(s) from the world model:\n"
            + json.dumps(predicted_state_payloads, indent=2, ensure_ascii=False)
        )
    user_content = (
        f"Conversation so far:\n{render_messages(conversation)}\n\n"
        f"Current proposed tool calls:\n{json.dumps(proposed_calls, indent=2, ensure_ascii=False)}\n\n"
        f"World-model feedback for internal thinking iteration {iteration}/{max_iterations}:\n"
        f"{json.dumps(feedbacks, indent=2, ensure_ascii=False)}"
        f"{state_payload_text}\n\n"
        f"{directive}"
    )
    if rollouts_summary:
        user_content += "\n\nMulti-step lookahead rollouts (use these to inform your decision):\n" + rollouts_summary
    return [
        {
            "role": "system",
            "content": (
                "You are revising planned tool calls before actual MCP execution. "
                "Use the world-model feedback to fix likely failures. "
                "Return JSON only as {\"tool_calls\": [...]} and do not return a final answer."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def summarize_state_for_planning(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(state, dict):
        return None
    context = state_context_from_any(state)
    process = state_process_from_any(state)
    summary = {
        "last_tool_execution_result": context.get("last_tool_execution_result"),
        "last_tool_name": context.get("last_tool_name"),
        "error_message": context.get("error_message"),
        "current_stage": process.get("current_stage"),
        "remaining_stages": process.get("remaining_stages"),
    }
    if is_enterprise_state_payload(state):
        body = state_body_from_any(state)
        delta = enterprise_state_delta_from_any(state)
        summary.update(
            {
                "state_schema": body.get("schema"),
                "mode": body.get("mode"),
                "outcome": delta.get("outcome") or body.get("outcome"),
                "process_state": delta.get("process_state") or body.get("process_state"),
                "last_tool_events": (
                    (delta.get("history_context") or {}).get("last_tool_events")
                    if isinstance(delta.get("history_context"), dict)
                    else (body.get("history_context") or {}).get("last_tool_events")
                ),
            }
        )
    return summary


def build_imagined_trajectory_prompt_message(imagined_steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": (
            "[IMAGINED_TRAJECTORY_FOR_PLANNING_ONLY]\n"
            + json.dumps(imagined_steps, ensure_ascii=False, indent=2)
        ),
    }


def build_imagined_trajectory_selection_judge_prompt() -> str:
    return (
        "You are selecting the best imagined enterprise-agent trajectory before any real tool execution.\n\n"
        "Choose the candidate most likely to succeed in the real environment. Prefer candidates that:\n"
        "- satisfy the user's requirements with the fewest missing dependencies\n"
        "- use tools and arguments coherently\n"
        "- avoid parse errors, empty actions, or repeated tool-call loops\n"
        "- make concrete progress toward a valid final answer\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        '  "selected_index": int,\n'
        '  "scores": [{"index": int, "score": float, "reason": "brief explanation"}],\n'
        '  "comments": "brief explanation"\n'
        "}"
    )


def _generate_with_optional_temperature(
    generator: Any,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
) -> str:
    try:
        return generator.generate_from_messages(messages, temperature=temperature)
    except TypeError:
        return generator.generate_from_messages(messages)


def score_imagined_trajectory_candidate(candidate_steps: list[dict[str, Any]]) -> float:
    score = 0.0
    if not candidate_steps:
        return score
    for step in candidate_steps:
        if step.get("final_answer"):
            score += 3.0
        if step.get("clarify"):
            score -= 1.0
        if step.get("parse_error"):
            score -= 2.0
        if step.get("error") == "empty_tool_calls":
            score -= 2.0
        if step.get("repeated_tool_call_loop"):
            score -= 1.5
        feedbacks = step.get("predicted_feedback") or []
        if feedbacks:
            score += sum(0.5 if fb.get("predicted_success") else -0.5 for fb in feedbacks)
        if step.get("tool_calls"):
            score += 0.2
    return score


def select_imagined_trajectory_with_llm_judge(
    agent_generator: Any,
    task: TaskTrajectory,
    conversation: list[dict[str, Any]],
    candidate_rollouts: list[dict[str, Any]],
) -> dict[str, Any]:
    if not candidate_rollouts:
        return {
            "selected_index": 0,
            "scores": [],
            "comments": "No candidates provided",
            "fallback_used": True,
        }

    messages = [
        {"role": "system", "content": build_imagined_trajectory_selection_judge_prompt()},
        {
            "role": "user",
            "content": (
                f"User task:\n{task.user_messages[-1] if task.user_messages else ''}\n\n"
                f"Conversation so far:\n{render_messages(conversation)}\n\n"
                "Candidate imagined trajectories:\n"
                + json.dumps(candidate_rollouts, ensure_ascii=False, indent=2)
            ),
        },
    ]
    try:
        raw = _generate_with_optional_temperature(agent_generator, messages, temperature=0.0)
        raw = raw.split("</think>\n", 1)[-1].strip()
        parsed = parse_jsonish(strip_code_fence(raw))
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object from imagined trajectory judge.")
        selected_index = int(parsed.get("selected_index", 0))
        if not 0 <= selected_index < len(candidate_rollouts):
            raise ValueError(f"Selected imagined rollout index out of range: {selected_index}")
        return {
            "selected_index": selected_index,
            "scores": parsed.get("scores") or [],
            "comments": str(parsed.get("comments", "")),
            "raw_response": parsed,
            "fallback_used": False,
        }
    except Exception as exc:
        scored = [
            {
                "index": index,
                "score": score_imagined_trajectory_candidate(rollout.get("imagined_steps") or []),
                "reason": "heuristic fallback",
            }
            for index, rollout in enumerate(candidate_rollouts)
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return {
            "selected_index": scored[0]["index"] if scored else 0,
            "scores": scored,
            "comments": f"LLM judge failed, used heuristic fallback: {exc}",
            "fallback_used": True,
        }

def build_agent_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are evaluating whether an agent chooses the correct next action in an enterprise tool-use task. "
                "Return JSON only. If the task requires a tool next, return "
                '{"tool_calls": [{"name": "<tool_name>", "arguments": {...}}]}. '
                'If the task is complete, return {"final_answer": "<answer>"}.'
            ),
        },
        {
            "role": "user",
            "content": f"Conversation (/no_think):\n{render_messages(messages)}\n\nNext action JSON:",
        },
    ]


def get_tool_schema_safe_local(tool: Any) -> dict[str, Any]:
    try:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is None:
            return {}
        if isinstance(args_schema, dict):
            return args_schema
        if hasattr(args_schema, "model_json_schema"):
            return args_schema.model_json_schema()
        if hasattr(args_schema, "schema"):
            return args_schema.schema()
    except Exception:
        return {}
    return {}


def build_react_tool_descriptions(tools: list[Any]) -> str:
    parts: list[str] = []
    for tool in tools:
        schema = get_tool_schema_safe_local(tool)
        description = getattr(tool, "description", "") or ""
        chunk = [f"Tool: {tool.name}", f"Description: {description}"]
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
        if properties:
            chunk.append("Parameters:")
            for name, info in properties.items():
                param_type = info.get("type", "string") if isinstance(info, dict) else "string"
                marker = "required" if name in required else "optional"
                chunk.append(f"- {name} ({param_type}, {marker})")
        parts.append("\n".join(chunk))
    return "\n\n".join(parts)


def build_react_system_prompt(tool_descriptions: str) -> str:
    return (
        "You are a ReAct reasoning agent. Follow this EXACT format:\n\n"
        f"AVAILABLE TOOLS:\n{tool_descriptions}\n\n"
        "RESPONSE FORMAT:\n"
        "Step 1: THINK NODE\n"
        "- Perform granular thinking.\n"
        "- Think about what you need to do next based on previous actions.\n"
        '- For THINK NODE return JSON: {"thought": "..."}\n\n'
        "Step 2: ACTION NODE\n"
        "- Output the best action.\n"
        "- For TOOL CALL return JSON:\n"
        '{"action": "<tool_name>", "action_input": {"param1": "value1"}}\n\n'
        "- For FINAL ANSWER return JSON:\n"
        '{"action": "Final Answer", "action_input": "Your complete response"}\n\n'
        "CRITICAL RULES:\n"
        "1. Use exact tool names.\n"
        "2. Use exact parameter names.\n"
        "3. Return JSON only.\n"
        "4. Call tools before answering when information is needed.\n"
        "5. Do not guess.\n"
        "6. Provide the complete final answer when done."
    )


def _trim_replay_messages_to_budget(
    messages: list[dict[str, str]],
    char_budget: int,
) -> list[dict[str, str]]:
    """Drop the oldest tool/assistant turns until the prompt fits the budget.

    Keeps the system message (if any) and the first user message intact so the
    agent always retains the task context, then re-adds the most recent turns
    backwards until the budget is exhausted.
    """
    if char_budget <= 0:
        return messages
    total = sum(len(msg.get("content", "")) for msg in messages)
    if total <= char_budget:
        return messages

    pinned: list[dict[str, str]] = []
    rest: list[dict[str, str]] = []
    seen_first_user = False
    for msg in messages:
        role = msg.get("role")
        if role == "system" and len(pinned) == 0:
            pinned.append(msg)
        elif role == "user" and not seen_first_user:
            pinned.append(msg)
            seen_first_user = True
        else:
            rest.append(msg)

    used = sum(len(msg.get("content", "")) for msg in pinned)
    kept_tail: list[dict[str, str]] = []
    for msg in reversed(rest):
        size = len(msg.get("content", ""))
        if used + size > char_budget and kept_tail:
            break
        kept_tail.append(msg)
        used += size
    kept_tail.reverse()

    if len(rest) > len(kept_tail):
        notice = {
            "role": "user",
            "content": (
                f"[CONTEXT TRIMMED] Dropped {len(rest) - len(kept_tail)} earlier "
                "thought/action/observation turns to fit the context window."
            ),
        }
        kept_tail.insert(0, notice)
    return pinned + kept_tail


def filter_messages_for_react_replay(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, str]]:
    observation_limit = _REPLAY_LIMITS["observation_chars"]
    history_budget = _REPLAY_LIMITS["history_budget_chars"]
    filtered: list[dict[str, str]] = []
    system_added = False

    for message in messages:
        role = message.get("role")
        if role == "system":
            content = str(message.get("content", ""))
            if content.startswith("[INTERNAL_WORLD_MODEL_THINKING]"):
                continue
            if not system_added:
                filtered.append({"role": "system", "content": system_prompt})
                system_added = True
            continue
        if role == "user":
            filtered.append({"role": "user", "content": str(message.get("content", ""))})
            continue
        if role == "assistant":
            if message.get("tool_calls"):
                tool_call = message["tool_calls"][0]
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_args = tool_call.get("function", {}).get("arguments", {})
                filtered.append(
                    {
                        "role": "assistant",
                        "content": f"Action: Using {tool_name} with args: {json.dumps(tool_args, ensure_ascii=False)}",
                    }
                )
            elif message.get("content"):
                filtered.append({"role": "assistant", "content": str(message.get("content"))})
            continue
        if role == "tool":
            tool_output = stringify_tool_output(message.get("content"))
            tool_output = truncate_for_replay(tool_output, observation_limit)
            filtered.append(
                {
                    "role": "user",
                    "content": (
                        f"Observation: Tool '{message.get('name', 'unknown')}' returned: "
                        f"{tool_output}"
                    ),
                }
            )

    if not system_added:
        filtered.insert(0, {"role": "system", "content": system_prompt})
    return _trim_replay_messages_to_budget(filtered, history_budget)


def build_conversation_summary_for_replay(messages: list[dict[str, Any]]) -> str:
    summary_parts: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("content"):
            summary_parts.append(f"Thought/Action: {message['content']}")
        elif role == "assistant" and message.get("tool_calls"):
            tool_call = message["tool_calls"][0]
            summary_parts.append(
                f"Action: {tool_call.get('function', {}).get('name', 'unknown')}("
                f"{json.dumps(tool_call.get('function', {}).get('arguments', {}), ensure_ascii=False)})"
            )
        elif role == "tool":
            summary_parts.append(
                f"Observation: Tool {message.get('name', 'unknown')} returned "
                f"{stringify_tool_output(message.get('content'))}"
            )
    return "\n".join(summary_parts[-5:])


IMAGINED_TRAJECTORY_AGENT_POLICY = """

IMAGINED TRAJECTORY POLICY
You are generating hypothetical next steps for planning only. Your goal is to explore useful possible next actions, not to finish early.
Prefer information-gathering and validation actions before irreversible updates.
Do not produce a final answer unless all required task conditions are explicitly satisfied by observed or imagined evidence.
When a world-model prediction is uncertain, incomplete, generic, or lacks concrete IDs/counts, treat it as non-authoritative and continue exploring.
For each step, identify the unresolved requirement and choose one action that would reduce uncertainty.
Prefer lookups that recover concrete IDs, labels, permissions, cases, files, events, or existing records.
Avoid repeating the same tool call with the same arguments.
If a tool call failed, try one alternative formulation or prerequisite lookup before giving up.
Do not assume absence from a predicted empty result unless the query used verified IDs and narrow filters.
Do not claim completion inside imagined trajectories unless every required mutation has been accounted for.
Return a tool call unless all requirements are explicitly satisfied, at least two distinct lookup/update strategies have failed, or the next step requires real user clarification.
If unsure, explore with a read-only lookup.
"""


def imagined_trajectory_system_prompt(system_prompt: str) -> str:
    if IMAGINED_TRAJECTORY_AGENT_POLICY.strip() in system_prompt:
        return system_prompt
    return (system_prompt or "").rstrip() + IMAGINED_TRAJECTORY_AGENT_POLICY


def build_react_think_messages(
    messages: list[dict[str, Any]],
    current_query: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    filtered_messages = filter_messages_for_react_replay(messages, system_prompt)
    conversation_summary = build_conversation_summary_for_replay(messages)
    think_prompt = (
        f"Current Query: {current_query}\n"
        f"Conversation Summary: {conversation_summary}\n\n"
        "Status: Analyzing task...\n\n"
        'Think: What should I do next based on the current query and progress to complete the task?\n'
        'Respond in JSON format: {"thought": "your reasoning here"}'
    )
    return filtered_messages + [{"role": "user", "content": think_prompt}]


def build_react_action_messages(
    messages: list[dict[str, Any]],
    current_query: str,
    system_prompt: str,
    avoid_action_signatures: list[str] | None = None,
    prefer_action_names: list[str] | None = None,
    avoid_action_names: list[str] | None = None,
) -> list[dict[str, str]]:
    filtered_messages = filter_messages_for_react_replay(messages, system_prompt)
    conversation_summary = build_conversation_summary_for_replay(messages)
    diversity_prompt = ""
    if avoid_action_signatures:
        diversity_prompt += (
            "\nPreviously generated candidate actions for this same beam expansion:\n"
            + "\n".join(f"- {signature}" for signature in avoid_action_signatures)
            + "\nDo not repeat any listed tool name and arguments exactly.\n"
        )
    if prefer_action_names:
        diversity_prompt += (
            "For this candidate, stay near the earlier action family by using the same tool name(s): "
            + ", ".join(prefer_action_names)
            + ". Be audacious with the arguments: try a materially different valid parameterization.\n"
        )
    if avoid_action_names:
        diversity_prompt += (
            "For this candidate, broaden the beam with a different action family. Avoid these tool name(s): "
            + ", ".join(avoid_action_names)
            + ". Choose a different useful tool/action if one can plausibly advance the task.\n"
        )
    elif avoid_action_signatures:
        diversity_prompt += "Generate a meaningfully different next action.\n"
    action_prompt = (
        f"Current Query: {current_query}\n"
        f"Conversation Summary: {conversation_summary}\n\n"
        "Based on your previous thought, select and execute the most appropriate action.\n"
        f"{diversity_prompt}"
        "Return JSON only in one of the allowed ACTION NODE formats. "
        "Do not return tool arguments by themselves; always include the tool name using "
        "{\"action\": \"<tool_name>\", \"action_input\": {...}} or {\"tool_calls\": [...]}."
    )
    return filtered_messages + [{"role": "user", "content": action_prompt}]

def build_react_action_batch_messages(
    messages: list[dict[str, Any]],
    current_query: str,
    system_prompt: str,
    candidate_action_count: int,
) -> list[dict[str, str]]:
    filtered_messages = filter_messages_for_react_replay(messages, system_prompt)
    conversation_summary = build_conversation_summary_for_replay(messages)
    candidate_action_count = max(1, int(candidate_action_count))
    same_action_name_count = max(1, candidate_action_count // 2)
    different_action_name_count = candidate_action_count - same_action_name_count
    action_prompt = (
        f"Current Query: {current_query}\n"
        f"Conversation Summary: {conversation_summary}\n\n"
        "Based on your previous thought, generate multiple candidate next actions for beam search.\n"
        f"Return exactly {candidate_action_count} candidate actions in one JSON object with this shape: "
        "{\"candidates\": [{\"action\": \"<tool_name>\", \"action_input\": {...}}]}.\n"
        f"Candidates 1 through {same_action_name_count} should use the same tool name/action family "
        "but materially different valid arguments.\n"
        f"Candidates {same_action_name_count + 1} through {candidate_action_count} should broaden the beam "
        f"with {different_action_name_count} different useful tool name(s)/action families whenever possible.\n"
        "Do not repeat the exact same tool name and arguments. Prefer concrete tool calls over final answers "
        "unless the task is already complete. Return JSON only."
    )
    return filtered_messages + [{"role": "user", "content": action_prompt}]


def parse_agent_candidate_decisions(
    raw_text: str,
    expected_count: int,
) -> list[tuple[dict[str, Any], str]]:
    cleaned = strip_code_fence(strip_action_wrappers(raw_text)).strip()
    parsed: Any = None
    try:
        parsed = parse_jsonish(cleaned)
    except Exception:
        parsed = None

    raw_candidates: list[Any] = []
    if isinstance(parsed, dict):
        for key in ("candidates", "actions", "action_candidates"):
            value = parsed.get(key)
            if isinstance(value, list):
                raw_candidates = value
                break
        if not raw_candidates and _is_agent_decision_shaped(parsed):
            raw_candidates = [parsed]
    elif isinstance(parsed, list):
        raw_candidates = parsed

    if not raw_candidates:
        for candidate in iter_balanced_json_objects(cleaned):
            if isinstance(candidate, dict):
                for key in ("candidates", "actions", "action_candidates"):
                    value = candidate.get(key)
                    if isinstance(value, list):
                        raw_candidates.extend(value)
                        break
                else:
                    if _is_agent_decision_shaped(candidate):
                        raw_candidates.append(candidate)
            elif isinstance(candidate, list):
                raw_candidates.extend(candidate)
            if len(raw_candidates) >= expected_count:
                break

    if not raw_candidates:
        return [(parse_agent_decision(raw_text), raw_text)]

    decisions: list[tuple[dict[str, Any], str]] = []
    for raw_candidate in raw_candidates:
        if len(decisions) >= expected_count:
            break
        candidate_payload = raw_candidate.get("candidate") if isinstance(raw_candidate, dict) else raw_candidate
        if candidate_payload is None:
            candidate_payload = raw_candidate
        candidate_text = (
            candidate_payload
            if isinstance(candidate_payload, str)
            else json.dumps(candidate_payload, ensure_ascii=False)
        )
        try:
            decisions.append((parse_agent_decision(candidate_text), candidate_text))
        except Exception:
            if isinstance(raw_candidate, str):
                raise
            raise ValueError(f"Unable to parse candidate action: {candidate_text[:200]}")
    return decisions


def build_agent_prompt(
    messages: list[dict[str, Any]],
    tokenizer: Any | None = None,
    disable_chat_template: bool = False,
) -> str:
    return apply_chat_template_or_fallback(
        tokenizer,
        build_agent_chat_messages(messages),
        add_generation_prompt=True,
        disable_chat_template=disable_chat_template,
    )


def build_world_model_reflection(predictions: list[dict[str, str]]) -> dict[str, Any]:
    lines = []
    for prediction in predictions:
        lines.append(f"{prediction['name']}: {prediction['content']}")
    return {
        "role": "system",
        "content": "[WORLD_MODEL_PREDICTION_FOR_PLANNING_ONLY]\n" + "\n\n".join(lines),
    }


class HFTextGenerator:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int,
        trust_remote_code: bool = False,
        dtype: str = "auto",
        disable_chat_template: bool = False,
        attn_implementation: str = "sdpa",
        device_map: str | dict | None = None,
    ) -> None:
        torch, _, AutoModelForCausalLM, AutoTokenizer, _, _, _ = require_training_stack()
        self.torch = torch
        self.disable_chat_template = disable_chat_template
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        model_cls = resolve_text_generation_model_class(
            model_path,
            trust_remote_code=trust_remote_code,
            causal_lm_class=AutoModelForCausalLM,
        )
        adapter_config_path = Path(model_path) / "adapter_config.json"
        if adapter_config_path.is_file():
            with adapter_config_path.open("r", encoding="utf-8") as handle:
                adapter_config = json.load(handle)
            base_model_path = adapter_config.get("base_model_name_or_path")
            if not base_model_path:
                raise ValueError(f"Missing base_model_name_or_path in {adapter_config_path}")
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "Loading a LoRA adapter for evaluation requires `peft`."
                ) from exc
            ensure_peft_weight_converter_compatibility()
            base_model = model_cls.from_pretrained(
                base_model_path,
                dtype=resolve_torch_dtype(torch, dtype),
                trust_remote_code=trust_remote_code,
                device_map=resolve_inference_device_map(torch, override=device_map),
                attn_implementation=attn_implementation,
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = model_cls.from_pretrained(
                model_path,
                dtype=resolve_torch_dtype(torch, dtype),
                trust_remote_code=trust_remote_code,
                device_map=resolve_inference_device_map(torch, override=device_map),
                attn_implementation=attn_implementation,
            )
        self.input_device = next(self.model.parameters()).device
        self.max_new_tokens = max_new_tokens
        model_config = getattr(self.model, "config", None)
        base_model_config = getattr(getattr(self.model, "base_model", None), "config", None)
        self.disable_generation_cache = "nemotron_h" in {
            getattr(model_config, "model_type", None),
            getattr(base_model_config, "model_type", None),
        }
        if self.disable_generation_cache:
            # Nemotron-H remote code can receive `past_key_values` with
            # `cache_position=None` under Transformers generation, which crashes
            # in `prepare_inputs_for_generation`. Disabling cache is slower but
            # keeps evaluation/replay generation correct.
            self.model.config.use_cache = False
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.use_cache = False

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.disable_generation_cache:
            generation_kwargs["use_cache"] = False
        if temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.95})
        else:
            generation_kwargs.update({"do_sample": False})
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return strip_model_thinking_output(
            raw_text,
            special_tokens=getattr(self.tokenizer, "all_special_tokens", None),
        )

    def generate_from_messages(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        prompt = apply_chat_template_or_fallback(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
            disable_chat_template=self.disable_chat_template,
        )
        return self.generate(prompt, temperature=temperature)


class LLMTextGenerator:
    def __init__(
        self,
        method: str,
        max_new_tokens: int,
        vllm_server_port: int | None = None,
    ) -> None:
        LLM = require_llm_class()
        self.method = method
        self.max_new_tokens = max_new_tokens
        self.client = LLM(method, vllm_server_port=vllm_server_port)
        self._openai = None
        self._gpt5_cap_fallback: OpenAIChatGenerator | None = None
        self._printed_runtime_usage = False
        self.usage_metadata: dict[str, Any] = {}
        self.response_metadata: dict[str, Any] = {}

    @staticmethod
    def _is_gpt5_cap_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            pattern in message
            for pattern in (
                "daily cap",
                "daily limit",
                "usage cap",
                "quota",
                "insufficient_quota",
                "resource_exhausted",
                "rate limit",
                "too many requests",
                "request limit",
                "429",
            )
        )

    def _get_gpt5_cap_fallback(self) -> "OpenAIChatGenerator":
        if self._gpt5_cap_fallback is None:
            print(
                "[gpt5-agent] GPT-5 endpoint cap reached; falling back to OpenAI gpt-5.1."
            )
            self._gpt5_cap_fallback = OpenAIChatGenerator(
                "gpt-5.1",
                max_new_tokens=self.max_new_tokens,
            )
        return self._gpt5_cap_fallback

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        if self._gpt5_cap_fallback is not None:
            return self._gpt5_cap_fallback.generate(prompt, temperature=temperature)
        self.client.system_prompt_enable = False
        self.client.system_prompt = None
        try:
            return strip_model_thinking_output(
                self.client(prompt, None, temperature=temperature)
            )
        except Exception as exc:
            if self.method == "gpt5" and self._is_gpt5_cap_error(exc):
                return self._get_gpt5_cap_fallback().generate(prompt, temperature=temperature)
            raise

    def generate_from_messages(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        system_parts = [str(message.get("content", "")) for message in messages if message.get("role") == "system"]
        system_prompt = "\n\n".join(part for part in system_parts if part).strip() or None
        non_system_messages = [message for message in messages if message.get("role") != "system"]
        prompt_messages = non_system_messages or messages
        prompt = apply_chat_template_or_fallback(
            None,
            prompt_messages,
            add_generation_prompt=True,
            disable_chat_template=True,
        )
        if self.method == "claude" and system_prompt:
            prompt = f"SYSTEM: {system_prompt}\n\n{prompt}"
        if self._gpt5_cap_fallback is not None:
            return self._gpt5_cap_fallback.generate_from_messages(
                messages,
                temperature=temperature,
            )
        self.client.system_prompt_enable = bool(system_prompt)
        self.client.system_prompt = system_prompt
        try:
            return strip_model_thinking_output(
                self.client(prompt, system_prompt, temperature=temperature)
            )
        except Exception as exc:
            if self.method == "gpt5" and self._is_gpt5_cap_error(exc):
                return self._get_gpt5_cap_fallback().generate_from_messages(
                    messages,
                    temperature=temperature,
                )
            raise

    def _vllm_chat_base_url(self) -> str:
        endpoint = getattr(self.client, "vllm_endpoint", "") or ""
        suffix = "/chat/completions"
        if endpoint.endswith(suffix):
            return endpoint[: -len(suffix)]
        return endpoint

    def _get_openai_client(self) -> Any:
        if self._openai is None:
            try:
                import openai
            except ImportError as exc:
                raise RuntimeError(
                    "The `openai` package is required for vLLM native tool calling."
                ) from exc
            self._openai = openai
        return self._openai

    def _to_chat_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chat_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if role == "assistant":
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    formatted_calls = []
                    for index, tool_call in enumerate(tool_calls):
                        function = tool_call.get("function") if isinstance(tool_call, dict) else None
                        if not isinstance(function, dict):
                            function = tool_call if isinstance(tool_call, dict) else {}
                        arguments = function.get("arguments", {})
                        if not isinstance(arguments, str):
                            arguments = json.dumps(arguments, ensure_ascii=False)
                        formatted_calls.append(
                            {
                                "id": tool_call.get("id", f"call_{len(chat_messages)}_{index}"),
                                "type": "function",
                                "function": {
                                    "name": function.get("name", ""),
                                    "arguments": arguments,
                                },
                            }
                        )
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": content or None,
                            "tool_calls": formatted_calls,
                        }
                    )
                    continue
            elif role == "tool":
                chat_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.get("tool_call_id", ""),
                        "content": content,
                    }
                )
                continue
            if role not in ("system", "user", "assistant"):
                role = "user"
            chat_messages.append({"role": role, "content": content})
        return chat_messages

    def _clean_json_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}, "required": []}

        if "oneOf" in schema:
            for option in schema["oneOf"]:
                if isinstance(option, dict) and option.get("type") == "object":
                    schema = option
                    break
            else:
                return {"type": "object", "properties": {}, "required": []}

        if "allOf" in schema:
            merged_schema = {"type": "object", "properties": {}, "required": []}
            for sub_schema in schema["allOf"]:
                if not isinstance(sub_schema, dict):
                    continue
                if "properties" in sub_schema:
                    merged_schema["properties"].update(sub_schema["properties"])
                if "required" in sub_schema:
                    merged_schema["required"].extend(sub_schema["required"])
            schema = merged_schema

        if "anyOf" in schema:
            for option in schema["anyOf"]:
                if isinstance(option, dict) and option.get("type") == "object":
                    schema = option
                    break
            else:
                return {"type": "object", "properties": {}, "required": []}

        cleaned = dict(schema)
        cleaned.setdefault("type", "object")
        if cleaned["type"] == "object":
            cleaned.setdefault("properties", {})
            cleaned.setdefault("required", [])
        return cleaned

    def _to_openai_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted_tools: list[dict[str, Any]] = []
        for tool in tools:
            input_schema = tool.get("inputSchema") or tool.get("input_schema") or {}
            formatted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": self._clean_json_schema(input_schema),
                    },
                }
            )
        return formatted_tools

    def supports_native_tool_calling(self) -> bool:
        return self.method.startswith("vllm/")

    def invoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        if self._gpt5_cap_fallback is not None:
            return self._gpt5_cap_fallback.invoke_with_tools(messages, tools)
        if not self.method.startswith("vllm/"):
            raise NotImplementedError(
                f"Native tool calling is not implemented for agent backend {self.method!r}."
            )
        openai = self._get_openai_client()
        client = openai.OpenAI(
            base_url=self._vllm_chat_base_url(),
            api_key=getattr(self.client, "vllm_api_key", None) or os.environ.get("VLLM_API_KEY") or "not-needed",
        )
        request_kwargs: dict[str, Any] = {
            "model": getattr(self.client, "vllm_model", self.method.split("/", 1)[1]),
            "messages": self._to_chat_messages(messages),
            "tools": self._to_openai_tools(tools),
            "tool_choice": "auto",
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
        }
        response = client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        tool_calls = []
        for index, tool_call in enumerate(message.tool_calls or []):
            arguments: Any = tool_call.function.arguments or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = arguments.strip()
            tool_calls.append(
                {
                    "id": tool_call.id or f"call_{index}",
                    "name": tool_call.function.name,
                    "args": arguments,
                    "type": "tool_call",
                }
            )
        return SimpleNamespace(
            content=message.content or "",
            tool_calls=tool_calls,
            usage_metadata={},
            response_metadata={},
        )


def _is_openai_reasoning_model(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith(("o1", "o3", "o4", "gpt-5"))


def _openai_reasoning_effort_for_model(name: str) -> str | None:
    configured = os.environ.get("OPENAI_REASONING_EFFORT")
    if configured:
        return configured
    if name.lower().startswith("gpt-5.1"):
        return "none"
    return None


_OPENAI_AGENT_MODEL_NAME_RE = re.compile(
    r"^(?:gpt-\d|o[134](?:[-_]|$)|chatgpt-)",
    re.IGNORECASE,
)

_GEMINI_AGENT_MODEL_NAME_RE = re.compile(
    r"^gemini-\d",
    re.IGNORECASE,
)


def _looks_like_openai_agent_model(name: str) -> bool:
    """Detect bare OpenAI model names (no `openai/` prefix needed).

    Matches public OpenAI identifiers like `gpt-4o`, `gpt-4o-mini`, `gpt-5`,
    `gpt-5-mini`, `gpt-3.5-turbo`, `o1`, `o1-mini`, `o3`, `o3-mini`, `o4-mini`,
    `chatgpt-4o-latest`. Skips names containing `/` (those are HuggingFace
    repos like `openai-community/gpt2`) and names ending in `.gguf` etc.
    """
    if "/" in name or name.endswith((".gguf", ".bin", ".pt", ".safetensors")):
        return False
    return bool(_OPENAI_AGENT_MODEL_NAME_RE.match(name))


def _looks_like_gemini_agent_model(name: str) -> bool:
    if "/" in name or name.endswith((".gguf", ".bin", ".pt", ".safetensors")):
        return False
    return bool(_GEMINI_AGENT_MODEL_NAME_RE.match(name))


class OpenAIChatGenerator:
    """Direct OpenAI Chat Completions wrapper for use as the agent model.

    Invoked when `--agent-model` is given as `openai/<model>` (e.g.
    `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-5`). Reads credentials
    from the standard `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`) env
    vars via the `openai` SDK. Reasoning models (`gpt-5*`, `o1*`, `o3*`,
    `o4*`) automatically use `max_completion_tokens` and skip `temperature`,
    matching the SDK's expectations for those models.
    """

    def __init__(self, model: str, max_new_tokens: int) -> None:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "The `openai` package is required for `--agent-model openai/...`."
            ) from exc
        self._openai = openai
        self.model = model
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self.client = openai.OpenAI()
        self.is_reasoning_model = _is_openai_reasoning_model(model)

    def _to_chat_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        chat_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if role == "assistant":
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    formatted_calls = []
                    for index, tool_call in enumerate(tool_calls):
                        function = tool_call.get("function") if isinstance(tool_call, dict) else None
                        if not isinstance(function, dict):
                            function = tool_call if isinstance(tool_call, dict) else {}
                        arguments = function.get("arguments", {})
                        if not isinstance(arguments, str):
                            arguments = json.dumps(arguments, ensure_ascii=False)
                        formatted_calls.append(
                            {
                                "id": tool_call.get("id", f"call_{len(chat_messages)}_{index}"),
                                "type": "function",
                                "function": {
                                    "name": function.get("name", ""),
                                    "arguments": arguments,
                                },
                            }
                        )
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": content or None,
                            "tool_calls": formatted_calls,
                        }
                    )
                    continue
            elif role == "tool":
                chat_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.get("tool_call_id", ""),
                        "content": content,
                    }
                )
                continue
            if role not in ("system", "user", "assistant"):
                role = "user"
            chat_messages.append({"role": role, "content": content})
        return chat_messages

    def _clean_json_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}, "required": []}

        if "oneOf" in schema:
            for option in schema["oneOf"]:
                if isinstance(option, dict) and option.get("type") == "object":
                    schema = option
                    break
            else:
                return {"type": "object", "properties": {}, "required": []}

        if "allOf" in schema:
            merged_schema = {"type": "object", "properties": {}, "required": []}
            for sub_schema in schema["allOf"]:
                if not isinstance(sub_schema, dict):
                    continue
                if "properties" in sub_schema:
                    merged_schema["properties"].update(sub_schema["properties"])
                if "required" in sub_schema:
                    merged_schema["required"].extend(sub_schema["required"])
            schema = merged_schema

        if "anyOf" in schema:
            for option in schema["anyOf"]:
                if isinstance(option, dict) and option.get("type") == "object":
                    schema = option
                    break
            else:
                return {"type": "object", "properties": {}, "required": []}

        cleaned = dict(schema)
        cleaned.setdefault("type", "object")
        if cleaned["type"] == "object":
            cleaned.setdefault("properties", {})
            cleaned.setdefault("required", [])
        return cleaned

    def _to_openai_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted_tools: list[dict[str, Any]] = []
        for tool in tools:
            input_schema = tool.get("inputSchema") or tool.get("input_schema") or {}
            formatted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": self._clean_json_schema(input_schema),
                    },
                }
            )
        return formatted_tools

    def invoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_chat_messages(messages),
            "tools": self._to_openai_tools(tools),
            "tool_choice": "auto",
        }
        if self.is_reasoning_model:
            request_kwargs["max_completion_tokens"] = self.max_new_tokens
            reasoning_effort = _openai_reasoning_effort_for_model(self.model)
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = reasoning_effort
        else:
            request_kwargs["max_tokens"] = self.max_new_tokens
            request_kwargs["temperature"] = 0.0

        response = self.client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        tool_calls = []
        for index, tool_call in enumerate(message.tool_calls or []):
            arguments: Any = tool_call.function.arguments or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = arguments.strip()
            tool_calls.append(
                {
                    "id": tool_call.id or f"call_{index}",
                    "name": tool_call.function.name,
                    "args": arguments,
                    "type": "tool_call",
                }
            )
        return SimpleNamespace(
            content=message.content or "",
            tool_calls=tool_calls,
            usage_metadata={},
            response_metadata={},
        )

    def generate_from_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        chat_messages = self._to_chat_messages(messages)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
        }
        if self.is_reasoning_model:
            request_kwargs["max_completion_tokens"] = self.max_new_tokens
            reasoning_effort = _openai_reasoning_effort_for_model(self.model)
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = reasoning_effort
        else:
            request_kwargs["max_tokens"] = self.max_new_tokens
            request_kwargs["temperature"] = temperature

        max_attempts = int(os.environ.get("LLM_RATE_LIMIT_MAX_RETRIES", "30"))
        attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                return strip_model_thinking_output(
                    response.choices[0].message.content or ""
                )
            except self._openai.BadRequestError as exc:
                message = str(exc).lower()
                if "max_tokens" in message and "max_completion_tokens" in message:
                    request_kwargs["max_completion_tokens"] = request_kwargs.pop("max_tokens", self.max_new_tokens)
                    self.is_reasoning_model = True
                    continue
                if "temperature" in message and "unsupported" in message:
                    request_kwargs.pop("temperature", None)
                    continue
                raise
            except self._openai.RateLimitError as exc:
                attempt += 1
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"OpenAI rate limit did not clear after {max_attempts} retries."
                    ) from exc
                wait = min(60, 2 ** min(attempt, 6))
                print(
                    f"[openai-agent] rate-limit, sleeping {wait}s "
                    f"(attempt {attempt}/{max_attempts}): {exc}"
                )
                time.sleep(wait)
            except (
                self._openai.APITimeoutError,
                self._openai.APIConnectionError,
                self._openai.InternalServerError,
            ) as exc:
                attempt += 1
                if attempt >= max_attempts:
                    raise
                wait = min(30, 2 ** min(attempt, 5))
                print(f"[openai-agent] transient error, sleeping {wait}s: {exc}")
                time.sleep(wait)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self.generate_from_messages(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
        )


class AzureOpenAIChatGenerator(OpenAIChatGenerator):
    """Direct Azure OpenAI chat-completions wrapper for agent generation.

    Supports three Azure endpoint shapes:

    1. **Resource endpoint** (default Azure) ``https://<resource>.openai.azure.com``
       — the SDK constructs ``{endpoint}/openai/deployments/{deployment}/chat/completions``
       automatically using ``AZURE_OPENAI_API_VERSION``.
    2. **OpenAI-compatible** endpoint ending in ``/openai/v1`` — routed via the
       ``openai.OpenAI`` client with ``base_url``.
    3. **Direct base URL** — used when the deployment is baked into the
       endpoint path (e.g. proxy gateways like
       ``https://api.<vendor>.com/ai-foundation/chat-ai/gpt/gpt-5.1``).
       Trigger by setting ``AZURE_OPENAI_DEPLOYMENT_IN_URL=1`` (or by passing
       the full chat-completions URL ending with ``/chat/completions``). The
       SDK then POSTs to ``{endpoint}/chat/completions?api-version=...`` and
       does not append the ``/openai/deployments/{deployment}/`` prefix.
    """

    def __init__(self, deployment: str, max_new_tokens: int) -> None:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "The `openai` package is required for `--agent-model azureopenai/...`."
            ) from exc

        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = (
            os.environ.get("AZURE_OPENAI_ENDPOINT")
            or os.environ.get("AZURE_OPENAI_BASE_URL")
        )
        api_version = (
            os.environ.get("AZURE_OPENAI_API_VERSION")
            or os.environ.get("OPENAI_API_VERSION")
        )
        missing = []
        if not api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if missing:
            missing_str = ", ".join(missing)
            raise RuntimeError(
                "Azure OpenAI agent models require environment variables: "
                f"{missing_str}."
            )

        normalized_endpoint = endpoint.rstrip("/")
        deployment_in_url_env = os.environ.get(
            "AZURE_OPENAI_DEPLOYMENT_IN_URL", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        endpoint_targets_chat_completions = normalized_endpoint.endswith(
            "/chat/completions"
        )
        if endpoint_targets_chat_completions:
            # Strip the explicit suffix so the SDK can append it itself.
            normalized_endpoint = normalized_endpoint[: -len("/chat/completions")]

        if normalized_endpoint.endswith("/openai/v1"):
            client = openai.OpenAI(
                api_key=api_key,
                base_url=normalized_endpoint + "/",
            )
        elif deployment_in_url_env or endpoint_targets_chat_completions:
            # Direct base-URL mode: the deployment is already in the URL path
            # (typical for vendor proxy gateways). We use
            # AzureOpenAI(base_url=...) so the SDK still attaches the `api-key`
            # header + `api-version` query param but does NOT construct
            # `/openai/deployments/{deployment}/` itself.
            if not api_version:
                raise RuntimeError(
                    "Azure OpenAI direct-base-URL mode requires "
                    "`AZURE_OPENAI_API_VERSION`."
                )
            client = openai.AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                base_url=normalized_endpoint,
            )
        else:
            if not api_version:
                raise RuntimeError(
                    "Azure OpenAI agent models require `AZURE_OPENAI_API_VERSION` "
                    "when `AZURE_OPENAI_ENDPOINT` is a resource endpoint."
                )
            client = openai.AzureOpenAI(
                azure_endpoint=normalized_endpoint,
                api_key=api_key,
                api_version=api_version,
            )

        self._openai = openai
        self.model = deployment
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self.client = client
        self.is_reasoning_model = _is_openai_reasoning_model(deployment)


class GeminiChatGenerator:
    """Direct Gemini API wrapper for agent generation and native tool calling."""

    def __init__(self, model: str, max_new_tokens: int) -> None:
        self.model = model
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.api_base_url = os.environ.get(
            "GEMINI_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ).rstrip("/")
        if not self.api_key:
            raise RuntimeError(
                "Gemini agent models require `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)."
            )

    def _api_url(self) -> str:
        model_name = self.model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        encoded_model = urllib.parse.quote(model_name, safe="/")
        encoded_key = urllib.parse.quote(self.api_key, safe="")
        return f"{self.api_base_url}/{encoded_model}:generateContent?key={encoded_key}"

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self._api_url(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        max_attempts = int(os.environ.get("LLM_RATE_LIMIT_MAX_RETRIES", "30"))
        attempt = 0
        while True:
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    raw_body = response.read().decode("utf-8")
                return json.loads(raw_body)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                attempt += 1
                if exc.code == 429 and attempt < max_attempts:
                    wait = min(60, 2 ** min(attempt, 6))
                    print(
                        f"[gemini-agent] rate-limit, sleeping {wait}s "
                        f"(attempt {attempt}/{max_attempts}): {body}"
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Gemini API request failed with HTTP {exc.code}: {body}"
                ) from exc
            except urllib.error.URLError as exc:
                attempt += 1
                if attempt >= max_attempts:
                    raise RuntimeError(f"Gemini API request failed: {exc}") from exc
                wait = min(30, 2 ** min(attempt, 5))
                print(f"[gemini-agent] transient error, sleeping {wait}s: {exc}")
                time.sleep(wait)

    def _coerce_message_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    def _extract_system_prompt(self, messages: list[dict[str, Any]]) -> str | None:
        system_parts = [
            self._coerce_message_content(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        ]
        combined = "\n\n".join(part.strip() for part in system_parts if part and part.strip()).strip()
        return combined or None

    def _to_gemini_contents(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                continue

            if role == "assistant":
                parts: list[dict[str, Any]] = []
                content = self._coerce_message_content(message.get("content", "")).strip()
                if content:
                    parts.append({"text": content})
                for index, tool_call in enumerate(message.get("tool_calls") or []):
                    function = tool_call.get("function") if isinstance(tool_call, dict) else None
                    if not isinstance(function, dict):
                        function = tool_call if isinstance(tool_call, dict) else {}
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {"raw": arguments}
                    elif arguments is None:
                        arguments = {}
                    elif not isinstance(arguments, dict):
                        arguments = {"value": arguments}
                    parts.append(
                        {
                            "functionCall": {
                                "id": tool_call.get("id", f"call_{len(contents)}_{index}"),
                                "name": function.get("name", ""),
                                "args": arguments,
                            }
                        }
                    )
                if parts:
                    contents.append({"role": "model", "parts": parts})
                continue

            if role == "tool":
                tool_name = str(message.get("name", "")).strip()
                tool_call_id = str(message.get("tool_call_id", "")).strip()
                raw_content = message.get("content", "")
                if isinstance(raw_content, str):
                    stripped = raw_content.strip()
                    if stripped:
                        try:
                            response_payload: Any = json.loads(stripped)
                        except json.JSONDecodeError:
                            response_payload = {"content": stripped}
                    else:
                        response_payload = {"content": ""}
                else:
                    response_payload = raw_content
                function_response = {
                    "name": tool_name or "tool",
                    "response": {"result": response_payload},
                }
                if tool_call_id:
                    function_response["id"] = tool_call_id
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"functionResponse": function_response}],
                    }
                )
                continue

            content = self._coerce_message_content(message.get("content", "")).strip()
            if content:
                contents.append({"role": "user", "parts": [{"text": content}]})

        return contents

    def _clean_json_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}, "required": []}

        if "oneOf" in schema:
            for option in schema["oneOf"]:
                if isinstance(option, dict) and option.get("type") == "object":
                    schema = option
                    break
            else:
                return {"type": "object", "properties": {}, "required": []}

        if "allOf" in schema:
            merged_schema = {"type": "object", "properties": {}, "required": []}
            for sub_schema in schema["allOf"]:
                if not isinstance(sub_schema, dict):
                    continue
                if "properties" in sub_schema:
                    merged_schema["properties"].update(sub_schema["properties"])
                if "required" in sub_schema:
                    merged_schema["required"].extend(sub_schema["required"])
            schema = merged_schema

        if "anyOf" in schema:
            for option in schema["anyOf"]:
                if isinstance(option, dict) and option.get("type") == "object":
                    schema = option
                    break
            else:
                return {"type": "object", "properties": {}, "required": []}

        cleaned = dict(schema)
        cleaned.setdefault("type", "object")
        if cleaned["type"] == "object":
            cleaned.setdefault("properties", {})
            cleaned.setdefault("required", [])
        return cleaned

    def _to_gemini_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declarations: list[dict[str, Any]] = []
        for tool in tools:
            input_schema = tool.get("inputSchema") or tool.get("input_schema") or {}
            declarations.append(
                {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": self._clean_json_schema(input_schema),
                }
            )
        return [{"function_declarations": declarations}] if declarations else []

    def _base_payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contents": self._to_gemini_contents(messages),
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": self.max_new_tokens,
            },
        }
        system_prompt = self._extract_system_prompt(messages)
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    def _extract_response_text(self, response: dict[str, Any]) -> str:
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return ""
        content = candidates[0].get("content")
        if not isinstance(content, dict):
            return ""
        text_parts: list[str] = []
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        return "\n".join(text_parts).strip()

    def _extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return []
        content = candidates[0].get("content")
        if not isinstance(content, dict):
            return []
        tool_calls: list[dict[str, Any]] = []
        for index, part in enumerate(content.get("parts") or []):
            if not isinstance(part, dict):
                continue
            function_call = part.get("functionCall") or part.get("function_call")
            if not isinstance(function_call, dict):
                continue
            arguments = function_call.get("args", {})
            if arguments is None:
                arguments = {}
            elif not isinstance(arguments, dict):
                arguments = {"value": arguments}
            tool_calls.append(
                {
                    "id": function_call.get("id") or f"call_{index}",
                    "name": function_call.get("name", ""),
                    "args": arguments,
                    "type": "tool_call",
                }
            )
        return tool_calls

    def invoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        payload = self._base_payload(messages, temperature=0.0)
        payload["tools"] = self._to_gemini_tools(tools)
        payload["tool_config"] = {
            "function_calling_config": {
                "mode": "AUTO",
            }
        }
        response = self._request(payload)
        return SimpleNamespace(
            content=self._extract_response_text(response),
            tool_calls=self._extract_tool_calls(response),
            usage_metadata={},
            response_metadata={"gemini_response": response},
        )

    def generate_from_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        payload = self._base_payload(messages, temperature=temperature)
        response = self._request(payload)
        return self._extract_response_text(response)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self.generate_from_messages(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
        )


def build_agent_generator(
    model_path: str,
    max_new_tokens: int,
    trust_remote_code: bool = False,
    dtype: str = "auto",
    disable_chat_template: bool = False,
    attn_implementation: str = "sdpa",
    device_map: str | dict | None = None,
    vllm_server_port: int | None = None,
):
    openai_alias = OPENAI_AGENT_MODEL_ALIASES.get(model_path)
    if openai_alias:
        return OpenAIChatGenerator(openai_alias, max_new_tokens=max_new_tokens)
    llm_method = LLM_AGENT_MODEL_ALIASES.get(model_path)
    if llm_method:
        return LLMTextGenerator(
            llm_method,
            max_new_tokens=max_new_tokens,
            vllm_server_port=vllm_server_port,
        )
    if model_path.startswith("azureopenai/") or model_path.startswith("azureopenai:"):
        separator = "/" if "/" in model_path else ":"
        azure_deployment = model_path.split(separator, 1)[1].strip()
        if not azure_deployment:
            raise ValueError(
                f"Empty Azure OpenAI deployment name in {model_path!r}. "
                "Use e.g. `azureopenai/my-gpt-5-deployment`."
            )
        return AzureOpenAIChatGenerator(
            azure_deployment,
            max_new_tokens=max_new_tokens,
        )
    if model_path.startswith("gemini/") or model_path.startswith("gemini:"):
        separator = "/" if "/" in model_path else ":"
        gemini_model = model_path.split(separator, 1)[1].strip()
        if not gemini_model:
            raise ValueError(
                f"Empty Gemini model name in {model_path!r}. "
                "Use e.g. `gemini/gemini-2.5-pro`."
            )
        return GeminiChatGenerator(gemini_model, max_new_tokens=max_new_tokens)
    if model_path.startswith("openai/") or model_path.startswith("openai:"):
        separator = "/" if "/" in model_path else ":"
        openai_model = model_path.split(separator, 1)[1].strip()
        if not openai_model:
            raise ValueError(
                f"Empty OpenAI model name in {model_path!r}. "
                "Use e.g. `openai/gpt-4o-mini`."
            )
        return OpenAIChatGenerator(openai_model, max_new_tokens=max_new_tokens)
    if _looks_like_openai_agent_model(model_path):
        return OpenAIChatGenerator(model_path, max_new_tokens=max_new_tokens)
    if _looks_like_gemini_agent_model(model_path):
        return GeminiChatGenerator(model_path, max_new_tokens=max_new_tokens)
    if model_path.startswith("vllm/") or model_path in API_AGENT_MODEL_METHODS:
        return LLMTextGenerator(
            model_path,
            max_new_tokens=max_new_tokens,
            vllm_server_port=vllm_server_port,
        )
    return HFTextGenerator(
        model_path,
        max_new_tokens=max_new_tokens,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        disable_chat_template=disable_chat_template,
        attn_implementation=attn_implementation,
        device_map=device_map,
    )


def build_tool_lookup(tools: list[Any]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    exact_lookup: dict[str, Any] = {}
    alias_lookup: dict[str, list[Any]] = {}
    for tool in tools:
        exact_lookup[tool.name] = tool
        original_name = getattr(tool, "_original_tool_name", tool.name)
        alias_lookup.setdefault(original_name, []).append(tool)
    return exact_lookup, alias_lookup


def resolve_tool_for_execution(
    requested_name: str,
    exact_lookup: dict[str, Any],
    alias_lookup: dict[str, list[Any]],
) -> tuple[Any | None, str | None]:
    if requested_name in exact_lookup:
        return exact_lookup[requested_name], None
    matches = alias_lookup.get(requested_name, [])
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        resolved_names = ", ".join(tool.name for tool in matches)
        return None, f"Ambiguous tool name `{requested_name}` matched multiple tools: {resolved_names}"
    return None, f"Unknown tool `{requested_name}`"


def predict_world_model_feedback(
    world_model_generator: Any,
    task: TaskTrajectory,
    previous_state: dict[str, Any],
    predicted_calls: list[dict[str, Any]],
    interaction_index: int,
    world_model_target: str,
    include_error_message_in_target: bool = False,
    include_stage_in_target: bool = False,
    include_world_model_history: bool = False,
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> list[dict[str, Any]]:
    prediction_example = WorldModelStateExample(
        trajectory_id=str(task.trajectory_index),
        trajectory_index=task.trajectory_index,
        interaction_index=interaction_index,
        system_prompt=task.system_prompt,
        user_prompt=task.user_messages[-1] if task.user_messages else "",
        action={"tool_calls": to_openai_tool_calls(predicted_calls)},
        state_history=list(state_history or []),
        input_history=list(input_history or []),
        previous_state=previous_state,
        state=make_blank_state_like(previous_state),
    )
    prediction = world_model_generator.generate_from_messages(
        build_state_prediction_chat_messages(
            prediction_example,
            target_mode=world_model_target,
            include_error_message=include_error_message_in_target,
            include_stage=include_stage_in_target,
            include_input_history=include_world_model_history,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
        )
    )
    prediction = prediction.split("</think>\n", 1)[-1].strip()
    emit_progress(
        "WORLD_MODEL_RAW",
        interaction_index=interaction_index,
        target=world_model_target,
        tool_calls=preview_tool_calls(predicted_calls),
        raw_prediction=prediction,
    )
    predicted_state = None
    predicted_tool_output = ""
    predicted_current_stage = None
    predicted_remaining_stages = None
    parse_error = None
    try:
        if is_tool_execution_result_target(world_model_target):
            parsed_payload = parse_binary_world_model_prediction(prediction)
            predicted_result = normalize_tool_execution_result_for_target(
                parsed_payload.get("success"),
                target_mode=world_model_target,
            )
            if predicted_result is None:
                raise ValueError(f"Unable to parse tool-result prediction: {prediction[:200]}")
            predicted_success = predicted_result == 1
            error_message = (parsed_payload.get("error_message") or "").strip()
            predicted_current_stage = parsed_payload.get("current_stage")
            predicted_remaining_stages = parsed_payload.get("remaining_stages")
            predicted_state = make_tool_execution_prediction_state(
                predicted_success=predicted_success,
                predicted_result=predicted_result,
                error_message=error_message,
                current_stage=predicted_current_stage,
                remaining_stages=predicted_remaining_stages,
            )
        elif is_tool_output_target(world_model_target):
            predicted_tool_output = prediction
            looks_like_failure = tool_output_looks_like_failure(prediction)
            predicted_success = not looks_like_failure
            error_message = prediction if looks_like_failure else ""
        else:
            predicted_state = sanitize_state_content(parse_jsonish(prediction))
            context = state_context_from_any(predicted_state)
            predicted_success = normalize_last_tool_execution_result(
                context.get("last_tool_execution_result")
            ) == 1
            error_message = context.get("error_message", "")
            predicted_current_stage = state_current_stage(predicted_state)
            predicted_remaining_stages = state_remaining_stages(predicted_state)
    except Exception as exc:
        predicted_success = False
        error_message = str(exc)
        parse_error = str(exc)

    emit_progress(
        "WORLD_MODEL_RESULT",
        interaction_index=interaction_index,
        predicted_success=predicted_success,
        predicted_error_message=error_message,
        predicted_tool_output=predicted_tool_output,
        predicted_current_stage=predicted_current_stage,
        predicted_remaining_stages=predicted_remaining_stages,
        parse_error=parse_error,
    )

    return [
        {
            "tool_calls": predicted_calls,
            "predicted_success": predicted_success,
            "predicted_state": predicted_state,
            "predicted_tool_output": predicted_tool_output,
            "predicted_error_message": error_message,
            "predicted_current_stage": predicted_current_stage,
            "predicted_remaining_stages": predicted_remaining_stages,
            "raw_prediction": prediction,
            "parse_error": parse_error,
        }
    ]


def imagine_trajectory(
    agent_generator: Any,
    world_model_generator: Any,
    task: TaskTrajectory,
    conversation: list[dict[str, Any]],
    previous_state: dict[str, Any],
    react_system_prompt: str,
    max_imagined_steps: int,
    world_model_target: str,
    include_error_message_in_target: bool = False,
    include_stage_in_target: bool = False,
    include_world_model_history: bool = False,
    start_interaction_index: int = 0,
    rollout_index: int = 0,
    rollout_temperature: float = 0.0,
    observation_source: str = "world_model",
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> list[dict[str, Any]]:
    imagined_conversation = [dict(message) for message in conversation]
    imagined_react_system_prompt = imagined_trajectory_system_prompt(react_system_prompt)
    imagined_state = sanitize_state_content(previous_state)
    imagined_state_history = append_state_history(
        list(state_history or []), imagined_state, max_items=state_history_size
    )
    imagined_input_history = list(input_history or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:]
    imagined_steps: list[dict[str, Any]] = []
    seen_tool_call_signatures: dict[str, int] = {}

    for imagined_index in range(max_imagined_steps):
        raw_thought = _generate_with_optional_temperature(
            agent_generator,
            build_react_think_messages(
                imagined_conversation,
                current_query=task.user_messages[-1] if task.user_messages else "",
                system_prompt=imagined_react_system_prompt,
            ),
            temperature=rollout_temperature,
        )
        raw_thought = strip_model_thinking_output(raw_thought)
        thought_payload = parse_thought_payload(raw_thought)
        emit_progress(
            "IMAGINED_THOUGHT",
            imagined_step=imagined_index + 1,
            thought=thought_payload.get("thought"),
        )
        imagined_conversation.append({"role": "assistant", "content": json.dumps(thought_payload, ensure_ascii=False)})

        raw_action = _generate_with_optional_temperature(
            agent_generator,
            build_react_action_messages(
                imagined_conversation,
                current_query=task.user_messages[-1] if task.user_messages else "",
                system_prompt=imagined_react_system_prompt,
            ),
            temperature=rollout_temperature,
        )
        raw_action = strip_model_thinking_output(raw_action)
        try:
            decision = parse_agent_decision(raw_action)
        except Exception as exc:
            emit_progress(
                "IMAGINED_ACTION_PARSE_ERROR",
                imagined_step=imagined_index + 1,
                error=str(exc),
                raw_action=raw_action,
            )
            imagined_steps.append(
                {
                    "imagined_step": imagined_index + 1,
                    "thought": thought_payload,
                    "raw_action": raw_action,
                    "parse_error": str(exc),
                }
            )
            break

        if "final_answer" in decision:
            emit_progress(
                "IMAGINED_FINAL_ANSWER",
                imagined_step=imagined_index + 1,
                final_answer=decision["final_answer"],
            )
            imagined_steps.append(
                {
                    "imagined_step": imagined_index + 1,
                    "thought": thought_payload,
                    "final_answer": decision["final_answer"],
                    "predicted_state": full_world_model_state_for_agent(imagined_state),
                    "predicted_state_summary": summarize_state_for_planning(imagined_state),
                }
            )
            break
        if "clarify" in decision:
            emit_progress(
                "IMAGINED_CLARIFY",
                imagined_step=imagined_index + 1,
                clarify=decision["clarify"],
            )
            imagined_steps.append(
                {
                    "imagined_step": imagined_index + 1,
                    "thought": thought_payload,
                    "clarify": decision["clarify"],
                    "predicted_state": full_world_model_state_for_agent(imagined_state),
                    "predicted_state_summary": summarize_state_for_planning(imagined_state),
                }
            )
            break

        planned_calls = [normalize_tool_call(call) for call in decision.get("tool_calls", [])]
        if not planned_calls:
            emit_progress(
                "IMAGINED_EMPTY_ACTION",
                imagined_step=imagined_index + 1,
                raw_action=raw_action,
            )
            imagined_steps.append(
                {
                    "imagined_step": imagined_index + 1,
                    "thought": thought_payload,
                    "raw_action": raw_action,
                    "error": "empty_tool_calls",
                }
            )
            break

        tool_call_signature = json_compact(planned_calls)
        emit_progress(
            "IMAGINED_ACTION",
            imagined_step=imagined_index + 1,
            tool_calls=preview_tool_calls(planned_calls),
            signature=tool_call_signature,
        )
        repeated_tool_call_count = seen_tool_call_signatures.get(tool_call_signature, 0) + 1
        seen_tool_call_signatures[tool_call_signature] = repeated_tool_call_count
        feedbacks: list[dict[str, Any]] = []
        predicted_state = None
        if observation_source == "world_model":
            feedbacks = predict_world_model_feedback(
                world_model_generator,
                task,
                imagined_state,
                planned_calls,
                interaction_index=start_interaction_index + imagined_index,
                world_model_target=world_model_target,
                include_error_message_in_target=include_error_message_in_target,
                include_stage_in_target=include_stage_in_target,
                include_world_model_history=include_world_model_history,
                state_history=imagined_state_history,
                input_history=imagined_input_history,
                system_prompt_max_chars=system_prompt_max_chars,
                action_max_chars=action_max_chars,
            )
            predicted_state = feedbacks[-1].get("predicted_state") if feedbacks else None
            imagined_input_history = append_world_model_input_history(
                imagined_input_history,
                make_imagined_world_model_history_entry(
                    imagined_step=imagined_index + 1,
                    action=planned_calls,
                    state=predicted_state,
                ),
            )
        imagined_steps.append(
            {
                "imagined_step": imagined_index + 1,
                "thought": thought_payload,
                "tool_calls": planned_calls,
                "repeated_tool_call_loop": repeated_tool_call_count > 1,
                "repeated_tool_call_count": repeated_tool_call_count,
                "predicted_feedback": feedbacks,
                "predicted_state": full_world_model_state_for_agent(predicted_state),
                "predicted_state_summary": (
                    summarize_state_for_planning(predicted_state) if predicted_state else None
                ),
                "raw_world_model_prediction": (
                    feedbacks[-1].get("raw_prediction") if feedbacks else None
                ),
                "observation_source": observation_source,
            }
        )
        if repeated_tool_call_count > 1:
            break
        imagined_conversation.append({"role": "assistant", "tool_calls": to_openai_tool_calls(planned_calls)})
        if observation_source == "world_model":
            predicted_tool_output_text = (
                feedbacks[-1].get("predicted_tool_output") if feedbacks else None
            ) or ""
            if predicted_tool_output_text:
                imagined_conversation.append(
                    {
                        "role": "tool",
                        "name": planned_calls[0].get("name", "") if planned_calls else "",
                        "content": (
                            "[IMAGINED_TOOL_OUTPUT_FROM_WORLD_MODEL]\n"
                            + predicted_tool_output_text
                        ),
                    }
                )
            else:
                imagined_conversation.append(
                    {
                        "role": "user",
                        "content": (
                            "Imagined observation based on world-model prediction:\n"
                            + json.dumps(
                                world_model_state_observation_payload(
                                    predicted_state,
                                    feedbacks[-1].get("predicted_error_message") if feedbacks else None,
                                    feedbacks[-1].get("raw_prediction") if feedbacks else None,
                                ),
                                ensure_ascii=False,
                            )
                        ),
                    }
                )
        else:
            imagined_conversation.append(
                {
                    "role": "user",
                    "content": (
                        "[IMAGINED_LOOKAHEAD_WITHOUT_WORLD_MODEL]\n"
                        "No imagined tool result is available for the previous step. "
                        "Continue planning the next likely step using only the task requirements, "
                        "the prior conversation, and the previously imagined tool calls."
                    ),
                }
            )
        if observation_source == "world_model" and predicted_state is not None:
            imagined_state = predicted_state
            imagined_state_history = append_state_history(
                imagined_state_history, imagined_state, max_items=state_history_size
            )
            if state_is_finished(predicted_state):
                break

    return imagined_steps


def append_imagined_observation_for_planning(
    imagined_conversation: list[dict[str, Any]],
    planned_calls: list[dict[str, Any]],
    feedbacks: list[dict[str, Any]],
    predicted_state: Any,
    observation_source: str,
) -> None:
    imagined_conversation.append({"role": "assistant", "tool_calls": to_openai_tool_calls(planned_calls)})
    if observation_source != "world_model":
        imagined_conversation.append(
            {
                "role": "user",
                "content": (
                    "[IMAGINED_LOOKAHEAD_WITHOUT_WORLD_MODEL]\n"
                    "No imagined tool result is available for the previous step. "
                    "Continue planning the next likely step using only the task requirements, "
                    "the prior conversation, and the previously imagined tool calls."
                ),
            }
        )
        return

    last_feedback = feedbacks[-1] if feedbacks else {}
    predicted_tool_output_text = (last_feedback.get("predicted_tool_output") or "").strip()
    if predicted_tool_output_text:
        imagined_conversation.append(
            {
                "role": "tool",
                "name": planned_calls[0].get("name", "") if planned_calls else "",
                "content": "[IMAGINED_TOOL_OUTPUT_FROM_WORLD_MODEL]\n" + predicted_tool_output_text,
            }
        )
        return

    imagined_conversation.append(
        {
            "role": "user",
            "content": (
                "Imagined observation based on world-model prediction:\n"
                + json.dumps(
                    world_model_state_observation_payload(
                        predicted_state,
                        last_feedback.get("predicted_error_message"),
                        last_feedback.get("raw_prediction"),
                    ),
                    ensure_ascii=False,
                )
            ),
        }
    )


def imagined_stage_value(state: Any) -> Any:
    if state is None:
        return None
    return state_current_stage(state)


def score_topk_imagined_step(
    *,
    previous_state: Any,
    predicted_state: Any,
    feedbacks: list[dict[str, Any]],
    repeated_tool_call_count: int,
    final_answer: bool = False,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if final_answer:
        if state_is_finished(previous_state):
            return 1.0, ["final_answer_after_finished_stage=1"]
        return 0.0, ["final_answer_but_stage_remains=0"]

    if repeated_tool_call_count > 1:
        return 0.0, ["same_tool_names_and_arguments=0"]

    score = 0.0
    if feedbacks and all(bool(feedback.get("predicted_success")) for feedback in feedbacks):
        score += 1.0
        reasons.append("success=1")
    else:
        reasons.append("failure=0")

    previous_stage = imagined_stage_value(previous_state)
    next_stage = imagined_stage_value(predicted_state)
    if next_stage is None and feedbacks:
        next_stage = feedbacks[-1].get("predicted_current_stage")
    if next_stage is not None and next_stage != previous_stage:
        score += 1.0
        reasons.append("current_stage_change=1")
    else:
        reasons.append("current_stage_unchanged=0")
    return score, reasons


def make_topk_rollout_record(
    index: int,
    branch: dict[str, Any],
    *,
    observation_source: str,
) -> dict[str, Any]:
    return {
        "rollout_index": index,
        "rollout_temperature": branch.get("rollout_temperature"),
        "observation_source": observation_source,
        "selection_strategy": "topk_search",
        "score": branch.get("score", 0.0),
        "depth": branch.get("depth", 0),
        "terminal": bool(branch.get("terminal")),
        "imagined_steps": branch.get("imagined_steps") or [],
        "terminal_state": full_world_model_state_for_agent(branch.get("state")),
        "terminal_state_summary": summarize_state_for_planning(branch.get("state")),
    }


def imagine_trajectory_topk_search(
    agent_generator: Any,
    world_model_generator: Any,
    task: TaskTrajectory,
    conversation: list[dict[str, Any]],
    previous_state: dict[str, Any],
    react_system_prompt: str,
    max_imagined_steps: int,
    world_model_target: str,
    include_error_message_in_target: bool = False,
    include_stage_in_target: bool = False,
    include_world_model_history: bool = False,
    start_interaction_index: int = 0,
    candidate_action_count: int = 3,
    top_k: int = 3,
    rollout_temperature: float = 0.7,
    observation_source: str = "world_model",
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidate_action_count = max(1, int(candidate_action_count))
    top_k = max(1, int(top_k))
    imagined_react_system_prompt = imagined_trajectory_system_prompt(react_system_prompt)
    root_state = sanitize_state_content(previous_state)
    beams: list[dict[str, Any]] = [
        {
            "branch_id": "0",
            "parent_id": None,
            "depth": 0,
            "conversation": [dict(message) for message in conversation],
            "state": root_state,
            "state_history": append_state_history(
                list(state_history or []), root_state, max_items=state_history_size
            ),
            "input_history": list(input_history or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:],
            "imagined_steps": [],
            "seen_signatures": {},
            "score": 0.0,
            "terminal": False,
            "rollout_temperature": 0.0,
        }
    ]
    all_partial_rollouts: list[dict[str, Any]] = []
    branch_counter = 1

    for imagined_index in range(max(0, int(max_imagined_steps))):
        expanded: list[dict[str, Any]] = []
        for branch in beams:
            if branch.get("terminal"):
                expanded.append(branch)
                continue

            branch_conversation = [dict(message) for message in branch["conversation"]]
            raw_thought = _generate_with_optional_temperature(
                agent_generator,
                build_react_think_messages(
                    branch_conversation,
                    current_query=task.user_messages[-1] if task.user_messages else "",
                    system_prompt=imagined_react_system_prompt,
                ),
                temperature=0.0,
            )
            raw_thought = strip_model_thinking_output(raw_thought)
            thought_payload = parse_thought_payload(raw_thought)
            thought_conversation = branch_conversation + [
                {"role": "assistant", "content": json.dumps(thought_payload, ensure_ascii=False)}
            ]

            seen_candidate_signatures: set[str] = set()
            same_action_name_budget = max(1, candidate_action_count // 2)
            raw_action_batch = _generate_with_optional_temperature(
                agent_generator,
                build_react_action_batch_messages(
                    thought_conversation,
                    current_query=task.user_messages[-1] if task.user_messages else "",
                    system_prompt=imagined_react_system_prompt,
                    candidate_action_count=candidate_action_count,
                ),
                temperature=rollout_temperature if candidate_action_count > 1 else 0.0,
            )
            raw_action_batch = strip_model_thinking_output(raw_action_batch)
            try:
                candidate_decisions = parse_agent_candidate_decisions(
                    raw_action_batch,
                    expected_count=candidate_action_count,
                )
                batch_parse_error = None
            except Exception as exc:
                candidate_decisions = []
                batch_parse_error = str(exc)

            for candidate_index in range(candidate_action_count):
                temperature = rollout_temperature if candidate_action_count > 1 else 0.0
                if candidate_index < same_action_name_budget:
                    candidate_diversity_mode = "same_action_name_different_arguments"
                else:
                    candidate_diversity_mode = "different_action_name"
                if candidate_index < len(candidate_decisions):
                    decision, raw_action = candidate_decisions[candidate_index]
                    parse_error = None
                else:
                    decision = None
                    raw_action = raw_action_batch
                    parse_error = batch_parse_error or (
                        f"Batched action response returned {len(candidate_decisions)} "
                        f"candidate(s), expected {candidate_action_count}."
                    )
                child_id = f"{branch['branch_id']}.{candidate_index}"
                base_step = {
                    "imagined_step": imagined_index + 1,
                    "topk_branch_id": child_id,
                    "topk_parent_id": branch.get("branch_id"),
                    "topk_candidate_index": candidate_index,
                    "topk_candidate_diversity_mode": candidate_diversity_mode,
                    "topk_same_action_name_budget": same_action_name_budget,
                    "thought": thought_payload,
                    "raw_action": raw_action,
                    "raw_action_batch": raw_action_batch,
                }

                if decision is None:
                    step_score, score_reasons = 0.0, [f"parse_error=0:{parse_error}"]
                    step = {
                        **base_step,
                        "parse_error": parse_error,
                        "topk_score_delta": step_score,
                        "topk_score_reasons": score_reasons,
                        "topk_total_score": branch["score"] + step_score,
                    }
                    expanded.append(
                        {
                            **branch,
                            "branch_id": child_id,
                            "parent_id": branch.get("branch_id"),
                            "depth": imagined_index + 1,
                            "imagined_steps": branch["imagined_steps"] + [step],
                            "score": branch["score"] + step_score,
                            "terminal": True,
                            "rollout_temperature": temperature,
                        }
                    )
                    continue

                if "final_answer" in decision:
                    step_score, score_reasons = score_topk_imagined_step(
                        previous_state=branch["state"],
                        predicted_state=branch["state"],
                        feedbacks=[],
                        repeated_tool_call_count=0,
                        final_answer=True,
                    )
                    step = {
                        **base_step,
                        "final_answer": decision["final_answer"],
                        "predicted_state": full_world_model_state_for_agent(branch["state"]),
                        "predicted_state_summary": summarize_state_for_planning(branch["state"]),
                        "topk_score_delta": step_score,
                        "topk_score_reasons": score_reasons,
                        "topk_total_score": branch["score"] + step_score,
                    }
                    expanded.append(
                        {
                            **branch,
                            "branch_id": child_id,
                            "parent_id": branch.get("branch_id"),
                            "depth": imagined_index + 1,
                            "imagined_steps": branch["imagined_steps"] + [step],
                            "score": branch["score"] + step_score,
                            "terminal": True,
                            "rollout_temperature": temperature,
                        }
                    )
                    continue

                if "clarify" in decision:
                    step = {
                        **base_step,
                        "clarify": decision["clarify"],
                        "predicted_state": full_world_model_state_for_agent(branch["state"]),
                        "predicted_state_summary": summarize_state_for_planning(branch["state"]),
                        "topk_score_delta": 0.0,
                        "topk_score_reasons": ["clarify=0"],
                        "topk_total_score": branch["score"],
                    }
                    expanded.append(
                        {
                            **branch,
                            "branch_id": child_id,
                            "parent_id": branch.get("branch_id"),
                            "depth": imagined_index + 1,
                            "imagined_steps": branch["imagined_steps"] + [step],
                            "terminal": True,
                            "rollout_temperature": temperature,
                        }
                    )
                    continue

                planned_calls = [normalize_tool_call(call) for call in decision.get("tool_calls", [])]
                if not planned_calls:
                    step = {
                        **base_step,
                        "error": "empty_tool_calls",
                        "topk_score_delta": 0.0,
                        "topk_score_reasons": ["empty_tool_calls=0"],
                        "topk_total_score": branch["score"],
                    }
                    expanded.append(
                        {
                            **branch,
                            "branch_id": child_id,
                            "parent_id": branch.get("branch_id"),
                            "depth": imagined_index + 1,
                            "imagined_steps": branch["imagined_steps"] + [step],
                            "terminal": True,
                            "rollout_temperature": temperature,
                        }
                    )
                    continue

                signature = json_compact(planned_calls)
                local_duplicate_candidate = signature in seen_candidate_signatures
                seen_candidate_signatures.add(signature)
                seen_signatures = dict(branch["seen_signatures"])
                repeated_tool_call_count = seen_signatures.get(signature, 0) + 1
                seen_signatures[signature] = repeated_tool_call_count
                if local_duplicate_candidate:
                    repeated_tool_call_count = max(repeated_tool_call_count, 2)
                    step_score, score_reasons = score_topk_imagined_step(
                        previous_state=branch["state"],
                        predicted_state=branch["state"],
                        feedbacks=[],
                        repeated_tool_call_count=repeated_tool_call_count,
                    )
                    step = {
                        **base_step,
                        "tool_calls": planned_calls,
                        "duplicate_candidate_action": True,
                        "repeated_tool_call_loop": True,
                        "repeated_tool_call_count": repeated_tool_call_count,
                        "predicted_feedback": [],
                        "predicted_state": full_world_model_state_for_agent(branch["state"]),
                        "predicted_state_summary": summarize_state_for_planning(branch["state"]),
                        "raw_world_model_prediction": None,
                        "observation_source": observation_source,
                        "topk_score_delta": step_score,
                        "topk_score_reasons": score_reasons,
                        "topk_total_score": branch["score"] + step_score,
                    }
                    expanded.append(
                        {
                            **branch,
                            "branch_id": child_id,
                            "parent_id": branch.get("branch_id"),
                            "depth": imagined_index + 1,
                            "imagined_steps": branch["imagined_steps"] + [step],
                            "score": branch["score"] + step_score,
                            "terminal": True,
                            "rollout_temperature": temperature,
                        }
                    )
                    continue

                feedbacks: list[dict[str, Any]] = []
                predicted_state = branch["state"]
                if observation_source == "world_model":
                    feedbacks = predict_world_model_feedback(
                        world_model_generator,
                        task,
                        branch["state"],
                        planned_calls,
                        interaction_index=start_interaction_index + imagined_index,
                        world_model_target=world_model_target,
                        include_error_message_in_target=include_error_message_in_target,
                        include_stage_in_target=include_stage_in_target,
                        include_world_model_history=include_world_model_history,
                        state_history=branch["state_history"],
                        input_history=branch["input_history"],
                        system_prompt_max_chars=system_prompt_max_chars,
                        action_max_chars=action_max_chars,
                    )
                    predicted_state = feedbacks[-1].get("predicted_state") if feedbacks else branch["state"]

                step_score, score_reasons = score_topk_imagined_step(
                    previous_state=branch["state"],
                    predicted_state=predicted_state,
                    feedbacks=feedbacks,
                    repeated_tool_call_count=repeated_tool_call_count,
                )
                next_score = branch["score"] + step_score
                next_input_history = append_world_model_input_history(
                    branch["input_history"],
                    make_imagined_world_model_history_entry(
                        imagined_step=imagined_index + 1,
                        action=planned_calls,
                        state=predicted_state,
                    ),
                )
                next_conversation = [dict(message) for message in thought_conversation]
                append_imagined_observation_for_planning(
                    next_conversation,
                    planned_calls,
                    feedbacks,
                    predicted_state,
                    observation_source,
                )
                next_state_history = branch["state_history"]
                if predicted_state is not None:
                    next_state_history = append_state_history(
                        next_state_history, predicted_state, max_items=state_history_size
                    )
                terminal = repeated_tool_call_count > 1 or state_is_finished(predicted_state)
                step = {
                    **base_step,
                    "tool_calls": planned_calls,
                    "repeated_tool_call_loop": repeated_tool_call_count > 1,
                    "repeated_tool_call_count": repeated_tool_call_count,
                    "predicted_feedback": feedbacks,
                    "predicted_state": full_world_model_state_for_agent(predicted_state),
                    "predicted_state_summary": (
                        summarize_state_for_planning(predicted_state) if predicted_state else None
                    ),
                    "raw_world_model_prediction": feedbacks[-1].get("raw_prediction") if feedbacks else None,
                    "observation_source": observation_source,
                    "topk_score_delta": step_score,
                    "topk_score_reasons": score_reasons,
                    "topk_total_score": next_score,
                }
                expanded.append(
                    {
                        "branch_id": child_id or str(branch_counter),
                        "parent_id": branch.get("branch_id"),
                        "depth": imagined_index + 1,
                        "conversation": next_conversation,
                        "state": predicted_state,
                        "state_history": next_state_history,
                        "input_history": next_input_history,
                        "imagined_steps": branch["imagined_steps"] + [step],
                        "seen_signatures": seen_signatures,
                        "score": next_score,
                        "terminal": terminal,
                        "rollout_temperature": temperature,
                    }
                )
                branch_counter += 1

        if not expanded:
            break
        expanded.sort(
            key=lambda item: (
                float(item.get("score", 0.0)),
                int(item.get("depth", 0)),
                0 if item.get("terminal") else 1,
            ),
            reverse=True,
        )
        beams = expanded[:top_k]
        all_partial_rollouts.extend(
            make_topk_rollout_record(
                len(all_partial_rollouts) + index,
                branch,
                observation_source=observation_source,
            )
            for index, branch in enumerate(beams)
        )
        if all(branch.get("terminal") for branch in beams):
            break

    final_rollouts = [
        make_topk_rollout_record(index, branch, observation_source=observation_source)
        for index, branch in enumerate(beams)
    ]
    if not final_rollouts:
        selection = {
            "selected_index": 0,
            "scores": [],
            "comments": "topk_search produced no candidates",
            "fallback_used": True,
            "selection_strategy": "topk_search",
            "candidate_action_count": candidate_action_count,
            "top_k": top_k,
        }
        return [], [], selection

    scores = [
        {
            "index": index,
            "score": rollout.get("score", 0.0),
            "reason": " + ".join(
                str(reason)
                for step in rollout.get("imagined_steps", [])
                for reason in step.get("topk_score_reasons", [])
            ),
        }
        for index, rollout in enumerate(final_rollouts)
    ]
    selection = {
        "selected_index": 0,
        "scores": scores,
        "comments": "Selected highest-scoring top-k imagined trajectory",
        "fallback_used": False,
        "selection_strategy": "topk_search",
        "candidate_action_count": candidate_action_count,
        "top_k": top_k,
        "partial_frontier_records": all_partial_rollouts,
    }
    return final_rollouts[0]["imagined_steps"], final_rollouts, selection


def imagine_trajectory_candidates(
    agent_generator: Any,
    world_model_generator: Any,
    task: TaskTrajectory,
    conversation: list[dict[str, Any]],
    previous_state: dict[str, Any],
    react_system_prompt: str,
    max_imagined_steps: int,
    world_model_target: str,
    include_error_message_in_target: bool = False,
    include_stage_in_target: bool = False,
    include_world_model_history: bool = False,
    start_interaction_index: int = 0,
    num_rollouts: int = 1,
    rollout_temperature: float = 0.7,
    selection_strategy: str = "llm_judge",
    observation_source: str = "world_model",
    candidate_action_count: int = 3,
    top_k: int = 3,
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if selection_strategy == "topk_search":
        return imagine_trajectory_topk_search(
            agent_generator=agent_generator,
            world_model_generator=world_model_generator,
            task=task,
            conversation=conversation,
            previous_state=previous_state,
            react_system_prompt=react_system_prompt,
            max_imagined_steps=max_imagined_steps,
            world_model_target=world_model_target,
            include_error_message_in_target=include_error_message_in_target,
            include_stage_in_target=include_stage_in_target,
            include_world_model_history=include_world_model_history,
            start_interaction_index=start_interaction_index,
            candidate_action_count=candidate_action_count,
            top_k=top_k,
            rollout_temperature=rollout_temperature,
            observation_source=observation_source,
            state_history=state_history,
            input_history=input_history,
            state_history_size=state_history_size,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
        )

    candidate_rollouts: list[dict[str, Any]] = []
    for rollout_index in range(max(num_rollouts, 1)):
        temperature = 0.0 if rollout_index == 0 else rollout_temperature
        imagined_steps = imagine_trajectory(
            agent_generator=agent_generator,
            world_model_generator=world_model_generator,
            task=task,
            conversation=conversation,
            previous_state=previous_state,
            react_system_prompt=react_system_prompt,
            max_imagined_steps=max_imagined_steps,
            world_model_target=world_model_target,
            include_error_message_in_target=include_error_message_in_target,
            include_stage_in_target=include_stage_in_target,
            include_world_model_history=include_world_model_history,
            start_interaction_index=start_interaction_index,
            rollout_index=rollout_index,
            rollout_temperature=temperature,
            observation_source=observation_source,
            state_history=state_history,
            input_history=input_history,
            state_history_size=state_history_size,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
        )
        candidate_rollouts.append(
            {
                "rollout_index": rollout_index,
                "rollout_temperature": temperature,
                "observation_source": observation_source,
                "imagined_steps": imagined_steps,
            }
        )

    if len(candidate_rollouts) == 1 or selection_strategy == "first":
        selection = {
            "selected_index": 0,
            "scores": [],
            "comments": "Selected first imagined trajectory",
            "fallback_used": False,
            "selection_strategy": selection_strategy,
        }
        return (
            candidate_rollouts[0]["imagined_steps"],
            candidate_rollouts,
            selection,
        )

    selection = select_imagined_trajectory_with_llm_judge(
        agent_generator=agent_generator,
        task=task,
        conversation=conversation,
        candidate_rollouts=candidate_rollouts,
    )
    selection["selection_strategy"] = selection_strategy
    selected_index = int(selection.get("selected_index", 0))
    return (
        candidate_rollouts[selected_index]["imagined_steps"],
        candidate_rollouts,
        selection,
    )


def imagine_revision_rollout(
    agent_generator: Any,
    world_model_generator: Any,
    task: TaskTrajectory,
    conversation: list[dict[str, Any]],
    previous_state: dict[str, Any],
    react_system_prompt: str,
    initial_planned_calls: list[dict[str, Any]],
    lookahead_steps: int,
    world_model_target: str,
    include_error_message_in_target: bool,
    include_stage_in_target: bool,
    include_world_model_history: bool,
    start_interaction_index: int,
    rollout_index: int,
    rollout_temperature: float,
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> dict[str, Any]:
    """Roll out `lookahead_steps` ahead, starting from `initial_planned_calls`.

    Step 1 is the agent's already-decided `initial_planned_calls`. Steps 2..K
    are imagined: the agent picks the next tool calls based on the world
    model's prediction of the previous step. The first step's feedbacks are
    returned separately so the caller can keep the existing single-step
    revision telemetry intact.
    """
    imagined_conversation = [dict(message) for message in conversation]
    imagined_react_system_prompt = imagined_trajectory_system_prompt(react_system_prompt)
    imagined_state = sanitize_state_content(previous_state)
    imagined_state_history = append_state_history(
        list(state_history or []), imagined_state, max_items=state_history_size
    )
    imagined_input_history = list(input_history or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:]
    steps: list[dict[str, Any]] = []
    first_step_feedbacks: list[dict[str, Any]] = []
    seen_signatures: dict[str, int] = {}

    current_planned_calls = initial_planned_calls
    for step_offset in range(max(lookahead_steps, 1)):
        feedbacks = predict_world_model_feedback(
            world_model_generator,
            task,
            imagined_state,
            current_planned_calls,
            interaction_index=start_interaction_index + step_offset,
            world_model_target=world_model_target,
            include_error_message_in_target=include_error_message_in_target,
            include_stage_in_target=include_stage_in_target,
            include_world_model_history=include_world_model_history,
            state_history=imagined_state_history,
            input_history=imagined_input_history,
            system_prompt_max_chars=system_prompt_max_chars,
            action_max_chars=action_max_chars,
        )
        if step_offset == 0:
            first_step_feedbacks = feedbacks

        last_feedback = feedbacks[-1] if feedbacks else {}
        predicted_state = last_feedback.get("predicted_state")
        predicted_tool_output = (last_feedback.get("predicted_tool_output") or "").strip()
        predicted_error_message = (last_feedback.get("predicted_error_message") or "").strip()
        imagined_input_history = append_world_model_input_history(
            imagined_input_history,
            make_imagined_world_model_history_entry(
                imagined_step=step_offset + 1,
                action=current_planned_calls,
                state=predicted_state,
            ),
        )

        steps.append(
            {
                "step_offset": step_offset,
                "tool_calls": current_planned_calls,
                "predicted_feedback": feedbacks,
                "predicted_state": full_world_model_state_for_agent(predicted_state),
                "predicted_state_summary": (
                    summarize_state_for_planning(predicted_state) if predicted_state else None
                ),
                "raw_world_model_prediction": last_feedback.get("raw_prediction"),
                "predicted_tool_output_preview": predicted_tool_output[:400] if predicted_tool_output else "",
                "predicted_error_message": predicted_error_message,
            }
        )

        imagined_conversation.append(
            {"role": "assistant", "tool_calls": to_openai_tool_calls(current_planned_calls)}
        )
        if predicted_tool_output:
            imagined_conversation.append(
                {
                    "role": "tool",
                    "name": current_planned_calls[0].get("name", "") if current_planned_calls else "",
                    "content": (
                        "[IMAGINED_TOOL_OUTPUT_FROM_WORLD_MODEL]\n" + predicted_tool_output
                    ),
                }
            )
        else:
            imagined_conversation.append(
                {
                    "role": "user",
                    "content": (
                        "Imagined observation based on world-model prediction:\n"
                        + json.dumps(
                            world_model_state_observation_payload(
                                predicted_state,
                                predicted_error_message or None,
                                last_feedback.get("raw_prediction"),
                            ),
                            ensure_ascii=False,
                        )
                    ),
                }
            )

        if predicted_state is not None:
            imagined_state = predicted_state
            imagined_state_history = append_state_history(
                imagined_state_history, imagined_state, max_items=state_history_size
            )
            if state_is_finished(predicted_state):
                break

        if step_offset + 1 >= max(lookahead_steps, 1):
            break

        # Agent picks the next imagined step based on the imagined conversation.
        try:
            raw_action = agent_generator.generate_from_messages(
                build_react_action_messages(
                    imagined_conversation,
                    current_query=task.user_messages[-1] if task.user_messages else "",
                    system_prompt=imagined_react_system_prompt,
                ),
                temperature=rollout_temperature,
            )
        except TypeError:
            # Generator does not accept temperature kwarg; fall back to default.
            raw_action = agent_generator.generate_from_messages(
                build_react_action_messages(
                    imagined_conversation,
                    current_query=task.user_messages[-1] if task.user_messages else "",
                    system_prompt=imagined_react_system_prompt,
                )
            )
        raw_action = strip_model_thinking_output(raw_action)
        try:
            decision = parse_agent_decision(raw_action)
        except Exception as exc:
            steps.append(
                {
                    "step_offset": step_offset + 1,
                    "imagined_parse_error": str(exc),
                }
            )
            break
        if "final_answer" in decision or "clarify" in decision:
            steps.append(
                {
                    "step_offset": step_offset + 1,
                    "imagined_decision": {
                        k: v for k, v in decision.items() if k in {"final_answer", "clarify"}
                    },
                }
            )
            break
        next_calls = [normalize_tool_call(call) for call in decision.get("tool_calls", [])]
        if not next_calls:
            break
        signature = json_compact(next_calls)
        seen_signatures[signature] = seen_signatures.get(signature, 0) + 1
        if seen_signatures[signature] > 1:
            steps.append(
                {
                    "step_offset": step_offset + 1,
                    "imagined_repeated_tool_call_loop": True,
                    "tool_calls": next_calls,
                }
            )
            break
        current_planned_calls = next_calls

    all_predicted_success = bool(steps) and all(
        all(fb.get("predicted_success", False) for fb in step.get("predicted_feedback", []))
        for step in steps
        if step.get("predicted_feedback")
    )
    any_predicted_failure = any(
        any(not fb.get("predicted_success", True) for fb in step.get("predicted_feedback", []))
        for step in steps
        if step.get("predicted_feedback")
    )

    return {
        "rollout_index": rollout_index,
        "rollout_temperature": rollout_temperature,
        "lookahead_steps_taken": len(steps),
        "lookahead_steps_target": lookahead_steps,
        "steps": steps,
        "first_step_feedbacks": first_step_feedbacks,
        "all_predicted_success": all_predicted_success,
        "any_predicted_failure": any_predicted_failure,
        "terminal_state": full_world_model_state_for_agent(imagined_state),
        "terminal_state_summary": (
            summarize_state_for_planning(imagined_state) if imagined_state else None
        ),
    }


def imagine_revision_rollouts(
    agent_generator: Any,
    world_model_generator: Any,
    task: TaskTrajectory,
    conversation: list[dict[str, Any]],
    previous_state: dict[str, Any],
    react_system_prompt: str,
    initial_planned_calls: list[dict[str, Any]],
    lookahead_steps: int,
    num_rollouts: int,
    world_model_target: str,
    include_error_message_in_target: bool,
    include_stage_in_target: bool,
    include_world_model_history: bool,
    start_interaction_index: int,
    rollout_temperature: float,
    state_history: list[dict[str, Any]] | None = None,
    input_history: list[dict[str, Any]] | None = None,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
) -> list[dict[str, Any]]:
    """Run N independent K-step rollouts of the same `initial_planned_calls`.

    The first rollout is deterministic (temperature=0); the remaining rollouts
    use `rollout_temperature` for diversity in the agent's continuation choices.
    """
    rollouts: list[dict[str, Any]] = []
    for index in range(max(num_rollouts, 1)):
        temperature = 0.0 if index == 0 else rollout_temperature
        rollouts.append(
            imagine_revision_rollout(
                agent_generator=agent_generator,
                world_model_generator=world_model_generator,
                task=task,
                conversation=conversation,
                previous_state=previous_state,
                react_system_prompt=react_system_prompt,
                initial_planned_calls=initial_planned_calls,
                lookahead_steps=lookahead_steps,
                world_model_target=world_model_target,
                include_error_message_in_target=include_error_message_in_target,
                include_stage_in_target=include_stage_in_target,
                include_world_model_history=include_world_model_history,
                start_interaction_index=start_interaction_index,
                rollout_index=index,
                rollout_temperature=temperature,
                state_history=state_history,
                input_history=input_history,
                state_history_size=state_history_size,
                system_prompt_max_chars=system_prompt_max_chars,
                action_max_chars=action_max_chars,
            )
        )
    return rollouts


def imagined_step_is_unsatisfactory(imagined_step: dict[str, Any]) -> bool:
    if imagined_step.get("repeated_tool_call_loop"):
        return True
    feedbacks = imagined_step.get("predicted_feedback") or []
    if not feedbacks:
        return True
    final_feedback = feedbacks[-1]
    if final_feedback.get("parse_error"):
        return True
    if (
        final_feedback.get("predicted_state") is None
        and final_feedback.get("predicted_current_stage") is None
        and final_feedback.get("predicted_remaining_stages") is None
        and final_feedback.get("predicted_success") is None
    ):
        return True
    return not final_feedback.get("predicted_success", False)


def update_state_from_actual_execution(
    previous_state: dict[str, Any],
    execution_results: list[dict[str, Any]],
    predicted_feedbacks: list[dict[str, Any]] | None = None,
    trust_predicted_state: bool = True,
) -> dict[str, Any]:
    predicted_state = None
    if predicted_feedbacks:
        predicted_state = predicted_feedbacks[-1].get("predicted_state")

    state_seed = predicted_state if trust_predicted_state and predicted_state is not None else previous_state
    next_state = sanitize_state_content(state_seed if state_seed is not None else make_blank_state())

    final_result = execution_results[-1] if execution_results else None
    if final_result is not None:
        tool_name = final_result.get("resolved_name") or final_result.get("requested_name")
        success = infer_execution_result_success(final_result)
        content = final_result.get("content", "")
        if is_enterprise_state_payload(next_state):
            state_root = next_state.setdefault("state", {})
            target = state_root.get("diff_from_previous_state")
            if not isinstance(target, dict):
                target = state_root
            outcome = target.setdefault("outcome", {})
            if isinstance(outcome, dict):
                outcome["status"] = "success" if success else "failure"
                outcome["summary"] = str(content)[:1000] if content is not None else ""
                outcome.setdefault("failure_category", "none" if success else "tool_error")
                outcome.setdefault("recoverable", not success)
            history = target.setdefault("history_context", {})
            if isinstance(history, dict):
                events = history.setdefault("last_tool_events", [])
                if isinstance(events, list):
                    events.append(
                        {
                            "tool_name": tool_name,
                            "status": "success" if success else "failure",
                            "operation": "execute",
                            "summary": str(content)[:1000] if content is not None else "",
                            "error": "" if success else str(content),
                        }
                    )
        elif is_compact_tool_execution_state(next_state):
            next_state["success"] = success
            next_state["last_tool_execution_result"] = 1 if success else 0
            next_state["last_tool_name"] = tool_name
            next_state["error_message"] = "" if success else str(content)
            next_state.setdefault("current_stage", state_current_stage(previous_state))
            next_state.setdefault("remaining_stages", state_remaining_stages(previous_state) or [])
        else:
            state_root = next_state.setdefault("state", {})
            context = state_root.setdefault("context", {})
            context["last_tool_execution_result"] = 1 if success else 0
            context["last_tool_name"] = tool_name
            context["error_message"] = "" if success else str(content)

    return next_state


def summarize_mode_metrics(
    mode_name: str,
    use_world_model_internal_thinking: bool,
    max_steps: int,
    completed_tasks: int,
    completed_step_counts: list[int],
    completed_tool_call_counts: list[int],
    total_tool_steps_taken: int,
    total_tool_calls_taken: int,
    total_internal_thinking_iterations: int,
    task_records: list[dict[str, Any]],
) -> dict[str, Any]:
    average_steps = sum(completed_step_counts) / len(completed_step_counts) if completed_step_counts else None
    average_tool_calls_per_trajectory = (
        sum(record["tool_calls_taken"] for record in task_records) / len(task_records) if task_records else None
    )
    average_imagined_rollouts = (
        sum(record.get("imagined_rollouts_used", 0) for record in task_records) / len(task_records)
        if task_records
        else None
    )
    average_tool_calls_to_complete = (
        sum(completed_tool_call_counts) / len(completed_tool_call_counts)
        if completed_tool_call_counts
        else None
    )
    average_internal_thinking_iterations = (
        total_internal_thinking_iterations / len(task_records) if task_records else None
    )
    return {
        "mode": mode_name,
        "use_world_model_internal_thinking": use_world_model_internal_thinking,
        "evaluated_tasks": len(task_records),
        "max_steps_per_task": max_steps,
        "total_tool_steps_taken": total_tool_steps_taken,
        "total_tool_calls_taken": total_tool_calls_taken,
        "total_internal_thinking_iterations": total_internal_thinking_iterations,
        "average_internal_thinking_iterations_per_task": average_internal_thinking_iterations,
        "completed_tasks": completed_tasks,
        "completion_rate": completed_tasks / len(task_records) if task_records else 0.0,
        "average_steps_to_complete": average_steps,
        "average_tool_calls_per_trajectory": average_tool_calls_per_trajectory,
        "average_tool_calls_to_complete": average_tool_calls_to_complete,
        "average_imagined_rollouts_per_task": average_imagined_rollouts,
        "task_records": task_records,
    }


def evaluate_world_model_predictions(
    generator: HFTextGenerator,
    examples: list[WorldModelStateExample],
    sample_limit: int,
    target_mode: str = "state",
    include_error_message: bool = False,
    include_stage: bool = False,
    include_input_history: bool = False,
) -> dict[str, Any]:
    target_mode = canonicalize_world_model_target(target_mode)
    stage_metrics_enabled = (
        target_mode == WORLD_MODEL_TARGET_STATE
        or (include_stage and target_mode == WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY)
    )
    canonical_event_state_target = is_canonical_event_state_target(target_mode)
    canonical_event_with_nudge_target = is_canonical_event_with_nudge_target(target_mode)
    canonical_target = canonical_event_state_target or canonical_event_with_nudge_target
    if not examples:
        run_outcome_error_judge = (
            include_error_message and is_tool_execution_result_target(target_mode)
        )
        empty_metrics = {
            "sampled_examples": 0,
            "exact_match": 0.0,
            "last_tool_execution_result_accuracy": 0.0,
            "binary_classification_accuracy": (
                0.0 if is_tool_execution_result_target(target_mode) else None
            ),
            "outcome_with_error_judge_accuracy": 0.0 if run_outcome_error_judge else None,
            "outcome_with_error_judge_score": 0.0 if run_outcome_error_judge else None,
            "outcome_with_error_judge_evaluated": 0 if run_outcome_error_judge else None,
            "llm_judge_score": (
                0.0
                if target_mode == WORLD_MODEL_TARGET_STATE or is_tool_output_target(target_mode)
                else None
            ),
            "llm_judge_match_rate": (
                0.0
                if target_mode == WORLD_MODEL_TARGET_STATE or is_tool_output_target(target_mode)
                else None
            ),
            "field_level_llm_judge_score": 0.0 if target_mode == WORLD_MODEL_TARGET_STATE else None,
            "field_level_llm_judge_match_rate": 0.0 if target_mode == WORLD_MODEL_TARGET_STATE else None,
            "field_level_llm_judge_scores": {} if target_mode == WORLD_MODEL_TARGET_STATE else None,
            "current_stage_accuracy": 0.0 if stage_metrics_enabled else None,
            "remaining_stages_accuracy": 0.0 if stage_metrics_enabled else None,
            "canonical_field_accuracy": 0.0 if canonical_target else None,
            "canonical_field_macro_accuracy": 0.0 if canonical_target else None,
            "canonical_full_match_rate": 0.0 if canonical_target else None,
            "canonical_scored_examples": 0 if canonical_target else None,
            "canonical_field_accuracies": {} if canonical_target else None,
        }
        return empty_metrics

    sample_size = len(examples) if sample_limit <= 0 else min(sample_limit, len(examples))
    sampled_examples = examples[:sample_size]
    exact_matches = 0
    parse_successes = 0
    last_tool_execution_result_matches = 0
    llm_judge_score_sum = 0.0
    llm_judge_matches = 0
    field_level_judge_score_sum = 0.0
    field_level_judge_matches = 0
    field_level_judge_count = 0
    field_level_judge_score_sums: dict[str, float] = {}
    field_level_judge_match_sums: dict[str, int] = {}
    field_level_judge_counts: dict[str, int] = {}
    current_stage_matches = 0
    remaining_stages_matches = 0
    outcome_error_judge_score_sum = 0.0
    outcome_error_judge_matches = 0
    outcome_error_judge_evaluated = 0
    canonical_field_correct: dict[str, int] = {}
    canonical_field_total: dict[str, int] = {}
    canonical_full_matches = 0
    canonical_scored_examples = 0
    records = []
    run_outcome_error_judge = (
        include_error_message and is_tool_execution_result_target(target_mode)
    )

    for example in tqdm(sampled_examples, total=len(sampled_examples)):
        prediction = generator.generate_from_messages(
            build_state_prediction_chat_messages(
                example,
                target_mode=target_mode,
                include_error_message=include_error_message,
                include_stage=include_stage,
                include_input_history=include_input_history,
            )
        )
        prediction = strip_model_thinking_output(prediction)
        gold_last_tool_execution_result = normalize_tool_execution_result_for_target(
            extract_last_tool_execution_result_from_state(example.state),
            target_mode=target_mode,
        )
        gold_canonical_payload: dict[str, Any] | None = None
        if target_mode == WORLD_MODEL_TARGET_STATE:
            gold_target = normalize_state_text(example.state)
        elif is_tool_output_target(target_mode):
            gold_target = example.tool_output or ""
        elif canonical_target:
            if example.canonical_label is not None:
                gold_canonical_payload = example.canonical_label
            elif canonical_event_with_nudge_target:
                gold_canonical_payload = canonical_event_with_nudge_from_action_state(
                    example.action, example.state
                )
            else:
                gold_canonical_payload = canonical_event_from_action_state(
                    example.action, example.state
                )
            gold_target = json.dumps(gold_canonical_payload, ensure_ascii=False, sort_keys=True)
        elif gold_last_tool_execution_result is None:
            gold_target = "None"
        else:
            gold_target = format_tool_execution_result_target(
                gold_last_tool_execution_result,
                example.error_payload,
                include_error_message=include_error_message,
                include_stage=(
                    include_stage
                    and target_mode == WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY
                ),
                current_stage=state_current_stage(example.state),
                remaining_stages=state_remaining_stages(example.state),
            )
        exact_match = False
        parse_success = False
        predicted_last_tool_execution_result = None
        predicted_error_message = ""
        predicted_current_stage = None
        predicted_remaining_stages = None
        if is_tool_output_target(target_mode):
            judge_zero_scores = {key: 0.0 for key in TOOL_OUTPUT_MATCH_JUDGE_KEYS}
        else:
            judge_zero_scores = {key: 0.0 for key in STATE_MATCH_JUDGE_KEYS}
        llm_judge_result = {
            "scores": judge_zero_scores,
            "overall_score": 0.0,
            "match": False,
            "comments": "Prediction could not be judged",
            "raw_response": {},
        }
        outcome_error_judge_result = None
        canonical_field_comparisons: dict[str, dict[str, Any]] | None = None
        parse_error = None
        try:
            if is_tool_execution_result_target(target_mode):
                parsed_prediction_payload = parse_binary_world_model_prediction(prediction)
                predicted_last_tool_execution_result = normalize_tool_execution_result_for_target(
                    parsed_prediction_payload.get("success"),
                    target_mode=target_mode,
                )
                predicted_error_message = parsed_prediction_payload.get("error_message", "") or ""
                predicted_current_stage = parsed_prediction_payload.get("current_stage")
                predicted_remaining_stages = parsed_prediction_payload.get("remaining_stages")
                parse_success = predicted_last_tool_execution_result is not None
                exact_match = parse_success and predicted_last_tool_execution_result == gold_last_tool_execution_result
                if include_stage and target_mode == WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY:
                    gold_payload = parse_binary_world_model_prediction(gold_target)
                    exact_match = exact_match and predicted_current_stage == gold_payload.get("current_stage")
                    exact_match = exact_match and predicted_remaining_stages == gold_payload.get("remaining_stages")
                    if include_error_message:
                        exact_match = exact_match and predicted_error_message == gold_payload.get("error_message", "")
            elif is_tool_output_target(target_mode):
                normalized_prediction = (prediction or "").strip()
                parse_success = True
                exact_match = normalized_prediction == (gold_target or "").strip()
                llm_judge_result = evaluate_tool_output_match_quality(
                    system_prompt=example.system_prompt,
                    user_prompt=example.user_prompt,
                    previous_state=example.previous_state,
                    action=example.action,
                    predicted_tool_output=normalized_prediction,
                    target_tool_output=(gold_target or "").strip(),
                )
            elif canonical_target:
                parsed_prediction = parse_jsonish(prediction)
                parse_success = isinstance(parsed_prediction, dict)
                normalized_prediction = json.dumps(
                    parsed_prediction, ensure_ascii=False, sort_keys=True
                )
                exact_match = normalized_prediction == gold_target
                canonical_field_comparisons = canonical_field_matches(
                    gold_canonical_payload or {},
                    parsed_prediction if parse_success else {},
                    include_nudge=canonical_event_with_nudge_target,
                )
            else:
                parsed_prediction = sanitize_state_content(parse_jsonish(prediction))
                normalized_prediction = json.dumps(parsed_prediction, ensure_ascii=False, sort_keys=True)
                exact_match = normalized_prediction == gold_target
                parse_success = True
                llm_judge_result = evaluate_state_match_quality(
                    system_prompt=example.system_prompt,
                    user_prompt=example.user_prompt,
                    previous_state=example.previous_state,
                    action=example.action,
                    predicted_state_text=normalized_prediction,
                    target_state_text=gold_target,
                )
                predicted_last_tool_execution_result = normalize_tool_execution_result_for_target(
                    extract_last_tool_execution_result_from_state(parsed_prediction),
                    target_mode=target_mode,
                )
                predicted_current_stage = state_current_stage(parsed_prediction)
                predicted_remaining_stages = state_remaining_stages(parsed_prediction)
        except Exception as exc:
            parse_error = str(exc)
            print(parse_error)

        if run_outcome_error_judge:
            outcome_error_judge_result = evaluate_outcome_error_match_quality(
                system_prompt=example.system_prompt,
                user_prompt=example.user_prompt,
                previous_state=example.previous_state,
                action=example.action,
                predicted_label=predicted_last_tool_execution_result,
                predicted_error_message=predicted_error_message,
                gold_label=gold_last_tool_execution_result,
                gold_error_message=example.error_payload,
            )
            outcome_error_judge_evaluated += 1
            outcome_error_judge_score_sum += outcome_error_judge_result["overall_score"]
            outcome_error_judge_matches += int(outcome_error_judge_result["match"])

        exact_matches += int(exact_match)
        parse_successes += int(parse_success)
        llm_judge_score_sum += llm_judge_result["overall_score"]
        llm_judge_matches += int(llm_judge_result["match"])
        if canonical_target:
            if canonical_field_comparisons is None:
                # Parsing raised before comparison; count every gold field wrong
                # rather than silently dropping the example from the denominator.
                canonical_field_comparisons = canonical_field_matches(
                    gold_canonical_payload or {},
                    {},
                    include_nudge=canonical_event_with_nudge_target,
                )
            canonical_scored_examples += 1
            all_fields_match = bool(canonical_field_comparisons)
            for field_path, comparison in canonical_field_comparisons.items():
                canonical_field_total[field_path] = canonical_field_total.get(field_path, 0) + 1
                if comparison["match"]:
                    canonical_field_correct[field_path] = (
                        canonical_field_correct.get(field_path, 0) + 1
                    )
                else:
                    all_fields_match = False
            canonical_full_matches += int(all_fields_match)
        if target_mode == WORLD_MODEL_TARGET_STATE:
            for field_path, field_result in llm_judge_result.get("field_scores", {}).items():
                field_score = clamp_score(field_result.get("score", 0.0))
                field_match = bool(field_result.get("match", field_score >= 0.8))
                field_level_judge_score_sum += field_score
                field_level_judge_matches += int(field_match)
                field_level_judge_count += 1
                field_level_judge_score_sums[field_path] = (
                    field_level_judge_score_sums.get(field_path, 0.0) + field_score
                )
                field_level_judge_match_sums[field_path] = (
                    field_level_judge_match_sums.get(field_path, 0) + int(field_match)
                )
                field_level_judge_counts[field_path] = field_level_judge_counts.get(field_path, 0) + 1
        gold_current_stage = state_current_stage(example.state)
        gold_remaining_stages = state_remaining_stages(example.state)
        last_tool_execution_result_match = (
            predicted_last_tool_execution_result == gold_last_tool_execution_result
        )
        current_stage_match = predicted_current_stage == gold_current_stage
        remaining_stages_match = predicted_remaining_stages == gold_remaining_stages
        last_tool_execution_result_matches += int(last_tool_execution_result_match)
        current_stage_matches += int(current_stage_match)
        remaining_stages_matches += int(remaining_stages_match)

        record = {
            "trajectory_id": example.trajectory_id,
            "trajectory_index": example.trajectory_index,
            "interaction_index": example.interaction_index,
            "parse_success": parse_success,
            "exact_match": exact_match,
            "prediction": prediction,
            "gold": gold_target,
            "llm_judge": llm_judge_result,
            "predicted_last_tool_execution_result": predicted_last_tool_execution_result,
            "gold_last_tool_execution_result": gold_last_tool_execution_result,
            "last_tool_execution_result_match": last_tool_execution_result_match,
            "predicted_current_stage": predicted_current_stage,
            "gold_current_stage": gold_current_stage,
            "current_stage_match": current_stage_match,
            "predicted_remaining_stages": predicted_remaining_stages,
            "gold_remaining_stages": gold_remaining_stages,
            "remaining_stages_match": remaining_stages_match,
        }
        if run_outcome_error_judge:
            record["predicted_error_message"] = predicted_error_message
            record["gold_error_message"] = example.error_payload
            record["outcome_with_error_judge"] = outcome_error_judge_result
        if canonical_target:
            record["canonical_field_comparisons"] = canonical_field_comparisons
            record["canonical_full_match"] = (
                bool(canonical_field_comparisons)
                and all(c["match"] for c in canonical_field_comparisons.values())
            )
        if parse_error:
            record["parse_error"] = parse_error

        print(record)

        records.append(record)

    metrics = {
        "sampled_examples": sample_size,
        "exact_match": exact_matches / sample_size,
        "parse_success_rate": parse_successes / sample_size,
        "last_tool_execution_result_accuracy": last_tool_execution_result_matches / sample_size,
        "binary_classification_accuracy": (
            last_tool_execution_result_matches / sample_size
            if is_tool_execution_result_target(target_mode)
            else None
        ),
        "outcome_with_error_judge_accuracy": (
            outcome_error_judge_matches / outcome_error_judge_evaluated
            if run_outcome_error_judge and outcome_error_judge_evaluated > 0
            else None
        ),
        "outcome_with_error_judge_score": (
            outcome_error_judge_score_sum / outcome_error_judge_evaluated
            if run_outcome_error_judge and outcome_error_judge_evaluated > 0
            else None
        ),
        "outcome_with_error_judge_evaluated": (
            outcome_error_judge_evaluated if run_outcome_error_judge else None
        ),
        "llm_judge_score": (
            llm_judge_score_sum / sample_size
            if target_mode == WORLD_MODEL_TARGET_STATE or is_tool_output_target(target_mode)
            else None
        ),
        "llm_judge_match_rate": (
            llm_judge_matches / sample_size
            if target_mode == WORLD_MODEL_TARGET_STATE or is_tool_output_target(target_mode)
            else None
        ),
        "field_level_llm_judge_score": (
            field_level_judge_score_sum / field_level_judge_count
            if target_mode == WORLD_MODEL_TARGET_STATE and field_level_judge_count > 0
            else None
        ),
        "field_level_llm_judge_match_rate": (
            field_level_judge_matches / field_level_judge_count
            if target_mode == WORLD_MODEL_TARGET_STATE and field_level_judge_count > 0
            else None
        ),
        "field_level_llm_judge_scores": (
            {
                field_path: {
                    "average_score": field_level_judge_score_sums[field_path] / field_level_judge_counts[field_path],
                    "match_rate": field_level_judge_match_sums[field_path] / field_level_judge_counts[field_path],
                    "count": field_level_judge_counts[field_path],
                }
                for field_path in sorted(field_level_judge_counts)
            }
            if target_mode == WORLD_MODEL_TARGET_STATE
            else None
        ),
        "current_stage_accuracy": current_stage_matches / sample_size if stage_metrics_enabled else None,
        "remaining_stages_accuracy": remaining_stages_matches / sample_size if stage_metrics_enabled else None,
        "canonical_field_accuracy": (
            sum(canonical_field_correct.values()) / sum(canonical_field_total.values())
            if canonical_target and sum(canonical_field_total.values()) > 0
            else (0.0 if canonical_target else None)
        ),
        "canonical_field_macro_accuracy": (
            sum(
                canonical_field_correct.get(field, 0) / canonical_field_total[field]
                for field in canonical_field_total
            )
            / len(canonical_field_total)
            if canonical_target and canonical_field_total
            else (0.0 if canonical_target else None)
        ),
        "canonical_full_match_rate": (
            canonical_full_matches / canonical_scored_examples
            if canonical_target and canonical_scored_examples > 0
            else (0.0 if canonical_target else None)
        ),
        "canonical_scored_examples": canonical_scored_examples if canonical_target else None,
        "canonical_field_accuracies": (
            {
                field: {
                    "accuracy": canonical_field_correct.get(field, 0) / canonical_field_total[field],
                    "correct": canonical_field_correct.get(field, 0),
                    "count": canonical_field_total[field],
                }
                for field in sorted(canonical_field_total)
            }
            if canonical_target
            else None
        ),
        "records": records,
    }
    print(metrics["last_tool_execution_result_accuracy"])
    return metrics


def evaluate_agent_replay_via_enterpriseops_gym(
    agent_generator: Any,
    world_model_generator: Any,
    tasks: list[TaskTrajectory],
    max_steps: int,
    internal_thinking_max_iterations: int,
    imagined_trajectory_max_steps: int,
    imagined_trajectory_rollouts: int,
    imagined_rollout_temperature: float,
    imagined_trajectory_selection_strategy: str,
    imagined_trajectory_observation_source: str,
    final_answer_f1_threshold: float,
    world_model_target: str,
    include_error_message_in_target: bool,
    include_stage_in_target: bool,
    include_world_model_history: bool,
    agent_max_observation_chars: int,
    agent_replay_history_budget_chars: int,
    gym_task_configs_dir: Path,
    gym_repo_path: Path | None,
    replay_modes: list[str] | tuple[str, ...] | None = None,
    imagined_trajectory_candidate_actions: int = 3,
    imagined_trajectory_top_k: int = 3,
    revision_lookahead_steps: int = 1,
    revision_imagined_rollouts: int = 1,
    revision_rollout_temperature: float = 0.7,
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE,
    system_prompt_max_chars: int = 0,
    action_max_chars: int = 0,
    react_wm_k_steps: int = 0,
    react_wm_kmax: int = 3,
    react_wm_foresight_temperature: float = 0.0,
    react_wm_foresight_observation_source: str = "world_model",
    k_controller: Any | None = None,
) -> dict[str, Any]:
    """Run agent replay against EnterpriseOps-Gym tasks via BenchmarkExecutor.

    For each held-out task with a `gym_task_config_name`, loads the matching
    gym task JSON, instantiates `BenchmarkExecutor` with our
    `WorldModelAssistedOrchestrator`, and runs the selected replay mode(s).
    Verifier scores from the gym replace our internal
    final-answer F1 threshold for completion.
    """
    if gym_repo_path is None:
        gym_repo_path = DEFAULT_ENTERPRISEOPS_GYM_REPO_PATH
    if gym_repo_path is not None:
        gym_path_str = str(gym_repo_path.resolve())
        if not gym_repo_path.exists():
            return {
                "skipped": True,
                "reason": f"gym_repo_path_missing:{gym_repo_path}",
                "evaluated_tasks": 0,
                "gym_repo_path": str(gym_repo_path),
            }
        if gym_path_str not in sys.path:
            sys.path.insert(0, gym_path_str)

    try:
        from benchmark.executor import BenchmarkExecutor
        from benchmark.models import BenchmarkConfig, LLMConfig
        from src.enterpriseops_gym.enterpriseops_gym_orchestrator import (
            build_world_model_assisted_orchestrator_class,
        )
    except ImportError as exc:
        return {
            "skipped": True,
            "reason": f"enterpriseops_gym_import_failed:{exc}",
            "evaluated_tasks": 0,
            "gym_repo_path": str(gym_repo_path) if gym_repo_path else None,
        }

    if not gym_task_configs_dir.exists():
        return {
            "skipped": True,
            "reason": f"gym_task_configs_dir_missing:{gym_task_configs_dir}",
            "evaluated_tasks": 0,
        }

    OrchestratorClass = build_world_model_assisted_orchestrator_class()

    matched_tasks: list[tuple[TaskTrajectory, Path]] = []
    unmatched_tasks: list[dict[str, Any]] = []
    for task in tasks:
        if not task.gym_task_config_name:
            unmatched_tasks.append(
                {"trajectory_index": task.trajectory_index, "reason": "no_gym_task_config_name"}
            )
            continue
        config_path = gym_task_configs_dir / task.gym_task_config_name
        if not config_path.exists():
            unmatched_tasks.append(
                {
                    "trajectory_index": task.trajectory_index,
                    "reason": "gym_task_config_not_found",
                    "expected_path": str(config_path),
                }
            )
            continue
        matched_tasks.append((task, config_path))

    if not matched_tasks:
        return {
            "skipped": True,
            "reason": "no_tasks_matched_gym_configs",
            "evaluated_tasks": 0,
            "unmatched_tasks": unmatched_tasks,
        }

    # The orchestrator never calls llm_client.invoke_with_tools, but BenchmarkExecutor
    # still constructs an LLMClient during initialize(). Provide a minimal-but-valid
    # config that initializes a LangChain client without making network calls.
    stub_llm_config = LLMConfig(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_api_key="not-used-by-world-model-assisted-orchestrator",
        temperature=0.0,
        max_tokens=1,
    )

    allowed_modes = (
        "baseline",
        "revision",
        "imagined",
        "react_wm",
        "react_wm_decide_k",
        "react_wm_rl_k",
    )
    selected_modes = tuple(replay_modes or allowed_modes)
    unknown_modes = [mode for mode in selected_modes if mode not in allowed_modes]
    if unknown_modes:
        return {
            "skipped": True,
            "reason": f"unknown_replay_modes:{','.join(unknown_modes)}",
            "evaluated_tasks": 0,
        }
    if any(mode != "baseline" for mode in selected_modes) and world_model_generator is None:
        return {
            "skipped": True,
            "reason": "world_model_required_for_non_baseline_replay",
            "evaluated_tasks": 0,
            "selected_replay_modes": list(selected_modes),
        }
    if "react_wm_rl_k" in selected_modes and k_controller is None:
        return {
            "skipped": True,
            "reason": "k_controller_required_for_react_wm_rl_k",
            "evaluated_tasks": 0,
            "selected_replay_modes": list(selected_modes),
        }

    mode_results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in selected_modes}
    mode_plan = [
        ("baseline", {}),
        (
            "revision",
            {"internal_thinking_max_iterations": internal_thinking_max_iterations},
        ),
        (
            "imagined",
            {"imagined_trajectory_max_steps": imagined_trajectory_max_steps},
        ),
        (
            "react_wm",
            {
                "react_wm_k_steps": react_wm_k_steps,
                "react_wm_kmax": react_wm_kmax,
                "react_wm_foresight_temperature": react_wm_foresight_temperature,
                "react_wm_foresight_observation_source": react_wm_foresight_observation_source,
            },
        ),
        (
            "react_wm_decide_k",
            {
                "react_wm_kmax": react_wm_kmax,
                "react_wm_foresight_temperature": react_wm_foresight_temperature,
                "react_wm_foresight_observation_source": react_wm_foresight_observation_source,
            },
        ),
        (
            "react_wm_rl_k",
            {
                "react_wm_kmax": react_wm_kmax,
                "react_wm_foresight_temperature": react_wm_foresight_temperature,
                "react_wm_foresight_observation_source": react_wm_foresight_observation_source,
                "k_controller": k_controller,
            },
        ),
    ]

    async def _run_one_task(task: TaskTrajectory, config_path: Path) -> None:
        with config_path.open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
        raw_config = {k: v for k, v in raw_config.items() if not k.startswith("_")}
        raw_config = override_enterpriseops_gym_mcp_urls(raw_config)
        bench_config = BenchmarkConfig(**raw_config)

        for mode, mode_overrides in mode_plan:
            if mode not in mode_results:
                continue
            orchestrator_kwargs = {
                "max_iterations": max_steps,
                "agent_generator": agent_generator,
                "world_model_generator": world_model_generator,
                "mode": mode,
                "world_model_target": world_model_target,
                "include_error_message_in_target": include_error_message_in_target,
                "include_stage_in_target": include_stage_in_target,
                "include_world_model_history": include_world_model_history,
                "internal_thinking_max_iterations": 0,
                "imagined_trajectory_max_steps": 0,
                "imagined_trajectory_rollouts": imagined_trajectory_rollouts if mode == "imagined" else 1,
                "imagined_rollout_temperature": imagined_rollout_temperature,
                "imagined_trajectory_selection_strategy": imagined_trajectory_selection_strategy,
                "imagined_trajectory_observation_source": imagined_trajectory_observation_source,
                "imagined_trajectory_candidate_actions": imagined_trajectory_candidate_actions,
                "imagined_trajectory_top_k": imagined_trajectory_top_k,
                "final_answer_f1_threshold": final_answer_f1_threshold,
                "agent_max_observation_chars": agent_max_observation_chars,
                "agent_replay_history_budget_chars": agent_replay_history_budget_chars,
                "revision_lookahead_steps": revision_lookahead_steps,
                "revision_imagined_rollouts": revision_imagined_rollouts,
                "revision_rollout_temperature": revision_rollout_temperature,
                "state_history_size": state_history_size,
                "system_prompt_max_chars": system_prompt_max_chars,
                "action_max_chars": action_max_chars,
                "initial_state": task.initial_state,
            }
            orchestrator_kwargs.update(mode_overrides)
            executor = BenchmarkExecutor(
                bench_config,
                llm_config=stub_llm_config,
                orchestrator_class=OrchestratorClass,
                orchestrator_kwargs=orchestrator_kwargs,
                config_path=str(config_path),
            )
            try:
                result = await executor.execute_benchmark()
            except Exception as exc:
                mode_results[mode].append(
                    {
                        "trajectory_index": task.trajectory_index,
                        "gym_task_config_name": task.gym_task_config_name,
                        "error": str(exc),
                    }
                )
                continue
            mode_results[mode].append(
                {
                    "trajectory_index": task.trajectory_index,
                    "gym_task_config_name": task.gym_task_config_name,
                    "result": result,
                }
            )

    async def _run_all() -> None:
        for task, config_path in tqdm(matched_tasks, desc="enterpriseops_gym_replay"):
            await _run_one_task(task, config_path)

    asyncio.run(_run_all())

    def _summarize_mode(records: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(records)
        errored = sum(1 for r in records if "error" in r)
        completed = 0
        verifier_scores: list[float] = []
        for record in records:
            result = record.get("result")
            if not result:
                continue
            statistics = result.get("statistics") or {}
            completed += int(statistics.get("successful_runs") or 0)
            verifier_level_pass_rate = statistics.get("verifier_level_pass_rate")
            if isinstance(verifier_level_pass_rate, (int, float)):
                verifier_scores.append(float(verifier_level_pass_rate))
                continue
            for run in result.get("runs", []) or []:
                verification_results = run.get("verification_results") or {}
                if isinstance(verification_results, dict):
                    run_passed = bool(verification_results) and all(
                        bool(v.get("passed", False))
                        for v in verification_results.values()
                        if isinstance(v, dict)
                    )
                    if not statistics:
                        completed += int(run_passed)
                verification_summary = run.get("verification_summary") or {}
                pass_rate = verification_summary.get("pass_rate")
                if isinstance(pass_rate, (int, float)):
                    verifier_scores.append(float(pass_rate))
        return {
            "evaluated_tasks": total,
            "errored_tasks": errored,
            "completed_runs": completed,
            "average_verifier_score": (
                sum(verifier_scores) / len(verifier_scores) if verifier_scores else None
            ),
            "task_records": records,
        }

    return {
        "gym_task_configs_dir": str(gym_task_configs_dir),
        "matched_tasks": len(matched_tasks),
        "unmatched_tasks": unmatched_tasks,
        "internal_thinking_max_iterations": internal_thinking_max_iterations,
        "imagined_trajectory_max_steps": imagined_trajectory_max_steps,
        "imagined_trajectory_rollouts": imagined_trajectory_rollouts,
        "imagined_rollout_temperature": imagined_rollout_temperature,
        "imagined_trajectory_selection_strategy": imagined_trajectory_selection_strategy,
        "imagined_trajectory_observation_source": imagined_trajectory_observation_source,
        "imagined_trajectory_candidate_actions": imagined_trajectory_candidate_actions,
        "imagined_trajectory_top_k": imagined_trajectory_top_k,
        "revision_lookahead_steps": revision_lookahead_steps,
        "revision_imagined_rollouts": revision_imagined_rollouts,
        "revision_rollout_temperature": revision_rollout_temperature,
        "react_wm_k_steps": react_wm_k_steps,
        "react_wm_kmax": react_wm_kmax,
        "react_wm_foresight_temperature": react_wm_foresight_temperature,
        "react_wm_foresight_observation_source": react_wm_foresight_observation_source,
        "world_model_target": world_model_target,
        "include_stage_in_target": include_stage_in_target,
        "include_world_model_history": include_world_model_history,
        "selected_replay_modes": list(selected_modes),
        **{mode: _summarize_mode(records) for mode, records in mode_results.items()},
    }


def _is_standard_azure_resource_endpoint(endpoint: str) -> bool:
    """Return True if `endpoint` is an endpoint the Azure SDK can route on its own.

    The Azure SDK auto-constructs `/openai/deployments/{deployment}/...` when
    given a resource endpoint like `https://<resource>.openai.azure.com`, and
    `AzureOpenAIChatGenerator` already routes `/openai/v1` endpoints to the
    OpenAI-compatible client. For everything else (e.g. vendor proxy gateways
    that already contain the deployment in the URL path), we need to switch
    to direct base-URL mode and set `AZURE_OPENAI_DEPLOYMENT_IN_URL=1`.
    """
    from urllib.parse import urlparse

    normalized = endpoint.rstrip("/")
    if normalized.endswith("/openai/v1"):
        return True
    parsed = urlparse(normalized)
    if not parsed.hostname:
        return False
    if not parsed.hostname.endswith(".openai.azure.com"):
        return False
    path = (parsed.path or "").strip("/")
    return path in {"", "openai"}


def _apply_llm_config_file(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Load an `--llm-config` JSON file and populate the matching env vars.

    Supported JSON schema (extra fields are ignored)::

        {
          "llm_provider":      "azureopenai" | "openai" | "gemini",
          "llm_model":         "<deployment-or-model-name>",
          "llm_api_key":       "<key>",
          "llm_api_endpoint":  "<endpoint URL>",
          "llm_api_version":   "<azure api version>",   # azureopenai only
          "temperature":       <float, optional, ignored>,
          "max_tokens":        <int, optional, ignored>
        }

    The loader maps:
    * Azure: ``llm_api_key`` → ``AZURE_OPENAI_API_KEY``,
             ``llm_api_endpoint`` → ``AZURE_OPENAI_ENDPOINT``,
             ``llm_api_version`` → ``AZURE_OPENAI_API_VERSION``,
             and auto-enables ``AZURE_OPENAI_DEPLOYMENT_IN_URL`` when the
             endpoint is not a standard ``*.openai.azure.com`` resource URL.
    * OpenAI: ``llm_api_key`` → ``OPENAI_API_KEY``,
              ``llm_api_endpoint`` → ``OPENAI_BASE_URL``.
    * Gemini: ``llm_api_key`` → ``GEMINI_API_KEY``,
              ``llm_api_endpoint`` → ``GEMINI_API_BASE_URL``.

    If ``--agent-model`` was not explicitly passed and the JSON provides both
    ``llm_provider`` and ``llm_model``, this function sets
    ``args.agent_model`` to ``"<provider>/<model>"`` (e.g.
    ``azureopenai/gpt-5.1``). Existing CLI values always win.

    Returns the parsed JSON dict so the caller can record/log it.
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"--llm-config: file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"--llm-config: failed to parse {config_path}: {exc}")
    if not isinstance(cfg, dict):
        raise SystemExit(f"--llm-config: expected a JSON object at {config_path}")

    provider = str(cfg.get("llm_provider") or "").strip().lower()
    model = str(cfg.get("llm_model") or "").strip()
    api_key = cfg.get("llm_api_key")
    endpoint = cfg.get("llm_api_endpoint")
    api_version = cfg.get("llm_api_version")

    def _set_if_unset(name: str, value: str) -> None:
        # Existing env values always win, so a developer-set env doesn't get
        # silently overwritten by a config file.
        if value is None:
            return
        value_str = str(value)
        if not value_str.strip():
            return
        if not os.environ.get(name):
            os.environ[name] = value_str

    if provider == "azureopenai":
        _set_if_unset("AZURE_OPENAI_API_KEY", api_key)
        _set_if_unset("AZURE_OPENAI_ENDPOINT", endpoint)
        _set_if_unset("AZURE_OPENAI_API_VERSION", api_version)
        if endpoint and not _is_standard_azure_resource_endpoint(str(endpoint)):
            _set_if_unset("AZURE_OPENAI_DEPLOYMENT_IN_URL", "1")
    elif provider == "openai":
        _set_if_unset("OPENAI_API_KEY", api_key)
        if endpoint:
            _set_if_unset("OPENAI_BASE_URL", endpoint)
    elif provider == "gemini":
        _set_if_unset("GEMINI_API_KEY", api_key)
        if endpoint:
            _set_if_unset("GEMINI_API_BASE_URL", endpoint)
    elif provider:
        raise SystemExit(
            f"--llm-config: unsupported llm_provider {provider!r}. "
            "Expected one of azureopenai | openai | gemini."
        )

    if (
        provider
        and model
        and getattr(args, "agent_model", None) in (None, "")
    ):
        if provider == "azureopenai":
            args.agent_model = f"azureopenai/{model}"
        elif provider == "openai":
            args.agent_model = model
        elif provider == "gemini":
            args.agent_model = f"gemini/{model}"

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an EnterpriseOps-Gym world model.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Fallback base model for the agent when --agent-model is not set.")
    parser.add_argument(
        "--world-model-path",
        default=None,
        help=(
            "Fine-tuned checkpoint or adapter path to evaluate. Required unless "
            "--world-model-method is provided."
        ),
    )
    parser.add_argument(
        "--world-model-method",
        default=None,
        help=(
            "Optional generator backend for the world model. Uses HF loading from "
            "--world-model-path by default; accepts the same hosted backends as --agent-model."
        ),
    )
    parser.add_argument(
        "--agent-model",
        default=None,
        help="Agent model used for EnterpriseOps-Gym replay. Defaults to --model.",
    )
    parser.add_argument(
        "--trajectory-dataset",
        choices=sorted(TRAJECTORY_DATASET_PRESETS.keys()),
        default="enterpriseops_gym",
    )
    parser.add_argument("--eval-data-path", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--canonical-eval-examples-path",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Path(s) to a materialized canonical-event examples JSONL (as produced by "
            "src/data_preparation/label_canonical_events_with_llm.py). Each line carries "
            "the model input plus a gold `canonical_event_state`/`canonical_event_with_nudge` "
            "label. When provided, next-state evaluation scores per-field categorical "
            "classification accuracy against these stored labels and agent replay is skipped "
            "unless --gym-task-configs is also set. Use with "
            "--world-model-target canonical_event_with_nudge (or canonical_event_state)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-history-size", type=int, default=DEFAULT_STATE_HISTORY_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--inference-device-map", default=None)
    parser.add_argument("--disable-chat-template", action="store_true")
    parser.add_argument(
        "--world-model-target",
        default=WORLD_MODEL_TARGET_STATE,
        choices=[
            WORLD_MODEL_TARGET_STATE,
            WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_TERNARY,
            WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
            WORLD_MODEL_TARGET_TOOL_OUTPUT,
            WORLD_MODEL_TARGET_CANONICAL_EVENT_STATE,
            WORLD_MODEL_TARGET_CANONICAL_EVENT_WITH_NUDGE,
            LEGACY_WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY,
        ],
    )
    parser.add_argument("--include-error-message-in-target", action="store_true")
    parser.add_argument("--include-stage-in-target", action="store_true")
    parser.add_argument("--include-world-model-history", action="store_true")
    parser.add_argument("--world-model-system-prompt-max-chars", type=int, default=0)
    parser.add_argument("--world-model-action-max-chars", type=int, default=0)
    parser.add_argument(
        "--vllm-server-port",
        type=int,
        default=None,
        help=(
            "Port for vLLM OpenAI-compatible chat completions when using vllm/... "
            "agent or world-model methods. Defaults to VLLM_SERVER_PORT or 9000."
        ),
    )
    parser.add_argument("--world-model-eval-samples", type=int, default=2000)
    parser.add_argument("--skip-next-state-eval", action="store_true")
    parser.add_argument("--skip-agent-replay", action="store_true")
    parser.add_argument(
        "--agent-replay-mode",
        choices=(
            "all",
            "all_with_itp",
            "itp_only",
            "baseline",
            "revision",
            "imagined",
            "react_wm",
            "react_wm_decide_k",
            "react_wm_rl_k",
        ),
        default="all",
        help=(
            "EnterpriseOps-Gym replay mode to run. `all` runs the three "
            "original ewm modes (baseline / revision / imagined). "
            "`all_with_itp` additionally runs the three `react_wm` family "
            "modes (react_wm / react_wm_decide_k / react_wm_rl_k). "
            "`itp_only` runs just the three `react_wm` family modes. "
            "Use `baseline` with --skip-next-state-eval to run agent replay "
            "without constructing a world model."
        ),
    )
    parser.add_argument(
        "--max-agent-tasks",
        type=int,
        default=0,
        help=(
            "Maximum EnterpriseOps-Gym replay tasks to run after applying the task split. "
            "Use 0 or a negative value for no cap. Defaults to all tasks in the split."
        ),
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=None,
        help=(
            "Optional path to a JSON file containing API credentials for the "
            "agent's LLM. Supported keys: llm_provider (azureopenai|openai|"
            "gemini), llm_model, llm_api_key, llm_api_endpoint, llm_api_version. "
            "Populates the matching env vars (e.g. AZURE_OPENAI_API_KEY) and "
            "sets --agent-model from llm_provider+llm_model when --agent-model "
            "is not passed explicitly. Existing env vars are NOT overwritten."
        ),
    )
    parser.add_argument("--agent-max-steps", type=int, default=15)
    parser.add_argument("--internal-thinking-max-iters", type=int, default=3)
    parser.add_argument("--imagined-trajectory-max-steps", type=int, default=3)
    parser.add_argument("--imagined-trajectory-rollouts", type=int, default=1)
    parser.add_argument("--imagined-rollout-temperature", type=float, default=0.7)
    parser.add_argument(
        "--imagined-trajectory-selection-strategy",
        choices=("first", "llm_judge", "topk_search"),
        default="llm_judge",
    )
    parser.add_argument("--imagined-trajectory-candidate-actions", type=int, default=3)
    parser.add_argument("--imagined-trajectory-top-k", type=int, default=3)
    parser.add_argument(
        "--imagined-trajectory-observation-source",
        choices=("world_model", "none"),
        default="world_model",
    )
    parser.add_argument("--revision-lookahead-steps", type=int, default=1)
    parser.add_argument("--revision-imagined-rollouts", type=int, default=1)
    parser.add_argument("--revision-rollout-temperature", type=float, default=0.7)
    parser.add_argument(
        "--react-wm-k",
        type=int,
        default=2,
        help=(
            "Fixed foresight depth K used by --agent-replay-mode react_wm. "
            "After turn 0, the world model rolls forward K steps and the "
            "imagined trajectory is injected as a [World-model foresight] "
            "user message before the next agent call."
        ),
    )
    parser.add_argument(
        "--react-wm-kmax",
        type=int,
        default=3,
        help=(
            "Maximum K used by --agent-replay-mode react_wm_decide_k / "
            "react_wm_rl_k. The chosen K is clipped to [0, kmax]."
        ),
    )
    parser.add_argument(
        "--react-wm-foresight-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the agent's imagined actions inside the foresight loop.",
    )
    parser.add_argument(
        "--react-wm-foresight-observation-source",
        choices=("world_model", "none"),
        default="world_model",
        help="Whether to use the world model for imagined observations in the foresight loop.",
    )
    parser.add_argument(
        "--k-controller-path",
        type=Path,
        default=None,
        help=(
            "Path to a trained K-controller checkpoint (a `policy_sft_khead` "
            "directory produced by `src.itp.training.train_adaptive_k sft`). "
            "Required for --agent-replay-mode react_wm_rl_k."
        ),
    )
    parser.add_argument("--k-controller-device", type=str, default="auto")
    parser.add_argument("--k-controller-dtype", type=str, default="auto")
    parser.add_argument(
        "--k-controller-max-seq-len",
        type=int,
        default=2048,
        help="Max sequence length the K-controller tokenizes the state prompt at.",
    )
    parser.add_argument(
        "--k-controller-do-sample",
        action="store_true",
        help="Sample K from the K-head softmax instead of taking the argmax.",
    )
    parser.add_argument(
        "--k-controller-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature when --k-controller-do-sample is set.",
    )
    parser.add_argument("--final-answer-f1-threshold", type=float, default=0.35)
    parser.add_argument("--agent-max-observation-chars", type=int, default=2000)
    parser.add_argument("--agent-replay-history-budget-chars", type=int, default=60000)
    parser.add_argument("--record-replay-trajectories", action="store_true")
    parser.add_argument(
        "--gym-task-configs",
        type=Path,
        default=None,
        help=(
            "Folder of EnterpriseOps-Gym task config JSONs. "
            f"Local generated configs usually live at {DEFAULT_ENTERPRISEOPS_GYM_TASK_CONFIGS_DIR}."
        ),
    )
    parser.add_argument(
        "--gym-repo-path",
        type=Path,
        default=DEFAULT_ENTERPRISEOPS_GYM_REPO_PATH,
        help=f"Path to the local EnterpriseOps-Gym repository. Defaults to {DEFAULT_ENTERPRISEOPS_GYM_REPO_PATH}.",
    )
    parser.add_argument(
        "--gym-task-split-manifest",
        type=Path,
        default=DEFAULT_ENTERPRISEOPS_GYM_TASK_SPLIT_MANIFEST,
    )
    parser.add_argument("--no-gym-task-split-manifest", action="store_true")
    args = parser.parse_args()
    args.llm_config_payload = None
    if args.llm_config is not None:
        args.llm_config_payload = _apply_llm_config_file(args.llm_config, args)
    if args.agent_replay_mode == "all":
        args.agent_replay_modes = ["baseline", "revision", "imagined"]
    elif args.agent_replay_mode == "all_with_itp":
        args.agent_replay_modes = [
            "baseline",
            "revision",
            "imagined",
            "react_wm",
            "react_wm_decide_k",
            "react_wm_rl_k",
        ]
    elif args.agent_replay_mode == "itp_only":
        args.agent_replay_modes = [
            "react_wm",
            "react_wm_decide_k",
            "react_wm_rl_k",
        ]
    else:
        args.agent_replay_modes = [args.agent_replay_mode]
    if "react_wm_rl_k" in args.agent_replay_modes and args.k_controller_path is None:
        parser.error(
            "--k-controller-path is required when --agent-replay-mode includes react_wm_rl_k."
        )
    args.requires_world_model = (
        not args.skip_next_state_eval
        or (
            not args.skip_agent_replay
            and any(mode != "baseline" for mode in args.agent_replay_modes)
        )
    )
    if args.requires_world_model and not args.world_model_path and not args.world_model_method:
        parser.error(
            "--world-model-path is required unless --world-model-method is provided "
            "when next-state evaluation or world-model-assisted replay is enabled"
        )
    args.world_model_target = canonicalize_world_model_target(args.world_model_target)
    if args.no_gym_task_split_manifest:
        args.gym_task_split_manifest = None
    _, preset_eval = TRAJECTORY_DATASET_PRESETS[args.trajectory_dataset]
    if args.eval_data_path is None:
        args.eval_data_path = list(preset_eval)
    elif isinstance(args.eval_data_path, Path):
        args.eval_data_path = [args.eval_data_path]
    return args


def _load_eval_trajectories(paths: list[Path]) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for path in paths:
        loaded = load_json(path)
        if not isinstance(loaded, list):
            raise SystemExit(f"Expected a list of trajectories in {path}")
        trajectories.extend(normalize_loaded_trajectories(loaded))
    return trajectories


def _load_canonical_eval_examples(
    paths: list[Path],
    target_mode: str,
) -> list[WorldModelStateExample]:
    """Load materialized canonical-event examples (JSONL) with their gold labels.

    Each record carries the model input (system prompt, task prompt, action, and
    history) plus a pre-computed gold label. We reconstruct a
    `WorldModelStateExample` per record and attach the label to `canonical_label`
    so `evaluate_world_model_predictions` scores against the same target the model
    was trained on rather than re-deriving one from `action`/`state`.
    """
    target_mode = canonicalize_world_model_target(target_mode)
    include_nudge = is_canonical_event_with_nudge_target(target_mode)
    examples: list[WorldModelStateExample] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Malformed JSON at {path}:{line_number + 1}: {exc}") from exc
                if not isinstance(record, dict):
                    raise SystemExit(f"Expected a JSON object at {path}:{line_number + 1}")
                if include_nudge:
                    label = record.get("canonical_event_with_nudge")
                    if not isinstance(label, dict):
                        event = record.get("canonical_event_state")
                        nudge = record.get("nudge")
                        if isinstance(event, dict) and isinstance(nudge, dict):
                            label = {"canonical_event_state": event, "nudge": nudge}
                else:
                    label = record.get("canonical_event_state")
                if not isinstance(label, dict):
                    raise SystemExit(
                        f"Missing gold label for target `{target_mode}` at {path}:{line_number + 1}"
                    )
                examples.append(
                    WorldModelStateExample(
                        trajectory_id=str(record.get("trajectory_id", len(examples))),
                        trajectory_index=int(record.get("trajectory_index", len(examples))),
                        interaction_index=int(record.get("interaction_index", 0)),
                        system_prompt=record.get("system_prompt", "") or "",
                        user_prompt=record.get("task_prompt", "") or "",
                        action=record.get("action"),
                        state_history=list(record.get("state_history") or []),
                        input_history=list(record.get("input_history") or []),
                        previous_state=record.get("previous_state"),
                        state={},
                        canonical_label=label,
                    )
                )
    return examples


def _build_world_model_generator(args: argparse.Namespace) -> Any:
    if args.world_model_method:
        return build_agent_generator(
            args.world_model_method,
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            disable_chat_template=args.disable_chat_template,
            attn_implementation=args.attn_implementation,
            device_map=args.inference_device_map,
            vllm_server_port=args.vllm_server_port,
        )
    if not args.world_model_path:
        raise ValueError("world_model_path is required when world_model_method is not provided")
    return HFTextGenerator(
        args.world_model_path,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        disable_chat_template=args.disable_chat_template,
        attn_implementation=args.attn_implementation,
        device_map=args.inference_device_map,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.canonical_eval_examples_path:
        eval_trajectories = []
        eval_examples = _load_canonical_eval_examples(
            args.canonical_eval_examples_path,
            args.world_model_target,
        )
        replay_tasks = []
        eval_data_paths = args.canonical_eval_examples_path
    else:
        eval_trajectories = _load_eval_trajectories(args.eval_data_path)
        eval_examples = extract_state_examples(
            eval_trajectories,
            state_history_size=args.state_history_size,
        )
        replay_tasks = extract_replay_tasks_from_state_trajectories(eval_trajectories)
        eval_data_paths = args.eval_data_path

    world_model_generator = _build_world_model_generator(args) if args.requires_world_model else None
    redacted_llm_config: dict[str, Any] | None = None
    if args.llm_config_payload is not None:
        redacted_llm_config = {
            k: ("<redacted>" if "key" in k.lower() else v)
            for k, v in args.llm_config_payload.items()
        }
        redacted_llm_config["__path__"] = str(args.llm_config)
    metrics: dict[str, Any] = {
        "eval_data_paths": [str(path) for path in eval_data_paths],
        "eval_trajectory_count": len(eval_trajectories),
        "eval_examples": len(eval_examples),
        "replay_tasks": len(replay_tasks),
        "agent_replay_mode": args.agent_replay_mode,
        "agent_replay_modes": args.agent_replay_modes,
        "agent_model": args.agent_model,
        "llm_config": redacted_llm_config,
        "requires_world_model": args.requires_world_model,
        "world_model_path": args.world_model_path,
        "world_model_method": args.world_model_method,
        "world_model_target": args.world_model_target,
    }

    if not args.skip_next_state_eval:
        metrics["world_model_state_eval"] = evaluate_world_model_predictions(
            world_model_generator,
            eval_examples,
            sample_limit=args.world_model_eval_samples,
            target_mode=args.world_model_target,
            include_error_message=args.include_error_message_in_target,
            include_stage=args.include_stage_in_target,
            include_input_history=args.include_world_model_history,
        )

    if not args.skip_agent_replay:
        if args.gym_task_configs is None:
            metrics["agent_replay_eval"] = {
                "skipped": True,
                "reason": "gym_task_configs_not_provided",
                "evaluated_tasks": 0,
            }
        else:
            candidate_tasks = list(replay_tasks)
            if args.gym_task_split_manifest is not None:
                selected_tasks, split_filter = filter_replay_tasks_by_gym_task_split(
                    candidate_tasks,
                    args.gym_task_split_manifest,
                )
            else:
                selected_tasks = candidate_tasks
                split_filter = {"enabled": False}
            selected_tasks_before_cap = len(selected_tasks)
            if args.max_agent_tasks > 0:
                selected_tasks = selected_tasks[: args.max_agent_tasks]
            agent_generator = build_agent_generator(
                args.agent_model or args.model,
                max_new_tokens=args.max_new_tokens,
                trust_remote_code=args.trust_remote_code,
                dtype=args.dtype,
                disable_chat_template=args.disable_chat_template,
                attn_implementation=args.attn_implementation,
                device_map=args.inference_device_map,
                vllm_server_port=args.vllm_server_port,
            )
            k_controller = None
            if "react_wm_rl_k" in args.agent_replay_modes:
                from src.itp.k_controller import KController

                k_controller = KController(
                    model_path=str(args.k_controller_path),
                    kmax=int(args.react_wm_kmax),
                    device_str=args.k_controller_device,
                    dtype_str=args.k_controller_dtype,
                    do_sample=args.k_controller_do_sample,
                    temperature=args.k_controller_temperature,
                    max_seq_len=args.k_controller_max_seq_len,
                )
            replay_eval = evaluate_agent_replay_via_enterpriseops_gym(
                agent_generator=agent_generator,
                world_model_generator=world_model_generator,
                tasks=selected_tasks,
                max_steps=args.agent_max_steps,
                internal_thinking_max_iterations=args.internal_thinking_max_iters,
                imagined_trajectory_max_steps=args.imagined_trajectory_max_steps,
                imagined_trajectory_rollouts=args.imagined_trajectory_rollouts,
                imagined_rollout_temperature=args.imagined_rollout_temperature,
                imagined_trajectory_selection_strategy=args.imagined_trajectory_selection_strategy,
                imagined_trajectory_observation_source=args.imagined_trajectory_observation_source,
                final_answer_f1_threshold=args.final_answer_f1_threshold,
                world_model_target=args.world_model_target,
                include_error_message_in_target=args.include_error_message_in_target,
                include_stage_in_target=args.include_stage_in_target,
                include_world_model_history=args.include_world_model_history,
                agent_max_observation_chars=args.agent_max_observation_chars,
                agent_replay_history_budget_chars=args.agent_replay_history_budget_chars,
                gym_task_configs_dir=args.gym_task_configs,
                gym_repo_path=args.gym_repo_path,
                replay_modes=args.agent_replay_modes,
                imagined_trajectory_candidate_actions=args.imagined_trajectory_candidate_actions,
                imagined_trajectory_top_k=args.imagined_trajectory_top_k,
                revision_lookahead_steps=args.revision_lookahead_steps,
                revision_imagined_rollouts=args.revision_imagined_rollouts,
                revision_rollout_temperature=args.revision_rollout_temperature,
                state_history_size=args.state_history_size,
                system_prompt_max_chars=args.world_model_system_prompt_max_chars,
                action_max_chars=args.world_model_action_max_chars,
                react_wm_k_steps=args.react_wm_k,
                react_wm_kmax=args.react_wm_kmax,
                react_wm_foresight_temperature=args.react_wm_foresight_temperature,
                react_wm_foresight_observation_source=args.react_wm_foresight_observation_source,
                k_controller=k_controller,
            )
            replay_eval["gym_task_split_filter"] = split_filter
            replay_eval["selected_tasks_before_max_agent_task_cap"] = selected_tasks_before_cap
            replay_eval["max_agent_tasks"] = args.max_agent_tasks
            replay_eval["selected_tasks_after_split_filter"] = len(selected_tasks)
            replay_metrics = dump_agent_replay_metrics(args.output_dir, replay_eval)
            replay_eval["replay_metrics"] = replay_metrics
            metrics["agent_replay_eval"] = replay_eval
            metrics["agent_replay_metrics"] = replay_metrics
            dump_agent_replay_strategy_records(args.output_dir, replay_eval)

    dump_json(args.output_dir / "evaluation_metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
