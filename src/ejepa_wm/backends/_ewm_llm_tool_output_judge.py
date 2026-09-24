"""Tool-output LLM world model with same-agent LLM judge scoring for beam_plan."""
from __future__ import annotations

import json
import logging
import math
from typing import Any

from ejepa_wm.backends import _ewm_runtime as ewm
from ejepa_wm.backends._ewm_finetuning import (
    WORLD_MODEL_INPUT_HISTORY_SIZE,
    normalize_world_model_input_history_text,
    parse_jsonish,
    strip_code_fence,
    strip_model_thinking_output,
    tool_output_looks_like_failure,
)
from ejepa_wm.backends._ewm_qwen_agentworld import (
    render_qwen_agentworld_system_prompt,
    resolve_qwen_agentworld_domain,
)

logger = logging.getLogger(__name__)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


def _tool_name_and_arguments(action: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(action, dict):
        return "", {}
    calls = action.get("tool_calls")
    if isinstance(calls, list) and calls:
        call = calls[0] if isinstance(calls[0], dict) else {}
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        name = str(function.get("name") or call.get("name") or "")
        arguments = function.get("arguments", call.get("arguments", call.get("args", {})))
    else:
        name = str(action.get("name") or action.get("tool") or action.get("tool_name") or "")
        arguments = action.get("arguments", action.get("args", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    return name, arguments if isinstance(arguments, dict) else {}


def _qwen_agentworld_user_prompt(
    domain: str,
    *,
    user_prompt: str,
    history_text: str,
    action: Any,
) -> str:
    name, arguments = _tool_name_and_arguments(action)
    if domain == "terminal" and name in {"run_shell", "execute_bash", "execute_command"}:
        command = str(arguments.get("command", arguments.get("cmd", "")))
        duration = min(60.0, float(arguments.get("timeout", arguments.get("duration", 1.0))))
        terminal_action = [
            {
                "keystrokes": command + ("" if command.endswith("\n") else "\n"),
                "duration": duration,
            }
        ]
        action_text = _safe_json(terminal_action)
        action_heading = "Current terminal action"
    else:
        action_text = _safe_json(action)
        action_heading = "Current tool call"
    return (
        f"Task:\n{user_prompt}\n\nHistorical context:\n{history_text}\n\n"
        f"{action_heading}:\n{action_text}\n\n"
        "Predict the exact next environment observation only."
    )


def _json_response(text: str) -> dict[str, Any]:
    cleaned = strip_model_thinking_output(strip_code_fence(text or ""))
    parsed = parse_jsonish(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("judge response was not a JSON object")
    return parsed


def _normalize_step_judgment(raw: dict[str, Any], *, predicted_tool_output: str) -> dict[str, Any]:
    heuristic_failure = tool_output_looks_like_failure(predicted_tool_output)
    failure_score = _clamp01(
        raw.get("failure_score", raw.get("failure_probability")),
        1.0 if heuristic_failure else 0.0,
    )
    progress_score = _clamp01(raw.get("progress_score", raw.get("task_progress_score")), 0.0)
    finish_score = _clamp01(raw.get("finish_score", raw.get("terminal_score")), 0.0)
    can_proceed = bool(raw.get("can_proceed", failure_score < 0.5))
    enough_to_finish = bool(raw.get("enough_to_finish", finish_score >= 0.75))
    step_score_raw = raw.get("step_score", raw.get("score"))
    try:
        step_score = max(-1.0, min(1.0, float(step_score_raw)))
    except (TypeError, ValueError):
        step_score = progress_score + 0.5 * finish_score - failure_score
        step_score = max(-1.0, min(1.0, step_score))
    return {
        "can_proceed": can_proceed,
        "step_score": step_score,
        "failure": bool(raw.get("failure", failure_score >= 0.5)),
        "failure_score": failure_score,
        "progress_score": progress_score,
        "enough_to_finish": enough_to_finish,
        "finish_score": finish_score,
        "reason": str(raw.get("reason", raw.get("comments", "")) or ""),
    }


def _fallback_step_judgment(predicted_tool_output: str, error: str = "") -> dict[str, Any]:
    failed = tool_output_looks_like_failure(predicted_tool_output)
    return {
        "can_proceed": not failed,
        "step_score": -1.0 if failed else 0.1,
        "failure": failed,
        "failure_score": 1.0 if failed else 0.0,
        "progress_score": 0.0 if failed else 0.3,
        "enough_to_finish": False,
        "finish_score": 0.0,
        "reason": error or ("heuristic failure from predicted tool output" if failed else "heuristic fallback"),
    }


def _state_from_judgment(judgment: dict[str, Any]) -> dict[str, Any]:
    failure = _clamp01(judgment.get("failure_score"))
    progress = _clamp01(judgment.get("progress_score"))
    finish = _clamp01(judgment.get("finish_score"))
    if failure >= 0.5 or judgment.get("failure"):
        execution_status = "failure"
    elif finish >= 0.75 or judgment.get("enough_to_finish"):
        execution_status = "success"
    elif progress >= 0.3:
        execution_status = "partial"
    else:
        execution_status = "no_op"
    if failure >= 0.5:
        progress_signal = "negative"
    elif progress >= 0.6:
        progress_signal = "positive"
    else:
        progress_signal = "neutral"
    return {
        "execution_status": execution_status,
        "progress_signal": progress_signal,
        "information_sufficiency": "sufficient" if finish >= 0.75 else "insufficient",
        "error_signature": "runtime_error" if failure >= 0.5 else "none",
        "side_effect_type": "unknown",
        "terminal": "finished" if finish >= 0.75 else "not_finished",
    }


def _field_probs_from_judgment(judgment: dict[str, Any]) -> dict[str, dict[str, float]]:
    failure = _clamp01(judgment.get("failure_score"))
    progress = _clamp01(judgment.get("progress_score"))
    finish = _clamp01(judgment.get("finish_score"))
    success = max(0.0, min(1.0 - failure, finish))
    partial = max(0.0, min(1.0 - failure - success, progress * (1.0 - failure)))
    no_op = max(0.0, 1.0 - failure - success - partial)
    return {
        "execution_status": {
            "failure": failure,
            "success": success,
            "partial": partial,
            "no_op": no_op,
            "unknown": 0.0,
        },
        "progress_signal": {
            "positive": progress * (1.0 - failure),
            "neutral": max(0.0, 1.0 - progress) * (1.0 - failure),
            "negative": failure,
            "unknown": 0.0,
        },
        "information_sufficiency": {
            "sufficient": finish,
            "insufficient": max(0.0, 1.0 - finish),
            "unknown": 0.0,
        },
        "error_signature": {
            "runtime_error": failure,
            "none": max(0.0, 1.0 - failure),
        },
        "side_effect_type": {"unknown": 1.0},
        "terminal": {"finished": finish, "not_finished": max(0.0, 1.0 - finish)},
    }


def _normalized_scores(records: list[dict[str, Any]]) -> None:
    live = [record for record in records if not record.get("vetoed")]
    if not live:
        for record in records:
            record["normalized_score"] = 0.0
        return
    highest = max(float(record.get("score", 0.0)) for record in live)
    exps = {id(record): math.exp(float(record.get("score", 0.0)) - highest) for record in live}
    denom = sum(exps.values()) or 1.0
    for record in records:
        record["normalized_score"] = exps[id(record)] / denom if not record.get("vetoed") else 0.0


class LlmToolOutputJudgeGenerator:
    """Beam-plan scorer for a general LLM world model that predicts tool-output text.

    The world model generates imagined tool outputs. The judge is intentionally the same
    generator as the policy agent, so critic firing and beam trajectory selection use the
    same model family as the acting agent.
    """

    canonical_event_available = True

    def __init__(
        self,
        world_model_generator: Any,
        judge_generator: Any,
        *,
        max_workers: int = 8,
        max_tool_output_chars: int = 2000,
        qwen_agentworld: bool = False,
        benchmark_name: str = "",
        domain_override: str = "",
        world_model_temperature: float = 0.0,
    ) -> None:
        self.world_model_generator = world_model_generator
        self.judge_generator = judge_generator
        self.max_workers = max(1, int(max_workers))
        self.max_tool_output_chars = max(200, int(max_tool_output_chars))
        self.qwen_agentworld = bool(qwen_agentworld)
        self.benchmark_name = benchmark_name
        self.domain_override = domain_override
        self.world_model_temperature = float(world_model_temperature)

    def _tool_output_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        input_history: list[dict[str, Any]],
        action: Any,
    ) -> list[dict[str, str]]:
        history_text = normalize_world_model_input_history_text(
            list(input_history or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:]
        )
        if self.qwen_agentworld:
            tool_name, _ = _tool_name_and_arguments(action)
            domain = resolve_qwen_agentworld_domain(
                self.benchmark_name,
                override=self.domain_override,
                first_tool_name=tool_name,
            )
            return [
                {
                    "role": "system",
                    "content": render_qwen_agentworld_system_prompt(
                        domain,
                        tool_context=system_prompt,
                        benchmark_name=self.benchmark_name,
                    ),
                },
                {
                    "role": "user",
                    "content": _qwen_agentworld_user_prompt(
                        domain,
                        user_prompt=user_prompt,
                        history_text=history_text,
                        action=action,
                    ),
                },
            ]
        return [
            {
                "role": "system",
                "content": (
                    "You are a world model for enterprise tool execution. Given the task, "
                    "recent action/observation history, and proposed tool call, predict the "
                    "tool output that would be observed. Return only the predicted tool output "
                    "text or JSON payload, with no analysis."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"System prompt:\n{system_prompt}\n\n"
                    f"User task:\n{user_prompt}\n\n"
                    "Recent action/observation history:\n"
                    f"{history_text}\n\n"
                    f"Proposed action:\n{_safe_json(action)}\n\n"
                    "Predict the resulting tool output. /no_think"
                ),
            },
        ]

    def _step_judge_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        input_history: list[dict[str, Any]],
        action: Any,
        predicted_tool_output: str,
    ) -> list[dict[str, str]]:
        history_text = normalize_world_model_input_history_text(
            list(input_history or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:]
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are judging an imagined enterprise-agent step before real execution. "
                    "Use the predicted tool output to decide whether this step is safe/useful, "
                    "whether it indicates failure, whether it makes task progress, and whether "
                    "the task is now finished. Respond only with JSON containing: "
                    "can_proceed boolean, step_score float from -1.0 to 1.0, failure boolean, "
                    "failure_score float 0.0-1.0, progress_score float 0.0-1.0, "
                    "enough_to_finish boolean, finish_score float 0.0-1.0, reason string."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"System prompt:\n{system_prompt}\n\n"
                    f"User task:\n{user_prompt}\n\n"
                    f"Recent history:\n{history_text}\n\n"
                    f"Proposed action:\n{_safe_json(action)}\n\n"
                    "Predicted tool output:\n"
                    f"{predicted_tool_output[: self.max_tool_output_chars]}\n\n"
                    "Judge this imagined step. /no_think"
                ),
            },
        ]

    def _trajectory_judge_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        input_history: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        history_text = normalize_world_model_input_history_text(
            list(input_history or [])[-WORLD_MODEL_INPUT_HISTORY_SIZE:]
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are selecting the best imagined enterprise-agent trajectory before "
                    "real tool execution. Choose the trajectory most likely to complete the "
                    "user task correctly while avoiding failures and incoherent tool usage. "
                    "Respond only with JSON: {\"selected_index\": int, "
                    "\"scores\": [{\"index\": int, \"score\": float, \"reason\": string}], "
                    "\"comments\": string}. Scores should be 0.0-1.0."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"System prompt:\n{system_prompt}\n\n"
                    f"User task:\n{user_prompt}\n\n"
                    f"Recent history:\n{history_text}\n\n"
                    "Candidate imagined trajectories:\n"
                    f"{_safe_json(candidates)}\n\n"
                    "Select the best trajectory. /no_think"
                ),
            },
        ]

    def _generate_many(
        self,
        generator: Any,
        messages_batch: list[list[dict[str, str]]],
        *,
        temperature: float = 0.0,
    ) -> list[str]:
        return ewm.generate_many(
            generator,
            messages_batch,
            [temperature] * len(messages_batch),
            max_workers=self.max_workers,
        )

    def _judge_steps(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        active_payloads: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages = [
            self._step_judge_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                input_history=payload["history"],
                action=payload["action"],
                predicted_tool_output=payload["predicted_tool_output"],
            )
            for payload in active_payloads
        ]
        raw_judgments = self._generate_many(self.judge_generator, messages)
        out: list[dict[str, Any]] = []
        for raw, payload in zip(raw_judgments, active_payloads, strict=False):
            try:
                out.append(
                    _normalize_step_judgment(
                        _json_response(raw), predicted_tool_output=payload["predicted_tool_output"]
                    )
                )
            except Exception as exc:
                logger.warning("llm_tool_output_judge: step judge parse failed (%s); using fallback", exc)
                out.append(_fallback_step_judgment(payload["predicted_tool_output"], str(exc)))
        return out

    def _trajectory_scores(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        input_history: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        fallback_scores: list[float],
    ) -> tuple[dict[int, float], dict[int, str], dict[str, Any]]:
        if len(candidates) <= 1:
            return {0: fallback_scores[0] if fallback_scores else 0.0}, {0: "single candidate"}, {"fallback_used": True}
        try:
            raw = self.judge_generator.generate_from_messages(
                self._trajectory_judge_messages(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    input_history=input_history,
                    candidates=candidates,
                ),
                temperature=0.0,
            )
            parsed = _json_response(raw)
            scores: dict[int, float] = {}
            reasons: dict[int, str] = {}
            for item in parsed.get("scores") or []:
                if not isinstance(item, dict):
                    continue
                index = int(item.get("index"))
                if 0 <= index < len(candidates):
                    scores[index] = _clamp01(item.get("score"), fallback_scores[index])
                    reasons[index] = str(item.get("reason", "") or "")
            for index, fallback in enumerate(fallback_scores):
                scores.setdefault(index, fallback)
                reasons.setdefault(index, "judge omitted candidate; used step-score fallback")
            return scores, reasons, {"raw_response": parsed, "fallback_used": False}
        except Exception as exc:
            logger.warning("llm_tool_output_judge: trajectory judge failed (%s); using fallback", exc)
            return (
                {index: score for index, score in enumerate(fallback_scores)},
                {index: "trajectory judge fallback" for index in range(len(fallback_scores))},
                {"error": str(exc), "fallback_used": True},
            )

    def score_action_plans_canonical_event(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        input_history: list[dict[str, Any]],
        action_plans: list[list[Any]],
        goal_text_override: str | None = None,
        score_config: Any = None,
    ) -> list[dict[str, Any]]:
        if not action_plans:
            return []
        num_plans = len(action_plans)
        histories: list[list[dict[str, Any]]] = [list(input_history or []) for _ in range(num_plans)]
        per_step_field_probs: list[list[dict[str, dict[str, float]]]] = [[] for _ in range(num_plans)]
        per_step_states: list[list[dict[str, Any]]] = [[] for _ in range(num_plans)]
        per_step_scores: list[list[dict[str, Any]]] = [[] for _ in range(num_plans)]
        per_step_tool_outputs: list[list[str]] = [[] for _ in range(num_plans)]
        terminal_probs: list[list[float]] = [[] for _ in range(num_plans)]
        max_len = max(len(plan) for plan in action_plans)
        veto_failure_prob = float(
            getattr(score_config, "veto_failure_prob", 0.6) if score_config is not None else 0.6
        )

        for step in range(max_len):
            active = [i for i, plan in enumerate(action_plans) if step < len(plan)]
            if not active:
                break
            output_messages = [
                self._tool_output_messages(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    input_history=histories[i],
                    action=action_plans[i][step],
                )
                for i in active
            ]
            tool_outputs = self._generate_many(
                self.world_model_generator,
                output_messages,
                temperature=self.world_model_temperature,
            )
            if self.qwen_agentworld:
                tool_outputs = [
                    strip_model_thinking_output(output).strip() for output in tool_outputs
                ]
            payloads = [
                {
                    "plan_index": plan_index,
                    "history": histories[plan_index],
                    "action": action_plans[plan_index][step],
                    "predicted_tool_output": output,
                }
                for plan_index, output in zip(active, tool_outputs, strict=False)
            ]
            judgments = self._judge_steps(
                system_prompt=system_prompt, user_prompt=user_prompt, active_payloads=payloads
            )
            for payload, judgment in zip(payloads, judgments, strict=False):
                plan_index = payload["plan_index"]
                predicted_output = payload["predicted_tool_output"]
                state = _state_from_judgment(judgment)
                per_step_field_probs[plan_index].append(_field_probs_from_judgment(judgment))
                per_step_states[plan_index].append(state)
                per_step_scores[plan_index].append(
                    {
                        "step": step + 1,
                        "score": float(judgment["step_score"]),
                        "contributions": {
                            "llm_step_judge": float(judgment["step_score"]),
                            "progress_score": float(judgment["progress_score"]),
                            "failure_score": -float(judgment["failure_score"]),
                            "finish_score": float(judgment["finish_score"]),
                        },
                        "judge": judgment,
                    }
                )
                per_step_tool_outputs[plan_index].append(predicted_output)
                terminal_probs[plan_index].append(float(judgment["finish_score"]))
                histories[plan_index] = histories[plan_index] + [
                    {
                        "step": len(histories[plan_index]) + 1,
                        "action": action_plans[plan_index][step],
                        "observation": predicted_output,
                    }
                ]

        candidates: list[dict[str, Any]] = []
        fallback_scores: list[float] = []
        for index, plan in enumerate(action_plans):
            trajectory_sum = sum(float(step.get("score", 0.0)) for step in per_step_scores[index])
            fallback_scores.append(
                max(0.0, min(1.0, (trajectory_sum + max_len) / max(1, 2 * max_len)))
            )
            candidates.append(
                {
                    "index": index,
                    "plan": plan,
                    "predicted_tool_outputs": per_step_tool_outputs[index],
                    "predicted_states": per_step_states[index],
                    "step_judgments": [step.get("judge") for step in per_step_scores[index]],
                    "fallback_score": fallback_scores[-1],
                }
            )
        judge_scores, judge_reasons, judge_detail = self._trajectory_scores(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_history=input_history,
            candidates=candidates,
            fallback_scores=fallback_scores,
        )

        records: list[dict[str, Any]] = []
        for index, plan in enumerate(action_plans):
            failure_scores = [
                float((step.get("judge") or {}).get("failure_score", 0.0))
                for step in per_step_scores[index]
            ]
            vetoed = any(score >= veto_failure_prob for score in failure_scores)
            score = float(judge_scores.get(index, fallback_scores[index]))
            records.append(
                {
                    "index": index,
                    "plan_index": index,
                    "plan": plan,
                    "score": score,
                    "vetoed": vetoed,
                    "veto_reasons": [
                        {"step": step_index + 1, "reasons": [f"judge_failure_score={score_value:.2f}"]}
                        for step_index, score_value in enumerate(failure_scores)
                        if score_value >= veto_failure_prob
                    ],
                    "reason": judge_reasons.get(index, "llm tool-output judge"),
                    "per_step": per_step_scores[index],
                    "per_step_predicted_state": per_step_states[index],
                    "predicted_state": per_step_states[index][-1] if per_step_states[index] else {},
                    "per_step_field_probs": per_step_field_probs[index],
                    "per_step_terminal_prob": terminal_probs[index],
                    "terminal_probability": terminal_probs[index][-1] if terminal_probs[index] else None,
                    "per_step_predicted_tool_output": per_step_tool_outputs[index],
                    "predicted_tool_output": per_step_tool_outputs[index][-1] if per_step_tool_outputs[index] else "",
                    "llm_tool_output_judge": judge_detail,
                }
            )
        _normalized_scores(records)
        records.sort(key=lambda record: (record["vetoed"], -float(record.get("score", 0.0))))
        return records


__all__ = ["LlmToolOutputJudgeGenerator"]
