#!/usr/bin/env python3
"""GRPO/ECHO finetuning for EnterpriseOps-Gym.

ECHO combines an on-policy policy-gradient objective with an auxiliary
cross-entropy objective on environment observation tokens. The upstream
microsoft/echo-rl implementation does this by extending SkyRL: the rollout
generator emits action-token masks plus world-model masks, and the trainer adds
``world_model_coeff * CE(env_tokens)`` to the GRPO/PPO policy loss.

This local implementation keeps the same objective shape without depending on
SkyRL. It runs grouped multi-turn EnterpriseOps-Gym rollouts with the current
HF policy, computes GRPO advantages within each task group, and trains on:

    GRPO(action tokens) + echo_coeff * CE(environment observation tokens)

Only EnterpriseOps-Gym is supported.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.enterpriseops_gym.enterpriseops_gym_orchestrator import build_react_tool_descriptions_from_gym
from src.finetuning import (
    DEFAULT_STATE_HISTORY_SIZE,
    ENTERPRISEOPS_GYM_EVAL_DATA_PATH,
    ENTERPRISEOPS_GYM_TRAIN_DATA_PATH,
    apply_chat_template_or_fallback,
    build_lora_config,
    build_react_action_messages,
    build_react_system_prompt,
    common_prefix_length,
    default_lora_target_modules,
    dump_json,
    dump_jsonl,
    extract_replay_tasks_from_state_trajectories,
    load_json,
    maybe_disable_nemotron_fast_mamba_kernels,
    normalize_loaded_trajectories,
    normalize_tool_call,
    parse_agent_decision,
    parse_csv_arg,
    preview_text,
    resolve_gym_task_config_name,
    resolve_text_generation_model_class,
    resolve_torch_dtype,
    resolve_training_device_map,
    stringify_tool_output,
    to_openai_tool_calls,
    validate_disjoint_task_sets,
)


@dataclass
class GeneratedAction:
    prompt_messages: list[dict[str, Any]]
    prompt_ids: list[int]
    action_ids: list[int]
    action_logprobs: list[float]
    text: str
    parse_error: str = ""


@dataclass
class EchoTurn:
    prompt_ids: list[int]
    response_ids: list[int]
    action_mask: list[int]
    world_loss_mask: list[int]
    old_action_logprobs: list[float]
    action_text: str
    observation_text: str
    parse_error: str = ""


@dataclass
class EchoRollout:
    task_id: str
    rollout_id: int
    gym_task_config_name: str
    reward: float
    correct: bool
    stop_reason: str
    turns: list[EchoTurn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EchoTrainingRow:
    input_ids: list[int]
    attention_mask: list[int]
    action_mask: list[int]
    world_loss_mask: list[int]
    old_logprobs: list[float]
    advantage: float
    reward: float
    task_id: str
    rollout_id: int
    turn_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO/ECHO finetuning on EnterpriseOps-Gym.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Base Hugging Face causal-LM checkpoint.")
    parser.add_argument("--train-data-path", type=Path, default=ENTERPRISEOPS_GYM_TRAIN_DATA_PATH)
    parser.add_argument("--eval-data-path", type=Path, default=ENTERPRISEOPS_GYM_EVAL_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("data_echo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-source", choices=("online_gym", "offline_replay"), default="online_gym")
    parser.add_argument("--gym-task-configs", type=Path, default=None, help="EnterpriseOps-Gym task config directory. Required for --rollout-source online_gym.")
    parser.add_argument("--gym-repo-path", type=Path, default=Path.home() / "programs" / "tools" / "EnterpriseOps-Gym")
    parser.add_argument(
        "--csm-mcp-url-override",
        default=None,
        help=(
            "Optional replacement URL for EnterpriseOps-Gym CSM MCP configs. "
            "By default the script now preserves the task config URL instead of "
            "forcing localhost:8001 to localhost:8010."
        ),
    )
    parser.add_argument("--max-train-tasks", type=int, default=32)
    parser.add_argument("--num-rollouts-per-task", type=int, default=4, help="GRPO group size per task.")
    parser.add_argument("--num-rl-iterations", type=int, default=1)
    parser.add_argument("--max-agent-steps", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--observation-max-chars", type=int, default=6000)
    parser.add_argument(
        "--success-observation-mode",
        choices=("status", "summary", "full"),
        default="summary",
        help=(
            "How much successful tool output to include in ECHO observation CE. "
            "Failures always keep only `API Error: ...`."
        ),
    )
    parser.add_argument("--state-history-size", type=int, default=DEFAULT_STATE_HISTORY_SIZE)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs-per-iteration", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--echo-coeff", type=float, default=0.5, help="Weight on environment-observation CE.")
    parser.add_argument("--action-loss-coeff", type=float, default=1.0, help="Weight on GRPO action loss.")
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-zero-advantage-groups", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-fast-mamba-kernels", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa", choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--disable-chat-template", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-bias", default="none", choices=("none", "all", "lora_only"))
    parser.add_argument("--lora-target-modules", default="auto")
    parser.add_argument("--lora-modules-to-save", default="")
    parser.add_argument(
        "--validate-train-eval-disjoint",
        action="store_true",
        help=(
            "Fail if train/eval trajectory files overlap on task_index, "
            "gym_task_config_name, or exact first system/user prompt pair. "
            "Disabled by default because online ECHO trains only from train tasks."
        ),
    )
    parser.add_argument("--skip-training", action="store_true", help="Collect/materialize rollouts but do not optimize.")
    return parser.parse_args()


def maybe_override_csm_mcp_url(raw_config: dict[str, Any], override_url: str | None) -> dict[str, Any]:
    if not override_url:
        return raw_config
    gym_servers = raw_config.get("gym_servers_config")
    if isinstance(gym_servers, list):
        for server in gym_servers:
            if isinstance(server, dict) and server.get("mcp_server_name") == "sn-csm-server":
                server["mcp_server_url"] = override_url
    if raw_config.get("mcp_server_name") == "sn-csm-server":
        raw_config["mcp_server_url"] = override_url
    return raw_config


def load_enterpriseops_trajectories(path: Path) -> list[dict[str, Any]]:
    records = load_json(path)
    if not isinstance(records, list):
        raise SystemExit(f"Expected a list of EnterpriseOps-Gym trajectories in {path}")
    trajectories = normalize_loaded_trajectories(records)
    bad = []
    for index, trajectory in enumerate(trajectories[:25]):
        config_name = trajectory.get("gym_task_config_name") or resolve_gym_task_config_name(trajectory)
        source_path = str(trajectory.get("source_path", "")).lower()
        if not config_name and "enterpriseops_gym" not in source_path:
            bad.append(index)
    if bad:
        raise SystemExit(f"{path} does not look like EnterpriseOps-Gym data; missing gym metadata near rows {bad[:5]}")
    return trajectories


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - (2 * half)
    return text[:half] + f"\n[observation truncated: omitted {omitted} chars]\n" + text[-half:]


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def tool_result_success(tool_result: Any) -> bool:
    if not isinstance(tool_result, dict):
        return True
    raw_success = tool_result.get("success")
    raw_payload = tool_result.get("result", tool_result)
    raw_is_error = isinstance(raw_payload, dict) and raw_payload.get("isError") is True
    if raw_success is not None:
        return bool(raw_success) and not raw_is_error
    if tool_result.get("error"):
        return False
    return not raw_is_error


def extract_api_error_message(tool_result: Any) -> str:
    if not isinstance(tool_result, dict):
        return stringify_tool_output(tool_result).strip() or "unknown tool error"
    candidates = [
        tool_result.get("error"),
        tool_result.get("message"),
        tool_result.get("detail"),
    ]
    raw_payload = tool_result.get("result")
    if isinstance(raw_payload, dict):
        candidates.extend(
            [
                raw_payload.get("error"),
                raw_payload.get("message"),
                raw_payload.get("detail"),
                raw_payload.get("content"),
            ]
        )
    elif raw_payload is not None:
        candidates.append(raw_payload)
    for candidate in candidates:
        text = stringify_tool_output(candidate).strip() if candidate is not None else ""
        if text:
            return text
    return stringify_tool_output(tool_result).strip() or "unknown tool error"


def compact_success_payload(payload: Any, mode: str) -> str:
    if mode == "status":
        return "OK"
    if mode == "full":
        return stringify_tool_output(payload)
    if isinstance(payload, list):
        preview_items = payload[:3]
        item_keys = []
        for item in preview_items:
            if isinstance(item, dict):
                item_keys.append(sorted(str(key) for key in item.keys())[:12])
        return safe_json(
            {
                "status": "OK",
                "type": "list",
                "count": len(payload),
                "sample_keys": item_keys,
                "sample": preview_items,
            }
        )
    if isinstance(payload, dict):
        id_like = {
            key: value
            for key, value in payload.items()
            if isinstance(key, str)
            and (key.lower() == "id" or key.lower().endswith("_id") or key.lower().endswith("id"))
        }
        scalar_preview = {
            key: value
            for key, value in payload.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        return safe_json(
            {
                "status": "OK",
                "type": "object",
                "keys": sorted(str(key) for key in payload.keys())[:30],
                "ids": id_like,
                "scalars": dict(list(scalar_preview.items())[:20]),
            }
        )
    text = stringify_tool_output(payload).strip()
    return safe_json({"status": "OK", "value": preview_text(text, 500)})


def format_echo_observation(tool_name: str, tool_result: Any, args: argparse.Namespace) -> tuple[str, str]:
    raw_payload = tool_result.get("result", tool_result) if isinstance(tool_result, dict) else tool_result
    payload_text = stringify_tool_output(raw_payload)
    if not tool_result_success(tool_result):
        error_message = extract_api_error_message(tool_result)
        return payload_text, f"{tool_name}: API Error: {error_message}"
    compact_text = compact_success_payload(raw_payload, args.success_observation_mode)
    return payload_text, f"{tool_name}: {compact_text}"


def normalize_reward_from_result(result: dict[str, Any] | None) -> tuple[float, bool]:
    if not result:
        return 0.0, False
    statistics = result.get("statistics") or {}
    if isinstance(statistics.get("verifier_level_pass_rate"), (int, float)):
        reward = float(statistics["verifier_level_pass_rate"])
        return reward, reward >= 1.0
    if isinstance(statistics.get("successful_runs"), (int, float)):
        successful = float(statistics["successful_runs"])
        total = float(statistics.get("total_runs") or 1.0)
        reward = successful / max(total, 1.0)
        return reward, successful > 0
    run_rewards: list[float] = []
    for run in result.get("runs", []) or []:
        summary = run.get("verification_summary") or {}
        if isinstance(summary.get("pass_rate"), (int, float)):
            run_rewards.append(float(summary["pass_rate"]))
            continue
        verification_results = run.get("verification_results") or {}
        if isinstance(verification_results, dict) and verification_results:
            checks = [bool(v.get("passed")) for v in verification_results.values() if isinstance(v, dict)]
            if checks:
                run_rewards.append(sum(checks) / len(checks))
    if run_rewards:
        reward = sum(run_rewards) / len(run_rewards)
        return reward, reward >= 1.0
    return 0.0, False


class EchoPolicy:
    def __init__(self, model: Any, tokenizer: Any, args: argparse.Namespace) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.input_device = next(model.parameters()).device
        self.disable_generation_cache = "nemotron_h" in {
            getattr(getattr(model, "config", None), "model_type", None),
            getattr(getattr(getattr(model, "base_model", None), "config", None), "model_type", None),
        }

    def encode_messages(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
        text = apply_chat_template_or_fallback(
            self.tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
            disable_chat_template=self.args.disable_chat_template,
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def observation_token_ids(
        self,
        prompt_messages: list[dict[str, Any]],
        action_text: str,
        observation_text: str,
    ) -> list[int]:
        action_messages = prompt_messages + [{"role": "assistant", "content": action_text}]
        observation_messages = action_messages + [
            {"role": "user", "content": "Environment observation:\n" + observation_text}
        ]
        action_ids = self.encode_messages(action_messages, add_generation_prompt=False)
        full_ids = self.encode_messages(observation_messages, add_generation_prompt=False)
        prefix = common_prefix_length(action_ids, full_ids)
        return full_ids[prefix:]

    def generate_action(self, prompt_messages: list[dict[str, Any]], *, rollout_seed: int) -> GeneratedAction:
        import torch

        self.model.eval()
        prompt_ids = self.encode_messages(prompt_messages, add_generation_prompt=True)
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=self.input_device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.args.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if self.args.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False
        if self.disable_generation_cache:
            generation_kwargs["use_cache"] = False
        if self.args.temperature > 0:
            torch.manual_seed(int(rollout_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(rollout_seed))
        with torch.no_grad():
            output = self.model.generate(inputs, **generation_kwargs)
            sequence = output.sequences[0]
            action_ids = sequence[len(prompt_ids) :].detach().cpu().tolist()
            if output.scores:
                transition = self.model.compute_transition_scores(
                    output.sequences,
                    output.scores,
                    normalize_logits=True,
                )[0]
                action_logprobs = transition[: len(action_ids)].detach().cpu().float().tolist()
            else:
                action_logprobs = [0.0] * len(action_ids)
        text = self.tokenizer.decode(action_ids, skip_special_tokens=True).strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()
        return GeneratedAction(
            prompt_messages=prompt_messages,
            prompt_ids=prompt_ids,
            action_ids=action_ids,
            action_logprobs=action_logprobs,
            text=text,
        )


def build_echo_rollout_orchestrator_class(policy: EchoPolicy, rollout_sink: list[EchoRollout], args: argparse.Namespace):
    from orchestrators.base import AgentOrchestrator

    class EchoRolloutOrchestrator(AgentOrchestrator):
        def __init__(self, *inner_args: Any, rollout_task_id: str, rollout_id: int, gym_task_config_name: str, **kwargs: Any) -> None:
            super().__init__(*inner_args, **kwargs)
            self.rollout_task_id = rollout_task_id
            self.rollout_id = rollout_id
            self.gym_task_config_name = gym_task_config_name
            self._rollout = EchoRollout(
                task_id=rollout_task_id,
                rollout_id=rollout_id,
                gym_task_config_name=gym_task_config_name,
                reward=0.0,
                correct=False,
                stop_reason="running",
            )

        def get_result_metadata(self) -> dict[str, Any]:
            return {
                "echo_rollout_task_id": self.rollout_task_id,
                "echo_rollout_id": self.rollout_id,
                "echo_turn_count": len(self._rollout.turns),
                "echo_stop_reason": self._rollout.stop_reason,
            }

        async def execute(self) -> dict[str, Any]:
            react_system_prompt = build_react_system_prompt(
                build_react_tool_descriptions_from_gym(self.available_tools)
            )
            user_query = self.config.user_prompt or ""
            conversation: list[dict[str, Any]] = [
                {"role": "system", "content": self.config.system_prompt or ""},
                {"role": "user", "content": user_query},
            ]
            conversation_flow: list[dict[str, Any]] = [
                {"type": "system_message", "content": self.config.system_prompt or ""},
                {"type": "user_message", "content": user_query},
            ]
            tools_used: list[str] = []
            tool_results: list[dict[str, Any]] = []
            final_answer = ""
            stop_reason = "max_steps"

            for step_index in range(args.max_agent_steps):
                prompt_messages = build_react_action_messages(
                    conversation,
                    current_query=user_query,
                    system_prompt=react_system_prompt,
                )
                generated = policy.generate_action(
                    prompt_messages,
                    rollout_seed=(args.seed * 1_000_003) + (self.rollout_id * 10_007) + step_index,
                )
                try:
                    decision = parse_agent_decision(generated.text)
                except Exception as exc:
                    generated.parse_error = str(exc)
                    self._rollout.turns.append(
                        EchoTurn(
                            prompt_ids=generated.prompt_ids,
                            response_ids=generated.action_ids,
                            action_mask=[1] * len(generated.action_ids),
                            world_loss_mask=[0] * len(generated.action_ids),
                            old_action_logprobs=generated.action_logprobs,
                            action_text=generated.text,
                            observation_text="",
                            parse_error=str(exc),
                        )
                    )
                    final_answer = ""
                    stop_reason = "parse_error"
                    break

                if "final_answer" in decision:
                    final_answer = str(decision["final_answer"])
                    conversation.append({"role": "assistant", "content": final_answer})
                    conversation_flow.append({"type": "ai_message", "content": final_answer, "tool_calls": []})
                    self._rollout.turns.append(
                        EchoTurn(
                            prompt_ids=generated.prompt_ids,
                            response_ids=generated.action_ids,
                            action_mask=[1] * len(generated.action_ids),
                            world_loss_mask=[0] * len(generated.action_ids),
                            old_action_logprobs=generated.action_logprobs,
                            action_text=generated.text,
                            observation_text="",
                        )
                    )
                    stop_reason = "final_answer"
                    break

                planned_calls = [normalize_tool_call(call) for call in decision.get("tool_calls", [])]
                if not planned_calls:
                    self._rollout.turns.append(
                        EchoTurn(
                            prompt_ids=generated.prompt_ids,
                            response_ids=generated.action_ids,
                            action_mask=[1] * len(generated.action_ids),
                            world_loss_mask=[0] * len(generated.action_ids),
                            old_action_logprobs=generated.action_logprobs,
                            action_text=generated.text,
                            observation_text="",
                            parse_error="empty_tool_calls",
                        )
                    )
                    stop_reason = "empty_tool_calls"
                    break

                planned_calls = [
                    {**call, "id": str(call.get("id") or f"call_{step_index}_{idx}")}
                    for idx, call in enumerate(planned_calls)
                ]
                conversation.append(
                    {
                        "role": "assistant",
                        "content": generated.text,
                        "tool_calls": to_openai_tool_calls(planned_calls),
                    }
                )
                conversation_flow.append(
                    {
                        "type": "ai_message",
                        "content": generated.text,
                        "tool_calls": [
                            {"id": call["id"], "name": call["name"], "args": call.get("arguments", {})}
                            for call in planned_calls
                        ],
                    }
                )

                observations: list[str] = []
                for call in planned_calls:
                    tool_name = call["name"]
                    tool_args = call.get("arguments", {})
                    try:
                        exec_result = await self._execute_tool_call(tool_name, tool_args)
                    except Exception as exc:
                        exec_result = {"result": {"success": False, "error": str(exc), "result": {}}, "gym_server": None}
                    tool_result = exec_result.get("result", {})
                    target_gym = exec_result.get("gym_server")
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
                    tool_results.append(
                        {
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "result": tool_result,
                            "gym_server": target_gym,
                        }
                    )
                    payload_text, echo_observation_text = format_echo_observation(tool_name, tool_result, args)
                    observations.append(echo_observation_text)
                    conversation.append(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "tool_call_id": call["id"],
                            "content": payload_text,
                        }
                    )
                    conversation_flow.append(
                        {
                            "type": "tool_result",
                            "tool_name": tool_name,
                            "result": tool_result,
                            "gym_server": target_gym,
                        }
                    )

                observation_text = truncate_text("\n\n".join(observations), args.observation_max_chars)
                observation_ids = policy.observation_token_ids(prompt_messages, generated.text, observation_text)
                self._rollout.turns.append(
                    EchoTurn(
                        prompt_ids=generated.prompt_ids,
                        response_ids=generated.action_ids + observation_ids,
                        action_mask=[1] * len(generated.action_ids) + [0] * len(observation_ids),
                        world_loss_mask=[0] * len(generated.action_ids) + [1] * len(observation_ids),
                        old_action_logprobs=generated.action_logprobs + [0.0] * len(observation_ids),
                        action_text=generated.text,
                        observation_text=observation_text,
                    )
                )
            else:
                stop_reason = "max_steps"

            self._rollout.stop_reason = stop_reason
            self._rollout.metadata = {
                "tools_used": tools_used,
                "tool_result_count": len(tool_results),
                "final_answer_preview": preview_text(final_answer, 300),
            }
            rollout_sink.append(self._rollout)
            return {
                "final_response": final_answer,
                "conversation_flow": conversation_flow,
                "tools_used": tools_used,
                "tool_results": tool_results,
                "messages": conversation,
            }

    return EchoRolloutOrchestrator


async def collect_online_rollouts_async(
    args: argparse.Namespace,
    policy: EchoPolicy,
    tasks: list[Any],
    iteration: int,
) -> list[EchoRollout]:
    if args.gym_task_configs is None:
        raise SystemExit("--gym-task-configs is required for --rollout-source online_gym")
    if args.gym_repo_path is not None and args.gym_repo_path.exists():
        gym_path_str = str(args.gym_repo_path)
        if gym_path_str not in sys.path:
            sys.path.insert(0, gym_path_str)
    try:
        from benchmark.executor import BenchmarkExecutor
        from benchmark.models import BenchmarkConfig, LLMConfig
    except ImportError as exc:
        raise SystemExit(
            f"Unable to import EnterpriseOps-Gym benchmark package: {exc}. "
            "Pass --gym-repo-path pointing at the EnterpriseOps-Gym repo."
        ) from exc

    eligible_tasks = [task for task in tasks if task.gym_task_config_name]
    if not eligible_tasks:
        raise SystemExit("No EnterpriseOps-Gym tasks with gym_task_config_name were found.")
    rng = random.Random(args.seed + (iteration * 1_000_003))
    selected = list(eligible_tasks)
    rng.shuffle(selected)
    if args.max_train_tasks > 0:
        selected = selected[: min(args.max_train_tasks, len(selected))]
    selected_names = [str(task.gym_task_config_name) for task in selected]
    print(
        "[echo_selected_tasks] "
        + json.dumps(
            {
                "iteration": iteration,
                "eligible_tasks": len(eligible_tasks),
                "selected_tasks": len(selected),
                "max_train_tasks": args.max_train_tasks,
                "sample": selected_names[:10],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    stub_llm_config = LLMConfig(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_api_key="not-used-by-echo-rollout-orchestrator",
        temperature=0.0,
        max_tokens=1,
    )
    rollouts: list[EchoRollout] = []
    OrchestratorClass = build_echo_rollout_orchestrator_class(policy, rollouts, args)

    rollout_counter = 0
    for task in tqdm(selected, desc="echo_online_rollout_tasks"):
        config_path = args.gym_task_configs / task.gym_task_config_name
        if not config_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
        raw_config = {key: value for key, value in raw_config.items() if not key.startswith("_")}
        raw_config = maybe_override_csm_mcp_url(raw_config, args.csm_mcp_url_override)
        bench_config = BenchmarkConfig(**raw_config)
        task_group_rollouts: list[EchoRollout] = []
        for group_index in range(args.num_rollouts_per_task):
            before = len(rollouts)
            executor = BenchmarkExecutor(
                bench_config,
                llm_config=stub_llm_config,
                orchestrator_class=OrchestratorClass,
                orchestrator_kwargs={
                    "max_iterations": args.max_agent_steps,
                    "rollout_task_id": str(task.gym_task_config_name),
                    "rollout_id": rollout_counter,
                    "gym_task_config_name": str(task.gym_task_config_name),
                },
                config_path=str(config_path),
            )
            result = None
            try:
                result = await executor.execute_benchmark()
            except Exception as exc:
                if len(rollouts) == before:
                    rollouts.append(
                        EchoRollout(
                            task_id=str(task.gym_task_config_name),
                            rollout_id=rollout_counter,
                            gym_task_config_name=str(task.gym_task_config_name),
                            reward=0.0,
                            correct=False,
                            stop_reason="benchmark_error",
                            metadata={"error": str(exc)},
                        )
                    )
            if len(rollouts) == before:
                rollouts.append(
                    EchoRollout(
                        task_id=str(task.gym_task_config_name),
                        rollout_id=rollout_counter,
                        gym_task_config_name=str(task.gym_task_config_name),
                        reward=0.0,
                        correct=False,
                        stop_reason="missing_orchestrator_rollout",
                    )
                )
            reward, correct = normalize_reward_from_result(result)
            new_rollout = rollouts[-1]
            new_rollout.reward = reward
            new_rollout.correct = correct
            new_rollout.metadata["group_index"] = group_index
            task_group_rollouts.append(new_rollout)
            rollout_counter += 1
        if len(task_group_rollouts) != args.num_rollouts_per_task:
            continue
    return rollouts


def collect_online_rollouts(
    args: argparse.Namespace,
    policy: EchoPolicy,
    tasks: list[Any],
    iteration: int,
) -> list[EchoRollout]:
    return asyncio.run(collect_online_rollouts_async(args, policy, tasks, iteration))


def build_offline_replay_rollouts(args: argparse.Namespace, trajectories: list[dict[str, Any]], tokenizer: Any) -> list[EchoRollout]:
    """Debug fallback: materialize logged EnterpriseOps-Gym turns without GRPO sampling."""
    from src.finetuning import extract_state_examples, build_state_prediction_chat_messages

    examples = extract_state_examples(trajectories, state_history_size=args.state_history_size)
    rollouts: list[EchoRollout] = []
    for index, example in enumerate(examples):
        prompt_messages = build_state_prediction_chat_messages(example, target_mode="tool_output")
        prompt_text = apply_chat_template_or_fallback(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
            disable_chat_template=args.disable_chat_template,
        )
        action_text = safe_json(example.action)
        action_messages = prompt_messages + [{"role": "assistant", "content": action_text}]
        observation_text = truncate_text(example.tool_output or "", args.observation_max_chars)
        full_messages = action_messages + [{"role": "user", "content": "Environment observation:\n" + observation_text}]
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        action_full_ids = tokenizer.encode(
            apply_chat_template_or_fallback(tokenizer, action_messages, disable_chat_template=args.disable_chat_template),
            add_special_tokens=False,
        )
        full_ids = tokenizer.encode(
            apply_chat_template_or_fallback(tokenizer, full_messages, disable_chat_template=args.disable_chat_template),
            add_special_tokens=False,
        )
        prompt_len = common_prefix_length(prompt_ids, action_full_ids)
        action_ids = action_full_ids[prompt_len:]
        obs_start = common_prefix_length(action_full_ids, full_ids)
        obs_ids = full_ids[obs_start:]
        rollouts.append(
            EchoRollout(
                task_id=example.trajectory_id,
                rollout_id=index,
                gym_task_config_name="offline_replay",
                reward=0.0,
                correct=False,
                stop_reason="offline_replay",
                turns=[
                    EchoTurn(
                        prompt_ids=prompt_ids[:prompt_len],
                        response_ids=action_ids + obs_ids,
                        action_mask=[1] * len(action_ids) + [0] * len(obs_ids),
                        world_loss_mask=[0] * len(action_ids) + [1] * len(obs_ids),
                        old_action_logprobs=[0.0] * (len(action_ids) + len(obs_ids)),
                        action_text=action_text,
                        observation_text=observation_text,
                    )
                ],
            )
        )
    return rollouts


def compute_group_advantages(rollouts: list[EchoRollout], args: argparse.Namespace) -> dict[tuple[str, int], float]:
    by_task: dict[str, list[EchoRollout]] = {}
    for rollout in rollouts:
        by_task.setdefault(rollout.task_id, []).append(rollout)
    advantages: dict[tuple[str, int], float] = {}
    for task_id, group in by_task.items():
        rewards = [float(item.reward) for item in group]
        mean = sum(rewards) / max(len(rewards), 1)
        variance = sum((reward - mean) ** 2 for reward in rewards) / max(len(rewards), 1)
        std = math.sqrt(variance)
        if args.skip_zero_advantage_groups and std <= args.advantage_eps:
            continue
        for rollout in group:
            advantage = float(rollout.reward) - mean
            if args.normalize_advantages:
                advantage /= max(std, args.advantage_eps)
            advantages[(rollout.task_id, rollout.rollout_id)] = advantage
    return advantages


def build_training_rows(rollouts: list[EchoRollout], args: argparse.Namespace) -> tuple[list[EchoTrainingRow], dict[str, Any]]:
    advantages = compute_group_advantages(rollouts, args)
    rows: list[EchoTrainingRow] = []
    dropped_long = 0
    dropped_no_action = 0
    dropped_no_advantage = 0
    for rollout in rollouts:
        key = (rollout.task_id, rollout.rollout_id)
        if key not in advantages:
            dropped_no_advantage += len(rollout.turns)
            continue
        for turn_index, turn in enumerate(rollout.turns):
            input_ids = turn.prompt_ids + turn.response_ids
            if len(input_ids) > args.max_seq_length:
                dropped_long += 1
                continue
            if not any(turn.action_mask):
                dropped_no_action += 1
                continue
            rows.append(
                EchoTrainingRow(
                    input_ids=input_ids,
                    attention_mask=[1] * len(input_ids),
                    action_mask=[0] * len(turn.prompt_ids) + turn.action_mask,
                    world_loss_mask=[0] * len(turn.prompt_ids) + turn.world_loss_mask,
                    old_logprobs=[0.0] * len(turn.prompt_ids) + turn.old_action_logprobs,
                    advantage=advantages[key],
                    reward=float(rollout.reward),
                    task_id=rollout.task_id,
                    rollout_id=rollout.rollout_id,
                    turn_index=turn_index,
                )
            )
    row_advantages = [float(row.advantage) for row in rows]
    row_advantage_mean = sum(row_advantages) / max(len(row_advantages), 1)
    row_advantage_variance = (
        sum((value - row_advantage_mean) ** 2 for value in row_advantages)
        / max(len(row_advantages), 1)
    )
    stats = {
        "rollouts": len(rollouts),
        "groups": len({rollout.task_id for rollout in rollouts}),
        "rows": len(rows),
        "dropped_long": dropped_long,
        "dropped_no_action": dropped_no_action,
        "dropped_no_advantage": dropped_no_advantage,
        "mean_reward": sum(float(r.reward) for r in rollouts) / max(len(rollouts), 1),
        "correct_rollouts": sum(1 for r in rollouts if r.correct),
        "action_tokens": sum(sum(row.action_mask) for row in rows),
        "world_tokens": sum(sum(row.world_loss_mask) for row in rows),
        "advantage_mean": row_advantage_mean,
        "advantage_std": math.sqrt(row_advantage_variance),
        "advantage_min": min(row_advantages) if row_advantages else None,
        "advantage_max": max(row_advantages) if row_advantages else None,
    }
    return rows, stats


class EchoDataset:
    def __init__(self, rows: list[EchoTrainingRow]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return asdict(self.rows[index])


class EchoGRPOCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch: dict[str, list[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "action_mask": [],
            "world_loss_mask": [],
            "old_logprobs": [],
            "advantages": [],
            "rewards": [],
        }
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["action_mask"].append(feature["action_mask"] + [0] * padding)
            batch["world_loss_mask"].append(feature["world_loss_mask"] + [0] * padding)
            batch["old_logprobs"].append(feature["old_logprobs"] + [0.0] * padding)
            batch["advantages"].append(float(feature["advantage"]))
            batch["rewards"].append(float(feature["reward"]))
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "action_mask": torch.tensor(batch["action_mask"], dtype=torch.float),
            "world_loss_mask": torch.tensor(batch["world_loss_mask"], dtype=torch.float),
            "old_logprobs": torch.tensor(batch["old_logprobs"], dtype=torch.float),
            "advantages": torch.tensor(batch["advantages"], dtype=torch.float),
            "rewards": torch.tensor(batch["rewards"], dtype=torch.float),
        }


def build_echo_grpo_trainer_class(base_trainer_class: Any, torch_module: Any, args: argparse.Namespace) -> Any:
    class EchoGRPOTrainer(base_trainer_class):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **_: Any) -> Any:
            action_mask = inputs.pop("action_mask")
            world_loss_mask = inputs.pop("world_loss_mask")
            old_logprobs = inputs.pop("old_logprobs")
            advantages = inputs.pop("advantages")
            inputs.pop("rewards", None)
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs["input_ids"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_action_mask = action_mask[..., 1:].contiguous()
            shift_world_mask = world_loss_mask[..., 1:].contiguous()
            shift_old_logprobs = old_logprobs[..., 1:].contiguous()

            log_probs = torch_module.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

            token_advantages = advantages.view(-1, 1).expand_as(token_log_probs)
            ratio = torch_module.exp(token_log_probs - shift_old_logprobs)
            unclipped = ratio * token_advantages
            clipped = torch_module.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * token_advantages
            surrogate = torch_module.minimum(unclipped, clipped)
            action_den = shift_action_mask.sum().clamp_min(1.0)
            grpo_action_loss = -((surrogate * shift_action_mask).sum() / action_den)

            world_den = shift_world_mask.sum().clamp_min(1.0)
            world_loss_unscaled = -((token_log_probs * shift_world_mask).sum() / world_den)
            loss = (args.action_loss_coeff * grpo_action_loss) + (args.echo_coeff * world_loss_unscaled)

            with torch_module.no_grad():
                clipped_fraction = (
                    ((ratio - 1.0).abs() > args.clip_range).float() * shift_action_mask
                ).sum() / action_den
                mean_action_logprob = (token_log_probs * shift_action_mask).sum() / action_den
                advantage_std = (
                    advantages.float().std(unbiased=False)
                    if advantages.numel() > 1
                    else torch_module.zeros((), device=advantages.device)
                )
                metrics = {
                    "echo/total_loss": float(loss.detach().cpu()),
                    "echo/grpo_action_loss": float(grpo_action_loss.detach().cpu()),
                    "echo/observation_loss": float(world_loss_unscaled.detach().cpu()),
                    "echo/observation_loss_scaled": float((args.echo_coeff * world_loss_unscaled).detach().cpu()),
                    "echo/action_tokens": float(shift_action_mask.sum().detach().cpu()),
                    "echo/observation_tokens": float(shift_world_mask.sum().detach().cpu()),
                    "echo/advantage_mean": float(advantages.mean().detach().cpu()),
                    "echo/advantage_std": float(advantage_std.detach().cpu()),
                    "echo/advantage_min": float(advantages.min().detach().cpu()),
                    "echo/advantage_max": float(advantages.max().detach().cpu()),
                    "echo/policy_ratio_mean": float((ratio * shift_action_mask).sum().detach().cpu() / action_den.detach().cpu()),
                    "echo/policy_clip_fraction": float(clipped_fraction.detach().cpu()),
                    "echo/action_logprob_mean": float(mean_action_logprob.detach().cpu()),
                }
            self._latest_echo_metrics = metrics
            current_step = int(getattr(self.state, "global_step", 0))
            last_logged_step = getattr(self, "_last_echo_logged_step", None)
            should_log = (
                last_logged_step != current_step
                and (current_step == 0 or current_step % max(1, int(args.logging_steps)) == 0)
            )
            if should_log:
                self._last_echo_logged_step = current_step
                self.log(metrics)
            return (loss, outputs) if return_outputs else loss

    return EchoGRPOTrainer


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    model_cls = resolve_text_generation_model_class(
        args.model,
        trust_remote_code=args.trust_remote_code,
        causal_lm_class=AutoModelForCausalLM,
    )
    model = model_cls.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=resolve_torch_dtype(torch, args.dtype),
        device_map=resolve_training_device_map(torch),
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    runtime_kernel_overrides = maybe_disable_nemotron_fast_mamba_kernels(
        model,
        torch_module=torch,
        gradient_checkpointing=args.gradient_checkpointing,
        allow_fast_mamba_kernels=args.allow_fast_mamba_kernels,
    )
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora_config = build_lora_config(args)
    if lora_config is not None:
        from peft import get_peft_model

        model = get_peft_model(model, lora_config)
        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tokenizer, runtime_kernel_overrides


def train_one_iteration(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    rows: list[EchoTrainingRow],
    iteration_output_dir: Path,
) -> dict[str, Any]:
    import torch
    from transformers import Trainer, TrainingArguments

    if not rows:
        raise ValueError("No GRPO/ECHO training rows were produced from rollouts.")
    training_kwargs = dict(
        output_dir=str(iteration_output_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs_per_iteration,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        report_to=[],
        remove_unused_columns=False,
        warmup_steps=0,
        lr_scheduler_type="constant",
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=1.0,
        optim="adamw_torch",
    )
    if "gradient_checkpointing_kwargs" in inspect.signature(TrainingArguments.__init__).parameters:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    trainer = build_echo_grpo_trainer_class(Trainer, torch, args)(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=EchoDataset(rows),
        data_collator=EchoGRPOCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    return {
        "train_metrics": trainer.state.log_history[-1] if trainer.state.log_history else {},
        "latest_echo_metrics": getattr(trainer, "_latest_echo_metrics", {}),
    }


def serializable_rollout(rollout: EchoRollout) -> dict[str, Any]:
    payload = asdict(rollout)
    for turn in payload.get("turns", []):
        turn["prompt_ids"] = {"length": len(turn.get("prompt_ids") or [])}
        turn["response_ids"] = {"length": len(turn.get("response_ids") or [])}
        turn["old_action_logprobs"] = {"length": len(turn.get("old_action_logprobs") or [])}
    return payload


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_trajectories = load_enterpriseops_trajectories(args.train_data_path)
    eval_trajectories = load_enterpriseops_trajectories(args.eval_data_path)
    if args.validate_train_eval_disjoint:
        validate_disjoint_task_sets(train_trajectories, eval_trajectories, dataset_name="enterpriseops_gym")
    train_tasks = extract_replay_tasks_from_state_trajectories(train_trajectories)

    model, tokenizer, runtime_kernel_overrides = load_model_and_tokenizer(args)
    policy = EchoPolicy(model, tokenizer, args)

    run_metrics: dict[str, Any] = {
        "method": "echo_grpo_enterpriseops_gym",
        "rollout_source": args.rollout_source,
        "echo_coeff": args.echo_coeff,
        "action_loss_coeff": args.action_loss_coeff,
        "num_rollouts_per_task": args.num_rollouts_per_task,
        "max_train_tasks": args.max_train_tasks,
        "runtime_kernel_overrides": runtime_kernel_overrides,
        "iterations": [],
        "use_lora": args.use_lora,
        "lora_config": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "bias": args.lora_bias,
            "target_modules": (
                default_lora_target_modules(args.model)
                if args.lora_target_modules == "auto"
                else parse_csv_arg(args.lora_target_modules)
            ) if args.use_lora else None,
            "modules_to_save": parse_csv_arg(args.lora_modules_to_save) if args.use_lora else None,
        },
    }

    for iteration in range(args.num_rl_iterations):
        iteration_dir = args.output_dir / f"iteration_{iteration:04d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        if args.rollout_source == "online_gym":
            rollouts = collect_online_rollouts(args, policy, train_tasks, iteration)
        else:
            rollouts = build_offline_replay_rollouts(args, train_trajectories, tokenizer)
        rows, row_stats = build_training_rows(rollouts, args)
        dump_jsonl(iteration_dir / "echo_rollouts.jsonl", (serializable_rollout(r) for r in rollouts))
        dump_jsonl(iteration_dir / "echo_training_rows.jsonl", (asdict(row) for row in rows))
        dump_json(iteration_dir / "echo_row_stats.json", row_stats)
        print("[echo_row_stats] " + json.dumps({"iteration": iteration, **row_stats}, ensure_ascii=False), flush=True)
        iteration_metrics: dict[str, Any] = {
            "iteration": iteration,
            "rollout_count": len(rollouts),
            "row_stats": row_stats,
        }
        if args.skip_training:
            iteration_metrics["training_skipped"] = True
        elif args.rollout_source == "offline_replay":
            iteration_metrics["training_skipped"] = True
            iteration_metrics["reason"] = "offline_replay has no on-policy GRPO old logprobs/rewards; use online_gym for ECHO training"
        else:
            iteration_metrics.update(train_one_iteration(args, model, tokenizer, rows, iteration_dir))
            print(
                "[echo_train_latest] "
                + json.dumps(iteration_metrics.get("latest_echo_metrics", {}), ensure_ascii=False),
                flush=True,
            )
        run_metrics["iterations"].append(iteration_metrics)
        dump_json(args.output_dir / "echo_run_metrics.json", run_metrics)

    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    dump_json(args.output_dir / "echo_run_metrics.json", run_metrics)


if __name__ == "__main__":
    main()
