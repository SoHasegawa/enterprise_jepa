"""``ewm_imagined`` World-Model backend — generative EWM imagined-trajectory planning.

This is the unification of the former standalone ``imagined`` strategy into the
pluggable WM contract: it is a ``prompt_injection`` backend whose
:meth:`advise` runs an imagined rollout with a generative Enterprise World Model
and returns the ``[IMAGINED_TRAJECTORY_FOR_PLANNING_ONLY]`` block to inject.

Selected via ``WM_STRATEGY=prompt_injection WM_BACKEND=ewm_imagined`` (the factory
also maps the convenience alias ``WM_STRATEGY=imagined`` to this).

Two models are involved:

* the **agent** that proposes candidate actions — the injected ``chat_fn`` (the
  policy LLM; same one the executor commits actions with), wrapped to the
  ``generate_from_messages`` interface the rollout helpers expect; and
* the **world model** itself — a fine-tuned model served over a vLLM
  OpenAI-compatible endpoint (``WM_EWM_MODEL`` / ``WM_VLLM_BASE_URL``).

The evolving enterprise state is **reconstructed from the conversation flow**
(the last ``tool_result``), so this fits the generic ``advise(conversation_flow)``
contract with no extra state-tracking hook (option (a)).

Axes (env): ``WM_STATE`` (binary_error | binary_error_stage | tool_output | canonical_nudge),
``ACTION_OPTIMIZER`` (``topk_search`` | unset→plain), ``K_CONTROLLER``
(``react_wm_decide_k`` | ``react_wm_rl_k`` | unset→static depth
``WM_IMAGINED_MAX_STEPS``).
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import re
import time
from collections.abc import Callable
from typing import Any

from ejepa_wm.backends import _ewm_runtime as ewm
from ejepa_wm.backends._ewm_qwen_agentworld import is_qwen_agentworld_model
from ejepa_wm.base import AdviseResult, ChatFn, SelectResult, WMConfig, WorldModel, render_flow

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _jepa_arch_defaults() -> dict[str, Any]:
    """Architecture fallbacks for a JEPA checkpoint whose ``jepa_data_manifest.json``
    is absent (e.g. a canonical-event-head-only checkpoint). Read from
    ``WM_JEPA_ARCH_DEFAULTS`` (a JSON object of the ``finetuning_jepa`` arch flags:
    ``backbone_type``, ``pooling``, ``latent_type``, ``latent_dim``,
    ``memory_tokens``, ``goal_conditioning``, ``canonical_event_head_hidden_size``,
    ``max_input_length``, ...). A value present in the checkpoint manifest always wins.
    """
    raw = (os.getenv("WM_JEPA_ARCH_DEFAULTS") or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("ewm_imagined: WM_JEPA_ARCH_DEFAULTS is not valid JSON; ignoring")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_k_controller(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in {"", "none", "static", "off"}:
        return ""
    if v in {"react_wm_decide_k", "decide_k", "agent", "decide"}:
        return "react_wm_decide_k"
    if v in {"react_wm_rl_k", "rl_k", "rl", "controller"}:
        return "react_wm_rl_k"
    logger.warning("ewm_imagined: unknown K_CONTROLLER=%s; using static max steps", value)
    return ""


class _ChatFnAgent:
    """Adapt a ``ChatFn`` to the rollout generation interface.

    Plain callbacks are still just ``messages -> str``. If a benchmark callback also accepts
    ``temperature`` and ``num_samples``/``n`` keywords, ``sample_many`` can use it for one shared
    prompt with OpenAI/vLLM ``n=k`` sampling.
    """

    def __init__(self, chat_fn: ChatFn) -> None:
        self._chat_fn = chat_fn
        self._accepts_temperature = self._accepts_parameter("temperature")
        self._accepts_num_samples = self._accepts_parameter("num_samples")
        self._accepts_n = self._accepts_parameter("n")
        self._accepts_response_format = self._accepts_parameter("response_format")
        self.supports_parallel_requests = (
            os.getenv("WM_AGENT_PARALLEL_REQUESTS") or "1"
        ).strip().lower() not in {"0", "false", "no", "off"}

    def _accepts_parameter(self, name: str) -> bool:
        try:
            signature = inspect.signature(self._chat_fn)
        except (TypeError, ValueError):
            return False
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == name
            for parameter in signature.parameters.values()
        )

    def generate_from_messages(
        self, messages, temperature: float = 0.0, response_format: dict[str, Any] | None = None
    ) -> str:
        kwargs: dict[str, Any] = {}
        if self._accepts_temperature:
            kwargs["temperature"] = temperature
        if response_format is not None and self._accepts_response_format:
            kwargs["response_format"] = response_format
        if kwargs:
            return self._chat_fn(messages, **kwargs)
        return self._chat_fn(messages)

    def generate_samples(
        self,
        messages,
        temperature: float = 0.0,
        num_samples: int = 1,
        response_format: dict[str, Any] | None = None,
    ) -> list[str]:
        sampler = getattr(self._chat_fn, "generate_samples", None)
        if sampler is not None:
            if response_format is not None:
                try:
                    return [
                        str(text)
                        for text in sampler(
                            messages,
                            temperature=temperature,
                            num_samples=num_samples,
                            response_format=response_format,
                        )
                    ]
                except TypeError:
                    pass
            return [
                str(text)
                for text in sampler(messages, temperature=temperature, num_samples=num_samples)
            ]
        if not self._accepts_num_samples and not self._accepts_n:
            raise NotImplementedError
        kwargs: dict[str, Any] = {}
        if self._accepts_temperature:
            kwargs["temperature"] = temperature
        if self._accepts_num_samples:
            kwargs["num_samples"] = num_samples
        else:
            kwargs["n"] = num_samples
        if response_format is not None and self._accepts_response_format:
            kwargs["response_format"] = response_format
        result = self._chat_fn(messages, **kwargs)
        if isinstance(result, list):
            return [str(text) for text in result]
        return [str(result)]


# --- shared model weights ---------------------------------------------------
#
# The per-episode MPC state (``_beam_imagined_plan``, ``_beam_plan_cursor``,
# critic counters, ...) lives on the ``EwmImagined`` INSTANCE, so running two
# tasks concurrently against one instance corrupts both. The fix is one world
# model per task -- but a JEPA checkpoint is multi-GB, so constructing the
# generator per task would reload the weights every time (and hold N copies on
# the GPU under ``max_parallel=N``).
#
# These caches make "one world model per task" cheap: the *weights* are built
# once per process and shared, while each task gets its own episode state. The
# generators' own caches are content-addressed (keyed by text), so sharing them
# across tasks is correctness-neutral.
_SHARED_GENERATORS: dict[str, Any] = {}


def _generator_cache_key(spec: Any) -> str:
    """Stable string key for a generator's construction parameters."""
    try:
        return json.dumps(spec, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return repr(spec)


def weight_sharing_enabled() -> bool:
    """Whether generators may be shared process-wide (``WM_SHARE_MODEL_WEIGHTS``).

    Off by default: sharing is only correct when the caller builds one world model
    per task purely for episode isolation. A caller that reuses a single world model
    for everything gains nothing, and tests install their own fake generators and
    must never receive another test's instance.
    """
    return (os.getenv("WM_SHARE_MODEL_WEIGHTS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def shared_generator(spec: Any, factory: Callable[[], Any]) -> Any:
    """Return the process-wide generator for ``spec``, building it on first use.

    Falls back to a fresh instance unless :func:`weight_sharing_enabled`.
    """
    if not weight_sharing_enabled():
        return factory()
    key = _generator_cache_key(spec)
    generator = _SHARED_GENERATORS.get(key)
    if generator is None:
        generator = factory()
        _SHARED_GENERATORS[key] = generator
        logger.info("ejepa_wm: built shared world-model generator (%s)", key[:200])
    return generator


def clear_shared_generators() -> None:
    """Drop the cached generators (tests; frees GPU memory between checkpoints)."""
    _SHARED_GENERATORS.clear()


class EwmImaginedWorldModel(WorldModel):
    """prompt_injection backend: inject an EWM-optimized imagined trajectory."""

    name = "ewm_imagined"

    def __init__(self, config: WMConfig, chat_fn: ChatFn) -> None:
        super().__init__(config)
        if chat_fn is None:
            raise ValueError("backend='ewm_imagined' requires a chat_fn (the policy LLM).")
        self._agent = _ChatFnAgent(chat_fn)

        # ITP-I inference harness. These settings affect only strategy=itp_i; unlike the
        # upstream training recipe, they use the caller's existing, unmodified policy model.
        self.itp_max_k = max(0, _env_int("WM_ITP_MAX_K", 5))
        self.itp_fixed_k = _env_int("WM_ITP_FIXED_K", -1)
        self.itp_decision_temperature = max(0.0, _env_float("WM_ITP_DECISION_TEMPERATURE", 0.8))
        self.itp_world_model_temperature = max(
            0.0, _env_float("WM_ITP_WORLD_MODEL_TEMPERATURE", 0.7)
        )
        self.itp_foresight_tokens = max(1, _env_int("WM_ITP_FORESIGHT_TOKENS", 256))

        # The imagined rollout's ReAct system prompt (text protocol, tool catalog embedded)
        # is built per-call in :meth:`advise` from the ``tools`` event carried in the
        # ``conversation_flow`` (see ``_react_system_prompt_from_flow``) — keeping the WM
        # executor-agnostic under the uniform ``build_world_model(config, chat_fn)`` contract.

        self.wm_state = (os.getenv("WM_STATE") or "binary_error").strip().lower()
        if self.wm_state not in ewm.WM_STATES:
            logger.warning(
                "ewm_imagined: unknown WM_STATE=%s; using %s",
                self.wm_state,
                ewm.WM_STATE_BINARY_ERROR,
            )
            self.wm_state = ewm.WM_STATE_BINARY_ERROR
        self.action_optimizer = (os.getenv("ACTION_OPTIMIZER") or "").strip().lower()
        self.k_controller_mode = _normalize_k_controller(os.getenv("K_CONTROLLER"))
        self.max_steps = _env_int("WM_IMAGINED_MAX_STEPS", 3)
        self.candidate_actions = _env_int("WM_IMAGINED_CANDIDATE_ACTIONS", 3)
        self.top_k = _env_int("WM_IMAGINED_TOP_K", 3)
        self.temperature = _env_float("WM_IMAGINED_TEMPERATURE", 0.7)
        # Open-loop beam candidates normally use one n=k sampling request.  The ladder keeps
        # one near-greedy plan and spreads the remaining candidates across temperatures, which
        # reduces duplicate plans at the cost of k individual generations.
        self.sample_temperature_ladder = (
            os.getenv("WM_SAMPLE_TEMPERATURE_LADDER") or ""
        ).strip().lower() in {"1", "true", "yes"}
        self.sample_temperature_ladder_max = max(
            0.0, _env_float("WM_SAMPLE_TEMPERATURE_LADDER_MAX", 1.2)
        )
        self.state_history_size = _env_int("WM_STATE_HISTORY_SIZE", 3)
        # Multi-rollout + LLM-judge selection (decoder-friendly path for the seq2seq JEPA on
        # terminal tasks): generate WM_IMAGINED_ROLLOUTS decoded rollouts and let the agent LLM
        # judge pick the best (WM_IMAGINED_SELECTION=llm_judge). Default = single rollout / first.
        self.imagined_rollouts = max(1, _env_int("WM_IMAGINED_ROLLOUTS", 1))
        self.imagined_selection = (os.getenv("WM_IMAGINED_SELECTION") or "first").strip().lower()
        # open_loop: sample complete candidate plans up front from one diversity-menu prompt
        # (symbolic "$stepK.field" refs for values not yet known), then one batched world-model
        # scoring pass, instead of one agent call + one WM call per horizon/rollout step.
        # Shared knob: affects beam_plan_step (via _beam_plan_open_loop) and the plain closed-loop
        # rollout dispatched through ewm.optimize_imagined_trajectory. hier_latent_cem is
        # unaffected -- it is already structurally "1 LLM call + 1 batched WM pass".
        self.imagined_rollout_mode = (
            (os.getenv("WM_IMAGINED_ROLLOUT_MODE") or "closed_loop").strip().lower()
        )
        if self.imagined_rollout_mode not in {"closed_loop", "open_loop"}:
            logger.warning(
                "ewm_imagined: unknown WM_IMAGINED_ROLLOUT_MODE=%s; using closed_loop",
                self.imagined_rollout_mode,
            )
            self.imagined_rollout_mode = "closed_loop"
        # Concurrent-request cap for generate_many's thread-pool dispatch (open_loop's batched
        # world-model calls, text-WM backend only -- a no-op serial loop until a backend opts
        # into supports_parallel_requests; see generate_many's docstring).
        self.llm_batch_parallelism = max(1, _env_int("WM_LLM_BATCH_PARALLELISM", 8))
        # Tier-1 CLOSED-loop imagined-rollout speedups -- only matter when the multi-rollout
        # path is active (WM_IMAGINED_ROLLOUTS>1 or WM_IMAGINED_SELECTION=llm_judge); the
        # default single-rollout path always uses imagine_trajectory directly and reads
        # neither knob (matches the reference: both are wins for the *multi-rollout* path only).
        # imagined_parallel_rollouts: advance all rollouts in lockstep, batching each step's
        # agent/world-model calls across rollouts (imagine_trajectories_lockstep) instead of
        # running each rollout fully before starting the next.
        self.imagined_parallel_rollouts = (
            os.getenv("WM_IMAGINED_PARALLEL_ROLLOUTS") or "1"
        ).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        # imagined_single_call_step: collapse each lockstep step's think+act into ONE combined
        # generation (build_react_step_messages), halving the per-step agent call count again.
        self.imagined_single_call_step = (
            os.getenv("WM_IMAGINED_SINGLE_CALL_STEP") or "1"
        ).strip().lower() in {
            "1",
            "true",
            "yes",
        }

        # beam_plan (MPC) knobs — used only for WM_STRATEGY=beam_plan (see beam_plan_step).
        # Mirror the EnterpriseOps-Gym orchestrator's --latent-plan-* / --latent-mpc-* replay args.
        self.beam_samples = _env_int("WM_BEAM_PLAN_SAMPLES", 5)  # m: candidates per horizon step
        self.beam_horizon = _env_int("WM_BEAM_PLAN_HORIZON", 3)  # n: lookahead depth
        self.beam_execute_steps = max(
            1, _env_int("WM_BEAM_MPC_EXECUTE_STEPS", 3)
        )  # steps between re-plans
        # Defaults mirror the EWM CLI (finetuning_jepa.py --latent-plan-*), not the orchestrator
        # class defaults, so an unset ejepa beam_plan run matches `python -m src.finetuning_jepa`.
        self.beam_score_margin = max(0.0, _env_float("WM_BEAM_PLAN_SCORE_MARGIN", 1e-3))
        self.beam_diversity_multiplier = max(1, _env_int("WM_BEAM_PLAN_DIVERSITY_MULTIPLIER", 2))
        # SSoT-style diversity keeps the open-loop beam prompt identical across samples so vLLM
        # can share prefill, but asks each sampled continuation to internally generate a random
        # string and map it to one diversity option before emitting the JSON plan.
        self.beam_ssot_diversity = (
            os.getenv("WM_BEAM_PLAN_SSOT_DIVERSITY") or ""
        ).strip().lower() in {"1", "true", "yes"}
        # Optional iterative refinement: after scoring one open-loop candidate set, show the
        # generator the best prior trajectories and their WM scores, then ask it to improve them.
        # One round is exactly the historical behavior. More rounds imply open-loop planning,
        # because refinement operates on complete trajectories rather than one action at a time.
        self.beam_refinement_rounds = max(1, _env_int("WM_BEAM_PLAN_REFINEMENT_ROUNDS", 1))
        self.beam_refinement_top_k = max(1, _env_int("WM_BEAM_PLAN_REFINEMENT_TOP_K", 4))
        if self.beam_refinement_rounds > 1 and self.imagined_rollout_mode != "open_loop":
            logger.info(
                "ewm_imagined: iterative beam refinement requires complete trajectories; "
                "switching imagined rollout mode to open_loop"
            )
            self.imagined_rollout_mode = "open_loop"
        # Advisory by default (Fix 2): the agent's own action always executes and the imagined
        # trajectory is injected only as guidance. WM_BEAM_PLAN_HARD_OVERRIDE=1 restores the old
        # behaviour of forcing the beam's confident recommendation over the agent's choice.
        self.beam_hard_override = (
            os.getenv("WM_BEAM_PLAN_HARD_OVERRIDE") or ""
        ).strip().lower() in {"1", "true", "yes"}
        # Optional policy handoff: at every live horizon step, ask the same acting model to
        # choose between its fresh action and the corresponding beam action. Choosing the
        # policy action invalidates the imagined suffix; choosing the plan advances its cursor.
        self.beam_revision = (os.getenv("WM_BEAM_PLAN_REVISION") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.beam_revision_temperature = max(
            0.0, _env_float("WM_BEAM_PLAN_REVISION_TEMPERATURE", 0.0)
        )
        self.beam_read_saturation_threshold = max(
            0, _env_int("WM_BEAM_PLAN_READ_SATURATION_THRESHOLD", 2)
        )
        self.beam_read_penalty = max(0.0, _env_float("WM_BEAM_PLAN_READ_PENALTY", 1.0))
        self.beam_read_after_progress_scale = max(
            0.0, _env_float("WM_BEAM_PLAN_READ_AFTER_PROGRESS_SCALE", 0.5)
        )
        self.beam_first_step_weight = max(0.0, _env_float("WM_BEAM_PLAN_FIRST_STEP_WEIGHT", 0.5))
        self.beam_required_action_coverage_bonus = max(
            0.0, _env_float("WM_BEAM_PLAN_REQUIRED_ACTION_COVERAGE_BONUS", 1.0)
        )
        self.beam_first_required_action_bonus = max(
            0.0, _env_float("WM_BEAM_PLAN_FIRST_REQUIRED_ACTION_BONUS", 0.75)
        )
        self.beam_schema_missing_required_penalty = max(
            0.0, _env_float("WM_BEAM_PLAN_SCHEMA_MISSING_REQUIRED_PENALTY", 0.15)
        )
        self.beam_schema_invalid_type_penalty = max(
            0.0, _env_float("WM_BEAM_PLAN_SCHEMA_INVALID_TYPE_PENALTY", 0.25)
        )
        # Decode the predicted TOOL OUTPUT text (not just canonical-event labels) for beam_plan's
        # winning, confidence-gated trajectory. Off by default: it's a generative greedy decode
        # per horizon step (one obs_grounding forward pass per step, run once per re-plan cycle --
        # not per candidate), and requires a checkpoint with a trained obs_grounding decoder
        # (native, or merged in via WM_JEPA_MERGE_CHECKPOINT).
        self.beam_decode_tool_output = (
            os.getenv("WM_BEAM_PLAN_DECODE_TOOL_OUTPUT") or ""
        ).strip().lower() in {"1", "true", "yes"}
        self.beam_decode_max_new_tokens = max(1, _env_int("WM_BEAM_PLAN_DECODE_MAX_NEW_TOKENS", 96))
        # When beam_plan spends a planning cycle. "interval" (default) re-plans every
        # WM_BEAM_MPC_EXECUTE_STEPS steps regardless of need. "critic" first scores the action
        # the agent already produced with the world model (ONE latent forward, no LLM call) and
        # only plans -- revising the current action AND looking several steps ahead -- when that
        # score says the action is bad (see _beam_plan_critic). Amortized cost becomes
        # critic + fire_rate * planning.
        self.beam_plan_trigger = (os.getenv("WM_BEAM_PLAN_TRIGGER") or "interval").strip().lower()
        if self.beam_plan_trigger not in {"interval", "critic"}:
            logger.warning(
                "ewm_imagined: unknown WM_BEAM_PLAN_TRIGGER=%s; using interval",
                self.beam_plan_trigger,
            )
            self.beam_plan_trigger = "interval"
        # critic trigger: plan when the predicted P(execution_status=failure) reaches this.
        self.beam_critic_failure_prob = _env_float("WM_BEAM_PLAN_CRITIC_FAILURE_PROB", 0.3)
        # critic trigger: plan when the action looks like it will not advance the task, i.e.
        # 1 - P(progress_signal=positive) reaches this.
        self.beam_critic_stall_prob = _env_float("WM_BEAM_PLAN_CRITIC_STALL_PROB", 0.7)
        # critic trigger: also plan whenever the score config's own P(failure)/P(deleted) safety
        # limits veto the action. This is a separate signal from the two thresholds above and is
        # not bounded by them, so it sets a floor on the fire rate -- on an LLM world model,
        # whose field probabilities are one-hot, that floor is most of the fire rate. Disable to
        # make WM_BEAM_PLAN_CRITIC_FAILURE_PROB/STALL_PROB the only levers.
        self.beam_critic_veto_fires = (
            os.getenv("WM_BEAM_PLAN_CRITIC_VETO_FIRES") or "true"
        ).strip().lower() in {"1", "true", "yes"}
        # critic trigger safety valve: force a planning cycle after this many consecutive
        # non-firing steps, so a mis-calibrated critic cannot disable lookahead for a whole
        # episode. 0 (default) is purely event-driven.
        self.beam_critic_max_quiet_steps = max(
            0, _env_int("WM_BEAM_PLAN_CRITIC_MAX_QUIET_STEPS", 0)
        )
        # Optional terminal predictor harness: when a checkpoint carries a terminal head,
        # surface P(done) as advisory guidance. At/above threshold means "verify and finish";
        # below threshold means "the task may not be finished; keep working/verifying."
        self.beam_terminal_advice = (
            os.getenv("WM_BEAM_PLAN_TERMINAL_ADVICE") or ""
        ).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.beam_terminal_advice_threshold = min(
            1.0, max(0.0, _env_float("WM_BEAM_PLAN_TERMINAL_ADVICE_THRESHOLD", 0.75))
        )
        # Optional dedicated action sampler. This is intentionally separate from both the
        # policy agent and the world model: a diffusion LLM can propose open-loop tool-call
        # plans while JEPA still predicts and scores their outcomes. It can run behind an
        # OpenAI-compatible endpoint or directly with Hugging Face Transformers.
        action_sampler_base_url = (
            (os.getenv("WM_BEAM_ACTION_SAMPLER_BASE_URL") or "").strip().rstrip("/")
        )
        action_sampler_backend = (
            (
                os.getenv("WM_BEAM_ACTION_SAMPLER_BACKEND")
                or ("openai" if action_sampler_base_url else "")
            )
            .strip()
            .lower()
        )
        if action_sampler_backend in {"hf", "transformers"}:
            action_sampler_backend = "huggingface"
        if action_sampler_backend not in {"", "openai", "huggingface"}:
            raise ValueError(
                "WM_BEAM_ACTION_SAMPLER_BACKEND must be openai or huggingface, got "
                f"{action_sampler_backend!r}"
            )
        self._beam_action_sampler = None
        self.beam_action_sampler_backend = action_sampler_backend
        self.beam_action_sampler_model = ""
        self.beam_action_sampler_max_new_tokens = max(
            1, _env_int("WM_BEAM_ACTION_SAMPLER_MAX_NEW_TOKENS", 256)
        )
        if action_sampler_backend == "huggingface":
            from ejepa_wm.backends._hf_diffusion_action_sampler import (
                HuggingFaceDiffusionActionSampler,
            )

            self.beam_action_sampler_model = (
                os.getenv("WM_BEAM_ACTION_SAMPLER_MODEL") or "google/diffusiongemma-26B-A4B-it"
            ).strip()
            denoising_steps = _env_int("WM_BEAM_ACTION_SAMPLER_MAX_DENOISING_STEPS", 0)
            self._beam_action_sampler = HuggingFaceDiffusionActionSampler(
                self.beam_action_sampler_model,
                max_new_tokens=self.beam_action_sampler_max_new_tokens,
                device_map=(os.getenv("WM_BEAM_ACTION_SAMPLER_DEVICE_MAP") or "auto").strip(),
                dtype=(os.getenv("WM_BEAM_ACTION_SAMPLER_DTYPE") or "bfloat16").strip(),
                max_denoising_steps=denoising_steps or None,
                trust_remote_code=(os.getenv("WM_BEAM_ACTION_SAMPLER_TRUST_REMOTE_CODE") or "")
                .strip()
                .lower()
                in {"1", "true", "yes"},
            )
        elif action_sampler_backend == "openai":
            if not action_sampler_base_url:
                raise ValueError(
                    "WM_BEAM_ACTION_SAMPLER_BASE_URL is required when "
                    "WM_BEAM_ACTION_SAMPLER_BACKEND=openai"
                )
            self.beam_action_sampler_model = (
                os.getenv("WM_BEAM_ACTION_SAMPLER_MODEL")
                or "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
            ).strip()
            self._beam_action_sampler = ewm.EwmGenerator(
                self.beam_action_sampler_model,
                action_sampler_base_url,
                os.getenv("WM_BEAM_ACTION_SAMPLER_API_KEY") or "not-needed",
                max_new_tokens=self.beam_action_sampler_max_new_tokens,
                timeout=_env_float("WM_BEAM_ACTION_SAMPLER_TIMEOUT", config.timeout),
            )
            # Temperature ladders require separate requests. urllib requests carry no shared
            # mutable client state, so vLLM can continuously batch them safely.
            self._beam_action_sampler.supports_parallel_requests = True
        # Per-episode MPC state (reset between tasks via reset_episode()).
        self._beam_imagined_plan: list[dict[str, Any]] = []  # cached imagined trajectory
        self._beam_plan_cursor = 0  # steps executed since last re-plan attempt
        self._beam_plan_history_offset = 0  # real history index where the cached plan starts
        self._beam_has_planned = (
            False  # forces a first re-plan attempt; after that the cooldown alone governs
        )
        self._beam_recent_tool_names: list[tuple] = []  # executed tool-name sets (anti-repetition)
        self._beam_quiet_steps = 0  # consecutive non-firing critic checks
        self._beam_critic_checks = 0  # critic evaluations this task (denominator of the fire rate)
        self._beam_critic_fires = 0  # critic evaluations that escalated to planning
        self._beam_pending_terminal_advice: dict[str, Any] | None = None
        self._beam_terminal_advice_count = 0
        self._beam_terminal_finish_advice_count = 0
        self._beam_terminal_middle_advice_count = 0
        self._agent_call_count = 0  # agent/LLM generations this episode

        # hier_latent_cem (MPC) knobs — used only for WM_STRATEGY=hier_latent_cem (see
        # hier_latent_cem_step). Mirror the EnterpriseOps-Gym orchestrator's --hier-cem-* args.
        self.hier_cem_anchors = max(1, _env_int("WM_HIER_CEM_ANCHORS", 8))
        self.hier_cem_samples = max(1, _env_int("WM_HIER_CEM_SAMPLES", 256))
        self.hier_cem_elites = max(1, _env_int("WM_HIER_CEM_ELITES", 16))
        self.hier_cem_iters = max(1, _env_int("WM_HIER_CEM_ITERS", 3))
        self.hier_cem_horizon = max(1, _env_int("WM_HIER_CEM_HORIZON", 5))
        self.hier_cem_init_std = _env_float("WM_HIER_CEM_INIT_STD", 0.2)
        self.hier_cem_min_std = _env_float("WM_HIER_CEM_MIN_STD", 0.02)
        self.hier_cem_smoothing = _env_float("WM_HIER_CEM_SMOOTHING", 1.0)
        self.hier_cem_min_elite_agreement = _env_float("WM_HIER_CEM_MIN_ELITE_AGREEMENT", 0.5)
        decode_strategy = (
            (os.getenv("WM_HIER_CEM_DECODE_STRATEGY") or "nearest_anchor").strip().lower()
        )
        if decode_strategy not in {"nearest_anchor", "learned_decoder"}:
            logger.warning(
                "ewm_imagined: unknown WM_HIER_CEM_DECODE_STRATEGY=%s; using nearest_anchor",
                decode_strategy,
            )
            decode_strategy = "nearest_anchor"
        self.hier_cem_decode_strategy = decode_strategy
        self.hier_cem_decode_max_new_tokens = max(
            1, _env_int("WM_HIER_CEM_DECODE_MAX_NEW_TOKENS", 96)
        )
        # Per-episode MPC state, same shape as beam_plan's (reset via reset_episode()).
        self._hier_imagined_plan: list[dict[str, Any]] = []
        self._hier_plan_cursor = 0
        self._hier_has_planned = False

        port = os.getenv("WM_VLLM_SERVER_PORT") or os.getenv("VLLM_SERVER_PORT") or "9000"
        base_url = (os.getenv("WM_VLLM_BASE_URL") or f"http://127.0.0.1:{port}/v1").rstrip("/")
        api_key = os.getenv("WM_VLLM_API_KEY") or "not-needed"
        self.ewm_model = os.getenv("WM_EWM_MODEL") or config.model or "gymops_world_model"
        qwen_agentworld_selected = is_qwen_agentworld_model(self.ewm_model) or (
            os.getenv("WM_QWEN_AGENTWORLD") or ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.qwen_agentworld = qwen_agentworld_selected
        # World-model generator selection (all reuse the same agent↔WM rollout loop):
        #  * WM_EWM_JEPA_CHECKPOINT set (or WM_EWM_BACKEND=jepa) -> a JEPA world model;
        #  * llm_canonical_trained -> reduced canonical-event JSON + terminal from a trained LLM,
        #    scored with one-hot probabilities for generated categories;
        #  * llm_canonical_zeroshot -> the same reduced canonical-event JSON + terminal prompt,
        #    but served by a general LLM with no task-specific state training;
        #  * llm_tool_output_judge -> a general LLM predicts tool outputs, then the policy-agent
        #    model judges step quality and whole trajectories;
        #  * WM_EWM_MCP_URL set -> route WM generation to a remote EWM MCP `generate` tool;
        #  * otherwise -> the original text LLM WM over vLLM (EwmGenerator).
        jepa_checkpoint = (os.getenv("WM_EWM_JEPA_CHECKPOINT") or "").strip()
        llm_canonical_event_checkpoint = (
            os.getenv("WM_EWM_LLM_CANONICAL_EVENT_CHECKPOINT") or ""
        ).strip()
        ewm_backend = (
            (os.getenv("WM_LLM_EWM_MODE") or os.getenv("WM_EWM_BACKEND") or "").strip().lower()
        )
        self.llm_ewm_mode = ewm_backend
        llm_canonical_trained_modes = {
            "llm_canonical_trained",
            "llm_canonical_event",
            "llm_canonical",
            "canonical_trained",
        }
        llm_canonical_zeroshot_modes = {
            "llm_canonical_zeroshot",
            "llm_canonical_zero_shot",
            "canonical_zeroshot",
            "canonical_zero_shot",
        }
        llm_tool_output_judge_modes = {
            "llm_tool_output_judge",
            "tool_output_judge",
            "llm_judge_tool_output",
        }
        mcp_url = (os.getenv("WM_EWM_MCP_URL") or "").strip()
        if jepa_checkpoint or ewm_backend == "jepa":
            if not jepa_checkpoint:
                raise ValueError(
                    "WM_EWM_BACKEND=jepa requires WM_EWM_JEPA_CHECKPOINT (the checkpoint dir)."
                )
            from ejepa_wm.backends._ewm_jepa import JepaEwmGenerator  # torch imported here

            jepa_kwargs = {
                "max_new_tokens": _env_int("WM_EWM_MAX_NEW_TOKENS", 512),
                "trust_remote_code": (os.getenv("WM_JEPA_TRUST_REMOTE_CODE") or "").strip().lower()
                in {"1", "true", "yes"},
                "dtype": os.getenv("WM_JEPA_DTYPE") or "auto",
                "imagined_observation_backend": os.getenv("WM_JEPA_OBSERVATION_BACKEND") or "auto",
                "arch_defaults": _jepa_arch_defaults(),
                # Merge in action_decoder_*/obs_ground_* weights from a separate checkpoint --
                # for checkpoints whose canonical-event-head training run silently dropped these
                # optional modules (see JepaEwmGenerator.__init__'s merge_checkpoint docstring).
                "merge_checkpoint": (os.getenv("WM_JEPA_MERGE_CHECKPOINT") or "").strip() or None,
                "merge_decode_max_new_tokens": _env_int("WM_JEPA_MERGE_DECODE_MAX_NEW_TOKENS", 96),
            }
            self._wm = shared_generator(
                ("jepa", jepa_checkpoint, jepa_kwargs),
                lambda: JepaEwmGenerator(jepa_checkpoint, **jepa_kwargs),
            )
            self.ewm_model = f"jepa:{jepa_checkpoint}"
            self._wm_endpoint = jepa_checkpoint
        elif llm_canonical_event_checkpoint or ewm_backend in llm_canonical_trained_modes:
            from ejepa_wm.backends._ewm_llm_canonical_event import (
                LlmCanonicalEventGenerator,
                ServedLlmCanonicalEventGenerator,
            )

            if llm_canonical_event_checkpoint:
                self._wm = LlmCanonicalEventGenerator(
                    llm_canonical_event_checkpoint,
                    max_new_tokens=_env_int("WM_EWM_MAX_NEW_TOKENS", 128),
                    trust_remote_code=(os.getenv("WM_JEPA_TRUST_REMOTE_CODE") or "").strip().lower()
                    in {"1", "true", "yes"},
                    dtype=os.getenv("WM_JEPA_DTYPE") or "auto",
                )
                self.ewm_model = f"llm_canonical_trained:{llm_canonical_event_checkpoint}"
                self._wm_endpoint = llm_canonical_event_checkpoint
            else:
                served = ewm.EwmGenerator(
                    self.ewm_model,
                    base_url,
                    api_key,
                    max_new_tokens=_env_int("WM_EWM_MAX_NEW_TOKENS", 256),
                    timeout=config.timeout,
                )
                served.supports_parallel_requests = True
                self._wm = ServedLlmCanonicalEventGenerator(
                    served, mode="llm_canonical_trained", max_workers=self.llm_batch_parallelism
                )
                self.ewm_model = f"llm_canonical_trained:{self.ewm_model}"
                self._wm_endpoint = base_url
        elif ewm_backend in llm_canonical_zeroshot_modes:
            from ejepa_wm.backends._ewm_llm_canonical_event import ServedLlmCanonicalEventGenerator

            served = ewm.EwmGenerator(
                self.ewm_model,
                base_url,
                api_key,
                max_new_tokens=_env_int("WM_EWM_MAX_NEW_TOKENS", 256),
                timeout=config.timeout,
            )
            served.supports_parallel_requests = True
            self._wm = ServedLlmCanonicalEventGenerator(
                served, mode="llm_canonical_zeroshot", max_workers=self.llm_batch_parallelism
            )
            self.ewm_model = f"llm_canonical_zeroshot:{self.ewm_model}"
            self._wm_endpoint = base_url
        elif ewm_backend in llm_tool_output_judge_modes or (
            qwen_agentworld_selected and not ewm_backend and not mcp_url
        ):
            from ejepa_wm.backends._ewm_llm_tool_output_judge import LlmToolOutputJudgeGenerator

            served = ewm.EwmGenerator(
                self.ewm_model,
                base_url,
                api_key,
                max_new_tokens=_env_int(
                    "WM_EWM_MAX_NEW_TOKENS", 32768 if qwen_agentworld_selected else 1024
                ),
                top_p=(
                    _env_float("WM_QWEN_AGENTWORLD_TOP_P", 0.95)
                    if qwen_agentworld_selected
                    else None
                ),
                top_k=(
                    _env_int("WM_QWEN_AGENTWORLD_TOP_K", 20) if qwen_agentworld_selected else None
                ),
                timeout=config.timeout,
            )
            served.supports_parallel_requests = True
            if qwen_agentworld_selected and not self.llm_ewm_mode:
                self.llm_ewm_mode = "llm_tool_output_judge"
            self._wm = LlmToolOutputJudgeGenerator(
                served,
                self._agent,
                max_workers=self.llm_batch_parallelism,
                max_tool_output_chars=_env_int("WM_LLM_TOOL_OUTPUT_JUDGE_MAX_CHARS", 2000),
                qwen_agentworld=self.qwen_agentworld,
                benchmark_name=os.getenv("BENCHMARK_NAME") or "",
                domain_override=os.getenv("WM_QWEN_AGENTWORLD_DOMAIN") or "",
                world_model_temperature=(
                    _env_float("WM_QWEN_AGENTWORLD_TEMPERATURE", 0.6)
                    if qwen_agentworld_selected
                    else 0.0
                ),
            )
            self.ewm_model = f"llm_tool_output_judge:{self.ewm_model}"
            self._wm_endpoint = base_url
        elif mcp_url:
            from ejepa_wm.backends._ewm_generators import McpEwmGenerator

            self._wm = McpEwmGenerator(
                mcp_url,
                tool_name=os.getenv("WM_EWM_MCP_TOOL") or "generate",
                api_key=os.getenv("WM_EWM_MCP_TOKEN") or None,
                timeout=config.timeout,
            )
            self._wm_endpoint = self._wm.url
        else:
            self._wm = ewm.EwmGenerator(
                self.ewm_model,
                base_url,
                api_key,
                max_new_tokens=_env_int("WM_EWM_MAX_NEW_TOKENS", 4096),
                timeout=config.timeout,
            )
            self._wm_endpoint = base_url

        # A JEPA is a transition model: the policy proposes hypothetical actions and JEPA
        # predicts their canonical states. Other backends retain ITP-I's free-form generative
        # WM request. The latter client holds no model weights and makes no request at startup.
        self._itp_uses_jepa_canonical = bool(
            jepa_checkpoint and getattr(self._wm, "canonical_event_available", False)
        )
        if config.strategy == "itp_i" and jepa_checkpoint and not self._itp_uses_jepa_canonical:
            raise ValueError(
                "JEPA-backed ITP-I requires trained canonical-event heads in the checkpoint."
            )
        self._itp_world_model = None
        if not self._itp_uses_jepa_canonical:
            self._itp_world_model = ewm.EwmGenerator(
                os.getenv("WM_ITP_WORLD_MODEL_MODEL")
                or os.getenv("WM_EWM_MODEL")
                or config.model
                or "gymops_world_model",
                (os.getenv("WM_ITP_WORLD_MODEL_BASE_URL") or base_url).rstrip("/"),
                os.getenv("WM_ITP_WORLD_MODEL_API_KEY") or api_key,
                max_new_tokens=self.itp_foresight_tokens,
                timeout=config.timeout,
            )

        self._k_controller = None
        if self.k_controller_mode == "react_wm_rl_k":
            model_path = (os.getenv("K_CONTROLLER_MODEL_PATH") or "").strip()
            if not model_path:
                logger.warning(
                    "ewm_imagined: react_wm_rl_k but K_CONTROLLER_MODEL_PATH unset; static depth"
                )
                self.k_controller_mode = ""
            else:
                try:
                    from ejepa_wm.backends import _ewm_k_controller as kc  # torch imported here

                    self._k_controller = kc.KController(
                        model_path=model_path,
                        kmax=self.max_steps,
                        device_str=os.getenv("K_CONTROLLER_DEVICE") or "auto",
                        dtype_str=os.getenv("K_CONTROLLER_DTYPE") or "auto",
                    )
                except Exception as exc:
                    logger.warning("ewm_imagined: K-controller load failed (%s); static depth", exc)
                    self.k_controller_mode = ""

        logger.info(
            "ewm_imagined: wm_state=%s optimizer=%s k_controller=%s kmax=%d model=%s endpoint=%s",
            self.wm_state,
            self.action_optimizer or "none",
            self.k_controller_mode or "static",
            self.max_steps,
            self.ewm_model,
            self._wm_endpoint,
        )
        if self._beam_action_sampler is not None:
            logger.info(
                "ewm_imagined: dedicated beam action sampler model=%s max_new_tokens=%d",
                self.beam_action_sampler_model,
                self.beam_action_sampler_max_new_tokens,
            )

    @staticmethod
    def _prompt_from_flow(conversation_flow: list[dict[str, Any]], etype: str) -> str:
        for event in conversation_flow or []:
            if isinstance(event, dict) and event.get("type") == etype:
                return str(event.get("content", "") or "")
        return ""

    def _resolve_k(
        self, conversation, previous_state, state_history, system_prompt, user_prompt
    ) -> int:
        kmax = self.max_steps
        if self.k_controller_mode == "react_wm_decide_k":
            return ewm.decide_k_via_agent(
                self._agent, conversation, previous_state, kmax=kmax, fallback_k=kmax
            )
        if self.k_controller_mode == "react_wm_rl_k" and self._k_controller is not None:
            state_text = ewm.build_k_controller_state_text(
                system_prompt, user_prompt, previous_state, state_history
            )
            try:
                return max(0, min(kmax, int(self._k_controller.decide_k(state_text).k)))
            except Exception as exc:
                logger.warning("ewm_imagined: decide_k failed (%s); static kmax", exc)
                return kmax
        return kmax

    def _react_system_prompt_from_flow(self, conversation_flow) -> str:
        """Build the imagined-rollout ReAct system prompt from a ``tools`` event in the flow.

        The tool catalog travels in the ``conversation_flow`` (not a constructor arg), so the
        WM stays executor-agnostic while the text-protocol imagined rollout still gets the
        training-matched ReAct-with-tools system prompt it needs to emit valid ``tool_call``s
        (matches evaluation.py / the EWM training data). Returns "" when no tools are present.
        """
        for event in conversation_flow or []:
            if isinstance(event, dict) and event.get("type") == "tools":
                tools = event.get("tools")
                if not tools:
                    return ""
                try:
                    return ewm.build_react_system_prompt(
                        ewm.build_react_tool_descriptions_from_gym(tools)
                    )
                except Exception as exc:
                    logger.warning("ewm_imagined: react prompt from flow tools failed (%s)", exc)
                    return ""
        return ""

    def advise(self, conversation_flow, *, history=None) -> AdviseResult:
        if self.config.strategy == "itp_i":
            return self._itp_i_advise(conversation_flow)
        conversation = ewm.to_ewm_conversation(conversation_flow)
        system_prompt = self._prompt_from_flow(conversation_flow, "system_message")
        user_prompt = self._prompt_from_flow(conversation_flow, "user_message")
        # The imagined agent rolls out via the *text* ReAct protocol, so it needs the tool
        # catalog embedded in its system prompt to emit valid tool_calls (else the rollouts
        # hallucinate and the WM scores garbage). Built from the `tools` event the executor
        # puts in the conversation_flow; fall back to the agent's own system prompt only if
        # no tool catalog is present.
        react_system_prompt = (
            self._react_system_prompt_from_flow(conversation_flow)
            or system_prompt
            or "You are a ReAct agent. Return JSON actions only."
        )
        # Reconstruct the ENTERPRISE state + multi-step history from the flow (matches
        # what the WM was trained on), instead of a compact single-state shortcut.
        previous_state, state_history = ewm.enterprise_state_from_flow(
            conversation_flow, max_items=self.state_history_size
        )

        k_steps = self._resolve_k(
            conversation, previous_state, state_history, system_prompt, user_prompt
        )
        base_detail = {
            "backend": "ewm_imagined",
            "wm_state": self.wm_state,
            "action_optimizer": self.action_optimizer or "none",
            "k_controller": self.k_controller_mode or "static",
            "k_steps": k_steps,
            "num_rollouts": self.imagined_rollouts,
            "selection": self.imagined_selection,
            "imagined_rollout_mode": self.imagined_rollout_mode,
            "imagined_parallel_rollouts": self.imagined_parallel_rollouts,
            "imagined_single_call_step": self.imagined_single_call_step,
        }
        if k_steps <= 0:
            return AdviseResult(text="", detail={**base_detail, "imagined_step_count": 0})

        try:
            steps = ewm.optimize_imagined_trajectory(
                self.action_optimizer,
                self._agent,
                self._wm,
                conversation,
                previous_state,
                react_system_prompt,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                wm_state=self.wm_state,
                max_imagined_steps=k_steps,
                candidate_action_count=self.candidate_actions,
                top_k=self.top_k,
                temperature=self.temperature,
                state_history=state_history,
                state_history_size=self.state_history_size,
                num_rollouts=self.imagined_rollouts,
                selection_strategy=self.imagined_selection,
                rollout_mode=self.imagined_rollout_mode,
                llm_batch_parallelism=self.llm_batch_parallelism,
                parallel_rollouts=self.imagined_parallel_rollouts,
                single_call_step=self.imagined_single_call_step,
            )
        except Exception as exc:
            # The EWM client raises urllib errors; the agent (chat_fn → openai/httpx)
            # raises its own connection types. Either way a transient rollout failure
            # must degrade to plain ReAct (no injection this step), not error the task.
            logger.warning("ewm_imagined: rollout failed (%s); no injection this step", exc)
            return AdviseResult(
                text="", detail={**base_detail, "error": str(exc), "imagined_step_count": 0}
            )

        text = ewm.build_imagined_trajectory_message(steps)["content"] if steps else ""
        return AdviseResult(
            text=text, detail={**base_detail, "imagined_step_count": len(steps or [])}
        )

    def _itp_i_advise(self, conversation_flow) -> AdviseResult:
        """Run ITP-I's adaptive-K imagination stage and return reflection guidance.

        The executor's subsequent tool-bound policy call is the reflect-and-act stage. Keeping
        that stage outside this backend preserves each benchmark's native tool-call protocol.
        """
        from ejepa_wm.backends._ewm_itp_i import (
            canonical_action_step_messages,
            decision_messages,
            imagination_messages,
            parse_lookahead,
            reflection_guidance,
        )

        started = time.monotonic()
        task = self._prompt_from_flow(conversation_flow, "user_message")
        observed = render_flow(conversation_flow, max_chars=4000)
        tools: list[Any] = []
        for event in conversation_flow or []:
            if isinstance(event, dict) and event.get("type") == "tools":
                tools = list(event.get("tools") or [])
                break

        decision_raw = ""
        decision_error = ""
        if self.itp_fixed_k >= 0:
            k = max(0, min(self.itp_max_k, self.itp_fixed_k))
            decision_mode = "fixed"
        else:
            decision_mode = "adaptive"
            try:
                self._agent_call_count += 1
                decision_raw = self._agent.generate_from_messages(
                    decision_messages(task, observed, max_k=self.itp_max_k),
                    temperature=self.itp_decision_temperature,
                )
                k = parse_lookahead(decision_raw, max_k=self.itp_max_k)
            except Exception as exc:
                decision_error = str(exc)
                k = min(self.itp_max_k, 1)
                logger.warning("ewm_imagined: ITP-I K selection failed (%s); using K=%d", exc, k)

        foresight = ""
        action_plan_raw: list[str] = []
        action_plan: list[dict[str, Any]] = []
        imagined_plan: list[dict[str, Any]] = []
        wm_error = ""
        wm_calls = 0
        action_proposal_calls = 0
        action_proposal_seconds = 0.0
        wm_seconds = 0.0
        if k > 0:
            try:
                if self._itp_uses_jepa_canonical:
                    from ejepa_wm.backends._ewm_beam_plan import BeamPlanConfig, parse_single_plan

                    system_prompt = self._prompt_from_flow(conversation_flow, "system_message")
                    input_history = self._input_history_from_flow(conversation_flow)
                    allowed_tools = set(self._tool_names_from_flow(conversation_flow))
                    plan_config = BeamPlanConfig(num_candidates=1, horizon=1)
                    for imagined_step in range(1, k + 1):
                        prefix_text = (
                            self._render_plan_message(imagined_plan) if imagined_plan else ""
                        )
                        action_proposal_calls += 1
                        self._agent_call_count += 1
                        proposal_started = time.monotonic()
                        try:
                            raw_step = self._agent.generate_from_messages(
                                canonical_action_step_messages(
                                    task,
                                    observed,
                                    tools,
                                    imagined_prefix=prefix_text,
                                    step=imagined_step,
                                    k=k,
                                ),
                                temperature=self.itp_world_model_temperature,
                            )
                        finally:
                            action_proposal_seconds += time.monotonic() - proposal_started
                        action_plan_raw.append(raw_step)
                        parsed_step = parse_single_plan(raw_step, plan_config)
                        wrapped = self._wrap_step(parsed_step[0] if parsed_step else None)
                        if not wrapped:
                            raise ValueError(
                                f"policy returned no parseable action at imagined step {imagined_step}"
                            )
                        validation_error = self._plan_validation_error([wrapped], allowed_tools)
                        if validation_error:
                            raise ValueError(
                                "invalid hypothetical action at imagined step "
                                f"{imagined_step}: {validation_error}"
                            )
                        action_plan.append(wrapped)

                        # Re-score the full prefix so JEPA's latent rollout reaches the state
                        # on which the next policy action must condition. This is K alternating
                        # policy/WM interactions as specified by ITP, not open-loop planning.
                        wm_calls += 1
                        model_started = time.monotonic()
                        try:
                            scored = self._wm.score_action_plans_canonical_event(
                                system_prompt=system_prompt,
                                user_prompt=task,
                                input_history=input_history,
                                action_plans=[list(action_plan)],
                            )
                        finally:
                            wm_seconds += time.monotonic() - model_started
                        record = scored[0] if scored else {}
                        states = record.get("per_step_predicted_state") or []
                        terminal = record.get("per_step_terminal_prob") or []
                        imagined_plan.append(
                            {
                                "calls": [
                                    ewm.normalize_tool_call(call)
                                    for call in (wrapped.get("tool_calls") or [])
                                ],
                                "predicted_state": states[-1] if states else {},
                                "terminal_probability": terminal[-1] if terminal else None,
                            }
                        )
                    foresight = self._render_plan_message(imagined_plan)
                elif self._itp_world_model is not None:
                    wm_calls = 1
                    model_started = time.monotonic()
                    try:
                        foresight = self._itp_world_model.generate_from_messages(
                            imagination_messages(task, observed, tools, k=k),
                            temperature=self.itp_world_model_temperature,
                        )
                    finally:
                        wm_seconds = time.monotonic() - model_started
                else:
                    raise RuntimeError("ITP-I world model is unavailable")
            except Exception as exc:
                wm_error = str(exc)
                if imagined_plan:
                    foresight = self._render_plan_message(imagined_plan)
                logger.warning("ewm_imagined: ITP-I imagination failed (%s)", exc)
        elapsed = time.monotonic() - started
        detail = {
            "backend": "ewm_imagined",
            "strategy": "itp_i",
            "itp_i_k": k,
            "itp_i_max_k": self.itp_max_k,
            "itp_i_k_mode": decision_mode,
            "itp_i_k_raw": decision_raw,
            "itp_i_foresight": foresight,
            "itp_i_world_model_kind": (
                "jepa_canonical_event" if self._itp_uses_jepa_canonical else "generative_text"
            ),
            "itp_i_action_plan_raw": action_plan_raw,
            "itp_i_action_plan": action_plan,
            "itp_i_action_proposal_calls": action_proposal_calls,
            "itp_i_action_proposal_seconds": max(0.0, action_proposal_seconds),
            "itp_i_world_model_calls": wm_calls,
            "itp_i_policy_decision_calls": 0 if decision_mode == "fixed" else 1,
            "itp_i_world_model_seconds": max(0.0, wm_seconds),
            "itp_i_total_advice_seconds": max(0.0, elapsed),
            "itp_i_decision_error": decision_error,
            "itp_i_world_model_error": wm_error,
        }
        return AdviseResult(text=reflection_guidance(foresight, k=k), detail=detail)

    def select(self, conversation_flow, candidate_events) -> SelectResult:
        # imagined is a prompt_injection backend; selection is not supported.
        return SelectResult(index=0, detail={"backend": "ewm_imagined", "unsupported": "select"})

    # -- beam_plan (MPC) strategy ------------------------------------------
    #
    # Port of the EnterpriseOps-Gym orchestrator's ``mode == "beam_plan"`` block into the
    # pluggable WM contract. Every ``WM_BEAM_MPC_EXECUTE_STEPS`` steps this RE-PLANS: one
    # agent call per horizon step proposes ``m`` candidate next actions; the JEPA world model
    # scores them with the canonical-event heads (LLM-free) via
    # ``score_action_plans_canonical_event``; the best-per-step forms an imagined trajectory
    # that is cached and shown to the agent (``beam_injection_text``) so for the next
    # execute-steps steps the agent FOLLOWS the visible plan without re-invoking the WM. The
    # step-0 action can override the agent's baseline, but only when it beats the baseline by
    # more than ``WM_BEAM_PLAN_SCORE_MARGIN`` (margin-gated). Requires the JEPA canonical-event
    # backend; any other backend has no ``score_action_plans_canonical_event`` and callers
    # should fall back to the plain agent (see ``supports_beam_plan``).

    def supports_beam_plan(self) -> bool:
        return callable(getattr(self._wm, "score_action_plans_canonical_event", None))

    def supports_action_feedback(self) -> bool:
        """Whether this backend can predict a proposed action's next state or tool output."""
        return callable(getattr(self._wm, "score_action_plans_canonical_event", None))

    def action_feedback(
        self,
        conversation_flow,
        *,
        seed_calls,
        user_query: str = "",
        mode: str = "revision",
    ) -> AdviseResult:
        """Predict one proposed action and format feedback for revision or delayed reference."""
        mode = str(mode or "revision").strip().lower()
        if mode not in {"revision", "reference"}:
            raise ValueError(f"Unknown action-feedback mode: {mode}")
        seed_norm = [ewm.normalize_tool_call(call) for call in (seed_calls or [])]
        seed_norm = [call for call in seed_norm if str(call.get("name") or "").strip()]
        detail: dict[str, Any] = {
            "backend": "ewm_imagined",
            "strategy": mode,
            "calls": seed_norm,
            "world_model_calls": 0,
        }
        if not self.supports_action_feedback():
            return AdviseResult(text="", detail={**detail, "event": "unsupported"})
        if not seed_norm:
            return AdviseResult(text="", detail={**detail, "event": "no_action"})

        system_prompt = self._prompt_from_flow(conversation_flow, "system_message")
        task = user_query or self._prompt_from_flow(conversation_flow, "user_message")
        step = self._wrap_step({"tool_calls": seed_norm})
        if not step:
            return AdviseResult(text="", detail={**detail, "event": "invalid_action"})
        started = time.monotonic()
        scored = self._wm.score_action_plans_canonical_event(
            system_prompt=system_prompt,
            user_prompt=task,
            input_history=self._input_history_from_flow(conversation_flow),
            action_plans=[[step]],
        )
        elapsed = time.monotonic() - started
        record = scored[0] if scored else {}
        states = record.get("per_step_predicted_state") or []
        terminals = record.get("per_step_terminal_prob") or []
        predicted_state = states[0] if states else {}
        terminal_probability = terminals[0] if terminals else record.get("terminal_probability")
        tool_outputs = record.get("per_step_predicted_tool_output") or []
        predicted_tool_output = (
            tool_outputs[0] if tool_outputs else record.get("predicted_tool_output", "")
        )
        per_step = record.get("per_step") or []
        step_judge = (
            (per_step[0].get("judge") or {}) if per_step and isinstance(per_step[0], dict) else {}
        )
        tool_output_feedback = (
            "per_step_predicted_tool_output" in record or "predicted_tool_output" in record
        )
        rollout = self._render_plan_message(
            [
                {
                    "calls": seed_norm,
                    "predicted_state": predicted_state,
                    "terminal_probability": terminal_probability,
                    "predicted_tool_output": predicted_tool_output,
                }
            ]
        )
        judge_text = ""
        if step_judge:
            judge_text = (
                "\nSame-agent critic assessment: "
                f"can_proceed={step_judge.get('can_proceed')}, "
                f"failure_probability={float(step_judge.get('failure_score', 0.0)):.2f}, "
                f"progress_probability={float(step_judge.get('progress_score', 0.0)):.2f}, "
                f"enough_to_finish={step_judge.get('enough_to_finish')}, "
                f"finish_probability={float(step_judge.get('finish_score', 0.0)):.2f}"
            )
            reason = str(step_judge.get("reason") or "").strip()
            if reason:
                judge_text += f"; reason={reason[:300]}"
        if mode == "revision":
            text = (
                "[WM_ACTION_REVISION]\n"
                "The world model evaluated your proposed action before execution:\n"
                f"{rollout}{judge_text}\n\n"
                "Decide whether to PROCEED with the same action or REVISE it. Return the actual "
                "next action using the normal tool-call format. Treat the prediction as uncertain "
                "guidance, not an observed result.\n"
                "[/WM_ACTION_REVISION]"
            )
        else:
            text = (
                "[WM_PREVIOUS_ACTION_REFERENCE]\n"
                "For reference, the world model predicted the following consequence for the "
                "previous step's action:\n"
                f"{rollout}{judge_text}\n\n"
                "Compare this prediction with the actual tool result now available, and use it as "
                "uncertain context for the next action.\n"
                "[/WM_PREVIOUS_ACTION_REFERENCE]"
            )
        detail.update(
            {
                "event": "action_feedback",
                "prediction_kind": "tool_output" if tool_output_feedback else "canonical_state",
                "predicted_state": predicted_state,
                "predicted_tool_output": predicted_tool_output,
                "step_judge": step_judge,
                "terminal_probability": terminal_probability,
                "score": record.get("score"),
                "vetoed": record.get("vetoed"),
                "world_model_calls": 1,
                "judge_calls": 1 if tool_output_feedback else 0,
                "model_calls": 2 if tool_output_feedback else 1,
                "world_model_seconds": max(0.0, elapsed),
            }
        )
        return AdviseResult(text=text, detail=detail)

    def supports_hier_cem(self) -> bool:
        """hier_latent_cem needs direct access to the JEPA net (predict_latent /
        predict_canonical_event_logits), its tokenizer, and trained canonical-event heads --
        i.e. a JepaEwmGenerator, same requirement as beam_plan's scoring."""
        return (
            hasattr(self._wm, "model")
            and hasattr(self._wm, "tokenizer")
            and bool(getattr(self._wm, "canonical_event_available", False))
        )

    def reset_episode(self) -> None:
        """Clear per-episode MPC state; call at the start of each task."""
        self._beam_imagined_plan = []
        self._beam_plan_cursor = 0
        self._beam_plan_history_offset = 0
        self._beam_has_planned = False
        self._beam_recent_tool_names = []
        self._beam_quiet_steps = 0
        self._beam_critic_checks = 0
        self._beam_critic_fires = 0
        self._beam_pending_terminal_advice = None
        self._beam_terminal_advice_count = 0
        self._beam_terminal_finish_advice_count = 0
        self._beam_terminal_middle_advice_count = 0
        self._hier_imagined_plan = []
        self._hier_plan_cursor = 0
        self._hier_has_planned = False
        self._agent_call_count = 0

    @property
    def agent_call_count(self) -> int:
        return self._agent_call_count

    @staticmethod
    def _tool_names_from_flow(conversation_flow) -> list[str]:
        names: list[str] = []
        for event in conversation_flow or []:
            if isinstance(event, dict) and event.get("type") == "tools":
                for tool in event.get("tools") or []:
                    if not isinstance(tool, dict):
                        continue
                    name = tool.get("name") or (tool.get("function") or {}).get("name")
                    if name:
                        names.append(str(name))
        return names

    @staticmethod
    def _tool_schema_map_from_flow(conversation_flow) -> dict[str, dict[str, Any]]:
        schemas: dict[str, dict[str, Any]] = {}
        for event in conversation_flow or []:
            if not isinstance(event, dict) or event.get("type") != "tools":
                continue
            for tool in event.get("tools") or []:
                if not isinstance(tool, dict):
                    continue
                name = tool.get("name") or (tool.get("function") or {}).get("name")
                if not name:
                    continue
                schema = (
                    tool.get("inputSchema")
                    or tool.get("input_schema")
                    or ((tool.get("function") or {}).get("parameters"))
                    or {}
                )
                if isinstance(schema, dict):
                    schemas[str(name)] = schema
        return schemas

    @staticmethod
    def _input_history_from_flow(conversation_flow) -> list[dict[str, Any]]:
        """Reconstruct a ``{step, action, observation}`` history (the shape the JEPA
        current-state text renderer expects) from the agent's conversation_flow."""
        entries: list[dict[str, Any]] = []
        pending_action: dict[str, Any] | None = None
        step = 0
        for event in conversation_flow or []:
            if not isinstance(event, dict):
                continue
            etype = event.get("type")
            if etype == "ai_message":
                calls = event.get("tool_calls") or []
                pending_action = {"tool_calls": ewm.to_openai_tool_calls(calls)} if calls else None
            elif etype == "tool_result":
                entries.append(
                    {"step": step, "action": pending_action, "observation": event.get("result")}
                )
                step += 1
                pending_action = None
        return entries

    @staticmethod
    def _normalize_step_arguments(arguments: Any) -> Any:
        if isinstance(arguments, str):
            text = arguments.strip()
            if not text:
                return {}
            try:
                parsed = ewm.parse_jsonish(text)
            except Exception:
                return text
            return parsed
        return arguments

    @staticmethod
    def _wrap_step(step: dict[str, Any] | None) -> dict[str, Any] | None:
        """Normalize an LLM step into the training-action format
        ``{"tool_calls": [{"type": "function", "function": {"name", "arguments"}}]}``.
        Accepts ``{"tool_calls": [...]}``, ``{"name", "arguments"}``,
        ``{"function": {...}}`` and ``{"tool"/"tool_name": ...}`` shapes."""
        if not isinstance(step, dict):
            return None
        metadata: dict[str, Any] = {}
        for key in ("id", "bind"):
            if key in step:
                metadata[key] = step[key]
        if isinstance(step.get("tool_calls"), list) and step["tool_calls"]:
            calls = []
            for call in step["tool_calls"]:
                norm = ewm.normalize_tool_call(call)
                name = str(norm.get("name", "")).strip()
                if name:
                    calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": EwmImaginedWorldModel._normalize_step_arguments(
                                    norm.get("arguments", {})
                                ),
                            },
                        }
                    )
            return {**metadata, "tool_calls": calls} if calls else None
        function = step.get("function") if isinstance(step.get("function"), dict) else None
        source = function or step
        name = str(source.get("name") or step.get("tool") or step.get("tool_name") or "").strip()
        if not name:
            return None
        arguments = source.get("arguments") or step.get("args") or step.get("arguments") or {}
        return {
            **metadata,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": EwmImaginedWorldModel._normalize_step_arguments(arguments),
                    },
                }
            ],
        }

    @staticmethod
    def _step_tool_names(step: dict[str, Any]) -> tuple:
        if not isinstance(step, dict):
            return tuple()
        names = []
        for call in step.get("tool_calls") or []:
            if isinstance(call, dict):
                name = str((call.get("function") or {}).get("name") or call.get("name", "")).strip()
                if name:
                    names.append(name)
        return tuple(sorted(names))

    @classmethod
    def _candidate_score_diagnostics(
        cls,
        records: list[dict[str, Any]],
        path_names: list[tuple],
        repeat_penalty_scale: float,
    ) -> list[dict[str, Any]]:
        """Return compact, auditable score/rank telemetry for every beam candidate."""
        diagnostics = []
        for record in records:
            plan = record.get("plan") or []
            first_names = cls._step_tool_names(plan[0]) if plan else ()
            adjusted_score = float(record.get("adjusted_score", record.get("score", 0.0)))
            repeat_penalty = repeat_penalty_scale * path_names.count(first_names)
            schema_warning_delta = float(record.get("schema_warning_score_delta", 0.0))
            diagnostics.append(
                {
                    "plan_index": int(record.get("plan_index", record.get("index", -1))),
                    "first_action_names": list(first_names),
                    "raw_score": float(record.get("raw_score", adjusted_score)),
                    "raw_rank": record.get("raw_rank"),
                    "raw_vetoed": bool(record.get("raw_vetoed", record.get("vetoed", False))),
                    "adjusted_score": adjusted_score,
                    "adjusted_rank": record.get("adjusted_rank"),
                    "adjusted_vetoed": bool(
                        record.get("adjusted_vetoed", record.get("vetoed", False))
                    ),
                    "action_aware_score_delta": float(
                        record.get(
                            "action_aware_score_delta",
                            adjusted_score - float(record.get("raw_score", adjusted_score)),
                        )
                    ),
                    "action_aware_adjustments": record.get("action_aware_adjustments", []),
                    "schema_warning_count": int(record.get("schema_warning_count", 0) or 0),
                    "schema_warning_score_delta": schema_warning_delta,
                    "schema_warnings": record.get("schema_warnings", []),
                    "repeat_penalty": repeat_penalty,
                    "planner_score": adjusted_score - repeat_penalty,
                }
            )

        def _assign_rank(prefix: str, score_key: str, veto_key: str) -> None:
            ranked = sorted(
                diagnostics,
                key=lambda item: (item[veto_key], -item[score_key], item["plan_index"]),
            )
            for rank, item in enumerate(ranked, start=1):
                if item.get(f"{prefix}_rank") is None:
                    item[f"{prefix}_rank"] = rank

        _assign_rank("raw", "raw_score", "raw_vetoed")
        _assign_rank("adjusted", "adjusted_score", "adjusted_vetoed")
        _assign_rank("planner", "planner_score", "adjusted_vetoed")
        diagnostics.sort(key=lambda item: item["plan_index"])
        return diagnostics

    @staticmethod
    def _log_candidate_score_diagnostics(
        depth: int | str, diagnostics: list[dict[str, Any]]
    ) -> None:
        for candidate in diagnostics:
            logger.info(
                "ewm_imagined: beam candidate scores depth=%s candidate=%s",
                depth,
                json.dumps(candidate, ensure_ascii=False, default=str),
            )

    @classmethod
    def _argument_signature(cls, value: Any) -> Any:
        """Argument shape for open-loop duplicate detection.

        Keep the complete plan structure and argument keys, but collapse concrete scalar values so
        two samples that only differ by guessed IDs/names do not consume duplicate WM scoring.
        Symbolic refs keep their dependency shape because `$step1.id` vs `$step2.id` is meaningful.
        """
        if isinstance(value, dict):
            return tuple(
                sorted((str(key), cls._argument_signature(val)) for key, val in value.items())
            )
        if isinstance(value, list):
            if not value:
                return ("list", 0)
            return ("list", tuple(cls._argument_signature(item) for item in value[:3]))
        if isinstance(value, str):
            text = value.strip()
            if re.match(r"^\$step\d+(?:\.[A-Za-z_][A-Za-z0-9_]*)*$", text):
                return re.sub(r"\$step\d+", "$stepN", text)
            return "str"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, (int, float)):
            return "number"
        if value is None:
            return "null"
        return type(value).__name__

    @classmethod
    def _plan_signature(cls, steps: list[dict[str, Any]]) -> tuple:
        """Whole-plan dedup signature, not just first action.

        This preserves candidates that share the same first tool but diverge downstream, while
        collapsing exact strategy duplicates with superficial concrete-value variation.
        """
        signature = []
        for step in steps:
            calls = []
            for call in step.get("tool_calls") or []:
                norm = ewm.normalize_tool_call(call)
                name = str(norm.get("name", "")).strip()
                if name:
                    calls.append((name, cls._argument_signature(norm.get("arguments", {}))))
            signature.append(tuple(calls))
        return tuple(signature)

    @staticmethod
    def _schema_type_matches(value: Any, expected: str) -> bool:
        if expected == "string":
            return isinstance(value, str)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "object":
            return isinstance(value, dict)
        if expected == "array":
            return isinstance(value, list)
        return True

    @classmethod
    def _argument_schema_errors(
        cls, tool_name: str, arguments: dict[str, Any], tool_schemas: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        from ejepa_wm.backends._ewm_symbolic_plan import has_symbolic_references

        errors: list[dict[str, Any]] = []
        schema = tool_schemas.get(tool_name) or {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        for key in required:
            if str(key) not in arguments:
                errors.append(
                    {
                        "tool": tool_name,
                        "argument": str(key),
                        "reason": "missing_required_argument",
                    }
                )
        for key, value in arguments.items():
            prop = properties.get(key)
            if not isinstance(prop, dict):
                continue
            expected = prop.get("type")
            if isinstance(expected, list):
                expected_types = [str(item) for item in expected]
            elif isinstance(expected, str):
                expected_types = [expected]
            else:
                expected_types = []
            if not expected_types or has_symbolic_references(value):
                continue
            if not any(
                cls._schema_type_matches(value, expected_type) for expected_type in expected_types
            ):
                errors.append(
                    {
                        "tool": tool_name,
                        "argument": str(key),
                        "reason": "invalid_argument_type",
                        "expected": expected_types,
                        "actual": type(value).__name__,
                    }
                )
        return errors

    @classmethod
    def _argument_schema_error(
        cls, tool_name: str, arguments: dict[str, Any], tool_schemas: dict[str, dict[str, Any]]
    ) -> str | None:
        errors = cls._argument_schema_errors(tool_name, arguments, tool_schemas)
        if errors:
            return str(errors[0].get("reason") or "invalid_arguments")
        return None

    @classmethod
    def _plan_validation_detail(
        cls,
        steps: list[dict[str, Any]],
        allowed_tool_names: set[str],
        tool_schemas: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return hard structural errors plus soft schema warnings for a sampled plan.

        Argument values may contain unresolved symbolic references, so validation deliberately
        permits symbolic strings where a runtime JSON schema expects a concrete scalar. Tool names
        must exist, every argument payload must be a JSON object, and symbolic dependencies must
        point to earlier steps with compatible semantic types when they can be inferred. Missing
        required args and scalar type mismatches are soft warnings because generated lookahead
        plans are advisory/scored candidates, not necessarily directly executable actions.
        """
        from ejepa_wm.backends._ewm_symbolic_plan import validate_symbolic_references

        tool_schemas = tool_schemas or {}
        symbolic = validate_symbolic_references(steps)
        if symbolic.get("broken_references"):
            return {
                "hard_error": "broken_reference",
                "schema_warnings": [],
                "symbolic": symbolic,
            }
        schema_warnings: list[dict[str, Any]] = []
        for step_index, step in enumerate(steps, start=1):
            for call in step.get("tool_calls") or []:
                norm = ewm.normalize_tool_call(call)
                name = str(norm.get("name", "")).strip()
                if allowed_tool_names and name not in allowed_tool_names:
                    return {
                        "hard_error": "invalid_tool",
                        "schema_warnings": schema_warnings,
                        "symbolic": symbolic,
                    }
                arguments = norm.get("arguments", {})
                if not isinstance(arguments, dict):
                    return {
                        "hard_error": "invalid_arguments",
                        "schema_warnings": schema_warnings,
                        "symbolic": symbolic,
                    }
                for warning in cls._argument_schema_errors(name, arguments, tool_schemas):
                    schema_warnings.append({"step": step_index, **warning})
        return {"hard_error": None, "schema_warnings": schema_warnings, "symbolic": symbolic}

    @classmethod
    def _plan_validation_error(
        cls,
        steps: list[dict[str, Any]],
        allowed_tool_names: set[str],
        tool_schemas: dict[str, dict[str, Any]] | None = None,
    ) -> str | None:
        detail = cls._plan_validation_detail(steps, allowed_tool_names, tool_schemas)
        hard_error = detail.get("hard_error")
        if hard_error:
            return str(hard_error)
        return None

    @staticmethod
    def _schema_warning_penalty(
        warnings: list[dict[str, Any]],
        *,
        missing_required_penalty: float,
        invalid_type_penalty: float,
    ) -> float:
        penalty = 0.0
        for warning in warnings:
            reason = warning.get("reason")
            if reason == "missing_required_argument":
                penalty += missing_required_penalty
            elif reason == "invalid_argument_type":
                penalty += invalid_type_penalty
        return -penalty

    def _schema_warnings_by_step(
        self, plan: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[int, float]]:
        warnings: list[dict[str, Any]] = []
        per_step_deltas: dict[int, float] = {}
        for step_index, step in enumerate(plan, start=1):
            if not isinstance(step, dict):
                continue
            step_warnings = [
                item for item in (step.get("_schema_warnings") or []) if isinstance(item, dict)
            ]
            if step_warnings:
                warnings.extend(step_warnings)
                per_step_deltas[step_index] = self._schema_warning_penalty(
                    step_warnings,
                    missing_required_penalty=self.beam_schema_missing_required_penalty,
                    invalid_type_penalty=self.beam_schema_invalid_type_penalty,
                )
        return warnings, per_step_deltas

    @staticmethod
    def _apply_schema_warning_delta_to_record(
        record: dict[str, Any], warnings: list[dict[str, Any]], delta: float
    ) -> None:
        record["schema_warnings"] = warnings
        record["schema_warning_count"] = len(warnings)
        record["schema_warning_score_delta"] = delta
        record["score"] = float(record.get("score", 0.0)) + delta
        if "adjusted_score" in record:
            record["adjusted_score"] = float(record.get("adjusted_score", 0.0)) + delta
        contributions = record.get("field_contributions")
        if isinstance(contributions, dict):
            contributions["schema_soft_validation"] = (
                float(contributions.get("schema_soft_validation", 0.0)) + delta
            )
        reason = str(record.get("reason") or "")
        suffix = f"schema_soft_validation:{delta:.3f}"
        record["reason"] = f"{reason} | {suffix}" if reason else suffix

    @staticmethod
    def _apply_schema_warning_deltas_to_steps(
        record: dict[str, Any], per_step_deltas: dict[int, float]
    ) -> None:
        per_step = record.get("per_step")
        if not isinstance(per_step, list):
            return
        for item in per_step:
            if not isinstance(item, dict):
                continue
            step_delta = per_step_deltas.get(int(item.get("step") or 0), 0.0)
            if not step_delta:
                continue
            item["schema_warning_score_delta"] = step_delta
            item["score"] = float(item.get("score", 0.0)) + step_delta
            step_contributions = item.get("contributions")
            if isinstance(step_contributions, dict):
                step_contributions["schema_soft_validation"] = (
                    float(step_contributions.get("schema_soft_validation", 0.0)) + step_delta
                )

    @staticmethod
    def _renormalize_schema_penalized_scores(scored: list[dict[str, Any]]) -> None:
        live = [record for record in scored if not record.get("vetoed", False)]
        if live:
            highest = max(float(record.get("score", 0.0)) for record in live)
            exps = [math.exp(float(record.get("score", 0.0)) - highest) for record in live]
            denom = sum(exps) or 1.0
            for record, exp_value in zip(live, exps, strict=False):
                normalized = exp_value / denom
                record["normalized_score"] = normalized
                if "adjusted_normalized_score" in record:
                    record["adjusted_normalized_score"] = normalized
        for record in scored:
            if record.get("vetoed", False):
                record["normalized_score"] = 0.0
                if "adjusted_normalized_score" in record:
                    record["adjusted_normalized_score"] = 0.0
        scored.sort(
            key=lambda record: (
                bool(record.get("vetoed", False)),
                -float(record.get("score", 0.0)),
            )
        )

    def _apply_schema_warning_penalties(self, scored: list[dict[str, Any]]) -> None:
        """Softly penalize generated beam plans with schema-level argument issues."""
        for record in scored:
            warnings, per_step_deltas = self._schema_warnings_by_step(record.get("plan") or [])
            if not warnings:
                record.setdefault("schema_warnings", [])
                record.setdefault("schema_warning_count", 0)
                record.setdefault("schema_warning_score_delta", 0.0)
                continue
            delta = self._schema_warning_penalty(
                warnings,
                missing_required_penalty=self.beam_schema_missing_required_penalty,
                invalid_type_penalty=self.beam_schema_invalid_type_penalty,
            )
            self._apply_schema_warning_delta_to_record(record, warnings, delta)
            self._apply_schema_warning_deltas_to_steps(record, per_step_deltas)
        self._renormalize_schema_penalized_scores(scored)

    # Canonical-event fields rendered in the imagined-rollout injection, in priority order.
    _INJECTION_STATE_FIELDS = (
        "execution_status",
        "progress_signal",
        "side_effect_type",
        "error_signature",
        "information_sufficiency",
    )
    # Uninformative predicted-state values that are filtered out of the injection.
    _INJECTION_UNINFORMATIVE = (None, "", [], ["none"], "none", "unknown")
    # Predicted tool-output text (decode_plan_observations) longer than this is truncated in the
    # injected guidance -- long enough to be useful, short enough not to dominate the prompt.
    _INJECTION_TOOL_OUTPUT_CHARS = 300

    @classmethod
    def _render_plan_message(
        cls,
        remaining_plan: list[dict[str, Any]],
        *,
        terminal_advice: bool = False,
        terminal_advice_threshold: float = 0.75,
    ) -> str:
        """Render an alternating (predicted action -> predicted resulting state[ -> predicted
        tool output]) trajectory as advisory guidance text, shared by ``beam_plan`` and
        ``hier_latent_cem`` so both modes' injected guidance is visually identical to the agent
        regardless of which planner produced it. The tool-output line only appears for entries
        that carry a decoded ``predicted_tool_output`` (see ``WM_BEAM_PLAN_DECODE_TOOL_OUTPUT``).
        Ends with anti-surrender guidance (the predicted states are model guesses, not ground
        truth)."""
        lines = [
            "World-model imagined rollout for the next steps (predicted action -> predicted "
            "resulting state). Use it as guidance, not ground truth:"
        ]
        threshold = max(0.0, min(1.0, terminal_advice_threshold))
        finish_steps: list[tuple[int, float]] = []
        middle_steps: list[tuple[int, float]] = []
        for plan_i, entry in enumerate(remaining_plan, 1):
            calls_text = (
                ", ".join(
                    f"{c.get('name')}({json.dumps(c.get('arguments', {}), ensure_ascii=False)[:100]})"
                    for c in (entry.get("calls") or [])
                )
                or "(no-op)"
            )
            lines.append(f"  step {plan_i} action: {calls_text}")
            predicted_state = entry.get("predicted_state") or {}
            state_text = ", ".join(
                f"{field}={predicted_state[field]}"
                for field in cls._INJECTION_STATE_FIELDS
                if predicted_state.get(field) not in cls._INJECTION_UNINFORMATIVE
            )
            terminal_prob = entry.get("terminal_probability")
            if isinstance(terminal_prob, (int, float)):
                terminal_text = f"terminal_probability={terminal_prob:.2f}"
                state_text = f"{state_text}, {terminal_text}" if state_text else terminal_text
                if terminal_advice and terminal_prob >= threshold:
                    finish_steps.append((plan_i, float(terminal_prob)))
                elif terminal_advice:
                    middle_steps.append((plan_i, float(terminal_prob)))
            if state_text:
                lines.append(f"  step {plan_i} predicted state: {state_text}")
            tool_output = (entry.get("predicted_tool_output") or "").strip()
            if tool_output:
                truncated = tool_output[: cls._INJECTION_TOOL_OUTPUT_CHARS]
                if len(tool_output) > cls._INJECTION_TOOL_OUTPUT_CHARS:
                    truncated += "...(truncated)"
                lines.append(f"  step {plan_i} predicted tool output: {truncated}")
        if finish_steps:
            step_i, prob = max(finish_steps, key=lambda item: item[1])
            lines.append(
                "Terminal advisory: the world model predicts the task may be ready to finish after "
                f"step {step_i} (P(done)={prob:.2f}). If the actual state confirms every "
                "requirement is satisfied, stop calling tools and provide the final answer."
            )
        elif middle_steps:
            step_i, prob = min(middle_steps, key=lambda item: item[1])
            lines.append(
                "Progress advisory: the world model predicts the task may not be finished "
                f"after step {step_i} (P(done)={prob:.2f}, threshold={threshold:.2f}). Continue "
                "executing or verifying the "
                "required work; do not provide the final answer unless the actual state proves every "
                "requirement is satisfied."
            )
        lines.append(
            "Guidance: these predicted states are the world model's guesses, NOT confirmed results. "
            "Do NOT give a final answer or give up prematurely. A predicted 'finalize' only means the "
            "task MAY be near done -- first verify from the actual state that every requirement is met. "
            "If you are blocked, uncertain, or a step failed, examine the current state and try a "
            "different action or approach before concluding."
        )
        return "\n".join(lines)

    def beam_injection_text(self) -> str:
        """The cached imagined ROLLOUT to show the agent BEFORE it proposes its next action — an
        alternating (predicted action -> predicted resulting state) sequence, so it can follow the
        plan without re-invoking the WM. ``""`` when there is no live plan or pending terminal
        advice."""
        parts: list[str] = []
        if self._beam_pending_terminal_advice:
            advice = self._beam_pending_terminal_advice
            probability = float(advice["terminal_probability"])
            threshold = float(advice["threshold"])
            if advice.get("advice_type") == "middle":
                parts.append(
                    "Progress advisory: the world model predicted that the previous step may not "
                    f"have completed the task (P(done)={probability:.2f}, threshold={threshold:.2f}). "
                    "Continue executing or verifying the required work; do not provide the final "
                    "answer unless the actual state proves every requirement is satisfied."
                )
            else:
                parts.append(
                    "Terminal advisory: the world model predicted that the previous step may have "
                    f"completed the task (P(done)={probability:.2f}, threshold={threshold:.2f}). "
                    "If the actual state confirms every requirement is satisfied, stop calling "
                    "tools and provide the final answer."
                )
            self._beam_pending_terminal_advice = None
        if self._beam_imagined_plan and self._beam_plan_cursor < len(self._beam_imagined_plan):
            parts.append(
                self._render_plan_message(
                    self._beam_imagined_plan[self._beam_plan_cursor :],
                    terminal_advice=self.beam_terminal_advice,
                    terminal_advice_threshold=self.beam_terminal_advice_threshold,
                )
            )
        return "\n\n".join(part for part in parts if part)

    def hier_injection_text(self) -> str:
        """Same advisory injection as :meth:`beam_injection_text`, but the trajectory came from
        the hierarchical latent-action CEM (``_ewm_hier_cem``) instead of a discrete
        LLM-proposed candidate pool. ``""`` when there is no live plan."""
        if not self._hier_imagined_plan or self._hier_plan_cursor >= len(self._hier_imagined_plan):
            return ""
        return self._render_plan_message(self._hier_imagined_plan[self._hier_plan_cursor :])

    @staticmethod
    def _truncate_args(arguments: Any, limit: int = 300) -> Any:
        try:
            text = json.dumps(arguments, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(arguments)
        return arguments if len(text) <= limit else text[:limit] + "...(truncated)"

    @classmethod
    def _plan_preview(cls, imagined_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fix 3: log the ACTUAL injected trajectory (calls + predicted states, args truncated)
        so guidance quality can be audited directly from the run's telemetry."""
        return [
            {
                "calls": [
                    {"name": c.get("name"), "arguments": cls._truncate_args(c.get("arguments"))}
                    for c in (entry.get("calls") or [])
                ],
                **({"id": entry["id"]} if "id" in entry else {}),
                **({"bind": entry["bind"]} if "bind" in entry else {}),
                "score": entry.get("score"),
                "reason": entry.get("reason"),
                "predicted_state": entry.get("predicted_state"),
                "predicted_tool_output": (
                    entry["predicted_tool_output"][:300]
                    if "predicted_tool_output" in entry
                    else None
                ),
                "schema_warnings": entry.get("schema_warnings", []),
                "terminal_probability": entry.get("terminal_probability"),
            }
            for entry in imagined_plan
        ]

    def _terminal_advice_type(self, probability: Any) -> str | None:
        if not self.beam_terminal_advice or not isinstance(probability, (int, float)):
            return None
        prob = float(probability)
        if prob >= self.beam_terminal_advice_threshold:
            return "finish"
        return "middle"

    def _set_beam_terminal_advisory(
        self,
        *,
        probability: Any,
        source: str,
        step_index: int,
    ) -> bool:
        advice_type = self._terminal_advice_type(probability)
        if advice_type is None:
            return False
        prob = float(probability)
        threshold = self.beam_terminal_advice_threshold
        self._beam_pending_terminal_advice = {
            "source": source,
            "step_index": step_index,
            "terminal_probability": prob,
            "threshold": threshold,
            "advice_type": advice_type,
        }
        self._beam_terminal_advice_count += 1
        if advice_type == "finish":
            self._beam_terminal_finish_advice_count += 1
        else:
            self._beam_terminal_middle_advice_count += 1
        return True

    def _beam_plan_critic(
        self,
        *,
        seed_step,
        system_prompt,
        user_prompt,
        imagined_history,
        score_cfg,
    ) -> dict[str, Any]:
        """Should we spend a planning cycle on THIS step? One world-model forward, no LLM.

        The interval trigger re-plans every ``WM_BEAM_MPC_EXECUTE_STEPS`` steps whether or not
        anything is wrong, so the planning cost (an agent call plus the rollout) is paid on every
        step in amortized terms. This asks the cheaper question first: score the action the
        agent ALREADY produced -- that generation is sunk cost -- as a one-step plan, and only
        escalate to full planning when the prediction says the action is bad. Escalation then
        does the expensive thing properly: revise the current action AND look several steps
        ahead. Amortized cost becomes ``critic + fire_rate * planning``, so the fire rate is the
        lever, and it is returned/logged per step (fired or not) so it can be measured.

        Fires when the world model vetoes the action (the score config's own P(failure)/
        P(deleted) safety limits), when P(execution_status=failure) reaches
        ``WM_BEAM_PLAN_CRITIC_FAILURE_PROB``, or when the action looks like it will not advance
        the task (1 - P(progress_signal=positive) reaches ``WM_BEAM_PLAN_CRITIC_STALL_PROB``).
        """
        if not seed_step:
            return {"fires": True, "reason": "no_seed_action", "world_model_calls": 0}
        try:
            scored = self._wm.score_action_plans_canonical_event(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                input_history=imagined_history,
                action_plans=[[seed_step]],
                score_config=score_cfg,
            )
            record = scored[0] if scored else {}
            probs = (record.get("per_step_field_probs") or [{}])[0] or {}
            per_step = record.get("per_step") or []
            score = (
                float(per_step[0].get("score", 0.0))
                if per_step
                else float(record.get("score", 0.0))
            )
            failure_prob = float(probs.get("execution_status", {}).get("failure", 0.0))
            positive_prob = float(probs.get("progress_signal", {}).get("positive", 0.0))
            stall_prob = 1.0 - positive_prob
            terminal_probs = record.get("per_step_terminal_prob") or []
            terminal_prob = (
                terminal_probs[0] if terminal_probs else record.get("terminal_probability")
            )
            terminal_prob = (
                float(terminal_prob) if isinstance(terminal_prob, (int, float)) else None
            )
            vetoed = bool(record.get("vetoed"))
            reasons: list[str] = []
            if vetoed and self.beam_critic_veto_fires:
                flat = "; ".join(
                    ", ".join(entry.get("reasons", []))
                    for entry in (record.get("veto_reasons") or [])
                )
                reasons.append(f"veto({flat})" if flat else "veto")
            if failure_prob >= self.beam_critic_failure_prob:
                reasons.append(
                    f"P(failure)={failure_prob:.2f}>={self.beam_critic_failure_prob:.2f}"
                )
            if stall_prob >= self.beam_critic_stall_prob:
                reasons.append(
                    f"P(no_progress)={stall_prob:.2f}>={self.beam_critic_stall_prob:.2f}"
                )
            terminal_advice_type = self._terminal_advice_type(terminal_prob)
            return {
                "fires": bool(reasons),
                "reason": " | ".join(reasons) if reasons else "action_looks_good",
                "score": score,
                "vetoed": vetoed,
                "failure_prob": failure_prob,
                "stall_prob": stall_prob,
                "terminal_probability": terminal_prob,
                "terminal_advice": terminal_advice_type == "finish",
                "terminal_middle_advice": terminal_advice_type == "middle",
                "terminal_advice_type": terminal_advice_type,
                "predicted_state": record.get("predicted_state"),
                "world_model_calls": 1,
            }
        except Exception as exc:
            logger.warning(
                "ewm_imagined: beam_plan critic failed (%s); escalating to full planning", exc
            )
            return {
                "fires": True,
                "reason": f"critic_error:{type(exc).__name__}",
                "world_model_calls": 0,
            }

    @staticmethod
    def _beam_refinement_feedback(
        scored: list[dict[str, Any]], *, seed_offset: int, top_k: int, next_round: int
    ) -> str:
        """Serialize prior WM-ranked trajectories for the next shared generation prompt."""
        candidates = [
            record
            for record in scored
            if int(record.get("plan_index", record.get("index", -1))) >= seed_offset
            and (record.get("plan") or [])
        ][: max(1, top_k)]
        summaries: list[dict[str, Any]] = []
        for rank, record in enumerate(candidates, 1):
            steps: list[dict[str, Any]] = []
            for step in record.get("plan") or []:
                calls = step.get("tool_calls") if isinstance(step, dict) else []
                normalized = [ewm.normalize_tool_call(call) for call in (calls or [])]
                steps.extend(
                    {
                        "name": call.get("name"),
                        "arguments": call.get("arguments") or {},
                    }
                    for call in normalized
                    if str(call.get("name") or "").strip()
                )
            summaries.append(
                {
                    "rank": rank,
                    "score": round(float(record.get("score", 0.0)), 6),
                    "normalized_score": round(float(record.get("normalized_score", 0.0)), 6),
                    "vetoed": bool(record.get("vetoed", False)),
                    "reason": str(record.get("reason") or "")[:400],
                    "steps": steps,
                    "predicted_states": record.get("per_step_predicted_state") or [],
                }
            )
        return (
            f"\n\nITERATIVE TRAJECTORY REFINEMENT — GENERATION ROUND {next_round}:\n"
            "The world model scored trajectories from the previous generation round. Higher "
            "scores are better; vetoed trajectories should be repaired, not repeated.\n"
            f"PREVIOUS SCORED TRAJECTORIES:\n{json.dumps(summaries, ensure_ascii=False)}\n\n"
            "Generate a new complete trajectory that improves on these examples. Preserve useful "
            "steps, repair low-scoring or vetoed choices, and seek a materially better route. Do "
            "not merely copy a previous trajectory when a feasible improvement exists. Follow "
            "the original output schema exactly and output no analysis."
        )

    def _beam_plan_open_loop(
        self,
        *,
        beam_cfg,
        score_cfg,
        seed_step,
        system_prompt,
        user_prompt,
        imagined_history,
        tool_names,
        tool_schemas,
        state_text,
        path_names,
    ) -> dict[str, Any]:
        """Open-loop beam planning: one shared diversity-menu agent prompt, then ONE batched
        world-model pass for every step of every plan. This replaces
        ``beam_plan_step``'s closed-loop per-depth loop (one candidate-generation call per
        horizon step plus one WM scoring call per depth) with up-front plan generation and a
        single batched scoring pass.

        The agent emits m COMPLETE n-step plans up front (symbolic ``"$stepK.field"`` refs for
        values not yet known); actions never see a predicted state, so nothing serializes.
        :meth:`_ewm_jepa.JepaEwmGenerator.score_action_plans_canonical_event` already rolls every
        plan/step forward in one batched lockstep latent pass -- this just has to feed it m+1
        multi-step plans instead of m one-step plans at a time.

        Returns the same bookkeeping the depth loop produces (plus ``event``), so the confidence
        gate, override synthesis, injection and detail-dict construction in ``beam_plan_step`` are
        unchanged: ``imagined_plan``, ``per_depth_scores``, ``per_depth_normalized``,
        ``first_num_candidates``, ``first_best``, ``beam_calls``,
        ``first_recommend_calls_candidate``, ``first_recommend_reason_candidate``.
        """
        from ejepa_wm.backends._ewm_beam_plan import (
            build_single_plan_prompt,
            parse_single_plan,
        )

        empty = {
            "imagined_plan": [],
            "per_depth_scores": [],
            "per_depth_normalized": [],
            "first_num_candidates": 0,
            "first_best": {},
            "beam_calls": 1,
            "first_recommend_calls_candidate": None,
            "first_recommend_reason_candidate": "no_candidates",
            "beam_refinement_rounds_requested": self.beam_refinement_rounds,
            "beam_refinement_rounds_completed": 0,
            "beam_refinement_score_passes": 0,
            "beam_refinement_rounds": [],
        }

        # One shared prompt carries the diversity menu, then sample_many requests m independent
        # completions. A configured dedicated action sampler (for example DiffusionGemma) is used
        # only here; the policy agent still chooses real actions and JEPA still scores plans.
        # Backends exposing generate_samples use one OpenAI/vLLM n=k request. The temperature
        # ladder remains the explicit slower path because n=k accepts only one temperature.
        base_prompt = build_single_plan_prompt(
            system_prompt,
            state_text,
            beam_cfg,
            tool_names=tool_names or None,
            ssot_diversity=self.beam_ssot_diversity,
        )
        temperatures = (
            ewm.build_temperature_ladder(
                beam_cfg.num_candidates,
                self.temperature,
                self.sample_temperature_ladder_max,
            )
            if self.sample_temperature_ladder
            else None
        )
        candidate_generator = self._beam_action_sampler or self._agent
        response_format = {"type": "json_object"} if self.beam_ssot_diversity else None
        candidate_plans: list[list[dict[str, Any]]] = []
        seen_plan_signatures: set[tuple] = set()
        candidate_stats = {
            "requested": beam_cfg.num_candidates * self.beam_refinement_rounds,
            "raw_samples": 0,
            "parsed": 0,
            "wrapped": 0,
            "duplicates": 0,
            "invalid_tools": 0,
            "invalid_arguments": 0,
            "broken_references": 0,
            "schema_warned_candidates": 0,
            "schema_warnings": 0,
            "missing_required_arguments": 0,
            "invalid_argument_types": 0,
            "accepted": 0,
            "generator": (
                "dedicated_action_sampler"
                if self._beam_action_sampler is not None
                else "policy_agent"
            ),
            "generation_seconds": 0.0,
            "requests_issued": 0,
            "ssot_diversity": self.beam_ssot_diversity,
            "refinement_rounds_requested": self.beam_refinement_rounds,
            "rounds": [],
        }
        allowed_tool_names = set(tool_names or [])
        refinement_feedback = ""
        agent_requests = 0
        refinement_score_passes = 0
        for round_index in range(self.beam_refinement_rounds):
            prompt = base_prompt + refinement_feedback
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Output ONLY the JSON value requested by the user. No prose or markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            generation_started = time.perf_counter()
            raw_samples, round_requests = ewm.sample_many(
                candidate_generator,
                messages,
                temperature=self.temperature,
                num_samples=beam_cfg.num_candidates,
                temperatures=temperatures,
                response_format=response_format,
            )
            generation_seconds = time.perf_counter() - generation_started
            self._agent_call_count += round_requests
            agent_requests += round_requests
            round_stats = {
                "round": round_index + 1,
                "raw_samples": len(raw_samples),
                "accepted_new": 0,
                "duplicates": 0,
                "schema_warned_candidates": 0,
                "schema_warnings": 0,
                "missing_required_arguments": 0,
                "invalid_argument_types": 0,
                "generation_seconds": generation_seconds,
                "requests_issued": round_requests,
                "used_prior_scores": bool(refinement_feedback),
            }
            candidate_stats["raw_samples"] += len(raw_samples)
            candidate_stats["generation_seconds"] += generation_seconds
            candidate_stats["requests_issued"] += round_requests
            for raw in raw_samples:
                raw_text = raw if isinstance(raw, str) else str(getattr(raw, "content", raw) or "")
                parsed = parse_single_plan(raw_text, beam_cfg)
                if not parsed:
                    continue
                candidate_stats["parsed"] += 1
                steps = [
                    wrapped for wrapped in (self._wrap_step(step) for step in parsed) if wrapped
                ]
                if not steps:
                    continue
                candidate_stats["wrapped"] += 1
                validation = self._plan_validation_detail(steps, allowed_tool_names, tool_schemas)
                validation_error = validation.get("hard_error")
                if validation_error:
                    candidate_stats[
                        "invalid_tools"
                        if validation_error == "invalid_tool"
                        else "broken_references"
                        if validation_error == "broken_reference"
                        else "invalid_arguments"
                    ] += 1
                    continue
                schema_warnings = [
                    warning
                    for warning in (validation.get("schema_warnings") or [])
                    if isinstance(warning, dict)
                ]
                if schema_warnings:
                    candidate_stats["schema_warned_candidates"] += 1
                    candidate_stats["schema_warnings"] += len(schema_warnings)
                    round_stats["schema_warned_candidates"] += 1
                    round_stats["schema_warnings"] += len(schema_warnings)
                    for warning in schema_warnings:
                        reason = warning.get("reason")
                        if reason == "missing_required_argument":
                            candidate_stats["missing_required_arguments"] += 1
                            round_stats["missing_required_arguments"] += 1
                        elif reason == "invalid_argument_type":
                            candidate_stats["invalid_argument_types"] += 1
                            round_stats["invalid_argument_types"] += 1
                        step_index = int(warning.get("step") or 0)
                        if 1 <= step_index <= len(steps):
                            steps[step_index - 1].setdefault("_schema_warnings", []).append(warning)
                # Deduplicate complete plans across every refinement round. Concrete scalar
                # values remain collapsed so guessed identifiers do not create fake novelty.
                signature = self._plan_signature(steps)
                if signature in seen_plan_signatures:
                    candidate_stats["duplicates"] += 1
                    round_stats["duplicates"] += 1
                    continue
                seen_plan_signatures.add(signature)
                candidate_plans.append(steps)
                round_stats["accepted_new"] += 1
            candidate_stats["rounds"].append(round_stats)

            if round_index + 1 < self.beam_refinement_rounds and candidate_plans:
                intermediate_plans = list(candidate_plans)
                seed_offset = 0
                if seed_step:
                    intermediate_plans = [[seed_step], *intermediate_plans]
                    seed_offset = 1
                prior_scored = self._wm.score_action_plans_canonical_event(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    input_history=imagined_history,
                    action_plans=intermediate_plans,
                    score_config=score_cfg,
                )
                self._apply_schema_warning_penalties(prior_scored)
                refinement_score_passes += 1
                refinement_feedback = self._beam_refinement_feedback(
                    prior_scored,
                    seed_offset=seed_offset,
                    top_k=self.beam_refinement_top_k,
                    next_round=round_index + 2,
                )
        candidate_stats["accepted"] = len(candidate_plans)
        empty["open_loop_candidate_stats"] = candidate_stats
        empty["beam_calls"] = agent_requests
        empty["beam_refinement_rounds_completed"] = len(candidate_stats["rounds"])
        empty["beam_refinement_score_passes"] = refinement_score_passes
        empty["beam_refinement_rounds"] = candidate_stats["rounds"]
        if not candidate_plans:
            return {**empty, "event": "GYM_BEAM_PLAN_OPEN_LOOP_NO_CANDIDATES"}

        # The seed (the agent's own action) is scored as a one-step plan at index 0, exactly as
        # depth 0 does in closed-loop, so the override margin gate below compares like with like.
        # It is never selected as the injected trajectory: a one-step plan cannot reach the
        # horizon, and "keep the agent's action" is expressed by declining to override, not by
        # injecting a one-step plan.
        seed_offset = 0
        if seed_step:
            candidate_plans = [[seed_step]] + candidate_plans
            seed_offset = 1
        scored = self._wm.score_action_plans_canonical_event(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_history=imagined_history,
            action_plans=candidate_plans,
            score_config=score_cfg,
        )
        self._apply_schema_warning_penalties(scored)
        refinement_score_passes += 1
        by_index = {
            int(record.get("plan_index", record.get("index", -1))): record for record in scored
        }
        seed_record = by_index.get(0) if seed_offset else None

        repeat_penalty_scale = 0.75 * self.beam_diversity_multiplier

        candidate_score_diagnostics = self._candidate_score_diagnostics(
            scored, path_names, repeat_penalty_scale
        )
        for candidate in candidate_score_diagnostics:
            candidate["depth"] = "open_loop"
        self._log_candidate_score_diagnostics("open_loop", candidate_score_diagnostics)

        def adjusted(record: dict[str, Any]) -> float:
            plan = record.get("plan") or []
            names = self._step_tool_names(plan[0]) if plan else tuple()
            return float(record.get("score", 0.0)) - repeat_penalty_scale * path_names.count(names)

        def first_step_adjusted(record: dict[str, Any]) -> float:
            """Depth-0 score, penalized -- comparable across plans of different lengths, unlike
            the full-horizon trajectory score."""
            per_step = record.get("per_step") or []
            plan = record.get("plan") or []
            names = self._step_tool_names(plan[0]) if plan else tuple()
            head = float(per_step[0].get("score", 0.0)) if per_step else 0.0
            return head - repeat_penalty_scale * path_names.count(names)

        generated = [
            record
            for record in scored
            if int(record.get("plan_index", -1)) >= seed_offset
            and not record.get("vetoed", False)
            and (record.get("plan") or [])
        ]
        if not generated:
            return {**empty, "event": "GYM_BEAM_PLAN_OPEN_LOOP_VETOED"}
        winner = max(generated, key=adjusted)

        winner_plan = winner.get("plan") or []
        per_step = winner.get("per_step") or []
        per_step_states = winner.get("per_step_predicted_state") or []
        per_step_terminal = winner.get("per_step_terminal_prob") or []
        imagined_plan: list[dict[str, Any]] = []
        per_depth_scores: list[float] = []
        for depth, step in enumerate(winner_plan):
            calls = [
                ewm.normalize_tool_call(call)
                for call in (step.get("tool_calls", []) if isinstance(step, dict) else [])
            ]
            calls = [c for c in calls if str(c.get("name", "")).strip()]
            step_score = float(per_step[depth].get("score", 0.0)) if depth < len(per_step) else 0.0
            per_depth_scores.append(step_score)
            per_step_tool_outputs = winner.get("per_step_predicted_tool_output") or []
            imagined_entry = {
                "calls": calls,
                "score": step_score,
                "reason": winner.get("reason"),
                "predicted_state": (
                    per_step_states[depth]
                    if depth < len(per_step_states)
                    else winner.get("predicted_state")
                ),
                "terminal_probability": (
                    float(per_step_terminal[depth]) if depth < len(per_step_terminal) else None
                ),
            }
            for key in ("id", "bind"):
                if isinstance(step, dict) and key in step:
                    imagined_entry[key] = step[key]
            schema_warnings = (step.get("_schema_warnings") or []) if isinstance(step, dict) else []
            if schema_warnings:
                imagined_entry["schema_warnings"] = schema_warnings
            if depth < len(per_step_tool_outputs):
                imagined_entry["predicted_tool_output"] = per_step_tool_outputs[depth]
            imagined_plan.append(imagined_entry)

        # One softmax over the candidate PLANS replaces closed-loop's per-depth softmaxes. The
        # downstream gate averages per_depth_normalized, so repeating the winner's trajectory-
        # level normalized score once per depth keeps that average on the same scale
        # (probability mass among ~m candidates) that gate_flat_score_ratio is tuned for.
        winner_normalized = float(winner.get("normalized_score") or 0.0)
        per_depth_normalized = [winner_normalized] * max(1, len(imagined_plan))

        # Override margin, decided on depth-0 scores exactly as closed-loop's depth-0 branch does.
        first_calls = list(imagined_plan[0]["calls"]) if imagined_plan else []
        if not seed_record:
            recommend_calls = first_calls or None
            recommend_reason = "beam_best_beats_seed" if first_calls else "no_candidates"
        else:
            margin = max(0.0, self.beam_score_margin)
            if first_step_adjusted(winner) - first_step_adjusted(seed_record) > margin:
                recommend_calls = first_calls or None
                recommend_reason = "beam_best_beats_seed"
            else:
                recommend_calls = None
                recommend_reason = "kept_baseline_below_margin"

        return {
            "imagined_plan": imagined_plan,
            "per_depth_scores": per_depth_scores,
            "per_depth_normalized": per_depth_normalized,
            "first_num_candidates": len(candidate_plans),
            "first_best": winner,
            "beam_calls": agent_requests,
            "first_recommend_calls_candidate": recommend_calls,
            "first_recommend_reason_candidate": recommend_reason,
            "event": "GYM_BEAM_PLAN",
            "open_loop_candidate_stats": candidate_stats,
            "candidate_score_diagnostics": candidate_score_diagnostics,
            "beam_refinement_rounds_requested": self.beam_refinement_rounds,
            "beam_refinement_rounds_completed": len(candidate_stats["rounds"]),
            "beam_refinement_score_passes": refinement_score_passes,
            "beam_refinement_rounds": candidate_stats["rounds"],
        }

    @staticmethod
    def _imagined_entry_to_step(entry: dict[str, Any]) -> dict[str, Any]:
        step: dict[str, Any] = {"tool_calls": ewm.to_openai_tool_calls(entry.get("calls") or [])}
        for key in ("id", "bind"):
            if key in entry:
                step[key] = entry[key]
        return step

    def _resolve_beam_plan_entry(
        self,
        entry: dict[str, Any],
        *,
        current_history: list[dict[str, Any]],
        tool_schemas: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from ejepa_wm.backends._ewm_symbolic_plan import resolve_symbolic_references

        plan_prefix = [
            self._imagined_entry_to_step(prefix_entry)
            for prefix_entry in self._beam_imagined_plan[: self._beam_plan_cursor]
        ]
        observed_prefix = current_history[
            self._beam_plan_history_offset : self._beam_plan_history_offset + self._beam_plan_cursor
        ]
        observations = [
            item.get("observation") for item in observed_prefix if isinstance(item, dict)
        ]
        resolved, detail = resolve_symbolic_references(
            entry.get("calls") or [],
            observations=observations,
            plan_steps=plan_prefix,
        )
        calls = [
            call
            for call in (ewm.normalize_tool_call(call) for call in (resolved or []))
            if str(call.get("name", "")).strip()
        ]
        schema_errors: list[dict[str, Any]] = []
        for call in calls:
            arguments = call.get("arguments", {})
            if not isinstance(arguments, dict):
                schema_errors.append({"tool": call.get("name"), "reason": "invalid_arguments"})
                continue
            error = self._argument_schema_error(
                str(call.get("name") or ""), arguments, tool_schemas
            )
            if error:
                schema_errors.append({"tool": call.get("name"), "reason": error})
        if schema_errors:
            detail.setdefault("unresolved", []).extend(schema_errors)
        detail["resolved_calls"] = calls
        detail["plan_cursor"] = self._beam_plan_cursor
        detail["observed_prefix_len"] = len(observations)
        return calls, detail

    @classmethod
    def _calls_signature(cls, calls: list[dict[str, Any]]) -> str:
        normalized = [ewm.normalize_tool_call(call) for call in (calls or [])]
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)

    @classmethod
    def _calls_have_symbolic_references(cls, calls: list[dict[str, Any]]) -> bool:
        from ejepa_wm.backends._ewm_symbolic_plan import has_symbolic_references

        return has_symbolic_references(calls)

    def _beam_revision_choose(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        state_text: str,
        agent_calls: list[dict[str, Any]],
        planned_calls: list[dict[str, Any]],
        remaining_plan: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Use the acting policy as a binary arbiter for one MPC horizon step."""
        agent_norm = [ewm.normalize_tool_call(call) for call in (agent_calls or [])]
        plan_norm = [ewm.normalize_tool_call(call) for call in (planned_calls or [])]
        detail: dict[str, Any] = {
            "beam_revision_enabled": True,
            "beam_revision_attempted": False,
            "beam_revision_choice": "agent",
            "beam_revision_plan_selected": False,
            "beam_revision_agent_calls": 0,
        }
        if not plan_norm:
            detail["beam_revision_reason"] = "no_planned_action"
            return agent_norm, detail
        if self._calls_signature(agent_norm) == self._calls_signature(plan_norm):
            detail.update(
                {
                    "beam_revision_choice": "plan",
                    "beam_revision_plan_selected": True,
                    "beam_revision_reason": "agent_plan_agreement",
                }
            )
            return agent_norm, detail

        preview = self._plan_preview(remaining_plan)
        prompt = (
            "Choose the next action for the real agent. Compare exactly two options:\n"
            f"OPTION_AGENT={json.dumps(agent_norm, ensure_ascii=False)}\n"
            f"OPTION_PLAN={json.dumps(plan_norm, ensure_ascii=False)}\n"
            f"REMAINING_IMAGINED_PLAN={json.dumps(preview, ensure_ascii=False)}\n\n"
            f"Task: {user_prompt}\n"
            f"Current state: {state_text}\n\n"
            "The imagined plan is uncertain guidance, not an observed result. Select OPTION_PLAN "
            "only when it is the better executable next step toward unmet task requirements; "
            'otherwise select OPTION_AGENT. Return only JSON: {"choice":"agent"} or '
            '{"choice":"plan"}. Do not propose a third action.'
        )
        raw = ""
        try:
            raw = self._agent.generate_from_messages(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.beam_revision_temperature,
            )
            self._agent_call_count += 1
            detail["beam_revision_attempted"] = True
            detail["beam_revision_agent_calls"] = 1
            parsed = ewm.parse_jsonish(raw)
            choice = (
                str(parsed.get("choice", "")).strip().lower() if isinstance(parsed, dict) else ""
            )
        except Exception as exc:
            detail["beam_revision_reason"] = f"revision_error:{type(exc).__name__}"
            detail["beam_revision_raw"] = raw[:500]
            return agent_norm, detail

        detail["beam_revision_raw"] = raw[:500]
        if choice != "plan":
            detail["beam_revision_reason"] = (
                "policy_selected_agent" if choice == "agent" else "invalid_choice_fallback"
            )
            return agent_norm, detail

        if self._calls_have_symbolic_references(plan_norm):
            same_tools = self._step_tool_names(
                self._wrap_step({"tool_calls": agent_norm}) or {}
            ) == (self._step_tool_names(self._wrap_step({"tool_calls": plan_norm}) or {}))
            if same_tools and agent_norm and not self._calls_have_symbolic_references(agent_norm):
                detail.update(
                    {
                        "beam_revision_choice": "plan",
                        "beam_revision_plan_selected": True,
                        "beam_revision_reason": "plan_selected_agent_grounded_arguments",
                    }
                )
                return agent_norm, detail
            detail["beam_revision_reason"] = "unresolved_plan_reference_fallback"
            return agent_norm, detail

        detail.update(
            {
                "beam_revision_choice": "plan",
                "beam_revision_plan_selected": True,
                "beam_revision_reason": "policy_selected_plan",
            }
        )
        return plan_norm, detail

    def beam_plan_step(
        self, conversation_flow, *, seed_calls, user_query: str = ""
    ) -> AdviseResult:
        """One MPC decision. ``seed_calls`` is the agent's baseline next action (list of
        ``{"name", "arguments"}``). Returns an :class:`AdviseResult` whose ``detail`` carries:

        * ``calls`` — the calls to execute this step. **Advisory by default**: this is the agent's
          baseline; the beam only overrides it when ``WM_BEAM_PLAN_HARD_OVERRIDE`` is set AND the
          beam is *confident* AND beats the seed by the margin.
        * ``replanned`` / ``override_applied`` / ``override_reason`` / ``event`` — decision telemetry;
        * ``beam_confident`` / ``confidence_gate`` / ``advisory`` / ``hard_override`` / ``injected`` /
          ``recommended_calls`` / ``agent_calls`` / ``imagined_plan`` — audit fields: the beam only
          injects its imagined trajectory as guidance when confident.

        Confidence is gated on the FULL generated trajectory's aggregate normalized score, not the
        isolated first-step candidate spread: a flat-looking first step can still sit on a
        trajectory that clearly separates from alternatives once the horizon plays out, and
        per-step gating would throw that information away unseen. There is therefore no per-step
        early-exit — the full horizon always runs during a re-plan.

        MPC cooldown: every re-plan attempt (confident or not) starts a ``WM_BEAM_MPC_EXECUTE_STEPS``
        cooldown window before the next one is allowed. A rejected re-plan (flat/similar trajectory
        scores) is near-certain to look just as flat one step later — nothing about the environment
        changed drastically in a single agent step — so immediately re-invoking the world model
        there just re-spends horizon LLM calls on the same answer; instead the agent decides on its
        own for the rest of the cooldown (``GYM_BEAM_PLAN_COOLDOWN``), same cadence as if a
        confident plan HAD been injected. Only "ran out of cached plan before the cooldown elapsed"
        (execute_steps > horizon) forces an early re-plan.

        On any internal failure it degrades to the baseline (``calls == seed_calls``) so
        planning never crashes the run. ``text`` is unused (injection is via
        :meth:`beam_injection_text`)."""
        from ejepa_wm.backends._ewm_beam_plan import (
            BeamPlanConfig,
            build_step_candidates_prompt,
            parse_action_candidates,
        )
        from ejepa_wm.backends._ewm_canonical_event_scoring import CanonicalEventScoreConfig

        seed_norm = [ewm.normalize_tool_call(c) for c in (seed_calls or [])]
        seed_norm = [c for c in seed_norm if str(c.get("name", "")).strip()]
        base_detail = {"backend": "ewm_imagined", "strategy": "beam_plan", "calls": seed_norm}

        if not self.supports_beam_plan():
            return AdviseResult(
                text="", detail={**base_detail, "event": "GYM_BEAM_PLAN_UNSUPPORTED"}
            )
        if not seed_norm:
            return AdviseResult(text="", detail={**base_detail, "event": "GYM_BEAM_PLAN_NO_SEED"})

        system_prompt = self._prompt_from_flow(conversation_flow, "system_message")
        if self.qwen_agentworld:
            system_prompt = self._react_system_prompt_from_flow(conversation_flow) or system_prompt
        user_prompt = user_query or self._prompt_from_flow(conversation_flow, "user_message")
        tool_names = self._tool_names_from_flow(conversation_flow)
        tool_schemas = self._tool_schema_map_from_flow(conversation_flow)
        execute_steps = self.beam_execute_steps

        # Built before the re-plan decision because the critic trigger scores the seed action,
        # and reused by the planner after it either way.
        seed_step = self._wrap_step({"tool_calls": seed_norm})
        # Up-weight progress and make non-positive progress a penalty (a stagnant/no-progress
        # step -- e.g. a redundant read -- should score lower).
        score_cfg = CanonicalEventScoreConfig()
        score_cfg.weights["progress_signal"] = 1.2
        score_cfg.utilities["progress_signal"]["neutral"] = -0.3
        score_cfg.utilities["progress_signal"]["negative"] = -1.0
        score_cfg.read_saturation_threshold = self.beam_read_saturation_threshold
        score_cfg.read_after_saturation_penalty = self.beam_read_penalty
        score_cfg.read_after_progress_scale = self.beam_read_after_progress_scale
        score_cfg.first_step_score_weight = self.beam_first_step_weight
        score_cfg.required_action_coverage_bonus = self.beam_required_action_coverage_bonus
        score_cfg.first_required_action_bonus = self.beam_first_required_action_bonus
        imagined_history = self._input_history_from_flow(conversation_flow)
        current_state_text = json.dumps(
            imagined_history[-self.state_history_size :]
            if self.state_history_size
            else imagined_history,
            ensure_ascii=False,
            default=str,
        )[:8000]
        critic_result: dict[str, Any] | None = None

        plan_exhausted = bool(self._beam_imagined_plan) and self._beam_plan_cursor >= len(
            self._beam_imagined_plan
        )
        if self.beam_plan_trigger == "critic":
            # Ask the world model whether THIS action needs help (one forward, no LLM) instead
            # of re-planning on a fixed cadence.
            critic_result = self._beam_plan_critic(
                seed_step=seed_step,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                imagined_history=imagined_history,
                score_cfg=score_cfg,
            )
            self._beam_critic_checks += 1
            quiet_exceeded = (
                self.beam_critic_max_quiet_steps > 0
                and self._beam_quiet_steps >= self.beam_critic_max_quiet_steps
            )
            # ``plan_exhausted`` is not a cadence -- planning stays event-driven. It only covers
            # "the cached plan ran out": the non-firing branch below indexes
            # ``_beam_imagined_plan[_beam_plan_cursor]`` whenever a plan exists, so a critic that
            # stays quiet for more steps than the plan is long (reachable whenever
            # WM_BEAM_PLAN_CRITIC_MAX_QUIET_STEPS > WM_BEAM_PLAN_HORIZON) would index past its
            # end and lose the step to the caller's baseline fallback.
            need_replan = bool(critic_result["fires"]) or quiet_exceeded or plan_exhausted
            if need_replan:
                self._beam_critic_fires += 1
                self._beam_quiet_steps = 0
            else:
                self._beam_quiet_steps += 1
        else:
            need_replan = (
                not self._beam_has_planned
                or self._beam_plan_cursor >= execute_steps
                or plan_exhausted
            )
        critic_telemetry = {
            "trigger": self.beam_plan_trigger,
            "critic": critic_result,
            "critic_checks": self._beam_critic_checks,
            "critic_fires": self._beam_critic_fires,
            # The lever the critic trigger trades against the interval trigger's fixed
            # 1/WM_BEAM_MPC_EXECUTE_STEPS cadence: below that, the critic is the cheaper
            # trigger; above it, it re-plans more often than the interval it replaces.
            "critic_fire_rate": (
                self._beam_critic_fires / self._beam_critic_checks
                if self._beam_critic_checks
                else None
            ),
            "critic_break_even_fire_rate": 1.0 / execute_steps if execute_steps else None,
        }
        if not need_replan and self._beam_imagined_plan:
            terminal_advice_set = self._set_beam_terminal_advisory(
                probability=(critic_result or {}).get("terminal_probability"),
                source="critic_seed_action",
                step_index=self._beam_plan_cursor,
            )
            revision_detail: dict[str, Any] = {"beam_revision_enabled": self.beam_revision}
            binding_detail: dict[str, Any] = {}
            selected_calls = seed_norm
            planned_entry = self._beam_imagined_plan[self._beam_plan_cursor]
            planned_calls, binding_detail = self._resolve_beam_plan_entry(
                planned_entry,
                current_history=imagined_history,
                tool_schemas=tool_schemas,
            )
            if binding_detail.get("unresolved"):
                self._beam_imagined_plan = []
                self._beam_plan_cursor = 0
                self._beam_plan_history_offset = 0
                revision_detail["beam_revision_reason"] = "unresolved_plan_reference_fallback"
            elif self.beam_hard_override:
                selected_calls = planned_calls
                self._beam_plan_cursor += 1
            elif self.beam_revision:
                selected_calls, revision_detail = self._beam_revision_choose(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    state_text=current_state_text,
                    agent_calls=seed_norm,
                    planned_calls=planned_calls,
                    remaining_plan=self._beam_imagined_plan[self._beam_plan_cursor :],
                )
                if revision_detail.get("beam_revision_plan_selected"):
                    self._beam_plan_cursor += 1
                else:
                    self._beam_imagined_plan = []
                    self._beam_plan_cursor = 0
                    self._beam_plan_history_offset = 0
            else:
                self._beam_plan_cursor += 1
            override_applied = self._calls_signature(selected_calls) != self._calls_signature(
                seed_norm
            )
            return AdviseResult(
                text="",
                detail={
                    **base_detail,
                    "event": "GYM_BEAM_PLAN_FOLLOW",
                    "replanned": False,
                    "calls": selected_calls,
                    "agent_calls": seed_norm,
                    "override_applied": override_applied,
                    "override_reason": (
                        "beam_revision_selected_plan"
                        if override_applied
                        else "beam_revision_kept_agent"
                    ),
                    "plan_cursor": self._beam_plan_cursor,
                    "plan_len": len(self._beam_imagined_plan),
                    "terminal_advice_set": terminal_advice_set,
                    "symbolic_binding": binding_detail,
                    **revision_detail,
                    **critic_telemetry,
                },
            )
        if not need_replan:
            # Last re-plan was rejected (below the trajectory-confidence gate), or (critic
            # trigger) the critic didn't fire -- ride out the rest of this cooldown window with
            # no world-model call and no injected guidance.
            terminal_advice_set = self._set_beam_terminal_advisory(
                probability=(critic_result or {}).get("terminal_probability"),
                source="critic_seed_action",
                step_index=self._beam_plan_cursor,
            )
            self._beam_plan_cursor += 1
            return AdviseResult(
                text="",
                detail={
                    **base_detail,
                    "event": "GYM_BEAM_PLAN_COOLDOWN",
                    "replanned": False,
                    "override_applied": False,
                    "plan_cursor": self._beam_plan_cursor,
                    "execute_steps": execute_steps,
                    "terminal_advice_set": terminal_advice_set,
                    **critic_telemetry,
                },
            )

        try:
            beam_cfg = BeamPlanConfig(
                num_candidates=self.beam_samples, horizon=self.beam_horizon, top_k=self.top_k
            )
            imagined_history_start = list(
                imagined_history
            )  # snapshot for the post-hoc observation decode below
            repeat_penalty_scale = 0.75 * self.beam_diversity_multiplier
            override_margin = self.beam_score_margin
            path_names = list(self._beam_recent_tool_names)
            imagined_plan: list[dict[str, Any]] = []
            first_recommend_calls_candidate: list[dict[str, Any]] | None = None
            first_recommend_reason_candidate = "no_candidates"
            first_num_candidates = 0
            first_best: dict[str, Any] = {}
            beam_calls = 0
            per_depth_scores: list[float] = []
            per_depth_normalized: list[float] = []
            event = "GYM_BEAM_PLAN"
            open_loop_candidate_stats: dict[str, Any] | None = None
            candidate_score_diagnostics: list[dict[str, Any]] = []
            refinement_rounds_completed = 0
            refinement_score_passes = 0
            refinement_round_telemetry: list[dict[str, Any]] = []

            def _adjusted_rank(records):
                ranked = []
                for record in records:
                    plan = record.get("plan") or []
                    names = self._step_tool_names(plan[0]) if plan else tuple()
                    penalty = repeat_penalty_scale * path_names.count(names)
                    ranked.append((float(record.get("score", 0.0)) - penalty, record, names))
                ranked.sort(key=lambda item: -item[0])
                return ranked

            if self.imagined_rollout_mode == "open_loop":
                # One shared diversity-menu prompt sampled into candidate plans + ONE batched
                # world-model scoring pass, instead of one candidate-generation call and one
                # world-model call per horizon depth. Produces the same bookkeeping the depth
                # loop below does.
                state_text = json.dumps(
                    imagined_history[-self.state_history_size :]
                    if self.state_history_size
                    else imagined_history,
                    ensure_ascii=False,
                    default=str,
                )[:8000]
                _open = self._beam_plan_open_loop(
                    beam_cfg=beam_cfg,
                    score_cfg=score_cfg,
                    seed_step=seed_step,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    imagined_history=imagined_history,
                    tool_names=tool_names,
                    tool_schemas=tool_schemas,
                    state_text=state_text,
                    path_names=path_names,
                )
                imagined_plan = _open["imagined_plan"]
                per_depth_scores = _open["per_depth_scores"]
                per_depth_normalized = _open["per_depth_normalized"]
                first_num_candidates = _open["first_num_candidates"]
                first_best = _open["first_best"]
                beam_calls = _open["beam_calls"]
                first_recommend_calls_candidate = _open["first_recommend_calls_candidate"]
                first_recommend_reason_candidate = _open["first_recommend_reason_candidate"]
                event = _open["event"]
                open_loop_candidate_stats = _open.get("open_loop_candidate_stats")
                candidate_score_diagnostics = _open.get("candidate_score_diagnostics", [])
                refinement_rounds_completed = int(
                    _open.get("beam_refinement_rounds_completed") or 0
                )
                refinement_score_passes = int(_open.get("beam_refinement_score_passes") or 0)
                refinement_round_telemetry = list(_open.get("beam_refinement_rounds") or [])
            else:
                for depth in range(max(1, self.beam_horizon)):
                    state_text = json.dumps(
                        imagined_history[-self.state_history_size :]
                        if self.state_history_size
                        else imagined_history,
                        ensure_ascii=False,
                        default=str,
                    )[:8000]
                    prompt = build_step_candidates_prompt(
                        system_prompt, state_text, beam_cfg, depth, tool_names=tool_names or None
                    )
                    self._agent_call_count += 1
                    beam_calls += 1
                    raw = self._agent.generate_from_messages(
                        [
                            {
                                "role": "system",
                                "content": "You output ONLY a JSON array of tool-call actions. No prose, no markdown.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.temperature,
                    )
                    raw_text = (
                        raw if isinstance(raw, str) else str(getattr(raw, "content", raw) or "")
                    )
                    actions = parse_action_candidates(raw_text, beam_cfg)
                    candidate_steps = [w for w in (self._wrap_step(a) for a in actions) if w]
                    if (
                        depth == 0 and seed_step
                    ):  # always include the baseline action as a candidate
                        candidate_steps = [seed_step] + candidate_steps
                    if not candidate_steps:
                        if depth == 0:
                            event = "GYM_BEAM_PLAN_NO_CANDIDATES"
                        break
                    candidate_plans = [[step] for step in candidate_steps]
                    scored = self._wm.score_action_plans_canonical_event(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        input_history=imagined_history,
                        action_plans=candidate_plans,
                        score_config=score_cfg,
                    )
                    ranked = _adjusted_rank(scored)
                    depth_diagnostics = self._candidate_score_diagnostics(
                        scored, path_names, repeat_penalty_scale
                    )
                    for candidate in depth_diagnostics:
                        candidate["depth"] = depth
                    self._log_candidate_score_diagnostics(depth, depth_diagnostics)
                    candidate_score_diagnostics.extend(depth_diagnostics)

                    best_entry = next(
                        (
                            entry
                            for entry in ranked
                            if not entry[1].get("vetoed", False) and entry[1].get("plan")
                        ),
                        None,
                    )
                    if best_entry is None:
                        if depth == 0:
                            event = "GYM_BEAM_PLAN_VETOED"
                        break
                    best_adj_score, best, best_names = best_entry
                    best_step = best["plan"][0]
                    if depth == 0:
                        first_num_candidates = len(candidate_plans)
                        first_best = best
                        # Margin gate: the top candidate must beat the seed (candidate 0) by more than
                        # override_margin before we recommend replacing the agent's choice. This
                        # comparison can only be made at depth 0 (the seed is only defined for the
                        # CURRENT immediate action) -- but whether we ACT on it is decided after the
                        # full trajectory is built, below (the trajectory-level confidence gate).
                        seed_entry = next(
                            (entry for entry in ranked if entry[1].get("plan_index") == 0), None
                        )
                        seed_adj_score = seed_entry[0] if seed_entry else None
                        beats_seed = best.get("plan_index") != 0 and (
                            seed_adj_score is None
                            or best_adj_score - seed_adj_score > override_margin
                        )
                        if beats_seed:
                            new_calls = [
                                ewm.normalize_tool_call(call)
                                for call in (
                                    best_step.get("tool_calls", [])
                                    if isinstance(best_step, dict)
                                    else []
                                )
                            ]
                            first_recommend_calls_candidate = [
                                c for c in new_calls if str(c.get("name", "")).strip()
                            ] or None
                            first_recommend_reason_candidate = "beam_best_beats_seed"
                            recommend_step = best_step
                        else:
                            first_recommend_calls_candidate = None
                            first_recommend_reason_candidate = (
                                "seed_is_best"
                                if best.get("plan_index") == 0
                                else "kept_baseline_below_margin"
                            )
                            recommend_step = seed_step if isinstance(seed_step, dict) else best_step
                        best_step = recommend_step  # lookahead follows the recommended trajectory
                    path_names.append(self._step_tool_names(best_step))
                    chosen_calls = [
                        ewm.normalize_tool_call(call)
                        for call in (
                            best_step.get("tool_calls", []) if isinstance(best_step, dict) else []
                        )
                    ]
                    chosen_calls = [c for c in chosen_calls if str(c.get("name", "")).strip()]
                    per_depth_scores.append(float(best.get("score") or 0.0))
                    per_depth_normalized.append(float(best.get("normalized_score") or 0.0))
                    terminal_probs = best.get("per_step_terminal_prob") or []
                    step_tool_outputs = best.get("per_step_predicted_tool_output") or []
                    imagined_entry = {
                        "calls": chosen_calls,
                        "score": best.get("score"),
                        "reason": best.get("reason"),
                        "predicted_state": best.get("predicted_state"),
                        "terminal_probability": (
                            float(terminal_probs[0])
                            if terminal_probs
                            else best.get("terminal_probability")
                        ),
                    }
                    for key in ("id", "bind"):
                        if isinstance(best_step, dict) and key in best_step:
                            imagined_entry[key] = best_step[key]
                    if step_tool_outputs:
                        imagined_entry["predicted_tool_output"] = step_tool_outputs[0]
                    imagined_plan.append(imagined_entry)
                    imagined_history = imagined_history + [{"action": best_step, "observation": ""}]
                    # No per-step short-circuit here -- gating happens AFTER the full trajectory is
                    # built: a flat-looking first step can still sit on a trajectory that clearly
                    # separates from alternatives once the horizon plays out.

            # Trajectory-level gate: aggregate over the FULL generated trajectory, not the
            # isolated first-step candidate spread.
            horizon = max(1, self.beam_horizon)
            full_horizon_reached = len(imagined_plan) == horizon
            trajectory_score = sum(per_depth_scores)
            trajectory_avg_normalized = (
                sum(per_depth_normalized) / len(per_depth_normalized)
                if per_depth_normalized
                else 0.0
            )
            trajectory_confidence_gate = beam_cfg.gate_flat_score_ratio / max(
                first_num_candidates, 1
            )
            beam_confident = (
                full_horizon_reached and trajectory_avg_normalized >= trajectory_confidence_gate
            )

            first_step_schema_warnings = (
                imagined_plan[0].get("schema_warnings") if imagined_plan else []
            ) or []
            if (
                beam_confident
                and first_recommend_calls_candidate
                and self._calls_have_symbolic_references(first_recommend_calls_candidate)
            ):
                first_recommend_calls = None
                first_override_reason = "unresolved_first_step_reference"
            elif beam_confident and first_recommend_calls_candidate and first_step_schema_warnings:
                first_recommend_calls = None
                first_override_reason = "schema_warning_first_step"
            elif beam_confident and first_recommend_calls_candidate:
                first_recommend_calls = first_recommend_calls_candidate
                first_override_reason = "beam_best_beats_seed"
            elif first_num_candidates == 0:
                first_recommend_calls = None
                first_override_reason = "no_candidates"
            elif not beam_confident:
                first_recommend_calls = None
                first_override_reason = "below_confidence_gate"
            else:
                first_recommend_calls = None
                first_override_reason = first_recommend_reason_candidate

            # Only force the agent's action when hard-override is explicitly on. Keep the
            # ORIGINAL agent proposal in agent_calls_original before reassigning seed_norm below
            # -- seed_norm becomes "what actually executes" (used for both "calls" and the
            # anti-repetition memory), but detail["agent_calls"] must still reflect what the
            # agent itself proposed, or a hard-override run can never audit whether an override
            # actually changed anything (agent_calls would silently equal calls every time).
            agent_calls_original = seed_norm
            override_applied = False
            revision_detail: dict[str, Any] = {"beam_revision_enabled": self.beam_revision}
            if self.beam_hard_override and first_recommend_calls:
                seed_norm = first_recommend_calls  # planned_calls analogue: what actually executes
                override_applied = True
                if imagined_plan:
                    self._set_beam_terminal_advisory(
                        probability=imagined_plan[0].get("terminal_probability"),
                        source="beam_hard_override",
                        step_index=0,
                    )
            elif self.beam_revision and first_recommend_calls and imagined_plan:
                seed_norm, revision_detail = self._beam_revision_choose(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    state_text=current_state_text,
                    agent_calls=agent_calls_original,
                    planned_calls=first_recommend_calls,
                    remaining_plan=imagined_plan,
                )
                override_applied = self._calls_signature(seed_norm) != self._calls_signature(
                    agent_calls_original
                )
                if revision_detail.get("beam_revision_plan_selected"):
                    self._set_beam_terminal_advisory(
                        probability=imagined_plan[0].get("terminal_probability"),
                        source="beam_revision",
                        step_index=0,
                    )
            # Cross-replan anti-repetition memory tracks the EXECUTED action (now known, since the
            # hard-override decision above is final) -- deferred to here rather than depth 0, since
            # it depends on the trajectory-level confidence gate.
            self._beam_recent_tool_names.append(self._step_tool_names({"tool_calls": seed_norm}))
            self._beam_has_planned = (
                True  # a re-plan attempt happened; the cooldown alone governs from here
            )
            decoded_tool_output = False
            # Decode the predicted TOOL OUTPUT (not just the canonical-event labels) for the
            # winning trajectory -- but only once it's confirmed worthwhile (beam_confident), since
            # decoding is a generative greedy loop (one per horizon step), far more expensive than
            # the classification-head scoring above. Requires a checkpoint with a trained
            # obs_grounding decoder (native or merged in via WM_JEPA_MERGE_CHECKPOINT).
            if beam_confident and imagined_plan and self.beam_decode_tool_output:
                try:
                    if getattr(getattr(self._wm, "model", None), "obs_grounding", False):
                        plan_actions = [
                            {"tool_calls": ewm.to_openai_tool_calls(entry["calls"])}
                            for entry in imagined_plan
                        ]
                        decoded_texts = self._wm.decode_plan_observations(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            input_history=imagined_history_start,
                            plan=plan_actions,
                            max_new_tokens=self.beam_decode_max_new_tokens,
                        )
                        for entry, text in zip(imagined_plan, decoded_texts):
                            entry["predicted_tool_output"] = text
                        decoded_tool_output = True
                    else:
                        logger.warning(
                            "ewm_imagined: WM_BEAM_PLAN_DECODE_TOOL_OUTPUT=1 but this checkpoint has no "
                            "trained obs_grounding decoder; skipping (canonical-event labels only)."
                        )
                except Exception as exc:
                    logger.warning(
                        "ewm_imagined: beam_plan tool-output decode failed (%s); labels only", exc
                    )
            # Inject the imagined trajectory ONLY when the beam was confident. A flat/low-confidence
            # plan is more likely to nudge the agent toward generic-success safe reads than help.
            if beam_confident and imagined_plan:
                plan_selected_by_revision = bool(revision_detail.get("beam_revision_plan_selected"))
                keep_plan = (
                    self.beam_hard_override or not self.beam_revision or plan_selected_by_revision
                )
                if keep_plan:
                    self._beam_imagined_plan = imagined_plan
                    self._beam_plan_history_offset = len(imagined_history_start)
                    first_step_executed = (self.beam_hard_override and override_applied) or (
                        self.beam_revision and plan_selected_by_revision
                    )
                    self._beam_plan_cursor = 1 if first_step_executed else 0
                    injected = True
                else:
                    # The policy chose its own action. The remaining trajectory was conditioned
                    # on a different first transition and is therefore no longer valid.
                    self._beam_imagined_plan = []
                    self._beam_plan_cursor = 0
                    self._beam_plan_history_offset = 0
                    injected = False
            else:
                self._beam_imagined_plan = []
                self._beam_plan_cursor = 0
                self._beam_plan_history_offset = 0
                injected = False

            return AdviseResult(
                text="",
                detail={
                    "backend": "ewm_imagined",
                    "strategy": "beam_plan",
                    "calls": seed_norm,
                    "event": event,
                    "replanned": True,
                    "imagined_rollout_mode": self.imagined_rollout_mode,
                    "sample_temperature_ladder": self.sample_temperature_ladder,
                    "sample_temperature_ladder_max": self.sample_temperature_ladder_max,
                    "beam_plan_ssot_diversity": self.beam_ssot_diversity,
                    "beam_plan_refinement_rounds": self.beam_refinement_rounds,
                    "beam_plan_refinement_top_k": self.beam_refinement_top_k,
                    "beam_plan_revision": self.beam_revision,
                    "beam_plan_revision_temperature": self.beam_revision_temperature,
                    "beam_action_aware_score_config": {
                        "read_saturation_threshold": self.beam_read_saturation_threshold,
                        "read_penalty": self.beam_read_penalty,
                        "read_after_progress_scale": self.beam_read_after_progress_scale,
                        "first_step_weight": self.beam_first_step_weight,
                        "required_action_coverage_bonus": (
                            self.beam_required_action_coverage_bonus
                        ),
                        "first_required_action_bonus": self.beam_first_required_action_bonus,
                    },
                    "beam_schema_soft_validation_config": {
                        "missing_required_penalty": (self.beam_schema_missing_required_penalty),
                        "invalid_type_penalty": self.beam_schema_invalid_type_penalty,
                    },
                    "beam_refinement_rounds_completed": refinement_rounds_completed,
                    "beam_refinement_score_passes": refinement_score_passes,
                    "beam_refinement_round_telemetry": refinement_round_telemetry,
                    "llm_ewm_mode": self.llm_ewm_mode or None,
                    "beam_action_sampler_enabled": self._beam_action_sampler is not None,
                    "beam_action_sampler_backend": self.beam_action_sampler_backend or None,
                    "beam_action_sampler_model": self.beam_action_sampler_model or None,
                    "beam_action_sampler_max_new_tokens": (
                        self.beam_action_sampler_max_new_tokens
                        if self._beam_action_sampler is not None
                        else None
                    ),
                    **critic_telemetry,
                    "override_applied": override_applied,
                    "override_reason": (
                        "beam_revision_selected_plan"
                        if override_applied and self.beam_revision and not self.beam_hard_override
                        else first_override_reason
                    ),
                    **revision_detail,
                    "num_candidates": first_num_candidates,
                    # set only under a prediction-ablation control (WM_JEPA_PREDICTION_CONTROL)
                    "prediction_control": first_best.get("prediction_control"),
                    "beam_llm_calls": beam_calls,
                    "imagined_plan_len": len(imagined_plan),
                    "best_score": first_best.get("score"),
                    "best_normalized_score": first_best.get("normalized_score"),
                    "best_reason": first_best.get("reason"),
                    "trajectory_score": trajectory_score,
                    "trajectory_avg_normalized_score": trajectory_avg_normalized,
                    "full_horizon_reached": full_horizon_reached,
                    "confidence_gate": trajectory_confidence_gate,
                    "beam_confident": beam_confident,
                    "advisory": not self.beam_hard_override,
                    "hard_override": self.beam_hard_override,
                    "injected": injected,
                    "recommended_calls": first_recommend_calls,
                    "agent_calls": agent_calls_original,
                    "decoded_tool_output": decoded_tool_output,
                    "beam_plan_terminal_advice": self.beam_terminal_advice,
                    "beam_plan_terminal_advice_threshold": self.beam_terminal_advice_threshold,
                    "beam_plan_terminal_advice_count": self._beam_terminal_advice_count,
                    "beam_plan_terminal_finish_advice_count": self._beam_terminal_finish_advice_count,
                    "beam_plan_terminal_middle_advice_count": self._beam_terminal_middle_advice_count,
                    "open_loop_candidate_stats": open_loop_candidate_stats,
                    "candidate_score_diagnostics": candidate_score_diagnostics,
                    "imagined_plan": self._plan_preview(imagined_plan),
                },
            )
        except Exception as exc:
            logger.warning(
                "ewm_imagined: beam_plan failed (%s); falling back to baseline action", exc
            )
            return AdviseResult(
                text="", detail={**base_detail, "event": "GYM_BEAM_PLAN_ERROR", "error": str(exc)}
            )

    def hier_latent_cem_step(
        self, conversation_flow, *, seed_calls, user_query: str = ""
    ) -> AdviseResult:
        """One hierarchical latent-action CEM MPC decision (``_ewm_hier_cem.hierarchical_cem_plan``):
        ONE LLM call opens the search space with K diverse anchor actions; the world model then
        samples thousands of CONTINUOUS latent-action trajectories around them and CEM-refines a
        per-family Gaussian proposal toward the highest-scoring region -- LLM-free, decode-free
        (recursive rollout). Only the converged best trajectory is decoded (nearest-anchor or
        learned-decoder) and shown to the agent.

        Same cooldown-vs-rejection state machine and advisory-by-default semantics as
        :meth:`beam_plan_step` (see its docstring); the only difference is *how* the imagined
        trajectory is produced. Any failure degrades to the baseline (``calls == seed_calls``)."""
        from ejepa_wm.backends._ewm_canonical_event_scoring import CanonicalEventScoreConfig
        from ejepa_wm.backends._ewm_hier_cem import HierarchicalCEMConfig, hierarchical_cem_plan

        seed_norm = [ewm.normalize_tool_call(c) for c in (seed_calls or [])]
        seed_norm = [c for c in seed_norm if str(c.get("name", "")).strip()]
        base_detail = {"backend": "ewm_imagined", "strategy": "hier_latent_cem", "calls": seed_norm}

        if not self.supports_hier_cem():
            return AdviseResult(
                text="", detail={**base_detail, "event": "GYM_HIER_CEM_UNSUPPORTED"}
            )
        if not seed_norm:
            return AdviseResult(text="", detail={**base_detail, "event": "GYM_HIER_CEM_NO_SEED"})

        system_prompt = self._prompt_from_flow(conversation_flow, "system_message")
        user_prompt = user_query or self._prompt_from_flow(conversation_flow, "user_message")
        tool_names = self._tool_names_from_flow(conversation_flow)
        execute_steps = self.beam_execute_steps

        hier_plan_exhausted = bool(self._hier_imagined_plan) and self._hier_plan_cursor >= len(
            self._hier_imagined_plan
        )
        need_replan = (
            not self._hier_has_planned
            or self._hier_plan_cursor >= execute_steps
            or hier_plan_exhausted
        )
        if not need_replan and self._hier_imagined_plan:
            # Follow the visible imagined trajectory: no WM call, no override.
            self._hier_plan_cursor += 1
            return AdviseResult(
                text="",
                detail={
                    **base_detail,
                    "event": "GYM_HIER_CEM_FOLLOW",
                    "replanned": False,
                    "override_applied": False,
                    "plan_cursor": self._hier_plan_cursor,
                    "plan_len": len(self._hier_imagined_plan),
                },
            )
        if not need_replan:
            # Last re-plan was rejected (not confident) -- no world-model call, no injected
            # guidance, same cadence as if a confident plan had been injected.
            self._hier_plan_cursor += 1
            return AdviseResult(
                text="",
                detail={
                    **base_detail,
                    "event": "GYM_HIER_CEM_COOLDOWN",
                    "replanned": False,
                    "override_applied": False,
                    "plan_cursor": self._hier_plan_cursor,
                    "execute_steps": execute_steps,
                },
            )

        try:
            imagined_history = self._input_history_from_flow(conversation_flow)
            gen = self._wm
            context_text = gen._context_text(system_prompt, user_prompt)
            current_state_text = gen._current_state_text(
                system_prompt, user_prompt, imagined_history
            )

            def _hier_llm_generate(prompt: str) -> str:
                self._agent_call_count += 1
                raw = self._agent.generate_from_messages(
                    [
                        {
                            "role": "system",
                            "content": "You output ONLY a JSON array of tool-call actions. No prose, no markdown.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                )
                return raw if isinstance(raw, str) else str(getattr(raw, "content", raw) or "")

            # Same up-weight-progress tweak as beam_plan_step.
            score_cfg = CanonicalEventScoreConfig()
            score_cfg.weights["progress_signal"] = 1.2
            score_cfg.utilities["progress_signal"]["neutral"] = -0.3
            score_cfg.utilities["progress_signal"]["negative"] = -1.0
            cem_cfg = HierarchicalCEMConfig(
                num_llm_anchors=self.hier_cem_anchors,
                num_samples=self.hier_cem_samples,
                num_elites=self.hier_cem_elites,
                num_iters=self.hier_cem_iters,
                horizon=self.hier_cem_horizon,
                init_std=self.hier_cem_init_std,
                min_std=self.hier_cem_min_std,
                smoothing=self.hier_cem_smoothing,
                min_elite_agreement=self.hier_cem_min_elite_agreement,
                top_k=self.top_k,
                max_input_length=gen.max_input_length,
                max_action_length=gen.max_action_length,
                decode_strategy=self.hier_cem_decode_strategy,
                decode_max_new_tokens=self.hier_cem_decode_max_new_tokens,
                score_config=score_cfg,
            )
            result = hierarchical_cem_plan(
                model=gen.model,
                tokenizer=gen.tokenizer,
                vocab=gen.canonical_event_vocab,
                context_text=context_text,
                current_state_text=current_state_text,
                llm_generate=_hier_llm_generate,
                config=cem_cfg,
                tool_names=tool_names or None,
                input_history=imagined_history,
            )
            confident = bool(result.get("confident"))
            imagined_plan = []
            for entry in result.get("imagined_plan") or []:
                calls = [
                    c
                    for c in (ewm.normalize_tool_call(c) for c in (entry.get("calls") or []))
                    if str(c.get("name", "")).strip()
                ]
                imagined_plan.append(
                    {
                        "calls": calls,
                        "score": entry.get("score"),
                        "reason": entry.get("reason"),
                        "predicted_state": entry.get("predicted_state"),
                    }
                )
            recommended_calls = imagined_plan[0]["calls"] if imagined_plan else None
            # Keep the ORIGINAL agent proposal before seed_norm is potentially reassigned below
            # -- see the matching comment in _beam_plan_response for why detail["agent_calls"]
            # must not silently collapse to equal detail["calls"] whenever an override fires.
            agent_calls_original = seed_norm
            override_applied = False
            # Advisory by default: the agent's own action always executes unless
            # WM_BEAM_PLAN_HARD_OVERRIDE is set (shared knob with beam_plan).
            if self.beam_hard_override and confident and recommended_calls:
                seed_norm = recommended_calls
                override_applied = True
            self._hier_has_planned = (
                True  # a re-plan attempt happened; the cooldown alone governs from here
            )
            # Confidence-gated injection: only show a plan the CEM actually converged on
            # (elite_agreement >= hier_cem_min_elite_agreement).
            if confident and imagined_plan:
                self._hier_imagined_plan = imagined_plan
                self._hier_plan_cursor = 1 if (self.beam_hard_override and override_applied) else 0
                injected = True
            else:
                self._hier_imagined_plan = []
                self._hier_plan_cursor = 0
                injected = False

            return AdviseResult(
                text="",
                detail={
                    "backend": "ewm_imagined",
                    "strategy": "hier_latent_cem",
                    "calls": seed_norm,
                    "event": "GYM_HIER_CEM_PLAN",
                    "replanned": True,
                    "num_anchors": result.get("num_anchors"),
                    "num_llm_calls": result.get("num_llm_calls"),
                    "imagined_plan_len": len(imagined_plan),
                    "best_score": result.get("score"),
                    "elite_agreement": result.get("elite_agreement"),
                    "vetoed": result.get("vetoed"),
                    "reason": result.get("reason"),
                    "confident": confident,
                    "advisory": not self.beam_hard_override,
                    "hard_override": self.beam_hard_override,
                    "injected": injected,
                    "override_applied": override_applied,
                    "recommended_calls": recommended_calls,
                    "agent_calls": agent_calls_original,
                    "family_distribution": result.get("family_distribution"),
                    "imagined_plan": self._plan_preview(imagined_plan),
                },
            )
        except Exception as exc:
            logger.warning(
                "ewm_imagined: hier_latent_cem failed (%s); falling back to baseline action", exc
            )
            return AdviseResult(
                text="", detail={**base_detail, "event": "GYM_HIER_CEM_ERROR", "error": str(exc)}
            )


__all__ = ["EwmImaginedWorldModel"]
