#!/usr/bin/env python3
"""Measure world-model latency as a function of rollout horizon.

No existing run sweeps the horizon (every harness run to date is fixed at
``--wm-beam-plan-horizon 4`` / ``--wm-itp-fixed-k 4``), and the beam planner does
not time its scoring pass, so panel 1 needs its own measurement. This benchmarks
the one call every world model implements identically --
``score_action_plans_canonical_event`` -- over a grid of horizons and candidate
counts, and records what each architecture actually spends:

* Enterprise-JEPA: ``n_candidates x horizon`` latent transition forward passes,
  no decoding.
* LLM world models: the same rollout, but each transition is autoregressive
  decoding; the predicted-token count is recorded alongside the latency.

Each LLM world model can be measured on either serving backend, and both may be
requested in one sweep -- every measurement records which one it used:

* ``transformers`` (``--llm-*-checkpoint``): weights loaded in-process on the
  local GPU. This is the **like-for-like comparison against JEPA**, which also
  runs in-process; token counts come from the backend's own tokenizer.
* ``vllm`` (``--llm-*-model`` + ``--llm-base-url``): the served endpoint the
  harness actually runs against, with continuous batching and a warm server;
  token counts come from ``/metrics`` (``--metrics-url``).

JEPA only, no server needed::

    .venv/bin/python3 scripts/measure_wm_latency_vs_horizon.py \\
        --jepa-checkpoint checkpoints/jepa \\
        --horizons 1,2,4,8 --candidates 8 --repeats 5 \\
        --out results/wm_latency/wm_latency_vs_horizon.json

All three world models on the JEPA-matched (in-process) backend::

    .venv/bin/python3 scripts/measure_wm_latency_vs_horizon.py \\
        --jepa-checkpoint checkpoints/jepa \\
        --llm-state-checkpoint checkpoints/llm_wm_state \\
        --llm-tool-output-checkpoint Qwen/Qwen3.6-27B

Note that ``torch``/``transformers`` are needed for the in-process backends; the
repo venv has neither, so run these with a venv that does, e.g.
``assets/EnterpriseOps-Gym/purple/.venv/bin/python``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

SYSTEM_PROMPT = (
    "You are an enterprise operations assistant with access to MCP tools. "
    "Use the tools to satisfy the user request and verify the result."
)
USER_PROMPT = (
    "Create a channel named 'Project Updates' in the TechCorp Solutions team and "
    "start an audio call between James Wilson and Alice Johnson."
)
# Base action/observation shapes. A real beam scores DIFFERENT candidate actions
# against a state that is new at every agent step, and both matter for latency:
# the JEPA action cache is keyed by action text and its text cache by state text,
# so reusing either turns a backbone forward into a cache hit.
TOOL_NAMES = ("create_channel", "list_users", "create_call", "list_teams")
ACTION = {
    "tool_calls": [
        {
            "name": "create_channel",
            "arguments": {
                "team_id": "team_techcorp_001",
                "display_name": "Project Updates",
                "membership_type": "standard",
            },
        }
    ]
}
OBSERVATION = json.dumps(
    {
        "success": True,
        "result": {"channel_id": "channel_322f0569", "display_name": "Project Updates"},
    }
)


def action_variant(candidate: int, step: int) -> dict[str, Any]:
    """One candidate's action at one horizon step, distinct in tool and arguments."""
    return {
        "tool_calls": [
            {
                "name": TOOL_NAMES[candidate % len(TOOL_NAMES)],
                "arguments": {
                    "team_id": f"team_techcorp_{candidate:03d}",
                    "display_name": f"Project Updates {candidate}-{step}",
                    "membership_type": "standard",
                },
            }
        ]
    }


def build_plans(*, horizon: int, candidates: int, distinct: bool) -> list[list[Any]]:
    if not distinct:
        return [[dict(ACTION) for _ in range(horizon)] for _ in range(candidates)]
    return [
        [action_variant(candidate, step) for step in range(horizon)]
        for candidate in range(candidates)
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--jepa-checkpoint", help="Enterprise-JEPA checkpoint dir.")
    parser.add_argument(
        "--llm-state-checkpoint",
        help=(
            "State-output world model as a local transformers checkpoint -- the "
            "JEPA-matched backend (in-process weights, no server)."
        ),
    )
    parser.add_argument(
        "--llm-state-model",
        help="Served (vLLM) canonical-event model name (state output) on --llm-base-url.",
    )
    parser.add_argument(
        "--llm-tool-output-checkpoint",
        help=(
            "Tool-output world model as a local transformers checkpoint -- the "
            "JEPA-matched backend (in-process weights, no server)."
        ),
    )
    parser.add_argument(
        "--llm-tool-output-model",
        help="Served (vLLM) model name for the tool-output world model + judge.",
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=1024,
        help="Decode budget for the transformers tool-output world model.",
    )
    parser.add_argument("--llm-base-url", help="OpenAI-compatible base URL for served LLM WMs.")
    parser.add_argument("--llm-api-key", default="EMPTY")
    parser.add_argument(
        "--llm-tokenizer",
        type=Path,
        help=(
            "tokenizer.json of the SERVED LLM world model. When given, prompt and "
            "completion tokens are counted locally per call, which is exact and "
            "unaffected by other clients sharing the endpoint (the --metrics-url "
            "Prometheus delta is not)."
        ),
    )
    parser.add_argument("--horizons", default="1,2,4,8", help="Comma-separated rollout depths.")
    parser.add_argument(
        "--candidates",
        type=int,
        default=8,
        help="Candidate trajectories scored per call (the beam width).",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Timed repetitions per cell.")
    parser.add_argument(
        "--statistic",
        choices=("mean", "median"),
        default="mean",
        help=(
            "Which per-call statistic the derived per-trajectory / per-transition "
            "figures use. Both are always written to the JSON, together with the "
            "stdev and the min."
        ),
    )
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warmup calls per cell.")
    parser.add_argument(
        "--history-steps",
        type=int,
        default=4,
        help="Prior interactions in the input history (context length control).",
    )
    parser.add_argument(
        "--state-cache",
        choices=("cold", "warm"),
        default="cold",
        help=(
            "cold (default): every timed call gets a fresh state text, so the "
            "state/history encode is paid per call exactly as in a live run. "
            "warm: reuse one state across repeats, which measures only the "
            "marginal per-transition cost (JEPA caches the encoded latents)."
        ),
    )
    parser.add_argument(
        "--identical-candidates",
        action="store_true",
        help=(
            "Score N copies of one action instead of N distinct candidates. "
            "Lets the action-latent cache dedupe them; distinct is the default "
            "because a real beam proposes different actions."
        ),
    )
    parser.add_argument(
        "--metrics-url",
        help="vLLM /metrics endpoint used to attribute predicted tokens to LLM world models.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "out" / "wm_latency" / "wm_latency_vs_horizon.json",
    )
    return parser.parse_args(argv)


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    """Persist the sweep so far.

    Called after every measured cell: a slow leg (a 35B tool-output model can take
    hours at horizon 8) must not put the already-measured cells at risk if the run
    is interrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def runtime_environment() -> dict[str, Any]:
    """Versions and optional fast kernels, recorded with every sweep.

    Some checkpoints (e.g. the qwen-agentworld family) implement a linear-attention
    fast path that needs ``flash-linear-attention`` and ``causal-conv1d``. Without
    them transformers prints "The fast path is not available ... Falling back to
    torch implementation" and runs a reference kernel instead -- correct, but far
    slower, which would understate the in-process backend. Recorded rather than
    assumed, so a latency figure cannot silently mix kernel regimes.
    """
    environment: dict[str, Any] = {}
    for module in ("torch", "transformers"):
        try:
            environment[f"{module}_version"] = __import__(module).__version__
        except Exception:
            environment[f"{module}_version"] = None
    environment["fast_kernels"] = {
        name: importlib.util.find_spec(name) is not None
        for name in ("fla", "causal_conv1d", "flash_attn")
    }
    try:
        import torch

        environment["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        environment["cuda_device"] = None
    return environment


def input_history(steps: int, *, salt: str = "") -> list[dict[str, Any]]:
    """Prior interactions. ``salt`` makes the state/observation text unique.

    A non-empty salt reproduces the live condition: the current-state text is new
    at every agent step, so encoding it is a real backbone forward rather than a
    hit in ``_latent_text_cache``.
    """
    observation = OBSERVATION if not salt else OBSERVATION + f" /* {salt} */"
    return [
        {
            "step": index,
            "action": action_variant(0, index),
            "observation": observation,
            "state": observation,
        }
        for index in range(steps)
    ]


def generation_tokens(metrics_url: str | None) -> float | None:
    """Total generated tokens reported by a vLLM endpoint (None when unavailable)."""
    if not metrics_url:
        return None
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as response:
            body = response.read().decode("utf-8", "replace")
    except OSError:
        return None
    for line in body.splitlines():
        if line.startswith("vllm:generation_tokens_total{"):
            try:
                return float(line.rsplit(" ", 1)[-1])
            except ValueError:
                return None
    return None


def build_jepa(checkpoint: str):
    from ejepa_wm.backends._ewm_jepa import JepaEwmGenerator

    return JepaEwmGenerator(
        checkpoint,
        imagined_observation_backend="canonical_event",
        dtype=os.getenv("WM_JEPA_DTYPE", "auto"),
    )


def build_llm_state_transformers(checkpoint: str):
    """State-output world model, in-process transformers -- the JEPA-matched backend."""
    from ejepa_wm.backends._ewm_llm_canonical_event import LlmCanonicalEventGenerator

    return LlmCanonicalEventGenerator(checkpoint)


def build_llm_tool_output_transformers(checkpoint: str, max_new_tokens: int):
    """Tool-output world model + judge, in-process transformers.

    ``LlmToolOutputJudgeGenerator`` takes a world-model generator and a judge
    generator; in a real run the judge is the policy agent ("same model family as
    the acting agent"). A standalone benchmark has no policy agent, so the same
    local model serves both roles -- which is also the like-for-like comparison
    against JEPA, where one set of local weights answers every question.
    """
    from ejepa_wm.backends._ewm_generators import HFEwmGenerator
    from ejepa_wm.backends._ewm_llm_tool_output_judge import LlmToolOutputJudgeGenerator

    local = HFEwmGenerator(checkpoint, max_new_tokens=max_new_tokens)
    return LlmToolOutputJudgeGenerator(local, local)


class _LocalTokenizer:
    """``tokenizers`` tokenizer exposed with the ``tokenizer(text)["input_ids"]`` shape
    that :func:`attach_token_counter` expects from a transformers tokenizer."""

    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(path))

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": self._tok.encode(str(text)).ids}


def attach_prompt_counter(generator: Any, tokenizer: Any) -> Any:
    """Count PROMPT tokens sent by a served canonical-event generator.

    :func:`attach_token_counter` counts completions from the returned text; the
    prompt side is only visible on the way in, so wrap ``_generate_message_batch``
    (the single choke point for served requests) and tokenize every message.
    Returns the counter (``.prompt_tokens`` / ``.reset()``) or ``None``.
    """
    original = getattr(generator, "_generate_message_batch", None)
    if original is None or tokenizer is None:
        return None

    class PromptCounter:
        prompt_tokens = 0

        def reset(self) -> None:
            self.prompt_tokens = 0

    counter = PromptCounter()

    def counted(messages_batch: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        for messages in messages_batch or []:
            for message in messages or []:
                counter.prompt_tokens += len(tokenizer(message.get("content", ""))["input_ids"])
        return original(messages_batch, *call_args, **call_kwargs)

    generator._generate_message_batch = counted
    return counter


def build_served(
    model: str,
    base_url: str,
    api_key: str,
    *,
    tool_output: bool,
    tokenizer_path: Path | None = None,
):
    """Served (vLLM) canonical-event scorer, or tool-output + judge world model.

    Mirrors how ``ewm_imagined`` wires these backends in a real run (same client,
    parallel requests enabled, same default token budgets) so the measured
    latency is the one the harness would pay. With ``tokenizer_path`` the served
    generator gets a local tokenizer so prompt and completion tokens are counted
    exactly, independent of other clients on the endpoint.
    """
    from ejepa_wm.backends._ewm_runtime import EwmGenerator

    served = EwmGenerator(
        model,
        base_url,
        api_key,
        max_new_tokens=int(os.getenv("WM_EWM_MAX_NEW_TOKENS", "1024" if tool_output else "256")),
    )
    served.supports_parallel_requests = True
    tokenizer = _LocalTokenizer(tokenizer_path) if tokenizer_path else None
    if tool_output:
        from ejepa_wm.backends._ewm_llm_tool_output_judge import LlmToolOutputJudgeGenerator

        generator = LlmToolOutputJudgeGenerator(served, served)
        if tokenizer is not None:
            served.tokenizer = tokenizer
        return generator
    from ejepa_wm.backends._ewm_llm_canonical_event import ServedLlmCanonicalEventGenerator

    generator = ServedLlmCanonicalEventGenerator(served, mode="llm_canonical_trained")
    if tokenizer is not None:
        generator.tokenizer = tokenizer  # picked up by attach_token_counter (completions)
        generator.prompt_counter = attach_prompt_counter(generator, tokenizer)
    return generator


def world_model_specs(args: argparse.Namespace) -> list[tuple[str, str, Any]]:
    """``(world model, serving backend, factory)`` for every model requested.

    Both LLM world models can be measured either way, and both may be requested in
    one sweep -- the ``transformers`` variants are the like-for-like comparison
    against JEPA (in-process weights on the same GPU), while the ``vllm`` variants
    are what the harness actually runs against a served endpoint.
    """
    specs: list[tuple[str, str, Any]] = []
    if args.jepa_checkpoint:
        specs.append(("Enterprise-JEPA", "transformers", lambda: build_jepa(args.jepa_checkpoint)))
    if args.llm_state_checkpoint:
        specs.append(
            (
                "LLM-WM (state output)",
                "transformers",
                lambda: build_llm_state_transformers(args.llm_state_checkpoint),
            )
        )
    if args.llm_state_model:
        specs.append(
            (
                "LLM-WM (state output)",
                "vllm",
                lambda: build_served(
                    args.llm_state_model,
                    args.llm_base_url,
                    args.llm_api_key,
                    tool_output=False,
                    tokenizer_path=args.llm_tokenizer,
                ),
            )
        )
    if args.llm_tool_output_checkpoint:
        specs.append(
            (
                "LLM-WM (tool output)",
                "transformers",
                lambda: build_llm_tool_output_transformers(
                    args.llm_tool_output_checkpoint, args.llm_max_new_tokens
                ),
            )
        )
    if args.llm_tool_output_model:
        specs.append(
            (
                "LLM-WM (tool output)",
                "vllm",
                lambda: build_served(
                    args.llm_tool_output_model,
                    args.llm_base_url,
                    args.llm_api_key,
                    tool_output=True,
                    tokenizer_path=args.llm_tokenizer,
                ),
            )
        )
    return specs


def attach_token_counter(generator: Any) -> Any:
    """Count tokens generated by an in-process transformers backend.

    vLLM reports generated tokens through ``/metrics``; local backends do not, so
    wrap whichever method actually produces text and tokenize the completions with
    that backend's own tokenizer. Returns an object with ``.tokens`` / ``.reset()``,
    or ``None`` when the backend has no local tokenizer (served models).
    """

    class Counter:
        tokens = 0

        def reset(self) -> None:
            self.tokens = 0

    counter = Counter()

    def wrap(owner: Any, method_name: str, tokenizer: Any) -> bool:
        original = getattr(owner, method_name, None)
        if original is None or tokenizer is None:
            return False

        def counted(*call_args: Any, **call_kwargs: Any) -> Any:
            result = original(*call_args, **call_kwargs)
            texts = result if isinstance(result, list) else [result]
            for text in texts:
                if isinstance(text, str) and text:
                    counter.tokens += len(tokenizer(text)["input_ids"])
            return result

        setattr(owner, method_name, counted)
        return True

    # Canonical-event scorer (state output): batches prompts internally.
    if wrap(generator, "_generate_message_batch", getattr(generator, "tokenizer", None)):
        return counter
    # Tool-output judge: text comes from its world-model/judge generators.
    inner = getattr(generator, "world_model_generator", None)
    if inner is not None and wrap(
        inner, "generate_from_messages", getattr(inner, "tokenizer", None)
    ):
        judge = getattr(generator, "judge_generator", None)
        if judge is not None and judge is not inner:
            wrap(judge, "generate_from_messages", getattr(judge, "tokenizer", None))
        return counter
    return None


def time_cell(
    generator: Any,
    *,
    horizon: int,
    candidates: int,
    history_steps: int,
    repeats: int,
    warmup: int,
    metrics_url: str | None,
    token_counter: Any = None,
    state_cache: str = "cold",
    distinct_candidates: bool = True,
    statistic: str = "mean",
) -> dict[str, Any]:
    """Time ``repeats`` scoring calls for one (horizon, candidates) cell.

    Each call is rebuilt rather than reused: under ``state_cache="cold"`` the
    state/observation text carries a per-call salt, so the encode cost is paid
    every call the way a live agent step pays it.
    """
    from ejepa_wm.backends._ewm_canonical_event_scoring import CanonicalEventScoreConfig

    plans = build_plans(horizon=horizon, candidates=candidates, distinct=distinct_candidates)

    def call(tag: str) -> Any:
        salt = "" if state_cache == "warm" else f"h{horizon}-{tag}-{time.time_ns()}"
        return generator.score_action_plans_canonical_event(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT,
            input_history=input_history(history_steps, salt=salt),
            action_plans=plans,
            score_config=CanonicalEventScoreConfig(),
        )

    for index in range(max(0, warmup)):
        call(f"warmup{index}")
    tokens_before = generation_tokens(metrics_url)
    if token_counter is not None:
        token_counter.reset()
    prompt_counter = getattr(generator, "prompt_counter", None)
    if prompt_counter is not None:
        prompt_counter.reset()
    samples: list[float] = []
    for index in range(max(1, repeats)):
        started = time.perf_counter()
        call(f"rep{index}")
        samples.append(time.perf_counter() - started)
    tokens_after = generation_tokens(metrics_url)
    predicted_tokens = None
    token_source = None
    if token_counter is not None and token_counter.tokens:
        # Local tokenizer (in-process backend, or a served one given --llm-tokenizer):
        # counted from the completions themselves, immune to other clients.
        predicted_tokens = token_counter.tokens / max(1, repeats)
        token_source = "local_tokenizer"
    elif tokens_before is not None and tokens_after is not None:
        # Served backend without a local tokenizer: the endpoint's Prometheus
        # counter, which also counts every other client's traffic in the window.
        predicted_tokens = (tokens_after - tokens_before) / max(1, repeats)
        token_source = "vllm_metrics_delta"
    prompt_tokens = (
        prompt_counter.prompt_tokens / max(1, repeats)
        if prompt_counter is not None and prompt_counter.prompt_tokens
        else None
    )
    transitions = horizon * candidates
    mean_seconds = statistics.mean(samples)
    median_seconds = statistics.median(samples)
    reported = mean_seconds if statistic == "mean" else median_seconds
    return {
        "horizon": horizon,
        "candidates": candidates,
        "transitions": transitions,
        "state_cache": state_cache,
        "candidate_mode": "distinct" if distinct_candidates else "identical",
        "repeats": len(samples),
        "statistic": statistic,
        # The reported figures below use ``statistic``; every raw summary is kept so a
        # reader can recompute with the other choice.
        "seconds_per_call": reported,
        "seconds_per_call_mean": mean_seconds,
        "seconds_per_call_median": median_seconds,
        "seconds_per_call_stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "seconds_per_call_min": min(samples),
        "seconds_per_call_max": max(samples),
        "seconds_per_candidate_trajectory": reported / candidates,
        "seconds_per_transition": reported / transitions,
        "seconds_per_candidate_trajectory_median": median_seconds / candidates,
        "seconds_per_transition_median": median_seconds / transitions,
        "predicted_tokens_per_call": predicted_tokens,
        "predicted_tokens_per_transition": (
            predicted_tokens / transitions if predicted_tokens is not None else None
        ),
        "prompt_tokens_per_call": prompt_tokens,
        "token_source": token_source,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = world_model_specs(args)
    if not specs:
        print(
            "Nothing to measure: pass --jepa-checkpoint and/or the LLM world-model options.",
            file=sys.stderr,
        )
        return 1
    horizons = [int(value) for value in str(args.horizons).split(",") if value.strip()]

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidates": args.candidates,
        "horizons": horizons,
        "history_steps": args.history_steps,
        "state_cache": args.state_cache,
        "candidate_mode": "identical" if args.identical_candidates else "distinct",
        "statistic": args.statistic,
        "repeats": args.repeats,
        "environment": runtime_environment(),
        "measurements": [],
    }
    for name, backend, factory in specs:
        tag = f"{name} / {backend}"
        print(f"[{tag}] loading world model...", file=sys.stderr)
        try:
            generator = factory()
        except Exception as exc:
            print(f"[{tag}] SKIPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        # In-process backends carry their own tokenizer; a served backend gets one only
        # when --llm-tokenizer was given (see build_served), in which case counting
        # locally is preferred over the shared endpoint's Prometheus delta.
        counter = (
            attach_token_counter(generator)
            if backend == "transformers" or getattr(generator, "tokenizer", None) is not None
            else None
        )
        for horizon in horizons:
            try:
                cell = time_cell(
                    generator,
                    horizon=horizon,
                    candidates=args.candidates,
                    history_steps=args.history_steps,
                    repeats=args.repeats,
                    warmup=args.warmup,
                    metrics_url=args.metrics_url if backend == "vllm" else None,
                    token_counter=counter,
                    state_cache=args.state_cache,
                    distinct_candidates=not args.identical_candidates,
                    statistic=args.statistic,
                )
            except Exception as exc:
                print(f"[{tag}] horizon={horizon} FAILED: {exc}", file=sys.stderr)
                continue
            cell["world_model"] = name
            cell["backend"] = backend
            payload["measurements"].append(cell)
            write_payload(args.out, payload)  # partial results survive an interrupt
            print(
                f"[{tag}] horizon={horizon:>2} "
                f"{cell['seconds_per_call']:.3f}+-{cell['seconds_per_call_stdev']:.3f}s/call "
                f"({cell['statistic']} of {cell['repeats']}) "
                f"{cell['seconds_per_candidate_trajectory'] * 1000:.1f}ms/trajectory "
                f"{cell['seconds_per_transition'] * 1000:.2f}ms/transition"
                + (
                    f" {cell['predicted_tokens_per_call']:.0f} predicted tokens/call"
                    if cell.get("predicted_tokens_per_call")
                    else ""
                ),
                file=sys.stderr,
            )
        del generator

    write_payload(args.out, payload)
    print(f"Wrote {args.out}")
    return 0 if payload["measurements"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
