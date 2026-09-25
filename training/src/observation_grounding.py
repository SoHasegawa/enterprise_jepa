"""Per-benchmark isolation of the "essential" environment-output span for
Option-C observation-token grounding (see finetuning_jepa.py).

The world-model latent z_pred is grounded by predicting the first-N tokens of
the *isolated* environment output (ECHO / arXiv 2605.24517: supervise only the
genuinely environment-determined tokens where the outcome signal concentrates).
Observation formats differ per benchmark, so the split rule is benchmark-specific
-- this module holds that registry, derived from real sampled trajectories.

Each JepaExample is tagged with its benchmark (benchmark_key_from_path on the
source trajectory file); `observation_ground_target_ids` returns the token ids to
supervise (empty list -> this example contributes no grounding target).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ANSI CSI colour/erase codes and OSC title-set sequences (with or without the
# leading ESC, since some dumps drop it and leave a bare "]0;...").
_ANSI = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"          # CSI  e.g. \x1b[2K, \x1b[0m
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC  e.g. \x1b]0;title\x07
    r"|\][0-9]+;[^\x07\n]*(?:\x07)?"       # bare ]0;title (ESC stripped upstream)
    r"|\x1b[<-?]?[ -/]*[@-~]"              # stray ESC-prefixed control
)
# codeactinstruct trailing step-counter boilerplate.
_STEP_FOOTER = re.compile(r"\s*You have \d+ steps? left and \d+ chances?.*\Z", re.S)
# nebius SWE-agent trailing "(Open file: ...) (Current directory: ...) bash-$" footer.
_SWE_FOOTER = re.compile(r"\s*\(Open file:.*?\)\s*\(Current directory:.*?\)\s*bash-\$?\s*\Z", re.S)
# swe-play "EXECUTION RESULT of [tool]:" header.
_SWE_PLAY_HEADER = re.compile(r"\AEXECUTION RESULT of \[[^\]]*\]:\s*", re.S)


@dataclass(frozen=True)
class GroundingSpec:
    direction: str = "front"          # "front" | "front_tail" | "first_line" | "skip"
    n: int = 48                       # first-N tokens to supervise
    tail: int = 0                     # last-M tokens (front_tail only)
    strip_ansi: bool = False
    strip_prefix: str | None = None   # literal prefix to drop
    strip_prefix_re: Any = None       # compiled regex prefix to drop
    strip_suffix_re: Any = None       # compiled regex suffix to drop


# Registry keyed by benchmark_key_from_path(). See the per-benchmark plan; every
# rule is grounded in real sampled observation formats.
GROUNDING_SPECS: dict[str, GroundingSpec] = {
    # --- core ---
    # "tool_name: {pretty JSON}\n\n..." -- need enough to pass etag/kind boilerplate
    # and reach status/error/first data key.
    "enterpriseops_gym": GroundingSpec(direction="front", n=64),
    # '{"count": N, "data":[...]}' or a bare SQL error -- strongly front-loaded.
    "crmarenapro": GroundingSpec(direction="front", n=32),
    # "stdout:\n<output>" -- strip the stdout: wrapper; errors/first line at front.
    "terminalbench": GroundingSpec(direction="front", n=64, strip_prefix="stdout:\n"),
    # "An error occurred when calling tool X: <ErrType>: ..." or short result.
    "toucan": GroundingSpec(direction="front", n=48),
    # --- adp: terminal / shell ---
    "nemotron_terminal_corpus": GroundingSpec(direction="front", n=48, strip_ansi=True),
    "agenttuning_os": GroundingSpec(direction="front", n=32, strip_ansi=True),
    "openhands": GroundingSpec(direction="front", n=48),
    # --- adp: SWE / code editing ---
    "swe-smith": GroundingSpec(direction="front", n=48),
    # exit code lives in a "[Command finished with exit code N]" suffix -> keep tail.
    "swe-gym_openhands_sampled_trajectories": GroundingSpec(direction="front_tail", n=48, tail=16),
    "swe-play-trajectories": GroundingSpec(direction="front", n=48, strip_prefix_re=_SWE_PLAY_HEADER),
    "nebius_SWE-agent-trajectories": GroundingSpec(direction="front", n=48, strip_suffix_re=_SWE_FOOTER),
    "code_feedback": GroundingSpec(direction="front", n=32),
    "codeactinstruct": GroundingSpec(direction="front", n=24, strip_suffix_re=_STEP_FOOTER),
    "mini-coder": GroundingSpec(direction="front", n=48),
    # --- adp: agent-env text (folded user) ---
    "agenttuning_alfworld": GroundingSpec(direction="front", n=32),
    "agenttuning_db": GroundingSpec(direction="front", n=16),
    "agenttuning_kg": GroundingSpec(direction="front", n=24),
    "agenttuning_webshop": GroundingSpec(direction="front", n=32),
    # --- adp: web browsing (accessibility tree) -- first line is title+url outcome ---
    "go-browse-wa": GroundingSpec(direction="first_line", n=32),
    "nnetnav-live": GroundingSpec(direction="first_line", n=32),
    "nnetnav-wa": GroundingSpec(direction="first_line", n=32),
    "synatra": GroundingSpec(direction="first_line", n=32),
    # --- skip: raw DOM (no front-loaded outcome) or no extractable observations ---
    "mind2web": GroundingSpec(direction="skip"),
    "agenttuning_mind2web": GroundingSpec(direction="skip"),
    "coderforge_preview": GroundingSpec(direction="skip"),
    "orca_agentinstruct": GroundingSpec(direction="skip"),
    # fallback for any unregistered / new benchmark.
    "default": GroundingSpec(direction="front", n=48),
}

_TRAJECTORY_SUFFIXES = (
    "_world_model_train_trajectories.json",
    "_world_model_test_trajectories.json",
    "_world_model_enterprise_state_train_trajectories.json",
    "_world_model_enterprise_state_test_trajectories.json",
    "_world_model_trajectories.json",
)


def benchmark_key_from_path(path: str | Path) -> str:
    """Map a trajectory file path to a GROUNDING_SPECS key."""
    stem = Path(path).name
    for suffix in _TRAJECTORY_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem.startswith("enterpriseops_gym"):
        return "enterpriseops_gym"
    if stem.startswith("crmarenapro"):
        return "crmarenapro"
    if stem.startswith("terminalbench"):
        return "terminalbench"
    if stem.startswith("toucan"):
        return "toucan"
    return stem  # ADP datasets are named exactly by their key


def get_grounding_spec(benchmark: str) -> GroundingSpec:
    return GROUNDING_SPECS.get(benchmark, GROUNDING_SPECS["default"])


def isolate_observation_span(text: str, benchmark: str) -> str:
    """Return the essential environment-output span for `benchmark`, or "" if this
    benchmark is skipped or the text has no usable content."""
    spec = get_grounding_spec(benchmark)
    if spec.direction == "skip" or not text:
        return ""
    if spec.strip_ansi:
        text = _ANSI.sub("", text)
    if spec.strip_prefix and text.startswith(spec.strip_prefix):
        text = text[len(spec.strip_prefix):]
    if spec.strip_prefix_re is not None:
        text = spec.strip_prefix_re.sub("", text, count=1)
    if spec.strip_suffix_re is not None:
        text = spec.strip_suffix_re.sub("", text)
    if spec.direction == "first_line":
        text = text.lstrip("\n").split("\n", 1)[0]
    return text.strip()


def observation_ground_target_ids(
    text: str,
    benchmark: str,
    tokenizer: Any,
    *,
    max_tokens: int | None = None,
) -> list[int]:
    """Token ids to supervise for this observation: the first-N (and, for
    front_tail specs, last-M) tokens of the isolated span. Empty list => no target.

    max_tokens, when set, overrides the per-benchmark N as a global cap (the front
    budget); the tail budget is taken from the spec.
    """
    span = isolate_observation_span(text, benchmark)
    if not span:
        return []
    spec = get_grounding_spec(benchmark)
    n = spec.n if max_tokens is None else min(spec.n, max_tokens)
    ids = tokenizer(span, add_special_tokens=False)["input_ids"]
    if not ids:
        return []
    if spec.direction == "front_tail" and spec.tail > 0 and len(ids) > n:
        head = ids[:n]
        tail = ids[-spec.tail:]
        return head + tail
    return ids[:n]
