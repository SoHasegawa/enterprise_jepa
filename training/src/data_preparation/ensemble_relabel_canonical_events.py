#!/usr/bin/env python3
"""Multi-model ensemble re-annotation of the canonical_event_with_nudge labels.

Purpose: the existing labels come from a SINGLE model (gpt-5.4-mini). This re-labels the same
transitions with N independent frontier models using the IDENTICAL prompt, schema and payload
as the original run, then measures inter-model agreement and emits (a) consensus labels and
(b) a disagreement queue for human adjudication.

Why the prompt is vendored verbatim: agreement between models is only interpretable if every
rater saw exactly the same instructions. The system/user prompts, the compact example payload
and the JSON schema below are copied unchanged from
src/data_preparation/label_canonical_events_with_llm.py (branch feat/llama-factory) so the new
raters are directly comparable with the existing gpt-5.4-mini labels, which become an extra
rater in the agreement analysis rather than a reference "truth".

SCOPE: the 11-field state design is UNCHANGED -- same fields, same allowed values. This script
only re-measures and (optionally) re-derives the labels for that same schema.

Agreement statistics (per field):
  * raw agreement, Cohen's kappa (pairwise), Fleiss' kappa (all raters)
  * Gwet's AC1 -- reported because kappa is deflated by the severe class imbalance in these
    fields (risk_signal is ~96% one class), which was the confound flagged in review
  * per-class precision/recall of each model against the consensus, and confusion pairs
  * unanimity / majority coverage, and the rate at which the ORIGINAL label is outvoted

Outputs (under --outdir):
  ensemble_raw_labels.jsonl       one row per (transition, model) -- cached, resumable
  ensemble_agreement.json         all agreement statistics
  ensemble_consensus_labels.jsonl majority-vote label per transition (+ per-field vote detail)
  ensemble_disagreements.jsonl    transitions with no majority on >=1 field -> human queue

Usage:
  # plumbing check, no API calls, prints a full rendered prompt
  uv run python src/data_preparation/ensemble_relabel_canonical_events.py --dry-run --sample-size 3

  # real run (3 models x N transitions), resumable -- rerun to fill gaps
  uv run python src/data_preparation/ensemble_relabel_canonical_events.py \
      --models openai/gpt-5.2 anthropic/claude-opus-5 gemini-pro \
      --sample-size 400 --workers 4

  # statistics only, from the cache
  uv run python src/data_preparation/ensemble_relabel_canonical_events.py --stage analyze
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ------------------------------------------------------------------- model clients
# A --models entry is either a `<provider>/<model-id>` spec handled by the provider's own
# SDK, or a bare src/llm.py method name. The prefixed form exists because the agreement
# study needs the model ID pinned and the system prompt actually delivered; src/llm.py's
# `claude` method hardcodes claude-sonnet-4-6 AND drops the system prompt (see
# LLM._generate_claude), which would silently invalidate the comparison.
OPENAI_PREFIXES = ("openai/",)
AZURE_PREFIXES = ("azureopenai/", "azure/")
ANTHROPIC_PREFIXES = ("anthropic/", "claude/")


def _is_openai_reasoning_model(name: str) -> bool:
    """gpt-5* / o1 / o3 / o4 take `max_completion_tokens` and reject `temperature`."""
    return name.lower().startswith(("o1", "o3", "o4", "gpt-5"))


def _strip_prefix(spec: str, prefixes: tuple[str, ...]) -> str | None:
    for prefix in prefixes:
        if spec.startswith(prefix):
            return spec[len(prefix):]
    return None


class AnthropicClient:
    """Anthropic Messages API, called with the model ID exactly as given on the CLI.

    Text is joined across every `text` block rather than read from `content[0]`: on models
    where thinking is on by default (Opus 5 and later) the first block is a thinking block,
    so index-0 access returns the wrong thing or raises.
    """

    def __init__(self, model: str, max_output_tokens: int) -> None:
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_output_tokens = max_output_tokens
        # Opus 5 / Sonnet 5 / Fable 5 / Opus 4.8 / 4.7 reject sampling parameters with a 400;
        # older models accept them. Discovered on first use rather than hardcoded, so a new
        # model ID does not need a code change here.
        self.accepts_temperature = True

    def _create(self, prompt: str, system_prompt: str, temperature: float | None):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        return self.client.messages.create(**kwargs)

    def __call__(self, prompt: str, system_prompt: str, image_paths=None, temperature: float = 0.0) -> str:
        want_temperature = temperature if self.accepts_temperature else None
        try:
            message = self._create(prompt, system_prompt, want_temperature)
        except self._anthropic.BadRequestError as exc:
            if want_temperature is None or "temperature" not in str(exc).lower():
                raise
            self.accepts_temperature = False
            print(f"  [{self.model}] rejects `temperature`; retrying without it "
                  f"(labels from this model are sampled at the model default, not pinned to "
                  f"{temperature}).", flush=True)
            message = self._create(prompt, system_prompt, None)
        if message.stop_reason == "refusal":
            raise RuntimeError(f"{self.model} refused the request")
        return "".join(block.text for block in message.content if block.type == "text")


class OpenAIChatClient:
    """OpenAI (or Azure OpenAI) Chat Completions, called with the model/deployment as given."""

    def __init__(self, model: str, max_output_tokens: int, azure: bool = False) -> None:
        import openai

        if azure:
            import os

            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            self.client = (
                openai.OpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"], base_url=endpoint)
                if "/openai/v1" in endpoint
                else openai.AzureOpenAI(
                    api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    azure_endpoint=endpoint,
                    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                )
            )
        else:
            self.client = openai.OpenAI()
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.is_reasoning_model = _is_openai_reasoning_model(model)

    def __call__(self, prompt: str, system_prompt: str, image_paths=None, temperature: float = 0.0) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if self.is_reasoning_model:
            kwargs["max_completion_tokens"] = self.max_output_tokens
        else:
            kwargs["max_tokens"] = self.max_output_tokens
            kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def build_client(spec: str, args: argparse.Namespace):
    """Resolve one --models entry to a callable `(prompt, system_prompt, ...) -> str`."""
    model = _strip_prefix(spec, ANTHROPIC_PREFIXES)
    if model is not None:
        return AnthropicClient(model, args.max_output_tokens)
    model = _strip_prefix(spec, AZURE_PREFIXES)
    if model is not None:
        return OpenAIChatClient(model, args.max_output_tokens, azure=True)
    model = _strip_prefix(spec, OPENAI_PREFIXES)
    if model is not None:
        return OpenAIChatClient(model, args.max_output_tokens)
    from src.llm import LLM

    return LLM(spec)

TRAJ = REPO_ROOT / "trajectories"
STEM = "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro"
DEFAULT_INPUT = TRAJ / f"{STEM}_train_examples_cleaned_value_scored_recognition_probe.jsonl"
DEFAULT_OUTDIR = REPO_ROOT / "data" / "week1" / "ensemble_labels"
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# --- value sets, vendored verbatim from canonical_event_state.py (feat/llama-factory) --------
EXECUTION_STATUS_VALUES = {"success", "failure", "partial", "no_op", "unknown"}
ERROR_SIGNATURE_VALUES = {"none", "not_found", "permission_denied", "invalid_argument",
                          "dependency_missing", "test_failed", "timeout", "policy_risk",
                          "parse_error", "runtime_error", "unknown"}
ACTION_TYPE_VALUES = {"read", "search", "create", "update", "delete", "run", "test",
                      "communicate", "clarify", "validate", "unknown"}
OBJECT_TYPE_VALUES = {"file", "database_row", "ticket", "customer", "quote", "account", "package",
                      "process", "message", "calendar", "case", "permission", "comment", "label",
                      "repository", "branch", "unknown"}
SIDE_EFFECT_TYPE_VALUES = {"none", "retrieved", "modified", "created", "deleted", "sent",
                           "installed", "validated", "failed_validation", "executed", "unknown"}
PROGRESS_SIGNAL_VALUES = {"positive", "negative", "neutral", "unknown"}
RISK_SIGNAL_VALUES = {"none", "policy_risk", "confidentiality_risk", "destructive_action",
                      "irreversible_action"}
INFORMATION_SUFFICIENCY_VALUES = {"sufficient", "insufficient", "unknown"}
INFORMATION_GAIN_VALUES = {"high", "medium", "low", "negative", "unknown"}
RECOMMENDED_ABSTRACT_ACTION_VALUES = {"proceed", "search", "inspect", "retrieve", "validate",
                                      "clarify", "avoid", "rollback", "finalize", "unknown"}
MISSING_INFORMATION_TYPE_VALUES = {"none", "object_id", "schema", "policy", "file_context",
                                   "test_result", "user_intent", "dependency", "current_state",
                                   "permission", "unknown"}
# order matters: it is the order the original prompt listed them in
REQUIRED_CATEGORICAL_FIELDS = {
    "execution_status": EXECUTION_STATUS_VALUES,
    "error_signature": ERROR_SIGNATURE_VALUES,
    "action_type": ACTION_TYPE_VALUES,
    "object_type": OBJECT_TYPE_VALUES,
    "side_effect_type": SIDE_EFFECT_TYPE_VALUES,
    "progress_signal": PROGRESS_SIGNAL_VALUES,
    "risk_signal": RISK_SIGNAL_VALUES,
}
NUDGE_SINGLE_FIELDS = {
    "information_sufficiency": INFORMATION_SUFFICIENCY_VALUES,
    "information_gain": INFORMATION_GAIN_VALUES,
    "recommended_abstract_action": RECOMMENDED_ABSTRACT_ACTION_VALUES,
}
MULTI_FIELD = "missing_information_type"
STATE_FIELDS = tuple(REQUIRED_CATEGORICAL_FIELDS)
SINGLE_FIELDS = tuple(REQUIRED_CATEGORICAL_FIELDS) + tuple(NUDGE_SINGLE_FIELDS)
ALL_FIELDS = SINGLE_FIELDS + (MULTI_FIELD,)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("label", "analyze"), default="label")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument(
        "--models", nargs="+",
        default=["openai/gpt-5.2", "anthropic/claude-opus-5", "gemini-pro"],
        help=(
            "One entry per ensemble member. Two forms: `<provider>/<model-id>` calls that "
            "provider's SDK with the model ID verbatim -- `openai/<model>` (OPENAI_API_KEY), "
            "`azureopenai/<deployment>` (AZURE_OPENAI_*), `anthropic/<model>` "
            "(ANTHROPIC_API_KEY, e.g. anthropic/claude-opus-5); anything else is a src/llm.py "
            "method name (gemini, gemini-pro, gpt5, ...). Prefer the prefixed form for OpenAI "
            "and Anthropic: src/llm.py's `claude` method pins claude-sonnet-4-6 and does not "
            "forward the system prompt, which breaks the identical-prompt premise of the "
            "agreement study."
        ),
    )
    p.add_argument("--max-output-tokens", type=int, default=8000,
                   help="Per-call output cap. Needs headroom above the JSON label on models "
                        "where thinking is on by default (Claude Opus 5 and later), since the "
                        "cap covers thinking plus the response.")
    p.add_argument("--sample-size", type=int, default=400, help="0 = label every row.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-input-history-items", type=int, default=8)   # original default
    p.add_argument("--max-field-chars", type=int, default=4000)        # original default
    p.add_argument("--context", choices=("trajectory", "step"), default="trajectory",
                   help="trajectory (default): also show the whole episode + task outcome, so "
                        "task-relative fields are judged by this step's contribution. "
                        "step: exactly the original single-step prompt (comparable with the "
                        "existing gpt-5.4-mini labels).")
    p.add_argument("--max-trajectory-steps", type=int, default=30)
    p.add_argument("--trajectory-source", type=Path, default=None,
                   help="File to reconstruct full trajectories from; defaults to --input with "
                        "the _recognition_probe suffix removed (that file has every step).")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="Build prompts, make no API calls.")
    return p.parse_args()


# --------------------------------------------------------------------------- prompt (vendored)
def truncate_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "...<truncated>"
    if isinstance(value, list):
        return [truncate_value(v, max_chars) for v in value]
    if isinstance(value, dict):
        return {k: truncate_value(v, max_chars) for k, v in value.items()}
    return value


def build_trajectory_index(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct each trajectory as an ordered step list carrying BOTH action and observation.

    A row stores only its own action; the observation of step t lives in step t+1's
    input_history[-1] (verified alignment -- see build_canonical_event_recognition_probe.py).
    The final step of a trajectory therefore has no recoverable observation.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("trajectory_id"))].append(row)
    out: dict[str, list[dict[str, Any]]] = {}
    for trajectory_id, group in grouped.items():
        group.sort(key=lambda r: int(r.get("interaction_index") or 0))
        by_index = {int(r.get("interaction_index") or 0): r for r in group}
        steps = []
        for row in group:
            index = int(row.get("interaction_index") or 0)
            successor = by_index.get(index + 1)
            observation = None
            if successor is not None:
                history = successor.get("input_history") or []
                if history and isinstance(history[-1], dict):
                    observation = history[-1].get("observation")
            steps.append({"step": index, "action": row.get("action"), "observation": observation})
        out[trajectory_id] = steps
    return out


def compact_example_payload(
    row: dict[str, Any],
    *,
    max_input_history_items: int,
    max_field_chars: int,
    context: str = "trajectory",
    trajectory_steps: list[dict[str, Any]] | None = None,
    max_trajectory_steps: int = 30,
) -> dict[str, Any]:
    """The original labeler's payload, optionally extended with WHOLE-TRAJECTORY context.

    context="step"       -- exactly the original fields (comparable with the gpt-5.4-mini run).
    context="trajectory" -- additionally supplies the full step list (actions + observations),
                            the steps that FOLLOW this action, and the trajectory-level task
                            outcome, so task-relative fields (progress_signal, information_*,
                            recommended_abstract_action) can be judged by this step's actual
                            contribution to the task rather than from local evidence alone.
    """
    payload = {
        "trajectory_id": row.get("trajectory_id"),
        "benchmark": row.get("benchmark"),
        "trajectory_index": row.get("trajectory_index"),
        "interaction_index": row.get("interaction_index"),
        "system_prompt": row.get("system_prompt"),
        "task_prompt": row.get("task_prompt"),
        "recent_state_history": [],
        "recent_input_history": (row.get("input_history") or [])[-max_input_history_items:],
        "previous_state": row.get("previous_state"),
        "candidate_action": row.get("action"),
        "observed_resulting_state": None,
        "observed_tool_output": row.get("observation"),
        "observed_error_payload": None,
    }
    if context == "trajectory":
        index = int(row.get("interaction_index") or 0)
        steps = list(trajectory_steps or [])
        payload["full_trajectory"] = steps[:max_trajectory_steps]
        payload["total_steps_in_trajectory"] = len(steps)
        payload["steps_after_this_action"] = [s for s in steps if s["step"] > index][:max_trajectory_steps]
        payload["task_outcome"] = {
            "task_completed": row.get("trajectory_success"),
            "verifier_pass_rate": row.get("trajectory_pass_rate"),
            "outcome_evidence_source": row.get("success_source"),
        }
    return truncate_value(payload, max_field_chars)


def values_text(values: set[str]) -> str:
    return ", ".join(sorted(values))


TRAJECTORY_CONTEXT_GUIDANCE = "\n".join([
    "",
    "Whole-trajectory evidence (use it for the task-relative fields):",
    "- full_trajectory lists every step of this episode as (step, action, observation); "
    "steps_after_this_action lists only what happened AFTER the candidate action; "
    "task_outcome reports whether the task was ultimately completed and the verifier pass rate.",
    "- Judge progress_signal by this step's CONTRIBUTION TO THE TASK, not by whether the call "
    "returned cleanly: an action that executed fine but retrieved nothing useful, repeated an "
    "earlier step, or was later undone/redone is neutral or negative even though it succeeded.",
    "- Judge information_sufficiency / information_gain / missing_information_type against what "
    "the LATER steps reveal was actually needed: if a subsequent step had to look something up "
    "before the task could proceed, that information was missing here.",
    "- Judge recommended_abstract_action as the best next behavior given the whole episode, and "
    "note whether the trajectory's own next step turned out to be productive or wasteful.",
    "- IMPORTANT: task_outcome and later steps must NOT change execution_status, error_signature, "
    "action_type, object_type or side_effect_type. Those describe what THIS action did "
    "operationally and stay judged from this step's own action and observation. A failed task "
    "can contain successful actions, and a completed task can contain failed ones.",
])


def build_labeling_messages(example_payload: dict[str, Any]) -> list[dict[str, str]]:
    output_contract = "\n".join([
        "Return exactly one JSON object with top-level fields: canonical_event_state, nudge.",
        "Do not include schema_version anywhere in the output.",
        "canonical_event_state fields and allowed values:",
        *[f"- {f}: {values_text(v)}" for f, v in REQUIRED_CATEGORICAL_FIELDS.items()],
        "nudge fields and allowed values:",
        f"- information_sufficiency: {values_text(INFORMATION_SUFFICIENCY_VALUES)}",
        f"- information_gain: {values_text(INFORMATION_GAIN_VALUES)}",
        f"- recommended_abstract_action: {values_text(RECOMMENDED_ABSTRACT_ACTION_VALUES)}",
        "- missing_information_type: non-empty list using only "
        f"{values_text(MISSING_INFORMATION_TYPE_VALUES)}; use ['none'] only by itself.",
    ])
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
        "- Prefer unknown only when the provided evidence is genuinely insufficient."
        + (TRAJECTORY_CONTEXT_GUIDANCE if "full_trajectory" in example_payload else "")
        + "\n\nExample to label:\n"
        f"{json.dumps(example_payload, ensure_ascii=False, sort_keys=True)}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def parse_and_validate(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(text)
        if not match:
            raise ValueError("no JSON object in response")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    state = payload.get("canonical_event_state") or {}
    nudge = payload.get("nudge") or {}
    out_state, out_nudge = {}, {}
    for field, allowed in REQUIRED_CATEGORICAL_FIELDS.items():
        value = str(state.get(field, "")).strip()
        if value not in allowed:
            raise ValueError(f"canonical_event_state.{field}={value!r} not in schema")
        out_state[field] = value
    for field, allowed in NUDGE_SINGLE_FIELDS.items():
        value = str(nudge.get(field, "")).strip()
        if value not in allowed:
            raise ValueError(f"nudge.{field}={value!r} not in schema")
        out_nudge[field] = value
    missing = nudge.get(MULTI_FIELD)
    if not isinstance(missing, list) or not missing:
        raise ValueError("nudge.missing_information_type must be a non-empty list")
    values = [str(v).strip() for v in missing]
    if any(v not in MISSING_INFORMATION_TYPE_VALUES for v in values):
        raise ValueError(f"nudge.missing_information_type has out-of-schema values: {values}")
    out_nudge[MULTI_FIELD] = sorted(set(values))
    return {"canonical_event_state": out_state, "nudge": out_nudge}


def label_value(label: dict[str, Any], field: str) -> Any:
    block = label.get("canonical_event_state") if field in STATE_FIELDS else label.get("nudge")
    value = (block or {}).get(field)
    return tuple(sorted(value)) if isinstance(value, list) else value


# --------------------------------------------------------------------------- agreement stats
def cohen_kappa(a: list[Any], b: list[Any]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    return 1.0 if expected >= 1.0 else (observed - expected) / (1.0 - expected)


def gwet_ac1(ratings: list[list[Any]]) -> float:
    """Gwet's AC1 for R raters x N items. Less sensitive to prevalence than kappa, which is why
    it is reported alongside: these fields are severely imbalanced."""
    n = len(ratings)
    if not n:
        return float("nan")
    r = len(ratings[0])
    categories = sorted({v for row in ratings for v in row}, key=str)
    if len(categories) < 2:
        return 1.0
    pa = sum(sum(Counter(row)[c] * (Counter(row)[c] - 1) for c in categories) / (r * (r - 1)) for row in ratings) / n
    pi = {c: sum(Counter(row)[c] for row in ratings) / (n * r) for c in categories}
    pe = sum(p * (1 - p) for p in pi.values()) / (len(categories) - 1)
    return (pa - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss_kappa(ratings: list[list[Any]]) -> float:
    n = len(ratings)
    if not n:
        return float("nan")
    r = len(ratings[0])
    categories = sorted({v for row in ratings for v in row}, key=str)
    if len(categories) < 2:
        return 1.0
    pa = sum((sum(Counter(row)[c] ** 2 for c in categories) - r) / (r * (r - 1)) for row in ratings) / n
    pj = {c: sum(Counter(row)[c] for row in ratings) / (n * r) for c in categories}
    pe = sum(p ** 2 for p in pj.values())
    return (pa - pe) / (1 - pe) if pe < 1 else 1.0


# --------------------------------------------------------------------------- labeling
def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('trajectory_id')}::{row.get('interaction_index')}"


def resolve_trajectory_index(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    """Full trajectories come from the unfiltered file: the probe file drops terminal steps, so
    reconstructing episodes from it alone would hide each trajectory's ending."""
    source = args.trajectory_source
    if source is None:
        candidate = args.input.with_name(args.input.name.replace("_recognition_probe", ""))
        source = candidate if candidate.is_file() else args.input
    print(f"trajectory context from: {source}")
    return build_trajectory_index(load_rows(source))


def stage_label(args: argparse.Namespace) -> None:
    rows = load_rows(args.input)
    if args.sample_size and len(rows) > args.sample_size:
        rows = random.Random(args.seed).sample(rows, args.sample_size)
    trajectories = resolve_trajectory_index(args) if args.context == "trajectory" else {}
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_path = args.outdir / "ensemble_raw_labels.jsonl"

    done: set[tuple[str, str]] = set()
    if cache_path.is_file():
        for entry in load_rows(cache_path):
            if entry.get("label") is not None:
                done.add((entry["key"], entry["model"]))
        print(f"cache: {len(done)} (transition, model) pairs already labeled")

    jobs = [(row, model) for row in rows for model in args.models if (row_key(row), model) not in done]
    print(f"{len(rows)} transitions x {len(args.models)} models -> {len(jobs)} calls to make")

    if args.dry_run:
        row = rows[0]
        messages = build_labeling_messages(compact_example_payload(
            row, max_input_history_items=args.max_input_history_items,
            max_field_chars=args.max_field_chars, context=args.context,
            trajectory_steps=trajectories.get(str(row.get("trajectory_id"))),
            max_trajectory_steps=args.max_trajectory_steps))
        print("\n=== SYSTEM ===\n" + messages[0]["content"])
        print("\n=== USER (first 2500 chars) ===\n" + messages[1]["content"][:2500])
        print(f"\n[dry-run] would issue {len(jobs)} calls across models {args.models}; no API calls made.")
        return

    clients = {model: build_client(model, args) for model in args.models}
    lock = threading.Lock()
    handle = cache_path.open("a", encoding="utf-8")
    counters = Counter()

    def run(job):
        row, model = job
        payload = compact_example_payload(
            row, max_input_history_items=args.max_input_history_items,
            max_field_chars=args.max_field_chars, context=args.context,
            trajectory_steps=trajectories.get(str(row.get("trajectory_id"))),
            max_trajectory_steps=args.max_trajectory_steps)
        messages = build_labeling_messages(payload)
        label, error = None, None
        for attempt in range(args.max_retries):
            try:
                text = clients[model](messages[1]["content"], messages[0]["content"], temperature=args.temperature)
                label = parse_and_validate(text if isinstance(text, str) else str(text))
                break
            except Exception as exc:                      # noqa: BLE001 -- log and continue
                error = f"{type(exc).__name__}: {exc}"[:300]
        with lock:
            counters["ok" if label else "fail"] += 1
            handle.write(json.dumps({
                "key": row_key(row), "trajectory_id": row.get("trajectory_id"),
                "interaction_index": row.get("interaction_index"), "benchmark": row.get("benchmark"),
                "model": model, "label": label, "error": error, "context": args.context,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            total = counters["ok"] + counters["fail"]
            if total % 25 == 0:
                print(f"  {total}/{len(jobs)} done (ok={counters['ok']}, fail={counters['fail']})", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, jobs))
    handle.close()
    print(f"labeled: ok={counters['ok']} fail={counters['fail']} -> {cache_path}")
    stage_analyze(args)


# --------------------------------------------------------------------------- analysis
def stage_analyze(args: argparse.Namespace) -> None:
    cache_path = args.outdir / "ensemble_raw_labels.jsonl"
    if not cache_path.is_file():
        raise SystemExit(f"no cache at {cache_path}; run --stage label first")
    original = {row_key(r): (r.get("canonical_event_with_nudge") or {}) for r in load_rows(args.input)}

    by_key: dict[str, dict[str, dict]] = defaultdict(dict)
    for entry in load_rows(cache_path):
        if entry.get("label"):
            by_key[entry["key"]][entry["model"]] = entry["label"]
    models = sorted({m for v in by_key.values() for m in v})
    complete = [k for k, v in by_key.items() if len(v) == len(models)]
    print(f"transitions with all {len(models)} models: {len(complete)} (models: {models})\n")
    if not complete:
        raise SystemExit("no transition has labels from every model yet")

    raters = models + ["ORIGINAL(gpt-5.4-mini)"]
    report: dict[str, Any] = {"models": models, "n": len(complete), "fields": {}}
    consensus_rows, disagreement_rows = [], []

    print(f"{'field':28s} {'unanim':>7s} {'major':>7s} {'Fleiss':>7s} {'AC1':>7s} {'orig!=cons':>10s}")
    print("-" * 74)
    for field in ALL_FIELDS:
        ratings, orig_vals = [], []
        for key in complete:
            ratings.append([label_value(by_key[key][m], field) for m in models])
            orig_vals.append(label_value(original.get(key, {}), field))
        unanimous = sum(1 for r in ratings if len(set(r)) == 1) / len(ratings)
        majority = sum(1 for r in ratings if Counter(r).most_common(1)[0][1] >= 2) / len(ratings)
        cons = [Counter(r).most_common(1)[0][0] for r in ratings]
        orig_diff = sum(1 for c, o in zip(cons, orig_vals) if c != o) / len(cons)
        block = {
            "unanimity": unanimous, "majority_coverage": majority,
            "fleiss_kappa": fleiss_kappa(ratings), "gwet_ac1": gwet_ac1(ratings),
            "original_vs_consensus_disagreement": orig_diff,
            "pairwise": {}, "model_vs_consensus_accuracy": {},
            "top_confusions_original_vs_consensus": [
                {"original": o, "consensus": c, "count": n}
                for (o, c), n in Counter((o, c) for c, o in zip(cons, orig_vals) if c != o).most_common(5)
            ],
        }
        for i, a in enumerate(models):
            block["model_vs_consensus_accuracy"][a] = sum(
                1 for r, c in zip(ratings, cons) if r[i] == c) / len(cons)
            for j, b in enumerate(models):
                if i < j:
                    block["pairwise"][f"{a}|{b}"] = {
                        "raw": sum(1 for r in ratings if r[i] == r[j]) / len(ratings),
                        "cohen_kappa": cohen_kappa([r[i] for r in ratings], [r[j] for r in ratings]),
                    }
            block["pairwise"][f"{a}|ORIGINAL"] = {
                "raw": sum(1 for r, o in zip(ratings, orig_vals) if r[i] == o) / len(ratings),
                "cohen_kappa": cohen_kappa([r[i] for r in ratings], orig_vals),
            }
        report["fields"][field] = block
        print(f"{field:28s} {unanimous:>7.3f} {majority:>7.3f} {block['fleiss_kappa']:>7.3f} "
              f"{block['gwet_ac1']:>7.3f} {orig_diff:>10.3f}")

    # consensus + disagreement files
    for key in complete:
        state, nudge, votes, contested = {}, {}, {}, []
        for field in ALL_FIELDS:
            values = [label_value(by_key[key][m], field) for m in models]
            top, count = Counter(values).most_common(1)[0]
            votes[field] = {"values": [list(v) if isinstance(v, tuple) else v for v in values], "count": count}
            if count < 2:
                contested.append(field)
            resolved = list(top) if isinstance(top, tuple) else top
            (state if field in STATE_FIELDS else nudge)[field] = resolved
        entry = {"key": key, "canonical_event_with_nudge": {"canonical_event_state": state, "nudge": nudge},
                 "votes": votes, "contested_fields": contested,
                 "label_source": "ensemble_majority", "label_models": models}
        consensus_rows.append(entry)
        if contested:
            disagreement_rows.append(entry)

    (args.outdir / "ensemble_agreement.json").write_text(json.dumps(report, indent=1, default=str))
    with (args.outdir / "ensemble_consensus_labels.jsonl").open("w", encoding="utf-8") as f:
        for row in consensus_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    with (args.outdir / "ensemble_disagreements.jsonl").open("w", encoding="utf-8") as f:
        for row in disagreement_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"\nconsensus labels    -> {args.outdir/'ensemble_consensus_labels.jsonl'} ({len(consensus_rows)})")
    print(f"no-majority queue   -> {args.outdir/'ensemble_disagreements.jsonl'} ({len(disagreement_rows)})")
    print(f"agreement report    -> {args.outdir/'ensemble_agreement.json'}")
    print("\nReading: high Fleiss/AC1 + low original-vs-consensus disagreement => the existing\n"
          "gpt-5.4-mini labels are sound. Low unanimity on a field => that field is genuinely\n"
          "ambiguous, so its ceiling is a labeling-protocol limit, not a modeling one.")


def main() -> None:
    args = parse_args()
    (stage_label if args.stage == "label" else stage_analyze)(args)


if __name__ == "__main__":
    main()
