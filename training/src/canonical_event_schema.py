"""The 11-field canonical_event_with_nudge label schema, shared by both world-model trainers.

`src/finetuning_jepa.py` predicts these fields with classification heads on the latent; the
causal-LM trainer in `src/finetuning.py` predicts them as generated JSON
(`--world-model-target canonical_event_with_nudge`). Both read the field names, allowed values
and default JSONL paths from here so the two models are scored on exactly the same label space
-- otherwise the LLM-vs-JEPA comparison would not be like-for-like.

Imports nothing from either trainer (finetuning_jepa already imports finetuning at module
scope, so a back-import would be circular).

SCOPE: this is a description of the EXISTING label schema, not a redesign of it. The field
set and allowed values are vendored from canonical_event_state.py on feat/llama-factory, which
is what produced the labels in the JSONL files.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORIES_DIR = REPO_ROOT / "trajectories"

# --- field groups -------------------------------------------------------------------------
CANONICAL_EVENT_STATE_FIELDS: tuple[str, ...] = (
    "action_type",
    "error_signature",
    "execution_status",
    "object_type",
    "progress_signal",
    "risk_signal",
    "side_effect_type",
)
NUDGE_SINGLE_LABEL_FIELDS: tuple[str, ...] = (
    "information_gain",
    "information_sufficiency",
    "recommended_abstract_action",
)
NUDGE_MULTI_LABEL_FIELDS: tuple[str, ...] = ("missing_information_type",)
CANONICAL_EVENT_SINGLE_LABEL_FIELDS: tuple[str, ...] = (
    CANONICAL_EVENT_STATE_FIELDS + NUDGE_SINGLE_LABEL_FIELDS
)
CANONICAL_EVENT_ALL_FIELDS: tuple[str, ...] = (
    CANONICAL_EVENT_SINGLE_LABEL_FIELDS + NUDGE_MULTI_LABEL_FIELDS
)

# --- allowed values -------------------------------------------------------------------------
# Sorted so the generation prompt and any vocab built from this module are deterministic.
CANONICAL_EVENT_ALLOWED_VALUES: dict[str, tuple[str, ...]] = {
    "execution_status": ("failure", "no_op", "partial", "success", "unknown"),
    "error_signature": (
        "dependency_missing", "invalid_argument", "none", "not_found", "parse_error",
        "permission_denied", "policy_risk", "runtime_error", "test_failed", "timeout", "unknown",
    ),
    "action_type": (
        "clarify", "communicate", "create", "delete", "read", "run", "search", "test",
        "unknown", "update", "validate",
    ),
    "object_type": (
        "account", "branch", "calendar", "case", "comment", "customer", "database_row", "file",
        "label", "message", "package", "permission", "process", "quote", "repository", "ticket",
        "unknown",
    ),
    "side_effect_type": (
        "created", "deleted", "executed", "failed_validation", "installed", "modified", "none",
        "retrieved", "sent", "unknown", "validated",
    ),
    "progress_signal": ("negative", "neutral", "positive", "unknown"),
    "risk_signal": (
        "confidentiality_risk", "destructive_action", "irreversible_action", "none", "policy_risk",
    ),
    "information_sufficiency": ("insufficient", "sufficient", "unknown"),
    "information_gain": ("high", "low", "medium", "negative", "unknown"),
    "recommended_abstract_action": (
        "avoid", "clarify", "finalize", "inspect", "proceed", "retrieve", "rollback", "search",
        "unknown", "validate",
    ),
    "missing_information_type": (
        "current_state", "dependency", "file_context", "none", "object_id", "permission",
        "policy", "schema", "test_result", "unknown", "user_intent",
    ),
}

# --- default label files ----------------------------------------------------------------------
# Cleaned by src/data_preparation/clean_canonical_event_examples.py: drops uninformative
# (all-unknown) labels and duplicate-input records (many with conflicting labels) present in the
# raw LLM-labeled file. See the sibling _cleaned_manifest.json for exact counts.
_STEM = "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro"
DEFAULT_CANONICAL_EVENT_TRAIN_JSONL = DEFAULT_TRAJECTORIES_DIR / f"{_STEM}_train_examples_cleaned.jsonl"
DEFAULT_CANONICAL_EVENT_EVAL_JSONL = DEFAULT_TRAJECTORIES_DIR / f"{_STEM}_eval_examples_cleaned.jsonl"


def split_canonical_labels(row: dict) -> dict[str, object] | None:
    """Pull the flat 11-field label dict out of one canonical-event JSONL row.

    Rows carry the labels either nested under `canonical_event_with_nudge`
    ({canonical_event_state: {...}, nudge: {...}}) or as sibling `canonical_event_state` /
    `nudge` keys. Returns None when a required field is missing, so callers can skip the row
    rather than train on a partially-labeled target.
    """
    if not isinstance(row, dict):
        return None
    combined = row.get("canonical_event_with_nudge")
    if isinstance(combined, dict):
        state = combined.get("canonical_event_state") or {}
        nudge = combined.get("nudge") or {}
    else:
        state = row.get("canonical_event_state") or {}
        nudge = row.get("nudge") or {}
    if not isinstance(state, dict) or not isinstance(nudge, dict):
        return None
    merged = {**state, **nudge}
    labels: dict[str, object] = {}
    for field in CANONICAL_EVENT_ALL_FIELDS:
        if field not in merged:
            return None
        labels[field] = merged[field]
    return labels
