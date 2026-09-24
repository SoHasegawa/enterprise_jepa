"""Hierarchical latent-action CEM sampling for agentic MPC planning.

Inference-only **port** of the EWM repo's ``src/hierarchical_action_sampling.py``. Self-contained
(no imports from an external EWM checkout); imports ``torch`` at module top, so it is imported
lazily by ``ewm_imagined`` (mirroring ``_ewm_k_controller`` / ``_ewm_jepa``).

Replaces the discrete "LLM proposes m candidates per horizon step" pattern in
``_ewm_beam_plan`` with a hierarchical-latent-action Cross-Entropy Method (CEM): the LLM opens
the search space ONCE per planning cycle with a small set of diverse anchor actions; the world
model then samples thousands of CONTINUOUS latent-action trajectories around those anchors,
batch-rolls them out in latent space (no LLM, no text decode), and iteratively refines a
per-family Gaussian proposal distribution toward the highest-scoring region. Only the converged
best trajectory is decoded back to an executable action.

Two kinds of randomness are deliberately kept separate (do not conflate them):
  * SigReg (training-time only, not ported here) pulls the STATE embedding z toward an
    isotropic Gaussian to prevent representation collapse.
  * This module's CEM samples ACTION trajectories from a Gaussian PROPOSAL distribution that
    is refit every iteration toward the elites -- a search procedure, not a regularizer.

Action "family" (categorical layer): every action here arrives as an MCP-style tool call, but
the underlying backend varies by benchmark. Plain MCP tools (EnterpriseOps-Gym, ...) use
family = tool name -- interpolating between two `create_event` calls' argument latents is
meaningful (parameter variation); interpolating between `create_event` and `get_calendar_list`
is not. SQL and shell backends (CRMArenaPro's `execute_crm_sql`, Terminal-Bench/DevOps-Gym's
`run_shell`/`exec`) wrap free-form query/command TEXT inside one dominant tool name, so
tool-name-as-family would collapse every distinct SQL statement or shell command into a single
bucket -- exactly the degenerate case the family layer exists to prevent. ``_family_key``
detects these backends (SQL keyword in `arguments.query`/`sql`/`statement`, or a shell-hinting
tool name with `arguments.command`/`cmd`/`script`) and uses the operation VERB (SQL keyword;
base shell command, unwrapping `bash -lc '...'` and taking only the first pipeline segment) as
the family instead.

Explicit bindings (ids, paths, emails, literals) are NEVER sampled from the continuous latent.
Two decode strategies are available (``HierarchicalCEMConfig.decode_strategy``):
  * "nearest_anchor" (default): the *choice* of which LLM-proposed anchor to execute is guided
    by the latent search; the anchor's own bindings are used verbatim, never interpolated.
    Always valid (it's a real anchor), but can only ever recover an action the LLM already
    proposed.
  * "learned_decoder": decodes the CEM-sampled/interpolated latent directly via
    ``TextLeWorldModel.decode_action_latent`` (see ``_ewm_jepa.py``; trained with
    ``--action-decoder-loss-coeff``). This is what actually realizes "explore an interpolated
    point, not just re-rank LLM anchors" -- but a decoder trained on real anchors is not
    guaranteed to produce a valid action for an arbitrary latent, so every decode is passed
    through ``validate_decoded_action`` and falls back to nearest_anchor on failure.

Fast-LeWorldModel's action-prefix parallel predictor is intentionally NOT used here -- rollout
is the existing recursive one-step-at-a-time ``predict_latent`` chain (batched over all N
candidates), matching the autoregressive CEM in the original LeWorldModel paper.
"""
from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from ejepa_wm.backends import _ewm_runtime as ewm
from ejepa_wm.backends._ewm_canonical_event_scoring import (
    MISSING_INFO_FIELD,
    CanonicalEventScoreConfig,
    logits_to_field_probs_batched,
    rank_trajectories,
)
from ejepa_wm.backends._ewm_jepa import _extend_batch_frame_history


def render_action(action: Any) -> str:
    if isinstance(action, str):
        return action.strip()
    return json.dumps(action, ensure_ascii=False, sort_keys=True, default=str)


@dataclass
class HierarchicalCEMConfig:
    num_llm_anchors: int = 8       # K: the ONE LLM call opens the search space with K diverse actions
    num_samples: int = 256         # N: latent trajectories sampled per CEM iteration
    num_elites: int = 16           # M: elites kept each iteration to refit (pi, mean, std)
    num_iters: int = 3             # I: CEM refinement iterations
    horizon: int = 5               # H: lookahead steps
    init_std: float = 0.2          # initial per-family Gaussian std, as a fraction of |mean| for singleton families
    min_std: float = 0.02          # std floor -- prevents premature Gaussian collapse
    smoothing: float = 1.0         # CEM blend with previous iteration's params: 1.0 = full replace (classic CEM)
    min_elite_agreement: float = 0.5  # fraction of final-iteration elites that must share the winning step-0
                                       # family for the plan to be called "confident". CEM-native substitute for
                                       # beam_plan's softmax-flatness gate: by the final iteration elites are
                                       # EXPECTED to cluster near the optimum, so a flat softmax over N
                                       # near-duplicate converged candidates would wrongly read as "unconfident".
    top_k: int = 3
    max_input_length: int = 2048
    max_action_length: int = 512
    temperature: float = 1.0       # head-logit temperature for scoring calibration
    decode_strategy: str = "nearest_anchor"  # "nearest_anchor" | "learned_decoder"
    decode_max_new_tokens: int = 96
    score_config: CanonicalEventScoreConfig = field(default_factory=CanonicalEventScoreConfig)


def normalize_action_step(step: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize an LLM-produced action-step dict into the training-action format
    {"tool_calls": [{"type": "function", "function": {"name", "arguments"}}, ...]}. Accepts the
    several shapes real LLMs return: {"tool_calls":[...]}, {"name","arguments"},
    {"function":{"name","arguments"}}, {"tool"/"tool_name": ...}."""
    if not isinstance(step, dict):
        return None
    if isinstance(step.get("tool_calls"), list) and step["tool_calls"]:
        calls = []
        for call in step["tool_calls"]:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else call
            name = str(function.get("name") or call.get("name") or "").strip()
            if name:
                calls.append({"type": "function", "function": {"name": name, "arguments": function.get("arguments") or call.get("arguments") or {}}})
        return {"tool_calls": calls} if calls else None
    function = step.get("function") if isinstance(step.get("function"), dict) else None
    source = function or step
    name = str(source.get("name") or step.get("tool") or step.get("tool_name") or "").strip()
    if not name:
        return None
    arguments = source.get("arguments") or step.get("args") or step.get("arguments") or {}
    return {"tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


# SQL/shell backends (CRMArenaPro, Terminal-Bench/DevOps-Gym) wrap free-form query/command text
# inside a single dominant tool ("execute_crm_sql", "run_shell"/"exec"), so tool-name alone
# collapses every distinct SQL statement or shell command into ONE family -- exactly the
# degenerate case the family layer exists to avoid. Detect these backends and use the operation
# VERB (SQL keyword; base shell command) as the family instead, falling back to tool name for
# genuine MCP tool calls (where the name already IS the operation).
_SQL_ARG_KEYS = ("query", "sql", "statement")
_SHELL_ARG_KEYS = ("command", "cmd", "script")
_SHELL_NAME_HINTS = ("bash", "shell", "exec", "terminal")
_SQL_VERB_RE = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|DROP|ALTER|MERGE|REPLACE|TRUNCATE|EXPLAIN)\b", re.IGNORECASE
)
_SHELL_OUTER_WRAPPER_RE = re.compile(r"^\s*(?:bash|sh|zsh)\s+(?:-\w+\s+)*['\"]", re.IGNORECASE)
_SHELL_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||[;|\n]")
_SHELL_SKIP_TOKENS = {"sudo", "env", "nohup", "time"}
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Python-execution, browser-action, and SWE-code-edit backends. Generalized ahead of any live
# execution harness targeting them in this repo -- verified against real ADP action payloads in
# the original EWM exploration: every group already arrives as the same {"tool_calls":[...]}
# shape, so no new JSON-parsing is needed, only a name/argument -> family mapping.
_PYTHON_ARG_KEYS = ("code", "script", "cell")
_PYTHON_NAME_HINTS = ("python", "ipython", "jupyter", "repl")
_PYTHON_FIRST_TOKEN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)")
_CODE_EDIT_NAME_HINTS = ("editor", "edit_file", "apply_patch", "str_replace")
_CODE_EDIT_COMMAND_KEYS = ("command", "action")
# Playwright/browser-agent verb vocabulary: the tool NAME already IS the operation here (like
# plain MCP), so no argument-digging is needed -- just recognizing these names as belonging to
# the "browser" backend rather than an arbitrary MCP tool, for display/grouping consistency.
_BROWSER_VERBS = {
    "click", "dblclick", "fill", "type", "press", "hover", "scroll", "check", "uncheck",
    "select", "select_option", "goto", "go_back", "go_forward", "drag", "upload_file",
    "screenshot", "wait_for", "wait_for_selector", "focus", "clear",
}


def _first_string_arg(arguments: Any, keys: tuple[str, ...]) -> str | None:
    if not isinstance(arguments, dict):
        return None
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _sql_verb(query: str) -> str:
    match = _SQL_VERB_RE.match(query.strip())
    return match.group(1).upper() if match else "sql"


def _shell_command_verb(command: str) -> str:
    """Base command of a (possibly `bash -lc '...'`-wrapped, possibly `a && b; c`-chained)
    shell string: strip the outer wrapper, take only the FIRST pipeline segment (later segments
    are follow-up commands, not this action's primary effect), then the first token that isn't
    an env-var assignment or a no-op prefix (sudo/env/nohup/time)."""
    text = command.strip()
    wrapper = _SHELL_OUTER_WRAPPER_RE.match(text)
    if wrapper:
        text = text[wrapper.end():]
    segment = _SHELL_SEGMENT_SPLIT_RE.split(text, maxsplit=1)[0]
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        tokens = segment.split()
    for token in tokens:
        if token in _SHELL_SKIP_TOKENS or _ENV_ASSIGNMENT_RE.match(token):
            continue
        return token
    return "shell"


def _python_verb(code: str) -> str:
    """First identifier-like token of the first non-empty, non-comment line -- a rough but
    cheap analogue of the shell-verb heuristic for arbitrary Python/pandas/sql-hybrid snippets."""
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _PYTHON_FIRST_TOKEN_RE.match(stripped)
        return match.group(1) if match else "python"
    return "python"


def _family_key(action: dict[str, Any]) -> str:
    calls = action.get("tool_calls") or []
    if not calls or not isinstance(calls[0], dict):
        return "noop"
    call = calls[0]
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    name = str(function.get("name") or call.get("name") or "").strip()
    name_lower = name.lower()
    arguments = function.get("arguments") if isinstance(function.get("arguments"), dict) else {}

    query = _first_string_arg(arguments, _SQL_ARG_KEYS)
    if query is not None and ("sql" in name_lower or _SQL_VERB_RE.match(query)):
        # SQL content is an unambiguous signal on its own (few non-SQL strings start with a
        # SQL keyword) -- trust the query text even if the tool name doesn't hint "sql".
        return "sql:" + _sql_verb(query)

    command = _first_string_arg(arguments, _SHELL_ARG_KEYS)
    if command is not None and any(hint in name_lower for hint in _SHELL_NAME_HINTS):
        # A generic "command" argument key is not unambiguous on its own (an ordinary MCP tool
        # could plausibly have one) -- require the tool-name hint too.
        return "shell:" + _shell_command_verb(command)

    edit_command = _first_string_arg(arguments, _CODE_EDIT_COMMAND_KEYS)
    if edit_command is not None and any(hint in name_lower for hint in _CODE_EDIT_NAME_HINTS):
        # SWE-agent editor tools (str_replace_editor, edit_file, apply_patch, ...) already
        # expose a clean sub-operation enum (view/str_replace/create/insert/undo_edit) -- no
        # text parsing needed, just read it.
        return "code_edit:" + edit_command.strip().lower()

    code = _first_string_arg(arguments, _PYTHON_ARG_KEYS)
    if code is not None and any(hint in name_lower for hint in _PYTHON_NAME_HINTS):
        return "python:" + _python_verb(code)

    if name_lower in _BROWSER_VERBS:
        # The tool name already IS the operation (like plain MCP) -- just tag the backend for
        # grouping/display consistency with sql:/shell:/python:/code_edit:.
        return "browser:" + name_lower

    # Agent-env-text verbs (alfworld: go/take/put/open/...) don't map cleanly onto
    # MCP/SQL/shell/Python/browser -- the tool name already IS the operation there too, so the
    # plain tool-name fallback below is correct and sufficient, same as real MCP tools.
    names = sorted(
        {str((c.get("function") or {}).get("name") or c.get("name", "")).strip() for c in calls if isinstance(c, dict)}
    )
    return "+".join(name for name in names if name) or "noop"


@torch.no_grad()
def _encode_texts(net: Any, tokenizer: Any, texts: list[str], device: Any, max_length: int) -> "torch.Tensor":
    tokens = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True
    )
    latent, _ = net.encode_latent_and_logits(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
    return latent


@torch.no_grad()
def _encode_frame_history(
    net: Any,
    tokenizer: Any,
    input_history: list[dict[str, Any]] | None,
    device: Any,
    config: HierarchicalCEMConfig,
    goal_arg: "torch.Tensor | None",
) -> tuple | None:
    """Build ``(frames [1,T,D], producing_actions [1,T,C], valid [1,T])`` from the real logged
    ``{step, action, observation}`` history for ``predictor_arch="transformer"`` -- same
    representation as ``_ewm_jepa.JepaEwmGenerator._encode_frame_history``, rebuilt here from a
    bare ``net``/``tokenizer`` since this module has no generator instance to reuse. Returns
    ``None`` for ``predictor_arch="mlp"`` or when there is no usable history, in which case the
    model's own ``[context, z_current]`` degenerate case applies."""
    if getattr(net, "predictor_arch", "mlp") != "transformer":
        return None
    entries = [item for item in (input_history or []) if isinstance(item, dict) and item.get("action") is not None]
    if not entries:
        return None
    action_texts = [render_action(item["action"]) or " " for item in entries]
    observation_texts = [
        ewm.stringify_tool_output(item.get("observation") or item.get("state") or "") or " " for item in entries
    ]
    z_frames = _encode_texts(net, tokenizer, observation_texts, device, config.max_input_length).unsqueeze(0)
    z_actions = _encode_texts(net, tokenizer, action_texts, device, config.max_action_length)
    if getattr(net, "goal_conditioning", False):
        goal = goal_arg if goal_arg is not None else torch.zeros_like(z_actions[:1])
        z_actions = torch.cat([z_actions, goal.expand(z_actions.shape[0], -1)], dim=-1)
    z_actions = z_actions.unsqueeze(0)
    valid = torch.ones(1, len(entries), dtype=torch.bool, device=device)
    return z_frames, z_actions, valid


def build_anchor_prompt(
    context_text: str, current_state_text: str, config: HierarchicalCEMConfig, tool_names: list[str] | None = None
) -> str:
    """One prompt, one LLM call: ask for K diverse next-action anchors. Reuses beam_plan's
    step-candidates prompt (breadth/parameterization split) as the anchor proposal -- diversity
    among the K anchors is what lets the per-family Gaussians cover distinct strategies."""
    from ejepa_wm.backends._ewm_beam_plan import BeamPlanConfig, build_step_candidates_prompt

    proxy_cfg = BeamPlanConfig(num_candidates=config.num_llm_anchors, horizon=1)
    return build_step_candidates_prompt(context_text, current_state_text, proxy_cfg, step=0, tool_names=tool_names)


def parse_anchor_actions(text: str, config: HierarchicalCEMConfig) -> list[dict[str, Any]]:
    from ejepa_wm.backends._ewm_beam_plan import BeamPlanConfig, parse_action_candidates

    proxy_cfg = BeamPlanConfig(num_candidates=config.num_llm_anchors, horizon=1)
    return parse_action_candidates(text, proxy_cfg)


def _init_family_state(
    anchor_latents: "torch.Tensor", anchor_families: list[str], config: HierarchicalCEMConfig
) -> tuple[list[str], dict[str, float], dict[str, "torch.Tensor"], dict[str, "torch.Tensor"]]:
    """Seed (pi, per-family mean, per-family std) from the LLM anchors: a family's mean is the
    mean of its anchors' latents; its std is the empirical std if it has >=2 anchors, else an
    isotropic prior scaled to the anchor's own latent magnitude."""
    families = sorted(set(anchor_families))
    pi = {name: anchor_families.count(name) / len(anchor_families) for name in families}
    means: dict[str, "torch.Tensor"] = {}
    stds: dict[str, "torch.Tensor"] = {}
    for name in families:
        idx = [i for i, fam in enumerate(anchor_families) if fam == name]
        points = anchor_latents[idx]
        mean = points.mean(dim=0)
        if points.shape[0] > 1:
            std = points.std(dim=0).clamp_min(config.min_std)
        else:
            scale = (mean.abs().mean() * config.init_std).clamp_min(config.min_std)
            std = scale.expand_as(mean).clone()
        means[name] = mean
        stds[name] = std
    return families, pi, means, stds


def _broadcast_horizon_state(
    families: list[str], pi: dict[str, float], means: dict[str, "torch.Tensor"], stds: dict[str, "torch.Tensor"], horizon: int
) -> tuple[list[dict[str, float]], list[dict[str, "torch.Tensor"]], list[dict[str, "torch.Tensor"]]]:
    """The anchor set is proposed once per planning cycle, so every horizon step starts from the
    identical (pi, mean, std) -- CEM then lets different steps diverge as elites are refit."""
    return (
        [dict(pi) for _ in range(horizon)],
        [dict(means) for _ in range(horizon)],
        [dict(stds) for _ in range(horizon)],
    )


def _sample_trajectories(
    pi_t: list[dict[str, float]],
    means_t: list[dict[str, "torch.Tensor"]],
    stds_t: list[dict[str, "torch.Tensor"]],
    num_samples: int,
    horizon: int,
    dim: int,
    device: Any,
) -> tuple[list[list[str]], "torch.Tensor"]:
    """Sample N full-horizon latent-action trajectories: at each step, draw a family from the
    step's categorical, then a continuous latent action from that family's Gaussian. Vectorized
    per (step, family) group rather than per-sample."""
    u = torch.empty(num_samples, horizon, dim, device=device)
    family_ids: list[list[str]] = [[""] * horizon for _ in range(num_samples)]
    for t in range(horizon):
        fam_list = list(pi_t[t].keys())
        probs = torch.tensor([max(pi_t[t][name], 0.0) for name in fam_list], device=device)
        probs = probs / probs.sum().clamp_min(1e-8)
        chosen = torch.multinomial(probs, num_samples, replacement=True)
        for fi, name in enumerate(fam_list):
            mask = chosen == fi
            count = int(mask.sum().item())
            if count == 0:
                continue
            mean = means_t[t][name]
            std = stds_t[t][name]
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(count, dim, device=device)
            positions = mask.nonzero(as_tuple=True)[0]
            u[positions, t] = samples
            for pos in positions.tolist():
                family_ids[pos][t] = name
    return family_ids, u


@torch.no_grad()
def _rollout_and_score(
    net: Any,
    z_current0: "torch.Tensor",
    z_context: "torch.Tensor",
    u: "torch.Tensor",
    config: HierarchicalCEMConfig,
    vocab: dict[str, list[str]],
    goal_arg: "torch.Tensor | None",
    frame_history0: tuple | None = None,
) -> list[list[dict[str, dict[str, float]]]]:
    """Recursive (NOT Fast-LeWM prefix) latent rollout: one predict_latent + one
    predict_canonical_event_logits call per horizon step, batched over all N trajectories --
    zero LLM calls, zero text decoding.

    ``frame_history0`` -- the real logged history (see :func:`_encode_frame_history`), already
    expanded to N samples -- seeds ``predictor_arch="transformer"``'s multi-step context; it is
    then autoregressively extended every step with the just-predicted event and the action that
    produced it (every one of the N samples is "active" every step here, unlike beam_plan's
    variable-length plans, so no per-sample padding/masking is needed)."""
    num, horizon, _ = u.shape
    z_current = z_current0
    frame_history = frame_history0
    all_active = torch.arange(num, device=z_current0.device)
    transformer = getattr(net, "predictor_arch", "mlp") == "transformer"
    trajectories: list[list[dict[str, dict[str, float]]]] = [[] for _ in range(num)]
    for t in range(horizon):
        z_action = u[:, t, :]
        z_pred, _, z_state = net.predict_latent_with_state(
            z_current, z_action, z_context, goal_arg, frame_history=frame_history
        )
        logits = net.predict_canonical_event_logits(z_current, z_action, z_context, z_pred, z_state)
        # One host transfer per field instead of one per (sample, field, class) scalar -- at
        # N=256 CEM samples the per-scalar version is thousands of device syncs per horizon step.
        probs_rows = logits_to_field_probs_batched(logits, vocab, temperature=config.temperature)
        for i in range(num):
            trajectories[i].append(probs_rows[i])
        if frame_history is not None or transformer:
            producing_action = net._frame_conditioning(z_action, goal_arg)
            frame_history = _extend_batch_frame_history(
                frame_history, num, all_active, z_pred, producing_action, z_current0.device
            )
        z_current = z_pred
    return trajectories


def _cem_update(
    pi_t: list[dict[str, float]],
    means_t: list[dict[str, "torch.Tensor"]],
    stds_t: list[dict[str, "torch.Tensor"]],
    elite_family_ids: list[list[str]],
    elite_u: "torch.Tensor",
    config: HierarchicalCEMConfig,
) -> None:
    """Classic CEM update, per horizon step: refit each family's (weight, mean, std) from the
    elites that chose it this iteration. A family with zero elites at a step keeps its Gaussian
    unchanged (nothing to learn from) but its categorical weight decays toward 0 -- it can still
    recover later if elites revisit it."""
    horizon = len(pi_t)
    num_elites = elite_u.shape[0]
    alpha = config.smoothing
    for t in range(horizon):
        fam_list = list(pi_t[t].keys())
        counts = {name: 0 for name in fam_list}
        sums = {name: torch.zeros_like(means_t[t][name]) for name in fam_list}
        sq_sums = {name: torch.zeros_like(means_t[t][name]) for name in fam_list}
        for n in range(num_elites):
            name = elite_family_ids[n][t]
            if name not in counts:
                continue
            counts[name] += 1
            sums[name] += elite_u[n, t]
            sq_sums[name] += elite_u[n, t] ** 2
        for name in fam_list:
            new_pi = counts[name] / max(num_elites, 1)
            pi_t[t][name] = (1 - alpha) * pi_t[t][name] + alpha * new_pi
            if counts[name] > 0:
                mean = sums[name] / counts[name]
                var = (sq_sums[name] / counts[name] - mean**2).clamp_min(config.min_std**2)
                std = var.sqrt().clamp_min(config.min_std)
                means_t[t][name] = (1 - alpha) * means_t[t][name] + alpha * mean
                stds_t[t][name] = (1 - alpha) * stds_t[t][name] + alpha * std
        total = sum(pi_t[t].values()) or 1.0
        for name in fam_list:
            pi_t[t][name] /= total


def _decode_predicted_state(field_probs: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Same convention as ``_ewm_jepa.score_action_plans_canonical_event``: argmax per
    single-label field, >=0.5 threshold for the multi-label field -- so beam_plan and this
    module's imagined-plan entries render identically in the injected message."""
    predicted_state: dict[str, Any] = {}
    for field_name, probs in field_probs.items():
        if not probs:
            continue
        if field_name == MISSING_INFO_FIELD:
            predicted_state[field_name] = [category for category, prob in probs.items() if prob >= 0.5] or ["none"]
        else:
            predicted_state[field_name] = max(probs.items(), key=lambda kv: kv[1])[0]
    return predicted_state


def _format_step_reason(contributions: dict[str, float], top_fields: int = 3) -> str:
    """Per-step dominant +/- contributions, same short-reason style as
    ``_ewm_canonical_event_scoring.explain_trajectory`` but scoped to a single horizon step."""
    ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    positives = [f"{name}:+{value:.2f}" for name, value in ranked if value > 1e-6][:top_fields]
    negatives = [f"{name}:{value:.2f}" for name, value in reversed(ranked) if value < -1e-6][:top_fields]
    parts = []
    if positives:
        parts.append("+ " + ", ".join(positives))
    if negatives:
        parts.append("- " + ", ".join(negatives))
    return " | ".join(parts) if parts else "neutral"


def validate_decoded_action(text: str) -> dict[str, Any] | None:
    """Light per-decode sanity gate for the learned-decoder strategy: NOT a grammar/schema/
    permission validator (that needs live tool-spec/DB-schema access this generic module
    doesn't have) -- only rejects obviously-broken decodes (empty, unparseable JSON with no
    recognizable tool call, or truncated garbage) before an action is ever handed to the caller
    for execution. Returns the normalized action dict, or None if the decode should be
    discarded (caller falls back to nearest_anchor)."""
    if not text or not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    return normalize_action_step(parsed) if isinstance(parsed, dict) else None


def _decode_learned_action(
    net: Any,
    tokenizer: Any,
    u_step: "torch.Tensor",
    config: HierarchicalCEMConfig,
) -> dict[str, Any] | None:
    """Decode ONE sampled latent action via the trained action_decoder. Returns None (caller
    falls back to nearest_anchor) if the model has no decoder, decoding raises, or the output
    fails validate_decoded_action."""
    if not getattr(net, "action_decoder", False):
        return None
    try:
        texts = net.decode_action_latent(u_step.unsqueeze(0), tokenizer, max_new_tokens=config.decode_max_new_tokens)
    except Exception:
        return None
    return validate_decoded_action(texts[0]) if texts else None


def _nearest_anchor(
    u_step: "torch.Tensor",
    family_step: str,
    anchor_latents: "torch.Tensor",
    anchor_families: list[str],
    anchor_actions: list[dict[str, Any]],
) -> tuple[dict[str, Any], float, int]:
    """Decode one sampled latent action to an executable one: nearest LLM anchor sharing the
    same family (tool identity). Bindings (ids/paths/emails) are exactly the anchor's own --
    never interpolated -- only the *choice of anchor* is guided by the latent search."""
    candidates = [i for i, name in enumerate(anchor_families) if name == family_step]
    if not candidates:
        candidates = list(range(len(anchor_actions)))
    best_index, best_distance = candidates[0], float("inf")
    for i in candidates:
        distance = torch.norm(anchor_latents[i] - u_step).item()
        if distance < best_distance:
            best_distance, best_index = distance, i
    return anchor_actions[best_index], best_distance, best_index


@torch.no_grad()
def hierarchical_cem_plan(
    model: Any,
    tokenizer: Any,
    vocab: dict[str, list[str]],
    context_text: str,
    current_state_text: str,
    llm_generate: Callable[[str], str],
    config: HierarchicalCEMConfig | None = None,
    goal_text: str | None = None,
    tool_names: list[str] | None = None,
    input_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One planning cycle. Exactly ONE LLM call opens the search space with K anchors; the
    N*H*num_iters latent rollouts and scoring are LLM-free and decode-free. Returns the
    converged best trajectory (decoded step-by-step to the nearest anchor for display/execution)
    plus a CEM-native confidence signal."""
    config = config or HierarchicalCEMConfig()
    net = getattr(model, "module", model)
    net.eval()
    device = next(net.parameters()).device

    prompt = build_anchor_prompt(context_text, current_state_text, config, tool_names)
    raw = llm_generate(prompt)  # <-- the only LLM call in the cycle
    parsed_actions = parse_anchor_actions(raw, config)
    anchor_actions = [step for step in (normalize_action_step(action) for action in parsed_actions) if step]
    if not anchor_actions:
        return {
            "num_llm_calls": 1, "num_anchors": 0, "confident": False, "vetoed": True,
            "reason": "no_anchors_parsed", "recommended_action": None, "imagined_plan": [], "all": [],
        }

    action_texts = [render_action(action) or " " for action in anchor_actions]
    anchor_latents = _encode_texts(net, tokenizer, action_texts, device, config.max_action_length)
    anchor_families = [_family_key(action) for action in anchor_actions]

    z_context = _encode_texts(net, tokenizer, [context_text], device, config.max_input_length)
    z_current0 = _encode_texts(net, tokenizer, [current_state_text], device, config.max_input_length)
    goal_arg = None
    if goal_text and getattr(net, "goal_conditioning", False):
        goal_arg = _encode_texts(net, tokenizer, [goal_text], device, config.max_input_length)

    frame_history_seed = _encode_frame_history(net, tokenizer, input_history, device, config, goal_arg)

    dim = anchor_latents.shape[-1]
    families, pi, means, stds = _init_family_state(anchor_latents, anchor_families, config)
    pi_t, means_t, stds_t = _broadcast_horizon_state(families, pi, means, stds, config.horizon)

    family_ids: list[list[str]] = []
    u = torch.empty(0)
    trajectories: list[list[dict[str, dict[str, float]]]] = []
    all_scored: list[dict[str, Any]] = []
    elite_family_ids: list[list[str]] = []

    for _ in range(max(1, config.num_iters)):
        family_ids, u = _sample_trajectories(pi_t, means_t, stds_t, config.num_samples, config.horizon, dim, device)
        z_current_batch = z_current0.expand(config.num_samples, -1).contiguous()
        z_context_batch = z_context.expand(config.num_samples, -1).contiguous()
        goal_batch = goal_arg.expand(config.num_samples, -1).contiguous() if goal_arg is not None else None
        frame_history_batch = (
            tuple(t.expand(config.num_samples, *t.shape[1:]).contiguous() for t in frame_history_seed)
            if frame_history_seed is not None
            else None
        )
        trajectories = _rollout_and_score(
            net, z_current_batch, z_context_batch, u, config, vocab, goal_batch, frame_history_batch
        )
        _, all_scored = rank_trajectories(trajectories, config.score_config, top_k=config.num_samples)
        non_vetoed = [record for record in all_scored if not record["vetoed"]]
        elite_records = non_vetoed[: config.num_elites] if non_vetoed else all_scored[: config.num_elites]
        elite_idx = [record["index"] for record in elite_records]
        elite_family_ids = [family_ids[i] for i in elite_idx]
        elite_u = u[elite_idx]
        _cem_update(pi_t, means_t, stds_t, elite_family_ids, elite_u, config)

    if not all_scored:
        return {
            "num_llm_calls": 1, "num_anchors": len(anchor_actions), "confident": False, "vetoed": True,
            "reason": "no_scored_trajectories", "recommended_action": None, "imagined_plan": [], "all": [],
        }

    best = all_scored[0]
    best_family_ids = family_ids[best["index"]]
    best_u = u[best["index"]]

    imagined_plan = []
    for t in range(config.horizon):
        decode_source = "nearest_anchor"
        decoded: dict[str, Any] | None = None
        distance: float | None = None
        anchor_index: int | None = None
        if config.decode_strategy == "learned_decoder":
            decoded = _decode_learned_action(net, tokenizer, best_u[t], config)
            if decoded is not None:
                decode_source = "learned_decoder"
        if decoded is None:
            decoded, distance, anchor_index = _nearest_anchor(
                best_u[t], best_family_ids[t], anchor_latents, anchor_families, anchor_actions
            )
        field_probs = trajectories[best["index"]][t] if best["index"] < len(trajectories) else {}
        contributions = best["per_step"][t]["contributions"] if t < len(best["per_step"]) else {}
        imagined_plan.append(
            {
                "family": best_family_ids[t],
                "calls": decoded.get("tool_calls") if isinstance(decoded, dict) else None,
                "decode_source": decode_source,
                "decode_distance": distance,
                "decode_anchor_index": anchor_index,
                "score": best["per_step"][t]["score"] if t < len(best["per_step"]) else None,
                "reason": _format_step_reason(contributions),
                "predicted_state": _decode_predicted_state(field_probs),
                "field_probs": field_probs,
            }
        )

    elite_family0 = [fam[0] for fam in elite_family_ids] if elite_family_ids else []
    agreement = elite_family0.count(best_family_ids[0]) / max(len(elite_family0), 1)
    confident = (not best["vetoed"]) and agreement >= config.min_elite_agreement

    return {
        "num_llm_calls": 1,
        "num_anchors": len(anchor_actions),
        "confident": confident,
        "elite_agreement": agreement,
        "vetoed": best["vetoed"],
        "score": best["score"],
        "reason": best["reason"],
        "recommended_action": imagined_plan[0]["calls"] if imagined_plan else None,
        "recommended_action_json": json.dumps(imagined_plan[0]["calls"], ensure_ascii=False) if imagined_plan and imagined_plan[0]["calls"] else None,
        "imagined_plan": imagined_plan,
        "family_distribution": {name: pi_t[0].get(name, 0.0) for name in families},
        "top": all_scored[: config.top_k],
        "all": all_scored,
    }


__all__ = [
    "HierarchicalCEMConfig",
    "hierarchical_cem_plan",
    "normalize_action_step",
    "validate_decoded_action",
]
