"""Tests for the ``beam_plan`` (JEPA MPC lookahead) strategy — the ejepa_wm port of the
EnterpriseOps-Gym orchestrator's beam block.

Everything is exercised without torch: canonical-event scoring uses plain probability
dicts (torch is only needed by ``logits_to_field_probs``, not by the scorer/ranker), the
beam parsers are pure Python, and the ``beam_plan_step`` MPC controller is driven with a
fake JEPA generator (exposing ``score_action_plans_canonical_event``) injected the same
way ``test_ewm_jepa`` injects a fake ``_ewm_jepa`` module.
"""

from __future__ import annotations

import json
import re
import sys
import types
from typing import Any

import pytest

from ejepa_wm.backends._ewm_beam_plan import (
    BeamPlanConfig,
    build_single_plan_prompt,
    build_step_candidates_prompt,
    parse_action_candidates,
    parse_plans,
    parse_single_plan,
)
from ejepa_wm.backends._ewm_canonical_event_scoring import (
    MISSING_INFO_FIELD,
    CanonicalEventScoreConfig,
    finalize_scored_plans,
    rank_trajectories,
    score_step,
    score_trajectory,
)
from ejepa_wm.backends._ewm_symbolic_plan import (
    resolve_symbolic_references,
    validate_symbolic_references,
)
from ejepa_wm.factory import wm_config_from_env

# --- strategy / config resolution ----------------------------------------------


def test_beam_plan_strategy_defaults_to_ewm_imagined(monkeypatch):
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")
    monkeypatch.delenv("WM_BACKEND", raising=False)
    cfg = wm_config_from_env()
    assert cfg.strategy == "beam_plan"
    assert cfg.backend == "ewm_imagined"


def test_beam_plan_is_a_known_strategy():
    from ejepa_wm.base import STRATEGIES, normalize_strategy

    assert "beam_plan" in STRATEGIES
    assert normalize_strategy("beam_plan") == "beam_plan"


# --- canonical-event scoring ----------------------------------------------------


def test_rank_trajectories_orders_and_vetoes():
    good = [
        {
            "execution_status": {"success": 0.9, "failure": 0.1},
            "progress_signal": {"positive": 0.8, "neutral": 0.2},
        }
    ]
    failing = [
        {
            "execution_status": {"success": 0.1, "failure": 0.9},
            "progress_signal": {"positive": 0.1, "negative": 0.9},
        }
    ]
    top, all_scored = rank_trajectories([good, failing], CanonicalEventScoreConfig(), top_k=2)
    assert top[0]["index"] == 0  # the good trajectory wins
    assert all_scored[0]["index"] == 0
    # the failing trajectory is vetoed (P(failure)=0.9 > default veto 0.6) and sorts last
    vetoed = {r["index"]: r["vetoed"] for r in all_scored}
    assert vetoed[1] is True and vetoed[0] is False
    # normalized scores: vetoed -> 0, live sums to ~1
    assert all_scored[[r["index"] for r in all_scored].index(1)]["normalized_score"] == 0.0


def test_trajectory_scoring_does_not_discount_later_steps():
    config = CanonicalEventScoreConfig()
    positive = {"progress_signal": {"positive": 1.0}}
    scored = score_trajectory([positive, positive, positive], config)

    assert scored["score"] == pytest.approx(2.4)


def test_deleted_side_effect_is_vetoed():
    plan = [{"side_effect_type": {"deleted": 0.9, "none": 0.1}}]
    _, all_scored = rank_trajectories([plan], CanonicalEventScoreConfig(), top_k=1)
    assert all_scored[0]["vetoed"] is True


def test_score_step_ignores_weak_nudge_fields():
    score, contributions = score_step(
        {
            "recommended_abstract_action": {"avoid": 1.0, "proceed": 0.0},
            MISSING_INFO_FIELD: {"object_id": 1.0, "none": 0.0},
        },
        CanonicalEventScoreConfig(),
    )
    assert score == pytest.approx(0.0)
    assert "recommended_abstract_action" not in contributions
    assert MISSING_INFO_FIELD not in contributions


def test_action_type_head_penalizes_reads_after_information_saturation():
    history = [
        {
            "action": {"name": "search_records", "arguments": {"query": "alpha"}},
            "observation": {"items": [{"id": "1"}]},
        },
        {
            "action": {"name": "get_record", "arguments": {"id": "1"}},
            "observation": {"id": "1", "status": "active"},
        },
    ]
    common = {
        "execution_status": {"success": 0.9, "failure": 0.1},
        "progress_signal": {"positive": 0.8, "neutral": 0.2},
    }
    trajectories = [
        [{**common, "action_type": {"read": 0.9, "search": 0.05, "update": 0.05}}],
        [{**common, "action_type": {"read": 0.05, "search": 0.05, "update": 0.9}}],
    ]
    plans = [
        [{"name": "get_record", "arguments": {"id": "1"}}],
        [{"name": "update_record", "arguments": {"id": "1", "status": "closed"}}],
    ]

    scored = finalize_scored_plans(
        trajectories,
        plans,
        CanonicalEventScoreConfig(),
        task_text="Close record 1.",
        input_history=history,
    )

    assert scored[0]["plan_index"] == 1
    read_record = next(record for record in scored if record["plan_index"] == 0)
    adjustment = read_record["action_aware_adjustments"][0]
    assert adjustment["kind"] == "read_after_saturation"
    assert adjustment["read_probability"] == pytest.approx(0.95)
    write_record = next(record for record in scored if record["plan_index"] == 1)
    assert read_record["raw_rank"] == 1
    assert write_record["raw_rank"] == 2
    assert read_record["adjusted_rank"] == 2
    assert write_record["adjusted_rank"] == 1
    assert read_record["action_aware_score_delta"] < 0.0


def test_action_type_head_penalizes_read_when_information_is_sufficient():
    trajectories = [
        [
            {
                "action_type": {"read": 1.0},
                "execution_status": {"success": 1.0},
                "progress_signal": {"positive": 1.0},
                "information_sufficiency": {"sufficient": 0.8, "insufficient": 0.2},
            }
        ]
    ]
    scored = finalize_scored_plans(
        trajectories,
        [[{"name": "list_records", "arguments": {}}]],
        CanonicalEventScoreConfig(),
    )

    adjustment = scored[0]["action_aware_adjustments"][0]
    assert adjustment["information_sufficient_probability"] == pytest.approx(0.8)
    assert adjustment["penalty_pressure"] == pytest.approx(0.8)
    assert adjustment["score_delta"] == pytest.approx(-0.8)


def test_action_type_head_softly_penalizes_read_after_task_progress():
    history = [
        {
            "action": {"name": "update_record", "arguments": {"id": "1"}},
            "observation": {"success": True},
        }
    ]
    trajectories = [
        [
            {
                "action_type": {"read": 1.0},
                "execution_status": {"success": 1.0},
                "information_sufficiency": {"sufficient": 0.1, "insufficient": 0.9},
            }
        ]
    ]
    scored = finalize_scored_plans(
        trajectories,
        [[{"name": "list_records", "arguments": {}}]],
        CanonicalEventScoreConfig(),
        input_history=history,
    )

    adjustment = scored[0]["action_aware_adjustments"][0]
    assert adjustment["prior_progress_actions"] == 1
    assert adjustment["penalty_pressure"] == pytest.approx(0.5)
    assert adjustment["score_delta"] == pytest.approx(-0.5)


def test_required_action_coverage_and_first_action_bonus_prefer_task_write_plan():
    common = {
        "execution_status": {"success": 0.9, "failure": 0.1},
        "progress_signal": {"positive": 0.8, "neutral": 0.2},
        "information_sufficiency": {"sufficient": 0.7, "insufficient": 0.3},
    }
    trajectories = [[common, common], [common, common]]
    plans = [
        [
            {"name": "list_records", "arguments": {}},
            {"name": "get_record", "arguments": {"id": "1"}},
        ],
        [
            {"name": "update_record", "arguments": {"id": "1"}},
            {"name": "send_notification", "arguments": {"id": "1"}},
        ],
    ]
    config = CanonicalEventScoreConfig(
        read_after_saturation_penalty=0.0,
        required_action_coverage_bonus=2.0,
        first_required_action_bonus=1.0,
    )

    scored = finalize_scored_plans(
        trajectories,
        plans,
        config,
        task_text="Update record 1 and notify its owner.",
    )

    assert scored[0]["plan_index"] == 1
    kinds = {item["kind"] for item in scored[0]["action_aware_adjustments"]}
    assert {"required_action_coverage", "first_required_action"} <= kinds


def test_first_step_weight_prefers_immediately_useful_trajectory():
    success = {"execution_status": {"success": 1.0}}
    neutral = {}
    plans = [
        [{"name": "update_record", "arguments": {}}, {"name": "list_records", "arguments": {}}],
        [{"name": "list_records", "arguments": {}}, {"name": "update_record", "arguments": {}}],
    ]
    config = CanonicalEventScoreConfig(
        read_after_saturation_penalty=0.0,
        first_step_score_weight=1.0,
    )

    scored = finalize_scored_plans(
        [[success, neutral], [neutral, success]],
        plans,
        config,
    )

    assert scored[0]["plan_index"] == 0
    adjustment = next(
        item
        for item in scored[0]["action_aware_adjustments"]
        if item["kind"] == "first_step_priority"
    )
    assert adjustment["score_delta"] > 0.0


def test_action_type_head_requires_opt_in_for_requested_archive_delete_veto_exemption():
    trajectory = [
        {
            "action_type": {"delete": 0.9, "read": 0.1},
            "execution_status": {"success": 0.9, "failure": 0.1},
            "progress_signal": {"positive": 0.9, "neutral": 0.1},
            "side_effect_type": {"deleted": 0.9, "none": 0.1},
        }
    ]
    default_scored = finalize_scored_plans(
        [trajectory],
        [[{"name": "archive_channel", "arguments": {"channel_id": "1"}}]],
        CanonicalEventScoreConfig(),
        task_text="Archive channel 1.",
    )

    assert default_scored[0]["vetoed"] is True
    assert default_scored[0]["action_aware_adjustments"] == []

    scored = finalize_scored_plans(
        [trajectory],
        [[{"name": "archive_channel", "arguments": {"channel_id": "1"}}]],
        CanonicalEventScoreConfig(exempt_task_required_side_effects=True),
        task_text="Archive channel 1.",
    )

    assert scored[0]["vetoed"] is False
    assert scored[0]["veto_reasons"] == []
    assert scored[0]["action_aware_adjustments"][0]["kind"] == "task_required_side_effect"
    assert scored[0]["raw_vetoed"] is True
    assert scored[0]["adjusted_vetoed"] is False


def test_action_type_head_preserves_delete_veto_when_task_forbids_deletion():
    trajectory = [
        {
            "action_type": {"delete": 0.9, "read": 0.1},
            "execution_status": {"success": 0.9, "failure": 0.1},
            "progress_signal": {"positive": 0.9, "neutral": 0.1},
            "side_effect_type": {"deleted": 0.9, "none": 0.1},
        }
    ]
    scored = finalize_scored_plans(
        [trajectory],
        [[{"name": "delete_record", "arguments": {"id": "1"}}]],
        CanonicalEventScoreConfig(),
        task_text="Do not delete record 1; update it instead.",
    )

    assert scored[0]["vetoed"] is True


# --- beam parsers ---------------------------------------------------------------


def test_parse_action_candidates_tolerates_code_fence_and_prose():
    cfg = BeamPlanConfig(num_candidates=3, horizon=2)
    raw = (
        'Sure!\n```json\n[{"name":"ls","arguments":{"p":"/"}}, {"name":"cat","arguments":{}}]\n```'
    )
    actions = parse_action_candidates(raw, cfg)
    assert [a["name"] for a in actions] == ["ls", "cat"]


def test_parse_action_candidates_caps_at_m():
    cfg = BeamPlanConfig(num_candidates=2, horizon=1)
    raw = '[{"name":"a","arguments":{}},{"name":"b","arguments":{}},{"name":"c","arguments":{}}]'
    assert len(parse_action_candidates(raw, cfg)) == 2


def test_parse_plans_handles_plans_wrapper_and_single():
    cfg = BeamPlanConfig(num_candidates=3, horizon=3)
    assert parse_plans('{"plans": [[{"name":"x","arguments":{}}]]}', cfg) == [
        [{"name": "x", "arguments": {}}]
    ]
    assert parse_plans('{"name":"x","arguments":{}}', cfg) == [[{"name": "x", "arguments": {}}]]
    assert parse_plans("not json", cfg) == []


def test_build_step_candidates_prompt_embeds_tools_and_step():
    cfg = BeamPlanConfig(num_candidates=4, horizon=2)
    prompt = build_step_candidates_prompt("SYS", "STATE", cfg, 0, tool_names=["ls", "cat"])
    assert "Use ONLY these tools: ls, cat" in prompt
    assert "lookahead step 1" in prompt and "4 actions" in prompt


def test_build_single_plan_prompt_has_diversity_options_menu():
    cfg = BeamPlanConfig(num_candidates=4, horizon=2)
    prompt = build_single_plan_prompt("SYS", "STATE", cfg, tool_names=["read", "write"])
    assert "DIVERSITY OPTIONS:" in prompt
    assert prompt.rfind("DIVERSITY OPTIONS:") > prompt.find("CURRENT STATE:")
    assert "1. best_overall" in prompt
    assert "2. different_first_tool" in prompt
    assert "4. write_or_update_first" in prompt
    assert "DIVERSITY PROCEDURE:" in prompt
    assert "Select exactly one option from 1-4" in prompt
    assert "SSoT DIVERSITY PROCEDURE" not in prompt
    assert "Use ONLY these tools: read, write" in prompt
    assert "JSON array of up to 2 steps" in prompt
    assert "$vars.name" in prompt
    assert '"bind"' in prompt


def test_build_single_plan_prompt_uses_paper_style_ssot_seed_before_plan():
    cfg = BeamPlanConfig(num_candidates=4, horizon=2)
    prompt = build_single_plan_prompt("SYS", "STATE", cfg, ssot_diversity=True)
    assert "SSoT DIVERSITY PROCEDURE:" in prompt
    assert "This procedure is required." in prompt
    assert "generate a unique 16-character mixed-case alphanumeric random string" in prompt
    assert "sum the ASCII values and select option 1 + (sum modulo 4)" in prompt
    assert '"random_seed":"<16 mixed-case alphanumeric characters>"' in prompt
    assert '"diversity_slot":<integer 1-4>' in prompt
    assert prompt.find("SSoT DIVERSITY PROCEDURE") < prompt.find("DIVERSITY OPTIONS:")
    assert prompt.rstrip().endswith('"diversity_slot" before "steps":')


def test_build_single_plan_prompt_has_unique_slots_for_default_beam_size():
    cfg = BeamPlanConfig(num_candidates=8, horizon=2)
    prompt = build_single_plan_prompt("SYS", "STATE", cfg, ssot_diversity=True)
    assert "7. recovery_plan" in prompt
    assert "8. dependency_first" in prompt
    option_names = re.findall(r"^\d+\. ([a-z_]+):", prompt, flags=re.MULTILINE)
    assert len(option_names) == 8
    assert len(set(option_names)) == 8


def test_parse_single_plan_accepts_ssot_seed_wrapper():
    cfg = BeamPlanConfig(num_candidates=4, horizon=2)
    text = json.dumps(
        {
            "random_seed": "aB3dE5fG7hJ9kL2m",
            "diversity_slot": 3,
            "steps": [
                {"name": "read", "arguments": {"id": "x"}},
                {"name": "write", "arguments": {"id": "$step1.id"}},
            ],
        }
    )
    assert parse_single_plan(text, cfg) == [
        {"name": "read", "arguments": {"id": "x"}},
        {"name": "write", "arguments": {"id": "$step1.id"}},
    ]


def test_symbolic_plan_validation_accepts_late_bound_user_id():
    plan = [
        {
            "name": "find_user",
            "arguments": {"name": "Ethan Well"},
            "bind": {
                "ethan_user_id": {
                    "field": "sys_id",
                    "match": {"name": "Ethan Well"},
                }
            },
        },
        {"name": "assign_incident", "arguments": {"user_id": "$vars.ethan_user_id"}},
    ]

    validation = validate_symbolic_references(plan)

    assert validation["valid"] is True
    assert validation["valid_reference_count"] == 1
    assert validation["broken_references"] == []


def test_symbolic_plan_validation_rejects_forward_and_type_mismatch():
    forward = [
        {"name": "assign_incident", "arguments": {"user_id": "$step1.user_id"}},
    ]
    mismatch = [
        {"name": "find_asset", "arguments": {"name": "Laptop"}},
        {"name": "assign_incident", "arguments": {"user_id": "$step1.asset_id"}},
    ]

    assert validate_symbolic_references(forward)["broken_references"][0]["reason"] == (
        "producer_must_precede_consumer"
    )
    assert validate_symbolic_references(mismatch)["broken_references"][0]["reason"] == (
        "dependency_type_mismatch"
    )


def test_symbolic_reference_resolution_uses_actual_observation_bindings():
    plan_prefix = [
        {
            "name": "find_user",
            "arguments": {"name": "Ethan Well"},
            "bind": {
                "ethan_user_id": {
                    "field": "sys_id",
                    "match": {"name": "Ethan Well"},
                }
            },
        }
    ]
    action = [{"name": "assign_incident", "arguments": {"user_id": "$vars.ethan_user_id"}}]

    resolved, detail = resolve_symbolic_references(
        action,
        observations=[[{"name": "Ethan Well", "sys_id": "user-123456"}]],
        plan_steps=plan_prefix,
    )

    assert detail["unresolved"] == []
    assert detail["bindings"][0]["reference"] == "$vars.ethan_user_id"
    assert resolved == [{"name": "assign_incident", "arguments": {"user_id": "user-123456"}}]

    direct_resolved, direct_detail = resolve_symbolic_references(
        [{"name": "assign_incident", "arguments": {"user_id": "$step1.user_id"}}],
        observations=[{"user_id": "user-654321"}],
    )
    assert direct_detail["unresolved"] == []
    assert direct_resolved == [{"name": "assign_incident", "arguments": {"user_id": "user-654321"}}]


def test_symbolic_reference_resolution_marks_ambiguous_bindings_unresolved():
    plan_prefix = [
        {
            "name": "find_user",
            "arguments": {"name": "Ethan Well"},
            "bind": {
                "ethan_user_id": {
                    "field": "sys_id",
                    "match": {"name": "Ethan Well"},
                }
            },
        }
    ]
    action = [{"name": "assign_incident", "arguments": {"user_id": "$vars.ethan_user_id"}}]

    _resolved, detail = resolve_symbolic_references(
        action,
        observations=[
            [
                {"name": "Ethan Well", "sys_id": "user-123456"},
                {"name": "Ethan Well", "sys_id": "user-789012"},
            ]
        ],
        plan_steps=plan_prefix,
    )

    assert detail["unresolved"][0]["reason"] == "binding_ambiguous"


@pytest.mark.parametrize("m,n_diff,n_same", [(5, 3, 2), (6, 3, 3), (4, 2, 2), (3, 2, 1)])
def test_step_candidates_prompt_two_group_split(m, n_diff, n_same):
    cfg = BeamPlanConfig(num_candidates=m, horizon=1)
    prompt = build_step_candidates_prompt("SYS", "STATE", cfg, 0)
    assert f"GROUP A ({n_diff} actions)" in prompt  # ceil(m/2) different-tool candidates
    assert f"GROUP B ({n_same} actions)" in prompt  # floor(m/2) same-tool different-arg candidates
    assert "DIFFERENT tool name" in prompt and "SAME single tool" in prompt


# --- beam_plan_step MPC controller (fake JEPA generator) ------------------------


class _FakeDecodeModel:
    """Stand-in for TextLeWorldModel's ``obs_grounding`` flag (all decode_plan_observations
    needs to see on ``generator.model``)."""

    def __init__(self, obs_grounding: bool = False):
        self.obs_grounding = obs_grounding


class _FakeScoringJepa:
    """Fake JEPA generator scoring by first-tool-name: ``good_tool`` high, else low. Computes a
    softmax ``normalized_score`` so the confidence gate behaves realistically; ``high=None`` makes
    every candidate score equally (a flat/low-confidence step). Optionally fakes the
    decode_plan_observations (predicted-tool-output decode) seam too."""

    def __init__(
        self,
        high="good_tool",
        obs_grounding=False,
        decode_texts=None,
        decode_error=None,
        terminal_prob=None,
    ):
        self.high = high
        self.model = _FakeDecodeModel(obs_grounding=obs_grounding)
        self.decode_texts = decode_texts
        self.decode_error = decode_error
        self.terminal_prob = terminal_prob
        self.decode_calls: list[dict] = []

    def decode_plan_observations(
        self,
        *,
        system_prompt,
        user_prompt,
        input_history,
        plan,
        max_new_tokens=96,
        goal_text_override=None,
    ):
        self.decode_calls.append(
            {"plan": plan, "max_new_tokens": max_new_tokens, "input_history": input_history}
        )
        if self.decode_error is not None:
            raise self.decode_error
        if self.decode_texts is not None:
            return list(self.decode_texts)
        return [f"decoded step {i}" for i in range(len(plan))]

    def score_action_plans_canonical_event(
        self, *, system_prompt, user_prompt, input_history, action_plans, score_config=None
    ):
        import math

        raw = []
        for i, plan in enumerate(action_plans):
            call = (plan[0].get("tool_calls", [{}])[0].get("function", {}) if plan else {}) or {}
            name = call.get("name", "")
            raw.append((i, plan, name, 5.0 if (self.high and name == self.high) else 0.0))
        highest = max(s for *_, s in raw)
        exps = [math.exp(s - highest) for *_, s in raw]
        denom = sum(exps) or 1.0
        out = []
        for (i, plan, name, score), e in zip(raw, exps, strict=False):
            best = bool(self.high and name == self.high)
            predicted_state = {
                "execution_status": "success" if best else "failure",
                "progress_signal": "positive" if best else "neutral",
                "error_signature": "none",  # uninformative -> filtered from injection
                "missing_information_type": ["none"],  # uninformative -> filtered from injection
            }
            # Real per-step raw field probabilities (undecoded) -- what the critic trigger reads
            # for P(execution_status=failure) / P(progress_signal=positive).
            field_probs = {
                "execution_status": {"success": 0.9, "failure": 0.05}
                if best
                else {"success": 0.1, "failure": 0.8},
                "progress_signal": {"positive": 0.9, "neutral": 0.05}
                if best
                else {"positive": 0.1, "neutral": 0.1},
            }
            # Real score_action_plans_canonical_event decodes one predicted_state PER STEP of a
            # multi-step plan (per_step_predicted_state), a per-step score breakdown (per_step)
            # that beam_plan's open-loop margin gate/critic trigger read at index 0, and the raw
            # per-step field probabilities (per_step_field_probs) the critic trigger reads -- all
            # mirrored here so open-loop/critic tests exercise the real record shape.
            terminal_probs = (
                [float(self.terminal_prob) for _ in plan] if self.terminal_prob is not None else []
            )
            out.append(
                {
                    "index": i,
                    "plan_index": i,
                    "plan": plan,
                    "score": score,
                    "normalized_score": e / denom,
                    "vetoed": False,
                    "reason": name,
                    "predicted_state": predicted_state,
                    "per_step_predicted_state": [predicted_state for _ in plan],
                    "per_step": [{"step": step + 1, "score": score} for step in range(len(plan))],
                    "per_step_field_probs": [field_probs for _ in plan],
                    "per_step_terminal_prob": terminal_probs,
                    "terminal_probability": terminal_probs[-1] if terminal_probs else None,
                }
            )
        out.sort(key=lambda r: (r["vetoed"], -r["score"]))
        return out


@pytest.fixture()
def fake_jepa_module(monkeypatch):
    """Inject a torch-free fake ``_ewm_jepa`` so ewm_imagined's lazy import resolves to us."""
    mod = types.ModuleType("ejepa_wm.backends._ewm_jepa")
    mod.JepaEwmGenerator = lambda *a, **k: _FakeScoringJepa()
    monkeypatch.setitem(sys.modules, "ejepa_wm.backends._ewm_jepa", mod)
    return mod


def _candidate_chat_fn():
    """Policy LLM stub: proposes two distinct candidate actions per horizon step."""

    def chat_fn(messages):
        return json.dumps(
            [
                {"name": "good_tool", "arguments": {"x": 1}},
                {"name": "bad_tool", "arguments": {}},
            ]
        )

    return chat_fn


def _make_beam_wm(monkeypatch, fake_jepa_module, **env):
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/ckpt")
    monkeypatch.setenv("WM_IMAGINED_ROLLOUT_MODE", "closed_loop")  # defensive: a prior helper call
    # in the same test (e.g. the open-loop comparison) may have left this set to open_loop.
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    return EwmImaginedWorldModel(wm_config_from_env(), _candidate_chat_fn())


_FLOW = [
    {"type": "system_message", "content": "you are an agent"},
    {"type": "user_message", "content": "do the task"},
    {
        "type": "tools",
        "tools": [
            {"name": "good_tool"},
            {"name": "follow_up_tool"},
            {"name": "bad_tool"},
            {"name": "seed_tool"},
        ],
    },
]


def test_beam_plan_advisory_recommends_without_override(monkeypatch, fake_jepa_module):
    # Advisory is the DEFAULT (Fix 2): the confident beam recommends good_tool but the agent's
    # own action still executes (no override); the plan is injected as guidance.
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
        WM_BEAM_MPC_EXECUTE_STEPS=2,
    )
    assert wm.supports_beam_plan() is True
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="do the task"
    )
    d = res.detail
    assert d["event"] == "GYM_BEAM_PLAN" and d["replanned"] is True
    assert d["beam_confident"] is True and d["injected"] is True and d["advisory"] is True
    assert d["override_applied"] is False  # advisory: agent action stands
    assert d["calls"][0]["name"] == "seed_tool"  # the executed call is the agent's
    assert d["recommended_calls"][0]["name"] == "good_tool"  # what the beam would recommend
    assert d["imagined_plan_len"] == 2
    assert (
        d["imagined_plan"][0]["predicted_state"]["execution_status"] == "success"
    )  # Fix 3 audit log


def test_beam_plan_hard_override_forces_recommendation(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_HARD_OVERRIDE=1,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
        WM_BEAM_MPC_EXECUTE_STEPS=2,
    )
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="do the task"
    )
    d = res.detail
    assert d["hard_override"] is True and d["advisory"] is False
    assert d["override_applied"] is True
    assert d["calls"][0]["name"] == "good_tool"  # forced over the agent's action


def _set_revision_choice(wm, choice: str) -> None:
    candidate_chat = _candidate_chat_fn()

    def chat_fn(messages):
        if "Compare exactly two options" in messages[-1]["content"]:
            return json.dumps({"choice": choice})
        return candidate_chat(messages)

    wm._agent._chat_fn = chat_fn


def test_beam_revision_selects_each_planned_horizon_action(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_REVISION=1,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
        WM_BEAM_MPC_EXECUTE_STEPS=4,
    )
    _set_revision_choice(wm, "plan")
    seed = [{"name": "seed_tool", "arguments": {}}]

    first = wm.beam_plan_step(_FLOW, seed_calls=seed, user_query="do the task")
    assert first.detail["beam_revision_choice"] == "plan"
    assert first.detail["override_applied"] is True
    assert first.detail["calls"][0]["name"] == "good_tool"
    assert first.detail["injected"] is True
    assert wm._beam_plan_cursor == 1

    second = wm.beam_plan_step(_FLOW, seed_calls=seed, user_query="do the task")
    assert second.detail["event"] == "GYM_BEAM_PLAN_FOLLOW"
    assert second.detail["beam_revision_choice"] == "plan"
    assert second.detail["calls"][0]["name"] == "good_tool"
    assert second.detail["override_applied"] is True
    assert wm._beam_plan_cursor == 2


def test_beam_revision_uses_agent_grounding_for_symbolic_planned_action(
    monkeypatch, fake_jepa_module
):
    wm = _make_beam_wm(monkeypatch, fake_jepa_module, WM_BEAM_PLAN_REVISION=1)
    _set_revision_choice(wm, "plan")
    selected, detail = wm._beam_revision_choose(
        system_prompt="agent",
        user_prompt="update the record",
        state_text="record id is 42",
        agent_calls=[{"name": "follow_up_tool", "arguments": {"id": "42"}}],
        planned_calls=[{"name": "follow_up_tool", "arguments": {"id": "$step1.id"}}],
        remaining_plan=[
            {
                "calls": [{"name": "follow_up_tool", "arguments": {"id": "$step1.id"}}],
                "predicted_state": {},
            }
        ],
    )

    assert selected == [{"name": "follow_up_tool", "arguments": {"id": "42"}}]
    assert detail["beam_revision_plan_selected"] is True
    assert detail["beam_revision_reason"] == "plan_selected_agent_grounded_arguments"


def test_beam_revision_agent_choice_invalidates_imagined_suffix(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_REVISION=1,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=3,
        WM_BEAM_MPC_EXECUTE_STEPS=4,
    )
    _set_revision_choice(wm, "agent")
    seed = [{"name": "seed_tool", "arguments": {}}]

    result = wm.beam_plan_step(_FLOW, seed_calls=seed, user_query="do the task")

    assert result.detail["beam_revision_choice"] == "agent"
    assert result.detail["override_applied"] is False
    assert result.detail["calls"] == seed
    assert result.detail["injected"] is False
    assert wm._beam_imagined_plan == []
    assert wm._beam_plan_cursor == 0


def test_beam_plan_confidence_gate_withholds_flat_plan(monkeypatch, fake_jepa_module):
    # A flat step (all candidates score equally) is not confident: no injection, no plan cached.
    # The gate is trajectory-level (avg normalized score over the FULL horizon), not per-step, so
    # there is no early break -- the full horizon always runs during a re-plan.
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=4,
        WM_BEAM_MPC_EXECUTE_STEPS=2,
    )
    wm._wm = _FakeScoringJepa(high=None)  # every candidate scores the same -> flat
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    d = res.detail
    assert d["beam_confident"] is False
    assert d["override_reason"] == "below_confidence_gate"
    assert d["injected"] is False
    assert wm.beam_injection_text() == ""  # nothing cached to inject
    assert d["beam_llm_calls"] == 4  # full horizon always runs (no early break)
    assert d["full_horizon_reached"] is True


def test_beam_plan_cooldown_after_rejected_replan(monkeypatch, fake_jepa_module):
    # A rejected (flat) re-plan should NOT immediately re-trigger the CEM/beam on the very next
    # step -- it rides out the rest of the cooldown window autonomously (GYM_BEAM_PLAN_COOLDOWN),
    # then re-plans again once execute_steps has elapsed.
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
        WM_BEAM_MPC_EXECUTE_STEPS=2,
    )
    wm._wm = _FakeScoringJepa(high=None)  # flat -> every re-plan is rejected
    seed = [{"name": "seed_tool", "arguments": {}}]
    r1 = wm.beam_plan_step(_FLOW, seed_calls=seed, user_query="t")
    assert r1.detail["event"] == "GYM_BEAM_PLAN" and r1.detail["beam_confident"] is False
    calls_after_r1 = r1.detail["beam_llm_calls"]

    r2 = wm.beam_plan_step(_FLOW, seed_calls=seed, user_query="t")
    assert r2.detail["event"] == "GYM_BEAM_PLAN_COOLDOWN"  # no WM call, no re-trigger
    assert "beam_llm_calls" not in r2.detail

    r3 = wm.beam_plan_step(_FLOW, seed_calls=seed, user_query="t")
    assert r3.detail["event"] == "GYM_BEAM_PLAN_COOLDOWN"  # still cooling down (execute_steps=2)

    r4 = wm.beam_plan_step(_FLOW, seed_calls=seed, user_query="t")
    assert r4.detail["event"] == "GYM_BEAM_PLAN"  # cooldown elapsed -> re-plan again
    assert r4.detail["beam_llm_calls"] == calls_after_r1


def test_beam_plan_injection_then_follow_then_replan(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
        WM_BEAM_MPC_EXECUTE_STEPS=2,
    )
    seed = [{"name": "seed_tool", "arguments": {}}]
    r1 = wm.beam_plan_step(_FLOW, seed_calls=seed, user_query="do the task")
    assert r1.detail["replanned"] is True and r1.detail["injected"] is True

    injection = wm.beam_injection_text()
    # the cached imagined ROLLOUT is now visible: alternating action -> predicted state
    assert "good_tool" in injection
    assert "action:" in injection and "predicted state:" in injection
    assert "execution_status=success" in injection  # informative field rendered
    assert "error_signature" not in injection  # 'none' is filtered as uninformative
    assert "missing_information_type" not in injection  # ['none'] is filtered as uninformative
    assert "NOT confirmed results" in injection  # anti-surrender guidance appended

    # advisory replan sets cursor=0, so the plan (len 2) is followed for execute_steps steps
    # before the next re-plan.
    events = []
    for _ in range(4):
        r = wm.beam_plan_step(_FLOW, seed_calls=seed, user_query="do the task")
        events.append(r.detail["event"])
        if r.detail["event"] == "GYM_BEAM_PLAN_FOLLOW":
            assert r.detail["calls"][0]["name"] == "seed_tool"  # follow keeps the baseline
    assert "GYM_BEAM_PLAN_FOLLOW" in events  # at least one follow before re-planning
    assert "GYM_BEAM_PLAN" in events  # cadence eventually triggers a re-plan
    assert wm.agent_call_count > 0


def test_beam_plan_terminal_probability_injection(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
        WM_BEAM_MPC_EXECUTE_STEPS=2,
        WM_BEAM_PLAN_TERMINAL_ADVICE=1,
    )
    wm._wm = _FakeScoringJepa(terminal_prob=0.91)
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert res.detail["imagined_plan"][0]["terminal_probability"] == pytest.approx(0.91)
    assert res.detail["beam_plan_terminal_advice"] is True

    injection = wm.beam_injection_text()
    assert "terminal_probability=0.91" in injection
    assert "Terminal advisory" in injection
    assert "recommended_abstract_action" not in injection
    assert "missing_information_type" not in injection


def test_beam_plan_terminal_middle_probability_injection(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
        WM_BEAM_MPC_EXECUTE_STEPS=2,
        WM_BEAM_PLAN_TERMINAL_ADVICE=1,
        WM_BEAM_PLAN_TERMINAL_ADVICE_THRESHOLD=0.75,
    )
    wm._wm = _FakeScoringJepa(terminal_prob=0.10)
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert res.detail["imagined_plan"][0]["terminal_probability"] == pytest.approx(0.10)
    assert "beam_plan_terminal_middle_threshold" not in res.detail

    injection = wm.beam_injection_text()
    assert "terminal_probability=0.10" in injection
    assert "Progress advisory" in injection
    assert "may not be finished" in injection
    assert "threshold=0.75" in injection
    assert "Terminal advisory" not in injection


def test_beam_plan_critic_terminal_advice_sets_next_prompt(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_TRIGGER="critic",
        WM_BEAM_PLAN_TERMINAL_ADVICE=1,
        WM_BEAM_PLAN_TERMINAL_ADVICE_THRESHOLD=0.75,
    )
    wm._wm = _FakeScoringJepa(high="seed_tool", terminal_prob=0.88)
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert res.detail["event"] == "GYM_BEAM_PLAN_COOLDOWN"
    assert res.detail["critic"]["terminal_probability"] == pytest.approx(0.88)
    assert res.detail["terminal_advice_set"] is True

    injection = wm.beam_injection_text()
    assert "previous step may have completed the task" in injection
    assert "P(done)=0.88" in injection
    assert wm.beam_injection_text() == ""


def test_beam_plan_critic_terminal_middle_advice_sets_next_prompt(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_TRIGGER="critic",
        WM_BEAM_PLAN_TERMINAL_ADVICE=1,
        WM_BEAM_PLAN_TERMINAL_ADVICE_THRESHOLD=0.75,
    )
    wm._wm = _FakeScoringJepa(high="seed_tool", terminal_prob=0.08)
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert res.detail["event"] == "GYM_BEAM_PLAN_COOLDOWN"
    assert res.detail["critic"]["terminal_probability"] == pytest.approx(0.08)
    assert res.detail["critic"]["terminal_middle_advice"] is True
    assert res.detail["critic"]["terminal_advice_type"] == "middle"
    assert res.detail["terminal_advice_set"] is True

    injection = wm.beam_injection_text()
    assert "previous step may not have completed the task" in injection
    assert "P(done)=0.08" in injection
    assert "threshold=0.75" in injection
    assert wm.beam_injection_text() == ""


def test_beam_plan_reset_episode_clears_state(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(monkeypatch, fake_jepa_module, WM_BEAM_MPC_EXECUTE_STEPS=3)
    wm.beam_plan_step(_FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t")
    assert wm.beam_injection_text() != ""
    wm.reset_episode()
    assert wm.beam_injection_text() == ""
    assert wm.agent_call_count == 0


def test_beam_plan_score_margin_keeps_baseline(monkeypatch, fake_jepa_module):
    # A huge margin means the beam's better candidate never beats the seed by enough -> no override.
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=1,
        WM_BEAM_PLAN_SCORE_MARGIN=1000,
    )
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert res.detail["override_applied"] is False
    assert res.detail["override_reason"] == "kept_baseline_below_margin"
    assert res.detail["calls"][0]["name"] == "seed_tool"


def test_beam_plan_no_seed_is_noop(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(monkeypatch, fake_jepa_module)
    res = wm.beam_plan_step(_FLOW, seed_calls=[], user_query="t")
    assert res.detail["event"] == "GYM_BEAM_PLAN_NO_SEED"
    assert res.detail["calls"] == []


def test_beam_plan_unsupported_backend_falls_back():
    # A WM whose generator has no score_action_plans_canonical_event cannot beam-plan.
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    wm = EwmImaginedWorldModel.__new__(EwmImaginedWorldModel)
    wm._wm = object()  # no scoring method
    assert wm.supports_beam_plan() is False


# --- decoded tool output (obs_grounding decoder) --------------------------------


def test_beam_plan_decode_tool_output_disabled_by_default(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
        WM_BEAM_MPC_EXECUTE_STEPS=2,
    )
    wm._wm = _FakeScoringJepa(obs_grounding=True)  # decoder available, but the flag is off
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert res.detail["beam_confident"] is True
    assert res.detail["decoded_tool_output"] is False
    assert wm._wm.decode_calls == []
    assert all(entry.get("predicted_tool_output") is None for entry in res.detail["imagined_plan"])
    assert "predicted tool output" not in wm.beam_injection_text()


def test_beam_plan_decode_tool_output_enabled_with_obs_grounding(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
        WM_BEAM_MPC_EXECUTE_STEPS=2,
        WM_BEAM_PLAN_DECODE_TOOL_OUTPUT=1,
        WM_BEAM_PLAN_DECODE_MAX_NEW_TOKENS=64,
    )
    fake = _FakeScoringJepa(obs_grounding=True, decode_texts=["ok: created", "ok: sent"])
    wm._wm = fake
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert res.detail["beam_confident"] is True
    assert res.detail["decoded_tool_output"] is True
    assert len(fake.decode_calls) == 1
    assert fake.decode_calls[0]["max_new_tokens"] == 64
    assert len(fake.decode_calls[0]["plan"]) == res.detail["imagined_plan_len"]
    previews = res.detail["imagined_plan"]
    assert [entry["predicted_tool_output"] for entry in previews] == ["ok: created", "ok: sent"]
    injection = wm.beam_injection_text()
    assert "step 1 predicted tool output: ok: created" in injection
    assert "step 2 predicted tool output: ok: sent" in injection


def test_beam_plan_decode_tool_output_skips_without_obs_grounding(
    monkeypatch, fake_jepa_module, caplog
):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=1,
        WM_BEAM_MPC_EXECUTE_STEPS=2,
        WM_BEAM_PLAN_DECODE_TOOL_OUTPUT=1,
    )
    fake = _FakeScoringJepa(obs_grounding=False)  # checkpoint has no trained decoder
    wm._wm = fake
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert res.detail["beam_confident"] is True
    assert res.detail["injected"] is True  # canonical-event labels alone still inject fine
    assert res.detail["decoded_tool_output"] is False
    assert fake.decode_calls == []  # never attempted -- no decoder to call


def test_beam_plan_decode_tool_output_failure_falls_back(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=1,
        WM_BEAM_MPC_EXECUTE_STEPS=2,
        WM_BEAM_PLAN_DECODE_TOOL_OUTPUT=1,
    )
    fake = _FakeScoringJepa(obs_grounding=True, decode_error=RuntimeError("decode blew up"))
    wm._wm = fake
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert res.detail["beam_confident"] is True
    assert res.detail["injected"] is True  # the confident plan itself is unaffected
    assert res.detail["decoded_tool_output"] is False
    assert all(entry.get("predicted_tool_output") is None for entry in res.detail["imagined_plan"])


def test_beam_plan_decode_tool_output_truncated_in_preview(monkeypatch, fake_jepa_module):
    long_text = "x" * 500
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=1,
        WM_BEAM_MPC_EXECUTE_STEPS=2,
        WM_BEAM_PLAN_DECODE_TOOL_OUTPUT=1,
    )
    fake = _FakeScoringJepa(obs_grounding=True, decode_texts=[long_text])
    wm._wm = fake
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    preview_text = res.detail["imagined_plan"][0]["predicted_tool_output"]
    assert len(preview_text) == 300  # _plan_preview truncates to 300 chars
    injection = wm.beam_injection_text()
    assert "...(truncated)" in injection


# --- open-loop rollout mode (WM_IMAGINED_ROLLOUT_MODE=open_loop) ----------------


def _open_loop_plan_texts() -> list:
    """Two distinct single-plan replies, in the bare step-array shape build_single_plan_prompt
    actually asks for (one plan per sample -- NOT build_skeleton_prompt's old array-of-arrays-
    of-plans shape). The second step of the first plan uses a symbolic $step1.field ref
    (open-loop's whole point is that this survives verbatim -- nothing tries to resolve it)."""
    return [
        json.dumps(
            [
                {"name": "good_tool", "arguments": {"x": 1}},
                {"name": "follow_up_tool", "arguments": {"ref": "$step1.id"}},
            ]
        ),
        json.dumps([{"name": "bad_tool", "arguments": {}}]),
    ]


def _open_loop_chat_fn(calls: list):
    """Plain chat_fn with no ``generate_samples`` -- sample_many falls back to k serial/parallel
    single-plan requests (see generate_many's serial-loop path for a plain _ChatFnAgent).
    Alternates between the two distinct plan texts across those k calls, so tests requesting
    WM_BEAM_PLAN_SAMPLES=2 still see both candidate plans."""
    texts = _open_loop_plan_texts()

    def chat_fn(messages):
        calls.append(messages)
        return texts[(len(calls) - 1) % len(texts)]

    return chat_fn


class _FakeSamplingAgent:
    """Exposes generate_samples so open-loop can use one shared n=k request."""

    def __init__(self):
        self.calls: list = []
        self.sample_calls: list = []
        self.temperature_calls: list = []

    def generate_from_messages(self, messages, temperature=0.0):
        self.calls.append(messages)
        self.temperature_calls.append(temperature)
        return _open_loop_plan_texts()[0]

    def generate_samples(self, messages, temperature=0.0, num_samples=1, response_format=None):
        self.sample_calls.append((messages, temperature, num_samples, response_format))
        texts = _open_loop_plan_texts()
        return [texts[i % len(texts)] for i in range(num_samples)]


def _make_open_loop_beam_wm(monkeypatch, fake_jepa_module, chat_fn=None, **env):
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/ckpt")
    monkeypatch.setenv("WM_IMAGINED_ROLLOUT_MODE", "open_loop")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    calls: list = []
    wm = EwmImaginedWorldModel(wm_config_from_env(), chat_fn or _open_loop_chat_fn(calls))
    return wm, calls


def test_chat_fn_agent_parallel_requests_default_and_escape_hatch(monkeypatch, fake_jepa_module):
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/ckpt")
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    wm = EwmImaginedWorldModel(wm_config_from_env(), lambda messages: "[]")
    assert wm._agent.supports_parallel_requests is True

    monkeypatch.setenv("WM_AGENT_PARALLEL_REQUESTS", "0")
    wm_disabled = EwmImaginedWorldModel(wm_config_from_env(), lambda messages: "[]")
    assert wm_disabled._agent.supports_parallel_requests is False


def test_chat_fn_agent_delegates_num_samples_callback(monkeypatch, fake_jepa_module):
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/ckpt")
    monkeypatch.setenv("WM_IMAGINED_ROLLOUT_MODE", "open_loop")
    monkeypatch.setenv("WM_BEAM_PLAN_SAMPLES", "2")
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    calls: list = []

    def chat_fn(messages, *, temperature=0.0, num_samples=1):
        calls.append((messages, temperature, num_samples))
        return _open_loop_plan_texts()[:num_samples]

    wm = EwmImaginedWorldModel(wm_config_from_env(), chat_fn)
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert len(calls) == 1
    assert calls[0][1] == pytest.approx(0.7)
    assert calls[0][2] == 2
    assert "DIVERSITY OPTIONS:" in calls[0][0][1]["content"]
    assert res.detail["beam_llm_calls"] == 1
    assert wm.agent_call_count == 1


def test_open_loop_uses_shared_diversity_prompt_generate_samples(monkeypatch, fake_jepa_module):
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/ckpt")
    monkeypatch.setenv("WM_IMAGINED_ROLLOUT_MODE", "open_loop")
    monkeypatch.setenv("WM_BEAM_PLAN_SAMPLES", "2")
    monkeypatch.setenv("WM_BEAM_PLAN_HORIZON", "3")
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    agent = _FakeSamplingAgent()
    wm = EwmImaginedWorldModel(
        wm_config_from_env(), lambda messages: agent.generate_from_messages(messages)
    )
    wm._agent = agent
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert len(agent.sample_calls) == 1
    messages, temperature, num_samples, response_format = agent.sample_calls[0]
    assert temperature == pytest.approx(0.7)
    assert num_samples == 2
    assert response_format is None
    assert agent.calls == []
    assert "DIVERSITY OPTIONS:" in messages[1]["content"]
    assert "DIVERSITY SLOT" not in messages[1]["content"]
    assert res.detail["beam_llm_calls"] == 1
    assert wm.agent_call_count == 1
    assert res.detail["imagined_rollout_mode"] == "open_loop"
    assert res.detail["open_loop_candidate_stats"]["requested"] == 2
    diagnostics = res.detail["candidate_score_diagnostics"]
    assert len(diagnostics) == 3
    assert {candidate["depth"] for candidate in diagnostics} == {"open_loop"}
    assert all(
        {"raw_score", "raw_rank", "adjusted_score", "adjusted_rank", "planner_rank"}
        <= candidate.keys()
        for candidate in diagnostics
    )


def test_open_loop_iterative_refinement_uses_prior_scored_trajectories(
    monkeypatch, fake_jepa_module
):
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/ckpt")
    monkeypatch.setenv("WM_IMAGINED_ROLLOUT_MODE", "closed_loop")
    monkeypatch.setenv("WM_BEAM_PLAN_SAMPLES", "2")
    monkeypatch.setenv("WM_BEAM_PLAN_HORIZON", "2")
    monkeypatch.setenv("WM_BEAM_PLAN_REFINEMENT_ROUNDS", "2")
    monkeypatch.setenv("WM_BEAM_PLAN_REFINEMENT_TOP_K", "2")
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    class RefiningAgent(_FakeSamplingAgent):
        def generate_samples(self, messages, temperature=0.0, num_samples=1, response_format=None):
            self.sample_calls.append((messages, temperature, num_samples, response_format))
            if "PREVIOUS SCORED TRAJECTORIES" not in messages[1]["content"]:
                return _open_loop_plan_texts()[:num_samples]
            refined = json.dumps(
                [
                    {"name": "good_tool", "arguments": {"x": 1}},
                    {"name": "bad_tool", "arguments": {"ref": "$step1.id"}},
                ]
            )
            alternative = json.dumps(
                [
                    {"name": "follow_up_tool", "arguments": {}},
                    {"name": "good_tool", "arguments": {}},
                ]
            )
            return [refined, alternative][:num_samples]

    agent = RefiningAgent()
    wm = EwmImaginedWorldModel(
        wm_config_from_env(), lambda messages: agent.generate_from_messages(messages)
    )
    wm._agent = agent
    result = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )

    assert wm.imagined_rollout_mode == "open_loop"
    assert len(agent.sample_calls) == 2
    refinement_prompt = agent.sample_calls[1][0][1]["content"]
    assert "PREVIOUS SCORED TRAJECTORIES" in refinement_prompt
    assert '"score": 5.0' in refinement_prompt
    assert '"predicted_states"' in refinement_prompt
    assert result.detail["beam_plan_refinement_rounds"] == 2
    assert result.detail["beam_refinement_rounds_completed"] == 2
    assert result.detail["beam_refinement_score_passes"] == 2
    assert result.detail["beam_llm_calls"] == 2
    assert result.detail["open_loop_candidate_stats"]["accepted"] == 4
    assert result.detail["open_loop_candidate_stats"]["rounds"][1]["used_prior_scores"] is True


def test_open_loop_uses_dedicated_action_sampler(monkeypatch, fake_jepa_module):
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/ckpt")
    monkeypatch.setenv("WM_IMAGINED_ROLLOUT_MODE", "open_loop")
    monkeypatch.setenv("WM_BEAM_PLAN_SAMPLES", "2")
    monkeypatch.setenv("WM_BEAM_ACTION_SAMPLER_BASE_URL", "http://sampler.test/v1")
    monkeypatch.setenv("WM_BEAM_ACTION_SAMPLER_MODEL", "diffusion-action-sampler")
    monkeypatch.setenv("WM_BEAM_ACTION_SAMPLER_MAX_NEW_TOKENS", "192")
    from ejepa_wm.backends import _ewm_runtime as runtime
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    samplers = []

    class FakeDedicatedSampler(_FakeSamplingAgent):
        def __init__(self, model, base_url, api_key, max_new_tokens, timeout):
            super().__init__()
            self.model = model
            self.base_url = base_url
            self.api_key = api_key
            self.max_new_tokens = max_new_tokens
            self.timeout = timeout
            samplers.append(self)

    monkeypatch.setattr(runtime, "EwmGenerator", FakeDedicatedSampler)
    policy_calls = []
    wm = EwmImaginedWorldModel(wm_config_from_env(), _open_loop_chat_fn(policy_calls))
    result = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )

    sampler = samplers[0]
    assert policy_calls == []
    assert len(sampler.sample_calls) == 1
    assert sampler.sample_calls[0][2] == 2
    assert sampler.max_new_tokens == 192
    assert sampler.supports_parallel_requests is True
    assert result.detail["beam_action_sampler_enabled"] is True
    assert result.detail["beam_action_sampler_backend"] == "openai"
    assert result.detail["beam_action_sampler_model"] == "diffusion-action-sampler"
    stats = result.detail["open_loop_candidate_stats"]
    assert stats["generator"] == "dedicated_action_sampler"
    assert stats["requests_issued"] == 1
    assert stats["accepted"] == 2
    assert stats["generation_seconds"] >= 0.0


def test_open_loop_uses_huggingface_action_sampler(monkeypatch, fake_jepa_module):
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/ckpt")
    monkeypatch.setenv("WM_IMAGINED_ROLLOUT_MODE", "open_loop")
    monkeypatch.setenv("WM_BEAM_PLAN_SAMPLES", "2")
    monkeypatch.setenv("WM_BEAM_ACTION_SAMPLER_BACKEND", "huggingface")
    monkeypatch.setenv("WM_BEAM_ACTION_SAMPLER_MAX_DENOISING_STEPS", "24")
    from ejepa_wm.backends import _hf_diffusion_action_sampler as hf_sampler
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    samplers = []

    class FakeHuggingFaceSampler(_FakeSamplingAgent):
        def __init__(self, model, **kwargs):
            super().__init__()
            self.model = model
            self.kwargs = kwargs
            samplers.append(self)

    monkeypatch.setattr(hf_sampler, "HuggingFaceDiffusionActionSampler", FakeHuggingFaceSampler)
    policy_calls = []
    wm = EwmImaginedWorldModel(wm_config_from_env(), _open_loop_chat_fn(policy_calls))
    result = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )

    assert policy_calls == []
    assert samplers[0].model == "google/diffusiongemma-26B-A4B-it"
    assert samplers[0].kwargs["max_denoising_steps"] == 24
    assert len(samplers[0].sample_calls) == 1
    assert result.detail["beam_action_sampler_backend"] == "huggingface"
    assert result.detail["open_loop_candidate_stats"]["requests_issued"] == 1


def test_open_loop_rejects_invalid_sampled_tools_and_arguments(monkeypatch, fake_jepa_module):
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/ckpt")
    monkeypatch.setenv("WM_IMAGINED_ROLLOUT_MODE", "open_loop")
    monkeypatch.setenv("WM_BEAM_PLAN_SAMPLES", "3")
    monkeypatch.setenv("WM_BEAM_ACTION_SAMPLER_BASE_URL", "http://sampler.test/v1")
    from ejepa_wm.backends import _ewm_runtime as runtime
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    class InvalidSampleSampler:
        def __init__(self, *args, **kwargs):
            pass

        def generate_samples(self, messages, temperature=0.0, num_samples=1):
            return [
                json.dumps([{"name": "hallucinated_tool", "arguments": {}}]),
                json.dumps([{"name": "good_tool", "arguments": "not-an-object"}]),
                json.dumps([{"name": "bad_tool", "arguments": {}}]),
            ]

    monkeypatch.setattr(runtime, "EwmGenerator", InvalidSampleSampler)
    wm = EwmImaginedWorldModel(wm_config_from_env(), lambda messages: "[]")
    result = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )

    stats = result.detail["open_loop_candidate_stats"]
    assert stats["parsed"] == 3
    assert stats["wrapped"] == 3
    assert stats["invalid_tools"] == 1
    assert stats["invalid_arguments"] == 1
    assert stats["accepted"] == 1


def test_open_loop_soft_accepts_generated_schema_argument_warnings(
    monkeypatch, fake_jepa_module
):
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/ckpt")
    monkeypatch.setenv("WM_IMAGINED_ROLLOUT_MODE", "open_loop")
    monkeypatch.setenv("WM_BEAM_PLAN_SAMPLES", "2")
    monkeypatch.setenv("WM_BEAM_PLAN_HORIZON", "1")
    monkeypatch.setenv("WM_BEAM_PLAN_HARD_OVERRIDE", "1")
    monkeypatch.setenv("WM_BEAM_ACTION_SAMPLER_BASE_URL", "http://sampler.test/v1")
    from ejepa_wm.backends import _ewm_runtime as runtime
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    flow = [
        {"type": "system_message", "content": "you are an agent"},
        {"type": "user_message", "content": "do the task"},
        {
            "type": "tools",
            "tools": [
                {
                    "name": "good_tool",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "required": ["x"],
                    },
                },
                {
                    "name": "bad_tool",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                    },
                },
                {"name": "seed_tool", "inputSchema": {"type": "object", "properties": {}}},
            ],
        },
    ]

    class SchemaWarningSampler:
        def __init__(self, *args, **kwargs):
            pass

        def generate_samples(self, messages, temperature=0.0, num_samples=1, response_format=None):
            return [
                json.dumps([{"name": "good_tool", "arguments": {}}]),
                json.dumps([{"name": "bad_tool", "arguments": {"count": "two"}}]),
            ][:num_samples]

    monkeypatch.setattr(runtime, "EwmGenerator", SchemaWarningSampler)
    wm = EwmImaginedWorldModel(wm_config_from_env(), lambda messages: "[]")
    result = wm.beam_plan_step(
        flow, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )

    stats = result.detail["open_loop_candidate_stats"]
    assert stats["invalid_arguments"] == 0
    assert stats["accepted"] == 2
    assert stats["schema_warned_candidates"] == 2
    assert stats["missing_required_arguments"] == 1
    assert stats["invalid_argument_types"] == 1
    assert result.detail["calls"][0]["name"] == "seed_tool"
    assert result.detail["recommended_calls"] is None
    assert result.detail["override_reason"] == "schema_warning_first_step"
    assert result.detail["injected"] is True
    assert result.detail["beam_schema_soft_validation_config"] == {
        "missing_required_penalty": 0.15,
        "invalid_type_penalty": 0.25,
    }

    diagnostics = result.detail["candidate_score_diagnostics"]
    good_diag = next(item for item in diagnostics if item["first_action_names"] == ["good_tool"])
    assert good_diag["schema_warning_count"] == 1
    assert good_diag["schema_warning_score_delta"] == pytest.approx(-0.15)
    assert good_diag["schema_warnings"][0]["reason"] == "missing_required_argument"
    assert result.detail["imagined_plan"][0]["schema_warnings"][0]["reason"] == (
        "missing_required_argument"
    )


def test_open_loop_ssot_diversity_is_shared_prompt_telemetry(monkeypatch, fake_jepa_module):
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/ckpt")
    monkeypatch.setenv("WM_IMAGINED_ROLLOUT_MODE", "open_loop")
    monkeypatch.setenv("WM_BEAM_PLAN_SAMPLES", "2")
    monkeypatch.setenv("WM_BEAM_PLAN_HORIZON", "2")
    monkeypatch.setenv("WM_BEAM_PLAN_SSOT_DIVERSITY", "1")
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    agent = _FakeSamplingAgent()
    wm = EwmImaginedWorldModel(
        wm_config_from_env(), lambda messages: agent.generate_from_messages(messages)
    )
    wm._agent = agent
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )

    messages, temperature, num_samples, response_format = agent.sample_calls[0]
    assert temperature == pytest.approx(0.7)
    assert num_samples == 2
    assert response_format == {"type": "json_object"}
    assert "SSoT DIVERSITY PROCEDURE:" in messages[1]["content"]
    assert res.detail["beam_plan_ssot_diversity"] is True
    assert res.detail["open_loop_candidate_stats"]["ssot_diversity"] is True


def test_open_loop_temperature_ladder_forces_per_sample_temperatures(monkeypatch, fake_jepa_module):
    # The ladder must bypass generate_samples(n=k), because one request can carry only one
    # sampling configuration.  The reference ladder keeps one exploit slot and caps exploration.
    monkeypatch.setenv("WM_STRATEGY", "beam_plan")
    monkeypatch.setenv("WM_EWM_JEPA_CHECKPOINT", "/fake/ckpt")
    monkeypatch.setenv("WM_IMAGINED_ROLLOUT_MODE", "open_loop")
    monkeypatch.setenv("WM_BEAM_PLAN_SAMPLES", "2")
    monkeypatch.setenv("WM_BEAM_PLAN_HORIZON", "2")
    monkeypatch.setenv("WM_SAMPLE_TEMPERATURE_LADDER", "1")
    monkeypatch.setenv("WM_SAMPLE_TEMPERATURE_LADDER_MAX", "0.9")
    from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

    agent = _FakeSamplingAgent()
    wm = EwmImaginedWorldModel(
        wm_config_from_env(), lambda messages: agent.generate_from_messages(messages)
    )
    wm._agent = agent
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )

    assert agent.sample_calls == []
    assert agent.temperature_calls == pytest.approx([0.28, 0.9])
    assert res.detail["beam_llm_calls"] == 2
    assert res.detail["sample_temperature_ladder"] is True
    assert res.detail["sample_temperature_ladder_max"] == pytest.approx(0.9)


def test_open_loop_k_request_fallback_without_generate_samples(monkeypatch, fake_jepa_module):
    # A plain chat_fn wrapper cannot express n=k, so sample_many falls back to k logical requests
    # with the same diversity-menu prompt. generate_many may still dispatch them concurrently.
    calls: list = []
    wm_open, _ = _make_open_loop_beam_wm(
        monkeypatch,
        fake_jepa_module,
        chat_fn=_open_loop_chat_fn(calls),
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=3,
    )
    res_open = wm_open.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert len(calls) == 2
    assert res_open.detail["beam_llm_calls"] == 2
    assert wm_open.agent_call_count == 2
    assert res_open.detail["imagined_rollout_mode"] == "open_loop"

    wm_closed = _make_beam_wm(
        monkeypatch, fake_jepa_module, WM_BEAM_PLAN_SAMPLES=2, WM_BEAM_PLAN_HORIZON=3
    )
    res_closed = wm_closed.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert res_closed.detail["beam_llm_calls"] == 3
    assert wm_closed.agent_call_count == 3
    assert res_closed.detail["imagined_rollout_mode"] == "closed_loop"


def test_open_loop_recommends_winner_and_preserves_symbolic_ref(monkeypatch, fake_jepa_module):
    calls: list = []
    wm, _ = _make_open_loop_beam_wm(
        monkeypatch,
        fake_jepa_module,
        chat_fn=_open_loop_chat_fn(calls),
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
    )
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="do the task"
    )
    d = res.detail
    assert d["beam_confident"] is True
    assert d["recommended_calls"][0]["name"] == "good_tool"  # beats the seed and bad_tool
    assert d["imagined_plan_len"] == 2  # the full 2-step winning plan
    # the symbolic ref survives verbatim -- nothing tries to resolve/guess it
    assert d["imagined_plan"][1]["calls"][0]["arguments"]["ref"] == "$step1.id"
    assert d["calls"][0]["name"] == "seed_tool"  # advisory: seed still executes


def _symbolic_flow() -> list[dict[str, Any]]:
    return [
        {"type": "system_message", "content": "you are an agent"},
        {
            "type": "user_message",
            "content": "Find Ethan Well and assign the incident to that user.",
        },
        {
            "type": "tools",
            "tools": [
                {
                    "name": "find_user",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
                {
                    "name": "assign_incident",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"user_id": {"type": "string"}},
                        "required": ["user_id"],
                    },
                },
                {"name": "seed_tool", "inputSchema": {"type": "object", "properties": {}}},
            ],
        },
    ]


def _symbolic_plan_text() -> str:
    return json.dumps(
        [
            {
                "name": "find_user",
                "arguments": {"name": "Ethan Well"},
                "bind": {
                    "ethan_user_id": {
                        "field": "sys_id",
                        "match": {"name": "Ethan Well"},
                    }
                },
            },
            {
                "name": "assign_incident",
                "arguments": {"user_id": "$vars.ethan_user_id"},
            },
        ]
    )


def test_open_loop_hard_override_resolves_symbolic_cached_followup(monkeypatch, fake_jepa_module):
    def chat_fn(_messages):
        return _symbolic_plan_text()

    wm, _ = _make_open_loop_beam_wm(
        monkeypatch,
        fake_jepa_module,
        chat_fn=chat_fn,
        WM_BEAM_PLAN_SAMPLES=1,
        WM_BEAM_PLAN_HORIZON=2,
        WM_BEAM_MPC_EXECUTE_STEPS=4,
        WM_BEAM_PLAN_HARD_OVERRIDE=1,
    )
    wm._wm = _FakeScoringJepa(high="find_user")
    first = wm.beam_plan_step(
        _symbolic_flow(),
        seed_calls=[{"name": "seed_tool", "arguments": {}}],
        user_query="Find Ethan Well and assign the incident to that user.",
    )
    assert first.detail["calls"] == [{"name": "find_user", "arguments": {"name": "Ethan Well"}}]
    assert first.detail["imagined_plan"][1]["calls"][0]["arguments"]["user_id"] == (
        "$vars.ethan_user_id"
    )

    flow_with_result = [
        *_symbolic_flow(),
        {
            "type": "ai_message",
            "content": "",
            "tool_calls": [{"name": "find_user", "arguments": {"name": "Ethan Well"}}],
        },
        {
            "type": "tool_result",
            "tool_name": "find_user",
            "result": [{"name": "Ethan Well", "sys_id": "user-123456"}],
        },
    ]
    second = wm.beam_plan_step(
        flow_with_result,
        seed_calls=[{"name": "seed_tool", "arguments": {}}],
        user_query="Find Ethan Well and assign the incident to that user.",
    )

    assert second.detail["event"] == "GYM_BEAM_PLAN_FOLLOW"
    assert second.detail["calls"] == [
        {"name": "assign_incident", "arguments": {"user_id": "user-123456"}}
    ]
    assert second.detail["symbolic_binding"]["unresolved"] == []
    assert second.detail["symbolic_binding"]["bindings"][0]["reference"] == ("$vars.ethan_user_id")


def test_open_loop_hard_override_does_not_execute_unresolved_symbolic_followup(
    monkeypatch, fake_jepa_module
):
    def chat_fn(_messages):
        return _symbolic_plan_text()

    wm, _ = _make_open_loop_beam_wm(
        monkeypatch,
        fake_jepa_module,
        chat_fn=chat_fn,
        WM_BEAM_PLAN_SAMPLES=1,
        WM_BEAM_PLAN_HORIZON=2,
        WM_BEAM_MPC_EXECUTE_STEPS=4,
        WM_BEAM_PLAN_HARD_OVERRIDE=1,
    )
    wm._wm = _FakeScoringJepa(high="find_user")
    wm.beam_plan_step(
        _symbolic_flow(),
        seed_calls=[{"name": "seed_tool", "arguments": {}}],
        user_query="Find Ethan Well and assign the incident to that user.",
    )
    flow_with_ambiguous_result = [
        *_symbolic_flow(),
        {
            "type": "ai_message",
            "content": "",
            "tool_calls": [{"name": "find_user", "arguments": {"name": "Ethan Well"}}],
        },
        {
            "type": "tool_result",
            "tool_name": "find_user",
            "result": [
                {"name": "Ethan Well", "sys_id": "user-123456"},
                {"name": "Ethan Well", "sys_id": "user-789012"},
            ],
        },
    ]

    second = wm.beam_plan_step(
        flow_with_ambiguous_result,
        seed_calls=[{"name": "seed_tool", "arguments": {}}],
        user_query="Find Ethan Well and assign the incident to that user.",
    )

    assert second.detail["calls"] == [{"name": "seed_tool", "arguments": {}}]
    assert second.detail["symbolic_binding"]["unresolved"][0]["reason"] == "binding_ambiguous"
    assert wm._beam_imagined_plan == []


def test_open_loop_dedup_keeps_same_first_tool_different_later_steps(monkeypatch, fake_jepa_module):
    calls: list = []
    plan_texts = [
        json.dumps(
            [
                {"name": "good_tool", "arguments": {"x": 1}},
                {"name": "follow_up_tool", "arguments": {"ref": "$step1.id"}},
            ]
        ),
        json.dumps(
            [
                {"name": "good_tool", "arguments": {"x": 1}},
                {"name": "bad_tool", "arguments": {"ref": "$step1.id"}},
            ]
        ),
    ]

    def chat_fn(messages):
        calls.append(messages)
        return plan_texts[(len(calls) - 1) % len(plan_texts)]

    wm, _ = _make_open_loop_beam_wm(
        monkeypatch,
        fake_jepa_module,
        chat_fn=chat_fn,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
    )
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    stats = res.detail["open_loop_candidate_stats"]
    assert stats["accepted"] == 2
    assert stats["duplicates"] == 0
    assert res.detail["num_candidates"] == 3  # seed + two generated plans


def test_open_loop_dedup_collapses_scalar_only_variants(monkeypatch, fake_jepa_module):
    calls: list = []
    plan_texts = [
        json.dumps([{"name": "good_tool", "arguments": {"record_id": "A"}}]),
        json.dumps([{"name": "good_tool", "arguments": {"record_id": "B"}}]),
    ]

    def chat_fn(messages):
        calls.append(messages)
        return plan_texts[(len(calls) - 1) % len(plan_texts)]

    wm, _ = _make_open_loop_beam_wm(
        monkeypatch,
        fake_jepa_module,
        chat_fn=chat_fn,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=1,
    )
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    stats = res.detail["open_loop_candidate_stats"]
    assert stats["accepted"] == 1
    assert stats["duplicates"] == 1
    assert res.detail["num_candidates"] == 2  # seed + one unique generated plan


def test_open_loop_winner_is_never_the_seed_even_if_it_scores_highest(
    monkeypatch, fake_jepa_module
):
    calls: list = []
    wm, _ = _make_open_loop_beam_wm(
        monkeypatch,
        fake_jepa_module,
        chat_fn=_open_loop_chat_fn(calls),
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
    )
    wm._wm = _FakeScoringJepa(high="seed_tool")  # the seed's OWN tool scores highest
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    d = res.detail
    # the winner must be one of the GENERATED plans -- the seed is structurally excluded from
    # winner selection, so "no generated plan beats the seed" surfaces as a margin rejection,
    # never as the seed itself being reported as the winning trajectory.
    assert d["best_reason"] in {"good_tool", "bad_tool"}
    # Since the seed dominates the softmax here, the winning generated plan's normalized score
    # is tiny -> this actually fails the trajectory-level confidence gate before the margin
    # comparison is even consulted. Either rejection reason is a correct "don't override" outcome.
    assert d["override_reason"] in {"kept_baseline_below_margin", "below_confidence_gate"}
    assert d["override_applied"] is False
    assert d["calls"][0]["name"] == "seed_tool"


def test_open_loop_anti_repetition_flips_winner(monkeypatch, fake_jepa_module):
    calls: list = []
    wm, _ = _make_open_loop_beam_wm(
        monkeypatch,
        fake_jepa_module,
        chat_fn=_open_loop_chat_fn(calls),
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
    )
    wm._wm = _FakeScoringJepa(high=None)  # flat: every plan scores equally on raw score
    wm._beam_recent_tool_names = [("good_tool",)]  # good_tool was already executed this task
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    # good_tool's repeat penalty flips the winner to bad_tool despite equal raw scores
    assert res.detail["best_reason"] == "bad_tool"


def test_open_loop_per_depth_normalized_repeats_winner_score(monkeypatch, fake_jepa_module):
    calls: list = []
    wm, _ = _make_open_loop_beam_wm(
        monkeypatch,
        fake_jepa_module,
        chat_fn=_open_loop_chat_fn(calls),
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
    )
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    d = res.detail
    # one softmax over PLANS (not one per depth) -- repeating the winner's normalized score once
    # per depth keeps trajectory_avg_normalized_score on the same scale the confidence gate uses.
    assert d["trajectory_avg_normalized_score"] == pytest.approx(d["best_normalized_score"])
    assert d["full_horizon_reached"] is True
    assert d["beam_confident"] is True


def test_open_loop_garbage_output_falls_back_to_no_candidates(monkeypatch, fake_jepa_module):
    def garbage_chat_fn(messages):
        return "not json at all, sorry"

    wm, _ = _make_open_loop_beam_wm(
        monkeypatch,
        fake_jepa_module,
        chat_fn=garbage_chat_fn,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
    )
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    d = res.detail
    assert d["event"] == "GYM_BEAM_PLAN_OPEN_LOOP_NO_CANDIDATES"
    assert d["override_applied"] is False
    assert d["calls"][0]["name"] == "seed_tool"  # agent's own action stands, no crash


def test_open_loop_hard_override_forces_winner(monkeypatch, fake_jepa_module):
    calls: list = []
    wm, _ = _make_open_loop_beam_wm(
        monkeypatch,
        fake_jepa_module,
        chat_fn=_open_loop_chat_fn(calls),
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
        WM_BEAM_PLAN_HARD_OVERRIDE=1,
    )
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    d = res.detail
    assert d["override_applied"] is True
    assert d["calls"][0]["name"] == "good_tool"  # forced over the agent's own action
    assert d["agent_calls"][0]["name"] == "seed_tool"  # original proposal preserved in telemetry


# --- beam_plan_trigger=critic ---------------------------------------------------


def test_beam_plan_critic_does_not_fire_on_a_good_action(monkeypatch, fake_jepa_module):
    # The seed action's OWN tool matches `high` -> the critic scores it as low failure/stall
    # probability -> no re-plan, no agent (LLM) call spent at all.
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_TRIGGER="critic",
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
    )
    wm._wm = _FakeScoringJepa(high="seed_tool")  # the seed IS the "good" action here
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    d = res.detail
    assert d["event"] == "GYM_BEAM_PLAN_COOLDOWN"
    assert d["trigger"] == "critic"
    assert d["critic"]["fires"] is False
    assert d["critic"]["reason"] == "action_looks_good"
    assert d["critic_checks"] == 1 and d["critic_fires"] == 0
    assert d["calls"][0]["name"] == "seed_tool"
    assert wm.agent_call_count == 0  # critic is a WM-only forward, no LLM call


def test_beam_plan_critic_fires_on_a_bad_action_and_triggers_full_planning(
    monkeypatch, fake_jepa_module
):
    # The seed action's tool does NOT match `high` -> high failure/stall probability -> fires,
    # escalating to a full planning cycle (same mechanics as interval's re-plan).
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_TRIGGER="critic",
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
    )
    wm._wm = _FakeScoringJepa(high="good_tool")  # "seed_tool" != "good_tool" -> looks bad
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    d = res.detail
    assert d["event"] == "GYM_BEAM_PLAN"
    assert d["trigger"] == "critic"
    assert d["critic"]["fires"] is True
    assert "P(failure)" in d["critic"]["reason"] and "P(no_progress)" in d["critic"]["reason"]
    assert d["critic_checks"] == 1 and d["critic_fires"] == 1
    assert d["recommended_calls"][0]["name"] == "good_tool"  # full planning found + recommended it
    assert wm.agent_call_count == 2  # the full horizon-2 planning cycle's own LLM calls


def test_beam_plan_critic_max_quiet_steps_forces_a_replan(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_TRIGGER="critic",
        WM_BEAM_PLAN_CRITIC_MAX_QUIET_STEPS=2,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=1,
    )
    wm._wm = _FakeScoringJepa(high="seed_tool")  # never fires on its own
    seed = [{"name": "seed_tool", "arguments": {}}]
    r1 = wm.beam_plan_step(_FLOW, seed_calls=seed, user_query="t")
    assert r1.detail["event"] == "GYM_BEAM_PLAN_COOLDOWN"
    r2 = wm.beam_plan_step(_FLOW, seed_calls=seed, user_query="t")
    assert r2.detail["event"] == "GYM_BEAM_PLAN_COOLDOWN"
    # after 2 consecutive non-firing checks, the safety valve forces a planning cycle
    r3 = wm.beam_plan_step(_FLOW, seed_calls=seed, user_query="t")
    assert r3.detail["event"] == "GYM_BEAM_PLAN"
    assert r3.detail["critic"]["fires"] is False  # the critic itself still says "looks fine"
    assert r3.detail["critic_fires"] == 1  # but quiet_exceeded still counted as a fire


def test_beam_plan_critic_error_escalates_to_full_planning(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_TRIGGER="critic",
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=1,
    )

    class _BrokenScoringJepa(_FakeScoringJepa):
        def score_action_plans_canonical_event(self, **kwargs):
            if len(kwargs["action_plans"]) == 1:  # the critic's own one-step-plan probe
                raise RuntimeError("boom")
            return super().score_action_plans_canonical_event(**kwargs)

    wm._wm = _BrokenScoringJepa(high="good_tool")
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    d = res.detail
    assert d["critic"]["fires"] is True
    assert d["critic"]["reason"] == "critic_error:RuntimeError"
    assert d["event"] == "GYM_BEAM_PLAN"  # a broken critic must not silently disable planning


def test_beam_plan_interval_trigger_unaffected_by_critic_config(monkeypatch, fake_jepa_module):
    # interval is the default trigger; critic_* knobs must be no-ops when it's selected.
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_CRITIC_FAILURE_PROB=0.0,  # would fire on everything if read
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
        WM_BEAM_PLAN_HARD_OVERRIDE=1,
    )
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    d = res.detail
    assert d["trigger"] == "interval"
    assert d["critic"] is None
    assert d["critic_checks"] == 0 and d["critic_fires"] == 0


class _VetoingCriticJepa(_FakeScoringJepa):
    """Seed action that looks fine on both critic thresholds but trips the score config's veto."""

    def score_action_plans_canonical_event(self, **kwargs):
        out = super().score_action_plans_canonical_event(**kwargs)
        if len(kwargs["action_plans"]) == 1 and out:  # the critic's own one-step probe
            out[0]["vetoed"] = True
            out[0]["veto_reasons"] = [{"step": 1, "reasons": ["p_failure_limit"]}]
        return out


def test_beam_plan_critic_veto_fires_by_default(monkeypatch, fake_jepa_module):
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_TRIGGER="critic",
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=1,
    )
    wm._wm = _VetoingCriticJepa(high="seed_tool")  # thresholds say "fine", the veto says no
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert res.detail["critic"]["fires"] is True
    assert res.detail["critic"]["reason"].startswith("veto(")
    assert res.detail["event"] == "GYM_BEAM_PLAN"


def test_beam_plan_critic_veto_fires_can_be_disabled(monkeypatch, fake_jepa_module):
    # The veto is a signal the two probability thresholds cannot bound, so it sets a floor on the
    # fire rate. Disabling it leaves FAILURE_PROB/STALL_PROB as the only levers.
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_TRIGGER="critic",
        WM_BEAM_PLAN_CRITIC_VETO_FIRES=0,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=1,
    )
    wm._wm = _VetoingCriticJepa(high="seed_tool")
    res = wm.beam_plan_step(
        _FLOW, seed_calls=[{"name": "seed_tool", "arguments": {}}], user_query="t"
    )
    assert res.detail["critic"]["vetoed"] is True  # still reported...
    assert res.detail["critic"]["fires"] is False  # ...but no longer a trigger
    assert res.detail["event"] == "GYM_BEAM_PLAN_COOLDOWN"


def test_beam_plan_critic_quiet_longer_than_the_plan_replans(monkeypatch, fake_jepa_module):
    # MAX_QUIET_STEPS (6) > HORIZON (2): a critic that stays quiet outlives the cached plan. The
    # follow branch indexes _beam_imagined_plan[_beam_plan_cursor] for every non-firing step, so
    # running off the end must force a planning cycle rather than raise IndexError (which the
    # executor would swallow into its baseline fallback, silently losing lookahead).
    wm = _make_beam_wm(
        monkeypatch,
        fake_jepa_module,
        WM_BEAM_PLAN_TRIGGER="critic",
        WM_BEAM_PLAN_CRITIC_MAX_QUIET_STEPS=6,
        WM_BEAM_PLAN_SAMPLES=2,
        WM_BEAM_PLAN_HORIZON=2,
    )
    seed = [{"name": "seed_tool", "arguments": {}}]
    wm._wm = _FakeScoringJepa(high="good_tool")  # seed_tool looks bad -> the critic fires once
    assert wm.beam_plan_step(_FLOW, seed_calls=seed, user_query="t").detail["event"] == (
        "GYM_BEAM_PLAN"
    )
    plan_len = len(wm._beam_imagined_plan)
    assert plan_len > 0  # there is a cached plan to follow

    wm._wm = _FakeScoringJepa(high="seed_tool")  # now the seed looks fine -> the critic goes quiet
    events = [
        wm.beam_plan_step(_FLOW, seed_calls=seed, user_query="t").detail["event"]
        for _ in range(plan_len + 1)
    ]
    # Quiet steps follow the cached plan, then the exhausted plan forces one re-plan -- well
    # before MAX_QUIET_STEPS=6 would have.
    assert events.count("GYM_BEAM_PLAN") == 1
    assert wm._beam_quiet_steps < wm.beam_critic_max_quiet_steps
    assert wm._beam_plan_cursor <= len(wm._beam_imagined_plan)
