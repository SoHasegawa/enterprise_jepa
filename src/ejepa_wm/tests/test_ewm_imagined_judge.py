"""Tests for the multi-rollout + LLM-judge imagined-trajectory selection (the decoder-friendly
path used by the seq2seq JEPA on terminal tasks). No torch: the judge and the optimize dispatch
are exercised with fakes; ``imagine_trajectory`` is monkeypatched to canned rollouts."""
from __future__ import annotations

import json

from ejepa_wm.backends import _ewm_runtime as ewm


class _FakeAgent:
    """Records prompts; returns a scripted judge reply."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate_from_messages(self, messages, temperature=0.0):
        self.calls.append((messages, temperature))
        return self.reply


def _rollout(idx, **flags):
    step = {"imagined_step": 1, "tool_calls": [{"name": "run_shell", "args": {}}], **flags}
    return {"rollout_index": idx, "imagined_steps": [step]}


def test_score_candidate_rewards_final_penalizes_errors():
    good = [{"final_answer": "done", "tool_calls": [1]}]
    bad = [{"parse_error": "boom"}]
    assert ewm.score_imagined_trajectory_candidate(good) > ewm.score_imagined_trajectory_candidate(bad)


def test_llm_judge_picks_selected_index():
    agent = _FakeAgent(json.dumps({"selected_index": 2, "scores": [], "comments": "c"}))
    cands = [_rollout(0), _rollout(1), _rollout(2)]
    out = ewm.select_imagined_trajectory_with_llm_judge(agent, "do the task", [], cands)
    assert out["selected_index"] == 2 and out["fallback_used"] is False


def test_llm_judge_strips_think_and_code_fence():
    agent = _FakeAgent('<think>reasoning</think>\n```json\n{"selected_index": 1}\n```')
    out = ewm.select_imagined_trajectory_with_llm_judge(agent, "t", [], [_rollout(0), _rollout(1)])
    assert out["selected_index"] == 1 and out["fallback_used"] is False


def test_llm_judge_falls_back_to_heuristic_on_bad_output():
    agent = _FakeAgent("not json at all")
    # rollout 1 has a final_answer -> heuristic prefers it
    cands = [_rollout(0, parse_error="x"), _rollout(1, final_answer="done")]
    out = ewm.select_imagined_trajectory_with_llm_judge(agent, "t", [], cands)
    assert out["fallback_used"] is True
    assert out["selected_index"] == 1


def test_llm_judge_rejects_out_of_range_index():
    agent = _FakeAgent(json.dumps({"selected_index": 9}))
    out = ewm.select_imagined_trajectory_with_llm_judge(agent, "t", [], [_rollout(0), _rollout(1)])
    assert out["fallback_used"] is True  # out-of-range -> heuristic fallback


def test_optimize_llm_judge_generates_n_rollouts_and_selects_sequential(monkeypatch):
    # parallel_rollouts=False exercises the original sequential "one imagine_trajectory call
    # per rollout" path explicitly (parallel_rollouts defaults to True -- see the companion
    # lockstep test below for the default dispatch).
    temps = []

    def fake_imagine(*a, **k):
        temps.append(k.get("temperature"))
        # tag each rollout by its temperature so we can identify the selected one
        return [{"imagined_step": 1, "temp": k.get("temperature")}]

    monkeypatch.setattr(ewm, "imagine_trajectory", fake_imagine)
    monkeypatch.setattr(
        ewm, "select_imagined_trajectory_with_llm_judge",
        lambda agent, user_prompt, conversation, cands: {"selected_index": 2, "fallback_used": False},
    )
    steps = ewm.optimize_imagined_trajectory(
        "", _FakeAgent(""), object(), [], {}, "sys",
        system_prompt="sys", user_prompt="task", wm_state="binary_error",
        max_imagined_steps=2, num_rollouts=3, selection_strategy="llm_judge",
        temperature=0.7, parallel_rollouts=False,
    )
    assert temps == [0.0, 0.7, 0.7]      # 3 rollouts, first deterministic
    assert steps[0]["temp"] == 0.7        # judge picked rollout index 2 (a sampled one)


def test_optimize_llm_judge_uses_lockstep_by_default(monkeypatch):
    # parallel_rollouts=True is the default: N>1 rollouts dispatch through
    # imagine_trajectories_lockstep (one batched call site) instead of N sequential
    # imagine_trajectory calls.
    lockstep_kwargs = {}

    def fake_lockstep(*a, **k):
        lockstep_kwargs.update(k)
        temps = k["rollout_temperatures"]
        return [[{"imagined_step": 1, "temp": t}] for t in temps]

    monkeypatch.setattr(ewm, "imagine_trajectories_lockstep", fake_lockstep)
    monkeypatch.setattr(
        ewm, "select_imagined_trajectory_with_llm_judge",
        lambda agent, user_prompt, conversation, cands: {"selected_index": 2, "fallback_used": False},
    )
    steps = ewm.optimize_imagined_trajectory(
        "", _FakeAgent(""), object(), [], {}, "sys",
        system_prompt="sys", user_prompt="task", wm_state="binary_error",
        max_imagined_steps=2, num_rollouts=3, selection_strategy="llm_judge",
        temperature=0.7,
    )
    assert lockstep_kwargs["rollout_temperatures"] == [0.0, 0.7, 0.7]
    assert steps[0]["temp"] == 0.7


def test_optimize_single_rollout_default(monkeypatch):
    calls = []
    monkeypatch.setattr(ewm, "imagine_trajectory", lambda *a, **k: calls.append(k.get("temperature")) or [])
    ewm.optimize_imagined_trajectory(
        "", _FakeAgent(""), object(), [], {}, "sys",
        system_prompt="sys", user_prompt="task", wm_state="binary_error",
        max_imagined_steps=2,  # defaults: num_rollouts=1, selection=first
    )
    assert calls == [0.0]  # exactly one deterministic rollout, no judge
