"""Tests for the lockstep CLOSED-loop multi-rollout driver (``imagine_trajectories_lockstep``),
``build_react_step_messages`` (the single-call think+act prompt), and their wiring into
``optimize_imagined_trajectory`` via ``parallel_rollouts``/``single_call_step``. No torch.
"""
from __future__ import annotations

import json

from ejepa_wm.backends import _ewm_runtime as ewm


class _FakeAgentGenerator:
    """Plain agent generator (no batching/parallel-request seam) that answers deterministically
    based on which phase it's asked for: build_react_step_messages's combined prompt always
    mentions "BOTH your reasoning and the action"; build_react_think_messages's prompt asks to
    "Think:"; build_react_action_messages's asks to "select and execute". Distinguishing by
    prompt content lets one fake serve both single-call and two-call modes correctly."""

    def __init__(self, action_name: str = "good_tool", final_after: int | None = None):
        self.action_name = action_name
        self.final_after = final_after
        self.calls: list[tuple[list, float]] = []

    def generate_from_messages(self, messages, temperature=0.0):
        self.calls.append((messages, temperature))
        prompt = messages[-1]["content"]
        step_num = sum(1 for c in self.calls if "BOTH your reasoning" in c[0][-1]["content"] or "Think:" in c[0][-1]["content"])
        if self.final_after is not None and step_num > self.final_after:
            final_answer = True
        else:
            final_answer = False
        if "BOTH your reasoning" in prompt:
            if final_answer:
                return json.dumps({"thought": "done", "final_answer": "all set"})
            return json.dumps({"thought": "thinking", "tool_calls": [{"name": self.action_name, "arguments": {}}]})
        if "Think:" in prompt:
            return json.dumps({"thought": "thinking"})
        # action-only phase
        if final_answer:
            return json.dumps({"final_answer": "all set"})
        return json.dumps({"tool_calls": [{"name": self.action_name, "arguments": {}}]})


class _FakeWmGeneratorWithPredictFeedback:
    def __init__(self, finish_after: int = 99):
        self.finish_after = finish_after
        self.calls: list = []

    def predict_feedback(self, *, system_prompt, user_prompt, planned_calls, previous_state,
                          state_history, wm_state, interaction_index):
        self.calls.append(planned_calls)
        current_stage = "finished" if len(self.calls) >= self.finish_after else "in_progress"
        return [{
            "tool_calls": planned_calls,
            "predicted_success": True,
            "predicted_state": {"current_stage": current_stage},
            "predicted_tool_output": "",
            "predicted_error_message": "",
            "raw_prediction": "",
            "parse_error": None,
        }]


def test_build_react_step_messages_asks_for_combined_thought_and_action():
    messages = ewm.build_react_step_messages([], current_query="do the task", system_prompt="You are an agent.")
    prompt = messages[-1]["content"]
    assert "BOTH your reasoning and the action" in prompt
    assert '"thought"' in prompt


def test_lockstep_single_call_step_halves_agent_call_count_vs_two_call():
    agent_single = _FakeAgentGenerator(final_after=2)
    wm = _FakeWmGeneratorWithPredictFeedback()
    ewm.imagine_trajectories_lockstep(
        agent_single, wm, conversation=[], previous_state={}, react_system_prompt="You are an agent.",
        system_prompt="sys", user_prompt="do the task", wm_state="binary_error",
        max_imagined_steps=2, rollout_temperatures=[0.0, 0.7], single_call_step=True,
    )
    calls_single = len(agent_single.calls)

    agent_two = _FakeAgentGenerator(final_after=2)
    wm2 = _FakeWmGeneratorWithPredictFeedback()
    ewm.imagine_trajectories_lockstep(
        agent_two, wm2, conversation=[], previous_state={}, react_system_prompt="You are an agent.",
        system_prompt="sys", user_prompt="do the task", wm_state="binary_error",
        max_imagined_steps=2, rollout_temperatures=[0.0, 0.7], single_call_step=False,
    )
    calls_two = len(agent_two.calls)
    assert calls_two == calls_single * 2


def test_lockstep_batches_agent_calls_across_alive_rollouts_per_step():
    # generate_many is called once per phase per step with ALL alive rollouts' messages, not
    # once per rollout -- verify by wrapping generate_many and checking each call's batch size.
    batch_sizes = []
    original_generate_many = ewm.generate_many

    def counting_generate_many(generator, messages_list, temperatures, **kwargs):
        batch_sizes.append(len(messages_list))
        return original_generate_many(generator, messages_list, temperatures, **kwargs)

    agent = _FakeAgentGenerator(final_after=99)
    wm = _FakeWmGeneratorWithPredictFeedback()
    try:
        ewm.generate_many = counting_generate_many
        ewm.imagine_trajectories_lockstep(
            agent, wm, conversation=[], previous_state={}, react_system_prompt="You are an agent.",
            system_prompt="sys", user_prompt="do the task", wm_state="binary_error",
            max_imagined_steps=2, rollout_temperatures=[0.0, 0.7, 0.7], single_call_step=True,
        )
    finally:
        ewm.generate_many = original_generate_many
    # 3 rollouts alive for 2 steps -> each of the 2 agent-decision phase calls batches all 3
    assert batch_sizes == [3, 3]


def test_lockstep_terminates_rollouts_independently():
    agent = _FakeAgentGenerator(final_after=0)  # every rollout finalizes on its first real step
    wm = _FakeWmGeneratorWithPredictFeedback()
    steps = ewm.imagine_trajectories_lockstep(
        agent, wm, conversation=[], previous_state={}, react_system_prompt="You are an agent.",
        system_prompt="sys", user_prompt="do the task", wm_state="binary_error",
        max_imagined_steps=5, rollout_temperatures=[0.0, 0.7],
    )
    assert len(steps) == 2
    for rollout_steps in steps:
        assert "final_answer" in rollout_steps[-1]
        assert len(rollout_steps) == 1  # stopped immediately, didn't run all 5 steps


def test_lockstep_advances_state_via_predict_feedback_seam():
    agent = _FakeAgentGenerator(final_after=99)
    wm = _FakeWmGeneratorWithPredictFeedback(finish_after=2)
    steps = ewm.imagine_trajectories_lockstep(
        agent, wm, conversation=[], previous_state={}, react_system_prompt="You are an agent.",
        system_prompt="sys", user_prompt="do the task", wm_state="binary_error",
        max_imagined_steps=5, rollout_temperatures=[0.0],
    )
    rollout_steps = steps[0]
    assert rollout_steps[0]["tool_calls"][0]["name"] == "good_tool"
    assert rollout_steps[-1]["predicted_state"]["current_stage"] == "finished"
    assert len(rollout_steps) == 2  # stopped once state_is_finished


def test_optimize_imagined_trajectory_parallel_rollouts_toggle_off_uses_sequential(monkeypatch):
    calls = []

    def fake_imagine(*a, **k):
        calls.append(k.get("temperature"))
        return [{"imagined_step": 1}]

    monkeypatch.setattr(ewm, "imagine_trajectory", fake_imagine)

    def fail_lockstep(*a, **k):
        raise AssertionError("imagine_trajectories_lockstep should not be called when parallel_rollouts=False")

    monkeypatch.setattr(ewm, "imagine_trajectories_lockstep", fail_lockstep)

    class _Agent:
        def generate_from_messages(self, messages, temperature=0.0):
            return ""

    ewm.optimize_imagined_trajectory(
        "", _Agent(), object(), [], {}, "sys",
        system_prompt="sys", user_prompt="task", wm_state="binary_error",
        max_imagined_steps=2, num_rollouts=2, parallel_rollouts=False,
    )
    assert len(calls) == 2
