"""Tests for the open-loop text-WM imagined rollout (Variant A): the parser
(``parse_open_loop_plans``/``normalize_open_loop_plan_step``), the batched-generation
dispatcher (``generate_many``), and the driver (``imagine_trajectories_open_loop``) + its
wiring into ``optimize_imagined_trajectory``. No torch: ``_ewm_finetuning`` (aliased ``ft`` in
``_ewm_runtime``) has no torch dependency, so the real state/prompt helpers are used directly.
"""
from __future__ import annotations

import json

import pytest

from ejepa_wm.backends import _ewm_runtime as ewm

# ---------------------------------------------------------------------------
# parse_open_loop_plans / normalize_open_loop_plan_step
# ---------------------------------------------------------------------------


def test_parse_open_loop_plans_accepts_plans_wrapper():
    text = json.dumps({
        "plans": [
            {"strategy": "a", "steps": [{"tool_calls": [{"name": "toolA", "arguments": {"x": 1}}]}]},
            {"strategy": "b", "steps": [{"tool_calls": [{"name": "toolB", "arguments": {}}]}]},
        ]
    })
    plans = ewm.parse_open_loop_plans(text, num_plans=2, max_steps=3)
    assert len(plans) == 2
    assert plans[0]["strategy"] == "a"
    assert plans[0]["steps"][0]["tool_calls"][0]["name"] == "toolA"


def test_parse_open_loop_plans_accepts_bare_array_of_bare_step_arrays():
    text = json.dumps([
        [{"tool_calls": [{"name": "toolA", "arguments": {}}]}],
        [{"tool_calls": [{"name": "toolB", "arguments": {}}]}],
    ])
    plans = ewm.parse_open_loop_plans(text, num_plans=2, max_steps=3)
    assert len(plans) == 2
    assert plans[0]["strategy"] == ""  # bare arrays have no strategy string
    assert plans[1]["steps"][0]["tool_calls"][0]["name"] == "toolB"


def test_parse_open_loop_plans_tolerates_code_fence_and_prose():
    text = "Sure, here you go:\n```json\n" + json.dumps({
        "plans": [{"strategy": "a", "steps": [{"tool_calls": [{"name": "toolA", "arguments": {}}]}]}]
    }) + "\n```\nHope that helps!"
    plans = ewm.parse_open_loop_plans(text, num_plans=1, max_steps=3)
    assert len(plans) == 1
    assert plans[0]["steps"][0]["tool_calls"][0]["name"] == "toolA"


def test_parse_open_loop_plans_truncates_at_final_answer():
    text = json.dumps({
        "plans": [
            {
                "strategy": "a",
                "steps": [
                    {"tool_calls": [{"name": "toolA", "arguments": {}}]},
                    {"final_answer": "done"},
                    {"tool_calls": [{"name": "toolB", "arguments": {}}]},  # must be dropped
                ],
            }
        ]
    })
    plans = ewm.parse_open_loop_plans(text, num_plans=1, max_steps=5)
    assert len(plans[0]["steps"]) == 2
    assert "final_answer" in plans[0]["steps"][-1]


def test_parse_open_loop_plans_rejects_clarify_and_truncates_there():
    text = json.dumps({
        "plans": [
            {
                "strategy": "a",
                "steps": [
                    {"tool_calls": [{"name": "toolA", "arguments": {}}]},
                    {"action": "clarify", "action_input": {"question": "which one?"}},
                    {"tool_calls": [{"name": "toolB", "arguments": {}}]},  # must be dropped
                ],
            }
        ]
    })
    plans = ewm.parse_open_loop_plans(text, num_plans=1, max_steps=5)
    assert len(plans[0]["steps"]) == 1  # truncated before the clarify step


def test_parse_open_loop_plans_caps_at_max_steps():
    text = json.dumps([[
        {"tool_calls": [{"name": f"tool{i}", "arguments": {}}]} for i in range(5)
    ]])
    plans = ewm.parse_open_loop_plans(text, num_plans=1, max_steps=2)
    assert len(plans[0]["steps"]) == 2


def test_parse_open_loop_plans_returns_empty_on_garbage():
    assert ewm.parse_open_loop_plans("not json at all", num_plans=2, max_steps=3) == []
    assert ewm.parse_open_loop_plans("", num_plans=2, max_steps=3) == []


def test_normalize_open_loop_plan_step_rejects_clarify():
    assert ewm.normalize_open_loop_plan_step({"action": "clarify", "action_input": "q?"}) is None


def test_normalize_open_loop_plan_step_rejects_non_dict():
    assert ewm.normalize_open_loop_plan_step("not a dict") is None


# ---------------------------------------------------------------------------
# generate_many
# ---------------------------------------------------------------------------


class _CountingGenerator:
    def __init__(self, reply="ok"):
        self.reply = reply
        self.calls = []

    def generate_from_messages(self, messages, temperature=0.0):
        self.calls.append((messages, temperature))
        return self.reply


def test_generate_many_single_item_passthrough():
    gen = _CountingGenerator("hi")
    out = ewm.generate_many(gen, [[{"role": "user", "content": "x"}]], [0.0])
    assert out == ["hi"]
    assert len(gen.calls) == 1


def test_generate_many_serial_fallback_by_default():
    # No backend today declares supports_parallel_requests -- this is the path every existing
    # ejepa_wm backend actually takes, so it's the most important case to get right.
    gen = _CountingGenerator("hi")
    out = ewm.generate_many(gen, [[{"role": "user", "content": f"x{i}"}] for i in range(4)], [0.0] * 4)
    assert out == ["hi", "hi", "hi", "hi"]
    assert len(gen.calls) == 4


def test_generate_many_requires_matching_lengths():
    gen = _CountingGenerator()
    with pytest.raises(ValueError):
        ewm.generate_many(gen, [[{"role": "user", "content": "x"}]], [0.0, 0.7])


def test_generate_many_empty_input():
    assert ewm.generate_many(_CountingGenerator(), [], []) == []


class _ParallelGenerator:
    """Declares supports_parallel_requests; returns each message's own index as the reply so
    result ordering can be checked, and clones per worker thread."""

    supports_parallel_requests = True

    def __init__(self):
        self.clone_count = 0

    def clone_for_parallel_requests(self):
        self.clone_count += 1
        return self

    def generate_from_messages(self, messages, temperature=0.0):
        return messages[0]["content"]


def test_generate_many_thread_pool_path_preserves_order():
    gen = _ParallelGenerator()
    messages_list = [[{"role": "user", "content": str(i)}] for i in range(6)]
    out = ewm.generate_many(gen, messages_list, [0.0] * 6, max_workers=3)
    assert out == [str(i) for i in range(6)]


class _FlakyParallelGenerator:
    supports_parallel_requests = True

    def __init__(self):
        self.attempts = {}

    def generate_from_messages(self, messages, temperature=0.0):
        index = messages[0]["content"]
        self.attempts[index] = self.attempts.get(index, 0) + 1
        if index == "2" and self.attempts[index] == 1:
            raise RuntimeError("transient failure")
        return index


def test_generate_many_failed_parallel_request_retries_serially():
    gen = _FlakyParallelGenerator()
    messages_list = [[{"role": "user", "content": str(i)}] for i in range(4)]
    out = ewm.generate_many(gen, messages_list, [0.0] * 4, max_workers=4)
    assert out == ["0", "1", "2", "3"]
    assert gen.attempts["2"] == 2  # failed once, retried serially and succeeded


# ---------------------------------------------------------------------------
# imagine_trajectories_open_loop driver (via the predict_feedback local-adapter seam --
# avoids needing the full text-WM prompt/parse machinery to exercise the driver's own
# per-plan bookkeeping in isolation)
# ---------------------------------------------------------------------------


class _FakeAgentGenerator:
    """Plain agent generator with no ``generate_samples`` -- ``sample_many`` falls back to k
    serial calls via ``generate_many``. Cycles through ``replies`` across those calls (a single
    fixed ``reply`` is repeated for every call, matching the old fixed-reply behavior)."""

    def __init__(self, reply=None, replies=None):
        self.replies = replies if replies is not None else [reply]
        self.calls = []

    def generate_from_messages(self, messages, temperature=0.0):
        self.calls.append((messages, temperature))
        return self.replies[(len(self.calls) - 1) % len(self.replies)]


class _FakeSamplingAgentGenerator:
    """Exposes generate_samples: k plans in ONE request, matching EwmGenerator's real vLLM
    n=k capability -- used to test open-loop's "exactly 1 request" ideal path."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []
        self.sample_calls = []

    def generate_from_messages(self, messages, temperature=0.0):
        self.calls.append((messages, temperature))
        return self.replies[0]

    def generate_samples(self, messages, temperature=0.0, num_samples=1):
        self.sample_calls.append((messages, temperature, num_samples))
        return [self.replies[i % len(self.replies)] for i in range(num_samples)]


class _FakeWmGeneratorWithPredictFeedback:
    """Mirrors the JEPA generator's local-adapter seam: predict_feedback is a cheap in-process
    call, so imagine_trajectories_open_loop loops it directly with no generate_many batching."""

    def __init__(self, finish_after: int = 1):
        self.finish_after = finish_after
        self.calls = []

    def predict_feedback(self, *, system_prompt, user_prompt, planned_calls, previous_state,
                          state_history, wm_state, interaction_index):
        self.calls.append(planned_calls)
        step_num = len(self.calls)
        current_stage = "finished" if step_num >= self.finish_after else "in_progress"
        return [{
            "tool_calls": planned_calls,
            "predicted_success": True,
            "predicted_state": {"current_stage": current_stage},
            "predicted_tool_output": "",
            "predicted_error_message": "",
            "predicted_current_stage": current_stage,
            "predicted_remaining_stages": None,
            "predicted_canonical_event": None,
            "predicted_nudge": None,
            "raw_prediction": "",
            "parse_error": None,
        }]


# Two DISTINCT single-plan replies (the shape build_react_open_loop_plan_messages actually
# asks each sample for -- one plan per sample, NOT the old {"plans": [...]} multi-plan shape).
_SINGLE_PLAN_TEXTS = [
    json.dumps({"strategy": "good first", "steps": [
        {"tool_calls": [{"name": "good_tool", "arguments": {"x": 1}}]},
        {"tool_calls": [{"name": "follow_up", "arguments": {"ref": "$step1.id"}}]},
    ]}),
    json.dumps({"strategy": "bad first", "steps": [
        {"tool_calls": [{"name": "bad_tool", "arguments": {}}]},
    ]}),
]


def test_open_loop_driver_one_request_via_generate_samples_backend():
    agent = _FakeSamplingAgentGenerator(_SINGLE_PLAN_TEXTS)
    wm = _FakeWmGeneratorWithPredictFeedback(finish_after=99)  # never finishes early
    plans = ewm.imagine_trajectories_open_loop(
        agent, wm, conversation=[], previous_state={}, react_system_prompt="You are an agent.",
        system_prompt="sys", user_prompt="do the task", wm_state="binary_error",
        max_imagined_steps=2, num_rollouts=2,
    )
    assert len(agent.sample_calls) == 1
    assert agent.sample_calls[0][2] == 2  # num_samples == num_rollouts
    assert agent.calls == []              # never fell back to per-sample generate_from_messages
    assert plans is not None
    assert len(plans) == 2


def test_open_loop_driver_k_request_fallback_without_generate_samples():
    agent = _FakeAgentGenerator(replies=_SINGLE_PLAN_TEXTS)
    wm = _FakeWmGeneratorWithPredictFeedback(finish_after=99)
    plans = ewm.imagine_trajectories_open_loop(
        agent, wm, conversation=[], previous_state={}, react_system_prompt="You are an agent.",
        system_prompt="sys", user_prompt="do the task", wm_state="binary_error",
        max_imagined_steps=2, num_rollouts=2,
    )
    assert len(agent.calls) == 2  # k separate requests, still fewer than a per-rollout serial loop
    assert plans is not None
    assert len(plans) == 2


def test_open_loop_driver_preserves_symbolic_ref_and_step_shape():
    agent = _FakeAgentGenerator(replies=_SINGLE_PLAN_TEXTS)
    wm = _FakeWmGeneratorWithPredictFeedback(finish_after=99)
    plans = ewm.imagine_trajectories_open_loop(
        agent, wm, conversation=[], previous_state={}, react_system_prompt="You are an agent.",
        system_prompt="sys", user_prompt="do the task", wm_state="binary_error",
        max_imagined_steps=2, num_rollouts=2,
    )
    first_plan_steps = plans[0]
    assert len(first_plan_steps) == 2
    assert first_plan_steps[0]["open_loop"] is True
    assert first_plan_steps[0]["tool_calls"][0]["name"] == "good_tool"
    # the symbolic ref survives verbatim -- nothing tries to resolve/guess it
    assert first_plan_steps[1]["tool_calls"][0]["arguments"]["ref"] == "$step1.id"
    # same record shape imagine_trajectory produces, plus the open_loop marker
    for key in ("imagined_step", "thought", "tool_calls", "repeated_tool_call_loop",
                "predicted_feedback", "predicted_state"):
        assert key in first_plan_steps[0]


def test_open_loop_driver_stops_plan_when_state_is_finished():
    agent = _FakeAgentGenerator(replies=_SINGLE_PLAN_TEXTS)
    wm = _FakeWmGeneratorWithPredictFeedback(finish_after=1)  # every plan finishes after step 1
    plans = ewm.imagine_trajectories_open_loop(
        agent, wm, conversation=[], previous_state={}, react_system_prompt="You are an agent.",
        system_prompt="sys", user_prompt="do the task", wm_state="binary_error",
        max_imagined_steps=5, num_rollouts=2,
    )
    # the 2-step plan never gets to run its second step -- state_is_finished stopped it after 1
    assert len(plans[0]) == 1


def test_open_loop_driver_returns_none_on_total_parse_failure():
    agent = _FakeAgentGenerator("not json at all, sorry")
    wm = _FakeWmGeneratorWithPredictFeedback()
    plans = ewm.imagine_trajectories_open_loop(
        agent, wm, conversation=[], previous_state={}, react_system_prompt="You are an agent.",
        system_prompt="sys", user_prompt="do the task", wm_state="binary_error",
        max_imagined_steps=2, num_rollouts=2,
    )
    assert plans is None
    assert wm.calls == []  # never even attempted a world-model call


# ---------------------------------------------------------------------------
# optimize_imagined_trajectory: open_loop dispatch + closed-loop fallback/regression
# ---------------------------------------------------------------------------


def test_optimize_imagined_trajectory_open_loop_vs_closed_loop_call_counts():
    agent_open = _FakeAgentGenerator(_SINGLE_PLAN_TEXTS[0])  # num_rollouts defaults to 1 -> 1 sample
    wm = _FakeWmGeneratorWithPredictFeedback(finish_after=99)
    steps = ewm.optimize_imagined_trajectory(
        "", agent_open, wm, [], {}, "You are an agent.",
        system_prompt="sys", user_prompt="do the task", wm_state="binary_error",
        max_imagined_steps=2, rollout_mode="open_loop",
    )
    assert len(agent_open.calls) == 1
    assert len(steps) == 2  # the "first" (index 0) plan's 2 steps

    class _ClosedLoopAgent:
        def __init__(self):
            self.calls = []

        def generate_from_messages(self, messages, temperature=0.0):
            self.calls.append(messages)
            # alternate thought/action replies for imagine_trajectory's think+act loop
            if len(self.calls) % 2 == 1:
                return json.dumps({"thought": "thinking"})
            return json.dumps({"tool_calls": [{"name": "good_tool", "arguments": {}}]})

    agent_closed = _ClosedLoopAgent()
    wm_closed = _FakeWmGeneratorWithPredictFeedback(finish_after=99)
    ewm.optimize_imagined_trajectory(
        "", agent_closed, wm_closed, [], {}, "You are an agent.",
        system_prompt="sys", user_prompt="do the task", wm_state="binary_error",
        max_imagined_steps=2, rollout_mode="closed_loop",
    )
    # closed-loop: 2 agent calls (think + act) per imagined step, for 2 steps = 4 calls
    assert len(agent_closed.calls) == 4


def test_optimize_imagined_trajectory_falls_back_to_closed_loop_on_open_loop_parse_failure():
    agent = _FakeAgentGenerator("garbage, not parseable")
    wm = _FakeWmGeneratorWithPredictFeedback(finish_after=99)

    class _FallbackAgent:
        """First call (the open-loop plans prompt) returns garbage; subsequent calls (the
        closed-loop think/act fallback) return valid replies."""

        def __init__(self):
            self.calls = []

        def generate_from_messages(self, messages, temperature=0.0):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return "garbage, not parseable"
            if len(self.calls) % 2 == 0:
                return json.dumps({"thought": "thinking"})
            return json.dumps({"final_answer": "done"})

    fallback_agent = _FallbackAgent()
    steps = ewm.optimize_imagined_trajectory(
        "", fallback_agent, wm, [], {}, "You are an agent.",
        system_prompt="sys", user_prompt="do the task", wm_state="binary_error",
        max_imagined_steps=2, rollout_mode="open_loop",
    )
    assert len(fallback_agent.calls) > 1  # fell through to the closed-loop rollout, didn't give up
    assert steps and "final_answer" in steps[-1]


def test_optimize_imagined_trajectory_closed_loop_default_is_unaffected():
    # rollout_mode defaults to "closed_loop" -- omitting it must behave exactly as before.
    class _Agent:
        def __init__(self):
            self.calls = []

        def generate_from_messages(self, messages, temperature=0.0):
            self.calls.append(messages)
            if len(self.calls) % 2 == 1:
                return json.dumps({"thought": "thinking"})
            return json.dumps({"final_answer": "done"})

    agent = _Agent()
    wm = _FakeWmGeneratorWithPredictFeedback(finish_after=99)
    steps = ewm.optimize_imagined_trajectory(
        "", agent, wm, [], {}, "You are an agent.",
        system_prompt="sys", user_prompt="do the task", wm_state="binary_error",
        max_imagined_steps=2,
    )
    assert steps and "final_answer" in steps[-1]
    assert len(agent.calls) == 2  # one think + one act, then final_answer stops the rollout
