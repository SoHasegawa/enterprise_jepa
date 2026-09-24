"""Tests for LLM tool-output prediction plus same-agent judge beam scoring."""
from __future__ import annotations

import json

from ejepa_wm.backends._ewm_llm_tool_output_judge import LlmToolOutputJudgeGenerator


def _plan_step(name: str) -> dict:
    return {"tool_calls": [{"type": "function", "function": {"name": name, "arguments": {}}}]}


class _ToolOutputWorldModel:
    supports_parallel_requests = False

    def generate_from_messages(self, messages, temperature=0.0):
        prompt = messages[-1]["content"]
        if "bad_tool" in prompt:
            return "Error: missing required object id"
        return '{"status":"ok","record":"created"}'


class _SameAgentJudge:
    supports_parallel_requests = False

    def __init__(self):
        self.step_judges = 0
        self.trajectory_judges = 0

    def generate_from_messages(self, messages, temperature=0.0):
        system = messages[0]["content"]
        prompt = messages[-1]["content"]
        if "judging an imagined enterprise-agent step" in system:
            self.step_judges += 1
            if "Error:" in prompt:
                return json.dumps({
                    "can_proceed": False,
                    "step_score": -1.0,
                    "failure": True,
                    "failure_score": 0.95,
                    "progress_score": 0.0,
                    "enough_to_finish": False,
                    "finish_score": 0.0,
                    "reason": "predicted tool failure",
                })
            return json.dumps({
                "can_proceed": True,
                "step_score": 0.9,
                "failure": False,
                "failure_score": 0.0,
                "progress_score": 0.9,
                "enough_to_finish": True,
                "finish_score": 0.8,
                "reason": "useful completed step",
            })
        self.trajectory_judges += 1
        return json.dumps({
            "selected_index": 0,
            "scores": [
                {"index": 0, "score": 0.9, "reason": "best progress"},
                {"index": 1, "score": 0.1, "reason": "tool failure"},
            ],
            "comments": "choose non-failing plan",
        })


def test_tool_output_judge_scores_and_preserves_predicted_outputs():
    judge = _SameAgentJudge()
    wm = LlmToolOutputJudgeGenerator(_ToolOutputWorldModel(), judge)
    scored = wm.score_action_plans_canonical_event(
        system_prompt="sys",
        user_prompt="do the task",
        input_history=[],
        action_plans=[[_plan_step("good_tool")], [_plan_step("bad_tool")]],
    )

    assert judge.step_judges == 2
    assert judge.trajectory_judges == 1
    assert scored[0]["plan_index"] == 0
    assert scored[0]["score"] == 0.9
    assert scored[0]["terminal_probability"] == 0.8
    assert scored[0]["per_step_predicted_tool_output"] == ['{"status":"ok","record":"created"}']
    assert scored[0]["per_step_field_probs"][0]["execution_status"]["failure"] == 0.0
    bad = next(record for record in scored if record["plan_index"] == 1)
    assert bad["vetoed"] is True
    assert bad["per_step_field_probs"][0]["execution_status"]["failure"] == 0.95
