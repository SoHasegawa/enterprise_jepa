from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_wm_harnesses.py"
SPEC = importlib.util.spec_from_file_location("run_wm_harnesses", SCRIPT_PATH)
assert SPEC is not None
matrix = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)


def base_argv(tmp_path: Path) -> list[str]:
    return [
        "--dry-run",
        "--output-dir",
        str(tmp_path),
        "--",
        "ejepa",
        "bench",
        "run",
        "EnterpriseOps-Gym",
        "--executor",
        "mcp_react",
        "--config",
        "target=sample",
        "--wm-ewm-jepa-checkpoint",
        "/model/checkpoint",
    ]


def test_default_plan_runs_all_five_harnesses_once(tmp_path: Path) -> None:
    assert matrix.main(base_argv(tmp_path)) == 0

    summaries = list(tmp_path.glob("*.json"))
    assert len(summaries) == 1
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    names = [run["harness"] for run in payload["runs"]]
    assert names == list(matrix.HARNESS_NAMES)
    assert len(names) == len(set(names)) == 5
    assert all("capture_trajectory=true" in run["command"] for run in payload["runs"])


def test_periodic_and_critic_beam_commands_differ_only_by_trigger_controls(
    tmp_path: Path,
) -> None:
    args = matrix.parse_args(base_argv(tmp_path))
    commands = {item.name: item.arguments for item in matrix.harnesses(args)}

    assert "interval" in commands["beam_interval"]
    assert "critic" in commands["beam_critic"]
    assert "--wm-beam-plan-critic-failure-prob" not in commands["beam_interval"]
    assert "--wm-beam-plan-critic-failure-prob" in commands["beam_critic"]
    assert commands["revision"] == ("--wm-strategy", "revision")
    assert commands["reference"] == ("--wm-strategy", "reference")


def test_rejects_strategy_in_base_command(tmp_path: Path) -> None:
    argv = [*base_argv(tmp_path), "--wm-strategy", "revision"]
    assert matrix.main(argv) == 2
