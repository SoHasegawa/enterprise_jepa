"""Self-contained with/without-WM evaluation demo.

Runs a tiny "test" set through the *same* code path twice — once with the no-WM baseline
(``strategy=none``) and once with a ``selection`` WM (``LlmWorldModel``) — and prints the
with/without-WM score table. This exercises the real ``WorldModel.select`` path and
``ejepa_wm.report`` without needing any benchmark infra or network.

The agent is mocked: for each task it samples ``K`` candidate actions, exactly one of which is
correct (marked internally). The baseline WM always takes candidate 0; the ``selection`` WM uses
a chat function that reads the candidates and returns the correct index — standing in for a real
LLM world model (swap ``chat_fn`` for ``build_world_model(WMConfig(..., backend='served'))`` for a real
gateway).

Run:  ``PYTHONPATH=src python -m ejepa_wm.examples.eval_with_without_wm``
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ejepa_wm import WMConfig, build_world_model, report


def _make_tasks(n_tasks: int = 8, k: int = 4) -> list[dict]:
    """Toy tasks: K candidate actions, exactly one correct. The correct one is rarely index 0,
    so a WM that can read the candidates should beat the first-candidate baseline."""
    tasks = []
    for t in range(n_tasks):
        correct = (t % (k - 1)) + 1  # 1..k-1 → never 0
        candidates = [
            {"type": "ai_message",
             "content": ("CORRECT: run the right tool" if i == correct else "wrong / off-task action"),
             "tool_calls": []}
            for i in range(k)
        ]
        tasks.append({
            "task_id": f"task-{t:02d}",
            "flow": [{"type": "user_message", "content": f"Solve task {t}"}],
            "candidates": candidates,
            "correct": correct,
        })
    return tasks


def _oracle_chat_fn(messages: list[dict[str, str]]) -> str:
    """Stand-in for a competent WM: return the index marked CORRECT."""
    prompt = messages[-1]["content"]
    # The select prompt lists candidates as "[i] <text>"; pick the CORRECT one.
    best = 0
    for m in re.finditer(r"\[(\d+)\]\s*(.*)", prompt):
        if "CORRECT" in m.group(2):
            best = int(m.group(1))
            break
    return str(best)


def run_arm(arm: str, tasks: list[dict], wm) -> list[dict]:
    records = []
    for task in tasks:
        result = wm.select(task["flow"], task["candidates"])
        success = result.index == task["correct"]
        records.append({"task_id": task["task_id"], "success": success, "chosen": result.index})
    return records


def main() -> None:
    tasks = _make_tasks()

    off_wm = build_world_model(WMConfig(strategy="none"))
    on_wm = build_world_model(WMConfig(strategy="selection", backend="llm", n=4), chat_fn=_oracle_chat_fn)

    off = run_arm("off", tasks, off_wm)
    on = run_arm("on", tasks, on_wm)

    out = Path(__file__).resolve().parent / "_eval_out"
    (out / "off").mkdir(parents=True, exist_ok=True)
    (out / "on").mkdir(parents=True, exist_ok=True)
    (out / "off" / "wm_eval_records.json").write_text(json.dumps(off, indent=2))
    (out / "on" / "wm_eval_records.json").write_text(json.dumps(on, indent=2))

    cmp = report.compare(off, on)
    report.write_report(out, cmp)
    print(report.format_table(cmp))
    print(f"\nRecords + report written under: {out}")
    print("Reproduce the comparison via the CLI:")
    print(f"  ejepa result wm-compare --off {out/'off'} --on {out/'on'}")


if __name__ == "__main__":
    main()
