#!/usr/bin/env python3
"""Reconstruct end-to-end token cost per task from saved trajectories.

The harness summaries record wall-clock but no token counts, so the policy's share
of the cost has to be rebuilt from the final ``conversation_flow`` in each task's
``PurpleInternalRecord`` (EnterpriseOps-Gym layout) or ``tool_calls``/messages
(other executors). A ReAct policy re-reads its whole context on every step, so

    policy prompt tokens  = sum over policy steps of (tokens of the flow prefix)
    policy output tokens  = tokens of the assistant messages

World-model calls are read from ``wm_steps`` / ``wm_react.steps`` where present.
Tokenisation uses a local Qwen-family ``tokenizer.json`` (default: the JEPA
checkpoint's, same vocabulary family as the Qwen3.6-27B policy), so counts are
estimates with the same tokenizer applied to every arm -- exact enough for FLOPs
ratios, not for billing.

    uv run python scripts/count_policy_tokens.py \
        --run "baseline=<result_dir>" --run "JEPA beam=<result_dir>" --run "LLM-WM beam=<result_dir>"

Requires the ``tokenizers`` package (present in assets/AutomationBench/purple/.venv).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

DEFAULT_TOKENIZER = Path("checkpoints/jepa/tokenizer.json")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=RESULT_DIR",
        help="A labelled result directory (repeatable).",
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--csv", type=Path, help="Write per-task rows here.")
    return parser.parse_args(argv)


def load_tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        print(
            "tokenizers not importable; run with assets/AutomationBench/purple/.venv/bin/python",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return Tokenizer.from_file(str(path))


def last_record(path: Path) -> dict[str, Any] | None:
    """The executor's final internal record / trajectory payload for one task."""
    record = None
    for line in path.read_text(errors="replace").splitlines():
        if (
            "PurpleInternalRecord" not in line
            and "wm_steps" not in line
            and "tool_calls" not in line
        ):
            continue
        try:
            payload = json.loads(line).get("payload")
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and (
            payload.get("conversation_flow") or payload.get("tool_calls") or payload.get("wm_steps")
        ):
            record = payload
    return record


def text_of(entry: Any) -> str:
    if isinstance(entry, dict):
        content = entry.get("content")
        if isinstance(content, str):
            return content
        return json.dumps(entry, ensure_ascii=False)
    return str(entry)


def count_task(record: dict[str, Any], tok) -> dict[str, Any]:
    flow = record.get("conversation_flow")
    n_tokens = lambda s: len(tok.encode(s).ids)
    if flow:
        lens = [n_tokens(text_of(e)) for e in flow]
        steps = prompt = output = 0
        prefix = 0
        for entry, length in zip(flow, lens, strict=True):
            kind = entry.get("type") if isinstance(entry, dict) else ""
            if kind == "ai_message":
                steps += 1
                prompt += prefix  # this step re-read everything before it
                output += length
            prefix += length
        context = prefix
    else:
        # AutomationBench-style payload: only tool calls survive; approximate the
        # context as the serialised call log and count one step per call.
        calls = record.get("tool_calls") or []
        lens = [n_tokens(json.dumps(c, ensure_ascii=False)) for c in calls]
        steps, prompt, output, prefix = 0, 0, 0, 0
        for length in lens:
            steps += 1
            prompt += prefix
            prefix += length
        output = None
        context = prefix
    wm = record.get("wm_react") or {}
    wm_steps = (
        wm.get("steps")
        if isinstance(wm, dict) and wm.get("steps")
        else record.get("wm_steps") or []
    )
    wm_calls = 0
    for s in wm_steps:
        d = s.get("detail") if isinstance(s.get("detail"), dict) else s
        wm_calls += int(
            d.get("world_model_calls")
            or s.get("world_model_calls")
            or (1 if s.get("replanned") else 0)
            or 0
        )
    return {
        "policy_steps": steps,
        "policy_prompt_tokens": prompt,
        "policy_output_tokens": output,
        "final_context_tokens": context,
        "wm_steps": len(wm_steps),
        "wm_calls": wm_calls,
    }


def summarise(label: str, result_dir: Path, tok) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for path in sorted((result_dir / "trajectories").glob("*.jsonl")):
        record = last_record(path)
        if not record:
            continue
        row = count_task(record, tok)
        row.update({"run": label, "task": path.stem})
        rows.append(row)
    if not rows:
        return {"run": label, "tasks": 0}, rows

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.mean(vals) if vals else None

    return {
        "run": label,
        "tasks": len(rows),
        "policy_steps": mean("policy_steps"),
        "policy_prompt_tokens": mean("policy_prompt_tokens"),
        "policy_output_tokens": mean("policy_output_tokens"),
        "final_context_tokens": mean("final_context_tokens"),
        "wm_steps": mean("wm_steps"),
        "wm_calls": mean("wm_calls"),
    }, rows


def fmt(v, digits=0):
    return "-" if v is None else f"{v:,.{digits}f}"


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.run:
        print("pass at least one --run LABEL=RESULT_DIR", file=sys.stderr)
        return 1
    tok = load_tokenizer(args.tokenizer)
    summaries, all_rows = [], []
    for spec in args.run:
        label, _, path = spec.partition("=")
        summary, rows = summarise(label, Path(path), tok)
        summaries.append(summary)
        all_rows.extend(rows)

    print(f"tokenizer: {args.tokenizer}\n")
    print(
        f"{'run':<22s} {'tasks':>5s} {'steps':>6s} {'prompt tok':>11s} {'output tok':>11s} "
        f"{'final ctx':>10s} {'WM steps':>9s} {'WM calls':>9s}"
    )
    for s in summaries:
        print(
            f"{s['run']:<22s} {s['tasks']:>5d} {fmt(s.get('policy_steps'), 1):>6s} "
            f"{fmt(s.get('policy_prompt_tokens')):>11s} {fmt(s.get('policy_output_tokens')):>11s} "
            f"{fmt(s.get('final_context_tokens')):>10s} {fmt(s.get('wm_steps'), 1):>9s} {fmt(s.get('wm_calls'), 1):>9s}"
        )
    print("\n(per task, mean; prompt tokens = sum of context re-read at every policy step)")
    if args.csv and all_rows:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"per-task rows -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
