#!/usr/bin/env python3
"""Week 1.1 -- annotation audit: how trustworthy are the GPT canonical-event labels?

plan.md asks for (a) a stratified adjudication sample with full context, (b) per-field
agreement measured with macro-F1 and Cohen's kappa rather than accuracy alone, and (c)
deterministic labeling wherever the field is mechanically derivable.

NOTE ON SCOPE: this operates on the EXISTING 11-field schema (7 canonical_event_state + 4
nudge). plan.md's proposed transition/task/policy restructuring is deliberately NOT applied.

Three stages:

  --stage rules   Run deterministic labelers over the whole split and report GPT-vs-rule
                  agreement per field. Needs no human effort, so it runs today and already
                  localizes suspect fields: a field where a simple rule reproduces the GPT
                  label is cheap to fix; one where they diverge is either genuinely semantic
                  or genuinely noisy.
  --stage sample  Emit N stratified adjudication packets (JSONL, one row per transition)
                  carrying everything plan.md lists: system+task prompt, full pre-action
                  history, the action, the next observation, the remaining trajectory and the
                  trajectory-level outcome (for progress/sufficiency judgements). Writes a
                  companion *_adjudication_template.jsonl with blank label slots to fill in.
  --stage score   Given the filled-in template, compute per-field accuracy, macro-F1,
                  balanced accuracy and Cohen's kappa for GPT-vs-human and rules-vs-human.

Stratification (plan.md: "by dataset, field and rare failure class") is implemented as: for
every (benchmark, field, value) cell, take at least one example, then fill the remaining
budget proportional to sqrt(cell size) so rare classes are over-sampled relative to their
frequency without crowding out the head of the distribution.

Usage:
  uv run python src/analysis/week1_annotation_audit.py --stage rules
  uv run python src/analysis/week1_annotation_audit.py --stage sample --num-samples 250
  uv run python src/analysis/week1_annotation_audit.py --stage score \
      --adjudicated data/week1/audit_adjudication_template.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.finetuning_jepa as fj

TRAJ = REPO_ROOT / "trajectories"
STEM = "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro"
DEFAULT_EVAL = TRAJ / f"{STEM}_eval_examples_cleaned_value_scored_recognition_probe.jsonl"
DEFAULT_OUTDIR = REPO_ROOT / "data" / "week1"

SINGLE_FIELDS = list(fj.CANONICAL_EVENT_SINGLE_LABEL_FIELDS)
MULTI_FIELDS = list(fj.NUDGE_MULTI_LABEL_FIELDS)
STATE_FIELDS = list(fj.CANONICAL_EVENT_STATE_FIELDS)

# Fields plan.md flags as mechanically derivable; the rest ("genuinely semantic") are
# LLM-only and get no rule.
RULE_FIELDS = ("action_type", "object_type", "error_signature", "execution_status", "side_effect_type")

READ_PREFIXES = ("get", "list", "find", "retrieve", "read", "cat", "head", "tail", "describe", "show", "fetch")
SEARCH_PREFIXES = ("search", "query", "locate", "grep", "rg", "lookup", "filter")
CREATE_PREFIXES = ("create", "add", "insert", "register", "upload", "copy", "fork", "mkdir", "touch", "new", "send_invite")
UPDATE_PREFIXES = ("update", "patch", "modify", "edit", "write", "move", "rename", "chmod", "chown", "set", "put", "assign")
DELETE_PREFIXES = ("delete", "remove", "archive", "rm", "drop", "truncate", "clear", "revoke")
RUN_PREFIXES = ("run", "execute", "exec", "bash", "sh", "python", "invoke", "start", "trigger", "build", "install")
TEST_PREFIXES = ("test", "verify", "check", "validate", "assert")
COMMUNICATE_PREFIXES = ("send", "reply", "publish", "post", "message", "email", "notify", "comment")

OBJECT_KEYWORDS = {
    "calendar": ("calendar", "event", "freebusy", "acl"),
    "message": ("message", "mail", "email", "chat", "thread", "draft"),
    "file": ("file", "drive", "document", "folder", "blob", "attachment"),
    "ticket": ("ticket", "incident", "issue", "request"),
    "case": ("case",),
    "account": ("account", "user", "member", "identity", "profile"),
    "customer": ("customer", "contact", "lead", "client"),
    "quote": ("quote", "opportunity", "order", "invoice"),
    "permission": ("permission", "role", "acl", "grant", "scope", "policy"),
    "repository": ("repo", "repository", "git", "branch", "commit", "pull"),
    "process": ("process", "job", "task", "workflow", "pipeline", "run"),
    "package": ("package", "pip", "npm", "apt", "dependency", "install"),
    "database_row": ("sql", "query", "table", "record", "row", "select"),
    "label": ("label", "tag"),
    "comment": ("comment", "note"),
    "branch": ("branch",),
}

ERROR_PATTERNS = (
    ("not_found", r"\b(not[\s_-]?found|404|no such (file|user|record|entity)|does not exist|missing (record|entity))\b"),
    ("permission_denied", r"\b(permission denied|forbidden|403|unauthorized|401|access denied|not permitted)\b"),
    ("invalid_argument", r"\b(invalid[\s_-]?(argument|parameter|value|request)|400|bad request|validation (failed|error)|malformed)\b"),
    ("timeout", r"\b(timeout|timed out|deadline exceeded|504)\b"),
    ("parse_error", r"\b(parse error|json(decode)?error|syntax ?error|could not parse|unmarshal)\b"),
    ("dependency_missing", r"\b(module ?not ?found|no module named|command not found|dependency (missing|not)|importerror)\b"),
    ("test_failed", r"\b(test[s]? failed|assertion ?error|failed: \d+|\d+ failed)\b"),
    ("runtime_error", r"\b(traceback|exception|runtime ?error|internal server error|500|segmentation fault)\b"),
)
FAILURE_HINT = re.compile(r"\b(error|failed|failure|denied|invalid|exception|traceback|cannot|could not|unable to)\b", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("rules", "sample", "score"), required=True)
    p.add_argument("--input", type=Path, default=DEFAULT_EVAL,
                   help="Labeled JSONL (default: the recognition-probe eval file, which carries observations).")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--num-samples", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adjudicated", type=Path, default=None, help="Filled-in template, for --stage score.")
    return p.parse_args()


# --------------------------------------------------------------------------- helpers
def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def gpt_label(row: dict[str, Any], field: str) -> Any:
    label = row.get("canonical_event_with_nudge") or {}
    block = label.get("canonical_event_state") if field in STATE_FIELDS else label.get("nudge")
    return (block or {}).get(field)


def tool_names(row: dict[str, Any]) -> list[str]:
    action = row.get("action") or {}
    calls = action.get("tool_calls") if isinstance(action, dict) else None
    names = []
    for call in calls or []:
        if isinstance(call, dict):
            fn = call.get("function") if isinstance(call.get("function"), dict) else call
            if fn.get("name"):
                names.append(str(fn["name"]))
    return names


def tool_arguments(row: dict[str, Any]) -> dict[str, Any]:
    action = row.get("action") or {}
    for call in (action.get("tool_calls") or []) if isinstance(action, dict) else []:
        if isinstance(call, dict):
            fn = call.get("function") if isinstance(call.get("function"), dict) else call
            if isinstance(fn.get("arguments"), dict):
                return fn["arguments"]
    return {}


# --------------------------------------------------------------------------- rules
def rule_action_type(row: dict[str, Any]) -> str:
    names = tool_names(row)
    if not names:
        return "unknown"
    base = names[0].split("-")[-1].lower()
    head = re.split(r"[._]", base)[0]
    for prefixes, value in (
        (DELETE_PREFIXES, "delete"), (CREATE_PREFIXES, "create"), (UPDATE_PREFIXES, "update"),
        (TEST_PREFIXES, "test"), (COMMUNICATE_PREFIXES, "communicate"), (SEARCH_PREFIXES, "search"),
        (RUN_PREFIXES, "run"), (READ_PREFIXES, "read"),
    ):
        if head.startswith(prefixes) or base.startswith(prefixes):
            return value
    return "unknown"


def rule_object_type(row: dict[str, Any]) -> str:
    blob = " ".join(tool_names(row)).lower() + " " + " ".join(str(k) for k in tool_arguments(row)).lower()
    best, best_hits = "unknown", 0
    for value, keywords in OBJECT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in blob)
        if hits > best_hits:
            best, best_hits = value, hits
    return best


def rule_error_signature(row: dict[str, Any]) -> str:
    text = str(row.get("observation") or "")
    if not text.strip():
        return "unknown"
    low = text.lower()
    for value, pattern in ERROR_PATTERNS:
        if re.search(pattern, low):
            return value
    return "none" if not FAILURE_HINT.search(low) else "unknown"


def rule_execution_status(row: dict[str, Any]) -> str:
    text = str(row.get("observation") or "")
    if not text.strip():
        return "unknown"
    if rule_error_signature(row) not in ("none", "unknown"):
        return "failure"
    return "success" if not FAILURE_HINT.search(text) else "failure"


def rule_side_effect_type(row: dict[str, Any]) -> str:
    if rule_execution_status(row) == "failure":
        return "none"
    return {
        "read": "retrieved", "search": "retrieved", "create": "created", "update": "modified",
        "delete": "deleted", "communicate": "sent", "run": "executed", "test": "validated",
    }.get(rule_action_type(row), "unknown")


RULES = {
    "action_type": rule_action_type,
    "object_type": rule_object_type,
    "error_signature": rule_error_signature,
    "execution_status": rule_execution_status,
    "side_effect_type": rule_side_effect_type,
}


# --------------------------------------------------------------------------- metrics
def cohen_kappa(a: list[Any], b: list[Any]) -> float:
    """Unweighted Cohen's kappa over the union of observed categories."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    return 1.0 if expected >= 1.0 else (observed - expected) / (1.0 - expected)


def macro_f1(gold: list[Any], pred: list[Any]) -> float:
    labels = sorted(set(gold) | set(pred), key=str)
    scores = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else float("nan")


def balanced_accuracy(gold: list[Any], pred: list[Any]) -> float:
    recalls = []
    for label in sorted(set(gold), key=str):
        support = [i for i, g in enumerate(gold) if g == label]
        if support:
            recalls.append(sum(1 for i in support if pred[i] == label) / len(support))
    return sum(recalls) / len(recalls) if recalls else float("nan")


def agreement_block(gold: list[Any], pred: list[Any]) -> dict[str, Any]:
    return {
        "n": len(gold),
        "accuracy": sum(1 for g, p in zip(gold, pred) if g == p) / max(len(gold), 1),
        "macro_f1": macro_f1(gold, pred),
        "balanced_accuracy": balanced_accuracy(gold, pred),
        "cohen_kappa": cohen_kappa(gold, pred),
    }


# --------------------------------------------------------------------------- stages
def stage_rules(args: argparse.Namespace) -> None:
    rows = load_rows(args.input)
    print(f"rows: {len(rows)}  (deterministic rules vs GPT labels; no human effort required)\n")
    report: dict[str, Any] = {"input": str(args.input), "rows": len(rows), "fields": {}}
    print(f"{'field':24s} {'acc':>7s} {'macroF1':>8s} {'balAcc':>7s} {'kappa':>7s}")
    print("-" * 60)
    for field in RULE_FIELDS:
        gold = [gpt_label(r, field) for r in rows]
        pred = [RULES[field](r) for r in rows]
        keep = [i for i, g in enumerate(gold) if g is not None]
        gold = [gold[i] for i in keep]
        pred = [pred[i] for i in keep]
        block = agreement_block(gold, pred)
        report["fields"][field] = block
        # most common disagreements, to make the failure mode inspectable
        confusion = Counter((g, p) for g, p in zip(gold, pred) if g != p)
        report["fields"][field]["top_disagreements"] = [
            {"gpt": g, "rule": p, "count": c} for (g, p), c in confusion.most_common(5)
        ]
        print(f"{field:24s} {block['accuracy']:>7.3f} {block['macro_f1']:>8.3f} "
              f"{block['balanced_accuracy']:>7.3f} {block['cohen_kappa']:>7.3f}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "audit_rule_agreement.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print(f"\nreport -> {out}")
    print("Reading: high kappa => the field is mechanically derivable (replace the LLM label with the rule).\n"
          "         low kappa  => either genuinely semantic, or the GPT label is noisy; send it to adjudication.")


def stratified_sample(rows: list[dict[str, Any]], budget: int, seed: int) -> list[int]:
    """>=1 example per (benchmark, field, value) cell, then sqrt-proportional fill."""
    rng = random.Random(seed)
    cells: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        benchmark = str(row.get("benchmark") or "unknown")
        for field in SINGLE_FIELDS:
            value = gpt_label(row, field)
            if value is not None:
                cells[(benchmark, field, str(value))].append(i)
    chosen: set[int] = set()
    for key in sorted(cells):
        chosen.add(rng.choice(cells[key]))          # rare-class floor
        if len(chosen) >= budget:
            break
    weights = {k: math.sqrt(len(v)) for k, v in cells.items()}
    total = sum(weights.values()) or 1.0
    for key in sorted(cells, key=lambda k: -weights[k]):
        if len(chosen) >= budget:
            break
        extra = int(round(budget * weights[key] / total))
        for idx in rng.sample(cells[key], min(extra, len(cells[key]))):
            chosen.add(idx)
            if len(chosen) >= budget:
                break
    return sorted(chosen)


def stage_sample(args: argparse.Namespace) -> None:
    rows = load_rows(args.input)
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_rows(args.input.parent / args.input.name.replace("_recognition_probe", "")):
        by_trajectory[str(row["trajectory_id"])].append(row)
    for group in by_trajectory.values():
        group.sort(key=lambda r: int(r.get("interaction_index") or 0))

    picks = stratified_sample(rows, args.num_samples, args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    packets_path = args.outdir / "audit_adjudication_packets.jsonl"
    template_path = args.outdir / "audit_adjudication_template.jsonl"
    benchmarks = Counter()
    with packets_path.open("w", encoding="utf-8") as packets, template_path.open("w", encoding="utf-8") as template:
        for rank, index in enumerate(picks):
            row = rows[index]
            trajectory = by_trajectory.get(str(row["trajectory_id"]), [])
            step = int(row.get("interaction_index") or 0)
            future = [
                {"interaction_index": r.get("interaction_index"), "action": r.get("action"),
                 "gpt_execution_status": gpt_label(r, "execution_status")}
                for r in trajectory if int(r.get("interaction_index") or 0) > step
            ]
            benchmarks[row.get("benchmark")] += 1
            packets.write(json.dumps({
                "audit_id": f"audit-{rank:04d}",
                "trajectory_id": row["trajectory_id"],
                "interaction_index": step,
                "benchmark": row.get("benchmark"),
                # --- everything plan.md asks the annotator to see ---
                "system_prompt": row.get("system_prompt"),
                "task_prompt": row.get("task_prompt"),
                "pre_action_history": row.get("input_history"),
                "action": row.get("action"),
                "next_observation": row.get("observation"),
                "subsequent_trajectory": future,
                "trajectory_outcome": {
                    "trajectory_success": row.get("trajectory_success"),
                    "trajectory_pass_rate": row.get("trajectory_pass_rate"),
                    "success_source": row.get("success_source"),
                },
                "gpt_labels": row.get("canonical_event_with_nudge"),
                "rule_labels": {f: RULES[f](row) for f in RULE_FIELDS},
            }, ensure_ascii=False) + "\n")
            template.write(json.dumps({
                "audit_id": f"audit-{rank:04d}",
                "trajectory_id": row["trajectory_id"],
                "interaction_index": step,
                "adjudicated": {**{f: None for f in SINGLE_FIELDS}, **{f: None for f in MULTI_FIELDS}},
                "notes": "",
            }, ensure_ascii=False) + "\n")
    print(f"sampled {len(picks)} transitions from {len(rows)} rows")
    print(f"  per benchmark: {dict(benchmarks)}")
    print(f"  packets  -> {packets_path}")
    print(f"  template -> {template_path}  (fill `adjudicated`, then --stage score)")


def stage_score(args: argparse.Namespace) -> None:
    if args.adjudicated is None:
        raise SystemExit("--stage score requires --adjudicated <filled template>")
    rows = {(str(r["trajectory_id"]), int(r.get("interaction_index") or 0)): r for r in load_rows(args.input)}
    adjudicated = load_rows(args.adjudicated)
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in adjudicated:
        key = (str(entry["trajectory_id"]), int(entry.get("interaction_index") or 0))
        labels = entry.get("adjudicated") or {}
        if key in rows and any(v is not None for v in labels.values()):
            paired.append((rows[key], labels))
    if not paired:
        raise SystemExit("no adjudicated rows with filled labels found")
    print(f"adjudicated transitions: {len(paired)}\n")
    report: dict[str, Any] = {"n": len(paired), "gpt_vs_human": {}, "rules_vs_human": {}}
    print(f"{'field':24s} {'GPT acc':>8s} {'GPT F1':>7s} {'GPT k':>7s} | {'rule acc':>8s} {'rule k':>7s}")
    print("-" * 74)
    for field in SINGLE_FIELDS:
        pairs = [(row, labels[field]) for row, labels in paired if labels.get(field) is not None]
        if not pairs:
            continue
        human = [v for _, v in pairs]
        gpt = [gpt_label(r, field) for r, _ in pairs]
        report["gpt_vs_human"][field] = agreement_block(human, gpt)
        line = (f"{field:24s} {report['gpt_vs_human'][field]['accuracy']:>8.3f} "
                f"{report['gpt_vs_human'][field]['macro_f1']:>7.3f} "
                f"{report['gpt_vs_human'][field]['cohen_kappa']:>7.3f}")
        if field in RULES:
            rule = [RULES[field](r) for r, _ in pairs]
            report["rules_vs_human"][field] = agreement_block(human, rule)
            line += (f" | {report['rules_vs_human'][field]['accuracy']:>8.3f} "
                     f"{report['rules_vs_human'][field]['cohen_kappa']:>7.3f}")
        print(line)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "audit_human_agreement.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print(f"\nreport -> {out}")


def main() -> None:
    args = parse_args()
    {"rules": stage_rules, "sample": stage_sample, "score": stage_score}[args.stage](args)


if __name__ == "__main__":
    main()
