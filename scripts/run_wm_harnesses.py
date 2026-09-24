#!/usr/bin/env python3
"""Run every WM harness once for one benchmark and one configured model.

The command after ``--`` is the common ``ejepa bench run`` command. It must contain
the benchmark, executor, target, and exactly one model configuration, but no
``--wm-strategy`` or ``--wm-beam-plan-trigger``. This wrapper runs five sequential
experiments: ITP-I, periodic beam planning, critic-triggered beam planning,
revision, and delayed reference.

Example::

    python scripts/run_wm_harnesses.py \
      --result-root "$BENCHMARK_HOME/experiments" \
      --label enterpriseops-jepa \
      -- ejepa bench run EnterpriseOps-Gym --executor mcp_react \
      --config target=opsgym_80_test \
      --wm-ewm-jepa-checkpoint /path/to/checkpoint \
      --wm-jepa-observation-backend canonical_event
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_bench_repeated as repeat


@dataclass(frozen=True)
class Harness:
    name: str
    arguments: tuple[str, ...]


HARNESS_NAMES = (
    "itp_i",
    "beam_interval",
    "beam_critic",
    "revision",
    "reference",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run five WM harnesses once each for one model and benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/wm_harness_summaries"))
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--harnesses",
        nargs="+",
        choices=HARNESS_NAMES,
        default=list(HARNESS_NAMES),
        help="Harness subset; each selected harness runs exactly once.",
    )
    parser.add_argument("--itp-max-k", type=int, default=5)
    parser.add_argument("--beam-samples", type=int, default=8)
    parser.add_argument("--beam-horizon", type=int, default=4)
    parser.add_argument("--beam-execute-steps", type=int, default=4)
    parser.add_argument("--beam-score-margin", type=float, default=0.10)
    parser.add_argument("--beam-temperature", type=float, default=0.7)
    parser.add_argument("--beam-refinement-rounds", type=int, default=1)
    parser.add_argument("--beam-refinement-top-k", type=int, default=4)
    # Calibrated against the captured critic telemetry of a JEPA beam_critic run: the previous
    # 0.25/0.35 sat at the middle of the predicted-probability distribution and fired on ~47% of
    # steps, i.e. ~2x the interval trigger's fixed 1/--beam-execute-steps cadence, which is why
    # beam_critic ran ~2x slower than beam_interval instead of cheaper. 0.50/0.60 fires on ~18%.
    parser.add_argument("--critic-failure-prob", type=float, default=0.50)
    parser.add_argument("--critic-stall-prob", type=float, default=0.60)
    parser.add_argument("--critic-max-quiet-steps", type=int, default=6)
    parser.add_argument("--ssot-diversity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sample-temperature-ladder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--beam-hard-override", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-auto-capture-trajectory", action="store_true")
    parser.add_argument(
        "--exclude-domains",
        nargs="*",
        default=[],
        metavar="DOMAIN",
        help=(
            "Additionally report a clearly-labelled score over tasks whose task_id "
            "domain prefix (the part before the first '.') is NOT in this list, e.g. "
            "--exclude-domains marketing finance. The unfiltered score_rate is always "
            "reported as well, so the headline number never changes silently."
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("pass one base 'ejepa bench run' command after --")
    for name in (
        "itp_max_k",
        "beam_samples",
        "beam_horizon",
        "beam_execute_steps",
        "beam_refinement_rounds",
        "beam_refinement_top_k",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.critic_max_quiet_steps < 0:
        parser.error("--critic-max-quiet-steps must be >= 0")
    return args


def has_option(command: Sequence[str], option: str) -> bool:
    return option in command or any(arg.startswith(f"{option}=") for arg in command)


def validate_base_command(command: Sequence[str]) -> None:
    if not command:
        raise ValueError("base command is empty")
    for option in ("--wm-strategy", "--wm-beam-plan-trigger"):
        if has_option(command, option):
            raise ValueError(
                f"remove {option} from the base command; the wrapper sets it per harness"
            )


def beam_common_arguments(args: argparse.Namespace) -> list[str]:
    result = [
        "--wm-beam-plan-samples",
        str(args.beam_samples),
        "--wm-beam-plan-horizon",
        str(args.beam_horizon),
        "--wm-beam-mpc-execute-steps",
        str(args.beam_execute_steps),
        "--wm-beam-plan-score-margin",
        str(args.beam_score_margin),
        "--wm-imagined-temperature",
        str(args.beam_temperature),
        "--wm-beam-plan-refinement-rounds",
        str(args.beam_refinement_rounds),
        "--wm-beam-plan-refinement-top-k",
        str(args.beam_refinement_top_k),
    ]
    if args.ssot_diversity:
        result.append("--wm-beam-plan-ssot-diversity")
    if args.sample_temperature_ladder:
        result.append("--sample-temperature-ladder")
    if args.beam_hard_override:
        result.append("--wm-beam-plan-hard-override")
    return result


def harnesses(args: argparse.Namespace) -> list[Harness]:
    common = beam_common_arguments(args)
    available = {
        "itp_i": Harness(
            "itp_i", ("--wm-strategy", "itp_i", "--wm-itp-max-k", str(args.itp_max_k))
        ),
        "beam_interval": Harness(
            "beam_interval",
            ("--wm-strategy", "beam_plan", "--wm-beam-plan-trigger", "interval", *common),
        ),
        "beam_critic": Harness(
            "beam_critic",
            (
                "--wm-strategy",
                "beam_plan",
                "--wm-beam-plan-trigger",
                "critic",
                *common,
                "--wm-beam-plan-critic-failure-prob",
                str(args.critic_failure_prob),
                "--wm-beam-plan-critic-stall-prob",
                str(args.critic_stall_prob),
                "--wm-beam-plan-critic-max-quiet-steps",
                str(args.critic_max_quiet_steps),
            ),
        ),
        "revision": Harness("revision", ("--wm-strategy", "revision")),
        "reference": Harness("reference", ("--wm-strategy", "reference")),
    }
    return [available[name] for name in args.harnesses]


def safe_label(value: str | None) -> str:
    if not value:
        return "wm-harnesses"
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return normalized or "wm-harnesses"


def effective_base_command(args: argparse.Namespace) -> list[str]:
    command = repeat.effective_command(
        args.command,
        auto_capture_trajectory=not args.no_auto_capture_trajectory,
    )
    if args.result_root is not None and not has_option(command, "--result-root"):
        command.extend(["--result-root", str(repeat.expand_path(args.result_root))])
    return command


PASS_SCORE_EPSILON = 1e-9
_SCORED_ASSERTIONS_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+scored assertions")


def task_domain(task_id: Any) -> str:
    """Domain prefix of a task id (``finance.ap_aging_report`` -> ``finance``)."""
    text = str(task_id or "")
    return text.split(".", 1)[0] if "." in text else ""


def unique_per_task(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Per-task records de-duplicated by task_id (last write wins).

    The internal-trajectory artifact is re-emitted cumulatively, so ``per_task``
    can hold several rows per task (1032 rows for a 600-task AutomationBench run).
    Averaging the raw list double-counts.
    """
    records = summary.get("per_task")
    if not isinstance(records, list):
        return []
    deduped: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if isinstance(record, Mapping) and record.get("task_id") is not None:
            deduped[str(record["task_id"])] = record
    return list(deduped.values())


def scorable_assertions(record: Mapping[str, Any]) -> int | None:
    """Denominator from ``reason`` ("0/5 scored assertions satisfied"), else None."""
    match = _SCORED_ASSERTIONS_RE.search(str(record.get("reason") or ""))
    return int(match.group(2)) if match else None


def filtered_scores(summary: Mapping[str, Any], excluded: Sequence[str]) -> dict[str, Any] | None:
    """Score and pass rate over tasks outside ``excluded`` domains.

    Returns None when no domains are excluded or the run has no per-task records,
    so the caller simply omits the block rather than reporting a misleading zero.
    """
    if not excluded:
        return None
    records = unique_per_task(summary)
    if not records:
        return None
    drop = {str(name).strip().lower() for name in excluded if str(name).strip()}
    scores: list[float] = []
    passes = unscorable = 0
    for record in records:
        if task_domain(record.get("task_id")).lower() in drop:
            continue
        if scorable_assertions(record) == 0:
            # No scorable assertion can ever pass; excluding keeps the rate meaningful.
            unscorable += 1
            continue
        score = record.get("score")
        if not isinstance(score, int | float) or isinstance(score, bool):
            continue
        scores.append(float(score))
        if float(score) >= 1.0 - PASS_SCORE_EPSILON:
            passes += 1
    if not scores:
        return None
    return {
        "excluded_domains": sorted(drop),
        "tasks_total": len(records),
        "tasks_kept": len(scores),
        "tasks_dropped_by_domain": sum(
            1 for r in records if task_domain(r.get("task_id")).lower() in drop
        ),
        "tasks_dropped_unscorable": unscorable,
        "score_rate": sum(scores) / len(scores),
        "pass_rate": passes / len(scores),
        "passed": passes,
    }


def comparison_row(record: Mapping[str, Any], excluded: Sequence[str] = ()) -> dict[str, Any]:
    summary = record.get("result_summary")
    summary = summary if isinstance(summary, Mapping) else {}
    agentic = summary.get("agentic_task_metrics")
    agentic = agentic if isinstance(agentic, Mapping) else {}
    row: dict[str, Any] = {
        "harness": record.get("harness"),
        "returncode": record.get("returncode"),
        "status": summary.get("status"),
        "result_dir": record.get("result_dir"),
        "score_rate": summary.get("score_rate"),
        "duration_seconds": summary.get("duration_seconds"),
        "wrapper_elapsed_seconds": record.get("wrapper_elapsed_seconds"),
        "avg_task_execution_time_seconds": agentic.get("avg_task_execution_time_seconds"),
        "avg_tool_calls": agentic.get("avg_tool_calls"),
        "avg_failed_tool_calls": agentic.get("avg_failed_tool_calls"),
        "avg_wm_beam_planning_count": agentic.get("avg_wm_beam_planning_count"),
        "avg_wm_critic_check_count": agentic.get("avg_wm_critic_check_count"),
        "avg_wm_critic_fire_count": agentic.get("avg_wm_critic_fire_count"),
        "wm_critic_fire_rate": agentic.get("wm_critic_fire_rate"),
        "avg_wm_action_change_count": agentic.get("avg_wm_action_change_count"),
        "avg_wm_world_model_call_count": agentic.get("avg_wm_world_model_call_count"),
        "avg_wm_judge_call_count": agentic.get("avg_wm_judge_call_count"),
        "avg_wm_model_call_count": agentic.get("avg_wm_model_call_count"),
        "avg_wm_beam_refinement_round_count": agentic.get("avg_wm_beam_refinement_round_count"),
        "avg_wm_beam_refinement_score_pass_count": agentic.get(
            "avg_wm_beam_refinement_score_pass_count"
        ),
    }
    filtered = filtered_scores(summary, excluded)
    if filtered is not None:
        row["filtered"] = filtered
    return row


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    base_command: Sequence[str],
    records: Sequence[Mapping[str, Any]],
) -> None:
    payload = {
        "schema_version": "1.0",
        "created_at_utc": repeat.utc_now_iso(),
        "mode": "dry_run" if args.dry_run else "run",
        "single_run_per_harness": True,
        "base_command": list(base_command),
        "selected_harnesses": list(args.harnesses),
        "runs": list(records),
        "excluded_domains": list(args.exclude_domains or []),
        "comparison": [comparison_row(record, args.exclude_domains) for record in records],
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_base_command(args.command)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_dir = repeat.expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    basename = f"ejepa-wm-harnesses-{safe_label(args.label)}-{stamp}"
    summary_path = output_dir / f"{basename}.json"
    base_command = effective_base_command(args)
    records: list[dict[str, Any]] = []

    for index, harness in enumerate(harnesses(args), 1):
        command = [*base_command, *harness.arguments]
        log_path = output_dir / f"{basename}-{index:02d}-{harness.name}.log"
        print(f"=== WM harness {index}/{len(args.harnesses)}: {harness.name} ===")
        print("Command:", shlex.join(command))
        if args.dry_run:
            records.append(
                {
                    "run_index": index,
                    "harness": harness.name,
                    "command": command,
                    "returncode": None,
                    "result_dir": None,
                    "log_path": None,
                }
            )
            continue

        discovery = repeat.prepare_discovery(command, args.result_root)
        started_at = repeat.utc_now_iso()
        started_mono = time.monotonic()
        returncode = repeat.run_command(command, log_path)
        result_dir = repeat.discover_new_result_dir(discovery, started_at_monotonic=started_mono)
        record: dict[str, Any] = {
            "run_index": index,
            "harness": harness.name,
            "command": command,
            "started_at_utc": started_at,
            "completed_at_utc": repeat.utc_now_iso(),
            "wrapper_elapsed_seconds": time.monotonic() - started_mono,
            "returncode": returncode,
            "result_dir": str(result_dir) if result_dir else None,
            "log_path": str(log_path),
        }
        if result_dir is not None:
            try:
                record["result_summary"] = repeat.summarize_result_dir(result_dir)
            except Exception as exc:
                record["summary_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        write_summary(summary_path, args=args, base_command=base_command, records=records)
        print("Harness summary:", comparison_row(record, args.exclude_domains))
        print(f"Partial comparison written: {summary_path}")
        if returncode != 0 and args.stop_on_failure:
            break

    write_summary(summary_path, args=args, base_command=base_command, records=records)
    print(f"Wrote comparison: {summary_path}")
    if args.dry_run:
        return 0
    return 0 if records and all(record.get("returncode") == 0 for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
