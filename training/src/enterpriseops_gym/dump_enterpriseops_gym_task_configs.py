"""Materialize EnterpriseOps-Gym task configs from the HuggingFace dataset.

`evaluate.py` in the EnterpriseOps-Gym repo writes one task config per HF row to
`tempfile.mkdtemp(prefix="rl_gym_hf_")` and discards it after the run, leaving
only `results_*` files. That makes the task configs unrecoverable for downstream
evaluation paths (e.g. `finetuning.py --gym-task-configs`).

This dumper performs the same HF row → task config JSON conversion, but to a
caller-specified directory so the configs persist. Output filenames match the
gym convention: `<mode>__<domain>__<task_id>.json`.
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dump EnterpriseOps-Gym task configs from a HuggingFace dataset to a "
            "persistent directory. Mirrors the HF→JSON conversion `evaluate.py` does "
            "internally so the resulting files are interchangeable with the ones it "
            "feeds to BenchmarkExecutor."
        )
    )
    parser.add_argument(
        "--hf-dataset",
        default="ServiceNow-AI/EnterpriseOps-Gym",
        help="HuggingFace dataset repo ID.",
    )
    parser.add_argument(
        "--domain",
        nargs="+",
        required=True,
        help=(
            "One or more domains (HF dataset splits), e.g. `itsm csm teams email drive`."
        ),
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        default=["oracle"],
        help="One or more tool-set modes (HF dataset configs), e.g. `oracle +5_tools`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination folder. Created if it does not exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files. Without this flag, existing task configs are skipped.",
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        help="Optional cap on the number of rows materialized per (mode, domain) pair.",
    )
    return parser.parse_args()


def dump_task_configs(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required. Install with `uv add datasets` "
            "or run inside the EnterpriseOps-Gym environment."
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_string_fields = {"gym_servers_config", "verifiers"}
    hf_only_fields = {"task_id", "domain"}

    total_written = 0
    total_skipped = 0
    per_split_counts: dict[tuple[str, str], int] = {}

    for mode in args.mode:
        for domain in args.domain:
            print(
                f"Loading split: dataset={args.hf_dataset} config={mode} split={domain}",
                file=sys.stderr,
            )
            try:
                rows = hf_load_dataset(args.hf_dataset, mode, split=domain)
            except Exception as exc:
                print(
                    f"  failed to load (mode={mode}, domain={domain}): {exc}",
                    file=sys.stderr,
                )
                continue

            written_for_split = 0
            for row in rows:
                if (
                    args.max_per_split is not None
                    and written_for_split >= args.max_per_split
                ):
                    break
                task_id = row.get("task_id") or f"task_{id(row)}"
                file_name = f"{mode}__{domain}__{task_id}.json"
                file_path = args.output_dir / file_name

                if file_path.exists() and not args.overwrite:
                    total_skipped += 1
                    continue

                task_dict: dict = {}
                for k, v in row.items():
                    if k in hf_only_fields:
                        continue
                    if k in json_string_fields and isinstance(v, str):
                        try:
                            v = json.loads(v)
                        except json.JSONDecodeError:
                            print(
                                f"  warning: {file_name} field {k!r} did not parse as JSON; "
                                "stored as raw string.",
                                file=sys.stderr,
                            )
                    task_dict[k] = v

                with file_path.open("w", encoding="utf-8") as handle:
                    json.dump(task_dict, handle, ensure_ascii=False, indent=2)
                total_written += 1
                written_for_split += 1

            per_split_counts[(mode, domain)] = written_for_split
            print(
                f"  wrote {written_for_split} task configs for ({mode}, {domain})",
                file=sys.stderr,
            )

    print(
        f"\nTotal: wrote {total_written}, skipped {total_skipped} (existing).\n"
        f"Output dir: {args.output_dir}\n"
        f"Per-split counts: {per_split_counts}"
    )


def main() -> None:
    args = parse_args()
    dump_task_configs(args)


if __name__ == "__main__":
    main()
