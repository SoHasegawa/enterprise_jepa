"""Materialize EnterpriseOps-Gym task configs from the HuggingFace dataset.

`evaluate.py` in the EnterpriseOps-Gym repo writes one task config per HF row to
`tempfile.mkdtemp(prefix="rl_gym_hf_")` and discards it after the run, leaving
only `results_*` files. That makes the task configs unrecoverable for downstream
evaluation paths (e.g. `evaluation.py --gym-task-configs`).

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


def load_split_rows(args: argparse.Namespace, mode: str, domain: str):
    from datasets import load_dataset as hf_load_dataset

    print(
        f"Loading split: dataset={args.hf_dataset} config={mode} split={domain}",
        file=sys.stderr,
    )
    return hf_load_dataset(args.hf_dataset, mode, split=domain)


def decode_json_string_field(file_name: str, field_name: str, value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        print(
            f"  warning: {file_name} field {field_name!r} did not parse as JSON; "
            "stored as raw string.",
            file=sys.stderr,
        )
        return value


def build_task_config(row: dict, file_name: str) -> dict:
    json_string_fields = {"gym_servers_config", "verifiers"}
    hf_only_fields = {"task_id", "domain"}
    task_dict: dict = {}
    for key, value in row.items():
        if key in hf_only_fields:
            continue
        if key in json_string_fields:
            value = decode_json_string_field(file_name, key, value)
        task_dict[key] = value
    return task_dict


def dump_split_rows(args: argparse.Namespace, rows, mode: str, domain: str) -> tuple[int, int]:
    written_for_split = 0
    skipped_for_split = 0
    for row in rows:
        if args.max_per_split is not None and written_for_split >= args.max_per_split:
            break
        task_id = row.get("task_id") or f"task_{id(row)}"
        file_name = f"{mode}__{domain}__{task_id}.json"
        file_path = args.output_dir / file_name

        if file_path.exists() and not args.overwrite:
            skipped_for_split += 1
            continue

        task_dict = build_task_config(row, file_name)
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(task_dict, handle, ensure_ascii=False, indent=2)
        written_for_split += 1
    return written_for_split, skipped_for_split


def dump_task_configs(args: argparse.Namespace) -> None:
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required. Install with `uv add datasets` "
            "or run inside the EnterpriseOps-Gym environment."
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_skipped = 0
    per_split_counts: dict[tuple[str, str], int] = {}

    for mode in args.mode:
        for domain in args.domain:
            try:
                rows = load_split_rows(args, mode, domain)
            except Exception as exc:
                print(
                    f"  failed to load (mode={mode}, domain={domain}): {exc}",
                    file=sys.stderr,
                )
                continue

            written_for_split, skipped_for_split = dump_split_rows(args, rows, mode, domain)
            total_written += written_for_split
            total_skipped += skipped_for_split
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
