import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.generate_enterpriseops_gym_stateful_samples import (  # noqa: E402
    dump_json,
    load_jsonl,
    reconstruct_seed_trajectory,
)


DEFAULT_SEEDS_PATH = ROOT / "trajectories" / "imported_benchmark_seeds.jsonl"
DEFAULT_SOURCE_ROOT = Path("/data/user/enterprisegym/results/react/gpt-5/teams/oracle/run_1")
DEFAULT_OUTPUT_PATH = ROOT / "trajectories" / "enterpriseops_gym_stateful_trajectories.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate stateful trajectories for EnterpriseOps-Gym oracle runs by "
            "matching raw run JSON files to imported benchmark seeds."
        )
    )
    parser.add_argument("--seeds-path", type=Path, default=DEFAULT_SEEDS_PATH)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl"],
        default="json",
        help="Serialize trajectories as a JSON list or JSONL.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="",
        help="Optional comma-separated domain filter, for example `csm,itsm,teams`.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on the number of matched run files to convert.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any JSON file in the source root does not have a matching imported seed.",
    )
    return parser.parse_args()


def normalize_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def parse_domain_filter(raw_value: str) -> set[str]:
    if not raw_value.strip():
        return set()
    return {item.strip().lower() for item in raw_value.split(",") if item.strip()}


def load_enterpriseops_gym_seed_map(seeds_path: Path, domains: set[str]) -> dict[str, dict]:
    seed_map = {}
    for record in load_jsonl(seeds_path):
        if record.get("source_dataset") != "EnterpriseOps-Gym":
            continue
        domain = (record.get("environment") or {}).get("domain", "").lower()
        if domains and domain not in domains:
            continue
        source_path = (record.get("source_record") or {}).get("source_path")
        if not source_path:
            continue
        normalized = normalize_path(source_path)
        if normalized in seed_map:
            raise ValueError(f"Duplicate EnterpriseOps-Gym seed for source path: {normalized}")
        seed_map[normalized] = record
    return seed_map


def iter_source_files(source_root: Path) -> list[Path]:
    return sorted(path for path in source_root.glob("*.json") if path.is_file())


def dump_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_trajectory_id(seed: dict) -> str:
    source_path = seed["source_record"]["source_path"]
    return f"enterpriseops-gym-stateful-{Path(source_path).stem}"


def try_reconstruct_source_file(source_file: Path, seed: dict) -> tuple[dict | None, dict | None]:
    try:
        trajectory = reconstruct_seed_trajectory(
            seed,
            trajectory_id=build_trajectory_id(seed),
            selection_basis="matched_imported_seed_by_source_path",
        )
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return None, {
            "source_path": str(source_file),
            "seed_id": seed.get("seed_id"),
            "reason": str(exc),
        }
    return trajectory, None


def write_trajectories(path: Path, records: list[dict], output_format: str) -> None:
    if output_format == "jsonl":
        dump_jsonl(path, records)
    else:
        dump_json(path, records)


def report_summary(
    *,
    output_path: Path,
    trajectories: list[dict],
    matched_paths: list[str],
    missing_paths: list[str],
    skipped_records: list[dict],
) -> None:
    domain_counts = Counter(item["domain"] for item in trajectories)

    print(f"Wrote {len(trajectories)} trajectories to {output_path}")
    print(f"Matched run files: {len(matched_paths)}")
    print(f"Unmatched run files: {len(missing_paths)}")
    print(f"Skipped non-reconstructable runs: {len(skipped_records)}")
    if domain_counts:
        print("Domain counts:")
        for domain, count in sorted(domain_counts.items()):
            print(f"  {domain}: {count}")
    if skipped_records:
        print("Sample skipped runs:")
        for item in skipped_records[:5]:
            print(f"  {item['source_path']}: {item['reason']}")


def main():
    args = parse_args()
    domain_filter = parse_domain_filter(args.domains)
    seed_map = load_enterpriseops_gym_seed_map(args.seeds_path, domain_filter)
    source_files = iter_source_files(args.source_root)

    trajectories = []
    matched_paths = []
    missing_paths = []
    skipped_records = []

    for source_file in source_files:
        normalized = normalize_path(source_file)
        seed = seed_map.get(normalized)
        if seed is None:
            missing_paths.append(str(source_file))
            continue

        trajectory, skipped = try_reconstruct_source_file(source_file, seed)
        if skipped is not None:
            skipped_records.append(skipped)
            continue

        trajectories.append(trajectory)
        matched_paths.append(str(source_file))

        if args.max_records is not None and len(trajectories) >= args.max_records:
            break

    if args.strict and (missing_paths or skipped_records):
        preview_lines = missing_paths[:10]
        preview_lines.extend(item["source_path"] for item in skipped_records[:10])
        preview = "\n".join(preview_lines)
        raise ValueError(
            "Found EnterpriseOps-Gym run files without matching imported seeds or reconstructable runs:\n"
            f"{preview}"
        )

    if not trajectories:
        raise ValueError(
            f"No EnterpriseOps-Gym trajectories were generated from {args.source_root} "
            f"using seeds from {args.seeds_path}."
        )

    write_trajectories(args.output_path, trajectories, args.output_format)
    report_summary(
        output_path=args.output_path,
        trajectories=trajectories,
        matched_paths=matched_paths,
        missing_paths=missing_paths,
        skipped_records=skipped_records,
    )


if __name__ == "__main__":
    main()
