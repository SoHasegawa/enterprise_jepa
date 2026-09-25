#!/usr/bin/env python3
"""Convert a world-model trajectory `.json` (one big JSON list) to `.jsonl` (one per line).

Why: `json.load()` must materialize the entire list before extraction can start, so a 19 GB
corpus costs its full parsed size in RSS on top of the examples being built from it. A `.jsonl`
can be read a record at a time, which lets finetuning_jepa.py extract and free in chunks
(--trajectory-chunk-size).

Convert only the corpus that is actually too big. finetuning_jepa.py prefers a sibling `.jsonl`
automatically (resolve_streamable_path), so nothing else changes -- presets, manifests and the
other files keep pointing at `.json` and keep their existing behaviour.

Streams both sides: reads with ijson when available (constant memory), else falls back to
json.load for the read. Writing is always streamed.

Usage:
  uv run python src/data_preparation/convert_trajectories_to_jsonl.py \
      trajectories/toucan_1_5m_multiturn_world_model_trajectories.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="the .json trajectory file to convert")
    p.add_argument("--out", type=Path, default=None,
                   help="output path (default: the source with a .jsonl suffix, which is what "
                        "finetuning_jepa.py auto-detects)")
    p.add_argument("--progress-every", type=int, default=50000)
    return p.parse_args()


def iter_records_json_load(path: Path):
    """Whole-file read. Correct for any input, but needs the parsed file in RAM once."""
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise SystemExit(f"Expected a list of trajectories in {path}")
    yield from records


def iter_records_ijson(path: Path, backend_name: str):
    """Constant-memory read through a named ijson backend.

    The fast C backend (yajl2_c) rejects integers outside 64-bit range with
    "integer overflow" -- these corpora contain tool arguments like 14573837282918237419 --
    while ijson's pure-Python backend and stdlib json both handle arbitrary precision. Hence
    the backend ladder in main().
    """
    import importlib  # noqa: PLC0415

    backend = importlib.import_module(f"ijson.backends.{backend_name}")
    with path.open("rb") as handle:
        # use_float=True: ijson yields Decimal by default, which json.dumps cannot serialize.
        yield from backend.items(handle, "item", use_float=True)


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise SystemExit(f"not found: {args.source}")
    out = args.out or args.source.with_suffix(".jsonl")
    if out == args.source:
        raise SystemExit("refusing to overwrite the source file")
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing {out}; delete it first")

    tmp = out.with_suffix(out.suffix + ".partial")
    # Fast streaming first; fall back to the slower streaming backend on the 64-bit integer
    # limit; fall back to whole-file json.load only as a last resort.
    readers = [("ijson/yajl2_c", lambda: iter_records_ijson(args.source, "yajl2_c")),
               ("ijson/python", lambda: iter_records_ijson(args.source, "python")),
               ("json.load (whole file in RAM)", lambda: iter_records_json_load(args.source))]
    seen: Counter[str] = Counter()
    written = 0
    last_error: Exception | None = None
    for label, make in readers:
        seen.clear()
        written = 0
        try:
            print(f"[convert] reading with {label}", flush=True)
            with tmp.open("w", encoding="utf-8") as handle:
                for record in make():
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    key = str(record.get("trajectory_id", "")) if isinstance(record, dict) else ""
                    if key:
                        seen[key] += 1
                    if written % args.progress_every == 0:
                        print(f"  {written} records", flush=True)
            break
        except (ImportError, ValueError, TypeError) as exc:
            last_error = exc
            print(f"[convert] {label} failed after {written} records ({type(exc).__name__}: "
                  f"{str(exc)[:120]}); retrying with the next reader.", flush=True)
        except Exception as exc:                                    # noqa: BLE001
            # ijson raises its own IncompleteJSONError, which is not importable without ijson.
            if type(exc).__name__ not in {"IncompleteJSONError", "JSONError", "CommonJSONError"}:
                raise
            last_error = exc
            print(f"[convert] {label} failed after {written} records ({type(exc).__name__}: "
                  f"{str(exc)[:120]}); retrying with the next reader.", flush=True)
    else:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"every reader failed; last error: {last_error}")
    tmp.rename(out)

    repeated = sum(1 for count in seen.values() if count > 1)
    print(f"\nwrote {written} records -> {out}")
    print(f"  size: {args.source.stat().st_size / 2**30:.2f} GB -> {out.stat().st_size / 2**30:.2f} GB")
    # Chunked extraction restarts per-trajectory history at a chunk boundary, so repeated ids
    # are the one thing that makes chunking non-equivalent to whole-file extraction.
    if repeated:
        print(f"  WARNING: {repeated} trajectory_id(s) appear more than once. Chunked extraction "
              f"restarts their history at chunk boundaries -- keep --trajectory-chunk-size above "
              f"the largest repeat group, or leave this file as .json.")
    else:
        print("  all trajectory_ids unique -> chunked extraction is equivalent to whole-file")


if __name__ == "__main__":
    main()
