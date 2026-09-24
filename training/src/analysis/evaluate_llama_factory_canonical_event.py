#!/usr/bin/env python3
"""Evaluate next-state prediction of an LLM-based EWM checkpoint against the
LlamaFactory canonical_event_with_nudge eval sets.

Takes a trained model (a merged HF causal-LM checkpoint, or a LoRA adapter
directory with `adapter_config.json`) and runs it over every
`data/llama_factory/canonical_event_with_nudge/*eval.json` alpaca file (each
`{"instruction", "input", "output"}` record's `instruction`/`input` become the
system/user chat turns, exactly as they were built by
`convert_canonical_nudge_to_llama_factory.py` from `build_state_prediction_chat_messages`).
The model's generated JSON is parsed and scored per-field against the gold
`canonical_event_with_nudge` label with the same categorical-match logic
`src/evaluation.py` uses for the JEPA head classifier, so the resulting
`canonical_field_comparisons` records are drop-in compatible with
`src/analysis/calculate_canonical_field_accuracy.py`.

Usage:

    uv run python src/analysis/evaluate_llama_factory_canonical_event.py \\
        /path/to/merged_or_adapter_checkpoint \\
        --limit 200
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.calculate_canonical_field_accuracy import compute_field_accuracy
from src.data_preparation.canonical_event_state import canonical_field_matches
from src.finetuning import parse_jsonish

DEFAULT_EVAL_GLOB = str(ROOT / "data" / "llama_factory" / "canonical_event_with_nudge" / "*eval.json")
DATASET_PREFIX = "canonical_event_with_nudge_"


class VLLMChatGenerator:
    """Minimal client for a vLLM OpenAI-compatible /v1/chat/completions server.

    Exposes the same ``generate_from_messages(messages, temperature)`` surface as
    ``src.evaluation.HFTextGenerator`` so the scorer is agnostic to the backend.
    The vLLM server already applies the model's chat template (and, when started
    with ``--default-chat-template-kwargs '{"enable_thinking": false}'``, disables
    the thinking block), so we just forward the system/user turns.
    """

    def __init__(self, base_url: str, model: str, max_new_tokens: int, timeout: float = 600.0) -> None:
        import openai

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.client = openai.OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)

    def generate_from_messages(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content or ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "model_path",
        type=Path,
        nargs="?",
        default=None,
        help="HF checkpoint directory: a merged causal LM, or a LoRA adapter dir with adapter_config.json. "
        "Omit when scoring against a running vLLM server via --vllm-base-url.",
    )
    parser.add_argument(
        "--vllm-base-url",
        default=None,
        help="Base URL of a running vLLM OpenAI-compatible server, e.g. http://localhost:9010/v1. "
        "When set, requests are sent to the server instead of loading a local HF checkpoint.",
    )
    parser.add_argument(
        "--vllm-model",
        default="gymops_world_model",
        help="Served model name to request from the vLLM server (--served-model-name / LoRA module name).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent generation requests (only used with --vllm-base-url). Default 1.",
    )
    parser.add_argument(
        "--eval-glob",
        default=DEFAULT_EVAL_GLOB,
        help=f"Glob of alpaca eval JSON files to score. Default: {DEFAULT_EVAL_GLOB}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max examples scored per eval file (0 = score every example). Generation is one-example-at-a-time, so full crmarenapro/enterpriseops_gym eval files can take a long time.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-chat-template", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed applied before --limit truncates each file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "evaluation" / "llama_factory_canonical_event_eval_results.json",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def benchmark_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith(DATASET_PREFIX):
        stem = stem[len(DATASET_PREFIX):]
    return stem[: -len("_eval")] if stem.endswith("_eval") else stem


def load_alpaca_examples(path: Path, limit: int, seed: int) -> list[dict[str, str]]:
    examples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(examples, list):
        raise ValueError(f"{path} does not contain a JSON list of alpaca examples")
    if limit > 0 and len(examples) > limit:
        import random

        rng = random.Random(seed)
        examples = rng.sample(examples, limit)
    return examples


def score_example(generator: HFTextGenerator, example: dict[str, str], temperature: float) -> dict[str, Any]:
    gold = json.loads(example["output"])
    messages = [
        {"role": "system", "content": example.get("instruction", "")},
        {"role": "user", "content": example.get("input", "")},
    ]
    raw_prediction = generator.generate_from_messages(messages, temperature=temperature)
    try:
        parsed_prediction = parse_jsonish(raw_prediction)
    except (ValueError, TypeError):
        parsed_prediction = None
    predicted_is_dict = isinstance(parsed_prediction, dict)
    comparisons = canonical_field_matches(gold, parsed_prediction if predicted_is_dict else None, include_nudge=True)
    return {
        "gold": gold,
        "prediction_raw": raw_prediction,
        "prediction_parsed": parsed_prediction if predicted_is_dict else None,
        "prediction_valid_json": predicted_is_dict,
        "canonical_field_comparisons": comparisons,
        "all_fields_match": bool(comparisons) and all(c["match"] for c in comparisons.values()),
    }


def print_summary(label: str, records: list[dict[str, Any]]) -> None:
    result = compute_field_accuracy(records)
    print(f"\n=== {label} ({len(records)} records) ===")
    print(f"scored_records: {result['scored_records']} (skipped: {result['skipped_records']})")

    def _fmt(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "n/a"

    print(f"micro_field_accuracy: {_fmt(result['micro_field_accuracy'])}")
    print(f"macro_field_accuracy: {_fmt(result['macro_field_accuracy'])}")
    print(f"full_match_rate: {_fmt(result['full_match_rate'])}")
    for field, summary in result["per_field"].items():
        print(f"  {field}: {summary['accuracy']:.4f} ({summary['correct']}/{summary['count']})")


def main() -> None:
    args = parse_args()
    eval_paths = sorted(Path(p) for p in glob.glob(args.eval_glob))
    if not eval_paths:
        raise SystemExit(f"No eval files matched --eval-glob {args.eval_glob!r}")

    if args.vllm_base_url:
        print(f"Using vLLM server at {args.vllm_base_url} (model={args.vllm_model}) ...")
        generator = VLLMChatGenerator(
            base_url=args.vllm_base_url,
            model=args.vllm_model,
            max_new_tokens=args.max_new_tokens,
        )
        model_identifier = f"{args.vllm_base_url}#{args.vllm_model}"
    else:
        if args.model_path is None:
            raise SystemExit("Provide a model_path (HF checkpoint) or --vllm-base-url.")
        from src.evaluation import HFTextGenerator

        print(f"Loading model from {args.model_path} ...")
        generator = HFTextGenerator(
            str(args.model_path),
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            disable_chat_template=args.disable_chat_template,
            attn_implementation=args.attn_implementation,
            device_map=args.device_map,
        )
        model_identifier = str(args.model_path)

    all_records: list[dict[str, Any]] = []
    per_benchmark_records: dict[str, list[dict[str, Any]]] = {}
    invalid_json_count = 0
    started = time.time()

    concurrency = max(1, args.concurrency)
    for eval_path in eval_paths:
        benchmark = benchmark_name_from_path(eval_path)
        examples = load_alpaca_examples(eval_path, args.limit, args.seed)
        print(
            f"\nScoring {len(examples)} examples from {eval_path.name} "
            f"(benchmark={benchmark}, concurrency={concurrency}) ..."
        )
        bench_records: list[dict[str, Any]] = [None] * len(examples)  # type: ignore[list-item]
        completed = 0

        def _score(i: int, ex: dict[str, str]) -> tuple[int, dict[str, Any]]:
            rec = score_example(generator, ex, args.temperature)
            rec["benchmark"] = benchmark
            rec["source_file"] = eval_path.name
            return i, rec

        if concurrency == 1:
            for index, example in enumerate(examples):
                _, record = _score(index, example)
                bench_records[index] = record
                if not record["prediction_valid_json"]:
                    invalid_json_count += 1
                completed += 1
                if args.progress_every > 0 and completed % args.progress_every == 0:
                    elapsed = time.time() - started
                    print(f"  [{benchmark}] {completed}/{len(examples)} scored ({elapsed:.1f}s elapsed)")
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(_score, i, ex) for i, ex in enumerate(examples)]
                for future in as_completed(futures):
                    index, record = future.result()
                    bench_records[index] = record
                    if not record["prediction_valid_json"]:
                        invalid_json_count += 1
                    completed += 1
                    if args.progress_every > 0 and completed % args.progress_every == 0:
                        elapsed = time.time() - started
                        print(f"  [{benchmark}] {completed}/{len(examples)} scored ({elapsed:.1f}s elapsed)")

        per_benchmark_records[benchmark] = bench_records
        all_records.extend(bench_records)

    print(f"\nTotal examples scored: {len(all_records)} (invalid/unparseable JSON predictions: {invalid_json_count})")
    for benchmark, records in per_benchmark_records.items():
        print_summary(benchmark, records)
    print_summary("overall", all_records)

    output_payload = {
        "model_path": model_identifier,
        "eval_files": [str(p) for p in eval_paths],
        "limit_per_file": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "invalid_json_predictions": invalid_json_count,
        "overall": compute_field_accuracy(all_records),
        "per_benchmark": {
            benchmark: compute_field_accuracy(records) for benchmark, records in per_benchmark_records.items()
        },
        "records": all_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
