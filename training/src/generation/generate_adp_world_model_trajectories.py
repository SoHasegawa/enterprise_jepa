# Provenance: vendored from the EWM repo, branch `jepa` (commit 7b62196). Produces the
# expanded Stage-1 corpus of the paper (Appendix B, Table 6: Agent Data Protocol sources).
"""Convert Agent Data Protocol (ADP) standardized trajectories into the lean
world-model trajectory format consumed by ``src/finetuning.py``, with a
minimal ``state`` message (``context.last_tool_output`` only) recording each
action's observation.

Source
------
The ADP release ``neulab/agent-data-collection`` ships one subset per benchmark,
each with a ``full_std.jsonl`` in the ADP *standardized* schema: every line is a
``{"id", "content": [...], "details": {...}}`` record whose ``content`` items are
tagged with a ``class_`` discriminator:

  - ``text_observation``  {content, source in {user, agent, environment}, name?}
  - ``message_action``    {content, description?, reasoning_content?}
  - ``code_action``       {content, language, description?, reasoning_content?}
  - ``api_action``        {function, kwargs, description?, reasoning_content?}
  - ``web_observation``   {html, axtree, url, image_observation, ...}

Target
------
Each output trajectory is ``{"messages": [...]}`` with roles ``system`` / ``user``
/ ``assistant`` / ``action`` / ``state`` -- the same shape produced by the other
generators in this directory, but the ``state`` messages here are lean: unlike
the full six-aspect schema built elsewhere (see
``generate_toucan_world_model_trajectories.py``), each one carries only
``{"state": {"context": {"last_tool_output": ...}}}``. Environment observations
(``text_observation`` with ``source == environment`` and every
``web_observation``) are what populate these state messages -- for
``web_observation`` the text is taken from ``axtree``, falling back to ``html``
then ``url``, kept in full (respecting ``--max-content-chars`` if it is set).

Actions carry OpenAI-style ``tool_calls``:
``{"tool_calls": [{"type": "function", "function": {"name", "arguments"}}]}``.

One output file is written per benchmark:
``trajectories/<subset>_world_model_trajectories.json``.

Usage
-----
    # every subset in the local HF cache (skips toucan_1_5m, already generated)
    python src/generation/generate_adp_world_model_trajectories.py

    # a single benchmark, capped for a quick look
    python src/generation/generate_adp_world_model_trajectories.py \
        --dataset swe-smith --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The ADP standardized dump that ships with `neulab/agent-data-collection`.
DEFAULT_REPO_ID = "neulab/agent-data-collection"
DEFAULT_SNAPSHOT_PATH = Path(
    "/data/user/hub/datasets--neulab--agent-data-collection/snapshots/"
    "31a76bfb0124d77ae7322eabbb0171bf11ee2c67"
)
DEFAULT_OUTPUT_DIR = ROOT / "trajectories"

# toucan_1_5m already has trajectories/toucan_world_model_trajectories.json
# (generated from the raw Toucan source), so it is skipped by default.
DEFAULT_SKIP = {"toucan_1_5m"}

BASH_LANGUAGES = {"bash", "sh", "shell", "zsh"}
PYTHON_LANGUAGES = {"python", "python3", "py", "ipython"}

FINISH_RE = re.compile(r"<finish>(.*?)</finish>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=None,
        help="ADP dataset snapshot dir (one subdir per subset). Auto-detected from "
        "the HF cache when omitted.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="HF dataset repo id used for cache auto-detection.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Restrict to these subset names. Repeat to pass several. Default: all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where the per-benchmark trajectory files are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of trajectories kept per benchmark (for quick looks).",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=None,
        help="Subset name to skip. Repeat to pass several. Defaults to toucan_1_5m.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate benchmarks whose output file already exists.",
    )
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=0,
        help="If > 0, truncate any message string to this many characters.",
    )
    return parser.parse_args()


def resolve_snapshot_path(repo_id: str, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Snapshot path does not exist: {explicit}")
        return explicit
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_type == "dataset" and repo.repo_id == repo_id:
                for revision in repo.revisions:
                    if revision.snapshot_path is not None:
                        return Path(revision.snapshot_path)
    except Exception:
        pass
    if DEFAULT_SNAPSHOT_PATH.exists():
        return DEFAULT_SNAPSHOT_PATH
    raise FileNotFoundError(
        f"Could not locate a local snapshot for {repo_id!r}. "
        "Pass --snapshot-path explicitly."
    )


def discover_subsets(snapshot_path: Path) -> list[str]:
    subsets = []
    for entry in sorted(snapshot_path.iterdir()):
        if entry.is_dir() and (entry / "full_std.jsonl").exists():
            subsets.append(entry.name)
    return subsets


def truncate(text: str, limit: int) -> str:
    if limit and len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def system_prompt_for(dataset: str) -> str:
    return (
        "You are an autonomous agent. Given the user's task, reason about the "
        "problem and take actions (tool calls or code execution) to accomplish "
        f"it. You are operating in the '{dataset}' environment."
    )


def join_nonempty(*parts: str, sep: str = "\n\n") -> str:
    return sep.join(part for part in (p.strip() if p else "" for p in parts) if part)


def code_action_tool_call(language: str, content: str) -> dict[str, Any]:
    lang = (language or "").lower()
    if lang in BASH_LANGUAGES:
        return {"name": "execute_bash", "arguments": {"command": content}}
    if lang in PYTHON_LANGUAGES:
        return {"name": "execute_ipython_cell", "arguments": {"code": content}}
    return {"name": "execute_code", "arguments": {"language": language, "content": content}}


def make_action_message(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "action",
        "content": {
            "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": arguments}}
            ]
        },
    }


def make_state_message(last_tool_output: str) -> dict[str, Any]:
    return {
        "role": "state",
        "content": {"state": {"context": {"last_tool_output": last_tool_output}}},
    }


def web_observation_text(item: dict[str, Any]) -> str:
    for key in ("axtree", "html", "url"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def convert_record(
    record: dict[str, Any],
    *,
    dataset: str,
    index: int,
    max_content_chars: int,
) -> dict[str, Any] | None:
    content = record.get("content")
    if not isinstance(content, list) or not content:
        return None

    body: list[dict[str, Any]] = []
    action_count = 0

    def add_thought(item: dict[str, Any]) -> None:
        thought = join_nonempty(
            item.get("reasoning_content") or "", item.get("description") or ""
        )
        if thought:
            body.append({"role": "assistant", "content": truncate(thought, max_content_chars)})

    for item in content:
        if not isinstance(item, dict):
            continue
        cls = item.get("class_")

        if cls == "text_observation":
            source = item.get("source", "environment")
            text = (item.get("content") or "").strip()
            if not text:
                continue
            if source == "user":
                body.append({"role": "user", "content": truncate(text, max_content_chars)})
            elif source == "agent":
                body.append({"role": "assistant", "content": truncate(text, max_content_chars)})
            else:
                # source == "environment": this is the tool output for the
                # preceding action -> a lean state message.
                body.append(make_state_message(truncate(text, max_content_chars)))

        elif cls == "web_observation":
            text = web_observation_text(item)
            if text:
                body.append(make_state_message(truncate(text, max_content_chars)))

        elif cls == "message_action":
            text = (item.get("content") or "").strip()
            finish = FINISH_RE.search(text)
            if finish:
                text = finish.group(1).strip()
            merged = join_nonempty(
                item.get("reasoning_content") or "", item.get("description") or "", text
            )
            if merged:
                body.append({"role": "assistant", "content": truncate(merged, max_content_chars)})

        elif cls == "code_action":
            add_thought(item)
            call = code_action_tool_call(
                item.get("language", ""), str(item.get("content", ""))
            )
            body.append(make_action_message(call["name"], call["arguments"]))
            action_count += 1

        elif cls == "api_action":
            add_thought(item)
            arguments = item.get("kwargs")
            if not isinstance(arguments, dict):
                arguments = {}
            body.append(make_action_message(str(item.get("function", "")), arguments))
            action_count += 1

        # Unknown class_ -> skipped silently.

    # Require at least one user turn and at least one agent turn (action or text).
    has_user = any(m["role"] == "user" for m in body)
    has_agent = any(m["role"] in ("assistant", "action") for m in body)
    if not (has_user and has_agent):
        return None

    trajectory_id = record.get("id") or f"{dataset}-{index}"
    details = record.get("details") if isinstance(record.get("details"), dict) else {}
    return {
        "trajectory_id": str(trajectory_id),
        "source": f"adp:{dataset}",
        "domain": dataset,
        "dataset": dataset,
        "split": "all",
        "action_count": action_count,
        "available_apis": details.get("available_apis"),
        "messages": [{"role": "system", "content": system_prompt_for(dataset)}] + body,
    }


def iter_std_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def generate_for_subset(
    subset: str,
    snapshot_path: Path,
    output_dir: Path,
    *,
    limit: int | None,
    max_content_chars: int,
) -> dict[str, Any]:
    src = snapshot_path / subset / "full_std.jsonl"
    out = output_dir / f"{subset}_world_model_trajectories.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    seen = 0
    # Stream the (potentially multi-GB) input and write the JSON array as we go
    # so memory stays flat regardless of subset size.
    with out.open("w", encoding="utf-8") as writer:
        writer.write("[")
        first = True
        for record in iter_std_records(src):
            seen += 1
            trajectory = convert_record(
                record,
                dataset=subset,
                index=seen,
                max_content_chars=max_content_chars,
            )
            if trajectory is None:
                continue
            writer.write(("\n" if first else ",\n") + json.dumps(trajectory, ensure_ascii=False))
            first = False
            kept += 1
            if limit is not None and kept >= limit:
                break
        writer.write("\n]\n" if not first else "]\n")

    return {
        "subset": subset,
        "input": str(src),
        "output": str(out),
        "records_read": seen,
        "trajectories_written": kept,
    }


def main() -> int:
    args = parse_args()
    snapshot_path = resolve_snapshot_path(args.repo_id, args.snapshot_path)
    skip = set(args.skip) if args.skip is not None else set(DEFAULT_SKIP)

    subsets = discover_subsets(snapshot_path)
    if args.dataset:
        missing = [d for d in args.dataset if d not in subsets]
        if missing:
            raise SystemExit(f"Requested subset(s) not found: {', '.join(missing)}")
        subsets = list(args.dataset)
    else:
        subsets = [s for s in subsets if s not in skip]

    print(f"snapshot: {snapshot_path}", file=sys.stderr)
    print(f"benchmarks to process ({len(subsets)}): {', '.join(subsets)}", file=sys.stderr)

    summaries = []
    for subset in subsets:
        out = args.output_dir / f"{subset}_world_model_trajectories.json"
        if out.exists() and not args.overwrite and not args.dataset:
            print(f"[skip] {subset}: {out.name} already exists (use --overwrite)", file=sys.stderr)
            continue
        print(f"[run ] {subset} ...", file=sys.stderr, flush=True)
        summary = generate_for_subset(
            subset,
            snapshot_path,
            args.output_dir,
            limit=args.limit,
            max_content_chars=args.max_content_chars,
        )
        summaries.append(summary)
        print(
            f"[done] {subset}: {summary['trajectories_written']} trajectories "
            f"from {summary['records_read']} records -> {Path(summary['output']).name}",
            file=sys.stderr,
            flush=True,
        )

    print(json.dumps({"snapshot_path": str(snapshot_path), "results": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
