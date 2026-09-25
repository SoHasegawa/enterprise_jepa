"""Standardize identifier-like values in trajectory outputs and later tool args.

This script rewrites database-specific identifiers in world-model trajectories
into stable per-trajectory IDs that start at 0 and remain consistent across
later steps in the same trajectory.

It is intentionally conservative:
  - `last_tool_output` fields inside state messages are rewritten.
  - Later action-tool-call arguments are rewritten only when they use IDs that
    have already been observed in earlier tool outputs from the same trajectory.
  - Numeric values are rewritten only when they belong to identifier-like keys
    such as `id`, `*_id`, `*_ids`, or camelCase variants like `linkedObjectsIds`.
  - UUID-looking strings are always rewritten because they are almost always
    database-specific identifiers in these trajectories.
  - The original message structure is preserved and only the matched scalar
    spans are replaced, which works even when the tool output is a truncated
    JSON-like string rather than valid JSON.
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TRAJECTORIES_DIR = ROOT / "trajectories"
DEFAULT_PATTERNS = (
    "*_train_trajectories.json",
    "*_test_trajectories.json",
)
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
INTEGER_RE = re.compile(r"^-?(0|[1-9][0-9]*)$")
KEY_VALUE_ID_RE = re.compile(
    r"(?P<key>\b[A-Za-z_][A-Za-z0-9_]*\b)\s*=\s*(?P<value>[0-9]+|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
FAMILY_ALIASES = {
    "portal_user": "user",
    "assigned_to_user": "user",
    "assignment_group": "group",
    "new_cycle": "cycle",
}


@dataclass
class Replacement:
    start: int
    end: int
    text: str


@dataclass
class Frame:
    kind: str
    container_key: str | None
    expecting_key: bool = False
    pending_key: str | None = None


class IdentifierStandardizer:
    """Assigns stable placeholder IDs per trajectory and identifier family."""

    def __init__(self) -> None:
        self._family_maps: dict[str, dict[str, str]] = {}
        self._family_counters: dict[str, int] = {}

    def standardize_output(self, text: str) -> tuple[str, int]:
        replacements = self._collect_replacements(text)
        replacements.extend(self._collect_key_value_replacements(text))
        if not replacements:
            return text, 0

        chunks: list[str] = []
        cursor = 0
        for replacement in sorted(replacements, key=lambda item: item.start):
            if replacement.start < cursor:
                continue
            chunks.append(text[cursor:replacement.start])
            chunks.append(replacement.text)
            cursor = replacement.end
        chunks.append(text[cursor:])
        return "".join(chunks), len(replacements)

    def standardize_tool_arguments(self, payload: Any) -> tuple[Any, int]:
        updated, replacements = self._standardize_tool_argument_value(payload, path=[])
        return updated, replacements

    def _collect_replacements(self, text: str) -> list[Replacement]:
        replacements: list[Replacement] = []
        stack: list[Frame] = []
        index = 0
        expecting_value_for_key: str | None = None

        while index < len(text):
            char = text[index]

            if char.isspace():
                index += 1
                continue

            if char == "{":
                stack.append(Frame(kind="object", container_key=expecting_value_for_key, expecting_key=True))
                expecting_value_for_key = None
                index += 1
                continue

            if char == "[":
                stack.append(Frame(kind="array", container_key=expecting_value_for_key))
                expecting_value_for_key = None
                index += 1
                continue

            if char == "}" or char == "]":
                if stack:
                    stack.pop()
                expecting_value_for_key = None
                index += 1
                continue

            if char == ",":
                if stack and stack[-1].kind == "object":
                    stack[-1].expecting_key = True
                    stack[-1].pending_key = None
                expecting_value_for_key = None
                index += 1
                continue

            if char == ":":
                if stack and stack[-1].kind == "object":
                    expecting_value_for_key = stack[-1].pending_key
                    stack[-1].expecting_key = False
                index += 1
                continue

            if char == '"':
                token_end, decoded = self._parse_string(text, index)
                if stack and stack[-1].kind == "object" and stack[-1].expecting_key:
                    stack[-1].pending_key = decoded
                    stack[-1].expecting_key = False
                else:
                    path = self._current_path(stack, expecting_value_for_key)
                    replacement_text = self._replacement_for_scalar(path, decoded, is_string=True)
                    if replacement_text is not None:
                        replacements.append(Replacement(index, token_end, replacement_text))
                    expecting_value_for_key = None
                index = token_end
                continue

            if char == "-" or char.isdigit():
                token_end = self._parse_number(text, index)
                token = text[index:token_end]
                path = self._current_path(stack, expecting_value_for_key)
                replacement_text = self._replacement_for_scalar(path, token, is_string=False)
                if replacement_text is not None:
                    replacements.append(Replacement(index, token_end, replacement_text))
                expecting_value_for_key = None
                index = token_end
                continue

            if char.isalpha():
                token_end = self._parse_literal(text, index)
                expecting_value_for_key = None
                index = token_end
                continue

            index += 1

        return replacements

    def _collect_key_value_replacements(self, text: str) -> list[Replacement]:
        replacements: list[Replacement] = []
        for match in KEY_VALUE_ID_RE.finditer(text):
            key = match.group("key")
            value = match.group("value")
            if not self._is_identifier_key(key):
                continue
            family = self._family_for_path([key])
            normalized = self._lookup_standard_id(family, value)
            value_start, value_end = match.span("value")
            replacements.append(Replacement(value_start, value_end, normalized))
        return replacements

    def _standardize_tool_argument_value(self, value: Any, path: list[str]) -> tuple[Any, int]:
        if isinstance(value, dict):
            updated: dict[str, Any] = {}
            replacements = 0
            for key, item in value.items():
                new_item, item_replacements = self._standardize_tool_argument_value(item, path + [key])
                updated[key] = new_item
                replacements += item_replacements
            return updated, replacements

        if isinstance(value, list):
            updated_list: list[Any] = []
            replacements = 0
            for item in value:
                new_item, item_replacements = self._standardize_tool_argument_value(item, path)
                updated_list.append(new_item)
                replacements += item_replacements
            return updated_list, replacements

        if not path:
            return value, 0

        key = path[-1]
        family = self._family_for_path(path)

        if isinstance(value, int) and not isinstance(value, bool):
            normalized = self._existing_standard_id(family, str(value))
            if normalized is None:
                return value, 0
            return int(normalized), 1

        if isinstance(value, str):
            normalized = self._existing_standard_id(family, value)
            if normalized is None:
                return value, 0
            return normalized, 1

        return value, 0

    def _current_path(self, stack: list[Frame], current_key: str | None) -> list[str]:
        path = [frame.container_key for frame in stack if frame.container_key]
        if current_key:
            path.append(current_key)
        return path

    def _replacement_for_scalar(
        self,
        path: list[str],
        raw_value: str,
        *,
        is_string: bool,
    ) -> str | None:
        if not path:
            return None

        key = path[-1]
        family = self._family_for_path(path)

        if is_string:
            if not self._should_replace_string(key, raw_value):
                return None
            normalized = self._lookup_standard_id(family, raw_value)
            return json.dumps(normalized)

        if not INTEGER_RE.fullmatch(raw_value):
            return None
        if not self._is_identifier_key(key):
            return None

        normalized = self._lookup_standard_id(family, raw_value)
        return normalized

    def _lookup_standard_id(self, family: str, original: str) -> str:
        family_map = self._family_maps.setdefault(family, {})
        if original in family_map:
            return family_map[original]

        next_value = self._family_counters.get(family, 0)
        family_map[original] = str(next_value)
        self._family_counters[family] = next_value + 1
        return family_map[original]

    def _existing_standard_id(self, family: str, original: str) -> str | None:
        return self._family_maps.get(family, {}).get(original)

    def _should_replace_string(self, key: str, value: str) -> bool:
        if UUID_RE.fullmatch(value):
            return True
        return self._is_identifier_key(key)

    def _family_for_path(self, path: list[str]) -> str:
        key = self._normalize_key(path[-1])
        if key == "id":
            parent = next((self._normalize_key(item) for item in reversed(path[:-1]) if item), "")
            if parent and parent not in {"items", "results", "data", "nodes", "edges"}:
                return self._canonical_family(f"{parent}.id")
            return "id"
        if key.endswith("_ids"):
            return self._canonical_family(key[:-4])
        if key.endswith("_id"):
            return self._canonical_family(key[:-3])
        return self._canonical_family(key)

    def _is_identifier_key(self, key: str) -> bool:
        normalized = self._normalize_key(key)
        return normalized == "id" or normalized.endswith("_id") or normalized.endswith("_ids")

    def _normalize_key(self, key: str) -> str:
        snake_key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()
        return snake_key

    def _canonical_family(self, family: str) -> str:
        if family.endswith(".id"):
            prefix = family[:-3]
            return f"{FAMILY_ALIASES.get(prefix, prefix)}.id"
        return FAMILY_ALIASES.get(family, family)

    def _parse_string(self, text: str, start: int) -> tuple[int, str]:
        index = start + 1
        escaped = False
        while index < len(text):
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                raw = text[start : index + 1]
                try:
                    decoded = json.loads(raw)
                except json.JSONDecodeError:
                    decoded = raw[1:-1]
                return index + 1, decoded
            index += 1
        raw = text[start:]
        try:
            decoded = json.loads(raw + '"')
        except json.JSONDecodeError:
            decoded = raw[1:]
        return len(text), decoded

    def _parse_number(self, text: str, start: int) -> int:
        index = start
        while index < len(text) and text[index] in "-0123456789.eE+":
            index += 1
        return index

    def _parse_literal(self, text: str, start: int) -> int:
        index = start
        while index < len(text) and (text[index].isalpha() or text[index] == "_"):
            index += 1
        return index


def default_inputs() -> list[Path]:
    paths: list[Path] = []
    for pattern in DEFAULT_PATTERNS:
        paths.extend(sorted(TRAJECTORIES_DIR.glob(pattern)))
    return paths


def build_output_path(input_path: Path, output_dir: Path | None, in_place: bool) -> Path:
    if in_place:
        return input_path
    target_dir = output_dir or input_path.parent
    stem = input_path.stem + "_standardized"
    return target_dir / f"{stem}{input_path.suffix}"


def standardize_trajectory(trajectory: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    updated = deepcopy(trajectory)
    standardizer = IdentifierStandardizer()
    message_updates = 0
    replacement_count = 0

    for message in updated.get("messages") or []:
        content = message.get("content")
        if (
            message.get("role") == "action"
            and isinstance(content, dict)
            and isinstance(content.get("tool_calls"), list)
        ):
            action_updates = 0
            for tool_call in content["tool_calls"]:
                if not isinstance(tool_call, dict):
                    continue
                function_payload = tool_call.get("function")
                if not isinstance(function_payload, dict):
                    continue
                arguments = function_payload.get("arguments")
                updated_arguments, replacements = standardizer.standardize_tool_arguments(arguments)
                function_payload["arguments"] = updated_arguments
                action_updates += replacements
            if action_updates:
                message_updates += 1
                replacement_count += action_updates

        if not isinstance(content, dict):
            continue
        state = content.get("state")
        if not isinstance(state, dict):
            continue
        context = state.get("context")
        if not isinstance(context, dict):
            continue
        last_tool_output = context.get("last_tool_output")
        if not isinstance(last_tool_output, str) or not last_tool_output:
            continue

        standardized_output, replacements = standardizer.standardize_output(last_tool_output)
        if replacements == 0:
            continue
        context["last_tool_output"] = standardized_output
        message_updates += 1
        replacement_count += replacements

    return updated, message_updates, replacement_count


def process_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as handle:
        trajectories = json.load(handle)
    if not isinstance(trajectories, list):
        raise SystemExit(f"Expected a JSON list at {input_path}, got {type(trajectories).__name__}.")

    updated_trajectories = []
    updated_count = 0
    replacement_count = 0
    for trajectory in trajectories:
        updated, message_updates, replacements = standardize_trajectory(trajectory)
        updated_trajectories.append(updated)
        if message_updates:
            updated_count += 1
            replacement_count += replacements

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(updated_trajectories, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "trajectory_count": len(trajectories),
        "updated_trajectories": updated_count,
        "replacement_count": replacement_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Trajectory JSON files to rewrite. Defaults to all *_train/_test trajectory files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for rewritten files. Defaults to the input file directory.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input files instead of writing *_standardized.json siblings.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable run summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = args.inputs or default_inputs()
    if not input_paths:
        raise SystemExit("No input trajectory files found.")

    summaries = []
    for input_path in input_paths:
        output_path = build_output_path(input_path, args.output_dir, args.in_place)
        summaries.append(process_file(input_path, output_path))

    if args.json:
        print(json.dumps(summaries, indent=2))
        return

    for summary in summaries:
        print(f"input: {summary['input_path']}")
        print(f"output: {summary['output_path']}")
        print(f"trajectory_count: {summary['trajectory_count']}")
        print(f"updated_trajectories: {summary['updated_trajectories']}")
        print(f"replacement_count: {summary['replacement_count']}")
        print()


if __name__ == "__main__":
    main()
