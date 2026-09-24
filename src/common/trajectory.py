from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

TRUTHY_VALUES = {"1", "true", "yes", "on", "y"}
DEFAULT_TRAJECTORY_DIR_NAME = "trajectories"
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|authorization|bearer|password|secret|token|credential)",
    re.IGNORECASE,
)


def capture_trajectory_enabled(config: Mapping[str, Any] | None) -> bool:
    """request config / env から trajectory 保存の有効化を判定する。"""
    values = []
    if config:
        values.extend(
            [
                config.get("capture_trajectory"),
                config.get("capture_trajectories"),
                config.get("trajectory_capture"),
            ]
        )
    values.append(os.getenv("BENCHMARK_CAPTURE_TRAJECTORY"))

    for value in values:
        if isinstance(value, bool):
            return value
        if value is None:
            continue
        if str(value).strip().lower() in TRUTHY_VALUES:
            return True
    return False


def trajectory_root_for_result(result_dir: Path) -> Path:
    """1 run 分の trajectory 保存先ディレクトリを返す。"""
    return result_dir / DEFAULT_TRAJECTORY_DIR_NAME


def max_text_chars() -> int:
    """trajectory に保存する 1 文字列あたりの最大長。"""
    raw = os.getenv("BENCHMARK_TRAJECTORY_MAX_TEXT_CHARS", "16000")
    try:
        parsed = int(raw)
    except ValueError:
        return 16000
    return max(1000, parsed)


def _slugify(value: str, *, fallback: str = "item", max_length: int = 96) -> str:
    normalized = "".join(
        char if char.isascii() and (char.isalnum() or char in {"-", "_"}) else "_"
        for char in value.strip()
    )
    collapsed = re.sub(r"_+", "_", normalized).strip("_")
    return (collapsed or fallback)[:max_length]


def _redact_trajectory_text(value: str, *, limit: int) -> str:
    if value.startswith("data:") and ";base64," in value:
        return f"{value[:120]}...[TRUNCATED data URL {len(value)} chars]"
    if len(value) > limit:
        return f"{value[:limit]}...[TRUNCATED {len(value) - limit} chars]"
    return value


def redact_trajectory_payload(value: Any, *, max_chars: int | None = None) -> Any:
    """trajectory 用 payload から secret と巨大文字列を取り除く。"""
    limit = max_chars if max_chars is not None else max_text_chars()
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _SENSITIVE_KEY_PATTERN.search(key_text):
                redacted[key_text] = "[REDACTED]"
            else:
                redacted[key_text] = redact_trajectory_payload(item, max_chars=limit)
        return redacted
    if isinstance(value, list):
        return [redact_trajectory_payload(item, max_chars=limit) for item in value]
    if isinstance(value, tuple):
        return [redact_trajectory_payload(item, max_chars=limit) for item in value]
    if isinstance(value, str):
        return _redact_trajectory_text(value, limit=limit)
    return value


def write_task_trajectory(
    *,
    trajectory_root: Path,
    task_id: str,
    events: list[Mapping[str, Any]],
    label: str | None = None,
) -> Path | None:
    """1 task の trajectory events を JSON Lines で保存する。"""
    if not events:
        return None

    directory = trajectory_root
    if label:
        directory = directory / _slugify(label, fallback="phase")
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{_slugify(task_id, fallback='task')}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            redacted = redact_trajectory_payload(event)
            handle.write(json.dumps(redacted, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return path


def build_trajectory_capture_summary(
    *,
    enabled: bool,
    trajectory_root: Path | None,
) -> dict[str, Any]:
    """detail.json に入れる trajectory capture の設定サマリ。"""
    return {
        "enabled": enabled,
        "format": "jsonl",
        "event_source": "A2A client events and optional Purple executor internals",
        "directory": str(trajectory_root) if trajectory_root else None,
        "max_text_chars": max_text_chars(),
        "redaction": {
            "sensitive_key_pattern": _SENSITIVE_KEY_PATTERN.pattern,
            "data_url_base64": "truncated",
        },
    }
