from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

PURPLE_REQUEST_SCHEMA_VERSION = "benchmark.purple.request.v1"


def _normalized_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def build_purple_request_text(
    task_context: str,
    request_config: Mapping[str, Any] | None = None,
) -> str:
    """Wrap a benchmark task for request-scoped Purple executor routing.

    If the evaluation request does not specify routing fields, the original
    task text is returned to preserve compatibility with existing Purple
    executors.
    """
    config = request_config or {}
    executor = _normalized_text(config.get("executor") or config.get("executor_name"))
    executor_endpoint = _normalized_text(
        config.get("executor_endpoint") or config.get("purple_executor_endpoint")
    )
    raw_executor_config = config.get("executor_config")
    executor_config = raw_executor_config if isinstance(raw_executor_config, Mapping) else {}

    if not executor and not executor_endpoint and not executor_config:
        return task_context

    envelope = {
        "schema_version": PURPLE_REQUEST_SCHEMA_VERSION,
        "task_context": task_context,
    }
    if executor:
        envelope["executor"] = executor
    if executor_endpoint:
        envelope["executor_endpoint"] = executor_endpoint
    if executor_config:
        envelope["executor_config"] = dict(executor_config)
    return json.dumps(envelope, ensure_ascii=False)


def parse_purple_request_text(request_text: str) -> dict[str, Any] | None:
    """Return a Purple routing envelope, or None for legacy task text."""
    try:
        payload = json.loads(request_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != PURPLE_REQUEST_SCHEMA_VERSION:
        return None
    task_context = payload.get("task_context")
    if not isinstance(task_context, str) or not task_context.strip():
        raise ValueError("Purple request envelope requires task_context")
    return payload
