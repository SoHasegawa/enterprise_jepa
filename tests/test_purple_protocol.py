import json

from common.purple_protocol import (
    PURPLE_REQUEST_SCHEMA_VERSION,
    build_purple_request_text,
    parse_purple_request_text,
)


def test_build_purple_request_text_preserves_legacy_text_without_routing_config() -> None:
    assert (
        build_purple_request_text("# Question\n1 + 1", {"target": "mixed"}) == "# Question\n1 + 1"
    )


def test_build_purple_request_text_wraps_executor_endpoint() -> None:
    text = build_purple_request_text(
        "# Question\n1 + 1",
        {
            "target": "mixed",
            "executor": "llm",
            "executor_endpoint": "http://executor.svc.cluster.local:8080",
            "executor_config": {"model": "google/gemma-4-31B-it"},
        },
    )

    payload = json.loads(text)
    assert payload == {
        "schema_version": PURPLE_REQUEST_SCHEMA_VERSION,
        "task_context": "# Question\n1 + 1",
        "executor": "llm",
        "executor_endpoint": "http://executor.svc.cluster.local:8080",
        "executor_config": {"model": "google/gemma-4-31B-it"},
    }


def test_parse_purple_request_text_returns_none_for_legacy_text() -> None:
    assert parse_purple_request_text("# Question\n1 + 1") is None


def test_parse_purple_request_text_reads_valid_envelope() -> None:
    text = build_purple_request_text(
        "task",
        {"executor": "local"},
    )

    assert parse_purple_request_text(text) == {
        "schema_version": PURPLE_REQUEST_SCHEMA_VERSION,
        "task_context": "task",
        "executor": "local",
    }
