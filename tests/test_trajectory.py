import asyncio
import json

from common.purple_client import PurpleClient
from common.trajectory import (
    capture_trajectory_enabled,
    redact_trajectory_payload,
    write_task_trajectory,
)


def test_capture_trajectory_enabled_accepts_bool_and_string() -> None:
    assert capture_trajectory_enabled({"capture_trajectory": True})
    assert capture_trajectory_enabled({"capture_trajectories": "yes"})
    assert capture_trajectory_enabled({"trajectory_capture": "1"})
    assert not capture_trajectory_enabled({"capture_trajectory": False})


def test_redact_trajectory_payload_masks_secrets_and_truncates_data_urls() -> None:
    payload = {
        "api_key": "sk-secret",
        "nested": {"Authorization": "Bearer secret"},
        "image": "data:image/png;base64," + ("a" * 2000),
        "text": "hello",
    }

    redacted = redact_trajectory_payload(payload, max_chars=100)

    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["Authorization"] == "[REDACTED]"
    assert redacted["image"].endswith("[TRUNCATED data URL 2022 chars]")
    assert redacted["text"] == "hello"


def test_write_task_trajectory_writes_jsonl_with_redaction(tmp_path) -> None:
    path = write_task_trajectory(
        trajectory_root=tmp_path,
        task_id="QUIZ/001",
        label="pass 1",
        events=[
            {"sequence": 0, "api_token": "secret", "text": "request"},
            {"sequence": 1, "text": "response"},
        ],
    )

    assert path is not None
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {
        "api_token": "[REDACTED]",
        "sequence": 0,
        "text": "request",
    }
    assert json.loads(lines[1])["text"] == "response"


def test_purple_client_requests_streaming_when_capturing_trajectory(monkeypatch) -> None:
    calls = []

    async def fake_send_message_with_files(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "context_id": "ctx-1",
            "response": "answer",
            "events": [{"sequence": 0, "text": "event"}],
        }

    monkeypatch.setattr(
        "common.purple_client.send_message_with_files",
        fake_send_message_with_files,
    )

    client = PurpleClient()
    result = asyncio.run(
        client.send_message_with_trajectory(
            "question",
            [],
            "http://purple.example:8080",
            capture_trajectory=True,
        )
    )

    assert result["response"] == "answer"
    assert result["trajectory"] == [{"sequence": 0, "text": "event"}]
    assert calls[0]["streaming"] is True
    assert calls[0]["capture_events"] is True
