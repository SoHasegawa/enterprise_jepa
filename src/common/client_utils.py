import asyncio
import json
import os
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory, Consumer
from a2a.types import (
    DataPart,
    FilePart,
    FileWithBytes,
    FileWithUri,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
)

from common.logging_utils import get_logger
from common.trajectory import redact_trajectory_payload

LOGGER = get_logger(__name__)
DEFAULT_TIMEOUT = float(os.getenv("BENCHMARK_A2A_CLIENT_TIMEOUT", "900"))
DEFAULT_AGENT_CARD_FETCH_RETRIES = max(
    1,
    int(os.getenv("BENCHMARK_A2A_AGENT_CARD_FETCH_RETRIES", "5")),
)
DEFAULT_AGENT_CARD_FETCH_INITIAL_BACKOFF = max(
    0.0,
    float(os.getenv("BENCHMARK_A2A_AGENT_CARD_FETCH_INITIAL_BACKOFF", "1.0")),
)
DEFAULT_AGENT_CARD_FETCH_MAX_BACKOFF = max(
    DEFAULT_AGENT_CARD_FETCH_INITIAL_BACKOFF,
    float(os.getenv("BENCHMARK_A2A_AGENT_CARD_FETCH_MAX_BACKOFF", "8.0")),
)
FilePayload = FileWithBytes | FileWithUri
INTERNAL_TRAJECTORY_ARTIFACT_NAME = "internal_trajectory"


def create_message(
    *,
    role: Role = Role.user,
    text: str,
    context_id: str | None = None,
) -> Message:
    """テキストだけを持つ A2A メッセージを作る。"""
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def create_message_with_files(
    *,
    role: Role = Role.user,
    text: str,
    file_payloads: list[FilePayload],
    context_id: str | None = None,
) -> Message:
    """テキストと添付ファイルをまとめた A2A メッセージを作る。"""
    parts = [Part(TextPart(kind="text", text=text))]
    for file_payload in file_payloads:
        parts.append(Part(FilePart(kind="file", file=file_payload)))
    return Message(
        kind="message",
        role=role,
        parts=parts,
        message_id=uuid4().hex,
        context_id=context_id,
    )


def _stringify_data(value: Any) -> str:
    """A2A パーツから得た値を文字列へ正規化する。"""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def merge_parts(parts: list[Part]) -> str:
    """A2A の複数パーツをログしやすい 1 本の文字列へまとめる。"""
    chunks: list[str] = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(_stringify_data(part.root.data))
        elif isinstance(part.root, FilePart):
            chunks.append(_stringify_data(part.root.file.model_dump()))
    return "\n".join(chunks)


def _append_response_text(outputs: dict[str, Any], text: str) -> None:
    """複数の応答断片を改行区切りで結合する。"""
    if not text:
        return
    if outputs["response"]:
        outputs["response"] += "\n"
    outputs["response"] += text


def _is_retryable_agent_card_error(exc: Exception) -> bool:
    """agent-card 取得の一時的な通信失敗だけを再試行対象にする。"""
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ),
    ):
        return True

    message = str(exc).lower()
    retryable_markers = (
        "agent card",
        "agent-card",
        "all connection attempts failed",
        "503",
        "502",
        "504",
        "temporarily unavailable",
        "connection refused",
        "connection reset",
        "network communication error",
    )
    return any(marker in message for marker in retryable_markers)


async def _resolve_agent_card_with_retry(
    *,
    resolver: A2ACardResolver,
    base_url: str,
) -> Any:
    """agent-card 解決時の短い一時障害を吸収する。"""
    last_error: Exception | None = None
    for attempt in range(1, DEFAULT_AGENT_CARD_FETCH_RETRIES + 1):
        try:
            return await resolver.get_agent_card()
        except Exception as exc:
            last_error = exc
            if attempt >= DEFAULT_AGENT_CARD_FETCH_RETRIES or not _is_retryable_agent_card_error(
                exc
            ):
                raise

            delay = min(
                DEFAULT_AGENT_CARD_FETCH_INITIAL_BACKOFF * (2 ** (attempt - 1)),
                DEFAULT_AGENT_CARD_FETCH_MAX_BACKOFF,
            )
            LOGGER.warning(
                "Agent card fetch failed for %s (attempt %d/%d): %s; retrying in %.1fs",
                base_url,
                attempt,
                DEFAULT_AGENT_CARD_FETCH_RETRIES,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to resolve agent card for {base_url}")


def _model_dump(value: Any) -> Any:
    """Pydantic/A2A オブジェクトを JSON 化しやすい値へ落とす。"""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _model_dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_dump(item) for item in value]
    return value


def _event_text(event: Any) -> str:
    """trajectory で検索しやすい短い text 表現を抜き出す。"""
    match event:
        case Message() as message:
            return merge_parts(message.parts)
        case (_, TaskStatusUpdateEvent() as status_event):
            parts = status_event.status.message.parts if status_event.status.message else []
            return merge_parts(parts)
        case (_, TaskArtifactUpdateEvent() as artifact_event):
            return merge_parts(artifact_event.artifact.parts)
        case (task, None):
            parts = task.status.message.parts if task.status.message else []
            return merge_parts(parts)
        case _:
            return ""


def _serialize_trajectory_event(
    *,
    sequence: int,
    direction: str,
    event: Any,
) -> dict[str, Any]:
    """A2A client event を trajectory 保存用 dict へ変換する。"""
    record: dict[str, Any] = {
        "sequence": sequence,
        "direction": direction,
        "event_type": type(event).__name__,
    }
    if direction == "inbound":
        text = _event_text(event)
        if text:
            record["text"] = text

    match event:
        case Message() as message:
            record["event_type"] = "Message"
            record["payload"] = _model_dump(message)
        case (task, event_update):
            record["event_type"] = (
                type(event_update).__name__ if event_update is not None else "Task"
            )
            record["task"] = _model_dump(task)
            record["event"] = _model_dump(event_update)
        case _:
            record["payload"] = _model_dump(event)

    return redact_trajectory_payload(record)


def _initial_send_outputs(
    outbound_message: Message,
    *,
    capture_events: bool,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {"response": "", "context_id": None, "artifacts": []}
    if capture_events:
        outputs["events"] = [
            _serialize_trajectory_event(
                sequence=0,
                direction="outbound",
                event=outbound_message,
            )
        ]
    return outputs


def _append_trajectory_event(outputs: dict[str, Any], event: Any) -> None:
    events = outputs["events"]
    events.append(
        _serialize_trajectory_event(
            sequence=len(events),
            direction="inbound",
            event=event,
        )
    )


async def _collect_send_outputs(
    client: Any,
    outbound_message: Message,
    *,
    capture_events: bool,
) -> tuple[dict[str, Any], Any]:
    last_event = None
    outputs = _initial_send_outputs(outbound_message, capture_events=capture_events)
    async for event in client.send_message(outbound_message):
        last_event = event
        if capture_events:
            _append_trajectory_event(outputs, event)
    return outputs, last_event


def _apply_message_outputs(outputs: dict[str, Any], message: Message) -> None:
    outputs["context_id"] = message.context_id
    outputs["response"] = merge_parts(message.parts)


def _apply_task_outputs(outputs: dict[str, Any], task: Any) -> None:
    outputs["context_id"] = task.context_id
    outputs["status"] = task.status.state.value
    if task.status.message:
        _append_response_text(outputs, merge_parts(task.status.message.parts))
    if not task.artifacts:
        return

    for artifact in task.artifacts:
        artifact_dump = _model_dump(artifact)
        outputs["artifacts"].append(artifact_dump)
        if getattr(artifact, "name", None) != INTERNAL_TRAJECTORY_ARTIFACT_NAME:
            _append_response_text(outputs, merge_parts(artifact.parts))


def _apply_last_event_outputs(outputs: dict[str, Any], last_event: Any) -> None:
    match last_event:
        case Message() as message:
            _apply_message_outputs(outputs, message)
        case (task, _):
            _apply_task_outputs(outputs, task)
        case _:
            return


async def _send_message_impl(
    *,
    outbound_message: Message,
    base_url: str,
    streaming: bool,
    consumer: Consumer | None,
    capture_events: bool,
) -> dict[str, Any]:
    """A2A クライアントの共通送信処理を実行する。"""
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await _resolve_agent_card_with_retry(resolver=resolver, base_url=base_url)
            config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
            client = ClientFactory(config).create(agent_card)
            if consumer:
                await client.add_event_consumer(consumer)

            outputs, last_event = await _collect_send_outputs(
                client,
                outbound_message,
                capture_events=capture_events,
            )
            _apply_last_event_outputs(outputs, last_event)
            return outputs
    except Exception as exc:
        LOGGER.error("A2A 通信失敗: %s", exc)
        raise RuntimeError(f"Error communicating with agent at {base_url}: {exc}") from exc


async def send_message(
    message: str,
    base_url: str,
    context_id: str | None = None,
    streaming: bool = False,
    consumer: Consumer | None = None,
    capture_events: bool = False,
) -> dict[str, Any]:
    """テキストだけを送って A2A 応答を受け取る。"""
    outbound_message = create_message(text=message, context_id=context_id)
    return await _send_message_impl(
        outbound_message=outbound_message,
        base_url=base_url,
        streaming=streaming,
        consumer=consumer,
        capture_events=capture_events,
    )


async def send_message_with_files(
    message: str,
    file_payloads: list[FilePayload],
    base_url: str,
    context_id: str | None = None,
    streaming: bool = False,
    consumer: Consumer | None = None,
    capture_events: bool = False,
) -> dict[str, Any]:
    """添付ファイル付きで A2A 応答を受け取る。"""
    outbound_message = create_message_with_files(
        text=message,
        file_payloads=file_payloads,
        context_id=context_id,
    )
    return await _send_message_impl(
        outbound_message=outbound_message,
        base_url=base_url,
        streaming=streaming,
        consumer=consumer,
        capture_events=capture_events,
    )
