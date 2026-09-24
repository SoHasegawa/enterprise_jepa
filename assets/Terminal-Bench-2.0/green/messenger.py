import json
from uuid import uuid4

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
    Consumer,
)
from a2a.types import Message, Part, Role, TextPart, DataPart

from common.client_utils import INTERNAL_TRAJECTORY_ARTIFACT_NAME
from common.trajectory import redact_trajectory_payload


DEFAULT_TIMEOUT = 300


def create_message(
    *, role: Role = Role.user, text: str, context_id: str | None = None
) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def merge_parts(parts: list[Part]) -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return "\n".join(chunks)


def _model_dump(value: object) -> object:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _model_dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_dump(item) for item in value]
    return value


def _event_text(event: object) -> str:
    match event:
        case Message() as msg:
            return merge_parts(msg.parts)
        case (_, update):
            status = getattr(update, "status", None)
            message = getattr(status, "message", None)
            parts = getattr(message, "parts", []) if message else []
            return merge_parts(parts)
        case _:
            return ""


def _serialize_trajectory_event(sequence: int, direction: str, event: object) -> dict:
    record: dict = {
        "sequence": sequence,
        "direction": direction,
        "event_type": type(event).__name__,
    }
    text = _event_text(event)
    if direction == "inbound" and text:
        record["text"] = text

    match event:
        case Message() as msg:
            record["event_type"] = "Message"
            record["payload"] = _model_dump(msg)
        case (task, event_update):
            record["event_type"] = (
                type(event_update).__name__ if event_update is not None else "Task"
            )
            record["task"] = _model_dump(task)
            record["event"] = _model_dump(event_update)
        case _:
            record["payload"] = _model_dump(event)

    return redact_trajectory_payload(record)


async def send_message(
    message: str,
    base_url: str,
    context_id: str | None = None,
    streaming: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    consumer: Consumer | None = None,
    capture_events: bool = False,
) -> dict:
    """Returns dict with context_id, response, status, events, and artifacts."""
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        agent_card.url = base_url
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=streaming,
        )
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        if consumer:
            await client.add_event_consumer(consumer)

        outbound_msg = create_message(text=message, context_id=context_id)
        last_event = None
        outputs: dict = {"response": "", "context_id": None, "artifacts": []}
        if capture_events:
            outputs["events"] = [
                _serialize_trajectory_event(0, "outbound", outbound_msg)
            ]

        async for event in client.send_message(outbound_msg):
            last_event = event
            if capture_events:
                outputs["events"].append(
                    _serialize_trajectory_event(len(outputs["events"]), "inbound", event)
                )

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                outputs["response"] += merge_parts(msg.parts)

            case (task, _update):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                msg = task.status.message
                if msg:
                    outputs["response"] += merge_parts(msg.parts)
                if task.artifacts:
                    for artifact in task.artifacts:
                        outputs["artifacts"].append(_model_dump(artifact))
                        if getattr(artifact, "name", None) != INTERNAL_TRAJECTORY_ARTIFACT_NAME:
                            outputs["response"] += merge_parts(artifact.parts)

            case _:
                pass

        return outputs


class Messenger:
    def __init__(self) -> None:
        self._context_ids: dict[str, str | None] = {}

    async def talk_to_agent(
        self,
        message: str,
        url: str,
        new_conversation: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
        capture_trajectory: bool = False,
    ) -> str:
        outputs = await self.talk_to_agent_with_trajectory(
            message=message,
            url=url,
            new_conversation=new_conversation,
            timeout=timeout,
            capture_trajectory=capture_trajectory,
        )
        if outputs.get("status", "completed") != "completed":
            raise RuntimeError(f"{url} responded with: {outputs}")
        return str(outputs["response"])

    async def talk_to_agent_with_trajectory(
        self,
        message: str,
        url: str,
        new_conversation: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
        capture_trajectory: bool = False,
    ) -> dict:
        outputs = await send_message(
            message=message,
            base_url=url,
            context_id=None if new_conversation else self._context_ids.get(url),
            timeout=timeout,
            streaming=True,
            capture_events=capture_trajectory,
        )
        if outputs.get("status", "completed") != "completed":
            raise RuntimeError(f"{url} responded with: {outputs}")
        self._context_ids[url] = outputs.get("context_id")  # type: ignore[assignment]
        return outputs

    def reset(self) -> None:
        self._context_ids = {}
