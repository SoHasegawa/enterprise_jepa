import asyncio

import httpx

from common.client_utils import send_message


class _AsyncClientStub:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        return None


class _MessageClientStub:
    async def add_event_consumer(self, consumer) -> None:
        return None

    async def send_message(self, outbound_message):
        yield outbound_message.model_copy(
            update={"parts": outbound_message.parts, "context_id": "ctx-1"}
        )


def test_send_message_retries_transient_agent_card_fetch_failures(monkeypatch) -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    class ResolverStub:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def get_agent_card(self):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise httpx.ConnectError("temporary connect failure")
            return {"url": "http://agent.example:8080/"}

    class ClientFactoryStub:
        def __init__(self, config) -> None:
            self.config = config

        def create(self, agent_card):
            return _MessageClientStub()

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("common.client_utils.httpx.AsyncClient", _AsyncClientStub)
    monkeypatch.setattr("common.client_utils.A2ACardResolver", ResolverStub)
    monkeypatch.setattr("common.client_utils.ClientFactory", ClientFactoryStub)
    monkeypatch.setattr("common.client_utils.asyncio.sleep", fake_sleep)

    response = asyncio.run(send_message("hello", "http://agent.example:8080"))

    assert response["response"] == "hello"
    assert response["context_id"] == "ctx-1"
    assert attempts["count"] == 3
    assert sleeps == [1.0, 2.0]


def test_send_message_fails_fast_for_non_retryable_agent_card_error(monkeypatch) -> None:
    attempts = {"count": 0}

    class ResolverStub:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def get_agent_card(self):
            attempts["count"] += 1
            raise ValueError("bad response schema")

    monkeypatch.setattr("common.client_utils.httpx.AsyncClient", _AsyncClientStub)
    monkeypatch.setattr("common.client_utils.A2ACardResolver", ResolverStub)

    try:
        asyncio.run(send_message("hello", "http://agent.example:8080"))
    except RuntimeError as exc:
        assert "bad response schema" in str(exc)
    else:
        raise AssertionError("send_message should have failed")

    assert attempts["count"] == 1
