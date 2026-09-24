"""Minimal OpenAI-compatible chat client used by the ``served`` world-model backend.

The upstream monorepo built this client through an internal model factory. Here it is a
thin wrapper over the ``openai`` SDK so the backend works against any OpenAI-compatible
endpoint (local vLLM, a proxy, or the OpenAI API itself) with no extra dependency.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ChatClient:
    """``.chat(messages) -> str`` over an OpenAI-compatible ``/chat/completions``."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = self._client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()


def build_chat_client(overrides: Mapping[str, Any]) -> ChatClient:
    """Build a :class:`ChatClient` from the ``served`` backend's override mapping."""
    options = dict(overrides)
    options.pop("backend", None)
    return ChatClient(
        base_url=str(options.pop("base_url", "")),
        api_key=str(options.pop("api_key", "EMPTY")),
        model=str(options.pop("model", "")),
        temperature=options.pop("temperature", None),
        max_tokens=options.pop("max_tokens", None),
        timeout=options.pop("timeout", None),
    )
