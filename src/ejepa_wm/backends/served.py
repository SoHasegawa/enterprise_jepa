"""Served-model World Model.

Uses a **served** OpenAI-compatible model (a local vLLM endpoint, or any proxy) as the
World Model. This lets the same served model the EnterpriseOps baselines run
(see ``assets/EnterpriseOps-Gym/run.sh``) double as the WM.

Endpoint resolution (first non-empty wins), so it reuses the baseline's serving env by default:

* base_url: ``WM_BASE_URL`` -> ``ENTERPRISEOPS_LLM_BASE_URL`` -> ``OPENAI_BASE_URL`` ->
  ``LOCAL_VLLM_BASE`` (the baseline's ``http://127.0.0.1:8020/v1``)
* api_key:  ``WM_API_KEY``  -> ``ENTERPRISEOPS_LLM_API_KEY`` -> ``OPENAI_API_KEY`` -> ``"EMPTY"``
  (vLLM ignores the key but the client requires one)
* model:    ``WMConfig.model`` (``--wm-model``) -> ``WM_MODEL`` -> ``ENTERPRISEOPS_LLM_MODEL``
"""
from __future__ import annotations

from collections.abc import Mapping

from ejepa_wm.backends._openai_chat import build_chat_client
from ejepa_wm.backends.llm import LlmWorldModel
from ejepa_wm.base import ChatFn, WMConfig


def resolve_served(config: WMConfig, env: Mapping[str, str]) -> tuple[str, str, str]:
    """Resolve (base_url, api_key, model) for a served WM, reusing baseline serving env."""
    base_url = (
        env.get("WM_BASE_URL")
        or env.get("ENTERPRISEOPS_LLM_BASE_URL")
        or env.get("OPENAI_BASE_URL")
        or env.get("LOCAL_VLLM_BASE")
        or ""
    ).strip()
    api_key = (
        env.get("WM_API_KEY")
        or env.get("ENTERPRISEOPS_LLM_API_KEY")
        or env.get("OPENAI_API_KEY")
        or "EMPTY"
    ).strip()
    model = (
        (config.model or "")
        or env.get("WM_MODEL")
        or env.get("ENTERPRISEOPS_LLM_MODEL")
        or ""
    ).strip()
    if not base_url:
        raise ValueError(
            "served WM needs a base_url: set WM_BASE_URL (or reuse the baseline's "
            "ENTERPRISEOPS_LLM_BASE_URL / LOCAL_VLLM_BASE)."
        )
    if not model:
        raise ValueError("served WM needs a model: set --wm-model, WM_MODEL, or ENTERPRISEOPS_LLM_MODEL.")
    return base_url, api_key, model


def build_served_chat_fn(config: WMConfig, env: Mapping[str, str] | None = None) -> ChatFn:
    import os

    base_url, api_key, model = resolve_served(config, env if env is not None else os.environ)
    client = build_chat_client(
        {"backend": "openai", "base_url": base_url, "api_key": api_key, "model": model}
    )

    def chat(messages: list[dict[str, str]]) -> str:
        return client.chat(messages)

    return chat


class ServedWorldModel(LlmWorldModel):
    """:class:`LlmWorldModel` wired to a served OpenAI-compatible endpoint (vLLM or any proxy)."""

    name = "served"

    def __init__(self, config: WMConfig) -> None:
        super().__init__(config, build_served_chat_fn(config))
