"""LLM-backed Purple executor for the terminal-bench-shell-v1 protocol."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlencode

import httpx
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

SRC_DIR = Path(__file__).resolve().parents[4] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.client_utils import INTERNAL_TRAJECTORY_ARTIFACT_NAME


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_AZURE_DEPLOYMENT = "gpt-5.5"
DEFAULT_MAX_STEPS = 40
# GPT-5.x on Azure requires a recent chat-completions API version.
DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"
DEFAULT_OPENAI_API_VERSION = "2024-10-21"
DEFAULT_GPT5_MAX_COMPLETION_TOKENS = 16384
DEFAULT_MAX_COMPLETION_TOKENS = 2048
DEFAULT_HISTORY_TEXT_CHARS = 12000
SYSTEM_PROMPT = """You solve terminal tasks by proposing shell commands.

You communicate using JSON only. On each turn respond with exactly one object:

1. To run a command:
{"kind":"exec_request","command":"<shell command>","timeout":<seconds 1-300>}

2. When finished:
{"kind":"final","output":"<short summary>"}

Rules:
- Use standard Linux shell commands.
- Prefer short, focused commands; inspect before mutating.
- Do not wrap JSON in markdown fences.
- timeout must be an integer between 1 and 300.
"""


@dataclass
class _LlmConfig:
    provider: str  # "azure" | "openai"
    api_key: str
    model: str
    base_url: str | None = None
    deployment: str | None = None
    api_version: str | None = None


@dataclass
class _Session:
    instruction: str
    history: list[dict[str, str]] = field(default_factory=list)
    steps: int = 0


def _extract_request_text(parts: list[Part]) -> str:
    chunks: list[str] = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
    request_text = "\n".join(chunks).strip()
    if not request_text:
        raise ValueError("No text part found in request message")
    return request_text


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_json_object(text: str, *, source: str) -> dict[str, Any]:
    """Parse one JSON object from model or protocol text (tolerates fences/extra lines)."""
    normalized = _strip_markdown_fence(text)
    if not normalized:
        raise ValueError(f"{source}: empty payload")

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []

    try:
        payload = json.loads(normalized)
        if isinstance(payload, dict):
            candidates.append(payload)
    except json.JSONDecodeError:
        pass

    for index, char in enumerate(normalized):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(normalized, index)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(payload)

    for line in normalized.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(payload)

    for payload in candidates:
        if payload.get("kind") in {"task", "exec_request", "exec_result", "final"}:
            return payload

    if candidates:
        return candidates[0]

    raise ValueError(f"{source}: could not parse a JSON object from: {normalized[:200]!r}")


def _decode_payload(message: str) -> dict[str, Any]:
    return _parse_json_object(message, source="protocol message")


def _normalize_azure_openai_endpoint(base_url: str) -> str:
    endpoint = base_url.split("?", 1)[0].rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break
    return f"{endpoint}/"


def _is_azure_openai_endpoint(base_url: str | None) -> bool:
    if not base_url:
        return False
    host = base_url.lower()
    return "openai.azure.com" in host or "cognitiveservices.azure.com" in host


def _is_gpt5_family(model: str) -> bool:
    normalized = model.strip().lower()
    if normalized.startswith("openai/"):
        normalized = normalized.split("/", 1)[1]
    return normalized.startswith("gpt-5")


def _resolve_azure_deployment() -> str:
    return str(
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        or os.getenv("TERMINAL_BENCH_LLM_MODEL")
        or os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL_NAME")
        or DEFAULT_AZURE_DEPLOYMENT
    )


def _resolve_openai_model() -> str:
    return str(
        os.getenv("TERMINAL_BENCH_LLM_MODEL")
        or os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL_NAME")
        or DEFAULT_MODEL
    )


def _resolve_azure_api_version() -> str:
    return str(
        os.getenv("AZURE_OPENAI_API_VERSION")
        or os.getenv("OPENAI_API_VERSION")
        or DEFAULT_AZURE_API_VERSION
    )


def _resolve_llm_config() -> _LlmConfig:
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    # Prefer Azure OpenAI when endpoint is configured (GPT-5.5 default path).
    if azure_endpoint:
        api_key = (
            azure_api_key
            or os.getenv("TERMINAL_BENCH_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Azure OpenAI requires AZURE_OPENAI_API_KEY "
                "(or TERMINAL_BENCH_LLM_API_KEY / OPENAI_API_KEY)."
            )
        deployment = _resolve_azure_deployment()
        return _LlmConfig(
            provider="azure",
            api_key=api_key,
            model=deployment,
            deployment=deployment,
            base_url=_normalize_azure_openai_endpoint(azure_endpoint),
            api_version=_resolve_azure_api_version(),
        )

    terminal_base = os.getenv("TERMINAL_BENCH_LLM_BASE_URL")
    base_url = (
        terminal_base
        or os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or DEFAULT_OPENAI_BASE_URL
    ).rstrip("/")

    if _is_azure_openai_endpoint(base_url):
        api_key = (
            azure_api_key
            or os.getenv("TERMINAL_BENCH_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("API key required for Azure OpenAI endpoint.")
        deployment = _resolve_azure_deployment()
        return _LlmConfig(
            provider="azure",
            api_key=api_key,
            model=deployment,
            deployment=deployment,
            base_url=_normalize_azure_openai_endpoint(base_url),
            api_version=_resolve_azure_api_version(),
        )

    api_key = (
        os.getenv("TERMINAL_BENCH_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or azure_api_key
    )
    if not api_key:
        raise ValueError(
            "API key required. Set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT "
            "for Azure GPT-5.5, or OPENAI_API_KEY / LLM_API_KEY for OpenAI-compatible APIs."
        )
    return _LlmConfig(
        provider="openai",
        api_key=api_key,
        model=_resolve_openai_model(),
        base_url=base_url,
    )


def _max_steps() -> int:
    return int(os.getenv("TERMINAL_BENCH_MAX_STEPS", str(DEFAULT_MAX_STEPS)))


def _max_completion_tokens(config: _LlmConfig) -> int:
    raw = os.getenv("TERMINAL_BENCH_LLM_MAX_TOKENS")
    if raw:
        return max(256, int(raw))
    if _is_gpt5_family(config.model):
        return DEFAULT_GPT5_MAX_COMPLETION_TOKENS
    return DEFAULT_MAX_COMPLETION_TOKENS


def _truncate_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[TRUNCATED {len(text) - max_chars} chars]"


def _sanitize_exec_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep shell results within model context limits."""
    sanitized = dict(payload)
    limit = int(os.getenv("TERMINAL_BENCH_LLM_MAX_OUTPUT_CHARS", str(DEFAULT_HISTORY_TEXT_CHARS)))
    for key in ("stdout", "stderr"):
        value = sanitized.get(key)
        if isinstance(value, str):
            sanitized[key] = _truncate_text(value, max_chars=limit)
    return sanitized


def _history_entry_for_llm(entry: dict[str, str]) -> dict[str, str]:
    role = str(entry.get("role", "user"))
    content = str(entry.get("content", ""))
    if role == "tool":
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("kind") == "exec_result":
            payload = _sanitize_exec_result_payload(payload)
            content = json.dumps(payload, ensure_ascii=False)
        return {
            "role": "user",
            "content": f"Shell command result:\n{content}",
        }
    if role == "assistant":
        return {"role": "assistant", "content": content}
    return {"role": "user", "content": content}


def _build_chat_messages(session: _Session) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task instruction:\n{session.instruction}"},
    ]
    for entry in session.history:
        messages.append(_history_entry_for_llm(entry))
    return messages


def _message_content_to_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    chunks.append(str(part["text"]))
                elif part.get("text"):
                    chunks.append(str(part["text"]))
        return "\n".join(chunks).strip()
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    if content is None:
        return ""
    return str(content).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    return _parse_json_object(text, source="LLM output")


def _completion_kwargs(config: _LlmConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if _is_gpt5_family(config.model):
        kwargs["max_completion_tokens"] = _max_completion_tokens(config)
        reasoning_effort = (
            os.getenv("AZURE_OPENAI_REASONING_EFFORT")
            or os.getenv("TERMINAL_BENCH_LLM_REASONING_EFFORT")
        )
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort.strip().lower()
    else:
        kwargs["max_tokens"] = _max_completion_tokens(config)
        kwargs["temperature"] = float(os.getenv("TERMINAL_BENCH_LLM_TEMPERATURE", "0"))
    return kwargs


def _chat_completions_url(config: _LlmConfig) -> str:
    if config.provider == "azure":
        assert config.base_url and config.deployment and config.api_version
        query = urlencode({"api-version": config.api_version})
        return (
            f"{config.base_url}openai/deployments/{config.deployment}"
            f"/chat/completions?{query}"
        )
    assert config.base_url
    return f"{config.base_url.rstrip('/')}/chat/completions"


def _chat_completions_headers(config: _LlmConfig) -> dict[str, str]:
    if config.provider == "azure":
        return {
            "api-key": config.api_key,
            "Content-Type": "application/json",
        }
    return {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }


def _build_chat_body(config: _LlmConfig, messages: list[dict[str, str]]) -> dict[str, Any]:
    body: dict[str, Any] = {"messages": messages, **_completion_kwargs(config)}
    if config.provider != "azure":
        body["model"] = config.model
    return body


def _parse_completion_payload(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Unexpected LLM response: {payload}")
    message = choices[0].get("message") or {}
    content = _message_content_to_text(message)
    if not content:
        raise RuntimeError(f"LLM returned empty content: {payload}")
    return _extract_json_object(content)


def _http_error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err)
            return json.dumps(body, ensure_ascii=False)[:2000]
    except Exception:
        pass
    text = response.text.strip()
    return text[:2000] if text else response.reason_phrase


def _call_llm_http(session: _Session, config: _LlmConfig) -> dict[str, Any]:
    messages = _build_chat_messages(session)
    body = _build_chat_body(config, messages)
    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            _chat_completions_url(config),
            headers=_chat_completions_headers(config),
            json=body,
        )
        if response.is_error:
            detail = _http_error_detail(response)
            raise RuntimeError(
                f"LLM request failed ({response.status_code}): {detail}"
            )
        return _parse_completion_payload(response.json())


def _call_llm_azure_sdk(session: _Session, config: _LlmConfig) -> dict[str, Any]:
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Azure GPT-5.5 requires the openai package. "
            "Install with: uv sync --project assets/Terminal-Bench-2.0/purple --extra llm_shell"
        ) from exc

    assert config.base_url and config.deployment and config.api_version
    client = AzureOpenAI(
        api_key=config.api_key,
        azure_endpoint=config.base_url.rstrip("/"),
        api_version=config.api_version,
    )
    messages = _build_chat_messages(session)
    completion = client.chat.completions.create(
        model=config.deployment,
        messages=messages,
        **_completion_kwargs(config),
    )
    payload = completion.model_dump(mode="json")
    return _parse_completion_payload(payload)


def _maybe_write_runtime_snapshot(config: _LlmConfig) -> None:
    """Persist deployment metadata for ejepa result executor_runtime resolution."""
    result_dir_env = os.getenv("BENCHMARK_RESULT_DIR")
    if not result_dir_env:
        return
    runtime_path = (
        Path(result_dir_env).expanduser().resolve()
        / ".runtime"
        / "llm_shell"
        / "runtime.json"
    )
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "provider": config.provider,
        "deployment": config.deployment or config.model,
        "api_version": config.api_version,
        "llm_models": [
            {
                "role": "chat",
                "model_name": config.deployment or config.model,
                "provider": config.provider,
            }
        ],
    }
    runtime_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _call_llm(session: _Session, config: _LlmConfig) -> dict[str, Any]:
    if config.provider == "azure":
        prefer_sdk = os.getenv("TERMINAL_BENCH_LLM_USE_HTTP", "").strip().lower() not in {
            "1",
            "true",
            "yes",
        }
        if prefer_sdk:
            return _call_llm_azure_sdk(session, config)
    return _call_llm_http(session, config)


class LlmShellExecutor(AgentExecutor):
    """Stateful shell agent used by the green orchestrator."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._llm_config = _resolve_llm_config()
        _maybe_write_runtime_snapshot(self._llm_config)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task:
            task = context.current_task
        elif context.message:
            task = new_task(context.message)
        else:
            raise ServerError(error=InvalidParamsError(message="No message provided"))

        if not context.message:
            raise ServerError(error=InvalidParamsError(message="No message provided"))

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        try:
            request_text = _extract_request_text(context.message.parts)
            payload = _decode_payload(request_text)
            kind = str(payload.get("kind", ""))
            context_id = task.context_id or task.id

            if kind == "task":
                instruction = payload.get("instruction")
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError("task payload requires non-empty instruction")
                self._sessions[context_id] = _Session(instruction=instruction.strip())
                response_payload = self._next_action(self._sessions[context_id])
            elif kind == "exec_result":
                session = self._sessions.get(context_id)
                if session is None:
                    raise ValueError("exec_result received without an active session")
                sanitized = _sanitize_exec_result_payload(payload)
                session.history.append(
                    {
                        "role": "tool",
                        "content": json.dumps(sanitized, ensure_ascii=False),
                    }
                )
                response_payload = self._next_action(session)
            else:
                raise ValueError(f"Unsupported payload kind: {kind}")

            session = self._sessions.get(context_id)
            if session is not None and session.history:
                internal_trajectory = {
                    "executor": "llm_shell",
                    "format": "terminal-bench-shell-v1",
                    "payload": {
                        "info": {
                            "steps": session.steps,
                            "provider": self._llm_config.provider,
                            "deployment": self._llm_config.deployment,
                            "api_version": self._llm_config.api_version,
                        },
                        "messages": list(session.history),
                    },
                }
                await updater.add_artifact(
                    parts=[
                        Part(
                            root=TextPart(
                                text=json.dumps(
                                    internal_trajectory, ensure_ascii=False
                                )
                            )
                        )
                    ],
                    name=INTERNAL_TRAJECTORY_ARTIFACT_NAME,
                )
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=json.dumps(response_payload)))]
            )
            await updater.complete()
        except Exception as exc:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    f"llm_shell execution failed: {exc}",
                    task.context_id,
                    task.id,
                ),
                final=True,
            )
            raise ServerError(error=InternalError(message=str(exc))) from exc

    def _next_action(self, session: _Session) -> dict[str, Any]:
        session.steps += 1
        if session.steps > _max_steps():
            return {"kind": "final", "output": "step limit reached"}

        action = _call_llm(session, self._llm_config)
        kind = str(action.get("kind", ""))
        if kind == "exec_request":
            command = action.get("command")
            timeout = action.get("timeout", 30)
            if not isinstance(command, str) or not command.strip():
                raise ValueError("exec_request requires non-empty command")
            if not isinstance(timeout, int):
                timeout = 30
            timeout = max(1, min(int(timeout), 300))
            normalized = {
                "kind": "exec_request",
                "command": command.strip(),
                "timeout": timeout,
            }
            session.history.append(
                {"role": "assistant", "content": json.dumps(normalized)}
            )
            return normalized

        if kind == "final":
            output = action.get("output", "")
            return {"kind": "final", "output": str(output)}

        raise ValueError(f"LLM returned unsupported action: {action}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())


def build_executor() -> AgentExecutor:
    return LlmShellExecutor()
