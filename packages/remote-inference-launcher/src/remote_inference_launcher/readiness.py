"""Benchmark-ready OpenAI-compatible endpoint readiness checks."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from remote_inference_launcher.diagnostics import sanitized_excerpt


@dataclass(frozen=True)
class ReadinessConfig:
    """Configuration for the shared endpoint handoff readiness check."""

    smoke_test: str = "chat_completion"
    prompt: str = "Reply with OK."
    max_tokens: int = 4
    timeout_seconds: float = 2.0
    retry_interval_seconds: float = 1.0


@dataclass(frozen=True)
class ReadinessResult:
    """Result of a single shared endpoint readiness attempt."""

    models_endpoint_ok: bool
    smoke_test_ok: bool
    smoke_test_kind: str
    failure_code: str | None = None
    failure_message: str | None = None
    excerpt: str = ""

    @property
    def ok(self) -> bool:
        """Return whether every enabled readiness stage passed."""

        return self.models_endpoint_ok and self.smoke_test_ok

    def to_summary(self) -> dict[str, object]:
        """Return summary JSON fields."""

        return {
            "models_endpoint_ok": self.models_endpoint_ok,
            "smoke_test_ok": self.smoke_test_ok,
            "smoke_test_kind": self.smoke_test_kind,
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
            "excerpt": self.excerpt,
        }


def validate_readiness_config(config: ReadinessConfig) -> ReadinessConfig:
    """Reject readiness policies that cannot be evaluated deterministically."""

    if config.smoke_test not in {"chat_completion", "disabled"}:
        raise ValueError("readiness.smoke_test must be 'chat_completion' or 'disabled'.")
    if config.timeout_seconds <= 0:
        raise ValueError("readiness.timeout_seconds must be positive.")
    if config.retry_interval_seconds <= 0:
        raise ValueError("readiness.retry_interval_seconds must be positive.")
    if config.max_tokens < 1:
        raise ValueError("readiness.max_tokens must be positive.")
    return config


def check_openai_readiness(
    api_base: str,
    *,
    model: str,
    api_key: str = "",
    config: ReadinessConfig | None = None,
) -> ReadinessResult:
    """Run the shared OpenAI-compatible readiness check once."""

    readiness_config = validate_readiness_config(config or ReadinessConfig())
    models_ok, models_error = _check_models(api_base, api_key=api_key, readiness=readiness_config)
    if not models_ok:
        return ReadinessResult(
            models_endpoint_ok=False,
            smoke_test_ok=False,
            smoke_test_kind=readiness_config.smoke_test,
            failure_code="readiness_models_failed",
            failure_message=models_error,
            excerpt=sanitized_excerpt(models_error),
        )
    if readiness_config.smoke_test == "disabled":
        return ReadinessResult(
            models_endpoint_ok=True,
            smoke_test_ok=True,
            smoke_test_kind="disabled",
        )
    smoke_ok, smoke_error = _check_chat_completion(
        api_base,
        model=model,
        api_key=api_key,
        readiness=readiness_config,
    )
    if not smoke_ok:
        return ReadinessResult(
            models_endpoint_ok=True,
            smoke_test_ok=False,
            smoke_test_kind=readiness_config.smoke_test,
            failure_code="readiness_smoke_failed",
            failure_message=smoke_error,
            excerpt=sanitized_excerpt(smoke_error),
        )
    return ReadinessResult(
        models_endpoint_ok=True,
        smoke_test_ok=True,
        smoke_test_kind=readiness_config.smoke_test,
    )


def wait_for_openai_readiness(
    api_base: str,
    *,
    model: str,
    api_key: str = "",
    config: ReadinessConfig | None = None,
    timeout_seconds: float,
    should_continue: Callable[[], None] | None = None,
) -> ReadinessResult:
    """Poll until the endpoint passes shared readiness or the timeout expires."""

    readiness_config = validate_readiness_config(config or ReadinessConfig())
    started_at = time.monotonic()
    last_result = ReadinessResult(
        models_endpoint_ok=False,
        smoke_test_ok=False,
        smoke_test_kind=readiness_config.smoke_test,
        failure_code="readiness_models_failed",
        failure_message="readiness has not been checked yet",
    )
    while True:
        if should_continue is not None:
            should_continue()
        last_result = check_openai_readiness(
            api_base,
            model=model,
            api_key=api_key,
            config=readiness_config,
        )
        if last_result.ok:
            return last_result
        if timeout_seconds > 0 and time.monotonic() - started_at >= timeout_seconds:
            return last_result
        time.sleep(max(readiness_config.retry_interval_seconds, 0.1))


def _check_models(
    api_base: str,
    *,
    api_key: str,
    readiness: ReadinessConfig,
) -> tuple[bool, str]:
    request = urllib.request.Request(api_base.rstrip("/") + "/models")
    _add_auth(request, api_key)
    try:
        with urllib.request.urlopen(request, timeout=readiness.timeout_seconds) as response:
            if response.status >= 400:
                return False, f"/models returned HTTP {response.status}"
            return True, ""
    except (OSError, urllib.error.URLError) as error:
        return False, str(error)


def _check_chat_completion(
    api_base: str,
    *,
    model: str,
    api_key: str,
    readiness: ReadinessConfig,
) -> tuple[bool, str]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": readiness.prompt}],
        "max_tokens": readiness.max_tokens,
        "temperature": 0,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    _add_auth(request, api_key)
    try:
        with urllib.request.urlopen(request, timeout=readiness.timeout_seconds) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        return False, f"/chat/completions returned HTTP {error.code}: {detail}"
    except (OSError, urllib.error.URLError) as error:
        return False, str(error)
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        return False, f"/chat/completions returned non-JSON response: {response_body[:200]}"
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, f"/chat/completions returned no choices: {response_body[:200]}"
    return True, ""


def _add_auth(request: urllib.request.Request, api_key: str) -> None:
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
