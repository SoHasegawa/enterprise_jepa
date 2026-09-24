"""Core inference launcher interfaces."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class InferenceSession:
    """A ready OpenAI-compatible inference endpoint."""

    name: str
    api_base: str
    served_model_name: str
    model: str = ""
    api_key: str = ""
    local_port: int | None = None
    remote_port: int | None = None
    logs: str = ""
    pid: int | None = None
    job_id: str = ""
    node: str = ""
    cleanup_command: str = ""
    backend_kind: str = ""
    run_id: str = ""
    summary_path: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def env(self, *, prefix: str | None = None) -> dict[str, str]:
        """Return shell-style environment values for this endpoint."""

        if prefix:
            normalized = _env_prefix(prefix)
            values = {
                f"{normalized}_BASE_URL": self.api_base,
                f"{normalized}_MODEL": self.served_model_name,
            }
            if self.api_key:
                values[f"{normalized}_API_KEY"] = self.api_key
            return values

        values = {
            "OPENAI_BASE_URL": self.api_base,
            "OPENAI_MODEL_NAME": self.served_model_name,
        }
        if self.api_key:
            values["OPENAI_API_KEY"] = self.api_key
        return values


class InferenceLauncher(Protocol):
    """A lifecycle-managed launcher for one inference endpoint."""

    def start(self) -> InferenceSession:
        """Start the endpoint and return once it is ready."""

    def stop(self) -> None:
        """Stop the endpoint if this launcher owns it."""

    @contextmanager
    def running(self) -> Iterator[InferenceSession]:
        """Context manager that starts and stops the endpoint."""

        session = self.start()
        try:
            yield session
        finally:
            self.stop()


def _env_prefix(value: str) -> str:
    normalized = "".join(
        char.upper() if char.isascii() and char.isalnum() else "_" for char in value.strip()
    ).strip("_")
    if not normalized:
        raise ValueError("Environment prefix must contain at least one alphanumeric character.")
    return normalized
