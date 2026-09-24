"""Adapter for already-running OpenAI-compatible endpoints."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from remote_inference_launcher.config_types import DiagnosticsConfig
from remote_inference_launcher.core import InferenceSession
from remote_inference_launcher.diagnostics import sanitized_excerpt
from remote_inference_launcher.readiness import (
    ReadinessConfig,
    check_openai_readiness,
    validate_readiness_config,
)
from remote_inference_launcher.resource_budget import (
    ResourceBudgetReservation,
    acquire_current_resource_budget,
)
from remote_inference_launcher.summaries import SummaryWriter, endpoint_summary, new_run_id


@dataclass(frozen=True)
class ExistingEndpointConfig:
    """Configuration for an endpoint managed outside this launcher."""

    name: str = "default"
    api_base: str = ""
    served_model_name: str = ""
    model: str = ""
    api_key: str = ""
    discover_model: bool = True
    readiness: ReadinessConfig = field(default_factory=ReadinessConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    launch_summary_path: str = ""
    overwrite_launch_summary: bool = False


class ExistingEndpointLauncher:
    """Validate and expose an endpoint that is already running."""

    owns_resources = False

    def __init__(self, config: ExistingEndpointConfig) -> None:
        self.config = validate_existing_endpoint_config(config)
        self._run_id = new_run_id()
        self._summary = SummaryWriter(
            self.config.launch_summary_path,
            run_id=self._run_id,
            endpoint_name=self.config.name,
            backend_kind="existing_endpoint",
            overwrite=self.config.overwrite_launch_summary,
        )
        self._budget_token: ResourceBudgetReservation | None = None

    def start(self) -> InferenceSession:
        """Return a session after validating endpoint readiness."""

        api_base = self.config.api_base.rstrip("/")
        self._summary.reserve()
        self._budget_token = acquire_current_resource_budget(candidate_attempts=1)
        failure_summary_written = False
        served_model_name = self.config.served_model_name or self.config.model
        try:
            if not served_model_name and self.config.discover_model:
                served_model_name = discover_openai_model(api_base, api_key=self.config.api_key)
            readiness = check_openai_readiness(
                api_base,
                model=served_model_name or self.config.model,
                api_key=self.config.api_key,
                config=self.config.readiness,
            )
            if not readiness.ok:
                self._summary.write(
                    endpoint_summary(
                        endpoint_name=self.config.name,
                        backend_kind="existing_endpoint",
                        lifecycle_state="FAILED",
                        run_id=self._run_id,
                        api_base=api_base,
                        served_model_name=served_model_name,
                        model=self.config.model,
                        api_key_set=bool(self.config.api_key),
                        readiness=readiness,
                        failure_code=readiness.failure_code,
                        failure_message=readiness.failure_message,
                    )
                )
                failure_summary_written = True
                raise RuntimeError(
                    "OpenAI-compatible endpoint is not ready: "
                    f"{api_base} ({readiness.failure_code}: {readiness.failure_message})"
                )
            served_model_name = served_model_name or self.config.model
            if not served_model_name:
                raise RuntimeError(
                    "Existing endpoint is ready but no served model name was configured or "
                    "discovered."
                )
            session = InferenceSession(
                name=self.config.name,
                api_base=api_base,
                served_model_name=served_model_name,
                model=self.config.model or served_model_name,
                api_key=self.config.api_key,
                backend_kind="existing_endpoint",
                run_id=self._run_id,
                summary_path=str(self._summary.path),
                metadata={
                    "readiness": readiness.to_summary(),
                    "launch_summary_path": str(self._summary.path),
                },
            )
            self._summary.write(
                endpoint_summary(
                    endpoint_name=session.name,
                    backend_kind="existing_endpoint",
                    lifecycle_state="READY",
                    run_id=self._run_id,
                    api_base=session.api_base,
                    served_model_name=session.served_model_name,
                    model=session.model,
                    api_key_set=bool(session.api_key),
                    readiness=readiness,
                )
            )
            return session
        except BaseException as error:
            if not failure_summary_written:
                message = str(error)
                self._summary.write(
                    endpoint_summary(
                        endpoint_name=self.config.name,
                        backend_kind="existing_endpoint",
                        lifecycle_state="FAILED",
                        run_id=self._run_id,
                        api_base=api_base,
                        served_model_name=served_model_name,
                        model=self.config.model,
                        api_key_set=bool(self.config.api_key),
                        failure_code="readiness_models_failed",
                        failure_message=sanitized_excerpt(message),
                    )
                )
            self.stop()
            raise

    def reserve_summary(self):
        """Reserve the launch summary path before external lifecycle work starts."""

        return self._summary.reserve()

    def stop(self) -> None:
        """Do not stop endpoints owned by another process."""
        if self._budget_token is not None:
            self._budget_token.release()
            self._budget_token = None

    @contextmanager
    def running(self) -> Iterator[InferenceSession]:
        """Context manager matching owned launchers."""

        session = self.start()
        try:
            yield session
        finally:
            self.stop()


def validate_existing_endpoint_config(config: ExistingEndpointConfig) -> ExistingEndpointConfig:
    """Reject existing-endpoint configs that cannot be checked."""

    if not config.name.strip():
        raise ValueError("Existing endpoint name must be non-empty.")
    if not config.api_base.strip():
        raise ValueError("Existing endpoint api_base must be non-empty.")
    if not config.api_base.rstrip("/").endswith("/v1"):
        raise ValueError("Existing endpoint api_base must end with /v1.")
    try:
        validate_readiness_config(config.readiness)
    except ValueError as error:
        raise ValueError(f"Existing endpoint {error}") from error
    return config


def discover_openai_model(api_base: str, *, api_key: str = "", timeout_seconds: float = 5) -> str:
    """Return the first model id from an OpenAI-compatible `/models` response."""

    request = urllib.request.Request(api_base.rstrip("/") + "/models")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Failed to discover model from {api_base}: {error}") from error

    data = body.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
                return item["id"].strip()
    return ""
