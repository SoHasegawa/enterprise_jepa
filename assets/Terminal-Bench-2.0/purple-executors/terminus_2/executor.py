"""Terminus-2 Purple executor for the terminal-bench-shell-v1 protocol."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

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

_EXECUTOR_DIR = Path(__file__).resolve().parent
if str(_EXECUTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_EXECUTOR_DIR))

SRC_DIR = Path(__file__).resolve().parents[4] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.client_utils import INTERNAL_TRAJECTORY_ARTIFACT_NAME

from parsers import AgentParseResult, ParsedCommand, convert_parse_result, get_parser


DEFAULT_MODEL = "gpt-4.1"
DEFAULT_AZURE_DEPLOYMENT = "gpt-5.5"
DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"
DEFAULT_MAX_STEPS = 80
DEFAULT_MAX_OUTPUT_BYTES = 10000
DEFAULT_MAX_COMPLETION_TOKENS = 16384
DEFAULT_PARSE_ATTEMPTS = 3
DEFAULT_TEMPERATURE = 0.7


@dataclass
class _LiteLlmConfig:
    litellm_model: str
    api_key: str
    provider: str
    deployment: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    temperature: float = DEFAULT_TEMPERATURE


@dataclass
class _Session:
    instruction: str
    messages: list[dict[str, str]] = field(default_factory=list)
    command_queue: list[ParsedCommand] = field(default_factory=list)
    pending_completion: bool = False
    pending_prompt: str | None = None
    last_terminal_output: str = "(empty terminal)"
    last_exec_command: str = ""
    steps: int = 0
    turns: list[dict[str, Any]] = field(default_factory=list)


def _extract_request_text(parts: list[Part]) -> str:
    chunks: list[str] = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
    request_text = "\n".join(chunks).strip()
    if not request_text:
        raise ValueError("No text part found in request message")
    return request_text


def _parse_json_object(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("protocol payload must be a JSON object")
    return payload


def _decode_payload(message: str) -> dict[str, Any]:
    return _parse_json_object(message)


def _normalize_azure_endpoint(base_url: str) -> str:
    endpoint = base_url.split("?", 1)[0].rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break
    return f"{endpoint.rstrip('/')}/"


def _is_azure_endpoint(base_url: str | None) -> bool:
    if not base_url:
        return False
    host = base_url.lower()
    return "openai.azure.com" in host or "cognitiveservices.azure.com" in host


def _is_gpt5_family(model: str) -> bool:
    normalized = model.strip().lower()
    if normalized.startswith("azure/"):
        normalized = normalized.split("/", 1)[1]
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


def _litellm_model_for_openai_compatible(model: str, *, api_base: str | None) -> str:
    """Prefix OpenAI-compatible models so LiteLLM does not treat org/model ids as HuggingFace."""
    if not api_base:
        return model
    if model.startswith(("openai/", "azure/", "hosted_vllm/", "ollama/")):
        return model
    return f"openai/{model}"


def _resolve_litellm_config() -> _LiteLlmConfig:
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    temperature = float(os.getenv("TERMINUS_2_TEMPERATURE", str(DEFAULT_TEMPERATURE)))

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
        api_version = (
            os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("OPENAI_API_VERSION")
            or DEFAULT_AZURE_API_VERSION
        )
        return _LiteLlmConfig(
            litellm_model=f"azure/{deployment}",
            api_key=api_key,
            provider="azure",
            deployment=deployment,
            api_base=_normalize_azure_endpoint(azure_endpoint),
            api_version=api_version,
            temperature=temperature,
        )

    base_url = (
        os.getenv("TERMINAL_BENCH_LLM_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )
    if base_url and _is_azure_endpoint(base_url):
        api_key = (
            azure_api_key
            or os.getenv("TERMINAL_BENCH_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("API key required for Azure OpenAI endpoint.")
        deployment = _resolve_azure_deployment()
        api_version = (
            os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("OPENAI_API_VERSION")
            or DEFAULT_AZURE_API_VERSION
        )
        return _LiteLlmConfig(
            litellm_model=f"azure/{deployment}",
            api_key=api_key,
            provider="azure",
            deployment=deployment,
            api_base=_normalize_azure_endpoint(base_url),
            api_version=api_version,
            temperature=temperature,
        )

    api_key = (
        os.getenv("TERMINAL_BENCH_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or azure_api_key
    )
    if not api_key:
        raise ValueError(
            "API key required. Set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT "
            "or OPENAI_API_KEY / LLM_API_KEY for OpenAI-compatible APIs."
        )
    model = _resolve_openai_model()
    api_base = base_url.rstrip("/") if base_url else None
    return _LiteLlmConfig(
        litellm_model=_litellm_model_for_openai_compatible(model, api_base=api_base),
        api_key=api_key,
        provider="openai",
        api_base=api_base,
        temperature=temperature,
    )


def _parser_name() -> str:
    return os.getenv("TERMINUS_2_PARSER", "json").strip().lower()


def _max_steps() -> int:
    return int(os.getenv("TERMINAL_BENCH_MAX_STEPS", str(DEFAULT_MAX_STEPS)))


def _max_completion_tokens(config: _LiteLlmConfig) -> int:
    raw = os.getenv("TERMINAL_BENCH_LLM_MAX_TOKENS")
    if raw:
        return max(256, int(raw))
    if _is_gpt5_family(config.litellm_model):
        return DEFAULT_MAX_COMPLETION_TOKENS
    return 4096


def _prompt_template_path(parser_name: str) -> Path:
    if parser_name == "json":
        return _EXECUTOR_DIR / "prompt_templates" / "terminus-json-plain.txt"
    if parser_name == "xml":
        return _EXECUTOR_DIR / "prompt_templates" / "terminus-xml-plain.txt"
    raise ValueError(f"Unknown TERMINUS_2_PARSER: {parser_name!r}")


def _limit_output_length(output: str, max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> str:
    encoded = output.encode("utf-8")
    if len(encoded) <= max_bytes:
        return output
    portion_size = max_bytes // 2
    first = encoded[:portion_size].decode("utf-8", errors="ignore")
    last = encoded[-portion_size:].decode("utf-8", errors="ignore")
    omitted = len(encoded) - len(first.encode("utf-8")) - len(last.encode("utf-8"))
    return (
        f"{first}\n[... output limited to {max_bytes} bytes; "
        f"{omitted} interior bytes omitted ...]\n{last}"
    )


def _format_exec_result(payload: dict[str, Any]) -> str:
    stdout = payload.get("stdout")
    stderr = payload.get("stderr")
    exit_code = payload.get("exit_code")
    chunks: list[str] = []
    if isinstance(stdout, str) and stdout:
        chunks.append(stdout)
    if isinstance(stderr, str) and stderr:
        chunks.append(f"[stderr]\n{stderr}")
    if exit_code is not None:
        chunks.append(f"[exit_code={exit_code}]")
    if not chunks:
        return "(no output)"
    return "\n".join(chunks)


def _completion_confirmation_message(terminal_output: str, parser_name: str) -> str:
    if parser_name == "xml":
        return (
            f"Current terminal state:\n{terminal_output}\n\n"
            "Are you sure you want to mark the task as complete? "
            "This will trigger your solution to be graded and you won't be able to "
            "make any further corrections. If so, include "
            "<task_complete>true</task_complete> again."
        )
    return (
        f"Current terminal state:\n{terminal_output}\n\n"
        "Are you sure you want to mark the task as complete? "
        "This will trigger your solution to be graded and you won't be able to "
        "make any further corrections. If so, include \"task_complete\": true "
        "in your JSON response again."
    )


def _parse_error_response_type(parser_name: str) -> str:
    return "JSON response" if parser_name == "json" else "response"


def _maybe_write_runtime_snapshot(config: _LiteLlmConfig) -> None:
    result_dir_env = os.getenv("BENCHMARK_RESULT_DIR")
    if not result_dir_env:
        return
    runtime_path = (
        Path(result_dir_env).expanduser().resolve()
        / ".runtime"
        / "terminus_2"
        / "runtime.json"
    )
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "provider": config.provider,
        "deployment": config.deployment or config.litellm_model,
        "api_version": config.api_version,
        "parser": _parser_name(),
        "llm_models": [
            {
                "role": "chat",
                "model_name": config.deployment or config.litellm_model,
                "provider": config.provider,
            }
        ],
    }
    runtime_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _litellm_completion_kwargs(config: _LiteLlmConfig) -> dict[str, Any]:
    """Build LiteLLM kwargs; GPT-5 family rejects non-default temperature."""
    kwargs: dict[str, Any] = {
        "model": config.litellm_model,
        "api_key": config.api_key,
    }
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.api_version:
        kwargs["api_version"] = config.api_version
    if _is_gpt5_family(config.litellm_model):
        kwargs["max_completion_tokens"] = _max_completion_tokens(config)
        reasoning_effort = (
            os.getenv("AZURE_OPENAI_REASONING_EFFORT")
            or os.getenv("TERMINAL_BENCH_LLM_REASONING_EFFORT")
        )
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort.strip().lower()
    else:
        kwargs["max_tokens"] = _max_completion_tokens(config)
        kwargs["temperature"] = config.temperature
    return kwargs


def _call_litellm(config: _LiteLlmConfig, messages: list[dict[str, str]]) -> str:
    try:
        import litellm
    except ImportError as exc:
        raise RuntimeError(
            "terminus_2 requires litellm. Install with: "
            "uv sync --project assets/Terminal-Bench-2.0/purple --extra terminus_2"
        ) from exc

    drop_unsupported = os.getenv("TERMINUS_2_LITELLM_DROP_PARAMS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if drop_unsupported or _is_gpt5_family(config.litellm_model):
        litellm.drop_params = True

    kwargs = _litellm_completion_kwargs(config)
    kwargs["messages"] = messages

    response = litellm.completion(**kwargs)
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if content is None:
        return ""
    return str(content).strip()


class Terminus2Executor(AgentExecutor):
    """Terminus-2 style batch-command agent over the green shell protocol."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._llm_config = _resolve_litellm_config()
        self._parser_name = _parser_name()
        self._parser = get_parser(self._parser_name)
        self._prompt_template = _prompt_template_path(self._parser_name).read_text(encoding="utf-8")
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
                session = _Session(instruction=instruction.strip())
                self._sessions[context_id] = session
                response_payload = self._next_action(session)
            elif kind == "exec_result":
                session = self._sessions.get(context_id)
                if session is None:
                    raise ValueError("exec_result received without an active session")
                terminal_output = _limit_output_length(_format_exec_result(payload))
                if payload.get("timed_out"):
                    command = session.last_exec_command or "(unknown)"
                    timeout_sec = payload.get("timeout_sec", "?")
                    terminal_output = (
                        f"Previous command:\n{command}\n\n"
                        f"The previous command timed out after {timeout_sec} seconds.\n\n"
                        f"It may still be running, or you may need a longer-running command. "
                        f"Current output:\n{terminal_output}"
                    )
                session.last_terminal_output = terminal_output
                session.messages.append(
                    {"role": "user", "content": session.last_terminal_output}
                )
                session.pending_prompt = None
                response_payload = self._next_action(session)
            else:
                raise ValueError(f"Unsupported payload kind: {kind}")

            session = self._sessions.get(context_id)
            if session is not None and session.turns:
                internal_trajectory = {
                    "executor": "terminus_2",
                    "format": "terminal-bench-shell-v1",
                    "payload": {
                        "info": {
                            "steps": session.steps,
                            "provider": self._llm_config.provider,
                            "deployment": self._llm_config.deployment,
                            "parser": self._parser_name,
                        },
                        "turns": list(session.turns),
                    },
                }
                await updater.add_artifact(
                    parts=[
                        Part(
                            root=TextPart(
                                text=json.dumps(internal_trajectory, ensure_ascii=False)
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
                    f"terminus_2 execution failed: {exc}",
                    task.context_id,
                    task.id,
                ),
                final=True,
            )
            raise ServerError(error=InternalError(message=str(exc))) from exc

    def _build_initial_prompt(self, session: _Session) -> str:
        return self._prompt_template.format(
            instruction=session.instruction,
            terminal_state=_limit_output_length(session.last_terminal_output),
        )

    def _record_turn(self, session: _Session, *, prompt: str, response: str, parsed: AgentParseResult) -> None:
        session.turns.append(
            {
                "prompt": prompt,
                "response": response,
                "parsed": {
                    "commands": [
                        {"command": cmd.command, "timeout": cmd.timeout}
                        for cmd in parsed.commands
                    ],
                    "is_task_complete": parsed.is_task_complete,
                    "error": parsed.error,
                    "warning": parsed.warning,
                },
            }
        )

    def _query_and_parse(self, session: _Session, prompt: str) -> AgentParseResult:
        messages = list(session.messages)
        messages.append({"role": "user", "content": prompt})
        feedback = ""
        last_response = ""
        for attempt in range(DEFAULT_PARSE_ATTEMPTS):
            attempt_prompt = prompt if not feedback else (
                f"Previous response had parsing errors:\n{feedback}\n\n"
                f"Please fix these issues and provide a proper "
                f"{_parse_error_response_type(self._parser_name)}."
            )
            if attempt > 0:
                messages[-1] = {"role": "user", "content": attempt_prompt}

            last_response = _call_litellm(self._llm_config, messages)
            messages.append({"role": "assistant", "content": last_response})
            parsed = convert_parse_result(self._parser.parse_response(last_response))
            self._record_turn(session, prompt=attempt_prompt, response=last_response, parsed=parsed)

            feedback = ""
            if parsed.error:
                feedback = f"ERROR: {parsed.error}"
                if parsed.warning:
                    feedback += f"\nWARNINGS: {parsed.warning}"
            elif parsed.warning:
                feedback = f"WARNINGS: {parsed.warning}"

            if not parsed.error:
                session.messages = messages
                return parsed

        session.messages = messages
        return parsed

    def _next_action(self, session: _Session) -> dict[str, Any]:
        session.steps += 1
        if session.steps > _max_steps():
            return {"kind": "final", "output": "step limit reached"}

        if session.command_queue:
            command = session.command_queue.pop(0)
            session.last_exec_command = command.command
            return {
                "kind": "exec_request",
                "command": command.command,
                "timeout": command.timeout,
            }

        while True:
            if session.pending_prompt:
                prompt = session.pending_prompt
                session.pending_prompt = None
            elif not session.messages:
                prompt = self._build_initial_prompt(session)
            else:
                if session.turns and session.turns[-1]["parsed"].get("warning"):
                    prompt = (
                        f"Previous response had warnings:\n"
                        f"WARNINGS: {session.turns[-1]['parsed']['warning']}\n\n"
                        f"{_limit_output_length(session.last_terminal_output)}"
                    )
                else:
                    prompt = _limit_output_length(session.last_terminal_output)

            parsed = self._query_and_parse(session, prompt)

            if parsed.error and not parsed.commands and not parsed.is_task_complete:
                session.pending_prompt = (
                    f"Previous response had parsing errors:\nERROR: {parsed.error}\n\n"
                    f"Please fix these issues and provide a proper "
                    f"{_parse_error_response_type(self._parser_name)}."
                )
                continue

            if parsed.commands:
                session.command_queue = list(parsed.commands)
                session.pending_completion = False
                command = session.command_queue.pop(0)
                session.last_exec_command = command.command
                return {
                    "kind": "exec_request",
                    "command": command.command,
                    "timeout": command.timeout,
                }

            if parsed.is_task_complete:
                if session.pending_completion:
                    return {
                        "kind": "final",
                        "output": "Task marked complete by Terminus-2 agent.",
                    }
                session.pending_completion = True
                session.pending_prompt = _completion_confirmation_message(
                    _limit_output_length(session.last_terminal_output),
                    self._parser_name,
                )
                continue

            session.pending_completion = False
            session.pending_prompt = (
                "Your response did not include executable commands or task_complete. "
                "Provide a valid Terminus-2 response."
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())


def build_executor() -> AgentExecutor:
    return Terminus2Executor()
