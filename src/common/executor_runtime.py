from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_CRMARENAPRO_ANTHROPIC_MODEL_ID = "claude-3-5-sonnet-latest"
DEFAULT_CRMARENAPRO_OPENAI_COMPATIBLE_MODEL_ID = "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"
DEFAULT_CRMARENAPRO_OPENAI_MODEL_ID = "gpt-4o-mini"
DEFAULT_TERMINAL_BENCH_LLM_MODEL_ID = "gpt-5.5"
UNKNOWN_LLM_NOTE = "Could not determine the LLM model name from saved artifacts."
NO_LLM_NOTE = "This executor does not use an LLM."
TERMINAL_BENCH_BENCHMARK_KEYS = frozenset({"terminal-bench-2.0", "terminal-bench-2"})
REQUEST_CONFIG_VLLM_MODEL_ID_NOTE = "request_config.vllm_model_id"


def _normalize_model_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_note(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _model_from_openai_base_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    path = value.split("?", 1)[0].rstrip("/")
    if not path:
        return None
    return _normalize_model_name(path.rsplit("/", 1)[-1])


def _env_flag(environ: Mapping[str, str], name: str) -> bool:
    value = environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _request_config_flag(config: Mapping[str, Any], name: str) -> bool:
    value = config.get(name)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _append_model(
    entries: list[dict[str, str]],
    *,
    role: str,
    model_name: str | None,
    source: str,
    status: str,
) -> None:
    normalized_model = _normalize_model_name(model_name)
    if normalized_model is None:
        return

    for entry in entries:
        if entry["role"] == role and entry["model_name"] == normalized_model:
            return

    entries.append(
        {
            "role": role,
            "model_name": normalized_model,
            "source": source,
            "status": status,
        }
    )


RuntimeAppender = Callable[
    [list[dict[str, str]], list[str], Mapping[str, Any], Mapping[str, str]],
    None,
]


def _artifact_source(artifact_path: Path, result_dir: Path) -> str:
    try:
        return f"artifact:{artifact_path.relative_to(result_dir)}"
    except ValueError:
        return f"artifact:{artifact_path}"


def _read_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def _normalize_runtime_payload(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None

    models = _normalized_runtime_models(payload)
    notes = _normalized_runtime_notes(payload)

    if not models and not notes:
        return None

    return {
        "schema_version": "1.0",
        "llm_models": models,
        "notes": notes,
    }


def _normalized_runtime_models(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_models = payload.get("llm_models")
    if not isinstance(raw_models, list):
        raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return []

    models: list[dict[str, str]] = []
    for raw_model in raw_models:
        if not isinstance(raw_model, Mapping):
            continue
        _append_model(
            models,
            role=str(raw_model.get("role") or "chat"),
            model_name=_normalize_model_name(raw_model.get("model_name") or raw_model.get("name")),
            source=str(raw_model.get("source") or "recorded"),
            status=str(raw_model.get("status") or "recorded"),
        )
    return models


def _normalized_runtime_notes(payload: Mapping[str, Any]) -> list[str]:
    raw_notes = payload.get("notes")
    if not isinstance(raw_notes, list):
        return []

    notes: list[str] = []
    for raw_note in raw_notes:
        normalized_note = _normalize_note(raw_note)
        if normalized_note and normalized_note not in notes:
            notes.append(normalized_note)
    return notes


def _append_generation_metadata_model(
    entries: list[dict[str, str]],
    *,
    metadata: Mapping[str, Any],
    artifact_path: Path,
    result_dir: Path,
) -> None:
    _append_model(
        entries,
        role="chat",
        model_name=_normalize_model_name(metadata.get("model")),
        source=_artifact_source(artifact_path, result_dir),
        status="recorded",
    )


def _manifest_generation_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    generation_metadata = payload.get("generation_metadata")
    if not isinstance(generation_metadata, Mapping):
        return None
    return generation_metadata


def _response_generation_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    envelope = payload.get("envelope")
    if not isinstance(envelope, Mapping):
        return None
    generation_metadata = envelope.get("generation_metadata")
    if not isinstance(generation_metadata, Mapping):
        return None
    return generation_metadata


def _collect_generation_metadata_models(
    *,
    result_dir: Path,
    glob_pattern: str,
    metadata_getter: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for artifact_path in sorted(result_dir.glob(glob_pattern)):
        payload = _read_json_mapping(artifact_path)
        if payload is None:
            continue
        metadata = metadata_getter(payload)
        if metadata is None:
            continue
        _append_generation_metadata_model(
            entries,
            metadata=metadata,
            artifact_path=artifact_path,
            result_dir=result_dir,
        )
    return entries


def _terminal_bench_llm_shell_model(environ: Mapping[str, str]) -> tuple[str, str]:
    for env_name in (
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "TERMINAL_BENCH_LLM_MODEL",
        "LLM_MODEL",
        "OPENAI_MODEL_NAME",
    ):
        model_name = _normalize_model_name(environ.get(env_name))
        if model_name is not None:
            return model_name, f"environment:{env_name}"

    if _normalize_model_name(environ.get("AZURE_OPENAI_ENDPOINT")):
        return DEFAULT_TERMINAL_BENCH_LLM_MODEL_ID, "Terminal-Bench llm_shell Azure default"

    for env_name in ("TERMINAL_BENCH_LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_BASE_URL"):
        model_name = _model_from_openai_base_url(environ.get(env_name))
        if model_name is not None:
            return model_name, f"environment:{env_name}"

    return DEFAULT_TERMINAL_BENCH_LLM_MODEL_ID, "Terminal-Bench llm_shell default"


def _collect_terminal_bench_runtime_models(result_dir: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for runtime_path in _terminal_bench_runtime_paths(result_dir):
        payload = _read_json_mapping(runtime_path)
        if payload is None:
            continue
        _append_terminal_bench_runtime_payload_models(
            entries,
            payload,
            source=_artifact_source(runtime_path, result_dir),
        )
    return entries


def _terminal_bench_runtime_paths(result_dir: Path) -> list[Path]:
    runtime_globs = (
        ".runtime/llm_shell/runtime.json",
        ".runtime/terminus_2/runtime.json",
    )
    runtime_paths: list[Path] = []
    for pattern in runtime_globs:
        runtime_paths.extend(sorted(result_dir.glob(pattern)))
    return runtime_paths


def _append_terminal_bench_runtime_payload_models(
    entries: list[dict[str, str]],
    payload: Mapping[str, Any],
    *,
    source: str,
) -> None:
    raw_models = payload.get("llm_models")
    if isinstance(raw_models, list):
        _append_terminal_bench_runtime_model_list(entries, raw_models, source=source)
        return

    _append_model(
        entries,
        role="chat",
        model_name=_normalize_model_name(
            payload.get("deployment") or payload.get("model_name") or payload.get("model")
        ),
        source=source,
        status="recorded",
    )


def _append_terminal_bench_runtime_model_list(
    entries: list[dict[str, str]],
    raw_models: list[Any],
    *,
    source: str,
) -> None:
    for raw_model in raw_models:
        if not isinstance(raw_model, Mapping):
            continue
        _append_model(
            entries,
            role=str(raw_model.get("role") or "chat"),
            model_name=_normalize_model_name(
                raw_model.get("model_name") or raw_model.get("deployment")
            ),
            source=source,
            status="recorded",
        )


def _crmarenapro_baseline_model(environ: Mapping[str, str]) -> tuple[str, str]:
    model_name = _normalize_model_name(environ.get("LLM_MODEL"))
    if model_name is not None:
        return model_name, "environment:LLM_MODEL"

    model_name = _normalize_model_name(environ.get("OPENAI_MODEL_NAME"))
    if model_name is not None:
        return model_name, "environment:OPENAI_MODEL_NAME"

    provider = (environ.get("LLM_PROVIDER") or "").strip().lower()
    if provider == "anthropic":
        return DEFAULT_CRMARENAPRO_ANTHROPIC_MODEL_ID, "crmarenapro baseline anthropic default"
    if provider and provider != "anthropic":
        return (
            DEFAULT_CRMARENAPRO_OPENAI_COMPATIBLE_MODEL_ID,
            f"crmarenapro baseline {provider} default",
        )

    if environ.get("ANTHROPIC_API_KEY") or not (
        environ.get("OPENAI_API_KEY") or environ.get("NEBIUS_API_KEY")
    ):
        return DEFAULT_CRMARENAPRO_ANTHROPIC_MODEL_ID, "crmarenapro baseline anthropic default"

    if environ.get("OPENAI_API_KEY"):
        for env_name in ("OPENAI_API_BASE_URL", "OPENAI_BASE_URL"):
            model_name = _model_from_openai_base_url(environ.get(env_name))
            if model_name is not None:
                return model_name, f"environment:{env_name}"
        return DEFAULT_CRMARENAPRO_OPENAI_MODEL_ID, "crmarenapro baseline openai default"

    return (
        DEFAULT_CRMARENAPRO_OPENAI_COMPATIBLE_MODEL_ID,
        "crmarenapro baseline openai-compatible default",
    )


def _append_resolved_chat_model(
    entries: list[dict[str, str]],
    *,
    model_name: str,
    source: str,
) -> None:
    _append_model(
        entries,
        role="chat",
        model_name=model_name,
        source=source,
        status="default" if "default" in source else "configured",
    )


def _append_no_llm_runtime(
    _entries: list[dict[str, str]],
    notes: list[str],
    _request_config: Mapping[str, Any],
    _environ: Mapping[str, str],
) -> None:
    notes.append(NO_LLM_NOTE)


def _append_crmarenapro_baseline_runtime(
    entries: list[dict[str, str]],
    _notes: list[str],
    _request_config: Mapping[str, Any],
    environ: Mapping[str, str],
) -> None:
    model_name, source = _crmarenapro_baseline_model(environ)
    _append_resolved_chat_model(entries, model_name=model_name, source=source)


def _append_terminal_bench_runtime(
    entries: list[dict[str, str]],
    notes: list[str],
    request_config: Mapping[str, Any],
    environ: Mapping[str, str],
    *,
    result_dir: Path | None,
    executor_key: str,
) -> None:
    if _request_config_flag(request_config, "oracle"):
        notes.append("Oracle mode: solve.sh ran locally without the purple LLM.")
        return

    if result_dir is not None and result_dir.exists():
        entries.extend(_collect_terminal_bench_runtime_models(result_dir))
    if entries:
        return

    model_name, source = _terminal_bench_llm_shell_model(environ)
    if executor_key == "terminus_2" and "llm_shell" in source:
        source = source.replace("llm_shell", "terminus_2")
    _append_resolved_chat_model(entries, model_name=model_name, source=source)


RUNTIME_APPENDERS: dict[tuple[str, str], RuntimeAppender] = {
    ("crmarenapro", "baseline_crm_agent"): _append_crmarenapro_baseline_runtime,
}


def _collect_recorded_runtime_models(
    *,
    benchmark_key: str,
    executor_key: str,
    result_dir: Path | None,
) -> list[dict[str, str]]:
    if result_dir is None or not result_dir.exists():
        return []

    del benchmark_key, executor_key
    return []


def resolve_executor_runtime_payload(
    *,
    benchmark_name: str,
    executor_name: str,
    request_config: Mapping[str, Any] | None = None,
    result_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
    recorded_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_recorded = _normalize_runtime_payload(recorded_payload)
    if normalized_recorded is not None:
        return normalized_recorded

    resolved_request_config = request_config if isinstance(request_config, Mapping) else {}
    resolved_environ = environ if isinstance(environ, Mapping) else {}
    benchmark_key = benchmark_name.strip().lower()
    executor_key = executor_name.strip().lower()

    entries = _collect_recorded_runtime_models(
        benchmark_key=benchmark_key,
        executor_key=executor_key,
        result_dir=result_dir,
    )
    notes: list[str] = []

    if benchmark_key in TERMINAL_BENCH_BENCHMARK_KEYS and executor_key in {
        "llm_shell",
        "terminus_2",
    }:
        _append_terminal_bench_runtime(
            entries,
            notes,
            resolved_request_config,
            resolved_environ,
            result_dir=result_dir,
            executor_key=executor_key,
        )
    else:
        appender = RUNTIME_APPENDERS.get((benchmark_key, executor_key))
        if appender is not None:
            appender(entries, notes, resolved_request_config, resolved_environ)

    if not entries and not notes:
        notes.append(UNKNOWN_LLM_NOTE)

    return {
        "schema_version": "1.0",
        "llm_models": entries,
        "notes": notes,
    }


def format_executor_runtime_models(payload: Mapping[str, Any] | None) -> str:
    normalized_payload = _normalize_runtime_payload(payload)
    if normalized_payload is None:
        return "unknown"

    entries = normalized_payload["llm_models"]
    notes = normalized_payload["notes"]
    if not entries:
        if any(note == NO_LLM_NOTE for note in notes):
            return "not used"
        return "unknown"

    parts: list[str] = []
    for entry in entries:
        suffix = ""
        if entry["status"] != "recorded":
            suffix = f" ({entry['status']})"
        parts.append(f"{entry['role']}={entry['model_name']}{suffix}")
    return ", ".join(parts)


def format_executor_runtime_notes(payload: Mapping[str, Any] | None) -> str | None:
    normalized_payload = _normalize_runtime_payload(payload)
    if normalized_payload is None:
        return None

    notes = normalized_payload["notes"]
    if not notes:
        return None
    return "; ".join(notes)
