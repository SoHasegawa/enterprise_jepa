from __future__ import annotations

import json
from pathlib import Path

from common.executor_runtime import resolve_executor_runtime_payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_resolve_executor_runtime_payload_reports_crmarenapro_configured_model() -> None:
    payload = resolve_executor_runtime_payload(
        benchmark_name="crmarenapro",
        executor_name="baseline_crm_agent",
        environ={"LLM_MODEL": "custom-crm-model"},
    )

    assert payload["llm_models"] == [
        {
            "role": "chat",
            "model_name": "custom-crm-model",
            "source": "environment:LLM_MODEL",
            "status": "configured",
        }
    ]
    assert payload["notes"] == []


def test_resolve_executor_runtime_payload_reports_crmarenapro_default_model() -> None:
    payload = resolve_executor_runtime_payload(
        benchmark_name="crmarenapro",
        executor_name="baseline_crm_agent",
        environ={},
    )

    assert payload["llm_models"] == [
        {
            "role": "chat",
            "model_name": "claude-3-5-sonnet-latest",
            "source": "crmarenapro baseline anthropic default",
            "status": "default",
        }
    ]
    assert payload["notes"] == []


def test_resolve_executor_runtime_payload_reports_terminal_bench_terminus_2_model() -> None:
    payload = resolve_executor_runtime_payload(
        benchmark_name="Terminal-Bench-2.0",
        executor_name="terminus_2",
        environ={
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-5.5",
        },
    )

    assert payload["llm_models"] == [
        {
            "role": "chat",
            "model_name": "gpt-5.5",
            "source": "environment:AZURE_OPENAI_DEPLOYMENT_NAME",
            "status": "configured",
        }
    ]
    assert payload["notes"] == []


def test_resolve_executor_runtime_payload_reports_terminal_bench_llm_shell_model() -> None:
    payload = resolve_executor_runtime_payload(
        benchmark_name="Terminal-Bench-2.0",
        executor_name="llm_shell",
        environ={
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-5.5",
        },
    )

    assert payload["llm_models"] == [
        {
            "role": "chat",
            "model_name": "gpt-5.5",
            "source": "environment:AZURE_OPENAI_DEPLOYMENT_NAME",
            "status": "configured",
        }
    ]
    assert payload["notes"] == []


def test_resolve_executor_runtime_payload_does_not_report_unknown_executor_env_fallback() -> (
    None
):
    payload = resolve_executor_runtime_payload(
        benchmark_name="Terminal-Bench-2.0",
        executor_name="some_other_executor",
        environ={
            "TERMINAL_BENCH_SOME_OTHER_EXECUTOR_LLM_MODEL": "gpt-5.4-mini",
        },
    )

    assert payload["llm_models"] == []
    assert payload["notes"] == ["Could not determine the LLM model name from saved artifacts."]


def test_resolve_executor_runtime_payload_reads_terminal_bench_runtime_artifact(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "result"
    runtime_path = result_dir / ".runtime" / "llm_shell" / "runtime.json"
    _write_json(
        runtime_path,
        {
            "schema_version": "1.0",
            "deployment": "gpt-5.5",
            "llm_models": [{"role": "chat", "model_name": "gpt-5.5"}],
        },
    )

    payload = resolve_executor_runtime_payload(
        benchmark_name="Terminal-Bench-2.0",
        executor_name="llm_shell",
        result_dir=result_dir,
        environ={},
    )

    assert payload["llm_models"] == [
        {
            "role": "chat",
            "model_name": "gpt-5.5",
            "source": "artifact:.runtime/llm_shell/runtime.json",
            "status": "recorded",
        }
    ]
    assert payload["notes"] == []


def test_resolve_executor_runtime_payload_reads_terminal_bench_runtime_fallback_model(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "result"
    runtime_path = result_dir / ".runtime" / "terminus_2" / "runtime.json"
    _write_json(runtime_path, {"schema_version": "1.0", "model": "gpt-5.6"})

    payload = resolve_executor_runtime_payload(
        benchmark_name="Terminal-Bench-2.0",
        executor_name="terminus_2",
        result_dir=result_dir,
        environ={},
    )

    assert payload["llm_models"] == [
        {
            "role": "chat",
            "model_name": "gpt-5.6",
            "source": "artifact:.runtime/terminus_2/runtime.json",
            "status": "recorded",
        }
    ]
    assert payload["notes"] == []


def test_resolve_executor_runtime_payload_reports_crmarenapro_openai_base_url_model() -> None:
    payload = resolve_executor_runtime_payload(
        benchmark_name="crmarenapro",
        executor_name="baseline_crm_agent",
        environ={
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_BASE_URL": "https://example.invalid/v1/gpt/gpt-5.1",
        },
    )

    assert payload["llm_models"] == [
        {
            "role": "chat",
            "model_name": "gpt-5.1",
            "source": "environment:OPENAI_API_BASE_URL",
            "status": "configured",
        }
    ]
    assert payload["notes"] == []


def test_resolve_executor_runtime_payload_normalizes_recorded_payload() -> None:
    payload = resolve_executor_runtime_payload(
        benchmark_name="unused",
        executor_name="unused",
        recorded_payload={
            "models": [
                {"name": "gpt-5.1"},
                {"name": "gpt-5.1"},
                {"role": "embedding", "model_name": "text-embedding-3-large"},
                "ignored",
            ],
            "notes": [" recorded ", "recorded", "", 42],
        },
    )

    assert payload == {
        "schema_version": "1.0",
        "llm_models": [
            {
                "role": "chat",
                "model_name": "gpt-5.1",
                "source": "recorded",
                "status": "recorded",
            },
            {
                "role": "embedding",
                "model_name": "text-embedding-3-large",
                "source": "recorded",
                "status": "recorded",
            },
        ],
        "notes": ["recorded"],
    }
