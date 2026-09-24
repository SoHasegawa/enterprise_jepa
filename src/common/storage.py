from __future__ import annotations

import os
from pathlib import Path

BENCHMARK_HOME_ENV = "BENCHMARK_HOME"
# Root for benchmark results and any shared model/container artefacts. Override with
# $BENCHMARK_HOME (a shared filesystem on a cluster); otherwise results land in the
# repository's own results/ directory.
DEFAULT_SHARED_STORAGE_ROOT = Path(__file__).resolve().parents[2] / "results"
DEFAULT_VLLM_MODEL_ID = "Qwen/Qwen3.5-27B"
DEFAULT_VLLM_OPENAI_SIF_FILENAMES = {
    "Qwen/Qwen3.5-27B": "vllm-openai-qwen3.5-27b.sif",
    "Qwen/Qwen3.6-35B-A3B": "vllm-openai-qwen3.6-35b-a3b.sif",
    "google/gemma-4-31B": "vllm-openai-gemma-4-31b.sif",
}


def resolve_shared_storage_root(explicit_root: Path | None = None) -> Path:
    """ベンチマーク共通ストレージのルートを解決する。"""
    if explicit_root is not None:
        return explicit_root.expanduser().resolve()

    env_value = os.getenv(BENCHMARK_HOME_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()

    return DEFAULT_SHARED_STORAGE_ROOT


def default_result_root(shared_storage_root: Path | None = None) -> Path:
    """共通ストレージ上の既定結果保存先を返す。"""
    return resolve_shared_storage_root(shared_storage_root) / "experiments"


def vllm_model_slug(model_id: str) -> str:
    """MODEL_ID を shared storage 上の vLLM 用 slug に変換する。"""
    normalized = "".join(
        char.lower() if char.isascii() and char.isalnum() else "-" for char in model_id.strip()
    )
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized.strip("-") or "model"


def is_qwen3_model(model_id: str) -> bool:
    """Qwen3 系 model かどうかを返す。"""
    return model_id.strip().lower().startswith("qwen/qwen3")


def is_gemma4_model(model_id: str) -> bool:
    """Gemma 4 系 model かどうかを返す。"""
    return model_id.strip().lower().startswith("google/gemma-4")


def default_vllm_reasoning_parser(model_id: str) -> str | None:
    """model ごとの既定 reasoning parser を返す。"""
    if is_qwen3_model(model_id):
        return "qwen3"
    if is_gemma4_model(model_id):
        return "gemma4"
    return None


def default_vllm_tool_call_parser(model_id: str) -> str | None:
    """model ごとの既定 tool-call parser を返す。"""
    if is_qwen3_model(model_id):
        return "qwen3_xml"
    if is_gemma4_model(model_id):
        return "gemma4"
    return None


def default_vllm_dtype(model_id: str) -> str | None:
    """model ごとの既定 dtype を返す。"""
    if is_gemma4_model(model_id):
        return "bfloat16"
    return None


def default_vllm_trust_remote_code(model_id: str) -> bool | None:
    """model ごとの trust-remote-code 既定値を返す。"""
    if is_gemma4_model(model_id):
        return True
    return None


def default_vllm_model_dir(
    model_id: str = DEFAULT_VLLM_MODEL_ID,
    shared_storage_root: Path | None = None,
) -> Path:
    """共通ストレージ上の vLLM model directory 既定パスを返す。"""
    return resolve_shared_storage_root(shared_storage_root) / "models" / Path(model_id)


def default_vllm_openai_sif_path(
    model_id: str = DEFAULT_VLLM_MODEL_ID,
    shared_storage_root: Path | None = None,
) -> Path:
    """共通ストレージ上の vLLM OpenAI-compatible SIF 既定パスを返す。"""
    filename = DEFAULT_VLLM_OPENAI_SIF_FILENAMES.get(
        model_id,
        f"vllm-openai-{vllm_model_slug(model_id)}.sif",
    )
    return resolve_shared_storage_root(shared_storage_root) / "sifs" / filename
