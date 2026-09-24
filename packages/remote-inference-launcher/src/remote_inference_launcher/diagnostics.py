"""Shared startup diagnostics, capacity parsing, and failure classification."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityInfo:
    """Parsed vLLM serving capacity hints."""

    max_model_len: int | None = None
    max_num_seqs: int | None = None
    kv_cache_tokens: int | None = None
    vllm_max_concurrency: float | None = None
    recommended_benchmark_max_parallel: int | None = None
    excerpt: str = ""

    def to_summary(self) -> dict[str, object]:
        """Return summary JSON fields."""

        return {
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "kv_cache_tokens": self.kv_cache_tokens,
            "vllm_max_concurrency": self.vllm_max_concurrency,
            "recommended_benchmark_max_parallel": self.recommended_benchmark_max_parallel,
            "vllm_capacity_excerpt": self.excerpt,
        }


def parse_vllm_capacity(
    text: str,
    *,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
) -> CapacityInfo:
    """Parse known vLLM capacity log lines into benchmark-facing hints."""

    concurrency = _first_float(
        (
            r"Maximum concurrency[^:\n]*:\s*([0-9]+(?:\.[0-9]+)?)x",
            r"max(?:imum)?\s+concurrency[^0-9\n]*([0-9]+(?:\.[0-9]+)?)",
        ),
        text,
    )
    kv_tokens = _first_int(
        (
            r"GPU KV cache size:\s*([0-9,]+)\s*tokens",
            r"KV cache[^:\n]*:\s*([0-9,]+)\s*tokens",
        ),
        text,
    )
    excerpt = _capacity_excerpt(text)
    recommendation = recommended_benchmark_parallel(
        max_num_seqs=max_num_seqs,
        vllm_max_concurrency=concurrency,
    )
    return CapacityInfo(
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        kv_cache_tokens=kv_tokens,
        vllm_max_concurrency=concurrency,
        recommended_benchmark_max_parallel=recommendation,
        excerpt=excerpt,
    )


def recommended_benchmark_parallel(
    *,
    max_num_seqs: int | None = None,
    vllm_max_concurrency: float | None = None,
) -> int | None:
    """Return a conservative benchmark parallelism recommendation."""

    candidates: list[int] = []
    if max_num_seqs is not None and max_num_seqs > 0:
        candidates.append(max_num_seqs)
    if vllm_max_concurrency is not None and vllm_max_concurrency > 0:
        candidates.append(max(1, int(vllm_max_concurrency)))
    if not candidates:
        return None
    return max(1, min(candidates))


def classify_failure(text: str, *, fallback: str | None = None) -> str | None:
    """Classify a startup failure from logs or exception text."""

    lowered = text.lower()
    # Specific runtime and handoff causes must win over generic scheduler wording.
    patterns: tuple[tuple[str, str], ...] = (
        ("local_port_bind_failed", r"address already in use|port .*not available|bind failed"),
        ("vllm_oom", r"out of memory|oom|cuda.*memory|hip.*memory"),
        ("vllm_engine_dead", r"enginedeaderror|engine dead"),
        ("vllm_context_too_large", r"context length|max_model_len|maximum context"),
        (
            "vllm_rocm_device_error",
            r"hsa_status_error_invalid_packet_format|rocr_visible_devices|hip_visible_devices",
        ),
        ("vllm_tool_parser_or_template_error", r"tool.parser|chat template|reasoning parser"),
        ("remote_port_not_written", r"remote.*port.*not.*written|state.*port"),
        ("readiness_smoke_failed", r"chat/completions|smoke"),
        ("readiness_models_failed", r"/models|models endpoint"),
        ("ssh_connection_reset", r"connection reset|kex_exchange_identification"),
        ("ssh_tunnel_bind_failed", r"exitonforwardfailure|forward.*failed|bind.*port"),
        ("slurm_prolog_failure", r"prolog"),
        (
            "slurm_cancelled_or_failed",
            r"\b(cancelled|timeout|out_of_memory|node_fail|boot_fail|preempted)\b|"
            r"\bstate=(failed|cancelled|timeout|out_of_memory|node_fail|boot_fail|preempted)\b",
        ),
    )
    for code, pattern in patterns:
        if re.search(pattern, lowered):
            return code
    return fallback


def sanitized_excerpt(text: str, *, max_chars: int = 2000) -> str:
    """Return a bounded excerpt with obvious secrets redacted."""

    excerpt = text.strip()[-max_chars:]
    replacements = (
        (r"(?i)(authorization:\s*bearer\s+)[^\s]+", r"\1[REDACTED]"),
        (r"(?i)(api[_-]?key[\"'=:\s]+)[^\s,\"']+", r"\1[REDACTED]"),
        (r"(?i)(token[\"'=:\s]+)[^\s,\"']+", r"\1[REDACTED]"),
        (r"hf_[A-Za-z0-9_=-]+", "hf_[REDACTED]"),
        (r"sk-[A-Za-z0-9_=-]+", "sk-[REDACTED]"),
    )
    for pattern, replacement in replacements:
        excerpt = re.sub(pattern, replacement, excerpt)
    return excerpt


def _first_float(patterns: tuple[str, ...], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def _first_int(patterns: tuple[str, ...], text: str) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def _capacity_excerpt(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if "maximum concurrency" in line.lower() or "kv cache" in line.lower():
            lines.append(line.strip())
    return sanitized_excerpt("\n".join(lines), max_chars=1000)
