"""Endpoint capability normalization for pool handoff and scheduling."""

from __future__ import annotations

from collections.abc import Mapping

CAPABILITY_SCHEMA_VERSION = "ril-capability/v1"


def capability_payload(
    *,
    endpoint: Mapping[str, object],
    plan_endpoint: Mapping[str, object] | None = None,
    summary: Mapping[str, object] | None = None,
    model_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    """Return a normalized endpoint capability payload."""

    plan_endpoint = plan_endpoint or {}
    summary = summary or {}
    metadata = _mapping(plan_endpoint.get("metadata"))
    slurm = _mapping(plan_endpoint.get("slurm"))
    diagnostics = _mapping(summary.get("diagnostics"))

    served_model_name = _first_text(
        summary.get("served_model_name"),
        endpoint.get("served_model_name"),
        plan_endpoint.get("served_model_name"),
        summary.get("model"),
        plan_endpoint.get("model"),
    )
    max_model_len = _first_int(summary.get("max_model_len"), metadata.get("max_model_len"))
    max_num_seqs = _first_int(summary.get("max_num_seqs"), metadata.get("max_num_seqs"))
    num_gpus = _first_int(slurm.get("num_gpus"), metadata.get("tensor_parallel_size"))
    nodes = _first_int(slurm.get("nodes"))
    partition = _first_text(
        slurm.get("partition"), _mapping(endpoint.get("slurm")).get("partition")
    )
    kv_cache_tokens = _first_int(diagnostics.get("kv_cache_tokens"))
    vllm_max_concurrency = _first_float(summary.get("vllm_max_concurrency"))
    recommended = _first_int(summary.get("recommended_benchmark_max_parallel"))
    if recommended is None:
        benchmark_handoff = _mapping(summary.get("benchmark_handoff"))
        recommended = _first_int(benchmark_handoff.get("recommended_max_parallel"))
    capacity_excerpt = _first_text(
        diagnostics.get("vllm_capacity_excerpt"),
        summary.get("vllm_capacity_excerpt"),
    )

    sources = _capability_sources(
        summary=summary,
        metadata=metadata,
        slurm=slurm,
        endpoint=endpoint,
        diagnostics=diagnostics,
    )
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "served_model_name": served_model_name,
        "model_ids": list(model_ids),
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "num_gpus": num_gpus,
        "nodes": nodes,
        "partition": partition,
        "kv_cache_tokens": kv_cache_tokens,
        "vllm_max_concurrency": vllm_max_concurrency,
        "recommended_benchmark_max_parallel": recommended,
        "capacity_excerpt": capacity_excerpt,
        "sources": sources,
    }


def capability_matches(
    capability: Mapping[str, object],
    *,
    model: str = "",
    min_context: int | None = None,
) -> tuple[bool, str]:
    """Return whether a capability payload satisfies requested constraints."""

    if model:
        served_model = str(capability.get("served_model_name", "") or "")
        model_ids = capability.get("model_ids", ())
        known_ids = (
            {str(item) for item in model_ids} if isinstance(model_ids, list | tuple) else set()
        )
        if served_model != model and model not in known_ids:
            return False, "model_mismatch"
    if min_context is not None:
        max_context = _first_int(capability.get("max_model_len"))
        if max_context is None or max_context < min_context:
            return False, "capability_mismatch"
    return True, "ok"


def _capability_sources(
    *,
    summary: Mapping[str, object],
    metadata: Mapping[str, object],
    slurm: Mapping[str, object],
    endpoint: Mapping[str, object],
    diagnostics: Mapping[str, object],
) -> dict[str, str]:
    sources: dict[str, str] = {}
    _add_source(
        sources,
        "served_model_name",
        ("summary", summary.get("served_model_name")),
        ("registry", endpoint.get("served_model_name")),
    )
    for field in ("max_model_len", "max_num_seqs"):
        _add_source(sources, field, ("summary", summary.get(field)), ("plan", metadata.get(field)))
    for field in ("num_gpus", "nodes", "partition"):
        _add_source(sources, field, ("plan", slurm.get(field)))
    _add_source(sources, "kv_cache_tokens", ("summary", diagnostics.get("kv_cache_tokens")))
    _add_source(
        sources,
        "vllm_max_concurrency",
        ("summary", summary.get("vllm_max_concurrency")),
    )
    _add_source(
        sources,
        "recommended_benchmark_max_parallel",
        ("summary", summary.get("recommended_benchmark_max_parallel")),
    )
    _add_source(
        sources,
        "capacity_excerpt",
        ("summary", diagnostics.get("vllm_capacity_excerpt")),
    )
    return sources


def _add_source(
    sources: dict[str, str],
    field: str,
    *candidates: tuple[str, object],
) -> None:
    for source, value in candidates:
        if _is_present(value):
            sources[field] = source
            return


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _first_text(*values: object) -> str:
    for value in values:
        if _is_present(value):
            return str(value)
    return ""


def _first_int(*values: object) -> int | None:
    for value in values:
        if not _is_present(value):
            continue
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None


def _first_float(*values: object) -> float | None:
    for value in values:
        if not _is_present(value):
            continue
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None


def _is_present(value: object) -> bool:
    return value is not None and value != ""
