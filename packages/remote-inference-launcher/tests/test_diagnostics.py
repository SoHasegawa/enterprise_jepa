from __future__ import annotations

from remote_inference_launcher.diagnostics import classify_failure, parse_vllm_capacity


def test_parse_vllm_capacity_recommends_conservative_parallelism() -> None:
    capacity = parse_vllm_capacity(
        "GPU KV cache size: 524,288 tokens\n"
        "Maximum concurrency for 262144 tokens per request: 2.00x\n",
        max_model_len=262144,
        max_num_seqs=4,
    )

    assert capacity.kv_cache_tokens == 524288
    assert capacity.vllm_max_concurrency == 2.0
    assert capacity.recommended_benchmark_max_parallel == 2


def test_classify_common_startup_failures() -> None:
    assert classify_failure("EngineDeadError: worker died") == "vllm_engine_dead"
    assert classify_failure("HSA_STATUS_ERROR_INVALID_PACKET_FORMAT") == "vllm_rocm_device_error"
    assert classify_failure("kex_exchange_identification: read: Connection reset") == (
        "ssh_connection_reset"
    )


def test_specific_vllm_failures_win_over_generic_failed_text() -> None:
    assert classify_failure("vLLM engine failed: CUDA out of memory") == "vllm_oom"
    assert classify_failure("vLLM failed with EngineDeadError") == "vllm_engine_dead"
    assert classify_failure("startup failed: maximum context length exceeds max_model_len") == (
        "vllm_context_too_large"
    )
