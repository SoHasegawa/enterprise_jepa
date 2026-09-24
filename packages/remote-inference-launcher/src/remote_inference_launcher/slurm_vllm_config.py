"""Configuration, validation, and template rendering for Slurm vLLM launches."""

from __future__ import annotations

import math
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from importlib import resources
from string import Template

from remote_inference_launcher.config_types import (
    CandidateRaceConfig,
    DiagnosticsConfig,
    QueuePolicy,
)
from remote_inference_launcher.plan_paths import (
    generated_slurm_job_name,
    join_remote_path,
    model_label,
    safe_endpoint_label,
)
from remote_inference_launcher.readiness import (
    ReadinessConfig,
    validate_readiness_config,
)
from remote_inference_launcher.summaries import new_run_id
from remote_inference_launcher.verbosity import validate_verbosity

DEFAULT_JOB_NAME_PREFIX = "ril"
DEFAULT_REMOTE_OUT_DIR_ROOT = "tmp/remote-inference-launcher/slurm-vllm"
REMOTE_STATE_FILENAME = "remote-inference-state.json"


class SlurmPendingTimeoutError(TimeoutError):
    """Raised when a submitted Slurm job remains pending past its queue timeout."""


@dataclass(frozen=True)
class SlurmVllmConfig:
    """Configuration for one Slurm-backed vLLM server session."""

    name: str = "default"
    ssh_target: str = ""
    verbosity: str = "progress"
    model: str = ""
    served_model_name: str = ""
    remote_port: int | None = None
    local_port: int | None = None
    local_bind_host: str = "127.0.0.1"
    setup_cmd: str = ""
    python_bin: str = "python"
    sbatch_cmd: str = "sbatch"
    out_dir: str = ""
    remote_out_dir_root: str = ""
    runtime_tmp_root: str = ""
    hf_home: str = ""
    job_name: str = ""
    job_name_prefix: str = DEFAULT_JOB_NAME_PREFIX
    partition: str = ""
    walltime: str = ""
    num_gpus: int | None = None
    memory: str = ""
    cpus_per_task: int | None = None
    nodes: int = 1
    exclude: str = ""
    nodelist: str = ""
    target_device: str = ""
    rocm_eager_fallback: bool = True
    distributed_backend: str = ""
    head_node_port: int | None = None
    ray_port: int = 6379
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    data_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    api_key: str = ""
    extra_args: tuple[str, ...] = ()
    check_interval_seconds: int = 10
    queue_timeout_seconds: int = 7200
    ready_timeout_seconds: int = 7200
    keep_remote_job: bool = False
    readiness: ReadinessConfig = field(default_factory=ReadinessConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    queue_policy: QueuePolicy = field(default_factory=QueuePolicy)
    candidate_race: CandidateRaceConfig = field(default_factory=CandidateRaceConfig)
    resource_preferences: tuple[dict[str, object], ...] = ()
    include_base_resource_candidate: bool = False
    launch_summary_path: str = ""
    overwrite_launch_summary: bool = False


@dataclass(frozen=True)
class SlurmJobInfo:
    """Current Slurm scheduler state for the submitted vLLM job."""

    state: str
    node: str = ""
    reason: str = ""


_RESOURCE_PREFERENCE_PARENT_ONLY_FIELDS = {
    "resource_preferences",
    "include_base_resource_candidate",
    "readiness",
    "diagnostics",
    "queue_policy",
    "candidate_race",
    "launch_summary_path",
    "overwrite_launch_summary",
}
_RESOURCE_PREFERENCE_OVERRIDE_FIELDS = {
    field.name for field in dataclass_fields(SlurmVllmConfig)
} - _RESOURCE_PREFERENCE_PARENT_ONLY_FIELDS

_RESOURCE_PREFERENCE_PARENT_IDENTITY_FIELDS = (
    "job_name",
    "out_dir",
    "local_port",
    "remote_port",
    "head_node_port",
)


def with_slurm_vllm_defaults(
    config: SlurmVllmConfig,
    *,
    run_id: str | None = None,
) -> SlurmVllmConfig:
    """Fill derived defaults that depend on other Slurm vLLM config fields."""

    if not config.ssh_target:
        raise ValueError("Slurm vLLM config is missing: ssh_target.")
    if not config.model:
        raise ValueError("Slurm vLLM config is missing: model.")
    _require_scheduler_values(config)
    _validate_resource_preference_identity(config)
    tensor_parallel_size = (
        config.tensor_parallel_size if config.tensor_parallel_size is not None else config.num_gpus
    )
    served_model_name = config.served_model_name or config.model
    head_node_port = (
        config.head_node_port if config.head_node_port is not None else config.remote_port
    )
    job_name_prefix = config.job_name_prefix or DEFAULT_JOB_NAME_PREFIX
    resolved_run_id = run_id or new_run_id()
    job_name = config.job_name or _default_job_name(
        config.name,
        config.model,
        job_name_prefix,
        run_id=resolved_run_id,
    )
    return replace(
        config,
        served_model_name=served_model_name,
        head_node_port=head_node_port,
        job_name=job_name,
        job_name_prefix=job_name_prefix,
        tensor_parallel_size=tensor_parallel_size,
    )


def validate_slurm_vllm_config(config: SlurmVllmConfig) -> SlurmVllmConfig:
    """Reject Slurm vLLM configs that cannot produce a valid server session."""

    validate_verbosity(config.verbosity, label="Slurm vLLM")
    _validate_required_string_fields(config)
    _validate_port_fields(config)
    _validate_positive_int_fields(config)
    _validate_optional_positive_int_fields(config)
    _validate_gpu_memory_utilization(config)
    _validate_extra_args(config)
    _validate_distributed_config(config)
    return config


def _validate_required_string_fields(config: SlurmVllmConfig) -> None:
    for field_name in (
        "name",
        "ssh_target",
        "model",
        "served_model_name",
        "local_bind_host",
        "python_bin",
        "sbatch_cmd",
        "job_name",
        "job_name_prefix",
        "partition",
        "walltime",
        "memory",
    ):
        _require_non_empty_string(config, field_name)
    for field_name in ("distributed_backend", "api_key"):
        if getattr(config, field_name):
            _require_non_empty_string(config, field_name)


def _validate_port_fields(config: SlurmVllmConfig) -> None:
    _require_supported_local_bind_host(config.local_bind_host)
    if config.remote_port is not None:
        _require_port(config.remote_port, "remote_port")
    if config.local_port is not None:
        _require_port(config.local_port, "local_port")
    if config.head_node_port is not None:
        _require_port(config.head_node_port, "head_node_port")


def _validate_positive_int_fields(config: SlurmVllmConfig) -> None:
    for field_name in (
        "num_gpus",
        "cpus_per_task",
        "nodes",
        "ray_port",
        "check_interval_seconds",
        "queue_timeout_seconds",
        "ready_timeout_seconds",
    ):
        _require_positive_int(config, field_name)


def _validate_optional_positive_int_fields(config: SlurmVllmConfig) -> None:
    for field_name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
    ):
        _require_optional_positive_int(config, field_name)


def _validate_gpu_memory_utilization(config: SlurmVllmConfig) -> None:
    if config.gpu_memory_utilization is not None and _invalid_gpu_memory_utilization(
        config.gpu_memory_utilization
    ):
        raise ValueError("Slurm vLLM gpu_memory_utilization must be in (0, 1].")


def _validate_extra_args(config: SlurmVllmConfig) -> None:
    for item in config.extra_args:
        if not isinstance(item, str) or not item:
            raise ValueError("Slurm vLLM extra_args must contain only non-empty strings.")
    seen_candidate_names: set[str] = set()
    for index, item in enumerate(config.resource_preferences):
        if not isinstance(item, dict) or not item:
            raise ValueError("Slurm vLLM resource_preferences must contain non-empty mappings.")
        unknown = sorted(set(item) - _RESOURCE_PREFERENCE_OVERRIDE_FIELDS)
        if unknown:
            raise ValueError(
                f"Slurm vLLM resource preference contains unsupported fields: {', '.join(unknown)}."
            )
        candidate_name = _resource_preference_name(item, index=index)
        if candidate_name in seen_candidate_names:
            raise ValueError(
                f"Slurm vLLM resource_preferences contain duplicate candidate name: "
                f"{candidate_name}."
            )
        seen_candidate_names.add(candidate_name)
    _validate_candidate_race(config.candidate_race)
    _validate_queue_policy(
        config.queue_policy,
        has_resource_preferences=bool(config.resource_preferences),
    )
    try:
        validate_readiness_config(config.readiness)
    except ValueError as error:
        raise ValueError(f"Slurm vLLM {error}") from error


def render_slurm_vllm_sbatch(
    config: SlurmVllmConfig,
    *,
    out_dir: str,
    state_path: str | None = None,
) -> str:
    """Render the Slurm script used to start vLLM on the remote host."""

    config = validate_slurm_vllm_config(with_slurm_vllm_defaults(config))
    extra_args = effective_vllm_extra_args(config)
    template_name = (
        "launch_vllm_ray.sbatch" if config.distributed_backend == "ray" else "launch_vllm.sbatch"
    )
    template = _AtTemplate(_template_text(template_name))
    setup_cmd_block = f"\n{config.setup_cmd}\n" if config.setup_cmd else ""
    return template.substitute(
        sbatch_directives=_sbatch_directives(config, out_dir=out_dir),
        out_dir=shlex.quote(out_dir),
        model=shlex.quote(config.model),
        served_model_name=shlex.quote(config.served_model_name),
        remote_port="" if config.remote_port is None else str(config.remote_port),
        head_node_port="" if config.head_node_port is None else str(config.head_node_port),
        state_path=shlex.quote(state_path or f"{out_dir.rstrip('/')}/{REMOTE_STATE_FILENAME}"),
        ray_port=str(config.ray_port),
        num_gpus=str(config.num_gpus),
        python_bin=shlex.quote(config.python_bin),
        target_device=shlex.quote(config.target_device),
        runtime_tmp_root=shlex.quote(config.runtime_tmp_root),
        hf_home=shlex.quote(config.hf_home),
        api_key_line=_api_key_line(config.api_key),
        extra_args=_bash_array(extra_args),
        vllm_options=_bash_array(_vllm_serve_options(config)),
        setup_cmd_block=setup_cmd_block,
    )


def effective_vllm_extra_args(config: SlurmVllmConfig) -> tuple[str, ...]:
    """Return vLLM extra args after applying explicit package safety additions."""

    args = list(config.extra_args)
    if config.target_device == "rocm" and config.rocm_eager_fallback:
        if "--enforce-eager" not in args:
            args.append("--enforce-eager")
        if not _has_any_option(args, {"--compilation-config", "-cc"}):
            args.extend(["--compilation-config", '{"mode":0,"backend":"eager"}'])
    return tuple(args)


def _require_scheduler_values(config: SlurmVllmConfig) -> None:
    values = {
        "partition": config.partition,
        "walltime": config.walltime,
        "num_gpus": config.num_gpus,
        "memory": config.memory,
        "cpus_per_task": config.cpus_per_task,
    }
    missing = [name for name, value in values.items() if value in {"", None}]
    if missing:
        raise ValueError(
            f"Slurm vLLM scheduler values are missing for {config.ssh_target!r}: "
            f"set {', '.join(missing)}."
        )


def _require_non_empty_string(config: SlurmVllmConfig, field_name: str) -> None:
    value = getattr(config, field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Slurm vLLM {field_name} must be a non-empty string.")


def _require_supported_local_bind_host(local_bind_host: str) -> None:
    if any(character.isspace() for character in local_bind_host):
        raise ValueError("Slurm vLLM local_bind_host must not contain whitespace.")
    if ":" in local_bind_host:
        raise ValueError(
            "Slurm vLLM local_bind_host must be an IPv4 address, hostname, or '*'. "
            "IPv6 bind hosts are not supported."
        )


def _require_port(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError(f"Slurm vLLM {field_name} must be between 1 and 65535.")


def _require_positive_int(config: SlurmVllmConfig, field_name: str) -> None:
    value = getattr(config, field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Slurm vLLM {field_name} must be positive.")


def _require_optional_positive_int(config: SlurmVllmConfig, field_name: str) -> None:
    value = getattr(config, field_name)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Slurm vLLM {field_name} must be positive.")


def _invalid_gpu_memory_utilization(value: object) -> bool:
    return (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or value <= 0
        or value > 1
    )


def _validate_distributed_config(config: SlurmVllmConfig) -> None:
    if config.nodes > 1 and config.distributed_backend != "ray":
        raise ValueError("Slurm vLLM nodes > 1 requires distributed_backend='ray'.")
    if config.nodes == 1 and config.distributed_backend:
        raise ValueError("Slurm vLLM distributed_backend requires nodes > 1.")
    if config.distributed_backend and config.distributed_backend != "ray":
        raise ValueError("Slurm vLLM distributed_backend currently supports only 'ray'.")


def _validate_candidate_race(config: CandidateRaceConfig) -> None:
    if config.max_active_candidates < 1:
        raise ValueError("Slurm vLLM candidate_race.max_active_candidates must be positive.")
    if config.winner_condition != "endpoint_ready":
        raise ValueError("Slurm vLLM candidate_race.winner_condition must be 'endpoint_ready'.")
    if config.launch_stagger_seconds < 0:
        raise ValueError("Slurm vLLM candidate_race.launch_stagger_seconds must be non-negative.")
    if config.enabled and not config.cancel_losers:
        raise ValueError(
            "Slurm vLLM candidate_race.cancel_losers must be true; detached loser jobs are not "
            "supported."
        )


def _validate_queue_policy(
    config: QueuePolicy,
    *,
    has_resource_preferences: bool,
) -> None:
    if config.max_pending_seconds < 0:
        raise ValueError("Slurm vLLM queue_policy.max_pending_seconds must be non-negative.")
    if config.poll_interval_seconds < 1:
        raise ValueError("Slurm vLLM queue_policy.poll_interval_seconds must be positive.")
    if config.status_interval_seconds < 1:
        raise ValueError("Slurm vLLM queue_policy.status_interval_seconds must be positive.")
    if config.fallback_on_pending and config.max_pending_seconds < 1:
        raise ValueError(
            "Slurm vLLM queue_policy.fallback_on_pending requires max_pending_seconds."
        )
    if config.fallback_on_pending and not has_resource_preferences:
        raise ValueError(
            "Slurm vLLM queue_policy.fallback_on_pending requires resource_preferences."
        )


def _validate_resource_preference_identity(config: SlurmVllmConfig) -> None:
    if not config.resource_preferences:
        return
    shared_fields = [
        field_name
        for field_name in _RESOURCE_PREFERENCE_PARENT_IDENTITY_FIELDS
        if getattr(config, field_name) not in {"", None}
    ]
    if shared_fields:
        raise ValueError(
            "Slurm vLLM resource_preferences require per-candidate launch identity; do not "
            "set parent-level "
            f"{', '.join(shared_fields)}. Set unique values inside each resource preference "
            "or let the launcher generate them."
        )


def _vllm_serve_options(config: SlurmVllmConfig) -> tuple[str, ...]:
    options: list[str] = []
    if config.distributed_backend:
        options.extend(["--distributed-executor-backend", config.distributed_backend])
    for name, value in (
        ("--tensor-parallel-size", config.tensor_parallel_size),
        ("--pipeline-parallel-size", config.pipeline_parallel_size),
        ("--data-parallel-size", config.data_parallel_size),
        ("--gpu-memory-utilization", config.gpu_memory_utilization),
        ("--max-model-len", config.max_model_len),
        ("--max-num-seqs", config.max_num_seqs),
        ("--max-num-batched-tokens", config.max_num_batched_tokens),
    ):
        if value is not None:
            options.extend([name, str(value)])
    return tuple(options)


def _sbatch_directives(config: SlurmVllmConfig, *, out_dir: str) -> str:
    lines = [
        f"#SBATCH --job-name={config.job_name}",
        f"#SBATCH --partition={config.partition}",
        f"#SBATCH --nodes={config.nodes}",
        f"#SBATCH --ntasks={config.nodes}",
        f"#SBATCH --mem={config.memory}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task={config.cpus_per_task}",
        f"#SBATCH --gres=gpu:{config.num_gpus}",
        f"#SBATCH --time={config.walltime}",
        f"#SBATCH --output={out_dir.rstrip('/')}/std.log",
        f"#SBATCH --error={out_dir.rstrip('/')}/err.log",
    ]
    if config.exclude:
        lines.append(f"#SBATCH --exclude={config.exclude}")
    if config.nodelist:
        lines.append(f"#SBATCH --nodelist={config.nodelist}")
    return "\n".join(lines)


def _default_remote_out_dir(
    remote_home: str,
    config: SlurmVllmConfig,
    *,
    run_id: str,
) -> str:
    root = config.remote_out_dir_root or f"{remote_home.rstrip('/')}/{DEFAULT_REMOTE_OUT_DIR_ROOT}"
    return join_remote_path(
        root,
        model_label(config.model),
        f"{safe_endpoint_label(config.name)}-{run_id}",
    )


def _api_base_for_bind_host(local_bind_host: str, local_port: int) -> str:
    return f"http://{_api_host_for_bind_host(local_bind_host)}:{local_port}/v1"


def _api_host_for_bind_host(local_bind_host: str) -> str:
    if local_bind_host in {"*", "0.0.0.0"}:
        return "127.0.0.1"
    return local_bind_host


def _expand_remote_home(path: str, remote_home: str) -> str:
    if path == "~":
        return remote_home
    if path.startswith("~/"):
        return f"{remote_home.rstrip('/')}/{path[2:]}"
    return path


def _default_job_name(name: str, model: str, prefix: str, *, run_id: str) -> str:
    del model
    return generated_slurm_job_name(prefix, safe_endpoint_label(name), run_id)


def _bash_array(values: Sequence[str]) -> str:
    return "(" + " ".join(shlex.quote(value) for value in values) + ")"


def _api_key_line(api_key: str) -> str:
    if not api_key:
        return ""
    return f"  --api-key {shlex.quote(api_key)} \\\n"


def _has_any_option(args: Sequence[str], option_names: set[str]) -> bool:
    for arg in args:
        if arg in option_names:
            return True
        if any(arg.startswith(f"{name}=") for name in option_names):
            return True
    return False


def _resource_preference_name(preference: dict[str, object], *, index: int) -> str:
    if "name" not in preference:
        return f"candidate_{index}"
    raw_name = preference["name"]
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("Slurm vLLM resource preference name must be a non-empty string.")
    return raw_name.strip()


def _template_text(filename: str) -> str:
    return (
        resources.files("remote_inference_launcher.templates")
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )


class _AtTemplate(Template):
    delimiter = "@"
