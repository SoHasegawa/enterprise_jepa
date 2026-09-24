"""Canonical effective launch plans for remote inference endpoints."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from remote_inference_launcher.endpoint_race import EndpointRaceConfig
from remote_inference_launcher.existing_endpoint import ExistingEndpointConfig
from remote_inference_launcher.fleet import FleetConfig
from remote_inference_launcher.inference_config import inference_config_from_mapping
from remote_inference_launcher.local_vllm import (
    DEFAULT_OUT_DIR_ROOT,
    LocalVllmConfig,
    with_local_vllm_defaults,
)
from remote_inference_launcher.plan_paths import (
    EffectivePath,
)
from remote_inference_launcher.plan_paths import (
    generated_slurm_job_name as _generated_slurm_job_name,
)
from remote_inference_launcher.plan_paths import (
    join_remote_path as _join_remote_path,
)
from remote_inference_launcher.plan_paths import (
    model_label as _model_label,
)
from remote_inference_launcher.plan_paths import (
    normalize_local_path as _normalize_local_path,
)
from remote_inference_launcher.plan_paths import (
    optional_local_path as _optional_local_path,
)
from remote_inference_launcher.plan_paths import (
    optional_remote_path as _optional_remote_path,
)
from remote_inference_launcher.plan_paths import (
    remote_out_dir_path as _remote_out_dir_path,
)
from remote_inference_launcher.plan_paths import (
    unique_endpoint_labels as _unique_endpoint_labels,
)
from remote_inference_launcher.plan_paths import (
    without_empty_paths as _without_empty_paths,
)
from remote_inference_launcher.readiness import ReadinessConfig
from remote_inference_launcher.slurm_vllm import (
    DEFAULT_REMOTE_OUT_DIR_ROOT as DEFAULT_SLURM_REMOTE_OUT_DIR_ROOT,
)
from remote_inference_launcher.slurm_vllm import (
    REMOTE_STATE_FILENAME as SLURM_REMOTE_STATE_FILENAME,
)
from remote_inference_launcher.slurm_vllm import (
    SlurmVllmConfig,
    slurm_resource_preference_candidates,
    with_slurm_vllm_defaults,
)
from remote_inference_launcher.ssh_vllm import (
    DEFAULT_REMOTE_OUT_DIR_ROOT as DEFAULT_SSH_REMOTE_OUT_DIR_ROOT,
)
from remote_inference_launcher.ssh_vllm import (
    REMOTE_STATE_FILENAME as SSH_REMOTE_STATE_FILENAME,
)
from remote_inference_launcher.ssh_vllm import SshVllmConfig, with_ssh_vllm_defaults
from remote_inference_launcher.summaries import new_run_id
from remote_inference_launcher.yaml_config import load_yaml_mapping

PLAN_SCHEMA_VERSION = "ril-plan/v1"
DEFAULT_REGISTRY_ROOT = ".remote-inference-launcher/runs"
RemoteHomeResolver = Callable[[str], str]


@dataclass(frozen=True)
class EffectiveEndpointPlan:
    """Canonical launch facts for one concrete endpoint."""

    name: str
    endpoint_label: str
    kind: str
    backend_kind: str
    endpoint_index: int
    raw_config_source: str
    model: str
    served_model_name: str
    api_key_set: bool
    local_bind_host: str
    local_port: int | None
    local_port_strategy: str
    remote_port: int | None
    remote_port_strategy: str
    ssh_target: str
    ownership_policy: str
    keep_remote_job: bool
    job_name: str
    out_dir: str
    remote_state_path: str
    local_log_dir: Path
    summary_path: Path
    cleanup_command: str
    paths: dict[str, EffectivePath]
    slurm: dict[str, object]
    readiness: dict[str, object]
    metadata: dict[str, object]

    @property
    def source(self) -> str:
        """Compatibility label for validation callers."""

        return self.raw_config_source

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        payload = dataclasses.asdict(self)
        return _jsonable(payload)


@dataclass(frozen=True)
class EffectiveLaunchPlan:
    """Canonical launch plan for one top-level command invocation."""

    schema_version: str
    run_id: str
    created_at: str
    source_config_path: str
    raw_config_hash: str
    semantic_config_hash: str
    instance_hash: str
    registry_dir: Path
    summary_path: Path
    controller_policy: str
    ownership_policy: str
    endpoints: tuple[EffectiveEndpointPlan, ...]
    resource_budget: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return _launch_plan_payload(self, include_instance_hash=True)


@dataclass(frozen=True)
class _EndpointInput:
    config: object
    source: str


def build_effective_launch_plan_from_path(
    path: str | Path,
    *,
    cli_overrides: Mapping[str, object] | None = None,
    run_id: str | None = None,
    created_at: str | None = None,
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
    controller_policy: str = "foreground",
    ownership_policy: str = "owned",
    remote_home_resolver: RemoteHomeResolver | None = None,
    reserve_ports: bool = False,
) -> EffectiveLaunchPlan:
    """Load a YAML config and build its canonical effective plan."""

    raw_config = load_yaml_mapping(path, kind="inference config")
    source = str(Path(path))
    config = inference_config_from_mapping(raw_config, source=source)
    return build_effective_launch_plan(
        config,
        source_config_path=source,
        raw_config=raw_config,
        cli_overrides=cli_overrides,
        run_id=run_id,
        created_at=created_at,
        registry_root=registry_root,
        controller_policy=controller_policy,
        ownership_policy=ownership_policy,
        remote_home_resolver=remote_home_resolver,
        reserve_ports=reserve_ports,
    )


def build_effective_launch_plan(
    config: object,
    *,
    source_config_path: str,
    raw_config: Mapping[str, object] | None = None,
    cli_overrides: Mapping[str, object] | None = None,
    run_id: str | None = None,
    created_at: str | None = None,
    registry_root: str | Path = DEFAULT_REGISTRY_ROOT,
    controller_policy: str = "foreground",
    ownership_policy: str = "owned",
    remote_home_resolver: RemoteHomeResolver | None = None,
    reserve_ports: bool = False,
) -> EffectiveLaunchPlan:
    """Build an immutable effective plan before launch side effects."""

    resolved_run_id = run_id or new_run_id()
    resolved_created_at = created_at or _utc_now()
    registry_dir = Path(registry_root).expanduser() / resolved_run_id
    endpoint_inputs = _endpoint_inputs(config, source=source_config_path)
    endpoint_labels = _unique_endpoint_labels(
        [str(getattr(endpoint.config, "name", "default")) for endpoint in endpoint_inputs]
    )
    remote_home_cache: dict[str, str] = {}
    endpoints = tuple(
        _build_endpoint_plan(
            endpoint,
            endpoint_index=index,
            endpoint_label=endpoint_labels[index],
            run_id=resolved_run_id,
            registry_dir=registry_dir,
            ownership_policy=_endpoint_ownership_policy(endpoint.config, ownership_policy),
            remote_home_cache=remote_home_cache,
            remote_home_resolver=remote_home_resolver,
            reserve_ports=reserve_ports,
        )
        for index, endpoint in enumerate(endpoint_inputs)
    )
    raw_hash_payload = {
        "config": raw_config if raw_config is not None else _jsonable_config(config),
        "cli_overrides": dict(cli_overrides or {}),
    }
    resource_budget = _resource_budget_payload(config)
    semantic_hash = _hash_payload(
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "ownership_policy": ownership_policy,
            "resource_budget": resource_budget,
            "endpoints": [_semantic_endpoint_payload(endpoint) for endpoint in endpoints],
        }
    )
    plan = EffectiveLaunchPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        run_id=resolved_run_id,
        created_at=resolved_created_at,
        source_config_path=source_config_path,
        raw_config_hash=_hash_payload(raw_hash_payload),
        semantic_config_hash=semantic_hash,
        instance_hash="",
        registry_dir=registry_dir,
        summary_path=registry_dir / "summary.json",
        controller_policy=controller_policy,
        ownership_policy=ownership_policy,
        endpoints=endpoints,
        resource_budget=resource_budget,
    )
    return replace(plan, instance_hash=_hash_payload(_launch_plan_payload(plan)))


def _endpoint_inputs(config: object, *, source: str) -> tuple[_EndpointInput, ...]:
    if isinstance(config, FleetConfig):
        endpoints: list[_EndpointInput] = []
        for name, endpoint in config.endpoints.items():
            endpoints.extend(_endpoint_inputs(endpoint, source=f"{source} endpoint {name!r}"))
        return tuple(endpoints)
    if isinstance(config, EndpointRaceConfig):
        endpoints = []
        for name, candidate in config.candidates.items():
            endpoints.extend(_endpoint_inputs(candidate, source=f"{source} candidate {name!r}"))
        return tuple(endpoints)
    if isinstance(config, SlurmVllmConfig) and config.resource_preferences:
        return tuple(
            _EndpointInput(candidate, f"{source} resource {name!r}")
            for name, candidate in slurm_resource_preference_candidates(config)
        )
    if isinstance(
        config, ExistingEndpointConfig | LocalVllmConfig | SshVllmConfig | SlurmVllmConfig
    ):
        return (_EndpointInput(config, source),)
    raise TypeError(f"Unsupported inference config type: {type(config).__name__}")


def _build_endpoint_plan(
    endpoint: _EndpointInput,
    *,
    endpoint_index: int,
    endpoint_label: str,
    run_id: str,
    registry_dir: Path,
    ownership_policy: str,
    remote_home_cache: dict[str, str],
    remote_home_resolver: RemoteHomeResolver | None,
    reserve_ports: bool,
) -> EffectiveEndpointPlan:
    config = endpoint.config
    if isinstance(config, ExistingEndpointConfig):
        return _existing_endpoint_plan(
            config,
            source=endpoint.source,
            endpoint_index=endpoint_index,
            endpoint_label=endpoint_label,
            registry_dir=registry_dir,
            ownership_policy=ownership_policy,
        )
    if isinstance(config, LocalVllmConfig):
        return _local_vllm_endpoint_plan(
            config,
            source=endpoint.source,
            endpoint_index=endpoint_index,
            endpoint_label=endpoint_label,
            run_id=run_id,
            registry_dir=registry_dir,
            ownership_policy=ownership_policy,
            reserve_ports=reserve_ports,
        )
    if isinstance(config, SshVllmConfig):
        return _ssh_vllm_endpoint_plan(
            config,
            source=endpoint.source,
            endpoint_index=endpoint_index,
            endpoint_label=endpoint_label,
            run_id=run_id,
            registry_dir=registry_dir,
            ownership_policy=ownership_policy,
            remote_home_cache=remote_home_cache,
            remote_home_resolver=remote_home_resolver,
            reserve_ports=reserve_ports,
        )
    if isinstance(config, SlurmVllmConfig):
        return _slurm_vllm_endpoint_plan(
            config,
            source=endpoint.source,
            endpoint_index=endpoint_index,
            endpoint_label=endpoint_label,
            run_id=run_id,
            registry_dir=registry_dir,
            ownership_policy=ownership_policy,
            remote_home_cache=remote_home_cache,
            remote_home_resolver=remote_home_resolver,
            reserve_ports=reserve_ports,
        )
    raise TypeError(f"Unsupported inference config type: {type(config).__name__}")


def _existing_endpoint_plan(
    config: ExistingEndpointConfig,
    *,
    source: str,
    endpoint_index: int,
    endpoint_label: str,
    registry_dir: Path,
    ownership_policy: str,
) -> EffectiveEndpointPlan:
    summary_path = _endpoint_summary_path(registry_dir, endpoint_label)
    served_model_name = config.served_model_name or config.model
    return EffectiveEndpointPlan(
        name=config.name,
        endpoint_label=endpoint_label,
        kind="existing_endpoint",
        backend_kind="existing_endpoint",
        endpoint_index=endpoint_index,
        raw_config_source=source,
        model=config.model or served_model_name,
        served_model_name=served_model_name,
        api_key_set=bool(config.api_key),
        local_bind_host="",
        local_port=None,
        local_port_strategy="external",
        remote_port=None,
        remote_port_strategy="external",
        ssh_target="",
        ownership_policy=ownership_policy,
        keep_remote_job=False,
        job_name="",
        out_dir="",
        remote_state_path="",
        local_log_dir=summary_path.parent,
        summary_path=summary_path,
        cleanup_command="",
        paths={},
        slurm={},
        readiness=_readiness_payload(config.readiness),
        metadata={
            "api_base": config.api_base.rstrip("/"),
            "discover_model": config.discover_model,
            "job_name_strategy": "external",
            "out_dir_strategy": "external",
        },
    )


def _local_vllm_endpoint_plan(
    config: LocalVllmConfig,
    *,
    source: str,
    endpoint_index: int,
    endpoint_label: str,
    run_id: str,
    registry_dir: Path,
    ownership_policy: str,
    reserve_ports: bool,
) -> EffectiveEndpointPlan:
    resolved = with_local_vllm_defaults(config)
    summary_path = _endpoint_summary_path(registry_dir, endpoint_label)
    local_port, local_port_strategy = _local_port_value(config.port, reserve_ports=reserve_ports)
    out_dir_raw = config.out_dir or str(
        Path.home()
        / DEFAULT_OUT_DIR_ROOT
        / _model_label(resolved.model)
        / f"{endpoint_label}-{run_id}"
    )
    out_dir_path = _normalize_local_path(
        "out_dir",
        out_dir_raw,
        required=False,
        writable=True,
        generated=not bool(config.out_dir),
    )
    paths = _without_empty_paths(
        {
            "python_bin": _optional_local_path(
                "python_bin",
                resolved.python_bin,
                required=True,
                writable=False,
            ),
            "model": _optional_local_path(
                "model",
                resolved.model,
                required=True,
                writable=False,
            ),
            "hf_home": _optional_local_path(
                "hf_home",
                resolved.hf_home,
                required=False,
                writable=True,
            ),
            "runtime_tmp_root": _optional_local_path(
                "runtime_tmp_root",
                resolved.runtime_tmp_root,
                required=False,
                writable=True,
            ),
            "out_dir": out_dir_path,
        }
    )
    return EffectiveEndpointPlan(
        name=resolved.name,
        endpoint_label=endpoint_label,
        kind="local_vllm",
        backend_kind="local_vllm",
        endpoint_index=endpoint_index,
        raw_config_source=source,
        model=resolved.model,
        served_model_name=resolved.served_model_name,
        api_key_set=bool(resolved.api_key),
        local_bind_host=resolved.host,
        local_port=local_port,
        local_port_strategy=local_port_strategy,
        remote_port=None,
        remote_port_strategy="external",
        ssh_target="",
        ownership_policy=ownership_policy,
        keep_remote_job=resolved.keep_server,
        job_name="",
        out_dir=out_dir_path.resolved,
        remote_state_path="",
        local_log_dir=Path(out_dir_path.resolved),
        summary_path=summary_path,
        cleanup_command=f"remote-inference-launcher stop {registry_dir}",
        paths=paths,
        slurm={},
        readiness=_readiness_payload(resolved.readiness),
        metadata={
            "job_name_strategy": "external",
            "out_dir_strategy": "explicit" if config.out_dir else "generated",
            "target_device": resolved.target_device,
            "tensor_parallel_size": resolved.tensor_parallel_size,
            "pipeline_parallel_size": resolved.pipeline_parallel_size,
            "data_parallel_size": resolved.data_parallel_size,
            "gpu_memory_utilization": resolved.gpu_memory_utilization,
            "max_model_len": resolved.max_model_len,
            "max_num_seqs": resolved.max_num_seqs,
            "max_num_batched_tokens": resolved.max_num_batched_tokens,
            "extra_args": tuple(resolved.extra_args),
        },
    )


def _ssh_vllm_endpoint_plan(
    config: SshVllmConfig,
    *,
    source: str,
    endpoint_index: int,
    endpoint_label: str,
    run_id: str,
    registry_dir: Path,
    ownership_policy: str,
    remote_home_cache: dict[str, str],
    remote_home_resolver: RemoteHomeResolver | None,
    reserve_ports: bool,
) -> EffectiveEndpointPlan:
    resolved = with_ssh_vllm_defaults(config)
    summary_path = _endpoint_summary_path(registry_dir, endpoint_label)
    local_port, local_port_strategy = _local_port_value(
        config.local_port,
        reserve_ports=reserve_ports,
    )
    remote_port, remote_port_strategy = _generated_int_strategy(config.remote_port)
    remote_home = _remote_home(
        resolved.ssh_target,
        remote_home_cache=remote_home_cache,
        remote_home_resolver=remote_home_resolver,
        require_now=False,
    )
    out_dir_path = _remote_out_dir_path(
        label="out_dir",
        configured_out_dir=config.out_dir,
        configured_root=config.remote_out_dir_root,
        default_root=DEFAULT_SSH_REMOTE_OUT_DIR_ROOT,
        model=resolved.model,
        endpoint_label=endpoint_label,
        run_id=run_id,
        remote_home=remote_home,
    )
    remote_state_path = _join_remote_path(out_dir_path.resolved, SSH_REMOTE_STATE_FILENAME)
    paths = _remote_vllm_paths(
        resolved,
        remote_home=remote_home,
        out_dir_path=out_dir_path,
    )
    return EffectiveEndpointPlan(
        name=resolved.name,
        endpoint_label=endpoint_label,
        kind="ssh_vllm",
        backend_kind="ssh_vllm",
        endpoint_index=endpoint_index,
        raw_config_source=source,
        model=resolved.model,
        served_model_name=resolved.served_model_name,
        api_key_set=bool(resolved.api_key),
        local_bind_host=resolved.local_bind_host,
        local_port=local_port,
        local_port_strategy=local_port_strategy,
        remote_port=remote_port,
        remote_port_strategy=remote_port_strategy,
        ssh_target=resolved.ssh_target,
        ownership_policy=ownership_policy,
        keep_remote_job=resolved.keep_remote_process,
        job_name="",
        out_dir=out_dir_path.resolved,
        remote_state_path=remote_state_path,
        local_log_dir=summary_path.parent,
        summary_path=summary_path,
        cleanup_command=f"remote-inference-launcher stop {registry_dir}",
        paths=paths,
        slurm={},
        readiness=_readiness_payload(resolved.readiness),
        metadata={
            "job_name_strategy": "external",
            "out_dir_strategy": "explicit" if config.out_dir else "generated",
            "remote_out_dir_root_strategy": "explicit"
            if config.remote_out_dir_root
            else "generated",
            "setup_cmd": resolved.setup_cmd,
            "target_device": resolved.target_device,
            "tensor_parallel_size": resolved.tensor_parallel_size,
            "pipeline_parallel_size": resolved.pipeline_parallel_size,
            "data_parallel_size": resolved.data_parallel_size,
            "gpu_memory_utilization": resolved.gpu_memory_utilization,
            "max_model_len": resolved.max_model_len,
            "max_num_seqs": resolved.max_num_seqs,
            "max_num_batched_tokens": resolved.max_num_batched_tokens,
            "extra_args": tuple(resolved.extra_args),
        },
    )


def _slurm_vllm_endpoint_plan(
    config: SlurmVllmConfig,
    *,
    source: str,
    endpoint_index: int,
    endpoint_label: str,
    run_id: str,
    registry_dir: Path,
    ownership_policy: str,
    remote_home_cache: dict[str, str],
    remote_home_resolver: RemoteHomeResolver | None,
    reserve_ports: bool,
) -> EffectiveEndpointPlan:
    resolved_defaults = with_slurm_vllm_defaults(config, run_id=run_id)
    job_name = config.job_name or _generated_slurm_job_name(
        resolved_defaults.job_name_prefix,
        endpoint_label,
        run_id,
    )
    resolved = replace(resolved_defaults, job_name=job_name)
    summary_path = _endpoint_summary_path(registry_dir, endpoint_label)
    local_port, local_port_strategy = _local_port_value(
        config.local_port,
        reserve_ports=reserve_ports,
    )
    remote_port, remote_port_strategy = _generated_int_strategy(config.remote_port)
    remote_home = _remote_home(
        resolved.ssh_target,
        remote_home_cache=remote_home_cache,
        remote_home_resolver=remote_home_resolver,
        require_now=False,
    )
    out_dir_path = _remote_out_dir_path(
        label="out_dir",
        configured_out_dir=config.out_dir,
        configured_root=config.remote_out_dir_root,
        default_root=DEFAULT_SLURM_REMOTE_OUT_DIR_ROOT,
        model=resolved.model,
        endpoint_label=endpoint_label,
        run_id=run_id,
        remote_home=remote_home,
    )
    remote_state_path = _join_remote_path(out_dir_path.resolved, SLURM_REMOTE_STATE_FILENAME)
    paths = _remote_vllm_paths(
        resolved,
        remote_home=remote_home,
        out_dir_path=out_dir_path,
    )
    return EffectiveEndpointPlan(
        name=resolved.name,
        endpoint_label=endpoint_label,
        kind="slurm_vllm",
        backend_kind="slurm_vllm",
        endpoint_index=endpoint_index,
        raw_config_source=source,
        model=resolved.model,
        served_model_name=resolved.served_model_name,
        api_key_set=bool(resolved.api_key),
        local_bind_host=resolved.local_bind_host,
        local_port=local_port,
        local_port_strategy=local_port_strategy,
        remote_port=remote_port,
        remote_port_strategy=remote_port_strategy,
        ssh_target=resolved.ssh_target,
        ownership_policy=ownership_policy,
        keep_remote_job=resolved.keep_remote_job,
        job_name=job_name,
        out_dir=out_dir_path.resolved,
        remote_state_path=remote_state_path,
        local_log_dir=summary_path.parent,
        summary_path=summary_path,
        cleanup_command=f"remote-inference-launcher stop {registry_dir}",
        paths=paths,
        slurm=_slurm_payload(resolved),
        readiness=_readiness_payload(resolved.readiness),
        metadata={
            "job_name_strategy": "explicit" if config.job_name else "generated",
            "out_dir_strategy": "explicit" if config.out_dir else "generated",
            "remote_out_dir_root_strategy": "explicit"
            if config.remote_out_dir_root
            else "generated",
            "setup_cmd": resolved.setup_cmd,
            "target_device": resolved.target_device,
            "tensor_parallel_size": resolved.tensor_parallel_size,
            "pipeline_parallel_size": resolved.pipeline_parallel_size,
            "data_parallel_size": resolved.data_parallel_size,
            "gpu_memory_utilization": resolved.gpu_memory_utilization,
            "max_model_len": resolved.max_model_len,
            "max_num_seqs": resolved.max_num_seqs,
            "max_num_batched_tokens": resolved.max_num_batched_tokens,
            "extra_args": tuple(resolved.extra_args),
        },
    )


def _remote_vllm_paths(
    config: SshVllmConfig | SlurmVllmConfig,
    *,
    remote_home: str,
    out_dir_path: EffectivePath,
) -> dict[str, EffectivePath]:
    return _without_empty_paths(
        {
            "python_bin": _optional_remote_path(
                "python_bin",
                config.python_bin,
                required=True,
                writable=False,
                remote_home=remote_home,
            ),
            "model": _optional_remote_path(
                "model",
                config.model,
                required=True,
                writable=False,
                remote_home=remote_home,
            ),
            "hf_home": _optional_remote_path(
                "hf_home",
                config.hf_home,
                required=False,
                writable=True,
                remote_home=remote_home,
            ),
            "runtime_tmp_root": _optional_remote_path(
                "runtime_tmp_root",
                config.runtime_tmp_root,
                required=False,
                writable=True,
                remote_home=remote_home,
            ),
            "remote_out_dir_root": _optional_remote_path(
                "remote_out_dir_root",
                config.remote_out_dir_root,
                required=False,
                writable=True,
                remote_home=remote_home,
            ),
            "out_dir": out_dir_path,
        }
    )


def _remote_home(
    ssh_target: str,
    *,
    remote_home_cache: dict[str, str],
    remote_home_resolver: RemoteHomeResolver | None,
    require_now: bool,
) -> str:
    if not remote_home_resolver:
        if require_now:
            raise RuntimeError(f"Remote home resolver is required for {ssh_target}.")
        return ""
    if ssh_target not in remote_home_cache:
        remote_home_cache[ssh_target] = remote_home_resolver(ssh_target).strip()
    return remote_home_cache[ssh_target]


def _local_port_value(
    configured_port: int | None,
    *,
    reserve_ports: bool,
) -> tuple[int | None, str]:
    if configured_port is not None:
        return configured_port, "explicit"
    if reserve_ports:
        from remote_inference_launcher.ports import reserve_local_port

        reservation = reserve_local_port("127.0.0.1", None)
        try:
            return reservation.port, "reserved"
        finally:
            reservation.close()
    return None, "generated"


def _generated_int_strategy(configured_value: int | None) -> tuple[int | None, str]:
    if configured_value is not None:
        return configured_value, "explicit"
    return None, "generated"


def _endpoint_summary_path(registry_dir: Path, endpoint_label: str) -> Path:
    return registry_dir / "endpoints" / endpoint_label / "summary.json"


def _slurm_payload(config: SlurmVllmConfig) -> dict[str, object]:
    return {
        "partition": config.partition,
        "walltime": config.walltime,
        "num_gpus": config.num_gpus,
        "memory": config.memory,
        "cpus_per_task": config.cpus_per_task,
        "nodes": config.nodes,
        "exclude": config.exclude,
        "nodelist": config.nodelist,
        "sbatch_cmd": config.sbatch_cmd,
        "job_name_prefix": config.job_name_prefix,
        "distributed_backend": config.distributed_backend,
        "head_node_port": config.head_node_port,
        "ray_port": config.ray_port,
        "queue_timeout_seconds": config.queue_timeout_seconds,
        "queue_policy": _jsonable_config(config.queue_policy),
    }


def _readiness_payload(config: ReadinessConfig) -> dict[str, object]:
    return _jsonable_config(config)


def _resource_budget_payload(config: object) -> dict[str, object]:
    budget = getattr(config, "resource_budget", None)
    if budget is None:
        return {}
    return {key: value for key, value in _jsonable_config(budget).items() if value is not None}


def _endpoint_ownership_policy(config: object, ownership_policy: str) -> str:
    if isinstance(config, ExistingEndpointConfig):
        return "external"
    return ownership_policy


def _semantic_endpoint_payload(endpoint: EffectiveEndpointPlan) -> dict[str, object]:
    paths = {
        label: {
            "raw": path.raw,
            "resolved": path.resolved,
            "location": path.location,
            "required": path.required,
            "writable": path.writable,
            "expansion": path.expansion,
        }
        for label, path in endpoint.paths.items()
        if path.expansion != "generated" and label not in {"out_dir", "remote_out_dir_root"}
    }
    return {
        "name": endpoint.name,
        "kind": endpoint.kind,
        "backend_kind": endpoint.backend_kind,
        "model": endpoint.model,
        "served_model_name": endpoint.served_model_name,
        "api_key_set": endpoint.api_key_set,
        "ssh_target": endpoint.ssh_target,
        "ownership_policy": endpoint.ownership_policy,
        "keep_remote_job": endpoint.keep_remote_job,
        "paths": paths,
        "slurm": endpoint.slurm,
        "readiness": endpoint.readiness,
        "metadata": _semantic_metadata(endpoint),
    }


def _semantic_metadata(endpoint: EffectiveEndpointPlan) -> dict[str, object]:
    metadata = endpoint.metadata
    per_kind_fields = {
        "existing_endpoint": ("api_base", "discover_model"),
        "local_vllm": _VLLM_SEMANTIC_METADATA_FIELDS,
        "ssh_vllm": ("setup_cmd", *_VLLM_SEMANTIC_METADATA_FIELDS),
        "slurm_vllm": ("setup_cmd", *_VLLM_SEMANTIC_METADATA_FIELDS),
    }
    return {key: metadata[key] for key in per_kind_fields.get(endpoint.kind, ()) if key in metadata}


_VLLM_SEMANTIC_METADATA_FIELDS = (
    "target_device",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "extra_args",
)


def _launch_plan_payload(
    plan: EffectiveLaunchPlan,
    *,
    include_instance_hash: bool = False,
) -> dict[str, object]:
    payload = {
        "schema_version": plan.schema_version,
        "run_id": plan.run_id,
        "created_at": plan.created_at,
        "source_config_path": plan.source_config_path,
        "raw_config_hash": plan.raw_config_hash,
        "semantic_config_hash": plan.semantic_config_hash,
        "registry_dir": str(plan.registry_dir),
        "summary_path": str(plan.summary_path),
        "controller_policy": plan.controller_policy,
        "ownership_policy": plan.ownership_policy,
        "endpoints": [endpoint.to_dict() for endpoint in plan.endpoints],
        "resource_budget": plan.resource_budget,
    }
    if include_instance_hash:
        payload["instance_hash"] = plan.instance_hash
    return _jsonable(payload)


def _hash_payload(payload: object) -> str:
    encoded = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _jsonable_config(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable_config(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable_config(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [_jsonable_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _jsonable(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
