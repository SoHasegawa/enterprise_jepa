"""Preflight validation for managed inference configs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from remote_inference_launcher.endpoint_race import (
    EndpointRaceConfig,
    validate_endpoint_race_config,
)
from remote_inference_launcher.existing_endpoint import (
    ExistingEndpointConfig,
    validate_existing_endpoint_config,
)
from remote_inference_launcher.fleet import FleetConfig, validate_fleet_config
from remote_inference_launcher.inference_config import inference_config_from_mapping
from remote_inference_launcher.local_vllm import (
    LocalVllmConfig,
    validate_local_vllm_config,
    with_local_vllm_defaults,
)
from remote_inference_launcher.plans import (
    EffectiveEndpointPlan,
    EffectiveLaunchPlan,
    build_effective_launch_plan,
)
from remote_inference_launcher.ports import reserve_local_port
from remote_inference_launcher.preflight import PreflightCheck, run_preflight_checks_for_configs
from remote_inference_launcher.runtime_preflight import run_runtime_preflight_checks
from remote_inference_launcher.slurm_vllm import (
    SlurmVllmConfig,
    slurm_resource_preference_candidates,
    validate_slurm_vllm_config,
    with_slurm_vllm_defaults,
)
from remote_inference_launcher.ssh_vllm import (
    SshVllmConfig,
    validate_ssh_vllm_config,
    with_ssh_vllm_defaults,
)
from remote_inference_launcher.yaml_config import load_yaml_mapping


@dataclass(frozen=True)
class ValidationIssue:
    """One validation issue."""

    severity: str
    source: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "source": self.source, "message": self.message}


@dataclass(frozen=True)
class EndpointPlan:
    """Resolved validation facts for one concrete endpoint."""

    source: str
    name: str
    kind: str
    local_bind_host: str = ""
    local_port: int | None = None
    local_port_explicit: bool = False
    ssh_target: str = ""
    remote_port: int | None = None
    remote_port_explicit: bool = False
    job_name: str = ""
    job_name_explicit: bool = False
    out_dir: str = ""
    out_dir_explicit: bool = False
    num_gpus: int | None = None
    generated: Mapping[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "name": self.name,
            "kind": self.kind,
            "local_bind_host": self.local_bind_host,
            "local_port": self.local_port,
            "local_port_explicit": self.local_port_explicit,
            "ssh_target": self.ssh_target,
            "remote_port": self.remote_port,
            "remote_port_explicit": self.remote_port_explicit,
            "job_name": self.job_name,
            "job_name_explicit": self.job_name_explicit,
            "out_dir": self.out_dir,
            "out_dir_explicit": self.out_dir_explicit,
            "num_gpus": self.num_gpus,
            "generated": dict(self.generated),
        }


@dataclass(frozen=True)
class ValidationReport:
    """Validation output for one command invocation."""

    plans: tuple[EndpointPlan, ...]
    issues: tuple[ValidationIssue, ...]
    preflight_checks: tuple[PreflightCheck, ...] = ()
    effective_limits: Mapping[str, int | None] = field(default_factory=dict)
    effective_launch_plans: tuple[EffectiveLaunchPlan, ...] = ()
    strict_preflight: bool = False
    fail_on_unknown_preflight: bool = False
    runtime_preflight: bool = False

    @property
    def ok(self) -> bool:
        return self.result == "ok"

    @property
    def result(self) -> str:
        if any(issue.severity == "error" for issue in self.issues):
            return "failed"
        summary = self.preflight_summary
        if self.strict_preflight and (
            summary.get("transient_failure", 0) > 0 or summary.get("unknown", 0) > 0
        ):
            return "inconclusive"
        return "ok"

    @property
    def preflight_summary(self) -> dict[str, int]:
        counts = Counter(check.outcome for check in self.preflight_checks)
        return {
            "ok": counts.get("ok", 0),
            "skipped": counts.get("skipped", 0),
            "durable_failure": counts.get("durable_failure", 0),
            "transient_failure": counts.get("transient_failure", 0),
            "unknown": counts.get("unknown", 0),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "result": self.result,
            "issues": [issue.to_dict() for issue in self.issues],
            "preflight_checks": [check.to_dict() for check in self.preflight_checks],
            "preflight_summary": self.preflight_summary,
            "plans": [plan.to_dict() for plan in self.plans],
            "effective_limits": dict(self.effective_limits),
            "effective_launch_plans": [
                effective_plan.to_dict() for effective_plan in self.effective_launch_plans
            ],
            "runtime_preflight": self.runtime_preflight,
        }


@dataclass(frozen=True)
class ResourceUsage:
    """Peak resource usage implied by a launch plan subtree."""

    concurrent_logical_launches: int = 0
    active_candidate_attempts: int = 0
    submitted_slurm_jobs: int = 0
    total_requested_gpus: int = 0

    def to_limits(self) -> dict[str, int | None]:
        return {
            "max_concurrent_logical_launches": self.concurrent_logical_launches or None,
            "max_active_candidate_attempts": self.active_candidate_attempts or None,
            "max_submitted_slurm_jobs": self.submitted_slurm_jobs or None,
            "max_total_requested_gpus": self.total_requested_gpus or None,
        }


def validate_config_paths(
    paths: Iterable[str | Path],
    *,
    preflight: bool = False,
    strict_preflight: bool = False,
    fail_on_unknown_preflight: bool = False,
    runtime_preflight: bool = False,
) -> ValidationReport:
    """Validate one or more inference config paths."""

    plans: list[EndpointPlan] = []
    effective_launch_plans: list[EffectiveLaunchPlan] = []
    root_usages: list[ResourceUsage] = []
    issues: list[ValidationIssue] = []
    preflight_checks: list[PreflightCheck] = []
    loaded_configs: list[tuple[str, object]] = []
    runtime_preflight_inputs: list[tuple[str, object, EffectiveLaunchPlan]] = []
    for path in paths:
        source = str(path)
        try:
            raw_config = load_yaml_mapping(path, kind="inference config")
            config = inference_config_from_mapping(raw_config, source=source)
            loaded_configs.append((source, config))
            issues.extend(_policy_validation_issues(config, source=source))
            effective_plan = build_effective_launch_plan(
                config,
                source_config_path=source,
                raw_config=raw_config,
            )
            effective_launch_plans.append(effective_plan)
            plans.extend(_endpoint_plans_from_effective_plan(effective_plan))
            root_usages.append(_peak_usage_for_config(config))
            issues.extend(_resource_budget_issues(config, source=source))
            runtime_preflight_inputs.append((source, config, effective_plan))
        except Exception as error:
            issues.append(ValidationIssue("error", source, str(error)))
    if (preflight or strict_preflight) and loaded_configs:
        checks = run_preflight_checks_for_configs(tuple(loaded_configs))
        preflight_checks.extend(checks)
        issues.extend(
            _preflight_issues(
                checks,
                strict_preflight=strict_preflight,
                fail_on_unknown_preflight=fail_on_unknown_preflight,
            )
        )
    if runtime_preflight:
        for source, config, effective_plan in runtime_preflight_inputs:
            checks = run_runtime_preflight_checks(
                config,
                source=source,
                effective_plan=effective_plan,
            )
            preflight_checks.extend(checks)
            issues.extend(
                _preflight_issues(
                    checks,
                    strict_preflight=strict_preflight,
                    fail_on_unknown_preflight=fail_on_unknown_preflight,
                )
            )
    issues.extend(_duplicate_issues(plans))
    issues.extend(_explicit_local_bind_issues(plans))
    return ValidationReport(
        plans=tuple(plans),
        issues=tuple(issues),
        preflight_checks=tuple(preflight_checks),
        effective_limits=_effective_limits(root_usages),
        effective_launch_plans=tuple(effective_launch_plans),
        strict_preflight=strict_preflight,
        fail_on_unknown_preflight=fail_on_unknown_preflight,
        runtime_preflight=runtime_preflight,
    )


def _policy_validation_issues(config: object, *, source: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        _validate_config_policy(config)
    except Exception as error:
        issues.append(ValidationIssue("error", source, str(error)))
    if isinstance(config, FleetConfig):
        for name, endpoint in config.endpoints.items():
            issues.extend(_policy_validation_issues(endpoint, source=f"{source} endpoint {name!r}"))
    elif isinstance(config, EndpointRaceConfig):
        for name, candidate in config.candidates.items():
            issues.extend(
                _policy_validation_issues(candidate, source=f"{source} candidate {name!r}")
            )
    elif isinstance(config, SlurmVllmConfig) and config.resource_preferences:
        for candidate_name, candidate_config in slurm_resource_preference_candidates(config):
            try:
                issues.extend(
                    _policy_validation_issues(
                        candidate_config,
                        source=f"{source} resource {candidate_name!r}",
                    )
                )
            except Exception as error:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"{source} resource {candidate_name!r}",
                        str(error),
                    )
                )
    return issues


def _validate_config_policy(config: object) -> None:
    if isinstance(config, FleetConfig):
        validate_fleet_config(config)
        return
    if isinstance(config, EndpointRaceConfig):
        validate_endpoint_race_config(config)
        return
    if isinstance(config, ExistingEndpointConfig):
        validate_existing_endpoint_config(config)
        return
    if isinstance(config, LocalVllmConfig):
        validate_local_vllm_config(with_local_vllm_defaults(config))
        return
    if isinstance(config, SshVllmConfig):
        validate_ssh_vllm_config(with_ssh_vllm_defaults(config))
        return
    if isinstance(config, SlurmVllmConfig):
        validate_slurm_vllm_config(with_slurm_vllm_defaults(config))
        return
    raise TypeError(f"Unsupported inference config type: {type(config).__name__}")


def format_validation_report(report: ValidationReport) -> str:
    """Return human-readable validation output."""

    lines = [f"Validation result: {report.result.upper()}"]
    lines.append("")
    lines.extend(_format_effective_launch_plans(report.effective_launch_plans))
    lines.append("")
    lines.extend(_format_endpoint_plans(report))
    lines.append("")
    lines.append("Effective limits:")
    for name, value in sorted(report.effective_limits.items()):
        lines.append(f"  {name}: {value if value is not None else 'unbounded'}")
    lines.append("")
    lines.append("Preflight checks:")
    if not report.preflight_checks:
        lines.append("  (not run)")
    for layer, checks in _preflight_checks_by_layer(report.preflight_checks):
        lines.append(f"  {layer}:")
        for check in checks:
            detail = f": {check.detail}" if check.detail else ""
            lines.append(
                f"    [{check.outcome} {check.code}] {check.source}: "
                f"{check.name} attempts={check.attempts}{detail}"
            )
    if report.preflight_checks:
        lines.append("")
        lines.append("Preflight summary:")
        for outcome, count in report.preflight_summary.items():
            lines.append(f"  {outcome}: {count}")
    lines.append("")
    lines.append("Issues:")
    if not report.issues:
        lines.append("  (none)")
    for issue in report.issues:
        lines.append(f"  [{issue.severity}] {issue.source}: {issue.message}")
    return "\n".join(lines)


def validation_report_json(report: ValidationReport) -> str:
    """Return JSON validation output."""

    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


_PREFLIGHT_LAYER_ORDER = (
    "local",
    "remote-control",
    "slurm-scheduler",
    "compute-runtime",
    "serving-readiness",
)


def _preflight_checks_by_layer(
    checks: tuple[PreflightCheck, ...],
) -> list[tuple[str, list[PreflightCheck]]]:
    grouped: dict[str, list[PreflightCheck]] = defaultdict(list)
    for check in checks:
        grouped[check.resolved_layer].append(check)
    ordered: list[tuple[str, list[PreflightCheck]]] = [
        (layer, grouped.pop(layer)) for layer in _PREFLIGHT_LAYER_ORDER if layer in grouped
    ]
    ordered.extend((layer, grouped[layer]) for layer in sorted(grouped))
    return ordered


def _format_effective_launch_plans(
    effective_plans: tuple[EffectiveLaunchPlan, ...],
) -> list[str]:
    lines = ["Effective launch plans:"]
    if not effective_plans:
        lines.append("  (none)")
    for effective_plan in effective_plans:
        lines.append(
            "  - "
            f"{effective_plan.source_config_path}: run_id={effective_plan.run_id} "
            f"semantic_hash={effective_plan.semantic_config_hash} "
            f"instance_hash={effective_plan.instance_hash}"
        )
        lines.append(f"    registry_dir={effective_plan.registry_dir}")
        lines.append(f"    summary_path={effective_plan.summary_path}")
    return lines


def _format_endpoint_plans(report: ValidationReport) -> list[str]:
    lines = ["Effective endpoint plans:"]
    if not report.plans:
        lines.append("  (none)")
    for plan in report.plans:
        lines.extend(_format_endpoint_plan(report.effective_launch_plans, plan))
    return lines


def _format_endpoint_plan(
    effective_plans: tuple[EffectiveLaunchPlan, ...],
    plan: EndpointPlan,
) -> list[str]:
    lines = [
        "  - "
        f"{plan.source}: {plan.name} ({plan.kind}) "
        f"local={_local_strategy(plan)} remote={_remote_strategy(plan)} "
        f"job={plan.job_name or _generated_label(plan, 'job_name')} "
        f"out_dir={plan.out_dir or _generated_label(plan, 'out_dir')}"
    ]
    matching_endpoint = _matching_effective_endpoint(effective_plans, plan)
    if matching_endpoint is None:
        return lines
    lines.append(f"    label={matching_endpoint.endpoint_label}")
    lines.append(f"    summary_path={matching_endpoint.summary_path}")
    if matching_endpoint.remote_state_path:
        lines.append(f"    remote_state_path={matching_endpoint.remote_state_path}")
    for path in matching_endpoint.paths.values():
        lines.append(
            f"    {path.label}: {path.resolved} ({path.location}, expansion={path.expansion})"
        )
    return lines


def _endpoint_plans_from_effective_plan(
    effective_plan: EffectiveLaunchPlan,
) -> list[EndpointPlan]:
    plans: list[EndpointPlan] = []
    for endpoint in effective_plan.endpoints:
        plans.append(
            EndpointPlan(
                source=endpoint.raw_config_source,
                name=endpoint.name,
                kind=endpoint.kind,
                local_bind_host=endpoint.local_bind_host,
                local_port=endpoint.local_port,
                local_port_explicit=endpoint.local_port_strategy == "explicit",
                ssh_target=endpoint.ssh_target,
                remote_port=endpoint.remote_port,
                remote_port_explicit=endpoint.remote_port_strategy == "explicit",
                job_name=endpoint.job_name,
                job_name_explicit=endpoint.metadata.get("job_name_strategy") == "explicit",
                out_dir=endpoint.out_dir,
                out_dir_explicit=endpoint.metadata.get("out_dir_strategy") == "explicit",
                num_gpus=_endpoint_num_gpus(endpoint),
                generated={
                    "job_name": endpoint.metadata.get("job_name_strategy") == "generated",
                    "out_dir": endpoint.metadata.get("out_dir_strategy") == "generated",
                    "local_port": endpoint.local_port_strategy in {"generated", "reserved"},
                    "remote_port": endpoint.remote_port_strategy == "generated",
                },
            )
        )
    return plans


def _endpoint_num_gpus(endpoint: EffectiveEndpointPlan) -> int | None:
    slurm_payload = endpoint.slurm
    if isinstance(slurm_payload, Mapping) and slurm_payload.get("num_gpus") is not None:
        return int(slurm_payload["num_gpus"])
    if endpoint.metadata.get("tensor_parallel_size") is not None:
        return int(endpoint.metadata["tensor_parallel_size"])
    return None


def _matching_effective_endpoint(
    effective_plans: tuple[EffectiveLaunchPlan, ...],
    plan: EndpointPlan,
) -> EffectiveEndpointPlan | None:
    for effective_plan in effective_plans:
        for endpoint in effective_plan.endpoints:
            if (
                endpoint.raw_config_source == plan.source
                and endpoint.name == plan.name
                and endpoint.kind == plan.kind
            ):
                return endpoint
    return None


def _plans_for_config(config: object, *, source: str) -> list[EndpointPlan]:  # noqa: C901
    if isinstance(config, FleetConfig):
        plans: list[EndpointPlan] = []
        for name, endpoint in sorted(config.endpoints.items()):
            plans.extend(_plans_for_config(endpoint, source=f"{source} endpoint {name!r}"))
        return plans
    if isinstance(config, EndpointRaceConfig):
        plans = []
        for name, candidate in sorted(config.candidates.items()):
            plans.extend(_plans_for_config(candidate, source=f"{source} candidate {name!r}"))
        return plans
    if isinstance(config, ExistingEndpointConfig):
        return [
            EndpointPlan(
                source=source,
                name=config.name,
                kind="existing_endpoint",
                generated={},
            )
        ]
    if isinstance(config, LocalVllmConfig):
        resolved = with_local_vllm_defaults(config)
        return [
            EndpointPlan(
                source=source,
                name=resolved.name,
                kind="local_vllm",
                local_bind_host=resolved.host,
                local_port=resolved.port,
                local_port_explicit=config.port is not None,
                generated={"local_port": config.port is None},
                num_gpus=resolved.tensor_parallel_size,
            )
        ]
    if isinstance(config, SshVllmConfig):
        resolved = with_ssh_vllm_defaults(config)
        return [
            EndpointPlan(
                source=source,
                name=resolved.name,
                kind="ssh_vllm",
                local_bind_host=resolved.local_bind_host,
                local_port=resolved.local_port,
                local_port_explicit=resolved.local_port is not None,
                ssh_target=resolved.ssh_target,
                remote_port=resolved.remote_port,
                remote_port_explicit=config.remote_port is not None,
                out_dir=resolved.out_dir,
                out_dir_explicit=bool(config.out_dir),
                generated={
                    "local_port": resolved.local_port is None,
                    "remote_port": config.remote_port is None,
                    "out_dir": not bool(config.out_dir),
                },
                num_gpus=resolved.tensor_parallel_size,
            )
        ]
    if isinstance(config, SlurmVllmConfig):
        if config.resource_preferences:
            plans = []
            for candidate_name, candidate_config in slurm_resource_preference_candidates(config):
                plans.extend(
                    _plans_for_config(
                        candidate_config,
                        source=f"{source} resource {candidate_name!r}",
                    )
                )
            return plans
        resolved = with_slurm_vllm_defaults(config)
        return [
            EndpointPlan(
                source=source,
                name=resolved.name,
                kind="slurm_vllm",
                local_bind_host=resolved.local_bind_host,
                local_port=resolved.local_port,
                local_port_explicit=resolved.local_port is not None,
                ssh_target=resolved.ssh_target,
                remote_port=resolved.remote_port,
                remote_port_explicit=config.remote_port is not None,
                job_name=resolved.job_name,
                job_name_explicit=bool(config.job_name),
                out_dir=resolved.out_dir,
                out_dir_explicit=bool(config.out_dir),
                generated={
                    "job_name": not bool(config.job_name),
                    "out_dir": not bool(config.out_dir),
                    "local_port": config.local_port is None,
                    "remote_port": config.remote_port is None,
                },
                num_gpus=resolved.num_gpus,
            )
        ]
    raise TypeError(f"Unsupported inference config type: {type(config).__name__}")


def _duplicate_issues(plans: list[EndpointPlan]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    local_bindings: list[EndpointPlan] = []
    remote_ports: dict[tuple[str, int], list[EndpointPlan]] = defaultdict(list)
    job_names: dict[tuple[str, str], list[EndpointPlan]] = defaultdict(list)
    out_dirs: dict[tuple[str, str], list[EndpointPlan]] = defaultdict(list)
    for plan in plans:
        if plan.local_port_explicit and plan.local_port is not None:
            local_bindings.append(plan)
        if plan.remote_port_explicit and plan.remote_port is not None and plan.ssh_target:
            remote_ports[(plan.ssh_target, plan.remote_port)].append(plan)
        if plan.job_name_explicit and plan.job_name and plan.ssh_target:
            job_names[(plan.ssh_target, plan.job_name)].append(plan)
        if plan.out_dir_explicit and plan.out_dir and plan.ssh_target:
            out_dirs[(plan.ssh_target, plan.out_dir)].append(plan)
    issues.extend(_duplicate_local_bind_issues(local_bindings))
    for label, grouped in (
        ("explicit remote server port", remote_ports),
        ("explicit Slurm job name", job_names),
        ("explicit remote output directory", out_dirs),
    ):
        for key, matches in grouped.items():
            if len(matches) > 1:
                sources = ", ".join(match.source for match in matches)
                issues.append(
                    ValidationIssue(
                        "error",
                        sources,
                        f"Duplicate {label}: {key!r}.",
                    )
                )
    return issues


def _duplicate_local_bind_issues(plans: list[EndpointPlan]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for index, first in enumerate(plans):
        for second in plans[index + 1 :]:
            if not _local_bindings_overlap(first, second):
                continue
            issues.append(
                ValidationIssue(
                    "error",
                    f"{first.source}, {second.source}",
                    "Duplicate explicit local bind endpoint: "
                    f"{_local_bind_label(first)} overlaps {_local_bind_label(second)}.",
                )
            )
    return issues


def _local_bindings_overlap(first: EndpointPlan, second: EndpointPlan) -> bool:
    if first.local_port != second.local_port:
        return False
    first_host = _normalized_bind_host(first.local_bind_host)
    second_host = _normalized_bind_host(second.local_bind_host)
    return first_host == "*" or second_host == "*" or first_host == second_host


def _normalized_bind_host(host: str) -> str:
    normalized = host.strip().lower()
    if normalized in {"", "*", "0.0.0.0"}:
        return "*"
    if normalized == "localhost":
        return "127.0.0.1"
    return normalized


def _local_bind_label(plan: EndpointPlan) -> str:
    return f"{plan.local_bind_host}:{plan.local_port}"


def _explicit_local_bind_issues(plans: list[EndpointPlan]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for plan in plans:
        if not plan.local_port_explicit or plan.local_port is None:
            continue
        try:
            reservation = reserve_local_port(plan.local_bind_host, plan.local_port)
        except OSError as error:
            issues.append(
                ValidationIssue(
                    "error",
                    plan.source,
                    f"Explicit local port {plan.local_port} is not bindable on "
                    f"{plan.local_bind_host!r}: {error}.",
                )
            )
        else:
            reservation.close()
    return issues


def _effective_limits(usages: list[ResourceUsage]) -> dict[str, int | None]:
    return _sum_usage(usages).to_limits()


def _resource_budget_issues(config: object, *, source: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    budget = getattr(config, "resource_budget", None)
    if budget is not None:
        for field_name, value in budget.__dict__.items():
            if value is not None and value < 1:
                issues.append(
                    ValidationIssue(
                        "error",
                        source,
                        f"resource_budget.{field_name} must be positive.",
                    )
                )
        issues.extend(
            _budget_fit_issues(
                source=source,
                budget=budget,
                usage=_peak_usage_for_config(config),
            )
        )
    if isinstance(config, FleetConfig):
        child_usages = [_peak_usage_for_config(endpoint) for endpoint in config.endpoints.values()]
        active_logical = _fleet_concurrent_logical_launches(config)
        max_child_candidate_attempts = max(
            (usage.active_candidate_attempts for usage in child_usages),
            default=1,
        )
        if active_logical > 1 and max_child_candidate_attempts > 1 and budget is None:
            issues.append(
                ValidationIssue(
                    "error",
                    source,
                    "Fleets with multiple concurrent logical launches and candidate racing above "
                    "one require an explicit resource_budget.",
                )
            )
        for name, endpoint in config.endpoints.items():
            issues.extend(_resource_budget_issues(endpoint, source=f"{source} endpoint {name!r}"))
    elif isinstance(config, EndpointRaceConfig):
        for name, candidate in config.candidates.items():
            issues.extend(_resource_budget_issues(candidate, source=f"{source} candidate {name!r}"))
    return issues


def _budget_fit_issues(
    *,
    source: str,
    budget: object | None,
    usage: ResourceUsage,
) -> list[ValidationIssue]:
    if budget is None:
        return []
    checks = {
        "max_concurrent_logical_launches": usage.concurrent_logical_launches,
        "max_active_candidate_attempts": usage.active_candidate_attempts,
        "max_submitted_slurm_jobs": usage.submitted_slurm_jobs,
        "max_total_requested_gpus": usage.total_requested_gpus,
    }
    issues: list[ValidationIssue] = []
    for field_name, required in checks.items():
        cap = getattr(budget, field_name, None)
        if cap is not None and required > cap:
            issues.append(
                ValidationIssue(
                    "error",
                    source,
                    f"resource_budget.{field_name}={cap} cannot fit required {required}.",
                )
            )
    return issues


def _peak_usage_for_config(config: object) -> ResourceUsage:
    if isinstance(config, FleetConfig):
        child_usages = [_peak_usage_for_config(endpoint) for endpoint in config.endpoints.values()]
        active_count = _fleet_concurrent_logical_launches(config)
        launch_usage = ResourceUsage(
            concurrent_logical_launches=active_count,
            active_candidate_attempts=_top_sum(
                child_usages,
                "active_candidate_attempts",
                active_count,
            ),
            submitted_slurm_jobs=_top_sum(
                child_usages,
                "submitted_slurm_jobs",
                active_count,
            ),
            total_requested_gpus=_top_sum(
                child_usages,
                "total_requested_gpus",
                active_count,
            ),
        )
        held_usage = _sum_usage(child_usages)
        return ResourceUsage(
            concurrent_logical_launches=active_count,
            active_candidate_attempts=max(
                launch_usage.active_candidate_attempts,
                held_usage.active_candidate_attempts,
            ),
            submitted_slurm_jobs=max(
                launch_usage.submitted_slurm_jobs,
                held_usage.submitted_slurm_jobs,
            ),
            total_requested_gpus=max(
                launch_usage.total_requested_gpus,
                held_usage.total_requested_gpus,
            ),
        )
    if isinstance(config, EndpointRaceConfig):
        candidate_usages = [
            _peak_usage_for_config(candidate) for candidate in config.candidates.values()
        ]
        active_count = min(config.max_active_candidates, len(candidate_usages))
        return ResourceUsage(
            concurrent_logical_launches=1,
            active_candidate_attempts=_top_sum(
                candidate_usages,
                "active_candidate_attempts",
                active_count,
            ),
            submitted_slurm_jobs=_top_sum(
                candidate_usages,
                "submitted_slurm_jobs",
                active_count,
            ),
            total_requested_gpus=_top_sum(
                candidate_usages,
                "total_requested_gpus",
                active_count,
            ),
        )
    if isinstance(config, ExistingEndpointConfig):
        return ResourceUsage(concurrent_logical_launches=1, active_candidate_attempts=1)
    if isinstance(config, LocalVllmConfig):
        return ResourceUsage(
            concurrent_logical_launches=1,
            active_candidate_attempts=1,
            total_requested_gpus=_vllm_requested_gpus(
                target_device=config.target_device,
                tensor_parallel_size=config.tensor_parallel_size,
            ),
        )
    if isinstance(config, SshVllmConfig):
        return ResourceUsage(
            concurrent_logical_launches=1,
            active_candidate_attempts=1,
            total_requested_gpus=_vllm_requested_gpus(
                target_device=config.target_device,
                tensor_parallel_size=config.tensor_parallel_size,
            ),
        )
    if isinstance(config, SlurmVllmConfig):
        if config.resource_preferences:
            candidates = slurm_resource_preference_candidates(config)
            candidate_usages = [
                _peak_usage_for_config(candidate_config)
                for _candidate_name, candidate_config in candidates
            ]
            active_count = (
                config.candidate_race.max_active_candidates if config.candidate_race.enabled else 1
            )
            active_count = min(active_count, len(candidate_usages))
            return ResourceUsage(
                concurrent_logical_launches=1,
                active_candidate_attempts=_top_sum(
                    candidate_usages,
                    "active_candidate_attempts",
                    active_count,
                ),
                submitted_slurm_jobs=_top_sum(
                    candidate_usages,
                    "submitted_slurm_jobs",
                    active_count,
                ),
                total_requested_gpus=_top_sum(
                    candidate_usages,
                    "total_requested_gpus",
                    active_count,
                ),
            )
        return ResourceUsage(
            concurrent_logical_launches=1,
            active_candidate_attempts=1,
            submitted_slurm_jobs=1,
            total_requested_gpus=config.num_gpus or 0,
        )
    raise TypeError(f"Unsupported inference config type: {type(config).__name__}")


def _fleet_concurrent_logical_launches(config: FleetConfig) -> int:
    if not config.endpoints:
        return 0
    return min(config.max_active_launches or len(config.endpoints), len(config.endpoints))


def _sum_usage(usages: Iterable[ResourceUsage]) -> ResourceUsage:
    usage_list = list(usages)
    return ResourceUsage(
        concurrent_logical_launches=sum(usage.concurrent_logical_launches for usage in usage_list),
        active_candidate_attempts=sum(usage.active_candidate_attempts for usage in usage_list),
        submitted_slurm_jobs=sum(usage.submitted_slurm_jobs for usage in usage_list),
        total_requested_gpus=sum(usage.total_requested_gpus for usage in usage_list),
    )


def _top_sum(
    usages: Iterable[ResourceUsage],
    field_name: str,
    count: int,
) -> int:
    values = sorted((getattr(usage, field_name) for usage in usages), reverse=True)
    return sum(values[: max(count, 0)])


def _vllm_requested_gpus(
    *,
    target_device: str,
    tensor_parallel_size: int | None,
) -> int:
    if target_device == "cpu":
        return 0
    return tensor_parallel_size or 1


def _preflight_issues(
    checks: Iterable[PreflightCheck],
    *,
    strict_preflight: bool,
    fail_on_unknown_preflight: bool,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for check in checks:
        if check.ok:
            continue
        severity = "warning"
        if strict_preflight and check.outcome == "durable_failure":
            severity = "error"
        if strict_preflight and check.outcome == "unknown" and fail_on_unknown_preflight:
            severity = "error"
        issues.append(
            ValidationIssue(
                severity,
                check.source,
                f"Preflight check {check.name!r} {check.outcome} ({check.code}): {check.detail}",
            )
        )
    return issues


def _local_strategy(plan: EndpointPlan) -> str:
    if plan.local_port_explicit:
        return f"{plan.local_bind_host}:{plan.local_port}"
    if plan.kind in {"local_vllm", "ssh_vllm", "slurm_vllm"}:
        return f"{plan.local_bind_host}:generated-at-launch"
    return "external"


def _remote_strategy(plan: EndpointPlan) -> str:
    if not plan.ssh_target:
        return "n/a"
    if plan.remote_port_explicit:
        return f"{plan.ssh_target}:{plan.remote_port}"
    return f"{plan.ssh_target}:generated-at-launch"


def _generated_label(plan: EndpointPlan, field_name: str) -> str:
    return "generated-at-launch" if plan.generated.get(field_name) else "n/a"
