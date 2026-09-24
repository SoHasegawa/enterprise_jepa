"""Strict YAML config loading for inference launchers."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from remote_inference_launcher.endpoint_race import EndpointRaceConfig, EndpointRaceLauncher
from remote_inference_launcher.existing_endpoint import (
    ExistingEndpointConfig,
    ExistingEndpointLauncher,
)
from remote_inference_launcher.fleet import FleetConfig, InferenceFleetLauncher
from remote_inference_launcher.local_vllm import LocalVllmConfig, LocalVllmLauncher
from remote_inference_launcher.slurm_vllm import SlurmVllmConfig, SlurmVllmLauncher
from remote_inference_launcher.ssh_vllm import SshVllmConfig, SshVllmLauncher
from remote_inference_launcher.yaml_config import load_yaml_mapping

KIND_TO_CONFIG: dict[str, type] = {
    "existing_endpoint": ExistingEndpointConfig,
    "local_vllm": LocalVllmConfig,
    "slurm_vllm": SlurmVllmConfig,
    "ssh_vllm": SshVllmConfig,
}
ConfigValueCoercer = Callable[[Any, str, str, object], object]


def load_inference_config(path: str | Path) -> object:
    """Load a strict inference launcher config from YAML."""

    raw_config = load_yaml_mapping(path, kind="inference config")
    return inference_config_from_mapping(raw_config, source=str(Path(path)))


def inference_config_from_mapping(
    values: Mapping[str, Any],
    *,
    source: str = "inference config",
    default_name: str | None = None,
) -> object:
    """Build a launcher config object from a strict mapping."""

    config_values = dict(values)
    if default_name and "name" not in config_values:
        config_values["name"] = default_name
    kind = config_values.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"{source} field 'kind' must be a non-empty string.")
    normalized_kind = kind.strip()
    if normalized_kind == "fleet":
        return fleet_config_from_mapping(config_values, source=source)
    if normalized_kind == "endpoint_race":
        return endpoint_race_config_from_mapping(config_values, source=source)
    config_class = KIND_TO_CONFIG.get(normalized_kind)
    if config_class is None:
        raise ValueError(
            f"{source} field 'kind' must be one of: "
            f"{', '.join(sorted([*KIND_TO_CONFIG, 'endpoint_race', 'fleet']))}."
        )
    leaf_values = {key: value for key, value in config_values.items() if key != "kind"}
    return dataclass_config_from_mapping(config_class, leaf_values, source=source)


def fleet_config_from_mapping(values: Mapping[str, Any], *, source: str) -> FleetConfig:
    """Build a fleet config from a strict mapping."""

    unknown = sorted(
        set(values)
        - {
            "kind",
            "name",
            "endpoints",
            "max_active_launches",
            "launch_stagger_seconds",
            "failure_policy",
            "handoff_mode",
            "endpoint_lifetime_policy",
            "resource_budget",
            "launch_summary_path",
            "overwrite_launch_summary",
        }
    )
    if unknown:
        raise ValueError(f"{source} contains unsupported fleet fields: {', '.join(unknown)}.")
    endpoints = values.get("endpoints")
    if not isinstance(endpoints, Mapping) or not endpoints:
        raise ValueError(f"{source} field 'endpoints' must be a non-empty mapping.")
    parsed: dict[str, object] = {}
    for name, raw_endpoint in endpoints.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{source} endpoint names must be non-empty strings.")
        if not isinstance(raw_endpoint, Mapping):
            raise ValueError(f"{source} endpoint {name!r} must be a mapping.")
        parsed[name] = inference_config_from_mapping(
            raw_endpoint,
            source=f"{source} endpoint {name!r}",
            default_name=name,
        )
    config_values = {
        key: value for key, value in values.items() if key not in {"kind", "endpoints"}
    }
    config_values["endpoints"] = parsed
    return dataclass_config_from_mapping(FleetConfig, config_values, source=source)


def endpoint_race_config_from_mapping(
    values: Mapping[str, Any],
    *,
    source: str,
) -> EndpointRaceConfig:
    """Build an endpoint-race config from a strict mapping."""

    allowed = {
        "kind",
        "name",
        "candidates",
        "max_active_candidates",
        "launch_stagger_seconds",
        "winner_condition",
        "cancel_losers",
        "resource_budget",
        "launch_summary_path",
        "overwrite_launch_summary",
    }
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            f"{source} contains unsupported endpoint_race fields: {', '.join(unknown)}."
        )
    raw_candidates = values.get("candidates")
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, str | bytes):
        raise ValueError(f"{source} field 'candidates' must be a non-empty list of mappings.")
    candidates: dict[str, object] = {}
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, Mapping):
            raise ValueError(f"{source} candidate {index} must be a mapping.")
        candidate_name = raw_candidate.get("name", f"candidate_{index}")
        if not isinstance(candidate_name, str) or not candidate_name.strip():
            raise ValueError(f"{source} candidate {index} name must be a non-empty string.")
        if candidate_name in candidates:
            raise ValueError(f"{source} contains duplicate candidate name: {candidate_name}.")
        candidates[candidate_name] = inference_config_from_mapping(
            raw_candidate,
            source=f"{source} candidate {candidate_name!r}",
            default_name=candidate_name,
        )
    config_values = {
        key: value for key, value in values.items() if key not in {"kind", "candidates"}
    }
    config_values["candidates"] = candidates
    return dataclass_config_from_mapping(EndpointRaceConfig, config_values, source=source)


def dataclass_config_from_mapping(
    config_class: type,
    values: Mapping[str, Any],
    *,
    source: str,
) -> object:
    """Build a dataclass config from a strict mapping."""

    if not is_dataclass(config_class):
        raise TypeError(f"{config_class!r} is not a dataclass.")
    field_names = {field.name for field in fields(config_class)}
    unknown = sorted(set(values) - field_names)
    if unknown:
        raise ValueError(f"{source} contains unsupported fields: {', '.join(unknown)}.")
    hints = get_type_hints(config_class)
    converted = {
        field.name: _coerce_config_value(
            values[field.name],
            annotation=hints[field.name],
            field_name=field.name,
            source=source,
        )
        for field in fields(config_class)
        if field.name in values
    }
    return config_class(**converted)


def launcher_from_config(config: object, *, run_id: str | None = None):
    """Build a launcher from an already parsed config object."""

    if isinstance(config, FleetConfig):
        return InferenceFleetLauncher(config, run_id=run_id)
    if isinstance(config, EndpointRaceConfig):
        return EndpointRaceLauncher(config, run_id=run_id)
    if isinstance(config, ExistingEndpointConfig):
        return ExistingEndpointLauncher(config)
    if isinstance(config, LocalVllmConfig):
        return LocalVllmLauncher(config, run_id=run_id)
    if isinstance(config, SlurmVllmConfig):
        return SlurmVllmLauncher(config, run_id=run_id)
    if isinstance(config, SshVllmConfig):
        return SshVllmLauncher(config, run_id=run_id)
    if hasattr(config, "start") and hasattr(config, "stop"):
        return config
    raise TypeError(f"Unsupported inference launcher config: {type(config).__name__}")


def load_inference_launcher(
    path: str | Path,
    *,
    launch_summary_path: str | None = None,
    overwrite_launch_summary: bool = False,
    run_id: str | None = None,
    endpoint_plan: object | None = None,
):
    """Load a YAML config and return its launcher."""

    config = load_inference_config(path)
    if endpoint_plan is not None:
        config = _with_effective_endpoint_overrides(config, endpoint_plan=endpoint_plan)
    if launch_summary_path is not None or overwrite_launch_summary:
        config = _with_runtime_summary_overrides(
            config,
            launch_summary_path=launch_summary_path,
            overwrite_launch_summary=overwrite_launch_summary,
        )
    return launcher_from_config(config, run_id=run_id)


def _with_effective_endpoint_overrides(config: object, *, endpoint_plan: object) -> object:
    values: dict[str, object] = {}
    for config_field, plan_field in (
        ("launch_summary_path", "summary_path"),
        ("port", "local_port"),
        ("local_port", "local_port"),
        ("remote_port", "remote_port"),
        ("job_name", "job_name"),
        ("out_dir", "out_dir"),
    ):
        if not hasattr(config, config_field):
            continue
        value = _endpoint_plan_value(endpoint_plan, plan_field)
        if value in {"", None}:
            continue
        values[config_field] = (
            str(value) if config_field in {"launch_summary_path", "out_dir"} else value
        )
    if values and is_dataclass(config):
        return replace(config, **values)
    return config


def _endpoint_plan_value(endpoint_plan: object, field_name: str) -> object:
    if isinstance(endpoint_plan, Mapping):
        return endpoint_plan.get(field_name)
    return getattr(endpoint_plan, field_name, None)


def _with_runtime_summary_overrides(
    config: object,
    *,
    launch_summary_path: str | None,
    overwrite_launch_summary: bool,
) -> object:
    values: dict[str, object] = {}
    if launch_summary_path is not None and hasattr(config, "launch_summary_path"):
        values["launch_summary_path"] = launch_summary_path
    if overwrite_launch_summary and hasattr(config, "overwrite_launch_summary"):
        values["overwrite_launch_summary"] = True
    if values and is_dataclass(config):
        return replace(config, **values)
    return config


def _coerce_config_value(
    value: Any,
    *,
    annotation: object,
    field_name: str,
    source: str,
) -> object:
    coercer = _coercer_for_annotation(annotation, field_name=field_name)
    return coercer(value, field_name, source, annotation)


def _coercer_for_annotation(annotation: object, *, field_name: str) -> ConfigValueCoercer:
    coercer = _scalar_coercer_for_annotation(annotation)
    if coercer is not None:
        return coercer
    coercer = _collection_coercer_for_annotation(annotation)
    if coercer is not None:
        return coercer
    coercer = _dataclass_coercer_for_annotation(annotation)
    if coercer is not None:
        return coercer
    raise TypeError(f"Unsupported config field annotation for {field_name}: {annotation}")


def _scalar_coercer_for_annotation(annotation: object) -> ConfigValueCoercer | None:
    if annotation is str:
        return _coerce_string_field
    if _accepts_int(annotation):
        return _coerce_int_field
    if annotation is bool:
        return _coerce_bool_field
    if _accepts_float(annotation):
        return _coerce_float_field
    return None


def _collection_coercer_for_annotation(annotation: object) -> ConfigValueCoercer | None:
    if _is_string_tuple(annotation):
        return _coerce_string_tuple_field
    if _is_mapping(annotation):
        return _coerce_mapping_field
    if _is_mapping_tuple(annotation):
        return _coerce_mapping_tuple_field
    if _is_optional_str(annotation):
        return _coerce_optional_string_field
    return None


def _dataclass_coercer_for_annotation(annotation: object) -> ConfigValueCoercer | None:
    if _is_optional_dataclass(annotation):
        return _coerce_optional_dataclass_field
    if is_dataclass(annotation):
        return _coerce_dataclass_field
    return None


def _coerce_string_field(value: Any, field_name: str, source: str, _annotation: object) -> str:
    return _string_value(value, field_name=field_name, source=source)


def _coerce_int_field(value: Any, field_name: str, source: str, annotation: object) -> int | None:
    return _int_value(
        value,
        field_name=field_name,
        source=source,
        optional=_is_optional_int(annotation),
    )


def _coerce_bool_field(value: Any, field_name: str, source: str, _annotation: object) -> bool:
    return _bool_value(value, field_name=field_name, source=source)


def _coerce_float_field(
    value: Any, field_name: str, source: str, annotation: object
) -> float | None:
    return _float_value(
        value,
        field_name=field_name,
        source=source,
        optional=_is_optional_float(annotation),
    )


def _coerce_string_tuple_field(
    value: Any, field_name: str, source: str, _annotation: object
) -> tuple[str, ...]:
    return _string_tuple(value, field_name=field_name, source=source)


def _coerce_mapping_field(
    value: Any, field_name: str, source: str, _annotation: object
) -> dict[str, object]:
    return _mapping_value(value, field_name=field_name, source=source)


def _coerce_mapping_tuple_field(
    value: Any, field_name: str, source: str, _annotation: object
) -> tuple[dict[str, object], ...]:
    return _mapping_tuple(value, field_name=field_name, source=source)


def _coerce_optional_string_field(
    value: Any, field_name: str, source: str, _annotation: object
) -> str | None:
    return None if value is None else _string_value(value, field_name=field_name, source=source)


def _coerce_optional_dataclass_field(
    value: Any, field_name: str, source: str, annotation: object
) -> object | None:
    if value is None:
        return None
    dataclass_type = _optional_inner(annotation)
    return _coerce_dataclass_value(
        value,
        field_name=field_name,
        source=source,
        dataclass_type=dataclass_type,
    )


def _coerce_dataclass_field(value: Any, field_name: str, source: str, annotation: object) -> object:
    return _coerce_dataclass_value(
        value,
        field_name=field_name,
        source=source,
        dataclass_type=annotation,
    )


def _coerce_dataclass_value(
    value: Any,
    *,
    field_name: str,
    source: str,
    dataclass_type: type,
) -> object:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} field {field_name!r} must be a mapping.")
    return dataclass_config_from_mapping(
        dataclass_type,
        value,
        source=f"{source} field {field_name!r}",
    )


def _accepts_int(annotation: object) -> bool:
    return annotation is int or _is_optional_int(annotation)


def _accepts_float(annotation: object) -> bool:
    return annotation is float or _is_optional_float(annotation)


def _is_optional_int(annotation: object) -> bool:
    return set(get_args(annotation)) == {int, type(None)}


def _is_optional_float(annotation: object) -> bool:
    return set(get_args(annotation)) == {float, type(None)}


def _is_optional_str(annotation: object) -> bool:
    return set(get_args(annotation)) == {str, type(None)}


def _is_optional_dataclass(annotation: object) -> bool:
    args = set(get_args(annotation))
    if type(None) not in args or len(args) != 2:
        return False
    return is_dataclass(next(arg for arg in args if arg is not type(None)))


def _optional_inner(annotation: object) -> type:
    return next(arg for arg in get_args(annotation) if arg is not type(None))


def _is_string_tuple(annotation: object) -> bool:
    return get_origin(annotation) is tuple and get_args(annotation) == (str, Ellipsis)


def _is_mapping(annotation: object) -> bool:
    return get_origin(annotation) in {dict, Mapping}


def _is_mapping_tuple(annotation: object) -> bool:
    return get_origin(annotation) is tuple and get_args(annotation) in {
        (dict[str, object], Ellipsis),
        (dict[str, Any], Ellipsis),
        (Mapping[str, object], Ellipsis),
        (Mapping[str, Any], Ellipsis),
    }


def _string_value(value: Any, *, field_name: str, source: str) -> str:
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{source} field {field_name!r} must be a non-empty string.")
        return value
    if field_name == "memory" and isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise ValueError(f"{source} field {field_name!r} must be a non-empty string.")


def _int_value(
    value: Any,
    *,
    field_name: str,
    source: str,
    optional: bool,
) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source} field {field_name!r} must be an integer.")
    if field_name.endswith(("port", "_port")):
        if not 1 <= value <= 65535:
            raise ValueError(f"{source} field {field_name!r} must be between 1 and 65535.")
        return value
    if value < 1:
        raise ValueError(f"{source} field {field_name!r} must be a positive integer.")
    return value


def _bool_value(value: Any, *, field_name: str, source: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{source} field {field_name!r} must be a boolean.")
    return value


def _float_value(
    value: Any,
    *,
    field_name: str,
    source: str,
    optional: bool,
) -> float | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{source} field {field_name!r} must be a finite number.")
    return float(value)


def _string_tuple(value: Any, *, field_name: str, source: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{source} field {field_name!r} must be a list of non-empty strings.")
    items = tuple(value)
    for item in items:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{source} field {field_name!r} must be a list of non-empty strings.")
    return items


def _mapping_value(value: Any, *, field_name: str, source: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} field {field_name!r} must be a mapping.")
    return dict(value)


def _mapping_tuple(value: Any, *, field_name: str, source: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{source} field {field_name!r} must be a list of mappings.")
    items: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{source} field {field_name!r} must be a list of mappings.")
        items.append(dict(item))
    return tuple(items)
