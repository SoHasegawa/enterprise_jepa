"""Path and generated-name helpers for effective launch plans."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from remote_inference_launcher.summaries import safe_label

ENDPOINT_LABEL_LIMIT = 48


@dataclass(frozen=True)
class EffectivePath:
    """One normalized local or remote path used by an effective launch plan."""

    label: str
    raw: str
    resolved: str
    location: str
    required: bool
    writable: bool
    expansion: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return dataclasses.asdict(self)


def optional_local_path(
    label: str,
    value: str,
    *,
    required: bool,
    writable: bool,
) -> EffectivePath | None:
    """Return a local path record only when the value is path-like."""

    if not should_record_path(value):
        return None
    return normalize_local_path(label, value, required=required, writable=writable)


def optional_remote_path(
    label: str,
    value: str,
    *,
    required: bool,
    writable: bool,
    remote_home: str,
) -> EffectivePath | None:
    """Return a remote path record only when the value is path-like."""

    if not should_record_path(value):
        return None
    return normalize_remote_path(
        label,
        value,
        required=required,
        writable=writable,
        remote_home=remote_home,
    )


def without_empty_paths(paths: Mapping[str, EffectivePath | None]) -> dict[str, EffectivePath]:
    """Drop unset optional paths from a path mapping."""

    return {label: path for label, path in paths.items() if path is not None}


def normalize_local_path(
    label: str,
    value: str,
    *,
    required: bool,
    writable: bool,
    generated: bool = False,
) -> EffectivePath:
    """Normalize a local path and record how expansion happened."""

    if generated:
        expansion = "generated"
        resolved = str(Path(value).expanduser())
    elif value == "~" or value.startswith("~/"):
        expansion = "local_expanduser"
        resolved = str(Path(value).expanduser())
    else:
        expansion = "none"
        resolved = value
    return EffectivePath(
        label=label,
        raw=value,
        resolved=resolved,
        location="local",
        required=required,
        writable=writable,
        expansion=expansion,
    )


def normalize_remote_path(
    label: str,
    value: str,
    *,
    required: bool,
    writable: bool,
    remote_home: str,
    generated: bool = False,
) -> EffectivePath:
    """Normalize a remote path and record remote-home expansion state."""

    expansion = "none"
    resolved = value
    if value.startswith("~") and value not in {"~"} and not value.startswith("~/"):
        raise ValueError(f"Unsupported remote path for {label}: {value!r}.")
    if value == "~" or value.startswith("~/"):
        if remote_home:
            resolved = remote_home if value == "~" else f"{remote_home.rstrip('/')}/{value[2:]}"
            expansion = "remote_home"
        else:
            resolved = value
            expansion = "deferred_remote_home"
    elif generated:
        expansion = "generated"
    elif value and not value.startswith("/"):
        raise ValueError(f"Remote path for {label} must be absolute or start with '~': {value!r}.")
    return EffectivePath(
        label=label,
        raw=value,
        resolved=resolved,
        location="remote",
        required=required,
        writable=writable,
        expansion=expansion,
    )


def remote_out_dir_path(
    *,
    label: str,
    configured_out_dir: str,
    configured_root: str,
    default_root: str,
    model: str,
    endpoint_label: str,
    run_id: str,
    remote_home: str,
) -> EffectivePath:
    """Return the effective generated or explicit remote output directory."""

    if configured_out_dir:
        return normalize_remote_path(
            label,
            configured_out_dir,
            required=False,
            writable=True,
            remote_home=remote_home,
        )
    root = configured_root or f"~/{default_root}"
    root_path = normalize_remote_path(
        "remote_out_dir_root",
        root,
        required=False,
        writable=True,
        remote_home=remote_home,
        generated=not configured_root,
    )
    raw = join_remote_path(root, model_label(model), f"{endpoint_label}-{run_id}")
    resolved = join_remote_path(
        root_path.resolved,
        model_label(model),
        f"{endpoint_label}-{run_id}",
    )
    return EffectivePath(
        label=label,
        raw=raw,
        resolved=resolved,
        location="remote",
        required=False,
        writable=True,
        expansion=root_path.expansion,
    )


def unique_endpoint_labels(names: list[str]) -> tuple[str, ...]:
    """Return unique filesystem-safe endpoint labels."""

    bases = [safe_endpoint_label(name) for name in names]
    duplicate_bases = {base for base in bases if bases.count(base) > 1}
    labels: list[str] = []
    for index, (name, base) in enumerate(zip(names, bases, strict=True)):
        if base not in duplicate_bases:
            labels.append(base)
            continue
        digest = hashlib.sha1(f"{index}:{name}".encode()).hexdigest()[:8]
        suffix = f"-{digest}"
        labels.append(f"{base[: ENDPOINT_LABEL_LIMIT - len(suffix)]}{suffix}")
    return tuple(labels)


def safe_endpoint_label(value: str) -> str:
    """Return the base endpoint label before collision disambiguation."""

    return safe_label(value)[:ENDPOINT_LABEL_LIMIT] or "default"


def generated_slurm_job_name(prefix: str, endpoint_label: str, run_id: str) -> str:
    """Return the generated Slurm job name for an endpoint."""

    safe_prefix = safe_label(prefix)
    run_rand = run_random_suffix(run_id)
    base = f"{safe_prefix}-{endpoint_label}-{run_rand}"
    return base[:64]


def run_random_suffix(run_id: str) -> str:
    """Return the random suffix portion of a launcher run ID."""

    suffix = run_id.rsplit("-", 1)[-1]
    return "".join(char for char in suffix if char.isalnum())[:8] or "00000000"


def model_label(model: str) -> str:
    """Return a filesystem-safe model label."""

    return safe_label(model.replace("/", "-"))[:64] or "model"


def join_remote_path(*parts: str) -> str:
    """Join POSIX path fragments without changing '~' semantics."""

    cleaned: list[str] = []
    for index, part in enumerate(parts):
        if not part:
            continue
        if index == 0:
            cleaned.append(part.rstrip("/"))
        else:
            cleaned.append(part.strip("/"))
    if not cleaned:
        return ""
    if cleaned[0] in {"", "/"}:
        return "/" + "/".join(cleaned[1:])
    return "/".join(cleaned)


def should_record_path(value: str) -> bool:
    """Return whether a config string should be treated as a path."""

    if not value:
        return False
    if value == "~" or value.startswith(("/", "~/", "./", "../")):
        return True
    path = PurePosixPath(value)
    return len(path.parts) > 1 and value.count("/") != 1
