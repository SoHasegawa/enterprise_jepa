from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


@dataclass(slots=True)
class BenchmarkComponentVersions:
    """ベンチマークを構成する各コンポーネントの版情報。"""

    benchmark_version: str | None
    green_agent_version: str | None
    purple_agent_version: str | None
    executor_versions: dict[str, str | None]


def normalize_semver(raw_value: str, *, source: str) -> str:
    """semver 文字列を正規化し、形式不正なら例外を送出する。"""
    normalized = raw_value.strip()
    if not normalized:
        raise ValueError(f"Empty semantic version: {source}")
    if not SEMVER_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid semantic version '{raw_value}' from {source}")
    return normalized


def normalize_optional_semver(raw_value: Any, *, source: str) -> str | None:
    """未設定を許容しつつ、設定済みなら semver として検証する。"""
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ValueError(f"Semantic version must be a string in {source}")

    normalized = raw_value.strip()
    if not normalized:
        return None
    return normalize_semver(normalized, source=source)


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_config = tomllib.load(handle)
    if not isinstance(raw_config, dict):
        raise ValueError(f"TOML root must be a table: {path}")
    return raw_config


def load_benchmark_version(benchmark_dir: Path) -> str | None:
    """benchmark.toml から benchmark 自体の版情報を読む。"""
    config_path = benchmark_dir / "benchmark.toml"
    if not config_path.exists():
        return None
    raw_config = _load_toml(config_path)
    return normalize_optional_semver(
        raw_config.get("version"),
        source=str(config_path),
    )


def load_component_version(component_dir: Path, *, env_name: str | None = None) -> str | None:
    """component dir 直下の version.toml / pyproject.toml から版情報を読む。"""
    if env_name:
        env_value = os.getenv(env_name)
        normalized_env = normalize_optional_semver(
            env_value,
            source=f"environment:{env_name}",
        )
        if normalized_env is not None:
            return normalized_env

    version_toml_path = component_dir / "version.toml"
    if version_toml_path.exists():
        raw_config = _load_toml(version_toml_path)
        return normalize_optional_semver(
            raw_config.get("version"),
            source=str(version_toml_path),
        )

    pyproject_path = component_dir / "pyproject.toml"
    if pyproject_path.exists():
        raw_config = _load_toml(pyproject_path)
        project = raw_config.get("project")
        if not isinstance(project, Mapping):
            raise ValueError(f"Missing [project] table in {pyproject_path}")
        return normalize_optional_semver(
            project.get("version"),
            source=str(pyproject_path),
        )

    return None


def resolve_benchmark_component_versions(
    benchmark_dir: Path,
    *,
    executor_names: list[str] | None = None,
) -> BenchmarkComponentVersions:
    """ベンチマーク配下の green / purple / executor 版情報をまとめて返す。"""
    executor_versions: dict[str, str | None] = {}
    for executor_name in executor_names or []:
        executor_versions[executor_name] = load_component_version(
            benchmark_dir / "purple-executors" / executor_name,
        )

    return BenchmarkComponentVersions(
        benchmark_version=load_benchmark_version(benchmark_dir),
        green_agent_version=load_component_version(benchmark_dir / "green"),
        purple_agent_version=load_component_version(benchmark_dir / "purple"),
        executor_versions=executor_versions,
    )


def format_executor_versions(executor_versions: Mapping[str, str | None]) -> str:
    """executor 名と版情報の対応を 1 行表示向けに整形する。"""
    if not executor_versions:
        return "—"
    return ", ".join(
        f"{executor_name}@{version or '—'}" for executor_name, version in executor_versions.items()
    )


def format_component_version_summary(
    *,
    green_agent_version: str | None,
    purple_agent_version: str | None,
    executor_version: str | None,
) -> str:
    """実行結果表示向けに green / purple / executor の版情報を整形する。"""
    return (
        f"G={green_agent_version or '—'} "
        f"P={purple_agent_version or '—'} "
        f"E={executor_version or '—'}"
    )
