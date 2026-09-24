from __future__ import annotations

# ruff: noqa: E402
import asyncio
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

load_dotenv()


def _absolute_cli_path(value: str | None) -> str | None:
    """Resolve path-like CLI options before child processes can change cwd."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())

SRC_DIR = Path(__file__).resolve().parents[1]
SRC_DIR_STR = str(SRC_DIR)
if SRC_DIR_STR not in sys.path:
    sys.path.insert(0, SRC_DIR_STR)

from common.executor_runtime import (
    format_executor_runtime_models,
    format_executor_runtime_notes,
    resolve_executor_runtime_payload,
)
from common.inference_runtime import prepare_inference_runtime_config
from common.inference_runtime import (
    translate_termination_signals as _translate_termination_signals,
)
from common.logging_utils import configure_logging, get_logger
from common.network_env import ensure_no_proxy
from common.storage import (
    default_result_root,
)
from common.versioning import format_executor_versions, resolve_benchmark_component_versions

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
DEFAULT_ASSETS_ROOT = Path(
    os.getenv(
        "BENCHMARK_ASSETS_ROOT",
        str(REPO_ROOT / "assets"),
    )
)
DEFAULT_RESULT_ROOT = Path(
    os.getenv(
        "BENCHMARK_RESULT_ROOT",
        str(default_result_root()),
    )
)

configure_logging()
LOGGER = get_logger(__name__)
console = Console()
TABLE_HEADER_STYLE = "bold cyan"
RESULT_MANIFEST_FILENAME = "manifest.json"
RUN_ID_COLUMN = "Run ID"
PASS_AT_K_LABEL = "PASS@k"


def _load_cli_version_from_pyproject() -> str | None:
    """repo root の pyproject.toml から CLI version を読む。"""
    if not PYPROJECT_PATH.exists():
        return None

    with PYPROJECT_PATH.open("rb") as handle:
        raw_config = tomllib.load(handle)

    project = raw_config.get("project")
    if not isinstance(project, dict):
        return None

    raw_version = project.get("version")
    if not isinstance(raw_version, str):
        return None

    normalized = raw_version.strip()
    return normalized or None


def _resolve_cli_version() -> str:
    """`ejepa --version` で表示する CLI version を解決する。"""
    return _load_cli_version_from_pyproject() or "unknown"


def _version_callback(value: bool) -> None:
    """`--version` が指定されたら version を表示して終了する。"""
    if not value:
        return
    typer.echo(_resolve_cli_version())
    raise typer.Exit()


class OutputFormat(str, Enum):
    """CLI 出力形式。"""

    table = "table"
    json = "json"


class ResultStatusFilter(str, Enum):
    """結果一覧で使う status フィルタ。"""

    all = "all"
    completed = "completed"
    failed = "failed"


class ExecutionLauncher(str, Enum):
    """`bench run` の実行経路。"""

    local = "local"
    slurm = "slurm"


@dataclass(slots=True)
class AppState:
    """CLI 全体で共有するルートパス。"""

    assets_root: Path
    result_root: Path


@dataclass(slots=True)
class BenchmarkRecord:
    """`benchmark.toml` から引いたベンチマーク要約。"""

    name: str
    version: str | None
    benchmark_dir: Path
    config_path: Path
    default_executor: str | None
    available_executors: list[str]
    default_target: str | None
    targets: list[str]
    task_counts: dict[str, int]
    roles: list[str]
    green_entrypoint: str | None
    participant_entrypoints: dict[str, str]
    green_agent_version: str | None = None
    purple_agent_version: str | None = None
    executor_versions: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "benchmark_dir": str(self.benchmark_dir),
            "config_path": str(self.config_path),
            "default_executor": self.default_executor,
            "available_executors": self.available_executors,
            "default_target": self.default_target,
            "targets": self.targets,
            "task_counts": self.task_counts,
            "roles": self.roles,
            "green_entrypoint": self.green_entrypoint,
            "participant_entrypoints": self.participant_entrypoints,
            "green_agent_version": self.green_agent_version,
            "purple_agent_version": self.purple_agent_version,
            "executor_versions": self.executor_versions,
        }


@dataclass(slots=True)
class SlurmSubmissionResult:
    """Slurm 送信の実行結果。"""

    command: list[str]
    script_path: Path
    stdout: str
    stderr: str
    job_id: str | None


@dataclass(slots=True)
class RuntimeConfigOverrides:
    """`bench run` の benchmark config 上書き指定。"""

    target: str | None
    config_overrides: list[str] | None
    task_ids: list[str] | None
    max_parallel: int | None
    vllm_model_id: str | None
    green_host: str | None
    green_port: int | None
    purple_host: str | None
    purple_port: int | None


@dataclass(slots=True)
class SlurmOptions:
    """`sbatch` へ渡す任意 option 群。"""

    partition: str | None
    job_name: str | None
    output: str | None
    time_limit: str | None
    mem: str | None
    cpus_per_task: int | None
    gpus: int | None
    account: str | None
    qos: str | None
    constraint: str | None
    exclude: str | None
    extra_args: list[str]


@dataclass(slots=True)
class BenchmarkRunContext:
    """`bench run` の実行前に解決済みの共通情報。"""

    state: AppState
    record: BenchmarkRecord
    benchmark_dir: Path
    raw_config: dict[str, Any]
    runtime_config: dict[str, Any]
    executor_name: str
    launcher: ExecutionLauncher
    workdir: Path
    ready_timeout: int
    show_logs: bool
    serve_only: bool
    inference_config: Path | None


@dataclass(slots=True)
class AgentLaunchSpec:
    """local launcher で解決する agent endpoint 指定。"""

    config: Mapping[str, Any]
    label: str
    entrypoint: Path
    override_env_names: list[str]


@dataclass(slots=True)
class ProcessEnvConfig:
    """benchmark subprocess に渡す環境変数の入力値。"""

    benchmark_name: str
    benchmark_version: str | None
    target: str
    benchmark_dir: Path
    assets_root: Path
    executor_name: str | None
    request_config: dict[str, Any] | None
    result_root: Path
    result_dir: Path
    run_id: str
    user_name: str
    config_hash: str


app = typer.Typer(
    name="ejepa",
    help="ベンチマーク定義の閲覧と実行、結果探索を行う CLI。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
benchmark_app = typer.Typer(no_args_is_help=True, help="ベンチマーク定義の一覧・検索・実行")
result_app = typer.Typer(no_args_is_help=True, help="保存済み結果の一覧・検索・詳細表示")
app.add_typer(benchmark_app, name="bench")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(result_app, name="result")


@app.callback()
def main_callback(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="pyproject.toml に記載された CLI version を表示する。",
        ),
    ] = False,
    assets_root: Annotated[
        Path,
        typer.Option(
            "--assets-root",
            help="ベンチマーク asset を探索するルートディレクトリ。",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = DEFAULT_ASSETS_ROOT,
    result_root: Annotated[
        Path,
        typer.Option(
            "--result-root",
            help="Green Agent が結果を書き出すルートディレクトリ。",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = DEFAULT_RESULT_ROOT,
) -> None:
    """ルートオプションを保持する。"""
    ctx.obj = AppState(
        assets_root=assets_root.expanduser().resolve(),
        result_root=result_root.expanduser().resolve(),
    )


def _state_from_ctx(ctx: typer.Context) -> AppState:
    """Typer context から共有状態を取り出す。"""
    if not isinstance(ctx.obj, AppState):
        raise RuntimeError("CLI state is not initialized")
    return ctx.obj


def _json_dumps(value: Any) -> str:
    """Path を含む値も JSON 表示しやすく整形する。"""
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _relative_path_text(path: Path, root: Path) -> str:
    """root 配下なら相対パス、それ以外は絶対パスで返す。"""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _load_benchmark_config(assets_root: Path, benchmark_name: str) -> tuple[Path, dict[str, Any]]:
    """ベンチマーク設定ファイルを読み込む。"""
    benchmark_dir = assets_root / benchmark_name
    config_path = benchmark_dir / "benchmark.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"benchmark.toml not found: {config_path}")
    with config_path.open("rb") as handle:
        raw_config = tomllib.load(handle)
    return benchmark_dir, raw_config


def _load_task_counts(benchmark_dir: Path) -> dict[str, int]:
    """`green/tasks/task_ids.toml` があれば target ごとの件数を読む。"""
    task_ids_path = benchmark_dir / "green" / "tasks" / "task_ids.toml"
    if not task_ids_path.exists():
        return {}
    with task_ids_path.open("rb") as handle:
        raw_task_ids = tomllib.load(handle)
    task_counts: dict[str, int] = {}
    for target, task_ids in raw_task_ids.items():
        if isinstance(task_ids, list):
            task_counts[str(target)] = len(task_ids)
    return task_counts


def _load_declared_targets(raw_config: dict[str, Any]) -> list[str]:
    """`benchmark.toml` 上で明示宣言された target 名を読む。"""
    raw_metadata = raw_config.get("benchmark_metadata")
    if not isinstance(raw_metadata, dict):
        return []

    raw_targets = raw_metadata.get("targets")
    if not isinstance(raw_targets, list):
        return []

    declared_targets: list[str] = []
    seen: set[str] = set()
    for item in raw_targets:
        target = str(item).strip()
        if target and target not in seen:
            declared_targets.append(target)
            seen.add(target)
    return declared_targets


def _discover_purple_executor_dirs(benchmark_dir: Path) -> set[str]:
    discovered: set[str] = set()
    purple_executors_dir = benchmark_dir / "purple-executors"
    if not purple_executors_dir.exists():
        return discovered
    for candidate in purple_executors_dir.iterdir():
        if candidate.is_dir() and (candidate / "executor.py").exists():
            discovered.add(candidate.name)
    return discovered


def _discover_slurm_executors(raw_config: dict[str, Any]) -> set[str]:
    raw_slurm_config = raw_config.get("slurm")
    if not isinstance(raw_slurm_config, dict):
        return set()
    executor_scripts = raw_slurm_config.get("executors")
    if not isinstance(executor_scripts, dict):
        return set()
    return {
        str(executor_name)
        for executor_name, script_path in executor_scripts.items()
        if isinstance(script_path, str) and script_path.strip()
    }


def _ordered_executors(discovered: set[str], default_executor: Any) -> list[str]:
    ordered: list[str] = []
    if isinstance(default_executor, str) and default_executor.strip() in discovered:
        ordered.append(default_executor.strip())
    for executor_name in sorted(discovered):
        if executor_name not in ordered:
            ordered.append(executor_name)
    return ordered


def _discover_available_executors(benchmark_dir: Path, raw_config: dict[str, Any]) -> list[str]:
    """`bench run --executor` で選べる executor 名を集約する。"""
    discovered = _discover_purple_executor_dirs(benchmark_dir)
    discovered.update(_discover_slurm_executors(raw_config))

    default_executor = raw_config.get("default_executor")
    if isinstance(default_executor, str) and default_executor.strip():
        discovered.add(default_executor.strip())

    return _ordered_executors(discovered, default_executor)


def _benchmark_targets(raw_config: dict[str, Any], task_counts: Mapping[str, int]) -> list[str]:
    default_target = raw_config.get("config", {}).get("target")
    targets = _load_declared_targets(raw_config)
    for target in task_counts:
        if target not in targets:
            targets.append(target)
    if isinstance(default_target, str) and default_target and default_target not in targets:
        targets.insert(0, default_target)
    if not targets and isinstance(default_target, str) and default_target:
        return [default_target]
    return targets


def _participant_entrypoints(raw_config: dict[str, Any]) -> dict[str, str]:
    participants = raw_config.get("participants", [])
    return {
        str(participant.get("role")): str(participant.get("entrypoint"))
        for participant in participants
        if participant.get("role") and participant.get("entrypoint")
    }


def _benchmark_record_from_config(
    benchmark_dir: Path,
    config_path: Path,
    raw_config: dict[str, Any],
) -> BenchmarkRecord:
    available_executors = _discover_available_executors(benchmark_dir, raw_config)
    task_counts = _load_task_counts(benchmark_dir)
    targets = _benchmark_targets(raw_config, task_counts)
    default_target = raw_config.get("config", {}).get("target")
    participant_entrypoints = _participant_entrypoints(raw_config)
    component_versions = resolve_benchmark_component_versions(
        benchmark_dir,
        executor_names=available_executors,
    )
    return BenchmarkRecord(
        name=str(raw_config.get("name") or benchmark_dir.name),
        version=(
            str(raw_config["version"])
            if isinstance(raw_config.get("version"), str) and raw_config["version"].strip()
            else None
        ),
        benchmark_dir=benchmark_dir,
        config_path=config_path,
        default_executor=(
            str(raw_config["default_executor"])
            if raw_config.get("default_executor") is not None
            else None
        ),
        available_executors=available_executors,
        default_target=str(default_target) if default_target is not None else None,
        targets=targets,
        task_counts=task_counts,
        roles=sorted(participant_entrypoints),
        green_entrypoint=(
            str(raw_config.get("green_agent", {}).get("entrypoint"))
            if raw_config.get("green_agent", {}).get("entrypoint") is not None
            else None
        ),
        participant_entrypoints=participant_entrypoints,
        green_agent_version=component_versions.green_agent_version,
        purple_agent_version=component_versions.purple_agent_version,
        executor_versions=component_versions.executor_versions,
    )


def _discover_benchmarks(assets_root: Path) -> list[BenchmarkRecord]:
    """assets 配下の `benchmark.toml` を走査する。"""
    if not assets_root.exists():
        raise FileNotFoundError(f"assets root not found: {assets_root}")

    records: list[BenchmarkRecord] = []
    for candidate in sorted(assets_root.iterdir()):
        config_path = candidate / "benchmark.toml"
        if not candidate.is_dir() or not config_path.exists():
            continue

        with config_path.open("rb") as handle:
            raw_config = tomllib.load(handle)

        records.append(_benchmark_record_from_config(candidate, config_path, raw_config))
    return records


def _benchmark_search_blob(record: BenchmarkRecord) -> str:
    """検索しやすいように主要フィールドを 1 行へ畳み込む。"""
    values = [
        record.name,
        record.version or "",
        record.benchmark_dir.name,
        record.default_executor or "",
        *record.available_executors,
        record.default_target or "",
        *record.targets,
        *record.roles,
        *record.participant_entrypoints.values(),
        str(record.benchmark_dir),
    ]
    return " ".join(values).lower()


def _filter_benchmarks(
    records: Iterable[BenchmarkRecord], query: str | None
) -> list[BenchmarkRecord]:
    """クエリ指定があればベンチマーク一覧を絞り込む。"""
    if not query:
        return list(records)
    normalized_query = query.lower()
    return [record for record in records if normalized_query in _benchmark_search_blob(record)]


def _resolve_benchmark(records: list[BenchmarkRecord], name: str) -> BenchmarkRecord:
    """名前またはディレクトリ名からベンチマークを 1 件解決する。"""
    lowered = name.lower()
    exact_matches = [
        record
        for record in records
        if record.name.lower() == lowered or record.benchmark_dir.name.lower() == lowered
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise typer.BadParameter(f"Ambiguous benchmark name: {name}")

    partial_matches = [
        record
        for record in records
        if lowered in record.name.lower() or lowered in record.benchmark_dir.name.lower()
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]
    raise typer.BadParameter(
        f"Benchmark not found: {name}. Available: {', '.join(record.name for record in records)}"
    )


def _targets_summary(record: BenchmarkRecord) -> str:
    """target と件数を簡潔に表す。"""
    if not record.targets:
        return "—"
    labels: list[str] = []
    for target in record.targets:
        count = record.task_counts.get(target)
        labels.append(f"{target}({count})" if count is not None else target)
    return ", ".join(labels)


def _executor_label(record: BenchmarkRecord) -> str:
    if record.executor_versions:
        return format_executor_versions(record.executor_versions)
    if record.available_executors:
        return ", ".join(record.available_executors)
    return "—"


def _render_benchmark_table(records: list[BenchmarkRecord], assets_root: Path) -> None:
    """ベンチマーク一覧を表形式で表示する。"""
    table = Table(show_header=True, header_style=TABLE_HEADER_STYLE)
    table.add_column("Benchmark")
    table.add_column("Version")
    table.add_column("Green")
    table.add_column("Purple")
    table.add_column("Executors")
    table.add_column("Default Exec")
    table.add_column("Default Target")
    table.add_column("Targets")
    table.add_column("Roles")
    table.add_column("Path")
    for record in records:
        executor_label = _executor_label(record)
        table.add_row(
            record.name,
            record.version or "—",
            record.green_agent_version or "—",
            record.purple_agent_version or "—",
            executor_label,
            record.default_executor or "—",
            record.default_target or "—",
            _targets_summary(record),
            ", ".join(record.roles) if record.roles else "—",
            _relative_path_text(record.benchmark_dir, assets_root),
        )
    console.print(table)
    console.print(f"\nTotal benchmarks: {len(records)}")


def _build_endpoint(host: str, port: int) -> str:
    """ホストとポートから HTTP エンドポイントを組み立てる。"""
    return f"http://{host}:{port}"


def _should_start_agent_process(host: str, port: int) -> bool:
    """local bind address なら agent subprocess を起動し、remote endpoint なら既存 service を使う。"""
    normalized_host = host.strip().lower()
    local_hosts = {"", "127.0.0.1", "localhost", "0.0.0.0", "::1", "::"}
    return port == 0 or normalized_host in local_hosts


def _parse_config_override(raw_override: str) -> tuple[str, Any]:
    """`--config key=value` を request config へ入れる値に変換する。"""
    if "=" not in raw_override:
        raise ValueError(f"--config must be KEY=VALUE: {raw_override}")
    key, raw_value = raw_override.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"--config key is empty: {raw_override}")

    value_text = raw_value.strip()
    if value_text == "":
        return key, ""
    try:
        value = json.loads(value_text)
    except json.JSONDecodeError:
        value = value_text
    return key, value


def _apply_agent_endpoint_overrides(
    *,
    green_config: dict[str, Any],
    participants: list[dict[str, Any]],
    overrides: RuntimeConfigOverrides,
) -> None:
    if overrides.green_host:
        green_config["host"] = overrides.green_host
    if overrides.green_port is not None:
        green_config["port"] = overrides.green_port
    if overrides.purple_host:
        participants[0]["host"] = overrides.purple_host
    if overrides.purple_port is not None:
        participants[0]["port"] = overrides.purple_port


def _apply_request_config_overrides(
    request_config: dict[str, Any],
    overrides: RuntimeConfigOverrides,
) -> None:
    for raw_override in overrides.config_overrides or []:
        key, value = _parse_config_override(raw_override)
        request_config[key] = value

    simple_overrides = {
        "task_ids": overrides.task_ids,
        "max_parallel": overrides.max_parallel,
    }
    for key, value in simple_overrides.items():
        if value is not None and value != []:
            request_config[key] = value

    if isinstance(overrides.vllm_model_id, str) and overrides.vllm_model_id.strip():
        request_config["vllm_model_id"] = overrides.vllm_model_id.strip()

def _build_runtime_config(
    raw_config: dict[str, Any],
    *,
    overrides: RuntimeConfigOverrides,
) -> dict[str, Any]:
    """設定ファイルと CLI 上書きを統合した実行時設定を作る。"""
    green_config = dict(raw_config.get("green_agent", {}))
    if not green_config:
        raise ValueError("green_agent is required in benchmark.toml")
    green_config["host"] = str(green_config.get("host") or "127.0.0.1")
    green_config["port"] = int(green_config.get("port") or 0)

    participants = [dict(participant) for participant in raw_config.get("participants", [])]
    if not participants:
        raise ValueError("participants are required in benchmark.toml")
    for participant in participants:
        participant["host"] = str(participant.get("host") or "127.0.0.1")
        participant["port"] = int(participant.get("port") or 0)

    _apply_agent_endpoint_overrides(
        green_config=green_config,
        participants=participants,
        overrides=overrides,
    )

    request_config = dict(raw_config.get("config", {}))
    if overrides.target:
        request_config["target"] = overrides.target
    elif "target" not in request_config:
        raise ValueError("config.target is required in benchmark.toml or CLI")
    _apply_request_config_overrides(request_config, overrides)

    return {
        "default_executor": raw_config.get("default_executor"),
        "green_agent": green_config,
        "participants": participants,
        "config": request_config,
    }


def _attach_runtime_green_endpoint(
    request_config: Mapping[str, Any] | None,
    *,
    green_endpoint: str,
) -> dict[str, Any]:
    """Expose the resolved Green endpoint without mutating the source config."""
    resolved = dict(request_config or {})
    if green_endpoint and not str(resolved.get("green_endpoint") or "").strip():
        resolved["green_endpoint"] = green_endpoint
    return resolved


def _build_process_env(config: ProcessEnvConfig) -> dict[str, str]:
    """子プロセスへ渡す環境変数を整える。"""
    env = os.environ.copy()
    ensure_no_proxy(
        env,
        (
            env.get("OPENAI_BASE_URL"),
            env.get("OPENAI_EMBEDDING_BASE_URL"),
        ),
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["BENCHMARK_NAME"] = config.benchmark_name
    if config.benchmark_version:
        env["BENCHMARK_VERSION"] = config.benchmark_version
    env["BENCHMARK_TARGET"] = config.target
    env["BENCHMARK_DIR"] = str(config.benchmark_dir)
    env["BENCHMARK_ASSETS_ROOT"] = str(config.assets_root)
    env["BENCHMARK_RESULT_ROOT"] = str(config.result_root)
    env["BENCHMARK_RESULT_DIR"] = str(config.result_dir)
    env["BENCHMARK_RUN_ID"] = config.run_id
    env["BENCHMARK_USER_NAME"] = config.user_name
    env["BENCHMARK_CONFIG_HASH"] = config.config_hash
    if config.executor_name:
        env["BENCHMARK_EXECUTOR"] = config.executor_name

    parent_bin = str(Path(sys.executable).parent)
    env["PATH"] = parent_bin + os.pathsep + env.get("PATH", "")
    return env


def _resolve_project_python(project_dir: Path) -> Path | None:
    """uv 既定の `.venv` にある Python 実行ファイルを返す。"""
    candidates = [
        project_dir / ".venv" / "bin" / "python",
        project_dir / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _install_hint_for_project(project_dir: Path) -> str:
    """project path に対応する install helper を返す。"""
    try:
        relative = project_dir.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return "scripts/install.sh"

    component_hints = {
        ".": "scripts/install.sh cli",
        "assets/AutomationBench/green": "scripts/install.sh automationbench",
        "assets/AutomationBench/purple": "scripts/install.sh automationbench",
        "assets/crmarenapro/green": "scripts/install.sh crmarenapro",
        "assets/crmarenapro/purple": "scripts/install.sh crmarenapro",
        "assets/EnterpriseOps-Gym/green": "scripts/install.sh enterpriseops-gym",
        "assets/EnterpriseOps-Gym/purple": "scripts/install.sh enterpriseops-gym",
        "assets/Terminal-Bench-2.0/green": "scripts/install.sh terminal-bench-2.0",
        "assets/Terminal-Bench-2.0/purple": "scripts/install.sh terminal-bench-2.0",
        "assets/WorkBench/green": "scripts/install.sh workbench",
        "assets/WorkBench/purple": "scripts/install.sh workbench",
    }
    return component_hints.get(relative, "scripts/install.sh")


def _resolve_override_python(override_env_names: list[str]) -> str | None:
    for env_name in override_env_names:
        override = os.getenv(env_name)
        if not override:
            continue
        override_path = Path(override).expanduser()
        if not override_path.is_absolute():
            override_path = Path.cwd() / override_path
        return str(override_path)
    return None


def _iter_entrypoint_project_dirs(entrypoint: Path, benchmark_dir: Path) -> Iterable[Path]:
    current_dir = entrypoint.parent.resolve()
    benchmark_root = benchmark_dir.resolve()
    while True:
        yield current_dir
        if current_dir == benchmark_root:
            break
        if benchmark_root not in current_dir.parents:
            break
        current_dir = current_dir.parent


def _resolve_entrypoint_python(
    *,
    entrypoint: Path,
    benchmark_dir: Path,
    override_env_names: list[str],
) -> str:
    """entrypoint に最も近い uv project の Python を解決する。"""
    override_python = _resolve_override_python(override_env_names)
    if override_python is not None:
        return override_python

    for current_dir in _iter_entrypoint_project_dirs(entrypoint, benchmark_dir):
        pyproject_path = current_dir / "pyproject.toml"
        if pyproject_path.exists():
            project_python = _resolve_project_python(current_dir)
            if project_python is None:
                raise FileNotFoundError(
                    f"Python environment for {entrypoint} is not ready. "
                    f"Run: {_install_hint_for_project(current_dir)}"
                )
            return str(project_python)

    return sys.executable


def _build_agent_command(
    python_executable: str,
    entrypoint: Path,
    host: str,
    port: int,
    *,
    port_file: Path | None = None,
) -> list[str]:
    """Agent サーバー起動コマンドを組み立てる。"""
    command = [python_executable, str(entrypoint), "--host", host, "--port", str(port)]
    if port_file is not None:
        command.extend(["--port-file", str(port_file)])
    return command


def _agent_port_file(result_dir: Path, agent_label: str) -> Path:
    """動的割り当て port の受け渡しファイルパスを返す。"""
    safe_label = (
        "".join(
            char if char.isascii() and (char.isalnum() or char in {"-", "_"}) else "_"
            for char in agent_label
        ).strip("_")
        or "agent"
    )
    return result_dir / ".runtime" / "agent-ports" / f"{safe_label}.port"


def _read_agent_port_file(port_file: Path) -> int | None:
    if not port_file.exists():
        return None
    raw_value = port_file.read_text(encoding="utf-8").strip()
    if not raw_value:
        return None
    try:
        resolved_port = int(raw_value)
    except ValueError:
        return None
    return resolved_port if resolved_port > 0 else None


def _resolve_actual_agent_port(
    *,
    requested_port: int,
    port_file: Path | None,
    process: subprocess.Popen[str],
    timeout_seconds: int,
    agent_label: str,
) -> int:
    """固定 port または port file から実際の待受 port を解決する。"""
    if requested_port > 0:
        return requested_port
    if port_file is None:
        raise RuntimeError(f"{agent_label} requires a port file when requested port is 0")

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{agent_label} exited before publishing a dynamic port: {port_file}"
            )
        resolved_port = _read_agent_port_file(port_file)
        if resolved_port is not None:
            return resolved_port
        time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for {agent_label} port file: {port_file}")


DEFAULT_READY_TIMEOUT_SECONDS = 30


def _default_ready_timeout_for_run(benchmark_name: str, executor_name: str) -> int:
    """Default readiness timeout for a local run."""
    del benchmark_name, executor_name
    return DEFAULT_READY_TIMEOUT_SECONDS


def _default_ready_timeout_for_launcher(
    benchmark_name: str,
    executor_name: str,
    launcher: ExecutionLauncher,
) -> int:
    """Default readiness timeout, including the execution route."""
    del launcher
    return _default_ready_timeout_for_run(benchmark_name, executor_name)


def _default_slurm_exclude_for_launcher(
    benchmark_name: str,
    executor_name: str,
    launcher: ExecutionLauncher,
) -> str | None:
    """Default Slurm exclude nodelist, including the execution route."""
    del benchmark_name, executor_name, launcher
    return None


def _configured_slurm_script(
    executor_name: str,
    raw_config: dict[str, Any] | None,
) -> str | None:
    raw_slurm_config = raw_config.get("slurm") if isinstance(raw_config, dict) else None
    if not isinstance(raw_slurm_config, dict):
        return None

    executor_scripts = raw_slurm_config.get("executors")
    if isinstance(executor_scripts, dict):
        candidate = executor_scripts.get(executor_name)
        if isinstance(candidate, str) and candidate.strip():
            return candidate

    candidate = raw_slurm_config.get("script")
    if isinstance(candidate, str) and candidate.strip():
        return candidate
    return None


def _resolve_script_path(raw_script: str | None, benchmark_dir: Path) -> Path:
    if raw_script:
        script_path = Path(raw_script.strip()).expanduser()
        if not script_path.is_absolute():
            return benchmark_dir / script_path
        return script_path
    return REPO_ROOT / "slurm" / "run-benchmark.sbatch.sh"


def _resolve_slurm_run_script(
    *,
    benchmark_name: str,
    executor_name: str,
    benchmark_dir: Path,
    raw_config: dict[str, Any] | None = None,
) -> Path:
    """benchmark 設定付き sbatch script、または共通 wrapper を返す。"""
    del benchmark_name

    script_path = _resolve_script_path(
        _configured_slurm_script(executor_name, raw_config),
        benchmark_dir,
    )
    if not script_path.exists():
        raise FileNotFoundError(f"Slurm run script not found: {script_path}")
    return script_path.resolve()


def _build_slurm_passthrough_args(
    *,
    runtime_config: dict[str, Any],
    workdir: Path,
    show_logs: bool,
    serve_only: bool,
    inference_config: Path | None,
) -> list[str]:
    """組み込み sbatch script へ渡す `ejepa bench run` 引数を返す。"""
    args = [
        "--workdir",
        str(workdir),
        "--target",
        str(runtime_config["config"]["target"]),
        "--green-host",
        str(runtime_config["green_agent"]["host"]),
        "--green-port",
        str(int(runtime_config["green_agent"]["port"])),
    ]

    if runtime_config["participants"]:
        primary_participant = runtime_config["participants"][0]
        args.extend(
            [
                "--purple-host",
                str(primary_participant["host"]),
                "--purple-port",
                str(int(primary_participant["port"])),
            ]
        )

    vllm_model_id = runtime_config["config"].get("vllm_model_id")
    if isinstance(vllm_model_id, str) and vllm_model_id.strip():
        args.extend(["--vllm-model-id", vllm_model_id.strip()])

    max_parallel = runtime_config["config"].get("max_parallel")
    if max_parallel is not None:

        args.extend(["--max-parallel", str(int(max_parallel))])

    for task_id in runtime_config["config"].get("task_ids", []) or []:
        args.extend(["--task-id", str(task_id)])

    if show_logs:
        args.append("--show-logs")
    if serve_only:
        args.append("--serve-only")
    if inference_config is not None:
        args.extend(["--inference-config", str(inference_config)])
    return args


def _build_sbatch_command(
    *,
    sbatch_bin: str,
    script_path: Path,
    script_args: list[str],
    slurm_options: SlurmOptions,
) -> list[str]:
    """`sbatch` 実行コマンドを組み立てる。"""
    command = [sbatch_bin, "--parsable"]

    def add_option(flag: str, value: str | int | None) -> None:
        if value is None or value == "":
            return
        command.extend([flag, str(value)])

    add_option("--partition", slurm_options.partition)
    add_option("--job-name", slurm_options.job_name)
    add_option("--output", slurm_options.output)
    add_option("--time", slurm_options.time_limit)
    add_option("--mem", slurm_options.mem)
    add_option("--cpus-per-task", slurm_options.cpus_per_task)
    add_option("--account", slurm_options.account)
    add_option("--qos", slurm_options.qos)
    add_option("--constraint", slurm_options.constraint)
    add_option("--exclude", slurm_options.exclude)
    if slurm_options.gpus is not None:
        command.extend(["--gres", f"gpu:{slurm_options.gpus}"])

    command.extend(slurm_options.extra_args)
    command.append(str(script_path))
    command.extend(script_args)
    return command


def _parse_sbatch_job_id(stdout: str) -> str | None:
    """`sbatch --parsable` 出力から job id を抜き出す。"""
    stripped = stdout.strip()
    if not stripped:
        return None
    last_line = stripped.splitlines()[-1].strip()
    token = last_line.split(";", 1)[0].strip()
    if token.isdigit():
        return token

    match = re.search(r"\b(\d+)\b", last_line)
    if match:
        return match.group(1)
    return None


def _submit_benchmark_run_via_slurm(
    *,
    run_context: BenchmarkRunContext,
    slurm_options: SlurmOptions,
) -> SlurmSubmissionResult:
    """benchmark 設定付き、または共通 sbatch wrapper で Slurm へ送信する。"""
    sbatch_bin = shutil.which("sbatch")
    if sbatch_bin is None:
        raise FileNotFoundError("sbatch not found in PATH.")

    script_path = _resolve_slurm_run_script(
        benchmark_name=run_context.record.name,
        executor_name=run_context.executor_name,
        benchmark_dir=run_context.benchmark_dir,
        raw_config=run_context.raw_config,
    )
    script_args = _build_slurm_passthrough_args(
        runtime_config=run_context.runtime_config,
        workdir=run_context.workdir,
        show_logs=run_context.show_logs,
        serve_only=run_context.serve_only,
        inference_config=run_context.inference_config,
    )
    command = _build_sbatch_command(
        sbatch_bin=sbatch_bin,
        script_path=script_path,
        script_args=script_args,
        slurm_options=slurm_options,
    )

    submission_env = os.environ.copy()
    submission_env["BENCHMARK_ASSETS_ROOT"] = str(run_context.state.assets_root)
    submission_env["BENCHMARK_RESULT_ROOT"] = str(run_context.state.result_root)
    submission_env["BENCHMARK_SLURM_REPO_ROOT"] = str(REPO_ROOT)
    submission_env["BENCHMARK_SLURM_BENCHMARK_NAME"] = run_context.record.name
    submission_env["BENCHMARK_SLURM_EXECUTOR_NAME"] = run_context.executor_name
    submission_env["BENCHMARK_SLURM_READY_TIMEOUT"] = str(run_context.ready_timeout)

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=submission_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"exit code {completed.returncode}"
        )
        raise RuntimeError(f"sbatch failed: {detail}")

    return SlurmSubmissionResult(
        command=command,
        script_path=script_path,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
        job_id=_parse_sbatch_job_id(completed.stdout),
    )


def _start_agent_process(
    command: list[str],
    *,
    python_executable: str,
    workdir: Path,
    env: dict[str, str],
    show_logs: bool,
) -> subprocess.Popen[str]:
    """Agent サーバーを子プロセスとして起動する。"""
    sink = None if show_logs else subprocess.DEVNULL
    process_env = env.copy()
    ensure_no_proxy(process_env)
    process_bin = str(Path(python_executable).resolve().parent)
    process_env["PATH"] = process_bin + os.pathsep + env.get("PATH", "")
    return subprocess.Popen(
        command,
        cwd=workdir,
        env=process_env,
        stdout=sink,
        stderr=sink,
        text=True,
        start_new_session=True,
    )


def _signal_process_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    if sys.platform == "win32":
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
        return
    os.killpg(process.pid, sig)


def _signal_live_processes(processes: list[subprocess.Popen[str]], sig: signal.Signals) -> None:
    for process in processes:
        if process.poll() is not None:
            continue
        with suppress(ProcessLookupError):
            _signal_process_group(process, sig)


def _terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    """起動した子プロセスを順に停止する。"""
    grace_seconds = float(os.getenv("BENCHMARK_PROCESS_TERM_GRACE_SECONDS", "10"))
    _signal_live_processes(processes, signal.SIGTERM)

    deadline = time.time() + max(grace_seconds, 0.0)
    while time.time() < deadline:
        if all(process.poll() is not None for process in processes):
            return
        time.sleep(0.2)

    _signal_live_processes(processes, signal.SIGKILL)


def _merge_text_parts(parts: list[Any]) -> str:
    """イベントのパーツを読みやすい文字列へ整形する。"""
    from a2a.types import DataPart, TextPart

    text_parts: list[str] = []
    data_parts: list[str] = []
    for part in parts:
        if isinstance(part.root, TextPart):
            text = part.root.text.strip()
            if text:
                try:
                    data_parts.append(_json_dumps(json.loads(text)))
                except json.JSONDecodeError:
                    text_parts.append(text)
        elif isinstance(part.root, DataPart):
            data_parts.append(_json_dumps(part.root.data))
    return "\n".join([*text_parts, *data_parts]).strip()


async def _is_endpoint_ready(endpoint: str) -> bool:
    """A2A の AgentCard が取得できるかで起動完了を判定する。"""
    import httpx
    from a2a.client import A2ACardResolver

    try:
        async with httpx.AsyncClient(timeout=2) as client:
            resolver = A2ACardResolver(httpx_client=client, base_url=endpoint)
            await resolver.get_agent_card()
            return True
    except Exception:
        return False


def _format_process_exit(process: subprocess.Popen[str]) -> str:
    """ready 待ち中に落ちた子プロセスの情報を整形する。"""
    args = process.args
    command = " ".join(str(part) for part in args) if isinstance(args, (list, tuple)) else str(args)
    return f"pid={process.pid} exit_code={process.returncode} command={command}"


async def _wait_for_agents(
    endpoints: list[str],
    timeout: int,
    *,
    processes: list[subprocess.Popen[str]] | None = None,
) -> bool:
    """全エージェントの起動完了を待つ。"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if processes:
            exited = [process for process in processes if process.poll() is not None]
            if exited:
                details = "; ".join(_format_process_exit(process) for process in exited)
                raise RuntimeError(
                    f"Agent process exited before readiness check completed: {details}"
                )
        ready_count = 0
        for endpoint in endpoints:
            if await _is_endpoint_ready(endpoint):
                ready_count += 1
        if ready_count == len(endpoints):
            return True
        LOGGER.info("Agents ready: %s/%s", ready_count, len(endpoints))
        await asyncio.sleep(1)
    return False


def _log_client_stream_event(
    event: Any,
    *,
    message_type: type[Any],
    status_update_type: type[Any],
    artifact_update_type: type[Any],
) -> None:
    match event:
        case event_message if isinstance(event_message, message_type):
            rendered = _merge_text_parts(event_message.parts)
            if rendered:
                LOGGER.info("%s", rendered)
        case (_, status_event) if isinstance(status_event, status_update_type):
            parts = status_event.status.message.parts if status_event.status.message else []
            rendered = _merge_text_parts(parts)
            if rendered:
                LOGGER.info("[%s] %s", status_event.status.state.value, rendered)
        case (_, artifact_event) if isinstance(artifact_event, artifact_update_type):
            artifact_text = _merge_text_parts(artifact_event.artifact.parts)
            if artifact_text:
                LOGGER.info("[artifact] %s", artifact_text)
        case (task, None):
            parts = task.status.message.parts if task.status.message else []
            rendered = _merge_text_parts(parts)
            if rendered:
                LOGGER.info("[%s] %s", task.status.state.value, rendered)
        case _:
            LOGGER.info("Unhandled stream event: %s", type(event).__name__)


async def _run_client(eval_request: Any, green_endpoint: str) -> dict[str, Any]:
    """Green Agent へ評価要求を送り、最終応答を受け取る。"""
    from a2a.types import (
        AgentCard,
        Message,
        TaskArtifactUpdateEvent,
        TaskStatusUpdateEvent,
    )

    from common.client_utils import send_message

    async def event_consumer(event: Any, card: AgentCard) -> None:
        del card
        _log_client_stream_event(
            event,
            message_type=Message,
            status_update_type=TaskStatusUpdateEvent,
            artifact_update_type=TaskArtifactUpdateEvent,
        )

    return await send_message(
        eval_request.model_dump_json(),
        green_endpoint,
        streaming=True,
        consumer=event_consumer,
    )


def _load_result_manifests(result_root: Path) -> list[Any]:
    """結果 manifest を読み込み、完了時刻降順で返す。"""
    from common.result_store import iter_result_manifest_paths, load_result_manifest

    manifests: list[Any] = []
    for manifest_path in iter_result_manifest_paths(result_root):
        try:
            manifests.append(load_result_manifest(manifest_path))
        except Exception as exc:
            LOGGER.warning("Skipping invalid manifest %s: %s", manifest_path, exc)
    manifests.sort(key=lambda manifest: manifest.completed_at_utc, reverse=True)
    return manifests


def _filter_result_manifests(
    manifests: Iterable[Any],
    *,
    query: str | None,
    status: ResultStatusFilter,
    benchmark_name: str | None = None,
    executor_name: str | None = None,
    target: str | None = None,
) -> list[Any]:
    """検索語と属性で結果一覧を絞り込む。"""
    from common.result_store import build_manifest_search_blob

    normalized_query = query.lower() if query else None
    return [
        manifest
        for manifest in manifests
        if _result_manifest_matches_filters(
            manifest,
            normalized_query=normalized_query,
            status=status,
            benchmark_name=benchmark_name,
            executor_name=executor_name,
            target=target,
            search_blob=build_manifest_search_blob,
        )
    ]


def _result_manifest_matches_filters(
    manifest: Any,
    *,
    normalized_query: str | None,
    status: ResultStatusFilter,
    benchmark_name: str | None,
    executor_name: str | None,
    target: str | None,
    search_blob: Callable[[Any], str],
) -> bool:
    if status is not ResultStatusFilter.all and manifest.status != status.value:
        return False
    if benchmark_name and benchmark_name.lower() not in manifest.benchmark_name.lower():
        return False
    if executor_name and executor_name.lower() not in manifest.executor_name.lower():
        return False
    if target and target.lower() not in manifest.target.lower():
        return False
    return not (normalized_query and normalized_query not in search_blob(manifest))


def _has_result_selection_filters(
    *,
    query: str | None,
    status: Any,
    benchmark_name: str | None,
    executor_name: str | None,
    target: str | None,
) -> bool:
    """result 選択用の filter option が 1 つでも指定されたか判定する。"""
    return bool(
        query or benchmark_name or executor_name or target or status is not ResultStatusFilter.all
    )


def _resolve_result_identifier(identifier: str, result_root: Path, manifests: list[Any]) -> Any:
    """run_id またはファイルパスから結果 manifest を 1 件解決する。"""
    from common.result_store import load_result_manifest

    candidate_path = Path(identifier).expanduser()
    if candidate_path.exists():
        if candidate_path.is_dir():
            manifest_path = candidate_path / RESULT_MANIFEST_FILENAME
        elif candidate_path.name == RESULT_MANIFEST_FILENAME:
            manifest_path = candidate_path
        elif candidate_path.name == "detail.json":
            manifest_path = candidate_path.with_name(RESULT_MANIFEST_FILENAME)
        else:
            raise typer.BadParameter(f"Unsupported result path: {candidate_path}")
        if not manifest_path.exists():
            raise typer.BadParameter(f"{RESULT_MANIFEST_FILENAME} not found near: {candidate_path}")
        return load_result_manifest(manifest_path)

    exact_matches = [manifest for manifest in manifests if manifest.run_id == identifier]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise typer.BadParameter(f"Ambiguous run id: {identifier}")

    prefix_matches = [manifest for manifest in manifests if manifest.run_id.startswith(identifier)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        raise typer.BadParameter(f"Multiple run ids start with: {identifier}")

    try:
        relative_matches = [
            manifest
            for manifest in manifests
            if _relative_path_text(manifest.result_dir, result_root) == identifier
        ]
    except Exception:
        relative_matches = []
    if len(relative_matches) == 1:
        return relative_matches[0]

    raise typer.BadParameter(f"Result not found: {identifier}")


def _select_result_manifests(
    manifests: list[Any],
    *,
    result_root: Path,
    identifiers: Iterable[str],
    query: str | None,
    status: Any,
    benchmark_name: str | None,
    executor_name: str | None,
    target: str | None,
) -> list[Any]:
    """識別子指定と filter 指定から比較対象の run 群を選ぶ。"""
    selected: list[Any] = []
    seen_run_ids: set[str] = set()
    identifier_list = [identifier for identifier in identifiers if identifier]
    has_filters = _has_result_selection_filters(
        query=query,
        status=status,
        benchmark_name=benchmark_name,
        executor_name=executor_name,
        target=target,
    )

    def append_manifest(manifest: Any) -> None:
        """run_id 単位で重複なく manifest を追加する。"""
        run_id = str(manifest.run_id)
        if run_id in seen_run_ids:
            return
        seen_run_ids.add(run_id)
        selected.append(manifest)

    if not identifier_list and not has_filters:
        raise typer.BadParameter("Specify at least one result identifier or one filter option.")

    for identifier in identifier_list:
        append_manifest(_resolve_result_identifier(identifier, result_root, manifests))

    if has_filters:
        filtered_manifests = _filter_result_manifests(
            manifests,
            query=query,
            status=status,
            benchmark_name=benchmark_name,
            executor_name=executor_name,
            target=target,
        )
        for manifest in filtered_manifests:
            append_manifest(manifest)

    if not selected:
        raise typer.BadParameter("No results matched the given identifiers or filters.")
    return selected


def _coerce_non_negative_int(value: Any, *, default: int = 0) -> int:
    """表示用に非負整数へ丸める。"""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _load_result_detail_payload(detail_path: Path) -> dict[str, Any] | None:
    """detail.json を安全に読み込む。"""
    if not detail_path.exists():
        return None
    try:
        payload = json.loads(detail_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Skipping invalid detail payload %s: %s", detail_path, exc)
        return None
    if not isinstance(payload, dict):
        LOGGER.warning("Skipping non-object detail payload %s", detail_path)
        return None
    return payload


def _coerce_score_value(value: Any) -> float | None:
    """task result の score を float へ正規化する。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_compare_k_values(k_values: list[int] | None, run_count: int) -> list[int]:
    """compare で集計する k 一覧を検証して返す。"""
    if run_count <= 0:
        return []

    if not k_values:
        return list(range(1, run_count + 1))

    normalized = sorted({value for value in k_values if value > 0})
    if not normalized:
        raise typer.BadParameter("At least one positive --k is required.")
    if normalized[-1] > run_count:
        raise typer.BadParameter(f"--k must be <= number of selected results ({run_count}).")
    return normalized


def _estimate_pass_at_k(total_samples: int, passed_samples: int, k: int) -> float:
    """標準的な unbiased estimator で PASS@k を計算する。"""
    if k < 1:
        raise ValueError("k must be >= 1")
    if total_samples < 1 or passed_samples <= 0:
        return 0.0
    if passed_samples >= total_samples:
        return 1.0
    if k > total_samples:
        raise ValueError("k must be <= total_samples")

    remaining_failures = total_samples - passed_samples
    if remaining_failures < k:
        return 1.0
    return 1.0 - (math.comb(remaining_failures, k) / math.comb(total_samples, k))


def _index_task_results(
    task_results: Iterable[dict[str, Any]],
    *,
    run_id: str,
) -> dict[str, dict[str, Any]]:
    """1 run 分の task_results を task_id キーで引ける辞書へ変換する。"""
    indexed: dict[str, dict[str, Any]] = {}
    for task_result in task_results:
        task_id = str(task_result.get("task_id") or "").strip()
        if not task_id:
            raise typer.BadParameter(f"Run {run_id} contains a task result without task_id.")
        if task_id in indexed:
            raise typer.BadParameter(f"Run {run_id} contains duplicate task_id: {task_id}")
        indexed[task_id] = dict(task_result)
    return indexed


def _validate_selected_result_manifests(
    manifests: list[Any],
    *,
    min_run_count: int,
) -> None:
    """選択された run 群が同じ benchmark / target を向いているか確認する。"""
    if len(manifests) < min_run_count:
        raise typer.BadParameter(f"At least {min_run_count} result(s) are required.")

    benchmark_names = {str(manifest.benchmark_name) for manifest in manifests}
    if len(benchmark_names) != 1:
        raise typer.BadParameter("All compared results must share the same benchmark.")

    targets = {str(manifest.target) for manifest in manifests}
    if len(targets) != 1:
        raise typer.BadParameter("All compared results must share the same target.")

    seen_run_ids: set[str] = set()
    for manifest in manifests:
        if manifest.run_id in seen_run_ids:
            raise typer.BadParameter(f"Duplicate result selected: {manifest.run_id}")
        seen_run_ids.add(str(manifest.run_id))


def _compare_run_records_and_task_order(
    manifests: list[Any],
    *,
    result_root: Path,
) -> tuple[list[dict[str, dict[str, Any]]], list[dict[str, Any]], list[str], list[str | None]]:
    task_indexes: list[dict[str, dict[str, Any]]] = []
    run_records: list[dict[str, Any]] = []
    task_order: list[str] = []
    seen_task_ids: set[str] = set()
    benchmark_versions: list[str | None] = []

    for index, manifest in enumerate(manifests, start=1):
        task_index = _index_task_results(
            manifest.eval_result.task_results,
            run_id=str(manifest.run_id),
        )
        task_indexes.append(task_index)

        benchmark_version = getattr(manifest, "benchmark_version", None)
        if benchmark_version not in benchmark_versions:
            benchmark_versions.append(benchmark_version)

        run_records.append(
            {
                "label": f"R{index}",
                "run_id": manifest.run_id,
                "status": manifest.status,
                "benchmark_version": benchmark_version,
                "executor_name": manifest.executor_name,
                "task_selection_label": manifest.task_selection_label,
                "total_tasks": manifest.eval_result.total_tasks,
                "score_rate": manifest.eval_result.score_rate,
                "completed_at_utc": manifest.completed_at_utc.isoformat(),
                "result_dir": _relative_path_text(Path(manifest.result_dir), result_root),
                "detail_file_path": _relative_path_text(
                    Path(manifest.detail_file_path),
                    result_root,
                ),
                "fatal_error": manifest.fatal_error,
            }
        )

        for task_id in task_index:
            if task_id not in seen_task_ids:
                seen_task_ids.add(task_id)
                task_order.append(task_id)

    return task_indexes, run_records, task_order, benchmark_versions


def _compare_task_record(
    task_id: str,
    *,
    manifests: list[Any],
    run_records: list[dict[str, Any]],
    task_indexes: list[dict[str, dict[str, Any]]],
    pass_threshold: float,
) -> dict[str, Any]:
    run_outcomes: list[dict[str, Any]] = []
    available_run_count = 0
    pass_count = 0

    for run_record, manifest, task_index in zip(
        run_records,
        manifests,
        task_indexes,
        strict=True,
    ):
        task_result = task_index.get(task_id)
        if task_result is None:
            run_outcomes.append(
                {
                    "label": run_record["label"],
                    "run_id": manifest.run_id,
                    "status": "MISSING",
                }
            )
            continue

        score = _coerce_score_value(task_result.get("score"))
        passed = score is not None and score >= pass_threshold
        pass_count += int(passed)
        available_run_count += 1
        run_outcomes.append(
            {
                "label": run_record["label"],
                "run_id": manifest.run_id,
                "status": "PASS" if passed else "FAIL",
                "score": score,
                "eval_func": task_result.get("eval_func"),
                "reason": task_result.get("reason"),
                "error": task_result.get("error"),
            }
        )

    return {
        "task_id": task_id,
        "available_run_count": available_run_count,
        "pass_count": pass_count,
        "runs": run_outcomes,
    }


def _build_compare_task_records(
    task_order: list[str],
    *,
    manifests: list[Any],
    run_records: list[dict[str, Any]],
    task_indexes: list[dict[str, dict[str, Any]]],
    pass_threshold: float,
) -> list[dict[str, Any]]:
    return [
        _compare_task_record(
            task_id,
            manifests=manifests,
            run_records=run_records,
            task_indexes=task_indexes,
            pass_threshold=pass_threshold,
        )
        for task_id in task_order
    ]


def _build_pass_at_k_records(
    tasks: list[dict[str, Any]],
    normalized_ks: list[int],
) -> list[dict[str, Any]]:
    total_tasks = len(tasks)
    records: list[dict[str, Any]] = []
    for k in normalized_ks:
        eligible_tasks = [task for task in tasks if int(task["available_run_count"]) >= k]
        score = None
        if eligible_tasks:
            score = sum(
                _estimate_pass_at_k(
                    int(task["available_run_count"]),
                    int(task["pass_count"]),
                    k,
                )
                for task in eligible_tasks
            ) / len(eligible_tasks)
        records.append(
            {
                "k": k,
                "score": score,
                "eligible_tasks": len(eligible_tasks),
                "total_tasks": total_tasks,
                "coverage_rate": (len(eligible_tasks) / total_tasks) if total_tasks else 0.0,
            }
        )
    return records


def _build_result_compare_payload(
    manifests: list[Any],
    *,
    result_root: Path,
    pass_threshold: float,
    k_values: list[int] | None,
    min_run_count: int = 2,
) -> dict[str, Any]:
    """複数 run 比較用の PASS/FAIL 行列と PASS@k 集計を作る。"""
    _validate_selected_result_manifests(
        manifests,
        min_run_count=min_run_count,
    )

    normalized_ks = _normalize_compare_k_values(k_values, len(manifests))
    task_indexes, run_records, task_order, benchmark_versions = _compare_run_records_and_task_order(
        manifests,
        result_root=result_root,
    )
    tasks = _build_compare_task_records(
        task_order,
        manifests=manifests,
        run_records=run_records,
        task_indexes=task_indexes,
        pass_threshold=pass_threshold,
    )
    total_tasks = len(tasks)

    return {
        "benchmark_name": manifests[0].benchmark_name,
        "target": manifests[0].target,
        "benchmark_versions": benchmark_versions,
        "pass_threshold": pass_threshold,
        "run_count": len(manifests),
        "task_count": total_tasks,
        "has_partial_coverage": any(
            int(task["available_run_count"]) != len(manifests) for task in tasks
        ),
        "runs": run_records,
        "tasks": tasks,
        "pass_at_k": _build_pass_at_k_records(tasks, normalized_ks),
    }


def _build_result_pass_at_k_payload(compare_payload: dict[str, Any]) -> dict[str, Any]:
    """compare 集計結果から PASS@k 専用の要約を切り出す。"""
    return {
        "benchmark_name": compare_payload["benchmark_name"],
        "target": compare_payload["target"],
        "benchmark_versions": compare_payload["benchmark_versions"],
        "pass_threshold": compare_payload["pass_threshold"],
        "run_count": compare_payload["run_count"],
        "task_count": compare_payload["task_count"],
        "has_partial_coverage": compare_payload["has_partial_coverage"],
        "runs": compare_payload["runs"],
        "pass_at_k": compare_payload["pass_at_k"],
    }


def _format_compare_task_status(status: str) -> str:
    """PASS/FAIL/MISSING を比較テーブル用の表示へ変換する。"""
    if status == "PASS":
        return "[green]PASS[/green]"
    if status == "FAIL":
        return "[red]FAIL[/red]"
    return "[dim]—[/dim]"


def _render_result_compare(payload: dict[str, Any]) -> None:
    """複数 run の比較結果を表形式で表示する。"""
    version_text = ", ".join(version or "—" for version in payload["benchmark_versions"])
    task_basis = "union of task ids" if payload["has_partial_coverage"] else "shared task set"
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Benchmark   : {payload['benchmark_name']}",
                    f"Target      : {payload['target']}",
                    f"Versions    : {version_text}",
                    f"Runs        : {payload['run_count']}",
                    f"Tasks       : {payload['task_count']}",
                    f"Task Basis  : {task_basis}",
                    f"Pass Rule   : score >= {payload['pass_threshold']:g}",
                ]
            ),
            title="Result Compare",
            border_style="cyan",
        )
    )

    run_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE)
    run_table.add_column("Label")
    run_table.add_column(RUN_ID_COLUMN)
    run_table.add_column("Executor")
    run_table.add_column("Status")
    run_table.add_column("Score", justify="right")
    run_table.add_column("Tasks", justify="right")
    run_table.add_column("Completed")
    run_table.add_column("Path")
    for run_record in payload["runs"]:
        score_rate = run_record.get("score_rate")
        score_text = "—" if score_rate is None else f"{float(score_rate):.2%}"
        run_table.add_row(
            str(run_record["label"]),
            str(run_record["run_id"]),
            str(run_record["executor_name"]),
            str(run_record["status"]),
            score_text,
            str(run_record["total_tasks"]),
            str(run_record["completed_at_utc"]),
            str(run_record["result_dir"]),
        )
    console.print(run_table)

    task_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE)
    task_table.add_column("Task ID")
    task_table.add_column("Passes", justify="right")
    for run_record in payload["runs"]:
        task_table.add_column(str(run_record["label"]), justify="center")

    for task_record in payload["tasks"]:
        row = [
            str(task_record["task_id"]),
            f"{task_record['pass_count']}/{task_record['available_run_count']}",
        ]
        row.extend(
            _format_compare_task_status(str(run_outcome["status"]))
            for run_outcome in task_record["runs"]
        )
        task_table.add_row(*row)
    console.print(task_table)

    pass_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE)
    pass_table.add_column("k", justify="right")
    pass_table.add_column(PASS_AT_K_LABEL, justify="right")
    pass_table.add_column("Eligible", justify="right")
    for pass_record in payload["pass_at_k"]:
        score = pass_record.get("score")
        score_text = "—" if score is None else f"{float(score):.2%}"
        pass_table.add_row(
            str(pass_record["k"]),
            score_text,
            f"{pass_record['eligible_tasks']}/{pass_record['total_tasks']}",
        )
    console.print(pass_table)

    if payload["has_partial_coverage"]:
        console.print(
            "[yellow]Note:[/yellow] PASS@k excludes tasks that do not have at least k runs."
        )


def _render_result_pass_at_k(payload: dict[str, Any]) -> None:
    """PASS@k 専用コマンドの結果を表形式で表示する。"""
    version_text = ", ".join(version or "—" for version in payload["benchmark_versions"])
    task_basis = "union of task ids" if payload["has_partial_coverage"] else "shared task set"
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Benchmark   : {payload['benchmark_name']}",
                    f"Target      : {payload['target']}",
                    f"Versions    : {version_text}",
                    f"Runs        : {payload['run_count']}",
                    f"Tasks       : {payload['task_count']}",
                    f"Task Basis  : {task_basis}",
                    f"Pass Rule   : score >= {payload['pass_threshold']:g}",
                ]
            ),
            title=PASS_AT_K_LABEL,
            border_style="cyan",
        )
    )

    run_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE)
    run_table.add_column("Label")
    run_table.add_column(RUN_ID_COLUMN)
    run_table.add_column("Executor")
    run_table.add_column("Score", justify="right")
    run_table.add_column("Tasks", justify="right")
    for run_record in payload["runs"]:
        score_rate = run_record.get("score_rate")
        score_text = "—" if score_rate is None else f"{float(score_rate):.2%}"
        run_table.add_row(
            str(run_record["label"]),
            str(run_record["run_id"]),
            str(run_record["executor_name"]),
            score_text,
            str(run_record["total_tasks"]),
        )
    console.print(run_table)

    pass_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE)
    pass_table.add_column("k", justify="right")
    pass_table.add_column(PASS_AT_K_LABEL, justify="right")
    pass_table.add_column("Eligible", justify="right")
    pass_table.add_column("Coverage", justify="right")
    for pass_record in payload["pass_at_k"]:
        score = pass_record.get("score")
        score_text = "—" if score is None else f"{float(score):.2%}"
        pass_table.add_row(
            str(pass_record["k"]),
            score_text,
            f"{pass_record['eligible_tasks']}/{pass_record['total_tasks']}",
            f"{float(pass_record['coverage_rate']):.2%}",
        )
    console.print(pass_table)

    if payload["has_partial_coverage"]:
        console.print(
            "[yellow]Note:[/yellow] PASS@k excludes tasks that do not have at least k runs."
        )


def _render_result_table(manifests: list[Any], result_root: Path) -> None:
    """結果一覧を表形式で表示する。"""
    table = Table(show_header=True, header_style=TABLE_HEADER_STYLE)
    table.add_column(RUN_ID_COLUMN)
    table.add_column("User")
    table.add_column("Benchmark")
    table.add_column("Version")
    table.add_column("Executor")
    table.add_column("Target")
    table.add_column("Status")
    table.add_column("Score", justify="right")
    table.add_column("Tasks", justify="right")
    table.add_column("Completed")
    table.add_column("Path")
    for manifest in manifests:
        table.add_row(
            manifest.run_id,
            getattr(manifest, "user_name", None) or "—",
            manifest.benchmark_name,
            getattr(manifest, "benchmark_version", None) or "—",
            manifest.executor_name,
            manifest.target,
            manifest.status,
            f"{manifest.eval_result.score_rate:.2%}",
            str(manifest.eval_result.total_tasks),
            manifest.completed_at_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            _relative_path_text(manifest.result_dir, result_root),
        )
    console.print(table)
    console.print(f"\nTotal results: {len(manifests)}")


def _build_result_summary_lines(
    manifest: Any,
    result_root: Path,
    executor_runtime: Mapping[str, Any],
) -> list[str]:
    green_ver = getattr(manifest, "green_agent_version", None)
    purple_ver = getattr(manifest, "purple_agent_version", None)
    exec_ver = getattr(manifest, "executor_version", None)
    lines = [
        f"Run ID      : {manifest.run_id}",
        f"User        : {getattr(manifest, 'user_name', None) or '—'}",
        f"Status      : {manifest.status}",
        f"Benchmark   : {manifest.benchmark_name}",
        f"Version     : {getattr(manifest, 'benchmark_version', None) or '—'}",
        *(
            [
                f"Green Ver   : {green_ver}",
                f"Purple Ver  : {purple_ver}",
                f"Exec Ver    : {exec_ver or '—'}",
            ]
            if green_ver or purple_ver or exec_ver
            else []
        ),
        f"Executor    : {manifest.executor_name}",
        f"Target      : {manifest.target}",
        f"Tasks       : {manifest.task_selection_label}",
        f"LLM Models  : {format_executor_runtime_models(executor_runtime)}",
        f"Score       : {manifest.score_summary}",
        f"Started     : {manifest.created_at_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Completed   : {manifest.completed_at_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Result Dir  : {_relative_path_text(manifest.result_dir, result_root)}",
        f"Detail File : {_relative_path_text(manifest.detail_file_path, result_root)}",
    ]
    llm_notes = format_executor_runtime_notes(executor_runtime)
    if llm_notes:
        lines.append(f"LLM Notes   : {llm_notes}")
    return lines


def _render_result_summary_panel(summary_lines: list[str]) -> None:
    console.print(
        Panel.fit(
            "\n".join(summary_lines),
            title="Benchmark Result",
            border_style="cyan",
        )
    )


def _render_task_result_table(task_results: Iterable[Mapping[str, Any]]) -> None:
    task_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE)
    task_table.add_column("Task ID")
    task_table.add_column("Score", justify="right")
    task_table.add_column("Eval Func")
    task_table.add_column("Reason / Error")
    for task_result in task_results:
        task_table.add_row(
            str(task_result.get("task_id", "—")),
            str(task_result.get("score", "—")),
            str(task_result.get("eval_func", "—")),
            str(task_result.get("reason") or task_result.get("error") or "—"),
        )
    console.print(task_table)


def _build_result_detail_summary(
    detail_payload: Mapping[str, Any],
    executor_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    detail_summary: dict[str, Any] = {
        "request_config": detail_payload.get("request_config"),
        "participants": detail_payload.get("participants"),
        "fatal_error": detail_payload.get("fatal_error"),
        "executor_runtime": executor_runtime,
    }
    return detail_summary


def _render_detail_summary_json(detail_summary: Mapping[str, Any]) -> None:
    console.print(
        Syntax(
            _json_dumps(detail_summary),
            "json",
            word_wrap=True,
        )
    )


def _render_result_detail(manifest: Any, result_root: Path) -> None:
    """結果 1 件の詳細を表示する。"""
    detail_payload = _load_result_detail_payload(Path(manifest.detail_file_path))
    executor_runtime = resolve_executor_runtime_payload(
        benchmark_name=manifest.benchmark_name,
        executor_name=manifest.executor_name,
        request_config=getattr(manifest, "request_config", {}),
        result_dir=Path(manifest.result_dir),
        recorded_payload=detail_payload.get("executor_runtime")
        if isinstance(detail_payload, dict)
        else None,
    )
    _render_result_summary_panel(
        _build_result_summary_lines(manifest, result_root, executor_runtime)
    )

    if manifest.fatal_error:
        console.print(f"[red]Fatal Error:[/red] {manifest.fatal_error}")

    _render_task_result_table(manifest.eval_result.task_results)

    if detail_payload is not None:
        _render_detail_summary_json(
            _build_result_detail_summary(detail_payload, executor_runtime)
        )


def _translate_legacy_argv(argv: list[str]) -> list[str]:
    """旧 `--benchmark ...` 形式を `bench run` サブコマンドへ寄せる。"""
    if not argv or "--benchmark" not in argv:
        return argv

    root_args: list[str] = []
    run_args: list[str] = ["bench", "run"]
    benchmark_name: str | None = None
    valued_options = {
        "--executor",
        "--launcher",
        "--workdir",
        "--target",
        "--task-id",
        "--max-self-evolutions",
        "--max-benchmark-self-evolution-cycles",
        "--self-evolve-strategy",
        "--vllm-model-id",
        "--green-host",
        "--green-port",
        "--purple-host",
        "--purple-port",
        "--slurm-partition",
        "--slurm-job-name",
        "--slurm-output",
        "--slurm-time",
        "--slurm-mem",
        "--slurm-cpus-per-task",
        "--slurm-gpus",
        "--slurm-account",
        "--slurm-qos",
        "--slurm-constraint",
        "--slurm-exclude",
        "--slurm-arg",
        "--ready-timeout",
    }
    root_valued_options = {"--assets-root", "--result-root"}

    first_command_index = 0
    while first_command_index < len(argv):
        candidate = argv[first_command_index]
        if candidate in {
            "--assets-root",
            "--result-root",
        } and first_command_index + 1 < len(argv):
            first_command_index += 2
            continue
        break
    if first_command_index < len(argv) and argv[first_command_index] in {
        "bench",
        "benchmark",
        "result",
        "check",
    }:
        return argv

    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--benchmark":
            benchmark_name = argv[index + 1]
            index += 2
            continue
        if arg in root_valued_options:
            root_args.extend([arg, argv[index + 1]])
            index += 2
            continue
        if arg == "--eval-detail-dir":
            root_args.extend(["--result-root", argv[index + 1]])
            index += 2
            continue
        run_args.append(arg)
        if arg in valued_options:
            run_args.append(argv[index + 1])
            index += 2
            continue
        index += 1

    if benchmark_name is None:
        return argv
    return [*root_args, "bench", "run", benchmark_name, *run_args[2:]]


@benchmark_app.command("list")
def benchmark_list(
    ctx: typer.Context,
    query: Annotated[
        str | None, typer.Option("--query", help="名前・target・role などで絞り込む。")
    ] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="出力形式。")
    ] = OutputFormat.table,
) -> None:
    """登録済みベンチマーク一覧を表示する。"""
    state = _state_from_ctx(ctx)
    records = _filter_benchmarks(_discover_benchmarks(state.assets_root), query)
    if output_format is OutputFormat.json:
        console.print_json(_json_dumps([record.to_dict() for record in records]))
        return
    _render_benchmark_table(records, state.assets_root)


@benchmark_app.command("search")
def benchmark_search(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="検索文字列。")],
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="出力形式。")
    ] = OutputFormat.table,
) -> None:
    """ベンチマーク定義を検索する。"""
    benchmark_list(ctx=ctx, query=query, output_format=output_format)


@benchmark_app.command("show")
def benchmark_show(
    ctx: typer.Context,
    benchmark_name: Annotated[str, typer.Argument(help="ベンチマーク名またはディレクトリ名。")],
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="出力形式。")
    ] = OutputFormat.table,
) -> None:
    """ベンチマーク定義の詳細を表示する。"""
    state = _state_from_ctx(ctx)
    records = _discover_benchmarks(state.assets_root)
    record = _resolve_benchmark(records, benchmark_name)
    _, raw_config = _load_benchmark_config(state.assets_root, record.benchmark_dir.name)

    if output_format is OutputFormat.json:
        console.print_json(
            _json_dumps(
                {
                    "summary": record.to_dict(),
                    "benchmark_toml": raw_config,
                }
            )
        )
        return

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Benchmark      : {record.name}",
                    f"Version        : {record.version or '—'}",
                    f"Directory      : {_relative_path_text(record.benchmark_dir, state.assets_root)}",
                    f"Config         : {_relative_path_text(record.config_path, state.assets_root)}",
                    "Executors      : "
                    f"{', '.join(record.available_executors) if record.available_executors else '—'}",
                    f"Default Exec   : {record.default_executor or '—'}",
                    f"Default Target : {record.default_target or '—'}",
                    f"Targets        : {_targets_summary(record)}",
                    f"Roles          : {', '.join(record.roles) if record.roles else '—'}",
                    f"Green Entrypoint: {record.green_entrypoint or '—'}",
                ]
            ),
            title="Benchmark",
            border_style="cyan",
        )
    )
    console.print(Syntax(_json_dumps(raw_config), "json", word_wrap=True))


def _render_slurm_submission(
    run_context: BenchmarkRunContext,
    submission: SlurmSubmissionResult,
    *,
    slurm_exclude: str | None,
) -> None:
    console.print(
        Panel.fit(
            "\n".join(
                filter(
                    None,
                    [
                        f"Benchmark   : {run_context.record.name}",
                        f"Executor    : {run_context.executor_name}",
                        f"Launcher    : {run_context.launcher.value}",
                        f"Target      : {run_context.runtime_config['config']['target']}",
                        (
                            "Task IDs    : "
                            f"{', '.join(run_context.runtime_config['config'].get('task_ids', [])) or 'all'}"
                        ),
                        (
                            "Max Parallel: "
                            f"{run_context.runtime_config['config'].get('max_parallel', 1)}"
                        ),
                        (
                            f"vLLM Model  : {run_context.runtime_config['config']['vllm_model_id']}"
                            if run_context.runtime_config["config"].get("vllm_model_id")
                            else None
                        ),
                        f"Workdir     : {run_context.workdir}",
                        f"Result Root : {run_context.state.result_root}",
                        f"Ready Timeout: {run_context.ready_timeout}",
                        f"Slurm Script: {submission.script_path}",
                        f"Slurm Exclude: {slurm_exclude or '—'}",
                        (
                            f"Inference  : {run_context.inference_config}"
                            if run_context.inference_config is not None
                            else None
                        ),
                        f"Job ID      : {submission.job_id or 'unknown'}",
                        f"sbatch      : {submission.stdout or 'submitted'}",
                    ],
                )
            ),
            title="Slurm Submission",
            border_style="cyan",
        )
    )
    if submission.stderr:
        console.print(f"[yellow]sbatch stderr:[/yellow] {submission.stderr}")


def _run_benchmark_via_slurm(
    run_context: BenchmarkRunContext,
    slurm_options: SlurmOptions,
) -> None:
    try:
        submission = _submit_benchmark_run_via_slurm(
            run_context=run_context,
            slurm_options=slurm_options,
        )
    except Exception as exc:
        LOGGER.error("Slurm submission failed: %s", exc)
        console.print(f"[red]Slurm submission failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    LOGGER.info("Submitted Slurm job: %s", " ".join(submission.command))
    _render_slurm_submission(
        run_context,
        submission,
        slurm_exclude=slurm_options.exclude,
    )


def _render_benchmark_run_panel(run_context: BenchmarkRunContext, result_dir: Path) -> None:
    runtime_config = run_context.runtime_config
    console.print(
        Panel.fit(
            "\n".join(
                filter(
                    None,
                    [
                        f"Benchmark  : {run_context.record.name}",
                        f"Executor   : {run_context.executor_name}",
                        f"Target     : {runtime_config['config']['target']}",
                        f"Task IDs   : {', '.join(runtime_config['config'].get('task_ids', [])) or 'all'}",
                        f"Max Parallel : {runtime_config['config'].get('max_parallel', 1)}",
                        (
                            f"vLLM Model : {runtime_config['config']['vllm_model_id']}"
                            if runtime_config["config"].get("vllm_model_id")
                            else None
                        ),
                        f"Ready Timeout : {run_context.ready_timeout}",
                        f"Workdir    : {run_context.workdir}",
                        f"Result Dir : {result_dir}",
                    ],
                )
            ),
            title="Benchmark Run",
            border_style="cyan",
        )
    )


def _start_inference_runtime(
    inference_config: Path | None,
    env: dict[str, str],
    *,
    launch_summary_path: Path | None = None,
) -> tuple[Any, Any]:
    if inference_config is None:
        return None, None

    from common.inference_runtime import managed_inference

    inference_context = managed_inference(
        inference_config,
        launch_summary_path=launch_summary_path,
    )
    inference_runtime = inference_context.__enter__()
    inference_runtime.apply_to(env)
    console.print(
        Panel.fit(
            "\n".join(
                filter(
                    None,
                    [
                        *[
                            f"{name}: {getattr(session, 'api_base', '')}"
                            for name, session in sorted(inference_runtime.sessions.items())
                        ],
                        (
                            f"Summary: {launch_summary_path}"
                            if launch_summary_path is not None
                            else None
                        ),
                    ],
                )
            ),
            title="Inference Ready",
            border_style="green",
        )
    )
    return inference_context, inference_runtime


def _resolve_agent_endpoint(
    *,
    run_context: BenchmarkRunContext,
    launch_spec: AgentLaunchSpec,
    env: dict[str, str],
    result_dir: Path,
    processes: list[subprocess.Popen[str]],
) -> str:
    requested_port = int(launch_spec.config["port"])
    host = str(launch_spec.config["host"])
    if not _should_start_agent_process(host, requested_port):
        LOGGER.info(
            "Using existing %s endpoint: %s",
            launch_spec.label,
            _build_endpoint(host, requested_port),
        )
        return _build_endpoint(host, requested_port)

    python_executable = _resolve_entrypoint_python(
        entrypoint=launch_spec.entrypoint,
        benchmark_dir=run_context.benchmark_dir,
        override_env_names=launch_spec.override_env_names,
    )
    port_file = _agent_port_file(result_dir, launch_spec.label) if requested_port == 0 else None
    if port_file is not None:
        port_file.unlink(missing_ok=True)
    command = _build_agent_command(
        python_executable,
        launch_spec.entrypoint,
        host,
        requested_port,
        port_file=port_file,
    )
    LOGGER.info("Starting %s: %s", launch_spec.label, " ".join(command))
    process = _start_agent_process(
        command,
        python_executable=python_executable,
        workdir=run_context.workdir,
        env=env,
        show_logs=run_context.show_logs,
    )
    processes.append(process)
    actual_port = _resolve_actual_agent_port(
        requested_port=requested_port,
        port_file=port_file,
        process=process,
        timeout_seconds=run_context.ready_timeout,
        agent_label=launch_spec.label,
    )
    return _build_endpoint(host, actual_port)


def _resolve_participant_endpoints(
    run_context: BenchmarkRunContext,
    *,
    env: dict[str, str],
    result_dir: Path,
    processes: list[subprocess.Popen[str]],
) -> dict[str, str]:
    endpoints: dict[str, str] = {}
    for index, participant in enumerate(run_context.runtime_config["participants"]):
        role = str(participant["role"])
        endpoint = _resolve_agent_endpoint(
            run_context=run_context,
            launch_spec=AgentLaunchSpec(
                config=participant,
                label=f"participant-{index}-{role}",
                entrypoint=run_context.benchmark_dir / participant["entrypoint"],
                override_env_names=[
                    f"BENCHMARK_{role.upper()}_PYTHON",
                    "BENCHMARK_PURPLE_PYTHON",
                ],
            ),
            env=env,
            result_dir=result_dir,
            processes=processes,
        )
        endpoints[role] = endpoint
    return endpoints


def _render_resolved_endpoints(
    *,
    green_endpoint: str,
    participant_endpoints: Mapping[str, str],
) -> None:
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Green Endpoint : {green_endpoint}",
                    *[
                        f"Participant[{role}] : {endpoint}"
                        for role, endpoint in sorted(participant_endpoints.items())
                    ],
                ]
            ),
            title="Resolved Endpoints",
            border_style="green",
        )
    )


def _prepare_local_benchmark_run(run_context: BenchmarkRunContext):
    from common.result_store import (
        build_execution_identity,
        ensure_result_dir,
    )

    requested_participants = {
        str(participant["role"]): _build_endpoint(participant["host"], int(participant["port"]))
        for participant in run_context.runtime_config["participants"]
    }
    result_paths = build_execution_identity(
        benchmark_name=run_context.record.name,
        executor_name=run_context.executor_name,
        request_config=run_context.runtime_config["config"],
        participants=requested_participants,
        result_root=run_context.state.result_root,
    )
    ensure_result_dir(result_paths)
    env = _build_process_env(
        ProcessEnvConfig(
            benchmark_name=run_context.record.name,
            benchmark_version=run_context.record.version,
            target=str(run_context.runtime_config["config"]["target"]),
            benchmark_dir=run_context.benchmark_dir,
            assets_root=run_context.state.assets_root,
            executor_name=run_context.executor_name,
            request_config=run_context.runtime_config["config"],
            result_root=run_context.state.result_root,
            result_dir=result_paths.result_dir,
            run_id=result_paths.run_id,
            user_name=result_paths.user_name,
            config_hash=result_paths.config_hash,
        )
    )
    inference_summary_path = (
        result_paths.result_dir / "inference-launch-summary.json"
        if run_context.inference_config is not None
        else None
    )
    run_context.workdir.mkdir(parents=True, exist_ok=True)
    _render_benchmark_run_panel(run_context, result_paths.result_dir)
    return result_paths, env, inference_summary_path


def _prepare_inference_runtime_for_local_run(
    run_context: BenchmarkRunContext,
    env: dict[str, str],
    result_paths: Any,
    inference_summary_path: Path | None,
):
    inference_context, inference_runtime = _start_inference_runtime(
        run_context.inference_config,
        env,
        launch_summary_path=inference_summary_path,
    )
    if inference_runtime is not None:
        prepare_inference_runtime_config(
            runtime_config=run_context.runtime_config,
            env=env,
            result_paths=result_paths,
            inference_summary_path=inference_summary_path,
            inference_runtime=inference_runtime,
            warn=console.print,
            repo_root=REPO_ROOT,
        )
    return inference_context


def _start_local_benchmark_agents(
    run_context: BenchmarkRunContext,
    *,
    env: dict[str, str],
    result_dir: Path,
    processes: list[subprocess.Popen[str]],
) -> tuple[str, Mapping[str, str]]:
    green_endpoint = _resolve_agent_endpoint(
        run_context=run_context,
        launch_spec=AgentLaunchSpec(
            config=run_context.runtime_config["green_agent"],
            label="green",
            entrypoint=run_context.benchmark_dir
            / run_context.runtime_config["green_agent"]["entrypoint"],
            override_env_names=["BENCHMARK_GREEN_PYTHON"],
        ),
        env=env,
        result_dir=result_dir,
        processes=processes,
    )
    participant_endpoints = _resolve_participant_endpoints(
        run_context,
        env=env,
        result_dir=result_dir,
        processes=processes,
    )
    return green_endpoint, participant_endpoints


def _wait_for_local_benchmark_agents(
    endpoints: list[str],
    run_context: BenchmarkRunContext,
    processes: list[subprocess.Popen[str]],
) -> None:
    with console.status("Waiting for agents to become ready..."):
        ready = asyncio.run(
            _wait_for_agents(
                endpoints,
                run_context.ready_timeout,
                processes=processes,
            )
        )
    if not ready:
        raise RuntimeError(
            f"Agents did not become ready within {run_context.ready_timeout} seconds"
        )
    console.print("[green]All agents are ready.[/green]")


def _serve_local_benchmark_until_interrupted(run_context: BenchmarkRunContext) -> None:
    if not run_context.serve_only:
        return
    console.print("Serve-only mode. Press Ctrl+C to stop.")
    while True:
        time.sleep(1)


def _run_local_benchmark_eval(
    run_context: BenchmarkRunContext,
    *,
    green_endpoint: str,
    participant_endpoints: Mapping[str, str],
    result_paths: Any,
) -> None:
    from common.models import EvalRequest
    from common.result_store import load_result_manifest

    eval_request = EvalRequest(
        participants=participant_endpoints,
        config=_attach_runtime_green_endpoint(
            run_context.runtime_config["config"],
            green_endpoint=green_endpoint,
        ),
    )
    response = asyncio.run(_run_client(eval_request, green_endpoint))
    if response.get("status") not in {None, "completed"}:
        LOGGER.info("Final response: %s", response.get("response", ""))

    if result_paths.manifest_path.exists():
        manifest = load_result_manifest(result_paths.manifest_path)
        _render_result_detail(manifest, run_context.state.result_root)
        return

    console.print(
        Panel.fit(
            "\n".join(
                [
                    "Benchmark run finished, but manifest.json was not found yet.",
                    f"Expected manifest: {result_paths.manifest_path}",
                ]
            ),
            title="Run Output",
            border_style="yellow",
        )
    )


def _cleanup_local_benchmark_run(
    run_context: BenchmarkRunContext,
    *,
    processes: list[subprocess.Popen[str]],
    inference_context: Any,
    result_dir: Path,
) -> None:
    active_exc_type = sys.exc_info()[0]
    cleanup_error: Exception | None = None
    _terminate_processes(processes)
    if inference_context is not None:
        try:
            inference_context.__exit__(None, None, None)
        except Exception as exc:
            cleanup_error = exc
            LOGGER.exception("Failed to stop inference runtime.")
            console.print(f"[red]Failed to stop inference runtime:[/red] {exc}")
    if cleanup_error is not None and active_exc_type is None:
        raise typer.Exit(code=1) from cleanup_error


def _run_benchmark_locally(run_context: BenchmarkRunContext) -> None:
    result_paths, env, inference_summary_path = _prepare_local_benchmark_run(run_context)
    processes: list[subprocess.Popen[str]] = []
    inference_context = None
    try:
        inference_context = _prepare_inference_runtime_for_local_run(
            run_context,
            env,
            result_paths,
            inference_summary_path,
        )
        green_endpoint, participant_endpoints = _start_local_benchmark_agents(
            run_context,
            env=env,
            result_dir=result_paths.result_dir,
            processes=processes,
        )
        _wait_for_local_benchmark_agents(
            [green_endpoint, *participant_endpoints.values()],
            run_context,
            processes,
        )
        _render_resolved_endpoints(
            green_endpoint=green_endpoint,
            participant_endpoints=participant_endpoints,
        )
        _serve_local_benchmark_until_interrupted(run_context)
        _run_local_benchmark_eval(
            run_context,
            green_endpoint=green_endpoint,
            participant_endpoints=participant_endpoints,
            result_paths=result_paths,
        )
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except typer.Exit:
        raise
    except Exception as exc:
        LOGGER.error("Benchmark run failed: %s", exc)
        console.print(f"[red]Benchmark run failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        _cleanup_local_benchmark_run(
            run_context,
            processes=processes,
            inference_context=inference_context,
            result_dir=result_paths.result_dir,
        )


@benchmark_app.command("run")
def benchmark_run(
    ctx: typer.Context,
    benchmark_name: Annotated[str, typer.Argument(help="ベンチマーク名またはディレクトリ名。")],
    executor: Annotated[str | None, typer.Option("--executor", help="Purple executor 名。")] = None,
    launcher: Annotated[
        ExecutionLauncher,
        typer.Option("--launcher", help="実行経路。`local` または `slurm`。"),
    ] = ExecutionLauncher.local,
    workdir: Annotated[
        Path,
        typer.Option(
            "--workdir",
            help="Green/Purple サーバー起動時の作業ディレクトリ。",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = Path.cwd(),
    target: Annotated[
        str | None,
        typer.Option("--target", help="benchmark.toml の config.target を上書きする。"),
    ] = None,
    config_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--config",
            help="request config を KEY=VALUE 形式で上書きする。複数指定可。",
        ),
    ] = None,
    task_ids: Annotated[
        list[str] | None,
        typer.Option("--task-id", help="評価対象 task_id を絞り込む。"),
    ] = None,
    max_parallel: Annotated[
        int | None,
        typer.Option("--max-parallel", min=1, help="同時に進める benchmark task 数。"),
    ] = None,
    vllm_model_id: Annotated[
        str | None,
        typer.Option(
            "--vllm-model-id",
            help="managed vLLM の model id。",
        ),
    ] = None,
    wm_strategy: Annotated[
        str | None,
        typer.Option(
            "--wm-strategy",
            help="World Model strategy: none | prompt_injection | selection | itp_i | revision | "
            "reference | beam_plan | hier_latent_cem (aliases: best_of_n, imagined, itp-i). "
            "itp_i is the training-free "
            "Imagine-Then-Plan implicit-feedback harness. beam_plan/hier_latent_cem are "
            "JEPA-only MPC lookahead (discrete-candidate vs. hierarchical latent-action CEM "
            "search). Sets WM_STRATEGY.",
        ),
    ] = None,
    wm_backend: Annotated[
        str | None,
        typer.Option(
            "--wm-backend",
            help="World Model backend: noop | served | llm | ewm_predict | ewm_imagined. Sets WM_BACKEND.",
        ),
    ] = None,
    wm_n: Annotated[
        int | None,
        typer.Option("--wm-n", min=1, help="Candidate samples per step for selection strategy. Sets WM_N."),
    ] = None,
    wm_model: Annotated[
        str | None,
        typer.Option("--wm-model", help="World Model model id (llm/served backends). Sets WM_MODEL."),
    ] = None,
    wm_base_url: Annotated[
        str | None,
        typer.Option("--wm-base-url", help="OpenAI-compatible base URL for the served WM backend. Sets WM_BASE_URL."),
    ] = None,
    wm_ewm_mcp_url: Annotated[
        str | None,
        typer.Option(
            "--wm-ewm-mcp-url",
            help="Route the EWM world model over an MCP server (the `generate` tool), e.g. "
            "http://127.0.0.1:12072. Unset → in-process vLLM. Sets WM_EWM_MCP_URL.",
        ),
    ] = None,
    wm_ewm_model: Annotated[
        str | None,
        typer.Option(
            "--wm-ewm-model",
            help="EWM model id served by the world-model endpoint (default gymops_world_model). Sets WM_EWM_MODEL.",
        ),
    ] = None,
    wm_state: Annotated[
        str | None,
        typer.Option(
            "--wm-state",
            help="EWM target mode: binary_error | binary_error_stage | tool_output | canonical_nudge "
            "(schema-based categorical event state + epistemic nudge). Sets WM_STATE.",
        ),
    ] = None,
    wm_action_optimizer: Annotated[
        str | None,
        typer.Option(
            "--wm-action-optimizer",
            help="Imagined-rollout action optimizer: topk_search | (unset → plain). Sets ACTION_OPTIMIZER.",
        ),
    ] = None,
    wm_k_controller: Annotated[
        str | None,
        typer.Option(
            "--wm-k-controller",
            help="Imagined depth controller: react_wm_decide_k | react_wm_rl_k | (unset → static). Sets K_CONTROLLER.",
        ),
    ] = None,
    wm_k_controller_model_path: Annotated[
        str | None,
        typer.Option(
            "--wm-k-controller-model-path",
            help="Checkpoint path for the react_wm_rl_k controller. Sets K_CONTROLLER_MODEL_PATH.",
        ),
    ] = None,
    wm_k_controller_device: Annotated[
        str | None,
        typer.Option(
            "--wm-k-controller-device", help="Device for the RL K-controller (auto|cpu|cuda…). Sets K_CONTROLLER_DEVICE."
        ),
    ] = None,
    wm_k_controller_dtype: Annotated[
        str | None,
        typer.Option(
            "--wm-k-controller-dtype", help="dtype for the RL K-controller (auto|float16…). Sets K_CONTROLLER_DTYPE."
        ),
    ] = None,
    wm_imagined_max_steps: Annotated[
        int | None,
        typer.Option(
            "--wm-imagined-max-steps", min=0, help="Imagined rollout depth (static K / Kmax). Sets WM_IMAGINED_MAX_STEPS."
        ),
    ] = None,
    wm_itp_max_k: Annotated[
        int | None,
        typer.Option(
            "--wm-itp-max-k", min=0,
            help="ITP-I maximum adaptive lookahead depth (default 5). Sets WM_ITP_MAX_K.",
        ),
    ] = None,
    wm_itp_fixed_k: Annotated[
        int | None,
        typer.Option(
            "--wm-itp-fixed-k", min=-1,
            help="ITP-I fixed K; -1 uses policy-selected adaptive K. Sets WM_ITP_FIXED_K.",
        ),
    ] = None,
    wm_itp_decision_temperature: Annotated[
        float | None,
        typer.Option(
            "--wm-itp-decision-temperature", min=0.0,
            help="Temperature for the policy's adaptive-K decision. Sets WM_ITP_DECISION_TEMPERATURE.",
        ),
    ] = None,
    wm_itp_world_model_temperature: Annotated[
        float | None,
        typer.Option(
            "--wm-itp-world-model-temperature", min=0.0,
            help="Temperature for ITP-I K-step WM imagination. Sets WM_ITP_WORLD_MODEL_TEMPERATURE.",
        ),
    ] = None,
    wm_itp_foresight_tokens: Annotated[
        int | None,
        typer.Option(
            "--wm-itp-foresight-tokens", min=1,
            help="Maximum tokens in one ITP-I imagined trajectory. Sets WM_ITP_FORESIGHT_TOKENS.",
        ),
    ] = None,
    wm_itp_world_model_model: Annotated[
        str | None,
        typer.Option(
            "--wm-itp-world-model-model",
            help="Served generative WM model used by ITP-I. Defaults to --wm-ewm-model. Sets WM_ITP_WORLD_MODEL_MODEL.",
        ),
    ] = None,
    wm_itp_world_model_base_url: Annotated[
        str | None,
        typer.Option(
            "--wm-itp-world-model-base-url",
            help="OpenAI-compatible endpoint for ITP-I imagination. Sets WM_ITP_WORLD_MODEL_BASE_URL.",
        ),
    ] = None,
    wm_imagined_candidate_actions: Annotated[
        int | None,
        typer.Option(
            "--wm-imagined-candidate-actions",
            min=1,
            help="Candidate actions per imagined step (topk_search). Sets WM_IMAGINED_CANDIDATE_ACTIONS.",
        ),
    ] = None,
    wm_imagined_top_k: Annotated[
        int | None,
        typer.Option(
            "--wm-imagined-top-k", min=1, help="Beams kept per imagined step (topk_search). Sets WM_IMAGINED_TOP_K."
        ),
    ] = None,
    wm_imagined_temperature: Annotated[
        float | None,
        typer.Option(
            "--wm-imagined-temperature",
            help="Sampling temperature for imagined candidate actions. Sets WM_IMAGINED_TEMPERATURE.",
        ),
    ] = None,
    wm_sample_temperature_ladder: Annotated[
        bool | None,
        typer.Option(
            "--wm-sample-temperature-ladder",
            "--sample-temperature-ladder",
            help="Spread open-loop beam-plan samples over a temperature ladder instead of one temperature. "
            "Forfeits the single-request n=k path. Sets WM_SAMPLE_TEMPERATURE_LADDER.",
        ),
    ] = None,
    wm_sample_temperature_ladder_max: Annotated[
        float | None,
        typer.Option(
            "--wm-sample-temperature-ladder-max",
            "--sample-temperature-ladder-max",
            help="Upper bound for the open-loop sample-temperature ladder. Sets WM_SAMPLE_TEMPERATURE_LADDER_MAX.",
        ),
    ] = None,
    wm_beam_plan_ssot_diversity: Annotated[
        bool | None,
        typer.Option(
            "--wm-beam-plan-ssot-diversity",
            help="Use String-Seed-of-Thought-style shared-prompt diversity for open-loop beam-plan samples. "
            "Sets WM_BEAM_PLAN_SSOT_DIVERSITY.",
        ),
    ] = None,
    wm_beam_plan_refinement_rounds: Annotated[
        int | None,
        typer.Option(
            "--wm-beam-plan-refinement-rounds",
            min=1,
            help="Total open-loop trajectory generation/scoring rounds. Rounds after the first "
            "receive prior trajectories and WM scores and refine them. Values above 1 imply "
            "open-loop planning. Sets WM_BEAM_PLAN_REFINEMENT_ROUNDS.",
        ),
    ] = None,
    wm_beam_plan_refinement_top_k: Annotated[
        int | None,
        typer.Option(
            "--wm-beam-plan-refinement-top-k",
            min=1,
            help="Number of prior WM-ranked trajectories included in each iterative refinement "
            "prompt. Sets WM_BEAM_PLAN_REFINEMENT_TOP_K.",
        ),
    ] = None,
    wm_beam_action_sampler_backend: Annotated[
        str | None,
        typer.Option(
            "--wm-beam-action-sampler-backend",
            help="Dedicated open-loop action-sampler backend: openai | huggingface. "
            "The Hugging Face backend loads DiffusionGemma in the executor process. "
            "Sets WM_BEAM_ACTION_SAMPLER_BACKEND.",
        ),
    ] = None,
    wm_beam_action_sampler_base_url: Annotated[
        str | None,
        typer.Option(
            "--wm-beam-action-sampler-base-url",
            help="OpenAI-compatible endpoint dedicated to open-loop beam action sampling. "
            "When set, candidate plans come from this server while the policy agent and JEPA "
            "world model remain unchanged. Sets WM_BEAM_ACTION_SAMPLER_BASE_URL.",
        ),
    ] = None,
    wm_beam_action_sampler_model: Annotated[
        str | None,
        typer.Option(
            "--wm-beam-action-sampler-model",
            help="Dedicated sampler model. Defaults to NVIDIA NVFP4 for openai and "
            "google/diffusiongemma-26B-A4B-it for huggingface. "
            "Sets WM_BEAM_ACTION_SAMPLER_MODEL.",
        ),
    ] = None,
    wm_beam_action_sampler_device_map: Annotated[
        str | None,
        typer.Option(
            "--wm-beam-action-sampler-device-map",
            help="Hugging Face device map (default: auto). Sets WM_BEAM_ACTION_SAMPLER_DEVICE_MAP.",
        ),
    ] = None,
    wm_beam_action_sampler_dtype: Annotated[
        str | None,
        typer.Option(
            "--wm-beam-action-sampler-dtype",
            help="Hugging Face model dtype (default: bfloat16). Sets WM_BEAM_ACTION_SAMPLER_DTYPE.",
        ),
    ] = None,
    wm_beam_action_sampler_max_new_tokens: Annotated[
        int | None,
        typer.Option(
            "--wm-beam-action-sampler-max-new-tokens",
            min=1,
            help="Maximum output tokens for each dedicated action-sampler plan (default 256, "
            "one DiffusionGemma canvas). Sets WM_BEAM_ACTION_SAMPLER_MAX_NEW_TOKENS.",
        ),
    ] = None,
    wm_beam_action_sampler_timeout: Annotated[
        float | None,
        typer.Option(
            "--wm-beam-action-sampler-timeout",
            min=0.1,
            help="Request timeout in seconds for the dedicated action sampler. "
            "Sets WM_BEAM_ACTION_SAMPLER_TIMEOUT.",
        ),
    ] = None,
    wm_beam_action_sampler_max_denoising_steps: Annotated[
        int | None,
        typer.Option(
            "--wm-beam-action-sampler-max-denoising-steps",
            min=1,
            help="Optional Hugging Face DiffusionGemma denoising-step limit. "
            "By default the checkpoint generation config is used. "
            "Sets WM_BEAM_ACTION_SAMPLER_MAX_DENOISING_STEPS.",
        ),
    ] = None,
    wm_imagined_rollouts: Annotated[
        int | None,
        typer.Option(
            "--wm-imagined-rollouts",
            min=1,
            help="Number of independent imagined rollouts to generate (first deterministic, rest "
            "sampled). >1 pairs with --wm-imagined-selection llm_judge. Sets WM_IMAGINED_ROLLOUTS.",
        ),
    ] = None,
    wm_imagined_selection: Annotated[
        str | None,
        typer.Option(
            "--wm-imagined-selection",
            help="How to pick among imagined rollouts: first | llm_judge (LLM reranks decoded "
            "action->state rollouts; no world-model scoring). Sets WM_IMAGINED_SELECTION.",
        ),
    ] = None,
    wm_state_history_size: Annotated[
        int | None,
        typer.Option(
            "--wm-state-history-size",
            min=0,
            help="Number of prior states in the WM prompt history. Sets WM_STATE_HISTORY_SIZE.",
        ),
    ] = None,
    wm_ewm_max_new_tokens: Annotated[
        int | None,
        typer.Option(
            "--wm-ewm-max-new-tokens",
            min=1,
            help="Max new tokens the EWM/JEPA world model generates per prediction. Sets WM_EWM_MAX_NEW_TOKENS.",
        ),
    ] = None,
    wm_ewm_backend: Annotated[
        str | None,
        typer.Option(
            "--wm-ewm-backend",
            help="EWM world-model flavor for ewm_imagined: jepa | llm_canonical_trained | "
            "llm_canonical_zeroshot | llm_tool_output_judge | llm_canonical_event | "
            "(unset -> text LLM over vLLM/MCP). Sets WM_EWM_BACKEND.",
        ),
    ] = None,
    wm_llm_ewm_mode: Annotated[
        str | None,
        typer.Option(
            "--wm-llm-ewm-mode",
            "--wm-ewm-llm-mode",
            help="Explicit LLM-EWM mode: llm_canonical_trained | llm_canonical_zeroshot | "
            "llm_tool_output_judge. Overrides WM_EWM_BACKEND for LLM modes. Sets WM_LLM_EWM_MODE.",
        ),
    ] = None,
    wm_ewm_jepa_checkpoint: Annotated[
        str | None,
        typer.Option(
            "--wm-ewm-jepa-checkpoint",
            help="Use a text-JEPA world model for ewm_imagined instead of a text LLM: path to the JEPA "
            "checkpoint dir (text_leworldmodel.pt + backbone/). Sets WM_EWM_JEPA_CHECKPOINT.",
        ),
    ] = None,
    wm_ewm_llm_canonical_event_checkpoint: Annotated[
        str | None,
        typer.Option(
            "--wm-ewm-llm-canonical-event-checkpoint",
            help="Use a fine-tuned causal-LM world model for ewm_imagined's beam_plan/critic "
            "(scored via forced-choice next-token softmax instead of JEPA's classifier heads -- "
            "see _ewm_llm_canonical_event.py): path to the HF checkpoint dir (a model trained with "
            "world_model_target=canonical_event_with_nudge). Sets WM_EWM_LLM_CANONICAL_EVENT_CHECKPOINT.",
        ),
    ] = None,
    wm_jepa_observation_backend: Annotated[
        str | None,
        typer.Option(
            "--wm-jepa-observation-backend",
            help="Which JEPA head reconstructs the imagined observation: "
            "auto | canonical_event | success | decoder. Sets WM_JEPA_OBSERVATION_BACKEND.",
        ),
    ] = None,
    wm_jepa_dtype: Annotated[
        str | None,
        typer.Option(
            "--wm-jepa-dtype",
            help="torch dtype for the JEPA world model (auto|float16|bfloat16|float32). Sets WM_JEPA_DTYPE.",
        ),
    ] = None,
    wm_jepa_trust_remote_code: Annotated[
        bool | None,
        typer.Option(
            "--wm-jepa-trust-remote-code/--no-wm-jepa-trust-remote-code",
            help="Pass trust_remote_code to the JEPA tokenizer/backbone loaders. Sets WM_JEPA_TRUST_REMOTE_CODE.",
        ),
    ] = None,
    wm_jepa_arch_defaults: Annotated[
        str | None,
        typer.Option(
            "--wm-jepa-arch-defaults",
            help="JSON object of JEPA architecture fallbacks used only when the checkpoint has no "
            'jepa_data_manifest.json (e.g. \'{"backbone_type":"seq2seq","goal_conditioning":true}\'). '
            "A value present in the manifest always wins. Sets WM_JEPA_ARCH_DEFAULTS.",
        ),
    ] = None,
    wm_jepa_merge_checkpoint: Annotated[
        str | None,
        typer.Option(
            "--wm-jepa-merge-checkpoint",
            help="Merge action_decoder_*/obs_ground_* weights from a SEPARATE JEPA checkpoint into "
            "the primary --wm-ewm-jepa-checkpoint. For checkpoints whose canonical-event-head "
            "training run dropped these optional modules (the base checkpoint it was initialized "
            "from had no jepa_data_manifest.json to preserve them). Architecture dims are inferred "
            "from the merge checkpoint's own tensor shapes -- no extra config needed. "
            "Sets WM_JEPA_MERGE_CHECKPOINT.",
        ),
    ] = None,
    wm_jepa_merge_decode_max_new_tokens: Annotated[
        int | None,
        typer.Option(
            "--wm-jepa-merge-decode-max-new-tokens", min=1,
            help="Default max tokens generated per decode_action_latent/decode_observation_latent "
            "call when using a merged decoder (overridable per-call). Sets WM_JEPA_MERGE_DECODE_MAX_NEW_TOKENS.",
        ),
    ] = None,
    wm_beam_plan_decode_tool_output: Annotated[
        bool | None,
        typer.Option(
            "--wm-beam-plan-decode-tool-output/--no-wm-beam-plan-decode-tool-output",
            help="beam_plan: also decode the predicted TOOL OUTPUT text (not just canonical-event "
            "labels) for the winning, confidence-gated trajectory, and inject it alongside the "
            "predicted state. Requires a checkpoint with a trained obs_grounding decoder (native, "
            "or merged via --wm-jepa-merge-checkpoint). Default off (decoding is a generative "
            "greedy loop, run once per re-plan cycle, not per candidate). "
            "Sets WM_BEAM_PLAN_DECODE_TOOL_OUTPUT.",
        ),
    ] = None,
    wm_beam_plan_decode_max_new_tokens: Annotated[
        int | None,
        typer.Option(
            "--wm-beam-plan-decode-max-new-tokens", min=1,
            help="Max tokens generated per decoded tool-output step when "
            "--wm-beam-plan-decode-tool-output is set. Sets WM_BEAM_PLAN_DECODE_MAX_NEW_TOKENS.",
        ),
    ] = None,
    wm_beam_plan_samples: Annotated[
        int | None,
        typer.Option(
            "--wm-beam-plan-samples",
            min=1,
            help="beam_plan: m candidate next-actions the agent proposes per horizon step. "
            "Sets WM_BEAM_PLAN_SAMPLES.",
        ),
    ] = None,
    wm_beam_plan_horizon: Annotated[
        int | None,
        typer.Option(
            "--wm-beam-plan-horizon",
            min=1,
            help="beam_plan: n lookahead steps the world model rolls each plan forward. "
            "Sets WM_BEAM_PLAN_HORIZON.",
        ),
    ] = None,
    wm_beam_mpc_execute_steps: Annotated[
        int | None,
        typer.Option(
            "--wm-beam-mpc-execute-steps",
            min=1,
            help="beam_plan: MPC cadence — steps the agent follows the cached imagined plan "
            "before re-planning. Sets WM_BEAM_MPC_EXECUTE_STEPS.",
        ),
    ] = None,
    wm_beam_plan_score_margin: Annotated[
        float | None,
        typer.Option(
            "--wm-beam-plan-score-margin",
            help="beam_plan: keep the agent's baseline action unless the top beam candidate beats "
            "it by more than this margin. Sets WM_BEAM_PLAN_SCORE_MARGIN.",
        ),
    ] = None,
    wm_beam_plan_diversity_multiplier: Annotated[
        int | None,
        typer.Option(
            "--wm-beam-plan-diversity-multiplier",
            min=1,
            help="beam_plan: scales the anti-repetition penalty on candidates whose tool-name set "
            "was already executed. Sets WM_BEAM_PLAN_DIVERSITY_MULTIPLIER.",
        ),
    ] = None,
    wm_beam_plan_hard_override: Annotated[
        bool | None,
        typer.Option(
            "--wm-beam-plan-hard-override/--no-wm-beam-plan-hard-override",
            help="beam_plan / hier_latent_cem: force the confident recommendation over the "
            "agent's own action (default off = advisory: the agent acts, the plan is injected "
            "only as guidance). Shared knob for both MPC strategies. Sets WM_BEAM_PLAN_HARD_OVERRIDE.",
        ),
    ] = None,
    wm_beam_plan_revision: Annotated[
        bool | None,
        typer.Option(
            "--wm-beam-plan-revision/--no-wm-beam-plan-revision",
            help="beam_plan: at every live horizon step, ask the acting policy to choose between "
            "its fresh action and the corresponding planned action. Choosing the policy action "
            "invalidates the remaining plan. Sets WM_BEAM_PLAN_REVISION.",
        ),
    ] = None,
    wm_beam_plan_revision_temperature: Annotated[
        float | None,
        typer.Option(
            "--wm-beam-plan-revision-temperature",
            min=0.0,
            help="Temperature for the binary same-agent beam revision decision. "
            "Sets WM_BEAM_PLAN_REVISION_TEMPERATURE.",
        ),
    ] = None,
    wm_beam_plan_read_saturation_threshold: Annotated[
        int | None,
        typer.Option(
            "--wm-beam-plan-read-saturation-threshold",
            min=0,
            help="Distinct consecutive successful reads before beam scoring applies the full "
            "read/search penalty. Sets WM_BEAM_PLAN_READ_SATURATION_THRESHOLD.",
        ),
    ] = None,
    wm_beam_plan_read_penalty: Annotated[
        float | None,
        typer.Option(
            "--wm-beam-plan-read-penalty",
            min=0.0,
            help="Maximum per-step read/search penalty after information saturation. "
            "Sets WM_BEAM_PLAN_READ_PENALTY.",
        ),
    ] = None,
    wm_beam_plan_read_after_progress_scale: Annotated[
        float | None,
        typer.Option(
            "--wm-beam-plan-read-after-progress-scale",
            min=0.0,
            help="Read/search penalty pressure after a successful non-read action. "
            "Sets WM_BEAM_PLAN_READ_AFTER_PROGRESS_SCALE.",
        ),
    ] = None,
    wm_beam_plan_first_step_weight: Annotated[
        float | None,
        typer.Option(
            "--wm-beam-plan-first-step-weight",
            min=0.0,
            help="Additional weight on the immediately executable first-step score. "
            "Sets WM_BEAM_PLAN_FIRST_STEP_WEIGHT.",
        ),
    ] = None,
    wm_beam_plan_required_action_coverage_bonus: Annotated[
        float | None,
        typer.Option(
            "--wm-beam-plan-required-action-coverage-bonus",
            min=0.0,
            help="Bonus for covering operation types explicitly required by the task. "
            "Sets WM_BEAM_PLAN_REQUIRED_ACTION_COVERAGE_BONUS.",
        ),
    ] = None,
    wm_beam_plan_first_required_action_bonus: Annotated[
        float | None,
        typer.Option(
            "--wm-beam-plan-first-required-action-bonus",
            min=0.0,
            help="Bonus when the first planned step performs an explicitly required operation "
            "type. Sets WM_BEAM_PLAN_FIRST_REQUIRED_ACTION_BONUS.",
        ),
    ] = None,
    wm_beam_plan_trigger: Annotated[
        str | None,
        typer.Option(
            "--wm-beam-plan-trigger",
            help="beam_plan: when to spend a planning cycle. 'interval' (default) re-plans every "
            "--wm-beam-mpc-execute-steps steps regardless of need. 'critic' first scores the "
            "agent's own action with the world model (one latent forward, no LLM call) and only "
            "plans when that score says the action is bad -- see --wm-beam-plan-critic-*. "
            "Sets WM_BEAM_PLAN_TRIGGER.",
        ),
    ] = None,
    wm_beam_plan_critic_failure_prob: Annotated[
        float | None,
        typer.Option(
            "--wm-beam-plan-critic-failure-prob",
            help="critic trigger: plan when the predicted P(execution_status=failure) reaches "
            "this. Sets WM_BEAM_PLAN_CRITIC_FAILURE_PROB.",
        ),
    ] = None,
    wm_beam_plan_critic_stall_prob: Annotated[
        float | None,
        typer.Option(
            "--wm-beam-plan-critic-stall-prob",
            help="critic trigger: plan when the action looks like it will not advance the task, "
            "i.e. 1 - P(progress_signal=positive) reaches this. Sets WM_BEAM_PLAN_CRITIC_STALL_PROB.",
        ),
    ] = None,
    wm_beam_plan_critic_veto_fires: Annotated[
        bool | None,
        typer.Option(
            "--wm-beam-plan-critic-veto-fires/--no-wm-beam-plan-critic-veto-fires",
            help="critic trigger: also plan whenever the score config's own P(failure)/P(deleted) "
            "safety limits veto the action (default). This signal is independent of the two "
            "thresholds above and sets a floor on the fire rate; disable to make "
            "--wm-beam-plan-critic-failure-prob/--wm-beam-plan-critic-stall-prob the only levers. "
            "Sets WM_BEAM_PLAN_CRITIC_VETO_FIRES.",
        ),
    ] = None,
    wm_beam_plan_critic_max_quiet_steps: Annotated[
        int | None,
        typer.Option(
            "--wm-beam-plan-critic-max-quiet-steps",
            min=0,
            help="critic trigger safety valve: force a planning cycle after this many consecutive "
            "non-firing steps, so a mis-calibrated critic cannot disable lookahead for a whole "
            "episode. 0 (default) is purely event-driven. Sets WM_BEAM_PLAN_CRITIC_MAX_QUIET_STEPS.",
        ),
    ] = None,
    wm_beam_plan_terminal_advice: Annotated[
        bool | None,
        typer.Option(
            "--wm-beam-plan-terminal-advice/--no-wm-beam-plan-terminal-advice",
            help="beam_plan: use a JEPA terminal head, when present, to add a prompt advisory "
            "when P(done) is high. Sets WM_BEAM_PLAN_TERMINAL_ADVICE.",
        ),
    ] = None,
    wm_beam_plan_terminal_advice_threshold: Annotated[
        float | None,
        typer.Option(
            "--wm-beam-plan-terminal-advice-threshold",
            help="P(done) threshold for --wm-beam-plan-terminal-advice. "
            "Sets WM_BEAM_PLAN_TERMINAL_ADVICE_THRESHOLD.",
        ),
    ] = None,
    wm_imagined_rollout_mode: Annotated[
        str | None,
        typer.Option(
            "--wm-imagined-rollout-mode",
            help="beam_plan / plain imagined rollout: 'closed_loop' (default) chains one agent "
            "call + one world-model call per horizon/rollout step; 'open_loop' samples complete "
            "candidate plans from one diversity-menu prompt up front (symbolic \"$stepK.field\" "
            "refs for values not yet known) and scores them in one batched world-model pass. Falls back "
            "to treating the cycle as 'no candidates' if the plan payload can't be parsed. "
            "No-op for hier_latent_cem (already 1 LLM call + 1 batched WM pass). "
            "Sets WM_IMAGINED_ROLLOUT_MODE.",
        ),
    ] = None,
    wm_llm_batch_parallelism: Annotated[
        int | None,
        typer.Option(
            "--wm-llm-batch-parallelism",
            min=1,
            help="open_loop rollout mode, text-WM backend only: max concurrent requests when "
            "batching per-plan world-model calls against a backend that opts into "
            "supports_parallel_requests (no-op serial loop otherwise). Sets WM_LLM_BATCH_PARALLELISM.",
        ),
    ] = None,
    wm_imagined_parallel_rollouts: Annotated[
        bool | None,
        typer.Option(
            "--wm-imagined-parallel-rollouts/--no-wm-imagined-parallel-rollouts",
            help="closed_loop multi-rollout path only (WM_IMAGINED_ROLLOUTS>1 or "
            "WM_IMAGINED_SELECTION=llm_judge): advance all rollouts in lockstep, batching each "
            "step's agent/world-model calls across rollouts, instead of running each rollout "
            "fully before starting the next. Default on; the single-rollout default path is "
            "unaffected either way. Sets WM_IMAGINED_PARALLEL_ROLLOUTS.",
        ),
    ] = None,
    wm_imagined_single_call_step: Annotated[
        bool | None,
        typer.Option(
            "--wm-imagined-single-call-step/--no-wm-imagined-single-call-step",
            help="Only read by the lockstep multi-rollout path (see "
            "--wm-imagined-parallel-rollouts): generate each imagined step's thought AND action "
            "in one combined LLM call instead of the two-call think-then-act sequence, halving "
            "the per-step agent call count again. Default on. Sets WM_IMAGINED_SINGLE_CALL_STEP.",
        ),
    ] = None,
    wm_hier_cem_anchors: Annotated[
        int | None,
        typer.Option(
            "--wm-hier-cem-anchors", min=1,
            help="hier_latent_cem: K diverse anchor actions proposed by the ONE LLM call per "
            "planning cycle. Sets WM_HIER_CEM_ANCHORS.",
        ),
    ] = None,
    wm_hier_cem_samples: Annotated[
        int | None,
        typer.Option(
            "--wm-hier-cem-samples", min=1,
            help="hier_latent_cem: N continuous latent-action trajectories sampled per CEM "
            "iteration (LLM-free). Sets WM_HIER_CEM_SAMPLES.",
        ),
    ] = None,
    wm_hier_cem_elites: Annotated[
        int | None,
        typer.Option(
            "--wm-hier-cem-elites", min=1,
            help="hier_latent_cem: M elite trajectories kept each CEM iteration to refit "
            "(pi, mean, std). Sets WM_HIER_CEM_ELITES.",
        ),
    ] = None,
    wm_hier_cem_iters: Annotated[
        int | None,
        typer.Option(
            "--wm-hier-cem-iters", min=1,
            help="hier_latent_cem: CEM refinement iterations per planning cycle. Sets WM_HIER_CEM_ITERS.",
        ),
    ] = None,
    wm_hier_cem_horizon: Annotated[
        int | None,
        typer.Option(
            "--wm-hier-cem-horizon", min=1,
            help="hier_latent_cem: lookahead steps per latent-action trajectory. Sets WM_HIER_CEM_HORIZON.",
        ),
    ] = None,
    wm_hier_cem_init_std: Annotated[
        float | None,
        typer.Option(
            "--wm-hier-cem-init-std",
            help="hier_latent_cem: initial per-family Gaussian std (fraction of |mean| for "
            "singleton-anchor families). Sets WM_HIER_CEM_INIT_STD.",
        ),
    ] = None,
    wm_hier_cem_min_std: Annotated[
        float | None,
        typer.Option(
            "--wm-hier-cem-min-std",
            help="hier_latent_cem: std floor preventing premature Gaussian collapse. Sets WM_HIER_CEM_MIN_STD.",
        ),
    ] = None,
    wm_hier_cem_smoothing: Annotated[
        float | None,
        typer.Option(
            "--wm-hier-cem-smoothing",
            help="hier_latent_cem: CEM update blend with the previous iteration's params "
            "(1.0 = classic full replace). Sets WM_HIER_CEM_SMOOTHING.",
        ),
    ] = None,
    wm_hier_cem_min_elite_agreement: Annotated[
        float | None,
        typer.Option(
            "--wm-hier-cem-min-elite-agreement",
            help="hier_latent_cem: fraction of final-iteration elites that must share the "
            "winning step-0 family for the plan to be called confident. Sets WM_HIER_CEM_MIN_ELITE_AGREEMENT.",
        ),
    ] = None,
    wm_hier_cem_decode_strategy: Annotated[
        str | None,
        typer.Option(
            "--wm-hier-cem-decode-strategy",
            help="hier_latent_cem: how to turn the converged latent trajectory into an "
            "executable action: nearest_anchor (default) | learned_decoder (requires a "
            "checkpoint trained with --action-decoder-loss-coeff > 0; falls back to "
            "nearest_anchor on failure). Sets WM_HIER_CEM_DECODE_STRATEGY.",
        ),
    ] = None,
    wm_hier_cem_decode_max_new_tokens: Annotated[
        int | None,
        typer.Option(
            "--wm-hier-cem-decode-max-new-tokens", min=1,
            help="hier_latent_cem: max tokens generated per learned-decoder decode call. "
            "Sets WM_HIER_CEM_DECODE_MAX_NEW_TOKENS.",
        ),
    ] = None,
    green_host: Annotated[
        str | None, typer.Option("--green-host", help="Green host を上書きする。")
    ] = None,
    green_port: Annotated[
        int | None, typer.Option("--green-port", help="Green port を上書きする。")
    ] = None,
    purple_host: Annotated[
        str | None, typer.Option("--purple-host", help="Purple host を上書きする。")
    ] = None,
    purple_port: Annotated[
        int | None, typer.Option("--purple-port", help="Purple port を上書きする。")
    ] = None,
    slurm_partition: Annotated[
        str | None,
        typer.Option("--slurm-partition", help="launcher=slurm 時の Slurm partition。"),
    ] = None,
    slurm_job_name: Annotated[
        str | None,
        typer.Option("--slurm-job-name", help="launcher=slurm 時の Slurm job name。"),
    ] = None,
    slurm_output: Annotated[
        str | None,
        typer.Option("--slurm-output", help="launcher=slurm 時の Slurm output path。"),
    ] = None,
    slurm_time: Annotated[
        str | None,
        typer.Option("--slurm-time", help="launcher=slurm 時の Slurm time limit。"),
    ] = None,
    slurm_mem: Annotated[
        str | None,
        typer.Option("--slurm-mem", help="launcher=slurm 時の Slurm memory 指定。"),
    ] = None,
    slurm_cpus_per_task: Annotated[
        int | None,
        typer.Option("--slurm-cpus-per-task", min=1, help="launcher=slurm 時の CPU 数。"),
    ] = None,
    slurm_gpus: Annotated[
        int | None,
        typer.Option("--slurm-gpus", min=1, help="launcher=slurm 時の GPU 数。"),
    ] = None,
    slurm_account: Annotated[
        str | None,
        typer.Option("--slurm-account", help="launcher=slurm 時の Slurm account。"),
    ] = None,
    slurm_qos: Annotated[
        str | None,
        typer.Option("--slurm-qos", help="launcher=slurm 時の Slurm qos。"),
    ] = None,
    slurm_constraint: Annotated[
        str | None,
        typer.Option("--slurm-constraint", help="launcher=slurm 時の Slurm constraint。"),
    ] = None,
    slurm_exclude: Annotated[
        str | None,
        typer.Option("--slurm-exclude", help="launcher=slurm 時の Slurm exclude nodelist。"),
    ] = None,
    slurm_args: Annotated[
        list[str] | None,
        typer.Option(
            "--slurm-arg",
            help="launcher=slurm 時に `sbatch` へそのまま追加する引数。複数指定可。",
        ),
    ] = None,
    ready_timeout: Annotated[
        int | None,
        typer.Option(
            "--ready-timeout",
            help="エージェント起動待ち秒数。未指定時は実行条件に応じた既定値を使う。",
        ),
    ] = None,
    show_logs: Annotated[
        bool,
        typer.Option("--show-logs/--no-show-logs", help="子プロセスの stdout/stderr を表示する。"),
    ] = False,
    serve_only: Annotated[
        bool, typer.Option("--serve-only", help="サーバーだけ起動して待機する。")
    ] = False,
    inference_config: Annotated[
        Path | None,
        typer.Option(
            "--inference-config",
            exists=True,
            dir_okay=False,
            resolve_path=True,
            help="benchmark 実行前に起動する inference launcher YAML config。",
        ),
    ] = None,
) -> None:
    """ベンチマークを実行、または Slurm へ送信する。"""
    # The World Model is pluggable regardless of executor: surface it as first-class CLI
    # args that map to the WM_* env vars ejepa_wm reads (wm_config_from_env). Setting them on
    # os.environ here propagates to the green/purple subprocesses (the env builder does
    # os.environ.copy()) and to the slurm submission env, so any executor that consults
    # ejepa_wm picks up the requested WM without needing a WM-specific executor.
    _wm_cli_env: dict[str, str | None] = {
        "WM_STRATEGY": wm_strategy,
        "WM_BACKEND": wm_backend,
        "WM_N": str(wm_n) if wm_n is not None else None,
        "WM_MODEL": wm_model,
        "WM_BASE_URL": wm_base_url,
        "WM_EWM_MCP_URL": wm_ewm_mcp_url,
        "WM_EWM_MODEL": wm_ewm_model,
        # ewm_imagined / imagined-trajectory options.
        "WM_STATE": wm_state,
        "ACTION_OPTIMIZER": wm_action_optimizer,
        "K_CONTROLLER": wm_k_controller,
        "K_CONTROLLER_MODEL_PATH": _absolute_cli_path(wm_k_controller_model_path),
        "K_CONTROLLER_DEVICE": wm_k_controller_device,
        "K_CONTROLLER_DTYPE": wm_k_controller_dtype,
        "WM_IMAGINED_MAX_STEPS": str(wm_imagined_max_steps) if wm_imagined_max_steps is not None else None,
        "WM_ITP_MAX_K": str(wm_itp_max_k) if wm_itp_max_k is not None else None,
        "WM_ITP_FIXED_K": str(wm_itp_fixed_k) if wm_itp_fixed_k is not None else None,
        "WM_ITP_DECISION_TEMPERATURE": (
            str(wm_itp_decision_temperature) if wm_itp_decision_temperature is not None else None
        ),
        "WM_ITP_WORLD_MODEL_TEMPERATURE": (
            str(wm_itp_world_model_temperature)
            if wm_itp_world_model_temperature is not None else None
        ),
        "WM_ITP_FORESIGHT_TOKENS": (
            str(wm_itp_foresight_tokens) if wm_itp_foresight_tokens is not None else None
        ),
        "WM_ITP_WORLD_MODEL_MODEL": wm_itp_world_model_model,
        "WM_ITP_WORLD_MODEL_BASE_URL": wm_itp_world_model_base_url,
        "WM_IMAGINED_CANDIDATE_ACTIONS": (
            str(wm_imagined_candidate_actions) if wm_imagined_candidate_actions is not None else None
        ),
        "WM_IMAGINED_TOP_K": str(wm_imagined_top_k) if wm_imagined_top_k is not None else None,
        "WM_IMAGINED_TEMPERATURE": (
            str(wm_imagined_temperature) if wm_imagined_temperature is not None else None
        ),
        "WM_SAMPLE_TEMPERATURE_LADDER": (
            None if wm_sample_temperature_ladder is None else ("1" if wm_sample_temperature_ladder else "0")
        ),
        "WM_SAMPLE_TEMPERATURE_LADDER_MAX": (
            str(wm_sample_temperature_ladder_max)
            if wm_sample_temperature_ladder_max is not None else None
        ),
        "WM_BEAM_PLAN_SSOT_DIVERSITY": (
            None if wm_beam_plan_ssot_diversity is None else ("1" if wm_beam_plan_ssot_diversity else "0")
        ),
        "WM_BEAM_PLAN_REFINEMENT_ROUNDS": (
            str(wm_beam_plan_refinement_rounds)
            if wm_beam_plan_refinement_rounds is not None
            else None
        ),
        "WM_BEAM_PLAN_REFINEMENT_TOP_K": (
            str(wm_beam_plan_refinement_top_k)
            if wm_beam_plan_refinement_top_k is not None
            else None
        ),
        "WM_BEAM_ACTION_SAMPLER_BACKEND": wm_beam_action_sampler_backend,
        "WM_BEAM_ACTION_SAMPLER_BASE_URL": wm_beam_action_sampler_base_url,
        "WM_BEAM_ACTION_SAMPLER_MODEL": wm_beam_action_sampler_model,
        "WM_BEAM_ACTION_SAMPLER_DEVICE_MAP": wm_beam_action_sampler_device_map,
        "WM_BEAM_ACTION_SAMPLER_DTYPE": wm_beam_action_sampler_dtype,
        "WM_BEAM_ACTION_SAMPLER_MAX_NEW_TOKENS": (
            str(wm_beam_action_sampler_max_new_tokens)
            if wm_beam_action_sampler_max_new_tokens is not None else None
        ),
        "WM_BEAM_ACTION_SAMPLER_TIMEOUT": (
            str(wm_beam_action_sampler_timeout)
            if wm_beam_action_sampler_timeout is not None else None
        ),
        "WM_BEAM_ACTION_SAMPLER_MAX_DENOISING_STEPS": (
            str(wm_beam_action_sampler_max_denoising_steps)
            if wm_beam_action_sampler_max_denoising_steps is not None else None
        ),
        "WM_IMAGINED_ROLLOUTS": str(wm_imagined_rollouts) if wm_imagined_rollouts is not None else None,
        "WM_IMAGINED_SELECTION": wm_imagined_selection,
        "WM_STATE_HISTORY_SIZE": str(wm_state_history_size) if wm_state_history_size is not None else None,
        "WM_EWM_MAX_NEW_TOKENS": str(wm_ewm_max_new_tokens) if wm_ewm_max_new_tokens is not None else None,
        # JEPA world model for ewm_imagined (predicts the imagined observation/state in latent
        # space via its heads instead of an LLM's text output).
        "WM_EWM_BACKEND": wm_ewm_backend,
        "WM_LLM_EWM_MODE": wm_llm_ewm_mode,
        "WM_EWM_JEPA_CHECKPOINT": _absolute_cli_path(wm_ewm_jepa_checkpoint),
        "WM_EWM_LLM_CANONICAL_EVENT_CHECKPOINT": _absolute_cli_path(
            wm_ewm_llm_canonical_event_checkpoint
        ),
        "WM_JEPA_OBSERVATION_BACKEND": wm_jepa_observation_backend,
        "WM_JEPA_DTYPE": wm_jepa_dtype,
        "WM_JEPA_TRUST_REMOTE_CODE": (
            None if wm_jepa_trust_remote_code is None else ("1" if wm_jepa_trust_remote_code else "0")
        ),
        "WM_JEPA_ARCH_DEFAULTS": wm_jepa_arch_defaults,
        "WM_JEPA_MERGE_CHECKPOINT": _absolute_cli_path(wm_jepa_merge_checkpoint),
        "WM_JEPA_MERGE_DECODE_MAX_NEW_TOKENS": (
            str(wm_jepa_merge_decode_max_new_tokens) if wm_jepa_merge_decode_max_new_tokens is not None else None
        ),
        "WM_BEAM_PLAN_DECODE_TOOL_OUTPUT": (
            None if wm_beam_plan_decode_tool_output is None else ("1" if wm_beam_plan_decode_tool_output else "0")
        ),
        "WM_BEAM_PLAN_DECODE_MAX_NEW_TOKENS": (
            str(wm_beam_plan_decode_max_new_tokens) if wm_beam_plan_decode_max_new_tokens is not None else None
        ),
        # beam_plan (JEPA-only MPC lookahead) knobs.
        "WM_BEAM_PLAN_SAMPLES": str(wm_beam_plan_samples) if wm_beam_plan_samples is not None else None,
        "WM_BEAM_PLAN_HORIZON": str(wm_beam_plan_horizon) if wm_beam_plan_horizon is not None else None,
        "WM_BEAM_MPC_EXECUTE_STEPS": (
            str(wm_beam_mpc_execute_steps) if wm_beam_mpc_execute_steps is not None else None
        ),
        "WM_BEAM_PLAN_SCORE_MARGIN": (
            str(wm_beam_plan_score_margin) if wm_beam_plan_score_margin is not None else None
        ),
        "WM_BEAM_PLAN_DIVERSITY_MULTIPLIER": (
            str(wm_beam_plan_diversity_multiplier) if wm_beam_plan_diversity_multiplier is not None else None
        ),
        "WM_BEAM_PLAN_HARD_OVERRIDE": (
            None if wm_beam_plan_hard_override is None else ("1" if wm_beam_plan_hard_override else "0")
        ),
        "WM_BEAM_PLAN_REVISION": (
            None if wm_beam_plan_revision is None else ("1" if wm_beam_plan_revision else "0")
        ),
        "WM_BEAM_PLAN_REVISION_TEMPERATURE": (
            str(wm_beam_plan_revision_temperature)
            if wm_beam_plan_revision_temperature is not None
            else None
        ),
        "WM_BEAM_PLAN_READ_SATURATION_THRESHOLD": (
            str(wm_beam_plan_read_saturation_threshold)
            if wm_beam_plan_read_saturation_threshold is not None
            else None
        ),
        "WM_BEAM_PLAN_READ_PENALTY": (
            str(wm_beam_plan_read_penalty) if wm_beam_plan_read_penalty is not None else None
        ),
        "WM_BEAM_PLAN_READ_AFTER_PROGRESS_SCALE": (
            str(wm_beam_plan_read_after_progress_scale)
            if wm_beam_plan_read_after_progress_scale is not None
            else None
        ),
        "WM_BEAM_PLAN_FIRST_STEP_WEIGHT": (
            str(wm_beam_plan_first_step_weight)
            if wm_beam_plan_first_step_weight is not None
            else None
        ),
        "WM_BEAM_PLAN_REQUIRED_ACTION_COVERAGE_BONUS": (
            str(wm_beam_plan_required_action_coverage_bonus)
            if wm_beam_plan_required_action_coverage_bonus is not None
            else None
        ),
        "WM_BEAM_PLAN_FIRST_REQUIRED_ACTION_BONUS": (
            str(wm_beam_plan_first_required_action_bonus)
            if wm_beam_plan_first_required_action_bonus is not None
            else None
        ),
        "WM_BEAM_PLAN_TRIGGER": wm_beam_plan_trigger,
        "WM_BEAM_PLAN_CRITIC_FAILURE_PROB": (
            str(wm_beam_plan_critic_failure_prob) if wm_beam_plan_critic_failure_prob is not None else None
        ),
        "WM_BEAM_PLAN_CRITIC_STALL_PROB": (
            str(wm_beam_plan_critic_stall_prob) if wm_beam_plan_critic_stall_prob is not None else None
        ),
        "WM_BEAM_PLAN_CRITIC_VETO_FIRES": (
            None
            if wm_beam_plan_critic_veto_fires is None
            else ("1" if wm_beam_plan_critic_veto_fires else "0")
        ),
        "WM_BEAM_PLAN_CRITIC_MAX_QUIET_STEPS": (
            str(wm_beam_plan_critic_max_quiet_steps) if wm_beam_plan_critic_max_quiet_steps is not None else None
        ),
        "WM_BEAM_PLAN_TERMINAL_ADVICE": (
            None if wm_beam_plan_terminal_advice is None else ("1" if wm_beam_plan_terminal_advice else "0")
        ),
        "WM_BEAM_PLAN_TERMINAL_ADVICE_THRESHOLD": (
            str(wm_beam_plan_terminal_advice_threshold)
            if wm_beam_plan_terminal_advice_threshold is not None else None
        ),
        "WM_IMAGINED_ROLLOUT_MODE": wm_imagined_rollout_mode,
        "WM_LLM_BATCH_PARALLELISM": str(wm_llm_batch_parallelism) if wm_llm_batch_parallelism is not None else None,
        "WM_IMAGINED_PARALLEL_ROLLOUTS": (
            None if wm_imagined_parallel_rollouts is None else ("1" if wm_imagined_parallel_rollouts else "0")
        ),
        "WM_IMAGINED_SINGLE_CALL_STEP": (
            None if wm_imagined_single_call_step is None else ("1" if wm_imagined_single_call_step else "0")
        ),
        # hier_latent_cem (JEPA-only hierarchical latent-action CEM MPC) knobs.
        "WM_HIER_CEM_ANCHORS": str(wm_hier_cem_anchors) if wm_hier_cem_anchors is not None else None,
        "WM_HIER_CEM_SAMPLES": str(wm_hier_cem_samples) if wm_hier_cem_samples is not None else None,
        "WM_HIER_CEM_ELITES": str(wm_hier_cem_elites) if wm_hier_cem_elites is not None else None,
        "WM_HIER_CEM_ITERS": str(wm_hier_cem_iters) if wm_hier_cem_iters is not None else None,
        "WM_HIER_CEM_HORIZON": str(wm_hier_cem_horizon) if wm_hier_cem_horizon is not None else None,
        "WM_HIER_CEM_INIT_STD": str(wm_hier_cem_init_std) if wm_hier_cem_init_std is not None else None,
        "WM_HIER_CEM_MIN_STD": str(wm_hier_cem_min_std) if wm_hier_cem_min_std is not None else None,
        "WM_HIER_CEM_SMOOTHING": str(wm_hier_cem_smoothing) if wm_hier_cem_smoothing is not None else None,
        "WM_HIER_CEM_MIN_ELITE_AGREEMENT": (
            str(wm_hier_cem_min_elite_agreement) if wm_hier_cem_min_elite_agreement is not None else None
        ),
        "WM_HIER_CEM_DECODE_STRATEGY": wm_hier_cem_decode_strategy,
        "WM_HIER_CEM_DECODE_MAX_NEW_TOKENS": (
            str(wm_hier_cem_decode_max_new_tokens) if wm_hier_cem_decode_max_new_tokens is not None else None
        ),
    }
    for _wm_key, _wm_val in _wm_cli_env.items():
        if _wm_val is not None and str(_wm_val).strip():
            os.environ[_wm_key] = str(_wm_val)
    state = _state_from_ctx(ctx)
    records = _discover_benchmarks(state.assets_root)
    record = _resolve_benchmark(records, benchmark_name)
    benchmark_dir, raw_config = _load_benchmark_config(state.assets_root, record.benchmark_dir.name)
    overrides = RuntimeConfigOverrides(
        target=target,
        config_overrides=config_overrides,
        task_ids=task_ids,
        max_parallel=max_parallel,
        vllm_model_id=vllm_model_id,
        green_host=green_host,
        green_port=green_port,
        purple_host=purple_host,
        purple_port=purple_port,
    )
    runtime_config = _build_runtime_config(
        raw_config,
        overrides=overrides,
    )
    executor_name = executor or runtime_config.get("default_executor") or "default"
    effective_ready_timeout = ready_timeout or _default_ready_timeout_for_launcher(
        record.name,
        executor_name,
        launcher,
    )
    effective_slurm_exclude = slurm_exclude or _default_slurm_exclude_for_launcher(
        record.name,
        executor_name,
        launcher,
    )
    resolved_workdir = workdir.expanduser().resolve()
    run_context = BenchmarkRunContext(
        state=state,
        record=record,
        benchmark_dir=benchmark_dir,
        raw_config=raw_config,
        runtime_config=runtime_config,
        executor_name=executor_name,
        launcher=launcher,
        workdir=resolved_workdir,
        ready_timeout=effective_ready_timeout,
        show_logs=show_logs,
        serve_only=serve_only,
        inference_config=inference_config,
    )

    if launcher is ExecutionLauncher.slurm:
        slurm_options = SlurmOptions(
            partition=slurm_partition,
            job_name=slurm_job_name,
            output=slurm_output,
            time_limit=slurm_time,
            mem=slurm_mem,
            cpus_per_task=slurm_cpus_per_task,
            gpus=slurm_gpus,
            account=slurm_account,
            qos=slurm_qos,
            constraint=slurm_constraint,
            exclude=effective_slurm_exclude,
            extra_args=list(slurm_args or []),
        )
        _run_benchmark_via_slurm(run_context, slurm_options)
        return

    _run_benchmark_locally(run_context)


@result_app.command("list")
def result_list(
    ctx: typer.Context,
    query: Annotated[str | None, typer.Option("--query", help="任意の文字列で絞り込む。")] = None,
    status: Annotated[
        ResultStatusFilter, typer.Option("--status", help="status で絞り込む。")
    ] = ResultStatusFilter.all,
    benchmark_name: Annotated[
        str | None, typer.Option("--benchmark", help="ベンチマーク名で絞り込む。")
    ] = None,
    executor_name: Annotated[
        str | None, typer.Option("--executor", help="executor 名で絞り込む。")
    ] = None,
    target: Annotated[str | None, typer.Option("--target", help="target で絞り込む。")] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="出力形式。")
    ] = OutputFormat.table,
) -> None:
    """保存済みベンチマーク結果を一覧表示する。"""
    state = _state_from_ctx(ctx)
    manifests = _filter_result_manifests(
        _load_result_manifests(state.result_root),
        query=query,
        status=status,
        benchmark_name=benchmark_name,
        executor_name=executor_name,
        target=target,
    )
    if output_format is OutputFormat.json:
        console.print_json(
            _json_dumps([manifest.model_dump(mode="json") for manifest in manifests])
        )
        return
    _render_result_table(manifests, state.result_root)


@result_app.command("search")
def result_search(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="検索文字列。")],
    status: Annotated[
        ResultStatusFilter, typer.Option("--status", help="status で絞り込む。")
    ] = ResultStatusFilter.all,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="出力形式。")
    ] = OutputFormat.table,
) -> None:
    """保存済み結果を検索する。"""
    result_list(ctx=ctx, query=query, status=status, output_format=output_format)


@result_app.command("compare")
def result_compare(
    ctx: typer.Context,
    identifiers: Annotated[
        list[str] | None,
        typer.Argument(
            help="比較する run_id, その prefix, または結果ディレクトリ/manifest パス。省略時は filter option で選ぶ。"
        ),
    ] = None,
    query: Annotated[str | None, typer.Option("--query", help="任意の文字列で絞り込む。")] = None,
    status: Annotated[
        ResultStatusFilter, typer.Option("--status", help="status で絞り込む。")
    ] = ResultStatusFilter.all,
    benchmark_name: Annotated[
        str | None, typer.Option("--benchmark", help="ベンチマーク名で絞り込む。")
    ] = None,
    executor_name: Annotated[
        str | None, typer.Option("--executor", help="executor 名で絞り込む。")
    ] = None,
    target: Annotated[str | None, typer.Option("--target", help="target で絞り込む。")] = None,
    pass_threshold: Annotated[
        float,
        typer.Option("--pass-threshold", help="PASS とみなす score の閾値。"),
    ] = 1.0,
    k_values: Annotated[
        list[int] | None,
        typer.Option("--k", min=1, help="表示する PASS@k の k。複数指定可。"),
    ] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="出力形式。")
    ] = OutputFormat.table,
) -> None:
    """識別子または filter 指定で選んだ複数 run の比較結果を表示する。"""
    state = _state_from_ctx(ctx)
    manifests = _load_result_manifests(state.result_root)
    selected_manifests = _select_result_manifests(
        manifests,
        result_root=state.result_root,
        identifiers=identifiers or [],
        query=query,
        status=status,
        benchmark_name=benchmark_name,
        executor_name=executor_name,
        target=target,
    )
    payload = _build_result_compare_payload(
        selected_manifests,
        result_root=state.result_root,
        pass_threshold=pass_threshold,
        k_values=k_values,
    )
    if output_format is OutputFormat.json:
        console.print_json(_json_dumps(payload))
        return
    _render_result_compare(payload)


@result_app.command("pass-at-k")
def result_pass_at_k(
    ctx: typer.Context,
    identifiers: Annotated[
        list[str] | None,
        typer.Argument(
            help="集計対象の run_id, その prefix, または結果ディレクトリ/manifest パス。省略時は filter option で選ぶ。"
        ),
    ] = None,
    query: Annotated[str | None, typer.Option("--query", help="任意の文字列で絞り込む。")] = None,
    status: Annotated[
        ResultStatusFilter, typer.Option("--status", help="status で絞り込む。")
    ] = ResultStatusFilter.all,
    benchmark_name: Annotated[
        str | None, typer.Option("--benchmark", help="ベンチマーク名で絞り込む。")
    ] = None,
    executor_name: Annotated[
        str | None, typer.Option("--executor", help="executor 名で絞り込む。")
    ] = None,
    target: Annotated[str | None, typer.Option("--target", help="target で絞り込む。")] = None,
    pass_threshold: Annotated[
        float,
        typer.Option("--pass-threshold", help="PASS とみなす score の閾値。"),
    ] = 1.0,
    k_values: Annotated[
        list[int] | None,
        typer.Option("--k", min=1, help="計算する PASS@k の k。複数指定可。"),
    ] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="出力形式。")
    ] = OutputFormat.table,
) -> None:
    """識別子または filter 指定で選んだ run 群から PASS@k を直接計算する。"""
    state = _state_from_ctx(ctx)
    manifests = _load_result_manifests(state.result_root)
    selected_manifests = _select_result_manifests(
        manifests,
        result_root=state.result_root,
        identifiers=identifiers or [],
        query=query,
        status=status,
        benchmark_name=benchmark_name,
        executor_name=executor_name,
        target=target,
    )
    compare_payload = _build_result_compare_payload(
        selected_manifests,
        result_root=state.result_root,
        pass_threshold=pass_threshold,
        k_values=k_values,
        min_run_count=1,
    )
    payload = _build_result_pass_at_k_payload(compare_payload)
    if output_format is OutputFormat.json:
        console.print_json(_json_dumps(payload))
        return
    _render_result_pass_at_k(payload)


@result_app.command("show")
def result_show(
    ctx: typer.Context,
    identifier: Annotated[
        str,
        typer.Argument(help="run_id, その prefix, または結果ディレクトリ/manifest パス。"),
    ],
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="出力形式。")
    ] = OutputFormat.table,
) -> None:
    """結果 1 件の詳細を表示する。"""
    state = _state_from_ctx(ctx)
    manifests = _load_result_manifests(state.result_root)
    manifest = _resolve_result_identifier(identifier, state.result_root, manifests)
    if output_format is OutputFormat.json:
        detail_payload: dict[str, Any] | None = None
        detail_path = Path(manifest.detail_file_path)
        if detail_path.exists():
            detail_payload = json.loads(detail_path.read_text(encoding="utf-8"))
        executor_runtime = resolve_executor_runtime_payload(
            benchmark_name=manifest.benchmark_name,
            executor_name=manifest.executor_name,
            request_config=getattr(manifest, "request_config", {}),
            result_dir=Path(manifest.result_dir),
            recorded_payload=detail_payload.get("executor_runtime")
            if isinstance(detail_payload, dict)
            else None,
        )
        console.print_json(
            _json_dumps(
                {
                    "manifest": manifest.model_dump(mode="json"),
                    "detail": detail_payload,
                    "executor_runtime": executor_runtime,
                }
            )
        )
        return
    _render_result_detail(manifest, state.result_root)


def main(argv: list[str] | None = None) -> None:
    """CLI エントリポイント。"""
    effective_argv = sys.argv[1:] if argv is None else argv
    with _translate_termination_signals():
        app(args=_translate_legacy_argv(effective_argv))


if __name__ == "__main__":
    main()
