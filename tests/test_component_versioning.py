from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from common.versioning import load_component_version, resolve_benchmark_component_versions
from ejepa_cli import cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def plain_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rich の ANSI 制御シーケンスを抑止する。"""
    monkeypatch.setattr(
        cli,
        "console",
        Console(force_terminal=False, color_system=None, width=140),
    )


def _write_benchmark_fixture(
    assets_root: Path,
    *,
    benchmark_version: str = "1.2.3",
    green_version: str = "0.2.0",
    purple_version: str = "0.3.0",
    executor_version: str = "0.4.0",
) -> Path:
    benchmark_dir = assets_root / "ExampleBench"
    (benchmark_dir / "green").mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "purple").mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "purple-executors" / "local").mkdir(parents=True, exist_ok=True)

    (benchmark_dir / "benchmark.toml").write_text(
        "\n".join(
            [
                'name = "ExampleBench"',
                f'version = "{benchmark_version}"',
                'default_executor = "local"',
                "",
                "[config]",
                'target = "sample"',
                "",
                "[green_agent]",
                'entrypoint = "green/example_green_agent.py"',
                'host = "127.0.0.1"',
                "port = 0",
                "",
                "[[participants]]",
                'role = "agent"',
                'entrypoint = "purple/example_purple_agent.py"',
                'host = "127.0.0.1"',
                "port = 0",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (benchmark_dir / "green" / "pyproject.toml").write_text(
        f'[project]\nname = "example-green"\nversion = "{green_version}"\n',
        encoding="utf-8",
    )
    (benchmark_dir / "purple" / "pyproject.toml").write_text(
        f'[project]\nname = "example-purple"\nversion = "{purple_version}"\n',
        encoding="utf-8",
    )
    (benchmark_dir / "purple-executors" / "local" / "version.toml").write_text(
        f'version = "{executor_version}"\n',
        encoding="utf-8",
    )
    (benchmark_dir / "purple-executors" / "local" / "executor.py").write_text(
        "def build_executor():\n    return None\n",
        encoding="utf-8",
    )
    return benchmark_dir


def test_resolve_benchmark_component_versions_reads_component_semvers(tmp_path: Path) -> None:
    """benchmark / green / purple / executor の semver を個別に解決できる。"""
    benchmark_dir = _write_benchmark_fixture(tmp_path)

    versions = resolve_benchmark_component_versions(
        benchmark_dir,
        executor_names=["local"],
    )

    assert versions.benchmark_version == "1.2.3"
    assert versions.green_agent_version == "0.2.0"
    assert versions.purple_agent_version == "0.3.0"
    assert versions.executor_versions == {"local": "0.4.0"}


def test_load_component_version_rejects_invalid_semver(tmp_path: Path) -> None:
    """component version は semver 以外を受け付けない。"""
    component_dir = tmp_path / "component"
    component_dir.mkdir(parents=True)
    (component_dir / "version.toml").write_text('version = "latest"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid semantic version"):
        load_component_version(component_dir)


def test_benchmark_show_json_includes_component_versions(tmp_path: Path) -> None:
    """`ejepa bench show --format json` に component version を含める。"""
    assets_root = tmp_path / "assets"
    _write_benchmark_fixture(assets_root)

    result = runner.invoke(
        cli.app,
        [
            "--assets-root",
            str(assets_root),
            "bench",
            "show",
            "ExampleBench",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["version"] == "1.2.3"
    assert payload["summary"]["green_agent_version"] == "0.2.0"
    assert payload["summary"]["purple_agent_version"] == "0.3.0"
    assert payload["summary"]["executor_versions"] == {"local": "0.4.0"}


def test_benchmark_list_table_shows_component_versions(tmp_path: Path) -> None:
    """`ejepa bench list` が green / purple / executor の版情報を表示する。"""
    assets_root = tmp_path / "assets"
    _write_benchmark_fixture(assets_root)

    result = runner.invoke(
        cli.app,
        [
            "--assets-root",
            str(assets_root),
            "bench",
            "list",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Green" in result.output
    assert "Purple" in result.output
    assert "0.2.0" in result.output
    assert "0.3.0" in result.output
    assert "local@0.4.0" in result.output
