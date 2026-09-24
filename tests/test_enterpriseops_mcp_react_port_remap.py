"""Tests for the EnterpriseOps-Gym mcp_react executor's MCP-server port remap.

``ENTERPRISEOPS_MCP_PORT_REMAP`` rewrites fixed ``mcp_server_url`` host ports in the
task's ``gym_servers_config`` (dataset-sourced, often non-local), for when a required
host port is taken by an unrelated service and the domain server is reachable
elsewhere. The executor imports ``benchmark.*`` lazily, so it loads without the gym repo.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTOR_PATH = (
    REPO_ROOT / "assets" / "EnterpriseOps-Gym" / "purple-executors" / "mcp_react" / "executor.py"
)


def _load_executor():
    spec = importlib.util.spec_from_file_location("eops_mcp_react_port_remap_uut", EXECUTOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_port_remap(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_executor()
    monkeypatch.delenv("ENTERPRISEOPS_MCP_PORT_REMAP", raising=False)
    assert module._parse_mcp_port_remap() == {}
    monkeypatch.setenv("ENTERPRISEOPS_MCP_PORT_REMAP", "8008=8010, 8001=8011 , bad, x=y")
    assert module._parse_mcp_port_remap() == {"8008": "8010", "8001": "8011"}


def test_remap_mcp_server_url() -> None:
    module = _load_executor()
    remap = {"8008": "8010"}
    assert module._remap_mcp_server_url("http://localhost:8008", remap) == "http://localhost:8010"
    assert module._remap_mcp_server_url("http://localhost:8008/mcp", remap) == "http://localhost:8010/mcp"
    assert module._remap_mcp_server_url("http://127.0.0.1:8008/", remap) == "http://127.0.0.1:8010/"
    # Ports not in the map, and non-strings, are left untouched.
    assert module._remap_mcp_server_url("http://localhost:8001", remap) == "http://localhost:8001"
    assert module._remap_mcp_server_url(None, remap) is None
    assert module._remap_mcp_server_url("http://localhost:8008", {}) == "http://localhost:8008"


def test_resolve_seed_database_paths_applies_remap(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_executor()
    # Entries without seed_database_file skip DB-backed token normalization.
    cfg = [
        {"mcp_server_url": "http://localhost:8008"},  # hr
        {"mcp_server_url": "http://localhost:8001"},  # csm (unchanged)
    ]

    monkeypatch.setenv("ENTERPRISEOPS_MCP_PORT_REMAP", "8008=8010")
    out = module._resolve_seed_database_paths(cfg, repo_path=Path("/gym"), verifiers=[])
    assert out[0]["mcp_server_url"] == "http://localhost:8010"
    assert out[1]["mcp_server_url"] == "http://localhost:8001"

    # No env -> no rewrite (default behaviour unchanged).
    monkeypatch.delenv("ENTERPRISEOPS_MCP_PORT_REMAP", raising=False)
    out = module._resolve_seed_database_paths(cfg, repo_path=Path("/gym"), verifiers=[])
    assert out[0]["mcp_server_url"] == "http://localhost:8008"
