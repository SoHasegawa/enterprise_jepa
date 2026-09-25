"""Lightweight registry over TOUCAN's `mcp_servers/*_labeled.json` metadata.

The TOUCAN multi-turn trajectories carry server *names* (e.g. ``"Exa Search"``)
in `requested_mcp_servers`/`matched_mcp_servers` and target tools as
``"<server>::<tool>"`` strings, but the actual Smithery URL templates and tool
schemas live alongside in ``Toucan/mcp_servers/<rank>.<qualified>_labeled.json``.

This module loads that directory once, indexes by server name, and provides
helpers that render a Smithery-ready URL by substituting an API key, profile,
and (base64-encoded) per-server config into the template.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REGISTRY_DIR = Path.home() / "program" / "tools" / "Toucan" / "mcp_servers"


@dataclass(frozen=True)
class ToucanMcpServer:
    server_name: str
    qualified_name: str
    url_template: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    source_path: Path | None = None


def _normalize(name: str) -> str:
    return (name or "").strip().lower()


class ToucanMcpRegistry:
    """In-memory index of TOUCAN MCP server metadata, keyed by server name."""

    def __init__(self, servers: list[ToucanMcpServer]) -> None:
        self._by_name: dict[str, ToucanMcpServer] = {}
        self._by_qualified: dict[str, ToucanMcpServer] = {}
        for server in servers:
            self._by_name.setdefault(_normalize(server.server_name), server)
            if server.qualified_name:
                self._by_qualified.setdefault(_normalize(server.qualified_name), server)

    @classmethod
    def load(cls, registry_dir: Path = DEFAULT_REGISTRY_DIR) -> "ToucanMcpRegistry":
        if not registry_dir.exists():
            raise FileNotFoundError(f"TOUCAN MCP registry directory not found: {registry_dir}")
        servers: list[ToucanMcpServer] = []
        for path in sorted(registry_dir.glob("*_labeled.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            metadata = payload.get("metadata") or {}
            server_name = metadata.get("server_name") or ""
            remote = metadata.get("remote_server_response") or {}
            url_template = remote.get("url") or ""
            qualified = path.stem
            if "." in qualified:
                qualified = qualified.split(".", 1)[1]
            if qualified.endswith("_labeled"):
                qualified = qualified[: -len("_labeled")]
            if not server_name or not url_template:
                continue
            tools = remote.get("tools") or []
            if not isinstance(tools, list):
                tools = []
            servers.append(
                ToucanMcpServer(
                    server_name=server_name,
                    qualified_name=qualified,
                    url_template=url_template,
                    tools=tools,
                    source_path=path,
                )
            )
        return cls(servers)

    def lookup(self, server_name: str) -> ToucanMcpServer | None:
        norm = _normalize(server_name)
        if norm in self._by_name:
            return self._by_name[norm]
        return self._by_qualified.get(norm)

    def lookup_many(self, names: Iterable[str]) -> tuple[list[ToucanMcpServer], list[str]]:
        resolved: list[ToucanMcpServer] = []
        missing: list[str] = []
        seen: set[str] = set()
        for name in names:
            entry = self.lookup(name)
            if entry is None:
                missing.append(name)
                continue
            key = _normalize(entry.server_name)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(entry)
        return resolved, missing


def render_smithery_url(
    url_template: str,
    *,
    api_key: str,
    profile: str,
    server_config: dict[str, Any] | None = None,
) -> str:
    """Substitute Smithery placeholders in a TOUCAN MCP URL template.

    Mirrors `construct_mcp_server_url` in TOUCAN's `completion_qwen_agent.py`:
    expands `{config_b64}`, `{smithery_api_key}`, `{smithery_profile}`, and
    appends a `profile=` query parameter when neither placeholder is present
    but the URL is missing the parameter entirely.
    """
    if server_config is None or not server_config:
        # Match TOUCAN's default — keep `debug: false` so the placeholder still
        # renders to a valid base64 blob even when no per-server config is set.
        server_config = {"debug": False}
    config_b64 = base64.b64encode(json.dumps(server_config).encode()).decode()
    rendered = url_template
    if "{config_b64}" in rendered:
        rendered = rendered.replace("{config_b64}", config_b64)
    if "{smithery_api_key}" in rendered:
        rendered = rendered.replace("{smithery_api_key}", api_key or "")
    if "{smithery_profile}" in rendered:
        rendered = rendered.replace("{smithery_profile}", profile or "")
    elif "&profile=" not in rendered and "?profile=" not in rendered:
        separator = "&" if "?" in rendered else "?"
        rendered = f"{rendered}{separator}profile={profile or ''}"
    return rendered


def safe_server_slug(server_name: str) -> str:
    """Slugify a TOUCAN server name for use as a per-task MCP key.

    Matches the convention used in TOUCAN's `create_agent_for_item` —
    lowercase, spaces → dashes — so server identifiers stay stable when we
    later transform tool calls back into TOUCAN-style `function_call` records.
    """
    return (server_name or "unknown-server").strip().replace(" ", "-").lower()


def load_smithery_api_pool(pool_path: Path) -> list[dict[str, str]]:
    """Read TOUCAN's `smithery_api_pool.json` into a list of `{api_key, profile}` entries."""
    with pool_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "api_pool" in payload:
        entries = payload["api_pool"]
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError(
            f"Unexpected Smithery API pool shape in {pool_path}: "
            "expected a list or {'api_pool': [...]}."
        )
    cleaned: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        api_key = (entry.get("api_key") or "").strip()
        profile = (entry.get("profile") or "").strip()
        if not api_key or not profile:
            continue
        cleaned.append({"api_key": api_key, "profile": profile})
    return cleaned
