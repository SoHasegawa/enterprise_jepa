from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from urllib.parse import urlparse

DEFAULT_NO_PROXY_ENTRIES = (
    "127.0.0.1",
    "localhost",
    "::1",
    "0.0.0.0",
)


def _split_no_proxy(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _normalize_host(candidate: str | None) -> str | None:
    if candidate is None:
        return None

    raw = candidate.strip()
    if not raw:
        return None

    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    hostname = parsed.hostname
    if hostname:
        return hostname

    if "/" in raw:
        return None

    host = raw
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if ":" in host and host.count(":") == 1:
        maybe_host, maybe_port = host.rsplit(":", 1)
        if maybe_port.isdigit():
            return maybe_host or None
    return host or None


def _is_proxy_bypass_host(host: str) -> bool:
    normalized = host.strip().strip("[]").lower()
    if not normalized:
        return False
    if normalized in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        return True

    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return normalized.endswith((".localhost", ".local"))

    return (
        address.is_loopback or address.is_private or address.is_link_local or address.is_unspecified
    )


def ensure_no_proxy(
    env: dict[str, str],
    extra_hosts_or_urls: Iterable[str | None] = (),
) -> dict[str, str]:
    """ローカル通信が proxy を経由しないよう `NO_PROXY` / `no_proxy` を正規化する。"""
    entries: list[str] = []
    seen: set[str] = set()

    def add(entry: str | None) -> None:
        if entry is None:
            return
        normalized = entry.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        entries.append(normalized)

    for current in (
        *_split_no_proxy(env.get("NO_PROXY")),
        *_split_no_proxy(env.get("no_proxy")),
        *DEFAULT_NO_PROXY_ENTRIES,
    ):
        add(current)

    for candidate in extra_hosts_or_urls:
        host = _normalize_host(candidate)
        if host and _is_proxy_bypass_host(host):
            add(host)

    value = ",".join(entries)
    env["NO_PROXY"] = value
    env["no_proxy"] = value
    return env
