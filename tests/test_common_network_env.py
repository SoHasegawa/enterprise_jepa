from __future__ import annotations

from common.network_env import ensure_no_proxy


def test_ensure_no_proxy_merges_existing_entries_and_defaults() -> None:
    env = {
        "NO_PROXY": "api.internal, localhost,",
        "no_proxy": "service.local,api.internal",
    }

    result = ensure_no_proxy(env)

    entries = result["NO_PROXY"].split(",")
    assert entries == [
        "api.internal",
        "localhost",
        "service.local",
        "127.0.0.1",
        "::1",
        "0.0.0.0",
    ]
    assert result["no_proxy"] == result["NO_PROXY"]


def test_ensure_no_proxy_adds_only_proxy_bypass_hosts_from_extra_urls() -> None:
    env: dict[str, str] = {}

    result = ensure_no_proxy(
        env,
        [
            "http://127.0.0.1:8080/path",
            "https://10.10.0.5/v1",
            "[::1]:9000",
            "worker.local",
            "https://public.example.com",
            "not/a/host",
            "",
            None,
        ],
    )

    entries = result["NO_PROXY"].split(",")
    assert "127.0.0.1" in entries
    assert "10.10.0.5" in entries
    assert "::1" in entries
    assert "worker.local" in entries
    assert "public.example.com" not in entries
    assert "not/a/host" not in entries
    assert result["no_proxy"] == result["NO_PROXY"]


def test_ensure_no_proxy_handles_ipv6_and_private_network_literals() -> None:
    env = {"NO_PROXY": "existing"}

    result = ensure_no_proxy(
        env,
        [
            "http://[fd00::1]:7000",
            "172.16.200.122:9000",
            "192.168.1.10",
            "8.8.8.8",
        ],
    )

    entries = result["NO_PROXY"].split(",")
    assert "fd00::1" in entries
    assert "172.16.200.122" in entries
    assert "192.168.1.10" in entries
    assert "8.8.8.8" not in entries


def test_ensure_no_proxy_ignores_empty_and_public_candidates() -> None:
    env: dict[str, str] = {}

    result = ensure_no_proxy(
        env,
        [
            "   ",
            "http://public.example.com/path",
            "public.example.com:443",
            "path/without/scheme",
            "http://",
        ],
    )

    assert result["NO_PROXY"].split(",") == ["127.0.0.1", "localhost", "::1", "0.0.0.0"]
    assert result["no_proxy"] == result["NO_PROXY"]
