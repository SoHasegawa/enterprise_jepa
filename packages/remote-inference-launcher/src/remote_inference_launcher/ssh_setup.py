"""Render SSH setup snippets for remote launcher hosts."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO

SHELL_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_@%+=:,./~-]+$")


@dataclass(frozen=True)
class RemoteHost:
    """SSH-reachable host details."""

    alias: str
    hostname: str
    port: int


@dataclass(frozen=True)
class HostSshSettings:
    """User-specific SSH settings for one remote host."""

    host: RemoteHost
    user: str
    identity_file: str


# Presets for the clusters this launcher is pointed at. Replace the placeholder
# hostnames/ports with your own, or pass --hostname/--port explicitly.
REMOTE_HOST_PRESETS = {
    "gpu-cluster": RemoteHost(
        alias="gpu-cluster",
        hostname=os.getenv("RIL_GPU_CLUSTER_HOSTNAME", "gpu-cluster.example.invalid"),
        port=int(os.getenv("RIL_GPU_CLUSTER_PORT", "22")),
    ),
    "slurm-login": RemoteHost(
        alias="slurm-login",
        hostname=os.getenv("RIL_SLURM_LOGIN_HOSTNAME", "slurm-login.example.invalid"),
        port=int(os.getenv("RIL_SLURM_LOGIN_PORT", "22")),
    ),
}


def render_setup(settings: Sequence[HostSshSettings]) -> str:
    """Render SSH config, ssh-agent commands, and verification commands."""

    _validate_unique_aliases(setting.host.alias for setting in settings)
    ssh_config_lines = ["Paste this into your SSH config (~/.ssh/config):", ""]
    for setting in settings:
        validate_host_settings(setting)
        ssh_config_lines.extend(_host_config_lines(setting))
        ssh_config_lines.append("")

    agent_lines = ["Run this once per login session:", "", 'eval "$(ssh-agent -s)"']
    agent_lines.extend(
        f"ssh-add {identity_file}" for identity_file in _unique_identity_files(settings)
    )
    verify_lines = ["", "Verify each host:", ""]
    verify_lines.extend(
        f"ssh {setting.host.alias} 'hostname && squeue --version'" for setting in settings
    )
    return "\n".join([*ssh_config_lines, *agent_lines, *verify_lines])


def settings_from_values(
    *,
    alias: str,
    hostname: str,
    port: int,
    user: str,
    identity_file: str,
) -> HostSshSettings:
    """Build validated SSH settings from explicit values."""

    setting = HostSshSettings(
        host=RemoteHost(alias=alias, hostname=hostname, port=port),
        user=user,
        identity_file=identity_file,
    )
    validate_host_settings(setting)
    return setting


def settings_from_preset(
    preset: str,
    *,
    user: str,
    identity_file: str,
) -> HostSshSettings:
    """Build settings for a known internal team host preset."""

    if preset not in REMOTE_HOST_PRESETS:
        raise ValueError(f"Unknown SSH setup preset: {preset}.")
    setting = HostSshSettings(
        host=REMOTE_HOST_PRESETS[preset],
        user=user,
        identity_file=identity_file,
    )
    validate_host_settings(setting)
    return setting


def prompt_missing_settings(
    host: RemoteHost,
    *,
    user: str = "",
    identity_file: str = "",
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> HostSshSettings:
    """Prompt interactively for missing user-specific settings."""

    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    resolved_user = user or _prompt_value(
        f"SSH username for {host.alias}: ",
        stdin=input_stream,
        stdout=output_stream,
    )
    resolved_identity_file = identity_file or _prompt_value(
        f"Private key path for {host.alias}: ",
        stdin=input_stream,
        stdout=output_stream,
    )
    return HostSshSettings(
        host=host,
        user=resolved_user,
        identity_file=resolved_identity_file,
    )


def validate_host_settings(setting: HostSshSettings) -> None:
    """Reject SSH setup values that cannot produce a safe shell/config snippet."""

    for value, label in (
        (setting.host.alias, "SSH host alias"),
        (setting.host.hostname, f"hostname for {setting.host.alias}"),
        (setting.user, f"SSH username for {setting.host.alias}"),
        (setting.identity_file, f"identity file for {setting.host.alias}"),
    ):
        _validate_config_token(value, label=label)
    if isinstance(setting.host.port, bool) or not isinstance(setting.host.port, int):
        raise ValueError(f"port for {setting.host.alias} must be an integer.")
    if not 1 <= setting.host.port <= 65535:
        raise ValueError(f"port for {setting.host.alias} must be between 1 and 65535.")


def _prompt_value(prompt: str, *, stdin: TextIO, stdout: TextIO) -> str:
    stdout.write(prompt)
    stdout.flush()
    raw_value = stdin.readline()
    if raw_value == "":
        raise ValueError(f"missing input for {prompt.rstrip(': ')}.")
    value = raw_value.strip()
    if not value:
        raise ValueError(f"{prompt.rstrip(': ')} must not be empty.")
    return value


def _validate_config_token(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must not be empty.")
    if any(character.isspace() for character in value):
        raise ValueError(f"{label} must not contain whitespace.")
    if not SHELL_TOKEN_PATTERN.fullmatch(value):
        raise ValueError(f"{label} contains characters that are unsafe for this shell snippet.")


def _validate_unique_aliases(aliases: Sequence[str]) -> None:
    aliases = tuple(aliases)
    duplicate_aliases = sorted({alias for alias in aliases if aliases.count(alias) > 1})
    if duplicate_aliases:
        raise ValueError(f"duplicate SSH host alias: {', '.join(duplicate_aliases)}.")


def _host_config_lines(setting: HostSshSettings) -> list[str]:
    return [
        f"Host {setting.host.alias}",
        f"  HostName {setting.host.hostname}",
        f"  Port {setting.host.port}",
        f"  User {setting.user}",
        f"  IdentityFile {setting.identity_file}",
        "  AddKeysToAgent yes",
    ]


def _unique_identity_files(settings: Sequence[HostSshSettings]) -> list[str]:
    identity_files: list[str] = []
    for setting in settings:
        if setting.identity_file not in identity_files:
            identity_files.append(setting.identity_file)
    return identity_files
