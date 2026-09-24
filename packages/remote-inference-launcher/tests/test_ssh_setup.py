from __future__ import annotations

import io

import pytest

from remote_inference_launcher.cli import main
from remote_inference_launcher.ssh_setup import (
    HostSshSettings,
    RemoteHost,
    render_setup,
    settings_from_preset,
    settings_from_values,
)


def test_generic_ssh_setup_rendering() -> None:
    output = render_setup(
        [
            settings_from_values(
                alias="custom",
                hostname="example.internal",
                port=10022,
                user="alice",
                identity_file="~/.ssh/custom-key",
            )
        ]
    )
    assert "Host custom" in output
    assert "  HostName example.internal" in output
    assert "  Port 10022" in output
    assert "  User alice" in output
    assert "  IdentityFile ~/.ssh/custom-key" in output
    assert 'eval "$(ssh-agent -s)"' in output
    assert "ssh-add ~/.ssh/custom-key" in output
    assert "ssh custom 'hostname && squeue --version'" in output


def test_presets_render_configured_hosts() -> None:
    output = render_setup(
        [
            settings_from_preset(
                "slurm-login",
                user="alice",
                identity_file="~/.ssh/team-key",
            ),
            settings_from_preset(
                "gpu-cluster",
                user="alice",
                identity_file="~/.ssh/team-key",
            ),
        ]
    )
    assert "Host slurm-login" in output
    assert "  HostName slurm-login.example.invalid" in output
    assert "  Port 22" in output
    assert "Host gpu-cluster" in output
    assert "  HostName gpu-cluster.example.invalid" in output
    assert output.count("ssh-add ~/.ssh/team-key") == 1


def test_duplicate_aliases_fail_loudly() -> None:
    settings = [
        HostSshSettings(
            host=RemoteHost(alias="dup", hostname="one.example", port=22),
            user="alice",
            identity_file="~/.ssh/key1",
        ),
        HostSshSettings(
            host=RemoteHost(alias="dup", hostname="two.example", port=22),
            user="alice",
            identity_file="~/.ssh/key2",
        ),
    ]
    with pytest.raises(ValueError, match="duplicate SSH host alias"):
        render_setup(settings)


def test_unsafe_tokens_fail_loudly() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        settings_from_values(
            alias="bad;alias",
            hostname="example.internal",
            port=22,
            user="alice",
            identity_file="~/.ssh/key",
        )


def test_non_interactive_cli_requires_missing_user_values(capsys) -> None:
    status = main(["ssh-setup", "--preset", "slurm-login"])
    captured = capsys.readouterr()
    assert status == 2
    assert "--user is required" in captured.err


def test_cli_accepts_explicit_non_interactive_host_values(capsys) -> None:
    status = main(
        [
            "ssh-setup",
            "--alias",
            "custom",
            "--hostname",
            "example.internal",
            "--port",
            "10022",
            "--user",
            "alice",
            "--identity-file",
            "~/.ssh/custom-key",
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "Host custom" in captured.out
    assert "  HostName example.internal" in captured.out
    assert "  Port 10022" in captured.out


def test_cli_interactive_mode_prompts_for_preset_user_values(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("alice\n~/.ssh/team-key\n"))
    status = main(["ssh-setup", "--preset", "slurm-login", "--interactive"])
    captured = capsys.readouterr()
    assert status == 0
    assert "SSH username for slurm-login:" in captured.out
    assert "Private key path for slurm-login:" in captured.out
    assert "Host slurm-login" in captured.out
