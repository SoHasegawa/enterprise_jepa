from __future__ import annotations

import json
import os
import signal
from types import SimpleNamespace

import pytest

from remote_inference_launcher.cli import (
    build_parser,
    local_vllm_bootstrap_config_from_args,
    main,
    slurm_vllm_bootstrap_config_from_args,
    slurm_vllm_config_from_args,
    ssh_vllm_bootstrap_config_from_args,
)
from remote_inference_launcher.slurm_vllm import SlurmVllmConfig


def test_cli_flags_override_yaml_values(tmp_path) -> None:
    config_path = tmp_path / "slurm-vllm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "kind: slurm_vllm",
                "ssh_target: cluster",
                "partition: batch",
                "walltime: '6:00:00'",
                "num_gpus: 2",
                "memory: 384GB",
                "cpus_per_task: 20",
                "local_port: 8123",
                "keep_remote_job: false",
            ]
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "slurm-vllm",
            "--config",
            str(config_path),
            "--local-port",
            "8124",
            "--keep-remote-job",
            "--extra-arg=--disable-log-requests",
        ]
    )
    config = slurm_vllm_config_from_args(args)
    assert config.local_port == 8124
    assert config.keep_remote_job
    assert config.extra_args == ("--disable-log-requests",)


def test_boolean_cli_flags_can_force_false_from_yaml(tmp_path) -> None:
    config_path = tmp_path / "slurm-vllm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "kind: slurm_vllm",
                "ssh_target: cluster",
                "partition: batch",
                "walltime: '6:00:00'",
                "num_gpus: 2",
                "memory: 384GB",
                "cpus_per_task: 20",
                "keep_remote_job: true",
            ]
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["slurm-vllm", "--config", str(config_path), "--no-keep-remote-job"]
    )
    config = slurm_vllm_config_from_args(args)
    assert not config.keep_remote_job


def test_slurm_vllm_cli_loads_generic_structured_config(tmp_path) -> None:
    config_path = tmp_path / "slurm-vllm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "kind: slurm_vllm",
                "ssh_target: cluster",
                "model: vendor/model",
                "partition: batch",
                "walltime: '6:00:00'",
                "num_gpus: 2",
                "memory: 384GB",
                "cpus_per_task: 20",
                "readiness:",
                "  smoke_test: disabled",
                "queue_policy:",
                "  max_pending_seconds: 300",
                "  poll_interval_seconds: 15",
            ]
        ),
        encoding="utf-8",
    )

    args = build_parser().parse_args(["slurm-vllm", "--config", str(config_path)])
    config = slurm_vllm_config_from_args(args)

    assert config.readiness.smoke_test == "disabled"
    assert config.queue_policy.max_pending_seconds == 300
    assert config.queue_policy.poll_interval_seconds == 15


def test_slurm_vllm_cli_rejects_non_slurm_inference_config(tmp_path) -> None:
    config_path = tmp_path / "local.yaml"
    config_path.write_text("kind: local_vllm\nmodel: vendor/model\n", encoding="utf-8")
    args = build_parser().parse_args(["slurm-vllm", "--config", str(config_path)])

    with pytest.raises(ValueError, match="kind: slurm_vllm"):
        slurm_vllm_config_from_args(args)


def test_repeated_extra_args_preserve_order() -> None:
    args = build_parser().parse_args(
        [
            "slurm-vllm",
            "--extra-arg=--reasoning-parser",
            "--extra-arg=qwen3",
            "--extra-arg=--disable-log-requests",
        ]
    )
    config = slurm_vllm_config_from_args(args)
    assert config.extra_args == ("--reasoning-parser", "qwen3", "--disable-log-requests")


def test_slurm_vllm_cli_rejects_unused_env_name_flag() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["slurm-vllm", "--env-name", "gpu-cluster"])


def test_slurm_vllm_cli_verbosity_aliases_set_config() -> None:
    quiet_args = build_parser().parse_args(["slurm-vllm", "--quiet"])
    verbose_args = build_parser().parse_args(["slurm-vllm", "--verbose"])

    assert slurm_vllm_config_from_args(quiet_args).verbosity == "quiet"
    assert slurm_vllm_config_from_args(verbose_args).verbosity == "verbose"


def test_bootstrap_cli_verbosity_aliases_set_config() -> None:
    quiet_args = build_parser().parse_args(["slurm-vllm-bootstrap", "--quiet"])
    verbose_args = build_parser().parse_args(["slurm-vllm-bootstrap", "--verbose"])
    local_args = build_parser().parse_args(["local-vllm-bootstrap", "--quiet"])
    ssh_args = build_parser().parse_args(["ssh-vllm-bootstrap", "--verbose"])

    assert slurm_vllm_bootstrap_config_from_args(quiet_args).verbosity == "quiet"
    assert slurm_vllm_bootstrap_config_from_args(verbose_args).verbosity == "verbose"
    assert local_vllm_bootstrap_config_from_args(local_args).verbosity == "quiet"
    assert ssh_vllm_bootstrap_config_from_args(ssh_args).verbosity == "verbose"


def test_cli_rejects_verbosity_and_alias_together() -> None:
    args = build_parser().parse_args(["slurm-vllm", "--verbosity", "quiet", "--verbose"])

    with pytest.raises(ValueError, match="either --verbosity or --quiet/--verbose"):
        slurm_vllm_config_from_args(args)


def test_validate_cli_strict_preflight_transient_returns_inconclusive(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    from remote_inference_launcher.preflight import PreflightCheck

    config_path = tmp_path / "endpoint.yaml"
    config_path.write_text(
        "kind: existing_endpoint\n"
        "api_base: http://127.0.0.1:1/v1\n"
        "readiness: {smoke_test: disabled}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "remote_inference_launcher.validation.run_preflight_checks_for_configs",
        lambda config_sources: (
            PreflightCheck(
                source=next(iter(config_sources))[0],
                endpoint_name="default",
                name="ssh-reachable",
                outcome="transient_failure",
                code="ssh_connection_reset",
                attempts=3,
                duration_seconds=0.03,
                detail="Connection reset during SSH transport.",
                command_summary="ssh <ssh-target>",
            ),
        ),
    )

    status = main(["validate", str(config_path), "--strict-preflight"])

    captured = capsys.readouterr()
    assert status == 2
    assert "Validation result: INCONCLUSIVE" in captured.out


def test_validate_cli_runtime_preflight_flag(monkeypatch, tmp_path, capsys) -> None:
    from remote_inference_launcher.preflight import PreflightCheck

    calls: list[str] = []
    config_path = tmp_path / "endpoint.yaml"
    config_path.write_text(
        "kind: existing_endpoint\n"
        "api_base: http://127.0.0.1:1/v1\n"
        "readiness: {smoke_test: disabled}\n",
        encoding="utf-8",
    )

    def runtime_preflight(_config, *, source, effective_plan):
        del effective_plan
        calls.append(source)
        return (
            PreflightCheck(
                source=source,
                endpoint_name="default",
                name="runtime-vllm-import",
                outcome="durable_failure",
                code="runtime_vllm_import_failed",
                attempts=1,
                duration_seconds=0.01,
                detail="No module named vllm",
                layer="compute-runtime",
            ),
        )

    monkeypatch.setattr(
        "remote_inference_launcher.validation.run_runtime_preflight_checks",
        runtime_preflight,
    )

    status = main(["validate", str(config_path), "--runtime-preflight"])

    captured = capsys.readouterr()
    assert status == 0
    assert calls == [str(config_path)]
    assert "compute-runtime:" in captured.out
    assert "runtime-vllm-import" in captured.out


def test_slurm_vllm_cli_reports_cleanup_failure(monkeypatch, capsys) -> None:
    monkeypatch.setattr("remote_inference_launcher.cli.SlurmVllmLauncher", _StopFailureLauncher)

    status = main(["slurm-vllm"])

    captured = capsys.readouterr()
    assert status == 2
    assert "OPENAI_BASE_URL=http://127.0.0.1:8123/v1" in captured.out
    assert "INFERENCE_LAUNCH_SUMMARY_PATH=/tmp/slurm-summary.json" in captured.out
    assert "failed to stop Slurm vLLM launcher: cleanup failed" in captured.err


def test_generic_start_cli_writes_env_file_after_readiness(monkeypatch, tmp_path, capsys) -> None:
    import remote_inference_launcher.inference_config as inference_config

    monkeypatch.chdir(tmp_path)
    launcher = _GenericLauncher()
    load_kwargs: dict[str, object] = {}

    def load_launcher(_path: object, **kwargs: object) -> object:
        load_kwargs.update(kwargs)
        return launcher

    monkeypatch.setattr(
        inference_config,
        "load_inference_launcher",
        load_launcher,
    )
    config_path = tmp_path / "inference.yaml"
    config_path.write_text("kind: local_vllm\nmodel: example\n", encoding="utf-8")
    env_file = tmp_path / "inference.env"

    status = main(
        [
            "start",
            "--config",
            str(config_path),
            "--env-file",
            str(env_file),
            "--format",
            "json",
            "--exit-after-ready",
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert launcher.stopped
    assert '"actor"' in captured.out
    assert '"OPENAI_BASE_URL": "http://127.0.0.1:8123/v1"' in captured.out
    assert '"summary_path": "/tmp/launch-summary.json"' in captured.out
    content = env_file.read_text(encoding="utf-8")
    assert "export INFERENCE_ACTOR_BASE_URL='http://127.0.0.1:8123/v1'" in content
    assert "export INFERENCE_ACTOR_LAUNCH_SUMMARY_PATH='/tmp/launch-summary.json'" in content
    assert "export INFERENCE_LAUNCH_SUMMARY_PATH='/tmp/launch-summary.json'" in content
    assert "export OPENAI_MODEL_NAME='actor-model'" in content
    assert "API_KEY" not in content
    registry_dirs = list((tmp_path / ".remote-inference-launcher" / "runs").iterdir())
    assert len(registry_dirs) == 1
    registry_dir = registry_dirs[0]
    assert load_kwargs["run_id"] == registry_dir.name
    endpoint_plan = load_kwargs["endpoint_plan"]
    assert endpoint_plan.summary_path.parent.name == "default"
    state = json.loads((registry_dir / "state.json").read_text(encoding="utf-8"))
    plan = json.loads((registry_dir / "plan.json").read_text(encoding="utf-8"))
    registry_env = (registry_dir / "env.sh").read_text(encoding="utf-8")
    assert plan["schema_version"] == "ril-plan/v1"
    assert state["schema_version"] == "ril-state/v1"
    assert state["lifecycle_state"] == "READY"
    assert state["endpoints"]["actor"]["api_base"] == "http://127.0.0.1:8123/v1"
    assert "export OPENAI_API_KEY" not in registry_env
    assert "OPENAI_BASE_URL='http://127.0.0.1:8123/v1'" in registry_env
    assert (registry_dir / "events.jsonl").read_text(encoding="utf-8")


def test_generic_start_detach_prints_registry_and_log_commands(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    import remote_inference_launcher.controller as controller
    import remote_inference_launcher.registry as registry

    monkeypatch.chdir(tmp_path)

    def fake_spawn_detached_controller(**kwargs: object) -> SimpleNamespace:
        registry_dir = kwargs["registry_dir"]
        log_path = registry_dir / "controller.log"
        log_path.write_text("controller started\n", encoding="utf-8")
        registry.update_controller_pid(registry_dir, os.getpid())
        registry.update_state(registry_dir, lifecycle_state="SUBMITTING")
        return SimpleNamespace(pid=os.getpid(), log_path=log_path)

    monkeypatch.setattr(controller, "spawn_detached_controller", fake_spawn_detached_controller)
    monkeypatch.setattr(
        controller,
        "wait_for_controller_start",
        lambda registry_dir: registry.read_json(registry_dir / "state.json"),
    )
    config_path = tmp_path / "inference.yaml"
    config_path.write_text("kind: local_vllm\nmodel: example\n", encoding="utf-8")

    status = main(["start", "--config", str(config_path), "--detach"])

    output = capsys.readouterr().out
    assert status == 0
    assert "run_id=" in output
    assert "registry=.remote-inference-launcher/runs/" in output
    assert "logs_command=remote-inference-launcher logs" in output


def test_generic_start_detach_wait_ready_prints_registry_env(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    import remote_inference_launcher.controller as controller
    import remote_inference_launcher.registry as registry

    monkeypatch.chdir(tmp_path)

    def fake_spawn_detached_controller(**kwargs: object) -> SimpleNamespace:
        registry_dir = kwargs["registry_dir"]
        log_path = registry_dir / "controller.log"
        log_path.write_text("ready\n", encoding="utf-8")
        registry.update_controller_pid(registry_dir, os.getpid())
        registry.record_ready_sessions(
            registry_dir,
            {
                "actor": SimpleNamespace(
                    name="actor",
                    api_base="http://127.0.0.1:8123/v1",
                    served_model_name="actor-model",
                    summary_path="/tmp/launch-summary.json",
                )
            },
        )
        return SimpleNamespace(pid=os.getpid(), log_path=log_path)

    monkeypatch.setattr(controller, "spawn_detached_controller", fake_spawn_detached_controller)
    monkeypatch.setattr(
        controller,
        "wait_for_controller_start",
        lambda registry_dir: registry.read_json(registry_dir / "state.json"),
    )
    monkeypatch.setattr(
        controller,
        "wait_for_ready_state",
        lambda registry_dir, **_kwargs: registry.read_json(registry_dir / "state.json"),
    )
    config_path = tmp_path / "inference.yaml"
    config_path.write_text("kind: local_vllm\nmodel: example\n", encoding="utf-8")
    env_file = tmp_path / "inference.env"

    status = main(
        [
            "start",
            "--config",
            str(config_path),
            "--detach",
            "--wait-ready",
            "--env-file",
            str(env_file),
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert "OPENAI_BASE_URL='http://127.0.0.1:8123/v1'" in output
    assert "export OPENAI_API_KEY" not in env_file.read_text(encoding="utf-8")


def test_generic_start_cli_keeps_owned_launcher_alive_by_default(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    import remote_inference_launcher.cli as launcher_cli
    import remote_inference_launcher.inference_config as inference_config

    monkeypatch.chdir(tmp_path)
    launcher = _GenericLauncher()
    held_launchers: list[object] = []

    def fake_hold(held_launcher: object) -> None:
        held_launchers.append(held_launcher)
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        inference_config,
        "load_inference_launcher",
        lambda _path, **_kwargs: launcher,
    )
    monkeypatch.setattr(launcher_cli, "_hold_launcher", fake_hold)
    config_path = tmp_path / "inference.yaml"
    config_path.write_text("kind: local_vllm\nmodel: example\n", encoding="utf-8")

    status = main(["start", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert status == 130
    assert launcher.stopped
    assert held_launchers == [launcher]
    assert "OPENAI_BASE_URL=http://127.0.0.1:8123/v1" in captured.out
    assert "INFERENCE_LAUNCH_SUMMARY_PATH=/tmp/launch-summary.json" in captured.out


def test_generic_start_cli_shields_cleanup_from_termination_signal(
    monkeypatch,
    tmp_path,
) -> None:
    import remote_inference_launcher.cli as launcher_cli
    import remote_inference_launcher.inference_config as inference_config

    monkeypatch.chdir(tmp_path)
    launcher = _SignalDuringStopLauncher()

    def fake_hold(_held_launcher: object) -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        inference_config,
        "load_inference_launcher",
        lambda _path, **_kwargs: launcher,
    )
    monkeypatch.setattr(launcher_cli, "_hold_launcher", fake_hold)
    config_path = tmp_path / "inference.yaml"
    config_path.write_text("kind: local_vllm\nmodel: example\n", encoding="utf-8")

    status = main(["start", "--config", str(config_path)])

    assert status == 130
    assert launcher.stopped


def test_generic_start_cli_does_not_hold_external_launcher_by_default(
    monkeypatch,
    tmp_path,
) -> None:
    import remote_inference_launcher.cli as launcher_cli
    import remote_inference_launcher.inference_config as inference_config

    monkeypatch.chdir(tmp_path)
    launcher = _ExternalLauncher()
    monkeypatch.setattr(
        inference_config,
        "load_inference_launcher",
        lambda _path, **_kwargs: launcher,
    )
    monkeypatch.setattr(
        launcher_cli,
        "_hold_launcher",
        lambda _launcher: pytest.fail("external launcher should not be held"),
    )
    config_path = tmp_path / "inference.yaml"
    config_path.write_text("kind: existing_endpoint\napi_base: http://127.0.0.1:8123/v1\n")

    status = main(["start", "--config", str(config_path)])

    assert status == 0
    assert launcher.stopped


def test_generic_check_cli_prints_ready_endpoint_env(monkeypatch, capsys) -> None:
    import remote_inference_launcher.existing_endpoint as existing_endpoint

    monkeypatch.setattr(existing_endpoint, "ExistingEndpointLauncher", _CheckLauncher)

    status = main(["check", "--base-url", "http://127.0.0.1:8123/v1"])

    captured = capsys.readouterr()
    assert status == 0
    assert "OPENAI_BASE_URL=http://127.0.0.1:8123/v1" in captured.out
    assert "OPENAI_MODEL_NAME=checked-model" in captured.out


def test_generic_bootstrap_cli_prints_payload(monkeypatch, tmp_path, capsys) -> None:
    import remote_inference_launcher.vllm_bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "load_vllm_bootstrapper", lambda _path: _Bootstrapper())
    config_path = tmp_path / "bootstrap.yaml"
    config_path.write_text("kind: local_vllm_bootstrap\n", encoding="utf-8")

    status = main(["bootstrap", "--config", str(config_path), "--format", "json"])

    captured = capsys.readouterr()
    assert status == 0
    assert '"kind": "local_vllm_bootstrap"' in captured.out
    assert '"python_bin": "/tmp/qwen-vllm/bin/python"' in captured.out


class _StopFailureLauncher:
    def __init__(self, config: SlurmVllmConfig) -> None:
        self.config = config

    def start(self) -> SimpleNamespace:
        return SimpleNamespace(
            api_base="http://127.0.0.1:8123/v1",
            job_id="12345",
            logs="/remote/out",
            summary_path="/tmp/slurm-summary.json",
        )

    def stop(self) -> None:
        raise RuntimeError("cleanup failed")


class _GenericLauncher:
    owns_resources = True
    stopped = False

    def start(self) -> SimpleNamespace:
        return SimpleNamespace(
            name="actor",
            api_base="http://127.0.0.1:8123/v1",
            served_model_name="actor-model",
            model="raw-model",
            api_key="secret",
            local_port=8123,
            logs="/tmp/logs",
            summary_path="/tmp/launch-summary.json",
        )

    def stop(self) -> None:
        self.stopped = True


class _SignalDuringStopLauncher(_GenericLauncher):
    def stop(self) -> None:
        os.kill(os.getpid(), signal.SIGTERM)
        self.stopped = True


class _ExternalLauncher(_GenericLauncher):
    owns_resources = False


class _CheckLauncher:
    def __init__(self, config: object) -> None:
        self.config = config

    def start(self) -> SimpleNamespace:
        return SimpleNamespace(
            name="default",
            api_base="http://127.0.0.1:8123/v1",
            served_model_name="checked-model",
        )


class _Bootstrapper:
    def run(self) -> SimpleNamespace:
        return SimpleNamespace(
            kind="local_vllm_bootstrap",
            target="local",
            out_dir="/tmp/bootstrap",
            manifest={
                "venv_path": "/tmp/qwen-vllm",
                "python_bin": "/tmp/qwen-vllm/bin/python",
                "manifest_path": "/tmp/qwen-vllm/remote-inference-launcher-bootstrap-manifest.json",
            },
        )
