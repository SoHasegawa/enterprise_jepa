from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from remote_inference_launcher.preflight import PreflightCheck
from remote_inference_launcher.validation import (
    ValidationReport,
    format_validation_report,
    validate_config_paths,
)


def _first_config_source(config_sources: Iterable[tuple[str, object]]) -> str:
    return next(iter(config_sources))[0]


def test_validate_reports_duplicate_explicit_local_ports(tmp_path: Path) -> None:
    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    first.write_text(
        "\n".join(
            [
                "kind: local_vllm",
                "name: a",
                "model: vendor/a",
                "host: 127.0.0.1",
                "port: 34567",
                "readiness:",
                "  smoke_test: disabled",
            ]
        ),
        encoding="utf-8",
    )
    second.write_text(
        "\n".join(
            [
                "kind: ssh_vllm",
                "name: b",
                "ssh_target: host",
                "model: vendor/b",
                "local_port: 34567",
                "remote_port: 18817",
                "readiness:",
                "  smoke_test: disabled",
            ]
        ),
        encoding="utf-8",
    )

    report = validate_config_paths([first, second])

    assert not report.ok
    assert any("Duplicate explicit local bind endpoint" in issue.message for issue in report.issues)


def test_validate_reports_wildcard_local_bind_collision(tmp_path: Path) -> None:
    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    first.write_text(
        "\n".join(
            [
                "kind: local_vllm",
                "name: a",
                "model: vendor/a",
                "host: 0.0.0.0",
                "port: 34568",
                "readiness:",
                "  smoke_test: disabled",
            ]
        ),
        encoding="utf-8",
    )
    second.write_text(
        "\n".join(
            [
                "kind: ssh_vllm",
                "name: b",
                "ssh_target: host",
                "model: vendor/b",
                "local_bind_host: 127.0.0.1",
                "local_port: 34568",
                "remote_port: 18817",
                "readiness:",
                "  smoke_test: disabled",
            ]
        ),
        encoding="utf-8",
    )

    report = validate_config_paths([first, second])

    assert not report.ok
    assert any("overlaps" in issue.message for issue in report.issues)


def test_validate_slurm_resource_preferences_use_only_listed_candidates_by_default(
    tmp_path: Path,
) -> None:
    config = tmp_path / "slurm.yaml"
    config.write_text(
        """
kind: slurm_vllm
name: logical
ssh_target: cluster
model: vendor/model
partition: gpu
walltime: "01:00:00"
num_gpus: 2
memory: 64G
cpus_per_task: 8
candidate_race:
  enabled: true
  max_active_candidates: 2
resource_preferences:
  - name: preferred
    num_gpus: 1
""",
        encoding="utf-8",
    )

    report = validate_config_paths([config])

    assert report.ok
    assert len(report.plans) == 1
    assert {plan.source.rsplit(" ", 1)[-1] for plan in report.plans} == {"'preferred'"}
    assert report.effective_limits["max_active_candidate_attempts"] == 1
    assert report.effective_limits["max_submitted_slurm_jobs"] == 1
    assert report.effective_limits["max_total_requested_gpus"] == 1


def test_validate_slurm_resource_preferences_include_base_when_requested(
    tmp_path: Path,
) -> None:
    config = tmp_path / "slurm.yaml"
    config.write_text(
        """
kind: slurm_vllm
name: logical
ssh_target: cluster
model: vendor/model
partition: gpu
walltime: "01:00:00"
num_gpus: 2
memory: 64G
cpus_per_task: 8
include_base_resource_candidate: true
candidate_race:
  enabled: true
  max_active_candidates: 2
resource_preferences:
  - name: preferred
    num_gpus: 1
""",
        encoding="utf-8",
    )

    report = validate_config_paths([config])

    assert report.ok
    assert len(report.plans) == 2
    assert {plan.source.rsplit(" ", 1)[-1] for plan in report.plans} == {
        "'preferred'",
        "'logical-base'",
    }
    assert report.effective_limits["max_active_candidate_attempts"] == 2
    assert report.effective_limits["max_submitted_slurm_jobs"] == 2
    assert report.effective_limits["max_total_requested_gpus"] == 3


def test_validate_requires_budget_for_fleet_candidate_racing(tmp_path: Path) -> None:
    config = tmp_path / "fleet.yaml"
    config.write_text(
        """
kind: fleet
max_active_launches: 2
endpoints:
  a:
    kind: endpoint_race
    name: a
    max_active_candidates: 2
    candidates:
      - name: a1
        kind: existing_endpoint
        api_base: http://127.0.0.1:1/v1
        readiness: {smoke_test: disabled}
      - name: a2
        kind: existing_endpoint
        api_base: http://127.0.0.1:2/v1
        readiness: {smoke_test: disabled}
  b:
    kind: existing_endpoint
    api_base: http://127.0.0.1:3/v1
    readiness: {smoke_test: disabled}
""",
        encoding="utf-8",
    )

    report = validate_config_paths([config])

    assert not report.ok
    assert any("require an explicit resource_budget" in issue.message for issue in report.issues)


def test_validate_uses_peak_budget_for_sequential_endpoint_race(tmp_path: Path) -> None:
    config = tmp_path / "race.yaml"
    config.write_text(
        """
kind: endpoint_race
max_active_candidates: 1
resource_budget:
  max_concurrent_logical_launches: 1
  max_active_candidate_attempts: 1
  max_submitted_slurm_jobs: 1
  max_total_requested_gpus: 1
candidates:
  - name: a
    kind: slurm_vllm
    ssh_target: cluster
    model: vendor/a
    partition: gpu
    walltime: "01:00:00"
    num_gpus: 1
    memory: 64G
    cpus_per_task: 8
  - name: b
    kind: slurm_vllm
    ssh_target: cluster
    model: vendor/b
    partition: gpu
    walltime: "01:00:00"
    num_gpus: 1
    memory: 64G
    cpus_per_task: 8
""",
        encoding="utf-8",
    )

    report = validate_config_paths([config])

    assert report.ok
    assert report.effective_limits["max_concurrent_logical_launches"] == 1
    assert report.effective_limits["max_active_candidate_attempts"] == 1
    assert report.effective_limits["max_submitted_slurm_jobs"] == 1
    assert report.effective_limits["max_total_requested_gpus"] == 1


def test_validate_rejects_parallel_endpoint_race_over_budget(tmp_path: Path) -> None:
    config = tmp_path / "race.yaml"
    config.write_text(
        """
kind: endpoint_race
max_active_candidates: 2
resource_budget:
  max_active_candidate_attempts: 1
candidates:
  - name: a
    kind: existing_endpoint
    api_base: http://127.0.0.1:1/v1
    readiness: {smoke_test: disabled}
  - name: b
    kind: existing_endpoint
    api_base: http://127.0.0.1:2/v1
    readiness: {smoke_test: disabled}
""",
        encoding="utf-8",
    )

    report = validate_config_paths([config])

    assert not report.ok
    assert any("max_active_candidate_attempts" in issue.message for issue in report.issues)


def test_validate_rejects_endpoint_race_allocated_winner_condition(tmp_path: Path) -> None:
    config = tmp_path / "race.yaml"
    config.write_text(
        """
kind: endpoint_race
winner_condition: allocated
candidates:
  - name: a
    kind: existing_endpoint
    api_base: http://127.0.0.1:1/v1
    readiness: {smoke_test: disabled}
""",
        encoding="utf-8",
    )

    report = validate_config_paths([config])

    assert not report.ok
    assert any(
        "winner_condition must be 'endpoint_ready'" in issue.message for issue in report.issues
    )


def test_validate_rejects_unknown_readiness_smoke_test(tmp_path: Path) -> None:
    config = tmp_path / "invalid-readiness.yaml"
    config.write_text(
        """
kind: local_vllm
model: vendor/model
readiness:
  smoke_test: models_only
""",
        encoding="utf-8",
    )

    report = validate_config_paths([config])

    assert not report.ok
    assert any("readiness.smoke_test" in issue.message for issue in report.issues)


def test_validate_rejects_slurm_candidate_race_allocated_winner_condition(
    tmp_path: Path,
) -> None:
    config = tmp_path / "slurm.yaml"
    config.write_text(
        """
kind: slurm_vllm
ssh_target: cluster
model: vendor/model
partition: gpu
walltime: "01:00:00"
num_gpus: 1
memory: 64G
cpus_per_task: 8
candidate_race:
  enabled: true
  max_active_candidates: 2
  winner_condition: allocated
resource_preferences:
  - name: preferred
    partition: gpu
  - name: fallback
    partition: gpu-long
""",
        encoding="utf-8",
    )

    report = validate_config_paths([config])

    assert not report.ok
    assert any(
        "winner_condition must be 'endpoint_ready'" in issue.message for issue in report.issues
    )


def test_validate_rejects_slurm_duplicate_resource_preference_names(
    tmp_path: Path,
) -> None:
    config = tmp_path / "slurm.yaml"
    config.write_text(
        """
kind: slurm_vllm
ssh_target: cluster
model: vendor/model
partition: gpu
walltime: "01:00:00"
num_gpus: 1
memory: 64G
cpus_per_task: 8
resource_preferences:
  - name: preferred
    partition: gpu
  - name: preferred
    partition: gpu-long
""",
        encoding="utf-8",
    )

    report = validate_config_paths([config])

    assert not report.ok
    assert any("duplicate candidate name" in issue.message for issue in report.issues)


def test_validate_rejects_unsupported_slurm_resource_preference_fields(
    tmp_path: Path,
) -> None:
    config = tmp_path / "slurm.yaml"
    config.write_text(
        """
kind: slurm_vllm
ssh_target: cluster
model: vendor/model
partition: gpu
walltime: "01:00:00"
num_gpus: 1
memory: 64G
cpus_per_task: 8
resource_preferences:
  - name: preferred
    partition: gpu
    candidate_race: {}
""",
        encoding="utf-8",
    )

    report = validate_config_paths([config])

    assert not report.ok
    assert any("unsupported fields: candidate_race" in issue.message for issue in report.issues)


def test_validate_fleet_budget_counts_held_endpoint_resources(tmp_path: Path) -> None:
    config = tmp_path / "fleet.yaml"
    config.write_text(
        """
kind: fleet
max_active_launches: 1
resource_budget:
  max_concurrent_logical_launches: 1
  max_active_candidate_attempts: 1
  max_submitted_slurm_jobs: 1
  max_total_requested_gpus: 3
endpoints:
  a:
    kind: slurm_vllm
    ssh_target: cluster
    model: vendor/a
    partition: gpu
    walltime: "01:00:00"
    num_gpus: 2
    memory: 64G
    cpus_per_task: 8
  b:
    kind: slurm_vllm
    ssh_target: cluster
    model: vendor/b
    partition: gpu
    walltime: "01:00:00"
    num_gpus: 3
    memory: 64G
    cpus_per_task: 8
""",
        encoding="utf-8",
    )

    report = validate_config_paths([config])

    assert not report.ok
    assert report.effective_limits["max_concurrent_logical_launches"] == 1
    assert report.effective_limits["max_active_candidate_attempts"] == 2
    assert report.effective_limits["max_submitted_slurm_jobs"] == 2
    assert report.effective_limits["max_total_requested_gpus"] == 5
    messages = [issue.message for issue in report.issues]
    assert any(
        "max_submitted_slurm_jobs=1 cannot fit required 2" in message for message in messages
    )
    assert any(
        "max_total_requested_gpus=3 cannot fit required 5" in message for message in messages
    )


def test_validate_counts_default_local_vllm_gpu_request(tmp_path: Path) -> None:
    config = tmp_path / "race.yaml"
    config.write_text(
        """
kind: endpoint_race
resource_budget:
  max_total_requested_gpus: 1
candidates:
  - name: local
    kind: local_vllm
    model: vendor/model
    readiness: {smoke_test: disabled}
""",
        encoding="utf-8",
    )

    report = validate_config_paths([config])

    assert report.ok
    assert report.effective_limits["max_total_requested_gpus"] == 1


def test_validate_counts_cpu_vllm_as_zero_gpu_request(tmp_path: Path) -> None:
    config = tmp_path / "race.yaml"
    config.write_text(
        """
kind: endpoint_race
resource_budget:
  max_total_requested_gpus: 1
candidates:
  - name: local
    kind: local_vllm
    model: vendor/model
    target_device: cpu
    readiness: {smoke_test: disabled}
""",
        encoding="utf-8",
    )

    report = validate_config_paths([config])

    assert report.ok
    assert report.effective_limits["max_total_requested_gpus"] is None


def test_validate_preflight_is_advisory_by_default(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "endpoint.yaml"
    config.write_text(
        "\n".join(
            [
                "kind: existing_endpoint",
                "api_base: http://127.0.0.1:1/v1",
                "readiness: {smoke_test: disabled}",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "remote_inference_launcher.validation.run_preflight_checks_for_configs",
        lambda config_sources: (
            PreflightCheck(
                source=_first_config_source(config_sources),
                endpoint_name="default",
                name="ssh-reachable",
                outcome="durable_failure",
                code="remote_python_missing",
                attempts=1,
                duration_seconds=0.01,
                detail="nope",
                command_summary="ssh <ssh-target>",
            ),
        ),
    )

    report = validate_config_paths([config], preflight=True)

    assert report.ok
    assert report.preflight_checks[0].outcome == "durable_failure"
    assert any(issue.severity == "warning" for issue in report.issues)


def test_validate_preflight_groups_multiple_config_paths(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    for path, port in ((first, 1), (second, 2)):
        path.write_text(
            "\n".join(
                [
                    "kind: existing_endpoint",
                    f"api_base: http://127.0.0.1:{port}/v1",
                    "readiness: {smoke_test: disabled}",
                ]
            ),
            encoding="utf-8",
        )
    captured_sources: list[str] = []

    def capture_sources(
        config_sources: Iterable[tuple[str, object]],
    ) -> tuple[PreflightCheck, ...]:
        captured_sources.extend(source for source, _config in config_sources)
        return ()

    monkeypatch.setattr(
        "remote_inference_launcher.validation.run_preflight_checks_for_configs",
        capture_sources,
    )

    report = validate_config_paths([first, second], preflight=True)

    assert report.ok
    assert captured_sources == [str(first), str(second)]


def test_validate_strict_preflight_fails(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "endpoint.yaml"
    config.write_text(
        "\n".join(
            [
                "kind: existing_endpoint",
                "api_base: http://127.0.0.1:1/v1",
                "readiness: {smoke_test: disabled}",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "remote_inference_launcher.validation.run_preflight_checks_for_configs",
        lambda config_sources: (
            PreflightCheck(
                source=_first_config_source(config_sources),
                endpoint_name="default",
                name="vllm-import",
                outcome="durable_failure",
                code="vllm_import_failed",
                attempts=1,
                duration_seconds=0.01,
                detail="missing",
                command_summary="python -c import vllm",
            ),
        ),
    )

    report = validate_config_paths([config], strict_preflight=True)

    assert not report.ok
    assert report.result == "failed"
    assert any(issue.severity == "error" for issue in report.issues)


def test_validate_runtime_preflight_is_advisory_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "slurm.yaml"
    config.write_text(
        "\n".join(
            [
                "kind: slurm_vllm",
                "ssh_target: cluster",
                "model: vendor/model",
                "partition: gpu",
                "walltime: '00:10:00'",
                "num_gpus: 1",
                "memory: 32GB",
                "cpus_per_task: 4",
                "readiness: {smoke_test: disabled}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "remote_inference_launcher.validation.run_runtime_preflight_checks",
        lambda _config, *, source, effective_plan: (
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
        ),
    )

    report = validate_config_paths([config], runtime_preflight=True)

    assert report.ok
    assert report.runtime_preflight
    assert report.preflight_checks[0].resolved_layer == "compute-runtime"
    assert any(issue.severity == "warning" for issue in report.issues)


def test_validate_strict_runtime_preflight_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "slurm.yaml"
    config.write_text(
        "\n".join(
            [
                "kind: slurm_vllm",
                "ssh_target: cluster",
                "model: vendor/model",
                "partition: gpu",
                "walltime: '00:10:00'",
                "num_gpus: 1",
                "memory: 32GB",
                "cpus_per_task: 4",
                "readiness: {smoke_test: disabled}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "remote_inference_launcher.validation.run_runtime_preflight_checks",
        lambda _config, *, source, effective_plan: (
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
        ),
    )
    monkeypatch.setattr(
        "remote_inference_launcher.validation.run_preflight_checks_for_configs",
        lambda _config_sources: (),
    )

    report = validate_config_paths(
        [config],
        strict_preflight=True,
        runtime_preflight=True,
    )

    assert not report.ok
    assert report.result == "failed"
    assert any(issue.severity == "error" for issue in report.issues)


def test_format_validation_report_groups_preflight_checks_by_layer() -> None:
    report = ValidationReport(
        plans=(),
        issues=(),
        preflight_checks=(
            PreflightCheck(
                source="ssh.yaml",
                endpoint_name="default",
                name="ssh-reachable",
                outcome="ok",
                code="ok",
                attempts=1,
                duration_seconds=0.01,
            ),
            PreflightCheck(
                source="slurm.yaml",
                endpoint_name="default",
                name="slurm-partition-walltime",
                outcome="durable_failure",
                code="slurm_partition_walltime_exceeded",
                attempts=1,
                duration_seconds=0.01,
                detail="requested walltime exceeds partition max",
            ),
            PreflightCheck(
                source="slurm.yaml",
                endpoint_name="default",
                name="vllm-import",
                outcome="skipped",
                code="requires_compute_allocation",
                attempts=0,
                duration_seconds=0,
            ),
        ),
    )

    output = format_validation_report(report)

    assert "  remote-control:\n    [ok ok] ssh.yaml: ssh-reachable attempts=1" in output
    assert "  slurm-scheduler:" in output
    assert "slurm-partition-walltime attempts=1: requested walltime exceeds partition max" in output
    assert "  compute-runtime:\n    [skipped requires_compute_allocation]" in output
    assert report.to_dict()["preflight_checks"][2]["layer"] == "compute-runtime"


def test_validate_strict_preflight_transient_is_inconclusive(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "endpoint.yaml"
    config.write_text(
        "\n".join(
            [
                "kind: existing_endpoint",
                "api_base: http://127.0.0.1:1/v1",
                "readiness: {smoke_test: disabled}",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "remote_inference_launcher.validation.run_preflight_checks_for_configs",
        lambda config_sources: (
            PreflightCheck(
                source=_first_config_source(config_sources),
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

    report = validate_config_paths([config], strict_preflight=True)

    assert not report.ok
    assert report.result == "inconclusive"
    assert report.preflight_summary["transient_failure"] == 1
    assert not any(issue.severity == "error" for issue in report.issues)


def test_validate_strict_preflight_unknown_can_fail(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "endpoint.yaml"
    config.write_text(
        "\n".join(
            [
                "kind: existing_endpoint",
                "api_base: http://127.0.0.1:1/v1",
                "readiness: {smoke_test: disabled}",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "remote_inference_launcher.validation.run_preflight_checks_for_configs",
        lambda config_sources: (
            PreflightCheck(
                source=_first_config_source(config_sources),
                endpoint_name="default",
                name="setup-command",
                outcome="unknown",
                code="unknown",
                attempts=1,
                duration_seconds=0.01,
                detail="exit 42",
                command_summary="ssh <ssh-target>",
            ),
        ),
    )

    report = validate_config_paths(
        [config],
        strict_preflight=True,
        fail_on_unknown_preflight=True,
    )

    assert not report.ok
    assert report.result == "failed"
    assert any(issue.severity == "error" for issue in report.issues)
