from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest
from slurm_vllm_fakes import (
    TEST_MODEL,
    _FailingLauncher,
    _FinalTunnelFailureLauncher,
    _OutDirCaptureLauncher,
    _ReadyLauncher,
    _RecordingSummary,
    _RetryTunnelLauncher,
    _ScancelAlreadyGoneLauncher,
    _ScancelFailureLauncher,
    _ScancelTransientThenSuccessLauncher,
    _ScontrolFailureLauncher,
    _SqueueFailureLauncher,
    _StartFailureAndScancelFailureLauncher,
    base_config,
)

from remote_inference_launcher.ports import reserve_local_port
from remote_inference_launcher.slurm_vllm import (
    SlurmVllmConfig,
    SlurmVllmLauncher,
    effective_vllm_extra_args,
    render_slurm_vllm_sbatch,
    validate_slurm_vllm_config,
    with_slurm_vllm_defaults,
)


class TestSlurmVllm:
    def test_scheduler_values_require_explicit_config_even_for_known_hosts(self) -> None:
        with pytest.raises(ValueError, match="partition, walltime, num_gpus"):
            with_slurm_vllm_defaults(SlurmVllmConfig(ssh_target="slurm-login", model=TEST_MODEL))

    def test_defaults_fill_served_model_name_tensor_parallel_size_and_job_name(self) -> None:
        config = with_slurm_vllm_defaults(
            base_config(model="vendor/model", served_model_name="", job_name_prefix="test-vllm")
        )
        assert config.served_model_name == "vendor/model"
        assert config.tensor_parallel_size == 2
        assert config.job_name.startswith("test-vllm-default-")

    def test_explicit_setup_command_is_preserved(self) -> None:
        config = with_slurm_vllm_defaults(base_config(setup_cmd="module load vllm"))
        assert config.setup_cmd == "module load vllm"

    def test_rocm_eager_fallback_adds_vllm_safety_flags(self) -> None:
        config = with_slurm_vllm_defaults(
            base_config(target_device="rocm", extra_args=("--reasoning-parser", "qwen3"))
        )
        args = effective_vllm_extra_args(config)
        assert "--reasoning-parser" in args
        assert "--enforce-eager" in args
        assert "--compilation-config" in args

    def test_rocm_sbatch_moves_rocr_visible_devices_to_hip_visible_devices(self) -> None:
        script = render_slurm_vllm_sbatch(
            base_config(target_device="rocm"),
            out_dir="/remote/out",
        )

        assert (
            'export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$ROCR_VISIBLE_DEVICES}"' in script
        )
        assert "unset ROCR_VISIBLE_DEVICES" in script

    def test_rocm_ray_sbatch_wraps_python_for_ray_visible_devices(self) -> None:
        script = render_slurm_vllm_sbatch(
            base_config(
                nodes=2,
                distributed_backend="ray",
                target_device="rocm",
                num_gpus=1,
            ),
            out_dir="/remote/out",
        )

        assert "REMOTE_VLLM_NUM_GPUS=1" in script
        assert 'ROCM_PYTHON_WRAPPER="${OUT_DIR}/python-rocm-visible-devices"' in script
        assert (
            'export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$ROCR_VISIBLE_DEVICES}"' in script
        )
        assert "unset ROCR_VISIBLE_DEVICES" in script
        assert '--num-gpus="${REMOTE_VLLM_NUM_GPUS}"' in script

    def test_sbatch_script_serves_model_over_openai_endpoint(self) -> None:
        config = base_config(remote_port=18817, extra_args=("--reasoning-parser", "qwen3"))
        script = render_slurm_vllm_sbatch(config, out_dir="/remote/out")
        assert "#SBATCH --partition=batch-2gpu" in script
        assert "MODEL_ID=vendor/test-model" in script
        assert "SERVED_MODEL_NAME=vendor/test-model" in script
        assert "--served-model-name" in script
        assert "REQUESTED_REMOTE_VLLM_API_PORT=18817" in script
        assert '--port "${REMOTE_VLLM_API_PORT}"' in script
        assert "remote-inference-state.json" in script
        assert "--reasoning-parser qwen3" in script

    def test_sbatch_script_omits_model_sizing_options_unless_explicit(self) -> None:
        script = render_slurm_vllm_sbatch(base_config(), out_dir="/remote/out")
        assert "--gpu-memory-utilization" not in script
        assert "--max-model-len" not in script
        assert "--max-num-seqs" not in script
        assert "--max-num-batched-tokens" not in script

    def test_sbatch_script_includes_explicit_model_sizing_options(self) -> None:
        script = render_slurm_vllm_sbatch(
            base_config(
                gpu_memory_utilization=0.9,
                max_model_len=8192,
                max_num_seqs=1,
                max_num_batched_tokens=4096,
            ),
            out_dir="/remote/out",
        )
        assert "--gpu-memory-utilization 0.9" in script
        assert "--max-model-len 8192" in script
        assert "--max-num-seqs 1" in script
        assert "--max-num-batched-tokens 4096" in script

    def test_sbatch_script_defaults_to_short_runtime_tmp_root(self) -> None:
        script = render_slurm_vllm_sbatch(base_config(), out_dir="/remote/out")
        assert 'RUNTIME_TMP_ROOT="/tmp/remote-inference-launcher-vllm-${USER:-unknown}"' in script
        assert 'TMP_ROOT="${RUNTIME_TMP_ROOT%/}/${SLURM_JOB_ID:-$$}"' in script
        assert 'export HF_HOME="${OUT_DIR}' not in script
        assert "XDG_CACHE_HOME" not in script

    def test_sbatch_script_does_not_embed_local_hf_token(self, monkeypatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_secret_value")

        script = render_slurm_vllm_sbatch(base_config(), out_dir="/remote/out")

        assert "hf_secret_value" not in script
        assert "HF_TOKEN" not in script

    def test_sbatch_script_expands_home_shortcut_python_bin(self) -> None:
        script = render_slurm_vllm_sbatch(
            base_config(python_bin="~/.remote-vllm/bin/python"),
            out_dir="/remote/out",
        )
        assert "REMOTE_VLLM_PYTHON_BIN='~/.remote-vllm/bin/python'" in script
        assert 'REMOTE_VLLM_PYTHON_BIN="$(expand_home_path' in script
        assert 'elif [ "${1#\\~/}" != "$1" ]; then' in script

    def test_sbatch_script_expands_home_shortcut_out_dir(self) -> None:
        script = render_slurm_vllm_sbatch(base_config(), out_dir="~/remote-out")

        assert "OUT_DIR='~/remote-out'" in script
        assert 'OUT_DIR="$(expand_home_path' in script
        assert script.index('OUT_DIR="$(expand_home_path') < script.index('LOG_PATH="${OUT_DIR}')

    def test_start_expands_home_shortcut_out_dir_before_copying_script(self) -> None:
        launcher = _OutDirCaptureLauncher(base_config(local_port=8123, out_dir="~/remote-out"))

        try:
            session = launcher.start()
        finally:
            launcher.stop()

        assert launcher.remote_script == "/remote/home/remote-out/launch_vllm.sbatch"
        assert session.logs == "/remote/home/remote-out"

    def test_generated_local_port_retry_reports_final_binding(self) -> None:
        launcher = _RetryTunnelLauncher(base_config(local_port=None))

        try:
            session = launcher.start()
        finally:
            launcher.stop()

        assert len(launcher.tunnel_ports) == 2
        assert session.local_port == launcher.tunnel_ports[-1]
        assert session.api_base == f"http://127.0.0.1:{session.local_port}/v1"
        assert launcher.ready_api_base == session.api_base

    def test_final_tunnel_failure_refreshes_slurm_state_before_reraising(self) -> None:
        launcher = _FinalTunnelFailureLauncher(base_config(local_port=None))
        launcher._job_id = "12345"
        launcher._remote_node = "node001"
        reservation = reserve_local_port(launcher.config.local_bind_host, None)

        with pytest.raises(RuntimeError, match="tunnel bind failed"):
            launcher._start_tunnel_with_reservation(reservation, 18817)

        assert launcher.squeue_queries == 1
        assert launcher._latest_slurm_state == "RUNNING"
        assert launcher._latest_slurm_reason == "None"

    def test_sbatch_script_places_extra_args_after_launcher_defaults(self) -> None:
        script = render_slurm_vllm_sbatch(
            base_config(extra_args=("--no-enable-prefix-caching",)),
            out_dir="/remote/out",
        )
        assert script.index("--enable-prefix-caching") < script.index('"${EXTRA_ARGS[@]}"')

    def test_remote_config_rejects_invalid_local_port(self) -> None:
        for local_port in (0, -1, 65536):
            with pytest.raises(ValueError, match="local_port"):
                validate_slurm_vllm_config(
                    with_slurm_vllm_defaults(base_config(local_port=local_port))
                )

    def test_remote_config_requires_explicit_model(self) -> None:
        with pytest.raises(ValueError, match="model"):
            with_slurm_vllm_defaults(base_config(model=""))

    def test_remote_config_rejects_invalid_optional_model_sizing(self) -> None:
        for field_name in ("pipeline_parallel_size", "max_model_len", "max_num_batched_tokens"):
            with pytest.raises(ValueError, match=field_name):
                validate_slurm_vllm_config(with_slurm_vllm_defaults(base_config(**{field_name: 0})))

    def test_remote_config_rejects_invalid_verbosity(self) -> None:
        with pytest.raises(ValueError, match="verbosity"):
            validate_slurm_vllm_config(with_slurm_vllm_defaults(base_config(verbosity="debug")))

    def test_remote_config_rejects_single_node_distributed_backend(self) -> None:
        with pytest.raises(ValueError, match="distributed_backend requires nodes > 1"):
            validate_slurm_vllm_config(
                with_slurm_vllm_defaults(base_config(distributed_backend="ray"))
            )

    def test_quiet_suppresses_launcher_progress(self, capsys) -> None:
        launcher = _ReadyLauncher(base_config(local_port=8123, verbosity="quiet"))

        try:
            session = launcher.start()
        finally:
            launcher.stop()

        captured = capsys.readouterr()
        assert session.api_base == "http://127.0.0.1:8123/v1"
        assert captured.out == ""

    def test_progress_prints_launcher_progress(self, capsys) -> None:
        launcher = _ReadyLauncher(base_config(local_port=8123))

        try:
            launcher.start()
        finally:
            launcher.stop()

        captured = capsys.readouterr()
        assert "Submitted remote vLLM job 12345." in captured.out
        assert "Remote vLLM is ready at http://127.0.0.1:8123/v1." in captured.out

    def test_start_uses_loopback_api_base_by_default(self) -> None:
        launcher = _ReadyLauncher(base_config(local_port=8123))

        try:
            session = launcher.start()
        finally:
            launcher.stop()

        assert session.api_base == "http://127.0.0.1:8123/v1"

    def test_start_uses_configured_bind_host_for_api_base(self) -> None:
        launcher = _ReadyLauncher(base_config(local_port=8123, local_bind_host="localhost"))

        try:
            session = launcher.start()
        finally:
            launcher.stop()

        assert session.api_base == "http://localhost:8123/v1"

    def test_wildcard_bind_host_reports_loopback_api_base(self) -> None:
        launcher = _ReadyLauncher(base_config(local_port=8123, local_bind_host="*"))

        try:
            session = launcher.start()
        finally:
            launcher.stop()

        assert session.api_base == "http://127.0.0.1:8123/v1"

    @pytest.mark.parametrize("local_bind_host", ["bad host", "::1"])
    def test_remote_config_rejects_unsupported_bind_hosts(self, local_bind_host: str) -> None:
        with pytest.raises(ValueError, match="local_bind_host"):
            validate_slurm_vllm_config(
                with_slurm_vllm_defaults(base_config(local_bind_host=local_bind_host))
            )

    def test_verbose_prints_remote_command_diagnostics(self, monkeypatch, capsys) -> None:
        def fake_run(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

        monkeypatch.setattr("remote_inference_launcher.slurm_vllm.subprocess.run", fake_run)
        launcher = SlurmVllmLauncher(base_config(verbosity="verbose"))

        assert launcher._ssh_text("hostname") == "ok\n"

        captured = capsys.readouterr()
        assert "Remote command on cluster: hostname" in captured.out

    def test_ssh_text_retries_transient_transport_failure(self, monkeypatch) -> None:
        calls = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            if len(calls) == 1:
                return subprocess.CompletedProcess(
                    command,
                    255,
                    stdout="",
                    stderr="kex_exchange_identification: read: Connection reset by peer",
                )
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

        monkeypatch.setattr(
            "remote_inference_launcher.slurm_vllm_remote_ops.subprocess.run", fake_run
        )
        monkeypatch.setattr(
            "remote_inference_launcher.slurm_vllm_remote_ops.time.sleep", lambda _seconds: None
        )
        launcher = SlurmVllmLauncher(base_config())

        assert launcher._ssh_text("hostname") == "ok\n"
        assert len(calls) == 2

    def test_submit_remote_job_does_not_retry_transient_transport_failure(
        self,
        monkeypatch,
    ) -> None:
        calls = []

        def fake_run(command, **_kwargs):
            calls.append(command[-1])
            if "squeue" in command[-1]:
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(
                command,
                255,
                stdout="",
                stderr="kex_exchange_identification: read: Connection reset by peer",
            )

        monkeypatch.setattr("remote_inference_launcher.slurm_vllm.subprocess.run", fake_run)
        launcher = SlurmVllmLauncher(base_config())

        with pytest.raises(RuntimeError, match="refusing to replay sbatch"):
            launcher._submit_remote_job("/remote/out/launch_vllm.sbatch")

        assert sum("sbatch" in command for command in calls) == 1

    def test_submit_remote_job_recovers_job_id_after_transient_transport_failure(
        self,
        monkeypatch,
    ) -> None:
        calls = []

        def fake_run(command, **_kwargs):
            calls.append(command[-1])
            if "squeue" in command[-1]:
                return subprocess.CompletedProcess(command, 0, stdout="12345\n", stderr="")
            return subprocess.CompletedProcess(
                command,
                255,
                stdout="",
                stderr="kex_exchange_identification: read: Connection reset by peer",
            )

        monkeypatch.setattr("remote_inference_launcher.slurm_vllm.subprocess.run", fake_run)
        launcher = SlurmVllmLauncher(base_config())

        assert launcher._submit_remote_job("/remote/out/launch_vllm.sbatch") == "12345"
        assert sum("sbatch" in command for command in calls) == 1
        assert sum("squeue" in command for command in calls) == 1

    def test_start_cleans_up_submitted_job_when_readiness_fails(self) -> None:
        launcher = _FailingLauncher(base_config(local_port=8123))

        with pytest.raises(TimeoutError, match="not ready"):
            launcher.start()

        assert launcher.cancelled_jobs == ["12345"]
        assert launcher.job_info().state == "NOT_SUBMITTED"
        assert launcher._temp_dir is None

    def test_start_updates_summary_as_launch_facts_become_known(self, tmp_path: Path) -> None:
        launcher = _ReadyLauncher(base_config(local_port=8123))
        recording = _RecordingSummary(tmp_path / "summary.json")
        launcher._summary = recording

        try:
            launcher.start()
        finally:
            launcher.stop()

        assert [payload["lifecycle_state"] for payload in recording.payloads] == [
            "SUBMITTING",
            "SUBMITTING",
            "SUBMITTING",
            "SUBMITTED",
            "ALLOCATED",
            "REMOTE_PORT_READY",
            "TUNNEL_READY",
            "READY",
            "RELEASED",
        ]
        assert [payload.get("registry_event") for payload in recording.payloads[:3]] == [
            "remote_home_resolved",
            "remote_output_directory_created",
            "slurm_script_uploaded",
        ]
        assert recording.payloads[1]["remote_state_path"].endswith("/remote-inference-state.json")
        submitted = recording.payloads[3]
        assert submitted["job_id"] == "12345"
        assert submitted["cleanup_command"] == "ssh cluster 'scancel 12345'"
        assert submitted["remote_state_path"].endswith("/remote-inference-state.json")

    def test_failed_start_summary_remains_failed_after_cleanup(self, tmp_path: Path) -> None:
        launcher = _FailingLauncher(base_config(local_port=8123))
        recording = _RecordingSummary(tmp_path / "summary.json")
        launcher._summary = recording

        with pytest.raises(TimeoutError, match="not ready"):
            launcher.start()

        assert recording.payloads[-1]["lifecycle_state"] == "FAILED"
        assert recording.payloads[-1]["job_id"] == "12345"
        assert recording.payloads[-1]["cleanup_command"] == "ssh cluster 'scancel 12345'"

    def test_job_info_query_failures_raise_clear_error(self) -> None:
        launcher = _SqueueFailureLauncher(base_config())

        with pytest.raises(RuntimeError, match="squeue unavailable"):
            launcher._get_remote_job_info("12345")

    def test_node_resolution_failures_raise_clear_error(self) -> None:
        launcher = _ScontrolFailureLauncher(base_config())

        with pytest.raises(RuntimeError, match="scontrol unavailable"):
            launcher._get_remote_job_info("12345")

    def test_stop_cleans_local_temp_dir_when_remote_cancel_fails(self) -> None:
        launcher = _ScancelFailureLauncher(base_config())
        launcher._job_id = "12345"
        launcher._temp_dir = tempfile.TemporaryDirectory(prefix="ril-test.")
        temp_path = Path(launcher._temp_dir.name)

        with pytest.raises(RuntimeError, match="scancel failed"):
            launcher.stop()

        assert not temp_path.exists()
        assert launcher._job_id == "12345"

    def test_stop_allows_cancel_race_when_job_is_already_absent(self) -> None:
        launcher = _ScancelAlreadyGoneLauncher(base_config())
        launcher._job_id = "12345"
        launcher._temp_dir = tempfile.TemporaryDirectory(prefix="ril-test.")
        temp_path = Path(launcher._temp_dir.name)

        launcher.stop()

        assert not temp_path.exists()
        assert launcher._job_id == ""

    def test_stop_retries_cancel_when_job_is_still_present(self) -> None:
        launcher = _ScancelTransientThenSuccessLauncher(base_config())
        launcher._job_id = "12345"

        launcher.stop()

        assert launcher.scancel_attempts == 2
        assert launcher._job_id == ""

    def test_start_reports_cleanup_failure_when_cancel_fails(self) -> None:
        launcher = _StartFailureAndScancelFailureLauncher(base_config(local_port=8123))

        with pytest.raises(RuntimeError, match=r"scancel failed.*not ready"):
            launcher.start()

        assert launcher._temp_dir is None
        assert launcher._job_id == "12345"
