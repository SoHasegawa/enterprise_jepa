from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

from remote_inference_launcher.slurm_vllm import SlurmVllmConfig, SlurmVllmLauncher
from remote_inference_launcher.slurm_vllm_bootstrap import (
    SlurmVllmBootstrapConfig,
    SlurmVllmBootstrapper,
)
from remote_inference_launcher.ssh import (
    SSH_TRANSPORT_FAILURE_ATTEMPTS,
    is_transient_ssh_failure,
    ssh_options,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RIL_RUN_INTEGRATION") != "1",
        reason="set RIL_RUN_INTEGRATION=1 to run live Slurm integration tests",
    ),
]


def test_live_ssh_targets_can_run_slurm() -> None:
    for target in _required_targets("RIL_INTEGRATION_SSH_TARGETS"):
        completed = _ssh(target, "squeue --version | head -n 1")
        assert "slurm" in completed.stdout.lower()


def test_live_python_bootstrap_reuses_existing_environment() -> None:
    target = _required_env("RIL_INTEGRATION_BOOTSTRAP_SSH_TARGET")
    config = _bootstrap_config(
        target=target,
        environment_name=f"ril-it-py-{_suffix()}",
        job_name=f"ril-it-bootstrap-py-{_suffix()}",
    )

    result = SlurmVllmBootstrapper(config).run()

    try:
        assert result.manifest["environment_name"] == config.environment_name
        assert result.manifest["requested_vllm_package"] == config.vllm_package
        assert result.manifest["venv_path"]
    finally:
        assert _squeue_job(target, result.job_id) == ""


def test_live_cli_bootstrap_yaml_and_overrides(tmp_path: Path) -> None:
    target = _required_env("RIL_INTEGRATION_BOOTSTRAP_SSH_TARGET")
    environment_name = f"ril-it-cli-{_suffix()}"
    job_name = f"ril-it-bootstrap-cli-{_suffix()}"
    config_path = tmp_path / "bootstrap.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"ssh_target: {target}",
                "environment_name: from-yaml",
                f"venv_path: {_required_env('RIL_INTEGRATION_BOOTSTRAP_VENV_PATH')}",
                f"vllm_package: {_required_env('RIL_INTEGRATION_BOOTSTRAP_VLLM_PACKAGE')}",
                f"backend: {os.getenv('RIL_INTEGRATION_BOOTSTRAP_BACKEND', 'auto')}",
                f"partition: {_required_env('RIL_INTEGRATION_BOOTSTRAP_PARTITION')}",
                "walltime: '00:10:00'",
                "num_gpus: 1",
                "memory: 32GB",
                "cpus_per_task: 4",
                "check_interval_seconds: 5",
                "timeout_seconds: 600",
            ]
        ),
        encoding="utf-8",
    )

    command = [
        _cli_path(),
        "slurm-vllm-bootstrap",
        "--config",
        str(config_path),
        "--environment-name",
        environment_name,
        "--job-name",
        job_name,
        "--out-dir",
        f"~/tmp/remote-inference-launcher/integration/{job_name}",
    ]
    try:
        completed = _run_cli_command(command, timeout_seconds=900)

        assert completed.returncode == 0, completed.stderr or completed.stdout
        assert "REMOTE_LOGS=" in completed.stdout
        assert environment_name in _remote_bootstrap_manifest(target).stdout
    finally:
        _scancel_jobs_by_name(target, job_name)


def test_live_serving_cli_sigterm_cleans_submitted_job(tmp_path: Path) -> None:
    target = _required_env("RIL_INTEGRATION_SERVING_SSH_TARGET")
    job_name = f"ril-it-serving-kill-{_suffix()}"
    log_path = tmp_path / "launcher.log"
    command = [
        _cli_path(),
        "slurm-vllm",
        "--ssh-target",
        target,
        "--model",
        "integration/smoke-model",
        "--served-model-name",
        "integration/smoke-model",
        "--python-bin",
        os.getenv("RIL_INTEGRATION_SERVING_PYTHON_BIN", "python"),
        "--partition",
        _required_env("RIL_INTEGRATION_SERVING_PARTITION"),
        "--walltime",
        "00:10:00",
        "--num-gpus",
        os.getenv("RIL_INTEGRATION_SERVING_NUM_GPUS", "1"),
        "--memory",
        os.getenv("RIL_INTEGRATION_SERVING_MEMORY", "32GB"),
        "--cpus-per-task",
        os.getenv("RIL_INTEGRATION_SERVING_CPUS_PER_TASK", "4"),
        "--job-name",
        job_name,
        "--setup-cmd",
        "sleep 600",
        "--check-interval-seconds",
        "2",
        "--queue-timeout-seconds",
        "600",
        "--ready-timeout-seconds",
        "600",
    ]
    _append_optional_cli_arg(
        command,
        "--remote-port",
        os.getenv("RIL_INTEGRATION_SERVING_REMOTE_PORT"),
    )
    _append_optional_cli_arg(
        command,
        "--local-port",
        os.getenv("RIL_INTEGRATION_SERVING_LOCAL_PORT"),
    )
    process = subprocess.Popen(
        command,
        stderr=subprocess.STDOUT,
        stdout=log_path.open("w", encoding="utf-8"),
        text=True,
    )
    job_id = ""
    try:
        job_id = _wait_for_job_id(target, job_name)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=60) == 130
        assert _wait_for_job_to_leave_queue(target, job_id)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=60)
        if job_id:
            _scancel(target, job_id)
        assert _squeue_job(target, job_id) == ""


def test_live_generic_start_sigterm_cleans_submitted_job(tmp_path: Path) -> None:
    target = _required_env("RIL_INTEGRATION_SERVING_SSH_TARGET")
    job_name = f"ril-it-generic-start-kill-{_suffix()}"
    config_path = tmp_path / "slurm-vllm.yaml"
    log_path = tmp_path / "generic-start.log"
    python_bin = os.getenv("RIL_INTEGRATION_SERVING_PYTHON_BIN", "python")
    config_lines = [
        "kind: slurm_vllm",
        f"ssh_target: {_yaml_scalar(target)}",
        "model: integration/smoke-model",
        "served_model_name: integration/smoke-model",
        f"python_bin: {_yaml_scalar(python_bin)}",
        f"partition: {_yaml_scalar(_required_env('RIL_INTEGRATION_SERVING_PARTITION'))}",
        "walltime: '00:10:00'",
        f"num_gpus: {os.getenv('RIL_INTEGRATION_SERVING_NUM_GPUS', '1')}",
        f"memory: {_yaml_scalar(os.getenv('RIL_INTEGRATION_SERVING_MEMORY', '32GB'))}",
        f"cpus_per_task: {os.getenv('RIL_INTEGRATION_SERVING_CPUS_PER_TASK', '4')}",
        f"job_name: {_yaml_scalar(job_name)}",
        "setup_cmd: 'sleep 600'",
        "check_interval_seconds: 2",
        "queue_timeout_seconds: 600",
        "ready_timeout_seconds: 600",
    ]
    if os.getenv("RIL_INTEGRATION_GENERIC_START_REMOTE_PORT"):
        config_lines.append(
            f"remote_port: {os.getenv('RIL_INTEGRATION_GENERIC_START_REMOTE_PORT')}"
        )
    if os.getenv("RIL_INTEGRATION_GENERIC_START_LOCAL_PORT"):
        config_lines.append(f"local_port: {os.getenv('RIL_INTEGRATION_GENERIC_START_LOCAL_PORT')}")
    config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    process = subprocess.Popen(
        [
            _cli_path(),
            "start",
            "--config",
            str(config_path),
        ],
        stderr=subprocess.STDOUT,
        stdout=log_path.open("w", encoding="utf-8"),
        text=True,
    )
    job_id = ""
    try:
        job_id = _wait_for_job_id(target, job_name)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=60) == 130
        assert _wait_for_job_to_leave_queue(target, job_id)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=60)
        if job_id:
            _scancel(target, job_id)
        assert _squeue_job(target, job_id) == ""


def test_live_serving_python_starts_model_and_answers_chat() -> None:
    target = _required_env("RIL_INTEGRATION_SERVING_SSH_TARGET")
    job_name = f"ril-it-serving-chat-{_suffix()}"
    config = _serving_config(
        target=target,
        job_name=job_name,
        remote_port=_optional_int_env("RIL_INTEGRATION_SERVING_CHAT_REMOTE_PORT"),
    )
    job_id = ""
    launcher = SlurmVllmLauncher(config)

    try:
        with launcher.running() as session:
            job_id = session.job_id
            response = _chat_completion(
                session.api_base,
                model=config.served_model_name,
                timeout_seconds=_int_env("RIL_INTEGRATION_CHAT_TIMEOUT_SECONDS", 120),
            )
            assert _chat_response_text(response)
    finally:
        if job_id:
            assert _wait_for_job_to_leave_queue(target, job_id)


def _bootstrap_config(
    *,
    target: str,
    environment_name: str,
    job_name: str,
) -> SlurmVllmBootstrapConfig:
    return SlurmVllmBootstrapConfig(
        ssh_target=target,
        environment_name=environment_name,
        venv_path=_required_env("RIL_INTEGRATION_BOOTSTRAP_VENV_PATH"),
        vllm_package=_required_env("RIL_INTEGRATION_BOOTSTRAP_VLLM_PACKAGE"),
        backend=os.getenv("RIL_INTEGRATION_BOOTSTRAP_BACKEND", "auto"),
        partition=_required_env("RIL_INTEGRATION_BOOTSTRAP_PARTITION"),
        walltime="00:10:00",
        num_gpus=1,
        memory="32GB",
        cpus_per_task=4,
        job_name=job_name,
        out_dir=f"~/tmp/remote-inference-launcher/integration/{job_name}",
        check_interval_seconds=5,
        timeout_seconds=600,
    )


def _serving_config(
    *,
    target: str,
    job_name: str,
    remote_port: int | None,
) -> SlurmVllmConfig:
    model = _required_env("RIL_INTEGRATION_SERVING_MODEL")
    return SlurmVllmConfig(
        ssh_target=target,
        verbosity=os.getenv("RIL_INTEGRATION_SERVING_VERBOSITY", "progress"),
        model=model,
        served_model_name=os.getenv("RIL_INTEGRATION_SERVING_SERVED_MODEL_NAME", model),
        remote_port=remote_port,
        python_bin=os.getenv("RIL_INTEGRATION_SERVING_PYTHON_BIN", "python"),
        partition=_required_env("RIL_INTEGRATION_SERVING_PARTITION"),
        walltime=os.getenv("RIL_INTEGRATION_SERVING_WALLTIME", "00:30:00"),
        num_gpus=_int_env("RIL_INTEGRATION_SERVING_NUM_GPUS", 1),
        memory=os.getenv("RIL_INTEGRATION_SERVING_MEMORY", "32GB"),
        cpus_per_task=_int_env("RIL_INTEGRATION_SERVING_CPUS_PER_TASK", 4),
        job_name=job_name,
        out_dir=f"~/tmp/remote-inference-launcher/integration/{job_name}",
        setup_cmd=os.getenv("RIL_INTEGRATION_SERVING_SETUP_CMD", ""),
        target_device=os.getenv("RIL_INTEGRATION_SERVING_TARGET_DEVICE", ""),
        gpu_memory_utilization=_float_env("RIL_INTEGRATION_SERVING_GPU_MEMORY_UTILIZATION", 0.9),
        max_model_len=_int_env("RIL_INTEGRATION_SERVING_MAX_MODEL_LEN", 8192),
        max_num_seqs=_int_env("RIL_INTEGRATION_SERVING_MAX_NUM_SEQS", 1),
        max_num_batched_tokens=_int_env(
            "RIL_INTEGRATION_SERVING_MAX_NUM_BATCHED_TOKENS",
            4096,
        ),
        extra_args=tuple(shlex.split(os.getenv("RIL_INTEGRATION_SERVING_EXTRA_ARGS", ""))),
        check_interval_seconds=_int_env("RIL_INTEGRATION_SERVING_CHECK_INTERVAL_SECONDS", 5),
        queue_timeout_seconds=_int_env("RIL_INTEGRATION_SERVING_QUEUE_TIMEOUT_SECONDS", 1800),
        ready_timeout_seconds=_int_env("RIL_INTEGRATION_SERVING_READY_TIMEOUT_SECONDS", 1800),
    )


def _required_targets(name: str) -> tuple[str, ...]:
    raw_value = _required_env(name)
    targets = tuple(target.strip() for target in raw_value.split(",") if target.strip())
    if not targets:
        pytest.fail(f"{name} must include at least one SSH target")
    return targets


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.fail(f"{name} is required for live Slurm integration tests")
    return value


def _append_optional_cli_arg(command: list[str], flag: str, value: str | None) -> None:
    if value:
        command.extend([flag, value])


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


def _optional_int_env(name: str) -> int | None:
    raw_value = os.getenv(name)
    if raw_value in {None, ""}:
        return None
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


def _float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number.") from error


def _cli_path() -> str:
    path = shutil.which("remote-inference-launcher")
    if not path:
        pytest.fail("remote-inference-launcher CLI was not found on PATH")
    return path


def _run_cli_command(
    command: list[str],
    *,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=60)
        raise TimeoutError(
            f"CLI timed out after {timeout_seconds}s: "
            f"stdout={stdout.strip()!r} stderr={stderr.strip()!r}"
        ) from None
    return subprocess.CompletedProcess(command, process.returncode, stdout=stdout, stderr=stderr)


def _ssh(target: str, command: str) -> subprocess.CompletedProcess[str]:
    completed = _ssh_allow_failure(target, command)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed


def _ssh_allow_failure(target: str, command: str) -> subprocess.CompletedProcess[str]:
    ssh_command = [
        "ssh",
        *ssh_options(),
        target,
        f"bash -lc {shlex.quote(command)}",
    ]
    completed = subprocess.CompletedProcess(ssh_command, 255, stdout="", stderr="SSH was not run")
    for attempt in range(1, SSH_TRANSPORT_FAILURE_ATTEMPTS + 1):
        completed = subprocess.run(
            ssh_command,
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
        if attempt == SSH_TRANSPORT_FAILURE_ATTEMPTS or not is_transient_ssh_failure(completed):
            break
        time.sleep(min(2 * attempt, 10))
    return completed


def _chat_completion(
    api_base: str,
    *,
    model: str,
    timeout_seconds: int,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return _post_chat_completion(api_base, model=model)
        except (OSError, RuntimeError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(5)
    raise RuntimeError(f"Timed out waiting for chat completion response: {last_error}")


def _post_chat_completion(api_base: str, *, model: str) -> dict[str, object]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with one short sentence confirming the test request.",
            }
        ],
        "max_tokens": 32,
        "temperature": 0,
    }
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Chat completion failed with HTTP {error.code}: {body}") from error
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Chat completion response is not a JSON object: {body}")
    return parsed


def _chat_response_text(response: dict[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Chat completion response has no choices: {response}")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError(f"Chat completion choice is not an object: {response}")
    message = first_choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    text = first_choice.get("text")
    if isinstance(text, str):
        return text.strip()
    raise RuntimeError(f"Chat completion choice has no text content: {response}")


def _remote_bootstrap_manifest(target: str) -> subprocess.CompletedProcess[str]:
    manifest_path = (
        f"{_required_env('RIL_INTEGRATION_BOOTSTRAP_VENV_PATH').rstrip('/')}"
        "/remote-inference-launcher-bootstrap-manifest.json"
    )
    command = "\n".join(
        [
            f"manifest_path={shlex.quote(manifest_path)}",
            'case "${manifest_path}" in',
            '  "~") manifest_path="${HOME}" ;;',
            '  "~/"*) manifest_path="${HOME}/${manifest_path#\\~/}" ;;',
            "esac",
            'cat "${manifest_path}"',
        ]
    )
    return _ssh(target, command)


def _wait_for_job_id(target: str, job_name: str) -> str:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        job_id = _ssh(
            target,
            f"squeue -u \"$USER\" -h -n {shlex.quote(job_name)} -o '%i' | head -n 1",
        ).stdout.strip()
        if job_id:
            return job_id
        time.sleep(2)
    pytest.fail(f"Slurm job {job_name!r} did not appear in squeue")


def _wait_for_job_to_leave_queue(target: str, job_id: str) -> bool:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if not _squeue_job(target, job_id):
            return True
        time.sleep(2)
    return False


def _squeue_job(target: str, job_id: str) -> str:
    if not job_id:
        return ""
    return _ssh(target, f"squeue -j {shlex.quote(job_id)} -h -o '%i|%T|%j|%R'").stdout.strip()


def _squeue_jobs_by_name(target: str, job_name: str) -> tuple[str, ...]:
    output = _ssh(
        target,
        f"squeue -u \"$USER\" -h -n {shlex.quote(job_name)} -o '%i'",
    ).stdout
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def _scancel_jobs_by_name(target: str, job_name: str) -> None:
    job_ids = _squeue_jobs_by_name(target, job_name)
    for job_id in job_ids:
        _scancel(target, job_id)
    for job_id in job_ids:
        if not _wait_for_job_to_leave_queue(target, job_id):
            pytest.fail(f"Slurm job {job_id} for {job_name!r} did not leave the queue")


def _scancel(target: str, job_id: str) -> None:
    completed = _ssh_allow_failure(target, f"scancel {shlex.quote(job_id)}")
    if completed.returncode == 0:
        return
    if _squeue_job(target, job_id):
        pytest.fail(
            f"failed to cancel Slurm job {job_id}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def _yaml_scalar(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _suffix() -> str:
    return uuid.uuid4().hex[:10]
