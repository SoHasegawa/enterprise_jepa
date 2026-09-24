from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from pathlib import Path

import pytest

from remote_inference_launcher.core import InferenceSession
from remote_inference_launcher.existing_endpoint import (
    ExistingEndpointConfig,
    ExistingEndpointLauncher,
)
from remote_inference_launcher.inference_config import load_inference_launcher
from remote_inference_launcher.local_vllm import LocalVllmConfig, LocalVllmLauncher
from remote_inference_launcher.ports import find_available_port
from remote_inference_launcher.slurm_vllm import SlurmVllmConfig, SlurmVllmLauncher
from remote_inference_launcher.ssh_vllm import SshVllmConfig, SshVllmLauncher

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RIL_RUN_INTEGRATION") != "1",
        reason="set RIL_RUN_INTEGRATION=1 to run live serving integration tests",
    ),
]


def test_live_existing_endpoint_serves_models_and_answers_chat() -> None:
    _require_mode("existing")
    api_key = _env("RIL_EXISTING_API_KEY", "RIL_INTEGRATION_API_KEY")
    session = ExistingEndpointLauncher(
        ExistingEndpointConfig(
            api_base=_required_env("RIL_EXISTING_BASE_URL"),
            served_model_name=_env("RIL_EXISTING_MODEL"),
            api_key=api_key,
        )
    ).start()

    _assert_served_model_and_chat(session)


def test_live_generic_cli_existing_endpoint_writes_env_file_and_answers_chat(
    tmp_path: Path,
) -> None:
    _require_mode("generic_existing")
    api_key = _env("RIL_EXISTING_API_KEY", "RIL_INTEGRATION_API_KEY")
    model = _env("RIL_EXISTING_MODEL")
    config_path = tmp_path / "existing-endpoint.yaml"
    env_file = tmp_path / "inference.env"
    config_lines = [
        "kind: existing_endpoint",
        f"api_base: {_yaml_scalar(_required_env('RIL_EXISTING_BASE_URL'))}",
    ]
    if model:
        config_lines.append(f"served_model_name: {_yaml_scalar(model)}")
    if api_key:
        config_lines.append(f"api_key: {_yaml_scalar(api_key)}")
    config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    completed = _run_cli_command(
        [
            _cli_path(),
            "start",
            "--config",
            str(config_path),
            "--env-file",
            str(env_file),
            "--format",
            "json",
        ],
        timeout_seconds=_int_env("RIL_GENERIC_EXISTING_CLI_TIMEOUT_SECONDS", 60),
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["env"]["OPENAI_BASE_URL"] == _required_env("RIL_EXISTING_BASE_URL").rstrip("/")
    assert payload["env"]["OPENAI_MODEL_NAME"]
    exports = _read_env_exports(env_file)
    assert exports["OPENAI_BASE_URL"] == payload["env"]["OPENAI_BASE_URL"]
    assert exports["OPENAI_MODEL_NAME"] == payload["env"]["OPENAI_MODEL_NAME"]
    assert exports["INFERENCE_DEFAULT_BASE_URL"] == payload["env"]["OPENAI_BASE_URL"]

    _assert_served_model_and_chat(
        InferenceSession(
            name="default",
            api_base=exports["OPENAI_BASE_URL"],
            served_model_name=exports["OPENAI_MODEL_NAME"],
            api_key=exports.get("OPENAI_API_KEY", ""),
        )
    )


def test_live_local_vllm_serves_model_and_answers_chat() -> None:
    _require_mode("local")
    config = LocalVllmConfig(
        name="integration-local",
        model=_required_model("RIL_LOCAL_MODEL"),
        served_model_name=_env("RIL_LOCAL_SERVED_MODEL_NAME"),
        host=_env("RIL_LOCAL_HOST", default="0.0.0.0"),
        port=_optional_int_env("RIL_LOCAL_PORT") or find_available_port(),
        api_key=_env("RIL_LOCAL_API_KEY", "RIL_INTEGRATION_API_KEY"),
        python_bin=_required_env("RIL_LOCAL_VLLM_PYTHON_BIN"),
        out_dir=_env("RIL_LOCAL_OUT_DIR")
        or f"~/.remote-inference-launcher/integration/local-{_suffix()}",
        runtime_tmp_root=_env("RIL_LOCAL_RUNTIME_TMP_ROOT"),
        hf_home=_env("RIL_LOCAL_HF_HOME"),
        target_device=_env("RIL_LOCAL_TARGET_DEVICE"),
        tensor_parallel_size=_optional_int_env("RIL_LOCAL_TENSOR_PARALLEL_SIZE"),
        pipeline_parallel_size=_optional_int_env("RIL_LOCAL_PIPELINE_PARALLEL_SIZE"),
        data_parallel_size=_optional_int_env("RIL_LOCAL_DATA_PARALLEL_SIZE"),
        gpu_memory_utilization=_optional_float_env("RIL_LOCAL_GPU_MEMORY_UTILIZATION"),
        max_model_len=_optional_int_env("RIL_LOCAL_MAX_MODEL_LEN"),
        max_num_seqs=_optional_int_env("RIL_LOCAL_MAX_NUM_SEQS"),
        max_num_batched_tokens=_optional_int_env("RIL_LOCAL_MAX_NUM_BATCHED_TOKENS"),
        extra_args=_extra_args("RIL_LOCAL_EXTRA_ARGS"),
        check_interval_seconds=_int_env("RIL_LOCAL_CHECK_INTERVAL_SECONDS", 5),
        ready_timeout_seconds=_int_env("RIL_LOCAL_READY_TIMEOUT_SECONDS", 1800),
    )

    with LocalVllmLauncher(config).running() as session:
        _assert_served_model_and_chat(session)


def test_live_ssh_vllm_serves_model_and_answers_chat() -> None:
    _require_mode("ssh")
    config = SshVllmConfig(
        name="integration-ssh",
        ssh_target=_required_env("RIL_SSH_TARGET"),
        model=_required_model("RIL_SSH_MODEL"),
        served_model_name=_env("RIL_SSH_SERVED_MODEL_NAME"),
        remote_port=_optional_int_env("RIL_SSH_REMOTE_PORT"),
        local_port=_optional_int_env("RIL_SSH_LOCAL_PORT"),
        api_key=_env("RIL_SSH_API_KEY", "RIL_INTEGRATION_API_KEY"),
        setup_cmd=_env("RIL_SSH_SETUP_CMD"),
        python_bin=_required_env("RIL_SSH_VLLM_PYTHON_BIN"),
        out_dir=_env("RIL_SSH_OUT_DIR"),
        remote_out_dir_root=_env("RIL_SSH_REMOTE_OUT_DIR_ROOT"),
        runtime_tmp_root=_env("RIL_SSH_RUNTIME_TMP_ROOT"),
        hf_home=_env("RIL_SSH_HF_HOME"),
        target_device=_env("RIL_SSH_TARGET_DEVICE"),
        tensor_parallel_size=_optional_int_env("RIL_SSH_TENSOR_PARALLEL_SIZE"),
        pipeline_parallel_size=_optional_int_env("RIL_SSH_PIPELINE_PARALLEL_SIZE"),
        data_parallel_size=_optional_int_env("RIL_SSH_DATA_PARALLEL_SIZE"),
        gpu_memory_utilization=_optional_float_env("RIL_SSH_GPU_MEMORY_UTILIZATION"),
        max_model_len=_optional_int_env("RIL_SSH_MAX_MODEL_LEN"),
        max_num_seqs=_optional_int_env("RIL_SSH_MAX_NUM_SEQS"),
        max_num_batched_tokens=_optional_int_env("RIL_SSH_MAX_NUM_BATCHED_TOKENS"),
        extra_args=_extra_args("RIL_SSH_EXTRA_ARGS"),
        check_interval_seconds=_int_env("RIL_SSH_CHECK_INTERVAL_SECONDS", 5),
        ready_timeout_seconds=_int_env("RIL_SSH_READY_TIMEOUT_SECONDS", 1800),
    )

    with SshVllmLauncher(config).running() as session:
        _assert_served_model_and_chat(session)


def test_live_slurm_vllm_serves_model_and_answers_chat() -> None:
    _require_mode("slurm")
    job_name = f"ril-it-serving-{_suffix()}"
    config = SlurmVllmConfig(
        name="integration-slurm",
        ssh_target=_required_env("RIL_SLURM_TARGET", "RIL_INTEGRATION_SERVING_SSH_TARGET"),
        model=_required_model("RIL_SLURM_MODEL", "RIL_INTEGRATION_SERVING_MODEL"),
        served_model_name=_env(
            "RIL_SLURM_SERVED_MODEL_NAME",
            "RIL_INTEGRATION_SERVING_SERVED_MODEL_NAME",
        ),
        remote_port=_optional_int_env_any(
            ("RIL_SLURM_REMOTE_PORT", "RIL_INTEGRATION_SERVING_REMOTE_PORT"),
        ),
        local_port=_optional_int_env("RIL_SLURM_LOCAL_PORT"),
        api_key=_env("RIL_SLURM_API_KEY", "RIL_INTEGRATION_API_KEY"),
        setup_cmd=_env("RIL_SLURM_SETUP_CMD", "RIL_INTEGRATION_SERVING_SETUP_CMD"),
        python_bin=_required_env(
            "RIL_SLURM_VLLM_PYTHON_BIN",
            "RIL_INTEGRATION_SERVING_PYTHON_BIN",
        ),
        out_dir=_env("RIL_SLURM_OUT_DIR") or f"~/tmp/remote-inference-launcher/{job_name}",
        partition=_required_env("RIL_SLURM_PARTITION", "RIL_INTEGRATION_SERVING_PARTITION"),
        walltime=_env(
            "RIL_SLURM_WALLTIME",
            "RIL_INTEGRATION_SERVING_WALLTIME",
            default="00:30:00",
        ),
        num_gpus=_int_env_any(("RIL_SLURM_NUM_GPUS", "RIL_INTEGRATION_SERVING_NUM_GPUS"), 1),
        memory=_env("RIL_SLURM_MEMORY", "RIL_INTEGRATION_SERVING_MEMORY", default="32GB"),
        cpus_per_task=_int_env_any(
            ("RIL_SLURM_CPUS_PER_TASK", "RIL_INTEGRATION_SERVING_CPUS_PER_TASK"),
            4,
        ),
        exclude=_env("RIL_SLURM_EXCLUDE"),
        nodelist=_env("RIL_SLURM_NODELIST"),
        target_device=_env("RIL_SLURM_TARGET_DEVICE", "RIL_INTEGRATION_SERVING_TARGET_DEVICE"),
        tensor_parallel_size=_optional_int_env("RIL_SLURM_TENSOR_PARALLEL_SIZE"),
        pipeline_parallel_size=_optional_int_env("RIL_SLURM_PIPELINE_PARALLEL_SIZE"),
        data_parallel_size=_optional_int_env("RIL_SLURM_DATA_PARALLEL_SIZE"),
        gpu_memory_utilization=_optional_float_env_any(
            (
                "RIL_SLURM_GPU_MEMORY_UTILIZATION",
                "RIL_INTEGRATION_SERVING_GPU_MEMORY_UTILIZATION",
            )
        ),
        max_model_len=_optional_int_env_any(
            ("RIL_SLURM_MAX_MODEL_LEN", "RIL_INTEGRATION_SERVING_MAX_MODEL_LEN")
        ),
        max_num_seqs=_optional_int_env_any(
            ("RIL_SLURM_MAX_NUM_SEQS", "RIL_INTEGRATION_SERVING_MAX_NUM_SEQS")
        ),
        max_num_batched_tokens=_optional_int_env_any(
            (
                "RIL_SLURM_MAX_NUM_BATCHED_TOKENS",
                "RIL_INTEGRATION_SERVING_MAX_NUM_BATCHED_TOKENS",
            )
        ),
        extra_args=_extra_args("RIL_SLURM_EXTRA_ARGS", "RIL_INTEGRATION_SERVING_EXTRA_ARGS"),
        check_interval_seconds=_int_env_any(
            ("RIL_SLURM_CHECK_INTERVAL_SECONDS", "RIL_INTEGRATION_SERVING_CHECK_INTERVAL_SECONDS"),
            5,
        ),
        queue_timeout_seconds=_int_env_any(
            ("RIL_SLURM_QUEUE_TIMEOUT_SECONDS", "RIL_INTEGRATION_SERVING_QUEUE_TIMEOUT_SECONDS"),
            1800,
        ),
        ready_timeout_seconds=_int_env_any(
            ("RIL_SLURM_READY_TIMEOUT_SECONDS", "RIL_INTEGRATION_SERVING_READY_TIMEOUT_SECONDS"),
            1800,
        ),
        job_name=job_name,
    )

    with SlurmVllmLauncher(config).running() as session:
        _assert_served_model_and_chat(session)


def test_live_fleet_serves_all_named_models_and_answers_chat() -> None:
    _require_mode("fleet")
    config_path = Path(_required_env("RIL_FLEET_CONFIG")).expanduser()
    launcher = load_inference_launcher(config_path)
    sessions: dict[str, InferenceSession] = {}

    try:
        sessions = _normalize_sessions(launcher.start())
        if len(sessions) < _int_env("RIL_FLEET_MIN_ENDPOINTS", 2):
            pytest.fail(f"Fleet config started too few endpoints: {sorted(sessions)}")
        for session in sessions.values():
            _assert_served_model_and_chat(session)
    finally:
        launcher.stop()


def test_live_multinode_slurm_vllm_serves_model_and_answers_chat() -> None:
    _require_mode("multinode_slurm")
    config_path = Path(_required_env("RIL_MULTINODE_SLURM_CONFIG")).expanduser()
    launcher = load_inference_launcher(config_path)

    try:
        sessions = _normalize_sessions(launcher.start())
        for session in sessions.values():
            _assert_served_model_and_chat(session)
    finally:
        launcher.stop()


def _assert_served_model_and_chat(session: InferenceSession) -> None:
    model_ids = _model_ids(session.api_base, api_key=session.api_key)
    if session.served_model_name and model_ids:
        assert session.served_model_name in model_ids
    response = _chat_completion(
        session.api_base,
        model=session.served_model_name,
        api_key=session.api_key,
        timeout_seconds=_int_env("RIL_CHAT_TIMEOUT_SECONDS", 120),
    )
    assert _chat_response_text(response)


def _model_ids(api_base: str, *, api_key: str) -> set[str]:
    response = _get_json(api_base.rstrip("/") + "/models", api_key=api_key)
    data = response.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"/models response data is not a list: {response}")
    ids = set()
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.add(item["id"])
    return ids


def _chat_completion(
    api_base: str,
    *,
    model: str,
    api_key: str,
    timeout_seconds: int,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return _post_chat_completion(api_base, model=model, api_key=api_key)
        except (OSError, RuntimeError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(5)
    raise RuntimeError(f"Timed out waiting for chat completion response: {last_error}")


def _post_chat_completion(api_base: str, *, model: str, api_key: str) -> dict[str, object]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with one short sentence confirming the integration test.",
            }
        ],
        "max_tokens": _int_env("RIL_CHAT_MAX_TOKENS", 32),
        "temperature": 0,
    }
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=_json_headers(api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_int_env("RIL_CHAT_HTTP_TIMEOUT", 60)) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Chat completion failed with HTTP {error.code}: {body}") from error
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Chat completion response is not a JSON object: {body}")
    return parsed


def _get_json(url: str, *, api_key: str) -> dict[str, object]:
    request = urllib.request.Request(url, headers=_json_headers(api_key))
    with urllib.request.urlopen(request, timeout=_int_env("RIL_MODELS_HTTP_TIMEOUT", 30)) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"JSON response is not an object: {body}")
    return parsed


def _json_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


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


def _normalize_sessions(started: object) -> dict[str, InferenceSession]:
    if isinstance(started, Mapping):
        return {str(name): session for name, session in started.items()}
    name = str(getattr(started, "name", "default") or "default")
    return {name: started}


def _require_mode(mode: str) -> None:
    modes = _enabled_modes()
    if not modes:
        pytest.skip("set RIL_INTEGRATION_SERVING_MODES to select live serving modes")
    if "all" not in modes and mode not in modes:
        pytest.skip(f"set RIL_INTEGRATION_SERVING_MODES to include {mode!r}")


def _enabled_modes() -> set[str]:
    raw_value = os.getenv("RIL_INTEGRATION_SERVING_MODES") or os.getenv(
        "RIL_INTEGRATION_MODES",
        "",
    )
    return {mode.strip() for mode in raw_value.split(",") if mode.strip()}


def _required_model(*names: str) -> str:
    return _required_env(*names, "RIL_INTEGRATION_MODEL")


def _required_env(*names: str) -> str:
    value = _env(*names)
    if not value:
        pytest.fail(f"One of these environment variables is required: {', '.join(names)}")
    return value


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


def _int_env_any(names: tuple[str, ...], default: int) -> int:
    for name in names:
        raw_value = os.getenv(name)
        if raw_value:
            try:
                return int(raw_value)
            except ValueError as error:
                raise ValueError(f"{name} must be an integer.") from error
    return default


def _optional_int_env(name: str) -> int | None:
    raw_value = os.getenv(name)
    if raw_value in {None, ""}:
        return None
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


def _optional_int_env_any(names: tuple[str, ...]) -> int | None:
    for name in names:
        value = _optional_int_env(name)
        if value is not None:
            return value
    return None


def _optional_float_env(name: str) -> float | None:
    raw_value = os.getenv(name)
    if raw_value in {None, ""}:
        return None
    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number.") from error


def _optional_float_env_any(names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _optional_float_env(name)
        if value is not None:
            return value
    return None


def _extra_args(*names: str) -> tuple[str, ...]:
    raw_value = _env(*names)
    return tuple(shlex.split(raw_value)) if raw_value else ()


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


def _read_env_exports(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("export "):
            continue
        name, raw_value = line.removeprefix("export ").split("=", 1)
        parsed = shlex.split(raw_value)
        values[name] = parsed[0] if parsed else ""
    return values


def _yaml_scalar(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _suffix() -> str:
    return uuid.uuid4().hex[:8]
