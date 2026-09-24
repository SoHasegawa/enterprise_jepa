from __future__ import annotations

import importlib
import json
import os
import socket
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        write_manifest(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def write_manifest(args: list[str]) -> None:
    if len(args) != 2:
        raise ValueError("bootstrap manifest helper requires manifest_path and success_path.")
    manifest_path = Path(args[0])
    success_path = Path(args[1])
    package = required_env("RIL_BOOTSTRAP_VLLM_PACKAGE")
    expected = expected_vllm_version(package)
    actual, diagnostics = verify_vllm(expected=expected)
    print(f"Verified vllm=={actual}")
    ray_package = required_env("RIL_BOOTSTRAP_RAY_PACKAGE")
    ray_version = verify_ray(ray_package)
    if ray_package:
        print(f"Verified ray=={ray_version}")

    venv_path = required_env("RIL_BOOTSTRAP_VENV_PATH")
    data = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "environment_name": required_env("RIL_BOOTSTRAP_ENVIRONMENT_NAME"),
        "venv_path": venv_path,
        "python_bin": f"{venv_path}/bin/python",
        "requested_vllm_package": package,
        "installed_vllm_version": actual,
        "requested_ray_package": ray_package,
        "installed_ray_version": ray_version,
        "backend": required_env("RIL_BOOTSTRAP_BACKEND"),
        "uv_version": required_env("RIL_BOOTSTRAP_UV_VERSION"),
        "install_uv_if_missing": required_env("RIL_BOOTSTRAP_INSTALL_UV_IF_MISSING") == "1",
        "reused_existing_env": required_env("RIL_BOOTSTRAP_REUSED_EXISTING_ENV") == "1",
        "force": required_env("RIL_BOOTSTRAP_FORCE") == "1",
        "manifest_path": str(manifest_path),
        "diagnostics": diagnostics,
    }
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    atomic_write_text(manifest_path, payload)
    atomic_write_text(success_path, payload)


def atomic_write_text(path: Path, payload: str) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


def expected_vllm_version(package: str) -> str:
    prefix = "vllm=="
    if not package.startswith(prefix) or package == prefix:
        raise RuntimeError(
            "Remote bootstrap manifest verification requires an exact vllm==VERSION package."
        )
    return package.removeprefix(prefix)


def verify_vllm(*, expected: str) -> tuple[str, dict[str, Any]]:
    try:
        actual = metadata.version("vllm")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("vLLM is not installed in the bootstrap environment.") from error
    if not vllm_version_matches(actual, expected):
        raise RuntimeError(f"Expected vllm=={expected}, found vllm=={actual}")

    diagnostics = runtime_diagnostics()
    try:
        importlib.import_module("vllm")
        importlib.import_module("vllm._C")
        verify_torch_accelerator(diagnostics)
    except Exception:
        print(
            "vLLM diagnostics: " + json.dumps(diagnostics, sort_keys=True),
            file=sys.stderr,
        )
        traceback.print_exc()
        raise RuntimeError("vLLM import verification failed.") from None
    return actual, diagnostics


def vllm_version_matches(actual: str, expected: str) -> bool:
    if "+" in expected:
        return actual == expected
    return actual.split("+", maxsplit=1)[0] == expected


def verify_torch_accelerator(diagnostics: dict[str, Any]) -> None:
    """Fail bootstrap when torch imports but cannot initialize the allocated GPU."""

    try:
        import torch
    except Exception as error:
        diagnostics["torch_import_error"] = repr(error)
        raise RuntimeError("Torch import verification failed.") from error
    if not torch.cuda.is_available():
        diagnostics["torch_cuda_available"] = False
        raise RuntimeError("Torch CUDA/HIP accelerator is not available.")
    try:
        device_count = torch.cuda.device_count()
        diagnostics["torch_cuda_device_count"] = device_count
        if device_count < 1:
            raise RuntimeError("Torch reported zero CUDA/HIP devices.")
        torch.cuda.set_device(0)
        diagnostics["torch_cuda_device_name_0"] = torch.cuda.get_device_name(0)
        torch.empty((1,), device="cuda").cpu()
    except Exception as error:
        diagnostics["torch_cuda_init_error"] = repr(error)
        raise RuntimeError("Torch CUDA/HIP accelerator verification failed.") from error


def verify_ray(package: str) -> str:
    if not package:
        return ""
    try:
        actual = metadata.version("ray")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("Ray is not installed in the bootstrap environment.") from error
    try:
        importlib.import_module("ray")
    except Exception as error:
        raise RuntimeError("Ray import verification failed.") from error
    exact_prefix = "ray=="
    if package.startswith(exact_prefix) and package != exact_prefix:
        expected = package.removeprefix(exact_prefix)
        if actual != expected:
            raise RuntimeError(f"Expected ray=={expected}, found ray=={actual}")
    return actual


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Required environment variable is unset: {name}")
    return value


def runtime_diagnostics() -> dict[str, Any]:
    diagnostics: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch

        diagnostics["torch_version"] = torch.__version__
        diagnostics["torch_cuda_version"] = torch.version.cuda
        diagnostics["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            diagnostics["torch_cuda_device_count"] = torch.cuda.device_count()
    except Exception as error:
        diagnostics["torch_import_error"] = repr(error)

    native_extensions, native_needed = native_extension_dependencies()
    if native_extensions:
        diagnostics["vllm_native_extensions"] = native_extensions
    if native_needed:
        diagnostics["vllm_native_needed"] = native_needed
    return diagnostics


def native_extension_dependencies() -> tuple[list[str], dict[str, list[str]]]:
    native_extensions: list[str] = []
    native_needed: dict[str, list[str]] = {}
    for base in map(Path, sys.path):
        for path in base.glob("vllm/_C*.so"):
            native_extensions.append(str(path))
            try:
                readelf = subprocess.run(
                    ["readelf", "-d", str(path)],
                    capture_output=True,
                    check=False,
                    text=True,
                )
            except OSError:
                continue
            if readelf.returncode:
                continue
            dependencies = needed_libraries(readelf.stdout)
            if dependencies:
                native_needed[str(path)] = dependencies
    return native_extensions, native_needed


def needed_libraries(readelf_output: str) -> list[str]:
    libraries = []
    for line in readelf_output.splitlines():
        if "(NEEDED)" in line:
            library = line.rsplit("[", maxsplit=1)[1].split("]", maxsplit=1)[0]
            libraries.append(library)
    return sorted(set(libraries))


if __name__ == "__main__":
    raise SystemExit(main())
