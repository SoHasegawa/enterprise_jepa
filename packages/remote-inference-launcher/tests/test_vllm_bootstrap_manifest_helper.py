from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from remote_inference_launcher.templates import vllm_bootstrap_manifest as helper


def test_write_manifest_records_verified_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RIL_BOOTSTRAP_VLLM_PACKAGE", "vllm==1.2.3")
    monkeypatch.setenv("RIL_BOOTSTRAP_RAY_PACKAGE", "ray==2.9.0")
    monkeypatch.setenv("RIL_BOOTSTRAP_VENV_PATH", "/venv")
    monkeypatch.setenv("RIL_BOOTSTRAP_ENVIRONMENT_NAME", "bench-env")
    monkeypatch.setenv("RIL_BOOTSTRAP_BACKEND", "slurm")
    monkeypatch.setenv("RIL_BOOTSTRAP_UV_VERSION", "0.11.19")
    monkeypatch.setenv("RIL_BOOTSTRAP_INSTALL_UV_IF_MISSING", "1")
    monkeypatch.setenv("RIL_BOOTSTRAP_REUSED_EXISTING_ENV", "0")
    monkeypatch.setenv("RIL_BOOTSTRAP_FORCE", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(helper, "verify_vllm", lambda *, expected: ("1.2.3", {"ok": expected}))
    monkeypatch.setattr(helper, "verify_ray", lambda package: "2.9.0")
    monkeypatch.setattr(helper.socket, "gethostname", lambda: "host-a")

    manifest_path = tmp_path / "manifest.json"
    success_path = tmp_path / "success.json"
    helper.write_manifest([str(manifest_path), str(success_path)])

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["host"] == "host-a"
    assert payload["python_bin"] == "/venv/bin/python"
    assert payload["requested_vllm_package"] == "vllm==1.2.3"
    assert payload["installed_ray_version"] == "2.9.0"
    assert payload["install_uv_if_missing"] is True
    assert payload["reused_existing_env"] is False
    assert success_path.read_text(encoding="utf-8") == manifest_path.read_text(encoding="utf-8")


def test_bootstrap_manifest_main_returns_error_for_bad_args(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert helper.main(["only-one"]) == 2
    assert "requires manifest_path" in capsys.readouterr().err


def test_expected_version_and_required_env_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    assert helper.expected_vllm_version("vllm==0.9.1") == "0.9.1"
    with pytest.raises(RuntimeError, match="exact vllm==VERSION"):
        helper.expected_vllm_version("vllm")

    monkeypatch.delenv("RIL_MISSING", raising=False)
    with pytest.raises(RuntimeError, match="RIL_MISSING"):
        helper.required_env("RIL_MISSING")


def test_verify_vllm_reports_version_and_import_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper.metadata, "version", lambda package: "1.2.3+cuda")
    monkeypatch.setattr(helper, "runtime_diagnostics", lambda: {"python": "3.12"})
    monkeypatch.setattr(helper.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(helper, "verify_torch_accelerator", lambda diagnostics: None)
    assert helper.verify_vllm(expected="1.2.3") == ("1.2.3+cuda", {"python": "3.12"})

    with pytest.raises(RuntimeError, match=r"Expected vllm==9\.9\.9"):
        helper.verify_vllm(expected="9.9.9")

    def _raise_import_error(name: str):
        raise ImportError(name)

    monkeypatch.setattr(helper.importlib, "import_module", _raise_import_error)
    with pytest.raises(RuntimeError, match="vLLM import verification failed"):
        helper.verify_vllm(expected="1.2.3")


def test_verify_ray_handles_optional_and_exact_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    assert helper.verify_ray("") == ""
    monkeypatch.setattr(helper.metadata, "version", lambda package: "2.9.0")
    monkeypatch.setattr(helper.importlib, "import_module", lambda name: object())
    assert helper.verify_ray("ray==2.9.0") == "2.9.0"
    with pytest.raises(RuntimeError, match=r"Expected ray==2\.8\.0"):
        helper.verify_ray("ray==2.8.0")


def test_native_extension_dependency_parsing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extension = tmp_path / "vllm" / "_C.abi3.so"
    extension.parent.mkdir(parents=True)
    extension.write_text("", encoding="utf-8")

    def _fake_run(args, *, capture_output, check, text):
        assert args[-1] == str(extension)
        assert capture_output is True
        assert check is False
        assert text is True
        return subprocess.CompletedProcess(args, 0, stdout="0x1 (NEEDED) [libcuda.so]\n")

    monkeypatch.setattr(helper.sys, "path", [str(tmp_path)])
    monkeypatch.setattr(helper.subprocess, "run", _fake_run)

    extensions, needed = helper.native_extension_dependencies()

    assert extensions == [str(extension)]
    assert needed == {str(extension): ["libcuda.so"]}
    assert helper.needed_libraries("0x1 (NEEDED) [libb.so]\n0x1 (NEEDED) [libb.so]\n") == [
        "libb.so"
    ]
