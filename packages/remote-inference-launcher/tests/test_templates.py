from __future__ import annotations

from importlib import resources

from remote_inference_launcher.slurm_vllm import SlurmVllmConfig, render_slurm_vllm_sbatch
from remote_inference_launcher.slurm_vllm_bootstrap import (
    BOOTSTRAP_HELPER_NAME,
    SlurmVllmBootstrapConfig,
    render_bootstrap_sbatch_script,
)


def test_template_is_packaged_as_real_resource() -> None:
    template = (
        resources.files("remote_inference_launcher.templates")
        .joinpath("launch_vllm.sbatch")
        .read_text(encoding="utf-8")
    )
    assert "vllm.entrypoints.cli.main" in template
    assert "@sbatch_directives" in template


def test_ssh_template_is_packaged_as_real_resource() -> None:
    template = (
        resources.files("remote_inference_launcher.templates")
        .joinpath("launch_vllm_ssh.sh")
        .read_text(encoding="utf-8")
    )
    assert "vllm.entrypoints.cli.main" in template
    assert "@setup_cmd_block" in template


def test_rendered_template_replaces_launcher_placeholders() -> None:
    script = render_slurm_vllm_sbatch(
        SlurmVllmConfig(
            ssh_target="cluster",
            model="vendor/model",
            partition="batch",
            walltime="6:00:00",
            num_gpus=2,
            memory="384GB",
            cpus_per_task=20,
        ),
        out_dir="/remote/out",
    )
    assert "@sbatch_directives" not in script
    assert "@remote_port" not in script
    assert '"${EXTRA_ARGS[@]}"' in script


def test_bootstrap_template_and_helper_are_packaged_resources() -> None:
    template = (
        resources.files("remote_inference_launcher.templates")
        .joinpath("bootstrap_vllm.sbatch")
        .read_text(encoding="utf-8")
    )
    helper = (
        resources.files("remote_inference_launcher.templates")
        .joinpath(BOOTSTRAP_HELPER_NAME)
        .read_text(encoding="utf-8")
    )
    assert "RIL_BOOTSTRAP_ENVIRONMENT_NAME" in template
    assert "write_manifest" in helper
    assert "verify_torch_accelerator" in helper
    assert 'torch.empty((1,), device="cuda").cpu()' in helper


def test_rendered_bootstrap_template_replaces_launcher_placeholders() -> None:
    script = render_bootstrap_sbatch_script(
        SlurmVllmBootstrapConfig(
            ssh_target="cluster",
            environment_name="qwen35-vllm",
            vllm_package="vllm==0.19.1",
            partition="batch",
        ),
        out_dir="/remote/out",
    )
    assert "@sbatch_directives" not in script
    assert "@success_manifest_name" not in script
    assert "ENVIRONMENT_NAME=qwen35-vllm" in script
