"""Executor runtime for `mcp_react`.

Fail-fast only. Unlike benchmarks with a Docker/MCP/Redis stack, WorkBench has
no external service to bring up or probe here at all: it resolves the
upstream repository and checks that WorkBench's own required LLM credential
is present, matching `src.evals.agent._require_env`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path


def _executor_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_executor_module():
    """Import the sibling ``executor.py`` by path to reuse its repo resolver."""
    here = _executor_dir()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    path = here / "executor.py"
    spec = importlib.util.spec_from_file_location("workbench_mcp_react_executor_core", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


_PROVIDER_API_KEY_ENVS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
)
_VLLM_ENV_VARS = (
    "WORKBENCH_VLLM_BASE_URL",
    "WORKBENCH_VLLM_API_KEY",
    "WORKBENCH_VLLM_MODEL",
    "WORKBENCH_VLLM_MODEL_NAME",
)


def _missing_env_vars() -> list[str]:
    # Mirrors src.evals.agent.resolve_route(): a model's native-provider key
    # (OPENAI/ANTHROPIC/GEMINI_API_KEY) is used directly when present, and
    # OPENROUTER_API_KEY is only consulted as the fallback -- so no single key
    # is unconditionally required. This can only check that *some* credential
    # is present; whether it's the right one for the chosen model_name is
    # validated by resolve_route() itself at call time.
    if any(os.getenv(var, "").strip() for var in _VLLM_ENV_VARS):
        return []
    if any(os.getenv(var) for var in _PROVIDER_API_KEY_ENVS):
        return []
    return list(_PROVIDER_API_KEY_ENVS)


@contextmanager
def maybe_manage_executor_runtime():
    """Fail fast if the upstream repo or required credentials are missing.

    Never starts any process or container: WorkBench has none to manage.
    """
    core = _load_executor_module()
    core.resolve_repo_path()

    missing_env = _missing_env_vars()
    if missing_env:
        raise RuntimeError(
            "WorkBench mcp_react executor has no usable LLM credential set. Set one of: "
            f"{', '.join(missing_env)} (a native-provider key covers model_names on that "
            "provider directly; OPENROUTER_API_KEY is the fallback for everything else), "
            "or set WORKBENCH_VLLM_BASE_URL/WORKBENCH_VLLM_MODEL for a local vLLM endpoint. "
            "See assets/WorkBench/README.md."
        )

    yield
