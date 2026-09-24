"""Qwen-AgentWorld-specific domain prompt routing.

The prompt text is vendored from QwenLM/Qwen-AgentWorld under Apache-2.0. This
adapter only selects and fills those templates; non-AgentWorld models do not use it.
"""
from __future__ import annotations

import re
from importlib.resources import files

SUPPORTED_DOMAINS = frozenset({"mcp", "swe", "terminal"})

_MCP_BENCHMARKS = frozenset(
    {
        "crmarenapro",
        "enterpriseopsgym",
        "workbench",
        "workspacebench",
        "worldofworkflow",
        "worldofworkflows",
        "wow",
    }
)
_TERMINAL_BENCHMARKS = frozenset({"terminalbench2", "terminalbench20"})
_SWE_BENCHMARKS = frozenset({"devopsgym"})


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def is_qwen_agentworld_model(model_name: str) -> bool:
    """Recognize official IDs, local checkpoint paths, and descriptive served aliases."""
    normalized = (model_name or "").lower().replace("_", "-")
    return "qwen-agentworld" in normalized or "qwenagentworld" in _slug(normalized)


def resolve_qwen_agentworld_domain(
    benchmark_name: str,
    *,
    override: str = "",
    first_tool_name: str = "",
) -> str:
    """Map supported benchmark families to the closest AgentWorld training domain."""
    requested = (override or "").strip().lower()
    if requested:
        if requested not in SUPPORTED_DOMAINS:
            choices = ", ".join(sorted(SUPPORTED_DOMAINS))
            raise ValueError(
                f"unsupported WM_QWEN_AGENTWORLD_DOMAIN={override!r}; choose one of: {choices}"
            )
        return requested

    benchmark = _slug(benchmark_name)
    if benchmark in _TERMINAL_BENCHMARKS:
        return "terminal"
    if benchmark in _SWE_BENCHMARKS:
        return "swe"
    if benchmark in _MCP_BENCHMARKS:
        return "mcp"
    if _slug(first_tool_name) in {"runshell", "executebash", "executecommand"}:
        return "swe"
    return "mcp"


def render_qwen_agentworld_system_prompt(
    domain: str,
    *,
    tool_context: str,
    benchmark_name: str = "",
) -> str:
    """Load an official template and adapt its placeholders to the active benchmark."""
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"unsupported Qwen-AgentWorld domain: {domain}")
    template = (
        files("ejepa_wm.prompts.qwen_agentworld")
        .joinpath(f"{domain}_system_prompt.txt")
        .read_text(encoding="utf-8")
    )
    context = (tool_context or "").strip() or "No additional tool definitions were provided."
    has_tool_placeholder = "{tool_definitions}" in template
    rendered = template.replace("{tool_definitions}", context).replace("{demonstrations}", "")
    active_contract = (
        "" if has_tool_placeholder else f"\nActive tool/environment contract:\n{context}\n"
    )
    return (
        rendered.rstrip()
        + "\n\n---\n# EJEPA Benchmark Integration (adapted)\n"
        + f"Benchmark: {benchmark_name or 'unknown'}\n"
        + active_contract
        + "The historical context and current action are supplied in the user message. "
        + "Return only the predicted next environment observation in the exact format the "
        + "real tool would return. Do not add analysis, Markdown fences, or an observation label."
    )


__all__ = [
    "SUPPORTED_DOMAINS",
    "is_qwen_agentworld_model",
    "render_qwen_agentworld_system_prompt",
    "resolve_qwen_agentworld_domain",
]
