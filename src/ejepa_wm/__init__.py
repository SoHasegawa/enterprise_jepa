"""Pluggable World Model (WM) for ``ejepa``.

A benchmark-agnostic World Model an agent/executor consults at its decision points, with
injection, selection, ITP-I, and model-predictive planning strategies and swappable backends (``noop``/baseline,
``served``/LLM-as-WM, ``llm``, ``ewm_imagined``/JEPA). See ``README.md`` for how to plug in a
new WM and the served-model-as-both-agent-and-WM example.
"""

from __future__ import annotations

from ejepa_wm.base import (
    STRATEGIES,
    AdviseResult,
    ChatFn,
    SelectResult,
    WMConfig,
    WorldModel,
    normalize_strategy,
)
from ejepa_wm.factory import build_world_model, wm_config_from_env

__all__ = [
    "STRATEGIES",
    "AdviseResult",
    "ChatFn",
    "SelectResult",
    "WMConfig",
    "WorldModel",
    "build_world_model",
    "normalize_strategy",
    "wm_config_from_env",
]
