"""Centralized World-Model prompt store.

All WM prompt text lives here, organized one directory per backend, e.g.::

    prompts/llm/advise.md
    prompts/llm/select.md
    prompts/sidecar/inject.md

Backends MUST load their prompts via :func:`load_prompt` rather than inlining strings, so
that a new WM ships its own ``backends/<name>.py`` *and* its own ``prompts/<name>/*.md``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent


def prompt_path(backend: str, name: str) -> Path:
    return _ROOT / backend / f"{name}.md"


def load_prompt(backend: str, name: str, **variables: Any) -> str:
    """Read ``prompts/<backend>/<name>.md`` and fill ``{placeholders}`` from ``variables``.

    Raises ``FileNotFoundError`` with a clear message if the template is missing — a backend
    must never silently fall back to inline prompt text.
    """
    path = prompt_path(backend, name)
    if not path.is_file():
        raise FileNotFoundError(
            f"WM prompt not found: {path}. Add it under src/ejepa_wm/prompts/{backend}/{name}.md "
            "(prompts are centralized; do not inline prompt text in backend code)."
        )
    text = path.read_text(encoding="utf-8")
    if variables:
        text = text.format(**variables)
    return text
