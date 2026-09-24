"""Shared strict YAML configuration loading helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_yaml_mapping(path: str | Path, *, kind: str) -> Mapping[str, Any]:
    """Load a YAML document that must be a mapping."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"{config_path} must be a YAML mapping for {kind}.")
    return raw_config
