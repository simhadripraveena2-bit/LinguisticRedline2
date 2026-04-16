"""Helpers for loading project configuration from config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_PATH = Path("config.yaml")


def load_config() -> Dict[str, Any]:
    """Load and return configuration from the repository-level YAML file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
