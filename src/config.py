"""Configuration settings for LLM querying and app live scoring."""

from __future__ import annotations

import os
from dataclasses import dataclass

from config_loader import load_config


@dataclass(frozen=True)
class Settings:
    """Container for runtime-configurable settings used across the project."""

    api_key: str
    model_name: str = "llama-3.3-70b-versatile"
    request_timeout_seconds: int = 30
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    requests_per_minute: int = 30


CFG = load_config()

SETTINGS = Settings(
    api_key=CFG.get("groq_api_key", ""),
    model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
    request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
    max_retries=int(os.getenv("MAX_RETRIES", "5")),
    retry_backoff_seconds=float(os.getenv("RETRY_BACKOFF_SECONDS", "2.0")),
    requests_per_minute=int(os.getenv("REQUESTS_PER_MINUTE", "30")),
)
