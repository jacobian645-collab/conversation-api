"""Runtime settings, read from the environment (prefix CONVO_) or a .env file."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_CATALOG = _PACKAGE_DIR.parents[1] / "config" / "models.yaml"  # editable install / source checkout
_PACKAGED_CATALOG = _PACKAGE_DIR / "models.yaml"  # wheel install (force-included at build time)


def default_models_file() -> Path:
    return _REPO_CATALOG if _REPO_CATALOG.exists() else _PACKAGED_CATALOG


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONVO_", env_file=".env", extra="ignore")

    # Server
    host: str = "127.0.0.1"
    port: int = Field(default_factory=lambda: int(os.environ.get("PORT", "8080")))  # Cloud Run / ACA set PORT
    log_level: str = "info"

    # Auth: when set, every /v1 call needs `X-API-Key: <key>` (or `Authorization: Bearer <key>`).
    api_key: str | None = None

    # Models
    models_file: Path = Field(default_factory=default_models_file)
    default_model: str | None = None  # overrides `default:` in the catalog
    allow_adhoc_models: bool = True  # allow "provider:model-id" names that are not in the catalog
    request_timeout_s: float = 120.0

    # Conversation storage
    store: Literal["memory", "sqlite"] = "memory"
    sqlite_path: Path = Path("data/conversations.db")
    max_conversations: int = 1000  # memory store only; least recently used are evicted

    # Guardrails
    max_message_chars: int = 32_000
    max_system_chars: int = 16_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
