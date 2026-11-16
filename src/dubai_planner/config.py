"""Application settings loaded from environment variables and an optional `.env` file.

All secrets are typed :class:`pydantic.SecretStr` so they never appear in ``repr``/logs.
Nothing in this module (or anywhere else) may hardcode a credential.
"""

from __future__ import annotations

import os
from decimal import Decimal
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root (``src/dubai_planner/config.py`` → two levels up from the package).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class LLMProvider(StrEnum):
    """Supported LLM back-ends. ``mock`` is deterministic and fully offline."""

    OLLAMA = "ollama"
    GROQ = "groq"
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


class Settings(BaseSettings):
    """Typed, validated runtime configuration (see `.env.example` for documentation)."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `GROQ_API_KEY=` means "not set", not "empty secret"
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM provider ------------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    llm_model: str = "qwen2.5:7b"
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_base_url: str = "http://localhost:11434"

    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # --- Storage -----------------------------------------------------------
    db_path: Path = Path("data/app.db")
    checkpoint_db_path: Path = Path("data/checkpoints.db")
    cache_dir: Path = Path("data/cache")
    cache_ttl_hours: int = Field(default=24, ge=0)
    seed_dir: Path = Path("data/seed")

    # --- External data -----------------------------------------------------
    http_user_agent: str = "dubai-ai-travel-planner/0.1 (academic project)"
    http_timeout_s: float = Field(default=20.0, gt=0, le=120)
    #: When true, no network request is made: clients use cache, then seed fallbacks.
    http_offline: bool = False

    # --- Planner behaviour -------------------------------------------------
    max_review_iterations: int = Field(default=2, ge=0, le=10)
    budget_tolerance: Decimal = Field(default=Decimal("0.05"), ge=0, le=1)

    # --- Telemetry / logging -----------------------------------------------
    crewai_disable_telemetry: bool = True
    otel_sdk_disabled: bool = True
    log_level: str = "INFO"

    @field_validator("db_path", "checkpoint_db_path", "cache_dir", "seed_dir")
    @classmethod
    def _resolve_relative_to_root(cls, value: Path) -> Path:
        """Relative paths are anchored at the repo root, independent of the CWD."""
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid LOG_LEVEL {value!r}")
        return level

    def apply_runtime_env(self) -> None:
        """Export telemetry opt-outs to ``os.environ``.

        CrewAI/OpenTelemetry read these from the process environment at import time,
        so they must be set before ``import crewai``. Values from `.env` are not in
        ``os.environ`` by default, hence this explicit step. Secrets are never exported.
        """
        os.environ["CREWAI_DISABLE_TELEMETRY"] = str(self.crewai_disable_telemetry).lower()
        os.environ["OTEL_SDK_DISABLED"] = str(self.otel_sdk_disabled).lower()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
