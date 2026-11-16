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
from typing import Literal

from dotenv import dotenv_values
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


#: Model used when ``LLM_MODEL`` is empty. Hosted model ids change often **[verify]**.
DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OLLAMA: "qwen2.5:7b",
    LLMProvider.GROQ: "openai/gpt-oss-20b",
    LLMProvider.GEMINI: "gemini-2.5-flash",
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.ANTHROPIC: "claude-haiku-5-5",
    LLMProvider.MOCK: "mock",
}


ReasoningEffort = Literal["low", "medium", "high"]

#: Output-token cap when ``LLM_MAX_TOKENS`` is empty (absent = the provider's own default).
#: Groq's own default (2048) truncates reasoning models such as gpt-oss, whose hidden
#: reasoning counts as output; Groq also counts the cap against its per-minute token
#: limit, so it must stay well below the free tier's 8 000 TPM.
DEFAULT_MAX_TOKENS: dict[LLMProvider, int] = {LLMProvider.GROQ: 4096}
#: Model-id prefixes of reasoning models that get ``low`` effort when
#: ``LLM_REASONING_EFFORT`` is empty (structured extraction needs little reasoning).
LOW_EFFORT_MODEL_PREFIXES: dict[LLMProvider, tuple[str, ...]] = {
    LLMProvider.GROQ: ("openai/gpt-oss",),
}


def _dotenv_values(settings: BaseSettings) -> dict[str, str]:
    """Raw ``KEY=value`` pairs from the configured `.env` file (empty if absent)."""
    env_file = settings.model_config.get("env_file")
    if not isinstance(env_file, str | Path) or not Path(env_file).is_file():
        return {}
    return {k.upper(): v for k, v in dotenv_values(env_file).items() if v}


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
    #: Empty = the provider's default model (see ``DEFAULT_MODELS``).
    llm_model: str | None = None
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    #: Empty = the provider's default endpoint (Ollama: http://localhost:11434).
    llm_base_url: str | None = None
    #: Per-request timeout for LLM calls (seconds).
    llm_timeout_s: float = Field(default=120.0, gt=0, le=900)
    #: Transport-level retries inside the provider SDK (rate limits, 5xx). Bounded.
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    #: Repair attempts after an LLM output fails schema validation (hard rule 6: one).
    llm_repair_attempts: int = Field(default=1, ge=0, le=3)
    #: Max output tokens per call (empty = provider default, see ``DEFAULT_MAX_TOKENS``).
    llm_max_tokens: int | None = Field(default=None, ge=256, le=32_768)
    #: Reasoning effort for reasoning models (empty = see ``LOW_EFFORT_MODEL_PREFIXES``).
    llm_reasoning_effort: ReasoningEffort | None = None

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

    def model_for(self, agent_name: str | None = None) -> str:
        """Model name for ``agent_name``: ``LLM_MODEL__<AGENT>`` → ``LLM_MODEL`` → default.

        Per-agent overrides are read from the process environment first, then `.env`
        (e.g. ``LLM_MODEL__REQUIREMENTS=qwen2.5:3b`` for a smaller extraction model).
        """
        if agent_name:
            key = f"LLM_MODEL__{agent_name.upper()}"
            override = os.environ.get(key) or _dotenv_values(self).get(key)
            if override:
                return override
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]

    def max_tokens_for(self) -> int | None:
        return self.llm_max_tokens or DEFAULT_MAX_TOKENS.get(self.llm_provider)

    def reasoning_effort_for(self, model: str) -> ReasoningEffort | None:
        """Explicit ``LLM_REASONING_EFFORT``, else ``low`` for known reasoning models."""
        if self.llm_reasoning_effort is not None:
            return self.llm_reasoning_effort
        prefixes = LOW_EFFORT_MODEL_PREFIXES.get(self.llm_provider, ())
        return "low" if model.startswith(prefixes) else None

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
