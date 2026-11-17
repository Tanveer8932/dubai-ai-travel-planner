"""Remove configured secret values from text (defence in depth, hard rule 1).

Our own errors, traces and exports never *put* secrets anywhere, but a provider SDK
could echo a key in an exception message. Everything that leaves the process (UI text,
persisted trace events, exports) passes through :func:`redact`.
"""

from __future__ import annotations

from dubai_planner.config import Settings, get_settings

SECRET_FIELDS = ("groq_api_key", "gemini_api_key", "openai_api_key", "anthropic_api_key")
MASK = "***"


def secret_values(settings: Settings | None = None) -> list[str]:
    """The configured secret values long enough to be matched safely."""
    s = settings or get_settings()
    values: list[str] = []
    for name in SECRET_FIELDS:
        secret = getattr(s, name, None)
        value = secret.get_secret_value() if secret is not None else ""
        if len(value) >= 6:
            values.append(value)
    return values


def redact(text: str, settings: Settings | None = None) -> str:
    """``text`` with every configured API key value replaced by ``***``."""
    for value in secret_values(settings):
        text = text.replace(value, MASK)
    return text
