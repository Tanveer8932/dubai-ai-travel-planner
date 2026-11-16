"""LLM-layer exceptions. Messages name env vars and packages, never secret values."""

from __future__ import annotations


class LLMError(RuntimeError):
    """Base class for every LLM-layer failure."""


class LLMConfigError(LLMError):
    """Provider misconfigured: unknown provider, missing API key, bad base URL."""


class ProviderNotInstalledError(LLMConfigError):
    """The selected provider's optional integration package is not installed."""

    def __init__(self, provider: str, package: str, extra: str) -> None:
        super().__init__(
            f"LLM_PROVIDER={provider} needs the optional package {package!r}. "
            f"Install it with: pip install {package}  (or pip install -e \".[{extra}]\")"
        )
        self.provider = provider
        self.package = package


class LLMCallError(LLMError):
    """The provider call itself failed (timeout, connection refused, rate limit, 5xx)."""


class StructuredOutputError(LLMError):
    """LLM output still failed schema/semantic validation after the repair attempt(s)."""

    def __init__(self, agent: str, schema: str, errors: list[str], raw: str) -> None:
        summary = "; ".join(errors[-1:])[:600]
        super().__init__(
            f"{agent}: output did not validate as {schema} after "
            f"{len(errors)} attempt(s): {summary}"
        )
        self.agent = agent
        self.schema = schema
        self.errors = errors
        self.raw = raw[:4000]


class MockLLMError(LLMError):
    """The mock provider has no scripted response, fixture or generator for a request."""


_TRANSIENT_MARKERS = ("429", "rate limit", "ratelimit", "rate_limit", "timeout", "timed out",
                      "502", "503", "504", "overloaded", "temporarily unavailable",
                      "connection")


def is_transient(exc: BaseException) -> bool:
    """Provider hiccups worth retrying (rate limits, timeouts, 5xx); never bad output.

    Shared by both orchestration modes (LangGraph node ``RetryPolicy``, CrewAI tools).
    """
    if not isinstance(exc, LLMCallError):
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)
