"""Build LLM clients for both orchestrators from the same environment settings.

- :func:`get_chat_model` → LangChain ``BaseChatModel`` (LangGraph mode).
- :func:`get_crewai_llm` → CrewAI ``BaseLLM`` (CrewAI mode).
- :func:`get_chat_llm` → framework-neutral :class:`ChatLLM` for the agent core.

Switching ``LLM_PROVIDER`` (mock | ollama | groq | gemini | openai | anthropic) needs
no code change. Provider packages are imported lazily, so only the selected one has
to be installed; a missing package or API key raises :class:`LLMConfigError` naming
the package or env var. Keys are passed to SDKs as ``SecretStr`` where the SDK accepts
it and are never logged.

CrewAI 1.15 has native clients for OpenAI-compatible endpoints, so Groq and Gemini go
through their official OpenAI-compatible URLs and need neither LiteLLM nor extra
CrewAI extras. Anthropic needs ``crewai[anthropic]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

from dubai_planner.config import LLMProvider, Settings, get_settings
from dubai_planner.llm.base import ChatLLM, as_chat_llm
from dubai_planner.llm.errors import LLMConfigError, ProviderNotInstalledError

OLLAMA_DEFAULT_URL = "http://localhost:11434"
#: Official OpenAI-compatible endpoints **[verify]** if a provider changes them.
OPENAI_COMPATIBLE_URLS: dict[LLMProvider, str] = {
    LLMProvider.GROQ: "https://api.groq.com/openai/v1",
    LLMProvider.GEMINI: "https://generativelanguage.googleapis.com/v1beta/openai/",
}
KEY_ENV: dict[LLMProvider, str] = {
    LLMProvider.GROQ: "GROQ_API_KEY",
    LLMProvider.GEMINI: "GEMINI_API_KEY",
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
}


@dataclass(frozen=True)
class ProviderSpec:
    """Everything needed to build a client. ``repr`` never shows the key."""

    provider: LLMProvider
    model: str
    temperature: float
    timeout_s: float
    max_retries: int
    base_url: str | None = None
    api_key: SecretStr | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None

    def secret(self) -> str:
        return self.api_key.get_secret_value() if self.api_key is not None else ""


def resolve_spec(agent_name: str | None = None, settings: Settings | None = None) -> ProviderSpec:
    """Validate the configuration for ``agent_name`` without importing any provider SDK."""
    s = settings or get_settings()
    provider = s.llm_provider
    api_key: SecretStr | None = None
    if provider in KEY_ENV:
        env = KEY_ENV[provider]
        api_key = getattr(s, env.lower())
        if api_key is None or not api_key.get_secret_value().strip():
            raise LLMConfigError(
                f"LLM_PROVIDER={provider.value} requires {env} to be set (in the environment "
                f"or .env). Get a key from the provider, or use LLM_PROVIDER=ollama or mock."
            )
    if provider is LLMProvider.OLLAMA:
        base_url: str | None = s.llm_base_url or OLLAMA_DEFAULT_URL
    elif provider is LLMProvider.OPENAI:
        base_url = s.llm_base_url
    else:
        base_url = OPENAI_COMPATIBLE_URLS.get(provider)
    if base_url is not None and not base_url.startswith(("http://", "https://")):
        raise LLMConfigError(f"LLM_BASE_URL must start with http:// or https:// (got {base_url!r})")
    model = s.model_for(agent_name)
    return ProviderSpec(
        provider=provider,
        model=model,
        temperature=s.llm_temperature,
        timeout_s=s.llm_timeout_s,
        max_retries=s.llm_max_retries,
        base_url=base_url,
        api_key=api_key,
        max_tokens=s.max_tokens_for(),
        reasoning_effort=s.reasoning_effort_for(model),
    )


# --------------------------------------------------------------------------- LangChain


def _langchain_model(spec: ProviderSpec) -> Any:
    p = spec.provider
    common: dict[str, Any] = {"model": spec.model, "temperature": spec.temperature}
    limits: dict[str, Any] = {}  # only what is set, so provider defaults stay in charge
    if spec.reasoning_effort and p in (LLMProvider.GROQ, LLMProvider.OPENAI):
        limits["reasoning_effort"] = spec.reasoning_effort
    if spec.max_tokens:
        limits[{LLMProvider.OLLAMA: "num_predict",
                LLMProvider.GEMINI: "max_output_tokens"}.get(p, "max_tokens")] = spec.max_tokens
    try:
        if p is LLMProvider.MOCK:
            from dubai_planner.llm.langchain_mock import MockChatModel

            return MockChatModel()
        if p is LLMProvider.OLLAMA:
            from langchain_ollama import ChatOllama

            return ChatOllama(**common, **limits, base_url=spec.base_url,
                              client_kwargs={"timeout": spec.timeout_s})
        if p is LLMProvider.GROQ:
            from langchain_groq import ChatGroq

            return ChatGroq(**common, **limits, api_key=spec.api_key, timeout=spec.timeout_s,
                            max_retries=spec.max_retries)
        if p is LLMProvider.GEMINI:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(**common, **limits, google_api_key=spec.api_key,
                                          timeout=spec.timeout_s, max_retries=spec.max_retries)
        if p is LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(**common, **limits, api_key=spec.api_key, base_url=spec.base_url,
                              timeout=spec.timeout_s, max_retries=spec.max_retries)
        if p is LLMProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(**common, **limits, api_key=spec.api_key, timeout=spec.timeout_s,
                                 max_retries=spec.max_retries)
    except ImportError as exc:
        package = f"langchain-{p.value.replace('gemini', 'google-genai')}"
        raise ProviderNotInstalledError(p.value, package, "providers") from exc
    raise LLMConfigError(f"unsupported LLM_PROVIDER {p!r}")  # pragma: no cover


def get_chat_model(agent_name: str | None = None, settings: Settings | None = None) -> Any:
    """LangChain chat model for ``agent_name`` (honours ``LLM_MODEL__<AGENT>``)."""
    return _langchain_model(resolve_spec(agent_name, settings))


# --------------------------------------------------------------------------- CrewAI


def _crewai_llm(spec: ProviderSpec) -> Any:
    p = spec.provider
    if p is LLMProvider.MOCK:
        from dubai_planner.llm.crewai_mock import MockCrewAILLM

        return MockCrewAILLM()
    try:
        from crewai import LLM
    except ImportError as exc:
        raise ProviderNotInstalledError(p.value, "crewai", "crewai") from exc
    common: dict[str, Any] = {"temperature": spec.temperature, "timeout": spec.timeout_s,
                              "max_retries": spec.max_retries}
    try:
        if p is LLMProvider.OLLAMA:
            return LLM(model=f"ollama/{spec.model}", base_url=spec.base_url, **common)
        if p in OPENAI_COMPATIBLE_URLS:
            return LLM(model=spec.model, provider="openai", base_url=spec.base_url,
                       api_key=spec.secret(), **common)
        if p is LLMProvider.OPENAI:
            if spec.base_url:
                common["base_url"] = spec.base_url
            return LLM(model=spec.model, provider="openai", api_key=spec.secret(), **common)
        if p is LLMProvider.ANTHROPIC:
            return LLM(model=spec.model, provider="anthropic", api_key=spec.secret(), **common)
    except ImportError as exc:
        raise ProviderNotInstalledError(p.value, f"crewai[{p.value}]", "crewai") from exc
    raise LLMConfigError(f"unsupported LLM_PROVIDER {p!r}")  # pragma: no cover


def get_crewai_llm(agent_name: str | None = None, settings: Settings | None = None) -> Any:
    """CrewAI ``BaseLLM`` for ``agent_name`` from the same settings as LangGraph mode."""
    return _crewai_llm(resolve_spec(agent_name, settings))


def get_chat_llm(agent_name: str | None = None, settings: Settings | None = None) -> ChatLLM:
    """Framework-neutral client for direct agent calls (tests, scripts, evaluation).

    With ``LLM_PROVIDER=mock`` this is the shared :class:`MockLLM` engine itself (no
    LangChain needed); otherwise it wraps :func:`get_chat_model`.
    """
    spec = resolve_spec(agent_name, settings)
    if spec.provider is LLMProvider.MOCK:
        from dubai_planner.llm.mock import default_mock

        return default_mock()
    return as_chat_llm(_langchain_model(spec))
