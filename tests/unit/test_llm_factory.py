"""AC2.1: provider switching is configuration-only; missing keys/packages fail clearly."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from dubai_planner.config import DEFAULT_MODELS, LLMProvider, Settings
from dubai_planner.llm import (
    LLMConfigError,
    ProviderNotInstalledError,
    get_chat_llm,
    get_chat_model,
    get_crewai_llm,
    resolve_spec,
)
from dubai_planner.llm.factory import KEY_ENV, OPENAI_COMPATIBLE_URLS

FAKE_KEY = "test-key-not-real-0123456789"


class _FakeChat:
    """Records constructor kwargs; stands in for a LangChain chat class."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.fixture
def fake_langchain_providers(monkeypatch: pytest.MonkeyPatch) -> dict[str, type]:
    """Install fake provider packages so the factory's lazy imports succeed offline."""
    classes = {}
    for module, cls in [("langchain_ollama", "ChatOllama"), ("langchain_groq", "ChatGroq"),
                        ("langchain_google_genai", "ChatGoogleGenerativeAI"),
                        ("langchain_openai", "ChatOpenAI"),
                        ("langchain_anthropic", "ChatAnthropic")]:
        fake = types.ModuleType(module)
        klass = type(cls, (_FakeChat,), {})
        setattr(fake, cls, klass)
        monkeypatch.setitem(sys.modules, module, fake)
        classes[module] = klass
    return classes


def _settings(monkeypatch: pytest.MonkeyPatch, provider: str, **env: str) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", provider)
    for name in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                 "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("provider", "env", "module"),
    [
        ("ollama", {}, "langchain_ollama"),
        ("groq", {"GROQ_API_KEY": FAKE_KEY}, "langchain_groq"),
        ("gemini", {"GEMINI_API_KEY": FAKE_KEY}, "langchain_google_genai"),
        ("openai", {"OPENAI_API_KEY": FAKE_KEY}, "langchain_openai"),
        ("anthropic", {"ANTHROPIC_API_KEY": FAKE_KEY}, "langchain_anthropic"),
    ],
)
def test_switching_provider_needs_only_env(
    monkeypatch: pytest.MonkeyPatch, fake_langchain_providers: dict[str, type],
    provider: str, env: dict[str, str], module: str,
) -> None:
    s = _settings(monkeypatch, provider, **env)
    model = get_chat_model(settings=s)
    assert isinstance(model, fake_langchain_providers[module])
    assert model.kwargs["model"] == DEFAULT_MODELS[LLMProvider(provider)]
    assert model.kwargs["temperature"] == pytest.approx(0.2)
    if provider != "ollama":
        assert model.kwargs["max_retries"] == s.llm_max_retries
        assert model.kwargs["timeout"] == s.llm_timeout_s
        key = model.kwargs.get("api_key") or model.kwargs.get("google_api_key")
        assert key.get_secret_value() == FAKE_KEY  # passed as SecretStr, not plain text
    else:
        assert model.kwargs["base_url"] == "http://localhost:11434"
        assert model.kwargs["client_kwargs"] == {"timeout": s.llm_timeout_s}


def test_mock_provider_needs_no_packages_or_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch, "mock")
    from dubai_planner.llm.crewai_mock import MockCrewAILLM
    from dubai_planner.llm.langchain_mock import MockChatModel
    from dubai_planner.llm.mock import MockLLM

    assert isinstance(get_chat_model(settings=s), MockChatModel)
    assert isinstance(get_crewai_llm(settings=s), MockCrewAILLM)
    assert isinstance(get_chat_llm(settings=s), MockLLM)


@pytest.mark.parametrize("provider", ["groq", "gemini", "openai", "anthropic"])
def test_missing_key_names_the_env_var(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    s = _settings(monkeypatch, provider)
    env = KEY_ENV[LLMProvider(provider)]
    with pytest.raises(LLMConfigError, match=env):
        get_chat_model(settings=s)
    with pytest.raises(LLMConfigError, match=env):
        get_crewai_llm(settings=s)


def test_blank_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch, "groq", GROQ_API_KEY="   ")
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        resolve_spec(settings=s)


def test_missing_package_names_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langchain_groq", None)  # import raises ImportError
    s = _settings(monkeypatch, "groq", GROQ_API_KEY=FAKE_KEY)
    with pytest.raises(ProviderNotInstalledError, match="langchain-groq"):
        get_chat_model(settings=s)


def test_spec_repr_never_contains_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = resolve_spec(settings=_settings(monkeypatch, "openai", OPENAI_API_KEY=FAKE_KEY))
    assert FAKE_KEY not in repr(spec)
    assert FAKE_KEY not in str(spec)


def test_per_agent_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch, "ollama", LLM_MODEL="llama3.1:8b",
                  LLM_MODEL__REQUIREMENTS="qwen2.5:3b")
    assert resolve_spec("requirements", s).model == "qwen2.5:3b"
    assert resolve_spec("composer", s).model == "llama3.1:8b"
    assert resolve_spec(None, s).model == "llama3.1:8b"


def test_explicit_model_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch, "openai", OPENAI_API_KEY=FAKE_KEY, LLM_MODEL="my-model",
                  LLM_BASE_URL="http://localhost:8000/v1")
    spec = resolve_spec(settings=s)
    assert (spec.model, spec.base_url) == ("my-model", "http://localhost:8000/v1")


def test_invalid_base_url_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch, "ollama", LLM_BASE_URL="localhost:11434")
    with pytest.raises(LLMConfigError, match="LLM_BASE_URL"):
        resolve_spec(settings=s)


# ------------------------------------------------------------------ CrewAI (real classes)


@pytest.mark.parametrize(
    ("provider", "env"),
    [("ollama", {}), ("groq", {"GROQ_API_KEY": FAKE_KEY}),
     ("gemini", {"GEMINI_API_KEY": FAKE_KEY}), ("openai", {"OPENAI_API_KEY": FAKE_KEY})],
)
def test_crewai_llm_built_from_same_settings(
    monkeypatch: pytest.MonkeyPatch, provider: str, env: dict[str, str]
) -> None:
    """Constructing CrewAI's native clients makes no network call and needs no LiteLLM."""
    from crewai.llms.base_llm import BaseLLM

    s = _settings(monkeypatch, provider, **env)
    llm = get_crewai_llm("attractions", s)
    assert isinstance(llm, BaseLLM)
    assert DEFAULT_MODELS[LLMProvider(provider)] in llm.model
    if provider in ("groq", "gemini"):
        assert llm.base_url == OPENAI_COMPATIBLE_URLS[LLMProvider(provider)]
    if provider == "ollama":
        assert "11434" in str(llm.base_url)
    assert getattr(llm, "timeout", s.llm_timeout_s) == s.llm_timeout_s


def test_crewai_anthropic_without_extra_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("crewai")
    try:
        import anthropic  # noqa: F401
    except ImportError:
        s = _settings(monkeypatch, "anthropic", ANTHROPIC_API_KEY=FAKE_KEY)
        with pytest.raises(ProviderNotInstalledError, match=r"crewai\[anthropic\]"):
            get_crewai_llm(settings=s)
    else:  # pragma: no cover - extra installed on this machine
        pytest.skip("anthropic SDK installed; nothing to check")


# ------------------------------------------------------------------ output limits (Phase 3)


def _built(monkeypatch: pytest.MonkeyPatch, provider: str, **env: str) -> Any:
    from dubai_planner.llm.factory import get_chat_model

    for name in ("LLM_MAX_TOKENS", "LLM_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)
    return get_chat_model(settings=_settings(monkeypatch, provider, **env))


def test_groq_gpt_oss_defaults_avoid_truncated_reasoning(
        monkeypatch: pytest.MonkeyPatch, fake_langchain_providers: dict[str, type]) -> None:
    """Groq's own 2048-token default truncated gpt-oss JSON in the AC3.6 smoke run."""
    model = _built(monkeypatch, "groq", GROQ_API_KEY=FAKE_KEY)
    assert model.kwargs["max_tokens"] == 4096 and model.kwargs["reasoning_effort"] == "low"


def test_output_limits_follow_env_and_provider_names(
        monkeypatch: pytest.MonkeyPatch, fake_langchain_providers: dict[str, type]) -> None:
    groq = _built(monkeypatch, "groq", GROQ_API_KEY=FAKE_KEY, LLM_MAX_TOKENS="2000",
                  LLM_REASONING_EFFORT="high")
    assert groq.kwargs["max_tokens"] == 2000 and groq.kwargs["reasoning_effort"] == "high"
    llama = _built(monkeypatch, "groq", GROQ_API_KEY=FAKE_KEY, LLM_MODEL="llama-x")
    assert "reasoning_effort" not in llama.kwargs  # not a known reasoning model
    ollama = _built(monkeypatch, "ollama", LLM_MAX_TOKENS="1024")
    assert ollama.kwargs["num_predict"] == 1024 and "max_tokens" not in ollama.kwargs
    gemini = _built(monkeypatch, "gemini", GEMINI_API_KEY=FAKE_KEY, LLM_MAX_TOKENS="1024")
    assert gemini.kwargs["max_output_tokens"] == 1024
    plain = _built(monkeypatch, "anthropic", ANTHROPIC_API_KEY=FAKE_KEY)
    assert "max_tokens" not in plain.kwargs and "reasoning_effort" not in plain.kwargs


def test_invalid_reasoning_effort_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _settings(monkeypatch, "groq", LLM_REASONING_EFFORT="extreme")
