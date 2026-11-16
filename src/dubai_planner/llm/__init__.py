"""LLM provider layer: env-configured factory, structured output, deterministic mock."""

from __future__ import annotations

from dubai_planner.llm.base import ChatLLM, Completion, LLMMessage, as_chat_llm
from dubai_planner.llm.errors import (
    LLMCallError,
    LLMConfigError,
    LLMError,
    MockLLMError,
    ProviderNotInstalledError,
    StructuredOutputError,
)
from dubai_planner.llm.factory import (
    ProviderSpec,
    get_chat_llm,
    get_chat_model,
    get_crewai_llm,
    resolve_spec,
)
from dubai_planner.llm.mock import MockLLM, default_mock
from dubai_planner.llm.structured import StructuredResult, build_messages, invoke_structured

__all__ = [
    "ChatLLM",
    "Completion",
    "LLMCallError",
    "LLMConfigError",
    "LLMError",
    "LLMMessage",
    "MockLLM",
    "MockLLMError",
    "ProviderNotInstalledError",
    "ProviderSpec",
    "StructuredOutputError",
    "StructuredResult",
    "as_chat_llm",
    "build_messages",
    "default_mock",
    "get_chat_llm",
    "get_chat_model",
    "get_crewai_llm",
    "invoke_structured",
    "resolve_spec",
]
