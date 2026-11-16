"""The mock engine exposed as a CrewAI LLM (a ``BaseLLM`` subclass, no LiteLLM needed)."""

from __future__ import annotations

from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel, PrivateAttr

from dubai_planner.llm.base import LLMMessage, Role
from dubai_planner.llm.mock import MockLLM, default_mock

_ROLES: dict[str, Role] = {"system": "system", "user": "user", "assistant": "assistant"}


class MockCrewAILLM(BaseLLM):
    """Deterministic CrewAI LLM backed by :class:`MockLLM`. Makes no network calls."""

    _engine: MockLLM = PrivateAttr(default_factory=default_mock)

    def __init__(self, engine: MockLLM | None = None, **data: Any) -> None:
        data.setdefault("model", "mock")
        super().__init__(**data)
        if engine is not None:
            self._engine = engine

    @property
    def engine(self) -> MockLLM:
        return self._engine

    def call(
        self,
        messages: str | list[Any],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        if isinstance(messages, str):
            converted = [LLMMessage("user", messages)]
        else:
            converted = [
                LLMMessage(_ROLES.get(str(m.get("role", "user")), "user"), str(m.get("content")))
                for m in messages
            ]
        hint = response_model.__name__ if response_model is not None else None
        return self._engine.complete(converted, schema_hint=hint).text

    def supports_function_calling(self) -> bool:
        return False
