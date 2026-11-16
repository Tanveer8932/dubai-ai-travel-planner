"""The mock engine exposed as a LangChain chat model (used by LangGraph nodes)."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from dubai_planner.llm.base import LLMMessage, Role, _content_to_text
from dubai_planner.llm.mock import MockLLM, default_mock

_ROLES: dict[str, Role] = {"system": "system", "human": "user", "ai": "assistant"}


class MockChatModel(BaseChatModel):
    """Deterministic LangChain chat model backed by :class:`MockLLM`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "mock"
    engine: MockLLM = Field(default_factory=default_mock, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "dubai-planner-mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        converted = [
            LLMMessage(_ROLES.get(m.type, "user"), _content_to_text(m.content)) for m in messages
        ]
        completion = self.engine.complete(converted)
        message = AIMessage(
            content=completion.text,
            usage_metadata={
                "input_tokens": completion.tokens_in,
                "output_tokens": completion.tokens_out,
                "total_tokens": completion.tokens_in + completion.tokens_out,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
