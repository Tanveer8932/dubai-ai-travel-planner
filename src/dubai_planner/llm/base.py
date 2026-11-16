"""Framework-neutral chat interface used by the shared agent core.

Agents never import LangChain or CrewAI. They call a :class:`ChatLLM`, which is one of:

- a LangChain ``BaseChatModel`` (LangGraph mode), wrapped by :class:`LangChainChatLLM`;
- a CrewAI ``BaseLLM`` (CrewAI mode), wrapped by :class:`CrewAIChatLLM`;
- anything else implementing ``complete(messages) -> Completion`` (e.g. ``MockLLM``).

:func:`as_chat_llm` performs the wrapping, so ``get_chat_model()`` and
``get_crewai_llm()`` objects can both be passed straight to an agent's ``run``.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from dubai_planner.llm.errors import LLMCallError

Role = Literal["system", "user", "assistant"]

#: LangChain integrations whose ``with_structured_output`` we trust (JSON schema / tools).
NATIVE_STRUCTURED_PACKAGES = frozenset({
    "langchain_ollama", "langchain_groq", "langchain_google_genai", "langchain_openai",
    "langchain_anthropic",
})
#: ``with_structured_output(method=...)`` per integration when the default is unreliable.
#: Groq's default (forced tool call) is rejected with HTTP 400 ``tool_use_failed`` when a
#: model such as ``openai/gpt-oss-20b`` answers in plain JSON instead; ``json_schema``
#: sends a response format instead of forcing a tool.
NATIVE_STRUCTURED_METHODS: dict[str, str] = {"langchain_groq": "json_schema"}
#: Provider error codes meaning "the model replied, but not in the constrained format".
REJECTED_GENERATION_CODES = frozenset({"tool_use_failed", "json_validate_failed"})


@dataclass(frozen=True)
class LLMMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""


@dataclass(frozen=True)
class NativeResult:
    """Outcome of a provider-native structured-output call."""

    parsed: BaseModel | None
    completion: Completion
    error: str | None = None


@runtime_checkable
class ChatLLM(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, messages: Sequence[LLMMessage]) -> Completion: ...


@runtime_checkable
class NativeStructuredLLM(ChatLLM, Protocol):
    """A ChatLLM that can also ask the provider for schema-constrained output."""

    def complete_structured(
        self, messages: Sequence[LLMMessage], schema: type[BaseModel]
    ) -> NativeResult: ...


def estimate_tokens(text: str) -> int:
    """Rough token count (≈ 4 characters per token) when the provider reports none."""
    return max(1, len(text) // 4) if text else 0


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # LangChain content blocks
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
        return "".join(parts)
    return str(content)


def rejected_generation(exc: BaseException) -> tuple[str, str] | None:
    """``(code, generated_text)`` when a provider rejected a schema-constrained reply.

    OpenAI-compatible APIs (Groq) return HTTP 400 with ``error.code`` and the model's
    ``failed_generation`` when the reply did not match the requested tool/format. That is
    an output-format problem, not a transport failure, so it can go through the normal
    validate → repair path. Returns ``None`` for any other error.
    """
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return None
    code = str(error.get("code") or "")
    if code not in REJECTED_GENERATION_CODES:
        return None
    return code, str(error.get("failed_generation") or "")


class LangChainChatLLM:
    """Adapter: LangChain ``BaseChatModel`` → :class:`ChatLLM`."""

    def __init__(self, model: Any, *, native_structured: bool = False,
                 structured_method: str | None = None) -> None:
        self.model = model
        self.native_structured = native_structured
        self.structured_method = structured_method

    @property
    def name(self) -> str:
        return str(getattr(self.model, "model", None) or getattr(self.model, "model_name", None)
                   or type(self.model).__name__)

    @staticmethod
    def _to_lc(messages: Sequence[LLMMessage]) -> list[Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        kinds = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        return [kinds[m.role](content=m.content) for m in messages]

    def _completion(self, message: Any, prompt_text: str) -> Completion:
        text = _content_to_text(getattr(message, "content", ""))
        usage = getattr(message, "usage_metadata", None) or {}
        return Completion(
            text=text,
            tokens_in=int(usage.get("input_tokens") or estimate_tokens(prompt_text)),
            tokens_out=int(usage.get("output_tokens") or estimate_tokens(text)),
            model=self.name,
        )

    def complete(self, messages: Sequence[LLMMessage]) -> Completion:
        try:
            reply = self.model.invoke(self._to_lc(messages))
        except Exception as exc:
            raise LLMCallError(f"{self.name}: {type(exc).__name__}: {exc}"[:500]) from exc
        return self._completion(reply, "".join(m.content for m in messages))

    def complete_structured(
        self, messages: Sequence[LLMMessage], schema: type[BaseModel]
    ) -> NativeResult:
        prompt_text = "".join(m.content for m in messages)
        kwargs: dict[str, Any] = {"include_raw": True}
        if self.structured_method:
            kwargs["method"] = self.structured_method
        try:
            runnable = self.model.with_structured_output(schema, **kwargs)
            out = runnable.invoke(self._to_lc(messages))
        except NotImplementedError:
            raise
        except Exception as exc:
            rejected = rejected_generation(exc)
            if rejected is None:
                raise LLMCallError(f"{self.name}: {type(exc).__name__}: {exc}"[:500]) from exc
            code, text = rejected
            return NativeResult(
                parsed=None,
                completion=Completion(text=text, tokens_in=estimate_tokens(prompt_text),
                                      tokens_out=estimate_tokens(text), model=self.name),
                error=f"provider rejected the structured reply ({code})",
            )
        raw = out.get("raw") if isinstance(out, dict) else None
        completion = self._completion(raw, prompt_text)
        parsed = out.get("parsed") if isinstance(out, dict) else out
        error = out.get("parsing_error") if isinstance(out, dict) else None
        if parsed is not None and not isinstance(parsed, schema):
            parsed = None
            error = error or f"native output was not a {schema.__name__}"
        return NativeResult(parsed=parsed, completion=completion,
                            error=str(error) if error else None)


class CrewAIChatLLM:
    """Adapter: CrewAI ``BaseLLM`` → :class:`ChatLLM`."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    @property
    def name(self) -> str:
        return str(getattr(self.llm, "model", None) or type(self.llm).__name__)

    def complete(self, messages: Sequence[LLMMessage]) -> Completion:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        try:
            text = self.llm.call(payload)
        except Exception as exc:
            raise LLMCallError(f"{self.name}: {type(exc).__name__}: {exc}"[:500]) from exc
        text = _content_to_text(text)
        return Completion(
            text=text,
            tokens_in=estimate_tokens("".join(m.content for m in messages)),
            tokens_out=estimate_tokens(text),
            model=self.name,
        )


def _is_instance(obj: Any, module: str, cls_name: str) -> bool:
    """``isinstance`` against an optional framework class without importing it eagerly."""
    try:
        cls = getattr(importlib.import_module(module), cls_name)
    except ImportError:  # pragma: no cover - framework not installed
        return False
    return isinstance(obj, cls)


def as_chat_llm(obj: Any) -> ChatLLM:
    """Wrap a LangChain chat model or CrewAI LLM; pass a ChatLLM through unchanged."""
    if isinstance(obj, LangChainChatLLM | CrewAIChatLLM):
        return obj
    if _is_instance(obj, "langchain_core.language_models.chat_models", "BaseChatModel"):
        package = type(obj).__module__.split(".")[0]
        return LangChainChatLLM(obj, native_structured=package in NATIVE_STRUCTURED_PACKAGES,
                                structured_method=NATIVE_STRUCTURED_METHODS.get(package))
    if _is_instance(obj, "crewai.llms.base_llm", "BaseLLM"):
        return CrewAIChatLLM(obj)
    if isinstance(obj, ChatLLM):
        return obj
    raise TypeError(f"unsupported LLM object {type(obj).__name__}; expected a LangChain chat "
                    "model, a CrewAI LLM or a ChatLLM")
