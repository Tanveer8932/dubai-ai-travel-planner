"""Structured output: JSON extraction, validation, one repair retry, fail loudly, tracing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import BaseModel, Field

from dubai_planner.llm import LLMCallError, StructuredOutputError, invoke_structured
from dubai_planner.llm.base import (
    Completion,
    LangChainChatLLM,
    LLMMessage,
    NativeResult,
    as_chat_llm,
)
from dubai_planner.llm.mock import MockLLM
from dubai_planner.llm.structured import build_messages, extract_json, format_instructions
from dubai_planner.models.trace import TraceEventType
from dubai_planner.tracing import capture, run_context


class Answer(BaseModel):
    city: str
    days: int = Field(ge=1, le=30)


def _messages() -> list[LLMMessage]:
    return build_messages("You extract trips.", "Plan Dubai.", Answer, agent="tester")


@pytest.mark.parametrize(
    "text",
    [
        '{"city": "Dubai", "days": 3}',
        '```json\n{"city": "Dubai", "days": 3}\n```',
        'Sure! Here it is: {"city": "Dubai", "days": 3} Hope that helps.',
        '```\n{"city": "Dubai", "days": 3}\n```',
        '{"city": "Du}bai {x}", "days": 3}',
    ],
)
def test_extract_json_variants(text: str) -> None:
    assert Answer.model_validate_json(extract_json(text)).days == 3


@pytest.mark.parametrize("text", ["no json here", '{"city": "Dubai", "days": 3'])
def test_extract_json_rejects(text: str) -> None:
    with pytest.raises(ValueError, match="JSON"):
        extract_json(text)


def test_format_instructions_contain_markers_and_schema() -> None:
    text = format_instructions(Answer, "tester")
    assert "AGENT: tester" in text and "SCHEMA: Answer" in text and '"days"' in text


def test_valid_first_time(mock_llm: MockLLM) -> None:
    mock_llm.script("tester", '{"city": "Dubai", "days": 4}')
    result = invoke_structured(mock_llm, _messages(), Answer, agent="tester")
    assert result.value == Answer(city="Dubai", days=4)
    assert (result.calls, result.repaired) == (1, False)


@pytest.mark.parametrize(
    "bad",
    [
        "I think you should visit the Burj Khalifa!",  # prose, no JSON
        '{"city": "Dubai"}',  # missing field
        '{"city": "Dubai", "days": 99}',  # out of range
        '{"city": "Dubai", "days": "many"}',  # wrong type
        '{"city": "Dubai", "days": 3',  # truncated
        "",  # empty reply
    ],
)
def test_malformed_output_is_repaired_once(mock_llm: MockLLM, bad: str) -> None:
    mock_llm.script("tester", bad, '{"city": "Dubai", "days": 3}')
    result = invoke_structured(mock_llm, _messages(), Answer, agent="tester")
    assert result.value.days == 3
    assert (result.calls, result.repaired, len(result.errors)) == (2, True, 1)
    repair_turn = mock_llm.calls[-1].messages[-1]
    assert repair_turn.role == "user" and "not valid" in repair_turn.content
    assert mock_llm.calls[-1].messages[-2].role == "assistant"


def test_still_invalid_after_repair_fails_loudly(mock_llm: MockLLM) -> None:
    mock_llm.script("tester", "nope", '{"city": 1}')
    with pytest.raises(StructuredOutputError) as info:
        invoke_structured(mock_llm, _messages(), Answer, agent="tester")
    err = info.value
    assert err.agent == "tester" and err.schema == "Answer" and len(err.errors) == 2
    assert mock_llm.pending("tester") == 0  # exactly two calls: bounded


def test_repair_attempts_zero_means_single_call(mock_llm: MockLLM) -> None:
    mock_llm.script("tester", "nope", '{"city": "Dubai", "days": 3}')
    with pytest.raises(StructuredOutputError):
        invoke_structured(mock_llm, _messages(), Answer, agent="tester", repair_attempts=0)
    assert mock_llm.pending("tester") == 1


def test_repair_attempts_follow_settings(mock_llm: MockLLM, monkeypatch: pytest.MonkeyPatch
                                         ) -> None:
    from dubai_planner.config import get_settings

    monkeypatch.setenv("LLM_REPAIR_ATTEMPTS", "2")
    get_settings.cache_clear()
    mock_llm.script("tester", "x", "y", '{"city": "Dubai", "days": 3}')
    assert invoke_structured(mock_llm, _messages(), Answer, agent="tester").calls == 3


def test_postprocess_value_error_triggers_repair(mock_llm: MockLLM) -> None:
    mock_llm.script("tester", '{"city": "Paris", "days": 3}', '{"city": "Dubai", "days": 3}')

    def only_dubai(a: Answer) -> str:
        if a.city != "Dubai":
            raise ValueError("city must be Dubai")
        return a.city.upper()

    result = invoke_structured(mock_llm, _messages(), Answer, agent="tester",
                               postprocess=only_dubai)
    assert result.value == "DUBAI" and result.repaired
    assert "city must be Dubai" in mock_llm.calls[-1].messages[-1].content


class _Boom:
    name = "boom"

    def complete(self, messages: Sequence[LLMMessage]) -> Completion:
        raise LLMCallError("connection refused")


def test_transport_errors_are_not_repaired() -> None:
    with pytest.raises(LLMCallError, match="refused"):
        invoke_structured(_Boom(), _messages(), Answer, agent="tester")


def test_each_call_traced_with_tokens(mock_llm: MockLLM) -> None:
    mock_llm.script("tester", "bad", '{"city": "Dubai", "days": 2}')
    with run_context("run-1", mode="langgraph"), capture() as events:
        invoke_structured(mock_llm, _messages(), Answer, agent="tester")
    calls = [e for e in events if e.event is TraceEventType.LLM_CALL]
    assert [c.payload["valid"] for c in calls] == [False, True]
    assert all(c.run_id == "run-1" and c.mode == "langgraph" for c in calls)
    assert all(c.tokens_in > 0 and c.tokens_out > 0 for c in calls)
    assert all(c.ended_at is not None for c in calls)


# ------------------------------------------------------------------ native structured output


class _NativeModel:
    """Stands in for a LangChain chat model that supports with_structured_output."""

    model = "native-test"

    def __init__(self, outputs: list[dict[str, Any] | Exception], text_reply: str = "") -> None:
        self.outputs = outputs
        self.text_reply = text_reply
        self.text_calls = 0
        self.methods: list[str | None] = []

    def with_structured_output(self, schema: type, include_raw: bool = False,
                               method: str | None = None) -> Any:
        outer = self
        self.methods.append(method)

        class _Runnable:
            def invoke(self, _messages: Any) -> dict[str, Any]:
                out = outer.outputs.pop(0)
                if isinstance(out, Exception):
                    raise out
                return out

        return _Runnable()

    def invoke(self, _messages: Any) -> Any:
        from langchain_core.messages import AIMessage

        self.text_calls += 1
        return AIMessage(content=self.text_reply)


def test_native_structured_used_first() -> None:
    from langchain_core.messages import AIMessage

    raw = AIMessage(content="", usage_metadata={"input_tokens": 11, "output_tokens": 7,
                                                "total_tokens": 18})
    model = _NativeModel([{"raw": raw, "parsed": Answer(city="Dubai", days=5),
                           "parsing_error": None}])
    llm = LangChainChatLLM(model, native_structured=True)
    result = invoke_structured(llm, _messages(), Answer, agent="tester")
    assert result.value.days == 5 and result.calls == 1
    assert (result.tokens_in, result.tokens_out) == (11, 7)
    assert model.text_calls == 0


def test_native_parse_failure_repairs_via_json() -> None:
    from langchain_core.messages import AIMessage

    model = _NativeModel([{"raw": AIMessage(content="oops"), "parsed": None,
                           "parsing_error": "bad json"}],
                         text_reply='{"city": "Dubai", "days": 2}')
    result = invoke_structured(LangChainChatLLM(model, native_structured=True), _messages(),
                               Answer, agent="tester")
    assert result.value.days == 2 and result.repaired and model.text_calls == 1


class _ProviderError(Exception):
    """Shape of an OpenAI-compatible SDK error (groq/openai ``APIStatusError.body``)."""

    def __init__(self, body: Any) -> None:
        super().__init__("Error code: 400")
        self.body = body


def _rejected(code: str, generation: str) -> _ProviderError:
    return _ProviderError({"error": {"message": "Tool choice is required", "code": code,
                                     "type": "invalid_request_error",
                                     "failed_generation": generation}})


def test_rejected_native_reply_with_valid_json_is_accepted() -> None:
    """Groq 400 tool_use_failed: the model answered in plain JSON instead of a tool call."""
    model = _NativeModel([_rejected("tool_use_failed", '{"city": "Dubai", "days": 4}')])
    result = invoke_structured(LangChainChatLLM(model, native_structured=True), _messages(),
                               Answer, agent="tester")
    assert result.value.days == 4 and result.calls == 1 and model.text_calls == 0


def test_rejected_native_reply_with_bad_json_is_repaired() -> None:
    model = _NativeModel([_rejected("json_validate_failed", '{"city": "Dubai", "days": 99}')],
                         text_reply='{"city": "Dubai", "days": 3}')
    result = invoke_structured(LangChainChatLLM(model, native_structured=True), _messages(),
                               Answer, agent="tester")
    assert result.value.days == 3 and result.repaired and model.text_calls == 1
    assert "rejected" in result.errors[0]


@pytest.mark.parametrize("body", [
    {"error": {"code": "rate_limit_exceeded", "message": "429"}},
    {"error": "plain string"},
    None,
])
def test_other_provider_errors_stay_transport_errors(body: Any) -> None:
    model = _NativeModel([_ProviderError(body)])
    with pytest.raises(LLMCallError, match="native-test"):
        invoke_structured(LangChainChatLLM(model, native_structured=True), _messages(),
                          Answer, agent="tester")


def test_structured_method_is_passed_to_the_integration() -> None:
    from dubai_planner.llm.base import NATIVE_STRUCTURED_METHODS

    assert NATIVE_STRUCTURED_METHODS["langchain_groq"] == "json_schema"
    model = _NativeModel([{"raw": None, "parsed": Answer(city="Dubai", days=1),
                           "parsing_error": None}])
    llm = LangChainChatLLM(model, native_structured=True, structured_method="json_schema")
    invoke_structured(llm, _messages(), Answer, agent="tester")
    plain = _NativeModel([{"raw": None, "parsed": Answer(city="Dubai", days=1),
                           "parsing_error": None}])
    invoke_structured(LangChainChatLLM(plain, native_structured=True), _messages(), Answer,
                      agent="tester")
    assert model.methods == ["json_schema"] and plain.methods == [None]


def test_native_result_dataclass() -> None:
    res = NativeResult(parsed=None, completion=Completion(text="x"), error="e")
    assert res.error == "e"


def test_as_chat_llm_rejects_unknown_objects() -> None:
    with pytest.raises(TypeError, match="unsupported LLM"):
        as_chat_llm(object())
