"""The mock provider: deterministic, offline, usable through LangChain and CrewAI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from dubai_planner.llm import MockLLMError, as_chat_llm, invoke_structured
from dubai_planner.llm.base import CrewAIChatLLM, LangChainChatLLM, LLMMessage
from dubai_planner.llm.mock import MockLLM, parse_request
from dubai_planner.llm.structured import build_messages


class Answer(BaseModel):
    city: str
    days: int


def _msgs(user: str = "hi") -> list[LLMMessage]:
    return build_messages("system prompt", user, Answer, agent="tester")


def test_parse_request_reads_markers_and_context() -> None:
    msgs = build_messages("sys", '<context>\n{"a": 1}\n</context>\n<user_request>\nhello\n'
                          "</user_request>", Answer, agent="dining")
    req = parse_request(msgs)
    assert (req.agent, req.schema, req.context, req.user_text) == ("dining", "Answer",
                                                                    {"a": 1}, "hello")


def test_parse_request_tolerates_bad_context() -> None:
    req = parse_request(build_messages("s", "<context>{not json</context>", Answer, agent="x"))
    assert req.context is None


def test_script_takes_priority_and_is_consumed_in_order(mock_llm: MockLLM) -> None:
    mock_llm.script("tester", '{"city": "A", "days": 1}', Answer(city="B", days=2))
    first = mock_llm.complete(_msgs()).text
    second = mock_llm.complete(_msgs()).text
    assert json.loads(first)["city"] == "A" and json.loads(second)["city"] == "B"
    assert mock_llm.pending("tester") == 0


def test_fixture_files(tmp_path: Path) -> None:
    (tmp_path / "tester.json").write_text(json.dumps({"Answer": {"city": "Fix", "days": 3}}),
                                          encoding="utf-8")
    engine = MockLLM(fixtures_dir=tmp_path)
    assert invoke_structured(engine, _msgs(), Answer, agent="tester").value.city == "Fix"


def test_missing_reply_is_a_clear_error(mock_llm: MockLLM) -> None:
    with pytest.raises(MockLLMError, match="agent='tester'"):
        mock_llm.complete(_msgs())


def test_records_calls_and_estimates_tokens(mock_llm: MockLLM) -> None:
    mock_llm.script("tester", '{"city": "A", "days": 1}')
    completion = mock_llm.complete(_msgs("x" * 400))
    assert completion.tokens_in > 100 and completion.tokens_out > 0
    assert mock_llm.calls[0].agent == "tester"


def test_langchain_mock_chat_model(mock_llm: MockLLM) -> None:
    from langchain_core.messages import HumanMessage, SystemMessage

    from dubai_planner.llm.langchain_mock import MockChatModel

    mock_llm.script("tester", '{"city": "LC", "days": 2}')
    model = MockChatModel(engine=mock_llm)
    msgs = _msgs()
    reply = model.invoke([SystemMessage(content=msgs[0].content), HumanMessage(content="go")])
    assert json.loads(str(reply.content))["city"] == "LC"
    assert reply.usage_metadata and reply.usage_metadata["input_tokens"] > 0
    wrapped = as_chat_llm(model)
    assert isinstance(wrapped, LangChainChatLLM) and not wrapped.native_structured


def test_langchain_mock_through_invoke_structured(mock_llm: MockLLM) -> None:
    from dubai_planner.llm.langchain_mock import MockChatModel

    mock_llm.script("tester", "garbage", '{"city": "LC", "days": 2}')
    result = invoke_structured(MockChatModel(engine=mock_llm), _msgs(), Answer, agent="tester")
    assert result.value.city == "LC" and result.repaired


def test_crewai_mock_llm(mock_llm: MockLLM) -> None:
    from crewai.llms.base_llm import BaseLLM

    from dubai_planner.llm.crewai_mock import MockCrewAILLM

    llm = MockCrewAILLM(engine=mock_llm)
    assert isinstance(llm, BaseLLM) and llm.model == "mock"
    mock_llm.script("tester", '{"city": "Crew", "days": 1}')
    msgs = [{"role": m.role, "content": m.content} for m in _msgs()]
    assert json.loads(llm.call(msgs))["city"] == "Crew"
    assert isinstance(as_chat_llm(llm), CrewAIChatLLM)
    assert not llm.supports_function_calling()


def test_crewai_mock_uses_response_model_as_schema_hint(mock_llm: MockLLM, tmp_path: Path
                                                        ) -> None:
    from dubai_planner.llm.crewai_mock import MockCrewAILLM

    (tmp_path / "unknown.json").write_text(json.dumps({"Answer": {"city": "Hint", "days": 1}}),
                                           encoding="utf-8")
    llm = MockCrewAILLM(engine=MockLLM(fixtures_dir=tmp_path))
    assert json.loads(llm.call("plain prompt", response_model=Answer))["city"] == "Hint"


def test_crewai_agent_accepts_mock_llm(mock_llm: MockLLM) -> None:
    """Phase 4 compatibility: a CrewAI Agent can be built on the mock LLM."""
    from crewai import Agent

    from dubai_planner.llm.crewai_mock import MockCrewAILLM

    agent = Agent(role="Dining Specialist", goal="Pick restaurants", backstory="Local foodie",
                  llm=MockCrewAILLM(engine=mock_llm), allow_delegation=False)
    assert agent.llm is not None and agent.llm.model == "mock"
