"""The mock engine exposed as a CrewAI LLM (a ``BaseLLM`` subclass, no LiteLLM needed).

It answers two kinds of request:

- **Agent-core calls** (the prompt carries our ``AGENT:``/``SCHEMA:`` markers, sent by
  :class:`~dubai_planner.llm.base.CrewAIChatLLM`) and structured-output requests (a
  ``response_model``): the shared :class:`MockLLM` engine replies exactly as in
  LangGraph mode.
- **CrewAI agent reasoning** (CrewAI's ReAct prompt: role, tool list, current task):
  a deterministic, rule-following agent. It calls the tool the task names
  (``… the `research_attractions` tool …``), a manager delegates to the role named on
  the task's ``Responsible specialist:`` line, and once a tool result (observation)
  is in the conversation it returns it as the final answer. Tests can script raw
  replies per role with ``engine.script("crewai:<role>", ...)``.

Reasoning calls emit CrewAI's own ``LLMCallStarted/CompletedEvent`` (with estimated
usage) like the real provider clients, so trace listeners see the same events offline.
"""

from __future__ import annotations

import json
import re
from typing import Any, cast

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM, llm_call_context
from pydantic import BaseModel, PrivateAttr

from dubai_planner.llm.base import LLMMessage, Role, estimate_tokens
from dubai_planner.llm.mock import MockLLM, default_mock

_ROLES: dict[str, Role] = {"system": "system", "user": "user", "assistant": "assistant"}
_MARKER = re.compile(r"^AGENT:\s*\S+", re.MULTILINE)
_TOOL_NAME = re.compile(r"^Tool Name:\s*(.+?)\s*$", re.MULTILINE)
_NAMED_TOOL = re.compile(r"`([a-z0-9_]+)` tool")
_SPECIALIST = re.compile(r"Responsible specialist:\s*(.+?)\s*$", re.MULTILINE)
_YOU_ARE = re.compile(r"^You are (.+?)\.", re.MULTILINE)
_CURRENT_TASK = re.compile(r"Current Task:\s*(.*?)(?:\n\nThis is the expected|\Z)", re.DOTALL)
DELEGATE_TOOL = "delegate_work_to_coworker"


def _text(messages: list[dict[str, Any]], role: str) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == role)


def _observation(messages: list[dict[str, Any]]) -> str | None:
    """The latest tool result in a ReAct conversation, if any."""
    for message in reversed(messages):
        content = str(message.get("content") or "")
        if message.get("role") == "tool":
            return content.strip()
        if message.get("role") == "assistant" and "Observation:" in content:
            return content.rsplit("Observation:", 1)[1].strip()
    return None


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
        msgs: list[dict[str, Any]] = (
            [{"role": "user", "content": messages}] if isinstance(messages, str)
            else [dict(m) for m in messages])
        if response_model is not None or any(
                _MARKER.search(str(m.get("content") or "")) for m in msgs):
            converted = [LLMMessage(_ROLES.get(str(m.get("role", "user")), "user"),
                                    str(m.get("content"))) for m in msgs]
            hint = response_model.__name__ if response_model is not None else None
            return self._engine.complete(converted, schema_hint=hint).text
        with llm_call_context():
            self._emit_call_started_event(messages=cast(list[Any], msgs), from_task=from_task,
                                          from_agent=from_agent)
            reply = self._react(msgs, from_agent)
            prompt = "".join(str(m.get("content") or "") for m in msgs)
            usage = {"prompt_tokens": estimate_tokens(prompt),
                     "completion_tokens": estimate_tokens(reply)}
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            self._emit_call_completed_event(reply, LLMCallType.LLM_CALL, from_task=from_task,
                                            from_agent=from_agent, messages=cast(list[Any], msgs),
                                            usage=usage)
        return reply

    def _react(self, msgs: list[dict[str, Any]], from_agent: Any | None) -> str:
        system, user = _text(msgs, "system"), _text(msgs, "user")
        match = _YOU_ARE.search(system)
        role = str(getattr(from_agent, "role", "") or (match.group(1) if match else "agent"))
        self._engine.log_react(role)
        scripted = self._engine.take_script(f"crewai:{role}")
        if scripted is not None:
            return scripted
        observed = _observation(msgs)
        if observed is not None:
            return f"Thought: I have the result.\nFinal Answer: {observed}"
        tools = _TOOL_NAME.findall(system)
        if DELEGATE_TOOL in tools:
            coworker = _SPECIALIST.search(user)
            task = _CURRENT_TASK.search(user)
            args = {"task": (task.group(1) if task else user).strip()[:4000], "context": "",
                    "coworker": coworker.group(1) if coworker else ""}
            return (f"Thought: The specialist for this task should do it.\n"
                    f"Action: {DELEGATE_TOOL}\nAction Input: {json.dumps(args)}")
        named = [t for t in _NAMED_TOOL.findall(user) if t in tools]
        tool = named[0] if named else (tools[0] if tools else None)
        if tool is None:
            return "Thought: I have no tool for this.\nFinal Answer: I could not do this task."
        return f"Thought: I will use my tool.\nAction: {tool}\nAction Input: {{}}"

    def supports_function_calling(self) -> bool:
        return False
