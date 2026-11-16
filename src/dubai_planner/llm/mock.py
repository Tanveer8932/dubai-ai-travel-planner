"""Deterministic, offline mock LLM (``LLM_PROVIDER=mock``).

One engine, :class:`MockLLM`, answers every request; thin wrappers expose it as a
LangChain chat model (:mod:`dubai_planner.llm.langchain_mock`) and as a CrewAI LLM
(:mod:`dubai_planner.llm.crewai_mock`), so tests and offline demos run through the
same code paths as real providers.

The engine reads the ``AGENT:`` / ``SCHEMA:`` markers that
:func:`~dubai_planner.llm.structured.format_instructions` puts in the system prompt,
and the JSON ``<context>`` block agents put in the user prompt. Resolution order:

1. **Scripted** replies queued per agent with :meth:`MockLLM.script` (raw strings, so
   tests can feed malformed JSON, unknown ids, over-budget plans…).
2. **Fixture files** ``<fixtures_dir>/<agent>.json`` mapping schema name → reply object.
3. **Generators** registered per schema by the agent modules. They are deterministic
   baselines computed from the context block (e.g. "pick the top-ranked candidates").
"""

from __future__ import annotations

import json
import re
import threading
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dubai_planner.llm.base import Completion, LLMMessage, estimate_tokens
from dubai_planner.llm.errors import MockLLMError

_AGENT = re.compile(r"^AGENT:\s*([\w.-]+)\s*$", re.MULTILINE)
_SCHEMA = re.compile(r"^SCHEMA:\s*([\w.-]+)\s*$", re.MULTILINE)
_CONTEXT = re.compile(r"<context>\s*(.*?)\s*</context>", re.DOTALL)
_USER_REQUEST = re.compile(r"<user_request>\s*(.*?)\s*</user_request>", re.DOTALL)


@dataclass(frozen=True)
class MockRequest:
    agent: str
    schema: str
    context: dict[str, Any] | None
    user_text: str
    messages: tuple[LLMMessage, ...]


Generator = Callable[[MockRequest], BaseModel | dict[str, Any] | str]
_GENERATORS: dict[str, Generator] = {}


def register_generator(schema_name: str, fn: Generator) -> Generator:
    """Register the deterministic mock reply for ``schema_name`` (called by agent modules)."""
    _GENERATORS[schema_name] = fn
    return fn


def _ensure_generators() -> None:
    if not _GENERATORS:
        import dubai_planner.agents  # noqa: F401  (agent modules register generators)


def parse_request(messages: Sequence[LLMMessage], schema_hint: str | None = None) -> MockRequest:
    system = "\n".join(m.content for m in messages if m.role == "system")
    users = [m.content for m in messages if m.role == "user"]
    agent = _AGENT.search(system)
    schema = _SCHEMA.search(system)
    context: dict[str, Any] | None = None
    user_text = ""
    for text in users:  # the first user turn carries the task; later ones are repairs
        if context is None and (match := _CONTEXT.search(text)):
            try:
                loaded = json.loads(match.group(1))
            except json.JSONDecodeError:
                loaded = None
            context = loaded if isinstance(loaded, dict) else None
        if not user_text and (req := _USER_REQUEST.search(text)):
            user_text = req.group(1)
    return MockRequest(
        agent=agent.group(1) if agent else "unknown",
        schema=schema.group(1) if schema else (schema_hint or "unknown"),
        context=context,
        user_text=user_text or (users[0] if users else ""),
        messages=tuple(messages),
    )


class MockLLM:
    """Fixture/script/generator-backed ``ChatLLM``. Thread-safe (parallel branches)."""

    def __init__(self, *, fixtures_dir: Path | None = None) -> None:
        self.fixtures_dir = fixtures_dir
        self._scripts: dict[str, deque[str]] = defaultdict(deque)
        self._lock = threading.Lock()
        self.calls: list[MockRequest] = []

    @property
    def name(self) -> str:
        return "mock"

    def script(self, agent: str, *replies: str | BaseModel | dict[str, Any]) -> MockLLM:
        """Queue raw replies for ``agent`` (consumed in order before anything else)."""
        with self._lock:
            for reply in replies:
                self._scripts[agent].append(_serialise(reply))
        return self

    def pending(self, agent: str) -> int:
        with self._lock:
            return len(self._scripts[agent])

    def _fixture(self, req: MockRequest) -> str | None:
        if self.fixtures_dir is None:
            return None
        path = self.fixtures_dir / f"{req.agent}.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and req.schema in data:
            return _serialise(data[req.schema])
        return None

    def respond(self, req: MockRequest) -> str:
        with self._lock:
            self.calls.append(req)
            queue = self._scripts.get(req.agent)
            if queue:
                return queue.popleft()
        fixture = self._fixture(req)
        if fixture is not None:
            return fixture
        _ensure_generators()
        generator = _GENERATORS.get(req.schema)
        if generator is None:
            raise MockLLMError(
                f"mock LLM has no script, fixture or generator for agent={req.agent!r} "
                f"schema={req.schema!r}"
            )
        return _serialise(generator(req))

    def complete(self, messages: Sequence[LLMMessage], *, schema_hint: str | None = None
                 ) -> Completion:
        req = parse_request(messages, schema_hint)
        text = self.respond(req)
        prompt = "".join(m.content for m in messages)
        return Completion(text=text, tokens_in=estimate_tokens(prompt),
                          tokens_out=estimate_tokens(text), model="mock")


def _serialise(reply: str | BaseModel | dict[str, Any]) -> str:
    if isinstance(reply, str):
        return reply
    if isinstance(reply, BaseModel):
        return reply.model_dump_json()
    return json.dumps(reply, default=str)


_default: MockLLM | None = None


def default_mock() -> MockLLM:
    """Process-wide engine used by the factory when ``LLM_PROVIDER=mock``."""
    global _default
    if _default is None:
        _default = MockLLM()
    return _default
