"""Structured LLM output: schema-validated, with one bounded repair retry (hard rule 6).

Flow for :func:`invoke_structured`:

1. Build messages: the agent's system prompt + JSON format instructions (the schema).
2. Call the model. If it supports provider-native structured output (LangChain
   ``with_structured_output``) that is used for the first attempt; otherwise the reply
   text is parsed: strip code fences → first balanced JSON object →
   ``Schema.model_validate_json``.
3. Run the agent's ``postprocess`` (semantic checks: ids exist in tool results, dates
   consistent…). Raising ``ValueError`` there counts as invalid output.
4. On failure, send the errors back once ("repair") and validate again.
5. Still invalid → :class:`StructuredOutputError` (fail loudly). Transport failures
   raise :class:`LLMCallError` immediately; they are never "repaired".

Every model call publishes an ``llm_call`` :class:`TraceEvent` with token usage.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel, ValidationError

from dubai_planner.config import get_settings
from dubai_planner.llm.base import (
    ChatLLM,
    Completion,
    LLMMessage,
    NativeStructuredLLM,
    as_chat_llm,
)
from dubai_planner.llm.errors import StructuredOutputError
from dubai_planner.models.trace import TraceEventType
from dubai_planner.tracing import emit

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
#: Markers the mock provider reads to know which agent/schema is being asked for.
SCHEMA_MARKER = "SCHEMA: {name}"
AGENT_MARKER = "AGENT: {name}"
MAX_ERRORS_SHOWN = 8


@dataclass
class StructuredResult(Generic[R]):
    value: R
    calls: int
    tokens_in: int
    tokens_out: int
    repaired: bool
    errors: list[str] = field(default_factory=list)


def extract_json(text: str) -> str:
    """Return the first balanced JSON object in ``text`` (code fences are stripped)."""
    candidate = text.strip()
    fence = _FENCE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    start = candidate.find("{")
    if start < 0:
        raise ValueError("no JSON object found in the reply")
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : i + 1]
    raise ValueError("unterminated JSON object in the reply")


def parse_output(text: str, schema: type[T]) -> T:
    return schema.model_validate_json(extract_json(text))


def format_errors(exc: Exception) -> str:
    """Compact, model-readable error list (no input echo, which could be large)."""
    if isinstance(exc, ValidationError):
        lines = []
        for err in exc.errors(include_url=False, include_input=False)[:MAX_ERRORS_SHOWN]:
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            lines.append(f"- {loc}: {err['msg']}")
        more = exc.error_count() - MAX_ERRORS_SHOWN
        if more > 0:
            lines.append(f"- … and {more} more")
        return "\n".join(lines)
    return f"- {exc}"


def format_instructions(schema: type[BaseModel], agent: str) -> str:
    schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    return (
        f"{AGENT_MARKER.format(name=agent)}\n{SCHEMA_MARKER.format(name=schema.__name__)}\n"
        "Respond with ONE JSON object only: no prose, no markdown fences, no comments. "
        "It must validate against this JSON schema:\n"
        f"{schema_json}"
    )


def repair_message(errors: str) -> str:
    return (
        "Your previous reply was not valid. Problems:\n"
        f"{errors}\n"
        "Return the corrected JSON object only, following the same schema."
    )


def build_messages(
    system: str, user: str, schema: type[BaseModel], agent: str,
    history: Sequence[LLMMessage] = (),
) -> list[LLMMessage]:
    return [
        LLMMessage("system", f"{system.strip()}\n\n{format_instructions(schema, agent)}"),
        *history,
        LLMMessage("user", user),
    ]


PROMPT_PREVIEW_CHARS = 1500
RESPONSE_PREVIEW_CHARS = 1500


def prompt_preview(messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
    """Each message's role, size and opening text (the inspector's "prompt" view).

    Prompts hold only our instructions and the traveller's own words, never keys.
    """
    return [{"role": m.role, "chars": len(m.content),
             "preview": m.content[:PROMPT_PREVIEW_CHARS]} for m in messages]


def _trace_call(agent: str, schema: str, completion: Completion, attempt: int, started: datetime,
                latency_s: float, ok: bool, native: bool,
                messages: Sequence[LLMMessage] = (), error: str | None = None) -> None:
    emit(
        agent,
        TraceEventType.LLM_CALL,
        started_at=started,
        ended_at=datetime.now(UTC),
        tokens_in=completion.tokens_in,
        tokens_out=completion.tokens_out,
        payload={
            "schema": schema,
            "attempt": attempt,
            "model": completion.model,
            "latency_s": round(latency_s, 3),
            "valid": ok,
            "native": native,
            "prompt": prompt_preview(messages),
            "response_preview": (completion.text or "")[:RESPONSE_PREVIEW_CHARS],
            **({"validation_error": error[:500]} if error else {}),
        },
    )


def invoke_structured(
    llm: ChatLLM | Any,
    messages: Sequence[LLMMessage],
    schema: type[T],
    *,
    agent: str,
    postprocess: Callable[[T], R] | None = None,
    repair_attempts: int | None = None,
    use_native: bool | None = None,
) -> StructuredResult[R]:
    """Call ``llm`` until its output validates as ``schema`` (and ``postprocess`` accepts it).

    ``messages`` should come from :func:`build_messages` so the schema is in the prompt.
    ``postprocess`` turns the parsed model into the agent's result and raises
    ``ValueError`` for semantic problems, which triggers the repair retry.
    """
    chat = as_chat_llm(llm)
    repairs = get_settings().llm_repair_attempts if repair_attempts is None else repair_attempts
    native = isinstance(chat, NativeStructuredLLM) and bool(
        getattr(chat, "native_structured", False) if use_native is None else use_native
    )
    convo = list(messages)
    errors: list[str] = []
    tokens_in = tokens_out = 0
    raw = ""
    for attempt in range(repairs + 1):
        started, t0 = datetime.now(UTC), time.perf_counter()
        parsed: T | None = None
        problem: Exception | None = None
        use_native_now = native and attempt == 0
        if use_native_now:
            result = cast(NativeStructuredLLM, chat).complete_structured(convo, schema)
            completion = result.completion
            if result.parsed is not None:
                parsed = cast(T, result.parsed)
                raw = parsed.model_dump_json()
            else:
                # The native parser failed or the provider rejected the format, but the
                # model may still have written valid JSON as text: validate it the same way.
                raw = completion.text
                try:
                    parsed = parse_output(raw, schema)
                except (ValidationError, ValueError):
                    problem = ValueError(result.error or "no structured output returned")
        else:
            completion = chat.complete(convo)
            raw = completion.text
            try:
                parsed = parse_output(raw, schema)
            except (ValidationError, ValueError) as exc:
                problem = exc
        value: R | None = None
        if parsed is not None:
            try:
                value = postprocess(parsed) if postprocess else cast(R, parsed)
            except ValueError as exc:  # includes pydantic.ValidationError
                problem = exc
        tokens_in += completion.tokens_in
        tokens_out += completion.tokens_out
        _trace_call(agent, schema.__name__, completion, attempt, started,
                    time.perf_counter() - t0, problem is None, use_native_now, convo,
                    None if problem is None else format_errors(problem))
        if problem is None:
            return StructuredResult(
                value=cast(R, value), calls=attempt + 1, tokens_in=tokens_in,
                tokens_out=tokens_out, repaired=attempt > 0, errors=errors,
            )
        message = format_errors(problem)
        errors.append(message)
        convo = [*convo, LLMMessage("assistant", raw or "<empty>"),
                 LLMMessage("user", repair_message(message))]
    raise StructuredOutputError(agent, schema.__name__, errors, raw)
