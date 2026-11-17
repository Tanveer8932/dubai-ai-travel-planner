"""Safe, JSON-friendly trace payloads for step inputs and outputs (Phase 6 inspector).

Orchestration spans (LangGraph nodes, CrewAI stage tools) attach

- ``inputs``: a compact *summary* of what the step read (scalars, counts, type names;
  long text cut), never the full shared state;
- ``output``: the step's validated output model as JSON (``output_schema`` names its
  type, so the inspector can validate it again), or only a summary when it is larger
  than :data:`MAX_OUTPUT_CHARS`.

Only domain data goes in: these helpers never see settings or provider objects, so
no key, header or provider metadata can end up in a trace.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, TypeAdapter

MAX_OUTPUT_CHARS = 150_000
MAX_TEXT = 200
_ANY: TypeAdapter[Any] = TypeAdapter(Any)


def schema_name(value: Any) -> str:
    """``Itinerary``, ``list[RuleViolation]``, ``dict`` … (for re-validation)."""
    if isinstance(value, BaseModel):
        return type(value).__name__
    if isinstance(value, list | tuple):
        inner = {schema_name(v) for v in value}
        return f"list[{inner.pop() if len(inner) == 1 else 'Any'}]"
    return type(value).__name__


def summarize(value: Any, depth: int = 0) -> Any:
    """A small, safe view of ``value``: scalars kept, collections counted, text cut."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + "…"
    if isinstance(value, Decimal | date | datetime | time | Enum):
        return str(value.value if isinstance(value, Enum) else value)
    if isinstance(value, list | tuple | set | frozenset):
        items = list(value)
        if depth >= 1 or len(items) > 8 or any(isinstance(v, BaseModel) for v in items):
            return f"{len(items)} item(s)"
        return [summarize(v, depth + 1) for v in items]
    if isinstance(value, BaseModel):
        if depth >= 2:
            return type(value).__name__
        return {k: summarize(getattr(value, k), depth + 1) for k in type(value).model_fields}
    if isinstance(value, dict):
        if depth >= 2:
            return f"{len(value)} key(s)"
        return {str(k): summarize(v, depth + 1) for k, v in list(value.items())[:20]}
    return type(value).__name__


def output_payload(value: Any) -> dict[str, Any]:
    """``output_schema`` + ``output`` (full JSON) or ``output_summary`` when too large."""
    data = _ANY.dump_python(value, mode="json")
    size = len(json.dumps(data, default=str))
    out: dict[str, Any] = {"output_schema": schema_name(value), "output_chars": size}
    if size <= MAX_OUTPUT_CHARS:
        out["output"] = data
    else:
        out["output_truncated"] = True
        out["output_summary"] = summarize(value)
    return out


def record_io(span: dict[str, Any], *, inputs: dict[str, Any] | None = None,
              output: Any = None) -> None:
    """Attach input summary and output to an ``agent_span`` result dict."""
    if inputs is not None:
        span["inputs"] = {k: summarize(v) for k, v in inputs.items()}
    if output is not None:
        span.update(output_payload(output))
