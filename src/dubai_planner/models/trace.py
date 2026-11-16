"""Observability and conversation models shared by both orchestration modes."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from dubai_planner.models.common import StrictModel


def utcnow() -> datetime:
    return datetime.now(UTC)


class TraceEventType(StrEnum):
    START = "start"
    END = "end"
    ERROR = "error"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    DELEGATION = "delegation"
    GUARDRAIL = "guardrail"
    INFO = "info"


class TraceEvent(StrictModel):
    run_id: str
    mode: str
    agent: str
    event: TraceEventType
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    iteration: int = Field(default=0, ge=0)
    payload_ref: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ordered(self) -> TraceEvent:
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("ended_at must not be before started_at")
        return self

    @property
    def duration_s(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ChatMessage(StrictModel):
    role: ChatRole
    content: str = Field(max_length=20_000)
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
