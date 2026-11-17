"""Persistence records returned by ``storage.repository`` (rows as typed models)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field

from dubai_planner.models.common import AED, StrictModel
from dubai_planner.models.itinerary import Itinerary, PlannerMode
from dubai_planner.models.request import TripRequest
from dubai_planner.models.trace import utcnow


class Session(StrictModel):
    id: str
    title: str = "New trip"
    mode: PlannerMode = PlannerMode.LANGGRAPH
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class StoredMessage(StrictModel):
    id: int
    session_id: str
    role: str
    content: str
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class ItineraryRecord(StrictModel):
    """A saved itinerary version with its request and review status."""

    id: UUID
    session_id: str
    version: int
    parent_id: UUID | None
    mode: PlannerMode
    run_id: str | None = None
    total_aed: AED
    approved: bool
    created_at: datetime
    request: TripRequest | None
    itinerary: Itinerary


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class RunRecord(StrictModel):
    id: str
    session_id: str | None
    mode: PlannerMode
    status: RunStatus = RunStatus.RUNNING
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    duration_s: float | None = None
    llm_calls: int = 0
    tokens: int = 0
    error: str | None = None
    # Phase 6 (schema v2): aggregated LLM usage and what a fair comparison needs.
    tokens_in: int = 0
    tokens_out: int = 0
    llm_latency_s: float = 0.0
    iterations: int | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class EvalRun(StrictModel):
    id: str
    created_at: datetime = Field(default_factory=utcnow)
    config: dict[str, Any] = Field(default_factory=dict)


class EvalResult(StrictModel):
    id: int | None = None
    eval_run_id: str
    scenario_id: str
    mode: PlannerMode
    seed: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
