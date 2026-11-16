"""Result contract of a planning run, shared by every orchestration mode.

LangGraph mode (Phase 3) produces these; CrewAI mode (Phase 4) must return the same
types so the UI and the evaluation runner never branch on the framework.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import Field

from dubai_planner.models.common import StrictModel
from dubai_planner.models.itinerary import Itinerary, PlannerMode
from dubai_planner.models.request import TripRequest, TripRequestDraft
from dubai_planner.models.review import ReviewReport, RuleViolation, Severity


class PlanStatus(StrEnum):
    """How a run ended.

    - ``approved``: the reviewer approved it and no error-severity rule is violated.
    - ``needs_revision``: the review-iteration cap was reached; the best attempt is
      returned **unapproved**, with its issues.
    - ``needs_clarification``: required request fields are missing; a question is asked.
    - ``failed``: an agent or provider error stopped the run (it can be resumed).
    - ``cancelled``: the caller stopped consuming the stream (it can be resumed).
    """

    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    NEEDS_CLARIFICATION = "needs_clarification"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AttemptSummary(StrictModel):
    """One compose → validate → review pass."""

    iteration: int = Field(ge=0)
    approved: bool
    score: float = Field(ge=0, le=1)
    total_aed: Decimal
    errors: int = Field(ge=0)
    warnings: int = Field(ge=0)
    blocking_rules: list[str] = Field(default_factory=list)
    locked_days: list[int] = Field(default_factory=list)


class Attempt(StrictModel):
    """A full attempt kept so the best one can be returned when the cap is reached."""

    iteration: int = Field(ge=0)
    itinerary: Itinerary
    review: ReviewReport
    violations: list[RuleViolation] = Field(default_factory=list)

    @property
    def errors(self) -> int:
        return sum(1 for v in self.violations if v.severity is Severity.ERROR) + len(
            [i for i in self.review.issues if i.severity is Severity.ERROR and i.rule_id is None]
        )

    @property
    def approved(self) -> bool:
        """Reviewer approval **and** no error-severity rule violation (never trust one alone)."""
        return self.review.approved and not any(
            v.severity is Severity.ERROR for v in self.violations)

    def rank(self) -> tuple[bool, int, float, int]:
        """Higher is better: approved, fewer errors, higher score, later iteration."""
        return (self.approved, -self.errors, self.review.score, self.iteration)


class PlanResult(StrictModel):
    """Final outcome of one planning request (or of a resumed one)."""

    session_id: str
    run_id: str
    mode: PlannerMode
    status: PlanStatus
    request: TripRequest | None = None
    draft: TripRequestDraft | None = None
    itinerary: Itinerary | None = None
    review: ReviewReport | None = None
    violations: list[RuleViolation] = Field(default_factory=list)
    attempts: list[AttemptSummary] = Field(default_factory=list)
    iterations: int = Field(default=0, ge=0, description="Refinements performed")
    clarification_question: str | None = None
    missing: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    saved_version: int | None = None
    error: str | None = None

    @property
    def approved(self) -> bool:
        return self.status is PlanStatus.APPROVED

    @property
    def blocking_issues(self) -> list[str]:
        issues = [v.message for v in self.violations if v.severity is Severity.ERROR]
        if self.review is not None:
            issues += [i.message for i in self.review.blocking_issues if i.rule_id is None]
        return issues


class PlannerUpdate(StrictModel):
    """One streamed progress event (a node finished, or the run ended)."""

    session_id: str
    run_id: str
    node: str
    iteration: int = Field(default=0, ge=0)
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    summary: dict[str, Any] = Field(default_factory=dict)
    result: PlanResult | None = Field(default=None, description="Set on the final update only")

    @property
    def is_final(self) -> bool:
        return self.result is not None
