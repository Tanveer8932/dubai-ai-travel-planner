"""Results of deterministic validation and of the Reviewer agent."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from dubai_planner.models.common import StrictModel


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class RuleViolation(StrictModel):
    """Output of a deterministic rule in ``validation/rules.py``, e.g. ``heat.midday_outdoor``."""

    rule_id: str = Field(pattern=r"^[a-z_]+\.[a-z_]+$")
    severity: Severity
    message: str
    day_index: int | None = Field(default=None, ge=0)
    slot_index: int | None = Field(default=None, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)


class ReviewIssue(StrictModel):
    severity: Severity
    message: str
    rule_id: str | None = None
    day_index: int | None = Field(default=None, ge=0)
    slot_index: int | None = Field(default=None, ge=0)
    suggested_fix: str | None = None


class ReviewReport(StrictModel):
    approved: bool
    score: float = Field(ge=0, le=1)
    issues: list[ReviewIssue] = Field(default_factory=list)
    summary: str = ""
    iteration: int = Field(default=0, ge=0)

    @property
    def blocking_issues(self) -> list[ReviewIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]
