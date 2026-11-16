"""Turn a failed review into a *targeted* revision request for the composer.

Shared by both orchestration modes (the LangGraph ``refine`` node and the CrewAI
outer loop), so the revision strategy is not duplicated per framework.

Strategy (deterministic, no LLM):

- **feedback**: every error, then warnings, as one line each with the rule id, the
  day, the message and the suggested fix (bounded, so prompts stay small);
- **locked days**: if every blocking (error) issue names a day, all *other* days are
  locked and kept verbatim from the previous attempt, so only the faulty days are
  re-planned; a blocking issue without a day (e.g. ``budget.total``) unlocks every day.
"""

from __future__ import annotations

from pydantic import Field

from dubai_planner.models.common import StrictModel
from dubai_planner.models.itinerary import Itinerary
from dubai_planner.models.request import TripRequest
from dubai_planner.models.review import ReviewIssue, ReviewReport, Severity

MAX_FEEDBACK_LINES = 12


class RevisionPlan(StrictModel):
    feedback: list[str] = Field(default_factory=list)
    locked_days: list[int] = Field(default_factory=list, description="0-based days to keep")
    target_days: list[int] = Field(default_factory=list, description="0-based days to redo")


def _line(issue: ReviewIssue) -> str:
    where = f" day {issue.day_index + 1}" if issue.day_index is not None else ""
    rule = f"[{issue.rule_id}]" if issue.rule_id else "[reviewer]"
    fix = f" Fix: {issue.suggested_fix}" if issue.suggested_fix else ""
    return f"{issue.severity.value.upper()} {rule}{where}: {issue.message}{fix}"[:400]


def applicable_locks(edit_locked: list[int], previous: Itinerary | None,
                     request: TripRequest) -> list[int]:
    """Days a user edit may keep: only while the request's dates still match ``previous``.

    A refinement that changed the dates or length makes the old days meaningless, so the
    locks are dropped and every day is re-planned.
    """
    if not edit_locked or previous is None or len(previous.days) != request.num_days:
        return []
    if previous.days[0].date != request.start_date:
        return []
    return sorted(d for d in set(edit_locked) if 0 <= d < request.num_days)


def plan_revision(review: ReviewReport, itinerary: Itinerary, *, num_days: int) -> RevisionPlan:
    """Feedback lines and the days to keep for the next compose attempt."""
    errors = [i for i in review.issues if i.severity is Severity.ERROR]
    warnings = [i for i in review.issues if i.severity is Severity.WARNING]
    feedback = [_line(i) for i in [*errors, *warnings]][:MAX_FEEDBACK_LINES]
    days = min(len(itinerary.days), num_days)
    if not errors or any(i.day_index is None for i in errors):
        return RevisionPlan(feedback=feedback, target_days=list(range(days)))
    targets = sorted({i.day_index for i in errors if i.day_index is not None and i.day_index
                      < days})
    if not targets:
        return RevisionPlan(feedback=feedback, target_days=list(range(days)))
    locked = [d for d in range(days) if d not in targets]
    return RevisionPlan(feedback=feedback, locked_days=locked, target_days=targets)
