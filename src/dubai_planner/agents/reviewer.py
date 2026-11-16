"""Reviewer agent: merges deterministic rule violations with an LLM quality review.

The LLM sees a compact itinerary, the request and the rule violations, and returns
issues it finds (relevance, variety, realism…). The final :class:`ReviewReport` is
decided in Python:

- every rule violation becomes an issue (with a suggested fix);
- ``approved`` ⇔ no ERROR-severity issue from either source (the model cannot approve a
  plan that breaks a hard rule, nor block one with only warnings);
- ``score`` = min(model score, rule-based ceiling), clamped to [0, 1].
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from dubai_planner.agents.base import LLMOutput, context_block, load_prompt
from dubai_planner.agents.selection import request_brief
from dubai_planner.llm.errors import StructuredOutputError
from dubai_planner.llm.mock import MockRequest, register_generator
from dubai_planner.llm.structured import build_messages, invoke_structured
from dubai_planner.models.common import StrictModel
from dubai_planner.models.itinerary import Itinerary
from dubai_planner.models.request import TripRequest
from dubai_planner.models.review import ReviewIssue, ReviewReport, RuleViolation, Severity
from dubai_planner.models.trace import TraceEventType
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import agent_span, emit

AGENT = "reviewer"
ERROR_PENALTY = 0.25
WARNING_PENALTY = 0.05

SUGGESTED_FIXES: dict[str, str] = {
    "budget.total": "Swap expensive attractions/restaurants for cheaper researched options "
                    "or drop one paid activity.",
    "budget.arithmetic": "Recompute costs with the cost calculator.",
    "time.overlap": "Shift the later slot so it starts after the earlier one ends.",
    "time.opening_hours": "Move the visit inside the venue's opening hours or another day.",
    "heat.midday_outdoor": "Move the outdoor slot before 11:00 or after 16:00, or swap it for "
                           "an indoor attraction.",
    "geo.travel_time": "Group the day's activities in nearby areas.",
    "pace.density": "Remove an activity from that day or move it to a lighter day.",
    "constraints.dietary": "Replace the restaurant with one that caters for the dietary needs.",
    "constraints.accessibility": "Replace the venue with a wheelchair-accessible option.",
    "grounding.known_entities": "Use only venues from the researched lists.",
}


class ReviewerInput(StrictModel):
    itinerary: Itinerary
    request: TripRequest
    violations: list[RuleViolation] = Field(default_factory=list)
    iteration: int = Field(default=0, ge=0)


class DraftIssue(LLMOutput):
    severity: Severity = Severity.WARNING
    message: str = Field(min_length=1, max_length=400)
    day: int | None = Field(default=None, ge=1, description="1-based day number")
    suggested_fix: str | None = Field(default=None, max_length=300)


class ReviewDraft(LLMOutput):
    approved: bool
    score: float = Field(ge=0, le=1)
    summary: str = Field(default="", max_length=800)
    issues: list[DraftIssue] = Field(default_factory=list, max_length=20)


class ReviewContext(StrictModel):
    request: dict[str, Any]
    itinerary: dict[str, Any]
    violations: list[dict[str, Any]]


def compact_itinerary(it: Itinerary) -> dict[str, Any]:
    return {
        "title": it.title,
        "hotel": it.hotel.name if it.hotel else None,
        "days": [
            {"day": i + 1, "date": d.date.isoformat(), "theme": d.theme,
             "slots": [f"{s.start:%H:%M}-{s.end:%H:%M} {s.activity.kind.value}: "
                       f"{s.activity.name} ({'indoor' if s.indoor else 'outdoor'}, "
                       f"AED {s.cost_aed})" for s in d.slots]}
            for i, d in enumerate(it.days)
        ],
        "cost": {"total": str(it.cost.total), "budget": str(it.cost.budget_aed),
                 "within_budget": it.cost.within_budget},
    }


def violation_issue(v: RuleViolation) -> ReviewIssue:
    return ReviewIssue(severity=v.severity, message=v.message, rule_id=v.rule_id,
                       day_index=v.day_index, slot_index=v.slot_index,
                       suggested_fix=SUGGESTED_FIXES.get(v.rule_id))


def merge(draft: ReviewDraft, violations: list[RuleViolation], iteration: int) -> ReviewReport:
    issues = [violation_issue(v) for v in violations]
    seen = {i.message.strip().lower() for i in issues}
    for d in draft.issues:
        if d.message.strip().lower() in seen:
            continue
        seen.add(d.message.strip().lower())
        issues.append(ReviewIssue(severity=d.severity, message=d.message,
                                  day_index=d.day - 1 if d.day else None,
                                  suggested_fix=d.suggested_fix))
    errors = sum(1 for i in issues if i.severity is Severity.ERROR)
    warnings = sum(1 for i in issues if i.severity is Severity.WARNING)
    ceiling = 1.0 - ERROR_PENALTY * errors - WARNING_PENALTY * warnings
    score = round(max(0.0, min(draft.score, ceiling)), 3)
    approved = errors == 0
    summary = draft.summary
    if draft.approved != approved:
        verdict = "approved" if approved else "rejected"
        summary = (f"{summary} [Decision: {verdict} by rule: "
                   f"{errors} blocking issue(s).]").strip()
    return ReviewReport(approved=approved, score=score, issues=issues, summary=summary,
                        iteration=iteration)


def deterministic_draft(violations: list[dict[str, Any]] | list[RuleViolation]) -> ReviewDraft:
    """Rule-only review (mock reply and fallback)."""
    severities = [v.severity if isinstance(v, RuleViolation) else Severity(v["severity"])
                  for v in violations]
    errors = severities.count(Severity.ERROR)
    warnings = severities.count(Severity.WARNING)
    return ReviewDraft(
        approved=errors == 0,
        score=max(0.0, 0.9 - ERROR_PENALTY * errors - WARNING_PENALTY * warnings),
        summary=(f"Rule check: {errors} error(s), {warnings} warning(s)."
                 + (" Ready to go." if errors == 0 else " Needs revision.")),
    )


def run(data: ReviewerInput, llm: Any, tools: Toolbox | None = None, *, fallback: bool = False
        ) -> ReviewReport:
    """``tools`` is unused (rules are run by the caller) but kept for a uniform signature."""
    with agent_span(AGENT, violations=len(data.violations)) as span:
        ctx = ReviewContext(request=request_brief(data.request),
                            itinerary=compact_itinerary(data.itinerary),
                            violations=[v.model_dump(mode="json", exclude={"details"})
                                        for v in data.violations])
        messages = build_messages(
            system=load_prompt("reviewer.system"),
            user=load_prompt("reviewer.user", context=context_block(ctx)),
            schema=ReviewDraft, agent=AGENT,
        )
        draft: ReviewDraft
        try:
            draft = invoke_structured(llm, messages, ReviewDraft, agent=AGENT).value
        except StructuredOutputError:
            if not fallback:
                raise
            emit(AGENT, TraceEventType.INFO, payload={"fallback": "rule_only_review"})
            draft = deterministic_draft(data.violations)
        report = merge(draft, data.violations, data.iteration)
        span.update(approved=report.approved, score=report.score, issues=len(report.issues))
        return report


def _mock_review(req: MockRequest) -> ReviewDraft:
    ctx = ReviewContext.model_validate(req.context or {})
    return deterministic_draft(ctx.violations)


register_generator(ReviewDraft.__name__, _mock_review)
