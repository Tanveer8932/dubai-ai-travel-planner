"""``PlannerState``: the shared, checkpointed state of the LangGraph planner (architecture §3).

Every value is a Pydantic domain model (or a plain type), so a checkpoint is the same
data contract as the rest of the system. Only two keys are written concurrently and
therefore have reducers:

- ``research``: the five research branches each write a *partial* dict
  (``{"hotels": [...]}``). :func:`merge_research` merges them key by key, so parallel
  branches never collide; writing ``None`` resets it for a new request.
- ``messages``: the conversation, append-only.

All other keys are written by exactly one node per super-step (overwrite semantics).
"""

from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, Any, TypedDict

from dubai_planner.models import (
    Attempt,
    AttemptSummary,
    BudgetPlan,
    ChatMessage,
    Itinerary,
    PlanStatus,
    ResearchBundle,
    ReviewReport,
    RuleViolation,
    TripRequest,
    TripRequestDraft,
)

#: Research keys and the node that owns each one (one writer per key).
RESEARCH_KEYS = ("attractions", "hotels", "dining", "transport", "weather", "events")


def merge_research(current: ResearchBundle | None,
                   update: ResearchBundle | dict[str, Any] | None) -> ResearchBundle | None:
    """Reducer for ``research``.

    - ``None`` → reset (a new request starts with no research);
    - a :class:`ResearchBundle` → replace;
    - a ``dict`` with some of :data:`RESEARCH_KEYS` → merge those keys into the current
      bundle (unknown keys are a programming error and raise).
    """
    if update is None:
        return None
    if isinstance(update, ResearchBundle):
        return update
    unknown = set(update) - set(RESEARCH_KEYS)
    if unknown:
        raise ValueError(f"unknown research keys {sorted(unknown)}")
    base = current if current is not None else ResearchBundle()
    return ResearchBundle.model_validate({**dict(base), **update})


class PlannerState(TypedDict, total=False):
    """State of one session's planning thread (``thread_id = session_id``)."""

    # --- identity / input
    session_id: str
    run_id: str
    today: date
    user_message: str
    messages: Annotated[list[ChatMessage], operator.add]

    # --- requirements
    draft: TripRequestDraft | None
    request: TripRequest | None
    missing: list[str]
    clarification_question: str | None
    clarifications: int
    assumptions: list[str]

    # --- research (parallel fan-out) and planning
    research: Annotated[ResearchBundle | None, merge_research]
    budget_plan: BudgetPlan | None
    itinerary: Itinerary | None
    validation: list[RuleViolation]
    review: ReviewReport | None

    # --- review loop
    iteration: int
    feedback: list[str]
    locked_days: list[int]
    best: Attempt | None
    attempts: list[AttemptSummary]
    status: PlanStatus | None
    saved_version: int | None


def new_run_input(*, session_id: str, run_id: str, message: str, today: date,
                  user_message: ChatMessage) -> dict[str, Any]:
    """Input for a new planning request on a (possibly existing) thread.

    The conversation (``messages``) and the request ``draft`` carry over, so a follow-up
    refines the previous request. Everything produced by the previous run is reset, so
    nothing from it can leak into this run's result.
    """
    return {
        "session_id": session_id, "run_id": run_id, "today": today,
        "user_message": message, "messages": [user_message],
        "request": None, "missing": [], "clarification_question": None,
        "clarifications": 0, "assumptions": [],
        "research": None, "budget_plan": None, "itinerary": None, "validation": [],
        "review": None, "iteration": 0, "feedback": [], "locked_days": [], "best": None,
        "attempts": [], "status": None, "saved_version": None,
    }
