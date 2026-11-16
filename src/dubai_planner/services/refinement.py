"""Conversational refinement (implementation plan, Phase 5 task 4; architecture §9).

A follow-up message on a session that already has an itinerary is classified as:

- **request delta**: budget, dates, length or travellers change → a full re-plan of
  the updated request. A *relative* budget change ("reduce the budget by 20%", "add
  AED 500") is resolved here, deterministically, against the saved request (hard rule
  4: no LLM arithmetic) and stated explicitly in the message the planner receives;
- **local edit**: the message names specific days ("replace day 2 with indoor
  activities") → the request is kept, every other day is locked and copied verbatim,
  and the named days are re-planned with the user's words as composer feedback plus
  deterministic day constraints (``indoor_only``);
- otherwise a **new request** (no saved itinerary yet), a **clarification answer**
  (the planner is waiting on a question) or a **general change** (re-plan with the
  saved request as the base).

The result is a :class:`PlanEdit` for either planner: nothing here depends on the
orchestration framework or on its state, only on SQLite.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from pydantic import Field

from dubai_planner.models import DayConstraint, ItineraryRecord, PlanEdit
from dubai_planner.models.common import StrictModel


class ChangeKind(StrEnum):
    NEW_REQUEST = "new_request"
    CLARIFICATION_ANSWER = "clarification_answer"
    REQUEST_DELTA = "request_delta"
    LOCAL_EDIT = "local_edit"
    GENERAL_CHANGE = "general_change"


class Refinement(StrictModel):
    """What to send to the planner for one follow-up message."""

    kind: ChangeKind
    message: str = Field(description="Text the planner receives (and stores as the turn)")
    edit: PlanEdit | None = None
    target_days: list[int] = Field(default_factory=list, description="0-based days to redo")
    explanation: str = ""


_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
             "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "last": -1}
_DAY_NUM = re.compile(r"\bdays?\s+(\d{1,2})((?:\s*(?:,|and|&|-|to)\s*\d{1,2})*)", re.I)
_DAY_ORD = re.compile(r"\b(" + "|".join(_ORDINALS) + r")\s+day\b", re.I)
_BUDGET = re.compile(r"\bbudget\b", re.I)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per\s*cent|percent)", re.I)
_AMOUNT = re.compile(r"(?:aed|dhs?|dirhams?)\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*"
                     r"(?:aed|dhs?|dirhams?)", re.I)
_DOWN = re.compile(r"\b(reduce|cut|lower|decrease|less|cheaper|shrink|drop|down|save)\b", re.I)
_UP = re.compile(r"\b(increase|raise|more|extra|add|up|boost)\b", re.I)
_REQUEST_TERMS = re.compile(
    r"\b(budget|aed|dirhams?|usd|eur|gbp|dates?|january|february|march|april|may\s+\d|june|"
    r"july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|"
    r"sept|oct|nov|dec|nights?|adults?|child(?:ren)?|kids?|travell?ers?|people|persons?|"
    r"longer|shorter|extend|week)\b", re.I)
_INDOOR = re.compile(r"\bindoors?\b", re.I)
_OUTDOOR = re.compile(r"\boutdoors?\b", re.I)


def _days(text: str, num_days: int) -> list[int]:
    """0-based days the message names ("day 2", "days 1 and 3", "the last day")."""
    found: set[int] = set()
    for match in _DAY_NUM.finditer(text):
        nums = [int(match.group(1)), *(int(n) for n in re.findall(r"\d{1,2}", match.group(2)))]
        if re.search(r"-|to", match.group(2) or "") and len(nums) == 2:
            nums = list(range(nums[0], nums[1] + 1))
        found.update(n - 1 for n in nums)
    for match in _DAY_ORD.finditer(text):
        n = _ORDINALS[match.group(1).lower()]
        found.add(num_days - 1 if n == -1 else n - 1)
    return sorted(found)


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)


def resolve_budget(text: str, current: Decimal) -> Decimal | None:
    """New total budget for a *relative* change ("by 20%", "add AED 500"), else None.

    An absolute target ("budget to AED 5000") is left to the requirements agent.
    """
    if not _BUDGET.search(text):
        return None
    direction = -1 if _DOWN.search(text) else (1 if _UP.search(text) else 0)
    if direction == 0:
        return None
    if percent := _PERCENT.search(text):
        share = Decimal(percent.group(1)) / 100
        if not 0 < share < 1:
            return None
        return _quantize(current * (1 + direction * share))
    amount = _AMOUNT.search(text)
    if amount and re.search(r"\bby\b|\badd\b|\bextra\b|\bmore\b", text, re.I):
        value = Decimal((amount.group(1) or amount.group(2)).replace(",", ""))
        new = current + direction * value
        return _quantize(new) if new > 0 else None
    return None


def classify(message: str, *, latest: ItineraryRecord | None, pending_question: bool
             ) -> Refinement:
    """Decide how ``message`` relates to the session's latest saved itinerary."""
    text = message.strip()
    if pending_question:
        return Refinement(kind=ChangeKind.CLARIFICATION_ANSWER, message=text,
                          explanation="Answering the planner's question.")
    if latest is None or latest.request is None:
        return Refinement(kind=ChangeKind.NEW_REQUEST, message=text,
                          explanation="A new trip request.")
    request, itinerary = latest.request, latest.itinerary
    base = {"base_version": latest.version, "base_request": request}
    budget = resolve_budget(text, request.total_budget_aed())
    if budget is not None:
        return Refinement(
            kind=ChangeKind.REQUEST_DELTA,
            message=f"{text.rstrip('.')}: the new total budget is AED {budget}.",
            edit=PlanEdit(**base),
            explanation=f"Budget AED {request.total_budget_aed()} → AED {budget}; full re-plan.")
    if _REQUEST_TERMS.search(text):
        return Refinement(kind=ChangeKind.REQUEST_DELTA, message=text, edit=PlanEdit(**base),
                          explanation="The trip request changes; full re-plan.")
    days = [d for d in _days(text, len(itinerary.days)) if 0 <= d < len(itinerary.days)]
    if days:
        indoor = bool(_INDOOR.search(text)) and not _OUTDOOR.search(text)
        constraints = [DayConstraint(day_index=d, indoor_only=indoor, note=text[:300])
                       for d in days]
        locked = [d for d in range(len(itinerary.days)) if d not in days]
        names = ", ".join(str(d + 1) for d in days)
        return Refinement(
            kind=ChangeKind.LOCAL_EDIT, message=text, target_days=days,
            edit=PlanEdit(**base, keep_request=True, previous=itinerary, locked_days=locked,
                          day_constraints=constraints,
                          instructions=[f"Traveller's change for day(s) {names}: {text[:300]}"]
                          + (["Use indoor attractions only on those day(s)."] if indoor else [])),
            explanation=f"Re-planning day(s) {names} only; the other days are kept.")
    return Refinement(kind=ChangeKind.GENERAL_CHANGE, message=text, edit=PlanEdit(**base),
                      explanation="Re-planning the trip with your change.")
