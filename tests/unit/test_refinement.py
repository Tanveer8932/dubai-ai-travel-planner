"""Phase 5: follow-up classification (request delta vs local edit) and budget maths."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from factories import itinerary as make_itinerary
from factories import trip_request as make_request

from dubai_planner.models import ItineraryRecord, PlannerMode
from dubai_planner.services.refinement import ChangeKind, classify, resolve_budget


def record(**kwargs: Any) -> ItineraryRecord:
    request = kwargs.pop("request", make_request())
    itinerary = kwargs.pop("itinerary", make_itinerary(days=3))
    return ItineraryRecord(id=itinerary.id, session_id="s", version=2, parent_id=None,
                           mode=PlannerMode.LANGGRAPH, total_aed=itinerary.cost.total,
                           approved=True, created_at=datetime.now(UTC), request=request,
                           itinerary=itinerary)


@pytest.mark.parametrize(("text", "expected"), [
    ("reduce the budget by 20%", Decimal("4800")),
    ("Cut the budget by 12.5 percent", Decimal("5250")),
    ("increase my budget by 10%", Decimal("6600")),
    ("add AED 500 to the budget", Decimal("6500")),
    ("lower the budget by 1,000 dirhams", Decimal("5000")),
    ("change the budget to AED 5000", None),  # absolute: the requirements agent does it
    ("make day 2 cheaper", None),  # no "budget"
    ("reduce the budget by 150%", None),  # nonsense
])
def test_resolve_budget(text: str, expected: Decimal | None) -> None:
    assert resolve_budget(text, Decimal("6000")) == expected


def test_no_saved_itinerary_is_a_new_request() -> None:
    r = classify("2 adults in Feb", latest=None, pending_question=False)
    assert r.kind is ChangeKind.NEW_REQUEST and r.edit is None


def test_pending_question_wins() -> None:
    r = classify("Replace day 2", latest=record(), pending_question=True)
    assert r.kind is ChangeKind.CLARIFICATION_ANSWER and r.edit is None


def test_relative_budget_change_is_resolved_into_the_message() -> None:
    base = record()
    r = classify("reduce the budget by 20%", latest=base, pending_question=False)
    assert r.kind is ChangeKind.REQUEST_DELTA
    assert r.message == "reduce the budget by 20%: the new total budget is AED 4800."
    assert r.edit is not None and r.edit.base_request == base.request
    assert not r.edit.keep_request and not r.edit.locked_days


@pytest.mark.parametrize("text", ["make it 5 days in March", "add one more adult",
                                  "two children are coming too", "change the dates"])
def test_request_changes_replan_everything(text: str) -> None:
    r = classify(text, latest=record(), pending_question=False)
    assert r.kind is ChangeKind.REQUEST_DELTA and r.edit is not None
    assert r.edit.base_draft() is not None and not r.edit.locked_days


def test_day_edit_locks_the_other_days() -> None:
    base = record()
    r = classify("Replace day 2 with indoor activities", latest=base, pending_question=False)
    assert r.kind is ChangeKind.LOCAL_EDIT and r.target_days == [1]
    edit = r.edit
    assert edit is not None and edit.keep_request and edit.previous == base.itinerary
    assert edit.locked_days == [0, 2]
    [constraint] = edit.day_constraints
    assert constraint.day_index == 1 and constraint.indoor_only
    assert any("indoor" in line for line in edit.instructions)


@pytest.mark.parametrize(("text", "days", "indoor"), [
    ("swap days 1 and 3 for something calmer", [0, 2], False),
    ("days 1-2 should be indoors", [0, 1], True),
    ("make the last day more relaxed", [2], False),
    ("the first day: no outdoor stuff, indoor only please", [0], False),
    ("day 7 should be a beach day", None, False),  # outside the trip: general change
])
def test_day_references(text: str, days: list[int] | None, indoor: bool) -> None:
    r = classify(text, latest=record(), pending_question=False)
    if days is None:
        assert r.kind is ChangeKind.GENERAL_CHANGE
        return
    assert r.kind is ChangeKind.LOCAL_EDIT and r.target_days == days
    assert r.edit is not None
    assert all(c.indoor_only is indoor for c in r.edit.day_constraints)


def test_general_change_replans_with_the_saved_request() -> None:
    r = classify("more museums please", latest=record(), pending_question=False)
    assert r.kind is ChangeKind.GENERAL_CHANGE and r.edit is not None
    assert r.edit.base_request is not None and not r.edit.keep_request
