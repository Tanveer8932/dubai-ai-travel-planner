"""Phase 5 composer additions: per-day constraints and no reuse of locked-day venues."""

from __future__ import annotations

from typing import Any

import pytest
from factories import compose, trip_request

from dubai_planner.agents import composer
from dubai_planner.llm.mock import MockLLM
from dubai_planner.models import ActivityKind, DayConstraint, PlannerMode
from dubai_planner.tools.toolbox import Toolbox


@pytest.fixture
def first(toolbox: Toolbox, mock_llm: MockLLM) -> tuple[Any, Any, Any]:
    return compose(trip_request(), mock_llm, toolbox)


def _input(first: tuple[Any, Any, Any], **kwargs: Any) -> composer.ComposerInput:
    research, plan, itinerary = first
    return composer.ComposerInput(session_id="s1", mode=PlannerMode.LANGGRAPH,
                                  request=trip_request(), research=research, budget=plan,
                                  previous=itinerary, **kwargs)


def _attractions(day: Any) -> list[Any]:
    return [s for s in day.slots if s.activity.kind is ActivityKind.ATTRACTION]


def test_indoor_only_day_is_replanned_indoors(first: Any, toolbox: Toolbox,
                                              mock_llm: MockLLM) -> None:
    data = _input(first, locked_days=[0, 2],
                  day_constraints=[DayConstraint(day_index=1, indoor_only=True)])
    ctx = composer.build_context(data, toolbox)
    assert ctx.days[1].indoor_only and not ctx.days[0].indoor_only
    assert set(ctx.already_used) == {s.activity.ref_id for d in (0, 2)
                                     for s in _attractions(first[2].days[d])}
    out = composer.run(data, mock_llm, toolbox)
    assert out.days[0] == first[2].days[0] and out.days[2] == first[2].days[2]
    day2 = _attractions(out.days[1])
    assert day2 and all(s.indoor for s in day2)
    assert not {s.activity.ref_id for s in day2} & set(ctx.already_used)


def test_assemble_rejects_outdoor_on_an_indoor_only_day(first: Any, toolbox: Toolbox) -> None:
    data = _input(first, locked_days=[0, 2],
                  day_constraints=[DayConstraint(day_index=1, indoor_only=True)])
    ctx = composer.build_context(data, toolbox)
    outdoor = next(a for a in ctx.attractions
                   if not a.indoor and not a.day_trip and a.id not in ctx.already_used)
    draft = composer.DraftItinerary(days=[composer.DraftDay(day=2, slots=[
        composer.DraftSlot(start="08:00", end="09:00", kind="attraction", ref_id=outdoor.id)])])
    with pytest.raises(ValueError, match="indoor activities only"):
        composer.assemble(draft, data, toolbox)


def test_assemble_rejects_reusing_a_locked_day_attraction(first: Any,
                                                          toolbox: Toolbox) -> None:
    data = _input(first, locked_days=[0, 2])
    reused = _attractions(first[2].days[0])[0]
    draft = composer.DraftItinerary(days=[composer.DraftDay(day=2, slots=[
        composer.DraftSlot(start=f"{reused.start:%H:%M}", end=f"{reused.end:%H:%M}",
                           kind="attraction", ref_id=reused.activity.ref_id)])])
    with pytest.raises(ValueError, match="more than once"):
        composer.assemble(draft, data, toolbox)
