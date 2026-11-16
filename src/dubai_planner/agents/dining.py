"""Dining agent: restaurants filtered by dietary needs (hard), ranked; LLM picks a set."""

from __future__ import annotations

from typing import Any

from dubai_planner.agents.selection import CandidateBrief, SelectionResult, select_venues
from dubai_planner.models.common import StrictModel
from dubai_planner.models.options import DiningOption
from dubai_planner.models.request import TripRequest
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import agent_span

AGENT = "dining"
CANDIDATE_LIMIT = 30
MEALS_PER_DAY = 2  # lunch and dinner; breakfast is assumed near the hotel
MAX_PICKS = 24

DiningResult = SelectionResult[DiningOption]


class DiningInput(StrictModel):
    request: TripRequest


def brief(venue: DiningOption) -> CandidateBrief:
    return CandidateBrief(
        id=venue.id, name=venue.name, area=venue.area,
        summary=(f"{', '.join(venue.cuisines[:3])}; {venue.price_band.value}; meals "
                 f"{'/'.join(m.value for m in venue.meal_types)}; caters for "
                 f"{', '.join(d.value for d in venue.dietary) or 'no listed dietary needs'}"),
        price=f"{venue.avg_spend.label()} per person", tool_score=venue.score or 0.0,
    )


def run(data: DiningInput, llm: Any, tools: Toolbox, *, fallback: bool = False) -> DiningResult:
    with agent_span(AGENT) as span:
        candidates = tools.search_restaurants(data.request, limit=CANDIDATE_LIMIT)
        wanted = min(MAX_PICKS, data.request.num_days * MEALS_PER_DAY + 2)
        result = select_venues(
            llm, agent=AGENT,
            task=("Choose varied restaurants for lunches and dinners near the likely "
                  "sightseeing areas; every option already meets the dietary needs"),
            request=data.request, candidates=candidates, picks_wanted=wanted, brief=brief,
            result_type=DiningResult, fallback=fallback,
        )
        span.update(candidates=len(candidates), selected=len(result.options),
                    rejected_ids=result.rejected_ids, fallback=result.used_fallback)
        return result
