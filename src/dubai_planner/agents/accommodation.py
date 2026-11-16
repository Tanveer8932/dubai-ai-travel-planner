"""Accommodation agent: hotels ranked by budget fit and area; LLM picks a short list."""

from __future__ import annotations

from typing import Any

from dubai_planner.agents.selection import CandidateBrief, SelectionResult, select_venues
from dubai_planner.models.common import StrictModel
from dubai_planner.models.options import HotelOption
from dubai_planner.models.request import TripRequest
from dubai_planner.tools.search import target_nightly_rate
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import agent_span

AGENT = "accommodation"
CANDIDATE_LIMIT = 12
PICKS = 3

AccommodationResult = SelectionResult[HotelOption]


class AccommodationInput(StrictModel):
    request: TripRequest


def brief(hotel: HotelOption) -> CandidateBrief:
    metro = f"; metro {hotel.nearest_metro}" if hotel.nearest_metro else ""
    return CandidateBrief(
        id=hotel.id, name=hotel.name, area=hotel.area,
        summary=f"{hotel.stars}-star {hotel.price_band.value}{metro}",
        price=f"{hotel.nightly_rate.label()} per room-night", tool_score=hotel.score or 0.0,
    )


def run(data: AccommodationInput, llm: Any, tools: Toolbox, *, fallback: bool = False
        ) -> AccommodationResult:
    """First option is the recommended hotel. Empty when the trip has no nights."""
    with agent_span(AGENT) as span:
        candidates = tools.search_hotels(data.request, limit=CANDIDATE_LIMIT)
        target = target_nightly_rate(data.request) if data.request.nights else 0
        result = select_venues(
            llm, agent=AGENT,
            task=(f"Recommend {PICKS} hotels, best first; the budget suggests about AED "
                  f"{target:,.0f} per room-night"),
            request=data.request, candidates=candidates, picks_wanted=PICKS, brief=brief,
            result_type=AccommodationResult, fallback=fallback,
        )
        span.update(candidates=len(candidates), selected=[h.id for h in result.options],
                    rejected_ids=result.rejected_ids, fallback=result.used_fallback)
        return result
