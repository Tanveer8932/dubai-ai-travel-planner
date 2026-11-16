"""Attractions agent: ranks seed attractions with the search tool, LLM picks and explains."""

from __future__ import annotations

from typing import Any

from dubai_planner.agents.selection import CandidateBrief, SelectionResult, select_venues
from dubai_planner.models.common import StrictModel
from dubai_planner.models.options import AttractionOption
from dubai_planner.models.request import TripRequest
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import agent_span
from dubai_planner.validation.thresholds import PACE_TARGET_ACTIVITIES

AGENT = "attractions"
CANDIDATE_LIMIT = 30
EXTRA_PICKS = 2

AttractionsResult = SelectionResult[AttractionOption]


class AttractionsInput(StrictModel):
    request: TripRequest


def brief(att: AttractionOption) -> CandidateBrief:
    setting = "indoor" if att.indoor else "outdoor"
    best = "/".join(t.value for t in att.best_time) or "any time"
    flags = [f for f, on in (("day trip", att.day_trip), ("booking", att.requires_booking)) if on]
    return CandidateBrief(
        id=att.id, name=att.name, area=att.area,
        summary=(f"{att.category.value}; {setting}; {att.typical_duration_min} min; best {best}; "
                 f"tags {', '.join(att.tags[:5])}" + (f"; {', '.join(flags)}" if flags else "")),
        price=att.price.label(), tool_score=att.score or 0.0,
    )


def picks_wanted(request: TripRequest) -> int:
    return min(CANDIDATE_LIMIT, request.num_days * PACE_TARGET_ACTIVITIES[request.pace]
               + EXTRA_PICKS)


def run(data: AttractionsInput, llm: Any, tools: Toolbox, *, fallback: bool = False
        ) -> AttractionsResult:
    with agent_span(AGENT) as span:
        candidates = tools.search_attractions(data.request, limit=CANDIDATE_LIMIT)
        result = select_venues(
            llm, agent=AGENT, task="Choose the attractions that best fit this trip",
            request=data.request, candidates=candidates,
            picks_wanted=picks_wanted(data.request), brief=brief,
            result_type=AttractionsResult, fallback=fallback,
        )
        span.update(candidates=len(candidates), selected=len(result.options),
                    rejected_ids=result.rejected_ids, fallback=result.used_fallback)
        return result
