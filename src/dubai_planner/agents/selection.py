"""Grounded selection shared by the three LLM research agents (attractions, hotels, dining).

The tool ranks candidates deterministically; the LLM picks ids from that list and
explains why. Grounding (hard rule 5): ids not in the candidate list are **rejected**:
they are dropped, recorded in ``rejected_ids`` and traced. If no valid id is left, the
output counts as invalid and goes through the repair retry. If the model returns fewer
picks than needed, the list is topped up from the deterministic ranking (and marked so).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from dubai_planner.agents.base import LLMOutput, context_block, load_prompt
from dubai_planner.llm.errors import StructuredOutputError
from dubai_planner.llm.mock import MockRequest, register_generator
from dubai_planner.llm.structured import build_messages, invoke_structured
from dubai_planner.models.common import StrictModel
from dubai_planner.models.options import Venue
from dubai_planner.models.request import TripRequest
from dubai_planner.models.trace import TraceEventType
from dubai_planner.tracing import emit

V = TypeVar("V", bound=Venue)
TOPPED_UP = "Added from the deterministic ranking (the model picked fewer options)."
FALLBACK = "Deterministic ranking (model output was invalid after repair)."


class Pick(LLMOutput):
    id: str = Field(min_length=3, max_length=80)
    score: float = Field(default=0.5, ge=0, le=1)
    rationale: str = Field(default="", max_length=400)


class Selection(LLMOutput):
    picks: list[Pick] = Field(min_length=1, max_length=40)


class CandidateBrief(StrictModel):
    id: str
    name: str
    area: str
    summary: str
    price: str
    tool_score: float


class SelectionContext(StrictModel):
    task: str
    picks_wanted: int
    request: dict[str, Any]
    candidates: list[CandidateBrief]


class SelectionResult(BaseModel, Generic[V]):
    options: list[V]
    rejected_ids: list[str] = Field(default_factory=list)
    used_fallback: bool = False
    llm_calls: int = 0


def request_brief(request: TripRequest) -> dict[str, Any]:
    """The request fields research agents need (compact, JSON-safe)."""
    return {
        "dates": f"{request.start_date.isoformat()} to {request.end_date.isoformat()}",
        "days": request.num_days,
        "adults": request.travellers.adults,
        "children_ages": request.travellers.children_ages,
        "budget_aed_total": str(request.total_budget_aed()),
        "interests": request.interests,
        "pace": request.pace.value,
        "dietary": [d.value for d in request.dietary],
        "accessibility": [a.value for a in request.accessibility],
        "preferred_areas": request.preferred_areas,
        "must_include": request.must_include,
        "avoid": request.avoid,
    }


def select_venues(
    llm: Any,
    *,
    agent: str,
    task: str,
    request: TripRequest,
    candidates: Sequence[V],
    picks_wanted: int,
    brief: Callable[[V], CandidateBrief],
    result_type: type[SelectionResult[V]],
    fallback: bool = False,
) -> SelectionResult[V]:
    """Let the LLM choose ``picks_wanted`` candidates by id (grounded, validated).

    ``result_type`` must be the concrete parametrisation (e.g.
    ``SelectionResult[AttractionOption]``) so options serialise with all their fields.
    """
    if not candidates:
        return result_type(options=[])
    picks_wanted = min(picks_wanted, len(candidates))
    by_id = {c.id: c for c in candidates}
    rejected: list[str] = []
    ctx = SelectionContext(task=task, picks_wanted=picks_wanted, request=request_brief(request),
                           candidates=[brief(c) for c in candidates])
    messages = build_messages(
        system=load_prompt(f"{agent}.system"),
        user=load_prompt("selection.user", task=task, picks_wanted=picks_wanted,
                         context=context_block(ctx)),
        schema=Selection, agent=agent,
    )

    def ground(selection: Selection) -> list[V]:
        chosen: list[V] = []
        for pick in selection.picks:
            venue = by_id.get(pick.id)
            if venue is None:
                if pick.id not in rejected:
                    rejected.append(pick.id)
                continue
            if any(c.id == venue.id for c in chosen):
                continue
            chosen.append(venue.model_copy(update={"score": round(pick.score, 3),
                                                   "rationale": pick.rationale or None}))
        if not chosen:
            raise ValueError(
                "none of the picked ids are in the candidate list; use ids exactly as given "
                f"(e.g. {', '.join(list(by_id)[:3])})")
        return chosen[:picks_wanted]

    try:
        result = invoke_structured(llm, messages, Selection, agent=agent, postprocess=ground)
        chosen, calls, used_fallback = result.value, result.calls, False
    except StructuredOutputError:
        if not fallback:
            raise
        emit(agent, TraceEventType.INFO, payload={"fallback": "deterministic_ranking"})
        chosen = [c.model_copy(update={"rationale": FALLBACK}) for c in candidates[:picks_wanted]]
        calls, used_fallback = 0, True
    if rejected:
        emit(agent, TraceEventType.GUARDRAIL,
             payload={"rule": "grounding.known_entities", "rejected_ids": rejected})
    for candidate in candidates:  # top up from the deterministic ranking
        if len(chosen) >= picks_wanted:
            break
        if all(c.id != candidate.id for c in chosen):
            chosen.append(candidate.model_copy(update={"rationale": TOPPED_UP}))
    return result_type(options=chosen, rejected_ids=rejected, used_fallback=used_fallback,
                       llm_calls=calls)


def _mock_selection(req: MockRequest) -> Selection:
    """Deterministic mock: take the top-ranked candidates in tool order."""
    ctx = SelectionContext.model_validate(req.context or {})
    return Selection(picks=[
        Pick(id=c.id, score=c.tool_score,
             rationale=f"Ranked #{i + 1} by the search tool for: {ctx.task}")
        for i, c in enumerate(ctx.candidates[: ctx.picks_wanted])
    ])


register_generator(Selection.__name__, _mock_selection)
