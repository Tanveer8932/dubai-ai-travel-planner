"""Optional LLM-as-judge (implementation plan, Phase 7 task 3; ``research.md §5``).

Two instruments, both through the same structured-output path as the agents (prompt
files, Pydantic validation, one repair retry):

- :func:`rubric`: 1-5 scores for relevance, feasibility, explanation and local
  insight; the judge score is their mean;
- :func:`pairwise`: A/B comparison **run twice with the positions swapped** (Zheng et
  al., 2023). A winner counts only if both orders agree; otherwise the result is a
  tie and the disagreement is reported as position bias.

The mock provider answers with a **constant placeholder** (all 3s, "tie"): mock judge
numbers carry no quality information and reports must say so. Use a different model
from the generator where possible (``LLM_MODEL__JUDGE``), since a model judging its
own output is biased.
"""

from __future__ import annotations

import statistics
from typing import Any, Literal

from pydantic import Field

from dubai_planner.agents.base import LLMOutput, context_block, load_prompt
from dubai_planner.agents.reviewer import compact_itinerary
from dubai_planner.agents.selection import request_brief
from dubai_planner.llm.mock import MockRequest, register_generator
from dubai_planner.llm.structured import build_messages, invoke_structured
from dubai_planner.models import Itinerary, TripRequest
from dubai_planner.models.common import StrictModel

AGENT = "judge"
MOCK_KIND = "mock placeholder (constant; not a quality judgment)"


class RubricScores(LLMOutput):
    relevance: int = Field(ge=1, le=5)
    feasibility: int = Field(ge=1, le=5)
    explanation: int = Field(ge=1, le=5)
    local_insight: int = Field(ge=1, le=5)
    rationale: str = Field(default="", max_length=800)

    @property
    def mean(self) -> float:
        return round(statistics.fmean(
            [self.relevance, self.feasibility, self.explanation, self.local_insight]), 3)


class PairwiseVerdict(LLMOutput):
    winner: Literal["A", "B", "tie"]
    rationale: str = Field(default="", max_length=800)


class PairResult(StrictModel):
    """Outcome of a position-swapped comparison of plan ``a`` and plan ``b``."""

    first: Literal["A", "B", "tie"]  # a shown as A
    swapped: Literal["a", "b", "tie"]  # b shown as A; mapped back to the plans
    winner: Literal["a", "b", "tie"]
    consistent: bool
    rationale: str = ""


def _plan(it: Itinerary) -> dict[str, Any]:
    plan = compact_itinerary(it)
    plan["summary"] = it.summary[:600]
    plan["notes"] = [n[:200] for n in it.notes[:8]]
    return plan


def rubric(llm: Any, request: TripRequest, itinerary: Itinerary) -> RubricScores:
    messages = build_messages(
        system=load_prompt("judge_rubric.system"),
        user=load_prompt("judge_rubric.user", context=context_block(
            {"request": request_brief(request), "itinerary": _plan(itinerary)})),
        schema=RubricScores, agent=AGENT)
    return invoke_structured(llm, messages, RubricScores, agent=AGENT).value


def _compare(llm: Any, request: TripRequest, a: Itinerary, b: Itinerary) -> PairwiseVerdict:
    messages = build_messages(
        system=load_prompt("judge_pairwise.system"),
        user=load_prompt("judge_pairwise.user", context=context_block(
            {"request": request_brief(request), "A": _plan(a), "B": _plan(b)})),
        schema=PairwiseVerdict, agent=AGENT)
    return invoke_structured(llm, messages, PairwiseVerdict, agent=AGENT).value


def pairwise(llm: Any, request: TripRequest, a: Itinerary, b: Itinerary) -> PairResult:
    """Compare ``a`` and ``b`` in both orders; only an order-independent win counts."""
    first = _compare(llm, request, a, b)
    second = _compare(llm, request, b, a)
    back: dict[str, Literal["a", "b", "tie"]] = {"A": "b", "B": "a", "tie": "tie"}
    fwd: dict[str, Literal["a", "b", "tie"]] = {"A": "a", "B": "b", "tie": "tie"}
    swapped = back[second.winner]
    consistent = fwd[first.winner] == swapped
    return PairResult(first=first.winner, swapped=swapped,
                      winner=swapped if consistent else "tie", consistent=consistent,
                      rationale=first.rationale[:400])


def _mock_rubric(_req: MockRequest) -> RubricScores:
    return RubricScores(relevance=3, feasibility=3, explanation=3, local_insight=3,
                        rationale="Mock judge: constant placeholder, not an assessment.")


def _mock_pairwise(_req: MockRequest) -> PairwiseVerdict:
    return PairwiseVerdict(winner="tie", rationale="Mock judge: constant placeholder.")


register_generator(RubricScores.__name__, _mock_rubric)
register_generator(PairwiseVerdict.__name__, _mock_pairwise)
