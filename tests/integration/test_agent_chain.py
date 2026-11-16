"""The whole Phase 2 agent chain, offline, with the mock provider (no orchestration yet).

requirements → 5 research agents → budget → composer → rules → reviewer. This is the
sequence LangGraph (Phase 3) and CrewAI (Phase 4) will orchestrate.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from factories import compose

from dubai_planner.agents import requirements, reviewer
from dubai_planner.llm import get_chat_llm
from dubai_planner.models import Severity
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import capture, run_context
from dubai_planner.validation.rules import validate_itinerary

TODAY = date(2026, 11, 16)

SCENARIOS = {
    "family_winter": "Family of 4 with kids aged 6 and 9, 10-14 Feb 2027, AED 14,000, "
                     "vegetarian, history and beach",
    "couple_summer": "Couple, 15-18 July 2027, AED 9,000, shopping and views, halal",
    "solo_budget": "Solo, 2027-01-20 to 2027-01-22, AED 2,500, culture and food",
    "luxury_week": "2 adults, a week in March 2027, AED 60,000, luxury, views and beach",
    "wheelchair": "Me and my husband, 3 days from 8 Jan 2027, AED 7000, he uses a "
                  "wheelchair, relaxed pace, museums and views",
    "vegan_friends": "3 friends, 4 nights starting 5 April 2027, AED 15000, vegan, art",
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_chain_produces_approved_grounded_itinerary(scenario: str, toolbox: Toolbox) -> None:
    llm = get_chat_llm()  # LLM_PROVIDER=mock in tests: no code path differs from real use
    with run_context(f"it-{scenario}", mode="langgraph"), capture() as events:
        parsed = requirements.run(requirements.RequirementsInput(
            message=SCENARIOS[scenario], today=TODAY), llm, toolbox)
        assert parsed.request is not None, parsed.missing
        req = parsed.request
        research, _plan, itin = compose(req, llm, toolbox, session_id=scenario)
        violations = validate_itinerary(itin, req, research, catalog=toolbox.catalog)
        report = reviewer.run(reviewer.ReviewerInput(itinerary=itin, request=req,
                                                     violations=violations), llm)

    errors = [v for v in violations if v.severity is Severity.ERROR]
    assert not errors, [v.message for v in errors]
    assert report.approved
    assert itin.cost.total <= req.total_budget_aed() * Decimal("1.05")  # AC3.1 precursor
    assert itin.venue_ids() <= research.known_ids()  # grounded
    assert itin.cost.is_consistent and itin.cost.is_estimate
    assert len(itin.days) == req.num_days
    if req.dietary:
        dining_ids = {s.activity.ref_id for d in itin.days for s in d.slots
                      if s.activity.kind.value == "dining"}
        assert all(toolbox.catalog.venue(i).caters_for(list(req.dietary))  # type: ignore[union-attr]
                   for i in dining_ids)
    assert all(e.run_id == f"it-{scenario}" for e in events)
    assert sum(e.tokens_in for e in events) > 0


def test_infeasible_budget_is_reported_not_hidden(toolbox: Toolbox) -> None:
    llm = get_chat_llm()
    parsed = requirements.run(requirements.RequirementsInput(
        message="Family of 5, 10-16 Feb 2027, AED 1,500", today=TODAY), llm, toolbox)
    assert parsed.request is not None
    research, plan, itin = compose(parsed.request, llm, toolbox)
    assert not plan.feasible
    violations = validate_itinerary(itin, parsed.request, research, catalog=toolbox.catalog)
    report = reviewer.run(reviewer.ReviewerInput(itinerary=itin, request=parsed.request,
                                                 violations=violations), llm)
    assert any(v.rule_id == "budget.total" for v in violations) and not report.approved


def test_chain_is_deterministic(toolbox: Toolbox) -> None:
    from dubai_planner.llm.mock import MockLLM

    outputs = []
    for _ in range(2):
        llm = MockLLM()
        parsed = requirements.run(requirements.RequirementsInput(
            message=SCENARIOS["family_winter"], today=TODAY), llm, toolbox)
        assert parsed.request is not None
        _r, _p, itin = compose(parsed.request, llm, toolbox)
        outputs.append(itin.model_dump(exclude={"id", "created_at"}))
    assert outputs[0] == outputs[1]
