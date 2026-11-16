"""AC2.4: deterministic budget allocation with exact Decimal arithmetic."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from factories import research_bundle, trip_request

from dubai_planner.agents import budget
from dubai_planner.llm.mock import MockLLM
from dubai_planner.models import BudgetCategory, BudgetPlan, ResearchBundle, TripRequest
from dubai_planner.models.trace import TraceEventType
from dubai_planner.tools.costs import food_daily_rates, hotel_cost, party_cost
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import capture

CENT = Decimal("0.01")


@pytest.fixture
def research(toolbox: Toolbox, mock_llm: MockLLM) -> ResearchBundle:
    return research_bundle(trip_request(), mock_llm, toolbox)


def _plan(request: TripRequest, research: ResearchBundle, toolbox: Toolbox) -> BudgetPlan:
    return budget.compute_allocation(request, research, toolbox.catalog,
                                     hotel=research.hotels[0] if research.hotels else None)


def test_allocation_is_exact_decimal(research: ResearchBundle, toolbox: Toolbox) -> None:
    req = trip_request()
    plan = _plan(req, research, toolbox)
    for alloc in plan.allocations:
        assert isinstance(alloc.amount_aed, Decimal)
        assert alloc.amount_aed == alloc.amount_aed.quantize(CENT)
    hotel = research.hotels[0]
    assert plan.allocation(BudgetCategory.ACCOMMODATION) == hotel_cost(hotel, req.nights,
                                                                       req.travellers)
    assert plan.allocation(BudgetCategory.TRANSPORT) == research.transport.estimated_total_cost_aed  # type: ignore[union-attr]
    subtotal = sum((plan.allocation(c) for c in BudgetCategory if c is not
                    BudgetCategory.CONTINGENCY), Decimal("0"))
    assert plan.allocation(BudgetCategory.CONTINGENCY) == (subtotal * Decimal("0.10")).quantize(
        CENT)


def test_categories_sum_to_total(research: ResearchBundle, toolbox: Toolbox) -> None:
    plan = _plan(trip_request(), research, toolbox)
    assert {a.category for a in plan.allocations} == set(BudgetCategory)
    assert sum((a.amount_aed for a in plan.allocations), Decimal("0")) == plan.allocated_total_aed
    assert plan.unallocated_aed == plan.budget_aed - plan.allocated_total_aed


def test_food_uses_seed_derived_rates(research: ResearchBundle, toolbox: Toolbox) -> None:
    req = trip_request()
    plan = _plan(req, research, toolbox)
    person_days = req.travellers.total * req.num_days
    rates = set(food_daily_rates(toolbox.catalog).values())
    assert plan.allocation(BudgetCategory.FOOD) / person_days in rates


def test_feasible_plan_fits_budget(research: ResearchBundle, toolbox: Toolbox) -> None:
    plan = _plan(trip_request(budget_aed=Decimal("15000")), research, toolbox)
    assert plan.feasible and plan.infeasibility_reason is None
    assert plan.allocated_total_aed <= plan.budget_aed * Decimal("1.05")


def test_activities_capped_to_fit(research: ResearchBundle, toolbox: Toolbox) -> None:
    req = trip_request(budget_aed=Decimal("5000"))
    plan = _plan(req, research, toolbox)
    planned = research.attractions[: req.num_days * 3]
    wanted = sum((party_cost(a.price, req.travellers) for a in planned), Decimal("0"))
    assert plan.allocation(BudgetCategory.ACTIVITIES) <= wanted
    if plan.allocation(BudgetCategory.ACTIVITIES) < wanted:
        assert "capped" in next(a.basis for a in plan.allocations
                                if a.category is BudgetCategory.ACTIVITIES)


def test_over_budget_is_flagged_with_reason(research: ResearchBundle, toolbox: Toolbox) -> None:
    plan = _plan(trip_request(budget_aed=Decimal("800")), research, toolbox)
    assert not plan.feasible
    assert plan.infeasibility_reason and "exceeds the budget" in plan.infeasibility_reason
    assert plan.allocated_total_aed > plan.budget_aed * Decimal("1.05")
    assert plan.allocation(BudgetCategory.ACTIVITIES) == 0  # nothing left for extras


def test_tolerance_boundary(research: ResearchBundle, toolbox: Toolbox) -> None:
    req = trip_request(budget_aed=Decimal("800"))
    plan = _plan(req, research, toolbox)
    needed = plan.allocated_total_aed
    exact = budget.compute_allocation(
        req.model_copy(update={"budget_aed": needed}), research, toolbox.catalog,
        hotel=research.hotels[0], tolerance=Decimal("0"))
    assert exact.feasible  # allocation == budget is within a 0% tolerance


def test_budget_scopes(research: ResearchBundle, toolbox: Toolbox) -> None:
    per_person = trip_request(budget_aed=Decimal("2000"), budget_scope="per_person")
    plan = _plan(per_person, research, toolbox)
    assert plan.budget_aed == Decimal("6000.00")  # 3 travellers
    per_day = trip_request(budget_aed=Decimal("1000"), budget_scope="per_day")
    assert _plan(per_day, research, toolbox).budget_aed == Decimal("3000.00")  # 3 days


def test_day_visit_has_no_accommodation(toolbox: Toolbox, mock_llm: MockLLM) -> None:
    req = trip_request(start_date=date(2027, 2, 10), end_date=date(2027, 2, 10))
    bundle = research_bundle(req, mock_llm, toolbox)
    plan = budget.run(budget.BudgetInput(request=req, research=bundle), mock_llm, toolbox)
    assert plan.allocation(BudgetCategory.ACCOMMODATION) == 0


def test_no_researched_hotel_uses_cheapest_floor(research: ResearchBundle, toolbox: Toolbox,
                                                 mock_llm: MockLLM) -> None:
    empty = research.model_copy(update={"hotels": []})
    plan = budget.run(budget.BudgetInput(request=trip_request(), research=empty), mock_llm,
                      toolbox)
    cheapest = min(toolbox.catalog.hotels, key=lambda h: h.nightly_rate.midpoint_aed)
    acc = next(a for a in plan.allocations if a.category is BudgetCategory.ACCOMMODATION)
    assert cheapest.name in acc.basis


def test_unknown_hotel_id_rejected(research: ResearchBundle, toolbox: Toolbox,
                                   mock_llm: MockLLM) -> None:
    with pytest.raises(ValueError, match="not among the researched hotels"):
        budget.run(budget.BudgetInput(request=trip_request(), research=research,
                                      hotel_id="htl_nowhere"), mock_llm, toolbox)


def test_rationale_from_llm_quotes_numbers(research: ResearchBundle, toolbox: Toolbox,
                                           mock_llm: MockLLM) -> None:
    plan = budget.run(budget.BudgetInput(request=trip_request(), research=research), mock_llm,
                      toolbox)
    assert str(plan.allocated_total_aed) in plan.rationale and "estimate" in plan.rationale


def test_malformed_rationale_falls_back(research: ResearchBundle, toolbox: Toolbox,
                                       mock_llm: MockLLM) -> None:
    mock_llm.script("budget", "not json", '{"wrong": 1}')
    with capture() as events:
        plan = budget.run(budget.BudgetInput(request=trip_request(), research=research),
                          mock_llm, toolbox)
    assert plan.rationale.startswith("For 3 traveller(s)")
    assert any(e.event is TraceEventType.INFO and e.payload.get("fallback") for e in events)


def test_llm_cannot_change_the_numbers(research: ResearchBundle, toolbox: Toolbox,
                                       mock_llm: MockLLM) -> None:
    mock_llm.script("budget", '{"rationale": "Total is AED 1. Trust me."}')
    plan = budget.run(budget.BudgetInput(request=trip_request(), research=research), mock_llm,
                      toolbox)
    expected = _plan(trip_request(), research, toolbox)
    assert [(a.category, a.amount_aed) for a in plan.allocations] == [
        (a.category, a.amount_aed) for a in expected.allocations]
    assert plan.feasible == expected.feasible
