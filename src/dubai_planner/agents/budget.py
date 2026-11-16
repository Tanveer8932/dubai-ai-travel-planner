"""Budget agent: deterministic per-category allocation (``Decimal`` AED) + LLM rationale.

Allocation (hard rule 4, no LLM arithmetic):

- **accommodation** = nights × nightly midpoint × rooms (recommended hotel);
- **transport** = the transport agent's estimate;
- **food** = per-person-per-day rate × travellers × days, using the most expensive
  price band the budget allows after the planned activities (rates derived from seed
  restaurants; the budget band is the floor);
- **activities** = party cost of the top ``days × pace target`` attractions, capped at
  what is left after the categories above;
- **contingency** = ``contingency_pct`` × subtotal.

The categories always sum to ``allocated_total_aed``. The plan is **infeasible** (and
says why) when that total exceeds ``budget × (1 + BUDGET_TOLERANCE)``. The LLM only
writes the rationale; if it fails, a deterministic rationale is used (non-critical).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import Field

from dubai_planner.agents.base import LLMOutput, context_block, load_prompt
from dubai_planner.config import get_settings
from dubai_planner.data.seed import SeedCatalog
from dubai_planner.llm.errors import LLMCallError, StructuredOutputError
from dubai_planner.llm.mock import register_generator
from dubai_planner.llm.structured import build_messages, invoke_structured
from dubai_planner.models.budget import BudgetCategory, BudgetPlan, CategoryAllocation
from dubai_planner.models.common import PriceBand, StrictModel, quantize_aed
from dubai_planner.models.options import HotelOption
from dubai_planner.models.request import TripRequest
from dubai_planner.models.research import ResearchBundle
from dubai_planner.models.trace import TraceEventType
from dubai_planner.tools.costs import food_daily_rates, hotel_cost, party_cost, rooms_needed
from dubai_planner.tools.toolbox import Toolbox
from dubai_planner.tracing import agent_span, emit
from dubai_planner.validation.thresholds import CONTINGENCY_PCT, PACE_TARGET_ACTIVITIES

AGENT = "budget"
ZERO = Decimal("0")


class BudgetInput(StrictModel):
    request: TripRequest
    research: ResearchBundle
    hotel_id: str | None = Field(default=None, description="Default: first researched hotel")


class BudgetRationale(LLMOutput):
    rationale: str = Field(min_length=1, max_length=1500)


class BudgetSummary(StrictModel):
    """What the rationale writer sees (all numbers already computed)."""

    budget_aed: str
    allocated_aed: str
    feasible: bool
    infeasibility_reason: str | None
    allocations: dict[str, str]
    bases: dict[str, str]
    travellers: int
    days: int


def _pick_hotel(data: BudgetInput, catalog: SeedCatalog) -> tuple[HotelOption | None, str]:
    if data.request.nights <= 0:
        return None, "no overnight stay"
    if data.hotel_id:
        hotel = next((h for h in data.research.hotels if h.id == data.hotel_id), None)
        if hotel is None:
            raise ValueError(f"hotel {data.hotel_id!r} is not among the researched hotels")
        return hotel, "selected hotel"
    if data.research.hotels:
        return data.research.hotels[0], "recommended hotel"
    cheapest = min(catalog.hotels, key=lambda h: h.nightly_rate.midpoint_aed)
    return cheapest, "no hotel matched the filters; cheapest seed hotel used as a floor"


def _food_band(rates: dict[PriceBand, Decimal], per_person_day: Decimal) -> PriceBand:
    fitting = [band for band in PriceBand if rates[band] <= per_person_day]
    return fitting[-1] if fitting else PriceBand.BUDGET


def compute_allocation(
    request: TripRequest,
    research: ResearchBundle,
    catalog: SeedCatalog,
    *,
    hotel: HotelOption | None,
    hotel_basis: str = "",
    contingency_pct: Decimal = CONTINGENCY_PCT,
    tolerance: Decimal | None = None,
) -> BudgetPlan:
    """Pure, deterministic allocation (``rationale`` left empty)."""
    tol = get_settings().budget_tolerance if tolerance is None else tolerance
    budget = request.total_budget_aed()
    party, days, nights = request.travellers, request.num_days, request.nights

    accommodation = hotel_cost(hotel, nights, party) if hotel is not None else ZERO
    acc_basis = (f"{nights} night(s) × {rooms_needed(party)} room(s) × AED "
                 f"{hotel.nightly_rate.midpoint_aed} ({hotel.name}; {hotel_basis})"
                 if hotel is not None else "no accommodation needed")

    planned = research.attractions[: days * PACE_TARGET_ACTIVITIES[request.pace]]
    wanted = quantize_aed(sum((party_cost(a.price, party) for a in planned), ZERO))

    if research.transport is not None:
        transport = research.transport.estimated_total_cost_aed
        tr_basis = f"transport plan, AED {research.transport.estimated_daily_cost_aed}/day"
    else:
        transport, tr_basis = ZERO, "transport not researched"

    # Priority: accommodation and transport, then food, then paid activities.
    spendable = budget / (1 + contingency_pct)
    remaining = spendable - accommodation - transport
    rates = food_daily_rates(catalog)
    person_days = party.total * days
    after_wanted = remaining - wanted
    band = _food_band(rates, after_wanted / person_days if after_wanted > 0 else ZERO)
    food = quantize_aed(rates[band] * person_days)
    food_basis = f"{band.value} band AED {rates[band]}/person/day × {party.total} × {days} day(s)"

    activities = quantize_aed(max(ZERO, min(wanted, remaining - food)))
    act_basis = f"{len(planned)} planned attraction(s) at midpoint prices for {party.total}"
    if activities < wanted:
        act_basis += f"; capped to fit the budget (the top picks would cost AED {wanted})"

    subtotal = accommodation + transport + food + activities
    contingency = quantize_aed(subtotal * contingency_pct)
    allocations = [
        CategoryAllocation(category=BudgetCategory.ACCOMMODATION, amount_aed=accommodation,
                           basis=acc_basis),
        CategoryAllocation(category=BudgetCategory.ACTIVITIES, amount_aed=activities,
                           basis=act_basis),
        CategoryAllocation(category=BudgetCategory.FOOD, amount_aed=food, basis=food_basis),
        CategoryAllocation(category=BudgetCategory.TRANSPORT, amount_aed=transport,
                           basis=tr_basis),
        CategoryAllocation(category=BudgetCategory.CONTINGENCY, amount_aed=contingency,
                           basis=f"{contingency_pct * 100:.0f}% of AED {quantize_aed(subtotal)}"),
    ]
    allocated = quantize_aed(sum((a.amount_aed for a in allocations), ZERO))
    limit = quantize_aed(budget * (1 + tol))
    feasible = allocated <= limit
    reason = None
    if not feasible:
        biggest = max(allocations, key=lambda a: a.amount_aed)
        reason = (f"Estimated cost AED {allocated:,.2f} exceeds the budget AED {budget:,.2f} "
                  f"(limit with {tol * 100:.0f}% tolerance: AED {limit:,.2f}) by AED "
                  f"{allocated - budget:,.2f}; largest category: {biggest.category.value} "
                  f"AED {biggest.amount_aed:,.2f}.")
    return BudgetPlan(budget_aed=budget, allocations=allocations, contingency_pct=contingency_pct,
                      feasible=feasible, infeasibility_reason=reason)


def summary_of(plan: BudgetPlan, request: TripRequest) -> BudgetSummary:
    return BudgetSummary(
        budget_aed=str(plan.budget_aed), allocated_aed=str(plan.allocated_total_aed),
        feasible=plan.feasible, infeasibility_reason=plan.infeasibility_reason,
        allocations={a.category.value: str(a.amount_aed) for a in plan.allocations},
        bases={a.category.value: a.basis for a in plan.allocations},
        travellers=request.travellers.total, days=request.num_days,
    )


def default_rationale(summary: BudgetSummary) -> str:
    """Deterministic rationale (mock reply and fallback)."""
    parts = ", ".join(f"{k} AED {v}" for k, v in summary.allocations.items())
    verdict = ("fits within the budget" if summary.feasible
               else f"does NOT fit: {summary.infeasibility_reason}")
    return (f"For {summary.travellers} traveller(s) over {summary.days} day(s), the estimated "
            f"cost is AED {summary.allocated_aed} against a budget of AED {summary.budget_aed} "
            f"and {verdict} Breakdown: {parts}. All prices are estimates from curated data.")


def run(data: BudgetInput, llm: Any, tools: Toolbox) -> BudgetPlan:
    with agent_span(AGENT) as span:
        hotel, basis = _pick_hotel(data, tools.catalog)
        plan = compute_allocation(data.request, data.research, tools.catalog, hotel=hotel,
                                  hotel_basis=basis)
        summary = summary_of(plan, data.request)
        messages = build_messages(
            system=load_prompt("budget.system"),
            user=load_prompt("budget.user", context=context_block(summary)),
            schema=BudgetRationale, agent=AGENT,
        )
        rationale: str
        try:
            rationale = invoke_structured(llm, messages, BudgetRationale, agent=AGENT
                                          ).value.rationale
        except (StructuredOutputError, LLMCallError) as exc:
            emit(AGENT, TraceEventType.INFO,
                 payload={"fallback": "deterministic_rationale", "error": type(exc).__name__})
            rationale = default_rationale(summary)
        plan = plan.model_copy(update={"rationale": rationale})
        span.update(allocated=str(plan.allocated_total_aed), budget=str(plan.budget_aed),
                    feasible=plan.feasible)
        return plan


register_generator(
    BudgetRationale.__name__,
    lambda req: BudgetRationale(
        rationale=default_rationale(BudgetSummary.model_validate(req.context or {}))),
)
