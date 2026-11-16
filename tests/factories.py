"""Builders for valid domain objects used across tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from dubai_planner.data.seed import load_catalog
from dubai_planner.models import (
    Activity,
    ActivityKind,
    BudgetCategory,
    CostBreakdown,
    CostItem,
    DataKind,
    DayPlan,
    Itinerary,
    PlannerMode,
    PriceRange,
    PriceUnit,
    Provenance,
    TimeSlot,
    TransportLeg,
    TransportMode,
    Travellers,
    TripRequest,
)

AS_OF = date(2026, 11, 15)


def provenance(kind: DataKind = DataKind.CURATED, **kw: Any) -> Provenance:
    return Provenance(kind=kind, source=kw.pop("source", "test"), as_of=AS_OF, **kw)


def price(lo: str = "10", hi: str = "20", unit: PriceUnit = PriceUnit.PER_PERSON) -> PriceRange:
    return PriceRange(
        min_aed=Decimal(lo),
        max_aed=Decimal(hi),
        unit=unit,
        provenance=provenance(DataKind.ESTIMATED),
    )


def trip_request(**overrides: Any) -> TripRequest:
    data: dict[str, Any] = {
        "start_date": date(2027, 2, 10),
        "end_date": date(2027, 2, 12),
        "travellers": Travellers(adults=2, children_ages=[7]),
        "budget_aed": Decimal("6000"),
        "interests": ["history", "food"],
        "dietary": ["vegetarian"],
    }
    data.update(overrides)
    return TripRequest.model_validate(data)


def slot(
    start: str,
    end: str,
    *,
    ref_id: str = "att_dubai_frame",
    cost: str = "50",
    indoor: bool = True,
    kind: ActivityKind = ActivityKind.ATTRACTION,
) -> TimeSlot:
    return TimeSlot(
        start=time.fromisoformat(start),
        end=time.fromisoformat(end),
        activity=Activity(kind=kind, name=ref_id, ref_id=ref_id),
        transport_from_prev=TransportLeg(
            mode=TransportMode.METRO, distance_km=5.2, duration_min=20, cost_aed=Decimal("5")
        ),
        cost_aed=Decimal(cost),
        indoor=indoor,
    )


def itinerary(session_id: str = "s1", *, days: int = 2, total: str | None = None) -> Itinerary:
    catalog = load_catalog()
    hotel = next(h for h in catalog.hotels if h.id == "htl_rove_downtown")
    start = date(2027, 2, 10)
    day_plans = [
        DayPlan(
            date=start + timedelta(days=i),
            theme=f"Day {i + 1}",
            slots=[
                slot("09:00", "10:30"),
                slot("12:30", "13:30", ref_id="res_xva_cafe", cost="60", kind=ActivityKind.DINING),
                slot("18:00", "19:00", ref_id="att_dubai_fountain", cost="0", indoor=False),
            ],
            day_cost_aed=Decimal("110"),
        )
        for i in range(days)
    ]
    items = [
        CostItem(
            category=BudgetCategory.ACCOMMODATION,
            description="Rove Downtown x2",
            amount_aed=Decimal("800"),
            ref_id=hotel.id,
        ),
        CostItem(
            category=BudgetCategory.ACTIVITIES,
            description="Dubai Frame x2",
            amount_aed=Decimal("100"),
            ref_id="att_dubai_frame",
        ),
        CostItem(category=BudgetCategory.FOOD, description="Meals", amount_aed=Decimal("120")),
        CostItem(category=BudgetCategory.TRANSPORT, description="Metro", amount_aed=Decimal("20")),
        CostItem(category=BudgetCategory.CONTINGENCY, description="10%", amount_aed=Decimal("104")),
    ]
    cost = CostBreakdown.from_items(items, budget_aed=Decimal("6000"))
    if total is not None:
        cost = cost.model_copy(update={"total": Decimal(total)})
    return Itinerary(
        session_id=session_id,
        mode=PlannerMode.LANGGRAPH,
        title="Test trip",
        days=day_plans,
        hotel=hotel,
        cost=cost,
        created_at=datetime(2026, 11, 15, 12, 0, tzinfo=UTC),
    )


# --------------------------------------------------------------------------- Phase 2 helpers


def research_bundle(request: TripRequest, llm: Any, tools: Any) -> Any:
    """Run the five research agents (mock LLM) and bundle their outputs."""
    from dubai_planner.agents import accommodation, attractions, dining, transport, weather_events
    from dubai_planner.models import ResearchBundle

    att = attractions.run(attractions.AttractionsInput(request=request), llm, tools)
    hotels = accommodation.run(accommodation.AccommodationInput(request=request), llm, tools)
    dine = dining.run(dining.DiningInput(request=request), llm, tools)
    trans = transport.run(transport.TransportInput(request=request), None, tools)
    wx = weather_events.run(weather_events.WeatherEventsInput(
        start_date=request.start_date, end_date=request.end_date), None, tools)
    return ResearchBundle(attractions=att.options, hotels=hotels.options, dining=dine.options,
                          transport=trans, weather=wx.weather, events=wx.events)


def compose(request: TripRequest, llm: Any, tools: Any, *, session_id: str = "s1",
            research: Any = None, **kwargs: Any) -> tuple[Any, Any, Itinerary]:
    """research → budget → composer. Returns (research, budget_plan, itinerary)."""
    from dubai_planner.agents import budget, composer

    bundle = research if research is not None else research_bundle(request, llm, tools)
    plan = budget.run(budget.BudgetInput(request=request, research=bundle), llm, tools)
    itin = composer.run(composer.ComposerInput(
        session_id=session_id, mode=PlannerMode.LANGGRAPH, request=request, research=bundle,
        budget=plan, **kwargs), llm, tools)
    return bundle, plan, itin
